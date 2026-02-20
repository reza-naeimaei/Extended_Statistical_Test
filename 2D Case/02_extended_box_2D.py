"""
2D interval-extended congruency test simulation (BOX-bias; stochastic + bounded systematics).

Model
-----
    d = mu_d + b + e,     e ~ N(0, Σ),   b ∈ B = [-Δx,Δx]×[-Δy,Δy]

For a fixed true bias b = (bx,by) (outer grid), we run Monte Carlo over stochastic noise e
(inner loop) and estimate conditional decision probabilities.

Interval-extended test statistic (BOX bias, level alpha, df=2):
    T(d,b) = (d - b)ᵀ Σ⁻¹ (d - b)
    Tmin(d) = min_{b∈B} T(d,b)
    Tmax(d) = max_{b∈B} T(d,b)

Decision (kα = χ²_{2,1-α}):
    strict accept  ⇔ Tmax(d) ≤ kα          (robustly inside)
    strict reject  ⇔ Tmin(d)  > kα          (robustly outside)
    ambiguous      otherwise

This script produces:
  - 2×3 bias maps (H0 / Ha): P(strict accept | b), P(ambiguous | b), P(strict reject | b)
  - diagnostics for selected biases: (A) scatter with Minkowski sum/diff contours, (B) histograms of Tmin/Tmax

Notes
-----
- The BOX minimiser uses component-wise clipping in measurement space (consistent with your previous script).
- The BOX maximiser is evaluated exactly by enumerating the 4 corners of the box.
- Uses common random numbers (same noise) across grid points.
"""

from __future__ import annotations

import os
import contextlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import chi2
from joblib import Parallel, delayed
from matplotlib.lines import Line2D
from joblib.parallel import BatchCompletionCallBack
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


# =============================
# Performance / thread control
# =============================
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


# =============================
# Configuration (edit here)
# =============================
@dataclass(frozen=True)
class Config2DExtendedBOX:
    # covariance principal std devs (meters) and rotation (deg)
    sigma1: float = 0.02
    sigma2: float = 0.01
    theta_deg: float = 0.0

    # hypothesis settings
    mu_h0: Tuple[float, float] = (0.0, 0.0)
    mu_ha: Tuple[float, float] = (0.02, 0.01)

    # bias box shown in maps (meters): b_x in [-Δx,Δx], b_y in [-Δy,Δy]
    Delta: Tuple[float, float] = (0.005, 0.005)
    n_grid: int = 201

    # Monte Carlo
    n_stoch: int = 10_000
    seed: int = 0
    alpha: float = 0.05

    # parallelization
    n_jobs: int = -1
    backend: str = "threading"
    verbose: int = 0  # keep 0; we use tqdm instead

    # diagnostics (meters)
    b_selected: Tuple[Tuple[float, float], ...] = ((-0.005, 0.005), (0.0, 0.0), (0.005, 0.005))

    # output folder (under script folder)
    out_dirname: str = "outputs/extended_box_2D"


# =============================
# Utilities
# =============================
def rotation_matrix(theta_deg: float) -> np.ndarray:
    th = np.deg2rad(theta_deg)
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s],
                     [s,  c]], dtype=float)


def make_covariance_matrix(sigma1: float, sigma2: float, *, theta_deg: float = 0.0) -> np.ndarray:
    """Build Σ = R diag(sigma1^2, sigma2^2) R^T."""
    R = rotation_matrix(theta_deg)
    S = np.diag([sigma1**2, sigma2**2]).astype(float)
    Sigma = R @ S @ R.T
    return 0.5 * (Sigma + Sigma.T)


def critical_value(alpha: float, df: int = 2) -> float:
    return float(chi2.ppf(1.0 - alpha, df=df))


def quadratic_form(d: np.ndarray, Sigma_inv: np.ndarray) -> np.ndarray:
    """Vectorised quadratic form for d with shape (...,2): dᵀ Σ⁻¹ d."""
    return np.einsum("...i,ij,...j->...", d, Sigma_inv, d)


def interval_stats_box_points(d: np.ndarray, Delta: np.ndarray, Sigma_inv: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Tmin(d), Tmax(d) for a set of points d (shape (N,2)).

    Consistent with your earlier script:
      - Tmin: b* via component-wise clipping (approx for rotated Σ)
      - Tmax: exact by enumerating 4 box corners
    """
    d = np.asarray(d, float)
    Delta = np.asarray(Delta, float).reshape(2)
    Sigma_inv = np.asarray(Sigma_inv, float).reshape(2, 2)

    # Tmin (clip)
    b_star = np.clip(d, -Delta[None, :], Delta[None, :])
    Tmin = quadratic_form(d - b_star, Sigma_inv)

    # Tmax (corners)
    corners = np.array([
        [-Delta[0], -Delta[1]],
        [-Delta[0], +Delta[1]],
        [+Delta[0], -Delta[1]],
        [+Delta[0], +Delta[1]],
    ], dtype=float)  # (4,2)

    # Evaluate all corners in one go: q[k,n] = (d[n]-corner[k])ᵀ S (d[n]-corner[k])
    diff = d[None, :, :] - corners[:, None, :]  # (4,N,2)
    q = np.einsum("kni,ij,knj->kn", diff, Sigma_inv, diff)  # (4,N)
    Tmax = np.max(q, axis=0)

    return Tmin, Tmax


def decide_extended(Tmin: np.ndarray, Tmax: np.ndarray, k: float) -> np.ndarray:
    """
    Return decision codes for each sample:
      0 = strict accept (Tmax <= k)
      1 = ambiguous    (otherwise, not strict reject)
      2 = strict reject (Tmin > k)
    """
    Tmin = np.asarray(Tmin, float)
    Tmax = np.asarray(Tmax, float)
    dec = np.ones_like(Tmin, dtype=int)  # default ambiguous
    dec[Tmax <= k] = 0
    dec[Tmin > k] = 2
    return dec


# =============================
# tqdm + joblib helper
# =============================
@contextlib.contextmanager
def tqdm_joblib(tqdm_object: tqdm):
    """Context manager to patch joblib to report into tqdm progress bar."""
    class TqdmBatchCompletionCallback(BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_callback = BatchCompletionCallBack
    try:
        import joblib.parallel
        joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback  # type: ignore[attr-defined]
        yield tqdm_object
    finally:
        import joblib.parallel
        joblib.parallel.BatchCompletionCallBack = old_callback  # type: ignore[attr-defined]
        tqdm_object.close()


# =============================
# Parallel worker (row-wise)
# =============================
def _worker_row_extended_box(
    ix: int,
    bxi: float,
    by: np.ndarray,
    noise: np.ndarray,
    mu_d: np.ndarray,
    Delta: np.ndarray,
    Sigma_inv: np.ndarray,
    alpha: float,
) -> Tuple[int, Dict[str, np.ndarray]]:
    """
    Process one row (fixed bx=bxi, loop over by).
    Returns ix and vectors of length len(by):
      P_strict_accept, P_ambiguous, P_strict_reject
    """
    mu_d = np.asarray(mu_d, float).reshape(2)
    by = np.asarray(by, float)
    Delta = np.asarray(Delta, float).reshape(2)
    k = critical_value(alpha, df=2)

    p_sa = np.zeros(by.size, float)
    p_am = np.zeros(by.size, float)
    p_sr = np.zeros(by.size, float)

    for iy, byj in enumerate(by):
        b_vec = np.array([bxi, byj], dtype=float)
        d = mu_d[None, :] + b_vec[None, :] + noise  # (n_stoch,2)

        Tmin, Tmax = interval_stats_box_points(d, Delta, Sigma_inv)
        dec = decide_extended(Tmin, Tmax, k)

        p_sa[iy] = float(np.mean(dec == 0))
        p_am[iy] = float(np.mean(dec == 1))
        p_sr[iy] = float(np.mean(dec == 2))

    return ix, dict(P_ext_strict_accept=p_sa, P_ext_ambiguous=p_am, P_ext_strict_reject=p_sr)


# =============================
# Bias maps
# =============================
def run_bias_map_extended_box(
    *,
    mu_d: np.ndarray,
    Sigma: np.ndarray,
    Delta: np.ndarray,
    alpha: float,
    n_stoch: int,
    n_grid: int,
    seed: int,
    n_jobs: int,
    backend: str,
    verbose: int,
    save_npz: Optional[Path] = None,
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    mu_d = np.asarray(mu_d, float).reshape(2)
    Sigma = np.asarray(Sigma, float).reshape(2, 2)
    Delta = np.asarray(Delta, float).reshape(2)
    Sigma_inv = np.linalg.inv(Sigma)

    bx = np.linspace(-Delta[0], Delta[0], n_grid)
    by = np.linspace(-Delta[1], Delta[1], n_grid)

    # Common random numbers: SAME noise for all grid points
    noise = rng.multivariate_normal(mean=np.zeros(2), cov=Sigma, size=n_stoch)

    out = {
        "P_ext_strict_accept": np.zeros((n_grid, n_grid), float),
        "P_ext_ambiguous": np.zeros((n_grid, n_grid), float),
        "P_ext_strict_reject": np.zeros((n_grid, n_grid), float),
    }

    tasks = [
        delayed(_worker_row_extended_box)(ix, float(bx[ix]), by, noise, mu_d, Delta, Sigma_inv, alpha)
        for ix in range(n_grid)
    ]

    with tqdm_joblib(tqdm(total=n_grid, desc=f"Bias map rows (n_grid={n_grid}, n_stoch={n_stoch})", unit="row")):
        results = Parallel(n_jobs=n_jobs, backend=backend, verbose=verbose)(tasks)

    for ix, row in results:
        out["P_ext_strict_accept"][:, ix] = row["P_ext_strict_accept"]
        out["P_ext_ambiguous"][:, ix] = row["P_ext_ambiguous"]
        out["P_ext_strict_reject"][:, ix] = row["P_ext_strict_reject"]

    pack = dict(
        bx=bx, by=by,
        mu_d=mu_d, Sigma=Sigma, Delta=Delta,
        alpha=np.array([alpha], float),
        n_stoch=np.array([n_stoch], int),
        seed=np.array([seed], int),
        **out
    )
    if save_npz is not None:
        np.savez(save_npz, **pack)
    return pack


# =============================
# Plotting
# =============================
def _extent_mm(out: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, list]:
    bx = out["bx"] * 1000.0
    by = out["by"] * 1000.0
    extent = [bx.min(), bx.max(), by.min(), by.max()]
    return bx, by, extent


def plot_decision_grid_extended(out_h0: Dict[str, np.ndarray], out_ha: Dict[str, np.ndarray], save_path: Optional[Path] = None) -> None:
    bx, by, extent = _extent_mm(out_h0)

    data_grid = [
        [out_h0["P_ext_strict_accept"], out_h0["P_ext_ambiguous"], out_h0["P_ext_strict_reject"]],
        [out_ha["P_ext_strict_accept"], out_ha["P_ext_ambiguous"], out_ha["P_ext_strict_reject"]],
    ]

    col_labels = [
        r"$P(\mathrm{Strict\ Accept}\mid \mathbf{b})$",
        r"$P(\mathrm{Ambiguous}\mid \mathbf{b})$",
        r"$P(\mathrm{Strict\ Reject}\mid \mathbf{b})$",
    ]
    row_labels = [r"$H_0:\ \mu_d=\mathbf{0}$", r"$H_a:\ \mu_d\neq \mathbf{0}$"]

    fig, axs = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True)
    cmap = "RdBu"

    for r in range(2):
        for c in range(3):
            ax = axs[r, c]
            Z = data_grid[r][c]

            im = ax.imshow(Z, origin="lower", extent=extent, cmap=cmap, interpolation="bilinear")
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cb.ax.tick_params(labelsize=14)

            if np.nanmax(Z) > np.nanmin(Z):
                cs = ax.contour(Z, origin="lower", extent=extent, colors="black", linewidths=0.8, alpha=0.6, levels=6)
                ax.clabel(cs, inline=True, fontsize=10, fmt="%.2f")

            if r == 1:
                ax.set_xlabel(r"$b_x$ [mm]", fontsize=14)
            if c == 0:
                ax.set_ylabel(r"$b_y$ [mm]", fontsize=14)
                ax.annotate(row_labels[r], xy=(-0.40, 0.5), xycoords="axes fraction",
                            fontsize=14, va="center", ha="center", rotation=90, weight="bold")
            if r == 0:
                ax.set_title(col_labels[c], fontsize=14, weight="bold")

            ax.tick_params(axis="both", labelsize=14)

    fig.suptitle("Extended (BOX) test decision probabilities", fontsize=16)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def _minkowski_contours(
    ax: plt.Axes,
    Sigma_inv: np.ndarray,
    Delta: np.ndarray,
    k: float,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    *,
    n_grid: int = 220,
    unit_scale: float = 100.0,
) -> None:
    """
    Plot Minkowski sum/diff contours for the ellipse (dᵀS d ≤ k) and box B:
      - outer boundary (blue): Tmin(d)=k  -> dilation (E ⊕ B)
      - inner boundary (green): Tmax(d)=k -> erosion  (E ⊖ B)
    """
    x = np.linspace(xlim[0], xlim[1], n_grid)
    y = np.linspace(ylim[0], ylim[1], n_grid)
    X, Y = np.meshgrid(x, y)
    pts = np.column_stack([X.ravel(), Y.ravel()])
    Tmin, Tmax = interval_stats_box_points(pts, Delta, Sigma_inv)
    Tmin = Tmin.reshape(n_grid, n_grid)
    Tmax = Tmax.reshape(n_grid, n_grid)

    ax.contour(X * unit_scale, Y * unit_scale, Tmin, levels=[k], colors="blue", linewidths=2.0)
    ax.contour(X * unit_scale, Y * unit_scale, Tmax, levels=[k], colors="green", linewidths=2.0, linestyles="--")




# ---------------------------------------------------------------------
# Minkowski helpers (for plotting boundaries of the interval test)
# ---------------------------------------------------------------------
def minkowski_sum_ellipse_box(
    ax: plt.Axes,
    Sigma_inv: np.ndarray,
    Delta: np.ndarray,
    k: float,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    *,
    unit_scale: float = 1.0,
    n_grid: int = 220,
    color: str = "blue",
    lw: float = 2.0,
    ls: str = "-",
    zorder: int = 6,
):
    """Plot the inner boundary (T_min = k).

    Interpretation (BOX bias):
      - T_min(d) = min_{b∈[-Δ,Δ]} (d-b)^T Σ^{-1} (d-b)
      - The level-set {d: T_min(d)=k} is the strict-accept boundary.
    """
    Sigma_inv = np.asarray(Sigma_inv, float).reshape(2, 2)
    Delta = np.asarray(Delta, float).reshape(2)
    x = np.linspace(float(xlim[0]), float(xlim[1]), n_grid)
    y = np.linspace(float(ylim[0]), float(ylim[1]), n_grid)
    X, Y = np.meshgrid(x, y, indexing="xy")
    pts = np.stack([X.ravel(), Y.ravel()], axis=1)
    Tmin, _ = interval_stats_box_points(pts, Sigma_inv=Sigma_inv, Delta=Delta)
    Z = Tmin.reshape(n_grid, n_grid)
    cs = ax.contour(X * unit_scale, Y * unit_scale, Z, levels=[k],
                    colors=[color], linewidths=[lw], linestyles=[ls], zorder=zorder)
    return cs


def minkowski_diff_ellipse_box(
    ax: plt.Axes,
    Sigma_inv: np.ndarray,
    Delta: np.ndarray,
    k: float,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    *,
    unit_scale: float = 1.0,
    n_grid: int = 220,
    color: str = "green",
    lw: float = 2.0,
    ls: str = "--",
    zorder: int = 6,
):
    """Plot the outer boundary (T_max = k).

    Interpretation (BOX bias):
      - T_max(d) = max_{b∈[-Δ,Δ]} (d-b)^T Σ^{-1} (d-b)
      - The level-set {d: T_max(d)=k} is the strict-reject boundary.
    """
    Sigma_inv = np.asarray(Sigma_inv, float).reshape(2, 2)
    Delta = np.asarray(Delta, float).reshape(2)
    x = np.linspace(float(xlim[0]), float(xlim[1]), n_grid)
    y = np.linspace(float(ylim[0]), float(ylim[1]), n_grid)
    X, Y = np.meshgrid(x, y, indexing="xy")
    pts = np.stack([X.ravel(), Y.ravel()], axis=1)
    _, Tmax = interval_stats_box_points(pts, Sigma_inv=Sigma_inv, Delta=Delta)
    Z = Tmax.reshape(n_grid, n_grid)
    cs = ax.contour(X * unit_scale, Y * unit_scale, Z, levels=[k],
                    colors=[color], linewidths=[lw], linestyles=[ls], zorder=zorder)
    return cs

def plot_selected_bias_scatter_and_hist_grid_extended(
    *,
    b_list: Tuple[Tuple[float, float], ...],
    mu_h0: np.ndarray,
    mu_ha: np.ndarray,
    Sigma: np.ndarray,
    Delta: np.ndarray,
    alpha: float,
    n_stoch: int,
    seed: int,
    unit_scale: float = 100.0,
    unit_label: str = "cm",
    save_prefix: Optional[Path] = None,
    show: bool = True,
    # --- styling / geometry resolution ---
    contour_grid: int = 240,
    # Minkowski sum/diff ellipse borders
    sum_color: str = "blue",
    sum_lw: float = 2.0,
    sum_ls: str = "-",
    diff_color: str = "green",
    diff_lw: float = 2.0,
    diff_ls: str = "--",
    # Pie inset
    pie_loc: str = "lower right",
    pie_width: str = "42%",
    pie_height: str = "42%",
    pie_borderpad: float = 0.1,
) -> None:
    """
    Style-aligned with plot_selected_bias_scatter_and_hist_grid (classic):

      - Same figsize policy: (4.2*n, 7.2) for hist and (4.2*n, 8.0) for scatter
      - Figure-level legend at top-center with same bbox
      - Fixed scatter axes limits (in cm): x in [-10,10], y in [-8,8]
      - No tight_layout / constrained_layout (avoid inset-axes warnings)
      - Same row-label placement on the left side

    Extended-specific differences:
      - scatter points colored by extended decision (strict accept / ambiguous / strict reject)
      - Minkowski sum/diff contours (Tmin=k and Tmax=k)
      - histograms: Tmin and Tmax (semi-transparent)
    """
    from matplotlib.lines import Line2D

    def _max_density(values: np.ndarray, bins: np.ndarray) -> float:
        h, _ = np.histogram(values, bins=bins, density=True)
        return float(np.max(h)) if h.size else 0.0

    rng = np.random.default_rng(seed)
    mu_h0 = np.asarray(mu_h0, float).reshape(2)
    mu_ha = np.asarray(mu_ha, float).reshape(2)
    Sigma = np.asarray(Sigma, float).reshape(2, 2)
    Delta = np.asarray(Delta, float).reshape(2)

    Sigma_inv = np.linalg.inv(Sigma)
    k = critical_value(alpha, df=2)

    n = len(b_list)

    # Common random numbers for all panels
    noise = rng.multivariate_normal(mean=np.zeros(2), cov=Sigma, size=n_stoch)

    # Fixed scatter axes limits (cm) to match classic
    xlim_cm = (-10.0, 10.0)
    ylim_cm = (-8.0, 8.0)

    # ------------------------------------------------------------
    # Precompute Tmin/Tmax for histograms (consistent x-range)
    # ------------------------------------------------------------
    Tmin_vals = [[None for _ in range(n)] for _ in range(2)]
    Tmax_vals = [[None for _ in range(n)] for _ in range(2)]

    Tmax_global = 0.0
    for col, b in enumerate(b_list):
        b_vec = np.asarray(b, float).reshape(2)
        for row, mu_row in enumerate([mu_h0, mu_ha]):
            d = mu_row[None, :] + b_vec[None, :] + noise
            Tmin, Tmax = interval_stats_box_points(d, Delta, Sigma_inv)
            Tmin_vals[row][col] = Tmin
            Tmax_vals[row][col] = Tmax
            Tmax_global = max(Tmax_global, float(np.max(Tmax)))

    # Histogram x-range (kept as you wrote)
    x_hi = max(Tmax_global, 2.0 * k) * 1.05
    bins = np.linspace(0.0, x_hi, 50)

    # y-range (shared across ALL panels in this figure)
    y_hi_hist = 0.0
    for row in range(2):
        for col in range(n):
            y_hi_hist = max(y_hi_hist, _max_density(Tmin_vals[row][col], bins))
            y_hi_hist = max(y_hi_hist, _max_density(Tmax_vals[row][col], bins))

    x_smooth = np.linspace(0.0, x_hi, 700)
    y_chi2 = chi2.pdf(x_smooth, df=2)
    y_hi = 1.05 * max(y_hi_hist, float(np.max(y_chi2)))

    # ============================================================
    # (A) HISTOGRAM FIGURE (2×N)
    # ============================================================
    fig_h, axs_h = plt.subplots(2, n, figsize=(4.2 * n, 7.2), constrained_layout=False)
    if n == 1:
        axs_h = np.array([[axs_h[0]], [axs_h[1]]])

    fig_h.text(0.012, 0.63, r"$H_0:\ \mu_d=\mathbf{0}$",
               rotation=90, va="center", ha="left", fontsize=14, weight="bold")
    fig_h.text(0.012, 0.23, r"$H_a:\ \mu_d\neq \mathbf{0}$",
               rotation=90, va="center", ha="left", fontsize=14, weight="bold")

    legend_handles_h = {}

    for row in range(2):
        for col, b in enumerate(b_list):
            ax = axs_h[row, col]
            b_vec = np.asarray(b, float).reshape(2)

            Tmin = Tmin_vals[row][col]
            Tmax = Tmax_vals[row][col]

            h_min = ax.hist(Tmin, bins=bins, density=True, color="#1f77b4", alpha=0.35, label=r"$T_{\min}$")
            h_max = ax.hist(Tmax, bins=bins, density=True, color="#ff7f0e", alpha=0.35, label=r"$T_{\max}$")

            h_ref, = ax.plot(
                x_smooth, y_chi2,
                color="black", linestyle="--", linewidth=2.0, alpha=0.9,
                label=r"$\chi^2_2$ (H0 ref)"
            )
            h_k = ax.axvline(k, color="red", linewidth=2, label=r"$k_\alpha$")

            ax.set_xlim(0.0, x_hi)
            ax.set_ylim(0.0, y_hi)
            ax.grid(True, alpha=0.25)
            ax.tick_params(axis="both", labelsize=12)

            if row == 1:
                ax.set_xlabel(r"$T_{\min},\ T_{\max}$", fontsize=13)
            if col == 0:
                ax.set_ylabel("Density", fontsize=13)

            hyp = r"$H_0$" if row == 0 else r"$H_a$"
            ax.set_title(
                rf"{hyp}: $b=({b_vec[0]*unit_scale:.1f},{b_vec[1]*unit_scale:.1f})$ {unit_label}",
                fontsize=13,
            )

            legend_handles_h.setdefault(r"$T_{\min}$", h_min[2][0] if len(h_min) > 2 else None)
            legend_handles_h.setdefault(r"$T_{\max}$", h_max[2][0] if len(h_max) > 2 else None)
            legend_handles_h.setdefault(r"$\chi^2_2$ (H0 ref)", h_ref)
            legend_handles_h.setdefault(r"$k_\alpha$", h_k)

    fig_h.suptitle("Extended (BOX) statistic histograms", fontsize=18, y=0.98)

    leg_order_h = [r"$T_{\min}$", r"$T_{\max}$", r"$\chi^2_2$ (H0 ref)", r"$k_\alpha$"]
    handles, labels = [], []
    for lab in leg_order_h:
        h = legend_handles_h.get(lab, None)
        if h is not None:
            handles.append(h)
            labels.append(lab)
    if handles:
        fig_h.legend(handles, labels, loc="upper center", ncol=4, frameon=True, fontsize=11,
                     bbox_to_anchor=(0.5, 0.93))

    fig_h.subplots_adjust(left=0.09, right=0.99, bottom=0.08, top=0.84, wspace=0.28, hspace=0.30)

    if save_prefix is not None:
        fig_h.savefig(save_prefix.with_suffix("").as_posix() + "_hist_grid.png", dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig_h)

    # ============================================================
    # (B) SCATTER FIGURE (2×N)
    # ============================================================
    fig_s, axs_s = plt.subplots(2, n, figsize=(4.2 * n, 8.0), constrained_layout=False)
    if n == 1:
        axs_s = np.array([[axs_s[0]], [axs_s[1]]])

    fig_s.text(0.012, 0.63, r"$H_0:\ \mu_d=\mathbf{0}$",
               rotation=90, va="center", ha="left", fontsize=14, weight="bold")
    fig_s.text(0.012, 0.23, r"$H_a:\ \mu_d\neq \mathbf{0}$",
               rotation=90, va="center", ha="left", fontsize=14, weight="bold")

    proxy_minus = Line2D([0], [0], color=sum_color, lw=sum_lw, ls=sum_ls, label=r"ellipse $\ominus$ box")
    proxy_plus = Line2D([0], [0], color=diff_color, lw=diff_lw, ls=diff_ls, label=r"ellipse $\oplus$ box")

    legend_handles_s = {}
    legend_handles_s[r"ellipse $\ominus$ box"] = proxy_minus
    legend_handles_s[r"ellipse $\oplus$ box"] = proxy_plus

    xlim_m = (xlim_cm[0] / unit_scale, xlim_cm[1] / unit_scale)
    ylim_m = (ylim_cm[0] / unit_scale, ylim_cm[1] / unit_scale)

    for row, mu_row in enumerate([mu_h0, mu_ha]):
        for col, b in enumerate(b_list):
            ax = axs_s[row, col]
            b_vec = np.asarray(b, float).reshape(2)
            mean_vec = mu_row + b_vec

            d = mu_row[None, :] + b_vec[None, :] + noise
            Tmin, Tmax = interval_stats_box_points(d, Delta, Sigma_inv)
            dec = decide_extended(Tmin, Tmax, k)

            sa = (dec == 0)
            am = (dec == 1)
            sr = (dec == 2)

            h_sa = ax.scatter(d[sa, 0] * unit_scale, d[sa, 1] * unit_scale, s=10, c="lightgreen", label="Strict Accept")
            h_am = ax.scatter(d[am, 0] * unit_scale, d[am, 1] * unit_scale, s=10, c="khaki", label="Ambiguous")
            h_sr = ax.scatter(d[sr, 0] * unit_scale, d[sr, 1] * unit_scale, s=10, c="lightcoral", label="Strict Reject")

            h_mean = ax.scatter(mean_vec[0] * unit_scale, mean_vec[1] * unit_scale, c="blue", marker="o", s=50, label=r"$\mu_d + b$")
            h_org = ax.scatter(0.0, 0.0, c="black", marker="+", s=60, label="Origin")

            minkowski_diff_ellipse_box(
                ax, Sigma_inv, Delta, k,
                xlim_m, ylim_m,
                unit_scale=unit_scale, n_grid=contour_grid,
                color=sum_color, lw=sum_lw, ls=sum_ls,
            )
            minkowski_sum_ellipse_box(
                ax, Sigma_inv, Delta, k,
                xlim_m, ylim_m,
                unit_scale=unit_scale, n_grid=contour_grid,
                color=diff_color, lw=diff_lw, ls=diff_ls,
            )

            ax_in = inset_axes(ax, width=pie_width, height=pie_height, loc=pie_loc, borderpad=pie_borderpad)
            ax_in.pie(
                (100.0 * float(np.mean(sa)), 100.0 * float(np.mean(am)), 100.0 * float(np.mean(sr))),
                colors=("lightgreen", "khaki", "lightcoral"),
                autopct="%1.0f%%",
                startangle=0,
                textprops=dict(fontsize=10),
            )
            ax_in.set_aspect("equal")
            ax_in.set_xticks([])
            ax_in.set_yticks([])

            ax.set_aspect("equal")
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.tick_params(axis="both", labelsize=14)

            if row == 1:
                ax.set_xlabel(rf"$d_x$ [{unit_label}]", fontsize=14)
            if col == 0:
                ax.set_ylabel(rf"$d_y$ [{unit_label}]", fontsize=14)

            hyp = r"$H_0$" if row == 0 else r"$H_a$"
            ax.set_title(
                rf"{hyp}: $b=({b_vec[0]*unit_scale:.1f},{b_vec[1]*unit_scale:.1f})$ {unit_label}",
                fontsize=13,
            )

            ax.set_xlim(xlim_cm[0], xlim_cm[1])
            ax.set_ylim(ylim_cm[0], ylim_cm[1])

            legend_handles_s.setdefault("Strict Accept", h_sa)
            legend_handles_s.setdefault("Ambiguous", h_am)
            legend_handles_s.setdefault("Strict Reject", h_sr)
            legend_handles_s.setdefault(r"$\mu_d + b$", h_mean)
            legend_handles_s.setdefault("Origin", h_org)

    fig_s.suptitle("Extended (BOX) diagnostics for selected bias vectors", fontsize=18, y=0.98)

    labels_order = [
        "Strict Accept", "Ambiguous", "Strict Reject",
        r"$\mu_d + b$", "Origin",
        r"ellipse $\ominus$ box",
        r"ellipse $\oplus$ box",
    ]
    handles = [legend_handles_s[k] for k in labels_order if k in legend_handles_s]
    labels = [k for k in labels_order if k in legend_handles_s]

    fig_s.legend(
        handles, labels,
        loc="upper center",
        ncol=len(handles),  # one row
        frameon=True,
        fontsize=11,
        bbox_to_anchor=(0.5, 0.93),
        columnspacing=1.1,
        handletextpad=0.5,
        borderaxespad=0.2,
    )

    fig_s.subplots_adjust(left=0.09, right=0.99, bottom=0.08, top=0.84, wspace=0.25, hspace=0.25)

    if save_prefix is not None:
        fig_s.savefig(save_prefix.with_suffix("").as_posix() + "_scatter_grid.png", dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig_s)


# =============================
# Main
# =============================
def main(cfg: Config2DExtendedBOX) -> None:
    here = Path(__file__).resolve().parent
    out_dir = here / cfg.out_dirname
    out_dir.mkdir(parents=True, exist_ok=True)

    Sigma = make_covariance_matrix(cfg.sigma1, cfg.sigma2, theta_deg=cfg.theta_deg)
    Delta = np.array(cfg.Delta, dtype=float)

    out_h0 = run_bias_map_extended_box(
        mu_d=np.array(cfg.mu_h0, float),
        Sigma=Sigma,
        Delta=Delta,
        alpha=cfg.alpha,
        n_stoch=cfg.n_stoch,
        n_grid=cfg.n_grid,
        seed=cfg.seed,
        n_jobs=cfg.n_jobs,
        backend=cfg.backend,
        verbose=cfg.verbose,
        save_npz=out_dir / "maps_H0_extended_BOX.npz",
    )
    out_ha = run_bias_map_extended_box(
        mu_d=np.array(cfg.mu_ha, float),
        Sigma=Sigma,
        Delta=Delta,
        alpha=cfg.alpha,
        n_stoch=cfg.n_stoch,
        n_grid=cfg.n_grid,
        seed=cfg.seed,
        n_jobs=cfg.n_jobs,
        backend=cfg.backend,
        verbose=cfg.verbose,
        save_npz=out_dir / "maps_Ha_extended_BOX.npz",
    )

    plot_decision_grid_extended(out_h0, out_ha, save_path=out_dir / "Map_Grid_Extended_BOX.png")

    # Selected-bias diagnostics: two figures (scatter-grid + hist-grid)
    plot_selected_bias_scatter_and_hist_grid_extended(
        b_list=cfg.b_selected,
        mu_h0=np.array(cfg.mu_h0, float),
        mu_ha=np.array(cfg.mu_ha, float),
        Sigma=Sigma,
        Delta=Delta,
        alpha=cfg.alpha,
        n_stoch=cfg.n_stoch,
        seed=cfg.seed,
        unit_scale=100.0,
        unit_label="cm",
        save_prefix=out_dir / "SelectedBias_Extended_BOX",
        show=True,
    )

    print(f"Done. Outputs written to: {out_dir}")


if __name__ == "__main__":
    main(Config2DExtendedBOX())
