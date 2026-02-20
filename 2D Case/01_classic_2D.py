"""
2D classical congruency test simulation (purely stochastic).

Model
-----
    d = mu_d + b + e,     e ~ N(0, Σ)

For a fixed true bias b = (bx,by) (outer grid), we run Monte Carlo over stochastic noise e
(inner loop) and estimate conditional rejection probabilities.

Classical chi-square test (level alpha, df=2):
    T_cls(d) = dᵀ Σ⁻¹ d
    reject ⇔ T_cls(d) > k_alpha
where k_alpha = χ²_{2, 1-α}.

This script produces:
  - 2×2 bias maps: P(Accept|b) and P(Reject|b) under H0 and Ha
  - classical MC vs analytic validation via noncentral χ² (ncχ²)
  - diagnostics for selected biases: scatter + statistic histogram
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import chi2, ncx2
from joblib import Parallel, delayed
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
class Config2DClassic:
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
    backend: str = "threading" # 'loky' is often more robust than 'threading' for numpy work
    verbose: int = 0 # Reduced verbose as tqdm handles progress

    # diagnostics
    b_selected: Tuple[Tuple[float, float], ...] = ((-0.005, 0.005), (0, 0), (0.005, 0.005))

    # output folder (under script folder)
    out_dirname: str = "outputs/classic_2D"


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


def T_classical(d: np.ndarray, Sigma_inv: np.ndarray) -> np.ndarray:
    """Vectorised classical statistic for d with shape (...,2):  dᵀ Σ⁻¹ d."""
    return np.einsum("...i,ij,...j->...", d, Sigma_inv, d)


def classical_reject_prob_ncx2_from_Sinv(mean_d: np.ndarray, Sigma_inv: np.ndarray, k: float) -> float:
    """Analytic classical P(reject) under mean shift mean_d = μ_d + b."""
    mean_d = np.asarray(mean_d, float).reshape(2)
    lam = float(mean_d.T @ Sigma_inv @ mean_d)
    return float(1.0 - ncx2.cdf(k, df=2, nc=lam))


# =============================
# Parallel worker (row-wise)
# =============================
def _worker_row_classical(
    ix: int,
    bxi: float,
    by: np.ndarray,
    noise: np.ndarray,
    mu_d: np.ndarray,
    Sigma_inv: np.ndarray,
    alpha: float,
) -> Tuple[int, Dict[str, np.ndarray]]:
    """
    Process one row (fixed bx=bxi, loop over by).
    Returns ix and vectors of length len(by).
    """
    mu_d = np.asarray(mu_d, float).reshape(2)
    by = np.asarray(by, float)
    k = critical_value(alpha, df=2)

    # analytic ncχ² curve for this row (vectorized in by)
    S = Sigma_inv
    s11, s12, s22 = float(S[0, 0]), float(S[0, 1]), float(S[1, 1])
    mx = float(mu_d[0] + bxi)
    my = mu_d[1] + by
    lam = s11 * mx * mx + 2.0 * s12 * mx * my + s22 * my * my
    p_ncx2_row = (1.0 - ncx2.cdf(k, df=2, nc=lam)).astype(float)

    p_acc = np.zeros(by.size, float)
    p_rej = np.zeros(by.size, float)

    for iy, byj in enumerate(by):
        b_vec = np.array([bxi, byj], dtype=float)
        d = mu_d[None, :] + b_vec[None, :] + noise  # (n_stoch,2)
        T = T_classical(d, Sigma_inv)
        dec_rej = T > k
        p_rej[iy] = float(np.mean(dec_rej))
        p_acc[iy] = 1.0 - p_rej[iy]

    return ix, dict(P_cls_accept=p_acc, P_cls_reject=p_rej, P_cls_reject_ncx2=p_ncx2_row)


# =============================
# Bias maps
# =============================
def run_bias_map_classical(
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
        "P_cls_accept": np.zeros((n_grid, n_grid), float),
        "P_cls_reject": np.zeros((n_grid, n_grid), float),
        "P_cls_reject_ncx2": np.zeros((n_grid, n_grid), float),
    }

    tasks = [
        delayed(_worker_row_classical)(ix, float(bx[ix]), by, noise, mu_d, Sigma_inv, alpha)
        for ix in range(n_grid)
    ]

    # Header log
    print(f"[run_bias_map_classical] n_grid={n_grid}, n_stoch={n_stoch}, n_jobs={n_jobs}")

    # Execute with tqdm progress bar
    results = Parallel(n_jobs=n_jobs, backend=backend, verbose=verbose)(
        tqdm(tasks, total=n_grid, desc=f"Mapping (mu_d={mu_d})")
    )

    for ix, row in results:
        out["P_cls_accept"][:, ix] = row["P_cls_accept"]
        out["P_cls_reject"][:, ix] = row["P_cls_reject"]
        out["P_cls_reject_ncx2"][:, ix] = row["P_cls_reject_ncx2"]

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


def plot_decision_grid_classical(out_h0: Dict[str, np.ndarray], out_ha: Dict[str, np.ndarray], save_path: Optional[Path] = None) -> None:
    bx, by, extent = _extent_mm(out_h0)

    data_grid = [
        [out_h0["P_cls_accept"], out_h0["P_cls_reject"]],
        [out_ha["P_cls_accept"], out_ha["P_cls_reject"]],
    ]

    col_labels = [r"$P(\mathrm{Accept}\mid \mathbf{b})$", r"$P(\mathrm{Reject}\mid \mathbf{b})$"]
    row_labels = [r"$H_0:\ \mu_d=\mathbf{0}$", r"$H_a:\ \mu_d\neq \mathbf{0}$"]

    fig, axs = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
    cmap = "RdBu"

    for r in range(2):
        for c in range(2):
            ax = axs[r, c]
            Z = data_grid[r][c]

            im = ax.imshow(Z, origin="lower", extent=extent, cmap=cmap, interpolation="bilinear")
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cb.ax.tick_params(labelsize=16)

            if np.nanmax(Z) > np.nanmin(Z):
                cs = ax.contour(Z, origin="lower", extent=extent, colors="black", linewidths=0.8, alpha=0.6, levels=6)
                ax.clabel(cs, inline=True, fontsize=10, fmt="%.2f")

            if r == 1:
                ax.set_xlabel(r"$b_x$ [mm]", fontsize=16)
            if c == 0:
                ax.set_ylabel(r"$b_y$ [mm]", fontsize=16)
                ax.annotate(row_labels[r], xy=(-0.45, 0.5), xycoords="axes fraction",
                            fontsize=16, va="center", ha="center", rotation=90, weight="bold")
            if r == 0:
                ax.set_title(col_labels[c], fontsize=16, weight="bold")

            ax.tick_params(axis='both', labelsize=16)

    fig.suptitle("Classical test decision probabilities", fontsize=16)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_classical_validation_mc_vs_ncx2(out_h0: Dict[str, np.ndarray], out_ha: Dict[str, np.ndarray], save_path: Optional[Path] = None) -> None:
    """MC-analytic error maps for the classical test (H0 and Ha)."""
    bx_mm, by_mm, extent = _extent_mm(out_h0)

    Sigma = out_h0["Sigma"]
    alpha = float(out_h0["alpha"][0])
    bx = out_h0["bx"]
    by = out_h0["by"]

    # Analytic grids
    k = critical_value(alpha, df=2)
    S = np.linalg.inv(Sigma)
    s11, s12, s22 = float(S[0, 0]), float(S[0, 1]), float(S[1, 1])

    Bx, By = np.meshgrid(bx, by, indexing="xy")
    def _an(mu_d: np.ndarray) -> np.ndarray:
        mu_d = np.asarray(mu_d, float).reshape(2)
        mx = mu_d[0] + Bx
        my = mu_d[1] + By
        lam = s11 * mx * mx + 2.0 * s12 * mx * my + s22 * my * my
        return (1.0 - ncx2.cdf(k, df=2, nc=lam)).astype(float)

    an_h0 = _an(out_h0["mu_d"])
    an_ha = _an(out_ha["mu_d"])

    err_h0 = out_h0["P_cls_reject"] - an_h0
    err_ha = out_ha["P_cls_reject"] - an_ha

    fig, axs = plt.subplots(1, 2, figsize=(13, 5), sharex=True, sharey=True)
    cmap = "viridis"

    for ax, Z, title in [
        (axs[0], err_h0, rf"H0: MC-analytic  (max |err|={float(np.max(np.abs(err_h0))):.3f})"),
        (axs[1], err_ha, rf"Ha: MC-analytic  (max |err|={float(np.max(np.abs(err_ha))):.3f})"),
    ]:
        im = ax.imshow(Z, origin="lower", extent=extent, cmap=cmap, interpolation="nearest")
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=8)
        ax.set_title(title, fontsize=12, weight="bold")
        ax.set_xlabel(r"$b_x$ [mm]")

    axs[0].set_ylabel(r"$b_y$ [mm]")
    fig.suptitle("Classical test validation: Monte Carlo vs Analytical", fontsize=15)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_selected_bias_scatter_and_hist_grid(
    *,
    b_list: Tuple[Tuple[float, float], ...],
    mu_d: np.ndarray,
    Sigma: np.ndarray,
    alpha: float,
    n_stoch: int,
    seed: int,
    bins_n: int = 45,
    unit_scale: float = 100.0,
    unit_label: str = "cm",
    save_prefix: Optional[Path] = None,
    show: bool = True,
    # styling
    ellipse_color: str = "black",
    ellipse_lw: float = 2.0,
    # Pie inset
    pie_loc: str = "lower right",
    pie_width: str = "42%",
    pie_height: str = "42%",
    pie_borderpad: float = 0.1,
) -> None:
    """
    Produce TWO figures (not stacked):
      (A) 2×N histograms of T_cls with k_alpha and chi2(H0 ref)
            - row 1: H0 (mu_d = 0)
            - row 2: Ha (mu_d = mu_d input)
      (B) 2×N scatter plots with classical acceptance ellipse + pie inset
            - row 1: H0 (mu_d = 0)
            - row 2: Ha (mu_d = mu_d input)

    Style harmonized with the extended (BOX) diagnostics:
      - Same figsize policy
      - Figure-level legend at top-center
      - Fixed scatter axes limits (in cm): x in [-10,10], y in [-8,8]
      - No tight_layout / constrained_layout to avoid inset-axes warnings
    """

    def _max_density(values: np.ndarray, bins: np.ndarray) -> float:
        h, _ = np.histogram(values, bins=bins, density=True)
        return float(np.max(h)) if h.size else 0.0

    rng = np.random.default_rng(seed)
    mu_ha = np.asarray(mu_d, float).reshape(2)
    mu_h0 = np.zeros(2, dtype=float)

    Sigma = np.asarray(Sigma, float).reshape(2, 2)
    Sigma_inv = np.linalg.inv(Sigma)
    k = critical_value(alpha, df=2)

    n = len(b_list)

    # Common random numbers for all panels
    noise = rng.multivariate_normal(mean=np.zeros(2), cov=Sigma, size=n_stoch)

    # Acceptance ellipse border: x(t)=sqrt(k)*L*[cos t, sin t]
    L = np.linalg.cholesky(Sigma)
    t = np.linspace(0.0, 2.0 * np.pi, 600)
    unit_circle = np.stack([np.cos(t), np.sin(t)], axis=0)
    ellipse = (np.sqrt(k) * (L @ unit_circle)).T  # (M,2)

    # Fixed scatter axes limits (cm)
    xlim_cm = (-10.0, 10.0)
    ylim_cm = (-8.0, 8.0)
    xlim = (xlim_cm[0], xlim_cm[1])
    ylim = (ylim_cm[0], ylim_cm[1])

    # -------------------------
    # Precompute T for both hypotheses (consistent histogram scaling)
    # -------------------------
    T_vals = [[None for _ in range(n)] for _ in range(2)]
    T_max_global = 0.0

    for col, b in enumerate(b_list):
        b_vec = np.asarray(b, float).reshape(2)
        for row, mu_row in enumerate([mu_h0, mu_ha]):
            d = mu_row[None, :] + b_vec[None, :] + noise
            T = T_classical(d, Sigma_inv)
            T_vals[row][col] = T
            T_max_global = max(T_max_global, float(np.max(T)))

    # x-range (kept as you wrote)
    x_hi = max(T_max_global, 2.0 * k)
    bins = np.linspace(0.0, x_hi, bins_n)

    # y-range (shared across ALL panels in this figure)
    y_hi_hist = 0.0
    for row in range(2):
        for col in range(n):
            y_hi_hist = max(y_hi_hist, _max_density(T_vals[row][col], bins))

    x_smooth = np.linspace(0.0, x_hi, 700)
    y_chi2 = chi2.pdf(x_smooth, df=2)
    y_hi = 1.05 * max(y_hi_hist, float(np.max(y_chi2)))

    # ============================================================
    # (A) HISTOGRAM FIGURE (2×N)
    # ============================================================
    fig_h, axs_h = plt.subplots(2, n, figsize=(4.2 * n, 7.2), constrained_layout=False)
    if n == 1:
        axs_h = np.array([[axs_h[0]], [axs_h[1]]])

    fig_h.text(0.012, 0.63, r"$H_0:\ \mu_d=\mathbf{0}$", rotation=90,
               va="center", ha="left", fontsize=14, weight="bold")
    fig_h.text(0.012, 0.23, r"$H_a:\ \mu_d\neq \mathbf{0}$", rotation=90,
               va="center", ha="left", fontsize=14, weight="bold")

    legend_handles_h = {}

    for row in range(2):
        for col, b in enumerate(b_list):
            ax = axs_h[row, col]
            b_vec = np.asarray(b, float).reshape(2)
            T = T_vals[row][col]

            h_mc = ax.hist(T, bins=bins, density=True, color="0.7", alpha=0.85, label="Monte Carlo")

            h_ref, = ax.plot(x_smooth, y_chi2, color="black", linestyle="--", linewidth=2.0,
                             alpha=0.9, label=r"$\chi^2_2$ (H0 ref)")
            h_k = ax.axvline(k, color="red", linewidth=2, label=r"$k_\alpha$")

            ax.set_xlim(0.0, x_hi)
            ax.set_ylim(0.0, y_hi)
            ax.grid(True, alpha=0.25)
            ax.tick_params(axis="both", labelsize=12)

            if row == 1:
                ax.set_xlabel(r"$T_{\mathrm{cls}}$", fontsize=13)
            if col == 0:
                ax.set_ylabel("Density", fontsize=13)

            hyp = r"$H_0$" if row == 0 else r"$H_a$"
            ax.set_title(
                rf"{hyp}: $b=({b_vec[0]*unit_scale:.1f},{b_vec[1]*unit_scale:.1f})$ {unit_label}",
                fontsize=13,
            )

            legend_handles_h.setdefault("Monte Carlo", h_mc[2][0] if len(h_mc) > 2 else None)
            legend_handles_h.setdefault(r"$\chi^2_2$ (H0 ref)", h_ref)
            legend_handles_h.setdefault(r"$k_\alpha$", h_k)

    fig_h.suptitle("Classical statistic histograms", fontsize=18, y=0.98)

    leg_order = ["Monte Carlo", r"$\chi^2_2$ (H0 ref)", r"$k_\alpha$"]
    handles, labels = [], []
    for lab in leg_order:
        h = legend_handles_h.get(lab, None)
        if h is not None:
            handles.append(h); labels.append(lab)
    if handles:
        fig_h.legend(handles, labels, loc="upper center", ncol=3, frameon=True, fontsize=11,
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

    fig_s.text(0.012, 0.63, r"$H_0:\ \mu_d=\mathbf{0}$", rotation=90,
               va="center", ha="left", fontsize=14, weight="bold")
    fig_s.text(0.012, 0.23, r"$H_a:\ \mu_d\neq \mathbf{0}$", rotation=90,
               va="center", ha="left", fontsize=14, weight="bold")

    legend_handles_s = {}

    for row, mu_row in enumerate([mu_h0, mu_ha]):
        for col, b in enumerate(b_list):
            ax = axs_s[row, col]
            b_vec = np.asarray(b, float).reshape(2)
            mean_vec = mu_row + b_vec

            d = mu_row[None, :] + b_vec[None, :] + noise
            T = T_classical(d, Sigma_inv)
            acc = (T <= k)
            rej = ~acc

            h_ell, = ax.plot(ellipse[:, 0] * unit_scale, ellipse[:, 1] * unit_scale,
                             color=ellipse_color, linewidth=ellipse_lw)

            h_acc = ax.scatter(d[acc, 0] * unit_scale, d[acc, 1] * unit_scale, s=10, c="lightgreen")
            h_rej = ax.scatter(d[rej, 0] * unit_scale, d[rej, 1] * unit_scale, s=10, c="lightcoral")
            h_mean = ax.scatter(mean_vec[0] * unit_scale, mean_vec[1] * unit_scale, c="blue", marker="o", s=50)
            h_org = ax.scatter(0.0, 0.0, c="black", marker="+", s=60)

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

            ax.set_xlim(xlim[0], xlim[1])
            ax.set_ylim(ylim[0], ylim[1])

            ax_in = inset_axes(ax, width=pie_width, height=pie_height, loc=pie_loc, borderpad=pie_borderpad)
            ax_in.pie(
                (100.0 * float(np.mean(acc)), 100.0 * float(np.mean(rej))),
                colors=("lightgreen", "lightcoral"),
                autopct="%1.0f%%",
                startangle=0,
                textprops=dict(fontsize=10),
            )
            ax_in.set_aspect("equal")
            ax_in.set_xticks([])
            ax_in.set_yticks([])

            legend_handles_s.setdefault("Accept", h_acc)
            legend_handles_s.setdefault("Reject", h_rej)
            legend_handles_s.setdefault(r"$\mu_d + b$", h_mean)
            legend_handles_s.setdefault("Origin", h_org)
            legend_handles_s.setdefault("Ellipse", h_ell)

    fig_s.suptitle("Classical diagnostics for selected bias vectors", fontsize=18, y=0.98)

    leg_order = ["Accept", "Reject", r"$\mu_d + b$", "Origin", "Ellipse"]
    handles, labels = [], []
    for lab in leg_order:
        h = legend_handles_s.get(lab, None)
        if h is not None:
            handles.append(h); labels.append(lab)
    if handles:
        fig_s.legend(handles, labels, loc="upper center", ncol=5, frameon=True, fontsize=11,
                     bbox_to_anchor=(0.5, 0.93))

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
def main(cfg: Config2DClassic) -> None:
    here = Path(__file__).resolve().parent
    out_dir = here / cfg.out_dirname
    out_dir.mkdir(parents=True, exist_ok=True)

    Sigma = make_covariance_matrix(cfg.sigma1, cfg.sigma2, theta_deg=cfg.theta_deg)
    Delta = np.array(cfg.Delta, dtype=float)

    # H0 Map
    out_h0 = run_bias_map_classical(
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
        save_npz=out_dir / "maps_H0_classical.npz",
    )

    # Ha Map
    out_ha = run_bias_map_classical(
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
        save_npz=out_dir / "maps_Ha_classical.npz",
    )

    plot_decision_grid_classical(out_h0, out_ha, save_path=out_dir / "Map_Grid_Classical.png")
    plot_classical_validation_mc_vs_ncx2(out_h0, out_ha, save_path=out_dir / "Validation_Classical_MC_vs_Analytical.png")

    plot_selected_bias_scatter_and_hist_grid(
        b_list=cfg.b_selected,
        mu_d=np.array(cfg.mu_ha, float),
        Sigma=Sigma,
        alpha=cfg.alpha,
        n_stoch=cfg.n_stoch,
        seed=cfg.seed,
        unit_scale=100.0,
        unit_label="cm",
        save_prefix=out_dir / "SelectedBias_Classical",
    )

    print(f"Done. Outputs written to: {out_dir}")


if __name__ == "__main__":
    main(Config2DClassic())