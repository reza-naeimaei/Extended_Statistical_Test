"""
2D interval-extended congruency test simulation (ZONOTOPE-bias; stochastic + bounded systematics).

Model
-----
    d = mu_d + b + e,     e ~ N(0, Σ),   b ∈ Z(G)

Zonotope bias set
-----------------
    Z(G) = { G ξ : ξ ∈ [-1,1]^m }
where G ∈ R^{2×m} is the generator matrix (each column is one generator vector).

For a fixed true bias b = (bx,by) (outer grid), we run Monte Carlo over stochastic noise e
(inner loop) and estimate conditional decision probabilities.

Interval-extended test statistic (zonotope bias, level alpha, df=2):
    T(d,b)   = (d - b)ᵀ Σ⁻¹ (d - b)
    Tmin(d)  = min_{b ∈ Z(G)} T(d,b)
    Tmax(d)  = max_{b ∈ Z(G)} T(d,b)

Decision (kα = χ²_{2,1-α}):
    strict accept  ⇔ Tmax(d) ≤ kα          (robustly inside)
    strict reject  ⇔ Tmin(d)  > kα          (robustly outside)
    ambiguous      otherwise

This script produces:
  - 2×3 bias maps (H0 / Ha): P(strict accept | b), P(ambiguous | b), P(strict reject | b)
  - diagnostics for selected biases: (A) scatter with Minkowski sum/diff contours, (B) histograms of Tmin/Tmax

Notes
-----
- Tmin/Tmax are computed over the zonotope bias set Z(G) by working in whitened space (Σ = L Lᵀ):
    u = L^{-1} d,    U = L^{-1} Z(G)
  Tmin(d) is the squared distance from u to the convex polygon U (0 if inside).
  Tmax(d) is the maximum squared distance from u to the vertices of U.
- Uses common random numbers (same noise) across grid points.
"""

from __future__ import annotations

import os
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Sequence

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import chi2
from scipy.spatial import ConvexHull

from joblib import Parallel, delayed
from tqdm.auto import tqdm
from contextlib import contextmanager
import joblib

from mpl_toolkits.axes_grid1.inset_locator import inset_axes


# =============================
# Performance / thread control
# =============================
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


@contextmanager
def tqdm_joblib(tqdm_object: tqdm):
    """Context manager to patch joblib to report into tqdm progress bar."""

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            try:
                tqdm_object.update(n=self.batch_size)
            except Exception:
                pass
            return super().__call__(*args, **kwargs)

    old_cb = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_cb
        tqdm_object.close()


# =============================
# Configuration (edit here)
# =============================
@dataclass(frozen=True)
class Config2DExtendedZonotope:
    # covariance principal std devs (meters) and rotation (deg)
    sigma1: float = 0.02
    sigma2: float = 0.01
    theta_deg: float = 0.0

    # hypothesis settings
    mu_h0: Tuple[float, float] = (0.0, 0.0)
    mu_ha: Tuple[float, float] = (0.02, 0.01)

    # Zonotope generators (meters): list of 2D vectors g_i
    # Z(G) = { sum_i g_i * ζ_i , ζ_i ∈ [-1,1] }
    generators: Tuple[Tuple[float, float], ...] = (
        (0.005, 0.0),
        (0.0, 0.005),
        (0.005, 0.005),
    )

    # bias map grid resolution (over bounding box of the zonotope)
    n_grid: int = 101

    # Monte Carlo
    n_stoch: int = 10_000
    seed: int = 0
    alpha: float = 0.05

    # parallelization
    n_jobs: int = -1
    backend: str = "threading"  # keep consistent with your BOX script

    # diagnostics
    b_selected: Tuple[Tuple[float, float], ...] = ((-0.005, 0.005), (0.0, 0.0), (0.005, 0.005))

    # output folder (under script folder)
    out_dirname: str = "outputs/extended_zonotope_2D"


# =============================
# Utilities
# =============================
def rotation_matrix(theta_deg: float) -> np.ndarray:
    th = np.deg2rad(theta_deg)
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s], [s, c]], dtype=float)


def make_covariance_matrix(sigma1: float, sigma2: float, *, theta_deg: float = 0.0) -> np.ndarray:
    """Build Σ = R diag(sigma1^2, sigma2^2) R^T."""
    R = rotation_matrix(theta_deg)
    S = np.diag([sigma1**2, sigma2**2]).astype(float)
    Sigma = R @ S @ R.T
    return 0.5 * (Sigma + Sigma.T)


def critical_value(alpha: float, df: int = 2) -> float:
    return float(chi2.ppf(1.0 - alpha, df=df))


def cholesky_lower(Sigma: np.ndarray) -> np.ndarray:
    return np.linalg.cholesky(np.asarray(Sigma, dtype=float))


def solve_lower(L: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Solve L Y = X for Y. Supports X shape (2,) or (n,2)."""
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        return np.linalg.solve(L, X)
    if X.ndim == 2 and X.shape[1] == 2:
        return np.linalg.solve(L, X.T).T
    raise ValueError("Unsupported shape for solve_lower")


# =============================
# Zonotope geometry
# =============================
def get_zonotope_generators(gen_list: Sequence[Sequence[float]]) -> np.ndarray:
    """Convert list of 2D generator vectors into generator matrix G ∈ R^{2×m}."""
    G = np.asarray(gen_list, dtype=float)
    if G.ndim != 2 or G.shape[1] != 2:
        raise ValueError("generators must be a list/tuple of 2D vectors")
    return G.T  # (m,2) -> (2,m)


def zonotope_vertices(G: np.ndarray) -> np.ndarray:
    """All vertices of zonotope {G ζ : ζ_i ∈ {-1,+1}}. Returns (2^m,2)."""
    G = np.asarray(G, dtype=float)
    m = G.shape[1]
    alphas = np.array(list(itertools.product([-1.0, 1.0], repeat=m)), dtype=float)  # (2^m, m)
    V = (G @ alphas.T).T
    return V


def convex_hull_polygon(points: np.ndarray) -> np.ndarray:
    """Return closed polygon (n+1,2) of convex hull."""
    pts = np.asarray(points, dtype=float)
    if pts.shape[0] < 3:
        poly = pts.copy()
        if poly.shape[0] > 0:
            poly = np.vstack([poly, poly[0]])
        return poly

    hull = ConvexHull(pts)
    poly = pts[hull.vertices]
    return np.vstack([poly, poly[0]])


def points_in_convex_polygon(points: np.ndarray, polygon: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Vectorised point-in-convex-polygon test. polygon is (M,2) without repeated last point."""
    P = np.asarray(points, dtype=float)
    V = np.asarray(polygon, dtype=float)
    if len(V) < 3:
        return np.zeros((len(P),), dtype=bool)

    A = V
    B = np.roll(V, -1, axis=0)
    E = B - A
    PA = P[:, None, :] - A[None, :, :]
    cross = E[None, :, 0] * PA[:, :, 1] - E[None, :, 1] * PA[:, :, 0]
    return np.all(cross >= -eps, axis=1) | np.all(cross <= eps, axis=1)


def _dist2_points_to_segment(P: np.ndarray, A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Squared distance from each point in P (n,2) to segment AB."""
    AB = B - A
    denom = float(AB @ AB)
    if denom <= 0.0:
        D = P - A[None, :]
        return np.einsum("ij,ij->i", D, D)

    AP = P - A[None, :]
    t = (AP @ AB) / denom
    t = np.clip(t, 0.0, 1.0)
    Q = A[None, :] + t[:, None] * AB[None, :]
    D = P - Q
    return np.einsum("ij,ij->i", D, D)


def _min_dist2_to_polygon_edges(P: np.ndarray, seg_a: np.ndarray, seg_b: np.ndarray) -> np.ndarray:
    """Minimum squared distance from points to polygon edges."""
    d2_min = np.full(P.shape[0], np.inf, dtype=float)
    for A, B in zip(seg_a, seg_b):
        d2 = _dist2_points_to_segment(P, A, B)
        d2_min = np.minimum(d2_min, d2)
    return d2_min


@dataclass(frozen=True)
class ZonotopeGeom:
    polygon_b: np.ndarray      # closed (n+1,2) in b-space
    vertices_u: np.ndarray     # (Nv,2) in u-space
    polygon_u: np.ndarray      # closed polygon in u-space
    seg_a_u: np.ndarray        # (Ns,2) edge start points in u-space
    seg_b_u: np.ndarray        # (Ns,2) edge end points in u-space


def precompute_zonotope_geometry(G: np.ndarray, Sigma: np.ndarray) -> Tuple[np.ndarray, ZonotopeGeom]:
    """Precompute zonotope polygon (b-space) and polygon/vertices/segments in whitened u-space."""
    L = cholesky_lower(Sigma)

    Vb = zonotope_vertices(G)
    Pb = convex_hull_polygon(Vb)

    Vu = solve_lower(L, Vb)
    Pu = convex_hull_polygon(Vu)

    seg_a = Pu[:-1]
    seg_b = Pu[1:]

    geom = ZonotopeGeom(
        polygon_b=Pb,
        vertices_u=Vu,
        polygon_u=Pu,
        seg_a_u=seg_a,
        seg_b_u=seg_b,
    )
    return L, geom


# =============================
# Interval statistics + decisions
# =============================
def interval_stats_zonotope_points(d: np.ndarray, L: np.ndarray, geom: ZonotopeGeom) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (Tmin, Tmax) for samples d in original space.

    In whitened space u = L^{-1} d:
      Tmin = 0 if u ∈ U, else squared distance to boundary of U
      Tmax = max_v ||u - v||^2 over zonotope vertices v in U
    """
    u = solve_lower(L, d)  # (n,2)

    # Tmax: farthest vertex
    diffs = u[:, None, :] - geom.vertices_u[None, :, :]
    d2 = np.einsum("nij,nij->ni", diffs, diffs)
    Tmax = np.max(d2, axis=1)

    # Tmin: distance to polygon (0 if inside)
    inside = points_in_convex_polygon(u, geom.polygon_u[:-1], eps=1e-12)
    Tmin = np.zeros(u.shape[0], dtype=float)
    if np.any(~inside):
        Tmin[~inside] = _min_dist2_to_polygon_edges(u[~inside], geom.seg_a_u, geom.seg_b_u)

    return Tmin, Tmax


def decide_extended(Tmin: np.ndarray, Tmax: np.ndarray, k: float) -> np.ndarray:
    """0 strict accept, 1 ambiguous, 2 strict reject."""
    Tmin = np.asarray(Tmin, float)
    Tmax = np.asarray(Tmax, float)
    out = np.empty_like(Tmin, dtype=int)
    out[Tmax <= k] = 0
    out[(Tmin <= k) & (k < Tmax)] = 1
    out[Tmin > k] = 2
    return out


# =============================
# Bias map (parallel row worker)
# =============================
def _worker_row_zonotope(
    ix: int,
    bxi: float,
    by: np.ndarray,
    inside_row: np.ndarray,
    noise: np.ndarray,
    mu_d: np.ndarray,
    L: np.ndarray,
    geom: ZonotopeGeom,
    k: float,
) -> Tuple[int, Dict[str, np.ndarray]]:
    """
    Process one row (fixed bx=bxi, loop over by) for b inside the zonotope.
    Returns ix and vectors of length len(by) (NaN outside the zonotope).
    """
    by = np.asarray(by, float)
    out_sa = np.full(by.size, np.nan, float)
    out_am = np.full(by.size, np.nan, float)
    out_sr = np.full(by.size, np.nan, float)

    for iy, byj in enumerate(by):
        if not bool(inside_row[iy]):
            continue

        b_vec = np.array([bxi, byj], dtype=float)
        d = mu_d[None, :] + b_vec[None, :] + noise

        Tmin, Tmax = interval_stats_zonotope_points(d, L, geom)
        dec = decide_extended(Tmin, Tmax, k)

        out_sa[iy] = float(np.mean(dec == 0))
        out_am[iy] = float(np.mean(dec == 1))
        out_sr[iy] = float(np.mean(dec == 2))

    return ix, dict(P_ext_strict_accept=out_sa, P_ext_ambiguous=out_am, P_ext_strict_reject=out_sr)


def run_bias_map_extended_zonotope(
    *,
    mu_d: np.ndarray,
    Sigma: np.ndarray,
    G: np.ndarray,
    alpha: float,
    n_stoch: int,
    n_grid: int,
    seed: int,
    n_jobs: int,
    backend: str,
    save_npz: Optional[Path] = None,
) -> Dict[str, np.ndarray]:
    """
    Bias map over the bounding box of the zonotope, masked to b ∈ Z(G).
    Outputs use NaN outside Z(G) so imshow shows blank there.
    """
    rng = np.random.default_rng(seed)
    mu_d = np.asarray(mu_d, float).reshape(2)
    Sigma = np.asarray(Sigma, float).reshape(2, 2)

    L, geom = precompute_zonotope_geometry(G, Sigma)
    Pb = geom.polygon_b

    bx = np.linspace(float(np.min(Pb[:, 0])), float(np.max(Pb[:, 0])), n_grid)
    by = np.linspace(float(np.min(Pb[:, 1])), float(np.max(Pb[:, 1])), n_grid)

    # inside mask in b-space
    BBx, BBy = np.meshgrid(bx, by, indexing="xy")
    b_grid = np.stack([BBx.ravel(), BBy.ravel()], axis=1)
    inside = points_in_convex_polygon(b_grid, Pb[:-1], eps=1e-12).reshape(n_grid, n_grid)

    # common random numbers across grid points
    noise = rng.multivariate_normal(mean=np.zeros(2), cov=Sigma, size=n_stoch)
    k = critical_value(alpha, df=2)

    out = {
        "P_ext_strict_accept": np.full((n_grid, n_grid), np.nan, float),
        "P_ext_ambiguous": np.full((n_grid, n_grid), np.nan, float),
        "P_ext_strict_reject": np.full((n_grid, n_grid), np.nan, float),
    }

    tasks = [
        delayed(_worker_row_zonotope)(
            ix,
            float(bx[ix]),
            by,
            inside[:, ix],
            noise,
            mu_d,
            L,
            geom,
            k,
        )
        for ix in range(n_grid)
    ]

    desc = f"Bias map rows (n_grid={n_grid}, n_stoch={n_stoch})"
    with tqdm_joblib(tqdm(total=n_grid, desc=desc, unit="row")):
        results = Parallel(n_jobs=n_jobs, backend=backend, verbose=0)(tasks)

    for ix, row in results:
        out["P_ext_strict_accept"][:, ix] = row["P_ext_strict_accept"]
        out["P_ext_ambiguous"][:, ix] = row["P_ext_ambiguous"]
        out["P_ext_strict_reject"][:, ix] = row["P_ext_strict_reject"]

    pack = dict(
        bx=bx,
        by=by,
        inside_b=inside,
        polygon_b=Pb,
        mu_d=mu_d,
        Sigma=Sigma,
        G=G,
        alpha=np.array([alpha], float),
        n_stoch=np.array([n_stoch], int),
        seed=np.array([seed], int),
        **out,
    )

    if save_npz is not None:
        np.savez(save_npz, **pack)

    return pack


# =============================
# Plotting: bias maps (same style as BOX version)
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

            # outline zonotope support region
            poly = out_h0.get("polygon_b", None)
            if poly is not None:
                ax.plot(poly[:, 0] * 1000.0, poly[:, 1] * 1000.0, color="black", lw=1.2)

            if r == 1:
                ax.set_xlabel(r"$b_x$ [mm]", fontsize=14)
            if c == 0:
                ax.set_ylabel(r"$b_y$ [mm]", fontsize=14)
                ax.annotate(row_labels[r], xy=(-0.40, 0.5), xycoords="axes fraction",
                            fontsize=14, va="center", ha="center", rotation=90, weight="bold")
            if r == 0:
                ax.set_title(col_labels[c], fontsize=14, weight="bold")

            ax.tick_params(axis="both", labelsize=14)

    fig.suptitle("Extended (Zonotope) test decision probabilities", fontsize=16)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


# =============================
# Diagnostics: selected-bias scatter + hist (match BOX outputs)
# =============================
def plot_selected_bias_scatter_and_hist_grid_extended(
    *,
    b_list: Tuple[Tuple[float, float], ...],
    mu_h0: np.ndarray,
    mu_ha: np.ndarray,
    Sigma: np.ndarray,
    G: np.ndarray,
    alpha: float,
    n_stoch: int,
    seed: int,
    unit_scale: float = 100.0,
    unit_label: str = "cm",
    save_prefix: Optional[Path] = None,
    show: bool = True,
    contour_grid: int = 240,
    sum_color: str = "blue",
    sum_lw: float = 2.0,
    sum_ls: str = "-",
    diff_color: str = "green",
    diff_lw: float = 2.0,
    diff_ls: str = "--",
    pie_loc: str = "lower right",
    pie_width: str = "42%",
    pie_height: str = "42%",
    pie_borderpad: float = 0.1,
) -> None:
    """
    Produce TWO figures (not stacked), matching your BOX figure style:
      (A) 2×N histograms of Tmin/Tmax with chi2(H0 ref) and kα
      (B) 2×N scatter with 3 decision colors + contours at kα:
            blue  : ellipse ⊖ zonotope  (Tmin = kα)
            green : ellipse ⊕ zonotope  (Tmax = kα)
    """
    def _max_density(values: np.ndarray, bins: np.ndarray) -> float:
        h, _ = np.histogram(values, bins=bins, density=True)
        return float(np.max(h)) if h.size else 0.0

    rng = np.random.default_rng(seed)
    mu_h0 = np.asarray(mu_h0, float).reshape(2)
    mu_ha = np.asarray(mu_ha, float).reshape(2)
    Sigma = np.asarray(Sigma, float).reshape(2, 2)
    G = np.asarray(G, float)

    L, geom = precompute_zonotope_geometry(G, Sigma)
    k = critical_value(alpha, df=2)

    n = len(b_list)
    noise = rng.multivariate_normal(mean=np.zeros(2), cov=Sigma, size=n_stoch)

    # fixed scatter limits (cm) identical to your classic/BOX scripts
    xlim_cm = (-10.0, 10.0)
    ylim_cm = (-8.0, 8.0)
    xlim_m = (xlim_cm[0] / unit_scale, xlim_cm[1] / unit_scale)
    ylim_m = (ylim_cm[0] / unit_scale, ylim_cm[1] / unit_scale)

    # precompute Tmin/Tmax for histogram scaling
    Tmin_vals = [[None for _ in range(n)] for _ in range(2)]
    Tmax_vals = [[None for _ in range(n)] for _ in range(2)]
    Tmax_global = 0.0

    for col, b in enumerate(b_list):
        b_vec = np.asarray(b, float).reshape(2)
        for row, mu_row in enumerate([mu_h0, mu_ha]):
            d = mu_row[None, :] + b_vec[None, :] + noise
            Tmin, Tmax = interval_stats_zonotope_points(d, L, geom)
            Tmin_vals[row][col] = Tmin
            Tmax_vals[row][col] = Tmax
            Tmax_global = max(Tmax_global, float(np.max(Tmax)))

    # histogram x/y limits
    x_hi = max(Tmax_global, 2.0 * k) * 1.05
    bins = np.linspace(0.0, x_hi, 50)
    y_hi_hist = 0.0
    for row in range(2):
        for col in range(n):
            y_hi_hist = max(y_hi_hist, _max_density(Tmin_vals[row][col], bins))
            y_hi_hist = max(y_hi_hist, _max_density(Tmax_vals[row][col], bins))
    x_smooth = np.linspace(0.0, x_hi, 700)
    y_chi2 = chi2.pdf(x_smooth, df=2)
    y_hi = 1.05 * max(y_hi_hist, float(np.max(y_chi2)))

    # ------------------------------------------------------------
    # (A) Histogram grid
    # ------------------------------------------------------------
    fig_h, axs_h = plt.subplots(2, n, figsize=(4.2 * n, 7.2), constrained_layout=False)
    if n == 1:
        axs_h = np.array([[axs_h[0]], [axs_h[1]]])

    fig_h.text(0.012, 0.63, r"$H_0:\ \mu_d=\mathbf{0}$", rotation=90,
               va="center", ha="left", fontsize=14, weight="bold")
    fig_h.text(0.012, 0.23, r"$H_a:\ \mu_d\neq \mathbf{0}$", rotation=90,
               va="center", ha="left", fontsize=14, weight="bold")

    legend_handles_h: Dict[str, object] = {}

    for row in range(2):
        for col, b in enumerate(b_list):
            ax = axs_h[row, col]
            b_vec = np.asarray(b, float).reshape(2)

            Tmin = Tmin_vals[row][col]
            Tmax = Tmax_vals[row][col]

            h_min = ax.hist(Tmin, bins=bins, density=True, color="#1f77b4", alpha=0.35, label=r"$T_{\min}$")
            h_max = ax.hist(Tmax, bins=bins, density=True, color="#ff7f0e", alpha=0.35, label=r"$T_{\max}$")
            h_ref, = ax.plot(x_smooth, y_chi2, color="black", linestyle="--", linewidth=2.0,
                             alpha=0.9, label=r"$\chi^2_2$ (H0 ref)")
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

    fig_h.suptitle("Extended (Zonotope) statistic histograms", fontsize=18, y=0.98)
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

    # ------------------------------------------------------------
    # (B) Scatter grid with contours (computed ONCE)
    # ------------------------------------------------------------
    # Precompute contour grids around origin
    gx = np.linspace(xlim_m[0], xlim_m[1], contour_grid)
    gy = np.linspace(ylim_m[0], ylim_m[1], contour_grid)
    XX, YY = np.meshgrid(gx, gy)
    pts = np.stack([XX.ravel(), YY.ravel()], axis=1)
    Tmin_g, Tmax_g = interval_stats_zonotope_points(pts, L, geom)
    Tmin_grid = Tmin_g.reshape(XX.shape)
    Tmax_grid = Tmax_g.reshape(XX.shape)

    fig_s, axs_s = plt.subplots(2, n, figsize=(4.2 * n, 8.0), constrained_layout=False)
    if n == 1:
        axs_s = np.array([[axs_s[0]], [axs_s[1]]])

    fig_s.text(0.012, 0.63, r"$H_0:\ \mu_d=\mathbf{0}$", rotation=90,
               va="center", ha="left", fontsize=14, weight="bold")
    fig_s.text(0.012, 0.23, r"$H_a:\ \mu_d\neq \mathbf{0}$", rotation=90,
               va="center", ha="left", fontsize=14, weight="bold")

    proxy_minus = Line2D([0], [0], color=sum_color, lw=sum_lw, ls=sum_ls, label=r"ellipse $\ominus$ zonotope")
    proxy_plus = Line2D([0], [0], color=diff_color, lw=diff_lw, ls=diff_ls, label=r"ellipse $\oplus$ zonotope")

    legend_handles_s: Dict[str, object] = {
        r"ellipse $\ominus$ zonotope": proxy_minus,
        r"ellipse $\oplus$ zonotope": proxy_plus,
    }

    for row, mu_row in enumerate([mu_h0, mu_ha]):
        for col, b in enumerate(b_list):
            ax = axs_s[row, col]
            b_vec = np.asarray(b, float).reshape(2)
            mean_vec = mu_row + b_vec

            d = mu_row[None, :] + b_vec[None, :] + noise
            Tmin, Tmax = interval_stats_zonotope_points(d, L, geom)
            dec = decide_extended(Tmin, Tmax, k)

            sa = (dec == 0)
            am = (dec == 1)
            sr = (dec == 2)

            h_sa = ax.scatter(d[sa, 0] * unit_scale, d[sa, 1] * unit_scale, s=10, c="lightgreen", label="Strict Accept")
            h_am = ax.scatter(d[am, 0] * unit_scale, d[am, 1] * unit_scale, s=10, c="khaki", label="Ambiguous")
            h_sr = ax.scatter(d[sr, 0] * unit_scale, d[sr, 1] * unit_scale, s=10, c="lightcoral", label="Strict Reject")
            h_mean = ax.scatter(mean_vec[0] * unit_scale, mean_vec[1] * unit_scale, c="blue", marker="o", s=50, label=r"$\mu_d + b$")
            h_org = ax.scatter(0.0, 0.0, c="black", marker="+", s=60, label="Origin")

            # Contours at k (same style as your BOX figure)
            ax.contour(XX * unit_scale, YY * unit_scale, Tmin_grid, levels=[k], colors=[sum_color], linewidths=sum_lw, linestyles=sum_ls)
            ax.contour(XX * unit_scale, YY * unit_scale, Tmax_grid, levels=[k], colors=[diff_color], linewidths=diff_lw, linestyles=diff_ls)

            # Pie inset
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

    fig_s.suptitle("Extended (Zonotope) diagnostics for selected bias vectors", fontsize=18, y=0.98)

    labels_order = [
        "Strict Accept", "Ambiguous", "Strict Reject",
        r"$\mu_d + b$", "Origin",
        r"ellipse $\ominus$ zonotope",
        r"ellipse $\oplus$ zonotope",
    ]
    handles = [legend_handles_s[k] for k in labels_order if k in legend_handles_s]
    labels = [k for k in labels_order if k in legend_handles_s]

    if handles:
        fig_s.legend(
            handles, labels,
            loc="upper center",
            ncol=len(handles),
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
def main(cfg: Config2DExtendedZonotope) -> None:
    here = Path(__file__).resolve().parent
    out_dir = here / cfg.out_dirname
    out_dir.mkdir(parents=True, exist_ok=True)

    Sigma = make_covariance_matrix(cfg.sigma1, cfg.sigma2, theta_deg=cfg.theta_deg)
    G = get_zonotope_generators(cfg.generators)

    out_h0 = run_bias_map_extended_zonotope(
        mu_d=np.array(cfg.mu_h0, float),
        Sigma=Sigma,
        G=G,
        alpha=cfg.alpha,
        n_stoch=cfg.n_stoch,
        n_grid=cfg.n_grid,
        seed=cfg.seed,
        n_jobs=cfg.n_jobs,
        backend=cfg.backend,
        save_npz=out_dir / "maps_H0_extended_zonotope.npz",
    )
    out_ha = run_bias_map_extended_zonotope(
        mu_d=np.array(cfg.mu_ha, float),
        Sigma=Sigma,
        G=G,
        alpha=cfg.alpha,
        n_stoch=cfg.n_stoch,
        n_grid=cfg.n_grid,
        seed=cfg.seed,
        n_jobs=cfg.n_jobs,
        backend=cfg.backend,
        save_npz=out_dir / "maps_Ha_extended_zonotope.npz",
    )

    plot_decision_grid_extended(out_h0, out_ha, save_path=out_dir / "Map_Grid_Extended_Zonotope.png")

    plot_selected_bias_scatter_and_hist_grid_extended(
        b_list=cfg.b_selected,
        mu_h0=np.array(cfg.mu_h0, float),
        mu_ha=np.array(cfg.mu_ha, float),
        Sigma=Sigma,
        G=G,
        alpha=cfg.alpha,
        n_stoch=cfg.n_stoch,
        seed=cfg.seed,
        unit_scale=100.0,
        unit_label="cm",
        save_prefix=out_dir / "SelectedBias_Extended_Zonotope",
        show=True,
    )

    print(f"Done. Outputs written to: {out_dir}")


if __name__ == "__main__":
    main(Config2DExtendedZonotope())
