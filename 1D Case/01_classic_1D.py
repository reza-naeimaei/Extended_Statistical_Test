"""
1D classical congruency test simulation (purely stochastic).

Model
-----
    d = mu + b + e,     e ~ N(0, sigma^2)

For a fixed true bias b (outer loop), we run Monte Carlo over stochastic noise e (inner loop)
and estimate conditional rejection probabilities.

Classical two-sided z-test (level alpha):
    reject if |d| > k_alpha * sigma
where k_alpha = Phi^{-1}(1 - alpha/2).

This script produces:
  - histograms/scatter for selected biases b
  - bias curve: P(Reject | b) under H0 and Ha
  - MC vs analytic validation (Normal CDF)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


# =============================================================================
# Configuration (edit here)
# =============================================================================
@dataclass(frozen=True)
class Config1DClassic:
    # Units: meters
    sigma: float = 0.01          # 1 cm
    alpha: float = 0.05

    # Bias grid (for bias-curve visualizations)
    Delta: float = 0.01          # shown range for b in [-Delta, Delta]
    n_grid: int = 500

    # Monte Carlo
    n_stoch: int = 10_000
    seed: int = 0

    # Scenarios
    mu_h0: float = 0.0
    mu_ha: float = 0.02          # 2 cm

    # Diagnostics: selected b values for hist/scatter
    b_list: tuple[float, ...] = (-0.01, -0.005, 0.0, 0.005, 0.01)

    # Output folder (created under this script folder)
    out_dirname: str = "outputs/classic_1D"


# =============================================================================
# Core math
# =============================================================================
def k_alpha_two_sided(alpha: float) -> float:
    """Two-sided N(0,1) critical value k such that P(|Z|>k)=alpha."""
    return float(norm.ppf(1.0 - alpha / 2.0))


def classical_decision(d: np.ndarray, sigma: float, alpha: float) -> np.ndarray:
    """
    Binary decision:
      0 = accept
      2 = reject
    """
    d = np.asarray(d, dtype=float)
    thr = k_alpha_two_sided(alpha) * float(sigma)
    return np.where(np.abs(d) <= thr, 0, 2).astype(int)


# =============================================================================
# Monte Carlo (bias–noise separated)
# =============================================================================
def run_mc_given_bias(
    b: float,
    *,
    mu: float,
    sigma: float,
    alpha: float,
    n_stoch: int,
    rng: np.random.Generator,
    reuse_noise: Optional[np.ndarray] = None,
) -> Dict[str, float | np.ndarray]:
    """
    Monte Carlo at fixed true bias b.

    Returns:
      p_cls_reject, p_cls_accept
      d_samples, cls_samples
    """
    if reuse_noise is None:
        noise = rng.normal(loc=0.0, scale=sigma, size=n_stoch)
    else:
        noise = reuse_noise
        if noise.shape != (n_stoch,):
            raise ValueError("reuse_noise must have shape (n_stoch,)")

    d = mu + b + noise
    cls = classical_decision(d, sigma, alpha)

    return {
        "b": float(b),
        "p_cls_accept": float(np.mean(cls == 0)),
        "p_cls_reject": float(np.mean(cls == 2)),
        "d_samples": d,
        "cls_samples": cls,
    }


def run_bias_curve(
    *,
    mu: float,
    sigma: float,
    alpha: float,
    Delta: float,
    n_stoch: int,
    n_grid: int,
    seed: int,
) -> Dict[str, np.ndarray]:
    """
    1D analog of a bias map: conditional curve over b in [-Delta, Delta].
    Uses common random numbers (same noise for all b) for variance reduction.
    """
    rng = np.random.default_rng(seed)
    b_grid = np.linspace(-Delta, Delta, n_grid)

    noise = rng.normal(loc=0.0, scale=sigma, size=n_stoch)

    P_reject = np.zeros(n_grid, dtype=float)
    for i, b in enumerate(b_grid):
        res = run_mc_given_bias(
            float(b),
            mu=mu,
            sigma=sigma,
            alpha=alpha,
            n_stoch=n_stoch,
            rng=rng,
            reuse_noise=noise,
        )
        P_reject[i] = float(res["p_cls_reject"])

    return {
        "b": b_grid,
        "P_cls_reject": P_reject,
        "mu": np.array([mu], float),
        "sigma": np.array([sigma], float),
        "Delta": np.array([Delta], float),
        "alpha": np.array([alpha], float),
        "n_stoch": np.array([n_stoch], int),
        "seed": np.array([seed], int),
    }


# =============================================================================
# Analytic reference (validation)
# =============================================================================
def p_cls_reject_analytic_normal(mu: float, b: np.ndarray, sigma: float, alpha: float) -> np.ndarray:
    """
    Classical two-sided rejection probability under:
        d ~ N(mu + b, sigma^2)
    Reject if |d| > k*sigma, where k = Phi^{-1}(1 - alpha/2).
    """
    b = np.asarray(b, dtype=float)
    k = k_alpha_two_sided(alpha)
    m = mu + b

    # P(d > k*sigma) + P(d < -k*sigma)
    z1 = (k * sigma - m) / sigma
    z2 = (-k * sigma - m) / sigma
    return (1.0 - norm.cdf(z1)) + norm.cdf(z2)


# =============================================================================
# Plotting
# =============================================================================
def plot_classical_curves(out_h0: Dict[str, np.ndarray], out_ha: Dict[str, np.ndarray], *, save_path: Optional[Path] = None) -> None:
    b_mm = out_h0["b"] * 1000.0
    p0_rej = out_h0["P_cls_reject"]
    pa_rej = out_ha["P_cls_reject"]

    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True, sharey=True)

    axs[0].plot(b_mm, 1.0 - p0_rej, color='blue', label="H0")
    axs[0].plot(b_mm, 1.0 - pa_rej, color='red', label="Ha")
    axs[0].set_title("Classical: P(Accept | b)")
    axs[0].set_xlabel(r"$b$ [mm]")
    axs[0].set_ylabel("Probability")
    axs[0].grid(True, alpha=0.3)
    axs[0].legend()

    axs[1].plot(b_mm, p0_rej, color='blue', label="H0")
    axs[1].plot(b_mm, pa_rej, color='red', label="Ha")
    axs[1].set_title("Classical: P(Reject | b)")
    axs[1].set_xlabel(r"$b$ [mm]")
    axs[1].grid(True, alpha=0.3)
    axs[1].legend()

    fig.suptitle("1D Classical Test (Bias Curve)", fontsize=14)
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def validate_classical_mc_vs_analytic(
    out_h0: Dict[str, np.ndarray],
    out_ha: Dict[str, np.ndarray],
    *,
    mu_h0: float,
    mu_ha: float,
    sigma: float,
    alpha: float,
    n_stoch: int,
    save_path: Optional[Path] = None,
) -> None:
    """Validate 1D classical test probabilities (MC vs analytic).

    Produces a 1×2 figure (Accept and Reject) in the same style as the extended-test
    validation plots.

    Notes
    -----
    Analytic probabilities are computed via Normal CDF (equivalently, central/noncentral χ² in 1D).
    """
    b = out_h0["b"]
    b_mm = b * 1000.0

    # Analytic rejection probabilities
    p_h0_rej_an = p_cls_reject_analytic_normal(mu_h0, b, sigma, alpha)
    p_ha_rej_an = p_cls_reject_analytic_normal(mu_ha, b, sigma, alpha)

    # MC rejection probabilities
    p_h0_rej_mc = out_h0["P_cls_reject"]
    p_ha_rej_mc = out_ha["P_cls_reject"]

    # Acceptance = 1 - rejection
    p_h0_acc_an = 1.0 - p_h0_rej_an
    p_ha_acc_an = 1.0 - p_ha_rej_an
    p_h0_acc_mc = 1.0 - p_h0_rej_mc
    p_ha_acc_mc = 1.0 - p_ha_rej_mc

    # Diagnostics
    err_h0 = p_h0_rej_mc - p_h0_rej_an
    err_ha = p_ha_rej_mc - p_ha_rej_an
    se_worst = 0.5 / np.sqrt(n_stoch)

    print("=== Classical test validation (Normal CDF) ===")
    print(f"n_stoch = {n_stoch}")
    print(f"Worst-case SE (p=0.5): {se_worst:.4f}")
    print(f"Max |MC-analytic| H0 (reject): {float(np.max(np.abs(err_h0))):.4f}")
    print(f"Max |MC-analytic| Ha (reject): {float(np.max(np.abs(err_ha))):.4f}")

    # Plot (two panels)
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.8), sharex=True, sharey=True)

    def _panel(ax, title: str, p_h0_mc, p_ha_mc, p_h0_an, p_ha_an, ylabel: str) -> None:
        ax.plot(b_mm, p_h0_mc, color="blue", linestyle="-", label="H0 MC")
        ax.plot(b_mm, p_ha_mc, color="red", linestyle="-", label="Ha MC")
        ax.plot(b_mm, p_h0_an, color="orange", linestyle="--", lw=2, label="H0 analytic")
        ax.plot(b_mm, p_ha_an, color="black", linestyle="--", lw=2, label="Ha analytic")
        ax.set_title(title)
        ax.set_xlabel(r"$b$ [mm]")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

    _panel(
        axs[0],
        r"$P(\mathrm{Accept}\mid b)$",
        p_h0_acc_mc,
        p_ha_acc_mc,
        p_h0_acc_an,
        p_ha_acc_an,
        r"$P(\mathrm{Accept}\mid b)$",
    )
    _panel(
        axs[1],
        r"$P(\mathrm{Reject}\mid b)$",
        p_h0_rej_mc,
        p_ha_rej_mc,
        p_h0_rej_an,
        p_ha_rej_an,
        r"$P(\mathrm{Reject}\mid b)$",
    )

    # --- FIX: dedicated top area for title+legend (no overlap) ---
    fig.suptitle("1D Classical Test: MC vs analytic probabilities", fontsize=14, y=0.98)

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="upper center",
        ncol=4,
        frameon=True,
        bbox_to_anchor=(0.5, 0.92),
    )

    # Reserve space at the top for suptitle+legend
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.14, top=0.8, wspace=0.12)

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_selected_bias_samples_hist(
    b_list: Sequence[float],
    *,
    sigma: float,
    alpha: float,
    mu_h0: float,
    mu_ha: float,
    n_stoch: int,
    seed: int,
    bins: int = 30,
    save_path: Optional[Path] = None,
) -> None:
    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=0.0, scale=sigma, size=n_stoch)

    k = k_alpha_two_sided(alpha)
    thr = k * sigma

    fig, axs = plt.subplots(2, len(b_list), figsize=(4.2 * len(b_list), 7), sharex=True, sharey=True)

    max_mean = max(abs(mu_h0), abs(mu_ha)) + max(abs(float(b)) for b in b_list)
    x_lim = max_mean + 4.5 * sigma

    def draw(ax, d: np.ndarray, title: str) -> None:
        ax.hist(d, bins=bins, color='lightblue', density=True, alpha=0.9)
        ax.axvline(-thr, color='red', linestyle="--", linewidth=1.8)
        ax.axvline(+thr, color='red', linestyle="--", linewidth=1.8)
        cls = classical_decision(d, sigma, alpha)
        ax.text(
            0.02, 0.98,
            f"P(reject)={float(np.mean(cls==2)):.3f}\n±kσ=±{thr:.4f}",
            transform=ax.transAxes, va="top", ha="left",
            fontsize=9, bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
        )
        ax.set_title(title, fontsize=11)
        ax.grid(True, alpha=0.25)
        ax.set_xlim(-x_lim, x_lim)

    for j, b in enumerate(b_list):
        d = mu_h0 + float(b) + noise
        draw(axs[0, j], d, rf"H0: $\mu={mu_h0:.3f}$, $b={b:.3f}$")

    for j, b in enumerate(b_list):
        d = mu_ha + float(b) + noise
        draw(axs[1, j], d, rf"Ha: $\mu={mu_ha:.3f}$, $b={b:.3f}$")
        axs[1, j].set_xlabel("d [m]")

    axs[0, 0].set_ylabel("Density")
    axs[1, 0].set_ylabel("Density")
    fig.suptitle("1D Monte Carlo samples for selected biases (Classical)", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.92])

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_selected_bias_samples_scatter(
    b_list: Sequence[float],
    *,
    sigma: float,
    alpha: float,
    mu_h0: float,
    mu_ha: float,
    n_stoch: int,
    seed: int,
    jitter: float = 0.20,
    max_points: int = 5000,
    save_path: Optional[Path] = None,
) -> None:
    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=0.0, scale=sigma, size=n_stoch)

    k = k_alpha_two_sided(alpha)
    thr = k * sigma

    if n_stoch > max_points:
        idx = rng.choice(n_stoch, size=max_points, replace=False)
    else:
        idx = np.arange(n_stoch)

    fig, axs = plt.subplots(2, len(b_list), figsize=(4.2 * len(b_list), 7), sharex=True, sharey=True)

    max_mean = max(abs(mu_h0), abs(mu_ha)) + max(abs(float(b)) for b in b_list)
    x_lim = max_mean + 4.5 * sigma

    def draw(ax, d_full: np.ndarray, title: str) -> None:
        d = d_full[idx]
        y = rng.uniform(-jitter, jitter, size=d.shape[0])
        cls_sub = classical_decision(d, sigma, alpha)
        acc = cls_sub == 0
        rej = cls_sub == 2
        if np.any(acc):
            ax.scatter(d[acc], y[acc], s=10, alpha=0.6, linewidths=0)
        if np.any(rej):
            ax.scatter(d[rej], y[rej], s=10, alpha=0.6, linewidths=0)
        ax.axvline(-thr, color='red', linestyle="--", linewidth=1.8)
        ax.axvline(+thr, color='red', linestyle="--", linewidth=1.8)
        cls_full = classical_decision(d_full, sigma, alpha)
        ax.text(
            0.02, 0.98,
            f"P(reject)={float(np.mean(cls_full==2)):.3f}\n±kσ=±{thr:.4f}",
            transform=ax.transAxes, va="top", ha="left",
            fontsize=9, bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
        )
        ax.set_title(title, fontsize=11)
        ax.grid(True, alpha=0.25)
        ax.set_xlim(-x_lim, x_lim)
        ax.set_ylim(-0.35, 0.35)
        ax.set_yticks([])

    for j, b in enumerate(b_list):
        draw(axs[0, j], mu_h0 + float(b) + noise, rf"H0: $\mu={mu_h0:.3f}$, $b={b:.3f}$")

    for j, b in enumerate(b_list):
        draw(axs[1, j], mu_ha + float(b) + noise, rf"Ha: $\mu={mu_ha:.3f}$, $b={b:.3f}$")
        axs[1, j].set_xlabel("d [m]")

    axs[0, 0].set_ylabel("Samples (jittered)")
    axs[1, 0].set_ylabel("Samples (jittered)")
    fig.suptitle("1D Monte Carlo samples for selected biases (Classical scatter)", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.92])

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


# =============================================================================
# Main
# =============================================================================
def main(cfg: Config1DClassic) -> None:
    here = Path(__file__).resolve().parent
    out_dir = here / cfg.out_dirname
    out_dir.mkdir(parents=True, exist_ok=True)

    # Bias curves
    print("Running 1D bias curve for H0 (classical)...")
    out_h0 = run_bias_curve(
        mu=cfg.mu_h0,
        sigma=cfg.sigma,
        alpha=cfg.alpha,
        Delta=cfg.Delta,
        n_stoch=cfg.n_stoch,
        n_grid=cfg.n_grid,
        seed=cfg.seed,
    )

    print("Running 1D bias curve for Ha (classical)...")
    out_ha = run_bias_curve(
        mu=cfg.mu_ha,
        sigma=cfg.sigma,
        alpha=cfg.alpha,
        Delta=cfg.Delta,
        n_stoch=cfg.n_stoch,
        n_grid=cfg.n_grid,
        seed=cfg.seed,
    )

    # Diagnostics + plots
    plot_selected_bias_samples_hist(
        cfg.b_list,
        sigma=cfg.sigma,
        alpha=cfg.alpha,
        mu_h0=cfg.mu_h0,
        mu_ha=cfg.mu_ha,
        n_stoch=cfg.n_stoch,
        seed=cfg.seed,
        bins=20,
        save_path=out_dir / "SelectedBiasSamples_Classical_Hist.png",
    )

    plot_selected_bias_samples_scatter(
        cfg.b_list,
        sigma=cfg.sigma,
        alpha=cfg.alpha,
        mu_h0=cfg.mu_h0,
        mu_ha=cfg.mu_ha,
        n_stoch=cfg.n_stoch,
        seed=cfg.seed,
        jitter=0.20,
        max_points=5000,
        save_path=out_dir / "SelectedBiasSamples_Classical_Scatter.png",
    )

    plot_classical_curves(out_h0, out_ha, save_path=out_dir / "Curves_Classical.png")

    validate_classical_mc_vs_analytic(
        out_h0,
        out_ha,
        mu_h0=cfg.mu_h0,
        mu_ha=cfg.mu_ha,
        sigma=cfg.sigma,
        alpha=cfg.alpha,
        n_stoch=cfg.n_stoch,
        save_path=out_dir / "Validation_Classical_MC_vs_Analytic.png",
    )

    print(f"Done. Outputs written to: {out_dir}")


if __name__ == "__main__":
    main(Config1DClassic())
