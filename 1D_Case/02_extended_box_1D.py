"""
1D interval-extended congruency test simulation (BOX-bounded remaining systematics).

Model
-----
    d = mu + b + e,     e ~ N(0, sigma^2),   b in [-Delta, Delta]

Extended (interval-extended, BOX bias):
    z_min(d) = max(|d|-Delta, 0)/sigma
    z_max(d) = (|d|+Delta)/sigma
    k        = Phi^{-1}(1-alpha/2)

3-valued decision:
    strict accept  if z_max <= k
    strict reject  if z_min >  k
    ambiguous      otherwise

This script produces:
  - extended histograms for selected biases b
  - extended scatter/strip plots for selected biases b
  - bias curves: P(SA), P(Amb), P(SR) under H0 and Ha
  - MC vs analytic validation (1D closed forms; Normal CDF)
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
class Config1DExtendedBox:
    # Units: meters
    sigma: float = 0.01          # 1 cm
    Delta: float = 0.01          # 1 cm (bias half-width)
    alpha: float = 0.05

    # Bias grid for bias-curve visualizations (b in [-Delta, Delta])
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
    out_dirname: str = "outputs/extend_1D"


# =============================================================================
# Core math
# =============================================================================
def k_alpha_two_sided(alpha: float) -> float:
    """Two-sided N(0,1) critical value k such that P(|Z|>k)=alpha."""
    return float(norm.ppf(1.0 - alpha / 2.0))


def extended_decision(d: np.ndarray, sigma: float, Delta: float, alpha: float) -> np.ndarray:
    """
    3-way decision for interval-extended BOX model:
      0 = strict accept
      1 = ambiguous
      2 = strict reject
    """
    d = np.asarray(d, dtype=float)
    sigma = float(sigma)
    Delta = float(Delta)

    k = k_alpha_two_sided(alpha)

    z_min = np.maximum(np.abs(d) - Delta, 0.0) / sigma
    z_max = (np.abs(d) + Delta) / sigma

    ext = np.empty_like(z_min, dtype=int)
    ext[z_max <= k] = 0
    ext[(z_min <= k) & (k < z_max)] = 1
    ext[z_min > k] = 2
    return ext


# =============================================================================
# Monte Carlo (bias–noise separated)
# =============================================================================
def run_mc_given_bias(
    b: float,
    *,
    mu: float,
    sigma: float,
    Delta: float,
    alpha: float,
    n_stoch: int,
    rng: np.random.Generator,
    reuse_noise: Optional[np.ndarray] = None,
) -> Dict[str, float | np.ndarray]:
    """
    Monte Carlo at fixed true bias b.

    Returns:
      P(SA), P(Amb), P(SR) and samples (d, ext) for diagnostics.
    """
    if reuse_noise is None:
        noise = rng.normal(loc=0.0, scale=sigma, size=n_stoch)
    else:
        noise = reuse_noise
        if noise.shape != (n_stoch,):
            raise ValueError("reuse_noise must have shape (n_stoch,)")

    d = float(mu) + float(b) + noise
    ext = extended_decision(d, sigma, Delta, alpha)

    return {
        "b": float(b),
        "p_ext_strict_accept": float(np.mean(ext == 0)),
        "p_ext_ambiguous": float(np.mean(ext == 1)),
        "p_ext_strict_reject": float(np.mean(ext == 2)),
        "d_samples": d,
        "ext_samples": ext,
    }


def run_bias_curve(
    *,
    mu: float,
    sigma: float,
    Delta: float,
    alpha: float,
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

    P_sa = np.zeros(n_grid, dtype=float)
    P_amb = np.zeros(n_grid, dtype=float)
    P_sr = np.zeros(n_grid, dtype=float)

    for i, b in enumerate(b_grid):
        res = run_mc_given_bias(
            float(b),
            mu=mu,
            sigma=sigma,
            Delta=Delta,
            alpha=alpha,
            n_stoch=n_stoch,
            rng=rng,
            reuse_noise=noise,
        )
        P_sa[i] = float(res["p_ext_strict_accept"])
        P_amb[i] = float(res["p_ext_ambiguous"])
        P_sr[i] = float(res["p_ext_strict_reject"])

    return {
        "b": b_grid,
        "P_ext_strict_accept": P_sa,
        "P_ext_amb": P_amb,
        "P_ext_strict_reject": P_sr,
        "mu": np.array([mu], float),
        "sigma": np.array([sigma], float),
        "Delta": np.array([Delta], float),
        "alpha": np.array([alpha], float),
        "n_stoch": np.array([n_stoch], int),
        "seed": np.array([seed], int),
    }


# =============================================================================
# Analytic reference (validation, 1D closed form)
# =============================================================================
def _p_abs_leq(t: float, m: np.ndarray, sigma: float) -> np.ndarray:
    """P(|X| <= t) for X ~ N(m, sigma^2), vectorized over m."""
    t = float(max(t, 0.0))
    m = np.asarray(m, dtype=float)
    return norm.cdf((t - m) / sigma) - norm.cdf((-t - m) / sigma)


def p_extended_analytic_normal(
    mu: float,
    b: np.ndarray,
    sigma: float,
    Delta: float,
    alpha: float,
) -> Dict[str, np.ndarray]:
    """
    Analytic probabilities for 1D interval-extended BOX test under:
        d ~ N(mu + b, sigma^2)

    Define c = k*sigma with k = Phi^{-1}(1-alpha/2).
      strict accept: |d| <= max(c-Delta, 0)
      strict reject: |d| >  c+Delta
      ambiguous: otherwise
    """
    b = np.asarray(b, dtype=float)
    k = k_alpha_two_sided(alpha)
    c = k * float(sigma)
    a = max(c - float(Delta), 0.0)   # SA threshold in |d|-domain
    r = c + float(Delta)             # SR threshold in |d|-domain

    m = float(mu) + b

    p_le_a = _p_abs_leq(a, m, float(sigma))
    p_le_r = _p_abs_leq(r, m, float(sigma))

    p_sa = np.clip(p_le_a, 0.0, 1.0)
    p_sr = np.clip(1.0 - p_le_r, 0.0, 1.0)
    p_amb = np.clip(p_le_r - p_le_a, 0.0, 1.0)

    return {
        "P_ext_strict_accept": p_sa,
        "P_ext_amb": p_amb,
        "P_ext_strict_reject": p_sr,
    }


# =============================================================================
# Plotting (keeps your color style)
# =============================================================================
def plot_extended_curves(
    out_h0: Dict[str, np.ndarray],
    out_ha: Dict[str, np.ndarray],
    *,
    save_path: Optional[Path] = None,
) -> None:
    b_mm = out_h0["b"] * 1000.0

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True, sharey=True)

    axs[0].plot(b_mm, out_h0["P_ext_strict_accept"], color="blue", label="H0")
    axs[0].plot(b_mm, out_ha["P_ext_strict_accept"], color="red", label="Ha")
    axs[0].set_title("Extended: P(Strict Accept | b)")
    axs[0].set_xlabel(r"$b$ [mm]")
    axs[0].set_ylabel("Probability")
    axs[0].grid(True, alpha=0.3)
    axs[0].legend()

    axs[1].plot(b_mm, out_h0["P_ext_amb"], color="blue", label="H0")
    axs[1].plot(b_mm, out_ha["P_ext_amb"], color="red", label="Ha")
    axs[1].set_title("Extended: P(Ambiguous | b)")
    axs[1].set_xlabel(r"$b$ [mm]")
    axs[1].grid(True, alpha=0.3)
    axs[1].legend()

    axs[2].plot(b_mm, out_h0["P_ext_strict_reject"], color="blue", label="H0")
    axs[2].plot(b_mm, out_ha["P_ext_strict_reject"], color="red", label="Ha")
    axs[2].set_title("Extended: P(Strict Reject | b)")
    axs[2].set_xlabel(r"$b$ [mm]")
    axs[2].grid(True, alpha=0.3)
    axs[2].legend()

    fig.suptitle("1D Interval-Extended Test (Bias Curve)", fontsize=14)
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def validate_extended_mc_vs_analytic(
    out_h0: Dict[str, np.ndarray],
    out_ha: Dict[str, np.ndarray],
    *,
    mu_h0: float,
    mu_ha: float,
    sigma: float,
    Delta: float,
    alpha: float,
    n_stoch: int,
    save_path: Optional[Path] = None,
) -> None:
    b = out_h0["b"]
    b_mm = b * 1000.0

    an_h0 = p_extended_analytic_normal(mu_h0, b, sigma, Delta, alpha)
    an_ha = p_extended_analytic_normal(mu_ha, b, sigma, Delta, alpha)

    mc_h0_sa, mc_h0_amb, mc_h0_sr = (
        out_h0["P_ext_strict_accept"],
        out_h0["P_ext_amb"],
        out_h0["P_ext_strict_reject"],
    )
    mc_ha_sa, mc_ha_amb, mc_ha_sr = (
        out_ha["P_ext_strict_accept"],
        out_ha["P_ext_amb"],
        out_ha["P_ext_strict_reject"],
    )

    def max_abs_err(mc: np.ndarray, an: np.ndarray) -> float:
        return float(np.max(np.abs(mc - an)))

    print("=== Extended test validation (Normal CDF; 1D closed form) ===")
    print(f"n_stoch = {n_stoch}")
    print(f"Max |MC-analytic| H0 strict accept: {max_abs_err(mc_h0_sa, an_h0['P_ext_strict_accept']):.4f}")
    print(f"Max |MC-analytic| H0 ambiguous:     {max_abs_err(mc_h0_amb, an_h0['P_ext_amb']):.4f}")
    print(f"Max |MC-analytic| H0 strict reject: {max_abs_err(mc_h0_sr, an_h0['P_ext_strict_reject']):.4f}")
    print(f"Max |MC-analytic| Ha strict accept: {max_abs_err(mc_ha_sa, an_ha['P_ext_strict_accept']):.4f}")
    print(f"Max |MC-analytic| Ha ambiguous:     {max_abs_err(mc_ha_amb, an_ha['P_ext_amb']):.4f}")
    print(f"Max |MC-analytic| Ha strict reject: {max_abs_err(mc_ha_sr, an_ha['P_ext_strict_reject']):.4f}")

    # --- style consistent with classical figure ---
    # MC: H0 blue, Ha red
    # Analytic: H0 orange dashed, Ha black dashed
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6), sharex=True, sharey=True)

    # Panel 1: Strict accept
    l1, = axs[0].plot(b_mm, mc_h0_sa, color="blue", lw=2, label="H0 MC")
    l2, = axs[0].plot(b_mm, mc_ha_sa, color="red",  lw=2, label="Ha MC")
    l3, = axs[0].plot(b_mm, an_h0["P_ext_strict_accept"], color="orange", ls="--", lw=2, label="H0 analytic")
    l4, = axs[0].plot(b_mm, an_ha["P_ext_strict_accept"], color="black",  ls="--", lw=2, label="Ha analytic")
    axs[0].set_title(r"$P(\mathrm{Strict\ Accept}\mid b)$", fontsize=13)
    axs[0].set_xlabel(r"$b$ [mm]")
    axs[0].set_ylabel("Probability")
    axs[0].grid(True, alpha=0.3)

    # Panel 2: Ambiguous
    axs[1].plot(b_mm, mc_h0_amb, color="blue", lw=2)
    axs[1].plot(b_mm, mc_ha_amb, color="red",  lw=2)
    axs[1].plot(b_mm, an_h0["P_ext_amb"], color="orange", ls="--", lw=2)
    axs[1].plot(b_mm, an_ha["P_ext_amb"], color="black",  ls="--", lw=2)
    axs[1].set_title(r"$P(\mathrm{Ambiguous}\mid b)$", fontsize=13)
    axs[1].set_xlabel(r"$b$ [mm]")
    axs[1].grid(True, alpha=0.3)

    # Panel 3: Strict reject
    axs[2].plot(b_mm, mc_h0_sr, color="blue", lw=2)
    axs[2].plot(b_mm, mc_ha_sr, color="red",  lw=2)
    axs[2].plot(b_mm, an_h0["P_ext_strict_reject"], color="orange", ls="--", lw=2)
    axs[2].plot(b_mm, an_ha["P_ext_strict_reject"], color="black",  ls="--", lw=2)
    axs[2].set_title(r"$P(\mathrm{Strict\ Reject}\mid b)$", fontsize=13)
    axs[2].set_xlabel(r"$b$ [mm]")
    axs[2].grid(True, alpha=0.3)

    # Suptitle + figure-level legend (like classic)
    fig.suptitle("1D Interval-Extended Test: MC vs analytic probabilities", fontsize=14, y=0.98)
    fig.legend(
        handles=[l1, l2, l3, l4],
        labels=["H0 MC", "Ha MC", "H0 analytic", "Ha analytic"],
        loc="upper center",
        ncol=4,
        frameon=True,
        fontsize=11,
        bbox_to_anchor=(0.5, 0.93),
    )

    # Reserve top space for title + legend
    fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.88])

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_selected_bias_samples_extended_hist(
    b_list: Sequence[float],
    *,
    sigma: float,
    Delta: float,
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
    thr_cls = k * sigma
    thr_sa = max(thr_cls - Delta, 0.0)  # |d| threshold for strict accept
    thr_sr = thr_cls + Delta            # |d| threshold for strict reject

    fig, axs = plt.subplots(2, len(b_list), figsize=(4.2 * len(b_list), 7), sharex=True, sharey=True)

    max_mean = max(abs(mu_h0), abs(mu_ha)) + max(abs(float(b)) for b in b_list)
    x_lim = max_mean + 4.5 * sigma

    def draw(ax, d: np.ndarray, title: str) -> None:
        ax.hist(d, bins=bins, color="lightblue", density=True, alpha=0.9)

        # Extended boundaries (your style request)
        ax.axvline(-thr_sa, color="green", linestyle="-.", linewidth=1.8)
        ax.axvline(+thr_sa, color="green", linestyle="-.", linewidth=1.8)
        ax.axvline(-thr_sr, color="green", linestyle="-.", linewidth=1.8)
        ax.axvline(+thr_sr, color="green", linestyle="-.", linewidth=1.8)

        # Classical reference
        ax.axvline(-thr_cls, color="red", linestyle="--", linewidth=1.6)
        ax.axvline(+thr_cls, color="red", linestyle="--", linewidth=1.6)

        ext = extended_decision(d, sigma, Delta, alpha)
        p_sa = float(np.mean(ext == 0))
        p_amb = float(np.mean(ext == 1))
        p_sr = float(np.mean(ext == 2))

        ax.text(
            0.02,
            0.98,
            f"P(SA)={p_sa:.3f}\nP(Amb)={p_amb:.3f}\nP(SR)={p_sr:.3f}\n"
            f"kσ={thr_cls:.4f}\nΔ={Delta:.4f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
        )

        ax.set_title(title, fontsize=11)
        ax.grid(True, alpha=0.25)
        ax.set_xlim(-x_lim, x_lim)

    for j, b in enumerate(b_list):
        draw(axs[0, j], mu_h0 + float(b) + noise, rf"H0: $\mu={mu_h0:.3f}$, $b={b:.3f}$")

    for j, b in enumerate(b_list):
        draw(axs[1, j], mu_ha + float(b) + noise, rf"Ha: $\mu={mu_ha:.3f}$, $b={b:.3f}$")
        axs[1, j].set_xlabel("d [m]")

    axs[0, 0].set_ylabel("Density")
    axs[1, 0].set_ylabel("Density")
    fig.suptitle(
        "1D Monte Carlo samples for selected biases (Extended)\n"
        "Dotted green: ±(kσ−Δ) strict accept, Dash-dot green: ±(kσ+Δ) strict reject, Dashed red: ±kσ classical",
        fontsize=13,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.92])

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_selected_bias_samples_scatter_1d_extended(
    b_list: list[float],
    sigma: float,
    Delta: float,
    alpha: float,
    mu_h0: float,
    mu_ha: float,
    n_stoch: int,
    seed: int = 0,
    jitter: float = 0.20,
    max_points: int = 5000,
    save_path: str | None = None,
    show_classical_lines: bool = True,
) -> None:
    """
    2xN scatter/strip plot for extended test.
    Same layout as classical scatter, but points are colored by extended class:
      0 strict accept, 1 ambiguous, 2 strict reject.

    Overlays:
      dotted:  ±(kσ−Δ)   (strict accept edge)
      dash-dot:±(kσ+Δ)   (strict reject edge)
      dashed red: ±kσ    (classical reference, optional)
    """
    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=0.0, scale=sigma, size=n_stoch)

    k = k_alpha_two_sided(alpha)
    thr_cls = k * sigma
    thr_sa = max(thr_cls - Delta, 0.0)
    thr_sr = thr_cls + Delta

    max_mean = max(abs(mu_h0), abs(mu_ha)) + max(abs(b) for b in b_list)
    x_lim = max_mean + 4.5 * sigma
    x_min, x_max = -x_lim, x_lim

    fig, axs = plt.subplots(2, len(b_list), figsize=(4.2 * len(b_list), 7), sharex=True, sharey=True)

    if n_stoch > max_points:
        idx = rng.choice(n_stoch, size=max_points, replace=False)
    else:
        idx = np.arange(n_stoch)

    def draw_panel(ax, d_full: np.ndarray, title: str, *, add_labels: bool) -> None:
        d = d_full[idx]
        y = rng.uniform(-jitter, jitter, size=d.shape[0])

        ext_sub = extended_decision(d, sigma, Delta, alpha)
        ext_full = extended_decision(d_full, sigma, Delta, alpha)

        for code, lab, col in [
            (0, "Strict accept", "green"),
            (1, "Ambiguous", "gray"),
            (2, "Strict reject", "orange"),
        ]:
            m = ext_sub == code
            if np.any(m):
                ax.scatter(
                    d[m],
                    y[m],
                    color=col,
                    s=10,
                    alpha=0.6,
                    linewidths=0,
                    label=(lab if add_labels else None),
                )

        # Extended boundaries
        ax.axvline(-thr_sa, color="green", linestyle="-.", linewidth=1.8)
        ax.axvline(+thr_sa, color="green", linestyle="-.", linewidth=1.8)
        ax.axvline(-thr_sr, color="green", linestyle="-.", linewidth=1.8)
        ax.axvline(+thr_sr, color="green", linestyle="-.", linewidth=1.8)

        # Classical reference
        if show_classical_lines:
            ax.axvline(-thr_cls, linestyle="--", linewidth=1.8, color="red")
            ax.axvline(+thr_cls, linestyle="--", linewidth=1.8, color="red")

        ax.set_title(title, fontsize=11)
        ax.grid(True, alpha=0.25)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(-0.35, 0.35)
        ax.set_yticks([])

        p_sa = float(np.mean(ext_full == 0))
        p_amb = float(np.mean(ext_full == 1))
        p_sr = float(np.mean(ext_full == 2))

        ax.text(
            0.02,
            0.98,
            f"P(SA)={p_sa:.3f}\nP(Amb)={p_amb:.3f}\nP(SR)={p_sr:.3f}\n"
            f"kσ={thr_cls:.4f}\nΔ={Delta:.4f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
        )

    for c, b in enumerate(b_list):
        d_full = mu_h0 + b + noise
        draw_panel(axs[0, c], d_full, title=rf"H0: $\mu={mu_h0:.3f}$, $b={b:.3f}$", add_labels=(c == 0))

    for c, b in enumerate(b_list):
        d_full = mu_ha + b + noise
        draw_panel(axs[1, c], d_full, title=rf"Ha: $\mu={mu_ha:.3f}$, $b={b:.3f}$", add_labels=False)
        axs[1, c].set_xlabel("d [m]")

    axs[0, 0].set_ylabel("Samples (jittered)")
    axs[1, 0].set_ylabel("Samples (jittered)")

    fig.suptitle(
        "1D Monte Carlo samples for selected biases (Extended test, scatter)\n"
        "Dotted green: ±(kσ−Δ) strict accept, Dash-dot green: ±(kσ+Δ) strict reject, Dashed red: ±kσ classical",
        fontsize=13,
    )

    # one global legend (from first panel)
    handles, labels = axs[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=3, frameon=True)

    plt.tight_layout(rect=[0, 0.06, 1, 0.90])
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


# =============================================================================
# Main
# =============================================================================
def main(cfg: Config1DExtendedBox) -> None:
    here = Path(__file__).resolve().parent
    out_dir = here / cfg.out_dirname
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Running 1D bias curve for H0 (extended)...")
    out_h0 = run_bias_curve(
        mu=cfg.mu_h0,
        sigma=cfg.sigma,
        Delta=cfg.Delta,
        alpha=cfg.alpha,
        n_stoch=cfg.n_stoch,
        n_grid=cfg.n_grid,
        seed=cfg.seed,
    )

    print("Running 1D bias curve for Ha (extended)...")
    out_ha = run_bias_curve(
        mu=cfg.mu_ha,
        sigma=cfg.sigma,
        Delta=cfg.Delta,
        alpha=cfg.alpha,
        n_stoch=cfg.n_stoch,
        n_grid=cfg.n_grid,
        seed=cfg.seed,
    )

    plot_selected_bias_samples_extended_hist(
        cfg.b_list,
        sigma=cfg.sigma,
        Delta=cfg.Delta,
        alpha=cfg.alpha,
        mu_h0=cfg.mu_h0,
        mu_ha=cfg.mu_ha,
        n_stoch=cfg.n_stoch,
        seed=cfg.seed,
        bins=20,
        save_path=out_dir / "SelectedBiasSamples_Extended_Hist.png",
    )

    plot_selected_bias_samples_scatter_1d_extended(
        list(cfg.b_list),
        sigma=cfg.sigma,
        Delta=cfg.Delta,
        alpha=cfg.alpha,
        mu_h0=cfg.mu_h0,
        mu_ha=cfg.mu_ha,
        n_stoch=cfg.n_stoch,
        seed=cfg.seed,
        jitter=0.20,
        max_points=5000,
        save_path=str(out_dir / "SelectedBiasSamples_Extended_Scatter.png"),
        show_classical_lines=True,
    )

    plot_extended_curves(out_h0, out_ha, save_path=out_dir / "Probability_Curves_Extended.png")

    validate_extended_mc_vs_analytic(
        out_h0,
        out_ha,
        mu_h0=cfg.mu_h0,
        mu_ha=cfg.mu_ha,
        sigma=cfg.sigma,
        Delta=cfg.Delta,
        alpha=cfg.alpha,
        n_stoch=cfg.n_stoch,
        save_path=out_dir / "Validation_Extended_MC_vs_Analytic.png",
    )

    print(f"Done. Outputs written to: {out_dir}")


if __name__ == "__main__":
    main(Config1DExtendedBox())
