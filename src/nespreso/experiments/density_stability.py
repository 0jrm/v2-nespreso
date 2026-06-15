"""Density stability and smoothness experiment (hoisted from monolith __main__)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from nespreso.analysis.density import (
    compute_density_profiles,
    compute_smoothness_metrics,
    compute_stability_metrics,
)
from nespreso.experiments.validation_context import ValidationContext


def run_density_stability(ctx: ValidationContext) -> dict:
    """Compare NeSPReSO 1.0 vs 1.1 vertical density metrics on validation profiles."""
    print("\n" + "=" * 80)
    print("VERTICAL METRICS ANALYSIS: NeSPReSO 1.0 vs 1.1")
    print("=" * 80)

    n_depth_data = ctx.pred_T.shape[0]
    depth_array = np.arange(ctx.min_depth, ctx.min_depth + n_depth_data)
    n_profiles = ctx.pred_T.shape[1]

    try:
        PRES_data = ctx.full_dataset.PRES[:, ctx.subset_indices]
        use_pres = True
        print("Using PRES data for EOS computation")
    except Exception:
        use_pres = False
        print("Using depth as pressure approximation (1 dbar ≈ 1 m)")

    T_11 = ctx.pred_T.T
    S_11 = ctx.pred_S.T
    T_10 = ctx.old_pred_T.T
    S_10 = ctx.old_pred_S.T
    T_orig = ctx.original_profiles[:, 0, :].T
    S_orig = ctx.original_profiles[:, 1, :].T

    lat_profiles = ctx.lat_val
    lon_profiles = ctx.lon_val

    print("\nComputing density profiles for vertical metrics...")

    pres_for_eos = PRES_data if use_pres else None
    rho_11 = compute_density_profiles(
        T_11, S_11, lat_profiles, lon_profiles, depth_array, "NeSPReSO 1.1", pres_for_eos
    )
    rho_10 = compute_density_profiles(
        T_10, S_10, lat_profiles, lon_profiles, depth_array, "NeSPReSO 1.0", pres_for_eos
    )
    rho_orig = compute_density_profiles(
        T_orig, S_orig, lat_profiles, lon_profiles, depth_array, "Argo", pres_for_eos
    )

    print("Density computation completed.")

    print("\nComputing static stability metrics...")

    stability_11 = compute_stability_metrics(rho_11, depth_array, "NeSPReSO 1.1")
    stability_10 = compute_stability_metrics(rho_10, depth_array, "NeSPReSO 1.0")
    stability_orig = compute_stability_metrics(rho_orig, depth_array, "Argo")

    print("Computing density smoothness metrics...")

    smoothness_11 = compute_smoothness_metrics(rho_11, depth_array, "NeSPReSO 1.1")
    smoothness_10 = compute_smoothness_metrics(rho_10, depth_array, "NeSPReSO 1.0")
    smoothness_orig = compute_smoothness_metrics(rho_orig, depth_array, "Argo")

    print("\n" + "-" * 80)
    print("VERTICAL METRICS COMPARISON")
    print("-" * 80)
    print(f"\nStatic Stability Metrics:")
    print(f"{'Metric':<30} {'Argo':<15} {'NeSPReSO 1.0':<15} {'NeSPReSO 1.1':<15}")
    print("-" * 75)
    print(
        f"{'Fraction Unstable':<30} {stability_orig['frac_unstable']:<15.6f} {stability_10['frac_unstable']:<15.6f} {stability_11['frac_unstable']:<15.6f}"
    )
    print(
        f"{'Min N² [s⁻²]':<30} {stability_orig['min_N2']:<15.6e} {stability_10['min_N2']:<15.6e} {stability_11['min_N2']:<15.6e}"
    )

    int_neg_orig = stability_orig["int_neg_N2"]
    int_neg_10 = stability_10["int_neg_N2"]
    int_neg_11 = stability_11["int_neg_N2"]

    mean_int_neg_orig = np.nanmean(int_neg_orig) if np.any(np.isfinite(int_neg_orig)) else np.nan
    mean_int_neg_10 = np.nanmean(int_neg_10) if np.any(np.isfinite(int_neg_10)) else np.nan
    mean_int_neg_11 = np.nanmean(int_neg_11) if np.any(np.isfinite(int_neg_11)) else np.nan

    print(
        f"{'Mean Int. Neg. N² [m/s²]':<30} {mean_int_neg_orig:<15.6e} {mean_int_neg_10:<15.6e} {mean_int_neg_11:<15.6e}"
    )

    print(f"\nDensity Smoothness Metrics (50-300 m):")
    print(f"{'Metric':<30} {'Argo':<15} {'NeSPReSO 1.0':<15} {'NeSPReSO 1.1':<15}")
    print("-" * 75)
    print(
        f"{'Var(d²ρ/dz²)':<30} {smoothness_orig['var_d2rho_dz2']:<15.6e} {smoothness_10['var_d2rho_dz2']:<15.6e} {smoothness_11['var_d2rho_dz2']:<15.6e}"
    )
    print(
        f"{'Mean Inflection Points':<30} {smoothness_orig['mean_inflections']:<15.2f} {smoothness_10['mean_inflections']:<15.2f} {smoothness_11['mean_inflections']:<15.2f}"
    )

    print("\nGenerating vertical metrics comparison plots...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle("Vertical Metrics Comparison: NeSPReSO 1.0 vs 1.1", fontsize=16, fontweight="bold")

    ax = axes[0, 0]
    models = ["Argo", "NeSPReSO 1.0", "NeSPReSO 1.1"]
    frac_vals = [stability_orig["frac_unstable"], stability_10["frac_unstable"], stability_11["frac_unstable"]]
    colors = ["blue", "green", "red"]
    bars = ax.bar(models, frac_vals, color=colors, alpha=0.7, edgecolor="black")
    ax.set_ylabel("Fraction Unstable Profiles")
    ax.set_title("Fraction of Profiles with N² < 0")
    ax.grid(True, alpha=0.3, axis="y")
    ax.tick_params(axis="x", labelsize=12)
    for bar, val in zip(bars, frac_vals):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, height, f"{val:.3f}", ha="center", va="bottom", fontsize=11)

    ax = axes[0, 1]
    var_d2_vals = [smoothness_orig["var_d2rho_dz2"], smoothness_10["var_d2rho_dz2"], smoothness_11["var_d2rho_dz2"]]
    bars = ax.bar(models, var_d2_vals, color=colors, alpha=0.7, edgecolor="black")
    ax.set_ylabel("Var(d²ρ/dz²)")
    ax.set_title("Density Curvature Variance (50-300 m)")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, axis="y", which="both")
    ax.tick_params(axis="x", labelsize=12)
    for bar, val in zip(bars, var_d2_vals):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, height, f"{val:.2e}", ha="center", va="bottom", fontsize=9)

    ax = axes[1, 0]
    int_neg_vals = [mean_int_neg_orig, mean_int_neg_10, mean_int_neg_11]
    bars = ax.bar(models, int_neg_vals, color=colors, alpha=0.7, edgecolor="black")
    ax.set_ylabel("Mean Integrated |N²<0| [m/s²]")
    ax.set_title("Spatial Mean of Integrated Negative N²")
    ax.grid(True, alpha=0.3, axis="y")
    ax.tick_params(axis="x", labelsize=12)
    for bar, val in zip(bars, int_neg_vals):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{val:.2e}",
            ha="center",
            va="bottom" if height < 0 else "top",
            fontsize=9,
        )

    ax = axes[1, 1]
    min_n2_vals = [stability_orig["min_N2"], stability_10["min_N2"], stability_11["min_N2"]]
    bars = ax.bar(models, min_n2_vals, color=colors, alpha=0.7, edgecolor="black")
    ax.set_ylabel("Minimum N² [s⁻²]")
    ax.set_title("Global Minimum N²")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.tick_params(axis="x", labelsize=12)
    for bar, val in zip(bars, min_n2_vals):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{val:.2e}",
            ha="center",
            va="bottom" if height < 0 else "top",
            fontsize=9,
        )

    plt.tight_layout()
    plt.show()

    fig2, ax2 = plt.subplots(1, 1, figsize=(8, 6))
    infl_vals = [
        smoothness_orig["mean_inflections"],
        smoothness_10["mean_inflections"],
        smoothness_11["mean_inflections"],
    ]
    bars = ax2.bar(models, infl_vals, color=colors, alpha=0.7, edgecolor="black")
    ax2.set_ylabel("Mean Inflection Points")
    ax2.set_title("Mean Number of Inflection Points (50-300 m)")
    ax2.grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars, infl_vals):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2.0, height, f"{val:.2f}", ha="center", va="bottom", fontsize=11)
    plt.tight_layout()
    plt.show()

    fig3, ax3 = plt.subplots(1, 1, figsize=(12, 6))

    int_neg_orig_flat = int_neg_orig.flatten()
    int_neg_10_flat = int_neg_10.flatten()
    int_neg_11_flat = int_neg_11.flatten()

    valid_orig = int_neg_orig_flat[np.isfinite(int_neg_orig_flat)]
    valid_10 = int_neg_10_flat[np.isfinite(int_neg_10_flat)]
    valid_11 = int_neg_11_flat[np.isfinite(int_neg_11_flat)]

    all_vals = np.concatenate([valid_orig, valid_10, valid_11])
    if len(all_vals) > 0:
        abs_vals = np.abs(all_vals)
        log_min = np.log10(np.max([abs_vals.min(), 1e-6]))
        log_max = np.log10(abs_vals.max())
        n_bins = 50
        log_bins = np.logspace(log_min, log_max, n_bins)
        bins = -log_bins[::-1]

        ax3.hist(valid_orig, bins=bins, alpha=0.5, label="Argo", color="blue", edgecolor="black")
        ax3.hist(valid_10, bins=bins, alpha=0.5, label="NeSPReSO 1.0", color="green", edgecolor="black")
        ax3.hist(valid_11, bins=bins, alpha=0.5, label="NeSPReSO 1.1", color="red", edgecolor="black")
        ax3.set_xlabel("Integrated |N²<0| [m/s²]")
        ax3.set_ylabel("Frequency")
        ax3.set_title("Distribution of Integrated Negative N²")
        ax3.set_xscale("symlog", linthresh=1e-5)
        ax3.legend(loc="upper left", fontsize=11)
        ax3.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.show()

    print("\nVertical metrics analysis completed!")
    print("=" * 80 + "\n")

    return {
        "stability_11": stability_11,
        "stability_10": stability_10,
        "stability_orig": stability_orig,
        "smoothness_11": smoothness_11,
        "smoothness_10": smoothness_10,
        "smoothness_orig": smoothness_orig,
        "rho_11": rho_11,
        "rho_10": rho_10,
        "rho_orig": rho_orig,
    }
