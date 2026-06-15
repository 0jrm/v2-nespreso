"""Depth-interval RMSE/bias/correlation table (hoisted from monolith __main__)."""

from __future__ import annotations

from nespreso.analysis import compute_depth_interval_metrics, default_depth_intervals
from nespreso.experiments.validation_context import ValidationContext


def run_depth_interval_stats(ctx: ValidationContext) -> None:
    """Print average bias and RMSE for configured depth ranges."""
    min_depth = ctx.min_depth
    max_depth = ctx.max_depth
    isop_depths = ctx.isop_depths
    ist = ctx.ist
    iss = ctx.iss
    original_profiles = ctx.original_profiles
    pred_T = ctx.pred_T
    pred_S = ctx.pred_S
    gem_temp = ctx.gem_temp
    gem_sal = ctx.gem_sal
    old_pred_T = ctx.old_pred_T
    old_pred_S = ctx.old_pred_S

        # calculate average bias and rmse for depth ranges
    print(
        "Depth range \t NeSPReSO 1.1 T RMSE \t GEM T RMSE \t NeSPReSO 1.0 T RMSE \t ISOP T RMSE \t NeSPReSO 1.1 T Bias \t GEM T Bias \t NeSPReSO 1.0 T Bias \t ISOP T Bias \t NeSPReSO 1.1 S RMSE \t GEM S RMSE \t NeSPReSO 1.0 S RMSE \t ISOP S RMSE \t NeSPReSO 1.1 S Bias \t GEM S Bias \t NeSPReSO 1.0 S Bias \t ISOP S Bias \t NeSPReSO 1.1 T R^2 \t GEM T R^2 \t NeSPReSO 1.0 T R^2 \t NeSPReSO 1.1 S R^2 \t GEM S R^2 \t NeSPReSO 1.0 S R^2"
    )
    intervals = default_depth_intervals(min_depth, max_depth)

    for min_d, max_d in intervals:
        metrics = compute_depth_interval_metrics(
            min_d,
            max_d,
            isop_depths,
            ist.rmse.values,
            ist.bias.values,
            iss.rmse.values,
            iss.bias.values,
            original_profiles,
            pred_T,
            pred_S,
            gem_temp,
            gem_sal,
            old_pred_T,
            old_pred_S,
        )

        print(
            f"[{metrics['min_d']}-{metrics['max_d']}] \t {metrics['nn_t_rmse']:.3f} \t {metrics['gem_t_rmse']:.3f} \t {metrics['mlr_t_rmse']:.3f} \t {metrics['isop_avg_t_rmse']:.3f} \t {metrics['nn_t_bias']:.3f} \t {metrics['gem_t_bias']:.3f} \t {metrics['mlr_t_bias']:.3f} \t {metrics['isop_avg_t_bias']:.3f} \t {metrics['nn_s_rmse']:.3f} \t {metrics['gem_s_rmse']:.3f} \t {metrics['mlr_s_rmse']:.3f} \t {metrics['isop_avg_s_rmse']:.3f} \t {metrics['nn_s_bias']:.3f} \t {metrics['gem_s_bias']:.3f} \t {metrics['mlr_s_bias']:.3f} \t {metrics['isop_avg_s_bias']:.3f} \t {metrics['nn_T_corr']:.3f} \t {metrics['gem_T_corr']:.3f} \t {metrics['mlr_T_corr']:.3f} \t {metrics['nn_S_corr']:.3f} \t {metrics['gem_S_corr']:.3f} \t {metrics['mlr_S_corr']:.3f}"
        )
        # print(f"[{min_d}-{max_d}] \t {nn_t_rmse:.3f} \t {gem_t_rmse:.3f} \t {isop_avg_t_rmse:.3f} \t {nn_t_bias:.3f} \t {gem_t_bias:.3f} \t {isop_avg_t_bias:.3f} \t {nn_s_rmse:.3f} \t {gem_s_rmse:.3f} \t {isop_avg_s_rmse:.3f} \t {nn_s_bias:.3f} \t {gem_s_bias:.3f} \t {isop_avg_s_bias:.3f} \t {nn_T_corr:.3f} \t {gem_T_corr:.3f} \t {nn_S_corr:.3f} \t {gem_S_corr:.3f}")
        # print("\hline")
