"""Post-training analysis helpers hoisted from the monolith __main__ block."""

from nespreso.analysis.comparison import (
    compute_depth_interval_metrics,
    compute_season_masked_depth_rmse_bias,
    default_depth_intervals,
    isop_depth_indices,
)
from nespreso.analysis.correlation import calculate_correlation
from nespreso.analysis.density import (
    compute_density_profiles,
    compute_smoothness_metrics,
    compute_stability_metrics,
)
from nespreso.analysis.depth_stats import (
    average_depth,
    equivalent_average_statistic,
    histogram_available_depths,
)
from nespreso.analysis.glider import bin_data, get_glider_predictions
from nespreso.analysis.mlr import (
    fit_pcs_regression_exact_gpu,
    predict_pcs_exact_gpu,
    prepare_features,
)
from nespreso.analysis.residuals import compute_depth_rmse_bias, compute_profile_residual

__all__ = [
    "average_depth",
    "bin_data",
    "calculate_correlation",
    "compute_depth_interval_metrics",
    "compute_depth_rmse_bias",
    "compute_density_profiles",
    "compute_profile_residual",
    "compute_season_masked_depth_rmse_bias",
    "compute_smoothness_metrics",
    "compute_stability_metrics",
    "default_depth_intervals",
    "equivalent_average_statistic",
    "fit_pcs_regression_exact_gpu",
    "get_glider_predictions",
    "histogram_available_depths",
    "isop_depth_indices",
    "predict_pcs_exact_gpu",
    "prepare_features",
]
