"""Post-training analysis helpers hoisted from the monolith __main__ block."""

from nespreso.analysis.comparison import (
    compute_depth_interval_metrics,
    compute_season_masked_depth_rmse_bias,
    default_depth_intervals,
    isop_depth_indices,
)
from nespreso.analysis.correlation import calculate_correlation
from nespreso.analysis.glider import bin_data, get_glider_predictions
from nespreso.analysis.residuals import compute_depth_rmse_bias, compute_profile_residual

__all__ = [
    "bin_data",
    "calculate_correlation",
    "compute_depth_interval_metrics",
    "compute_depth_rmse_bias",
    "compute_profile_residual",
    "compute_season_masked_depth_rmse_bias",
    "default_depth_intervals",
    "get_glider_predictions",
    "isop_depth_indices",
]
