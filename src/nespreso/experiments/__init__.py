"""Runnable post-training experiment pipelines."""

from nespreso.experiments.common import (
    build_experiment_parser,
    configure_matplotlib,
    load_cfg_and_artifacts,
)
from nespreso.experiments.compare_legacy_nespreso import (
    run_compare_legacy_nespreso,
    run_gem_inference_timing,
    run_legacy_prediction_comparison,
    run_nespreso_inference_timing,
)
from nespreso.experiments.density_stability import run_density_stability
from nespreso.experiments.depth_interval_stats import run_depth_interval_stats
from nespreso.experiments.glider_mission import run_glider_mission
from nespreso.experiments.monthly_distribution import run_monthly_distribution
from nespreso.experiments.pca_regression import run_pca_regression_baseline
from nespreso.experiments.steric_depth_stats import run_steric_depth_stats
from nespreso.experiments.validation_context import ValidationContext, build_validation_context
from nespreso.experiments.validation_maps import run_validation_maps

__all__ = [
    "ValidationContext",
    "build_experiment_parser",
    "build_validation_context",
    "configure_matplotlib",
    "load_cfg_and_artifacts",
    "run_compare_legacy_nespreso",
    "run_density_stability",
    "run_depth_interval_stats",
    "run_gem_inference_timing",
    "run_glider_mission",
    "run_legacy_prediction_comparison",
    "run_monthly_distribution",
    "run_nespreso_inference_timing",
    "run_pca_regression_baseline",
    "run_steric_depth_stats",
    "run_validation_maps",
]
