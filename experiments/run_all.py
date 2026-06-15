#!/usr/bin/env python3
"""Run full NeSPReSO training plus all post-training validation experiments."""

from __future__ import annotations

import sys
from pathlib import Path

from nespreso.experiments.common import build_experiment_parser, load_cfg_and_artifacts
from nespreso.experiments.density_stability import run_density_stability
from nespreso.experiments.depth_interval_stats import run_depth_interval_stats
from nespreso.experiments.glider_mission import run_glider_mission
from nespreso.experiments.monthly_distribution import run_monthly_distribution
from nespreso.experiments.pca_regression import run_pca_regression_baseline
from nespreso.experiments.steric_depth_stats import run_steric_depth_stats
from nespreso.experiments.validation_maps import run_validation_maps


def main(argv: list[str] | None = None) -> int:
    parser = build_experiment_parser("NeSPReSO full pipeline (train + all experiments)")
    args = parser.parse_args(argv)
    _cfg, ctx = load_cfg_and_artifacts(args.config, bin_size=args.bin_size)

    run_steric_depth_stats(ctx)
    run_pca_regression_baseline(ctx)
    run_validation_maps(ctx)
    run_glider_mission(ctx)
    run_depth_interval_stats(ctx)
    run_density_stability(ctx)
    run_monthly_distribution(ctx)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
