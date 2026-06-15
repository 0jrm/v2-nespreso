#!/usr/bin/env python3
"""PCA regression baseline experiment (MLR fit + depth RMSE/bias + coefficient heatmaps)."""

from __future__ import annotations

import sys

from nespreso.experiments.common import build_experiment_parser, load_cfg_and_artifacts
from nespreso.experiments.pca_regression import run_pca_regression_baseline


def main(argv: list[str] | None = None) -> int:
    parser = build_experiment_parser("NeSPReSO PCA regression baseline experiment")
    args = parser.parse_args(argv)
    _, ctx = load_cfg_and_artifacts(args.config, bin_size=args.bin_size)
    run_pca_regression_baseline(ctx)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
