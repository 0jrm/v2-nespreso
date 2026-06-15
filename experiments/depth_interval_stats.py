#!/usr/bin/env python3
"""Depth-interval RMSE/bias/correlation table for validation profiles."""

from __future__ import annotations

import sys

from nespreso.experiments.common import build_experiment_parser, load_cfg_and_artifacts
from nespreso.experiments.depth_interval_stats import run_depth_interval_stats


def main(argv: list[str] | None = None) -> int:
    parser = build_experiment_parser("NeSPReSO depth-interval statistics experiment")
    args = parser.parse_args(argv)
    _, ctx = load_cfg_and_artifacts(args.config, bin_size=args.bin_size)
    run_depth_interval_stats(ctx)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
