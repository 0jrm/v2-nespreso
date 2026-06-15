#!/usr/bin/env python3
"""Train/val/test profiles-per-month stacked bar chart."""

from __future__ import annotations

import sys

from nespreso.experiments.common import build_experiment_parser, load_cfg_and_artifacts
from nespreso.experiments.monthly_distribution import run_monthly_distribution


def main(argv: list[str] | None = None) -> int:
    parser = build_experiment_parser("NeSPReSO monthly profile distribution experiment")
    args = parser.parse_args(argv)
    _, ctx = load_cfg_and_artifacts(args.config, bin_size=args.bin_size)
    run_monthly_distribution(ctx)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
