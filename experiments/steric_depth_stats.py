#!/usr/bin/env python3
"""Steric-height / ISOP depth-bin statistics and pcolormesh plots."""

from __future__ import annotations

import sys

from nespreso.experiments.common import build_experiment_parser, load_cfg_and_artifacts
from nespreso.experiments.steric_depth_stats import run_steric_depth_stats


def main(argv: list[str] | None = None) -> int:
    parser = build_experiment_parser("NeSPReSO steric depth statistics experiment")
    args = parser.parse_args(argv)
    _, ctx = load_cfg_and_artifacts(args.config, bin_size=args.bin_size)
    run_steric_depth_stats(ctx)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
