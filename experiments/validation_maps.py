#!/usr/bin/env python3
"""RMSE/bias maps, seasonal depth curves, and comparison maps."""

from __future__ import annotations

import sys

from nespreso.experiments.common import build_experiment_parser, load_cfg_and_artifacts
from nespreso.experiments.validation_maps import run_validation_maps


def main(argv: list[str] | None = None) -> int:
    parser = build_experiment_parser("NeSPReSO validation maps experiment")
    args = parser.parse_args(argv)
    _, ctx = load_cfg_and_artifacts(args.config, bin_size=args.bin_size)
    run_validation_maps(ctx)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
