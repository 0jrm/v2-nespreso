#!/usr/bin/env python3
"""Density stability and smoothness comparison (NeSPReSO 1.0 vs 1.1)."""

from __future__ import annotations

import sys

from nespreso.experiments.common import build_experiment_parser, load_cfg_and_artifacts
from nespreso.experiments.density_stability import run_density_stability


def main(argv: list[str] | None = None) -> int:
    parser = build_experiment_parser("NeSPReSO density stability experiment")
    args = parser.parse_args(argv)
    _, ctx = load_cfg_and_artifacts(args.config, bin_size=args.bin_size)
    run_density_stability(ctx)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
