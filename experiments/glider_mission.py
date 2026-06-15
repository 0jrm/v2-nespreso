#!/usr/bin/env python3
"""Glider mission crossing validation experiment."""

from __future__ import annotations

import sys

from nespreso.experiments.common import build_experiment_parser, load_cfg_and_artifacts
from nespreso.experiments.glider_mission import run_glider_mission


def main(argv: list[str] | None = None) -> int:
    parser = build_experiment_parser("NeSPReSO glider mission experiment")
    args = parser.parse_args(argv)
    _, ctx = load_cfg_and_artifacts(args.config, bin_size=args.bin_size)
    run_glider_mission(ctx)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
