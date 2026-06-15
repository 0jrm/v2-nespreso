#!/usr/bin/env python3
"""Train NeSPReSO (build dataset pickle, split, train, save checkpoint)."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

from nespreso.config import load_config
from nespreso.runner import run_training


def build_train_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="NeSPReSO training")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="YAML config (default: configs/default.yaml in repo root)",
    )
    parser.add_argument(
        "--tensorboard",
        action="store_true",
        help="Enable TensorBoard logging (overrides config monitor.tensorboard)",
    )
    parser.add_argument("--log-dir", default=None, help="TensorBoard log directory")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_train_parser()
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    if args.tensorboard:
        log_dir = args.log_dir or cfg.monitor.log_dir
        cfg = replace(cfg, monitor=replace(cfg.monitor, tensorboard=True, log_dir=log_dir))
    run_training(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
