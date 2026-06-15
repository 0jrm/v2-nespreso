"""Shared helpers for runnable experiment scripts."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from nespreso.config import AppConfig, load_config
from nespreso.experiments.validation_context import ValidationContext, build_validation_context
from nespreso.runner import TrainingArtifacts, run_training


def configure_matplotlib() -> None:
    plt.rcParams.update({"font.size": 18})


def build_experiment_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="YAML config (default: configs/default.yaml in repo root)",
    )
    parser.add_argument(
        "--bin-size",
        type=float,
        default=1.0,
        help="Map bin size in degrees (monolith default: 1)",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Assume checkpoint is loaded via config; still builds validation context",
    )
    return parser


def load_cfg_and_artifacts(
    config_path: Path | None,
    *,
    bin_size: float = 1.0,
) -> tuple[AppConfig, ValidationContext]:
    configure_matplotlib()
    cfg = load_config(config_path)
    _, artifacts = run_training(cfg, return_artifacts=True)
    ctx = build_validation_context(cfg, artifacts, bin_size=bin_size)
    return cfg, ctx


def load_cfg_and_context_only(
    config_path: Path | None,
    artifacts: TrainingArtifacts,
    *,
    bin_size: float = 1.0,
) -> tuple[AppConfig, ValidationContext]:
    configure_matplotlib()
    cfg = load_config(config_path)
    ctx = build_validation_context(cfg, artifacts, bin_size=bin_size)
    return cfg, ctx
