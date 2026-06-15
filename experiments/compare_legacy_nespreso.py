#!/usr/bin/env python3
"""NeSPReSO 1.1 timing, ensemble, legacy 1.0 load, and GEM timing."""

from __future__ import annotations

import sys
from dataclasses import asdict

from nespreso.config import load_config
from nespreso.experiments.common import build_experiment_parser, configure_matplotlib
from nespreso.experiments.compare_legacy_nespreso import run_compare_legacy_nespreso
from nespreso.runner import run_training


def main(argv: list[str] | None = None) -> int:
    parser = build_experiment_parser("NeSPReSO legacy comparison (timing + NeSPReSO 1.0)")
    args = parser.parse_args(argv)
    configure_matplotlib()
    cfg = load_config(args.config)
    model_cfg = cfg.model
    runtime = cfg.runtime
    input_params = asdict(cfg.input_params)
    _, artifacts = run_training(cfg, return_artifacts=True)
    subset_indices = artifacts.val_loader.dataset.indices
    run_compare_legacy_nespreso(
        trained_model=artifacts.trained_model,
        val_loader=artifacts.val_loader,
        val_dataset=artifacts.val_dataset,
        device=artifacts.device,
        subset_indices=subset_indices,
        input_params=input_params,
        n_components=model_cfg.n_components,
        layers_config=list(model_cfg.layers_config),
        dropout_prob=model_cfg.dropout_prob,
        input_dim=artifacts.input_dim,
        nn_repeat_time=runtime.nn_repeat_time,
        gem_repeat_time=runtime.gem_repeat_time,
        ensemble_models=runtime.ensemble_models,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
