"""Config path validation tests."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from nespreso.config import load_config
from nespreso.runner import require_trained_model_path


def test_default_config_has_no_trained_model_path():
    cfg = load_config()
    assert cfg.runtime.load_trained_model is False
    assert cfg.paths.trained_model_path is None


def test_load_trained_model_false_allows_missing_path():
    cfg = load_config()
    cfg = replace(
        cfg,
        runtime=replace(cfg.runtime, load_trained_model=False),
        paths=replace(cfg.paths, trained_model_path=None),
    )
    assert cfg.runtime.load_trained_model is False
    assert cfg.paths.trained_model_path is None


def test_load_trained_model_true_requires_path():
    cfg = load_config()
    cfg = replace(
        cfg,
        runtime=replace(cfg.runtime, load_trained_model=True),
        paths=replace(cfg.paths, trained_model_path=None),
    )
    with pytest.raises(ValueError, match="trained_model_path is not set"):
        require_trained_model_path(cfg)


def test_load_trained_model_true_requires_existing_path(tmp_path: Path):
    missing = tmp_path / "missing.pth"
    cfg = load_config()
    cfg = replace(
        cfg,
        runtime=replace(cfg.runtime, load_trained_model=True),
        paths=replace(cfg.paths, trained_model_path=str(missing)),
    )
    with pytest.raises(FileNotFoundError, match="checkpoint not found"):
        require_trained_model_path(cfg)


def test_load_trained_model_true_accepts_existing_path(tmp_path: Path):
    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"stub")
    cfg = load_config()
    cfg = replace(
        cfg,
        runtime=replace(cfg.runtime, load_trained_model=True),
        paths=replace(cfg.paths, trained_model_path=str(checkpoint)),
    )
    assert require_trained_model_path(cfg) == str(checkpoint)
