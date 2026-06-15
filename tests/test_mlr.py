"""
Pin PCA regression baseline helpers hoisted from the monolith __main__ block.

GPU fit is pinned on CPU via DEVICE monkeypatch for deterministic goldens.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from nespreso.analysis.mlr import fit_pcs_regression_exact_gpu, predict_pcs_exact_gpu, prepare_features

TOL = 1e-6
GOLDEN_FILE = Path(__file__).parent / "golden" / "geo_depth_mlr_synthetic.json"


def _load_golden():
    return json.loads(GOLDEN_FILE.read_text())


@pytest.fixture(autouse=True)
def force_mlr_cpu():
    import nespreso.analysis.mlr as mlr_mod

    old_device = mlr_mod.DEVICE
    mlr_mod.DEVICE = torch.device("cpu")
    yield
    mlr_mod.DEVICE = old_device


def test_prepare_features_golden():
    golden = _load_golden()
    np.random.seed(408)
    inputs = np.random.randn(8, 3)
    X = prepare_features(inputs, max_degree=2)
    assert list(X.shape) == golden["prepare_features_shape"]
    assert np.allclose(X[:2, :4], golden["prepare_features_head"], rtol=0, atol=TOL)


def test_fit_pcs_regression_exact_gpu_golden(capsys):
    golden = _load_golden()
    torch.manual_seed(408)
    np.random.seed(408)
    inputs = np.random.randn(8, 3)
    X = prepare_features(inputs, max_degree=2)
    pcs = np.random.randn(8, 4)
    beta = fit_pcs_regression_exact_gpu(X, pcs)
    assert list(beta.shape) == golden["fit_pcs_beta_shape"]
    assert np.allclose(beta[:3, :2].cpu(), golden["fit_pcs_beta_head"], rtol=0, atol=TOL)


def test_predict_pcs_exact_gpu_golden():
    golden = _load_golden()
    torch.manual_seed(408)
    np.random.seed(408)
    inputs = np.random.randn(8, 3)
    X = prepare_features(inputs, max_degree=2)
    pcs = np.random.randn(8, 4)
    beta = fit_pcs_regression_exact_gpu(X, pcs)
    X_new = prepare_features(np.random.randn(3, 3), max_degree=2)
    pcs_pred = predict_pcs_exact_gpu(beta, X_new)
    assert list(pcs_pred.shape) == golden["predict_pcs_shape"]
    assert np.allclose(pcs_pred[:2, :2], golden["predict_pcs_head"], rtol=0, atol=TOL)
