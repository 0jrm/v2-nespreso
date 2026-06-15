"""
Pin post-training analysis helpers hoisted from the monolith __main__ block.

Golden values use independent fixed seeds per helper. ``get_glider_predictions``
is pinned on CPU with a stub inverse-transform callback.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from tests.monolith_loader import load_monolith

TOL = 1e-6
GOLDEN_FILE = Path(__file__).parent / "golden" / "analysis_synthetic.json"


def _load_golden():
    return json.loads(GOLDEN_FILE.read_text())


@pytest.fixture
def monolith_module():
    return load_monolith()


def test_bin_data_golden(monolith_module):
    m = monolith_module
    golden = _load_golden()
    np.random.seed(401)
    data = np.arange(50, dtype=float).reshape(10, 5) + np.random.randn(10, 5) * 0.01
    binned = m.bin_data(data, 5)
    assert np.allclose(binned, golden["bin_data"], rtol=0, atol=TOL)


def test_calculate_correlation_golden(monolith_module):
    m = monolith_module
    golden = _load_golden()
    obs = np.array([[1.0, 2.0, np.nan], [4.0, 5.0, 6.0]])
    pred = np.array([[1.1, 2.2, 3.3], [3.9, 5.1, 6.2]])
    corr = m.calculate_correlation(obs, pred)
    assert np.allclose(corr, golden["calculate_correlation"], rtol=0, atol=TOL)


def test_compute_depth_rmse_bias_golden(monolith_module):
    m = monolith_module
    golden = _load_golden()["depth_rmse_bias"]
    np.random.seed(402)
    resid = np.random.randn(8, 6)
    rmse_d, bias_d = m.compute_depth_rmse_bias(resid, axis=1)
    assert np.allclose(rmse_d[:4], golden["rmse_head"], rtol=0, atol=TOL)
    assert np.allclose(bias_d[:4], golden["bias_head"], rtol=0, atol=TOL)


def test_compute_season_masked_depth_rmse_bias_golden(monolith_module):
    m = monolith_module
    golden = _load_golden()["season_masked"]
    np.random.seed(402)
    resid = np.random.randn(8, 6)
    mask = np.array([True, False, True, True, False, True])
    rmse_d, bias_d = m.compute_season_masked_depth_rmse_bias(resid, mask)
    assert np.allclose(rmse_d[:4], golden["rmse_head"], rtol=0, atol=TOL)
    assert np.allclose(bias_d[:4], golden["bias_head"], rtol=0, atol=TOL)


def test_compute_depth_interval_metrics_golden(monolith_module):
    m = monolith_module
    golden = _load_golden()["depth_interval_metrics"]
    np.random.seed(403)
    depths = np.arange(0, 8)
    original_profiles = np.random.randn(8, 2, 6)
    pred_T = np.random.randn(8, 6)
    pred_S = np.random.randn(8, 6)
    gem_temp = np.random.randn(6, 8)
    gem_sal = np.random.randn(6, 8)
    mlr_T = np.random.randn(8, 6)
    mlr_S = np.random.randn(8, 6)
    ist_rmse = np.linspace(0.1, 0.8, 8)
    ist_bias = np.linspace(-0.05, 0.05, 8)
    iss_rmse = np.linspace(0.05, 0.4, 8)
    iss_bias = np.linspace(-0.02, 0.02, 8)

    metrics = m.compute_depth_interval_metrics(
        1,
        6,
        depths,
        ist_rmse,
        ist_bias,
        iss_rmse,
        iss_bias,
        original_profiles,
        pred_T,
        pred_S,
        gem_temp,
        gem_sal,
        mlr_T,
        mlr_S,
    )

    for key, expected in golden.items():
        if key in ("min_d", "max_d"):
            assert metrics[key] == expected
        else:
            assert np.allclose(metrics[key], expected, rtol=0, atol=TOL)


def _fake_inverse(pcs):
    batch = pcs.shape[0]
    depth = 12
    t = np.arange(batch, dtype=float)[:, None] * np.ones((1, depth)) + pcs[:, :1]
    s = t * 0.5 + 35
    return t.T, s.T


def test_get_glider_predictions_golden(monolith_module):
    m = monolith_module
    golden = _load_golden()["glider_predictions"]
    torch.manual_seed(404)
    device = torch.device("cpu")
    inputs = torch.randn(3, 4)
    model = m.PredictionModel(input_dim=4, layers_config=[6], output_dim=4, dropout_prob=0.0)
    model.eval()
    loader = DataLoader(TensorDataset(inputs, inputs), batch_size=3)

    T_pred, S_pred = m.get_glider_predictions(
        model, loader, inputs, device, _fake_inverse, max_depth=10, min_depth=1
    )

    assert list(T_pred.shape) == golden["T_shape"]
    assert list(S_pred.shape) == golden["S_shape"]
    assert np.allclose(T_pred[:3, 0], golden["T_head"], rtol=0, atol=TOL)
    assert np.allclose(S_pred[:3, 0], golden["S_head"], rtol=0, atol=TOL)


def test_default_depth_intervals_shape(monolith_module):
    m = monolith_module
    intervals = m.default_depth_intervals(20, 1800)
    assert len(intervals) == 8
    assert intervals[0] == (20, 20)
    assert intervals[-1] == (20, 1800)
