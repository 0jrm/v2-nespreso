"""
Pin depth-bin statistics helpers hoisted from the monolith __main__ block.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tests.monolith_loader import load_monolith

TOL = 1e-6
GOLDEN_FILE = Path(__file__).parent / "golden" / "geo_depth_mlr_synthetic.json"


def _load_golden():
    return json.loads(GOLDEN_FILE.read_text())


@pytest.fixture
def monolith_module():
    return load_monolith()


def test_average_depth_golden(monolith_module):
    m = monolith_module
    golden = _load_golden()
    np.random.seed(406)
    depths = np.arange(0, 25, 5)
    targets = np.random.randn(5, 4) + 20
    targets[2, 0] = np.nan
    avg = m.average_depth(targets, depths)
    assert np.allclose(avg, golden["average_depth"], rtol=0, atol=TOL)


def test_histogram_available_depths_golden(monolith_module):
    m = monolith_module
    golden = _load_golden()
    np.random.seed(406)
    depths = np.arange(0, 25, 5)
    targets = np.random.randn(5, 4) + 20
    targets[2, 0] = np.nan
    hist = m.histogram_available_depths(targets)
    assert np.allclose(hist, golden["histogram_available_depths"], rtol=0, atol=TOL)


def test_equivalent_average_statistic_golden(monolith_module):
    m = monolith_module
    golden = _load_golden()["equivalent_average_statistic"]
    np.random.seed(407)
    depths_1m = np.arange(0, 20, 5)
    pred = np.random.randn(20, 3)
    tgt = pred + np.random.randn(20, 3) * 0.1
    count = np.array([3, 3, 3, 3])
    eq_rmse, eq_corr = m.equivalent_average_statistic(pred, tgt, count, depths_1m, m.rmse)
    assert np.allclose(eq_rmse, golden["rmse"], rtol=0, atol=TOL)
    assert np.allclose(eq_corr, golden["corr"], rtol=0, atol=TOL)


def test_equivalent_average_statistic_mixed_depth_bins(monolith_module):
    m = monolith_module
    np.random.seed(407)
    depths_5m = np.arange(0, 25, 5)
    pred_1m = np.random.randn(25, 3)
    tgt_5m = pred_1m[::5, :] + np.random.randn(5, 3) * 0.1
    count = np.array([3, 3, 3, 3, 3])
    eq_rmse, _ = m.equivalent_average_statistic(pred_1m, tgt_5m, count, depths_5m, m.rmse)
    assert np.isfinite(eq_rmse)
