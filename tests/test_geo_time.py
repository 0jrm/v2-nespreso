"""
Pin geo/time helpers hoisted from the monolith __main__ block.
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


def test_haversine_golden(monolith_module):
    m = monolith_module
    golden = _load_golden()
    dist = m.haversine(28.5, -89.0, 29.1, -88.5)
    assert np.allclose(dist, golden["haversine"], rtol=0, atol=TOL)


def test_calculate_distances_golden(monolith_module):
    m = monolith_module
    golden = _load_golden()
    np.random.seed(405)
    lats = 28.0 + np.cumsum(np.random.randn(6) * 0.01)
    lons = -89.0 + np.cumsum(np.random.randn(6) * 0.01)
    dists = m.calculate_distances(lats, lons)
    assert np.allclose(dists[:4], golden["calculate_distances_head"], rtol=0, atol=TOL)
    assert np.allclose(dists[-1], golden["calculate_distances_tail"], rtol=0, atol=TOL)


def test_datenums_to_datetimes_golden(monolith_module):
    m = monolith_module
    golden = _load_golden()
    matlab_datenums = np.array([736863.0151, 736883.0033, 736903.4988])
    dts = m.datenums_to_datetimes(matlab_datenums)
    assert [dt.isoformat() for dt in dts] == golden["datenums_to_datetimes"]
