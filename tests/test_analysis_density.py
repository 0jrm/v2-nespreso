"""
Pin density/stability helpers hoisted from the monolith __main__ block.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from nespreso.analysis.density import (
    compute_density_profiles,
    compute_smoothness_metrics,
    compute_stability_metrics,
)

TOL = 1e-6
GOLDEN_FILE = Path(__file__).parent / "golden" / "analysis_density_synthetic.json"


def _load_golden():
    return json.loads(GOLDEN_FILE.read_text())


def test_compute_density_profiles_golden():
    golden = _load_golden()
    np.random.seed(409)
    depth_arr = np.arange(20, 80, 5, dtype=float)
    t_profiles = 20 + np.random.randn(2, len(depth_arr))
    s_profiles = 35 + np.random.randn(2, len(depth_arr)) * 0.1
    lat_arr = np.array([28.5, 29.0])
    lon_arr = np.array([-89.0, -88.5])
    rho = compute_density_profiles(t_profiles, s_profiles, lat_arr, lon_arr, depth_arr, "test")
    assert list(rho.shape) == golden["rho_shape"]
    assert np.allclose(rho[0, :4], golden["rho_head"], rtol=0, atol=TOL)


def test_compute_stability_metrics_golden():
    golden = _load_golden()
    np.random.seed(410)
    depth_arr = np.arange(20, 120, 5, dtype=float)
    rho_profiles = 1025 + np.cumsum(np.random.randn(3, len(depth_arr)) * 0.01, axis=1)
    stability = compute_stability_metrics(rho_profiles, depth_arr, "test")
    assert np.allclose(stability["frac_unstable"], golden["frac_unstable"], rtol=0, atol=TOL)
    assert np.allclose(stability["min_N2"], golden["min_N2"], rtol=0, atol=TOL)


def test_compute_smoothness_metrics_golden():
    golden = _load_golden()
    np.random.seed(410)
    depth_arr = np.arange(20, 120, 5, dtype=float)
    rho_profiles = 1025 + np.cumsum(np.random.randn(3, len(depth_arr)) * 0.01, axis=1)
    smoothness = compute_smoothness_metrics(rho_profiles, depth_arr, "test")
    assert np.allclose(smoothness["var_d2rho_dz2"], golden["var_d2rho_dz2"], rtol=0, atol=TOL)
    assert np.allclose(smoothness["mean_inflections"], golden["mean_inflections"], rtol=0, atol=TOL)
