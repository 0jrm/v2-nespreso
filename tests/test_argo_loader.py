"""Unit tests for ARGO .mat loading."""

from datetime import datetime
from unittest.mock import patch

import numpy as np

from nespreso.io.argo import load_argo_mat


def test_load_argo_mat_converts_time_and_returns_arrays():
    mat_data = {
        "TIME": np.array([737789.5, 737790.5]),
        "LAT": np.array([25.0, 26.0]),
        "LON": np.array([-90.0, -89.0]),
        "SH1950": np.array([0.1, 0.2]),
    }

    with patch("nespreso.io.argo.mat73.loadmat", return_value=mat_data):
        data, time, lat, lon, sh1950 = load_argo_mat("/fake/path.mat")

    assert data is mat_data
    assert len(time) == 2
    assert all(isinstance(t, datetime) for t in time)
    np.testing.assert_array_equal(lat, mat_data["LAT"])
    np.testing.assert_array_equal(lon, mat_data["LON"])
    np.testing.assert_array_equal(sh1950, mat_data["SH1950"])
