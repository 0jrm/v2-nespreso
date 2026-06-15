"""Shared synthetic fixtures for unit and characterization tests."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest


@pytest.fixture
def synthetic_ts_profiles():
    """Small T/S profile stack on a fixed depth grid."""
    depth = np.linspace(0, 500, 26)
    n_profiles = 8
    temp = 20.0 - 0.02 * depth[:, None] + 0.1 * np.random.randn(len(depth), n_profiles)
    sal = 36.0 + 0.001 * depth[:, None] + 0.01 * np.random.randn(len(depth), n_profiles)
    return depth, temp, sal


@pytest.fixture
def synthetic_lat_lon_dates():
    lats = np.linspace(20.0, 30.0, 8)
    lons = np.linspace(-95.0, -85.0, 8)
    dates = [datetime(2015, 6, 1)] * 8
    return lats, lons, dates


@pytest.fixture
def fitted_pca_pair(synthetic_ts_profiles):
    from sklearn.decomposition import PCA

    _, temp, sal = synthetic_ts_profiles
    n_components = 3
    pca_temp = PCA(n_components=n_components)
    pca_sal = PCA(n_components=n_components)
    temp_pcs = pca_temp.fit_transform(temp.T)
    sal_pcs = pca_sal.fit_transform(sal.T)
    return pca_temp, pca_sal, temp_pcs, sal_pcs, n_components


def pytest_addoption(parser):
    parser.addoption("--run-unity", action="store_true", help="Run /unity characterization tests")


def pytest_configure(config):
    config.addinivalue_line("markers", "requires_unity: needs /unity data on HPC")
