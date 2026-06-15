"""
Pin Phase 7 glider field plotting helpers before extraction from the monolith.

Uses the Agg backend and mocks ``plt.show`` so figure artifacts are deterministic.
Golden values use ``np.random.seed(200)`` on a fixed distance/depth grid.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from nespreso.viz.fields import plot_field, plot_field_subplot

TOL = 1e-6
GOLDEN_FILE = Path(__file__).parent / "golden" / "viz_fields_synthetic.json"


def _load_golden():
    return json.loads(GOLDEN_FILE.read_text())


@pytest.fixture
def field_grid_fixture():
    np.random.seed(200)
    n_dist, n_depth = 12, 10
    distances = np.linspace(0, 120, n_dist)
    depths = np.linspace(0, 200, n_depth)
    data = 20.0 + np.random.randn(n_depth, n_dist)
    return distances, depths, data


def test_plot_field_temperature_artifacts(field_grid_fixture):
    distances, depths, data = field_grid_fixture
    golden = _load_golden()["plot_field_temperature"]

    plt.close("all")
    with patch("matplotlib.pyplot.show"):
        plot_field(data, distances, depths, "Temperature", "test temperature field")

    fig = plt.figure(plt.get_fignums()[0])
    ax = fig.axes[0]
    assert ax.get_title() == golden["axes"][0]["title"]
    assert ax.get_xlabel() == golden["axes"][0]["xlabel"]
    assert ax.get_ylabel() == golden["axes"][0]["ylabel"]
    pcm_head = np.asarray(ax.collections[1].get_array()).ravel()[:8]
    assert np.allclose(pcm_head, golden["axes"][0]["collections"][1]["head"], rtol=0, atol=TOL)


def test_plot_field_salinity_artifacts(field_grid_fixture):
    distances, depths, data = field_grid_fixture
    golden = _load_golden()["plot_field_salinity"]

    plt.close("all")
    with patch("matplotlib.pyplot.show"):
        plot_field(data + 15, distances, depths, "Salinity", "test salinity field")

    fig = plt.figure(plt.get_fignums()[0])
    ax = fig.axes[0]
    assert ax.get_title() == golden["axes"][0]["title"]
    pcm_head = np.asarray(ax.collections[1].get_array()).ravel()[:8]
    assert np.allclose(pcm_head, golden["axes"][0]["collections"][1]["head"], rtol=0, atol=TOL)


def test_plot_field_subplot_temperature_artifacts(field_grid_fixture):
    distances, depths, data = field_grid_fixture
    golden = _load_golden()["plot_field_subplot_temperature"]

    plt.close("all")
    fig = plt.figure(figsize=(8, 6))
    plot_field_subplot(data, distances, depths, "Temperature", "Glider T", 111, fig)

    ax = fig.axes[0]
    assert ax.get_title() == golden["title"]
    assert ax.get_xlabel() == golden["xlabel"]
    assert ax.get_ylabel() == golden["ylabel"]
    pcm_head = np.asarray(ax.collections[0].get_array()).ravel()[:8]
    assert np.allclose(pcm_head, golden["collection_head"], rtol=0, atol=TOL)


def test_plot_field_subplot_t_difference_artifacts(field_grid_fixture):
    distances, depths, data = field_grid_fixture
    golden = _load_golden()["plot_field_subplot_t_diff"]

    plt.close("all")
    fig = plt.figure(figsize=(8, 6))
    plot_field_subplot(data - 20, distances, depths, "T Difference", "T Difference", 111, fig)

    ax = fig.axes[0]
    assert ax.get_title() == golden["title"]
    pcm_head = np.asarray(ax.collections[0].get_array()).ravel()[:8]
    assert np.allclose(pcm_head, golden["collection_head"], rtol=0, atol=TOL)


def test_plot_field_invalid_variable_raises(field_grid_fixture):
    distances, depths, data = field_grid_fixture
    with pytest.raises(ValueError, match="Invalid variable name"):
        plot_field(data, distances, depths, "Density", "bad")
