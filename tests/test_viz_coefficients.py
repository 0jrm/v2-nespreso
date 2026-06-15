"""
Pin Phase 7 coefficient heatmap helper before extraction from the monolith.

Uses the Agg backend and mocks ``plt.show`` so seaborn heatmap artifacts are
deterministic. Golden values use ``np.random.seed(300)`` and a fixed beta tensor.
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
import torch

from nespreso.viz.coefficients import plot_coefficients_heatmap

TOL = 1e-6
GOLDEN_FILE = Path(__file__).parent / "golden" / "viz_coefficients_synthetic.json"


def _load_golden():
    return json.loads(GOLDEN_FILE.read_text())


@pytest.fixture
def coefficients_fixture():
    np.random.seed(300)
    torch.manual_seed(300)
    beta = torch.tensor(
        [
            [0.8, -0.2, 0.05],
            [0.1, 0.5, -0.9],
            [-0.3, 0.02, 0.4],
            [0.0, 0.15, -0.1],
        ],
        dtype=torch.float32,
    )
    feature_names = ["f0", "f1", "f2", "f3"]
    return beta, feature_names


def test_plot_coefficients_heatmap_normalized_artifacts(coefficients_fixture):
    beta, feature_names = coefficients_fixture
    golden = _load_golden()

    plt.close("all")
    with patch("matplotlib.pyplot.show"):
        plot_coefficients_heatmap(beta, feature_names, "test coefficients heatmap", normalize=True, threshold=1e-4)

    ax = plt.figure(plt.get_fignums()[0]).axes[0]
    arr = np.asarray(ax.collections[0].get_array()).ravel()
    assert ax.get_title() == golden["title"]
    assert ax.get_xlabel() == golden["xlabel"]
    assert ax.get_ylabel() == golden["ylabel"]
    assert [t.get_text() for t in ax.get_xticklabels()] == golden["xticklabels"]
    assert [t.get_text() for t in ax.get_yticklabels()] == golden["yticklabels"]
    assert arr.size == golden["heatmap_size"]
    assert np.allclose(arr[:12], golden["heatmap_head"], rtol=0, atol=TOL)


def test_plot_coefficients_heatmap_unnormalized_artifacts(coefficients_fixture):
    beta, feature_names = coefficients_fixture
    golden = _load_golden()

    plt.close("all")
    with patch("matplotlib.pyplot.show"):
        plot_coefficients_heatmap(
            beta, feature_names, "test coefficients heatmap", normalize=False, threshold=0.2
        )

    ax = plt.figure(plt.get_fignums()[0]).axes[0]
    arr = np.asarray(ax.collections[0].get_array()).ravel()
    assert np.allclose(arr[:12], golden["heatmap_unnormalized_head"], rtol=0, atol=TOL)
