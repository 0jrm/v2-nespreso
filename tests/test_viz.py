"""
Pin Phase 7 visualization helpers before extraction from the monolith.

Uses the Agg backend and mocks ``plt.show`` so plotting pins are deterministic.
Golden values use ``np.random.seed(42)`` on fixed synthetic grids.
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

from tests.monolith_loader import load_monolith

TOL = 1e-6
GOLDEN_DIR = Path(__file__).parent / "golden"
MAPS_GOLDEN = GOLDEN_DIR / "viz_maps_synthetic.json"
PROFILES_GOLDEN = GOLDEN_DIR / "viz_profiles_synthetic.json"
PLOT_GOLDEN = GOLDEN_DIR / "viz_plot_artifacts.json"


def _load_json(path: Path):
    return json.loads(path.read_text())


def _assert_grid_close(actual, expected):
    actual = np.asarray(actual, dtype=float)
    expected_arr = np.array(
        [[np.nan if v is None else v for v in row] for row in expected],
        dtype=float,
    )
    assert actual.shape == expected_arr.shape
    for a, e in zip(actual.ravel(), expected_arr.ravel()):
        if e is None or (isinstance(e, float) and np.isnan(e)):
            assert np.isnan(a)
        else:
            assert np.allclose(a, e, rtol=0, atol=TOL)


@pytest.fixture
def viz_maps_fixture():
    load_monolith()
    np.random.seed(100)
    lon_bins = np.array([-95.0, -93.0, -91.0])
    lat_bins = np.array([20.0, 22.0, 24.0])
    lon_val = np.array([-94.5, -94.0, -92.5, -91.5])
    lat_val = np.array([20.5, 21.0, 23.0, 23.5])
    bias_values = np.random.randn(26, 4).astype(np.float64)
    dpt_range = np.arange(26)
    return lon_bins, lat_bins, lon_val, lat_val, bias_values, dpt_range


@pytest.fixture
def viz_profiles_fixture():
    m = load_monolith()
    np.random.seed(102)
    m.min_depth = 20
    m.max_depth = 45
    min_d, max_d = 20, 45
    depth_n = max_d - min_d + 1
    true_values = np.random.randn(depth_n, 2, 4)
    gem_temp = np.random.randn(4, depth_n)
    gem_sal = np.random.randn(4, depth_n)
    pred0 = np.random.randn(depth_n, 4)
    pred1 = np.random.randn(depth_n, 4)
    return m, true_values, [pred0, pred1], gem_temp, gem_sal


@pytest.fixture
def viz_bias_fixture():
    m = load_monolith()
    np.random.seed(101)
    m.min_depth = 20
    m.max_depth = 45
    true_values = np.random.randn(26, 2, 4)
    gem_temp = np.random.randn(4, 26)
    gem_sal = np.random.randn(4, 26)
    pred0 = np.random.randn(26, 4)
    pred1 = np.random.randn(26, 4)
    return m, true_values, [pred0, pred1], gem_temp, gem_sal


def _fig_summary():
    figs = [plt.figure(n) for n in plt.get_fignums()]
    out = []
    for fig in figs:
        entry = {"naxes": len(fig.axes), "axes": []}
        for ax in fig.axes:
            axd = {"title": ax.get_title(), "nlines": len(ax.lines), "lines": [], "collections": []}
            for line in ax.lines[:3]:
                axd["lines"].append(
                    {
                        "x_head": np.asarray(line.get_xdata()).ravel()[:4].tolist(),
                        "y_head": np.asarray(line.get_ydata()).ravel()[:4].tolist(),
                    }
                )
            for coll in ax.collections[:2]:
                arr = coll.get_array()
                if arr is not None:
                    flat = np.asarray(arr).ravel()
                    axd["collections"].append({"head": flat[:6].tolist(), "size": int(flat.size)})
            entry["axes"].append(axd)
        if fig._suptitle:
            entry["suptitle"] = fig._suptitle.get_text()
        out.append(entry)
    return out


def test_calculate_average_in_bin_golden(viz_maps_fixture):
    m = load_monolith()
    lon_bins, lat_bins, lon_val, lat_val, bias_values, dpt_range = viz_maps_fixture
    golden = _load_json(MAPS_GOLDEN)

    avg_rmse, num_prof = m.calculate_average_in_bin(
        lon_bins, lat_bins, lon_val, lat_val, bias_values, dpt_range=dpt_range, is_rmse=True
    )
    avg_bias, num_prof_b = m.calculate_average_in_bin(
        lon_bins, lat_bins, lon_val, lat_val, bias_values, dpt_range=dpt_range, is_rmse=False
    )

    _assert_grid_close(avg_rmse, golden["avg_rmse_grid"])
    assert np.allclose(num_prof, golden["num_prof_rmse"], rtol=0, atol=TOL)
    _assert_grid_close(avg_bias, golden["avg_bias_grid"])
    assert np.allclose(num_prof_b, golden["num_prof_bias"], rtol=0, atol=TOL)


def _assert_collection_head(actual_head, expected_head):
    assert len(actual_head) == len(expected_head)
    for a, e in zip(actual_head, expected_head):
        if e is None:
            assert np.isnan(a)
        else:
            assert np.allclose(a, e, rtol=0, atol=TOL)


def test_calculate_bias_golden(viz_bias_fixture):
    m, true_values, predicted_values, gem_temp, gem_sal = viz_bias_fixture
    golden = _load_json(PROFILES_GOLDEN)

    nn_t, nn_s, gem_t, gem_s = m.calculate_bias(true_values, predicted_values, gem_temp, gem_sal)

    assert np.allclose(nn_t[:3, :2], golden["nn_t_bias_head"], rtol=0, atol=TOL)
    assert np.allclose(nn_s[:3, :2], golden["nn_s_bias_head"], rtol=0, atol=TOL)
    assert np.allclose(gem_t[:3, :2], golden["gem_temp_bias_head"], rtol=0, atol=TOL)
    assert np.allclose(gem_s[:3, :2], golden["gem_sal_bias_head"], rtol=0, atol=TOL)


def test_filter_by_season_golden():
    m = load_monolith()
    golden = _load_json(PROFILES_GOLDEN)
    dates = [735964.0, 736055.0, 736146.0, 736237.0]

    assert m.filter_by_season(list(range(4)), dates, "Winter") == golden["winter_indices"]
    assert m.filter_by_season(list(range(4)), dates, "Summer") == golden["summer_indices"]


def test_visualize_combined_results_plot_artifacts(viz_profiles_fixture):
    m, true_values, predicted_values, gem_temp, gem_sal = viz_profiles_fixture
    golden = _load_json(PLOT_GOLDEN)["visualize_combined_results"]
    sst = np.array([28.1, 27.5, 29.0, 28.3])
    ssh = np.array([0.12, -0.05, 0.33, 0.08])

    plt.close("all")
    np.random.seed(1020)
    with patch.object(m.plt, "show"):
        m.visualize_combined_results(
            true_values,
            gem_temp,
            gem_sal,
            predicted_values,
            sst,
            ssh,
            min_depth=20,
            max_depth=45,
            num_samples=1,
        )

    summary = _fig_summary()
    assert len(summary) == len(golden)
    assert summary[0]["naxes"] == golden[0]["naxes"]
    assert summary[0]["axes"][0]["title"] == golden[0]["axes"][0]["title"]
    for line, gline in zip(summary[0]["axes"][0]["lines"], golden[0]["axes"][0]["lines"]):
        assert np.allclose(line["x_head"], gline["x_head"], rtol=0, atol=TOL)
        assert np.allclose(line["y_head"], gline["y_head"], rtol=0, atol=TOL)
    assert summary[0]["suptitle"] == golden[0]["suptitle"]


def test_plot_bin_map_plot_artifacts(viz_maps_fixture):
    m = load_monolith()
    lon_bins, lat_bins, lon_val, lat_val, bias_values, dpt_range = viz_maps_fixture
    golden = _load_json(PLOT_GOLDEN)["plot_bin_map"]

    avg_rmse, num_prof = m.calculate_average_in_bin(
        lon_bins, lat_bins, lon_val, lat_val, bias_values, dpt_range=dpt_range, is_rmse=True
    )

    plt.close("all")
    with patch.object(m.plt, "show"):
        m.plot_bin_map(lon_bins, lat_bins, avg_rmse, num_prof, "Temperature", "RMSE")

    summary = _fig_summary()
    assert summary[0]["naxes"] == golden[0]["naxes"]
    _assert_collection_head(
        summary[0]["axes"][0]["collections"][0]["head"],
        golden[0]["axes"][0]["collections"][0]["head"],
    )


def test_plot_rmse_on_ax_annotations(viz_maps_fixture):
    import cartopy.crs as ccrs

    m = load_monolith()
    lon_bins, lat_bins, lon_val, lat_val, bias_values, dpt_range = viz_maps_fixture
    golden = _load_json(PLOT_GOLDEN)["plot_rmse_on_ax"]

    avg_rmse, num_prof = m.calculate_average_in_bin(
        lon_bins, lat_bins, lon_val, lat_val, bias_values, dpt_range=dpt_range, is_rmse=True
    )
    lon_centers = (lon_bins[:-1] + lon_bins[1:]) / 2
    lat_centers = (lat_bins[:-1] + lat_bins[1:]) / 2

    plt.close("all")
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    m.plot_rmse_on_ax(ax, lon_centers, lat_centers, avg_rmse, num_prof, "test title")

    assert ax.get_title() == golden["title"]
    assert len(ax.lines) == golden["nlines"]
    assert len(ax.texts) == golden["ntexts"]


def test_plot_comparison_maps_plot_artifacts(viz_maps_fixture):
    m = load_monolith()
    lon_bins, lat_bins, lon_val, lat_val, bias_values, dpt_range = viz_maps_fixture
    golden = _load_json(PLOT_GOLDEN)["plot_comparison_maps"]

    avg_rmse, _ = m.calculate_average_in_bin(
        lon_bins, lat_bins, lon_val, lat_val, bias_values, dpt_range=dpt_range, is_rmse=True
    )
    avg_bias, _ = m.calculate_average_in_bin(
        lon_bins, lat_bins, lon_val, lat_val, bias_values, dpt_range=dpt_range, is_rmse=False
    )
    lon_c = (lon_bins[:-1] + lon_bins[1:]) / 2
    lat_c = (lat_bins[:-1] + lat_bins[1:]) / 2

    plt.close("all")
    with patch.object(m.plt, "show"):
        m.plot_comparison_maps(lon_c, lat_c, avg_rmse, avg_bias, "temperature", "GEM")

    summary = _fig_summary()
    assert summary[0]["naxes"] == golden[0]["naxes"]
    _assert_collection_head(
        summary[0]["axes"][0]["collections"][0]["head"],
        golden[0]["axes"][0]["collections"][0]["head"],
    )


def test_plot_residual_profiles_for_top_bins_plot_artifacts(viz_maps_fixture):
    m = load_monolith()
    lon_bins, lat_bins, lon_val, lat_val, bias_values, dpt_range = viz_maps_fixture
    golden = _load_json(PLOT_GOLDEN)["plot_residual_profiles_for_top_bins"]

    avg_rmse, num_prof = m.calculate_average_in_bin(
        lon_bins, lat_bins, lon_val, lat_val, bias_values, dpt_range=dpt_range, is_rmse=True
    )
    np.random.seed(103)
    residuals = np.random.randn(26, 4)

    plt.close("all")
    with patch.object(m.plt, "show"):
        m.plot_residual_profiles_for_top_bins(
            lon_bins,
            lat_bins,
            lon_val,
            lat_val,
            residuals,
            avg_rmse,
            num_prof,
            "temperature",
            20,
            45,
            top_n=4,
        )

    summary = _fig_summary()
    assert summary[0]["naxes"] == golden[0]["naxes"]
    assert summary[0]["axes"][0]["title"] == golden[0]["axes"][0]["title"]
