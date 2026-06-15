"""Unit tests for satellite data loading helpers."""

from datetime import datetime
from unittest.mock import MagicMock, patch

import numpy as np

from nespreso.io.satellite import load_satellite_data, load_satellite_data_for_dataset


def _mock_sss_reader(*_args, **_kwargs):
    sss = MagicMock()
    sss.sss_smap_40km.values = np.full((2, 2), 35.0)
    sss.filename = "sss.nc"
    lats = np.array([20.0, 30.0])
    lons = np.array([-95.0, -85.0])
    return sss, lats, lons


def _mock_aviso_reader(*_args, **_kwargs):
    aviso = MagicMock()
    aviso.adt.values = np.full((2, 2), 0.5)
    lats = np.array([20.0, 30.0])
    lons = np.array([-95.0, -85.0])
    return aviso, lats, lons


def _mock_sst_reader(*_args, **_kwargs):
    sst = MagicMock()
    sst.analysed_sst.values = np.full((1, 2, 2), 300.0)
    lats = np.array([20.0, 30.0])
    lons = np.array([-95.0, -85.0])
    return sst, lats, lons


@patch("nespreso.io.satellite.get_sst_by_date", side_effect=_mock_sst_reader)
@patch("nespreso.io.satellite.get_aviso_by_date", side_effect=_mock_aviso_reader)
@patch("nespreso.io.satellite.get_sss_by_date", side_effect=_mock_sss_reader)
def test_load_satellite_data_returns_aligned_arrays(_sss, _aviso, _sst):
    c_date = datetime(2015, 6, 1)
    time = [c_date, c_date]
    lat = np.array([25.0, 26.0])
    lon = np.array([-90.0, -89.0])

    sss, sst, aviso = load_satellite_data(time, lat, lon)

    assert sss.shape == (2,)
    assert sst.shape == (2,)
    assert aviso.shape == (2,)
    assert np.all(np.isfinite(sss))
    assert np.all(np.isfinite(sst))
    assert np.all(np.isfinite(aviso))


@patch("nespreso.io.satellite.tqdm", side_effect=lambda x, **kwargs: x)
@patch("nespreso.io.satellite.get_sst_by_date", side_effect=_mock_sst_reader)
@patch("nespreso.io.satellite.get_aviso_by_date", side_effect=_mock_aviso_reader)
@patch("nespreso.io.satellite.get_sss_by_date", side_effect=_mock_sss_reader)
def test_load_satellite_data_for_dataset_returns_aligned_arrays(_sss, _aviso, _sst, _tqdm):
    c_date = datetime(2015, 6, 1)
    time = [c_date, c_date]
    lat = np.array([25.0, 26.0])
    lon = np.array([-90.0, -89.0])

    sss, sst, aviso = load_satellite_data_for_dataset(
        time,
        lat,
        lon,
        aviso_folder="/aviso",
        sst_folder="/sst",
        sss_folder="/sss",
        min_lat=18.0,
        max_lat=31.0,
        min_lon=-98.0,
        max_lon=-81.0,
        ex_lat=23.0,
        ex_lon=-90.0,
        debug=False,
    )

    assert sss.shape == (2,)
    assert sst.shape == (2,)
    assert aviso.shape == (2,)
    assert np.all(np.isfinite(sss))
    assert np.all(np.isfinite(sst))
    assert np.all(np.isfinite(aviso))
