"""Satellite data readers ported from eoas_pyutils/io_utils/coaps_io_data.py."""

from __future__ import annotations

import os
from datetime import datetime
from os.path import join

import numpy as np
import xarray as xr


def get_aviso_by_month(aviso_folder: str, c_date, bbox=None):
    """Read AVISO monthly data for a given date, optionally cropped to bbox."""
    aviso_file_name = join(aviso_folder, f"{c_date.year}-{c_date.month:02d}.nc")
    aviso_data = xr.open_dataset(aviso_file_name)

    if bbox is not None:
        aviso_data = aviso_data.sel(
            latitude=slice(bbox[0], bbox[1]),
            longitude=slice(bbox[2], bbox[3]),
        )

    lats = aviso_data.latitude
    lons = aviso_data.longitude
    return aviso_data, lats, lons


def get_aviso_by_date(aviso_folder: str, c_date, bbox=None):
    """Read AVISO data for a specified date, optionally cropped to bbox."""
    alternative_folder = "/home/jmiranda/Data/SSH/SEALEVEL_GLO_PHY_L4_NRT_008_046/"
    standard_format = join(aviso_folder, f"{c_date.year}-{c_date.month:02d}.nc")
    alternative_format = f"nrt_global_allsat_phy_l4_{c_date.strftime('%Y%m%d')}"

    try:
        aviso_data = xr.open_dataset(standard_format)
    except FileNotFoundError:
        files = os.listdir(alternative_folder)
        matching_files = [file for file in files if alternative_format in file]

        if not matching_files:
            raise FileNotFoundError(f"No files found for date {c_date} in {alternative_format}")

        try:
            aviso_data = xr.open_dataset(join(alternative_folder, matching_files[0]))
        except Exception as exc:
            raise RuntimeError(f"Could not load AVISO data for {c_date}, {matching_files[0]}") from exc

    if bbox is not None:
        target_time = np.datetime64(c_date)
        time_diff = np.abs(aviso_data["time"] - target_time)
        closest_index = np.argmin(time_diff.values)
        aviso_data = aviso_data.sel(
            time=aviso_data["time"][closest_index],
            latitude=slice(bbox[0], bbox[1]),
            longitude=slice(bbox[2], bbox[3]),
        )

    lats = aviso_data.latitude
    lons = aviso_data.longitude
    return aviso_data, lats, lons


def get_sst_by_date(sst_folder: str, c_date, bbox=None, sst_file_name=None):
    """Read SST for a single day, optionally cropped to bbox."""
    c_date_str = c_date.strftime("%Y%m%d")
    if sst_file_name is None:
        sst_file_name = join(
            sst_folder,
            str(c_date.year),
            f"{c_date_str}090000-JPL-L4_GHRSST-SSTfnd-MUR-GLOB-v02.0-fv04.1_subset.nc",
        )
    try:
        sst_data = xr.open_dataset(sst_file_name)
    except Exception:
        sst_file_name = join(
            sst_folder,
            str(c_date.year),
            f"{c_date_str}090000-JPL-L4_GHRSST-SSTfnd-MUR-GLOB-v02.0-fv04.1.nc",
        )
        try:
            sst_data = xr.open_dataset(sst_file_name)
        except Exception as exc:
            raise RuntimeError(f"Could not load SST data for {c_date}, {sst_file_name}") from exc

    if bbox is not None:
        sst_data = sst_data.sel(lat=slice(bbox[0], bbox[1]), lon=slice(bbox[2], bbox[3]))

    lats = sst_data.lat
    lons = sst_data.lon
    return sst_data, lats, lons


def get_sss_by_date(sss_folder: str, c_date, bbox=None):
    """Read salinity for a single day, optionally cropped to bbox."""
    day_of_year = (c_date - datetime(c_date.year, 1, 1)).days + 1

    sss_file_name = join(
        sss_folder,
        str(c_date.year),
        f"RSS_smap_SSS_L3_8day_running_{c_date.year}_{day_of_year:03d}_FNL_v05.0.nc",
    )
    try:
        sss_data = xr.open_dataset(sss_file_name)
    except Exception:
        sss_file_name = join(
            sss_folder,
            str(c_date.year),
            f"RSS_smap_SSS_L3_8day_running_{c_date.year}_{day_of_year:03d}_FNL_v06.0.nc",
        )
        try:
            sss_data = xr.open_dataset(sss_file_name)
        except Exception as exc:
            raise RuntimeError(f"Could not load SSS data for {c_date}, {sss_file_name}") from exc

    if bbox is not None:
        sss_data = sss_data.sel(
            lat=slice(bbox[0], bbox[1]),
            lon=slice((bbox[2] + 360) % 360, (bbox[3] + 360) % 360),
        )

    lats = sss_data.lat
    lons = np.where(sss_data.lon > 180, sss_data.lon - 360, sss_data.lon)
    return sss_data, lats, lons
