"""ARGO profile .mat loading for NeSPReSO."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import mat73

from nespreso.utils.time import datenum_to_datetime


def load_argo_mat(data_path: str) -> tuple[dict[str, Any], list[datetime], Any, Any, Any]:
    """
    Load ARGO GoM .mat file and extract core profile metadata.

    Args:
        data_path: Path to the ARGO .mat file.

    Returns:
        Tuple of (data, TIME, LAT, LON, SH1950) where TIME is datetime-converted.
    """
    data = mat73.loadmat(data_path)
    time = [datenum_to_datetime(datenum) for datenum in data["TIME"]]
    lat = data["LAT"]
    lon = data["LON"]
    sh1950 = data["SH1950"]
    return data, time, lat, lon, sh1950
