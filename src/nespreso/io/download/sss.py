"""SSS SMAP download placeholder — delegates to CMEMS multi-sensor product."""

from __future__ import annotations

from datetime import datetime, timedelta

from nespreso.io.download.copernicus import download_sss_day


def download_sss_smap(
    output_folder: str,
    start_date: datetime,
    end_date: datetime,
    min_lon: float,
    max_lon: float,
    min_lat: float,
    max_lat: float,
    username: str | None = None,
    password: str | None = None,
) -> None:
    """
    Day-loop SSS download with bbox filtering.

    Uses copernicusmarine (SMAP RSS local files remain readable via
  ``satellite_readers.get_sss_by_date`` when already on disk).
    """
    current = start_date
    while current <= end_date:
        download_sss_day(
            output_folder,
            current,
            min_lon,
            max_lon,
            min_lat,
            max_lat,
            username=username,
            password=password,
        )
        current += timedelta(days=1)
