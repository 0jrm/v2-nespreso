"""Copernicus Marine downloaders (replaces dead PODAAC OISST path)."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path


def download_sss_day(
    output_folder: str | Path,
    day: datetime,
    min_lon: float,
    max_lon: float,
    min_lat: float,
    max_lat: float,
    username: str | None = None,
    password: str | None = None,
) -> Path:
    """Download one day of CMEMS multi-sensor SSS."""
    import copernicusmarine as cm

    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    date_str = day.strftime("%Y%m%d")
    output_file = output_folder / f"SSS_{date_str}.nc"
    if output_file.exists():
        print(f"[SSS] {date_str}: exists, skipping")
        return output_file

    ds = cm.open_dataset(
        dataset_id="cmems_obs-mob_glo_phy-sss_my_multi_P1D",
        variables=["dos", "dos_error", "sea_ice_fraction", "sos", "sos_error"],
        minimum_longitude=min_lon,
        maximum_longitude=max_lon,
        minimum_latitude=min_lat,
        maximum_latitude=max_lat,
        start_datetime=day.strftime("%Y-%m-%dT00:00:00"),
        end_datetime=day.strftime("%Y-%m-%dT23:59:59"),
        username=username,
        password=password,
    )
    ds.load()
    ds.to_netcdf(output_file)
    ds.close()
    return output_file


def download_sss_range(
    output_folder: str | Path,
    start_date: datetime,
    end_date: datetime,
    min_lon: float,
    max_lon: float,
    min_lat: float,
    max_lat: float,
    username: str | None = None,
    password: str | None = None,
) -> None:
    """Day-loop SSS download with bbox filtering."""
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


def download_ostia_sst(
    output_folder: str | Path,
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
    Download OSTIA SST via copernicusmarine (replacement for dead PODAAC OISST script).

    Writes one NetCDF per day under ``output_folder/<year>/``.
    """
    import copernicusmarine as cm

    output_folder = Path(output_folder)
    current = start_date
    delta = timedelta(days=1)

    while current <= end_date:
        year_dir = output_folder / str(current.year)
        year_dir.mkdir(parents=True, exist_ok=True)
        date_str = current.strftime("%Y%m%d")
        output_file = year_dir / f"OSTIA_{date_str}.nc"
        if not output_file.exists():
            ds = cm.open_dataset(
                dataset_id="cmems_obs-glo_phy-sst_my_0.25deg_P1D",
                variables=["analysed_sst", "analysis_error"],
                minimum_longitude=min_lon,
                maximum_longitude=max_lon,
                minimum_latitude=min_lat,
                maximum_latitude=max_lat,
                start_datetime=current.strftime("%Y-%m-%dT00:00:00"),
                end_datetime=current.strftime("%Y-%m-%dT23:59:59"),
                username=username,
                password=password,
            )
            ds.load()
            ds.to_netcdf(output_file)
            ds.close()
            print(f"[OSTIA] saved {output_file}")
        current += delta
