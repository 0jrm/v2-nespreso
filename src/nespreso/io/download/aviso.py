"""AVISO SSH downloader with year/month + bbox filtering."""

from __future__ import annotations

import os
import subprocess
from calendar import monthrange
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DownloadBbox:
    """Bounding box in degrees (lat/lon). Longitude may be 0-360 for CMEMS."""

    min_lon: float
    max_lon: float
    min_lat: float
    max_lat: float


def download_aviso(
    output_folder: str | Path,
    start_year: int,
    end_year: int,
    bbox: DownloadBbox,
    username: str | None = None,
    password: str | None = None,
    months: range | None = None,
) -> None:
    """
    Download AVISO SSH monthly files year-by-year (ported from eoas_pyutils).

    Credentials default to ~/.netrc ``AVISO`` host entry when not provided.
    """
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    if username is None or password is None:
        import netrc

        secrets = netrc.netrc()
        username, _, password = secrets.hosts["AVISO"]

    month_range = months if months is not None else range(1, 13)

    for year in range(start_year, end_year + 1):
        for month in month_range:
            outfile = output_folder / f"{year}-{month:02d}.nc"
            if outfile.exists():
                print(f"[AVISO] {year}-{month:02d}: exists, skipping")
                continue

            if year < 2022:
                args = (
                    f'--motu http://my.cmems-du.eu/motu-web/Motu '
                    f'--service-id SEALEVEL_GLO_PHY_L4_MY_008_047-TDS '
                    f'--product-id cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.25deg_P1D '
                    f'--longitude-min {bbox.min_lon} --longitude-max {bbox.max_lon} '
                    f'--latitude-min {bbox.min_lat} --latitude-max {bbox.max_lat} '
                    f'--date-min "{year}-{month:02d}-01 00:00:00" '
                    f'--date-max "{year}-{month:02d}-{monthrange(year, month)[1]} 00:00:00" '
                    f'--variable sla --variable adt --variable ugos --variable vgos '
                    f'--variable ugosa --variable vgosa --variable err_sla '
                    f'--out-dir {output_folder} --out-name {outfile.name} '
                    f'--user {username} --pwd {password}'
                )
            else:
                args = (
                    f'--motu http://nrt.cmems-du.eu/motu-web/Motu '
                    f'--service-id SEALEVEL_GLO_PHY_L4_NRT_OBSERVATIONS_008_046-TDS '
                    f'--product-id dataset-duacs-nrt-global-merged-allsat-phy-l4 '
                    f'--longitude-min {bbox.min_lon} --longitude-max {bbox.max_lon} '
                    f'--latitude-min {bbox.min_lat} --latitude-max {bbox.max_lat} '
                    f'--date-min "{year}-{month:02d}-01 00:00:00" '
                    f'--date-max "{year}-{month:02d}-{monthrange(year, month)[1]} 00:00:00" '
                    f'--variable adt --variable err_sla --variable err_ugosa --variable err_vgosa '
                    f'--variable flag_ice --variable sla --variable ugos --variable ugosa '
                    f'--variable vgos --variable vgosa '
                    f'--out-dir {output_folder} --out-name {outfile.name} '
                    f'--user {username} --pwd {password}'
                )

            print(f"[AVISO] Downloading {outfile.name}")
            subprocess.call(f"motuclient {args}", shell=True)
