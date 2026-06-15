"""Date/month + bbox filtered satellite downloaders."""

from nespreso.io.download.aviso import download_aviso
from nespreso.io.download.copernicus import download_ostia_sst, download_ssh_year, download_sss_day
from nespreso.io.download.sss import download_sss_smap

__all__ = [
    "download_aviso",
    "download_ostia_sst",
    "download_ssh_year",
    "download_sss_day",
    "download_sss_smap",
]
