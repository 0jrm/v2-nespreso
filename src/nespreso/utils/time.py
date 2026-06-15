"""MATLAB datenum conversion (canonical version; both monolith variants unified)."""

from __future__ import annotations

from datetime import datetime, timedelta


def datenum_to_datetime(matlab_datenum: float) -> datetime:
    """
    Convert MATLAB datenum to Python datetime.

    MATLAB's datenum (1) is January 1, year 0000; Python's minimum year is 1.
    Year 0 is a leap year in the proleptic ISO calendar (+366 day correction).
    """
    days_from_year_0_to_year_1 = 366
    return datetime.fromordinal(int(matlab_datenum) - days_from_year_0_to_year_1) + timedelta(
        days=matlab_datenum % 1
    )


def matlab2datetime(matlab_datenum: float) -> datetime:
    """Alias preserved for call sites that used the second monolith variant."""
    return datenum_to_datetime(matlab_datenum)
