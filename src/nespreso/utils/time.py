"""MATLAB datenum conversion (canonical version; both monolith variants unified)."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np


def datenum_to_datetime(matlab_datenum: float) -> datetime:
    """
    Convert MATLAB datenum to Python datetime.

    MATLAB's datenum (1) is January 1, year 0000; Python's minimum year is 1.
    Year 0 is a leap year in the proleptic ISO calendar (+366 day correction).
    """
    days_from_year_0_to_year_1 = 366
    return datetime.fromordinal(int(matlab_datenum) - days_from_year_0_to_year_1) + timedelta(days=matlab_datenum % 1)


def matlab2datetime(matlab_datenum: float) -> datetime:
    """Alias preserved for call sites that used the second monolith variant."""
    return datenum_to_datetime(matlab_datenum)


def get_month(t: datetime | float) -> int:
    """Return calendar month from a datetime or MATLAB datenum."""
    return t.month if hasattr(t, "month") else datenum_to_datetime(t).month


def get_season(date: datetime) -> str:
    """Map a datetime to meteorological season (Northern Hemisphere)."""
    month = date.month
    if month in [3, 4, 5]:
        return "Spring"
    if month in [6, 7, 8]:
        return "Summer"
    if month in [9, 10, 11]:
        return "Autumn"
    return "Winter"


def datenums_to_datetimes(matlab_datenums: np.ndarray) -> list[datetime]:
    """
    Convert an array of MATLAB datenum values to Python datetime objects.

    Parameters:
    matlab_datenums (np.array): Array of MATLAB datenum values

    Returns:
    list: List of Python datetime objects
    """
    python_datetimes = [datenum_to_datetime(datenum) for datenum in matlab_datenums]
    return python_datetimes
