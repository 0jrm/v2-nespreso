"""
Pin helpers hoisted for Phase 8 experiments (season/month/reporting).
"""

from __future__ import annotations

from datetime import datetime

import pytest

from nespreso.analysis.monthly import count_profiles_per_month
from nespreso.utils.time import get_month, get_season


def test_get_season_northern_hemisphere():
    assert get_season(datetime(2020, 3, 15)) == "Spring"
    assert get_season(datetime(2020, 7, 1)) == "Summer"
    assert get_season(datetime(2020, 10, 1)) == "Autumn"
    assert get_season(datetime(2020, 1, 1)) == "Winter"


def test_get_month_datetime_and_datenum():
    assert get_month(datetime(2020, 8, 15)) == 8
    from nespreso.utils.time import datenum_to_datetime

    august_datenum = 736863.0151  # golden glider timestamp (August)
    assert get_month(august_datenum) == datenum_to_datetime(august_datenum).month


def test_count_profiles_per_month_synthetic():
    import numpy as np

    class _Dataset:
        TIME = np.array([736863.0151, 736863.0151, 737142.9568])

    counts = count_profiles_per_month(_Dataset(), np.array([0, 1, 2]))
    assert int(counts.sum()) == 3
    assert len(counts) == 12
