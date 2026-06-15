"""Monthly profile-count helpers hoisted from the monolith __main__ block."""

from __future__ import annotations

from datetime import datetime

import pandas as pd


def count_profiles_per_month(dataset, indices):
    dates = [datetime.fromordinal(int(d)) for d in dataset.TIME[indices]]
    frame = pd.DataFrame({"date": dates})
    return frame.groupby(frame["date"].dt.month).size().reindex(range(1, 13), fill_value=0)
