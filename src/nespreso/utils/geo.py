"""Geospatial helpers hoisted from the monolith __main__ block."""

from __future__ import annotations

import numpy as np


def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance in kilometers between two points
    on the earth (specified in decimal degrees)
    """
    # Convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371  # Radius of earth in kilometers. Use 3956 for miles
    return c * r


def calculate_distances(latitudes, longitudes):
    """Calculate the cumulative distance between successive lat/long pairs."""
    n = len(latitudes)
    distances = np.zeros(n)
    for i in range(1, n):
        distances[i] = distances[i - 1] + haversine(
            latitudes[i - 1], longitudes[i - 1], latitudes[i], longitudes[i]
        )
    return distances
