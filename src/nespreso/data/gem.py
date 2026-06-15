"""GEM baseline profile fitting and evaluation."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from numpy.polynomial.polynomial import Polynomial
from scipy.stats import linregress

from nespreso.utils.time import datenum_to_datetime


def calc_gem(
    dataset: Any,
    ignore_indices: np.ndarray | list[int],
    degree: int = 7,
    sat_ssh: bool = False,
) -> None:
    """
    Calculates this dataset's polyfits for the GEM profiles for each month.

    Args:
    - degree: Degree of the polynomial fit. Default is 7.
    - sat_ssh: Flag to use satellite SSH instead of profile SSH. Uses measured SSH as default.

    Returns:
    - nothing, but saves the polyfits in the attributes `dataset.gem_T_polyfits` and `dataset.gem_S_polyfits` for each month.
    """

    dataset.pressure_grid = np.arange(dataset.min_depth, dataset.max_depth + 1)
    # Initialize dictionaries to hold polyfits for each month
    dataset.gem_T_polyfits = {}
    dataset.gem_S_polyfits = {}

    mask = np.ones(len(dataset.SH1950), dtype=bool)
    mask[ignore_indices] = False

    steric_height = dataset.SH1950[mask]
    SSH = dataset.AVISO_ADT[mask]
    TEMP = dataset.TEMP[:, mask]
    SAL = dataset.SAL[:, mask]
    TIME = np.array(dataset.TIME)[mask]  # Apply mask to TIME

    dataset.gem_slope, dataset.gem_intercept, _, _, _ = linregress(steric_height, SSH)

    sort_idx = np.argsort(steric_height)

    sh_sorted = (
        steric_height[sort_idx] + np.arange(len(steric_height)) * 1e-10
    )  # add small number to avoid duplicate values
    temp_sorted = TEMP[:, sort_idx]
    sal_sorted = SAL[:, sort_idx]
    time_sorted = TIME[sort_idx]  # Sort TIME based on steric_height sorting

    # Convert sorted TIME to months
    months_sorted = [int((datenum_to_datetime(datenum).month - 1) / 3) for datenum in time_sorted]

    # Start time
    start_time = time.perf_counter()

    # Iterate over each month
    for month in set(months_sorted):
        dataset.gem_T_polyfits[month] = []
        dataset.gem_S_polyfits[month] = []

        # Indices for the current month
        month_indices = [i for i, m in enumerate(months_sorted) if m == month]

        # For each pressure level
        for i, p in enumerate(dataset.pressure_grid):
            # Filter data for the current month
            TEMP_at_p = temp_sorted[i, month_indices]
            SAL_at_p = sal_sorted[i, month_indices]
            sh_at_p = sh_sorted[month_indices]

            # Polynomial fit for the current month
            TEMP_polyfit = Polynomial.fit(sh_at_p, TEMP_at_p, degree)
            SAL_polyfit = Polynomial.fit(sh_at_p, SAL_at_p, degree)

            # Append the polynomial fit to the lists for the current month
            dataset.gem_T_polyfits[month].append(TEMP_polyfit)
            dataset.gem_S_polyfits[month].append(SAL_polyfit)

    end_time = time.perf_counter()
    # Calculate elapsed time
    elapsed_time = end_time - start_time
    print(f"GEM fit: {elapsed_time:.2f} seconds.")

    return


def get_gem_profiles(
    dataset: Any,
    indices: np.ndarray | list[int],
    sat_ssh: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generates the GEM profiles for the given indices, month by month.

    Args:
    - indices (list or numpy.ndarray): Indices for which profiles are needed.
    - sat_ssh (bool): Flag to use satellite SSH instead of profile SSH. Uses measured SSH as default.

    Returns:
    - numpy.ndarray: concatenated temperature and salinity profiles in the required format for visualization.
    """

    # Initialize arrays to hold GEM profiles
    temp_GEM = np.empty((len(indices), dataset.max_depth + 1 - dataset.min_depth))
    sal_GEM = np.empty((len(indices), dataset.max_depth + 1 - dataset.min_depth))
    temp_GEM[:] = np.nan  # Initialize with NaNs
    sal_GEM[:] = np.nan

    for idx, index in enumerate(indices):
        # Determine the month for the current index
        month = int((datenum_to_datetime(dataset.TIME[index]).month - 1) / 3)

        # Check if there are polyfits for this month
        if month not in dataset.gem_T_polyfits:
            continue  # Skip if no polyfits available for the month

        # Select SSH based on the sat_ssh flag
        if sat_ssh:
            ssh = (dataset.AVISO_ADT[index] - dataset.gem_intercept) / dataset.gem_slope
        else:
            ssh = dataset.SH1950[index]

        # For each pressure level
        for i, p in enumerate(dataset.pressure_grid):
            # Evaluate the fitted polynomials at the given SSH value
            temp_GEM[idx, i] = dataset.gem_T_polyfits[month][i](ssh)
            sal_GEM[idx, i] = dataset.gem_S_polyfits[month][i](ssh)

    # Interpolate missing values in temp_GEM and sal_GEM (same as before)
    for array in [temp_GEM, sal_GEM]:
        for row in range(array.shape[0]):
            valid_mask = ~np.isnan(array[row])
            if not valid_mask.any():  # skip rows with only NaNs
                continue

            array[row] = np.interp(np.arange(array.shape[1]), np.where(valid_mask)[0], array[row, valid_mask])

            # If NaNs at the start, fill with the first non-NaN value
            first_valid_idx = valid_mask.argmax()
            array[row, :first_valid_idx] = array[row, first_valid_idx]

            # If NaNs at the end, fill with the last non-NaN value
            last_valid_idx = len(array[row]) - valid_mask[::-1].argmax() - 1
            array[row, last_valid_idx + 1 :] = array[row, last_valid_idx]

    return temp_GEM, sal_GEM
