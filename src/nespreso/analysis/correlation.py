"""Correlation helpers hoisted from the monolith __main__ block."""

from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr


def calculate_correlation(observation: np.ndarray, prediction: np.ndarray) -> float:
    """
    Calculate the Pearson correlation coefficient between two 2D matrices,
    ignoring positions with NaNs in the observation matrix.

    Args:
    - observation (np.array): 2D array of observed data with NaNs for missing values.
    - prediction (np.array): 2D array of predicted data.

    Returns:
    - float: Pearson correlation coefficient, or NaN if it cannot be calculated.
    """
    # Flatten the arrays to 1D
    obs_flat = observation.flatten()
    pred_flat = prediction.flatten()

    # Create a mask for non-NaN values
    valid_mask = ~np.isnan(obs_flat)

    # Filter both arrays to include only the valid (non-NaN) values
    valid_obs = obs_flat[valid_mask]
    valid_pred = pred_flat[valid_mask]

    # Calculate the Pearson correlation coefficient on the non-NaN values
    if valid_obs.size == 0:
        return np.nan  # Return NaN if no valid observations
    correlation, _ = pearsonr(valid_obs, valid_pred)

    return correlation
