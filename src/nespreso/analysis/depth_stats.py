"""Depth-bin statistics helpers hoisted from the monolith __main__ block."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from nespreso.analysis.correlation import calculate_correlation


def average_depth(targets: np.ndarray, depths: np.ndarray) -> float:
    return np.nansum(targets.T * depths) / np.nansum(targets)


def histogram_available_depths(targets: np.ndarray) -> np.ndarray:
    # counts all the available depths from all profiles
    return np.sum(-1 * (np.isnan(targets) - 1), axis=1)


def _bin_predictions_at_1m(predictions, depths, step, bin_index):
    start = depths[bin_index]
    indices = np.arange(start, start + step, 1).astype(int)
    indices = indices[indices < predictions.shape[0]]
    if indices.size == 0:
        return predictions[:0, :]
    return predictions[indices, :]


def _predictions_for_correlation(predictions, targets, depths, step):
    """Downsample 1 m predictions to match 5 m binned targets when needed."""
    if targets.shape[0] != len(depths) or predictions.shape[0] == targets.shape[0]:
        return predictions

    pred_binned = np.full_like(targets, np.nan, dtype=float)
    for i in range(len(depths) - 1):
        bin_predictions = _bin_predictions_at_1m(predictions, depths, step, i)
        if bin_predictions.size:
            pred_binned[i, :] = np.nanmean(bin_predictions, axis=0)
    return pred_binned


def equivalent_average_statistic(
    predictions: np.ndarray,
    targets: np.ndarray,
    count: np.ndarray,
    depths: np.ndarray,
    function: Callable[[np.ndarray, np.ndarray], float],
) -> tuple[float, float]:
    """
    Adjusts the calculation of an average statistic (e.g., RMSE or bias) to account for the
    depth binning of the primary dataset and uses the histogram of valid measurements to weight
    these statistics.

    :param predictions: 2D array of predictions with depth along axis 0 (1 m resolution).
    :param targets: 2D array of targets with depth along axis 0. May be 1 m or 5 m binned.
    :param count: Array of counts of valid measurements at each depth bin.
    :param depths: Depth bins corresponding to the histogram (e.g., every 5 meters).
    :param function: The statistical function to use (e.g., rmse or bias).
    :return: The weighted average statistic across the depth bins.
    """
    stats_per_bin = np.zeros(len(depths))
    step = depths[1] - depths[0]
    targets_are_binned = targets.shape[0] == len(depths) and predictions.shape[0] != targets.shape[0]

    for i in range(len(depths) - 1):
        bin_predictions = _bin_predictions_at_1m(predictions, depths, step, i)
        if bin_predictions.size == 0:
            continue

        if targets_are_binned:
            bin_targets = np.broadcast_to(targets[i, :], bin_predictions.shape)
        else:
            indices = np.arange(depths[i], depths[i] + step, 1).astype(int)
            indices = indices[indices < targets.shape[0]]
            bin_targets = targets[indices, :]

        stats_per_bin[i] = function(bin_predictions, bin_targets)

    weighted_stat = np.nansum(stats_per_bin * count) / np.sum(count)
    corr_stat = calculate_correlation(
        targets,
        _predictions_for_correlation(predictions, targets, depths, step),
    )

    return weighted_stat, corr_stat
