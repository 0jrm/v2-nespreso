"""Depth-interval and seasonal method comparison helpers from the monolith."""

from __future__ import annotations

import numpy as np

from nespreso.analysis.correlation import calculate_correlation
from nespreso.metrics import bias, rmse


def isop_depth_indices(isop_depths, min_d, max_d):
    return np.where((isop_depths >= min_d) & (isop_depths <= max_d))[0]


def default_depth_intervals(min_depth, max_depth):
    return [
        (min_depth, 20),
        (20, 100),
        (100, 200),
        (200, 500),
        (500, 1000),
        (1000, max_depth),
        (0, 1000),
        (min_depth, max_depth),
    ]


def compute_season_masked_depth_rmse_bias(residual, season_mask):
    rmse_by_depth = np.sqrt(np.nanmean((residual[:, season_mask]) ** 2, axis=1))
    bias_by_depth = np.nanmean(residual[:, season_mask], axis=1)
    return rmse_by_depth, bias_by_depth


def compute_depth_interval_metrics(
    min_d,
    max_d,
    isop_depths,
    ist_rmse_values,
    ist_bias_values,
    iss_rmse_values,
    iss_bias_values,
    original_profiles,
    pred_T,
    pred_S,
    gem_temp,
    gem_sal,
    temp_MLR_profiles,
    sal_MLR_profiles,
    correlation_fn=calculate_correlation,
):
    i_isop_dpt = isop_depth_indices(isop_depths, min_d, max_d)
    calc_depths = isop_depths[i_isop_dpt].astype(int)
    ori_t = original_profiles[calc_depths, 0, :]
    ori_s = original_profiles[calc_depths, 1, :]
    nn_t = pred_T[calc_depths, :]
    nn_s = pred_S[calc_depths, :]
    gem_t = gem_temp[:, calc_depths].T
    gem_s = gem_sal[:, calc_depths].T
    mlr_t = temp_MLR_profiles[calc_depths, :]
    mlr_s = sal_MLR_profiles[calc_depths, :]

    nn_t_rmse = rmse(nn_t, ori_t)
    gem_t_rmse = rmse(gem_t, ori_t)
    nn_t_bias = bias(nn_t, ori_t)
    gem_t_bias = bias(gem_t, ori_t)
    nn_s_rmse = rmse(nn_s, ori_s)
    gem_s_rmse = rmse(gem_s, ori_s)
    nn_s_bias = bias(nn_s, ori_s)
    gem_s_bias = bias(gem_s, ori_s)

    mlr_t_rmse = rmse(mlr_t, ori_t)
    mlr_t_bias = bias(mlr_t, ori_t)
    mlr_s_rmse = rmse(mlr_s, ori_s)
    mlr_s_bias = bias(mlr_s, ori_s)

    isop_avg_t_rmse = np.mean(ist_rmse_values[i_isop_dpt])
    isop_avg_t_bias = np.mean(ist_bias_values[i_isop_dpt])
    isop_avg_s_rmse = np.mean(iss_rmse_values[i_isop_dpt])
    isop_avg_s_bias = np.mean(iss_bias_values[i_isop_dpt])

    nn_T_corr = correlation_fn(nn_t, ori_t)
    gem_T_corr = correlation_fn(gem_t, ori_t)
    nn_S_corr = correlation_fn(nn_s, ori_s)
    gem_S_corr = correlation_fn(gem_s, ori_s)
    mlr_T_corr = correlation_fn(mlr_t, ori_t)
    mlr_S_corr = correlation_fn(mlr_s, ori_s)

    return {
        "min_d": min_d,
        "max_d": max_d,
        "nn_t_rmse": nn_t_rmse,
        "gem_t_rmse": gem_t_rmse,
        "mlr_t_rmse": mlr_t_rmse,
        "isop_avg_t_rmse": isop_avg_t_rmse,
        "nn_t_bias": nn_t_bias,
        "gem_t_bias": gem_t_bias,
        "mlr_t_bias": mlr_t_bias,
        "isop_avg_t_bias": isop_avg_t_bias,
        "nn_s_rmse": nn_s_rmse,
        "gem_s_rmse": gem_s_rmse,
        "mlr_s_rmse": mlr_s_rmse,
        "isop_avg_s_rmse": isop_avg_s_rmse,
        "nn_s_bias": nn_s_bias,
        "gem_s_bias": gem_s_bias,
        "mlr_s_bias": mlr_s_bias,
        "isop_avg_s_bias": isop_avg_s_bias,
        "nn_T_corr": nn_T_corr,
        "gem_T_corr": gem_T_corr,
        "mlr_T_corr": mlr_T_corr,
        "nn_S_corr": nn_S_corr,
        "gem_S_corr": gem_S_corr,
        "mlr_S_corr": mlr_S_corr,
    }
