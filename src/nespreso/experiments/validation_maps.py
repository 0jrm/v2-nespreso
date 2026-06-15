"""RMSE/bias maps, seasonal depth curves, and comparison maps (hoisted from monolith __main__)."""

from __future__ import annotations

from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np

from nespreso.analysis import compute_season_masked_depth_rmse_bias
from nespreso.experiments.validation_context import ValidationContext
from nespreso.utils.time import get_season
from nespreso.viz.maps import calculate_average_in_bin, plot_bin_map, plot_comparison_maps


def run_validation_maps(ctx: ValidationContext) -> None:
    """Plot binned RMSE/bias maps, seasonal depth curves, and method comparison maps."""
    isop_depths = ctx.isop_depths
    lon_centers = ctx.lon_centers
    lat_centers = ctx.lat_centers
    lon_val = ctx.lon_val
    lat_val = ctx.lat_val
    pred_T_resid = ctx.pred_T_resid
    pred_S_resid = ctx.pred_S_resid
    gems_T_resid = ctx.gems_T_resid
    gems_S_resid = ctx.gems_S_resid
    old_T_resid = ctx.old_T_resid
    old_S_resid = ctx.old_S_resid
    lon_bins = ctx.lon_bins
    lat_bins = ctx.lat_bins
    dates_val = ctx.dates_val
    data_ISOP = ctx.data_ISOP

    avg_rmse_isop_t = data_ISOP["t_rmse_syn"]
    avg_rmse_isop_s = data_ISOP["s_rmse_syn"]
    avg_bias_isop_t = data_ISOP["t_bias_syn"]
    avg_bias_isop_s = data_ISOP["s_bias_syn"]

        # Heatmaps of the RMSE
    # dpt_range = np.arange(0,201)
    upper_limit = 0
    lower_limit = 1800
    dpt_range = isop_depths[(isop_depths <= lower_limit) & (isop_depths >= upper_limit)].astype(int)

    print(f"Statistics/Plots for Depth range: [{lower_limit}, {upper_limit}]")

    # Calculate average temperature RMSE for NN and GEM !
    grid_avg_temp_rmse_nn, num_prof_nn = calculate_average_in_bin(
        lon_centers, lat_centers, lon_val, lat_val, pred_T_resid, dpt_range, is_rmse=True
    )
    grid_avg_temp_rmse_gem, num_prof_gem = calculate_average_in_bin(
        lon_centers, lat_centers, lon_val, lat_val, gems_T_resid, dpt_range, is_rmse=True
    )
    grid_avg_temp_rmse_gain = grid_avg_temp_rmse_nn - grid_avg_temp_rmse_gem
    # same for NeSPReSO 1.0
    grid_avg_temp_rmse_mlr, num_prof_mlr = calculate_average_in_bin(
        lon_centers, lat_centers, lon_val, lat_val, old_T_resid, dpt_range, is_rmse=True
    )
    grid_avg_temp_rmse_gain_mlr = grid_avg_temp_rmse_nn - grid_avg_temp_rmse_mlr

    plot_bin_map(lon_bins, lat_bins, grid_avg_temp_rmse_nn, num_prof_nn, "Temperature", "RMSE")

    # now let's do the same for salinity
    # Calculate average temperature RMSE for NN and GEM
    grid_avg_sal_rmse_nn, num_prof_nn = calculate_average_in_bin(
        lon_centers, lat_centers, lon_val, lat_val, pred_S_resid, dpt_range, is_rmse=True
    )
    grid_avg_sal_rmse_gem, num_prof_gem = calculate_average_in_bin(
        lon_centers, lat_centers, lon_val, lat_val, gems_S_resid, dpt_range, is_rmse=True
    )
    grid_avg_sal_rmse_gain = grid_avg_sal_rmse_nn - grid_avg_sal_rmse_gem
    # same for NeSPReSO 1.0
    grid_avg_sal_rmse_mlr, num_prof_mlr = calculate_average_in_bin(
        lon_centers, lat_centers, lon_val, lat_val, old_S_resid, dpt_range, is_rmse=True
    )
    grid_avg_sal_rmse_gain_mlr = grid_avg_sal_rmse_nn - grid_avg_sal_rmse_mlr

    plot_bin_map(lon_bins, lat_bins, grid_avg_sal_rmse_nn, num_prof_nn, "Salinity", "RMSE")

    # same maps, but bias

    avg_nn_t_bias, num_prof_nn = calculate_average_in_bin(
        lon_centers, lat_centers, lon_val, lat_val, pred_T_resid, dpt_range, is_rmse=False
    )
    avg_nn_s_bias, num_prof_nn = calculate_average_in_bin(
        lon_centers, lat_centers, lon_val, lat_val, pred_S_resid, dpt_range, is_rmse=False
    )
    avg_gem_t_bias, num_prof_gem = calculate_average_in_bin(
        lon_centers, lat_centers, lon_val, lat_val, gems_T_resid, dpt_range, is_rmse=False
    )
    avg_gem_s_bias, num_prof_gem = calculate_average_in_bin(
        lon_centers, lat_centers, lon_val, lat_val, gems_S_resid, dpt_range, is_rmse=False
    )
    # same for NeSPReSO 1.0
    avg_mlr_t_bias, num_prof_mlr = calculate_average_in_bin(
        lon_centers, lat_centers, lon_val, lat_val, old_T_resid, dpt_range, is_rmse=False
    )
    avg_mlr_s_bias, num_prof_mlr = calculate_average_in_bin(
        lon_centers, lat_centers, lon_val, lat_val, old_S_resid, dpt_range, is_rmse=False
    )

    # TODO: fix the bias color scale (negative values are not being shown properly)
    plot_bin_map(lon_bins, lat_bins, avg_nn_t_bias, num_prof_nn, "Temperature", "Bias")
    plot_bin_map(lon_bins, lat_bins, avg_nn_s_bias, num_prof_nn, "Salinity", "Bias")

    # now let's redo these maps, but for the different seasons (months spring: MAM, summer: JJA, fall: SON, winter: DJF)
    # Convert matlab dates to Python datetime objects
    python_dates = [datetime.fromordinal(int(d)) + timedelta(days=d % 1) - timedelta(days=366) for d in dates_val]

    # Define seasons
    seasons = [get_season(date) for date in python_dates]

    # Calculate and plot statistics for each season
    seasons = ["Spring", "Summer", "Autumn", "Winter"]

    # Initialize lists to store data for all seasons
    nn_temp_rmse_all = []
    nn_sal_rmse_all = []
    nn_temp_bias_all = []
    nn_sal_bias_all = []
    gem_temp_rmse_all = []
    gem_sal_rmse_all = []
    gem_temp_bias_all = []
    gem_sal_bias_all = []
    mlr_temp_rmse_all = []
    mlr_sal_rmse_all = []
    mlr_temp_bias_all = []
    mlr_sal_bias_all = []

    for season in seasons:
        season_mask = np.array([get_season(date) for date in python_dates]) == season

        # Calculate RMSE and bias by depth for each season for both NN and GEM
        nn_temp_rmse, nn_temp_bias = compute_season_masked_depth_rmse_bias(pred_T_resid, season_mask)
        nn_sal_rmse, nn_sal_bias = compute_season_masked_depth_rmse_bias(pred_S_resid, season_mask)
        gem_temp_rmse, gem_temp_bias = compute_season_masked_depth_rmse_bias(gems_T_resid, season_mask)
        gem_sal_rmse, gem_sal_bias = compute_season_masked_depth_rmse_bias(gems_S_resid, season_mask)
        mlr_temp_rmse, mlr_temp_bias = compute_season_masked_depth_rmse_bias(old_T_resid, season_mask)
        mlr_sal_rmse, mlr_sal_bias = compute_season_masked_depth_rmse_bias(old_S_resid, season_mask)

        # Append data to lists
        nn_temp_rmse_all.append(nn_temp_rmse)
        nn_sal_rmse_all.append(nn_sal_rmse)
        nn_temp_bias_all.append(nn_temp_bias)
        nn_sal_bias_all.append(nn_sal_bias)
        gem_temp_rmse_all.append(gem_temp_rmse)
        gem_sal_rmse_all.append(gem_sal_rmse)
        gem_temp_bias_all.append(gem_temp_bias)
        gem_sal_bias_all.append(gem_sal_bias)
        mlr_temp_rmse_all.append(mlr_temp_rmse)
        mlr_sal_rmse_all.append(mlr_sal_rmse)
        mlr_temp_bias_all.append(mlr_temp_bias)
        mlr_sal_bias_all.append(mlr_sal_bias)

    # Create the figures
    fig_rmse, axs_rmse = plt.subplots(4, 2, figsize=(20, 30))
    fig_bias, axs_bias = plt.subplots(4, 2, figsize=(20, 30))

    fig_rmse.suptitle("RMSE by Depth for Each Season", fontsize=20)
    fig_bias.suptitle("Bias by Depth for Each Season", fontsize=20)

    # Find max values for consistent x-axis scales
    max_temp_rmse = max(np.max(nn_temp_rmse_all), np.max(gem_temp_rmse_all))
    max_sal_rmse = max(np.max(nn_sal_rmse_all), np.max(gem_sal_rmse_all))
    max_temp_bias = max(np.max(np.abs(nn_temp_bias_all)), np.max(np.abs(gem_temp_bias_all)))
    max_sal_bias = max(np.max(np.abs(nn_sal_bias_all)), np.max(np.abs(gem_sal_bias_all)))

    for i, season in enumerate(seasons):
        # RMSE plots
        axs_rmse[i, 0].plot(gem_temp_rmse_all[i], np.arange(0, 1801), linewidth=3, label="GEM", color="xkcd:orange")
        axs_rmse[i, 0].plot(
            mlr_temp_rmse_all[i], np.arange(0, 1801), linewidth=3, label="NeSPReSO 1.0", color="xkcd:green"
        )
        axs_rmse[i, 0].plot(
            nn_temp_rmse_all[i], np.arange(0, 1801), linewidth=3, label="NeSPReSO 1.1", color="xkcd:gray"
        )
        axs_rmse[i, 0].invert_yaxis()
        if i == 3:
            axs_rmse[i, 0].set_xlabel("Temperature RMSE [°C]")
        axs_rmse[i, 0].set_ylabel("Depth [m]")
        axs_rmse[i, 0].set_title(f"{season} - Temperature RMSE")
        axs_rmse[i, 0].legend()
        axs_rmse[i, 0].grid(True)
        axs_rmse[i, 0].set_xlim(0, max_temp_rmse)

        axs_rmse[i, 1].plot(gem_sal_rmse_all[i], np.arange(0, 1801), linewidth=3, label="GEM", color="xkcd:orange")
        axs_rmse[i, 1].plot(
            mlr_sal_rmse_all[i], np.arange(0, 1801), linewidth=3, label="NeSPReSO 1.0", color="xkcd:green"
        )
        axs_rmse[i, 1].plot(
            nn_sal_rmse_all[i], np.arange(0, 1801), linewidth=3, label="NeSPReSO 1.1", color="xkcd:gray"
        )
        axs_rmse[i, 1].invert_yaxis()
        if i == 3:
            axs_rmse[i, 1].set_xlabel("Salinity RMSE [PSU]")
        axs_rmse[i, 1].set_title(f"{season} - Salinity RMSE")
        axs_rmse[i, 1].legend()
        axs_rmse[i, 1].grid(True)
        axs_rmse[i, 1].set_xlim(0, max_sal_rmse)

        # Bias plots
        axs_bias[i, 0].plot(gem_temp_bias_all[i], np.arange(0, 1801), linewidth=3, label="GEM", color="xkcd:orange")
        axs_bias[i, 0].plot(
            mlr_temp_bias_all[i], np.arange(0, 1801), linewidth=3, label="NeSPReSO 1.0", color="xkcd:green"
        )
        axs_bias[i, 0].plot(
            nn_temp_bias_all[i], np.arange(0, 1801), linewidth=3, label="NeSPReSO 1.1", color="xkcd:gray"
        )
        axs_bias[i, 0].invert_yaxis()
        if i == 3:
            axs_bias[i, 0].set_xlabel("Temperature Bias [°C]")
        axs_bias[i, 0].set_ylabel("Depth [m]")
        axs_bias[i, 0].set_title(f"{season} - Temperature Bias")
        axs_bias[i, 0].legend()
        axs_bias[i, 0].grid(True)
        axs_bias[i, 0].set_xlim(-max_temp_bias, max_temp_bias)

        axs_bias[i, 1].plot(gem_sal_bias_all[i], np.arange(0, 1801), linewidth=3, label="GEM", color="xkcd:orange")
        axs_bias[i, 1].plot(
            mlr_sal_bias_all[i], np.arange(0, 1801), linewidth=3, label="NeSPReSO 1.0", color="xkcd:green"
        )
        axs_bias[i, 1].plot(
            nn_sal_bias_all[i], np.arange(0, 1801), linewidth=3, label="NeSPReSO 1.1", color="xkcd:gray"
        )
        axs_bias[i, 1].invert_yaxis()
        if i == 3:
            axs_bias[i, 1].set_xlabel("Salinity Bias [PSU]")
        axs_bias[i, 1].set_title(f"{season} - Salinity Bias")
        axs_bias[i, 1].legend()
        axs_bias[i, 1].grid(True)
        axs_bias[i, 1].set_xlim(-max_sal_bias, max_sal_bias)

    plt.tight_layout()
    plt.show()
    # # Temperature RMSE
    # grid_avg_temp_rmse_nn_season, num_prof_nn_season = calculate_average_in_bin(
    #     lon_centers, lat_centers, lon_val[season_mask], lat_val[season_mask],
    #     pred_T_resid[:,season_mask], dpt_range, is_rmse=True
    # )

    # plot_bin_map(lon_bins, lat_bins, grid_avg_temp_rmse_nn_season, num_prof_nn_season,
    #              f"Temperature - {season}", "RMSE")

    # # Salinity RMSE
    # grid_avg_sal_rmse_nn_season, _ = calculate_average_in_bin(
    #     lon_centers, lat_centers, lon_val[season_mask], lat_val[season_mask],
    #     pred_S_resid[:,season_mask], dpt_range, is_rmse=True
    # )
    # plot_bin_map(lon_bins, lat_bins, grid_avg_sal_rmse_nn_season, num_prof_nn_season,
    #              f"Salinity - {season}", "RMSE")

    # # Temperature Bias
    # avg_nn_t_bias_season, _ = calculate_average_in_bin(
    #     lon_centers, lat_centers, lon_val[season_mask], lat_val[season_mask],
    #     pred_T_resid[:,season_mask], dpt_range, is_rmse=False
    # )
    # plot_bin_map(lon_bins, lat_bins, avg_nn_t_bias_season, num_prof_nn_season,
    #              f"Temperature - {season}", "Bias")

    # # Salinity Bias
    # avg_nn_s_bias_season, _ = calculate_average_in_bin(
    #     lon_centers, lat_centers, lon_val[season_mask], lat_val[season_mask],
    #     pred_S_resid[:,season_mask], dpt_range, is_rmse=False
    # )
    # plot_bin_map(lon_bins, lat_bins, avg_nn_s_bias_season, num_prof_nn_season,
    #              f"Salinity - {season}", "Bias")

    # Comparison maps
    avg_rmse_nn_t = grid_avg_temp_rmse_nn
    avg_rmse_nn_s = grid_avg_sal_rmse_nn
    avg_bias_nn_t = avg_nn_t_bias
    avg_bias_nn_s = avg_nn_s_bias
    avg_rmse_gem_t = grid_avg_temp_rmse_gem
    avg_rmse_gem_s = grid_avg_sal_rmse_gem
    avg_bias_gem_t = avg_gem_t_bias
    avg_bias_gem_s = avg_gem_s_bias
    avg_rmse_mlr_t = grid_avg_temp_rmse_mlr
    avg_rmse_mlr_s = grid_avg_sal_rmse_mlr
    avg_bias_mlr_t = avg_mlr_t_bias
    avg_bias_mlr_s = avg_mlr_s_bias

    lon_centr = lon_centers[:-1]
    lat_centr = lat_centers[:-1]

    # GEM
    plot_comparison_maps(lon_centr, lat_centr, avg_rmse_nn_t, avg_rmse_gem_t, "temperature", "GEM")
    plot_comparison_maps(lon_centr, lat_centr, avg_rmse_nn_s, avg_rmse_gem_s, "salinity", "GEM")

    # ISOP
    plot_comparison_maps(lon_centr, lat_centr, avg_rmse_nn_t, avg_rmse_isop_t, "temperature", "ISOP")
    plot_comparison_maps(lon_centr, lat_centr, avg_rmse_nn_s, avg_rmse_isop_s, "salinity", "ISOP")

    # NeSPReSO 1.0
    plot_comparison_maps(lon_centr, lat_centr, avg_rmse_nn_t, avg_rmse_mlr_t, "temperature", "NeSPReSO 1.0")
    plot_comparison_maps(lon_centr, lat_centr, avg_rmse_nn_s, avg_rmse_mlr_s, "salinity", "NeSPReSO 1.0")

    print("the following are bias plots")
    plot_comparison_maps(lon_centr, lat_centr, avg_bias_nn_t, avg_bias_isop_t, "temperature", "ISOP", "Bias")
    plot_comparison_maps(lon_centr, lat_centr, avg_bias_nn_s, avg_bias_isop_s, "salinity", "ISOP", "Bias")

    plot_comparison_maps(lon_centr, lat_centr, avg_bias_nn_t, avg_bias_gem_t, "temperature", "GEM", "Bias")
    plot_comparison_maps(lon_centr, lat_centr, avg_bias_nn_s, avg_bias_gem_s, "salinity", "GEM", "Bias")

    plot_comparison_maps(lon_centr, lat_centr, avg_bias_nn_t, avg_bias_mlr_t, "temperature", "NeSPReSO 1.0", "Bias")
    plot_comparison_maps(lon_centr, lat_centr, avg_bias_nn_s, avg_bias_mlr_s, "salinity", "NeSPReSO 1.0", "Bias")
