"""Profile and seasonal visualization helpers extracted from the monolith."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

from nespreso.utils.time import matlab2datetime

def visualize_combined_results(
    true_values: tuple[np.ndarray, np.ndarray],
    gem_temp: np.ndarray,
    gem_sal: np.ndarray,
    predicted_values: tuple[np.ndarray, np.ndarray],
    sst_values: np.ndarray,
    ssh_values: np.ndarray,
    min_depth: float = 20,
    max_depth: float = 2000,
    num_samples: int = 5,
) -> None:
    # TODO: add date to plot
    """
    Visualize the true vs. predicted vs. GEM approximated values for a sample of profiles and their differences.

    Parameters:
    - true_values: ground truth temperature and salinity profiles.
    - gem_temp: GEM approximated temperature profiles.
    - gem_sal: GEM approximated salinity profiles.
    - predicted_values: model's predicted temperature and salinity profiles.
    - sst_values: Sea Surface Temperature values for each profile.
    - ssh_values: Sea Surface Height (adt) values for each profile.
    - num_samples: number of random profiles to visualize.

    Returns:
    - None (plots the results).
    """
    n_depths = max_depth + 1
    depth_levels = np.arange(min_depth, n_depths)
    population_size = true_values.shape[2]

    if num_samples == population_size:
        indices = np.arange(num_samples)
    else:
        indices = np.random.choice(int(population_size), num_samples, replace=False)

    for idx in indices:
        fig, axs = plt.subplots(2, 2, figsize=(12, 12))

        # First row: Actual Profiles
        # Temperature profile
        axs[0][0].plot(gem_temp[idx], depth_levels, "g", label="GEM Profile", alpha=0.75)
        axs[0][0].plot(predicted_values[0][:, idx], depth_levels, "r", label="NeSPReSO 1.1 Profile", alpha=0.75)
        axs[0][0].plot(true_values[:, 0, idx], depth_levels, "k", label="Target", linewidth=0.7)
        axs[0][0].invert_yaxis()
        axs[0][0].set_title(f"Temperature Profile")
        axs[0][0].set_ylabel("Depth")
        axs[0][0].set_xlabel("Temperature")
        axs[0][0].legend(loc="lower right")
        axs[0][0].grid(color="gray", linestyle="--", linewidth=0.5)

        # Salinity profile
        axs[0][1].plot(gem_sal[idx], depth_levels, "g", label="GEM Profile", alpha=0.75)
        axs[0][1].plot(predicted_values[1][:, idx], depth_levels, "r", label="NeSPReSO 1.1 Profile", alpha=0.75)
        axs[0][1].plot(true_values[:, 1, idx], depth_levels, "k", label="Target", linewidth=0.7)
        axs[0][1].invert_yaxis()
        axs[0][1].set_title(f"Salinity Profile")
        axs[0][1].set_ylabel("Depth")
        axs[0][1].set_xlabel("Salinity")
        axs[0][1].legend(loc="lower right")
        axs[0][1].grid(color="gray", linestyle="--", linewidth=0.5)

        # Second row: Differences
        gem_temp_dif = gem_temp[idx] - true_values[:, 0, idx]
        gem_sal_dif = gem_sal[idx] - true_values[:, 1, idx]
        nn_temp_dif = predicted_values[0][:, idx] - true_values[:, 0, idx]
        nn_sal_dif = predicted_values[1][:, idx] - true_values[:, 1, idx]

        axs[1][0].plot(np.abs(gem_temp_dif), depth_levels, "g", label="GEM Profile", alpha=0.75)
        axs[1][0].plot(np.abs(nn_temp_dif), depth_levels, "r", label="NeSPReSO 1.1 Profile", alpha=0.75)
        axs[1][0].axvline(0, color="k", linestyle="--", linewidth=0.5)
        axs[1][0].invert_yaxis()
        axs[1][0].set_title(f"Temperature Differences")
        axs[1][0].set_ylabel("Depth")
        axs[1][0].set_xlabel("Absolute difference [°C]")
        axs[1][0].legend(loc="best")
        axs[1][0].grid(color="gray", linestyle="--", linewidth=0.5)

        # Salinity difference
        axs[1][1].plot(np.abs(gem_sal_dif), depth_levels, "g", label="GEM Profile", alpha=0.75)
        axs[1][1].plot(np.abs(nn_sal_dif), depth_levels, "r", label="NeSPReSO 1.1 Profile", alpha=0.75)
        axs[1][1].axvline(0, color="k", linestyle="--", linewidth=0.5)
        axs[1][1].invert_yaxis()
        axs[1][1].set_title(f"Salinity Differences")
        axs[1][1].set_ylabel("Depth")
        axs[1][1].set_xlabel("Absolute difference [PSU]")
        axs[1][1].legend(loc="best")
        axs[1][1].grid(color="gray", linestyle="--", linewidth=0.5)

        gem_temp_se_individual = np.sqrt(np.mean(gem_temp_dif**2))
        gem_sal_se_individual = np.sqrt(np.mean(gem_sal_dif**2))
        nn_temp_se_individual = np.sqrt(np.mean(nn_temp_dif**2))
        nn_sal_se_individual = np.sqrt(np.mean(nn_sal_dif**2))

        accuracy_gain_temp = 100 * (gem_temp_se_individual - nn_temp_se_individual) / gem_temp_se_individual
        accuracy_gain_sal = 100 * (gem_sal_se_individual - nn_sal_se_individual) / gem_sal_se_individual

        # Add sst, ssh and accuracy gain information to the suptitle
        plt.suptitle(
            f"Profile {idx} - SST: {sst_values[idx]:.2f}, SSH (adt): {ssh_values[idx]:.2f}\n"
            f"T prediction improvement: {accuracy_gain_temp:.2f}%, S prediction improvement: {accuracy_gain_sal:.2f}%",
            fontsize=16,
        )

        plt.tight_layout()
        plt.show()

    # RMSE Calculations and Accuracy Gain
    gem_temp_errors = (gem_temp.T - true_values[:, 0, :]) ** 2
    gem_sal_errors = (gem_sal.T - true_values[:, 1, :]) ** 2

    nn_temp_errors = (predicted_values[0][:, :] - true_values[:, 0, :]) ** 2
    nn_sal_errors = (predicted_values[1][:, :] - true_values[:, 1, :]) ** 2

    gem_temp_se = np.sqrt(np.mean(gem_temp_errors))
    gem_sal_se = np.sqrt(np.mean(gem_sal_errors))

    nn_temp_se = np.sqrt(np.mean(nn_temp_errors))
    nn_sal_se = np.sqrt(np.mean(nn_sal_errors))

    accuracy_gain_temp = 100 * (gem_temp_se - nn_temp_se) / gem_temp_se
    accuracy_gain_sal = 100 * (gem_sal_se - nn_sal_se) / gem_sal_se

    print(f"NeSPReSO 1.1 Average temperature RMSE: {nn_temp_se:.3f}°C")
    print(f"NeSPReSO 1.1 Average salinity RMSE: {nn_sal_se:.3f} PSU")
    print(f"GEM Average temperature RMSE: {gem_temp_se:.3f}°C")
    print(f"GEM Average salinity RMSE: {gem_sal_se:.3f} PSU")

    gem_temp_errors = (gem_temp.T[150:, :] - true_values[150:, 0, :]) ** 2
    gem_sal_errors = (gem_sal.T[150:, :] - true_values[150:, 1, :]) ** 2

    nn_temp_errors = (predicted_values[0][150:, :] - true_values[150:, 0, :]) ** 2
    nn_sal_errors = (predicted_values[1][150:, :] - true_values[150:, 1, :]) ** 2

    gem_temp_se = np.sqrt(np.mean(gem_temp_errors))
    gem_sal_se = np.sqrt(np.mean(gem_sal_errors))

    nn_temp_se = np.sqrt(np.mean(nn_temp_errors))
    nn_sal_se = np.sqrt(np.mean(nn_sal_errors))

    accuracy_gain_temp = 100 * (gem_temp_se - nn_temp_se) / gem_temp_se
    accuracy_gain_sal = 100 * (gem_sal_se - nn_sal_se) / gem_sal_se


def filter_by_season(data: list[Any], dates: list[datetime], season: str) -> list[Any]:
    SEASONS = {"Winter": [12, 1, 2], "Spring": [3, 4, 5], "Summer": [6, 7, 8], "Fall": [9, 10, 11]}
    months = SEASONS[season]
    indices = [i for i, date in enumerate(dates) if matlab2datetime(date).month in months]
    return [data[i] for i in indices]


def seasonal_plots(
    lat_val: np.ndarray,
    lon_val: np.ndarray,
    dates_val: list[datetime],
    original_profiles: tuple[np.ndarray, np.ndarray],
    gem_temp: np.ndarray,
    gem_sal: np.ndarray,
    val_predictions: tuple[np.ndarray, np.ndarray],
    sst_inputs: np.ndarray,
    ssh_inputs: np.ndarray,
    max_depth: float,
    num_samples: int,
) -> None:
    seasons = ["Winter", "Spring", "Summer", "Fall"]
    total_samples = len(lat_val)
    indexes = np.arange(total_samples)
    for season in seasons:
        idx = np.array(filter_by_season(indexes, dates_val, season))
        print(season)
        fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()}, figsize=(10, 10))
        ax.set_global()
        ax.coastlines()
        # Setting plot limits to the Gulf of Mexico region
        ax.set_extent([-98, -80, 18, 31])
        scatter = ax.scatter(
            lon_val[idx],
            lat_val[idx],
            c=ssh_inputs[idx],
            cmap="viridis",
            edgecolors="k",
            linewidth=0.5,
            transform=ccrs.PlateCarree(),
        )
        cbar = plt.colorbar(scatter, ax=ax, orientation="vertical", pad=0.02, shrink=1)
        cbar.set_label("SSH Value")

        ax.set_title(f"{season} profiles in validation", fontsize=16)
        plt.show()

        # Now plot some samples from this season
        sliced_val_pred = [array[:, idx] for array in val_predictions]
        visualize_combined_results(
            original_profiles[:, :, idx],
            gem_temp[idx],
            gem_sal[idx],
            sliced_val_pred,
            sst_inputs[idx],
            ssh_inputs[idx],
            max_depth=max_depth,
            num_samples=num_samples,
        )


def calculate_bias(
    true_values: np.ndarray,
    predicted_values: tuple[np.ndarray, np.ndarray],
    gem_temp: np.ndarray,
    gem_sal: np.ndarray,
    min_depth: float = 20,
    max_depth: float = 2000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    gem_temp_bias = gem_temp.T - true_values[:, 0, :]
    gem_sal_bias = gem_sal.T - true_values[:, 1, :]

    nn_t_bias = predicted_values[0][:, :] - true_values[:, 0, :]
    nn_s_bias = predicted_values[1][:, :] - true_values[:, 1, :]

    return nn_t_bias, nn_s_bias, gem_temp_bias, gem_sal_bias
