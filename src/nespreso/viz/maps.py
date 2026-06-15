"""Bin-map and residual visualization helpers extracted from the monolith."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib.axes import Axes

def calculate_average_in_bin(
    lon_bins: np.ndarray,
    lat_bins: np.ndarray,
    lon_val: np.ndarray,
    lat_val: np.ndarray,
    bias_values: np.ndarray,
    dpt_range: np.ndarray = np.arange(0, 1801),
    is_rmse: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    # dpt_min, dpt_max = dpt_range
    avg_rmse_grid = np.zeros((len(lat_bins) - 1, len(lon_bins) - 1))
    num_prof_grid = np.zeros((len(lat_bins) - 1, len(lon_bins) - 1))

    if is_rmse:
        input_vals = bias_values**2
    else:
        input_vals = bias_values

    for i in range(len(lon_bins) - 1):
        for j in range(len(lat_bins) - 1):
            # Find points that fall into the current bin
            # in_bin = (lon_val == lon_bins[i]) & (lat_val == lat_bins[j])
            in_bin = (
                (lon_val >= lon_bins[i])
                & (lon_val < lon_bins[i + 1])
                & (lat_val >= lat_bins[j])
                & (lat_val < lat_bins[j + 1])
            )
            # Calculate average RMSE for points in the bin
            rmses = input_vals[dpt_range, :]
            rmses = rmses[:, in_bin]
            avg_rmse_grid[j, i] = np.mean(rmses)
            num_prof_grid[j, i] = np.sum(in_bin)

    if is_rmse:
        return np.sqrt(avg_rmse_grid), num_prof_grid

    else:
        return avg_rmse_grid, num_prof_grid


def plot_bin_map(
    lon_bins: np.ndarray,
    lat_bins: np.ndarray,
    avg_rmse_nn: np.ndarray,
    num_prof: np.ndarray,
    title_prefix: str,
    variable_plotted: str,
) -> None:
    # Calculate centers of the bins
    lon_centers = (lon_bins[:-1] + lon_bins[1:]) / 2
    lat_centers = (lat_bins[:-1] + lat_bins[1:]) / 2

    vmin = 0

    # Set up color maps and limits
    if title_prefix.startswith("Temperature"):
        units = "[°C]"
        if variable_plotted == "Bias":
            cmap = "coolwarm"
            vmax = 1
            vmin = -1

        else:
            cmap = "YlOrRd"
            vmax = 2
            vmin = 0.3

    else:
        units = "[PSU]"
        if variable_plotted == "Bias":
            cmap = "PiYG_r"
            vmax = 0.2
            vmin = -0.2

        else:
            cmap = "PuBuGn"
            vmax = 0.35
            vmin = 0

    # Create subplot grid
    fig, ax1 = plt.subplots(1, 1, figsize=(15, 15), subplot_kw={"projection": ccrs.PlateCarree()})

    # Plot the maps
    plot_rmse_on_ax(
        ax1,
        lon_centers,
        lat_centers,
        avg_rmse_nn,
        num_prof,
        f"NeSPReSO 1.1 Average {variable_plotted} - {title_prefix}",
    )

    pcm = ax1.pcolormesh(lon_centers, lat_centers, avg_rmse_nn, cmap=cmap, vmin=vmin, vmax=vmax)
    fig.colorbar(
        pcm,
        ax=ax1,
        orientation="vertical",
        pad=0.04,
        fraction=0.465 * (1 / 15),
        label=f"Average {variable_plotted} {units}",
    )
    ax1.set_xlabel("Longitude")
    ax1.set_ylabel("Latitude")
    # Set x and y ticks,
    ax1.set_xticks(np.arange(-99, -81, 1))
    ax1.set_yticks(np.arange(18, 30, 1))
    # add grid
    ax1.grid(color="gray", linestyle="--", linewidth=0.5)
    plt.show()


def plot_rmse_on_ax(
    ax: Axes,
    lon_centers: np.ndarray,
    lat_centers: np.ndarray,
    avg_rmse_grid: np.ndarray,
    num_prof: np.ndarray,
    title: str,
) -> None:
    ax.set_extent([-99, -81, 18, 30])  # Set to your area of interest
    ax.coastlines()

    pcm = ax.pcolormesh(lon_centers, lat_centers, avg_rmse_grid, cmap="coolwarm", vmin=-3, vmax=3)
    ax.set_title(title, fontsize=18)

    # Annotate each cell with the average RMSE value
    for i, lon in enumerate(lon_centers):
        for j, lat in enumerate(lat_centers):
            value = avg_rmse_grid[j, i]
            number = num_prof[j, i]
            if not np.isnan(value):  # Check if the value is not NaN, and if there are more than 2 profiles in the bin
                ax.text(
                    lon,
                    lat + 0.2,
                    f"{number:.0f}",
                    color="gray",
                    ha="center",
                    va="center",
                    fontsize=12,
                    transform=ccrs.PlateCarree(),
                )
                ax.text(
                    lon,
                    lat - 0.2,
                    f"{value:.2f}",
                    color="black",
                    ha="center",
                    va="center",
                    fontsize=12,
                    transform=ccrs.PlateCarree(),
                )


def plot_comparison_maps(
    lon_centers: np.ndarray,
    lat_centers: np.ndarray,
    avg_var_nn: np.ndarray,
    avg_var_compare: np.ndarray,
    title_prefix: str,
    name_compare: str,
    variable_name: str = "RMSE",
) -> None:
    # Calculate the difference
    avg_var_diff = np.abs(avg_var_nn) - np.abs(avg_var_compare)

    # Set up color maps and limits
    if title_prefix == "temperature":
        units = "[°C]"
        if variable_name == "Bias":
            cmap = "coolwarm"
            n_plots = 3
            vmax = 1
            vmin = -1
        else:
            cmap = "YlOrRd"
            n_plots = 3
            vmax = 2
            vmin = 0.3

    else:
        units = "[PSU]"
        if variable_name == "Bias":
            cmap = "PiYG_r"
            n_plots = 3
            vmax = 0.2
            vmin = -0.2
        else:
            cmap = "PuBuGn"
            n_plots = 3
            vmax = 0.35
            vmin = 0

    # Custom colormap for difference plot
    diff_cmap = "bwr"
    norm_diff = plt.Normalize(-vmax, vmax)

    # Create subplot grid
    fig, axes = plt.subplots(1, n_plots, figsize=(n_plots * 10, 15), subplot_kw={"projection": ccrs.PlateCarree()})

    # Titles for each subplot
    if variable_name == "Bias":
        dif_name = "Difference of magnitude"
    else:
        dif_name = "Difference"
    titles = [f"NeSPReSO 1.1", f"{name_compare}", f"{dif_name} (lower is better)"]

    # Function to add values to bins
    def annotate_bins(ax, data):
        for i, lon in enumerate(lon_centers):
            for j, lat in enumerate(lat_centers):
                value = data[j, i]
                if not np.isnan(value):
                    ax.text(
                        lon,
                        lat,
                        f"{value:.2f}",
                        color="black",
                        ha="center",
                        va="center",
                        fontsize=9,
                        transform=ccrs.PlateCarree(),
                    )

    # Plotting NN RMSE, ISOP RMSE, and Difference
    for i, (data, title) in enumerate(zip([avg_var_nn, avg_var_compare, avg_var_diff], titles)):
        if i < 2:
            pcm = axes[i].pcolormesh(lon_centers, lat_centers, data, cmap=cmap, vmin=vmin, vmax=vmax)
        elif i >= 2 and n_plots == 3:  # For the difference plot
            pcm_diff = axes[i].pcolormesh(lon_centers, lat_centers, data, cmap=diff_cmap, norm=norm_diff)
        # if i < n_plots:
        #     annotate_bins(axes[i], data)

    for i in range(n_plots):
        axes[i].set_title(titles[i], weight="bold")
        axes[i].coastlines()
        axes[i].set_xticks(np.arange(-99, -81, 2))
        axes[i].set_yticks(np.arange(18, 32, 2))
        axes[i].grid(color="gray", linestyle="--", linewidth=0.5)

    # Adding colorbar for the first two plots
    fig.colorbar(pcm, ax=axes[1], orientation="vertical", pad=0.04, fraction=0.0315)
    fig.suptitle(f"Average {title_prefix} {variable_name} by region", fontsize=28, y=0.705, fontweight="bold")

    if n_plots == 3:
        # Adding colorbar for the difference plot
        fig.colorbar(pcm_diff, ax=axes[2], orientation="vertical", pad=0.04, fraction=0.0305).set_label(
            label=f"{dif_name} {units}", size=14
        )

    plt.show()


def plot_residual_profiles_for_top_bins(
    lon_bins: np.ndarray,
    lat_bins: np.ndarray,
    lon_val: np.ndarray,
    lat_val: np.ndarray,
    nn_profiles: np.ndarray,
    avg_rmse_grid: np.ndarray,
    num_prof_grid: np.ndarray,
    param: str,
    min_depth: float,
    max_depth: float,
    top_n: int = 9,
) -> None:
    """
    Plots residual profiles for the top bins with the highest number of profiles.

    Parameters:
    - lon_bins, lat_bins: Arrays of longitude and latitude bin edges.
    - lon_val, lat_val: Arrays of longitude and latitude values for each profile.
    - nn_profiles, target_profiles: Arrays of neural network predicted and target profiles.
    - avg_rmse_grid, num_prof_grid: Grids of average RMSE and number of profiles per bin.
    - top_n: Number of top bins to plot. Default is 9 (for a 3x3 grid).
    """
    # Flatten the grid and sort bins by the number of profiles
    num_profiles_flat = num_prof_grid.flatten()
    sorted_indices = np.argsort(num_profiles_flat)[::-1][:top_n]

    # Set up the 3x3 subplot
    fig, axs = plt.subplots(3, 3, figsize=(15, 15))
    axs = axs.flatten()

    for idx, ax in enumerate(axs):
        if idx >= len(sorted_indices):
            ax.axis("off")
            continue

        # Get the bin index
        bin_index = np.unravel_index(sorted_indices[idx], num_prof_grid.shape)
        j, i = bin_index

        # Find profiles in this bin
        in_bin = (
            (lon_val >= lon_bins[i])
            & (lon_val < lon_bins[i + 1])
            & (lat_val >= lat_bins[j])
            & (lat_val < lat_bins[j + 1])
        )

        # Check if there are any profiles in the bin
        if np.any(in_bin):
            residuals = nn_profiles[:, in_bin].T

            # Plotting each residual profile
            for residual in residuals:
                ax.plot(
                    residual,
                    np.arange(min_depth, max_depth + 1, 1),
                    label=f"Lat: {lat_bins[j]}-{lat_bins[j + 1]}, Lon: {lon_bins[i]}-{lon_bins[i + 1]}",
                    color="gray",
                    linewidth=0.5,
                )
        else:
            ax.axis("off")  # No data for this bin

        ax.axvline(x=0, color="k", linewidth=0.5)
        # Set title with lat/lon, number of profiles, and average RMSE
        ax.set_title(
            f"Bin: Lat: {lat_bins[j]:.0f} ~ {lat_bins[j + 1]:.0f}, Lon: {lon_bins[i]:.0f} ~ {lon_bins[i + 1]:.0f}\n"
            f"Profiles: {num_prof_grid[j, i]}, Avg RMSE: {avg_rmse_grid[j, i]:.2f}"
        )
        fig.suptitle(f"Residual profiles for {param} bins with the most profiles\n", fontsize=16, fontweight="bold")
        ax.set_xlabel("Residual")
        ax.set_ylabel("Depth")
        ax.invert_yaxis()
        ax.grid(True)

    plt.tight_layout()
    plt.show()
