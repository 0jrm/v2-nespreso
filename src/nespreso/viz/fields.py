"""Glider field visualization helpers extracted from the monolith."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import cmocean.cm as ccm
from matplotlib.figure import Figure

def plot_field(
    data: np.ndarray,
    distances: np.ndarray,
    depths: np.ndarray,
    variable_name: str,
    title: str,
) -> None:
    """
    Plot the temperature or salinity field over distance.

    Parameters:
    data (np.array): 2D array of temperature or salinity data
    distances (np.array): 1D array of distances corresponding to the data
    variable_name (str): Name of the variable ('Temperature' or 'Salinity')
    title (str): Title of the plot
    """
    if variable_name == "Temperature":
        vmin = 0
        vmax = 40
        step = 5
        cmap = ccm.thermal
    elif variable_name == "Salinity":
        vmin = 34
        vmax = 37
        step = 1
        cmap = ccm.haline
    elif variable_name == "T Differences":
        vmin = -4
        vmax = 4
        step = 0.2
        cmap = "coolwarm"
    elif variable_name == "S Differences":
        vmin = -1
        vmax = 1
        step = 0.1
        cmap = "PiYG"
    else:
        raise ValueError(f"Invalid variable name: {variable_name}")

    num_levels = int((vmax - vmin) / step + 1)
    cmap = plt.get_cmap(cmap, num_levels)

    plt.figure(figsize=(12, 6))
    plt.contour(
        distances, depths, data, levels=np.arange(vmin, np.ceil(vmax) + 1, step), colors="black", linewidths=0.1
    )
    plt.pcolormesh(distances, depths, data, shading="nearest", cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(label=f"{variable_name} [{variable_name[0]}]", extend="both")
    plt.xlabel("Distance (km)")
    plt.ylabel("Depth (index)")
    plt.title(title)
    ax = plt.gca()
    ax.invert_yaxis()
    plt.show()

def plot_field_subplot(
    data: np.ndarray,
    distances: np.ndarray,
    depths: np.ndarray,
    variable_name: str,
    title: str,
    subplot_pos: int,
    fig: Figure,
) -> None:
    """
    Plot a field as a subplot.

    Parameters:
    data (np.array): 2D array of temperature or salinity data
    distances (np.array): 1D array of distances corresponding to the data
    depths (np.array): 1D array of depths corresponding to the data
    variable_name (str): Name of the variable ('Temperature', 'Salinity', etc.)
    title (str): Title of the subplot
    subplot_pos (int): Position of the subplot in the figure
    fig (matplotlib.figure.Figure): Figure object to plot on
    """
    ax = fig.add_subplot(subplot_pos)

    if variable_name == "Temperature":
        vmin = 5
        vmax = 30
        step = 5
        cmap = ccm.thermal
        unit = "°C"
    elif variable_name == "Salinity":
        vmin = 35
        vmax = 37
        step = 0.25
        cmap = ccm.haline
        unit = "PSU"
    elif variable_name == "T Difference":
        vmin = -4
        vmax = 4
        step = 0.5
        cmap = "bwr"
        unit = "°C"
    elif variable_name == "S Difference":
        vmin = -1
        vmax = 1
        step = 0.125
        cmap = "PiYG"
        unit = "PSU"
    else:
        raise ValueError(f"Invalid variable name: {variable_name}")

    num_levels = int((vmax - vmin) / step + 1)
    cmap = plt.get_cmap(cmap, num_levels)

    # rows = subplot_pos//100
    # cols = (subplot_pos%100)//10
    # id = subplot_pos%10
    # isFirstColumn = id%rows == 1
    # isLastRow = idcols == 1

    contour = ax.contour(
        distances,
        depths,
        data,
        levels=np.arange(vmin + step / 2, np.ceil(vmax + step) + 1, step),
        colors="black",
        linewidths=0.2,
    )
    pcm = ax.pcolormesh(distances, depths, data, shading="nearest", cmap=cmap, vmin=vmin, vmax=vmax)
    # if isFirstColumn:
    ax.set_ylabel("Depth [m]")
    # else:
    fig.colorbar(pcm, ax=ax, label=f"{variable_name} [{unit}]", extend="both")
    # if isLastRow:
    ax.set_xlabel("Distance (km)")
    ax.grid(color="gray", linestyle="--", linewidth=0.7)
    ax.set_title(title)
    ax.invert_yaxis()

