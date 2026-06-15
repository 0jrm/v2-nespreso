# %%
import sys
import os
import glob
import numpy as np
import mat73
import matplotlib.pyplot as plt
import random
import torch
import torch.nn as nn
import torch.optim as optim
import xarray as xr
import scipy
from numpy.polynomial.polynomial import Polynomial
from torch.utils.data import DataLoader, random_split, Subset
from sklearn.decomposition import PCA
from scipy.interpolate import RegularGridInterpolator, interp1d, make_interp_spline, splrep, BSpline
import pickle
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime, timedelta
from sklearn.cluster import MiniBatchKMeans
import matplotlib.colors as mcolors
import gsw
import seaborn as sns
from scipy.stats import linregress
from scipy.spatial.distance import cdist
import cmocean.cm as ccm
from collections import Counter
from tqdm import tqdm  # For the progress bar
from pandas import DataFrame as df
import time
import pandas as pd
import calendar
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
import plotly.express as px
from scipy.stats import pearsonr
from matplotlib.ticker import FormatStrFormatter
from nespreso.physics_metrics import (
    density_smoothness_metrics,
    eos_from_SP_T,
    second_derivative,
    static_stability_metrics,
)
from nespreso.io.satellite_readers import get_aviso_by_date
from nespreso.io.satellite import load_satellite_data, load_satellite_data_for_dataset
from nespreso.io.argo import load_argo_mat
from nespreso.metrics import bias, mad, rmse
from nespreso.utils.time import datenum_to_datetime, datenums_to_datetimes, get_month, get_season, matlab2datetime
from nespreso.reporting import print_training_params
from nespreso.experiments.validation_context import build_validation_context
from nespreso.experiments.pca_regression import run_pca_regression_baseline
from nespreso.experiments.glider_mission import run_glider_mission
from nespreso.experiments.monthly_distribution import run_monthly_distribution
from nespreso.experiments.density_stability import run_density_stability
from nespreso.experiments.depth_interval_stats import run_depth_interval_stats
from nespreso.experiments.validation_maps import run_validation_maps
from nespreso.experiments.steric_depth_stats import run_steric_depth_stats
from nespreso.utils.geo import calculate_distances, haversine
from nespreso.determinism import get_device, set_seed
from nespreso.inference import (
    get_inputs,
    get_predictions,
    get_predictions_torchscript,
    load_all_models,
    predict_with_numpy,
)
from nespreso.train import evaluate_model, train_model
from nespreso.data.features import prepare_inputs
from nespreso.data.dataset import TemperatureSalinityDataset
from nespreso.data.pca import sklearn_inverse_transform_pcs
from nespreso.data.splits import IndexedSubset, split_dataset
from nespreso.losses import (
    CombinedPCALoss,
    PCALoss,
    WeightedMSELoss,
    genWeightedMSELoss,
    make_loss,
)
from nespreso.models.density import DensityConstraint, RhoMLP
from nespreso.models.mlp import PredictionModel
from nespreso.viz.maps import (
    calculate_average_in_bin,
    plot_bin_map,
    plot_comparison_maps,
    plot_residual_profiles_for_top_bins,
    plot_rmse_on_ax,
)
from nespreso.viz.profiles import (
    calculate_bias,
    filter_by_season,
    seasonal_plots,
    visualize_combined_results,
)
from nespreso.viz.fields import plot_field, plot_field_subplot
from nespreso.viz.coefficients import plot_coefficients_heatmap
from nespreso.analysis.density import (
    compute_density_profiles,
    compute_smoothness_metrics,
    compute_stability_metrics,
)
from nespreso.analysis import (
    average_depth,
    bin_data,
    calculate_correlation,
    compute_depth_interval_metrics,
    compute_depth_rmse_bias,
    compute_profile_residual,
    compute_season_masked_depth_rmse_bias,
    default_depth_intervals,
    equivalent_average_statistic,
    fit_pcs_regression_exact_gpu,
    get_glider_predictions,
    histogram_available_depths,
    isop_depth_indices,
    predict_pcs_exact_gpu,
    prepare_features,
)

plt.rcParams.update({"font.size": 18})
# Set the seed for reproducibility
# load_trained_model = False
load_trained_model = False
ensemble_models = False
# load_dataset_file = False
load_dataset_file = True
gen_paula_profiles = False
global debug
debug = False  # Set to False to disable debugging
seed = 42
n_runs = 1  # number of model runs
nn_repeat_time = 10  # number of nespreso runs for generation timing
gem_repeat_time = 1  # number of GEM runs for generation timing

set_seed(seed)
DEVICE = get_device()

coolwhitewarm = mcolors.LinearSegmentedColormap.from_list(
    name="red_white_blue", colors=[(0, 0, 1), (1, 1.0, 1), (1, 0, 0)]
)


def inverse_transform(pcs, pca_temp, pca_sal, n_components):
    return sklearn_inverse_transform_pcs(pcs, pca_temp, pca_sal, n_components)


# %%
if __name__ == "__main__":
    from nespreso.config import load_config
    from nespreso.runner import run_training

    cfg = load_config()
    bin_size = 1  # bin size in degrees (monolith-only visualization knob)

    _save_model_path, artifacts = run_training(cfg, return_artifacts=True)
    ctx = build_validation_context(cfg, artifacts, bin_size=bin_size)

    run_steric_depth_stats(ctx)
    run_pca_regression_baseline(ctx)
    run_validation_maps(ctx)
    run_glider_mission(ctx)
    run_depth_interval_stats(ctx)
    run_density_stability(ctx)

        # #create a netcdf file with the validation dataset
    # sst_val = get_inputs(val_loader, device)[:,-2]
    # lat_val = val_dataset.dataset.LAT[val_indices]
    # lon_val = val_dataset.dataset.LON[val_indices]
    # date_val = datenums_to_datetimes(val_dataset.dataset.TIME[val_indices])
    # T_profiles_val = original_profiles[:,0,:]
    # S_profiles_val = original_profiles[:,1,:]
    # depth = np.arange(0,1801)

    # sst_val.shape, lat_val.shape, lon_val.shape, type(date_val), type(date_val[0]), len(date_val), T_profiles_val.shape, S_profiles_val.shape, depth.shape

    # from netCDF4 import Dataset
    # # import numpy as np
    # from datetime import datetime, timedelta

    # def create_netcdf(filename, sst_val, lat_val, lon_val, date_val, T_profiles_val, S_profiles_val, depth):
    #     with Dataset(filename, 'w', format='NETCDF4') as nc:
    #         # Dimensions
    #         nc.createDimension('time', len(date_val))
    #         nc.createDimension('lat', len(lat_val))
    #         nc.createDimension('lon', len(lon_val))
    #         nc.createDimension('depth', len(depth))

    #         # Variables
    #         times = nc.createVariable('time', 'f8', ('time',))
    #         lats = nc.createVariable('lat', 'f4', ('lat',))
    #         lons = nc.createVariable('lon', 'f4', ('lon',))
    #         depths = nc.createVariable('depth', 'f4', ('depth',))
    #         sst = nc.createVariable('sst', 'f4', ('time',))
    #         T_profiles = nc.createVariable('T_profiles', 'f4', ('depth', 'time'))
    #         S_profiles = nc.createVariable('S_profiles', 'f4', ('depth', 'time'))

    #         # Convert datetime to numeric time values
    #         ref_date = datetime(1900, 1, 1)
    #         numeric_dates = [(d - ref_date).total_seconds() for d in date_val]
    #         times[:] = numeric_dates

    #         # Assign data
    #         lats[:] = lat_val
    #         lons[:] = lon_val
    #         depths[:] = depth
    #         sst[:] = sst_val
    #         T_profiles[:, :] = T_profiles_val
    #         S_profiles[:, :] = S_profiles_val

    #         # Add attributes
    #         nc.description = 'Dataset used for acquiring statistics for ISOP, GEM and NeSPReSO methods. Contains SST, latitude, longitude, date, temperature profiles, salinity profiles, and depth.'
    #         sst.units = 'Celsius'
    #         lats.units = 'degrees'
    #         lons.units = 'degrees'
    #         depths.units = 'meter'
    #         times.units = 'seconds since 1900-01-01 00:00:00'
    #         T_profiles.units = 'Celsius'
    #         S_profiles.units = 'PSU'

    #         print(f"NetCDF file '{filename}' created successfully.")

    # # making a histogream of missing dates
    # full_dataset.data['TIME']
    # full_dataset.TIME
    # dates = datenums_to_datetimes(np.sort(full_dataset.data['TIME'][np.isin(full_dataset.data['TIME'], full_dataset.TIME, invert=True)]))
    # # Extracting year and month for each date
    # date_counts = Counter([(date.year, date.month) for date in dates])

    # # Sorting the dates for plotting
    # sorted_date_counts = dict(sorted(date_counts.items()))

    # # Creating labels and values for the histogram
    # labels = [f"{year}-{month:02}" for year, month in sorted_date_counts.keys()]
    # values = list(sorted_date_counts.values())

    # # Plotting the histogram
    # plt.figure(figsize=(22, 14))
    # plt.bar(labels, values)
    # plt.xlabel('Year-Month')
    # plt.ylabel('Frequency')
    # plt.title('Monthly Histogram of Dates')
    # plt.xticks(rotation=45)
    # plt.tight_layout()
    # plt.show()

    # filename = "/unity/g2/jmiranda/SubsurfaceFields/GEM_SubsurfaceFields/Test_dataset.nc"
    # # Creating the NetCDF file
    # create_netcdf(filename, sst_val, lat_val, lon_val, date_val, T_profiles_val, S_profiles_val, depth)

    # xr.open_dataset(filename).depth

    # # %%
    # from scipy.spatial import cKDTree

    # # lon_min = -88
    # # lon_max = -82
    # lon_min = np.floor(np.min(full_dataset.LON))
    # lon_max =  np.ceil(np.max(full_dataset.LON))
    # lat_min = np.floor(np.min(full_dataset.LAT))
    # lat_max =  np.ceil(np.max(full_dataset.LAT))

    # # Define grid spacing
    # grid_spacing = 0.1  # degrees

    # # Create the grid within the bounding box
    # lats_grid = np.arange(lat_min, lat_max + grid_spacing, grid_spacing)
    # lons_grid = np.arange(lon_min, lon_max + grid_spacing, grid_spacing)

    # # Use meshgrid to create a grid of coordinates
    # lats_mesh, lons_mesh = np.meshgrid(lats_grid, lons_grid)

    # # Flatten the meshgrid arrays to obtain the full list of coordinates
    # grid_points = np.vstack([lats_mesh.ravel(), lons_mesh.ravel()]).T

    # # Build a KD-Tree with the original LAT and LON data
    # data_points = np.vstack([full_dataset.LAT, full_dataset.LON]).T
    # tree = cKDTree(data_points)

    # # Query the tree for each grid point to find the distance to the nearest data point
    # distances, _ = tree.query(grid_points, distance_upper_bound=0.5)

    # # Filter the grid points where the distance is infinity (no points within 0.2 degrees)
    # filtered_grid_points = grid_points[distances != np.inf]

    # # Plot original data points and the filtered grid points
    # plt.figure(figsize=(10, 8))
    # plt.scatter(filtered_grid_points[:, 1], filtered_grid_points[:, 0], color='red', label='Filtered Grid Points', s=0.2)
    # plt.xlabel('Longitude')
    # plt.ylabel('Latitude')
    # plt.title('Points within 0.5 degrees of original data')
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    # %%
    # def calculate_sound_speed_NPL(T, S, Z, Phi=45):
    # """
    # Calculate sound speed (in m/s) using the NPL equation.
    # T: Temperature in degrees Celsius
    # S: Salinity in PSU
    # Z: Depth in meters
    # Phi: Latitude in degrees (default 45)
    # """
    # c = (1402.5 + 5 * T - 5.44e-2 * T**2 + 2.1e-4 * T**3
    #      + 1.33 * S - 1.23e-2 * S * T + 8.7e-5 * S * T**2
    #      + 1.56e-2 * Z + 2.55e-7 * Z**2 - 7.3e-12 * Z**3
    #      + 1.2e-6 * Z * (Phi - 45) - 9.5e-13 * T * Z**3
    #      + 3e-7 * T**2 * Z + 1.43e-5 * S * Z)
    # return c

    # # Recalculate sound speed at each depth using the NPL equation
    # sound_speed_profile_NPL = np.array([calculate_sound_speed_NPL(T, S, z) for T, S, z in zip(temperature_profile, salinity_profile, depths)])

    # # Finding the Sonic Layer Depth (SLD) using the NPL equation
    # max_sound_speed_index_NPL = np.argmax(sound_speed_profile_NPL)
    # SLD_NPL = depths[max_sound_speed_index_NPL]
    # # Conversion factor from meters to feet
    # meters_to_feet = 3.28084

    # # Conversion factor for the gradient from per feet to per 100 meters
    # conversion_factor = meters_to_feet / 100

    # # Calculating the Below Layer Gradient (BLG) using the NPL equation
    # gradient_NPL = np.gradient(sound_speed_profile_NPL, depths_feet)
    # # Average gradient below MLD in m/s per 100 feet using the NPL equation
    # BLG_NPL = np.mean(gradient_NPL[MLD_index:]) * conversion_factor

    # ## Eddy experiment - nature run stuff

    # #compare ssh distributions:

    # from matplotlib.ticker import PercentFormatter
    # from scipy.io import loadmat

    # def aggregate_from_mat(folder_path, *variable_names):
    #     aggregated_data = {var_name: [] for var_name in variable_names}

    #     # Loop through all files in the directory
    #     for filename in os.listdir(folder_path):
    #         if filename.endswith('.mat'):
    #             file_path = os.path.join(folder_path, filename)
    #             mat_data = loadmat(file_path)

    #             # Check if each variable exists in the .mat file and aggregate
    #             for var_name in variable_names:
    #                 if var_name in mat_data:
    #                     var_data = mat_data[var_name]
    #                     aggregated_data[var_name].append(np.expand_dims(var_data, axis=-1))
    #                 else:
    #                     print(f"'{var_name}' not found in {filename}")

    #     # Combine all variable data into single numpy arrays along the new axis
    #     for var_name in variable_names:
    #         if aggregated_data[var_name]:
    #             aggregated_data[var_name] = np.concatenate(aggregated_data[var_name], axis=-1)
    #         else:
    #             print(f"No '{var_name}' data found in any .mat files.")

    #     return aggregated_data

    # # Example usage:
    # folder_path = '/unity/g2/jmiranda/SubsurfaceFields/Data/NatureRun/'
    # ssh_nature_run = aggregate_from_mat(folder_path, 'ssh10')['ssh10'].flatten()

    # a = full_dataset.AVISO_ADT
    # n, bins, _ = plt.hist(a, weights=np.ones(len(a))/len(a), bins=100, color='blue', label='Training AVISO SSH')
    # plt.hist(ssh_nature_run, weights=np.ones(len(ssh_nature_run))/len(ssh_nature_run), bins=bins, color='red', label='Nature run SSH')
    # plt.gca().yaxis.set_major_formatter(PercentFormatter(1))

    # # Set custom x-ticks every 0.1 from -0.4 to 0.9
    # plt.xticks(np.arange(-0.4, 1.0, 0.1), fontsize=11)
    # plt.yticks(fontsize=11)
    # plt.legend(fontsize=11)

    # # Compare T/S diagrams for ssh ranges
    # ssh_nature_run = aggregate_from_mat(folder_path, 'ssh10')['ssh10']
    # T_nature_run = aggregate_from_mat(folder_path, 'temp10')['temp10']
    # S_nature_run = aggregate_from_mat(folder_path, 'sal10')['sal10']

    # import matplotlib.colors as mcolors

    # def plot_ts_profiles(datasets, dataset_labels, sigma_theta, Sg, Tg, cores, cmap_name='viridis'):
    #     """
    #     Plots T-S profiles from multiple datasets on the same plot.

    #     Parameters:
    #     - datasets: List of tuples [(TEMP1, SAL1), (TEMP2, SAL2), ...]
    #                 Each tuple contains temperature and salinity data.
    #     - dataset_labels: List of labels corresponding to each dataset.
    #     - sigma_theta: 2D array of sigma_theta values for contour plotting.
    #     - Sg: 2D array of salinity grid values for contour plotting.
    #     - Tg: 2D array of temperature grid values for contour plotting.
    #     - cores: Dictionary containing core water mass points to be marked on the plot.
    #             Example: {"SAAIW": (34.9, 6.5), "GCW": (36.4, 22.3), "NASUW": (36.8, 22)}
    #     - cmap_name: Name of the color map to use for distinguishing datasets (default: 'viridis').

    #     Returns:
    #     - None
    #     """

    #     # Initialize the plot
    #     fig, ax = plt.subplots(figsize=(10, 8))

    #     # Plot sigma_theta contours
    #     cs = ax.contour(Sg, Tg, sigma_theta, colors='grey', zorder=1)

    #     # Create a color map
    #     cmap = plt.get_cmap(cmap_name)
    #     colors = cmap(np.linspace(0, 1, len(datasets)))

    #     # Plot T-S profiles for each dataset
    #     for idx, (TEMP, SAL) in enumerate(datasets):
    #         label = dataset_labels[idx]
    #         color = colors[idx]

    #         # Ensure TEMP and SAL are 2D arrays for plotting
    #         if TEMP.ndim == 1:
    #             TEMP = TEMP[:, np.newaxis]
    #         if SAL.ndim == 1:
    #             SAL = SAL[:, np.newaxis]

    #         for i in range(TEMP.shape[1]):  # Plot each profile in the dataset
    #             ax.plot(SAL[:, i], TEMP[:, i], color=color, linewidth=0.5, label=label if i == 0 else "")

    #     # Plot core water masses
    #     for label, (salinity, temperature) in cores.items():
    #         ax.plot(salinity, temperature, 'o', markersize=7, color='black')
    #         ax.text(salinity, temperature, label, fontsize=11, verticalalignment='bottom', horizontalalignment='right', fontweight='bold')

    #     # Configure the plot
    #     ax.set_xlim(34.5, 37.5)
    #     plt.clabel(cs, fontsize=10, inline=False, fmt='%.1f', colors='k')
    #     plt.xlabel('Salinity [PSU]')
    #     plt.ylabel('Temperature [°C]')
    #     plt.title('T-S Diagram')
    #     plt.legend(fontsize=10)
    #     plt.show()

    # def index_for_range(data, min_val, max_val):
    #     return np.where((data >= min_val) & (data <= max_val))[0]

    # # Filter data based on SSH ranges
    # ssh_nature_run = ssh_nature_run.flatten()  # Assuming SSH values need to be compared
    # # Remove NaN values and corresponding indices from b, T_nature_run, and S_nature_run
    # valid_indices = ~np.isnan(ssh_nature_run)
    # ssh_nature_run = ssh_nature_run[valid_indices]
    # T_nature_run = T_nature_run[valid_indices]
    # S_nature_run = S_nature_run[valid_indices]
    # ssh_05to_01 = index_for_range(ssh_nature_run, -0.05, -0.01)
    # ssh_01to01 = index_for_range(ssh_nature_run, -0.01, 0.01)
    # ssh_01to10 = index_for_range(ssh_nature_run, 0.01, 0.1)
    # ssh_10to30 = index_for_range(ssh_nature_run, 0.1, 0.3)

    # # Build datasets with correct dimensions
    # datasets = [
    #     (T_nature_run[ssh_05to_01], S_nature_run[ssh_05to_01]),
    #     (T_nature_run[ssh_01to01], S_nature_run[ssh_01to01]),
    #     (T_nature_run[ssh_01to10], S_nature_run[ssh_01to10]),
    #     (T_nature_run[ssh_10to30], S_nature_run[ssh_10to30])
    # ]

    # dataset_labels = ['SSH -0.05 to -0.01', 'SSH -0.01 to 0.01', 'SSH 0.01 to 0.1', 'SSH 0.1 to 0.3']

    # # Plotting the T-S profiles
    # plot_ts_profiles(datasets, dataset_labels, sigma_theta, Sg, Tg, cores, cmap_name='viridis')

    ## Reviews

    run_monthly_distribution(ctx)
