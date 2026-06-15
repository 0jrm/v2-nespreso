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
from nespreso.utils.time import datenum_to_datetime, datenums_to_datetimes, matlab2datetime
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
    from dataclasses import asdict

    from nespreso.config import load_config
    from nespreso.runner import run_training

    cfg = load_config()

    # Monolith-only visualization knobs (not in YAML config).
    bin_size = 1  # bin size in degrees
    num_samples = 1  # profiles that will be plotted

    model_cfg = cfg.model
    input_params = asdict(cfg.input_params)
    n_components = model_cfg.n_components
    layers_config = list(model_cfg.layers_config)
    batch_size = model_cfg.batch_size
    min_depth = model_cfg.min_depth
    max_depth = model_cfg.max_depth
    dropout_prob = model_cfg.dropout_prob
    learning_rate = model_cfg.learning_rate
    train_size = model_cfg.train_size
    val_size = model_cfg.val_size
    test_size = model_cfg.test_size

    def printParams():
        true_params = [param for param, value in input_params.items() if value]
        print(f"\nNumber of profiles: {len(full_dataset)}")
        print("Parameters used:", ", ".join(true_params))
        print(f"Min depth: {min_depth}, Max depth: {max_depth}")
        print(f"Number of components used: {n_components} x2")
        print(f"Batch size: {batch_size}")
        print(f"Learning rate: {learning_rate}")
        print(f"Dropout probability: {dropout_prob}")
        print(f"Train/test/validation split: {train_size}/{test_size}/{val_size}")
        print(f"Layer configuration: {layers_config}\n")

    _save_model_path, artifacts = run_training(cfg, return_artifacts=True)
    full_dataset = artifacts.full_dataset
    train_dataset = artifacts.train_dataset
    val_dataset = artifacts.val_dataset
    test_dataset = artifacts.test_dataset
    train_loader = artifacts.train_loader
    val_loader = artifacts.val_loader
    test_loader = artifacts.test_loader
    trained_model = artifacts.trained_model
    device = artifacts.device
    input_dim = artifacts.input_dim

    printParams()

    print("Statistics from the last run:")
    print("Method & # profiles & Time (ms) & Time per profile (µs)")
    n_val = len(val_dataset)
    # Start time
    start_time = time.perf_counter()
    # Get predictions for the validation dataset
    for i in range(nn_repeat_time):
        val_predictions_pcs = get_predictions(trained_model, val_loader, device)
        # Accessing the original dataset for inverse_transform
        val_predictions = val_dataset.dataset.inverse_transform(val_predictions_pcs)
    # print(f"val_predictions_pcs: {type(val_predictions_pcs)} {val_predictions_pcs.shape}")
    # End time
    end_time = time.perf_counter()
    # Calculate elapsed time
    elapsed_time = (end_time - start_time) / nn_repeat_time
    print(f"NeSPReSO 1.1 ({device}) & {n_val} & {elapsed_time * 1e3:.2f} & {((elapsed_time * 1e6) / n_val):.2f}")

    # repeat time calculation for other device, if available
    # Check if CUDA is available
    if torch.cuda.is_available():
        if next(trained_model.parameters()).is_cuda:
            cuda_device = torch.device("cpu")
        else:
            cuda_device = torch.device("cuda")

        # Move model to CUDA device
        trained_model = trained_model.to(cuda_device)

        # Create new DataLoader with CUDA device
        cuda_val_loader = DataLoader(val_dataset, batch_size=val_loader.batch_size, sampler=val_loader.sampler)

        # Start time
        start_time = time.perf_counter()

        # Get predictions for the validation dataset on CUDA
        for i in range(nn_repeat_time):
            cuda_val_predictions_pcs = get_predictions(trained_model, cuda_val_loader, cuda_device)
            # Accessing the original dataset for inverse_transform
            cuda_val_predictions = val_dataset.dataset.inverse_transform(cuda_val_predictions_pcs)

        # End time
        end_time = time.perf_counter()

        # Calculate elapsed time
        cuda_elapsed_time = (end_time - start_time) / nn_repeat_time
        print(
            f"NeSPReSO 1.1 ({cuda_device}) & {n_val} & {cuda_elapsed_time * 1e3:.2f} & {((cuda_elapsed_time * 1e6) / n_val):.2f}"
        )

        # Move model back to original device
        trained_model = trained_model.to(device)
    else:
        print("CUDA is not available. Skipping GPU time calculation.")

    # load ISOP results
    file_path_new = "/unity/g2/jmiranda/SubsurfaceFields/Data/ISOP1_rmse_bias_1deg_maps.nc"
    data_ISOP = xr.open_dataset(file_path_new)

    # Create bins for longitude and latitude
    lon_bins = np.arange(np.min(data_ISOP.lon) - 0.5, np.max(data_ISOP.lon) + 1.5, 1)
    lat_bins = np.arange(np.min(data_ISOP.lat) - 0.5, np.max(data_ISOP.lat) + 1.5, 1)

    # Calculate centers of the bins
    lon_centers = lon_bins + bin_size / 2
    lat_centers = lat_bins + bin_size / 2

    # Initialize a NaN array for the number of profiles
    num_prof = np.full((len(lat_centers), len(lon_centers)), np.nan)

    # Extracting RMSE data and ensuring it matches the dimensions of the bins
    avg_rmse_isop_t = data_ISOP["t_rmse_syn"]
    avg_rmse_isop_s = data_ISOP["s_rmse_syn"]

    avg_rmse_gdem_t = data_ISOP["t_rmse_gdem"]
    avg_rmse_gdem_s = data_ISOP["s_rmse_gdem"]

    avg_bias_isop_t = data_ISOP["t_bias_syn"]
    avg_bias_isop_s = data_ISOP["s_bias_syn"]

    subset_indices = val_loader.dataset.indices

    # For original profiles
    original_profiles = val_dataset.dataset.get_profiles(subset_indices, pca_approx=False)

    # For PCA approximated profiles
    pca_approx_profiles = val_dataset.dataset.get_profiles(subset_indices, pca_approx=True)

    if ensemble_models:
        # Directory where models are saved
        models_dir = "/unity/g2/jmiranda/SubsurfaceFields/GEM_SubsurfaceFields/saved_models/"

        start_time = time.perf_counter()

        # Load all models
        models = load_all_models(
            models_dir=models_dir,
            device=device,
            input_dim=input_dim,
            layers_config=layers_config,
            n_components=n_components,
            dropout_prob=dropout_prob,
        )
        # print(f"Loaded {len(models)} models.")

        # Initialize accumulators for predictions
        accumulated_pred_T = None
        accumulated_pred_S = None

        # Iterate over each model and accumulate predictions
        for model in models:
            # print(f"Generating predictions with model: {model}")
            # Get predictions for the validation dataset
            val_predictions_pcs = get_predictions(model, val_loader, device)
            # Inverse transform to get actual predictions
            val_predictions = val_dataset.dataset.inverse_transform(val_predictions_pcs)

            # Split predictions into T and S
            pred_T_current = val_predictions[0]  # Assuming index 0 for T
            pred_S_current = val_predictions[1]  # Assuming index 1 for S

            # Accumulate predictions
            if accumulated_pred_T is None:
                accumulated_pred_T = pred_T_current
                accumulated_pred_S = pred_S_current
            else:
                accumulated_pred_T += pred_T_current
                accumulated_pred_S += pred_S_current

        # Compute the average predictions
        avg_pred_T = accumulated_pred_T / len(models)
        avg_pred_S = accumulated_pred_S / len(models)

        # End time
        end_time = time.perf_counter()

        # Calculate elapsed time
        cuda_elapsed_time = end_time - start_time

        # Assign averaged predictions to final variables
        pred_T = avg_pred_T
        pred_S = avg_pred_S

        print(f"NeSPReSO 1.1 ensamble 15x & {cuda_elapsed_time * 1e3:.2f} & {((cuda_elapsed_time * 1e6) / n_val):.2f}")

    else:
        # if not ensamble, get predictions from model
        pred_T = val_predictions[0]
        pred_S = val_predictions[1]

    # Load old NeSPReSO 1.0 model for comparison (using the same model as the API)
    print("Loading NeSPReSO 1.0 model for comparison (using API model)...")
    old_model_path = "/unity/g2/jmiranda/nespreso_api/models/ocean_tensorscript.pt"
    old_pca_path = "/unity/g2/jmiranda/nespreso_api/models/pca_stats.pkl"

    # Load TorchScript model (same as API uses)
    old_model = torch.jit.load(old_model_path, map_location=torch.device(DEVICE))
    old_model.eval()

    # Load PCA objects from API (with warning suppression for sklearn version mismatch)
    import pickle
    import warnings
    from sklearn.base import InconsistentVersionWarning

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
        with open(old_pca_path, "rb") as f:
            old_pca_data = pickle.load(f)
    old_pca_temp = old_pca_data["pca_temp"]
    old_pca_sal = old_pca_data["pca_sal"]
    old_input_params = old_pca_data.get("input_params", input_params)

    print(f"NeSPReSO 1.0 model loaded from API (TorchScript format)")
    print(f"Old model input params: {old_input_params}")

    # Get predictions from old model using TorchScript inference
    # Need to prepare inputs in the same format as the API expects
    old_val_predictions_pcs = get_predictions_torchscript(old_model, val_loader, device, old_input_params)

    old_val_predictions = sklearn_inverse_transform_pcs(
        old_val_predictions_pcs, old_pca_temp, old_pca_sal, n_components
    )
    old_pred_T = old_val_predictions[0]
    old_pred_S = old_val_predictions[1]

    print("NeSPReSO 1.0 model loaded and predictions computed.")

    orig_T = original_profiles[:, 0, :]
    orig_S = original_profiles[:, 1, :]
    old_T_resid = compute_profile_residual(old_pred_T, orig_T)
    old_S_resid = compute_profile_residual(old_pred_S, orig_S)

    # Start time
    start_time = time.perf_counter()
    # Get predictions for the validation dataset
    for i in range(gem_repeat_time):
        gem_temp, gem_sal = val_dataset.dataset.get_gem_profiles(subset_indices)
    # End time
    end_time = time.perf_counter()
    # Calculate elapsed time
    elapsed_time = (end_time - start_time) / gem_repeat_time
    print(f"GEM (cpu) & {n_val} & {elapsed_time * 1e3:.2f} & {((elapsed_time * 1e6) / n_val):.2f}")
    # print(f"GEM predictions {n_val} - Elapsed time: {elapsed_time:.6f} seconds, ran {gem_repeat_time} times.")
    gems_T = gem_temp.T
    gems_S = gem_sal.T

    pred_T_resid = pred_T - orig_T
    pred_S_resid = pred_S - orig_S
    gems_T_resid = gems_T - orig_T
    gems_S_resid = gems_S - orig_S

    sst_inputs, ssh_inputs = val_dataset.dataset.get_inputs(subset_indices)

    gem_temp, gem_sal = val_dataset.dataset.get_gem_profiles(subset_indices)

    lat_val, lon_val, dates_val = val_dataset.dataset.get_lat_lon_date(subset_indices)
    lat_val = np.floor(lat_val) + bin_size / 2
    lon_val = np.floor(lon_val) + bin_size / 2

    # visualize_combined_results(pca_approx_profiles, gem_temp, gem_sal, val_predictions, sst_inputs, ssh_inputs, min_depth=min_depth, max_depth = max_depth, num_samples=num_samples)

    printParams()

    print("Let's investigate how the method compares against vanilla GEM with in-situ SSH")

    ist = xr.open_dataset("/unity/g2/jmiranda/SubsurfaceFields/Data/isop1_stats_temp.nc")
    iss = xr.open_dataset("/unity/g2/jmiranda/SubsurfaceFields/Data/isop1_stats_salt.nc")

    our_depths = np.arange(0, 1801)
    isop_depths = ist.depth.values
    avg_gem_temp_rmse, avg_gem_temp_bias = compute_depth_rmse_bias(gems_T_resid, axis=1)
    avg_nn_temp_rmse, avg_nn_temp_bias = compute_depth_rmse_bias(pred_T_resid, axis=1)
    avg_old_temp_rmse, avg_old_temp_bias = compute_depth_rmse_bias(old_T_resid, axis=1)

    avg_gem_sal_rmse, avg_gem_sal_bias = compute_depth_rmse_bias(gems_S_resid, axis=1)
    avg_nn_sal_rmse, avg_nn_sal_bias = compute_depth_rmse_bias(pred_S_resid, axis=1)
    avg_old_sal_rmse, avg_old_sal_bias = compute_depth_rmse_bias(old_S_resid, axis=1)

    # Identify indices for the training, validation, and testing datasets
    train_indices = train_dataset.indices
    val_indices = val_dataset.indices
    test_indices = test_dataset.indices

    ## # --- Steric height (900 dbar) binning and T/S statistics on ISOP depth bins (full dataset) ---
    import matplotlib as mpl

    STERIC_REF_PRESSURE_DBAR = 900
    STERIC_BIN_WIDTH_M = 0.04

    n_profiles_full = len(full_dataset)
    TEMP_argo = full_dataset.TEMP
    SAL_argo = full_dataset.SAL

    # Use a simple 1 m pressure/depth vector like you already do
    PRES_argo = np.arange(full_dataset.min_depth, full_dataset.max_depth + 1, dtype=float)
    LAT_argo = full_dataset.LAT
    LON_argo = full_dataset.LON

    argo_depths = np.arange(full_dataset.min_depth, full_dataset.max_depth + 1, dtype=float)
    n_depths = len(argo_depths)

    # ---- steric height per profile ----
    steric_height_per_profile = np.full(n_profiles_full, np.nan)

    for i in range(n_profiles_full):
        p_col = PRES_argo
        t_col = TEMP_argo[:, i]
        s_col = SAL_argo[:, i]

        mask = (p_col <= STERIC_REF_PRESSURE_DBAR) & np.isfinite(p_col) & np.isfinite(t_col) & np.isfinite(s_col)
        if not np.any(mask):
            continue

        p = p_col[mask]
        t = t_col[mask]
        s = s_col[mask]

        # ensure p starts at 0
        if p[0] > 0:
            p = np.concatenate([[0.0], p])
            t = np.concatenate([[t[0]], t])
            s = np.concatenate([[s[0]], s])

        lat_i = LAT_argo[i]
        lon_i = LON_argo[i]

        SA = gsw.SA_from_SP(s, p, lon_i, lat_i)
        CT = gsw.CT_from_t(SA, t, p)
        geo_strf_dyn = gsw.geo_strf_dyn_height(SA, CT, p, p_ref=STERIC_REF_PRESSURE_DBAR, axis=0)

        # convert dyn height to meters (your constant)
        steric_height_per_profile[i] = geo_strf_dyn[0] / 9.7963

    valid_steric = np.isfinite(steric_height_per_profile)
    steric_height_valid = steric_height_per_profile[valid_steric]
    steric_height_valid = steric_height_valid - np.nanmin(steric_height_valid)

    # ---- steric bins (EDGES are what we will plot with pcolormesh) ----
    steric_bin_edges = np.arange(
        np.floor(np.nanmin(steric_height_valid) / STERIC_BIN_WIDTH_M) * STERIC_BIN_WIDTH_M,
        np.ceil(np.nanmax(steric_height_valid) / STERIC_BIN_WIDTH_M) * STERIC_BIN_WIDTH_M + STERIC_BIN_WIDTH_M * 0.5,
        STERIC_BIN_WIDTH_M,
    )
    n_steric_bins = len(steric_bin_edges) - 1
    steric_bin_centers = 0.5 * (steric_bin_edges[:-1] + steric_bin_edges[1:])

    steric_bin_idx_all = np.full(n_profiles_full, -1, dtype=int)
    steric_bin_idx_all[valid_steric] = np.clip(
        np.digitize(steric_height_valid, steric_bin_edges, right=False) - 1, 0, n_steric_bins - 1
    )

    # ---- depth/ISOP bins (EDGES) ----
    isop_depth_values = ist.depth.values
    isop_depth_edges = np.sort(np.unique(isop_depth_values))

    # safety: need at least 2 edges
    if len(isop_depth_edges) < 2:
        isop_depth_edges = np.array([argo_depths[0], argo_depths[-1]], dtype=float)

    n_depth_bins = len(isop_depth_edges) - 1
    isop_depth_bin_centers = 0.5 * (isop_depth_edges[:-1] + isop_depth_edges[1:])

    depth_bin_idx_per_level = np.digitize(argo_depths, isop_depth_edges, right=False) - 1
    depth_bin_idx_per_level = np.clip(depth_bin_idx_per_level, 0, n_depth_bins - 1)

    # ---- model predictions back to full T/S ----
    full_dataset_subset = Subset(full_dataset, range(n_profiles_full))
    full_loader = DataLoader(full_dataset_subset, batch_size=batch_size, shuffle=False)
    full_predictions_pcs = get_predictions(trained_model, full_loader, device)
    T_nespreso_full, S_nespreso_full = full_dataset.inverse_transform(full_predictions_pcs)

    # ---- allocate stats ----
    mean_T_argo = np.full((n_steric_bins, n_depth_bins), np.nan)
    std_T_argo = np.full((n_steric_bins, n_depth_bins), np.nan)
    mean_S_argo = np.full((n_steric_bins, n_depth_bins), np.nan)
    std_S_argo = np.full((n_steric_bins, n_depth_bins), np.nan)

    mean_T_nespreso = np.full((n_steric_bins, n_depth_bins), np.nan)
    std_T_nespreso = np.full((n_steric_bins, n_depth_bins), np.nan)
    mean_S_nespreso = np.full((n_steric_bins, n_depth_bins), np.nan)
    std_S_nespreso = np.full((n_steric_bins, n_depth_bins), np.nan)

    # ---- compute pooled stats per (steric bin, depth bin) ----
    for sb in range(n_steric_bins):
        sb_mask = steric_bin_idx_all[None, :] == sb

        for db in range(n_depth_bins):
            db_mask = depth_bin_idx_per_level[:, None] == db
            mask = sb_mask & db_mask

            t_argo_pool = TEMP_argo[mask]
            s_argo_pool = SAL_argo[mask]
            t_nesp_pool = T_nespreso_full[mask]
            s_nesp_pool = S_nespreso_full[mask]

            if np.any(np.isfinite(t_argo_pool)):
                mean_T_argo[sb, db] = np.nanmean(t_argo_pool)
                std_T_argo[sb, db] = np.nanstd(t_argo_pool)

            if np.any(np.isfinite(s_argo_pool)):
                mean_S_argo[sb, db] = np.nanmean(s_argo_pool)
                std_S_argo[sb, db] = np.nanstd(s_argo_pool)

            if np.any(np.isfinite(t_nesp_pool)):
                mean_T_nespreso[sb, db] = np.nanmean(t_nesp_pool)
                std_T_nespreso[sb, db] = np.nanstd(t_nesp_pool)

            if np.any(np.isfinite(s_nesp_pool)):
                mean_S_nespreso[sb, db] = np.nanmean(s_nesp_pool)
                std_S_nespreso[sb, db] = np.nanstd(s_nesp_pool)

    diff_mean_T = mean_T_nespreso - mean_T_argo
    diff_mean_S = mean_S_nespreso - mean_S_argo

    # =============================================================================
    # PLOTTING: blocky tiles + shared colormaps/colorbars (like your first figure)
    # =============================================================================

    # pcolormesh wants Z shaped (ny, nx) = (n_depth_bins, n_steric_bins)
    Tstd_argo = std_T_argo.T
    Tstd_nesp = std_T_nespreso.T
    Sstd_argo = std_S_argo.T
    Sstd_nesp = std_S_nespreso.T

    # shared scaling within variable (T panels share; S panels share)
    vmin_T = np.nanmin([Tstd_argo, Tstd_nesp])
    vmax_T = np.nanmax([Tstd_argo, Tstd_nesp])
    vmin_S = np.nanmin([Sstd_argo, Sstd_nesp])
    vmax_S = np.nanmax([Sstd_argo, Sstd_nesp])

    norm_T = mpl.colors.Normalize(vmin=vmin_T, vmax=vmax_T)
    norm_S = mpl.colors.Normalize(vmin=vmin_S, vmax=vmax_S)

    fig_steric_std, axs_steric_std = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)

    # --- Argo T std ---
    mT0 = axs_steric_std[0, 0].pcolormesh(
        steric_bin_edges,
        isop_depth_edges,
        Tstd_argo,
        cmap="jet",
        norm=norm_T,
        shading="flat",
        linewidth=0,
        antialiased=False,
        rasterized=True,
    )
    axs_steric_std[0, 0].set_title("Argo T std")

    # --- Argo S std ---
    mS0 = axs_steric_std[0, 1].pcolormesh(
        steric_bin_edges,
        isop_depth_edges,
        Sstd_argo,
        cmap="jet",
        norm=norm_S,
        shading="flat",
        linewidth=0,
        antialiased=False,
        rasterized=True,
    )
    axs_steric_std[0, 1].set_title("Argo S std")

    # --- NeSPReSO T std ---
    mT1 = axs_steric_std[1, 0].pcolormesh(
        steric_bin_edges,
        isop_depth_edges,
        Tstd_nesp,
        cmap="jet",
        norm=norm_T,
        shading="flat",
        linewidth=0,
        antialiased=False,
        rasterized=True,
    )
    axs_steric_std[1, 0].set_title("NeSPReSO T std")

    # --- NeSPReSO S std ---
    mS1 = axs_steric_std[1, 1].pcolormesh(
        steric_bin_edges,
        isop_depth_edges,
        Sstd_nesp,
        cmap="jet",
        norm=norm_S,
        shading="flat",
        linewidth=0,
        antialiased=False,
        rasterized=True,
    )
    axs_steric_std[1, 1].set_title("NeSPReSO S std")

    for ax in axs_steric_std.ravel():
        ax.set_xlabel("Steric height (m, ref 900 dbar)")
        ax.set_ylabel("ISOP depth (m)")
        ax.set_ylim(0, 1000)
        ax.invert_yaxis()

    # one colorbar for BOTH T panels (left column)
    cbarT = fig_steric_std.colorbar(mT0, ax=[axs_steric_std[0, 0], axs_steric_std[1, 0]], pad=0.02)
    cbarT.set_label("°C")

    # one colorbar for BOTH S panels (right column)
    cbarS = fig_steric_std.colorbar(mS0, ax=[axs_steric_std[0, 1], axs_steric_std[1, 1]], pad=0.02)
    cbarS.set_label("PSU")

    plt.show()

    # =============================================================================
    # DIFF STD: blocky tiles (per-variable symmetric scaling)
    # =============================================================================

    diff_std_T = (std_T_nespreso - std_T_argo).T
    diff_std_S = (std_S_nespreso - std_S_argo).T

    vmax_diff_std_T = np.nanmax(np.abs(diff_std_T))
    vmax_diff_std_S = np.nanmax(np.abs(diff_std_S))
    if (not np.isfinite(vmax_diff_std_T)) or vmax_diff_std_T <= 0:
        vmax_diff_std_T = 0.01
    if (not np.isfinite(vmax_diff_std_S)) or vmax_diff_std_S <= 0:
        vmax_diff_std_S = 0.01

    fig_steric_diff_std, axs_steric_diff_std = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    mDT = axs_steric_diff_std[0].pcolormesh(
        steric_bin_edges,
        isop_depth_edges,
        diff_std_T,
        cmap=coolwhitewarm,
        norm=mpl.colors.Normalize(vmin=-vmax_diff_std_T, vmax=vmax_diff_std_T),
        shading="flat",
        linewidth=0,
        antialiased=False,
        rasterized=True,
    )
    axs_steric_diff_std[0].set_title("NeSPReSO − Argo std T")
    axs_steric_diff_std[0].set_xlabel("Steric height (m, ref 900 dbar)")
    axs_steric_diff_std[0].set_ylabel("ISOP depth (m)")
    axs_steric_diff_std[0].set_ylim(0, 1000)
    axs_steric_diff_std[0].invert_yaxis()
    fig_steric_diff_std.colorbar(mDT, ax=axs_steric_diff_std[0], label="°C", pad=0.02)

    mDS = axs_steric_diff_std[1].pcolormesh(
        steric_bin_edges,
        isop_depth_edges,
        diff_std_S,
        cmap=coolwhitewarm,
        norm=mpl.colors.Normalize(vmin=-vmax_diff_std_S, vmax=vmax_diff_std_S),
        shading="flat",
        linewidth=0,
        antialiased=False,
        rasterized=True,
    )
    axs_steric_diff_std[1].set_title("NeSPReSO − Argo std S")
    axs_steric_diff_std[1].set_xlabel("Steric height (m, ref 900 dbar)")
    axs_steric_diff_std[1].set_ylabel("ISOP depth (m)")
    axs_steric_diff_std[1].set_ylim(0, 1000)
    axs_steric_diff_std[1].invert_yaxis()
    fig_steric_diff_std.colorbar(mDS, ax=axs_steric_diff_std[1], label="PSU", pad=0.02)

    plt.show()

    ## repeat the same analysis for the month of august only (all years)
    def _get_month(t):
        return t.month if hasattr(t, "month") else datenum_to_datetime(t).month

    august_mask = np.array([_get_month(t) == 8 for t in full_dataset.TIME])
    august_indices = np.where(august_mask)[0]
    n_profiles_august = len(august_indices)

    TEMP_argo_aug = full_dataset.TEMP[:, august_indices]
    SAL_argo_aug = full_dataset.SAL[:, august_indices]
    LAT_argo_aug = full_dataset.LAT[august_indices]
    LON_argo_aug = full_dataset.LON[august_indices]

    steric_height_per_profile_aug = np.full(n_profiles_august, np.nan)
    for i in range(n_profiles_august):
        p_col = PRES_argo
        t_col = TEMP_argo_aug[:, i]
        s_col = SAL_argo_aug[:, i]

        mask = (p_col <= STERIC_REF_PRESSURE_DBAR) & np.isfinite(p_col) & np.isfinite(t_col) & np.isfinite(s_col)
        if not np.any(mask):
            continue

        p = p_col[mask]
        t = t_col[mask]
        s = s_col[mask]

        if p[0] > 0:
            p = np.concatenate([[0.0], p])
            t = np.concatenate([[t[0]], t])
            s = np.concatenate([[s[0]], s])

        lat_i = LAT_argo_aug[i]
        lon_i = LON_argo_aug[i]

        SA = gsw.SA_from_SP(s, p, lon_i, lat_i)
        CT = gsw.CT_from_t(SA, t, p)
        geo_strf_dyn = gsw.geo_strf_dyn_height(SA, CT, p, p_ref=STERIC_REF_PRESSURE_DBAR, axis=0)
        steric_height_per_profile_aug[i] = geo_strf_dyn[0] / 9.7963

    valid_steric_aug = np.isfinite(steric_height_per_profile_aug)
    steric_height_valid_aug = steric_height_per_profile_aug[valid_steric_aug]
    steric_height_valid_aug = steric_height_valid_aug - np.nanmin(steric_height_valid_aug)

    steric_bin_edges_aug = np.arange(
        np.floor(np.nanmin(steric_height_valid_aug) / STERIC_BIN_WIDTH_M) * STERIC_BIN_WIDTH_M,
        np.ceil(np.nanmax(steric_height_valid_aug) / STERIC_BIN_WIDTH_M) * STERIC_BIN_WIDTH_M
        + STERIC_BIN_WIDTH_M * 0.5,
        STERIC_BIN_WIDTH_M,
    )
    n_steric_bins_aug = len(steric_bin_edges_aug) - 1

    steric_bin_idx_all_aug = np.full(n_profiles_august, -1, dtype=int)
    steric_bin_idx_all_aug[valid_steric_aug] = np.clip(
        np.digitize(steric_height_valid_aug, steric_bin_edges_aug, right=False) - 1, 0, n_steric_bins_aug - 1
    )

    full_dataset_subset_august = Subset(full_dataset, august_indices)
    full_loader_august = DataLoader(full_dataset_subset_august, batch_size=batch_size, shuffle=False)
    full_predictions_pcs_august = get_predictions(trained_model, full_loader_august, device)
    T_nespreso_august, S_nespreso_august = full_dataset.inverse_transform(full_predictions_pcs_august)

    mean_T_argo_aug = np.full((n_steric_bins_aug, n_depth_bins), np.nan)
    std_T_argo_aug = np.full((n_steric_bins_aug, n_depth_bins), np.nan)
    mean_S_argo_aug = np.full((n_steric_bins_aug, n_depth_bins), np.nan)
    std_S_argo_aug = np.full((n_steric_bins_aug, n_depth_bins), np.nan)
    mean_T_nespreso_aug = np.full((n_steric_bins_aug, n_depth_bins), np.nan)
    std_T_nespreso_aug = np.full((n_steric_bins_aug, n_depth_bins), np.nan)
    mean_S_nespreso_aug = np.full((n_steric_bins_aug, n_depth_bins), np.nan)
    std_S_nespreso_aug = np.full((n_steric_bins_aug, n_depth_bins), np.nan)

    for sb in range(n_steric_bins_aug):
        sb_mask = steric_bin_idx_all_aug[None, :] == sb
        for db in range(n_depth_bins):
            db_mask = depth_bin_idx_per_level[:, None] == db
            mask = sb_mask & db_mask

            t_argo_pool = TEMP_argo_aug[mask]
            s_argo_pool = SAL_argo_aug[mask]
            t_nesp_pool = T_nespreso_august[mask]
            s_nesp_pool = S_nespreso_august[mask]

            if np.any(np.isfinite(t_argo_pool)):
                mean_T_argo_aug[sb, db] = np.nanmean(t_argo_pool)
                std_T_argo_aug[sb, db] = np.nanstd(t_argo_pool)
            if np.any(np.isfinite(s_argo_pool)):
                mean_S_argo_aug[sb, db] = np.nanmean(s_argo_pool)
                std_S_argo_aug[sb, db] = np.nanstd(s_argo_pool)
            if np.any(np.isfinite(t_nesp_pool)):
                mean_T_nespreso_aug[sb, db] = np.nanmean(t_nesp_pool)
                std_T_nespreso_aug[sb, db] = np.nanstd(t_nesp_pool)
            if np.any(np.isfinite(s_nesp_pool)):
                mean_S_nespreso_aug[sb, db] = np.nanmean(s_nesp_pool)
                std_S_nespreso_aug[sb, db] = np.nanstd(s_nesp_pool)

    Tstd_argo_aug = std_T_argo_aug.T
    Tstd_nesp_aug = std_T_nespreso_aug.T
    Sstd_argo_aug = std_S_argo_aug.T
    Sstd_nesp_aug = std_S_nespreso_aug.T

    vmin_T_aug = np.nanmin([Tstd_argo_aug, Tstd_nesp_aug])
    vmax_T_aug = np.nanmax([Tstd_argo_aug, Tstd_nesp_aug])
    vmin_S_aug = np.nanmin([Sstd_argo_aug, Sstd_nesp_aug])
    vmax_S_aug = np.nanmax([Sstd_argo_aug, Sstd_nesp_aug])

    norm_T_aug = mpl.colors.Normalize(vmin=vmin_T_aug, vmax=vmax_T_aug)
    norm_S_aug = mpl.colors.Normalize(vmin=vmin_S_aug, vmax=vmax_S_aug)

    fig_steric_std_aug, axs_steric_std_aug = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)

    mT0_aug = axs_steric_std_aug[0, 0].pcolormesh(
        steric_bin_edges_aug,
        isop_depth_edges,
        Tstd_argo_aug,
        cmap="jet",
        norm=norm_T_aug,
        shading="flat",
        linewidth=0,
        antialiased=False,
        rasterized=True,
    )
    axs_steric_std_aug[0, 0].set_title("Argo T std (August)")

    mS0_aug = axs_steric_std_aug[0, 1].pcolormesh(
        steric_bin_edges_aug,
        isop_depth_edges,
        Sstd_argo_aug,
        cmap="jet",
        norm=norm_S_aug,
        shading="flat",
        linewidth=0,
        antialiased=False,
        rasterized=True,
    )
    axs_steric_std_aug[0, 1].set_title("Argo S std (August)")

    mT1_aug = axs_steric_std_aug[1, 0].pcolormesh(
        steric_bin_edges_aug,
        isop_depth_edges,
        Tstd_nesp_aug,
        cmap="jet",
        norm=norm_T_aug,
        shading="flat",
        linewidth=0,
        antialiased=False,
        rasterized=True,
    )
    axs_steric_std_aug[1, 0].set_title("NeSPReSO T std (August)")

    mS1_aug = axs_steric_std_aug[1, 1].pcolormesh(
        steric_bin_edges_aug,
        isop_depth_edges,
        Sstd_nesp_aug,
        cmap="jet",
        norm=norm_S_aug,
        shading="flat",
        linewidth=0,
        antialiased=False,
        rasterized=True,
    )
    axs_steric_std_aug[1, 1].set_title("NeSPReSO S std (August)")

    for ax in axs_steric_std_aug.ravel():
        ax.set_xlabel("Steric height (m, ref 900 dbar)")
        ax.set_ylabel("ISOP depth (m)")
        ax.set_ylim(0, 1000)
        ax.invert_yaxis()

    cbarT_aug = fig_steric_std_aug.colorbar(mT0_aug, ax=[axs_steric_std_aug[0, 0], axs_steric_std_aug[1, 0]], pad=0.02)
    cbarT_aug.set_label("°C")
    cbarS_aug = fig_steric_std_aug.colorbar(mS0_aug, ax=[axs_steric_std_aug[0, 1], axs_steric_std_aug[1, 1]], pad=0.02)
    cbarS_aug.set_label("PSU")
    plt.show()

    diff_std_T_aug = (std_T_nespreso_aug - std_T_argo_aug).T
    diff_std_S_aug = (std_S_nespreso_aug - std_S_argo_aug).T

    vmax_diff_std_T_aug = np.nanmax(np.abs(diff_std_T_aug))
    vmax_diff_std_S_aug = np.nanmax(np.abs(diff_std_S_aug))
    if (not np.isfinite(vmax_diff_std_T_aug)) or vmax_diff_std_T_aug <= 0:
        vmax_diff_std_T_aug = 0.01
    if (not np.isfinite(vmax_diff_std_S_aug)) or vmax_diff_std_S_aug <= 0:
        vmax_diff_std_S_aug = 0.01

    fig_steric_diff_std_aug, axs_steric_diff_std_aug = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    mDT_aug = axs_steric_diff_std_aug[0].pcolormesh(
        steric_bin_edges_aug,
        isop_depth_edges,
        diff_std_T_aug,
        cmap=coolwhitewarm,
        norm=mpl.colors.Normalize(vmin=-vmax_diff_std_T_aug, vmax=vmax_diff_std_T_aug),
        shading="flat",
        linewidth=0,
        antialiased=False,
        rasterized=True,
    )
    axs_steric_diff_std_aug[0].set_title("NeSPReSO − Argo std T (August)")
    axs_steric_diff_std_aug[0].set_xlabel("Steric height (m, ref 900 dbar)")
    axs_steric_diff_std_aug[0].set_ylabel("ISOP depth (m)")
    axs_steric_diff_std_aug[0].set_ylim(0, 1000)
    axs_steric_diff_std_aug[0].invert_yaxis()
    fig_steric_diff_std_aug.colorbar(mDT_aug, ax=axs_steric_diff_std_aug[0], label="°C", pad=0.02)

    mDS_aug = axs_steric_diff_std_aug[1].pcolormesh(
        steric_bin_edges_aug,
        isop_depth_edges,
        diff_std_S_aug,
        cmap=coolwhitewarm,
        norm=mpl.colors.Normalize(vmin=-vmax_diff_std_S_aug, vmax=vmax_diff_std_S_aug),
        shading="flat",
        linewidth=0,
        antialiased=False,
        rasterized=True,
    )
    axs_steric_diff_std_aug[1].set_title("NeSPReSO − Argo std S (August)")
    axs_steric_diff_std_aug[1].set_xlabel("Steric height (m, ref 900 dbar)")
    axs_steric_diff_std_aug[1].set_ylabel("ISOP depth (m)")
    axs_steric_diff_std_aug[1].set_ylim(0, 1000)
    axs_steric_diff_std_aug[1].invert_yaxis()
    fig_steric_diff_std_aug.colorbar(mDS_aug, ax=axs_steric_diff_std_aug[1], label="PSU", pad=0.02)

    plt.show()

    ## Multiple linear regression

    MLR_degree = 1
    MLR_indices = train_indices + val_indices
    inputs_list_train, pcs_list_train = full_dataset.__getitem__(MLR_indices)

    inputs_array_train = np.array(inputs_list_train.T)
    pcs_T_train, pcs_S_train = np.hsplit(pcs_list_train, 2)
    pcs_T_train = pcs_T_train.T
    pcs_S_train = pcs_S_train.T

    X_train = prepare_features(inputs_array_train, max_degree=MLR_degree)
    scaler = StandardScaler()
    X_avgs = X_train.mean(axis=0)
    X_train_scaled = scaler.fit_transform(X_train) + 1

    print("Beginning multiple linear regression using PyTorch on GPU")
    start_time = time.perf_counter()
    beta_T = fit_pcs_regression_exact_gpu(X_train, pcs_T_train)
    beta_S = fit_pcs_regression_exact_gpu(X_train, pcs_S_train)
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"MLR fit completed in {elapsed_time:.2f} seconds.")

    inputs_list_val, _ = full_dataset.__getitem__(val_indices)
    inputs_array_val = np.array(inputs_list_val.T)
    X_val = prepare_features(inputs_array_val, max_degree=MLR_degree)
    X_val_scaled = scaler.fit_transform(X_val) + 1

    pcs_pred_val_T = predict_pcs_exact_gpu(beta_T, X_val)
    pcs_pred_val_S = predict_pcs_exact_gpu(beta_S, X_val)
    pcs_pred_val = np.hstack([pcs_pred_val_T, pcs_pred_val_S])

    pca_temp = full_dataset.pca_temp
    pca_sal = full_dataset.pca_sal
    temp_MLR_profiles, sal_MLR_profiles = inverse_transform(pcs_pred_val, pca_temp, pca_sal, n_components)

    # Extract the original temperature and salinity profiles
    original_temp_profiles = original_profiles[:, 0, :]  # Shape: (n_samples, depth_levels)
    original_sal_profiles = original_profiles[:, 1, :]  # Shape: (n_samples, depth_levels)

    # Calculate residuals
    mlr_T_resid = compute_profile_residual(temp_MLR_profiles, original_temp_profiles)
    mlr_S_resid = compute_profile_residual(sal_MLR_profiles, original_sal_profiles)

    # Compute average RMSE and bias
    avg_mlr_temp_rmse, avg_mlr_temp_bias = compute_depth_rmse_bias(mlr_T_resid, axis=1)
    avg_mlr_sal_rmse, avg_mlr_sal_bias = compute_depth_rmse_bias(mlr_S_resid, axis=1)

    fig = plt.figure(figsize=(18, 18))

    # Temperature RMSE Plot
    ax = fig.add_subplot(2, 2, 1)
    ax.axvline(0, color="k", linestyle="--", linewidth=0.5)
    ax.grid(color="gray", linestyle="--", linewidth=0.5)
    plt.plot(ist.rmse.values, ist.depth.values, linewidth=3, label="ISOP", color="xkcd:blue")
    plt.plot(avg_gem_temp_rmse, np.arange(0, 1801), linewidth=3, label="GEM", color="xkcd:orange")
    plt.plot(avg_mlr_temp_rmse, np.arange(0, 1801), linewidth=3, label="MLR", color="xkcd:green")
    plt.plot(avg_old_temp_rmse, np.arange(0, 1801), linewidth=3, label="NeSPReSO 1.0", color="xkcd:purple")
    plt.plot(avg_nn_temp_rmse, np.arange(0, 1801), linewidth=3, label="NeSPReSO 1.1", color="xkcd:gray")
    ax.invert_yaxis()
    plt.legend()
    plt.xlabel("Temperature RMSE [°C]")
    plt.ylabel("Depth [m]")
    plt.title("Average Temperature RMSE")

    # Salinity RMSE Plot
    ax = fig.add_subplot(2, 2, 2)
    ax.axvline(0, color="k", linestyle="--", linewidth=0.5)
    ax.grid(color="gray", linestyle="--", linewidth=0.5)
    plt.plot(iss.rmse.values, iss.depth.values, linewidth=3, label="ISOP", color="xkcd:blue")
    plt.plot(avg_gem_sal_rmse, np.arange(0, 1801), linewidth=3, label="GEM", color="xkcd:orange")
    plt.plot(avg_mlr_sal_rmse, np.arange(0, 1801), linewidth=3, label="MLR", color="xkcd:green")
    plt.plot(avg_old_sal_rmse, np.arange(0, 1801), linewidth=3, label="NeSPReSO 1.0", color="xkcd:purple")
    plt.plot(avg_nn_sal_rmse, np.arange(0, 1801), linewidth=3, label="NeSPReSO 1.1", color="xkcd:gray")
    ax.invert_yaxis()
    plt.legend()
    plt.xlabel("Salinity RMSE [PSU]")
    plt.title("Average Salinity RMSE")

    # Temperature Bias Plot
    ax = fig.add_subplot(2, 2, 3)
    ax.axvline(0, color="k", linestyle="--", linewidth=0.5)
    ax.grid(color="gray", linestyle="--", linewidth=0.5)
    plt.plot(ist.bias.values, ist.depth.values, linewidth=3, label="ISOP", color="xkcd:blue")
    plt.plot(avg_gem_temp_bias, np.arange(0, 1801), linewidth=3, label="GEM", color="xkcd:orange")
    plt.plot(avg_mlr_temp_bias, np.arange(0, 1801), linewidth=3, label="MLR", color="xkcd:green")
    plt.plot(avg_old_temp_bias, np.arange(0, 1801), linewidth=3, label="NeSPReSO 1.0", color="xkcd:purple")
    plt.plot(avg_nn_temp_bias, np.arange(0, 1801), linewidth=3, label="NeSPReSO 1.1", color="xkcd:gray")
    ax.invert_yaxis()
    plt.legend()
    plt.xlabel("Temperature Bias [°C]")
    plt.ylabel("Depth [m]")
    plt.title("Average Temperature Bias")

    # Salinity Bias Plot
    ax = fig.add_subplot(2, 2, 4)
    ax.axvline(0, color="k", linestyle="--", linewidth=0.5)
    ax.grid(color="gray", linestyle="--", linewidth=0.5)
    plt.plot(iss.bias.values, iss.depth.values, linewidth=3, label="ISOP", color="xkcd:blue")
    plt.plot(avg_gem_sal_bias, np.arange(0, 1801), linewidth=3, label="GEM", color="xkcd:orange")
    plt.plot(avg_mlr_sal_bias, np.arange(0, 1801), linewidth=3, label="MLR", color="xkcd:green")
    plt.plot(avg_old_sal_bias, np.arange(0, 1801), linewidth=3, label="NeSPReSO 1.0", color="xkcd:purple")
    plt.plot(avg_nn_sal_bias, np.arange(0, 1801), linewidth=3, label="NeSPReSO 1.1", color="xkcd:gray")
    ax.invert_yaxis()
    plt.legend()
    plt.xlabel("Salinity Bias [PSU]")
    plt.title("Average Salinity Bias")

    plt.tight_layout()
    plt.show()

    # visualize features
    feature_names = ["timecos", "timesin", "latcos", "latsin", "loncos", "lonsin", "sst", "sss", "ssh"]


    # MLR coefficient analysis
    beta_T_scaled = beta_T.cpu() / X_avgs[:, None]
    beta_S_scaled = beta_S.cpu() / X_avgs[:, None]
    beta_T_dropped = torch.cat((beta_T_scaled[:2], beta_T_scaled[6:]), dim=0)
    beta_S_dropped = torch.cat((beta_S_scaled[:2], beta_S_scaled[6:]), dim=0)
    feature_names_dropped = feature_names[:2] + feature_names[6:]

    plot_coefficients_heatmap(
        beta_T_dropped,
        feature_names_dropped,
        "Normalized Regression Coefficients for Temperature PCs",
        normalize=True,
    )

    plot_coefficients_heatmap(
        beta_S_dropped,
        feature_names_dropped,
        "Normalized Regression Coefficients for Salinity PCs",
        normalize=True,
    )

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
    def get_season(date):
        month = date.month
        if month in [3, 4, 5]:
            return "Spring"
        elif month in [6, 7, 8]:
            return "Summer"
        elif month in [9, 10, 11]:
            return "Autumn"
        else:
            return "Winter"

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

    # Residual calculations
    nn_temp_residuals = pred_T - original_profiles[:, 0, :]
    nn_sal_residuals = pred_S - original_profiles[:, 1, :]

    ## GLIDER: Load the MATLAB file
    file_path = "/unity/g2/jmiranda/SubsurfaceFields/Data/Glider_binned_data_for_heat_content_IA_mission_lowpass_LCE_Campeche_cyclone.mat"
    gl_data = scipy.io.loadmat(file_path)

    # Display the keys to understand the structure of the data
    print(gl_data.keys())

    # Variable	Shape   	nan_min	    nan_max	    nan_avg	    nan_count
    # T1	    (201,240)	5.043881	40.509893	12.093195	387
    # S1	    (201,240)	32.740974	36.657226	35.476389	387
    # lon1	    (1,	240)	-94.817717	-94.405167	-94.690244	0
    # lat1	    (1,	240)	24.02255	27.0873	    25.840945	0
    # t1	    (1,	240)	736863.0151	736883.0033	736873.3337	0
    # T2	    (201,238)	4.708326	32.524226	12.129467	426
    # S2	    (201,238)	30.45545	36.783535	35.478705	426
    # lon2	    (1,	238)	-95.757333	-93.456633	-94.694365	0
    # lat2	    (1,	238)	22.9795	    26.09425	24.404746	0
    # t2	    (1,	238)	736892.9703	736914.0272	736902.8855	0
    # T3	    (201,223)	4.648858	27.38052	10.070962	271
    # S3	    (201,223)	34.902126	36.851436	35.290017	271
    # lon3	    (1,	223)	-95.99222	-93.557308	-94.752642	0
    # lat3	    (1,	223)	19.702925	21.887858	20.131646	0
    # t3	    (1,	223)	737142.9568	737180.9581	737161.8109	0
    # T4	    (201,301)	5.099238	31.526334	13.782459	4914
    # S4	    (201,301)	34.867155	37.01149	35.656481	4914
    # lon4	    (1,	301)	-89.085857	-87.716895	-88.320707	0
    # lat4	    (1,	301)	24.371618	26.146342	25.573801	0
    # t4	    (1,	301)	737254.5333	737272.0137	737263.1334	0
    # T1l	    (201,240)	4.971843	29.91509	12.041973	0
    # S1l	    (201,240)	32.886428	41.790276	35.479336	0
    # T2l	    (201,238)	4.70553	    30.756818	12.094563	0
    # S2l	    (201,238)	29.426921	36.761195	35.464176	0
    # T3l	    (201,223)	4.753081	27.064557	10.053855	223
    # S3l	    (201,223)	34.903059	36.599977	35.288714	223
    # T4l	    (201,301)	5.125582	31.045829	13.877145	4914
    # S4l	    (201,301)	34.92949	37.019112	35.665196	4914
    # t1l	    (1,	240)	736863.0151	736883.0033	736873.0092	0
    # t2l	    (1,	238)	736892.9703	736914.0272	736903.4988	0
    # t3l	    (1,	223)	737142.9568	737180.9581	737161.9574	0
    # t4l	    (1,	301)	737254.5333	737272.0137	737263.2735	0

    # Extract locations and distance
    latitudes_T1 = gl_data["lat1"][0]
    longitudes_T1 = gl_data["lon1"][0]
    latitudes_T2 = gl_data["lat2"][0]
    longitudes_T2 = gl_data["lon2"][0]
    latitudes_T3 = gl_data["lat3"][0]
    longitudes_T3 = gl_data["lon3"][0]
    latitudes_T4 = gl_data["lat4"][0]
    longitudes_T4 = gl_data["lon4"][0]
    d1 = calculate_distances(latitudes_T1, longitudes_T1)
    d2 = calculate_distances(latitudes_T2, longitudes_T2)
    d3 = calculate_distances(latitudes_T3, longitudes_T3)
    d4 = calculate_distances(latitudes_T4, longitudes_T4)

    # times
    tt1 = gl_data["t1"][0]
    tt2 = gl_data["t2"][0]
    tt3 = gl_data["t3"][0]
    tt4 = gl_data["t4"][0]
    t1 = datenums_to_datetimes(tt1)
    t2 = datenums_to_datetimes(tt2)
    t3 = datenums_to_datetimes(tt3)
    t4 = datenums_to_datetimes(tt4)

    # Extract temperature and salinity
    T1 = gl_data["T1"]
    S1 = gl_data["S1"]
    T2 = gl_data["T2"]
    S2 = gl_data["S2"]
    T3 = gl_data["T3"]
    S3 = gl_data["S3"]
    T4 = gl_data["T4"]
    S4 = gl_data["S4"]

    if "sss1" in data:
        # with open(dataset_pickle_file, 'rb') as file:
        #     data = pickle.load(file)

        sss1 = data["sss1"]
        sss2 = data["sss2"]
        sss3 = data["sss3"]
        sss4 = data["sss4"]
        sst1 = data["sst1"]
        sst2 = data["sst2"]
        sst3 = data["sst3"]
        sst4 = data["sst4"]
        aviso1 = data["aviso1"]
        aviso2 = data["aviso2"]
        aviso3 = data["aviso3"]
        aviso4 = data["aviso4"]

    else:
        # Extract aviso, sst, sss
        sss1, sst1, aviso1 = load_satellite_data(t1, latitudes_T1, longitudes_T1)
        sss2, sst2, aviso2 = load_satellite_data(t2, latitudes_T2, longitudes_T2)
        sss3, sst3, aviso3 = load_satellite_data(t3, latitudes_T3, longitudes_T3)
        sss4, sst4, aviso4 = load_satellite_data(t4, latitudes_T4, longitudes_T4)

        data["sss1"], data["sst1"], data["aviso1"] = sss1, sst1, aviso1
        data["sss2"], data["sst2"], data["aviso2"] = sss2, sst2, aviso2
        data["sss3"], data["sst3"], data["aviso3"] = sss3, sst3, aviso3
        data["sss4"], data["sst4"], data["aviso4"] = sss4, sst4, aviso4

        with open(dataset_pickle_file, "wb") as file:
            data = {
                "min_depth": min_depth,
                "max_depth": max_depth,
                "epochs": epochs,
                "patience": patience,
                "n_components": n_components,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "dropout_prob": dropout_prob,
                "layers_config": layers_config,
                "input_params": input_params,
                "train_size": train_size,
                "val_size": val_size,
                "test_size": test_size,
                "full_dataset": full_dataset,
                "sss1": sss1,
                "sss2": sss2,
                "sss3": sss3,
                "sss4": sss4,
                "sst1": sst1,
                "sst2": sst2,
                "sst3": sst3,
                "sst4": sst4,
                "aviso1": aviso1,
                "aviso2": aviso2,
                "aviso3": aviso3,
                "aviso4": aviso4,
            }
            pickle.dump(data, file)

    # Prepare the inputs
    gld_tensor1 = prepare_inputs(tt1, latitudes_T1, longitudes_T1, sss1, sst1, aviso1, input_params)
    gld_tensor2 = prepare_inputs(tt2, latitudes_T2, longitudes_T2, sss2, sst2, aviso2, input_params)
    gld_tensor3 = prepare_inputs(tt3, latitudes_T3, longitudes_T3, sss3, sst3, aviso3, input_params)
    gld_tensor4 = prepare_inputs(tt4, latitudes_T4, longitudes_T4, sss4, sst4, aviso4, input_params)

    # depth vector
    gld_depths = np.arange(0, 201 * 5, 5)

    pred_max_depth = 1004

    T_pred1, S_pred1 = get_glider_predictions(
        trained_model,
        val_loader,
        gld_tensor1,
        device,
        val_dataset.dataset.inverse_transform,
        max_depth=pred_max_depth,
    )
    T_pred2, S_pred2 = get_glider_predictions(
        trained_model,
        val_loader,
        gld_tensor2,
        device,
        val_dataset.dataset.inverse_transform,
        max_depth=pred_max_depth,
    )
    T_pred3, S_pred3 = get_glider_predictions(
        trained_model,
        val_loader,
        gld_tensor3,
        device,
        val_dataset.dataset.inverse_transform,
        max_depth=pred_max_depth,
    )
    T_pred4, S_pred4 = get_glider_predictions(
        trained_model,
        val_loader,
        gld_tensor4,
        device,
        val_dataset.dataset.inverse_transform,
        max_depth=pred_max_depth,
    )

    pred_depths = np.arange(0, pred_max_depth + 1, 1)

    # Define the bin size
    bin_size = 5

    # Bin the predicted data
    T_pred1_binned = bin_data(T_pred1, bin_size)
    T_pred2_binned = bin_data(T_pred2, bin_size)
    T_pred3_binned = bin_data(T_pred3, bin_size)
    T_pred4_binned = bin_data(T_pred4, bin_size)

    S_pred1_binned = bin_data(S_pred1, bin_size)
    S_pred2_binned = bin_data(S_pred2, bin_size)
    S_pred3_binned = bin_data(S_pred3, bin_size)
    S_pred4_binned = bin_data(S_pred4, bin_size)

    # Calculate the differences
    T_diff1 = T_pred1_binned - T1
    T_diff2 = T_pred2_binned - T2
    T_diff3 = T_pred3_binned - T3
    T_diff4 = T_pred4_binned - T4
    S_diff1 = S_pred1_binned - S1
    S_diff2 = S_pred2_binned - S2
    S_diff3 = S_pred3_binned - S3
    S_diff4 = S_pred4_binned - S4


    # Create a figure for the combined plots
    fig = plt.figure(figsize=(18, 18))  # Adjust size as needed
    plot_field_subplot(T1, d1, gld_depths, "Temperature", "Glider T", 321, fig)
    plot_field_subplot(T_pred1, d1, pred_depths, "Temperature", "Synthetic T", 323, fig)
    plot_field_subplot(T_diff1, d1, gld_depths, "T Difference", "T Difference", 325, fig)
    plot_field_subplot(S1, d1, gld_depths, "Salinity", "Glider S", 322, fig)
    plot_field_subplot(S_pred1, d1, pred_depths, "Salinity", "Synthetic S", 324, fig)
    plot_field_subplot(S_diff1, d1, gld_depths, "S Difference", "S Difference", 326, fig)
    plt.suptitle(
        f"Poseidon Crossing #1 \n{t1[0].strftime('%Y-%m-%d')} to {t1[-1].strftime('%Y-%m-%d')}",
        fontsize=18,
        fontweight="bold",
    )
    plt.tight_layout()  # Adjusts subplot params so that subplots fit into the figure area
    plt.show()

    # 2
    fig = plt.figure(figsize=(18, 18))  # Adjust size as needed
    plot_field_subplot(T2, d2, gld_depths, "Temperature", "Glider T", 321, fig)
    plot_field_subplot(T_pred2, d2, pred_depths, "Temperature", "Synthetic T", 323, fig)
    plot_field_subplot(T_diff2, d2, gld_depths, "T Difference", "T Difference", 325, fig)
    plot_field_subplot(S2, d2, gld_depths, "Salinity", "Glider S", 322, fig)
    plot_field_subplot(S_pred2, d2, pred_depths, "Salinity", "Synthetic S", 324, fig)
    plot_field_subplot(S_diff2, d2, gld_depths, "S Difference", "S Difference", 326, fig)
    plt.suptitle(
        f"Poseidon Crossing #2 \n{t2[0].strftime('%Y-%m-%d')} to {t2[-1].strftime('%Y-%m-%d')}",
        fontsize=18,
        fontweight="bold",
    )
    plt.tight_layout()  # Adjusts subplot params so that subplots fit into the figure area
    plt.show()

    # 3
    fig = plt.figure(figsize=(18, 18))  # Adjust size as needed
    plot_field_subplot(T3, d3, gld_depths, "Temperature", "Glider T", 321, fig)
    plot_field_subplot(T_pred3, d3, pred_depths, "Temperature", "Synthetic T", 323, fig)
    plot_field_subplot(T_diff3, d3, gld_depths, "T Difference", "T Difference", 325, fig)
    plot_field_subplot(S3, d3, gld_depths, "Salinity", "Glider S", 322, fig)
    plot_field_subplot(S_pred3, d3, pred_depths, "Salinity", "Synthetic S", 324, fig)
    plot_field_subplot(S_diff3, d3, gld_depths, "S Difference", "S Difference", 326, fig)
    plt.suptitle(
        f"Campeche Crossing #1 and #2 \n{t3[0].strftime('%Y-%m-%d')} to {t3[-1].strftime('%Y-%m-%d')}",
        fontsize=18,
        fontweight="bold",
    )
    plt.tight_layout()  # Adjusts subplot params so that subplots fit into the figure area
    plt.show()

    # 4
    fig = plt.figure(figsize=(18, 18))  # Adjust size as needed
    plot_field_subplot(T4, d4, gld_depths, "Temperature", "Glider T", 321, fig)
    plot_field_subplot(T_pred4, d4, pred_depths, "Temperature", "Synthetic T", 323, fig)
    plot_field_subplot(T_diff4, d4, gld_depths, "T Difference", "T Difference", 325, fig)
    plot_field_subplot(S4, d4, gld_depths, "Salinity", "Glider S", 322, fig)
    plot_field_subplot(S_pred4, d4, pred_depths, "Salinity", "Synthetic S", 324, fig)
    plot_field_subplot(S_diff4, d4, gld_depths, "S Difference", "S Difference", 326, fig)
    plt.suptitle(
        f"Intense LCE \n{t4[0].strftime('%Y-%m-%d')} to {t4[-1].strftime('%Y-%m-%d')}", fontsize=18, fontweight="bold"
    )
    plt.tight_layout()  # Adjusts subplot params so that subplots fit into the figure area
    plt.show()

    avg_d1 = int(np.round(average_depth(T1, gld_depths))) + 1
    avg_d2 = int(np.round(average_depth(T2, gld_depths))) + 1
    avg_d3 = int(np.round(average_depth(T3, gld_depths))) + 1
    avg_d4 = int(np.round(average_depth(T4, gld_depths))) + 1

    h1 = histogram_available_depths(T1)
    h2 = histogram_available_depths(T2)
    h3 = histogram_available_depths(T3)
    h4 = histogram_available_depths(T4)

    eq_rmse_T1, eq_corr_T1 = equivalent_average_statistic(T_pred1, T1, h1, gld_depths, rmse)
    eq_bias_T1, eq_corr_T1 = equivalent_average_statistic(T_pred1, T1, h1, gld_depths, bias)
    eq_rmse_S1, eq_corr_S1 = equivalent_average_statistic(S_pred1, S1, h1, gld_depths, rmse)
    eq_bias_S1, eq_corr_S1 = equivalent_average_statistic(S_pred1, S1, h1, gld_depths, bias)
    eq_rmse_T2, eq_corr_T2 = equivalent_average_statistic(T_pred2, T2, h2, gld_depths, rmse)
    eq_bias_T2, eq_corr_T2 = equivalent_average_statistic(T_pred2, T2, h2, gld_depths, bias)
    eq_rmse_S2, eq_corr_S2 = equivalent_average_statistic(S_pred2, S2, h2, gld_depths, rmse)
    eq_bias_S2, eq_corr_S2 = equivalent_average_statistic(S_pred2, S2, h2, gld_depths, bias)
    eq_rmse_T3, eq_corr_T3 = equivalent_average_statistic(T_pred3, T3, h3, gld_depths, rmse)
    eq_bias_T3, eq_corr_T3 = equivalent_average_statistic(T_pred3, T3, h3, gld_depths, bias)
    eq_rmse_S3, eq_corr_S3 = equivalent_average_statistic(S_pred3, S3, h3, gld_depths, rmse)
    eq_bias_S3, eq_corr_S3 = equivalent_average_statistic(S_pred3, S3, h3, gld_depths, bias)
    eq_rmse_T4, eq_corr_T4 = equivalent_average_statistic(T_pred4, T4, h4, gld_depths, rmse)
    eq_bias_T4, eq_corr_T4 = equivalent_average_statistic(T_pred4, T4, h4, gld_depths, bias)
    eq_rmse_S4, eq_corr_S4 = equivalent_average_statistic(S_pred4, S4, h4, gld_depths, rmse)
    eq_bias_S4, eq_corr_S4 = equivalent_average_statistic(S_pred4, S4, h4, gld_depths, bias)

    correlation_T1 = calculate_correlation(T1, T_pred1_binned)
    correlation_S1 = calculate_correlation(S1, S_pred1_binned)
    correlation_T2 = calculate_correlation(T2, T_pred2_binned)
    correlation_S2 = calculate_correlation(S2, S_pred2_binned)
    correlation_T3 = calculate_correlation(T3, T_pred3_binned)
    correlation_S3 = calculate_correlation(S3, S_pred3_binned)
    correlation_T4 = calculate_correlation(T4, T_pred4_binned)
    correlation_S4 = calculate_correlation(S4, S_pred4_binned)

    print("Crossing & T RMSE & T Bias & T R^2 & S RMSE & S Bias & S R^2")
    print(
        f"Poseidon #1 & {rmse(T_pred1_binned, T1):.3f} & {bias(T_pred1_binned, T1):.3f} & {correlation_T1:.3f} & {rmse(S_pred1_binned, S1):.3f} & {bias(S_pred1_binned, S1):.3f} & {correlation_S1:.3f}\\\\"
    )
    print(
        f"Poseidon #2 & {rmse(T_pred2_binned, T2):.3f} & {bias(T_pred2_binned, T2):.3f} & {correlation_T2:.3f} & {rmse(S_pred2_binned, S2):.3f} & {bias(S_pred2_binned, S2):.3f} & {correlation_S2:.3f}\\\\"
    )
    print(
        f"Campeche #1  & {rmse(T_pred3_binned, T3):.3f} & {bias(T_pred3_binned, T3):.3f} & {correlation_T3:.3f} & {rmse(S_pred3_binned, S3):.3f} & {bias(S_pred3_binned, S3):.3f} & {correlation_S3:.3f} \\\\"
    )
    print(
        f"Intense LCE  & {rmse(T_pred4_binned, T4):.3f} & {bias(T_pred4_binned, T4):.3f} & {correlation_T4:.3f} & {rmse(S_pred4_binned, S4):.3f} & {bias(S_pred4_binned, S4):.3f} & {correlation_S4:.3f} \\\\"
    )

    lat_all = full_dataset.LAT
    lon_all = full_dataset.LON

    # Initialize lists for latitudes and longitudes of each dataset
    lat_train, lon_train = lat_all[train_indices], lon_all[train_indices]
    lat_val, lon_val = lat_all[val_indices], lon_all[val_indices]
    lat_test, lon_test = lat_all[test_indices], lon_all[test_indices]

    # Create a plot with cartopy
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND.with_scale("50m"), color="black")
    ax.coastlines(resolution="50m")

    # Plot points for each dataset in one go
    ax.scatter(lon_train, lat_train, s=3, color="k", alpha=0.7, label="ARGO - Train set", transform=ccrs.Geodetic())
    ax.scatter(lon_test, lat_test, s=3, color="b", alpha=0.7, label="ARGO - Validation set", transform=ccrs.Geodetic())
    ax.scatter(
        lon_val, lat_val, s=30, color="r", marker="x", alpha=0.5, label="ARGO - Test set", transform=ccrs.Geodetic()
    )
    ax.scatter(lon_train, lat_train, s=3, color="k", alpha=0.04, transform=ccrs.Geodetic())
    ax.scatter(lon_test, lat_test, s=3, color="b", alpha=0.04, transform=ccrs.Geodetic())  # Set x and y ticks,
    ax.plot(longitudes_T1, latitudes_T1, color="c", linewidth=2, transform=ccrs.Geodetic(), label="Glider tracks")
    ax.plot(longitudes_T2, latitudes_T2, color="c", linewidth=2, transform=ccrs.Geodetic())
    ax.plot(longitudes_T3, latitudes_T3, color="c", linewidth=2, transform=ccrs.Geodetic())
    ax.plot(longitudes_T4, latitudes_T4, color="c", linewidth=2, transform=ccrs.Geodetic())
    ax.set_xticks(np.arange(-99, -79, 2))
    ax.set_yticks(np.arange(18, 34, 2))
    # add grid
    ax.grid(color="gray", linestyle="--", linewidth=0.5)
    # Add a legend
    plt.legend(loc="lower right", fontsize=14)

    plt.title("Data availability", fontsize=22, fontweight="bold")
    plt.show()

    # get ssh data
    aviso_folder = "/unity/f1/ozavala/DATA/GOFFISH/AVISO/GoM/"
    bbox = (18, 32, -99, -81)
    # t1_date = datenum_to_datetime(np.median(gl_data['t1'][0]))
    # t2_date = datenum_to_datetime(np.median(gl_data['t2'][0]))
    # t3_date = datenum_to_datetime(np.median(gl_data['t3'][0]))
    # t4_date = datenum_to_datetime(np.median(gl_data['t4'][0]))
    t1_date = datenum_to_datetime(gl_data["t1"][0].mean())
    t2_date = datenum_to_datetime(gl_data["t2"][0].mean())
    t3_date = datenum_to_datetime(gl_data["t3"][0].mean())
    t4_date = datenum_to_datetime(gl_data["t4"][0].mean())
    # t1_date = datetime.combine(t1_date.date(), datetime.min.time())
    # t2_date = datetime.combine(t2_date.date(), datetime.min.time())
    # t3_date = datetime.combine(t3_date.date(), datetime.min.time())
    # t4_date = datetime.combine(t4_date.date(), datetime.min.time())
    aviso1_adt, aviso_lats, aviso_lons = get_aviso_by_date(aviso_folder, t1_date, bbox)
    X, Y = np.meshgrid(aviso_lons.values, aviso_lats.values)
    aviso2_adt, _, _ = get_aviso_by_date(aviso_folder, t2_date, bbox)
    aviso3_adt, _, _ = get_aviso_by_date(aviso_folder, t3_date, bbox)
    aviso4_adt, _, _ = get_aviso_by_date(aviso_folder, t4_date, bbox)

    # Create individual plots for glider tracks with the same colorbar scale
    fig = plt.figure(figsize=(12, 12))

    # Subplot 1
    ax1 = fig.add_subplot(2, 2, 1, projection=ccrs.PlateCarree())
    ax1.add_feature(cfeature.LAND.with_scale("50m"), color="black", zorder=0)
    ax1.coastlines(resolution="50m")
    cf1 = ax1.contourf(X, Y, aviso1_adt.adt.values, cmap="jet", levels=50, extend="both")
    ax1.plot(
        longitudes_T1,
        latitudes_T1,
        color="#FF69B4",
        linewidth=2,
        transform=ccrs.Geodetic(),
        label="Poseidon crossing #1",
    )
    ax1.set_xticks(np.arange(-99, -79, 2))
    ax1.set_yticks(np.arange(18, 34, 2))
    ax1.set_extent([-99, -81, 18, 32])
    ax1.grid(color="gray", linestyle="--", linewidth=0.5)
    ax1.tick_params(axis="both", which="major", labelsize=10)

    # Add colorbar for subplot 1
    cbar1 = plt.colorbar(cf1, ax=ax1, fraction=0.036, pad=0.04)
    # Format the colorbar ticks to two decimal places
    cbar1.ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    # Add colorbar title
    cbar1.set_label("ADT (m)", fontsize=10)
    cbar1.ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    cbar1.set_label("ADT (m)", fontsize=8)
    cbar1.ax.tick_params(labelsize=8)  # Make the colorbar ticks smaller
    ax1.title.set_text(t1_date.strftime("%Y-%m-%d"))

    # Subplot 2
    ax2 = fig.add_subplot(2, 2, 2, projection=ccrs.PlateCarree())
    ax2.add_feature(cfeature.LAND.with_scale("50m"), color="black", zorder=0)
    ax2.coastlines(resolution="50m")
    cf2 = ax2.contourf(X, Y, aviso2_adt.adt.values, cmap="jet", levels=50, extend="both")
    ax2.plot(
        longitudes_T2,
        latitudes_T2,
        color="#FF69B4",
        linewidth=2,
        transform=ccrs.Geodetic(),
        label="Poseidon crossing #2",
    )
    ax2.set_xticks(np.arange(-99, -79, 2))
    ax2.set_yticks(np.arange(18, 34, 2))
    ax2.set_extent([-99, -81, 18, 32])
    ax2.grid(color="gray", linestyle="--", linewidth=0.5)
    ax2.tick_params(axis="both", which="major", labelsize=10)

    # Add colorbar for subplot 2
    cbar2 = plt.colorbar(cf2, ax=ax2, fraction=0.036, pad=0.04)
    cbar2.ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    cbar2.set_label("ADT (m)", fontsize=8)
    cbar2.ax.tick_params(labelsize=8)
    ax2.title.set_text(t2_date.strftime("%Y-%m-%d"))

    # Subplot 3
    ax3 = fig.add_subplot(2, 2, 3, projection=ccrs.PlateCarree())
    ax3.add_feature(cfeature.LAND.with_scale("50m"), color="black", zorder=0)
    ax3.coastlines(resolution="50m")
    cf3 = ax3.contourf(X, Y, aviso3_adt.adt.values, cmap="jet", levels=50, extend="both")
    ax3.plot(longitudes_T3, latitudes_T3, color="#FF69B4", linewidth=2, transform=ccrs.Geodetic(), label="Campeche")
    ax3.set_xticks(np.arange(-99, -79, 2))
    ax3.set_yticks(np.arange(18, 34, 2))
    ax3.set_extent([-99, -81, 18, 32])
    ax3.grid(color="gray", linestyle="--", linewidth=0.5)
    ax3.tick_params(axis="both", which="major", labelsize=10)

    # Add colorbar for subplot 3
    cbar3 = plt.colorbar(cf3, ax=ax3, fraction=0.036, pad=0.04)
    cbar3.ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    cbar3.set_label("ADT (m)", fontsize=8)
    cbar3.ax.tick_params(labelsize=8)
    ax3.title.set_text(t3_date.strftime("%Y-%m-%d"))

    # Subplot 4
    ax4 = fig.add_subplot(2, 2, 4, projection=ccrs.PlateCarree())
    ax4.add_feature(cfeature.LAND.with_scale("50m"), color="black", zorder=0)
    ax4.coastlines(resolution="50m")
    cf4 = ax4.contourf(X, Y, aviso4_adt.adt.values, cmap="jet", levels=50, extend="both")
    ax4.plot(longitudes_T4, latitudes_T4, color="#FF69B4", linewidth=2, transform=ccrs.Geodetic(), label="Intense LCE")
    ax4.set_xticks(np.arange(-99, -79, 2))
    ax4.set_yticks(np.arange(18, 34, 2))
    ax4.set_extent([-99, -81, 18, 32])
    ax4.grid(color="gray", linestyle="--", linewidth=0.5)
    ax4.tick_params(axis="both", which="major", labelsize=10)

    # Add colorbar for subplot 4
    cbar4 = plt.colorbar(cf4, ax=ax4, fraction=0.036, pad=0.04)
    cbar4.ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    cbar4.set_label("ADT (m)", fontsize=8)
    cbar4.ax.tick_params(labelsize=8)
    ax4.title.set_text(t4_date.strftime("%Y-%m-%d"))

    # Add a title for the entire figure
    plt.suptitle("Gliders", fontsize=22, fontweight="bold")

    # Display the plot
    plt.show()

    # # Meunier data plots

    # # Perform linear regression
    # slope, intercept, r_value, p_value, std_err = linregress(full_dataset.SH1950, full_dataset.AVISO_ADT)

    # # Print the results
    # print(f"Slope: {slope}")
    # print(f"Intercept: {intercept}")
    # print(f"Coefficient of determination (R²): {r_value**2}")
    # print(f"P-value: {p_value}")
    # print(f"Standard error of the regression estimate: {std_err}")

    # # Normalize ADT values for comparison
    # ADT_normalized = full_dataset.SH1950*slope + intercept

    # # Plot histograms
    # plt.figure(figsize=(6, 6))

    # # Histogram of ADT
    # plt.hist(ADT_normalized, bins=100, alpha=0.5, density=True, label='SH1950', edgecolor='k')

    # # Generate a KDE plot for SSH
    # sns.kdeplot(full_dataset.AVISO_ADT, color="r", label='ADT')

    # plt.xlabel('SSH [m]')
    # plt.ylabel('Frequency')
    # plt.title('SSH Distribution')
    # plt.legend(loc='upper right')

    # plt.show()

    # # create a plot of the location and SH of the profiles:
    # fig = plt.figure(figsize=(10, 8))
    # ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    # ax.add_feature(cfeature.LAND.with_scale('50m'), color='black')
    # # ax.add_feature(cfeature.OCEAN.with_scale('110m'))  # Adds ocean feature, might include basic bathymetry
    # ax.coastlines(resolution='50m')

    # # Plot points for each dataset in one go
    # scatter = ax.scatter(full_dataset.LON, full_dataset.LAT, c = ADT_normalized, transform=ccrs.Geodetic(), s=2, cmap='jet')
    # ax.set_xticks(np.arange(-99, -79, 2))
    # ax.set_yticks(np.arange(18, 34, 2))
    # plt.xticks(fontsize=11)  # Adjust the font size as needed
    # plt.yticks(fontsize=11)  # Adjust the font size as needed
    # ax.grid(color='gray', linestyle='--', linewidth=0.51)
    # cbar = plt.colorbar(scatter, label='SH1950', shrink=0.78)
    # cbar.set_label('SH1950 [m]', fontsize=16)
    # cbar.ax.tick_params(labelsize=16)
    # plt.xlabel('Longitude', fontsize=16)
    # plt.ylabel('Latitude', fontsize=16)
    # plt.title('ARGO locations')
    # plt.show()

    # # T-S diagram

    # tempL=np.linspace(np.min(full_dataset.TEMP)-1,np.max(full_dataset.TEMP)+1,156)

    # salL=np.linspace(np.min(full_dataset.SAL)-1,np.max(full_dataset.SAL)+1,156)

    # Tg, Sg = np.meshgrid(tempL,salL)
    # sigma_theta = gsw.sigma0(Sg, Tg)
    # cnt = np.linspace(sigma_theta.min(), sigma_theta.max(),156)

    # # Normalize the ADT values for color mapping
    # norm = mcolors.Normalize(vmin=ADT_normalized.min(), vmax=ADT_normalized.max())
    # cmap = plt.cm.jet  # Choose a colormap

    # # Create the T-S plot
    # fig, ax = plt.subplots(figsize=(10, 8))
    # cs = ax.contour(Sg, Tg, sigma_theta, colors='grey', zorder=1)

    # # Plot each line
    # for i in range(full_dataset.TEMP.shape[1]):  # Assuming TEMP and SAL have the same second dimension
    #     # TEMP[:, i] and SAL[:, i] form the x and y coordinates of the ith line
    #     color = cmap(norm(ADT_normalized[i]))  # Map the ADT value to a color
    #     ax.plot(full_dataset.SAL[:, i], full_dataset.TEMP[:, i], color=color, linewidth=0.5)

    # for i in range(pred_T.shape[1]):  # Assuming TEMP and SAL have the same second dimension
    #     # TEMP[:, i] and SAL[:, i] form the x and y coordinates of the ith line
    #     # color = cmap(norm(ADT_normalized[i]))  # Map the ADT value to a color
    #     ax.plot(pred_S[:, i], pred_T[:, i], color='pink', linewidth=0.2)

    # sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    # sm.set_array([])  # This line is necessary for ScalarMappable to work with colorbar

    # # Mark the cores of the water masses with circles and labels
    # cores = {
    #     "SAAIW ": (34.9, 6.5),
    #     "GCW ": (36.4, 22.3),
    #     "NASUW ": (36.8, 22)
    # }

    # for label, (salinity, temperature) in cores.items():
    #     ax.plot(salinity, temperature, 'o', markersize=7, color='black')
    #     ax.text(salinity, temperature, label, fontsize=11, verticalalignment='bottom', horizontalalignment='right', fontweight='bold')

    # # cbar = fig.colorbar(sm, ax=ax, orientation='vertical', fraction=0.036, pad=0.04)
    # # cbar.set_label('SH1950', fontsize=12)
    # # cbar.ax.tick_params(labelsize=11)
    # # set x lims to 34.5 to 37.5
    # ax.set_xlim(34.5, 37.5)
    # cl=plt.clabel(cs,fontsize=10,inline=False,fmt='%.1f',colors='k')
    # plt.xlabel('Salinity [PSU]')
    # plt.ylabel('Temperature [°C]')
    # plt.title('T-S Diagram')

    # # ax.set_xticks(np.arange(34.5, 37.5, 0.5))

    # plt.show()

    # fig = plt.figure(figsize=(6, 6))
    # plt.plot(ADT_normalized, full_dataset.AVISO_ADT, '.', markersize=0.6)
    # # add a trend line
    # plt.plot([ADT_normalized.min(), ADT_normalized.max()], [ADT_normalized.min(), ADT_normalized.max()], 'k')
    # plt.xlabel('SH1950 [m]')
    # plt.ylabel('SSH [m]')
    # plt.title('SH1950 vs SSH')
    # plt.show()

    # # compare T and S profile against PCA reconstruction
    # prof_number = 300
    # prof_number = np.atleast_1d(prof_number)
    # depths = np.arange(0, 501, 1)
    # pca_prof = full_dataset.get_profiles(prof_number,True)
    # pca_T = pca_prof[depths,0,:]
    # pca_S = pca_prof[depths,1,:]
    # ori_prof = full_dataset.get_profiles(prof_number,False)
    # ori_T = ori_prof[depths,0,:]
    # ori_S = ori_prof[depths,1,:]

    # fig = plt.figure(figsize=(10, 8))
    # ax1 = fig.add_subplot(1, 2, 1)
    # ax1.plot(pca_T, depths, label='15 PCS recon', color='r', linewidth=2)
    # ax1.plot(ori_T, depths, label='Argo', color='k', linewidth=1)
    # ax1.set_xlabel('Temperature [°C]')
    # ax1.set_ylabel('Depth [m]')
    # ax1.set_title('Temperature')
    # ax1.grid()
    # #invert y axis
    # ax1.invert_yaxis()
    # ax1.legend(fontsize=12)

    # ax2 = fig.add_subplot(1, 2, 2)
    # ax2.plot(pca_S, depths, label='15 PCS recon', color='c', linewidth=2)
    # ax2.plot(ori_S, depths, label='Argo', color='k', linewidth=1)
    # ax2.set_xlabel('Salinity [PSU])')
    # # ax2.set_ylabel('Depth [m]')
    # ax2.set_title('Salinity')
    # ax2.grid()
    # #invert y axis
    # ax2.invert_yaxis()
    # ax2.legend(fontsize=12)

    # calculate average bias and rmse for depth ranges
    print(
        "Depth range \t NeSPReSO 1.1 T RMSE \t GEM T RMSE \t NeSPReSO 1.0 T RMSE \t ISOP T RMSE \t NeSPReSO 1.1 T Bias \t GEM T Bias \t NeSPReSO 1.0 T Bias \t ISOP T Bias \t NeSPReSO 1.1 S RMSE \t GEM S RMSE \t NeSPReSO 1.0 S RMSE \t ISOP S RMSE \t NeSPReSO 1.1 S Bias \t GEM S Bias \t NeSPReSO 1.0 S Bias \t ISOP S Bias \t NeSPReSO 1.1 T R^2 \t GEM T R^2 \t NeSPReSO 1.0 T R^2 \t NeSPReSO 1.1 S R^2 \t GEM S R^2 \t NeSPReSO 1.0 S R^2"
    )
    intervals = default_depth_intervals(min_depth, max_depth)

    for min_d, max_d in intervals:
        metrics = compute_depth_interval_metrics(
            min_d,
            max_d,
            isop_depths,
            ist.rmse.values,
            ist.bias.values,
            iss.rmse.values,
            iss.bias.values,
            original_profiles,
            pred_T,
            pred_S,
            gem_temp,
            gem_sal,
            old_pred_T,
            old_pred_S,
        )

        print(
            f"[{metrics['min_d']}-{metrics['max_d']}] \t {metrics['nn_t_rmse']:.3f} \t {metrics['gem_t_rmse']:.3f} \t {metrics['mlr_t_rmse']:.3f} \t {metrics['isop_avg_t_rmse']:.3f} \t {metrics['nn_t_bias']:.3f} \t {metrics['gem_t_bias']:.3f} \t {metrics['mlr_t_bias']:.3f} \t {metrics['isop_avg_t_bias']:.3f} \t {metrics['nn_s_rmse']:.3f} \t {metrics['gem_s_rmse']:.3f} \t {metrics['mlr_s_rmse']:.3f} \t {metrics['isop_avg_s_rmse']:.3f} \t {metrics['nn_s_bias']:.3f} \t {metrics['gem_s_bias']:.3f} \t {metrics['mlr_s_bias']:.3f} \t {metrics['isop_avg_s_bias']:.3f} \t {metrics['nn_T_corr']:.3f} \t {metrics['gem_T_corr']:.3f} \t {metrics['mlr_T_corr']:.3f} \t {metrics['nn_S_corr']:.3f} \t {metrics['gem_S_corr']:.3f} \t {metrics['mlr_S_corr']:.3f}"
        )
        # print(f"[{min_d}-{max_d}] \t {nn_t_rmse:.3f} \t {gem_t_rmse:.3f} \t {isop_avg_t_rmse:.3f} \t {nn_t_bias:.3f} \t {gem_t_bias:.3f} \t {isop_avg_t_bias:.3f} \t {nn_s_rmse:.3f} \t {gem_s_rmse:.3f} \t {isop_avg_s_rmse:.3f} \t {nn_s_bias:.3f} \t {gem_s_bias:.3f} \t {isop_avg_s_bias:.3f} \t {nn_T_corr:.3f} \t {gem_T_corr:.3f} \t {nn_S_corr:.3f} \t {gem_S_corr:.3f}")
        # print("\hline")

    # ========================================================================
    # Vertical Metrics Analysis: Compare NeSPReSO 1.0 vs 1.1
    # ========================================================================
    print("\n" + "=" * 80)
    print("VERTICAL METRICS ANALYSIS: NeSPReSO 1.0 vs 1.1")
    print("=" * 80)

    # Prepare data: predictions are (depth, n_profiles)
    # pred_T and pred_S are (depth, n_profiles) where depth is 0-1800
    # old_pred_T and old_pred_S are (depth, n_profiles)
    # original_profiles is (depth, 2, n_profiles) where 2 is [T, S]

    # Get depth array - ensure it matches the actual data dimensions
    n_depth_data = pred_T.shape[0]
    depth_array = np.arange(min_depth, min_depth + n_depth_data)  # Shape: (n_depth,)
    n_profiles = pred_T.shape[1]

    # Try to get PRES data if available, otherwise use depth as pressure
    try:
        PRES_data = full_dataset.PRES[:, subset_indices]  # (depth, n_profiles)
        use_pres = True
        print("Using PRES data for EOS computation")
    except:
        use_pres = False
        print("Using depth as pressure approximation (1 dbar ≈ 1 m)")

    # Prepare data for metrics computation
    # Need to transpose to (n_profiles, depth) for metrics functions
    T_11 = pred_T.T  # (n_profiles, depth)
    S_11 = pred_S.T  # (n_profiles, depth)
    T_10 = old_pred_T.T  # (n_profiles, depth)
    S_10 = old_pred_S.T  # (n_profiles, depth)
    T_orig = original_profiles[:, 0, :].T  # (n_profiles, depth)
    S_orig = original_profiles[:, 1, :].T  # (n_profiles, depth)

    # Get lat/lon for EOS computation
    lat_profiles = lat_val  # (n_profiles,)
    lon_profiles = lon_val  # (n_profiles,)

    # Compute density for each model
    print("\nComputing density profiles for vertical metrics...")

    pres_for_eos = PRES_data if use_pres else None
    rho_11 = compute_density_profiles(T_11, S_11, lat_profiles, lon_profiles, depth_array, "NeSPReSO 1.1", pres_for_eos)
    rho_10 = compute_density_profiles(T_10, S_10, lat_profiles, lon_profiles, depth_array, "NeSPReSO 1.0", pres_for_eos)
    rho_orig = compute_density_profiles(T_orig, S_orig, lat_profiles, lon_profiles, depth_array, "Argo", pres_for_eos)

    print("Density computation completed.")

    # Compute static stability metrics
    print("\nComputing static stability metrics...")

    stability_11 = compute_stability_metrics(rho_11, depth_array, "NeSPReSO 1.1")
    stability_10 = compute_stability_metrics(rho_10, depth_array, "NeSPReSO 1.0")
    stability_orig = compute_stability_metrics(rho_orig, depth_array, "Argo")

    # Compute density smoothness metrics
    print("Computing density smoothness metrics...")

    smoothness_11 = compute_smoothness_metrics(rho_11, depth_array, "NeSPReSO 1.1")
    smoothness_10 = compute_smoothness_metrics(rho_10, depth_array, "NeSPReSO 1.0")
    smoothness_orig = compute_smoothness_metrics(rho_orig, depth_array, "Argo")

    # Print comparison statistics
    print("\n" + "-" * 80)
    print("VERTICAL METRICS COMPARISON")
    print("-" * 80)
    print(f"\nStatic Stability Metrics:")
    print(f"{'Metric':<30} {'Argo':<15} {'NeSPReSO 1.0':<15} {'NeSPReSO 1.1':<15}")
    print("-" * 75)
    print(
        f"{'Fraction Unstable':<30} {stability_orig['frac_unstable']:<15.6f} {stability_10['frac_unstable']:<15.6f} {stability_11['frac_unstable']:<15.6f}"
    )
    print(
        f"{'Min N² [s⁻²]':<30} {stability_orig['min_N2']:<15.6e} {stability_10['min_N2']:<15.6e} {stability_11['min_N2']:<15.6e}"
    )

    # Compute mean integrated negative N² (spatial mean)
    int_neg_orig = stability_orig["int_neg_N2"]
    int_neg_10 = stability_10["int_neg_N2"]
    int_neg_11 = stability_11["int_neg_N2"]

    mean_int_neg_orig = np.nanmean(int_neg_orig) if np.any(np.isfinite(int_neg_orig)) else np.nan
    mean_int_neg_10 = np.nanmean(int_neg_10) if np.any(np.isfinite(int_neg_10)) else np.nan
    mean_int_neg_11 = np.nanmean(int_neg_11) if np.any(np.isfinite(int_neg_11)) else np.nan

    print(
        f"{'Mean Int. Neg. N² [m/s²]':<30} {mean_int_neg_orig:<15.6e} {mean_int_neg_10:<15.6e} {mean_int_neg_11:<15.6e}"
    )

    print(f"\nDensity Smoothness Metrics (50-300 m):")
    print(f"{'Metric':<30} {'Argo':<15} {'NeSPReSO 1.0':<15} {'NeSPReSO 1.1':<15}")
    print("-" * 75)
    print(
        f"{'Var(d²ρ/dz²)':<30} {smoothness_orig['var_d2rho_dz2']:<15.6e} {smoothness_10['var_d2rho_dz2']:<15.6e} {smoothness_11['var_d2rho_dz2']:<15.6e}"
    )
    print(
        f"{'Mean Inflection Points':<30} {smoothness_orig['mean_inflections']:<15.2f} {smoothness_10['mean_inflections']:<15.2f} {smoothness_11['mean_inflections']:<15.2f}"
    )

    # Create comparison plots
    print("\nGenerating vertical metrics comparison plots...")

    # Plot 1: Fraction Unstable Profiles
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle("Vertical Metrics Comparison: NeSPReSO 1.0 vs 1.1", fontsize=16, fontweight="bold")

    # 1. Fraction Unstable
    ax = axes[0, 0]
    models = ["Argo", "NeSPReSO 1.0", "NeSPReSO 1.1"]
    frac_vals = [stability_orig["frac_unstable"], stability_10["frac_unstable"], stability_11["frac_unstable"]]
    colors = ["blue", "green", "red"]
    bars = ax.bar(models, frac_vals, color=colors, alpha=0.7, edgecolor="black")
    ax.set_ylabel("Fraction Unstable Profiles")
    ax.set_title("Fraction of Profiles with N² < 0")
    ax.grid(True, alpha=0.3, axis="y")
    ax.tick_params(axis="x", labelsize=12)  # Make xtick labels smaller
    # Add value labels on bars
    for bar, val in zip(bars, frac_vals):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, height, f"{val:.3f}", ha="center", va="bottom", fontsize=11)

    # 2. Variance of Second Derivative
    ax = axes[0, 1]
    var_d2_vals = [smoothness_orig["var_d2rho_dz2"], smoothness_10["var_d2rho_dz2"], smoothness_11["var_d2rho_dz2"]]
    bars = ax.bar(models, var_d2_vals, color=colors, alpha=0.7, edgecolor="black")
    ax.set_ylabel("Var(d²ρ/dz²)")
    ax.set_title("Density Curvature Variance (50-300 m)")
    ax.set_yscale("log")  # Set y-axis to log scale
    ax.grid(True, alpha=0.3, axis="y", which="both")
    ax.tick_params(axis="x", labelsize=12)  # Make xtick labels smaller
    for bar, val in zip(bars, var_d2_vals):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, height, f"{val:.2e}", ha="center", va="bottom", fontsize=9)

    # 3. Mean Integrated Negative N²
    ax = axes[1, 0]
    int_neg_vals = [mean_int_neg_orig, mean_int_neg_10, mean_int_neg_11]
    bars = ax.bar(models, int_neg_vals, color=colors, alpha=0.7, edgecolor="black")
    ax.set_ylabel("Mean Integrated |N²<0| [m/s²]")
    ax.set_title("Spatial Mean of Integrated Negative N²")
    ax.grid(True, alpha=0.3, axis="y")
    ax.tick_params(axis="x", labelsize=12)  # Make xtick labels smaller
    for bar, val in zip(bars, int_neg_vals):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{val:.2e}",
            ha="center",
            va="bottom" if height < 0 else "top",
            fontsize=9,
        )

    # 4. Minimum N²
    ax = axes[1, 1]
    min_n2_vals = [stability_orig["min_N2"], stability_10["min_N2"], stability_11["min_N2"]]
    bars = ax.bar(models, min_n2_vals, color=colors, alpha=0.7, edgecolor="black")
    ax.set_ylabel("Minimum N² [s⁻²]")
    ax.set_title("Global Minimum N²")
    # ax.axhline(0, color='r', linestyle='--', alpha=0.5, label='Unstable threshold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.tick_params(axis="x", labelsize=12)  # Make xtick labels smaller
    for bar, val in zip(bars, min_n2_vals):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{val:.2e}",
            ha="center",
            va="bottom" if height < 0 else "top",
            fontsize=9,
        )

    plt.tight_layout()
    plt.show()

    # Plot 2: Mean Inflection Points
    fig2, ax2 = plt.subplots(1, 1, figsize=(8, 6))
    infl_vals = [
        smoothness_orig["mean_inflections"],
        smoothness_10["mean_inflections"],
        smoothness_11["mean_inflections"],
    ]
    bars = ax2.bar(models, infl_vals, color=colors, alpha=0.7, edgecolor="black")
    ax2.set_ylabel("Mean Inflection Points")
    ax2.set_title("Mean Number of Inflection Points (50-300 m)")
    ax2.grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars, infl_vals):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2.0, height, f"{val:.2f}", ha="center", va="bottom", fontsize=11)
    plt.tight_layout()
    plt.show()

    # Plot 3: Distribution of Integrated Negative N²
    fig3, ax3 = plt.subplots(1, 1, figsize=(12, 6))

    # Flatten spatial arrays and filter finite values
    int_neg_orig_flat = int_neg_orig.flatten()
    int_neg_10_flat = int_neg_10.flatten()
    int_neg_11_flat = int_neg_11.flatten()

    valid_orig = int_neg_orig_flat[np.isfinite(int_neg_orig_flat)]
    valid_10 = int_neg_10_flat[np.isfinite(int_neg_10_flat)]
    valid_11 = int_neg_11_flat[np.isfinite(int_neg_11_flat)]

    # Create bins for histogram
    all_vals = np.concatenate([valid_orig, valid_10, valid_11])
    if len(all_vals) > 0:
        # Use symmetric log scale for negative values
        abs_vals = np.abs(all_vals)
        log_min = np.log10(np.max([abs_vals.min(), 1e-6]))
        log_max = np.log10(abs_vals.max())
        n_bins = 50
        log_bins = np.logspace(log_min, log_max, n_bins)
        bins = -log_bins[::-1]  # Negative bins

        ax3.hist(valid_orig, bins=bins, alpha=0.5, label="Argo", color="blue", edgecolor="black")
        ax3.hist(valid_10, bins=bins, alpha=0.5, label="NeSPReSO 1.0", color="green", edgecolor="black")
        ax3.hist(valid_11, bins=bins, alpha=0.5, label="NeSPReSO 1.1", color="red", edgecolor="black")
        ax3.set_xlabel("Integrated |N²<0| [m/s²]")
        ax3.set_ylabel("Frequency")
        ax3.set_title("Distribution of Integrated Negative N²")
        ax3.set_xscale("symlog", linthresh=1e-5)
        ax3.legend(loc="upper left", fontsize=11)
        ax3.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.show()

    print("\nVertical metrics analysis completed!")
    print("=" * 80 + "\n")

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

    # Make a bar plot showing how many profiles are in the training, validation and test datasets per month

    def count_profiles_per_month(dataset, indices):
        dates = [datetime.fromordinal(int(d)) for d in dataset.TIME[indices]]
        df = pd.DataFrame({"date": dates})
        return df.groupby(df["date"].dt.month).size().reindex(range(1, 13), fill_value=0)

    train_counts = count_profiles_per_month(train_dataset.dataset, train_indices)
    val_counts = count_profiles_per_month(val_dataset.dataset, val_indices)
    test_counts = count_profiles_per_month(test_dataset.dataset, test_indices)

    # Combine all dates and get unique months
    all_months = sorted(set(train_counts.index) | set(val_counts.index) | set(test_counts.index))

    # Combine all counts into a single DataFrame
    df = pd.DataFrame({"Train": train_counts, "Validation": val_counts, "Test": test_counts})

    # Calculate the total number of profiles for each month
    df_total = df.sum(axis=1)
    # Calculate the percentage for each dataset
    df_percentage = df.div(df_total, axis=0) * 100

    # Update the index to display month abbreviations
    df_percentage.index = [calendar.month_abbr[i] for i in df_percentage.index]

    # Plot
    ax = df_percentage.plot(kind="bar", stacked=True, figsize=(15, 6), width=0.8)
    plt.title("Profiles per Month")
    plt.xlabel("Month")
    plt.ylabel("%")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), fancybox=True, shadow=True, ncols=3)

    # Rotate x-axis labels
    plt.xticks(rotation=45, ha="right")

    # Add total number labels on top of each bar
    for i, total in enumerate(df_total):
        ax.text(
            i,
            0,
            f"Total:\n{total:,.0f}",
            ha="center",
            va="bottom",
        )

    # Add percentage labels on each bar segment
    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f%%", label_type="center")

    # Set y-axis to show percentages from 0 to 100
    plt.ylim(0, 100)  # Increase to 105 to accommodate total labels?

    plt.tight_layout()
    plt.show()
