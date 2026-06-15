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
from nespreso.analysis.monthly import count_profiles_per_month
from nespreso.reporting import print_training_params
from nespreso.experiments.validation_context import build_validation_context
from nespreso.experiments.pca_regression import run_pca_regression_baseline
from nespreso.experiments.glider_mission import run_glider_mission
from nespreso.experiments.density_stability import run_density_stability
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

    _save_model_path, artifacts = run_training(cfg, return_artifacts=True)
    ctx = build_validation_context(cfg, artifacts, bin_size=bin_size)
    full_dataset = ctx.full_dataset
    train_dataset = ctx.train_dataset
    val_dataset = ctx.val_dataset
    test_dataset = ctx.test_dataset
    train_loader = ctx.train_loader
    val_loader = ctx.val_loader
    test_loader = ctx.test_loader
    trained_model = ctx.trained_model
    device = ctx.device
    input_dim = ctx.input_dim
    pred_T = ctx.pred_T
    pred_S = ctx.pred_S
    old_pred_T = ctx.old_pred_T
    old_pred_S = ctx.old_pred_S
    original_profiles = ctx.original_profiles
    pca_approx_profiles = ctx.pca_approx_profiles
    orig_T = ctx.orig_T
    orig_S = ctx.orig_S
    pred_T_resid = ctx.pred_T_resid
    pred_S_resid = ctx.pred_S_resid
    gems_T = ctx.gems_T
    gems_S = ctx.gems_S
    gems_T_resid = ctx.gems_T_resid
    gems_S_resid = ctx.gems_S_resid
    old_T_resid = ctx.old_T_resid
    old_S_resid = ctx.old_S_resid
    gem_temp = ctx.gem_temp
    gem_sal = ctx.gem_sal
    sst_inputs = ctx.sst_inputs
    ssh_inputs = ctx.ssh_inputs
    lat_val = ctx.lat_val
    lon_val = ctx.lon_val
    dates_val = ctx.dates_val
    subset_indices = ctx.subset_indices
    train_indices = ctx.train_indices
    val_indices = ctx.val_indices
    test_indices = ctx.test_indices
    data_ISOP = ctx.data_ISOP
    lon_bins = ctx.lon_bins
    lat_bins = ctx.lat_bins
    lon_centers = ctx.lon_centers
    lat_centers = ctx.lat_centers
    ist = ctx.ist
    iss = ctx.iss
    isop_depths = ctx.isop_depths
    avg_gem_temp_rmse = ctx.avg_gem_temp_rmse
    avg_gem_temp_bias = ctx.avg_gem_temp_bias
    avg_nn_temp_rmse = ctx.avg_nn_temp_rmse
    avg_nn_temp_bias = ctx.avg_nn_temp_bias
    avg_old_temp_rmse = ctx.avg_old_temp_rmse
    avg_old_temp_bias = ctx.avg_old_temp_bias
    avg_gem_sal_rmse = ctx.avg_gem_sal_rmse
    avg_gem_sal_bias = ctx.avg_gem_sal_bias
    avg_nn_sal_rmse = ctx.avg_nn_sal_rmse
    avg_nn_sal_bias = ctx.avg_nn_sal_bias
    avg_old_sal_rmse = ctx.avg_old_sal_rmse
    avg_old_sal_bias = ctx.avg_old_sal_bias
    val_predictions = ctx.val_predictions

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
    august_mask = np.array([get_month(t) == 8 for t in full_dataset.TIME])
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

    mlr_results = run_pca_regression_baseline(ctx)
    beta_T = mlr_results["beta_T"]
    beta_S = mlr_results["beta_S"]
    mlr_T_resid = mlr_results["mlr_T_resid"]
    mlr_S_resid = mlr_results["mlr_S_resid"]
    avg_mlr_temp_rmse = mlr_results["avg_mlr_temp_rmse"]
    avg_mlr_temp_bias = mlr_results["avg_mlr_temp_bias"]
    avg_mlr_sal_rmse = mlr_results["avg_mlr_sal_rmse"]
    avg_mlr_sal_bias = mlr_results["avg_mlr_sal_bias"]
    X_avgs = mlr_results["X_avgs"]

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

    run_glider_mission(ctx)

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

    # Make a bar plot showing how many profiles are in the training, validation and test datasets per month

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
