"""Steric-height / ISOP depth-bin statistics and plots (hoisted from monolith __main__)."""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import gsw
import numpy as np
from torch.utils.data import DataLoader, Subset

from nespreso.experiments.validation_context import ValidationContext
from nespreso.inference import get_predictions
from nespreso.utils.time import get_month

coolwhitewarm = mcolors.LinearSegmentedColormap.from_list(
    name="red_white_blue", colors=[(0, 0, 1), (1, 1.0, 1), (1, 0, 0)]
)


def run_steric_depth_stats(ctx: ValidationContext) -> None:
    """Steric height (900 dbar) binning and T/S statistics on ISOP depth bins."""
    full_dataset = ctx.full_dataset
    trained_model = ctx.trained_model
    device = ctx.device
    batch_size = ctx.batch_size
    ist = ctx.ist

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

