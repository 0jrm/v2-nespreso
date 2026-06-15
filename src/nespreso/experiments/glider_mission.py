"""Glider mission validation experiment (hoisted from monolith __main__)."""

from __future__ import annotations

import pickle

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from matplotlib.ticker import FormatStrFormatter

from nespreso.analysis.correlation import calculate_correlation
from nespreso.analysis.depth_stats import (
    average_depth,
    equivalent_average_statistic,
    histogram_available_depths,
)
from nespreso.analysis.glider import bin_data, get_glider_predictions
from nespreso.data.features import prepare_inputs
from nespreso.data.pickle_compat import load_dataset_pickle
from nespreso.experiments.validation_context import ValidationContext
from nespreso.io.satellite import load_satellite_data
from nespreso.io.satellite_readers import get_aviso_by_date
from nespreso.metrics import bias, rmse
from nespreso.utils.geo import calculate_distances
from nespreso.utils.time import datenum_to_datetime, datenums_to_datetimes
from nespreso.viz.fields import plot_field_subplot


def _load_dataset_pickle_dict(ctx: ValidationContext) -> dict:
    return load_dataset_pickle(ctx.cfg.paths.dataset_pickle)


def run_glider_mission(ctx: ValidationContext) -> dict:
    """Glider crossing comparison: synthetic T/S vs in-situ glider sections."""
    nn_temp_residuals = ctx.pred_T - ctx.original_profiles[:, 0, :]
    nn_sal_residuals = ctx.pred_S - ctx.original_profiles[:, 1, :]

    file_path = "/unity/g2/jmiranda/SubsurfaceFields/Data/Glider_binned_data_for_heat_content_IA_mission_lowpass_LCE_Campeche_cyclone.mat"
    gl_data = scipy.io.loadmat(file_path)

    print(gl_data.keys())

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

    tt1 = gl_data["t1"][0]
    tt2 = gl_data["t2"][0]
    tt3 = gl_data["t3"][0]
    tt4 = gl_data["t4"][0]
    t1 = datenums_to_datetimes(tt1)
    t2 = datenums_to_datetimes(tt2)
    t3 = datenums_to_datetimes(tt3)
    t4 = datenums_to_datetimes(tt4)

    T1 = gl_data["T1"]
    S1 = gl_data["S1"]
    T2 = gl_data["T2"]
    S2 = gl_data["S2"]
    T3 = gl_data["T3"]
    S3 = gl_data["S3"]
    T4 = gl_data["T4"]
    S4 = gl_data["S4"]

    dataset_pickle_file = ctx.cfg.paths.dataset_pickle
    model_cfg = ctx.cfg.model
    data = _load_dataset_pickle_dict(ctx)

    if "sss1" in data:
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
                "min_depth": ctx.min_depth,
                "max_depth": ctx.max_depth,
                "epochs": model_cfg.epochs,
                "patience": model_cfg.patience,
                "n_components": ctx.n_components,
                "batch_size": ctx.batch_size,
                "learning_rate": ctx.learning_rate,
                "dropout_prob": ctx.dropout_prob,
                "layers_config": ctx.layers_config,
                "input_params": ctx.input_params,
                "train_size": ctx.train_size,
                "val_size": ctx.val_size,
                "test_size": ctx.test_size,
                "full_dataset": ctx.full_dataset,
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

    input_params = ctx.input_params
    gld_tensor1 = prepare_inputs(tt1, latitudes_T1, longitudes_T1, sss1, sst1, aviso1, input_params)
    gld_tensor2 = prepare_inputs(tt2, latitudes_T2, longitudes_T2, sss2, sst2, aviso2, input_params)
    gld_tensor3 = prepare_inputs(tt3, latitudes_T3, longitudes_T3, sss3, sst3, aviso3, input_params)
    gld_tensor4 = prepare_inputs(tt4, latitudes_T4, longitudes_T4, sss4, sst4, aviso4, input_params)

    gld_depths = np.arange(0, 201 * 5, 5)
    pred_max_depth = 1004

    T_pred1, S_pred1 = get_glider_predictions(
        ctx.trained_model,
        ctx.val_loader,
        gld_tensor1,
        ctx.device,
        ctx.val_dataset.dataset.inverse_transform,
        max_depth=pred_max_depth,
    )
    T_pred2, S_pred2 = get_glider_predictions(
        ctx.trained_model,
        ctx.val_loader,
        gld_tensor2,
        ctx.device,
        ctx.val_dataset.dataset.inverse_transform,
        max_depth=pred_max_depth,
    )
    T_pred3, S_pred3 = get_glider_predictions(
        ctx.trained_model,
        ctx.val_loader,
        gld_tensor3,
        ctx.device,
        ctx.val_dataset.dataset.inverse_transform,
        max_depth=pred_max_depth,
    )
    T_pred4, S_pred4 = get_glider_predictions(
        ctx.trained_model,
        ctx.val_loader,
        gld_tensor4,
        ctx.device,
        ctx.val_dataset.dataset.inverse_transform,
        max_depth=pred_max_depth,
    )

    pred_depths = np.arange(0, pred_max_depth + 1, 1)
    bin_size = 5

    T_pred1_binned = bin_data(T_pred1, bin_size)
    T_pred2_binned = bin_data(T_pred2, bin_size)
    T_pred3_binned = bin_data(T_pred3, bin_size)
    T_pred4_binned = bin_data(T_pred4, bin_size)

    S_pred1_binned = bin_data(S_pred1, bin_size)
    S_pred2_binned = bin_data(S_pred2, bin_size)
    S_pred3_binned = bin_data(S_pred3, bin_size)
    S_pred4_binned = bin_data(S_pred4, bin_size)

    T_diff1 = T_pred1_binned - T1
    T_diff2 = T_pred2_binned - T2
    T_diff3 = T_pred3_binned - T3
    T_diff4 = T_pred4_binned - T4
    S_diff1 = S_pred1_binned - S1
    S_diff2 = S_pred2_binned - S2
    S_diff3 = S_pred3_binned - S3
    S_diff4 = S_pred4_binned - S4

    fig = plt.figure(figsize=(18, 18))
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
    plt.tight_layout()
    plt.show()

    fig = plt.figure(figsize=(18, 18))
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
    plt.tight_layout()
    plt.show()

    fig = plt.figure(figsize=(18, 18))
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
    plt.tight_layout()
    plt.show()

    fig = plt.figure(figsize=(18, 18))
    plot_field_subplot(T4, d4, gld_depths, "Temperature", "Glider T", 321, fig)
    plot_field_subplot(T_pred4, d4, pred_depths, "Temperature", "Synthetic T", 323, fig)
    plot_field_subplot(T_diff4, d4, gld_depths, "T Difference", "T Difference", 325, fig)
    plot_field_subplot(S4, d4, gld_depths, "Salinity", "Glider S", 322, fig)
    plot_field_subplot(S_pred4, d4, pred_depths, "Salinity", "Synthetic S", 324, fig)
    plot_field_subplot(S_diff4, d4, gld_depths, "S Difference", "S Difference", 326, fig)
    plt.suptitle(
        f"Intense LCE \n{t4[0].strftime('%Y-%m-%d')} to {t4[-1].strftime('%Y-%m-%d')}", fontsize=18, fontweight="bold"
    )
    plt.tight_layout()
    plt.show()

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

    lat_all = ctx.full_dataset.LAT
    lon_all = ctx.full_dataset.LON

    lat_train, lon_train = lat_all[ctx.train_indices], lon_all[ctx.train_indices]
    lat_val, lon_val = lat_all[ctx.val_indices], lon_all[ctx.val_indices]
    lat_test, lon_test = lat_all[ctx.test_indices], lon_all[ctx.test_indices]

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND.with_scale("50m"), color="black")
    ax.coastlines(resolution="50m")

    ax.scatter(lon_train, lat_train, s=3, color="k", alpha=0.7, label="ARGO - Train set", transform=ccrs.Geodetic())
    ax.scatter(lon_test, lat_test, s=3, color="b", alpha=0.7, label="ARGO - Validation set", transform=ccrs.Geodetic())
    ax.scatter(
        lon_val, lat_val, s=30, color="r", marker="x", alpha=0.5, label="ARGO - Test set", transform=ccrs.Geodetic()
    )
    ax.scatter(lon_train, lat_train, s=3, color="k", alpha=0.04, transform=ccrs.Geodetic())
    ax.scatter(lon_test, lat_test, s=3, color="b", alpha=0.04, transform=ccrs.Geodetic())
    ax.plot(longitudes_T1, latitudes_T1, color="c", linewidth=2, transform=ccrs.Geodetic(), label="Glider tracks")
    ax.plot(longitudes_T2, latitudes_T2, color="c", linewidth=2, transform=ccrs.Geodetic())
    ax.plot(longitudes_T3, latitudes_T3, color="c", linewidth=2, transform=ccrs.Geodetic())
    ax.plot(longitudes_T4, latitudes_T4, color="c", linewidth=2, transform=ccrs.Geodetic())
    ax.set_xticks(np.arange(-99, -79, 2))
    ax.set_yticks(np.arange(18, 34, 2))
    ax.grid(color="gray", linestyle="--", linewidth=0.5)
    plt.legend(loc="lower right", fontsize=14)
    plt.title("Data availability", fontsize=22, fontweight="bold")
    plt.show()

    aviso_folder = "/unity/f1/ozavala/DATA/GOFFISH/AVISO/GoM/"
    bbox = (18, 32, -99, -81)
    t1_date = datenum_to_datetime(gl_data["t1"][0].mean())
    t2_date = datenum_to_datetime(gl_data["t2"][0].mean())
    t3_date = datenum_to_datetime(gl_data["t3"][0].mean())
    t4_date = datenum_to_datetime(gl_data["t4"][0].mean())
    aviso1_adt, aviso_lats, aviso_lons = get_aviso_by_date(aviso_folder, t1_date, bbox)
    X, Y = np.meshgrid(aviso_lons.values, aviso_lats.values)
    aviso2_adt, _, _ = get_aviso_by_date(aviso_folder, t2_date, bbox)
    aviso3_adt, _, _ = get_aviso_by_date(aviso_folder, t3_date, bbox)
    aviso4_adt, _, _ = get_aviso_by_date(aviso_folder, t4_date, bbox)

    fig = plt.figure(figsize=(12, 12))

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
    cbar1 = plt.colorbar(cf1, ax=ax1, fraction=0.036, pad=0.04)
    cbar1.ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    cbar1.set_label("ADT (m)", fontsize=10)
    cbar1.ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    cbar1.set_label("ADT (m)", fontsize=8)
    cbar1.ax.tick_params(labelsize=8)
    ax1.title.set_text(t1_date.strftime("%Y-%m-%d"))

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
    cbar2 = plt.colorbar(cf2, ax=ax2, fraction=0.036, pad=0.04)
    cbar2.ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    cbar2.set_label("ADT (m)", fontsize=8)
    cbar2.ax.tick_params(labelsize=8)
    ax2.title.set_text(t2_date.strftime("%Y-%m-%d"))

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
    cbar3 = plt.colorbar(cf3, ax=ax3, fraction=0.036, pad=0.04)
    cbar3.ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    cbar3.set_label("ADT (m)", fontsize=8)
    cbar3.ax.tick_params(labelsize=8)
    ax3.title.set_text(t3_date.strftime("%Y-%m-%d"))

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
    cbar4 = plt.colorbar(cf4, ax=ax4, fraction=0.036, pad=0.04)
    cbar4.ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    cbar4.set_label("ADT (m)", fontsize=8)
    cbar4.ax.tick_params(labelsize=8)
    ax4.title.set_text(t4_date.strftime("%Y-%m-%d"))

    plt.suptitle("Gliders", fontsize=22, fontweight="bold")
    plt.show()

    return {
        "nn_temp_residuals": nn_temp_residuals,
        "nn_sal_residuals": nn_sal_residuals,
        "T_pred1": T_pred1,
        "S_pred1": S_pred1,
        "correlation_T1": correlation_T1,
        "correlation_S1": correlation_S1,
    }
