"""PCA regression baseline experiment (hoisted from monolith __main__)."""

from __future__ import annotations

import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from nespreso.analysis.mlr import (
    fit_pcs_regression_exact_gpu,
    predict_pcs_exact_gpu,
    prepare_features,
)
from nespreso.analysis.residuals import compute_depth_rmse_bias, compute_profile_residual
from nespreso.experiments.validation_context import ValidationContext
from nespreso.viz.coefficients import plot_coefficients_heatmap


def _inverse_transform(pcs, pca_temp, pca_sal, n_components):
    from nespreso.data.pca import sklearn_inverse_transform_pcs

    return sklearn_inverse_transform_pcs(pcs, pca_temp, pca_sal, n_components)


def run_pca_regression_baseline(ctx: ValidationContext) -> dict:
    """Fit GPU MLR baseline, plot depth RMSE/bias, and coefficient heatmaps."""
    MLR_degree = 1
    MLR_indices = ctx.train_indices + ctx.val_indices
    inputs_list_train, pcs_list_train = ctx.full_dataset.__getitem__(MLR_indices)

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

    inputs_list_val, _ = ctx.full_dataset.__getitem__(ctx.val_indices)
    inputs_array_val = np.array(inputs_list_val.T)
    X_val = prepare_features(inputs_array_val, max_degree=MLR_degree)
    X_val_scaled = scaler.fit_transform(X_val) + 1

    pcs_pred_val_T = predict_pcs_exact_gpu(beta_T, X_val)
    pcs_pred_val_S = predict_pcs_exact_gpu(beta_S, X_val)
    pcs_pred_val = np.hstack([pcs_pred_val_T, pcs_pred_val_S])

    pca_temp = ctx.full_dataset.pca_temp
    pca_sal = ctx.full_dataset.pca_sal
    temp_MLR_profiles, sal_MLR_profiles = _inverse_transform(
        pcs_pred_val, pca_temp, pca_sal, ctx.n_components
    )

    original_temp_profiles = ctx.original_profiles[:, 0, :]
    original_sal_profiles = ctx.original_profiles[:, 1, :]

    mlr_T_resid = compute_profile_residual(temp_MLR_profiles, original_temp_profiles)
    mlr_S_resid = compute_profile_residual(sal_MLR_profiles, original_sal_profiles)

    avg_mlr_temp_rmse, avg_mlr_temp_bias = compute_depth_rmse_bias(mlr_T_resid, axis=1)
    avg_mlr_sal_rmse, avg_mlr_sal_bias = compute_depth_rmse_bias(mlr_S_resid, axis=1)

    fig = plt.figure(figsize=(18, 18))

    ax = fig.add_subplot(2, 2, 1)
    ax.axvline(0, color="k", linestyle="--", linewidth=0.5)
    ax.grid(color="gray", linestyle="--", linewidth=0.5)
    plt.plot(ctx.ist.rmse.values, ctx.ist.depth.values, linewidth=3, label="ISOP", color="xkcd:blue")
    plt.plot(ctx.avg_gem_temp_rmse, np.arange(0, 1801), linewidth=3, label="GEM", color="xkcd:orange")
    plt.plot(avg_mlr_temp_rmse, np.arange(0, 1801), linewidth=3, label="MLR", color="xkcd:green")
    plt.plot(ctx.avg_old_temp_rmse, np.arange(0, 1801), linewidth=3, label="NeSPReSO 1.0", color="xkcd:purple")
    plt.plot(ctx.avg_nn_temp_rmse, np.arange(0, 1801), linewidth=3, label="NeSPReSO 1.1", color="xkcd:gray")
    ax.invert_yaxis()
    plt.legend()
    plt.xlabel("Temperature RMSE [°C]")
    plt.ylabel("Depth [m]")
    plt.title("Average Temperature RMSE")

    ax = fig.add_subplot(2, 2, 2)
    ax.axvline(0, color="k", linestyle="--", linewidth=0.5)
    ax.grid(color="gray", linestyle="--", linewidth=0.5)
    plt.plot(ctx.iss.rmse.values, ctx.iss.depth.values, linewidth=3, label="ISOP", color="xkcd:blue")
    plt.plot(ctx.avg_gem_sal_rmse, np.arange(0, 1801), linewidth=3, label="GEM", color="xkcd:orange")
    plt.plot(avg_mlr_sal_rmse, np.arange(0, 1801), linewidth=3, label="MLR", color="xkcd:green")
    plt.plot(ctx.avg_old_sal_rmse, np.arange(0, 1801), linewidth=3, label="NeSPReSO 1.0", color="xkcd:purple")
    plt.plot(ctx.avg_nn_sal_rmse, np.arange(0, 1801), linewidth=3, label="NeSPReSO 1.1", color="xkcd:gray")
    ax.invert_yaxis()
    plt.legend()
    plt.xlabel("Salinity RMSE [PSU]")
    plt.title("Average Salinity RMSE")

    ax = fig.add_subplot(2, 2, 3)
    ax.axvline(0, color="k", linestyle="--", linewidth=0.5)
    ax.grid(color="gray", linestyle="--", linewidth=0.5)
    plt.plot(ctx.ist.bias.values, ctx.ist.depth.values, linewidth=3, label="ISOP", color="xkcd:blue")
    plt.plot(ctx.avg_gem_temp_bias, np.arange(0, 1801), linewidth=3, label="GEM", color="xkcd:orange")
    plt.plot(avg_mlr_temp_bias, np.arange(0, 1801), linewidth=3, label="MLR", color="xkcd:green")
    plt.plot(ctx.avg_old_temp_bias, np.arange(0, 1801), linewidth=3, label="NeSPReSO 1.0", color="xkcd:purple")
    plt.plot(ctx.avg_nn_temp_bias, np.arange(0, 1801), linewidth=3, label="NeSPReSO 1.1", color="xkcd:gray")
    ax.invert_yaxis()
    plt.legend()
    plt.xlabel("Temperature Bias [°C]")
    plt.ylabel("Depth [m]")
    plt.title("Average Temperature Bias")

    ax = fig.add_subplot(2, 2, 4)
    ax.axvline(0, color="k", linestyle="--", linewidth=0.5)
    ax.grid(color="gray", linestyle="--", linewidth=0.5)
    plt.plot(ctx.iss.bias.values, ctx.iss.depth.values, linewidth=3, label="ISOP", color="xkcd:blue")
    plt.plot(ctx.avg_gem_sal_bias, np.arange(0, 1801), linewidth=3, label="GEM", color="xkcd:orange")
    plt.plot(avg_mlr_sal_bias, np.arange(0, 1801), linewidth=3, label="MLR", color="xkcd:green")
    plt.plot(ctx.avg_old_sal_bias, np.arange(0, 1801), linewidth=3, label="NeSPReSO 1.0", color="xkcd:purple")
    plt.plot(ctx.avg_nn_sal_bias, np.arange(0, 1801), linewidth=3, label="NeSPReSO 1.1", color="xkcd:gray")
    ax.invert_yaxis()
    plt.legend()
    plt.xlabel("Salinity Bias [PSU]")
    plt.title("Average Salinity Bias")

    plt.tight_layout()
    plt.show()

    feature_names = ["timecos", "timesin", "latcos", "latsin", "loncos", "lonsin", "sst", "sss", "ssh"]

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

    return {
        "beta_T": beta_T,
        "beta_S": beta_S,
        "mlr_T_resid": mlr_T_resid,
        "mlr_S_resid": mlr_S_resid,
        "avg_mlr_temp_rmse": avg_mlr_temp_rmse,
        "avg_mlr_temp_bias": avg_mlr_temp_bias,
        "avg_mlr_sal_rmse": avg_mlr_sal_rmse,
        "avg_mlr_sal_bias": avg_mlr_sal_bias,
        "X_avgs": X_avgs,
    }
