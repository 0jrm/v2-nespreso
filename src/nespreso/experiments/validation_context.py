"""Shared validation artifacts built after training (hoisted from monolith __main__)."""

from __future__ import annotations

import pickle
import time
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import xarray as xr
from sklearn.base import InconsistentVersionWarning
from torch.utils.data import DataLoader

from nespreso.analysis.residuals import compute_depth_rmse_bias, compute_profile_residual
from nespreso.config import AppConfig
from nespreso.data.pca import sklearn_inverse_transform_pcs
from nespreso.determinism import get_device
from nespreso.inference import (
    get_predictions,
    get_predictions_torchscript,
    load_all_models,
)
from nespreso.reporting import print_training_params
from nespreso.runner import TrainingArtifacts


@dataclass
class ValidationContext:
    """Validation predictions, residuals, and map bins for post-training experiments."""

    cfg: AppConfig
    full_dataset: Any
    train_dataset: Any
    val_dataset: Any
    test_dataset: Any
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    trained_model: Any
    device: torch.device
    input_dim: int
    input_params: dict[str, bool]
    n_components: int
    layers_config: list[int]
    batch_size: int
    min_depth: float
    max_depth: float
    dropout_prob: float
    learning_rate: float
    train_size: float
    val_size: float
    test_size: float
    bin_size: float
    pred_T: np.ndarray
    pred_S: np.ndarray
    old_pred_T: np.ndarray
    old_pred_S: np.ndarray
    original_profiles: np.ndarray
    pca_approx_profiles: np.ndarray
    orig_T: np.ndarray
    orig_S: np.ndarray
    pred_T_resid: np.ndarray
    pred_S_resid: np.ndarray
    gems_T: np.ndarray
    gems_S: np.ndarray
    gems_T_resid: np.ndarray
    gems_S_resid: np.ndarray
    old_T_resid: np.ndarray
    old_S_resid: np.ndarray
    gem_temp: np.ndarray
    gem_sal: np.ndarray
    sst_inputs: np.ndarray
    ssh_inputs: np.ndarray
    lat_val: np.ndarray
    lon_val: np.ndarray
    dates_val: np.ndarray
    subset_indices: np.ndarray
    train_indices: np.ndarray
    val_indices: np.ndarray
    test_indices: np.ndarray
    data_ISOP: xr.Dataset
    lon_bins: np.ndarray
    lat_bins: np.ndarray
    lon_centers: np.ndarray
    lat_centers: np.ndarray
    ist: xr.Dataset
    iss: xr.Dataset
    isop_depths: np.ndarray
    avg_gem_temp_rmse: np.ndarray
    avg_gem_temp_bias: np.ndarray
    avg_nn_temp_rmse: np.ndarray
    avg_nn_temp_bias: np.ndarray
    avg_old_temp_rmse: np.ndarray
    avg_old_temp_bias: np.ndarray
    avg_gem_sal_rmse: np.ndarray
    avg_gem_sal_bias: np.ndarray
    avg_nn_sal_rmse: np.ndarray
    avg_nn_sal_bias: np.ndarray
    avg_old_sal_rmse: np.ndarray
    avg_old_sal_bias: np.ndarray
    val_predictions: list[np.ndarray]


def build_validation_context(
    cfg: AppConfig,
    artifacts: TrainingArtifacts,
    *,
    bin_size: float = 1.0,
) -> ValidationContext:
    """Mirror monolith post-training validation setup (timing, preds, GEM, legacy 1.0)."""
    from dataclasses import asdict

    model_cfg = cfg.model
    runtime = cfg.runtime
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
    nn_repeat_time = runtime.nn_repeat_time
    gem_repeat_time = runtime.gem_repeat_time
    ensemble_models = runtime.ensemble_models

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

    print_training_params(
        full_dataset,
        input_params=input_params,
        n_components=n_components,
        batch_size=batch_size,
        min_depth=min_depth,
        max_depth=max_depth,
        learning_rate=learning_rate,
        dropout_prob=dropout_prob,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        layers_config=layers_config,
    )

    print("Statistics from the last run:")
    print("Method & # profiles & Time (ms) & Time per profile (µs)")
    n_val = len(val_dataset)
    start_time = time.perf_counter()
    for i in range(nn_repeat_time):
        val_predictions_pcs = get_predictions(trained_model, val_loader, device)
        val_predictions = val_dataset.dataset.inverse_transform(val_predictions_pcs)
    end_time = time.perf_counter()
    elapsed_time = (end_time - start_time) / nn_repeat_time
    print(f"NeSPReSO 1.1 ({device}) & {n_val} & {elapsed_time * 1e3:.2f} & {((elapsed_time * 1e6) / n_val):.2f}")

    if torch.cuda.is_available():
        if next(trained_model.parameters()).is_cuda:
            cuda_device = torch.device("cpu")
        else:
            cuda_device = torch.device("cuda")

        trained_model = trained_model.to(cuda_device)
        cuda_val_loader = DataLoader(val_dataset, batch_size=val_loader.batch_size, sampler=val_loader.sampler)

        start_time = time.perf_counter()
        for i in range(nn_repeat_time):
            cuda_val_predictions_pcs = get_predictions(trained_model, cuda_val_loader, cuda_device)
            cuda_val_predictions = val_dataset.dataset.inverse_transform(cuda_val_predictions_pcs)
        end_time = time.perf_counter()
        cuda_elapsed_time = (end_time - start_time) / nn_repeat_time
        print(
            f"NeSPReSO 1.1 ({cuda_device}) & {n_val} & {cuda_elapsed_time * 1e3:.2f} & {((cuda_elapsed_time * 1e6) / n_val):.2f}"
        )
        trained_model = trained_model.to(device)
    else:
        print("CUDA is not available. Skipping GPU time calculation.")

    file_path_new = "/unity/g2/jmiranda/SubsurfaceFields/Data/ISOP1_rmse_bias_1deg_maps.nc"
    data_ISOP = xr.open_dataset(file_path_new)

    lon_bins = np.arange(np.min(data_ISOP.lon) - 0.5, np.max(data_ISOP.lon) + 1.5, 1)
    lat_bins = np.arange(np.min(data_ISOP.lat) - 0.5, np.max(data_ISOP.lat) + 1.5, 1)
    lon_centers = lon_bins + bin_size / 2
    lat_centers = lat_bins + bin_size / 2

    subset_indices = val_loader.dataset.indices
    original_profiles = val_dataset.dataset.get_profiles(subset_indices, pca_approx=False)
    pca_approx_profiles = val_dataset.dataset.get_profiles(subset_indices, pca_approx=True)

    if ensemble_models:
        models_dir = "/unity/g2/jmiranda/SubsurfaceFields/GEM_SubsurfaceFields/saved_models/"
        start_time = time.perf_counter()
        models = load_all_models(
            models_dir=models_dir,
            device=device,
            input_dim=input_dim,
            layers_config=layers_config,
            n_components=n_components,
            dropout_prob=dropout_prob,
        )
        accumulated_pred_T = None
        accumulated_pred_S = None
        for model in models:
            val_predictions_pcs = get_predictions(model, val_loader, device)
            val_predictions = val_dataset.dataset.inverse_transform(val_predictions_pcs)
            pred_T_current = val_predictions[0]
            pred_S_current = val_predictions[1]
            if accumulated_pred_T is None:
                accumulated_pred_T = pred_T_current
                accumulated_pred_S = pred_S_current
            else:
                accumulated_pred_T += pred_T_current
                accumulated_pred_S += pred_S_current
        avg_pred_T = accumulated_pred_T / len(models)
        avg_pred_S = accumulated_pred_S / len(models)
        end_time = time.perf_counter()
        cuda_elapsed_time = end_time - start_time
        pred_T = avg_pred_T
        pred_S = avg_pred_S
        print(f"NeSPReSO 1.1 ensamble 15x & {cuda_elapsed_time * 1e3:.2f} & {((cuda_elapsed_time * 1e6) / n_val):.2f}")
    else:
        pred_T = val_predictions[0]
        pred_S = val_predictions[1]

    print("Loading NeSPReSO 1.0 model for comparison (using API model)...")
    old_model_path = "/unity/g2/jmiranda/nespreso_api/models/ocean_tensorscript.pt"
    old_pca_path = "/unity/g2/jmiranda/nespreso_api/models/pca_stats.pkl"
    old_model = torch.jit.load(old_model_path, map_location=torch.device(get_device()))
    old_model.eval()

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
        with open(old_pca_path, "rb") as f:
            old_pca_data = pickle.load(f)
    old_pca_temp = old_pca_data["pca_temp"]
    old_pca_sal = old_pca_data["pca_sal"]
    old_input_params = old_pca_data.get("input_params", input_params)

    print(f"NeSPReSO 1.0 model loaded from API (TorchScript format)")
    print(f"Old model input params: {old_input_params}")

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

    start_time = time.perf_counter()
    for i in range(gem_repeat_time):
        gem_temp, gem_sal = val_dataset.dataset.get_gem_profiles(subset_indices)
    end_time = time.perf_counter()
    elapsed_time = (end_time - start_time) / gem_repeat_time
    print(f"GEM (cpu) & {n_val} & {elapsed_time * 1e3:.2f} & {((elapsed_time * 1e6) / n_val):.2f}")
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

    print_training_params(
        full_dataset,
        input_params=input_params,
        n_components=n_components,
        batch_size=batch_size,
        min_depth=min_depth,
        max_depth=max_depth,
        learning_rate=learning_rate,
        dropout_prob=dropout_prob,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        layers_config=layers_config,
    )

    print("Let's investigate how the method compares against vanilla GEM with in-situ SSH")

    ist = xr.open_dataset("/unity/g2/jmiranda/SubsurfaceFields/Data/isop1_stats_temp.nc")
    iss = xr.open_dataset("/unity/g2/jmiranda/SubsurfaceFields/Data/isop1_stats_salt.nc")
    isop_depths = ist.depth.values
    avg_gem_temp_rmse, avg_gem_temp_bias = compute_depth_rmse_bias(gems_T_resid, axis=1)
    avg_nn_temp_rmse, avg_nn_temp_bias = compute_depth_rmse_bias(pred_T_resid, axis=1)
    avg_old_temp_rmse, avg_old_temp_bias = compute_depth_rmse_bias(old_T_resid, axis=1)
    avg_gem_sal_rmse, avg_gem_sal_bias = compute_depth_rmse_bias(gems_S_resid, axis=1)
    avg_nn_sal_rmse, avg_nn_sal_bias = compute_depth_rmse_bias(pred_S_resid, axis=1)
    avg_old_sal_rmse, avg_old_sal_bias = compute_depth_rmse_bias(old_S_resid, axis=1)

    train_indices = train_dataset.indices
    val_indices = val_dataset.indices
    test_indices = test_dataset.indices

    return ValidationContext(
        cfg=cfg,
        full_dataset=full_dataset,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        trained_model=trained_model,
        device=device,
        input_dim=input_dim,
        input_params=input_params,
        n_components=n_components,
        layers_config=layers_config,
        batch_size=batch_size,
        min_depth=min_depth,
        max_depth=max_depth,
        dropout_prob=dropout_prob,
        learning_rate=learning_rate,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        bin_size=bin_size,
        pred_T=pred_T,
        pred_S=pred_S,
        old_pred_T=old_pred_T,
        old_pred_S=old_pred_S,
        original_profiles=original_profiles,
        pca_approx_profiles=pca_approx_profiles,
        orig_T=orig_T,
        orig_S=orig_S,
        pred_T_resid=pred_T_resid,
        pred_S_resid=pred_S_resid,
        gems_T=gems_T,
        gems_S=gems_S,
        gems_T_resid=gems_T_resid,
        gems_S_resid=gems_S_resid,
        old_T_resid=old_T_resid,
        old_S_resid=old_S_resid,
        gem_temp=gem_temp,
        gem_sal=gem_sal,
        sst_inputs=sst_inputs,
        ssh_inputs=ssh_inputs,
        lat_val=lat_val,
        lon_val=lon_val,
        dates_val=dates_val,
        subset_indices=subset_indices,
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
        data_ISOP=data_ISOP,
        lon_bins=lon_bins,
        lat_bins=lat_bins,
        lon_centers=lon_centers,
        lat_centers=lat_centers,
        ist=ist,
        iss=iss,
        isop_depths=isop_depths,
        avg_gem_temp_rmse=avg_gem_temp_rmse,
        avg_gem_temp_bias=avg_gem_temp_bias,
        avg_nn_temp_rmse=avg_nn_temp_rmse,
        avg_nn_temp_bias=avg_nn_temp_bias,
        avg_old_temp_rmse=avg_old_temp_rmse,
        avg_old_temp_bias=avg_old_temp_bias,
        avg_gem_sal_rmse=avg_gem_sal_rmse,
        avg_gem_sal_bias=avg_gem_sal_bias,
        avg_nn_sal_rmse=avg_nn_sal_rmse,
        avg_nn_sal_bias=avg_nn_sal_bias,
        avg_old_sal_rmse=avg_old_sal_rmse,
        avg_old_sal_bias=avg_old_sal_bias,
        val_predictions=val_predictions,
    )
