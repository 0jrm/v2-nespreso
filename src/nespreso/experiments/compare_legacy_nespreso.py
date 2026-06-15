"""NeSPReSO 1.1 timing, ensemble, legacy 1.0, and GEM timing (hoisted from validation_context)."""

from __future__ import annotations

import pickle
import time
import warnings
from typing import Any

import numpy as np
import torch
from sklearn.base import InconsistentVersionWarning
from torch.utils.data import DataLoader

from nespreso.data.pca import sklearn_inverse_transform_pcs
from nespreso.determinism import get_device
from nespreso.inference import get_predictions, get_predictions_torchscript, load_all_models


def run_nespreso_inference_timing(
    *,
    trained_model: Any,
    val_loader: DataLoader,
    val_dataset: Any,
    device: torch.device,
    nn_repeat_time: int,
) -> tuple[list[np.ndarray], Any]:
    """Time NeSPReSO 1.1 inference on the current device and optional CUDA/CPU flip."""
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

    return val_predictions, trained_model


def run_legacy_prediction_comparison(
    *,
    val_predictions: list[np.ndarray],
    val_loader: DataLoader,
    val_dataset: Any,
    device: torch.device,
    input_params: dict[str, bool],
    n_components: int,
    layers_config: list[int],
    dropout_prob: float,
    input_dim: int,
    ensemble_models: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Resolve ensemble or single NeSPReSO 1.1 preds and load NeSPReSO 1.0 API model."""
    n_val = len(val_dataset)

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

    return pred_T, pred_S, old_pred_T, old_pred_S


def run_gem_inference_timing(
    *,
    val_dataset: Any,
    subset_indices: np.ndarray,
    gem_repeat_time: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Time GEM profile generation on the validation subset."""
    n_val = len(val_dataset)
    start_time = time.perf_counter()
    for i in range(gem_repeat_time):
        gem_temp, gem_sal = val_dataset.dataset.get_gem_profiles(subset_indices)
    end_time = time.perf_counter()
    elapsed_time = (end_time - start_time) / gem_repeat_time
    print(f"GEM (cpu) & {n_val} & {elapsed_time * 1e3:.2f} & {((elapsed_time * 1e6) / n_val):.2f}")
    return gem_temp, gem_sal


def run_compare_legacy_nespreso(
    *,
    trained_model: Any,
    val_loader: DataLoader,
    val_dataset: Any,
    device: torch.device,
    subset_indices: np.ndarray,
    input_params: dict[str, bool],
    n_components: int,
    layers_config: list[int],
    dropout_prob: float,
    input_dim: int,
    nn_repeat_time: int,
    gem_repeat_time: int,
    ensemble_models: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[np.ndarray], Any, np.ndarray, np.ndarray]:
    """Run full legacy comparison pipeline (timing, preds, GEM timing)."""
    val_predictions, trained_model = run_nespreso_inference_timing(
        trained_model=trained_model,
        val_loader=val_loader,
        val_dataset=val_dataset,
        device=device,
        nn_repeat_time=nn_repeat_time,
    )
    pred_T, pred_S, old_pred_T, old_pred_S = run_legacy_prediction_comparison(
        val_predictions=val_predictions,
        val_loader=val_loader,
        val_dataset=val_dataset,
        device=device,
        input_params=input_params,
        n_components=n_components,
        layers_config=layers_config,
        dropout_prob=dropout_prob,
        input_dim=input_dim,
        ensemble_models=ensemble_models,
    )
    gem_temp, gem_sal = run_gem_inference_timing(
        val_dataset=val_dataset,
        subset_indices=subset_indices,
        gem_repeat_time=gem_repeat_time,
    )
    return pred_T, pred_S, old_pred_T, old_pred_S, val_predictions, trained_model, gem_temp, gem_sal
