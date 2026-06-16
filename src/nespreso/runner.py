"""Training pipeline entrypoint driven by AppConfig."""

from __future__ import annotations

import os
import pickle
import time
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from nespreso.config import AppConfig, density_penalty_dict
from nespreso.data.dataset import TemperatureSalinityDataset
from nespreso.data.pickle_compat import load_dataset_pickle
from nespreso.data.splits import split_dataset
from nespreso.determinism import get_device, set_seed
from nespreso.losses import CombinedPCALoss
from nespreso.models.mlp import PredictionModel
from nespreso.train import evaluate_model, train_model


@dataclass(frozen=True)
class TrainingArtifacts:
    """Dataset split, loaders, and trained model produced by ``run_training``."""

    full_dataset: Any
    train_dataset: Any
    val_dataset: Any
    test_dataset: Any
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    trained_model: Any
    device: torch.device
    input_params: dict[str, bool]
    input_dim: int


def require_trained_model_path(cfg: AppConfig) -> str:
    """Return an existing checkpoint path when ``load_trained_model`` is enabled."""
    path = cfg.paths.trained_model_path
    if not path:
        raise ValueError(
            "runtime.load_trained_model is true but paths.trained_model_path is not set. "
            "Provide an explicit checkpoint path in the config."
        )
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(
            f"runtime.load_trained_model is true but checkpoint not found: {path}"
        )
    return str(resolved)


def _load_dataset_pickle(monolith_module: Any, pickle_path: str | Path) -> dict[str, Any]:
    """
    Backward-compatible alias for tests that still pass a monolith module.

    The monolith module is ignored; loading uses ``pickle_compat`` class remapping.
    """
    if monolith_module is not None:
        warnings.warn(
            "_load_dataset_pickle(monolith, path) is deprecated; monolith_module is ignored. "
            "Use nespreso.data.pickle_compat.load_dataset_pickle(path).",
            DeprecationWarning,
            stacklevel=2,
        )
    return load_dataset_pickle(pickle_path)


def apply_runtime_globals(cfg: AppConfig, monolith_module: Any | None = None) -> None:
    """
    Mirror runtime flags from config onto package modules.

    When ``monolith_module`` is provided (characterization tests), also sets legacy
    module-level attributes on the monolith shim for backward compatibility.
    """
    set_seed(cfg.runtime.seed)

    from nespreso.data import dataset as dataset_module

    dataset_module.debug = cfg.runtime.debug

    if monolith_module is None:
        return

    monolith_module.load_trained_model = cfg.runtime.load_trained_model
    monolith_module.ensemble_models = cfg.runtime.ensemble_models
    monolith_module.load_dataset_file = cfg.runtime.load_dataset_file
    monolith_module.gen_paula_profiles = cfg.runtime.gen_paula_profiles
    monolith_module.debug = cfg.runtime.debug
    monolith_module.seed = cfg.runtime.seed
    monolith_module.n_runs = cfg.runtime.n_runs
    monolith_module.nn_repeat_time = cfg.runtime.nn_repeat_time
    monolith_module.gem_repeat_time = cfg.runtime.gem_repeat_time
    monolith_module.DEVICE = get_device()


def _prepare_data_and_loaders(cfg: AppConfig) -> dict[str, Any]:
    """Build or load the dataset pickle and train/val/test loaders."""
    model_cfg = cfg.model
    input_params = asdict(cfg.input_params)
    density_penalty_config = density_penalty_dict(cfg)

    dataset_pickle_file = cfg.paths.dataset_pickle
    bbox = cfg.bbox

    if os.path.exists(dataset_pickle_file) and cfg.runtime.load_dataset_file:
        data = load_dataset_pickle(dataset_pickle_file)
        full_dataset = data["full_dataset"]
        full_dataset.n_components = model_cfg.n_components
        full_dataset.min_depth = model_cfg.min_depth
        full_dataset.max_depth = model_cfg.max_depth
        full_dataset.input_params = input_params
        if not cfg.runtime.load_trained_model:
            full_dataset.reload()
    else:
        full_dataset = TemperatureSalinityDataset(
            n_components=model_cfg.n_components,
            input_params=input_params,
            min_depth=model_cfg.min_depth,
            max_depth=model_cfg.max_depth,
            data_path=cfg.paths.argo_mat,
            aviso_folder=cfg.paths.aviso_folder,
            sst_folder=cfg.paths.sst_folder,
            sss_folder=cfg.paths.sss_folder,
            min_lat=bbox.min_lat,
            max_lat=bbox.max_lat,
            min_lon=bbox.min_lon,
            max_lon=bbox.max_lon,
            ex_lat=bbox.ex_lat,
            ex_lon=bbox.ex_lon,
        )
        os.makedirs(os.path.dirname(dataset_pickle_file), exist_ok=True)
        with open(dataset_pickle_file, "wb") as file:
            pickle.dump({"full_dataset": full_dataset}, file)

    train_dataset, val_dataset, test_dataset = split_dataset(
        full_dataset,
        model_cfg.train_size,
        model_cfg.val_size,
        model_cfg.test_size,
    )

    train_loader = DataLoader(train_dataset, batch_size=model_cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=model_cfg.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=model_cfg.batch_size, shuffle=False)

    subset_indices = val_loader.dataset.indices
    full_dataset.calc_gem(subset_indices)

    input_dim = sum(val for val in input_params.values()) - 1 * input_params["sat"]
    device = get_device()
    weights = full_dataset.get_pca_weights()

    print(
        f"Explained Variance - T: {(sum(full_dataset.pca_temp.explained_variance_ratio_) * 100):.1f}% - "
        f"S: {(100 * sum(full_dataset.pca_sal.explained_variance_ratio_)):.1f}%"
    )

    criterion = CombinedPCALoss(
        temp_pca=train_dataset.dataset,
        sal_pca=train_dataset.dataset,
        n_components=model_cfg.n_components,
        weights=weights,
        device=device,
        density_config=density_penalty_config,
    )

    return {
        "full_dataset": full_dataset,
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "test_dataset": test_dataset,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "criterion": criterion,
        "input_params": input_params,
        "input_dim": input_dim,
        "device": device,
    }


def run_training(cfg: AppConfig, *, return_artifacts: bool = False) -> Path | None | tuple[Path | None, TrainingArtifacts]:
    """
    Build/load dataset pickle, split 0.7/0.15/0.15, train with early stopping.

    Returns path to saved checkpoint when training; None when loading a trained model.
    When ``return_artifacts`` is true, also returns dataset split and trained model for
    post-training analysis experiments.
    """
    apply_runtime_globals(cfg)

    model_cfg = cfg.model
    prep = _prepare_data_and_loaders(cfg)
    full_dataset = prep["full_dataset"]
    train_loader = prep["train_loader"]
    val_loader = prep["val_loader"]
    test_loader = prep["test_loader"]
    criterion = prep["criterion"]
    input_params = prep["input_params"]
    input_dim = prep["input_dim"]
    device = prep["device"]

    summary_writer = None
    if cfg.monitor.tensorboard:
        from torch.utils.tensorboard import SummaryWriter

        summary_writer = SummaryWriter(log_dir=cfg.monitor.log_dir)

    save_model_path: Path | None = None
    trained_model = None

    if cfg.runtime.load_trained_model:
        trained_model_path = require_trained_model_path(cfg)
        checkpoint = torch.load(trained_model_path, map_location=device, weights_only=False)
        trained_model = PredictionModel(
            input_dim=input_dim,
            layers_config=list(model_cfg.layers_config),
            output_dim=model_cfg.n_components * 2,
            dropout_prob=model_cfg.dropout_prob,
        )
        trained_model.load_state_dict(checkpoint["model_state_dict"])
        trained_model.to(device)
        full_dataset.pca_temp = checkpoint["pca_temp"]
        full_dataset.pca_sal = checkpoint["pca_sal"]
        print(f"Using loaded model from: {trained_model_path}")
    else:
        for run in range(cfg.runtime.n_runs):
            print(f"Run {run + 1}/{cfg.runtime.n_runs}")
            model = PredictionModel(
                input_dim=input_dim,
                layers_config=list(model_cfg.layers_config),
                output_dim=model_cfg.n_components * 2,
                dropout_prob=model_cfg.dropout_prob,
            )
            model.to(device)
            optimizer = optim.Adam(model.parameters(), lr=model_cfg.learning_rate)

            start_time = time.perf_counter()
            trained_model = train_model(
                model,
                train_loader,
                val_loader,
                criterion,
                optimizer,
                device,
                model_cfg.epochs,
                model_cfg.patience,
                summary_writer=summary_writer,
            )
            elapsed_time = time.perf_counter() - start_time
            print(f"NeSPReSO 1.1 train: {elapsed_time:.2f} seconds.")

            test_loss = evaluate_model(trained_model, test_loader, criterion, device)
            print(f"Test Loss: {test_loss:.4f}")

            suffix = "_sat.pth" if input_params["sat"] else ".pth"
            save_model_path = Path(cfg.paths.saved_models_dir) / (
                f"ocp_model_Test Loss: {test_loss:.4f}_{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{suffix}"
            )
            save_model_path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint = {
                "model_state_dict": trained_model.state_dict(),
                "pca_temp": full_dataset.pca_temp,
                "pca_sal": full_dataset.pca_sal,
                "input_params": input_params,
            }
            torch.save(checkpoint, save_model_path)
            print(f"Saved model and PCA components to {save_model_path}")

    if summary_writer is not None:
        summary_writer.close()

    if return_artifacts:
        artifacts = TrainingArtifacts(
            full_dataset=prep["full_dataset"],
            train_dataset=prep["train_dataset"],
            val_dataset=prep["val_dataset"],
            test_dataset=prep["test_dataset"],
            train_loader=prep["train_loader"],
            val_loader=prep["val_loader"],
            test_loader=prep["test_loader"],
            trained_model=trained_model,
            device=prep["device"],
            input_params=prep["input_params"],
            input_dim=prep["input_dim"],
        )
        return save_model_path, artifacts

    return save_model_path
