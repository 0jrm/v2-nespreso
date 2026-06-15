"""Training pipeline entrypoint driven by AppConfig."""

from __future__ import annotations

import importlib.util
import os
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from dataclasses import asdict

from nespreso.config import AppConfig, density_penalty_dict
from nespreso.determinism import get_device, set_seed


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


def _load_monolith():
    """Import the research monolith from the repo root."""
    import importlib.util
    import sys
    from pathlib import Path

    loader_name = "nespreso_monolith_loader"
    if loader_name not in sys.modules:
        loader_path = Path(__file__).resolve().parents[2] / "tests" / "monolith_loader.py"
        spec = importlib.util.spec_from_file_location(loader_name, loader_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load monolith loader from {loader_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[loader_name] = module
        spec.loader.exec_module(module)
    return sys.modules[loader_name].load_monolith()


def _load_dataset_pickle(monolith_module, pickle_path: str | Path):
    """Unpickle dataset saved when the monolith was run as ``__main__``."""
    real_main = sys.modules["__main__"]
    sys.modules["__main__"] = monolith_module
    try:
        with open(pickle_path, "rb") as file:
            return pickle.load(file)
    finally:
        sys.modules["__main__"] = real_main


def apply_runtime_globals(m, cfg: AppConfig) -> None:
    """Mirror module-level flags from config (backward compat for monolith)."""
    m.load_trained_model = cfg.runtime.load_trained_model
    m.ensemble_models = cfg.runtime.ensemble_models
    m.load_dataset_file = cfg.runtime.load_dataset_file
    m.gen_paula_profiles = cfg.runtime.gen_paula_profiles
    m.debug = cfg.runtime.debug
    m.seed = cfg.runtime.seed
    m.n_runs = cfg.runtime.n_runs
    m.nn_repeat_time = cfg.runtime.nn_repeat_time
    m.gem_repeat_time = cfg.runtime.gem_repeat_time
    set_seed(cfg.runtime.seed)
    m.DEVICE = get_device()


def run_training(cfg: AppConfig) -> Path | None:
    """
    Build/load dataset pickle, split 0.7/0.15/0.15, train with early stopping.

    Returns path to saved checkpoint when training; None when loading a trained model.
    """
    m = _load_monolith()
    apply_runtime_globals(m, cfg)

    model_cfg = cfg.model
    input_params = asdict(cfg.input_params)
    density_penalty_config = density_penalty_dict(cfg)

    dataset_pickle_file = cfg.paths.dataset_pickle
    bbox = cfg.bbox

    if os.path.exists(dataset_pickle_file) and cfg.runtime.load_dataset_file:
        data = _load_dataset_pickle(m, dataset_pickle_file)
        full_dataset = data["full_dataset"]
        full_dataset.n_components = model_cfg.n_components
        full_dataset.min_depth = model_cfg.min_depth
        full_dataset.max_depth = model_cfg.max_depth
        full_dataset.input_params = input_params
        if not cfg.runtime.load_trained_model:
            full_dataset.reload()
    else:
        full_dataset = m.TemperatureSalinityDataset(
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
            pickle.dump(
                {
                    "min_depth": model_cfg.min_depth,
                    "max_depth": model_cfg.max_depth,
                    "epochs": model_cfg.epochs,
                    "patience": model_cfg.patience,
                    "n_components": model_cfg.n_components,
                    "batch_size": model_cfg.batch_size,
                    "learning_rate": model_cfg.learning_rate,
                    "dropout_prob": model_cfg.dropout_prob,
                    "layers_config": list(model_cfg.layers_config),
                    "input_params": input_params,
                    "train_size": model_cfg.train_size,
                    "val_size": model_cfg.val_size,
                    "test_size": model_cfg.test_size,
                    "full_dataset": full_dataset,
                },
                file,
            )

    train_dataset, val_dataset, test_dataset = m.split_dataset(
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
    device = m.DEVICE
    weights = full_dataset.get_pca_weights()

    print(
        f"Explained Variance - T: {(sum(full_dataset.pca_temp.explained_variance_ratio_) * 100):.1f}% - "
        f"S: {(100 * sum(full_dataset.pca_sal.explained_variance_ratio_)):.1f}%"
    )

    criterion = m.CombinedPCALoss(
        temp_pca=train_dataset.dataset,
        sal_pca=train_dataset.dataset,
        n_components=model_cfg.n_components,
        weights=weights,
        device=device,
        density_config=density_penalty_config,
    )

    summary_writer = None
    if cfg.monitor.tensorboard:
        from torch.utils.tensorboard import SummaryWriter

        summary_writer = SummaryWriter(log_dir=cfg.monitor.log_dir)

    save_model_path: Path | None = None

    if cfg.runtime.load_trained_model:
        trained_model_path = require_trained_model_path(cfg)
        checkpoint = torch.load(trained_model_path, map_location=device, weights_only=False)
        trained_model = m.PredictionModel(
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
            model = m.PredictionModel(
                input_dim=input_dim,
                layers_config=list(model_cfg.layers_config),
                output_dim=model_cfg.n_components * 2,
                dropout_prob=model_cfg.dropout_prob,
            )
            model.to(device)
            optimizer = optim.Adam(model.parameters(), lr=model_cfg.learning_rate)

            start_time = time.perf_counter()
            trained_model = m.train_model(
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

            test_loss = m.evaluate_model(trained_model, test_loader, criterion, device)
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

    return save_model_path
