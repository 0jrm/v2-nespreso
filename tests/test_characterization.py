"""
Characterization tests for refactor safety.

Golden-output tests (marked ``requires_unity``) must be captured on HPC where
``/unity`` data and GPU are available. Run:

    pytest tests/test_characterization.py -m requires_unity --run-unity

to record or verify golden files under ``tests/golden/``.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from nespreso.config import AppConfig, density_penalty_dict, load_config
from nespreso.data.pickle_compat import load_dataset_pickle
from nespreso.data.splits import split_dataset
from nespreso.determinism import get_device, set_seed
from nespreso.losses import CombinedPCALoss
from nespreso.models.mlp import PredictionModel
from nespreso.runner import apply_runtime_globals
from nespreso.train import train_model
from tests.monolith_loader import load_monolith

GOLDEN_DIR = Path(__file__).parent / "golden"
TOL = 1e-6
GOLDEN_TRAIN_EPOCHS = 5


def _capture_short_train_trajectory(cfg: AppConfig) -> list[dict[str, float]]:
    """Run a deterministic short training loop for golden characterization."""
    m = load_monolith()
    apply_runtime_globals(cfg, m)

    model_cfg = cfg.model
    input_params = asdict(cfg.input_params)
    density_penalty_config = density_penalty_dict(cfg)

    data = load_dataset_pickle(cfg.paths.dataset_pickle)
    full_dataset = data["full_dataset"]
    full_dataset.n_components = model_cfg.n_components
    full_dataset.min_depth = model_cfg.min_depth
    full_dataset.max_depth = model_cfg.max_depth
    full_dataset.input_params = input_params
    full_dataset.reload()

    train_dataset, val_dataset, _test_dataset = split_dataset(
        full_dataset,
        model_cfg.train_size,
        model_cfg.val_size,
        model_cfg.test_size,
    )
    train_loader = DataLoader(train_dataset, batch_size=model_cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=model_cfg.batch_size, shuffle=False)

    subset_indices = val_loader.dataset.indices
    full_dataset.calc_gem(subset_indices)

    input_dim = sum(val for val in input_params.values()) - 1 * input_params["sat"]
    device = get_device()
    weights = full_dataset.get_pca_weights()

    criterion = CombinedPCALoss(
        temp_pca=train_dataset.dataset,
        sal_pca=train_dataset.dataset,
        n_components=model_cfg.n_components,
        weights=weights,
        device=device,
        density_config=density_penalty_config,
    )

    set_seed(cfg.runtime.seed)
    model = PredictionModel(
        input_dim=input_dim,
        layers_config=list(model_cfg.layers_config),
        output_dim=model_cfg.n_components * 2,
        dropout_prob=model_cfg.dropout_prob,
    )
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=model_cfg.learning_rate)

    trajectory: list[dict[str, float]] = []
    train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device,
        GOLDEN_TRAIN_EPOCHS,
        GOLDEN_TRAIN_EPOCHS + 1,
        trajectory=trajectory,
    )
    return trajectory


def test_inverse_transform_roundtrip(fitted_pca_pair):
    pca_temp, pca_sal, temp_pcs, sal_pcs, n_components = fitted_pca_pair
    pcs = np.hstack([temp_pcs, sal_pcs])
    m = load_monolith()
    temp_profiles, sal_profiles = m.inverse_transform(pcs, pca_temp, pca_sal, n_components)
    assert temp_profiles.shape[0] == pca_temp.n_features_in_
    assert sal_profiles.shape[0] == pca_sal.n_features_in_
    recon_temp = pca_temp.inverse_transform(pcs[:, :n_components]).T
    assert np.nanmax(np.abs(temp_profiles - recon_temp)) < TOL


@pytest.mark.requires_unity
def test_dataset_getitem_golden(request):
    """Capture one __getitem__ output against real GoM pickle/dataset on HPC."""
    if not request.config.getoption("--run-unity", default=False):
        pytest.skip("HPC golden tests disabled; pass --run-unity on /unity host")

    golden_file = GOLDEN_DIR / "dataset_getitem_0.json"
    from nespreso.config import load_config

    cfg = load_config()
    if not Path(cfg.paths.dataset_pickle).exists():
        pytest.skip("dataset pickle not present")

    data = load_dataset_pickle(cfg.paths.dataset_pickle)
    ds = data["full_dataset"]
    inputs, labels = ds[0]
    payload = {
        "inputs": np.asarray(inputs).tolist(),
        "labels": np.asarray(labels).tolist(),
    }
    if not golden_file.exists():
        GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
        golden_file.write_text(json.dumps(payload))
        pytest.fail("Golden file created; re-run to verify")
    golden = json.loads(golden_file.read_text())
    assert np.allclose(payload["inputs"], golden["inputs"], rtol=0, atol=TOL)
    assert np.allclose(payload["labels"], golden["labels"], rtol=0, atol=TOL)


@pytest.mark.requires_unity
def test_short_train_loss_trajectory(request):
    """Few-epoch train_model loss trajectory on real data (HPC only)."""
    if not request.config.getoption("--run-unity", default=False):
        pytest.skip("HPC golden tests disabled; pass --run-unity on /unity host")

    golden_file = GOLDEN_DIR / "train_loss_trajectory.json"
    pytest.importorskip("torch")

    cfg = load_config()
    if not Path(cfg.paths.dataset_pickle).exists():
        pytest.skip(f"dataset pickle not present: {cfg.paths.dataset_pickle}")

    trajectory = _capture_short_train_trajectory(cfg)
    payload = {
        "seed": cfg.runtime.seed,
        "epochs": GOLDEN_TRAIN_EPOCHS,
        "trajectory": trajectory,
    }

    if not golden_file.exists():
        GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
        golden_file.write_text(json.dumps(payload, indent=2))
        pytest.fail("Golden file created; re-run to verify")

    golden = json.loads(golden_file.read_text())
    assert golden["seed"] == payload["seed"]
    assert golden["epochs"] == payload["epochs"]
    assert len(golden["trajectory"]) == len(payload["trajectory"])
    for actual, expected in zip(payload["trajectory"], golden["trajectory"], strict=True):
        assert actual["epoch"] == expected["epoch"]
        assert np.isclose(actual["train_loss"], expected["train_loss"], rtol=0, atol=TOL)
        assert np.isclose(actual["val_loss"], expected["val_loss"], rtol=0, atol=TOL)
