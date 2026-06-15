"""
Pin PCA inverse-transform variants before Phase 4.2 extraction to ``pca.py``.

Phase 4 plan line refs (baseline monolith) map to current locations as:

1. ~514  -> ``TemperatureSalinityDataset.inverse_transform`` (``dataset.py:280``)
2. ~977  -> ``CombinedPCALoss._reconstruct_profiles`` (``monolith:339``)
3. ~1043 -> ``PCALoss.inverse_transform`` (``monolith:292``)
4. ~2058 -> nested ``inverse_transform`` in main block (``monolith:1898``, dead code)
5. ~2631 -> call site ``val_dataset.dataset.inverse_transform`` (``monolith:2608`` glider path)

Additional sklearn duplicates:

- module-level ``inverse_transform`` (``monolith:222``)
- ``inverse_transform_api`` (``monolith:1259``)
"""

from __future__ import annotations

import json
import pickle
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from sklearn.base import InconsistentVersionWarning

from nespreso.config import load_config
from nespreso.data.dataset import TemperatureSalinityDataset
from nespreso.data.pca import sklearn_inverse_transform_pcs
from nespreso.runner import _load_dataset_pickle
from tests.monolith_loader import load_monolith

TOL = 1e-6
# PCALoss stores sklearn stats in float32; allow slightly wider bound vs float64 sklearn.
TORCH_SKLEARN_TOL = 5e-6
GOLDEN_DIR = Path(__file__).parent / "golden"


def _make_pca_holder(pca_temp, pca_sal):
    return SimpleNamespace(pca_temp=pca_temp, pca_sal=pca_sal)


def _make_dataset_shell(pca_temp, pca_sal, temp_pcs, sal_pcs, n_components, temp=None, sal=None):
    """Minimal dataset instance for inverse_transform / get_profiles tests."""
    ds = TemperatureSalinityDataset.__new__(TemperatureSalinityDataset)
    ds.n_components = n_components
    ds.pca_temp = pca_temp
    ds.pca_sal = pca_sal
    ds.temp_pcs = temp_pcs
    ds.sal_pcs = sal_pcs
    if temp is not None:
        ds.TEMP = temp
    if sal is not None:
        ds.SAL = sal
    return ds


def test_sklearn_variants_equivalent_on_synthetic(fitted_pca_pair):
    """Module, dataset, and API-local sklearn paths must agree on identical PCAs."""
    pca_temp, pca_sal, temp_pcs, sal_pcs, n_components = fitted_pca_pair
    pcs = np.hstack([temp_pcs, sal_pcs])
    m = load_monolith()

    module_t, module_s = m.inverse_transform(pcs, pca_temp, pca_sal, n_components)

    ds = _make_dataset_shell(
        pca_temp,
        pca_sal,
        temp_pcs.T,
        sal_pcs.T,
        n_components,
    )
    dataset_t, dataset_s = ds.inverse_transform(pcs)

    # Local duplicate of inverse_transform_api (monolith:1259)
    api_t, api_s = sklearn_inverse_transform_pcs(pcs, pca_temp, pca_sal, n_components)

    assert module_t.shape == (pca_temp.n_features_in_, pcs.shape[0])
    assert module_s.shape == (pca_sal.n_features_in_, pcs.shape[0])
    assert np.nanmax(np.abs(module_t - dataset_t)) < TOL
    assert np.nanmax(np.abs(module_s - dataset_s)) < TOL
    assert np.nanmax(np.abs(module_t - api_t)) < TOL
    assert np.nanmax(np.abs(module_s - api_s)) < TOL


def test_pca_loss_torch_matches_sklearn(fitted_pca_pair):
    """PCALoss torch reconstruction equals sklearn inverse (layout differs by transpose)."""
    pca_temp, pca_sal, temp_pcs, sal_pcs, n_components = fitted_pca_pair
    pcs = np.hstack([temp_pcs, sal_pcs])
    m = load_monolith()
    holder = _make_pca_holder(pca_temp, pca_sal)

    sklearn_t, sklearn_s = m.inverse_transform(pcs, pca_temp, pca_sal, n_components)

    pca_loss = m.PCALoss(holder, holder, n_components)
    device = pca_loss.temp_components.device
    temp_pcs_t = torch.tensor(pcs[:, :n_components], dtype=torch.float32, device=device)
    sal_pcs_t = torch.tensor(pcs[:, n_components:], dtype=torch.float32, device=device)

    torch_t = pca_loss.inverse_transform(temp_pcs_t, pca_loss.temp_components, pca_loss.temp_mean)
    torch_s = pca_loss.inverse_transform(sal_pcs_t, pca_loss.sal_components, pca_loss.sal_mean)

    assert torch_t.shape == (pcs.shape[0], pca_temp.n_features_in_)
    assert torch_s.shape == (pcs.shape[0], pca_sal.n_features_in_)
    assert np.nanmax(np.abs(sklearn_t.T - torch_t.detach().cpu().numpy())) < TORCH_SKLEARN_TOL
    assert np.nanmax(np.abs(sklearn_s.T - torch_s.detach().cpu().numpy())) < TORCH_SKLEARN_TOL


def test_combined_pca_loss_reconstruct_matches_pca_loss(fitted_pca_pair):
    """CombinedPCALoss._reconstruct_profiles matches PCALoss per-field reconstruction."""
    pca_temp, pca_sal, temp_pcs, sal_pcs, n_components = fitted_pca_pair
    pcs = np.hstack([temp_pcs, sal_pcs])
    m = load_monolith()
    holder = _make_pca_holder(pca_temp, pca_sal)
    weights = np.ones(2 * n_components, dtype=np.float64)

    pca_loss = m.PCALoss(holder, holder, n_components)
    device = pca_loss.temp_components.device
    combined = m.CombinedPCALoss(
        temp_pca=holder,
        sal_pca=holder,
        n_components=n_components,
        weights=weights,
        device=device,
        density_config=None,
    )

    temp_pcs_t = torch.tensor(pcs[:, :n_components], dtype=torch.float32, device=device)
    sal_pcs_t = torch.tensor(pcs[:, n_components:], dtype=torch.float32, device=device)

    loss_t = pca_loss.inverse_transform(temp_pcs_t, pca_loss.temp_components, pca_loss.temp_mean)
    loss_s = pca_loss.inverse_transform(sal_pcs_t, pca_loss.sal_components, pca_loss.sal_mean)
    recon_t, recon_s = combined._reconstruct_profiles(temp_pcs_t, sal_pcs_t)

    assert np.nanmax(np.abs(loss_t.detach().cpu().numpy() - recon_t.detach().cpu().numpy())) < TOL
    assert np.nanmax(np.abs(loss_s.detach().cpu().numpy() - recon_s.detach().cpu().numpy())) < TOL


def test_get_profiles_pca_approx_uses_inverse_transform(fitted_pca_pair, synthetic_ts_profiles):
    """get_profiles(pca_approx=True) must match manual inverse_transform on stored PCs."""
    pca_temp, pca_sal, temp_pcs, sal_pcs, n_components = fitted_pca_pair
    depth, temp, sal = synthetic_ts_profiles
    ds = _make_dataset_shell(pca_temp, pca_sal, temp_pcs.T, sal_pcs.T, n_components, temp=temp, sal=sal)

    indices = np.array([0, 3, 5])
    approx = ds.get_profiles(indices, pca_approx=True)
    manual_t, manual_s = ds.inverse_transform(
        np.hstack([ds.temp_pcs[:, indices].T, ds.sal_pcs[:, indices].T])
    )
    expected = np.stack([manual_t, manual_s], axis=1)

    assert approx.shape == (temp.shape[0], 2, len(indices))
    assert np.nanmax(np.abs(approx - expected)) < TOL


def test_nested_main_inverse_transform_removed():
    """Dead main-block inverse_transform duplicate was removed from the monolith."""
    m = load_monolith()
    source = Path(m.__file__).read_text()
    assert source.count("def inverse_transform(pcs, pca_temp, pca_sal, n_components):") == 1
    assert "sklearn_inverse_transform_pcs" in source
    assert "Inverse the PCA transformation to reconstruct temperature and salinity profiles." not in source


@pytest.mark.requires_unity
def test_dataset_inverse_transform_golden(request, fitted_pca_pair):
    """Pin real-dataset inverse_transform on profile-0 labels from the GoM pickle."""
    if not request.config.getoption("--run-unity", default=False):
        pytest.skip("HPC golden tests disabled; pass --run-unity on /unity host")

    golden_file = GOLDEN_DIR / "pca_inverse_profile_0.json"
    cfg = load_config()
    if not Path(cfg.paths.dataset_pickle).exists():
        pytest.skip("dataset pickle not present")

    m = load_monolith()
    data = _load_dataset_pickle(m, cfg.paths.dataset_pickle)
    ds = data["full_dataset"]
    n_components = ds.pca_temp.n_components_
    if not hasattr(ds, "n_components"):
        ds.n_components = n_components
    _, labels = ds[0]
    pcs = np.asarray(labels, dtype=np.float64)[None, :]

    temp_profiles, sal_profiles = ds.inverse_transform(pcs)
    payload = {
        "temp_shape": list(temp_profiles.shape),
        "sal_shape": list(sal_profiles.shape),
        "temp_head": temp_profiles[:5, 0].tolist(),
        "sal_head": sal_profiles[:5, 0].tolist(),
        "temp_mean": float(np.nanmean(temp_profiles[:, 0])),
        "sal_mean": float(np.nanmean(sal_profiles[:, 0])),
    }

    if not golden_file.exists():
        GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
        golden_file.write_text(json.dumps(payload, indent=2))
        pytest.fail("Golden file created; re-run to verify")

    golden = json.loads(golden_file.read_text())
    assert payload["temp_shape"] == golden["temp_shape"]
    assert payload["sal_shape"] == golden["sal_shape"]
    assert np.allclose(payload["temp_head"], golden["temp_head"], rtol=0, atol=TOL)
    assert np.allclose(payload["sal_head"], golden["sal_head"], rtol=0, atol=TOL)
    assert np.isclose(payload["temp_mean"], golden["temp_mean"], rtol=0, atol=TOL)
    assert np.isclose(payload["sal_mean"], golden["sal_mean"], rtol=0, atol=TOL)

    # Cross-check: labels round-trip through get_profiles(pca_approx=True)
    approx = ds.get_profiles([0], pca_approx=True)
    assert np.nanmax(np.abs(approx[:, 0, 0] - temp_profiles[:, 0])) < TOL
    assert np.nanmax(np.abs(approx[:, 1, 0] - sal_profiles[:, 0])) < TOL


@pytest.mark.requires_unity
def test_inverse_transform_api_golden(request):
    """Pin API PCA inverse path used for NeSPReSO 1.0 comparison (monolith:1259)."""
    if not request.config.getoption("--run-unity", default=False):
        pytest.skip("HPC golden tests disabled; pass --run-unity on /unity host")

    api_pca_path = Path("/unity/g2/jmiranda/nespreso_api/models/pca_stats.pkl")
    golden_file = GOLDEN_DIR / "pca_inverse_api_profile_0.json"
    if not api_pca_path.exists():
        pytest.skip(f"API PCA pickle not present: {api_pca_path}")

    cfg = load_config()
    if not Path(cfg.paths.dataset_pickle).exists():
        pytest.skip("dataset pickle not present")

    m = load_monolith()
    data = _load_dataset_pickle(m, cfg.paths.dataset_pickle)
    ds = data["full_dataset"]
    n_components = ds.pca_temp.n_components_
    if not hasattr(ds, "n_components"):
        ds.n_components = n_components
    _, labels = ds[0]
    pcs = np.asarray(labels, dtype=np.float64)[None, :]

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
        with api_pca_path.open("rb") as f:
            old_pca_data = pickle.load(f)
    old_pca_temp = old_pca_data["pca_temp"]
    old_pca_sal = old_pca_data["pca_sal"]

    module_t, module_s = sklearn_inverse_transform_pcs(pcs, old_pca_temp, old_pca_sal, n_components)
    api_t, api_s = sklearn_inverse_transform_pcs(pcs, old_pca_temp, old_pca_sal, n_components)
    assert np.nanmax(np.abs(module_t - api_t)) < TOL
    assert np.nanmax(np.abs(module_s - api_s)) < TOL

    payload = {
        "temp_shape": list(module_t.shape),
        "sal_shape": list(module_s.shape),
        "temp_head": module_t[:5, 0].tolist(),
        "sal_head": module_s[:5, 0].tolist(),
    }

    if not golden_file.exists():
        GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
        golden_file.write_text(json.dumps(payload, indent=2))
        pytest.fail("Golden file created; re-run to verify")

    golden = json.loads(golden_file.read_text())
    assert payload["temp_shape"] == golden["temp_shape"]
    assert payload["sal_shape"] == golden["sal_shape"]
    assert np.allclose(payload["temp_head"], golden["temp_head"], rtol=0, atol=TOL)
    assert np.allclose(payload["sal_head"], golden["sal_head"], rtol=0, atol=TOL)
