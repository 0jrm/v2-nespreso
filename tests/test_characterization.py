"""
Characterization tests for refactor safety.

Golden-output tests (marked ``requires_unity``) must be captured on HPC where
``/unity`` data and GPU are available. Run:

    pytest tests/test_characterization.py -m requires_unity --run-unity

to record or verify golden files under ``tests/golden/``.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from nespreso.metrics import bias, mad, rmse

GOLDEN_DIR = Path(__file__).parent / "golden"
TOL = 1e-6


def _load_monolith():
    root = Path(__file__).resolve().parents[1]
    path = root / "singleFileModel_SAT_stats4verticalProj_meeting20260203.py"
    spec = importlib.util.spec_from_file_location("nespreso_monolith", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["nespreso_monolith"] = module
    spec.loader.exec_module(module)
    return module


def test_inverse_transform_roundtrip(fitted_pca_pair):
    pca_temp, pca_sal, temp_pcs, sal_pcs, n_components = fitted_pca_pair
    pcs = np.hstack([temp_pcs, sal_pcs])
    m = _load_monolith()
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
    m = _load_monolith()
    from nespreso.config import load_config

    cfg = load_config()
    if not Path(cfg.paths.dataset_pickle).exists():
        pytest.skip("dataset pickle not present")

    import pickle

    with open(cfg.paths.dataset_pickle, "rb") as fh:
        data = pickle.load(fh)
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
    # TODO(verify-numerics): implement after first HPC golden capture
    if not golden_file.exists():
        pytest.skip("Golden train trajectory not yet captured on HPC")


def pytest_addoption(parser):
    parser.addoption("--run-unity", action="store_true", help="Run /unity characterization tests")


def pytest_configure(config):
    config.addinivalue_line("markers", "requires_unity: needs /unity data on HPC")
