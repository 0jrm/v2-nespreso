"""
Pin Phase 5 model and loss forward passes before extraction from the monolith.

Golden values were captured with ``np.random.seed(42)`` and ``torch.manual_seed(42)``
on the synthetic profile fixture (depth=26, n_profiles=8, n_components=3).
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tests.monolith_loader import load_monolith

TOL = 1e-6
GOLDEN_DIR = Path(__file__).parent / "golden"
GOLDEN_FILE = GOLDEN_DIR / "combined_pca_loss_synthetic.json"


@pytest.fixture
def seeded_pca_pair():
    """Deterministic PCA pair for loss/model pin tests (seed=42)."""
    from sklearn.decomposition import PCA

    np.random.seed(42)
    depth = np.linspace(0, 500, 26)
    n_profiles = 8
    temp = 20.0 - 0.02 * depth[:, None] + 0.1 * np.random.randn(len(depth), n_profiles)
    sal = 36.0 + 0.001 * depth[:, None] + 0.01 * np.random.randn(len(depth), n_profiles)
    n_components = 3
    pca_temp = PCA(n_components=n_components)
    pca_sal = PCA(n_components=n_components)
    temp_pcs = pca_temp.fit_transform(temp.T)
    sal_pcs = pca_sal.fit_transform(sal.T)
    return pca_temp, pca_sal, temp_pcs, sal_pcs, n_components


def _synthetic_loss_batch(pca_pair):
    """Fixed PCS batch shared by loss pin tests (seed=42 at capture time)."""
    pca_temp, pca_sal, temp_pcs, sal_pcs, n_components = pca_pair
    pcs_np = np.hstack([temp_pcs, sal_pcs])[:4].astype(np.float32)
    pred_np = pcs_np + np.array([[0.1, -0.05, 0.02, 0.03, -0.01, 0.04]], dtype=np.float32)
    return pca_temp, pca_sal, n_components, pred_np, pcs_np


def _load_golden():
    return json.loads(GOLDEN_FILE.read_text())


def test_combined_pca_loss_forward_golden(seeded_pca_pair):
    """Pin CombinedPCALoss scalar and reconstructed profile heads."""
    pca_temp, pca_sal, n_components, pred_np, target_np = _synthetic_loss_batch(seeded_pca_pair)
    m = load_monolith()
    holder = SimpleNamespace(pca_temp=pca_temp, pca_sal=pca_sal)
    weights = np.ones(2 * n_components, dtype=np.float64)
    device = m.DEVICE

    combined = m.CombinedPCALoss(
        temp_pca=holder,
        sal_pca=holder,
        n_components=n_components,
        weights=weights,
        device=device,
        density_config=None,
    )
    combined.eval()

    pcs = torch.tensor(pred_np, dtype=torch.float32, device=device)
    targets = torch.tensor(target_np, dtype=torch.float32, device=device)

    with torch.no_grad():
        combined_loss = combined(pcs, targets)
        pca_loss = combined.pca_loss(pcs, targets)
        weighted_mse = combined.weighted_mse_loss(pcs, targets)
        recon_t, recon_s = combined._reconstruct_profiles(
            pcs[:, :n_components],
            pcs[:, n_components:],
        )

    golden = _load_golden()
    assert np.isclose(combined_loss.item(), golden["combined_loss"], rtol=0, atol=TOL)
    assert np.isclose(pca_loss.item(), golden["pca_loss"], rtol=0, atol=TOL)
    assert np.isclose(weighted_mse.item(), golden["weighted_mse_loss"], rtol=0, atol=TOL)
    assert np.allclose(recon_t[0, :5].cpu().numpy(), golden["recon_temp_head"], rtol=0, atol=TOL)
    assert np.allclose(recon_s[0, :5].cpu().numpy(), golden["recon_sal_head"], rtol=0, atol=TOL)


def test_prediction_model_forward_golden():
    """Pin PredictionModel forward on a fixed input (dropout disabled)."""
    torch.manual_seed(42)
    m = load_monolith()
    device = m.DEVICE

    model = m.PredictionModel(input_dim=5, layers_config=[16, 8], output_dim=6, dropout_prob=0.0)
    model.eval()
    model.to(device)

    x = torch.tensor([[0.5, -0.3, 1.2, 0.0, -0.8]], dtype=torch.float32, device=device)
    with torch.no_grad():
        out = model(x)

    golden = _load_golden()
    assert np.allclose(out[0, :6].cpu().numpy(), golden["prediction_head"], rtol=0, atol=TOL)


def test_make_loss_matches_manual_combined(seeded_pca_pair):
    """make_loss factory must build the same criterion as manual CombinedPCALoss."""
    from nespreso.losses import make_loss

    pca_temp, pca_sal, n_components, pred_np, target_np = _synthetic_loss_batch(seeded_pca_pair)
    m = load_monolith()
    holder = SimpleNamespace(pca_temp=pca_temp, pca_sal=pca_sal)
    weights = np.ones(2 * n_components, dtype=np.float64)
    device = m.DEVICE

    manual = m.CombinedPCALoss(
        temp_pca=holder,
        sal_pca=holder,
        n_components=n_components,
        weights=weights,
        device=device,
        density_config=None,
    )
    factory = make_loss(
        temp_pca=holder,
        sal_pca=holder,
        n_components=n_components,
        weights=weights,
        device=device,
        density_config=None,
    )

    pcs = torch.tensor(pred_np, dtype=torch.float32, device=device)
    targets = torch.tensor(target_np, dtype=torch.float32, device=device)
    manual.eval()
    factory.eval()

    with torch.no_grad():
        manual_loss = manual(pcs, targets)
        factory_loss = factory(pcs, targets)

    assert np.isclose(manual_loss.item(), factory_loss.item(), rtol=0, atol=TOL)
