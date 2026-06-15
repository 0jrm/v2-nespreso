"""
Pin Phase 6 inference helpers before extraction from the monolith.

Golden values use ``torch.manual_seed(42)`` with a fixed ``TensorDataset`` batch.
``predict_with_numpy`` is pinned on CPU because the monolith only moves tensors
to GPU when ``device == "cuda"`` exactly (not ``cuda:0``).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from tests.monolith_loader import load_monolith

TOL = 1e-6
GOLDEN_FILE = Path(__file__).parent / "golden" / "inference_synthetic.json"


def _synthetic_loader(device):
    torch.manual_seed(42)
    m = load_monolith()
    inputs = torch.tensor(
        [[1.0, 0.5, -0.2], [0.3, -0.1, 0.8], [0.0, 0.2, 0.4], [-0.5, 0.9, -0.3]],
        dtype=torch.float32,
    )
    labels = torch.zeros(4, 4)
    loader = DataLoader(TensorDataset(inputs, labels), batch_size=2)
    model = m.PredictionModel(input_dim=3, layers_config=[8], output_dim=4, dropout_prob=0.0)
    model.eval()
    model.to(device)
    return m, model, loader, inputs


def _load_golden():
    return json.loads(GOLDEN_FILE.read_text())


def test_get_predictions_golden():
    m = load_monolith()
    device = m.DEVICE
    _, model, loader, _ = _synthetic_loader(device)

    preds = m.get_predictions(model, loader, device)
    golden = _load_golden()

    assert list(preds.shape) == golden["predictions_shape"]
    assert np.allclose(preds[0], golden["predictions_head"], rtol=0, atol=TOL)
    assert np.allclose(preds[1], golden["predictions_row1"], rtol=0, atol=TOL)


def test_get_inputs_golden():
    m = load_monolith()
    _, _, loader, _ = _synthetic_loader(m.DEVICE)

    gathered = m.get_inputs(loader, m.DEVICE)
    golden = _load_golden()

    assert list(gathered.shape) == golden["inputs_shape"]
    assert np.allclose(gathered[0], golden["inputs_head"], rtol=0, atol=TOL)


def test_predict_with_numpy_golden():
    m = load_monolith()
    _, model, _, inputs = _synthetic_loader(m.DEVICE)

    model_cpu = m.PredictionModel(input_dim=3, layers_config=[8], output_dim=4, dropout_prob=0.0)
    model_cpu.load_state_dict(model.state_dict())
    model_cpu.eval()

    single = m.predict_with_numpy(model_cpu, inputs[:1].numpy(), device="cpu")
    golden = _load_golden()
    assert np.allclose(single[0], golden["predict_numpy_head"], rtol=0, atol=TOL)


def test_get_predictions_torchscript_matches_get_predictions():
    """TorchScript helper is a verbatim duplicate of get_predictions."""
    m = load_monolith()
    if not hasattr(m, "get_predictions_torchscript"):
        pytest.skip("get_predictions_torchscript not yet extracted from __main__")

    device = m.DEVICE
    _, model, loader, _ = _synthetic_loader(device)

    preds = m.get_predictions(model, loader, device)
    ts_preds = m.get_predictions_torchscript(model, loader, device, input_params_check={})
    assert np.allclose(preds, ts_preds, rtol=0, atol=TOL)
