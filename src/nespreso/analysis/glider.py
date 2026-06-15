"""Glider prediction helpers hoisted from the monolith __main__ block."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch


def get_glider_predictions(
    model: torch.nn.Module,
    loader,
    tensor: torch.Tensor,
    device: torch.device,
    inverse_transform: Callable[..., tuple[np.ndarray, np.ndarray]],
    max_depth: int = 1004,
    min_depth: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    tensor = tensor.to(device)

    # Get predictions
    model.eval()
    with torch.no_grad():
        gld_predictions_pcs = model(tensor)
        gld_predictions_pcs_cpu = gld_predictions_pcs.cpu().numpy()
        gld_predictions = inverse_transform(gld_predictions_pcs_cpu)

    # crop at max depth
    T_predictions = gld_predictions[0][min_depth : max_depth + 1, :]
    S_predictions = gld_predictions[1][min_depth : max_depth + 1, :]
    return T_predictions, S_predictions


def bin_data(data: np.ndarray, bin_size: int) -> np.ndarray:
    """
    Bin data vertically to a given bin size.

    Args:
    - data (np.array): Data to be binned.
    - bin_size (int): Size of each bin.

    Returns:
    - np.array: Binned data.
    """
    n_rows = data.shape[0] // bin_size
    binned_data = np.mean(data[: n_rows * bin_size].reshape(n_rows, bin_size, -1), axis=1)
    return binned_data
