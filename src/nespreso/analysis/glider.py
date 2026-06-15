"""Glider prediction helpers hoisted from the monolith __main__ block."""

from __future__ import annotations

import numpy as np
import torch


def get_glider_predictions(model, loader, tensor, device, inverse_transform, max_depth=1004, min_depth=0):
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


def bin_data(data, bin_size):
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
