"""Residual profile statistics hoisted from the monolith __main__ block."""

from __future__ import annotations

import numpy as np


def compute_profile_residual(predicted: np.ndarray, original: np.ndarray) -> np.ndarray:
    return predicted - original


def compute_depth_rmse_bias(residual: np.ndarray, axis: int = 1) -> tuple[np.ndarray, np.ndarray]:
    se = residual**2
    avg_rmse = np.sqrt(np.mean(se, axis=axis))
    avg_bias = np.mean(residual, axis=axis)
    return avg_rmse, avg_bias
