"""Residual profile statistics hoisted from the monolith __main__ block."""

from __future__ import annotations

import numpy as np


def compute_profile_residual(predicted, original):
    return predicted - original


def compute_depth_rmse_bias(residual, axis=1):
    se = residual**2
    avg_rmse = np.sqrt(np.mean(se, axis=axis))
    avg_bias = np.mean(residual, axis=axis)
    return avg_rmse, avg_bias
