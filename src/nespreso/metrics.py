"""Profile error metrics (extracted from monolith lines ~90-100)."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def rmse(predictions: ArrayLike, targets: ArrayLike) -> float:
    """Root mean squared error, NaN-safe."""
    predictions = np.asarray(predictions)
    targets = np.asarray(targets)
    return float(np.sqrt(np.nanmean(((predictions - targets) ** 2))))


def bias(predictions: ArrayLike, targets: ArrayLike) -> float:
    """Mean bias (prediction minus target), NaN-safe."""
    predictions = np.asarray(predictions)
    targets = np.asarray(targets)
    return float(np.nanmean((predictions - targets)))


def mad(x: ArrayLike) -> NDArray[np.floating]:
    """Median absolute deviation along axis 1."""
    x = np.asarray(x)
    return np.nanmedian(
        np.absolute(x - np.nanmedian(x, axis=1, keepdims=True)),
        axis=1,
    )
