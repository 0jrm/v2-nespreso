"""Centralized seeding and device selection."""

from __future__ import annotations

import random

import numpy as np
import torch

DEFAULT_SEED = 42


def set_seed(seed: int = DEFAULT_SEED) -> None:
    """Set random seeds for torch, numpy, and the standard library."""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def get_device() -> torch.device:
    """Return CUDA device when available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
