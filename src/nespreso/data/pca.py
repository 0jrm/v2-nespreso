"""Canonical PCA reconstruction helpers for NeSPReSO."""

from __future__ import annotations

from typing import Any

import torch


def sklearn_inverse_transform_pcs(
    pcs: Any,
    pca_temp: Any,
    pca_sal: Any,
    n_components: int,
) -> tuple[Any, Any]:
    """
    Inverse the PCA transformation.

    Args:
    - pcs (numpy.ndarray): Concatenated PCA components for temperature and salinity.
    - pca_temp, pca_sal: PCA models for temperature and salinity respectively.
    - n_components (int): Number of PCA components.

    Returns:
    - tuple: Inversed temperature and salinity profiles.
    """
    temp_profiles = pca_temp.inverse_transform(pcs[:, :n_components]).T
    sal_profiles = pca_sal.inverse_transform(pcs[:, n_components:]).T
    return temp_profiles, sal_profiles


def torch_reconstruct_profile(
    pcs: torch.Tensor,
    components: torch.Tensor,
    mean: torch.Tensor,
) -> torch.Tensor:
    # Reconstruct profiles: (batch, n_components) @ (n_components, n_features) + (1, n_features)
    return pcs @ components + mean


def torch_reconstruct_profiles(
    temp_pcs: torch.Tensor,
    sal_pcs: torch.Tensor,
    temp_components: torch.Tensor,
    sal_components: torch.Tensor,
    temp_mean: torch.Tensor,
    sal_mean: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    temp_profiles = torch_reconstruct_profile(temp_pcs, temp_components, temp_mean)
    sal_profiles = torch_reconstruct_profile(sal_pcs, sal_components, sal_mean)
    return temp_profiles, sal_profiles
