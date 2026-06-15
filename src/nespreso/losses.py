"""Training objectives for PCA-space profile prediction."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn

from nespreso.data.pca import torch_reconstruct_profile, torch_reconstruct_profiles
from nespreso.determinism import get_device
from nespreso.models.density import DensityConstraint

DEVICE = get_device()


class WeightedMSELoss(nn.Module):
    """
    The code defines several loss functions for use in a PCA-based model, including a weighted MSE loss
    and a combined PCA loss.

    @param n_components The parameter `n_components` represents the number of principal components to
    consider in the PCA loss. It determines the dimensionality of the PCA space for both temperature and
    salinity profiles.
    @param device The "device" parameter in the code refers to the device on which the computations will
    be performed. It can be either "cuda" for GPU acceleration or "cpu" for CPU computation.
    @param weights The "weights" parameter is a list of weights that are used to assign different
    importance to each element in the loss calculation. These weights are used in the WeightedMSELoss
    class to multiply the squared differences between the predicted and true values. The weights are
    normalized so that they sum up to

    @return The `forward` method of the `CombinedPCALoss` class returns the combined loss, which is the
    sum of the PCA loss and the weighted MSE loss.
    """

    def __init__(self, weights, device):
        super(WeightedMSELoss, self).__init__()
        self.weights = torch.tensor(weights, dtype=torch.float32, device=device)

    def forward(self, input, target):
        squared_diff = (input - target) ** 2
        weighted_squared_diff = self.weights * squared_diff
        loss = weighted_squared_diff.mean()
        return loss


def genWeightedMSELoss(
    n_components: int,
    device: torch.device,
    weights: np.ndarray,
) -> WeightedMSELoss:
    # Normalizing weights so they sum up to 1
    normalized_weights = weights / np.sum(weights)
    return WeightedMSELoss(normalized_weights, device)


class PCALoss(nn.Module):  # min test loss ~ 13
    def __init__(self, temp_pca, sal_pca, n_components):
        super(PCALoss, self).__init__()
        self.n_components = n_components
        # Use true PCA components and means for proper reconstruction in profile space
        temp_components = torch.tensor(temp_pca.pca_temp.components_, dtype=torch.float32, device=DEVICE)
        sal_components = torch.tensor(sal_pca.pca_sal.components_, dtype=torch.float32, device=DEVICE)
        temp_mean = torch.tensor(temp_pca.pca_temp.mean_, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        sal_mean = torch.tensor(sal_pca.pca_sal.mean_, dtype=torch.float32, device=DEVICE).unsqueeze(0)

        # Register as buffers so they move with the module and are not trainable
        self.register_buffer("temp_components", temp_components)
        self.register_buffer("sal_components", sal_components)
        self.register_buffer("temp_mean", temp_mean)
        self.register_buffer("sal_mean", sal_mean)

    def inverse_transform(self, pcs, components, mean):
        return torch_reconstruct_profile(pcs, components, mean)

    def forward(self, pcs, targets):
        # Split the predicted and true pcs for temp and sal
        pred_temp_pcs, pred_sal_pcs = pcs[:, : self.n_components], pcs[:, self.n_components :]
        true_temp_pcs, true_sal_pcs = targets[:, : self.n_components], targets[:, self.n_components :]

        # Inverse transform the PCA components to get the profiles
        pred_temp_profiles = self.inverse_transform(pred_temp_pcs, self.temp_components, self.temp_mean)
        pred_sal_profiles = self.inverse_transform(pred_sal_pcs, self.sal_components, self.sal_mean)
        true_temp_profiles = self.inverse_transform(true_temp_pcs, self.temp_components, self.temp_mean)
        true_sal_profiles = self.inverse_transform(true_sal_pcs, self.sal_components, self.sal_mean)

        # Calculate the Avg Squared Error between the predicted and true profiles
        mse_temp = nn.functional.mse_loss(pred_temp_profiles, true_temp_profiles)
        mse_sal = nn.functional.mse_loss(pred_sal_profiles, true_sal_profiles)

        # Combine the MSE for temperature and salinity
        # Keep the original scaling but remove division by dataset size to avoid vanishing loss
        total_mse = mse_temp / (37.86) + mse_sal / (0.28)
        return total_mse


class CombinedPCALoss(nn.Module):
    def __init__(self, temp_pca, sal_pca, n_components, weights, device, density_config=None):
        super(CombinedPCALoss, self).__init__()
        self.pca_loss = PCALoss(temp_pca, sal_pca, n_components)
        self.weighted_mse_loss = genWeightedMSELoss(n_components, device, weights)

        temp_components = torch.tensor(temp_pca.pca_temp.components_, dtype=torch.float32, device=device)
        sal_components = torch.tensor(sal_pca.pca_sal.components_, dtype=torch.float32, device=device)
        temp_mean = torch.tensor(temp_pca.pca_temp.mean_, dtype=torch.float32, device=device).unsqueeze(0)
        sal_mean = torch.tensor(sal_pca.pca_sal.mean_, dtype=torch.float32, device=device).unsqueeze(0)

        self.register_buffer("temp_components", temp_components)
        self.register_buffer("sal_components", sal_components)
        self.register_buffer("temp_mean", temp_mean)
        self.register_buffer("sal_mean", sal_mean)
        self.n_components = n_components

        if density_config and density_config.get("enabled", False):
            self.density_helper = DensityConstraint(dataset=temp_pca, device=device, config=density_config)
        else:
            self.density_helper = None

    def _reconstruct_profiles(self, temp_pcs, sal_pcs):
        return torch_reconstruct_profiles(
            temp_pcs,
            sal_pcs,
            self.temp_components,
            self.sal_components,
            self.temp_mean,
            self.sal_mean,
        )

    def forward(self, pcs, targets, indices=None):
        # Calculate the PCA loss
        pca_loss = self.pca_loss(pcs, targets)

        # Calculate the weighted MSE loss
        weighted_mse_loss = self.weighted_mse_loss(pcs, targets)

        # Combine the losses - Choose scaling factor
        combined_loss = (
            pca_loss / 2.8294 + weighted_mse_loss / 0.0255
        ) / 2  # here I divide by the individual minimum loss, and divide by two

        if self.density_helper is not None and indices is not None:
            pred_temp_pcs = pcs[:, : self.n_components]
            pred_sal_pcs = pcs[:, self.n_components :]
            temp_profiles, sal_profiles = self._reconstruct_profiles(pred_temp_pcs, pred_sal_pcs)
            combined_loss = combined_loss + self.density_helper(temp_profiles, sal_profiles, indices)

        return combined_loss


def make_loss(
    *,
    temp_pca,
    sal_pca,
    n_components: int,
    weights,
    device,
    density_config: dict[str, Any] | None = None,
) -> CombinedPCALoss:
    """Build the training criterion (mirrors runner / monolith selection logic)."""
    return CombinedPCALoss(
        temp_pca=temp_pca,
        sal_pca=sal_pca,
        n_components=n_components,
        weights=weights,
        device=device,
        density_config=density_config,
    )
