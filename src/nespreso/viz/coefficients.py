"""Regression coefficient heatmap visualization extracted from the monolith."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def plot_coefficients_heatmap(beta, feature_names, title, normalize=True, threshold=1e-4):
    """
    Plots a heatmap of regression coefficients with optional per-PC normalization and thresholding.

    Args:
    - beta (torch.Tensor): Coefficient matrix, shape (n_features, n_components).
    - feature_names (list): List of feature names corresponding to the rows of beta.
    - title (str): Title of the heatmap.
    - normalize (bool): Whether to apply per-PC Max-Abs normalization. Default is True.
    - threshold (float): Threshold below which coefficients are set to zero. Default is 1e-4.
    """
    # Convert to NumPy
    beta_np = beta.cpu().numpy()

    if normalize:
        # Apply Max-Abs normalization per PC (column-wise)
        max_abs_per_pc = np.max(np.abs(beta_np), axis=0)  # Shape: (n_components,)

        # Handle cases where the maximum is zero to avoid division by zero
        max_abs_per_pc[max_abs_per_pc == 0] = 1

        # Normalize each column (PC) by its maximum absolute value
        beta_np = beta_np / max_abs_per_pc  # Broadcasting division

    # Thresholding: Set coefficients with abs < threshold to zero
    beta_np_thresholded = np.where(np.abs(beta_np) < threshold, 0, beta_np)

    # Create a DataFrame for seaborn
    df_beta = pd.DataFrame(
        beta_np_thresholded,
        index=feature_names,
        columns=[f"PC{i + 1}" for i in range(beta_np_thresholded.shape[1])],
    )

    plt.figure(figsize=(20, 10))
    sns.heatmap(
        df_beta,
        cmap="coolwarm",
        center=0,
        annot=False,
        fmt=".2f",
        vmin=-1,  # Since normalization scales coefficients between -1 and 1
        vmax=1,
        linewidths=0.5,
        linecolor="gray",
    )
    plt.title(title, fontsize=16)
    plt.xlabel("Principal Components", fontsize=14)
    plt.ylabel("Input Features", fontsize=14)
    plt.tight_layout()
    plt.show()

