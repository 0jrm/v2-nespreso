"""PCA regression baseline helpers hoisted from the monolith __main__ block."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.preprocessing import PolynomialFeatures

from nespreso.determinism import get_device

DEVICE = get_device()


def prepare_features(inputs_array, max_degree=3):
    """
    Prepare the feature matrix for regression by including polynomial terms.

    Args:
    - inputs_array (numpy.ndarray): Array of input features, shape (n_samples, n_features).
    - max_degree (int): Maximum degree of polynomial features.

    Returns:
    - X (numpy.ndarray): Feature matrix of shape (n_samples, n_features_expanded).
    """
    # Generate polynomial features up to the specified degree
    poly = PolynomialFeatures(degree=max_degree, include_bias=False)
    X = poly.fit_transform(inputs_array)
    return X


def fit_pcs_regression_exact_gpu(X, pcs):
    """
    Fit regression models to predict principal component scores from features using exact least squares on GPU.

    Args:
    - X (numpy.ndarray): Feature matrix, shape (n_samples, n_features_expanded).
    - pcs (numpy.ndarray): Principal component scores, shape (n_samples, n_components).

    Returns:
    - beta (torch.Tensor): Coefficient matrix, shape (n_features_expanded, n_components).
    """
    # Convert data to torch tensors and move to GPU
    X_tensor = torch.tensor(X, dtype=torch.float32).to(DEVICE)
    pcs_tensor = torch.tensor(pcs, dtype=torch.float32).to(DEVICE)

    # Compute the pseudoinverse of X
    # Note: For large matrices, torch.linalg.lstsq may be more efficient
    X_pinv = torch.pinverse(X_tensor)

    print(f"{X_pinv.shape=}")

    # Compute the coefficients (beta) analytically
    beta = X_pinv @ pcs_tensor

    return beta


def predict_pcs_exact_gpu(beta, X_new):
    """
    Predict principal component scores using the exact coefficients on GPU.

    Args:
    - beta (torch.Tensor): Coefficient matrix, shape (n_features_expanded, n_components).
    - X_new (numpy.ndarray): New feature matrix, shape (n_samples_new, n_features_expanded).

    Returns:
    - pcs_pred (numpy.ndarray): Predicted principal component scores, shape (n_samples_new, n_components).
    """
    with torch.no_grad():
        X_new_tensor = torch.tensor(X_new, dtype=torch.float32).to(DEVICE)
        pcs_pred_tensor = X_new_tensor @ beta
        pcs_pred = pcs_pred_tensor.cpu().numpy()
    return pcs_pred
