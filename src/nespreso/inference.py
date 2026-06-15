"""Inference helpers extracted from the monolith."""

from __future__ import annotations

import glob
import os

import numpy as np
import torch

from nespreso.determinism import get_device
from nespreso.models.mlp import PredictionModel

DEVICE = get_device()


def get_predictions(model, dataloader, device):
    """
    Get model's predictions on the provided data with CUDA support.

    Parameters:
    - model: the PyTorch model.
    - dataloader: the DataLoader for the data.
    - device: device to which data and model should be moved before getting predictions.

    Returns:
    - predictions: model's predictions.
    """
    model.to(device)
    model.eval()
    predictions = []

    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                inputs = batch[0]
            else:
                inputs = batch
            inputs = inputs.to(device)
            outputs = model(inputs)
            predictions.extend(outputs.cpu().numpy())

    return np.array(predictions)


def get_inputs(dataloader, device):
    """
    Get inputs from the provided dataloader with CUDA support.

    Parameters:
    - dataloader: the DataLoader for the data.
    - device: device to which data should be moved.

    Returns:
    - all_inputs: list of inputs from the dataloader.
    """
    all_inputs = []

    for batch in dataloader:
        if isinstance(batch, (list, tuple)):
            inputs = batch[0]
        else:
            inputs = batch
        inputs = inputs.to(device)
        all_inputs.extend(inputs.cpu().numpy())

    return np.array(all_inputs)


def predict_with_numpy(model, numpy_input, device=DEVICE):
    # Convert numpy array to tensor
    tensor_input = torch.tensor(numpy_input, dtype=torch.float32)

    # Check if CUDA is available and move tensor to the appropriate device
    if device == "cuda" and torch.cuda.is_available():
        tensor_input = tensor_input.cuda()
        model = model.cuda()

    # Make sure the model is in evaluation mode
    model.eval()

    # Make predictions
    with torch.no_grad():
        predictions = model(tensor_input)

    # Convert predictions back to numpy (if on GPU, move to CPU first)
    numpy_predictions = predictions.cpu().numpy()

    return numpy_predictions


def get_predictions_torchscript(model, dataloader, device, input_params_check):
    """Get predictions from TorchScript model."""
    model.to(device)
    model.eval()
    predictions = []

    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                inputs = batch[0]
            else:
                inputs = batch
            inputs = inputs.to(device)
            outputs = model(inputs)
            predictions.extend(outputs.cpu().numpy())

    return np.array(predictions)


def load_all_models(models_dir, device, input_dim, layers_config, n_components, dropout_prob):
    model_paths = sorted(glob.glob(os.path.join(models_dir, "model_Test Loss: *.pth")))
    models = []
    for model_path in model_paths:
        print(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path, map_location=device)

        # Initialize the model architecture
        model = PredictionModel(
            input_dim=input_dim, layers_config=layers_config, output_dim=n_components * 2, dropout_prob=dropout_prob
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()  # Set to evaluation mode
        models.append(model)
    return models
