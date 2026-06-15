import numpy as np
import torch


def prepare_inputs(time, lat, lon, sss, sst, ssh, input_params):
    """
    Transforms the individual data arrays into the format expected by the model.

    Args:
    - time (array): Time data.
    - lat (array): Latitude data.
    - lon (array): Longitude data.
    - sss (array): Sea Surface Salinity data.
    - sst (array): Sea Surface Temperature data.
    - ssh (array): Sea Surface Height data.
    - input_params (dict): Dictionary indicating which features to include.

    Returns:
    - torch.Tensor: Tensor of transformed input data.
    """
    try:
        num_samples = len(time)  # Assuming all arrays have the same length
    except:
        num_samples = 1

    inputs = []

    # Iterate over each sample and create input features
    for i in range(num_samples):
        sample_inputs = []

        if input_params.get("timecos", False):
            sample_inputs.append(np.cos(2 * np.pi * (time[i] % 365) / 365))

        if input_params.get("timesin", False):
            sample_inputs.append(np.sin(2 * np.pi * (time[i] % 365) / 365))

        if input_params.get("latcos", False):
            sample_inputs.append(np.cos(2 * np.pi * (lat[i] / 180)))

        if input_params.get("latsin", False):
            sample_inputs.append(np.sin(2 * np.pi * (lat[i] / 180)))

        if input_params.get("loncos", False):
            sample_inputs.append(np.cos(2 * np.pi * (lon[i] / 360)))

        if input_params.get("lonsin", False):
            sample_inputs.append(np.sin(2 * np.pi * (lon[i] / 360)))

        if input_params.get("sat", False):
            if input_params.get("sss", False):
                sample_inputs.append(sss[i])
            if input_params.get("sst", False):
                sample_inputs.append(sst[i] - 273.15)
            if input_params.get("ssh", False):
                sample_inputs.append(ssh[i])

        # Convert the list of inputs for this sample to a tensor and add to the main list
        inputs.append(torch.tensor(sample_inputs, dtype=torch.float32))

    # Convert the list of tensors to a single tensor
    inputs_tensor = torch.stack(inputs)

    return inputs_tensor
