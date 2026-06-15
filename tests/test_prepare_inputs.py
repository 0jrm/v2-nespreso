"""Unit tests for prepare_inputs feature construction."""

import numpy as np
import pytest
import torch

from nespreso.data.features import prepare_inputs

_DEFAULT_INPUT_PARAMS = {
    "timecos": True,
    "timesin": True,
    "latcos": True,
    "latsin": True,
    "loncos": True,
    "lonsin": True,
    "sat": True,
    "sst": True,
    "sss": True,
    "ssh": True,
}

_EXPECTED_BATCH = torch.tensor(
    [
        [
            -0.150055393576622,
            0.9886775612831116,
            0.6427876353263855,
            0.7660444378852844,
            6.123234262925839e-17,
            -1.0,
            35.0,
            22.0,
            0.10000000149011612,
        ],
        [
            -0.9549667835235596,
            -0.29671281576156616,
            0.5,
            0.8660253882408142,
            0.08715574443340302,
            -0.9961947202682495,
            36.0,
            27.0,
            0.20000000298023224,
        ],
    ],
    dtype=torch.float32,
)


def test_prepare_inputs_default_feature_stack():
    time = np.array([100.0, 200.0])
    lat = np.array([25.0, 30.0])
    lon = np.array([-90.0, -85.0])
    sss = np.array([35.0, 36.0])
    sst = np.array([295.15, 300.15])
    ssh = np.array([0.1, 0.2])

    result = prepare_inputs(time, lat, lon, sss, sst, ssh, _DEFAULT_INPUT_PARAMS)

    assert result.shape == (2, 9)
    assert result.dtype == torch.float32
    torch.testing.assert_close(result, _EXPECTED_BATCH)


def test_prepare_inputs_respects_feature_flags():
    time = np.array([100.0])
    lat = np.array([25.0])
    lon = np.array([-90.0])
    sss = np.array([35.0])
    sst = np.array([295.15])
    ssh = np.array([0.1])

    result = prepare_inputs(
        time,
        lat,
        lon,
        sss,
        sst,
        ssh,
        {"sat": True, "sst": True, "sss": False, "ssh": False},
    )

    assert result.shape == (1, 1)
    assert result.item() == pytest.approx(22.0)
