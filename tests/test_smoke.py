"""Import and pure-function smoke tests (run without /unity)."""

import numpy as np
import pytest

from nespreso.config import load_config
from nespreso.determinism import get_device, set_seed
from nespreso.metrics import bias, mad, rmse
from nespreso.utils.time import datenum_to_datetime, matlab2datetime


def test_import_package():
    import nespreso

    assert nespreso.__version__ == "0.1.0"


def test_default_config_matches_gom_defaults():
    cfg = load_config()
    assert cfg.model.n_components == 15
    assert cfg.model.layers_config == (512, 512)
    assert cfg.bbox.min_lat == 18.0
    assert cfg.input_params.sat is True


def test_rmse_bias_mad_fixed_array():
    pred = np.array([1.0, 2.0, 3.0])
    targ = np.array([1.5, 2.5, 2.5])
    assert rmse(pred, targ) == pytest.approx(0.5)
    assert bias(pred, targ) == pytest.approx(-0.16666666666666666)
    x = np.array([[1.0, 2.0, 3.0], [4.0, 6.0, 8.0]])
    result = mad(x)
    assert result.shape == (2,)
    assert np.all(np.isfinite(result))


def test_datenum_roundtrip_consistency():
    sample = 737789.5
    assert datenum_to_datetime(sample) == matlab2datetime(sample)


def test_determinism_helpers():
    set_seed(42)
    device = get_device()
    assert str(device) in {"cpu", "cuda"}


def test_cli_parser_builds():
    from nespreso.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(["--config", "configs/default.yaml", "train"])
    assert args.command == "train"
