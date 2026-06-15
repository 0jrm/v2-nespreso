"""Configuration dataclasses and YAML loading."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


def _env_override(key: str, value: str | None) -> str | None:
    """Resolve a path-like config value with optional env override."""
    if value is None:
        return None
    env_key = key.upper().replace(".", "_")
    return os.environ.get(env_key, value)


@dataclass(frozen=True)
class BboxConfig:
    min_lat: float = 18.0
    max_lat: float = 31.0
    min_lon: float = -98.0
    max_lon: float = -81.0
    ex_lat: float = 23.0
    ex_lon: float = -90.0


@dataclass(frozen=True)
class PathsConfig:
    argo_mat: str = "/unity/g2/jmiranda/SubsurfaceFields/Data/ARGO_GoM_20220920.mat"
    aviso_folder: str = "/unity/f1/ozavala/DATA/GOFFISH/AVISO/GoM/"
    sst_folder: str = "/unity/f1/ozavala/DATA/GOFFISH/SST/OISST"
    sss_folder: str = "/Net/work/ozavala/DATA/GOFFISH/SSS/SMAP_Global/"
    dataset_pickle: str = "/unity/g2/jmiranda/SubsurfaceFields/GEM_SubsurfaceFields/config_dataset_full.pkl"
    saved_models_dir: str = "/unity/g2/jmiranda/SubsurfaceFields/GEM_SubsurfaceFields/saved_models"
    trained_model_path: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PathsConfig:
        defaults = cls()
        fields = {}
        for name in cls.__dataclass_fields__:
            val = data.get(name, getattr(defaults, name))
            if name == "trained_model_path":
                val = None if val in (None, "") else val
            fields[name] = _env_override(f"paths.{name}", val)
        return cls(**fields)


@dataclass(frozen=True)
class InputParams:
    timecos: bool = True
    timesin: bool = True
    latcos: bool = True
    latsin: bool = True
    loncos: bool = True
    lonsin: bool = True
    sat: bool = True
    sst: bool = True
    sss: bool = True
    ssh: bool = True


@dataclass(frozen=True)
class DensityConfig:
    enabled: bool = True
    checkpoint: str = "/unity/g2/jmiranda/SubsurfaceFields/2025-2_OCP-project/TEOS-ML/rhoMLP_w32_d3_best.pt"
    stats_path: str = "/unity/g2/jmiranda/SubsurfaceFields/2025-2_OCP-project/TEOS-ML/rho_norm_stats.npz"
    stab_weight: float = 0.001
    smooth_weight: float = 0.001
    stability_tol: float = 1e-6
    smooth_window: tuple[int, int] = (0, 500)


@dataclass(frozen=True)
class ModelConfig:
    n_components: int = 15
    layers_config: tuple[int, ...] = (512, 512)
    batch_size: int = 512
    min_depth: float = 0
    max_depth: float = 1800
    dropout_prob: float = 0.2
    epochs: int = 8000
    patience: int = 500
    learning_rate: float = 0.001
    train_size: float = 0.7
    val_size: float = 0.15
    test_size: float = 0.15


@dataclass(frozen=True)
class RuntimeFlags:
    load_trained_model: bool = False
    ensemble_models: bool = False
    load_dataset_file: bool = True
    gen_paula_profiles: bool = False
    debug: bool = False
    seed: int = 42
    n_runs: int = 1
    nn_repeat_time: int = 10
    gem_repeat_time: int = 1


@dataclass(frozen=True)
class MonitorConfig:
    tensorboard: bool = False
    log_dir: str = "runs/nespreso"


@dataclass(frozen=True)
class AppConfig:
    paths: PathsConfig = field(default_factory=PathsConfig)
    bbox: BboxConfig = field(default_factory=BboxConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    input_params: InputParams = field(default_factory=InputParams)
    density: DensityConfig = field(default_factory=DensityConfig)
    runtime: RuntimeFlags = field(default_factory=RuntimeFlags)
    monitor: MonitorConfig = field(default_factory=MonitorConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AppConfig:
        density_data = dict(data.get("density", {}))
        smooth_window = density_data.get("smooth_window", (0, 500))
        if isinstance(smooth_window, list):
            smooth_window = tuple(smooth_window)
        density = DensityConfig(**{**DensityConfig().__dict__, **density_data, "smooth_window": smooth_window})

        model_data = data.get("model", {})
        layers = model_data.get("layers_config", [512, 512])
        model = ModelConfig(**{**ModelConfig().__dict__, **model_data, "layers_config": tuple(layers)})

        return cls(
            paths=PathsConfig.from_dict(data.get("paths", {})),
            bbox=BboxConfig(**{**BboxConfig().__dict__, **data.get("bbox", {})}),
            model=model,
            input_params=InputParams(**{**InputParams().__dict__, **data.get("input_params", {})}),
            density=density,
            runtime=RuntimeFlags(**{**RuntimeFlags().__dict__, **data.get("runtime", {})}),
            monitor=MonitorConfig(**{**MonitorConfig().__dict__, **data.get("monitor", {})}),
        )


def density_penalty_dict(cfg: AppConfig) -> dict[str, Any]:
    """Density penalty config dict expected by the monolith loss."""
    payload = asdict(cfg.density)
    payload["smooth_window"] = list(payload["smooth_window"])
    return payload


def load_config(path: str | Path | None = None) -> AppConfig:
    """Load YAML config; defaults match current GoM monolith behavior."""
    if path is None:
        path = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"
    path = Path(path)
    with path.open() as fh:
        data = yaml.safe_load(fh) or {}
    return AppConfig.from_dict(data)
