"""Configuration dataclasses and YAML loading."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
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

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.min_lat, self.max_lat, self.min_lon, self.max_lon)


@dataclass(frozen=True)
class PathsConfig:
    argo_mat: str = "/unity/g2/jmiranda/SubsurfaceFields/Data/ARGO_GoM_20220920.mat"
    aviso_folder: str = "/unity/f1/ozavala/DATA/GOFFISH/AVISO/GoM/"
    sst_folder: str = "/unity/f1/ozavala/DATA/GOFFISH/SST/OISST"
    sss_folder: str = "/Net/work/ozavala/DATA/GOFFISH/SSS/SMAP_Global/"
    dataset_pickle: str = "/unity/g2/jmiranda/SubsurfaceFields/GEM_SubsurfaceFields/config_dataset_full.pkl"
    saved_models_dir: str = "/unity/g2/jmiranda/SubsurfaceFields/GEM_SubsurfaceFields/saved_models"
    density_checkpoint: str = "/unity/g2/jmiranda/SubsurfaceFields/2025-2_OCP-project/TEOS-ML/rhoMLP_w32_d3_best.pt"
    density_stats: str = "/unity/g2/jmiranda/SubsurfaceFields/2025-2_OCP-project/TEOS-ML/rho_norm_stats.npz"
    isop_nc: str = "/unity/g2/jmiranda/SubsurfaceFields/Data/ISOP1_rmse_bias_1deg_maps.nc"
    trained_model_path: str = (
        "/unity/g2/jmiranda/SubsurfaceFields/GEM_SubsurfaceFields/saved_models/"
        "ocp_model_Test Loss: 0.9163_2025-11-16 10:03:45_sat.pth"
    )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PathsConfig:
        defaults = cls()
        fields = {}
        for name in cls.__dataclass_fields__:
            val = data.get(name, getattr(defaults, name))
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

    def as_dict(self) -> dict[str, bool]:
        return {
            "timecos": self.timecos,
            "timesin": self.timesin,
            "latcos": self.latcos,
            "latsin": self.latsin,
            "loncos": self.loncos,
            "lonsin": self.lonsin,
            "sat": self.sat,
            "sst": self.sst,
            "sss": self.sss,
            "ssh": self.ssh,
        }


@dataclass(frozen=True)
class DensityConfig:
    enabled: bool = True
    checkpoint: str = "/unity/g2/jmiranda/SubsurfaceFields/2025-2_OCP-project/TEOS-ML/rhoMLP_w32_d3_best.pt"
    stats_path: str = "/unity/g2/jmiranda/SubsurfaceFields/2025-2_OCP-project/TEOS-ML/rho_norm_stats.npz"
    stab_weight: float = 0.001
    smooth_weight: float = 0.001
    stability_tol: float = 1e-6
    smooth_window: tuple[int, int] = (0, 500)

    def as_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "checkpoint": self.checkpoint,
            "stats_path": self.stats_path,
            "stab_weight": self.stab_weight,
            "smooth_weight": self.smooth_weight,
            "stability_tol": self.stability_tol,
            "smooth_window": list(self.smooth_window),
        }


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
    bin_size: int = 1
    num_samples: int = 1


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
        if "paths" in data:
            paths = data["paths"]
            density_data.setdefault("checkpoint", paths.get("density_checkpoint"))
            density_data.setdefault("stats_path", paths.get("density_stats"))
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


def load_config(path: str | Path | None = None) -> AppConfig:
    """Load YAML config; defaults match current GoM monolith behavior."""
    if path is None:
        path = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"
    path = Path(path)
    with path.open() as fh:
        data = yaml.safe_load(fh) or {}
    return AppConfig.from_dict(data)
