# NeSPReSO

**Neural Subsurface Pressure and Salinity Reconstruction from Ocean surface fields**

NeSPReSO reconstructs subsurface temperature and salinity profiles from satellite
observations (SSH, SST, SSS) using PCA-compressed vertical structure and a neural
network trained with physics-informed density constraints. This repository is the
modular Python package (`nespreso`) refactored from the original research monolith
for the Gulf of Mexico (GoM) use case.

**Release:** `0.1.0a1` (alpha) — behavior matches the frozen monolith; see
[CHANGELOG.md](CHANGELOG.md).

## What this package does

1. **Ingest** ARGO profile data and co-located satellite fields (AVISO SSH, SST, SSS).
2. **Compress** vertical T/S structure with PCA and fit GEM (Gravest Empirical Mode) profiles.
3. **Train** an MLP to predict PCA coefficients from surface inputs and location/time features.
4. **Evaluate** with depth-resolved RMSE, steric height, density stability, glider missions, and map-based validation.

The default configuration reproduces the GoM experiment from
`singleFileModel_SAT_stats4verticalProj_meeting20260203.py`.

## Quick start

```bash
git clone <repo-url> v2-nespreso
cd v2-nespreso
pip install -e ".[dev]"
```

Train with GoM defaults (requires HPC data paths in `configs/default.yaml`):

```bash
python -m nespreso train --config configs/default.yaml
```

Run training plus all post-training validation experiments:

```bash
python experiments/run_all.py --config configs/default.yaml
```

## Installation

### Requirements

- Python ≥ 3.10
- PyTorch, NumPy &lt; 2, scikit-learn, xarray, and other scientific stack (see [pyproject.toml](pyproject.toml))
- GPU recommended for training; CPU works for smoke tests

### Editable install

```bash
pip install -e ".[dev]"
```

Optional extras:

| Extra | Purpose |
|-------|---------|
| `dev` | pytest, ruff |
| `download` | copernicusmarine, requests (satellite downloads) |
| `monitor` | tensorboard (training logs) |

```bash
pip install -e ".[dev,download,monitor]"
```

On an HPC cluster, wrap long jobs with resource limits, for example:

```bash
srun --ntasks=1 --cpus-per-task=8 --gres=gpu:1 \
  python -m nespreso train --config configs/default.yaml
```

## Training

### CLI

```bash
python -m nespreso train --config configs/default.yaml
```

Training workflow:

1. Load or build the dataset pickle (`paths.dataset_pickle`).
2. Split profiles 70% / 15% / 15% (train / val / test).
3. Fit GEM on the validation subset.
4. Train with Adam and early stopping (`patience: 500`, up to `epochs: 8000`).
5. Save a checkpoint under `paths.saved_models_dir`.

### Load a trained model instead of retraining

In your config YAML:

```yaml
runtime:
  load_trained_model: true
paths:
  trained_model_path: /path/to/ocp_model_....pth
```

### TensorBoard (opt-in)

TensorBoard is **off by default** so numerics match the monolith.

```bash
python -m nespreso train --config configs/default.yaml --tensorboard --log-dir runs/gom
tensorboard --logdir runs/gom
```

Or set `monitor.tensorboard: true` in YAML.

### Key hyperparameters

All training knobs live in `configs/default.yaml`. Paper / GoM defaults:

| Parameter | Default |
|-----------|---------|
| `n_components` | 15 |
| `layers_config` | [512, 512] |
| `learning_rate` | 0.001 |
| `batch_size` | 512 |
| `epochs` | 8000 |
| `patience` | 500 |
| `dropout_prob` | 0.2 |

For a new region, tune mainly `n_components` (PCA complexity) and `patience`
(early-stop horizon). See [docs/NEW_BASIN.md](docs/NEW_BASIN.md).

## Download satellite data

Date and bounding-box filtering is supported for all products:

```bash
# AVISO SSH — loops year/month
python -m nespreso download aviso --output /path/AVISO \
  --start-year 2014 --end-year 2022 \
  --min-lon 262 --max-lon 305 --min-lat 7.5 --max-lat 50

# OSTIA SST via copernicusmarine
python -m nespreso download ostia --output /path/SST \
  --start 2014-01-01 --end 2014-01-31 \
  --min-lon -99 --max-lon -74 --min-lat 17 --max-lat 31

# SSS day loop via copernicusmarine
python -m nespreso download sss --output /path/SSS \
  --start 2014-01-01 --end 2014-01-31 \
  --min-lon -180 --max-lon 180 --min-lat -90 --max-lat 90
```

**Credentials:** CMEMS via `~/.netrc` or `copernicusmarine login`; AVISO motuclient
uses netrc host `AVISO`.

Install download dependencies first: `pip install -e ".[download]"`.

## Post-training experiments

Individual validation scripts (each accepts `--config` and `--bin-size`):

| Script | Description |
|--------|-------------|
| `experiments/compare_legacy_nespreso.py` | Timing vs NeSPReSO 1.0 and GEM |
| `experiments/pca_regression_baseline.py` | PCA + linear regression baseline |
| `experiments/validation_maps.py` | Spatial validation maps |
| `experiments/steric_depth_stats.py` | Steric height depth statistics |
| `experiments/glider_mission.py` | Glider mission comparison |
| `experiments/depth_interval_stats.py` | Depth-interval RMSE/bias |
| `experiments/density_stability.py` | Density stability checks |
| `experiments/monthly_distribution.py` | Monthly residual distributions |

Full pipeline (train + all experiments):

```bash
python experiments/run_all.py --config configs/default.yaml
```

### Deprecated monolith entry point

The repo-root monolith filename still works but emits a `DeprecationWarning`:

```bash
python singleFileModel_SAT_stats4verticalProj_meeting20260203.py
```

Frozen source: [legacy/monolith/](legacy/monolith/).

## Configuration

Primary config: [configs/default.yaml](configs/default.yaml).

| Section | Purpose |
|---------|---------|
| `paths` | ARGO `.mat`, satellite folders, dataset pickle, model output |
| `bbox` | Domain bounds and exclusion point |
| `model` | Architecture, training schedule, split ratios |
| `input_params` | Seasonal harmonics and satellite input flags |
| `density` | Optional TEOS-ML density penalty |
| `runtime` | Seeds, debug, load-model flags |
| `monitor` | TensorBoard settings |

**New basin:** copy `configs/default.yaml`, adjust paths and bbox, then follow
[docs/NEW_BASIN.md](docs/NEW_BASIN.md).

**Dataset vs config:** YAML is the source of truth for hyperparameters; the pickle
holds the materialized dataset. Details in [docs/CONFIG_DATASET.md](docs/CONFIG_DATASET.md).

## Project layout

```
v2-nespreso/
├── README.md                 # this file
├── configs/default.yaml      # GoM defaults
├── src/nespreso/             # installable library
│   ├── cli.py                # python -m nespreso
│   ├── runner.py             # training orchestration
│   ├── data/                 # dataset, PCA, splits, GEM
│   ├── io/                   # ARGO/satellite loaders + downloaders
│   ├── models/               # MLP + density constraint
│   ├── train.py              # training loop
│   ├── inference.py          # prediction helpers
│   ├── analysis/             # validation statistics
│   ├── viz/                  # plotting helpers
│   └── experiments/          # experiment library modules
├── experiments/              # runnable CLI scripts
├── tests/                    # unit + characterization tests
├── notebooks/                # Jupyter cookbook
├── docs/                     # HOWTO, basin guide, migration notes
└── legacy/                   # frozen monolith + provenance
```

Architecture overview: [ARCHITECTURE.md](ARCHITECTURE.md).

## Notebook

[notebooks/NeSPReSO_cookbook.ipynb](notebooks/NeSPReSO_cookbook.ipynb) walks through
installation, config, training, and common workflows interactively.

## Testing

Smoke tests (no HPC data required):

```bash
pytest tests/test_smoke.py -q
```

Full unit suite:

```bash
pytest tests/ -q -m "not requires_unity"
```

Characterization tests (require `/unity` data paths and GPU):

```bash
srun --ntasks=1 --cpus-per-task=8 --gres=gpu:1 \
  pytest tests/test_characterization.py -m requires_unity --run-unity -q
```

Golden outputs assert numerical parity with the monolith within tolerance `1e-6`.

## Documentation index

| Document | Contents |
|----------|----------|
| [docs/HOWTO.md](docs/HOWTO.md) | Detailed runbook (install, train, download) |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Module map, data flow, domain quirks |
| [docs/NEW_BASIN.md](docs/NEW_BASIN.md) | Checklist for new regions |
| [docs/CONFIG_DATASET.md](docs/CONFIG_DATASET.md) | Config vs dataset artifact split |
| [docs/PICKLE_MIGRATION.md](docs/PICKLE_MIGRATION.md) | Legacy pickle class-path migration |
| [legacy/SOURCES.md](legacy/SOURCES.md) | Provenance of ported code |
| [CHANGELOG.md](CHANGELOG.md) | Release history |

## Contributing and development

This repo was built by **structural refactor**, not rewrite: preserve numerics,
move code verbatim first, and verify with characterization tests after each change.
See [CLAUDE.md](CLAUDE.md) / [AGENTS.md](AGENTS.md) for project rules.

## License

See repository license file if present; contact maintainers for data access
requirements (CMEMS, AVISO, ARGO).
