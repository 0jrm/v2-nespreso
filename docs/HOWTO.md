# HOWTO — NeSPReSO rework

## Install (HPC or dev)

```bash
cd rework
pip install -e ".[dev]"
```

Optional extras: `pip install -e ".[download,monitor]"` for copernicusmarine and TensorBoard.

## Train (CLI)

Defaults reproduce the GoM monolith (`singleFileModel_SAT_stats4verticalProj_meeting20260203.py`):

```bash
python -m nespreso --config configs/default.yaml train
```

Workflow:

1. Load or build `config_dataset_full.pkl` (0.7 / 0.15 / 0.15 split).
2. Fit GEM on the validation subset.
3. Train with early stopping (`patience: 500`, `epochs: 8000`).
4. Save one checkpoint under `paths.saved_models_dir`.

### Load a trained model instead

Set in config:

```yaml
runtime:
  load_trained_model: true
paths:
  trained_model_path: /path/to/ocp_model_....pth
```

### TensorBoard (opt-in)

```bash
python -m nespreso --config configs/default.yaml train --tensorboard --log-dir runs/gom
tensorboard --logdir runs/gom
```

Or set `monitor.tensorboard: true` in YAML. Default is **off** so numerics match the monolith.

## Hyperparameters

All training knobs live in `configs/default.yaml` under `model`, `input_params`, and `density`. Paper defaults:

| Parameter | Default |
|-----------|---------|
| `n_components` | 15 |
| `layers_config` | [512, 512] |
| `learning_rate` | 0.001 |
| `batch_size` | 512 |
| `epochs` | 8000 |
| `patience` | 500 |
| `dropout_prob` | 0.2 |

For a new region, tune mainly `n_components` (PCA complexity) and `patience` (early-stop horizon).

## Download satellite data

Date/month + bbox filtering is supported:

```bash
# AVISO SSH — loops year/month (ported from eoas_pyutils)
python -m nespreso download aviso --output /path/AVISO --start-year 2014 --end-year 2022 \
  --min-lon 262 --max-lon 305 --min-lat 7.5 --max-lat 50

# OSTIA SST via copernicusmarine (replaces dead PODAAC OISST script)
python -m nespreso download ostia --output /path/SST --start 2014-01-01 --end 2014-01-31 \
  --min-lon -99 --max-lon -74 --min-lat 17 --max-lat 31

# SSS day loop via copernicusmarine
python -m nespreso download sss --output /path/SSS --start 2014-01-01 --end 2014-01-31 \
  --min-lon -180 --max-lon 180 --min-lat -90 --max-lat 90
```

Credentials: CMEMS via `~/.netrc` or copernicusmarine login; AVISO motuclient uses netrc `AVISO` host.

## Run the legacy monolith directly

Still supported for full analysis/plotting pipeline:

```bash
python singleFileModel_SAT_stats4verticalProj_meeting20260203.py
```

## Verification

```bash
pytest tests/test_smoke.py
# On HPC with /unity:
pytest tests/test_characterization.py -m requires_unity --run-unity
```
