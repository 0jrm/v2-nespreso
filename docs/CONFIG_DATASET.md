# Config vs dataset artifact separation

## Current state (v1.1 / GoM default)

`configs/default.yaml` is the **training/runtime config** (paths, bbox, model
hyperparameters, density penalty, runtime flags).

`paths.dataset_pickle` (`config_dataset_full.pkl` on HPC) is the **dataset artifact**
path used by `runner._prepare_data_and_loaders`:

| Pickle key | Belongs in config YAML? | Notes |
|------------|-------------------------|-------|
| `full_dataset` | No — data artifact | `TemperatureSalinityDataset` + embedded PCA |
| `n_components`, `layers_config`, `epochs`, … | Yes — legacy only | Present in old combined pickles; ignored on load |
| `input_params` | Yes — legacy only | Overwritten on load from current config |

**New pickles** (written by current `runner`) contain only `{"full_dataset": ...}`.
Hyperparameters are read exclusively from YAML.

**Legacy combined pickles** (e.g. existing HPC `config_dataset_full.pkl`) still
load via `pickle_compat.load_dataset_pickle`; only `full_dataset` is used. On
load, v2 **re-applies** `model.*` and `input_params` from YAML and calls
`full_dataset.reload()` unless `runtime.load_trained_model` is true.

Legacy pickles store `__main__.TemperatureSalinityDataset`; see `PICKLE_MIGRATION.md`.

## Problem (legacy combined pickles)

Coupling config and dataset in one pickle file causes:

- Ambiguity about which hyperparameters are authoritative after a config change
- Large binary churn when only YAML changes
- `__main__` class paths tied to how the monolith was executed
- Difficulty versioning data builds independently of training runs

The write-path split (step 2 below) is implemented for new builds; existing HPC
artifacts were not resaved.

## Target (post–Phase 9)

Split responsibilities:

```
configs/default.yaml          # paths, bbox, model, runtime — single source of truth
data/artifacts/gom_dataset.pkl   # full_dataset only (or HDF5 via ingest/)
checkpoints/*.pth             # model_state_dict + PCA for inference (already separate)
```

Planned steps:

1. **Phase B** — optional `resave_dataset_pickle` so on-disk class path is
   `nespreso.data.dataset.TemperatureSalinityDataset` (`docs/PICKLE_MIGRATION.md`).
2. **Split write path** — **done.** `runner` writes `dataset.pkl` with only
   `full_dataset`; hyperparameters read exclusively from YAML (reload semantics
   preserved on load).
3. **Config keys** — add `paths.dataset_artifact` (data) distinct from training
   config; deprecate embedding hyperparams in pickle (read old pickles for compat).
4. **New basins** — prefer `nespreso.ingest` pipeline for dataset builds; YAML
   points at artifact path, not a monolith-era combined file.

## v1 defaults preserved

`configs/default.yaml` values reproduce GoM monolith behavior (`n_components=15`,
`layers_config=[512,512]`, `dataset_pickle` path, etc.). No numeric changes when
loading existing `config_dataset_full.pkl`.
