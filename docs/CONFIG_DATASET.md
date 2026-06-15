# Config vs dataset artifact separation

## Current state (v1.1 / GoM default)

`configs/default.yaml` is the **training/runtime config** (paths, bbox, model
hyperparameters, density penalty, runtime flags).

`paths.dataset_pickle` (`config_dataset_full.pkl` on HPC) is a **combined artifact**
written by `runner._prepare_data_and_loaders`:

| Pickle key | Belongs in config YAML? | Notes |
|------------|-------------------------|-------|
| `full_dataset` | No — data artifact | `TemperatureSalinityDataset` + embedded PCA |
| `n_components`, `layers_config`, `epochs`, … | Yes — duplicates YAML | Frozen at pickle creation time |
| `input_params` | Yes — duplicates YAML | Overwritten on load from current config |

On load, v2 **re-applies** `model.*` and `input_params` from YAML and calls
`full_dataset.reload()` unless `runtime.load_trained_model` is true. So YAML is
authoritative for hyperparameters at train time, but the pickle still embeds a
copy of whatever was used when it was built.

Legacy pickles store `__main__.TemperatureSalinityDataset`; see `PICKLE_MIGRATION.md`.

## Problem

Coupling config and dataset in one pickle file causes:

- Ambiguity about which hyperparameters are authoritative after a config change
- Large binary churn when only YAML changes
- `__main__` class paths tied to how the monolith was executed
- Difficulty versioning data builds independently of training runs

## Target (post–Phase 9, no behavior change until implemented)

Split responsibilities:

```
configs/default.yaml          # paths, bbox, model, runtime — single source of truth
data/artifacts/gom_dataset.pkl   # full_dataset only (or HDF5 via ingest/)
checkpoints/*.pth             # model_state_dict + PCA for inference (already separate)
```

Planned steps (not yet implemented — do not resave HPC artifacts without approval):

1. **Phase B** — optional `resave_dataset_pickle` so on-disk class path is
   `nespreso.data.dataset.TemperatureSalinityDataset` (`docs/PICKLE_MIGRATION.md`).
2. **Split write path** — `runner` writes `dataset.pkl` with only `full_dataset`;
   hyperparameters read exclusively from YAML (preserve current reload semantics).
3. **Config keys** — add `paths.dataset_artifact` (data) distinct from training
   config; deprecate embedding hyperparams in pickle (read old pickles for compat).
4. **New basins** — prefer `nespreso.ingest` pipeline for dataset builds; YAML
   points at artifact path, not a monolith-era combined file.

## v1 defaults preserved

Until split is implemented, `configs/default.yaml` values reproduce GoM monolith
behavior (`n_components=15`, `layers_config=[512,512]`, `dataset_pickle` path, etc.).
No numeric changes when loading existing `config_dataset_full.pkl`.
