# Changelog

All notable changes to the NeSPReSO package are documented here.

## [0.1.0a1] — 2026-06-15

First **alpha** release of the modular `nespreso` package. The monolith
(`singleFileModel_SAT_stats4verticalProj_meeting20260203.py`, ~4,550 lines) has
been refactored into an installable library with characterization-tested parity.

### Added

- Installable package under `src/nespreso/` (config, io, data, models, train,
  inference, metrics, analysis, viz, experiments).
- CLI: `python -m nespreso train|download`.
- Experiment scripts under `experiments/`; full pipeline via `experiments/run_all.py`.
- Characterization and unit tests (`tests/`); golden outputs for numerical parity.
- `ARCHITECTURE.md`, `docs/HOWTO.md`, `docs/NEW_BASIN.md`, `legacy/SOURCES.md`.
- Dataset pickle compat loader (`pickle_compat`) for legacy `__main__` class paths.

### Changed

- Repo-root monolith filename is a **deprecation shim** → `experiments/run_all.py`.
- Frozen monolith relic archived at `legacy/monolith/`.
- Six commented `__main__` blocks archived under `legacy/removed_experiments/`.

### Preserved (numerical parity)

- MATLAB datenum `+366` correction, PCA `n_components=15`, layer sizes `[512,512]`,
  train/val/test split 0.7/0.15/0.15, early stopping, combined PCA loss scaling,
  density penalty, GEM fitting, validation map binning.

### Deferred (post-alpha)

- Phase B dataset pickle re-save (`docs/PICKLE_MIGRATION.md`).
- Config vs dataset artifact split (`docs/CONFIG_DATASET.md`).
- `nespreso.ingest` port from `global_nespreso/old/data_extract/`.
- Remove repo-root deprecation shim after caller migration.

### Verification

```bash
pip install -e ".[dev]"
pytest tests/test_smoke.py tests/test_config_paths.py tests/test_pca_inverse.py \
  tests/test_losses.py tests/test_inference.py tests/test_viz.py -q
srun --ntasks=1 --cpus-per-task=8 --gres=gpu:1 \
  pytest tests/test_characterization.py -m requires_unity --run-unity -q
```
