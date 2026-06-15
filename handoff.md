# NeSPReSO Refactor Handoff — Phase 9 complete

We refactored the NeSPReSO monolith:

`singleFileModel_SAT_stats4verticalProj_meeting20260203.py`

Repo/HPC path:

`/unity/g2/jmiranda/v2-nespreso`

Branch:

`refactor/modularize`

## Phase 8 — complete

All `phase8.txt` experiment scripts exist under `experiments/` with library modules in `src/nespreso/experiments/`.

| Module | Contents |
|---|---|
| `validation_context.py` | `ValidationContext`, `build_validation_context` — ISOP bins, residuals, depth RMSE/bias |
| `compare_legacy_nespreso.py` | Timing, ensemble, NeSPReSO 1.0 load, GEM timing (called from validation_context) |
| `pca_regression.py` | `run_pca_regression_baseline` |
| `density_stability.py` | `run_density_stability` |
| `glider_mission.py` | `run_glider_mission` |
| `validation_maps.py` | `run_validation_maps` |
| `steric_depth_stats.py` | `run_steric_depth_stats` |
| `depth_interval_stats.py` | `run_depth_interval_stats` |
| `monthly_distribution.py` | `run_monthly_distribution` |
| `common.py` | `build_experiment_parser`, `load_cfg_and_artifacts` |

### Runnable scripts (`experiments/`)

- `train.py`
- `compare_legacy_nespreso.py`
- `pca_regression_baseline.py`
- `density_stability.py`
- `glider_mission.py`
- `validation_maps.py`
- `steric_depth_stats.py`
- `depth_interval_stats.py`
- `monthly_distribution.py`
- `run_all.py` (full pipeline)

## Phase 9 — complete

1. `ARCHITECTURE.md` — module map, data flow, runbook, domain quirks.
2. Dead-code pass — six commented `__main__` blocks archived under `legacy/removed_experiments/`.
3. Type hints / docstrings on public APIs — `analysis/*`, `utils/*`, `train.py`, `inference.py`, `config.py`, `data/*`, `io/*`, `viz/*`, `losses.py`, `experiments/*`, `runner.py`.
4. Monolith relocated to `legacy/monolith/`; forward path is `experiments/run_all.py` (repo root is a deprecation shim).

## Removed — archived

The following commented `__main__` blocks were deleted from the active monolith in
`2cbb2b1` and preserved under `legacy/removed_experiments/` (verbatim, still commented).
Active post-training logic is in `src/nespreso/experiments/` and `experiments/*.py`.

| Block | Archive file |
|---|---|
| NetCDF validation export | `01_netcdf_validation_export.py` |
| Missing-date histogram | `02_missing_date_histogram.py` |
| KD-tree grid filter | `03_kdtree_grid_filter.py` |
| NPL sound-speed / SLD / BLG | `04_npl_sound_speed.py` |
| Nature-run SSH histogram | `05_nature_run_ssh_histogram.py` |
| Nature-run T-S by SSH bin | `06_nature_run_ts_by_ssh_bin.py` |

Full monolith snapshot (pre-removal): `legacy/monolith/singleFileModel_SAT_stats4verticalProj_meeting20260203.py`.

## Monolith retirement

| Artifact | Role |
|---|---|
| `legacy/monolith/singleFileModel_SAT_stats4verticalProj_meeting20260203.py` | Frozen relic (470 lines, pre–Phase 9 dead-code cut); not imported |
| `singleFileModel_SAT_stats4verticalProj_meeting20260203.py` (repo root) | Deprecation shim → `experiments/run_all.py` |
| `experiments/run_all.py` | **Forward** full pipeline (train + all experiments) |

| Consumer | Status |
|---|---|
| `runner.run_training` | Package-only (`pickle_compat`) |
| Golden / characterization tests | Package-only (`565069c`) |
| Dataset pickle on disk | `__main__.TemperatureSalinityDataset`; `pickle_compat` remaps |

**Deferred (post–Phase 9, optional):**

See `docs/POST_ALPHA_REVIEW.md` for actionable commands and sign-off checklists.

1. Phase B re-save dataset pickle (`docs/PICKLE_MIGRATION.md`).
2. Split config vs dataset artifact (`docs/CONFIG_DATASET.md`).
3. Remove repo-root deprecation shim once callers migrate to `experiments/run_all.py`.
4. Infrastructure port: `ingest/` from `global_nespreso/old/data_extract/` (`plan.md` Step 4).

## Verification

```bash
python -m compileall src singleFileModel_SAT_stats4verticalProj_meeting20260203.py experiments

pytest tests/test_smoke.py tests/test_config_paths.py tests/test_prepare_inputs.py \
  tests/test_satellite_loader.py tests/test_argo_loader.py tests/test_splits.py \
  tests/test_pca_inverse.py tests/test_losses.py tests/test_inference.py \
  tests/test_viz.py tests/test_viz_fields.py tests/test_viz_coefficients.py \
  tests/test_analysis.py tests/test_geo_time.py tests/test_analysis_depth_stats.py \
  tests/test_mlr.py tests/test_analysis_density.py tests/test_experiment_helpers.py -q

srun --ntasks=1 --cpus-per-task=8 --gres=gpu:1 \
  pytest tests/test_characterization.py tests/test_inference.py \
  -m requires_unity --run-unity -q
```

## Needs human review

- **ISOP comparison maps**: `avg_rmse_isop_*` restored in `run_validation_maps` from `ctx.data_ISOP`.
- **Glider satellite cache**: `run_glider_mission` loads dataset pickle before `sss1` cache branch.
- **`lon_val` binning**: `lon_val = np.floor(lon_val) + bin_size / 2` (preserved verbatim).
- **Removed monolith blocks**: archived under `legacy/removed_experiments/`; revive via `experiments/` if needed.

## Rules reminder

- Preserve behavior; move verbatim first.
- One commit per logical move; repo importable after each.
- Run characterization tests on HPC after numerical moves.

## Prior handoffs

Archived under `old_handoffs/`.
