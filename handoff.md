# NeSPReSO Refactor Handoff — Phase 9 in progress

We are refactoring the NeSPReSO monolith:

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

Monolith `__main__` is a thin orchestrator: `run_training` → `build_validation_context` → experiment runners.

## Phase 9 — in progress (per `phase9.txt`)

1. ~~Write `ARCHITECTURE.md`~~ — done (`ARCHITECTURE.md`).
2. ~~Dead-code pass on monolith commented blocks~~ — done (see **Removed — confirm** below).
3. Type hints / docstrings on public APIs — in progress (`analysis/*`, `utils/geo.py`, `utils/time.py` annotated; `train.py` / `inference.py` / `config.py` already typed).
4. Confirm monolith can retire — **not yet** (see **Monolith retirement** below).

## Removed — confirm

The following commented `__main__` blocks were deleted from the monolith (previously ~lines 166–467). None were active code paths; all post-training logic they related to is now in `src/nespreso/experiments/` or was never wired into the orchestrator.

| Block | Summary | Disposition |
|---|---|---|
| NetCDF validation export | `create_netcdf(...)` writing `Test_dataset.nc` with SST, lat/lon, T/S profiles | **Removed** — one-off export script; not part of experiment pipeline |
| Missing-date histogram | Bar chart of year-month counts for profiles outside `full_dataset.TIME` | **Removed** — exploratory QA; no library equivalent |
| KD-tree grid filter | 0.1° grid + `cKDTree` scatter of points within 0.5° of training data | **Removed** — exploratory spatial coverage plot |
| NPL sound-speed / SLD / BLG | `calculate_sound_speed_NPL`, sonic-layer depth, below-layer gradient | **Removed** — standalone acoustics experiment; references undefined `temperature_profile` / `MLD_index` in commented context |
| Nature-run SSH histogram | Compare training AVISO SSH vs NatureRun `.mat` `ssh10` distributions | **Removed** — eddy/nature-run side experiment; hardcoded `/unity/.../NatureRun/` |
| Nature-run T-S by SSH bin | `plot_ts_profiles`, `aggregate_from_mat`, SSH-range T/S diagrams | **Removed** — same nature-run side experiment; needs `sigma_theta` / `cores` not in scope |

**Please confirm** if any of the above should be revived as a standalone script under `experiments/` before the monolith file is deleted.

## Monolith retirement

The monolith (~130 lines) is now a **re-export shim** plus `__main__` orchestrator.

| Consumer | Status |
|---|---|
| `runner.run_training` | **Decoupled** — uses `nespreso.*` imports and `pickle_compat.load_dataset_pickle` |
| `glider_mission.py` | **Decoupled** — uses `pickle_compat.load_dataset_pickle` |
| `tests/monolith_loader.py` + golden-pin tests | Still load monolith for namespace parity checks |
| Dataset pickle on disk | Still stores `__main__.TemperatureSalinityDataset`; compat unpickler remaps (see `docs/PICKLE_MIGRATION.md`) |

**Retirement steps (remaining):**

1. ~~Point `runner.py` at package imports~~ — done.
2. Optional: re-save dataset pickle with `resave_dataset_pickle` (Phase B in `docs/PICKLE_MIGRATION.md`).
3. Migrate characterization tests off `tests/monolith_loader.py` where only used for pickle load.
4. Delete monolith file after steps 2–3 pass HPC goldens.

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
- **Removed monolith blocks** (table above): confirm no revival needed before final monolith delete.

## Rules reminder

- Preserve behavior; move verbatim first.
- One commit per logical move; repo importable after each.
- Run characterization tests on HPC after numerical moves.

## Prior handoffs

Archived under `old_handoffs/`.
