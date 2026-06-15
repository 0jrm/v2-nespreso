# NeSPReSO Refactor Handoff — Phase 8 complete, start Phase 9

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

Monolith `__main__` is a thin orchestrator: `run_training` → `build_validation_context` → experiment runners. Large commented blocks remain for Phase 9 dead-code review.

## Phase 9 — next (per `phase9.txt`)

1. Dead-code pass on monolith commented blocks (~lines 165–465); list removals under `## Removed - confirm` and wait for sign-off on ambiguous blocks.
2. Finish type hints / docstrings on public APIs.
3. Write `ARCHITECTURE.md` (module map, data flow, experiment how-to, domain quirks).
4. Confirm monolith can retire (nothing imports it except characterization tests / runner shim).

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
- **Phase 9 commented monolith blocks**: netCDF export, nature-run T-S, sound-speed NPL, etc. — confirm before delete.

## Rules reminder

- Preserve behavior; move verbatim first.
- One commit per logical move; repo importable after each.
- Run characterization tests on HPC after numerical moves.

## Prior handoffs

Archived under `old_handoffs/`.
