# NeSPReSO Refactor Handoff — Continue from Phase 8 extraction

We are refactoring the NeSPReSO monolith:

`singleFileModel_SAT_stats4verticalProj_meeting20260203.py`

Repo/HPC path:

`/unity/g2/jmiranda/v2-nespreso`

Branch:

`refactor/modularize`

## Phase 8 — in progress (experiments peel)

### Helpers hoisted from nested `__main__`

| Symbol | Path |
|---|---|
| `get_month`, `get_season` | `src/nespreso/utils/time.py` |
| `count_profiles_per_month` | `src/nespreso/analysis/monthly.py` |
| `print_training_params` | `src/nespreso/reporting.py` |

### Experiment library (`src/nespreso/experiments/`)

| Module | Contents |
|---|---|
| `validation_context.py` | `ValidationContext`, `build_validation_context` — post-training preds, GEM, legacy 1.0, ISOP stats |
| `pca_regression.py` | `run_pca_regression_baseline` — MLR fit, depth RMSE/bias figure, coefficient heatmaps |
| `density_stability.py` | `run_density_stability` — vertical density/stability/smoothness comparison plots |
| `glider_mission.py` | `run_glider_mission` — four glider crossings + AVISO overlay maps |
| `validation_maps.py` | `run_validation_maps` — binned RMSE/bias maps, seasonal depth curves, comparison maps |
| `common.py` | `build_experiment_parser`, `load_cfg_and_artifacts`, matplotlib setup |

### Runnable scripts (`experiments/`)

- `experiments/train.py` — `--config`, optional `--tensorboard` / `--log-dir`
- `experiments/pca_regression_baseline.py` — `--config`, `--bin-size`
- `experiments/density_stability.py`
- `experiments/glider_mission.py`
- `experiments/validation_maps.py`

Monolith `__main__` delegates to `build_validation_context` + the experiment runners above. Steric-height, depth-interval tables, and monthly distribution remain inline until peeled.

### Pin tests added

- `tests/test_experiment_helpers.py` — `get_season`, `get_month`, `count_profiles_per_month`

## Phase 7 — complete

Per `phase7.txt`, all plotting and analysis helpers are under `src/nespreso/viz/` and `src/nespreso/analysis/`.

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

## Next sprint: Phase 8 continued

Remaining inline `__main__` blocks to peel per `phase8.txt`:

1. `experiments/steric_depth_stats.py` — steric-height / ISOP depth-bin orchestration (~lines 433–933)
2. `experiments/compare_legacy_nespreso.py` — timing / ensemble / legacy 1.0 load (now in `validation_context`; may split)
3. `experiments/monthly_distribution.py` — train/val/test profiles-per-month bar chart

Then Phase 9: dead-code pass, `ARCHITECTURE.md`, monolith retirement.

## Needs human review

- **ISOP comparison maps**: `avg_rmse_isop_*` / `avg_bias_isop_*` restored in `run_validation_maps` from `ctx.data_ISOP` (were dropped during the first validation-context peel).
- **Glider satellite cache**: monolith previously referenced undefined `data` / `dataset_pickle_file` after `run_training` refactor; `run_glider_mission` now loads the dataset pickle explicitly before the `sss1` cache branch.
- **`lon_val` binning** at validation setup: `lon_val = np.floor(lon_val) + bin_size / 2` (preserved verbatim from monolith).

## Rules reminder

- Preserve behavior; move verbatim first.
- One commit per logical move; repo importable after each.
- Run characterization tests on HPC after numerical moves.
- Read `AGENTS.md` / `CLAUDE.md` before every change.

## Prior handoffs

Archived under `old_handoffs/`.
