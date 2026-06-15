# NeSPReSO Refactor Handoff — Continue from Phase 7 extraction

We are refactoring the NeSPReSO monolith:

`singleFileModel_SAT_stats4verticalProj_meeting20260203.py`

Repo/HPC path:

`/unity/g2/jmiranda/v2-nespreso`

Branch:

`refactor/modularize`

## Phase 7 — in progress (viz + analysis helpers extracted)

### Viz (complete)

Per `phase7.txt`, all plotting helpers are now under `src/nespreso/viz/`:

| Module | Path | Contents |
|---|---|---|
| Profiles | `src/nespreso/viz/profiles.py` | `visualize_combined_results`, `filter_by_season`, `seasonal_plots`, `calculate_bias` |
| Maps | `src/nespreso/viz/maps.py` | `calculate_average_in_bin`, `plot_bin_map`, `plot_rmse_on_ax`, `plot_comparison_maps`, `plot_residual_profiles_for_top_bins` |
| Fields | `src/nespreso/viz/fields.py` | `plot_field`, `plot_field_subplot` |
| Coefficients | `src/nespreso/viz/coefficients.py` | `plot_coefficients_heatmap` |

### Analysis (geo/time, depth stats, MLR, density)

| Module | Path | Contents |
|---|---|---|
| Glider | `src/nespreso/analysis/glider.py` | `get_glider_predictions`, `bin_data` |
| Correlation | `src/nespreso/analysis/correlation.py` | `calculate_correlation` |
| Residuals | `src/nespreso/analysis/residuals.py` | `compute_profile_residual`, `compute_depth_rmse_bias` |
| Comparison | `src/nespreso/analysis/comparison.py` | `isop_depth_indices`, `default_depth_intervals`, `compute_season_masked_depth_rmse_bias`, `compute_depth_interval_metrics` |
| Depth stats | `src/nespreso/analysis/depth_stats.py` | `average_depth`, `histogram_available_depths`, `equivalent_average_statistic` |
| MLR baseline | `src/nespreso/analysis/mlr.py` | `prepare_features`, `fit_pcs_regression_exact_gpu`, `predict_pcs_exact_gpu` |
| Density | `src/nespreso/analysis/density.py` | `compute_density_profiles`, `compute_stability_metrics`, `compute_smoothness_metrics` |

### Utils (geo/time)

| Module | Path | Contents |
|---|---|---|
| Geo | `src/nespreso/utils/geo.py` | `haversine`, `calculate_distances` |
| Time | `src/nespreso/utils/time.py` | `datenum_to_datetime`, `matlab2datetime`, `datenums_to_datetimes` |

Monolith re-exports all symbols. `__main__` uses hoisted helpers throughout; MLR baseline reinstated from `singleFileModel_SAT.py`; NeSPReSO 1.0 comparisons use `old_pred_*` residuals separately from MLR.

### Pin tests added

- `tests/test_viz.py`, `tests/test_viz_fields.py`, `tests/test_viz_coefficients.py` — viz golden pins
- `tests/test_analysis.py` — analysis golden pins (seeds `401`–`404`)
- `tests/test_geo_time.py` — geo/time golden pins (seed `405`)
- `tests/test_analysis_depth_stats.py` — depth-bin stats golden pins (seeds `406`–`407`)
- `tests/test_mlr.py` — MLR baseline golden pins (seed `408`, CPU-pinned)
- `tests/test_analysis_density.py` — density/stability golden pins (seeds `409`–`410`)
- `tests/golden/analysis_synthetic.json`, `tests/golden/geo_depth_mlr_synthetic.json`, `tests/golden/analysis_density_synthetic.json`

### Resolved human-review items

- **`calculate_bias`**: now takes explicit `min_depth` / `max_depth` parameters (defaults `20` / `2000`); removed unused module-global `depths` variable.
- **MLR vs NeSPReSO 1.0**: real MLR pipeline restored (`prepare_features` → GPU fit → PCA inverse); NeSPReSO 1.0 kept on `old_pred_*` for maps, seasonal plots, and depth-interval tables; profile RMSE/bias figure shows ISOP, GEM, MLR, 1.0, 1.1.
- **`equivalent_average_statistic`**: fixed mixed 1 m prediction / 5 m glider-target binning; glider call sites now pass crossing predictions (`T_pred1`…`T_pred4`) instead of validation `pred_T`.
- **`plot_field`**: still re-exported but unused in `__main__` (only `plot_field_subplot` called); kept for API parity.
- **`get_glider_predictions`**: hoisted version correctly uses the `model` parameter; `loader` remains unused (preserved verbatim).

## Phase 6 — complete

Per `phase6.txt` (inference portion; `train.py` was already extracted):

| Module | Path | Contents |
|---|---|---|
| Inference | `src/nespreso/inference.py` | `get_predictions`, `get_inputs`, `predict_with_numpy`, `get_predictions_torchscript`, `load_all_models` |

## Verification

```bash
python -m compileall src singleFileModel_SAT_stats4verticalProj_meeting20260203.py

pytest tests/test_smoke.py tests/test_config_paths.py tests/test_prepare_inputs.py \
  tests/test_satellite_loader.py tests/test_argo_loader.py tests/test_splits.py \
  tests/test_pca_inverse.py tests/test_losses.py tests/test_inference.py \
  tests/test_viz.py tests/test_viz_fields.py tests/test_viz_coefficients.py \
  tests/test_analysis.py tests/test_geo_time.py tests/test_analysis_depth_stats.py \
  tests/test_mlr.py tests/test_analysis_density.py -q

srun --ntasks=1 --cpus-per-task=8 --gres=gpu:1 \
  pytest tests/test_characterization.py tests/test_inference.py \
  -m requires_unity --run-unity -q
```

Last run: all passed (65 unit tests, 2 skipped; sklearn unpickle-version warnings only; goldens at `1e-6`).

## Next sprint: Phase 8 — peel `__main__` orchestration into `experiments/`

Remaining nested `__main__` helpers:

- `printParams`, `get_season`, `_get_month`, `count_profiles_per_month`
- Steric-height / ISOP depth-bin orchestration blocks (helpers already hoisted)

Recommended sequence:

1. Create `experiments/pca_regression_baseline.py`, `experiments/density_stability.py`, `experiments/glider_mission.py` importing library helpers.
2. Peel remaining orchestration blocks per `phase8.txt`.
3. Leave monolith trunk importable until Phase 9 retirement.

## Rules reminder

- Preserve behavior; move verbatim first.
- One commit per logical move; repo importable after each.
- Run characterization tests on HPC after numerical moves.
- Read `AGENTS.md` / `CLAUDE.md` before every change.

## Prior handoffs

Archived under `old_handoffs/`.
