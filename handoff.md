# NeSPReSO Refactor Handoff — Continue from Phase 7 extraction

We are refactoring the NeSPReSO monolith:

`singleFileModel_SAT_stats4verticalProj_meeting20260203.py`

Repo/HPC path:

`/unity/g2/jmiranda/v2-nespreso`

Branch:

`refactor/modularize`

Recent commits:

```text
af40a31 refactor(viz): extract profile and map plotting helpers from monolith
e25b137 test(viz): pin monolith plotting helpers with Agg backend goldens
6c5419d docs: update handoff for Phase 5 and Phase 6 completion
```

## Phase 7 — in progress (profiles + maps extracted)

Per `phase7.txt` (profiles/maps portion; `fields.py` deferred):

| Module | Path | Contents |
|---|---|---|
| Profiles | `src/nespreso/viz/profiles.py` | `visualize_combined_results`, `filter_by_season`, `seasonal_plots`, `calculate_bias` |
| Maps | `src/nespreso/viz/maps.py` | `calculate_average_in_bin`, `plot_bin_map`, `plot_rmse_on_ax`, `plot_comparison_maps`, `plot_residual_profiles_for_top_bins` |

Monolith re-exports all symbols. `plt.show` / file writes remain at rendering edges.

### Pin tests added

- `tests/test_viz.py` — synthetic golden pins for bin statistics, bias helpers, and Agg-backend figure artifacts
- `tests/golden/viz_maps_synthetic.json`
- `tests/golden/viz_profiles_synthetic.json`
- `tests/golden/viz_plot_artifacts.json`

Uses independent seeds per fixture (`100` maps, `101` bias, `102`/`1020`/`103` plots).

### Needs human review

- `calculate_bias` still references module-level `min_depth` / `max_depth` globals (unused `depths` variable preserved verbatim). Pin tests set these on `nespreso.viz.profiles` before calling through the monolith re-export.

## Phase 6 — complete

Per `phase6.txt` (inference portion; `train.py` was already extracted):

| Module | Path | Contents |
|---|---|---|
| Inference | `src/nespreso/inference.py` | `get_predictions`, `get_inputs`, `predict_with_numpy`, `get_predictions_torchscript`, `load_all_models` |

Monolith re-exports all symbols. Nested `get_predictions_torchscript` in `__main__` removed.

### Pin tests added

- `tests/test_inference.py` — synthetic golden pins for dataloader inference helpers
- `tests/golden/inference_synthetic.json`

## Phase 5 — complete

Per `phase5.txt`, model and loss blocks were pinned then extracted:

| Module | Path | Contents |
|---|---|---|
| MLP | `src/nespreso/models/mlp.py` | `PredictionModel` |
| Density | `src/nespreso/models/density.py` | `RhoMLP`, `DensityConstraint` |
| Losses | `src/nespreso/losses.py` | `WeightedMSELoss`, `genWeightedMSELoss`, `PCALoss`, `CombinedPCALoss`, `make_loss` |

Monolith re-exports all symbols for runner/characterization compatibility.

### Pin tests added

- `tests/test_losses.py` — synthetic golden forward pins
- `tests/golden/combined_pca_loss_synthetic.json` — loss scalar, profile heads, `PredictionModel` output

Uses `seeded_pca_pair` fixture (`np.random.seed(42)`) for deterministic synthetic batches.

## Phase 4 — complete

All items from `phase4.txt` are done:

| Module | Path | Notes |
|---|---|---|
| Dataset | `src/nespreso/data/dataset.py` | `TemperatureSalinityDataset`; thin shims for PCA/GEM |
| PCA | `src/nespreso/data/pca.py` | `sklearn_inverse_transform_pcs`, `torch_reconstruct_profile(s)` |
| Splits | `src/nespreso/data/splits.py` | `split_dataset`, `IndexedSubset` |
| GEM | `src/nespreso/data/gem.py` | `calc_gem`, `get_gem_profiles` |
| Features | `src/nespreso/data/features.py` | `prepare_inputs` (Phase 4 prep) |

### PCA architecture (pinned + canonicalized)

Two helper families — do **not** merge into one function:

1. **sklearn** — `sklearn_inverse_transform_pcs(pcs, pca_temp, pca_sal, n_components)` → `(depth, samples)`
2. **torch** — `torch_reconstruct_profile` / `torch_reconstruct_profiles` → `(batch, depth)`, float32 on `DEVICE`

Monolith still exposes `inverse_transform(...)` as a shim delegating to the sklearn helper.

Dead nested `inverse_transform` in the main block (~1898) was **removed**.

### Tests added in Phase 4.2

- `tests/test_pca_inverse.py` — unit + Unity goldens
- `tests/golden/pca_inverse_profile_0.json`
- `tests/golden/pca_inverse_api_profile_0.json`

## Verification (Phase 7 extraction)

```bash
python -m compileall src singleFileModel_SAT_stats4verticalProj_meeting20260203.py

pytest tests/test_smoke.py tests/test_config_paths.py tests/test_prepare_inputs.py \
  tests/test_satellite_loader.py tests/test_argo_loader.py tests/test_splits.py \
  tests/test_pca_inverse.py tests/test_losses.py tests/test_inference.py \
  tests/test_viz.py -q

srun --ntasks=1 --cpus-per-task=8 --gres=gpu:1 \
  pytest tests/test_characterization.py tests/test_inference.py \
  -m requires_unity --run-unity -q
```

Last run: all passed (sklearn unpickle-version warnings only; goldens at `1e-6`).

## Next sprint: Phase 7 remainder — field plots + `__main__` analysis

High-risk monolith blocks still in `singleFileModel_SAT_stats4verticalProj_meeting20260203.py`:

- `plot_field` / `plot_field_subplot` (glider section; target `src/nespreso/viz/fields.py`)
- `plot_coefficients_heatmap` (nested in `__main__`)
- Most of the `__main__` post-training analysis block (orchestration vs library split)

Recommended sequence:

1. Pin glider field plot helpers with Agg-backend artifact tests.
2. Extract `src/nespreso/viz/fields.py`.
3. Peel `__main__` analysis orchestration toward `experiments/` or `runner` hooks without changing numerics.

## Prior accepted HEAD (Phase 4)

`9fbaf35b8877c51ed02fd522bd0e3b7b1628c282`

```text
9fbaf35 refactor(data): extract pca and gem helpers from data layer
b5e8ff9 refactor(data): pin PCA inverse transform variants
e8e07f2 refactor(data): extract TemperatureSalinityDataset from monolith
```

## Pickle compatibility (unchanged)

Old pickles reference `__main__.TemperatureSalinityDataset`. Runner maps `sys.modules["__main__"]` to the monolith, which re-exports:

```python
from nespreso.data.dataset import TemperatureSalinityDataset
```

Verified pickle: `/unity/g2/jmiranda/SubsurfaceFields/GEM_SubsurfaceFields/config_dataset_full.pkl`

Pickled datasets may lack `n_components`; tests patch via `ds.pca_temp.n_components_`. Production runner sets `n_components` before `reload()`.

## Rules reminder

- Preserve behavior; move verbatim first.
- One commit per logical move; repo importable after each.
- Run characterization tests on HPC after numerical moves.
- Read `AGENTS.md` / `CLAUDE.md` before every change.

## Prior handoffs

Archived under `old_handoffs/`:

- `03_handoff_phase4_dataset_e8e07f2.md` — pre–Phase 4.2 (dataset extraction accepted, PCA pinning deferred)
- `02_handoff_future_agents.md`, `01_prompt_skynet.md`, `00_read_before_resuming.20260615.txt`
