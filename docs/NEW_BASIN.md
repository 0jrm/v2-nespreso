# New basin checklist

Use this when moving NeSPReSO beyond the Gulf of Mexico defaults in `configs/default.yaml`.

## Must change

1. **ARGO profiles** — point `paths.argo_mat` at a regional `.mat` with `TIME`, `LAT`, `LON`, `TEMP`, `SAL`, `PRES`, `SH1950` in the same layout as the GoM file.
2. **Bounding box** — update `bbox` (`min_lat`, `max_lat`, `min_lon`, `max_lon`) and exclusion point (`ex_lat`, `ex_lon`) for your domain.
3. **Satellite coverage** — set `paths.aviso_folder`, `paths.sst_folder`, `paths.sss_folder` and download data with `python -m nespreso download ...` for the basin date range.
4. **Dataset pickle** — use a new `paths.dataset_pickle` path so you do not overwrite GoM artifacts.
5. **GEM polyfit** — refit GEM coefficients for the new region (handled inside `TemperatureSalinityDataset.calc_gem` on the validation split).

## Retrained automatically

- **PCA** — `TemperatureSalinityDataset.reload()` refits temperature/salinity PCA on the new profiles when you rebuild the pickle.

## Usually unchanged

- Model architecture (`layers_config`, `PredictionModel`)
- Loss stack (`CombinedPCALoss`, density penalty wiring)
- Training loop (Adam, early stopping, split ratios)
- `input_params` structure (seasonal harmonics + satellite flags)

## Suggested first experiment

1. Copy `configs/default.yaml` → `configs/my_basin.yaml`.
2. Adjust `paths`, `bbox`, and `dataset_pickle`.
3. Download satellite fields for your period.
4. Lower `model.epochs` for a smoke train; then restore `8000` / `patience: 500`.
5. Sweep `n_components` (10–20) on validation RMSE.

## Density surrogate

If using `density.enabled: true`, provide basin-appropriate `density.checkpoint` and `density.stats_path`, or disable the penalty until a regional ρ-MLP is trained.
