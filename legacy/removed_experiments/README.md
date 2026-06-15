# Archived monolith `__main__` experiments

These blocks were **commented dead code** inside the monolith `__main__` orchestrator.
They were removed in Phase 9 (`2cbb2b1`) and archived here verbatim (still commented).

None were wired into the active pipeline. Active post-training logic lives in
`src/nespreso/experiments/` and `experiments/*.py`.

| File | Summary | Notes |
|------|---------|-------|
| `01_netcdf_validation_export.py` | `create_netcdf(...)` → `Test_dataset.nc` | One-off NetCDF export |
| `02_missing_date_histogram.py` | Year-month histogram of profiles outside `full_dataset.TIME` | Exploratory QA |
| `03_kdtree_grid_filter.py` | 0.1° grid + `cKDTree` spatial coverage plot | Exploratory |
| `04_npl_sound_speed.py` | NPL sound speed, sonic-layer depth, BLG | References undefined `temperature_profile` / `MLD_index` in context |
| `05_nature_run_ssh_histogram.py` | Training AVISO SSH vs NatureRun `ssh10` | Hardcoded `/unity/.../NatureRun/` |
| `06_nature_run_ts_by_ssh_bin.py` | T–S diagrams by SSH range | Needs `sigma_theta`, `cores` not in scope |

To revive: port into `experiments/` with config-driven paths, then add characterization
pins if numerics matter.
