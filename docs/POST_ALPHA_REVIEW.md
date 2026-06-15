# Post-alpha review guide (`0.1.0a1`)

Operational checklist for deferred work and human sign-off after the modular
alpha release. Complements `CHANGELOG.md`, `handoff.md`, and
`docs/PICKLE_MIGRATION.md` / `docs/CONFIG_DATASET.md`.

---

## Alpha release on `main`

| Item | Detail |
|---|---|
| **Branch** | `main` (created from full refactor history) |
| **Tag** | `v0.1.0a1` |
| **Commit** | `c60e57a` — `release: tag alpha 0.1.0a1 and complete Phase 9 public API typing` |
| **Remote** | `origin/main`, `origin/v0.1.0a1` pushed |
| **Docs added/updated** | `CHANGELOG.md`, `pyproject.toml` → `0.1.0a1`, `docs/HOWTO.md`, `ARCHITECTURE.md`, `handoff.md` |

```bash
git checkout main
pip install -e ".[dev]"
python experiments/run_all.py --config configs/default.yaml   # full pipeline
python -m nespreso train --config configs/default.yaml        # training only
```

---

## Deferred / post-alpha (actionable)

### 1. Phase B — re-save dataset pickle (drop `__main__` class path)

**Why:** On-disk pickle still references `__main__.TemperatureSalinityDataset`. Runtime compat works via `pickle_compat`, but the file itself is legacy.

**Pre-flight audit (txt dump):**

```bash
cd /unity/g2/jmiranda/v2-nespreso
mkdir -p reports
python - <<'PY' | tee reports/pickle_audit_pre.txt
import pickletools, sys
from nespreso.config import load_config
p = load_config().paths.dataset_pickle
print("pickle:", p)
with open(p, "rb") as f:
    pickletools.dis(f, out=sys.stdout)
PY
```

**Re-save to sidecar (do not overwrite until verified):**

```bash
srun --ntasks=1 --cpus-per-task=8 python - <<'PY'
from pathlib import Path
from nespreso.config import load_config
from nespreso.data.pickle_compat import resave_dataset_pickle

cfg = load_config()
src = Path(cfg.paths.dataset_pickle)
dst = src.with_name(src.stem + "_nespreso.pkl")
print(f"Writing {dst}")
resave_dataset_pickle(src, dst)
PY
```

**Verify parity before switching config:**

```bash
mkdir -p reports
srun --ntasks=1 --cpus-per-task=8 --gres=gpu:1 \
  pytest tests/test_characterization.py -m requires_unity --run-unity -q \
  | tee reports/char_test_pre_switch.txt

# Point config at new pickle, re-run same command — must still pass
```

**Post-save class-path check:**

```bash
python - <<'PY' | tee reports/pickle_audit_post.txt
import pickletools, sys
from pathlib import Path
from nespreso.config import load_config
p = Path(load_config().paths.dataset_pickle).with_name(
    Path(load_config().paths.dataset_pickle).stem + "_nespreso.pkl"
)
with open(p, "rb") as f:
    pickletools.dis(f, out=sys.stdout)
PY
# Expect nespreso.data.dataset.TemperatureSalinityDataset, not __main__
```

---

### 2. Config vs dataset artifact split

**Why:** `config_dataset_full.pkl` duplicates YAML hyperparameters inside the binary. YAML is authoritative at load time, but the coupling causes ambiguity and large diffs.

**Current-state inventory (txt report):**

```bash
srun --ntasks=1 --cpus-per-task=8 python - <<'PY' | tee reports/pickle_keys_vs_yaml.txt
import yaml
from nespreso.config import load_config
from nespreso.data.pickle_compat import load_dataset_pickle

cfg = load_config()
data = load_dataset_pickle(cfg.paths.dataset_pickle)
print("=== pickle top-level keys ===")
for k, v in data.items():
    print(f"  {k}: {type(v).__name__}")
print("\n=== YAML model block ===")
print(yaml.dump({"model": cfg.model.__dict__ if hasattr(cfg.model,'__dict__') else cfg.model}))
print("=== overlap keys (should eventually leave pickle) ===")
overlap = {"n_components","layers_config","epochs","patience","batch_size",
           "learning_rate","dropout_prob","train_size","val_size","test_size","input_params"}
print(sorted(overlap & set(data.keys())))
PY
```

**Implementation order (when approved):** split write path in `runner._prepare_data_and_loaders` → add `paths.dataset_artifact` → keep compat reader for old combined pickles. No HPC artifact rewrite until characterization tests pass on a **copy**.

---

### 3. Remove repo-root deprecation shim

**Why:** `singleFileModel_SAT_stats4verticalProj_meeting20260203.py` only forwards to `experiments/run_all.py`.

**Find remaining callers:**

```bash
cd /unity/g2/jmiranda
rg -l 'singleFileModel_SAT_stats4verticalProj_meeting20260203' --glob '*.py' --glob '*.sh' --glob '*.sbatch'
rg -l 'singleFileModel_SAT_stats4verticalProj' configs/ slurm/ 2>/dev/null
```

**Safe removal checklist:** zero external references → delete shim → confirm `python experiments/run_all.py` and `python -m nespreso train` still work → re-run characterization tests.

---

### 4. `nespreso.ingest` port (infrastructure)

**Why:** `global_nespreso/old/data_extract/` has SQLite+HDF5 co-location for new basins; current path is `.mat`-based.

**Scope reconnaissance (txt):**

```bash
find /unity/g2/jmiranda/global_nespreso/old/data_extract -name '*.py' | head -30 \
  | tee reports/ingest_source_tree.txt
wc -l /unity/g2/jmiranda/global_nespreso/old/data_extract/**/*.py 2>/dev/null \
  | tee -a reports/ingest_source_tree.txt
```

**Not started in alpha.** Target module: `src/nespreso/ingest/` per `plan.md` Step 4. Needs its own characterization pins before touching GoM production pickle.

---

## Needs human review (actionable)

### A. ISOP comparison maps — grid alignment

**Concern:** NeSPReSO/GEM maps are binned from validation profiles onto `lon_centers`/`lat_centers` (derived from ISOP NetCDF bounds). ISOP reference fields (`t_rmse_syn`, etc.) are passed **directly** from `data_ISOP` without regridding. If ISOP `lon`/`lat` don't match `lon_centr`/`lat_centr`, comparison plots are misleading.

Relevant code:

- `src/nespreso/experiments/validation_context.py` — builds `lon_centers`, `lat_centers` from ISOP file bounds
- `src/nespreso/experiments/validation_maps.py` L290-299 — passes `avg_rmse_isop_*` from NetCDF into `plot_comparison_maps` alongside NN-binned grids

**Diagnostic script (saves numeric diff + optional PDF):**

```bash
mkdir -p reports/isop_grid_check
srun --ntasks=1 --cpus-per-task=8 python - <<'PY'
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from nespreso.config import load_config
from nespreso.runner import run_training
from nespreso.experiments.validation_context import build_validation_context

cfg = load_config()
cfg.runtime.load_trained_model = True  # skip retrain if checkpoint set
_, art = run_training(cfg, return_artifacts=True)
ctx = build_validation_context(cfg, art, bin_size=1.0)

isop = ctx.data_ISOP
lon_c = ctx.lon_centers[:-1]
lat_c = ctx.lat_centers[:-1]

print("ISOP lon range:", float(isop.lon.min()), float(isop.lon.max()), "n=", isop.lon.size)
print("ctx lon_centers:", lon_c.min(), lon_c.max(), "n=", lon_c.size)
print("ISOP lat range:", float(isop.lat.min()), float(isop.lat.max()), "n=", isop.lat.size)
print("ctx lat_centers:", lat_c.min(), lat_c.max(), "n=", lat_c.size)

# If sizes match, max coordinate delta:
if isop.lon.size == lon_c.size and isop.lat.size == lat_c.size:
    dlon = np.max(np.abs(isop.lon.values - lon_c))
    dlat = np.max(np.abs(isop.lat.values - lat_c))
    print(f"max |Δlon|={dlon:.6f}, max |Δlat|={dlat:.6f}")
    with open("reports/isop_grid_check/alignment.txt", "w") as f:
        f.write(f"max_dlon={dlon}\nmax_dlat={dlat}\n")
else:
    print("SIZE MISMATCH — comparison maps need regridding review")
    with open("reports/isop_grid_check/alignment.txt", "w") as f:
        f.write("SIZE_MISMATCH\n")

# Quick visual: ISOP T RMSE vs NN-binned field shape
with PdfPages("reports/isop_grid_check/isop_vs_ctx_grid.pdf") as pdf:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    isop["t_rmse_syn"].plot(ax=axes[0], cmap="YlOrRd")
    axes[0].set_title("ISOP t_rmse_syn (native grid)")
    axes[1].text(0.1, 0.5, f"ctx grid {len(lat_c)}x{len(lon_c)}\nISOP {isop.t_rmse_syn.shape}", fontsize=14)
    axes[1].axis("off")
    pdf.savefig(fig); plt.close()
print("Wrote reports/isop_grid_check/")
PY
```

**Full visual review (interactive plots):**

```bash
srun --ntasks=1 --cpus-per-task=8 --gres=gpu:1 \
  python experiments/validation_maps.py --config configs/default.yaml
# Inspect ISOP comparison panels (lines 290-299 in validation_maps.py)
```

**Sign-off question:** Are `lon_centr`/`lat_centr` identical to ISOP NetCDF coordinates? If `alignment.txt` shows `SIZE_MISMATCH` or `max_dlon`/`max_dlat` > 0.01°, add explicit reindex/interp before `plot_comparison_maps`.

---

### B. Glider satellite cache — pickle side effects

**Concern:** `run_glider_mission` always calls `load_dataset_pickle()` (L79). If `sss1`…`aviso4` are absent, it calls `load_satellite_data()` (hardcoded `/unity/f1/ozavala/...` paths in `io/satellite.py`, not config bbox) and **rewrites** the entire dataset pickle with glider cache keys embedded.

Relevant code: `src/nespreso/experiments/glider_mission.py` L77-134.

**Inspect current pickle for cache keys:**

```bash
python - <<'PY' | tee reports/glider_pickle_keys.txt
from nespreso.config import load_config
from nespreso.data.pickle_compat import load_dataset_pickle
data = load_dataset_pickle(load_config().paths.dataset_pickle)
glider_keys = [k for k in data if k.startswith(("sss","sst","aviso"))]
print("glider cache keys:", glider_keys or "(none — first run will mutate pickle)")
print("pickle size keys:", sorted(data.keys()))
PY
```

**Dry-run glider only (watch for pickle rewrite):**

```bash
# Backup first
cp /unity/g2/jmiranda/SubsurfaceFields/GEM_SubsurfaceFields/config_dataset_full.pkl \
   /unity/g2/jmiranda/SubsurfaceFields/GEM_SubsurfaceFields/config_dataset_full.pkl.bak

ls -la /unity/g2/jmiranda/SubsurfaceFields/GEM_SubsurfaceFields/config_dataset_full.pkl

srun --ntasks=1 --cpus-per-task=8 --gres=gpu:1 \
  python experiments/glider_mission.py --config configs/default.yaml \
  2>&1 | tee reports/glider_mission_run.log

ls -la /unity/g2/jmiranda/SubsurfaceFields/GEM_SubsurfaceFields/config_dataset_full.pkl
# mtime/size change => pickle was rewritten
```

**Sign-off questions:**

1. Should glider satellite cache live in a **separate artifact** (not the training pickle)?
2. Should `load_satellite_data` use config paths/bbox instead of hardcoded GoM paths?
3. Is re-loading the full pickle on every glider run acceptable for HPC I/O?

---

### C. `lon_val` / `lat_val` binning (`floor + bin_size/2`)

**Concern:** In `validation_context.py` L208-209, profile coordinates are snapped to bin centers before `calculate_average_in_bin`. This is verbatim monolith behavior but may not match physical profile locations or ISOP's native 1° grid.

```python
lat_val = np.floor(lat_val) + bin_size / 2
lon_val = np.floor(lon_val) + bin_size / 2
```

**Quantify snapping error:**

```bash
srun --ntasks=1 --cpus-per-task=8 python - <<'PY' | tee reports/binning_snap_error.txt
import numpy as np
from nespreso.config import load_config
from nespreso.runner import run_training
from nespreso.experiments.validation_context import build_validation_context

cfg = load_config()
_, art = run_training(cfg, return_artifacts=True)
ctx = build_validation_context(cfg, art, bin_size=1.0)
idx = art.val_loader.dataset.indices
lat_raw, lon_raw, _ = art.full_dataset.get_lat_lon_date(idx)
lat_snap = np.floor(lat_raw) + 0.5
lon_snap = np.floor(lon_raw) + 0.5
print(f"n profiles: {len(lat_raw)}")
print(f"lat snap error: mean={np.mean(np.abs(lat_snap-lat_raw)):.4f} max={np.max(np.abs(lat_snap-lat_raw)):.4f}")
print(f"lon snap error: mean={np.mean(np.abs(lon_snap-lon_raw)):.4f} max={np.max(np.abs(lon_snap-lon_raw)):.4f}")
print(f"unique snapped cells: {len(np.unique(np.round(lat_snap,2)*1000 + np.round(lon_snap,2)))}")
PY
```

**Visual: snapped vs raw positions (PDF):**

```bash
srun --ntasks=1 --cpus-per-task=8 python - <<'PY'
import numpy as np, matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from nespreso.config import load_config
from nespreso.runner import run_training
from nespreso.experiments.validation_context import build_validation_context

cfg = load_config()
_, art = run_training(cfg, return_artifacts=True)
ctx = build_validation_context(cfg, art)
idx = art.val_loader.dataset.indices
lat_raw, lon_raw, _ = art.full_dataset.get_lat_lon_date(idx)

with PdfPages("reports/binning_raw_vs_snapped.pdf") as pdf:
    fig, ax = plt.subplots(figsize=(8,8))
    ax.scatter(lon_raw, lat_raw, s=8, alpha=0.4, label="raw")
    ax.scatter(ctx.lon_val, ctx.lat_val, s=8, alpha=0.4, label="snapped")
    ax.set_xlim(-99,-81); ax.set_ylim(18,32)
    ax.legend(); ax.set_title("Validation profile positions")
    pdf.savefig(fig)
print("reports/binning_raw_vs_snapped.pdf")
PY
```

**Sign-off:** Keep verbatim for parity, or change to `np.digitize` / actual coords (would need new golden baselines).

---

### D. Archived monolith `__main__` blocks

**Location:** `legacy/removed_experiments/01_…` through `06_…` (see `legacy/removed_experiments/README.md`).

**Inventory report:**

```bash
mkdir -p reports
for f in legacy/removed_experiments/0*.py; do
  echo "=== $f ===" >> reports/removed_blocks_summary.txt
  head -5 "$f" >> reports/removed_blocks_summary.txt
  rg -c '^[^#]' "$f" >> reports/removed_blocks_summary.txt || true
done
cat reports/removed_blocks_summary.txt
```

| Block | Revive? | Blocker |
|---|---|---|
| `01_netcdf_validation_export` | Maybe | Needs `experiments/export_validation_nc.py` + config paths |
| `02_missing_date_histogram` | Low priority | QA only |
| `03_kdtree_grid_filter` | Low priority | QA only |
| `04_npl_sound_speed` | **No without fix** | References undefined `temperature_profile`, `MLD_index` in original context |
| `05_nature_run_ssh_histogram` | Maybe | Hardcoded NatureRun path `/unity/.../NatureRun/` |
| `06_nature_run_ts_by_ssh_bin` | Maybe | Needs `sigma_theta`, `cores` in scope |

**To port one block (example: missing-date histogram):**

1. Uncomment logic from `legacy/removed_experiments/02_missing_date_histogram.py`
2. Wrap in `experiments/missing_date_histogram.py` calling package loaders
3. Parameterize paths via `configs/default.yaml`
4. Add characterization pin only if numerics matter

---

## Suggested review bundle (one shot)

Creates a `reports/alpha_review/` folder with txt dumps you can attach to a PR or email:

```bash
cd /unity/g2/jmiranda/v2-nespreso
mkdir -p reports/alpha_review
srun --ntasks=1 --cpus-per-task=8 --gres=gpu:1 \
  pytest tests/ -q -m "not requires_unity" | tee reports/alpha_review/unit_tests.txt
srun --ntasks=1 --cpus-per-task=8 --gres=gpu:1 \
  pytest tests/test_characterization.py -m requires_unity --run-unity -q \
  | tee reports/alpha_review/char_tests.txt
git log --oneline -10 | tee reports/alpha_review/recent_commits.txt
pip show nespreso | tee reports/alpha_review/package_info.txt
tar czf reports/alpha_review_bundle.tar.gz reports/
```

---

## Summary

Alpha `0.1.0a1` is on `main` with full refactor history. Nothing in the deferred list changes numerics until you explicitly run Phase B or the config/dataset split. The four human-review items are about **visual/grid correctness** (ISOP, binning) and **operational safety** (glider pickle mutation, archived dead code). Run the diagnostic commands above and sign off before touching production pickles or removing the deprecation shim.
