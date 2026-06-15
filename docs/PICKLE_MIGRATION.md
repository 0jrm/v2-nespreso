# Dataset pickle migration (monolith retirement)

## Problem

The GoM training pickle (`paths.dataset_pickle`, default
`config_dataset_full.pkl`) was created by running the monolith as a script.
Pickle therefore stores the dataset object as:

```
__main__.TemperatureSalinityDataset
```

At unpickle time Python must resolve that class. The old workaround swapped
`sys.modules["__main__"]` to the monolith module (which re-exported the package
class). That tied production code (`runner.py`, `glider_mission.py`) to the
monolith file.

## Current fix (Phase A — no file rewrite)

`nespreso.data.pickle_compat.load_dataset_pickle` uses a custom
`pickle.Unpickler.find_class` hook:

| Legacy path | Resolved to |
|-------------|-------------|
| `__main__.TemperatureSalinityDataset` | `nespreso.data.dataset.TemperatureSalinityDataset` |

`runner.py` and `glider_mission.py` now call this loader directly. New pickles
written by `run_training` already embed the package class path because
`TemperatureSalinityDataset` is imported from `nespreso.data.dataset`.

Verified on HPC: `load_dataset_pickle` reproduces the same `full_dataset[0]`
inputs/labels as the monolith `__main__` swap (characterization goldens).

## Optional re-save (Phase B)

When you want on-disk pickles to drop the `__main__` reference entirely:

```python
from pathlib import Path
from nespreso.config import load_config
from nespreso.data.pickle_compat import resave_dataset_pickle

cfg = load_config()
src = Path(cfg.paths.dataset_pickle)
dst = src.with_name(src.stem + "_nespreso.pkl")
resave_dataset_pickle(src, dst)
```

Then point `paths.dataset_pickle` at the new file (or atomically replace after
backup). Re-run:

```bash
srun --ntasks=1 --cpus-per-task=8 --gres=gpu:1 \
  pytest tests/test_characterization.py -m requires_unity --run-unity -q
```

## Retirement sequence

1. **Done:** package imports in `runner.py`; compat loader for legacy path.
2. **Optional:** re-save pickle(s) with `resave_dataset_pickle`.
3. **Later:** migrate characterization tests off `tests/monolith_loader.py`.
4. **Last:** delete monolith file once no importer remains.

Do **not** delete the monolith until steps 3–4 are complete and goldens pass.
