# Monolith relic

Frozen snapshot of `singleFileModel_SAT_stats4verticalProj_meeting20260203.py` from
**before** Phase 9 dead-code removal (parent of commit `2cbb2b1`, 470 lines).

This file is **not imported** by the package or tests. It is kept for provenance and
diff archaeology only.

## Forward entrypoints

| Task | Command |
|------|---------|
| Train only | `python -m nespreso train --config configs/default.yaml` |
| Train + all experiments | `python experiments/run_all.py --config configs/default.yaml` |
| Single experiment | `python experiments/<name>.py --config configs/default.yaml` |

The repo-root `singleFileModel_SAT_stats4verticalProj_meeting20260203.py` is a thin
deprecation shim that delegates to `experiments/run_all.py`.

## Removed `__main__` blocks

Commented exploratory code cut from this file during Phase 9 is archived under
`legacy/removed_experiments/`.
