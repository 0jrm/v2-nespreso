# Project: modularizing a single-file ocean-ML research pipeline

You are refactoring `singleFileModel_SAT_stats4verticalProj_meeting20260203.py`
(~4,550 lines) into a typed, multi-file package. This is STRUCTURAL SURGERY,
NOT A REWRITE. Read these rules before every change.

## Non-negotiable rules

1. PRESERVE BEHAVIOR. Do not change any numerics, algorithm, hyperparameter,
   PCA component count, seed, or scientific variable name. Refactors must
   reproduce existing outputs. Move code verbatim first; adjust only imports.
2. VERIFY EVERY STEP against the characterization tests in tests/. If a move
   can't be verified (see "Execution environment" below), say so explicitly in
   your summary and leave a TODO marker for human numerical verification.
3. ONE COMMIT PER LOGICAL MOVE. Keep the repo importable after each commit.
   Conventional messages, e.g. "refactor(io): extract satellite loaders".
4. NEVER delete code you don't understand. Research code has load-bearing
   oddities (MATLAB datenum +366 correction, sklearn warning suppression,
   the dataset.reload() dance). Preserve them. List anything suspicious under
   a "## Needs human review" heading instead of removing it.
5. Do NOT add features, upgrade dependencies, or reformat beyond the one
   agreed formatter pass. Keep diffs reviewable.
6. When you find duplication (e.g. inverse_transform exists 5x), DO NOT
   consolidate until a test proves the canonical version reproduces every
   call site. Flag duplicates; consolidate only in the assigned phase.

## Architecture target

```
nespreso/
  pyproject.toml
  configs/
  src/nespreso/
    cli.py
    config.py
    runner.py
    train.py
    metrics.py
    physics_metrics.py
    determinism.py
    utils/time.py
    io/satellite_readers.py
    io/download/
    ingest/          # phase infra-port
    data/            # phases 4+
    models/          # phase 5+
    inference.py     # phase 6+
    viz/             # phase 7+
  experiments/
  tests/
  docs/
  legacy/SOURCES.md
  singleFileModel_SAT_stats4verticalProj_meeting20260203.py  # trunk until phase 9
```

**Refactor base:** `singleFileModel_SAT_stats4verticalProj_meeting20260203.py` (superset of paper `singleFileModel_SAT.py`; adds `RhoMLP` / `DensityConstraint`).

## Principles

Library vs experiments separation; injectable config/paths (no /unity hardcoding,
no sys.path hacks — use packaging); one responsibility per ~150-400 line module;
pure computation returns data, side effects (plotting, file writes) at the edges;
centralize seeding + device.

## Execution environment

- **CAN RUN:** real data at `/unity/...` and GPU are available. Characterization tests
  capture real golden outputs; assert numerical equality within tolerance 1e-6
  (or exact where deterministic). Run them after every move.

## Workflow each task

Plan → make the smallest move → wire imports → run verification → report what you
did, what you verified, and what still needs human review. Then stop for review.

## Engineering principles pack

For general engineering guidance, read `eng-principles-pack/INDEX.md` first. Load only the hook and skill files whose triggers match the current task. Prefer lifecycle hooks at commit/PR/bugfix/numerical-merge moments, then at most one or two skills. Project skills are also available under `.cursor/skills/` and `.claude/skills/`.
