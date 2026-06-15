Here’s the handoff converted into a practical execution cookbook.

# Ponytail Refactor Cookbook

## Core rule

Do **not** add more structure until the verification floor exists.

The package is already lean enough for this stage. The next leverage point is not more modules. It is a working golden test for training behavior, especially `train_loss_trajectory` captured on HPC.

## Current state

Demand sprint is mostly built.

`src/nespreso/` contains roughly 1,700 lines. The monolith is still the trunk at roughly 4,400 lines. That is acceptable for now.

Phase 0 is incomplete:

* [ ] No first salvage commit.
* [ ] No formatting pass.
* [ ] `train_loss_trajectory` golden test is still a stub.
* [ ] Verified moves have not really started.

Phases 3–9 are untouched.

## Prime directive

Ship the verification gate before shipping more modules.

Every extraction from the monolith must be protected by goldens. Without the HPC `train_loss_trajectory` golden, later moves are blind.

---

# Phase 0: Verification floor

## 0.1 Make the first salvage commit

Create a baseline commit before more cleanup.

Checklist:

* [ ] Confirm current package imports work.
* [ ] Confirm monolith still runs through the existing path.
* [ ] Commit current working state as the salvage baseline.
* [ ] Do not bundle more refactor work into this commit.

Acceptance criterion:

* [ ] There is a clean commit representing the current demand sprint state.

---

## 0.2 Run formatting

Checklist:

* [ ] Run formatter over package and tests.
* [ ] Do not mix semantic changes with formatting.
* [ ] Commit format-only changes separately.

Acceptance criterion:

* [ ] Formatting diff contains no behavior changes.

---

## 0.3 Capture HPC golden: `train_loss_trajectory`

This is the critical gate.

Checklist:

* [ ] Run the short training path on HPC.
* [ ] Capture the loss trajectory.
* [ ] Store the expected golden output.
* [ ] Replace the always-skipped test with a real assertion.
* [ ] Confirm the test fails on intentionally changed behavior.
* [ ] Confirm the test passes on the current baseline.

Acceptance criterion:

* [ ] `test_short_train_loss_trajectory` is no longer a stub.
* [ ] Training loss behavior is protected before more monolith extraction.

Do not proceed to Phase 1–6 extractions until this passes.

---

# Phase 1: Immediate lean cleanup

These are low-risk deletions and shrinkage items. Prefer small commits grouped by theme.

## 1.1 Delete dead config method

File:

`config.py:L30-L31`

Task:

* [ ] Delete `BboxConfig.as_tuple()`.

Reason:

Never called.

Acceptance criterion:

* [ ] Tests pass.
* [ ] No references remain.

---

## 1.2 Remove duplicate density path source

Files:

`config.py:L42-L47`
`config.py:L95-L100`
`default.yaml:9-10`
`runner.py:L64-L65`

Problem:

Density paths exist in both `PathsConfig` and `DensityConfig`. Runner already overwrites them.

Task:

* [ ] Pick one source of truth.
* [ ] Remove the duplicate density path fields from the other config section.
* [ ] Keep runner behavior unchanged.
* [ ] Update tests and YAML accordingly.

Acceptance criterion:

* [ ] Density paths are defined once.
* [ ] Runner still resolves the same paths.

---

## 1.3 Remove unused `isop_nc` from typed config

Files:

`config.py:L48`
`default.yaml:11`

Problem:

`cfg.paths.isop_nc` is not read. The monolith hardcodes the same path around `singleFileModel…:1789`.

Task:

* [ ] Delete `isop_nc` from typed config.
* [ ] Delete it from `default.yaml`.
* [ ] Confirm no imports or references break.

Acceptance criterion:

* [ ] No `cfg.paths.isop_nc` references remain.

---

## 1.4 Remove unused runtime flags

Files:

`config.py:L145-L146`
`default.yaml:65-L66`

Problem:

`bin_size` and `num_samples` are in `RuntimeFlags`, but runner never applies them. Only monolith `__main__` locals use them.

Task:

* [ ] Delete `bin_size` from `RuntimeFlags`.
* [ ] Delete `num_samples` from `RuntimeFlags`.
* [ ] Delete corresponding YAML defaults.
* [ ] Leave monolith locals alone until config reconciliation.

Acceptance criterion:

* [ ] Runner behavior unchanged.
* [ ] Runtime config contains only fields runner actually uses.

---

## 1.5 Replace hand-rolled `as_dict()`

Files:

`config.py:L77-L89`
`config.py:L106-L115`

Problem:

Frozen dataclasses have custom `as_dict()` implementations.

Task:

* [ ] Replace custom `as_dict()` methods with `dataclasses.asdict(self)`, or stop converting and pass config objects directly.
* [ ] Prefer the smaller implementation.
* [ ] Keep output shape identical where callers depend on dictionaries.

Acceptance criterion:

* [ ] No custom dataclass dictionary conversion remains unless strictly necessary.

---

## 1.6 Remove config triple source of truth

Files:

`config.py`
`default.yaml`
monolith `__main__`

Problem:

Defaults exist in three places:

1. Dataclass defaults.
2. YAML defaults.
3. Monolith inline `__main__` knobs.

Task:

* [ ] Choose the next source-of-truth reduction.
* [ ] Drop duplicated defaults from either YAML or dataclass defaults.
* [ ] Do not create a fourth source.
* [ ] Do not do a full config reconcile yet.

Preferred next step:

* [ ] Move monolith `__main__` toward `load_config()` + `run_training()`.
* [ ] Delete the inline duplicate knob block around `singleFileModel…:1516-L1554`.

Acceptance criterion:

* [ ] There are fewer duplicated config defaults.
* [ ] Config behavior remains equivalent.

---

# Phase 2: Test and loader cleanup

## 2.1 Remove duplicate monolith loader

Files:

`runner.py:L21-L34`
`test_characterization.py:L28-L35`

Problem:

`_load_monolith()` is duplicated.

Task:

* [ ] Create `tests/monolith_loader.py`.
* [ ] Move shared loader there.
* [ ] Import it from runner characterization tests as needed.
* [ ] Keep the helper around 12 lines.

Acceptance criterion:

* [ ] One monolith loader exists.
* [ ] Tests still locate the monolith correctly.

---

## 2.2 Replace or remove skipped golden stub

File:

`test_characterization.py:L82-L92`

Problem:

`test_short_train_loss_trajectory` always skips.

Task:

* [ ] Implement it after HPC capture.
* [ ] If HPC capture is not available yet, remove the empty test temporarily.
* [ ] Do not keep noise tests.

Acceptance criterion:

* [ ] No always-skipped placeholder test remains.

---

# Phase 3: Download code cleanup

## 3.1 Delete unused SSH downloader

File:

`io/download/copernicus.py:L10-L47`

Problem:

`download_ssh_year` has no CLI and no caller. AVISO uses `motu` via `aviso.py`, not this function.

Task:

* [ ] Delete `download_ssh_year`.
* [ ] Confirm no imports break.
* [ ] Confirm AVISO path still uses `aviso.py`.

Acceptance criterion:

* [ ] Dead SSH downloader is gone.

---

## 3.2 Merge SSS wrapper into Copernicus downloader

File:

`io/download/sss.py`

Problem:

The file is a 39-line wrapper around a `download_sss_day` loop.

Task:

* [ ] Move range behavior into `copernicus.py` as `download_sss_range()`.
* [ ] Update CLI to call `download_sss_range()` directly.
* [ ] Delete `sss.py`.

Acceptance criterion:

* [ ] SSS range download still works.
* [ ] Wrapper module is gone.

---

## 3.3 Remove download barrel exports

File:

`io/download/__init__.py`

Problem:

Re-export barrel is unnecessary. Only `cli.py` imports downloaders.

Task:

* [ ] Delete re-exports.
* [ ] Import downloader modules directly in `cli.py`.
* [ ] Keep imports explicit.

Acceptance criterion:

* [ ] No unnecessary barrel API remains.

---

## 3.4 Normalize bbox style

File:

`io/download/aviso.py:L12-L19`

Problem:

`DownloadBbox` dataclass exists only for AVISO while OSTIA and SSS take four floats.

Task:

* [ ] Delete `DownloadBbox`.
* [ ] Use one bbox style everywhere: `(min_lon, max_lon, min_lat, max_lat)`.
* [ ] Update AVISO caller and tests.

Acceptance criterion:

* [ ] All downloaders use the same bbox shape.

---

# Phase 4: CLI cleanup

## 4.1 Deduplicate bbox/date arguments

File:

`cli.py:L29-L45`

Problem:

Three products repeat the same bbox/date arguments.

Task:

* [ ] Add a shared helper such as `add_common_bbox(parser)`.
* [ ] Use parser parents or shared namespace setup.
* [ ] Remove repeated argument definitions.

Expected reduction:

* [ ] Around 25 lines.

Acceptance criterion:

* [ ] CLI behavior unchanged.
* [ ] Repeated bbox/date flags are declared once.

---

## 4.2 Replace custom date parser

File:

`cli.py:L70-L71`

Problem:

Custom `_parse_date` duplicates stdlib behavior.

Task:

* [ ] Replace `_parse_date` with `datetime.fromisoformat(value)` for `YYYY-MM-DD`.
* [ ] Preserve error behavior if tests expect it.

Acceptance criterion:

* [ ] Date parsing still accepts `YYYY-MM-DD`.
* [ ] Custom parser is gone.

---

# Phase 5: Keep as-is

Do not spend simplification effort here.

## Keep optional extras

File:

`pyproject.toml:L29-L32`

Checklist:

* [ ] Keep `[download]` extra.
* [ ] Keep `[monitor]` extra.

Reason:

Lazy dependencies are correct. They avoid bloating core installs.

---

## Keep lean extractions

Files:

`train.py`
`metrics.py`
`determinism.py`
`utils/time.py`

Checklist:

* [ ] Keep these modules.
* [ ] Do not inline them back into the monolith.
* [ ] Preserve monolith imports.

Reason:

These are right-sized extractions.

---

## Keep vendored science code

File:

`physics_metrics.py`

Checklist:

* [ ] Do not delete or aggressively refactor.
* [ ] Treat as vendored science code.

Reason:

At 863 lines it is large, but not simplification bloat.

---

## Keep demand sprint docs

Files:

`docs/HOWTO.md`
`docs/NEW_BASIN.md`

Checklist:

* [ ] Keep both docs.

Reason:

They are deliverables, not bloat.

---

# Phase 6: Explicit deferrals

Do not build these now.

## Defer SQLite/HDF5 ingest

Plan reference:

`plan.md:Step4`

Decision:

* [ ] Do not build `nespreso.ingest` yet.
* [ ] Continue using monolith `.mat` + pickle path.
* [ ] Reconsider only when `NEW_BASIN` becomes real operational work.

Reason:

Nobody is training a second basin yet.

---

## Defer checkpoint/resume/best-model machinery

Plan reference:

`plan.md:Step4`

Decision:

* [ ] Do not build checkpoint/resume flags now.
* [ ] Keep one checkpoint save matching current behavior.
* [ ] Add machinery only when training runs are long enough that failed runs hurt.

Reason:

Current behavior does not justify lifecycle machinery.

---

## Defer full legacy/source architecture docs

Plan reference:

`plan.md:Tier3`

Decision:

* [ ] Do not write full `legacy/SOURCES.md`.
* [ ] Do not write full `ARCHITECTURE.md`.
* [ ] Add only a stub `SOURCES.md` at the first salvage commit.
* [ ] Expand in Phase 8 or 9.

Reason:

Early exhaustive docs will rot before verified moves land.

---

## Defer full config reconciliation

Plan reference:

`phase2.txt`

Decision:

* [ ] Do not reconcile YAML, dataclasses, and monolith all at once.
* [ ] Do not create another config copy.
* [ ] Next move is specifically: monolith `__main__` → `load_config()` + `run_training()`.

Reason:

The duplicated monolith block is the real next target.

---

# Phase 7: Complexity worth keeping

## PCA dedup remains test-first

Plan reference:

`phase4.txt`

Checklist:

* [ ] Keep PCA dedup work test-first.
* [ ] Do not preemptively generalize.
* [ ] Add complexity only behind characterization tests.

Reason:

This is the right kind of complexity: science behavior protected by tests.

---

# Recommended commit order

## Commit 1: Salvage baseline

* [ ] Current working state.
* [ ] No behavior changes.

## Commit 2: Format-only pass

* [ ] Formatter output only.
* [ ] No semantic edits.

## Commit 3: Remove dead config fields

* [ ] Delete `BboxConfig.as_tuple()`.
* [ ] Delete unused `isop_nc`.
* [ ] Delete unused `bin_size` and `num_samples`.
* [ ] Replace or simplify `as_dict()`.

## Commit 4: Config source-of-truth reduction

* [ ] Remove density path duplication.
* [ ] Reduce YAML/dataclass duplicate defaults.
* [ ] Preserve runner behavior.

## Commit 5: Shared monolith loader

* [ ] Add `tests/monolith_loader.py`.
* [ ] Remove duplicate `_load_monolith()`.

## Commit 6: Downloader cleanup

* [ ] Delete `download_ssh_year`.
* [ ] Merge SSS range wrapper.
* [ ] Remove download barrel exports.
* [ ] Normalize bbox tuple style.

## Commit 7: CLI cleanup

* [ ] Deduplicate bbox/date args.
* [ ] Replace custom date parser with `datetime.fromisoformat`.

## Commit 8: HPC golden

* [ ] Capture `train_loss_trajectory`.
* [ ] Implement real golden test.
* [ ] Delete skip stub.

This commit is the gate before more extraction.

---

# Definition of done

The cleanup is done when:

* [ ] Current behavior is preserved.
* [ ] No always-skipped golden test remains.
* [ ] `train_loss_trajectory` is captured and asserted.
* [ ] Dead download code is deleted.
* [ ] Config has fewer duplicate sources of truth.
* [ ] CLI repetition is reduced.
* [ ] One monolith loader exists.
* [ ] Optional extras remain optional.
* [ ] No new ingest, checkpoint, experiment, or architecture-doc machinery is added.

---

# Red flags

Stop and reassess if any of these happen:

* [ ] A new module appears before the HPC golden passes.
* [ ] YAML, dataclasses, and monolith all define the same default again.
* [ ] A skipped characterization test is added “temporarily.”
* [ ] Checkpoint/resume machinery appears before long failed runs become a real pain.
* [ ] SQLite/HDF5 ingest appears before a second basin is real.
* [ ] Refactor commits mix formatting, deletion, and behavior changes.
