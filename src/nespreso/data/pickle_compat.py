"""Compatibility loading for dataset pickles saved from the monolith ``__main__`` block."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

# Class names stored as ``__main__.<name>`` in legacy pickles (pickletools audit, 2026-06).
_LEGACY_MAIN_CLASSES: dict[str, str] = {
    "TemperatureSalinityDataset": "nespreso.data.dataset.TemperatureSalinityDataset",
}


def _resolve_legacy_main_class(name: str) -> type:
    target = _LEGACY_MAIN_CLASSES.get(name)
    if target is None:
        raise pickle.UnpicklingError(f"Unknown legacy __main__ class: {name}")
    module_path, _, class_name = target.rpartition(".")
    module = __import__(module_path, fromlist=[class_name])
    return getattr(module, class_name)


class _LegacyDatasetUnpickler(pickle.Unpickler):
    """Remap monolith-era ``__main__`` class paths to ``nespreso`` package modules."""

    def find_class(self, module: str, name: str) -> type:
        if module == "__main__":
            if name in _LEGACY_MAIN_CLASSES:
                return _resolve_legacy_main_class(name)
        return super().find_class(module, name)


def load_dataset_pickle(pickle_path: str | Path) -> dict[str, Any]:
    """
    Load the training dataset pickle dict without importing the monolith.

    Legacy pickles reference ``__main__.TemperatureSalinityDataset`` because the
    monolith was executed as ``__main__``. The custom unpickler remaps that path
    to ``nespreso.data.dataset.TemperatureSalinityDataset``.
    """
    with open(pickle_path, "rb") as file:
        return _LegacyDatasetUnpickler(file).load()


def resave_dataset_pickle(
    source_path: str | Path,
    dest_path: str | Path,
    *,
    protocol: int | None = None,
) -> Path:
    """
    Re-write a legacy pickle so ``full_dataset`` is stored under the package class path.

    Safe to run on a copy first; verify with characterization tests after re-save.
    """
    data = load_dataset_pickle(source_path)
    dest = Path(dest_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as file:
        pickle.dump(data, file, protocol=protocol)
    return dest
