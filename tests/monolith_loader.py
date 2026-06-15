"""Shared monolith import helper for runner and characterization tests."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def load_monolith():
    """Import the research monolith from the repo root."""
    main = sys.modules.get("__main__")
    if main is not None:
        main_file = getattr(main, "__file__", None)
        if main_file and Path(main_file).name == "singleFileModel_SAT_stats4verticalProj_meeting20260203.py":
            return main

    root = Path(__file__).resolve().parents[1]
    monolith_path = root / "singleFileModel_SAT_stats4verticalProj_meeting20260203.py"
    if not monolith_path.exists():
        raise FileNotFoundError(f"Monolith not found: {monolith_path}")

    spec = importlib.util.spec_from_file_location("nespreso_monolith", monolith_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load monolith from {monolith_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["nespreso_monolith"] = module
    spec.loader.exec_module(module)
    return module
