"""
Deprecated monolith entrypoint.

The research monolith lives under ``legacy/monolith/``. The forward pipeline is
``experiments/run_all.py`` (or ``python -m nespreso train`` for training only).
"""

from __future__ import annotations

import runpy
import warnings
from pathlib import Path

warnings.warn(
    "singleFileModel_SAT_stats4verticalProj_meeting20260203.py is deprecated; "
    "use: python experiments/run_all.py",
    DeprecationWarning,
    stacklevel=1,
)

if __name__ == "__main__":
    target = Path(__file__).resolve().parent / "experiments" / "run_all.py"
    runpy.run_path(str(target), run_name="__main__")
