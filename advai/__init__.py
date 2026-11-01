"""CSBP711 final project: early warning of Poor sprints.

Importing this package puts my copied Phase 1-3 pipeline (folder ``pipeline/``)
on the path, so ``phase3``, ``phase2`` and the Phase 1 ``src`` package can be used.
The pipeline is copied unchanged from my Software Engineering project
(github.com/hani-ziad/SwEng_202610) so that this repository runs on its own.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "pipeline"
DATA = ROOT / "data"
RESULTS = ROOT / "results"
SEED = 42

_p3 = str(PIPELINE / "phase-3-in-sprint-early-warning")
if _p3 not in sys.path:
    sys.path.insert(0, _p3)

import phase3  # noqa: E402,F401  (also loads phase2 and the Phase 1 src package)
