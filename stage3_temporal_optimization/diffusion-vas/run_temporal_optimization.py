#!/usr/bin/env python3
"""CLI wrapper for CHOIR Stage 3 temporal optimization.

Implementation lives in ``choir_stage3.entrypoints.demo_fitting_5stages_sam3d_reset``.
"""

from __future__ import annotations

import runpy
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

if __name__ == "__main__":
    runpy.run_module(
        "choir_stage3.entrypoints.demo_fitting_5stages_sam3d_reset",
        run_name="__main__",
    )
