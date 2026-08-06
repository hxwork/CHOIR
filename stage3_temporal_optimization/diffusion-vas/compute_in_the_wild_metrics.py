#!/usr/bin/env python3
"""Compatibility CLI wrapper for in-the-wild metrics.

Implementation lives in ``choir_stage3.entrypoints.compute_in_the_wild_metrics``.
"""

from __future__ import annotations

from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from choir_stage3.entrypoints.compute_in_the_wild_metrics import main

if __name__ == "__main__":
    main()
