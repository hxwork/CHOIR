#!/usr/bin/env python3
"""Compatibility CLI wrapper for two-hand HOI merge (see CHOIR README).

Implementation: ``choir_stage3.entrypoints.convert_hoi_for_two_hand_one_obj``.
"""

from __future__ import annotations

from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from choir_stage3.entrypoints.convert_hoi_for_two_hand_one_obj import main

if __name__ == "__main__":
    raise SystemExit(main())
