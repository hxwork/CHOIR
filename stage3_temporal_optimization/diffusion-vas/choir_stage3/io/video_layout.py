"""Re-export CHOIR repo ``output_layout.VideoLayout`` for Stage 3 modules."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from output_layout import VideoLayout, discover_video_roots  # noqa: E402

__all__ = ["VideoLayout", "discover_video_roots"]
