"""Frame sampling helpers."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def build_adaptive_sample_indices(
    num_total_frames: int,
    *,
    target_stride: int = 3,
    min_sampled_frames: int = 64,
    max_sampled_frames: int = 128,
) -> tuple[list[int], dict[str, Any]]:
    """Build sampled frame indices from a target original-frame stride."""
    n = max(0, int(num_total_frames))
    stride = max(1, int(target_stride))
    min_frames = max(1, int(min_sampled_frames))
    max_frames = max(min_frames, int(max_sampled_frames))

    if n <= 0:
        indices: list[int] = []
        target = 0
        mode = "empty"
    elif n <= min_frames:
        indices = list(range(n))
        target = n
        mode = "all_frames"
    else:
        target = int(math.ceil(n / float(stride)))
        target = max(min_frames, min(max_frames, target))
        target = min(n, target)
        indices = np.linspace(0, n - 1, target, dtype=int).tolist()
        indices = sorted(set(int(i) for i in indices))
        if indices[0] != 0:
            indices.insert(0, 0)
        if indices[-1] != n - 1:
            indices.append(n - 1)
        mode = "adaptive_stride"

    info = {
        "mode": mode,
        "num_total_frames": int(n),
        "num_sampled_frames": int(len(indices)),
        "target_num_sampled_frames": int(target),
        "target_stride": int(stride),
        "min_sampled_frames": int(min_frames),
        "max_sampled_frames": int(max_frames),
        "sampled_indices": [int(i) for i in indices],
    }
    return indices, info
