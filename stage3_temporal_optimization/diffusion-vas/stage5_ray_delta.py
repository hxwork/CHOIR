"""Helpers for Stage 5 camera-ray hand depth refinement."""

from __future__ import annotations

import torch


DEFAULT_STAGE5_RAY_MIN_DELTA = -0.05
DEFAULT_STAGE5_RAY_MAX_DELTA = 0.05


def build_stage5_ray_delta(
    raw_delta: torch.Tensor,
    ray_dirs: torch.Tensor,
    interaction_mask: torch.Tensor,
    *,
    min_delta: float = DEFAULT_STAGE5_RAY_MIN_DELTA,
    max_delta: float = DEFAULT_STAGE5_RAY_MAX_DELTA,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-frame translation delta vectors and scalar deltas."""
    delta = raw_delta.reshape(-1).clamp(float(min_delta), float(max_delta))
    mask = interaction_mask.to(device=delta.device, dtype=delta.dtype).reshape(-1)
    delta = delta * mask
    delta_vec = ray_dirs.to(device=delta.device, dtype=delta.dtype) * delta[:, None]
    return delta_vec, delta


def clamp_stage5_ray_delta_(
    raw_delta: torch.Tensor,
    *,
    min_delta: float = DEFAULT_STAGE5_RAY_MIN_DELTA,
    max_delta: float = DEFAULT_STAGE5_RAY_MAX_DELTA,
) -> None:
    """Clamp Stage 5 ray deltas in-place."""
    with torch.no_grad():
        raw_delta.clamp_(float(min_delta), float(max_delta))
