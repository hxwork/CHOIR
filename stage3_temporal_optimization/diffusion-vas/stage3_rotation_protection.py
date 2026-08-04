"""Helpers for protecting Stage 3 SAM3D object rotations."""

from __future__ import annotations

from typing import Any

import torch


DEFAULT_STAGE3_ROT_SMOOTHNESS_SCALE = 0.05


def stage3_rotation_lr_for_step(
    step: int,
    total_steps: int,
    base_lr: float,
    *,
    freeze_fraction: float = 0.65,
    unlock_lr_scale: float = 0.05,
    protect_rotation: bool = True,
) -> float:
    """Return the Stage 3 object-rotation lr for a two-phase schedule."""
    if not protect_rotation:
        return float(base_lr)
    if total_steps <= 0:
        return 0.0
    unlock_step = int(round(float(total_steps) * float(freeze_fraction)))
    if step < unlock_step:
        return 0.0
    return float(base_lr) * float(unlock_lr_scale)


def stage3_rotation_smoothness_scale(scale: float = DEFAULT_STAGE3_ROT_SMOOTHNESS_SCALE) -> float:
    """Return the light Stage 3 rotation smoothness multiplier."""
    return max(0.0, float(scale))


def rotation_drift_degrees(current_R: torch.Tensor, anchor_R: torch.Tensor) -> dict[str, Any]:
    """Summarize geodesic rotation drift between current and anchor rotations."""
    if current_R.shape != anchor_R.shape:
        raise ValueError(f"current_R and anchor_R must share shape, got {current_R.shape} vs {anchor_R.shape}")
    if current_R.ndim != 3 or current_R.shape[-2:] != (3, 3):
        raise ValueError(f"rotation tensors must have shape (N, 3, 3), got {tuple(current_R.shape)}")
    if current_R.numel() == 0:
        return {"mean_deg": 0.0, "max_deg": 0.0}

    rel = current_R @ anchor_R.transpose(-1, -2)
    trace = rel.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    cos_theta = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
    angles_deg = torch.rad2deg(torch.acos(cos_theta))
    return {
        "mean_deg": float(angles_deg.mean().detach().item()),
        "max_deg": float(angles_deg.max().detach().item()),
    }
