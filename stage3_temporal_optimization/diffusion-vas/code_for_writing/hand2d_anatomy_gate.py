"""Utilities for anatomy-driven hand 2D loss gating."""

from __future__ import annotations

from typing import Any

import torch


def compute_anatomy_hand2d_frame_weights(
    anatomy_scores: torch.Tensor,
    *,
    extreme_z: float = 6.0,
    min_weight: float = 0.0,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Convert per-frame anatomy scores into detached hand2d frame weights.

    Only extreme anatomy scores mark frames where 2D joints likely pushed the
    hand into implausible articulation. Those frames are removed from the 2D
    hand reprojection loss and can be repaired from neighboring plausible
    frames before later optimisation stages.
    """
    if anatomy_scores.ndim != 1:
        raise ValueError(f"anatomy_scores must be 1D, got shape {tuple(anatomy_scores.shape)}")
    if not (0.0 <= min_weight <= 1.0):
        raise ValueError(f"min_weight must be in [0, 1], got {min_weight}")
    if extreme_z <= 0.0:
        raise ValueError(f"extreme_z must be > 0, got {extreme_z}")

    scores = anatomy_scores.detach().float()
    if scores.numel() == 0:
        return scores.clone(), {
            "median": 0.0,
            "mad": 0.0,
            "robust_scale": 0.0,
            "extreme_threshold": 0.0,
            "extreme_z": float(extreme_z),
            "n_low": 0,
            "low_indices": [],
            "extreme_indices": [],
        }

    finite = torch.isfinite(scores)
    if not finite.any():
        weights = torch.ones_like(scores)
        return weights, {
            "median": 0.0,
            "mad": 0.0,
            "robust_scale": 0.0,
            "extreme_threshold": 0.0,
            "extreme_z": float(extreme_z),
            "n_low": 0,
            "low_indices": [],
            "extreme_indices": [],
        }

    valid_scores = scores[finite]
    median = valid_scores.median()
    abs_dev = (valid_scores - median).abs()
    mad = abs_dev.median()
    if float(mad.item()) <= eps:
        q25 = torch.quantile(valid_scores, 0.25)
        q75 = torch.quantile(valid_scores, 0.75)
        robust_scale = (q75 - q25) / 1.349
    else:
        robust_scale = mad * 1.4826

    if float(robust_scale.item()) <= eps:
        weights = torch.ones_like(scores)
        return weights, {
            "median": float(median.item()),
            "mad": float(mad.item()),
            "robust_scale": float(robust_scale.item()),
            "extreme_threshold": float(median.item()),
            "extreme_z": float(extreme_z),
            "n_low": 0,
            "low_indices": [],
            "extreme_indices": [],
        }

    z = ((scores - median) / robust_scale).clamp(min=0.0)
    extreme = z >= float(extreme_z)
    weights = torch.where(extreme, torch.full_like(scores, float(min_weight)), torch.ones_like(scores))
    weights = torch.where(finite, weights, torch.ones_like(weights))
    low = finite & extreme
    low_indices = torch.nonzero(low, as_tuple=False).flatten().detach().cpu().tolist()

    return weights.detach(), {
        "median": float(median.item()),
        "mad": float(mad.item()),
        "robust_scale": float(robust_scale.item()),
        "extreme_threshold": float((median + extreme_z * robust_scale).item()),
        "extreme_z": float(extreme_z),
        "n_low": int(low.sum().item()),
        "low_indices": [int(i) for i in low_indices],
        "extreme_indices": [int(i) for i in low_indices],
    }
