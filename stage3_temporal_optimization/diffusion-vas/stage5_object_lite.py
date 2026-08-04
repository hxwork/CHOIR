"""Helpers for Stage 5 object-lite optimization."""

from __future__ import annotations

import torch


DEFAULT_OBJECT_LITE_MAX_TRANS_DELTA = 0.04


def project_object_translation_delta_(
    trans: torch.Tensor,
    anchor_trans: torch.Tensor,
    *,
    max_delta: float = DEFAULT_OBJECT_LITE_MAX_TRANS_DELTA,
) -> None:
    """Project object translations into a per-frame ball around the anchor."""
    max_delta_f = float(max_delta)
    if max_delta_f <= 0.0:
        with torch.no_grad():
            trans.copy_(anchor_trans.to(device=trans.device, dtype=trans.dtype))
        return
    with torch.no_grad():
        anchor = anchor_trans.to(device=trans.device, dtype=trans.dtype)
        delta = trans - anchor
        norm = torch.linalg.norm(delta, dim=-1, keepdim=True)
        scale = (max_delta_f / norm.clamp(min=1e-12)).clamp(max=1.0)
        trans.copy_(anchor + delta * scale)
