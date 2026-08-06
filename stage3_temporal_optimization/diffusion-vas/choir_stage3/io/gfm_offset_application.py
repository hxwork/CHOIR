"""Utilities for applying GraspFlowMatching camera-ray depth offsets."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch


def _smooth_depth_offsets_in_range(
    offsets: torch.Tensor,
    valid: torch.Tensor,
    smooth_range: tuple[int, int] | None,
    kernel: Sequence[float],
) -> torch.Tensor:
    """Smooth scalar offsets only inside the requested half-open frame range."""
    smoothed = offsets.clone()
    n = int(offsets.numel())
    if n == 0:
        return smoothed
    lo, hi = (0, n) if smooth_range is None else (int(smooth_range[0]), int(smooth_range[1]))
    lo = max(0, min(n, lo))
    hi = max(lo, min(n, hi))
    if hi <= lo:
        return smoothed
    kernel_t = torch.as_tensor(kernel, dtype=offsets.dtype, device=offsets.device).reshape(-1)
    if kernel_t.numel() == 0 or float(kernel_t.abs().sum().item()) <= 0.0:
        return smoothed
    radius = int(kernel_t.numel() // 2)
    for idx in range(lo, hi):
        if not bool(valid[idx].item()):
            continue
        numer = torch.zeros((), dtype=offsets.dtype, device=offsets.device)
        denom = torch.zeros((), dtype=offsets.dtype, device=offsets.device)
        for k_idx, weight in enumerate(kernel_t):
            src = idx + k_idx - radius
            if src < lo or src >= hi or not bool(valid[src].item()):
                continue
            numer = numer + offsets[src] * weight
            denom = denom + weight
        if float(denom.abs().item()) > 0.0:
            smoothed[idx] = numer / denom
    return smoothed


def apply_camera_ray_depth_offsets(
    mano_trans: torch.Tensor,
    sampled_indices: Sequence[int],
    depth_offsets: Mapping[str, Any],
    *,
    smooth_offsets: bool = False,
    smooth_range: tuple[int, int] | None = None,
    smooth_kernel: Sequence[float] = (1.0, 4.0, 6.0, 4.0, 1.0),
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Apply per-frame depth offsets along normalize(-mano_trans).

    Offset lookup prefers original frame ids from ``sampled_indices`` and falls
    back to sampled positions for legacy files.
    """
    corrected = mano_trans.detach().clone()
    camera_rays = -mano_trans.detach()
    ray_norm = camera_rays.norm(dim=-1)
    ray_dirs = torch.nn.functional.normalize(camera_rays, dim=-1, eps=1e-8)
    invalid_ray_mask = ray_norm < 1e-8

    matched_by_frame = 0
    matched_by_position = 0
    missing_count = 0
    skipped_invalid_ray_count = 0
    applied_depth_mags: list[float] = []
    raw_depth_mags: list[float] = []
    offset_values = torch.zeros(corrected.shape[0], dtype=mano_trans.dtype, device=mano_trans.device)
    offset_valid = torch.zeros(corrected.shape[0], dtype=torch.bool, device=mano_trans.device)

    for sp, frame_idx in enumerate(sampled_indices):
        k_frame = str(int(frame_idx))
        k_pos = str(int(sp))
        if k_frame in depth_offsets:
            depth_mag = float(depth_offsets[k_frame])
            matched_by_frame += 1
        elif k_pos in depth_offsets:
            depth_mag = float(depth_offsets[k_pos])
            matched_by_position += 1
        else:
            missing_count += 1
            continue

        if bool(invalid_ray_mask[sp].item()):
            skipped_invalid_ray_count += 1
            continue

        offset_values[sp] = depth_mag
        offset_valid[sp] = True
        raw_depth_mags.append(depth_mag)

    applied_offsets = (
        _smooth_depth_offsets_in_range(offset_values, offset_valid, smooth_range, smooth_kernel)
        if smooth_offsets
        else offset_values
    )

    for sp in range(corrected.shape[0]):
        if not bool(offset_valid[sp].item()):
            continue
        depth_mag = float(applied_offsets[sp].item())
        corrected[sp] = corrected[sp] + ray_dirs[sp] * depth_mag
        applied_depth_mags.append(depth_mag)

    if applied_depth_mags:
        mags = torch.tensor(applied_depth_mags, dtype=mano_trans.dtype)
        mean_abs = float(mags.abs().mean().item())
        max_abs = float(mags.abs().max().item())
    else:
        mean_abs = 0.0
        max_abs = 0.0
    if raw_depth_mags:
        raw_mags = torch.tensor(raw_depth_mags, dtype=mano_trans.dtype)
        raw_mean_abs = float(raw_mags.abs().mean().item())
        smoothing_delta = (applied_offsets - offset_values).abs()[offset_valid]
        mean_abs_smoothing_delta = float(smoothing_delta.mean().item()) if smoothing_delta.numel() > 0 else 0.0
    else:
        raw_mean_abs = 0.0
        mean_abs_smoothing_delta = 0.0

    stats = {
        "total_count": int(corrected.shape[0]),
        "applied_count": int(len(applied_depth_mags)),
        "matched_by_frame": int(matched_by_frame),
        "matched_by_position": int(matched_by_position),
        "missing_count": int(missing_count),
        "skipped_invalid_ray_count": int(skipped_invalid_ray_count),
        "mean_abs_depth_offset": mean_abs,
        "max_abs_depth_offset": max_abs,
        "smoothed": bool(smooth_offsets),
        "smooth_range": [int(smooth_range[0]), int(smooth_range[1])] if smooth_range is not None else None,
        "raw_mean_abs_depth_offset": raw_mean_abs,
        "smoothed_mean_abs_depth_offset": mean_abs,
        "mean_abs_smoothing_delta": mean_abs_smoothing_delta,
    }
    return corrected, stats
