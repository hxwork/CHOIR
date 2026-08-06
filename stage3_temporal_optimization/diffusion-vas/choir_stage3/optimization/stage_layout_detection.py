"""Helpers for detecting whether a video has full stages or interaction only."""

from __future__ import annotations

from typing import Any


LAYOUT_FIVE_STAGE = "five_stage"
LAYOUT_INTERACTION_ONLY = "interaction_only"


def is_hold_video_id(video_id: object) -> bool:
    """Return whether a video id belongs to the hold_* HO3D subset."""
    return str(video_id).startswith("hold_")


def classify_stage_layout_from_interaction_bounds(
    *,
    num_sampled_frames: int,
    interaction_start: int,
    interaction_end: int,
    short_side_threshold: int = 5,
) -> dict[str, Any]:
    """Classify layout from detected interaction bounds in sampled-frame space."""
    n = max(0, int(num_sampled_frames))
    lo = max(0, min(int(interaction_start), n))
    hi = max(lo, min(int(interaction_end), n))
    threshold = max(0, int(short_side_threshold))

    pre_len = lo
    post_len = n - hi
    is_interaction_only = pre_len < threshold and post_len < threshold
    return {
        "layout_mode": LAYOUT_INTERACTION_ONLY if is_interaction_only else LAYOUT_FIVE_STAGE,
        "reason": "short_pre_and_post" if is_interaction_only else "has_context_before_or_after",
        "pre_interaction_len": int(pre_len),
        "post_interaction_len": int(post_len),
        "interaction_start": int(lo),
        "interaction_end": int(hi),
        "num_sampled_frames": int(n),
        "short_side_threshold": int(threshold),
    }


def apply_hold_interaction_only_override(
    layout: dict[str, Any],
    *,
    video_id: object,
    num_sampled_frames: int,
) -> dict[str, Any]:
    """Force hold_* videos to use interaction-only layout."""
    if not is_hold_video_id(video_id):
        return layout
    n = max(0, int(num_sampled_frames))
    overridden = dict(layout)
    overridden.update(
        {
            "layout_mode": LAYOUT_INTERACTION_ONLY,
            "reason": "hold_video_forced_interaction_only",
            "pre_interaction_len": 0,
            "post_interaction_len": 0,
            "interaction_start": 0,
            "interaction_end": n,
            "num_sampled_frames": n,
            "hold_forced_interaction_only": True,
        }
    )
    return overridden
