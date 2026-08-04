"""Hold-video-specific Stage 3 hand fitting policy."""

from __future__ import annotations

import os


HOLD_MANO_TO_OPENPOSE = (0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20)


def is_hold_video_id(video_id: object) -> bool:
    """Return whether a video id belongs to the hold_* HO3D subset."""
    return str(video_id).startswith("hold_")


def stage3_joint2d_relative_weight(video_id: object, *, default: float = 1.0, hold: float = 5.0) -> float:
    """Return the Stage 3 EMA-normalized joint-2D relative weight."""
    return float(hold if is_hold_video_id(video_id) else default)


def hold_sam3d_rot_outlier_max_angle(video_id: object, *, default: float, hold: float = 60.0) -> float:
    """Return SAM3D dense-rotation outlier threshold for a video."""
    return float(hold if is_hold_video_id(video_id) else default)


def hold_stage3_hand_refine_steps(video_id: object, *, steps: int = 200) -> int:
    """Return extra Stage 3 hand-only refinement steps for hold videos."""
    return int(steps if is_hold_video_id(video_id) else 0)


def stage3_object_smoothness_scale(video_id: object, *, default: float = 1.0, hold: float = 0.0) -> float:
    """Return Stage 3 object temporal-smoothness scale."""
    return float(hold if is_hold_video_id(video_id) else default)


def hold_joint_target_indices() -> tuple[int, ...]:
    """Return target 2D joint indices matching AMANO's OpenPose-style output order."""
    return HOLD_MANO_TO_OPENPOSE


def hold_mano_init_path(seq_path: str) -> str:
    """Return the hold-fit MANO initialization path for a sequence directory."""
    return os.path.join(seq_path, "processed", "hold_fit.slerp.npy")


def hold_mano_init_param_groups() -> tuple[str, ...]:
    """Return MANO parameter groups overridden from hold-fit initialization."""
    return ("mano_pose",)


def hold_pose_hand_mean(flat_hand_mean):
    """Return MANO hands_mean in per-joint axis-angle shape."""
    return flat_hand_mean.reshape(15, 3)


def hold_pose_from_fit(hand_pose, hand_mean):
    """Return hold-fit hand pose in TemporalHandObjectPose's (N, 15, 3) shape."""
    return hand_pose.reshape(-1, 15, 3) + hand_mean.reshape(1, 15, 3)


def hold_reorder_valid_mask(valid_mask, joint_order):
    """Reorder joint-level masks; keep frame-level masks unchanged."""
    if valid_mask.ndim < 2:
        return valid_mask
    if hasattr(valid_mask, "index_select"):
        return valid_mask.index_select(1, joint_order)
    return valid_mask[:, joint_order]
