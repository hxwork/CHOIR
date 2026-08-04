"""Stage 5 optimization mode helpers."""

from __future__ import annotations


STAGE5_FULL = "full"
STAGE5_FROZEN = "frozen"
STAGE5_OBJECT_LITE = "object_lite"
STAGE5_POSE_ONLY = "pose_only"
STAGE5_POSE_RAY = "pose_ray"
STAGE5_MODES = (STAGE5_FULL, STAGE5_FROZEN, STAGE5_OBJECT_LITE, STAGE5_POSE_ONLY, STAGE5_POSE_RAY)


def stage5_trainable_params(mode: str) -> tuple[str, ...]:
    """Return logical parameter groups optimized by a Stage 5 mode."""
    if mode == STAGE5_FULL:
        return ("object", "mano_pose", "mano_root")
    if mode == STAGE5_FROZEN:
        return ()
    if mode == STAGE5_OBJECT_LITE:
        return ("object", "mano_pose")
    if mode == STAGE5_POSE_ONLY:
        return ("mano_pose",)
    if mode == STAGE5_POSE_RAY:
        return ("mano_pose", "hand_ray_delta")
    raise ValueError(f"Unknown Stage 5 mode: {mode!r}")


def stage5_hand2d_weight(mode: str, default_weight: float = 5e-1) -> float:
    """Return Stage 5 hand-2D loss weight for an optimization mode."""
    if mode in (STAGE5_FROZEN, STAGE5_OBJECT_LITE, STAGE5_POSE_ONLY, STAGE5_POSE_RAY):
        return 0.0
    if mode == STAGE5_FULL:
        return float(default_weight)
    raise ValueError(f"Unknown Stage 5 mode: {mode!r}")


def stage5_hand2d_grad_groups(mode: str) -> tuple[str, ...]:
    """Return logical parameter groups receiving hand-2D gradients."""
    if mode == STAGE5_FULL:
        return ("mano_pose", "mano_root")
    if mode == STAGE5_FROZEN:
        return ()
    if mode == STAGE5_OBJECT_LITE:
        return ()
    if mode == STAGE5_POSE_ONLY:
        return ()
    if mode == STAGE5_POSE_RAY:
        return ()
    raise ValueError(f"Unknown Stage 5 mode: {mode!r}")
