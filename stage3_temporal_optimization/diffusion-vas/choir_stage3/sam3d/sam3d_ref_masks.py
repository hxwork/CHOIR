"""Helpers for SAM3D reference-frame object masks."""

from __future__ import annotations

import os
from typing import Optional

import cv2
import numpy as np

from choir_stage3.io.video_layout import VideoLayout


_IMAGE_EXTS = (".png", ".jpg", ".jpeg")


def resolve_sam3d_ref_obj_mask_path(seq_path: str, clip_frame_idx: int) -> Optional[str]:
    """Return the object-mask path for a SAM3D reference frame."""
    mask_dir = str(VideoLayout.from_root(seq_path).object_masks_dir)
    frame = int(clip_frame_idx)
    for ext in _IMAGE_EXTS:
        path = os.path.join(mask_dir, f"{frame}{ext}")
        if os.path.exists(path):
            return path
    return None


def load_sam3d_ref_obj_mask(seq_path: str, clip_frame_idx: int) -> tuple[Optional[np.ndarray], Optional[str]]:
    """Load a binary object mask for a SAM3D reference frame from object masks."""
    path = resolve_sam3d_ref_obj_mask_path(seq_path, clip_frame_idx)
    if path is None:
        return None, None

    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None, path
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, 3]
    elif img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return (img > 128).astype(np.uint8), path
