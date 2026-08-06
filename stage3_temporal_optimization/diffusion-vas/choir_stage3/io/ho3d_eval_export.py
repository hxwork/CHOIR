"""Export CHOIR Stage 3 predictions in the MagicHOI/HOLD eval-data schema."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import torch

PathLike = Union[str, Path]

MANO_TO_OPENPOSE = (0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20)
OPENPOSE_TO_MANO = tuple(MANO_TO_OPENPOSE.index(i) for i in range(21))


def _as_float_tensor(value) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().float()
    return torch.from_numpy(np.asarray(value)).float()


def _as_long_tensor(value) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().long()
    return torch.from_numpy(np.asarray(value)).long()


def save_magic_hoi_eval_data(
    *,
    output_path: PathLike,
    seq_name: str,
    intrinsics,
    hand_verts,
    hand_joints_openpose,
    hand_faces,
    obj_verts,
    obj_faces,
) -> Path:
    """Write ``eval_data.npy`` compatible with MagicHOI/HOLD-style metrics.

    ``run_amano`` returns 21 hand joints in the OpenPose-style order used by the
    Stage 3 2D targets. MagicHOI metrics expect MANO joint order, so this helper
    performs the same inverse remapping as the legacy ``save_eval_data`` path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    hand_verts_t = _as_float_tensor(hand_verts)
    hand_joints_openpose_t = _as_float_tensor(hand_joints_openpose)
    obj_verts_t = _as_float_tensor(obj_verts)
    hand_faces_t = _as_long_tensor(hand_faces)
    obj_faces_t = _as_long_tensor(obj_faces)

    if hand_verts_t.ndim != 3 or hand_joints_openpose_t.ndim != 3 or obj_verts_t.ndim != 3:
        raise ValueError("hand_verts, hand_joints_openpose, and obj_verts must be frame-major 3D arrays")
    if hand_joints_openpose_t.shape[1] != 21:
        raise ValueError(f"expected 21 hand joints, got {hand_joints_openpose_t.shape[1]}")
    if not (hand_verts_t.shape[0] == hand_joints_openpose_t.shape[0] == obj_verts_t.shape[0]):
        raise ValueError("hand/object vertices and hand joints must have the same frame count")

    intrinsics_np = np.asarray(intrinsics, dtype=np.float32)
    if intrinsics_np.shape == (3, 3):
        intrinsics_np = intrinsics_np[None]
    elif intrinsics_np.shape != (1, 3, 3):
        raise ValueError(f"expected intrinsics shape (3, 3) or (1, 3, 3), got {intrinsics_np.shape}")

    hand_joints = hand_joints_openpose_t[:, OPENPOSE_TO_MANO, :]
    hand_root = hand_joints[:, 0, :]
    obj_root = obj_verts_t.mean(dim=1)

    num_frames = hand_verts_t.shape[0]
    out_dict = {
        "fnames": np.array([f"rgb/{i:04d}.png" for i in range(num_frames)]),
        "K": intrinsics_np,
        "full_seq_name": seq_name,
        "verts.right": hand_verts_t,
        "jnts.right": hand_joints,
        "root.right": hand_root,
        "j3d_ra.right": hand_joints - hand_root[:, None, :],
        "verts.object": obj_verts_t,
        "v3d_c.object": obj_verts_t,
        "root.object": obj_root,
        "v3d_ra.object": obj_verts_t - obj_root[:, None, :],
        "v3d_right.object": obj_verts_t - hand_root[:, None, :],
        "faces": {
            "object": obj_faces_t,
            "right": hand_faces_t,
        },
    }

    np.save(output_path, out_dict, allow_pickle=True)
    return output_path
