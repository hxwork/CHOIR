"""Helpers for exporting Stage 3 state as GraspFlowMatching input."""

import json
import shutil
from pathlib import Path

import numpy as np


def _to_numpy(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _jsonable(value):
    arr = _to_numpy(value)
    if arr.ndim == 0:
        return round(float(arr), 7)
    return np.round(arr.astype(float), 7).tolist()


def _write_obj(path: Path, verts, faces) -> None:
    verts_np = _to_numpy(verts)
    faces_np = _to_numpy(faces).astype(np.int64)
    with path.open("w", encoding="utf-8") as f:
        for v in verts_np:
            f.write(f"v {float(v[0]):.9g} {float(v[1]):.9g} {float(v[2]):.9g}\n")
        for face in faces_np:
            # OBJ indices are 1-based.
            f.write(f"f {int(face[0]) + 1} {int(face[1]) + 1} {int(face[2]) + 1}\n")


def export_graspflowmatching_sequence(
    seq_path,
    sampled_indices,
    canonical_verts,
    canonical_faces,
    obj_rot_mats,
    obj_trans,
    obj_scale,
    mano_root_orient,
    mano_pose,
    mano_trans,
    is_right,
    output_dir_name="optimized_hoi_seq",
    clean_output=False,
):
    """Export sparse Stage 3 pose state into GraspFlowMatching's expected layout."""
    output_dir = Path(seq_path) / output_dir_name
    if clean_output and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sampled_indices = [int(i) for i in sampled_indices]
    num_frames = len(sampled_indices)
    frame_arrays = {
        "obj_rot_mats": _to_numpy(obj_rot_mats),
        "obj_trans": _to_numpy(obj_trans),
        "mano_root_orient": _to_numpy(mano_root_orient),
        "mano_pose": _to_numpy(mano_pose),
        "mano_trans": _to_numpy(mano_trans),
        "is_right": _to_numpy(is_right),
    }
    for name, arr in frame_arrays.items():
        if arr.shape[0] != num_frames:
            raise ValueError(f"sampled_indices length ({num_frames}) does not match {name} length ({arr.shape[0]})")

    _write_obj(output_dir / "obj_canonical.obj", canonical_verts, canonical_faces)
    scale_value = _to_numpy(obj_scale)
    if scale_value.ndim == 0 or scale_value.size == 1:
        scale_value = np.repeat(float(scale_value.reshape(-1)[0]), 3)

    for local_idx, frame_idx in enumerate(sampled_indices):
        obj_pose_data = {
            "scale": _jsonable(scale_value),
            "rotation": _jsonable(frame_arrays["obj_rot_mats"][local_idx]),
            "translation": _jsonable(frame_arrays["obj_trans"][local_idx]),
        }
        with (output_dir / f"obj_{frame_idx:05d}.json").open("w", encoding="utf-8") as f:
            json.dump(obj_pose_data, f, indent=4)

        mano_data = {
            "root_orient": _jsonable(frame_arrays["mano_root_orient"][local_idx]),
            "pose": _jsonable(frame_arrays["mano_pose"][local_idx]),
            "trans": _jsonable(frame_arrays["mano_trans"][local_idx]),
            "is_right": _jsonable(frame_arrays["is_right"][local_idx].reshape(-1)[0]),
        }
        with (output_dir / f"mano_{frame_idx:05d}.json").open("w", encoding="utf-8") as f:
            json.dump(mano_data, f, indent=4)

    return output_dir
