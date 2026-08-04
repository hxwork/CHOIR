"""Sparse SAM3D with temporal guidance + milestone mesh overlays on full-res RGB.

Temporal mechanism matches ``sam-3d-objects/demo_proj_temporal.py`` via
``Sam3dResetController`` / ``TemporalController`` (see ``sam3d_reset.py``).

Milestone frames (sampled indices) written under ``sam3d_sparse_keyframes/``:

* first sampled frame (0)
* Stage1 auto mask-motion onset (``stage1_interaction_lo_auto``)
* padded interaction segment start (``approaching_end_idx``)
* every ``stride`` frames in ``[approaching, interaction_end)``
* last frame inside interaction (``interaction_end - 1``)
* mask-motion end (``hi_auto - 1`` and ``hi_auto`` when in range)
* last sampled frame

Each milestone runs SAM3D in **sorted index order** with keyframe at 0 and
sequential ``follow``; mesh is Hard-Phong–shaded and composited onto the raw RGB.

When ``interaction_motion_profile`` classifies **rotation_likely**, the demo skips
this milestone pass and instead runs ``run_sam3d_rotation_dense_from_mask_onset``
(gap=1 from Stage1 mask-motion onset) under ``sam3d_sparse_keyframes/rotation_dense/``.
"""

from __future__ import annotations

import json
import os
import re
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import imageio
import numpy as np
import torch


SAM3D_DENSE_POSE_CACHE_FILE = "pose_snapshots.pt"
SAM3D_DENSE_POSE_CACHE_META_FILE = "pose_snapshots_meta.json"
SAM3D_DENSE_POSE_CACHE_VERSION = 1


def build_sam3d_dense_cache_meta(
    *,
    sampled_indices: List[int],
    reference_sampled_idx: int,
    onset_sampled_idx: int,
    end_sampled_exclusive: int,
    interaction_segment_lo: Optional[int],
    interaction_segment_hi: Optional[int],
    outlier_filter: bool,
    outlier_max_angle_deg: float,
    outlier_max_iters: int,
    retry_count: int,
    global_frame_offset: int,
    seed: Optional[int] = None,
    lambda_temp: Optional[float] = None,
    config_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Build metadata used to validate the SAM3D dense pose cache."""
    return {
        "cache_version": SAM3D_DENSE_POSE_CACHE_VERSION,
        "sampled_indices": [int(i) for i in sampled_indices],
        "reference_sampled_idx": int(reference_sampled_idx),
        "onset_sampled_idx": int(onset_sampled_idx),
        "end_sampled_exclusive": int(end_sampled_exclusive),
        "interaction_segment_lo": int(interaction_segment_lo) if interaction_segment_lo is not None else None,
        "interaction_segment_hi": int(interaction_segment_hi) if interaction_segment_hi is not None else None,
        "outlier_filter": bool(outlier_filter),
        "outlier_max_angle_deg": float(outlier_max_angle_deg),
        "outlier_max_iters": int(outlier_max_iters),
        "retry_count": int(retry_count),
        "global_frame_offset": int(global_frame_offset),
        "seed": int(seed) if seed is not None else None,
        "lambda_temp": float(lambda_temp) if lambda_temp is not None else None,
        "config_path": str(config_path) if config_path is not None else None,
    }


def sam3d_dense_cache_matches(
    meta: Dict[str, Any],
    *,
    sampled_indices: List[int],
    reference_sampled_idx: int,
    onset_sampled_idx: int,
    end_sampled_exclusive: int,
    interaction_segment_lo: Optional[int],
    interaction_segment_hi: Optional[int],
    outlier_filter: bool,
    outlier_max_angle_deg: float,
    outlier_max_iters: int,
    retry_count: int,
    global_frame_offset: int,
) -> bool:
    """Return whether an existing SAM3D dense pose cache matches this run."""
    expected = build_sam3d_dense_cache_meta(
        sampled_indices=sampled_indices,
        reference_sampled_idx=reference_sampled_idx,
        onset_sampled_idx=onset_sampled_idx,
        end_sampled_exclusive=end_sampled_exclusive,
        interaction_segment_lo=interaction_segment_lo,
        interaction_segment_hi=interaction_segment_hi,
        outlier_filter=outlier_filter,
        outlier_max_angle_deg=outlier_max_angle_deg,
        outlier_max_iters=outlier_max_iters,
        retry_count=retry_count,
        global_frame_offset=global_frame_offset,
    )
    keys = [
        "cache_version",
        "sampled_indices",
        "reference_sampled_idx",
        "onset_sampled_idx",
        "end_sampled_exclusive",
        "interaction_segment_lo",
        "interaction_segment_hi",
        "outlier_filter",
        "outlier_max_iters",
        "retry_count",
        "global_frame_offset",
    ]
    for key in keys:
        if meta.get(key) != expected[key]:
            return False
    return abs(float(meta.get("outlier_max_angle_deg", -1.0)) - expected["outlier_max_angle_deg"]) < 1e-6


def _torch_load_cache(path: Path) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def save_sam3d_dense_pose_cache(
    out_dir: str | Path,
    meta: Dict[str, Any],
    summary: Dict[str, Any],
    pose_snapshots: List[Dict[str, Any]],
) -> None:
    """Persist final filtered/interpolated SAM3D dense poses for reuse."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    with open(out_path / SAM3D_DENSE_POSE_CACHE_META_FILE, "w") as f:
        json.dump(meta, f, indent=2)
    torch.save(
        {
            "summary": summary,
            "pose_snapshots": pose_snapshots,
        },
        out_path / SAM3D_DENSE_POSE_CACHE_FILE,
    )


def load_sam3d_dense_pose_cache(
    out_dir: str | Path,
    *,
    sampled_indices: List[int],
    reference_sampled_idx: int,
    onset_sampled_idx: int,
    end_sampled_exclusive: int,
    interaction_segment_lo: Optional[int],
    interaction_segment_hi: Optional[int],
    outlier_filter: bool,
    outlier_max_angle_deg: float,
    outlier_max_iters: int,
    retry_count: int,
    global_frame_offset: int,
) -> Optional[Tuple[Dict[str, Any], List[Dict[str, Any]]]]:
    """Load cached SAM3D dense poses if metadata matches this run."""
    out_path = Path(out_dir)
    meta_path = out_path / SAM3D_DENSE_POSE_CACHE_META_FILE
    cache_path = out_path / SAM3D_DENSE_POSE_CACHE_FILE
    if not meta_path.exists() or not cache_path.exists():
        return None
    try:
        with open(meta_path, "r") as f:
            meta = json.load(f)
        if not sam3d_dense_cache_matches(
            meta,
            sampled_indices=sampled_indices,
            reference_sampled_idx=reference_sampled_idx,
            onset_sampled_idx=onset_sampled_idx,
            end_sampled_exclusive=end_sampled_exclusive,
            interaction_segment_lo=interaction_segment_lo,
            interaction_segment_hi=interaction_segment_hi,
            outlier_filter=outlier_filter,
            outlier_max_angle_deg=outlier_max_angle_deg,
            outlier_max_iters=outlier_max_iters,
            retry_count=retry_count,
            global_frame_offset=global_frame_offset,
        ):
            return None
        payload = _torch_load_cache(cache_path)
    except Exception as exc:
        print(f"[SAM3D-ROT] cache load failed ({cache_path}): {exc}")
        return None
    return payload.get("summary", {}), payload.get("pose_snapshots", [])


def default_sparse_stride_for_profile(suggested_mode: Optional[str]) -> int:
    if suggested_mode == "rotation_likely":
        return 1
    if suggested_mode == "translation_likely":
        return 8
    return 8


def sparse_stride_for_motion_profile(
    suggested_mode: Optional[str],
    override: Optional[int],
) -> int:
    if override is not None and int(override) > 0:
        return int(override)
    return default_sparse_stride_for_profile(suggested_mode)


def build_sam3d_follow_retry_seeds(*, seed: int, retry_count: int) -> Tuple[int, ...]:
    """Return seeds for the first follow attempt plus extra retry attempts."""
    base = int(seed)
    n_retry = max(0, int(retry_count))
    return tuple(base + i for i in range(n_retry + 1))


def _flip_quat_to_match_torch(q: torch.Tensor, ref: Optional[torch.Tensor]) -> torch.Tensor:
    if ref is None:
        return q
    if (q.flatten() * ref.flatten()).sum().item() < 0.0:
        return -q
    return q


def _apply_keyframe_scale_torch(out: Dict[str, Any], frozen_pose: Optional[Dict[str, torch.Tensor]]) -> Dict[str, Any]:
    if frozen_pose is None or "scale" not in frozen_pose:
        return out
    out["scale"] = frozen_pose["scale"].to(out["scale"].device).clone()
    return out


def _quat_slerp_np(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation between two unit quaternions (w,x,y,z). Same as ``demo_proj_temporal``."""
    q0 = q0 / (np.linalg.norm(q0) + 1e-12)
    q1 = q1 / (np.linalg.norm(q1) + 1e-12)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        out = (1.0 - t) * q0 + t * q1
        return out / (np.linalg.norm(out) + 1e-12)
    omega = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_o = np.sin(omega)
    return (np.sin((1.0 - t) * omega) / sin_o) * q0 + (np.sin(t * omega) / sin_o) * q1


def _quat_angle_deg_np(q0: np.ndarray, q1: np.ndarray) -> float:
    """Angular distance between two unit quaternions, in degrees."""
    q0 = q0 / (np.linalg.norm(q0) + 1e-12)
    q1 = q1 / (np.linalg.norm(q1) + 1e-12)
    dot = abs(float(np.dot(q0, q1)))
    return float(np.degrees(2.0 * np.arccos(min(1.0, dot))))


def build_rejected_overlay_filename(
    *,
    sampled_idx: int,
    raw_frame_idx: int,
    anchor_idx: int,
    angle_deg: float,
) -> str:
    """Return the filename for a rejected raw SAM3D pose overlay."""
    return (
        f"overlay_rejected_sidx_{int(sampled_idx):04d}_raw_{int(raw_frame_idx):05d}_"
        f"anchor_{int(anchor_idx):04d}_angle_{float(angle_deg):05.1f}.png"
    )


def _filter_outlier_keyframes_rotation(
    kf_poses: Dict[int, Dict[str, torch.Tensor]],
    max_angle_deg: float,
    max_iters: int,
) -> List[Dict[str, Any]]:
    """Greedy SLERP residual peel; mutates ``kf_poses``. Endpoints (min/max keys) are never dropped."""
    dropped: List[Dict[str, Any]] = []
    if not kf_poses:
        return dropped

    for it in range(int(max_iters)):
        ordered = sorted(kf_poses.keys())
        if len(ordered) < 3:
            break

        worst = None
        for i in range(1, len(ordered) - 1):
            k = ordered[i]
            kp = ordered[i - 1]
            kn = ordered[i + 1]
            t = (k - kp) / float(kn - kp)
            q_pred = _quat_slerp_np(
                kf_poses[kp]["rotation"].detach().cpu().numpy().reshape(-1),
                kf_poses[kn]["rotation"].detach().cpu().numpy().reshape(-1),
                t,
            )
            q_obs = kf_poses[k]["rotation"].detach().cpu().numpy().reshape(-1)
            ang = _quat_angle_deg_np(q_obs, q_pred)
            if ang > max_angle_deg and (worst is None or ang > worst["angle_deg"]):
                worst = {"frame": k, "iter": it, "angle_deg": ang, "left": kp, "right": kn}

        if worst is None:
            break
        dropped.append(worst)
        del kf_poses[worst["frame"]]
        print(
            f"[SAM3D-ROT][outlier-global] iter {it}: dropped keyframe {worst['frame']} "
            f"(residual = {worst['angle_deg']:.1f}deg vs SLERP({worst['left']}, {worst['right']}))"
        )

    return dropped


def _interp_pose_torch(
    prev_kf: int,
    next_kf: int,
    idx: int,
    prev: Dict[str, torch.Tensor],
    nxt: Dict[str, torch.Tensor],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """SLERP rotation, lerp translation, log-lerp scale (``demo_proj_temporal._interp_pose``)."""
    t = (idx - prev_kf) / float(next_kf - prev_kf)
    q0 = prev["rotation"].detach().cpu().numpy().reshape(-1)
    q1 = nxt["rotation"].detach().cpu().numpy().reshape(-1)
    q = _quat_slerp_np(q0, q1, t).reshape(prev["rotation"].shape)

    tr = (1.0 - t) * prev["translation"].detach().cpu().numpy() + t * nxt["translation"].detach().cpu().numpy()

    s0 = prev["scale"].detach().cpu().numpy()
    s1 = nxt["scale"].detach().cpu().numpy()
    eps = 1e-12
    log_s = (1.0 - t) * np.log(np.maximum(s0, eps)) + t * np.log(np.maximum(s1, eps))
    s = np.exp(log_s)

    return {
        "rotation": torch.from_numpy(q).to(device=device, dtype=prev["rotation"].dtype),
        "translation": torch.from_numpy(tr).to(device=device, dtype=prev["translation"].dtype),
        "scale": torch.from_numpy(s).to(device=device, dtype=prev["scale"].dtype),
    }


def _sanitize_filename_tag(tags: List[str]) -> str:
    raw = "_".join(tags)
    raw = re.sub(r"[^a-zA-Z0-9_.-]+", "_", raw)
    return raw[:120] if len(raw) > 120 else raw


def _ensure_even_dimensions_rgb(rgb: np.ndarray) -> np.ndarray:
    """Crop/resize to even H×W — libx264 + yuv420p reject odd dimensions (e.g. 853×480)."""
    h, w = rgb.shape[:2]
    nh, nw = h - (h % 2), w - (w % 2)
    if nh < 2 or nw < 2:
        return np.ascontiguousarray(rgb)
    if nh == h and nw == w:
        return np.ascontiguousarray(rgb)
    return cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)


def build_milestone_sampled_indices(
    *,
    num_sampled_frames: int,
    stage1_lo_auto: Optional[int],
    stage1_hi_auto: Optional[int],
    approaching_padded: int,
    interaction_end_padded: int,
    stride_m: int,
) -> Tuple[List[int], Dict[int, List[str]]]:
    """Return sorted unique sampled indices and tags for each (diagnostic)."""
    tags: Dict[int, List[str]] = {}

    def add(idx: int, label: str) -> None:
        if 0 <= idx < num_sampled_frames:
            tags.setdefault(idx, []).append(label)

    add(0, "first_sampled")
    if stage1_lo_auto is not None:
        add(int(stage1_lo_auto), "mask_motion_onset_auto")
    add(int(approaching_padded), "interaction_start_padded")
    st = max(1, int(stride_m))
    ap, ih = int(approaching_padded), int(interaction_end_padded)
    if ih > ap:
        for t in range(ap, ih, st):
            add(t, f"interaction_every_{st}")
        add(ih - 1, "interaction_end_last_in_segment")
    if stage1_hi_auto is not None:
        ha = int(stage1_hi_auto)
        if ha - 1 >= 0:
            add(ha - 1, "mask_motion_last_auto")
        if ha < num_sampled_frames:
            add(ha, "mask_motion_first_stable_auto")
    add(num_sampled_frames - 1, "last_sampled")

    ordered = sorted(tags.keys())
    return ordered, tags


def _rotation_matrix_from_sam3d_out(out: Dict[str, Any], device: torch.device) -> torch.Tensor:
    from pytorch3d.transforms import quaternion_to_matrix

    q = out["rotation"]
    if q.dim() == 3:
        q = q.squeeze(1)
    R = quaternion_to_matrix(q.float().to(device))
    if R.dim() == 3 and R.shape[0] == 1:
        R = R[0]
    return R


def render_mesh_soft_overlay_on_rgb(
    rgb_u8: np.ndarray,
    verts: torch.Tensor,
    faces: torch.Tensor,
    out_pose: Dict[str, Any],
    intrinsics_3x3: torch.Tensor,
    device: torch.device,
    mesh_color: Tuple[float, float, float] = (0.25, 0.75, 1.0),
    alpha: float = 0.85,
) -> np.ndarray:
    """Hard-Phong shaded mesh composite of SAM3D pose onto full-resolution RGB (uint8)."""
    from pytorch3d.transforms import Transform3d

    R = _rotation_matrix_from_sam3d_out(out_pose, device)
    T = out_pose["translation"].float().to(device).reshape(-1)[:3]
    sc = out_pose["scale"].float().to(device).reshape(-1)[:3]

    verts_d = verts.to(device=device, dtype=torch.float32)
    faces_d = faces.to(device=device, dtype=torch.int64).long()
    tf = Transform3d(device=device).scale(sc.unsqueeze(0)).rotate(R.unsqueeze(0)).translate(T.unsqueeze(0))
    vpos = tf.transform_points(verts_d.unsqueeze(0))[0]
    return render_mesh_phong_overlay_world_vertices(
        rgb_u8, vpos, faces_d, intrinsics_3x3, device, mesh_color=mesh_color, alpha=alpha
    )


def render_mesh_phong_overlay_world_vertices(
    rgb_u8: np.ndarray,
    verts_world: torch.Tensor,
    faces: torch.Tensor,
    intrinsics_3x3: torch.Tensor,
    device: torch.device,
    mesh_color: Tuple[float, float, float] = (0.25, 0.75, 1.0),
    alpha: float = 0.85,
) -> np.ndarray:
    """Hard-Phong composite given **already posed** object vertices (world space)."""
    from pytorch3d.renderer import (
        MeshRasterizer,
        MeshRenderer,
        PerspectiveCameras,
        PointLights,
        RasterizationSettings,
        TexturesVertex,
    )
    from pytorch3d.renderer.mesh.shader import HardPhongShader
    from pytorch3d.structures import Meshes

    H, W = rgb_u8.shape[:2]
    intr = intrinsics_3x3.to(device=device, dtype=torch.float32)
    fx, fy = intr[0, 0].item(), intr[1, 1].item()
    px, py = intr[0, 2].item(), intr[1, 2].item()

    vpos = verts_world.to(device=device, dtype=torch.float32)
    faces_d = faces.to(device=device, dtype=torch.int64).long()
    col = torch.tensor(mesh_color, device=device, dtype=torch.float32).view(1, 1, 3).expand(1, vpos.shape[0], -1)
    mesh = Meshes(verts=[vpos], faces=[faces_d], textures=TexturesVertex(verts_features=col))

    cameras = PerspectiveCameras(
        focal_length=torch.tensor([[fx, fy]], device=device, dtype=torch.float32),
        principal_point=torch.tensor([[px, py]], device=device, dtype=torch.float32),
        image_size=((H, W),),
        in_ndc=False,
        device=device,
    )
    lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])
    raster_settings = RasterizationSettings(image_size=(H, W), blur_radius=0.0, faces_per_pixel=1)
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=HardPhongShader(device=device, cameras=cameras, lights=lights),
    )
    with torch.no_grad():
        rgba = renderer(mesh, cameras=cameras, lights=lights).clamp(0.0, 1.0)
        mesh_rgb = rgba[0, ..., :3]
        mesh_a = rgba[0, ..., 3:4]

    rgb = torch.from_numpy(np.ascontiguousarray(rgb_u8)).float().to(device) / 255.0
    if rgb.dim() == 2:
        rgb = rgb.unsqueeze(-1).expand(-1, -1, 3)
    blend = (mesh_a * float(alpha)).clamp(0.0, 1.0)
    comp = rgb * (1.0 - blend) + mesh_rgb * blend
    return (comp.clamp(0.0, 1.0).cpu().numpy() * 255.0).astype(np.uint8)


def world_vertices_from_pnp_row(
    verts: torch.Tensor,
    R_col_major: torch.Tensor,
    t_vec: torch.Tensor,
    scale_full: torch.Tensor,
) -> torch.Tensor:
    """Match ``compute_pnp_health`` / Stage 3: posed = (verts * scale) @ R_row + t with R_row = R_col^T."""
    R_row = R_col_major.float().mT
    return (verts * scale_full) @ R_row + t_vec


def compress_rotation_follow_indices_non_interaction(
    indices: List[int],
    *,
    interaction_lo: Optional[int],
    interaction_hi: Optional[int],
) -> List[int]:
    """Drop redundant ``follow`` targets in **non-interaction** sampled segments.

    Half-open interaction window ``[interaction_lo, interaction_hi)`` matches
    Stage 1 ``[approaching_end_idx, interaction_end_idx)`` (2–3 segment). Outside
    that window the object is treated as rigid / low motion: each **maximal**
    run of **consecutive** non-interaction sampled indices (``t[k+1]==t[k]+1``)
    is reduced to **only the last** index (one ``follow`` from the previous SAM3D
    anchor). List discontinuities start a new run. Inside the window, **every**
    index is kept (gap=1 temporal chain).
    """
    if not indices:
        return []
    lo_i = interaction_lo
    hi_i = interaction_hi
    if lo_i is None or hi_i is None or int(hi_i) <= int(lo_i):
        return list(indices)

    ap, ih = int(lo_i), int(hi_i)

    def in_interaction(t: int) -> bool:
        return ap <= int(t) < ih

    out: List[int] = []
    i = 0
    n = len(indices)
    while i < n:
        t0 = int(indices[i])
        if in_interaction(t0):
            out.append(t0)
            i += 1
            continue
        j = i
        while j < n:
            tj = int(indices[j])
            if in_interaction(tj):
                break
            if j > i and tj != int(indices[j - 1]) + 1:
                break
            j += 1
        out.append(int(indices[j - 1]))
        i = j
    return out


def run_sam3d_rotation_dense_from_mask_onset(
    *,
    enabled: bool,
    output_path: str,
    onset_sampled_idx: int,
    end_sampled_exclusive: int,
    sampled_indices: List[int],
    raw_rgbs_np: Optional[np.ndarray],
    raw_masks_np: Optional[np.ndarray],
    seed: int,
    quiet: bool,
    config_path: Optional[str],
    lambda_temp: float,
    intrinsics_full: Optional[torch.Tensor],
    mesh_verts: Optional[torch.Tensor],
    mesh_faces: Optional[torch.Tensor],
    device: Optional[str] = None,
    interaction_segment_lo: Optional[int] = None,
    interaction_segment_hi: Optional[int] = None,
    outlier_filter: bool = True,
    outlier_max_angle_deg: float = 60.0,
    outlier_max_iters: int = 3,
    retry_count: int = 3,
    mesh_overlay_alpha: float = 0.85,
    global_frame_offset: int = 0,
    overwrite_dense_cache: bool = False,
    reference_sampled_idx: int = 0,
) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    """Dense gap=1 SAM3D temporal tracking from mask onset, aligned with ``demo_proj_temporal.py``.

    Includes local angular outlier rejection (``end_follow(commit=False)``), greedy
    global SLERP-residual peeling, and SLERP / lerp / log-lerp fill for dense indices
    missing a committed keyframe. ``follow`` anchors use the last **committed** sampled
    index (required when non-interaction compression skips indices).

    ``pose_snapshots`` and overlays are built from **final** dense poses after filtering
    and interpolation (for PnP compare).
    """
    empty_poses: List[Dict[str, Any]] = []
    if not enabled:
        return None, empty_poses

    lo = max(0, int(onset_sampled_idx))
    hi = min(int(end_sampled_exclusive), len(sampled_indices))
    ref = int(np.clip(int(reference_sampled_idx), 0, max(0, len(sampled_indices) - 1)))
    lo = min(lo, ref)
    if hi <= 0:
        print("[SAM3D-ROT] skip: empty sequence")
        return None, empty_poses

    if ref > 0:
        dense_indices = list(range(lo, hi))
    elif lo > 0:
        dense_indices = list(range(lo, hi))
    else:
        dense_indices = list(range(1, hi))
    if len(dense_indices) == 0:
        print("[SAM3D-ROT] skip: no dense output indices (check onset vs end)")
        return None, empty_poses

    out_dir = os.path.join(output_path, "sam3d_sparse_keyframes", "rotation_dense")
    if not overwrite_dense_cache:
        cached = load_sam3d_dense_pose_cache(
            out_dir,
            sampled_indices=sampled_indices,
            reference_sampled_idx=ref,
            onset_sampled_idx=lo,
            end_sampled_exclusive=hi,
            interaction_segment_lo=interaction_segment_lo,
            interaction_segment_hi=interaction_segment_hi,
            outlier_filter=outlier_filter,
            outlier_max_angle_deg=outlier_max_angle_deg,
            outlier_max_iters=outlier_max_iters,
            retry_count=retry_count,
            global_frame_offset=global_frame_offset,
        )
        if cached is not None:
            summary, pose_snapshots = cached
            summary = dict(summary)
            summary["cache_hit"] = True
            summary["cache_path"] = os.path.join(out_dir, SAM3D_DENSE_POSE_CACHE_FILE)
            print(
                f"[SAM3D-ROT] cache hit → {summary['cache_path']} "
                f"({len(pose_snapshots)} dense pose snapshot(s)); skip SAM3D inference"
            )
            return summary, pose_snapshots
    elif os.path.exists(out_dir):
        print(f"[SAM3D-ROT] overwrite_dense_cache=True; ignoring cache under {out_dir}")

    if raw_rgbs_np is None or raw_masks_np is None:
        print("[SAM3D-ROT] skip: raw rgb/mask arrays not provided")
        return None, empty_poses

    from sam3d_reset import (
        SAM3D_CONFIG_PATH,
        _SAM3D_AVAILABLE,
        _SAM3D_IMPORT_ERROR,
        Sam3dResetController,
    )

    if not _SAM3D_AVAILABLE:
        print(f"[SAM3D-ROT] skip: SAM3D not importable ({_SAM3D_IMPORT_ERROR})")
        return None, empty_poses

    dev = torch.device(device or "cuda:0")
    overlay_ok = (
        intrinsics_full is not None
        and mesh_verts is not None
        and mesh_faces is not None
    )
    if not overlay_ok:
        print("[SAM3D-ROT] warning: missing intrinsics_full or mesh verts/faces; mesh overlays skipped")

    n_raw = len(raw_rgbs_np)
    _global_fo = int(global_frame_offset)
    os.makedirs(out_dir, exist_ok=True)
    rejected_overlay_dir = os.path.join(out_dir, "rejected_raw_overlay")
    cfg = config_path or SAM3D_CONFIG_PATH

    if ref > 0:
        warmup_indices_full = []
        warmup_plan = []
        dense_backward_plan = list(range(ref - 1, lo - 1, -1))
        dense_forward_plan = list(range(ref + 1, hi))
        dense_plan = dense_backward_plan + dense_forward_plan
        total_infer = 1 + len(dense_plan)
        _saved = 0
    else:
        warmup_indices_full = list(range(1, lo)) if lo > 1 else []
        warmup_plan = compress_rotation_follow_indices_non_interaction(
            warmup_indices_full,
            interaction_lo=interaction_segment_lo,
            interaction_hi=interaction_segment_hi,
        )
        dense_plan = compress_rotation_follow_indices_non_interaction(
            dense_indices,
            interaction_lo=interaction_segment_lo,
            interaction_hi=interaction_segment_hi,
        )
        dense_backward_plan = []
        dense_forward_plan = dense_plan
        total_infer = 1 + len(warmup_plan) + len(dense_plan)
        _saved = (len(warmup_indices_full) - len(warmup_plan)) + (len(dense_indices) - len(dense_plan))

    print("\n" + "=" * 80)
    if warmup_indices_full:
        print(
            f"[SAM3D-ROT] ref keyframe @{ref}; warmup sampled {warmup_indices_full[0]}..{warmup_indices_full[-1]} "
            f"→ {len(warmup_plan)} follow(s); dense output {dense_indices[0]}..{dense_indices[-1]} "
            f"→ {len(dense_plan)} follow(s)"
        )
    elif ref > 0:
        print(
            f"[SAM3D-ROT] ref keyframe @{ref}; dense backward {lo}..{ref - 1} "
            f"and forward {ref + 1}..{hi - 1} "
            f"→ {len(dense_plan)} follow(s)"
        )
    else:
        print(
            f"[SAM3D-ROT] ref keyframe @{ref}; dense output {dense_indices[0]}..{dense_indices[-1]} "
            f"→ {len(dense_plan)} follow(s)"
        )
    if _saved > 0:
        print(f"[SAM3D-ROT] non-interaction compression: skipped {_saved} redundant follow target(s)")
    print(f"[SAM3D-ROT] output → {out_dir}  (total SAM3D inferences = {total_infer})")
    print("=" * 80)

    ctrl = Sam3dResetController(config_path=cfg, lambda_temp=float(lambda_temp), quiet=quiet)
    entries: List[Dict[str, Any]] = []
    warmup_entries: List[Dict[str, Any]] = []
    pose_snapshots: List[Dict[str, Any]] = []
    panels: List[np.ndarray] = []
    frozen_pose: Optional[Dict[str, torch.Tensor]] = None
    trusted_quat: Optional[torch.Tensor] = None
    last_committed = ref
    kf_poses: Dict[int, Dict[str, torch.Tensor]] = {}
    local_outliers: List[Dict[str, Any]] = []
    ref_intrinsics: Optional[torch.Tensor] = None
    _smax = max(0, len(sampled_indices) - 1)
    _rmax = max(0, n_raw - 1)
    _infer_i = 0

    def _prepare_frame(raw_fi: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if raw_fi < 0 or raw_fi >= n_raw:
            return None, None
        img = np.ascontiguousarray(raw_rgbs_np[raw_fi])
        msk = np.ascontiguousarray(raw_masks_np[raw_fi])
        if img.dtype != np.uint8:
            if img.max() <= 1.0:
                img = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
            else:
                img = np.clip(img, 0, 255).astype(np.uint8)
        msk_u8 = (msk > 0).astype(np.uint8)
        if int(msk_u8.sum()) < 50:
            return None, None
        return img, msk_u8

    def _log_line(phase: str, sidx: int, raw_fi: int, mode_hint: str) -> None:
        nonlocal _infer_i
        _infer_i += 1
        # ``sidx`` = slot into ``sampled_indices`` (0..N-1). ``raw_fi`` = ``sampled_indices[sidx]`` = row in
        # ``raw_rgbs_np`` for this clip (often equals ``sidx`` when N<=64 and no linspace subsample).
        _ds = int(raw_fi) + _global_fo
        _ds_max = int(_rmax) + _global_fo
        _ds_part = f"  dataset_frame={_ds}/{_ds_max}" if _global_fo != 0 else ""
        print(
            f"[SAM3D-ROT] pose track {_infer_i}/{total_infer}  phase={phase}  "
            f"sampled_slot={sidx}/{_smax}  clip_idx={raw_fi}/{_rmax}{_ds_part}  {mode_hint}",
            flush=True,
        )

    # --- Reference keyframe (sampled 0 by default; optionally a user-selected slot) ---
    raw0 = int(sampled_indices[ref])
    img0, msk0 = _prepare_frame(raw0)
    if img0 is None:
        print(f"[SAM3D-ROT] skip: cannot load or mask too small at ref sampled_slot={ref} clip_idx={raw0}")
        return None, empty_poses
    _log_line("ref_keyframe", ref, raw0, "keyframe")
    try:
        out0 = ctrl.keyframe(img0, msk0, frame_idx=ref, seed=int(seed))
        frozen_pose = {
            "rotation": out0["rotation"].detach().clone(),
            "translation": out0["translation"].detach().clone(),
            "scale": out0["scale"].detach().clone(),
        }
        trusted_quat = out0["rotation"].detach().clone()
        intr0 = out0.get("intrinsics")
        if intr0 is not None and hasattr(intr0, "detach"):
            ref_intrinsics = intr0.detach().clone()
        else:
            ref_intrinsics = None
        kf_poses[ref] = {
            "rotation": frozen_pose["rotation"].detach().clone(),
            "translation": frozen_pose["translation"].detach().clone(),
            "scale": frozen_pose["scale"].detach().clone(),
        }
    except Exception as _e:
        print(f"[SAM3D-ROT] reference keyframe @{ref} failed: {_e}")
        traceback.print_exc()
        return None, empty_poses

    def _run_follow(sidx: int, phase: str, record_dense_outputs: bool) -> bool:
        nonlocal trusted_quat, last_committed
        assert frozen_pose is not None and trusted_quat is not None
        raw_fi = int(sampled_indices[sidx])
        img, msk_u8 = _prepare_frame(raw_fi)
        if img is None:
            print(f"[SAM3D-ROT] skip follow sampled_slot={sidx} clip_idx={raw_fi}: load/mask")
            return False
        anchor = int(last_committed)
        _log_line(phase, sidx, raw_fi, f"follow anchor_sampled_idx={anchor}")
        dec: Dict[str, Any]
        is_local_out = False
        residual_deg = 0.0
        attempt_seed = int(seed)
        attempt_idx = 0
        attempt_records: List[Dict[str, Any]] = []
        try:
            for attempt_idx, attempt_seed in enumerate(
                build_sam3d_follow_retry_seeds(seed=int(seed), retry_count=retry_count)
            ):
                ctrl.follow_begin(anchor)
                try:
                    out = ctrl.follow_run(img, msk_u8, seed=int(attempt_seed))
                except Exception:
                    ctrl.follow_abort()
                    raise
                dec = {
                    "rotation": out["rotation"].detach().clone(),
                    "translation": out["translation"].detach().clone(),
                    "scale": out["scale"].detach().clone(),
                    "intrinsics": out.get("intrinsics"),
                }
                dec = _apply_keyframe_scale_torch(dec, frozen_pose)
                dec["rotation"] = _flip_quat_to_match_torch(dec["rotation"], trusted_quat)
                is_local_out = False
                if outlier_filter:
                    residual_deg = _quat_angle_deg_np(
                        dec["rotation"].detach().cpu().numpy().reshape(-1),
                        trusted_quat.detach().cpu().numpy().reshape(-1),
                    )
                    if residual_deg > float(outlier_max_angle_deg):
                        is_local_out = True
                attempt_records.append(
                    {
                        "attempt": int(attempt_idx),
                        "seed": int(attempt_seed),
                        "angle_deg": float(residual_deg) if outlier_filter else None,
                        "local_outlier": bool(is_local_out),
                    }
                )
                if is_local_out and attempt_idx < max(0, int(retry_count)):
                    ctrl.follow_abort()
                    print(
                        f"[SAM3D-ROT][retry] sampled_slot={sidx} clip_idx={raw_fi}: "
                        f"attempt={attempt_idx} seed={attempt_seed} angle={residual_deg:.1f}deg "
                        f"> {float(outlier_max_angle_deg):.1f}deg; retrying"
                    )
                    continue
                ctrl.follow_end(int(sidx), commit=not is_local_out, decoded=dec)
                break
        except Exception as _e:
            print(f"[SAM3D-ROT] SAM3D follow failed at sampled_slot={sidx} clip_idx={raw_fi}: {_e}")
            traceback.print_exc()
            return False

        if is_local_out:
            rejected_overlay_path = None
            if overlay_ok:
                try:
                    os.makedirs(rejected_overlay_dir, exist_ok=True)
                    rejected_name = build_rejected_overlay_filename(
                        sampled_idx=sidx,
                        raw_frame_idx=raw_fi,
                        anchor_idx=anchor,
                        angle_deg=residual_deg,
                    )
                    rejected_overlay_path = os.path.join(rejected_overlay_dir, rejected_name)
                    ov_rejected = render_mesh_soft_overlay_on_rgb(
                        img,
                        mesh_verts,
                        mesh_faces,
                        {k: v.to(dev) if isinstance(v, torch.Tensor) else v for k, v in dec.items()},
                        intrinsics_full,
                        dev,
                        alpha=float(mesh_overlay_alpha),
                    )
                    cv2.imwrite(rejected_overlay_path, cv2.cvtColor(ov_rejected, cv2.COLOR_RGB2BGR))
                except Exception as _e_ov_rej:
                    rejected_overlay_path = None
                    print(
                        f"[SAM3D-ROT] rejected overlay failed sampled_slot={sidx} "
                        f"clip_idx={raw_fi}: {_e_ov_rej}"
                    )
                    traceback.print_exc()
            _lo: Dict[str, Any] = {
                "sampled_slot": int(sidx),
                "sampled_idx": int(sidx),
                "clip_frame_idx": int(raw_fi),
                "raw_frame_idx": int(raw_fi),
                "phase": phase,
                "angle_deg": float(residual_deg),
                "anchor_sampled_idx": anchor,
                "retry_attempts": int(attempt_idx),
                "attempt_seed": int(attempt_seed),
                "attempts": attempt_records,
            }
            if rejected_overlay_path is not None:
                _lo["rejected_overlay_path"] = rejected_overlay_path
            if _global_fo != 0:
                _lo["dataset_raw_frame_idx"] = int(raw_fi) + _global_fo
            local_outliers.append(_lo)
            print(
                f"[SAM3D-ROT][outlier-local] sampled_slot={sidx} clip_idx={raw_fi}: jump vs trusted = {residual_deg:.1f}deg "
                f"> {float(outlier_max_angle_deg):.1f}deg; not committing (anchor stays {anchor})"
            )
        else:
            trusted_quat = dec["rotation"].detach().clone()
            last_committed = int(sidx)
            kf_poses[int(sidx)] = {
                "rotation": dec["rotation"].detach().clone(),
                "translation": dec["translation"].detach().clone(),
                "scale": dec["scale"].detach().clone(),
            }

        tr = dec["translation"].detach().float().reshape(-1)[:3].cpu().numpy()
        sc = dec["scale"].detach().float().reshape(-1)[:3].cpu().numpy()
        row = {
            "sampled_slot": int(sidx),
            "sampled_idx": int(sidx),
            "clip_frame_idx": int(raw_fi),
            "raw_frame_idx": int(raw_fi),
            "milestone_tags": [f"rotation_{phase}"],
            "mode": "follow_local_outlier" if is_local_out else "follow",
            "anchor_sampled_idx": anchor,
            "translation": tr.tolist(),
            "translation_l2": float(np.linalg.norm(tr)),
            "scale": sc.tolist(),
            "local_outlier": bool(is_local_out),
            "local_residual_angle_deg": float(residual_deg) if outlier_filter else None,
            "retry_attempts": int(attempt_idx),
            "attempt_seed": int(attempt_seed),
            "attempts": attempt_records,
        }
        if _global_fo != 0:
            row["dataset_raw_frame_idx"] = int(raw_fi) + _global_fo
        if record_dense_outputs:
            entries.append(row)
        else:
            warmup_entries.append(row)
        return True

    for sidx in warmup_plan:
        if not _run_follow(sidx, "temporal_warmup", record_dense_outputs=False):
            return None, empty_poses

    if ref > 0:
        trusted_quat = kf_poses[ref]["rotation"].detach().clone()
        last_committed = ref
        for sidx in dense_backward_plan:
            if not _run_follow(sidx, "dense_backward_from_ref", record_dense_outputs=True):
                break
        trusted_quat = kf_poses[ref]["rotation"].detach().clone()
        last_committed = ref
        for sidx in dense_forward_plan:
            if not _run_follow(sidx, "dense_forward_from_ref", record_dense_outputs=True):
                break
    else:
        for sidx in dense_plan:
            if not _run_follow(sidx, "dense_from_onset", record_dense_outputs=True):
                break

    dropped_global: List[Dict[str, Any]] = []
    interp_meta: List[Dict[str, Any]] = []
    if outlier_filter and len(kf_poses) >= 3:
        dropped_global = _filter_outlier_keyframes_rotation(
            kf_poses,
            float(outlier_max_angle_deg),
            int(outlier_max_iters),
        )

    sorted_kfs = sorted(kf_poses.keys())
    prev_qh: Optional[torch.Tensor] = None
    for kf in sorted_kfs:
        q = kf_poses[kf]["rotation"]
        if prev_qh is not None:
            q = _flip_quat_to_match_torch(q, prev_qh)
            kf_poses[kf]["rotation"] = q
        prev_qh = q.detach().clone()

    final_dense: Dict[int, Dict[str, torch.Tensor]] = {}
    for sidx in dense_indices:
        if sidx in kf_poses:
            fd = {k: v.detach().clone() for k, v in kf_poses[sidx].items()}
        else:
            left: Optional[int] = None
            right: Optional[int] = None
            for kf in sorted_kfs:
                if kf <= sidx:
                    left = kf
                elif kf > sidx and right is None:
                    right = kf
                    break
            if left is None and right is None:
                continue
            if left is None:
                fd = {k: v.detach().clone() for k, v in kf_poses[int(right)].items()}
            elif right is None:
                fd = {k: v.detach().clone() for k, v in kf_poses[int(left)].items()}
            else:
                fd = _interp_pose_torch(int(left), int(right), int(sidx), kf_poses[int(left)], kf_poses[int(right)], dev)
            interp_meta.append(
                {
                    "sampled_idx": int(sidx),
                    "left": int(left) if left is not None else None,
                    "right": int(right) if right is not None else None,
                }
            )
        if ref_intrinsics is not None:
            fd["intrinsics"] = ref_intrinsics.to(device=fd["rotation"].device, dtype=fd["rotation"].dtype)
        final_dense[int(sidx)] = fd

    entries.clear()
    pose_snapshots.clear()
    panels.clear()
    for sidx in dense_indices:
        if int(sidx) not in final_dense:
            continue
        out = final_dense[int(sidx)]
        raw_fi = int(sampled_indices[int(sidx)])
        was_interp = any(m.get("sampled_idx") == int(sidx) for m in interp_meta)
        tr = out["translation"].detach().float().reshape(-1)[:3].cpu().numpy()
        sc = out["scale"].detach().float().reshape(-1)[:3].cpu().numpy()
        _entry_row: Dict[str, Any] = {
            "sampled_slot": int(sidx),
            "sampled_idx": int(sidx),
            "clip_frame_idx": int(raw_fi),
            "raw_frame_idx": int(raw_fi),
            "milestone_tags": ["rotation_dense_from_onset"],
            "mode": "interpolated" if was_interp else "follow_committed_or_keyframe_ref",
            "translation": tr.tolist(),
            "translation_l2": float(np.linalg.norm(tr)),
            "scale": sc.tolist(),
        }
        if _global_fo != 0:
            _entry_row["dataset_raw_frame_idx"] = int(raw_fi) + _global_fo
        entries.append(_entry_row)
        _snap_row: Dict[str, Any] = {
            "sampled_slot": int(sidx),
            "sampled_idx": int(sidx),
            "clip_frame_idx": int(raw_fi),
            "raw_frame_idx": int(raw_fi),
            "pose": {k: v.detach().cpu() for k, v in out.items() if isinstance(v, torch.Tensor)},
        }
        if _global_fo != 0:
            _snap_row["dataset_raw_frame_idx"] = int(raw_fi) + _global_fo
        pose_snapshots.append(_snap_row)
        if overlay_ok:
            img, _msk = _prepare_frame(raw_fi)
            if img is not None:
                try:
                    ov = render_mesh_soft_overlay_on_rgb(
                        img,
                        mesh_verts,
                        mesh_faces,
                        {k: v.to(dev) if isinstance(v, torch.Tensor) else v for k, v in out.items()},
                        intrinsics_full,
                        dev,
                        alpha=float(mesh_overlay_alpha),
                    )
                    out_png = os.path.join(out_dir, f"overlay_sam3d_dense_sidx_{sidx:04d}_raw_{raw_fi:05d}.png")
                    cv2.imwrite(out_png, cv2.cvtColor(ov, cv2.COLOR_RGB2BGR))
                    h0, w0 = ov.shape[:2]
                    scv = min(1.0, 480.0 / max(h0, 1))
                    panels.append(cv2.resize(ov, (int(w0 * scv), int(h0 * scv)), interpolation=cv2.INTER_AREA))
                except Exception as _e_ov:
                    print(f"[SAM3D-ROT] overlay failed sampled_slot={sidx} clip_idx={raw_fi}: {_e_ov}")
                    traceback.print_exc()

    log_path = os.path.join(out_dir, "run_log.json")
    summary: Dict[str, Any] = {
        "policy": "rotation_likely_gap1_demo_proj_temporal_aligned",
        "global_frame_offset": int(_global_fo),
        "frame_index_legend": {
            "sampled_slot": "Index into sampled_indices / temporal chain slot (0..N-1).",
            "clip_frame_idx": "sampled_indices[slot]; row in raw_rgbs_np/raw_masks_np for this clip.",
            "dataset_raw_frame_idx": "clip_frame_idx + global_frame_offset (full-video index when clip is a slice).",
        },
        "reference_keyframe_sampled_idx": int(ref),
        "reference_keyframe_clip_frame_idx": int(sampled_indices[ref]),
        "onset_sampled_idx": lo,
        "end_sampled_exclusive": hi,
        "interaction_segment_sampled_half_open": [
            int(interaction_segment_lo) if interaction_segment_lo is not None else None,
            int(interaction_segment_hi) if interaction_segment_hi is not None else None,
        ],
        "outlier_filter": bool(outlier_filter),
        "outlier_max_angle_deg": float(outlier_max_angle_deg),
        "outlier_max_iters": int(outlier_max_iters),
        "retry_count": int(retry_count),
        "mesh_overlay_alpha": float(mesh_overlay_alpha),
        "local_outliers": local_outliers,
        "dropped_keyframes_global": dropped_global,
        "interpolated_dense_indices": interp_meta,
        "warmup_sampled_indices_full": warmup_indices_full,
        "warmup_follow_plan": warmup_plan,
        "dense_output_sampled_indices_full": dense_indices,
        "dense_follow_plan": dense_plan,
        "dense_backward_follow_plan": dense_backward_plan,
        "dense_forward_follow_plan": dense_forward_plan,
        "non_interaction_follows_skipped": int(_saved),
        "warmup_entries": warmup_entries,
        "config_path": cfg,
        "lambda_temp": float(lambda_temp),
        "entries": entries,
        "keyframe_poses_after_filter_sampled": [int(k) for k in sorted(kf_poses.keys())],
        "cache_hit": False,
    }
    with open(log_path, "w") as _jf:
        json.dump(summary, _jf, indent=2)
    print(f"[SAM3D-ROT] wrote {log_path} ({len(entries)} dense row(s); {len(interp_meta)} interpolated)")

    cache_meta = build_sam3d_dense_cache_meta(
        sampled_indices=sampled_indices,
        reference_sampled_idx=ref,
        onset_sampled_idx=lo,
        end_sampled_exclusive=hi,
        interaction_segment_lo=interaction_segment_lo,
        interaction_segment_hi=interaction_segment_hi,
        outlier_filter=outlier_filter,
        outlier_max_angle_deg=outlier_max_angle_deg,
        outlier_max_iters=outlier_max_iters,
        retry_count=retry_count,
        global_frame_offset=_global_fo,
        seed=seed,
        lambda_temp=lambda_temp,
        config_path=cfg,
    )
    save_sam3d_dense_pose_cache(out_dir, cache_meta, summary, pose_snapshots)
    print(
        f"[SAM3D-ROT] cached dense poses → "
        f"{os.path.join(out_dir, SAM3D_DENSE_POSE_CACHE_FILE)} "
        f"({len(pose_snapshots)} snapshot(s))"
    )

    if len(panels) > 0:
        vid_path = os.path.join(out_dir, "preview_sam3d_dense.mp4")
        writer = None
        try:
            writer = imageio.get_writer(
                vid_path,
                fps=4.0,
                codec="libx264",
                pixelformat="yuv420p",
                ffmpeg_params=["-crf", "28", "-preset", "veryfast"],
                macro_block_size=None,
            )
            for _p in panels:
                writer.append_data(_ensure_even_dimensions_rgb(np.ascontiguousarray(_p)))
            print(f"[SAM3D-ROT] preview video → {vid_path}")
        except Exception as _e_vid:
            print(f"[SAM3D-ROT] preview video failed: {_e_vid}")
        finally:
            if writer is not None:
                writer.close()

    return summary, pose_snapshots


def run_sam3d_sparse_keyframes_motion_window(
    *,
    enabled: bool,
    output_path: str,
    sampled_indices: List[int],
    num_sampled_frames: int,
    stage1_interaction_lo_auto: Optional[int],
    stage1_interaction_hi_auto: Optional[int],
    approaching_end_idx_padded: int,
    interaction_end_idx_padded: int,
    raw_rgbs_np: Optional[np.ndarray],
    raw_masks_np: Optional[np.ndarray],
    stride: int,
    seed: int,
    quiet: bool,
    config_path: Optional[str],
    lambda_temp: float,
    intrinsics_full: Optional[torch.Tensor],
    mesh_verts: Optional[torch.Tensor],
    mesh_faces: Optional[torch.Tensor],
    device: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    from sam3d_reset import (
        SAM3D_CONFIG_PATH,
        _SAM3D_AVAILABLE,
        _SAM3D_IMPORT_ERROR,
        Sam3dResetController,
    )

    if not enabled:
        return None
    if raw_rgbs_np is None or raw_masks_np is None:
        print("[SAM3D-SPARSE] skip: raw rgb/mask arrays not provided")
        return None
    if not _SAM3D_AVAILABLE:
        print(f"[SAM3D-SPARSE] skip: SAM3D not importable ({_SAM3D_IMPORT_ERROR})")
        return None

    dev = torch.device(device or "cuda:0")
    overlay_ok = (
        intrinsics_full is not None
        and mesh_verts is not None
        and mesh_faces is not None
    )
    if not overlay_ok:
        print("[SAM3D-SPARSE] warning: missing intrinsics_full or mesh verts/faces; mesh overlays skipped")

    ordered, tag_map = build_milestone_sampled_indices(
        num_sampled_frames=num_sampled_frames,
        stage1_lo_auto=stage1_interaction_lo_auto,
        stage1_hi_auto=stage1_interaction_hi_auto,
        approaching_padded=int(approaching_end_idx_padded),
        interaction_end_padded=int(interaction_end_idx_padded),
        stride_m=max(1, int(stride)),
    )
    if len(ordered) == 0:
        print("[SAM3D-SPARSE] skip: no milestone indices")
        return None

    chain: List[int] = []
    if 0 not in ordered:
        chain = [0] + [x for x in ordered if x != 0]
    else:
        chain = sorted(set(ordered))

    n_raw = len(raw_rgbs_np)
    out_dir = os.path.join(output_path, "sam3d_sparse_keyframes")
    os.makedirs(out_dir, exist_ok=True)
    cfg = config_path or SAM3D_CONFIG_PATH

    print("\n" + "=" * 80)
    print("[SAM3D-SPARSE] temporal chain: keyframe @ sampled 0, then follow in milestone order")
    print(f"[SAM3D-SPARSE] milestones (sampled_idx -> tags): { {k: tag_map[k] for k in chain} }")
    print(f"[SAM3D-SPARSE] output → {out_dir}")
    print("=" * 80)

    ctrl = Sam3dResetController(config_path=cfg, lambda_temp=float(lambda_temp), quiet=quiet)
    entries: List[Dict[str, Any]] = []
    panels: List[np.ndarray] = []
    prev_sidx: Optional[int] = None
    frozen_pose: Optional[Dict[str, torch.Tensor]] = None
    trusted_quat: Optional[torch.Tensor] = None
    _chain_n = len(chain)

    _smax = max(0, int(num_sampled_frames) - 1)
    _rmax = max(0, n_raw - 1)
    for _step, sidx in enumerate(chain, start=1):
        raw_fi = int(sampled_indices[sidx])
        _mode_hint = "keyframe" if prev_sidx is None else "follow"
        print(
            f"[SAM3D-SPARSE] pose track {_step}/{_chain_n}  "
            f"sampled_idx={sidx}/{_smax}  raw_frame={raw_fi}/{_rmax}  {_mode_hint}",
            flush=True,
        )
        if raw_fi < 0 or raw_fi >= n_raw:
            print(f"[SAM3D-SPARSE] skip sidx={sidx}: raw {raw_fi} out of range")
            continue
        img = np.ascontiguousarray(raw_rgbs_np[raw_fi])
        msk = np.ascontiguousarray(raw_masks_np[raw_fi])
        if img.dtype != np.uint8:
            if img.max() <= 1.0:
                img = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
            else:
                img = np.clip(img, 0, 255).astype(np.uint8)
        msk_u8 = (msk > 0).astype(np.uint8)
        if int(msk_u8.sum()) < 50:
            print(f"[SAM3D-SPARSE] skip sidx={sidx} raw={raw_fi}: mask too small")
            continue
        try:
            if prev_sidx is None:
                out = ctrl.keyframe(img, msk_u8, frame_idx=int(sidx), seed=int(seed))
                mode = "keyframe"
                frozen_pose = {
                    "rotation": out["rotation"].detach().clone(),
                    "translation": out["translation"].detach().clone(),
                    "scale": out["scale"].detach().clone(),
                }
                trusted_quat = out["rotation"].detach().clone()
            else:
                out = ctrl.follow(img, msk_u8, anchor_idx=int(prev_sidx), frame_idx=int(sidx), seed=int(seed))
                mode = "follow"
                out = _apply_keyframe_scale_torch(dict(out), frozen_pose)
                out["rotation"] = _flip_quat_to_match_torch(out["rotation"], trusted_quat)
                trusted_quat = out["rotation"].detach().clone()
        except Exception as _e:
            print(f"[SAM3D-SPARSE] SAM3D failed at sampled_idx={sidx} raw={raw_fi}: {_e}")
            traceback.print_exc()
            break

        tr = out["translation"].detach().float().reshape(-1)[:3].cpu().numpy()
        sc = out["scale"].detach().float().reshape(-1)[:3].cpu().numpy()
        tags = tag_map.get(int(sidx), ["milestone"])
        entries.append(
            {
                "sampled_idx": int(sidx),
                "raw_frame_idx": int(raw_fi),
                "milestone_tags": tags,
                "mode": mode,
                "anchor_sampled_idx": None if prev_sidx is None else int(prev_sidx),
                "translation": tr.tolist(),
                "translation_l2": float(np.linalg.norm(tr)),
                "scale": sc.tolist(),
            }
        )
        prev_sidx = int(sidx)

        tag_fn = _sanitize_filename_tag(tags)
        if overlay_ok:
            try:
                ov = render_mesh_soft_overlay_on_rgb(
                    img,
                    mesh_verts,
                    mesh_faces,
                    out,
                    intrinsics_full,
                    dev,
                )
                out_png = os.path.join(
                    out_dir,
                    f"overlay_{tag_fn}_sidx_{sidx:04d}_raw_{raw_fi:05d}.png",
                )
                cv2.imwrite(out_png, cv2.cvtColor(ov, cv2.COLOR_RGB2BGR))
                h0, w0 = ov.shape[:2]
                scv = min(1.0, 480.0 / max(h0, 1))
                panels.append(
                    cv2.resize(ov, (int(w0 * scv), int(h0 * scv)), interpolation=cv2.INTER_AREA),
                )
            except Exception as _e_ov:
                print(f"[SAM3D-SPARSE] overlay failed sidx={sidx}: {_e_ov}")
                traceback.print_exc()

    log_path = os.path.join(out_dir, "run_log.json")
    summary: Dict[str, Any] = {
        "milestone_plan": {str(k): tag_map[k] for k in sorted(tag_map.keys())},
        "inference_chain_sampled": chain,
        "stride_interaction": int(stride),
        "stage1_lo_auto": stage1_interaction_lo_auto,
        "stage1_hi_auto": stage1_interaction_hi_auto,
        "approaching_padded": int(approaching_end_idx_padded),
        "interaction_end_padded": int(interaction_end_idx_padded),
        "config_path": cfg,
        "lambda_temp": float(lambda_temp),
        "temporal_reference": "sam-3d-objects/demo_proj_temporal.py",
        "entries": entries,
    }
    with open(log_path, "w") as _jf:
        json.dump(summary, _jf, indent=2)
    print(f"[SAM3D-SPARSE] wrote {log_path} ({len(entries)} inference(s))")

    if len(panels) > 0:
        vid_path = os.path.join(out_dir, "preview.mp4")
        writer = None
        try:
            writer = imageio.get_writer(
                vid_path,
                fps=2.0,
                codec="libx264",
                pixelformat="yuv420p",
                ffmpeg_params=["-crf", "28", "-preset", "veryfast"],
                macro_block_size=None,
            )
            for _p in panels:
                writer.append_data(_ensure_even_dimensions_rgb(np.ascontiguousarray(_p)))
            print(f"[SAM3D-SPARSE] preview video → {vid_path}")
        except Exception as _e_vid:
            print(f"[SAM3D-SPARSE] preview video failed: {_e_vid}")
        finally:
            if writer is not None:
                writer.close()

    return summary
