"""
in_the_wild_metric.py
In-the-wild evaluation metrics for hand-object interaction pose fitting.

Metrics
-------
1. avg_penetration_vol_cm3  : Average interpenetration volume between hand and object
                              over interaction frames (cm³).
2. avg_chamfer_dist_cm      : Average bidirectional Chamfer distance between hand and
                              object surfaces over interaction frames (cm).
3. avg_mask_iou_pct         : Average IoU between the rendered 3-D object silhouette
                              and the amodal segmentation mask over all frames (%).
4. obj_accel_cm_s2          : Mean vertex acceleration of the object over all frames (cm/s²).
   hand_accel_cm_s2         : Mean vertex acceleration of the hand over all frames (cm/s²).

Performance notes
-----------------
* Metric 3 (IoU)         : Batched GPU rasterization — all T frames in one shot (chunked to
                           avoid OOM).
* Metrics 1 & 2          : Per-interaction-frame trimesh operations run in parallel via a
                           thread pool (trimesh/embree releases the GIL).
* Metric 4 (acceleration): Pure NumPy, already vectorised.
"""

from __future__ import annotations

import os
from multiprocessing.pool import ThreadPool
from typing import List, Optional

import numpy as np
import torch
import trimesh
from scipy.spatial import cKDTree

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _make_trimesh(verts_np: np.ndarray, faces_np: np.ndarray) -> trimesh.Trimesh:
    """Create a Trimesh, attempting to fill holes for watertightness."""
    mesh = trimesh.Trimesh(vertices=verts_np, faces=faces_np, process=False)
    if not mesh.is_watertight:
        trimesh.repair.fill_holes(mesh)
    return mesh


# ---------------------------------------------------------------------------
# Metric 1: Interpenetration volume  (single frame — called from thread pool)
# ---------------------------------------------------------------------------


def compute_penetration_volume_cm3(
    hand_verts: np.ndarray,
    hand_faces: np.ndarray,
    obj_verts: np.ndarray,
    obj_faces: np.ndarray,
    n_samples: int = 10_000,
    unit_to_cm: float = 1.0,
) -> float:
    """
    Monte-Carlo estimate of the interpenetration volume between hand and object.

    Parameters
    ----------
    hand_verts / obj_verts : (V, 3) arrays in original scene units.
    unit_to_cm             : multiply to convert to cm (e.g. 100 if units are metres).
    n_samples              : random samples drawn inside the union bounding box.

    Returns
    -------
    Penetration volume in cm³.
    """
    hv = _to_numpy(hand_verts) * unit_to_cm
    ov = _to_numpy(obj_verts) * unit_to_cm
    hf = _to_numpy(hand_faces).astype(np.int32)
    of = _to_numpy(obj_faces).astype(np.int32)

    hand_mesh = _make_trimesh(hv, hf)
    obj_mesh = _make_trimesh(ov, of)

    bbox_min = np.minimum(hv.min(0), ov.min(0))
    bbox_max = np.maximum(hv.max(0), ov.max(0))
    bbox_vol_cm3 = float(np.prod(bbox_max - bbox_min))
    if bbox_vol_cm3 <= 0:
        return 0.0

    samples = np.random.uniform(bbox_min, bbox_max, (n_samples, 3)).astype(np.float32)

    try:
        inside_hand = hand_mesh.contains(samples)
        inside_obj = obj_mesh.contains(samples)
        frac_both = (inside_hand & inside_obj).mean()
    except Exception:
        return 0.0

    return float(frac_both * bbox_vol_cm3)


# ---------------------------------------------------------------------------
# Metric 2: Bidirectional Chamfer distance
# ---------------------------------------------------------------------------
# Two implementations are provided:
#   • compute_chamfer_distance_cm      — single frame, CPU (kept for API compat)
#   • compute_chamfer_distance_batch   — all interaction frames, GPU via
#                                        pytorch3d.ops.knn_points (fast path used
#                                        inside compute_all_metrics)


def compute_chamfer_distance_cm(
    hand_verts: np.ndarray,
    hand_faces: np.ndarray,
    obj_verts: np.ndarray,
    obj_faces: np.ndarray,
    n_samples: int = 3_000,
    unit_to_cm: float = 1.0,
) -> float:
    """Single-frame CPU fallback. Returns distance in cm."""
    hv = _to_numpy(hand_verts) * unit_to_cm
    ov = _to_numpy(obj_verts) * unit_to_cm
    hf = _to_numpy(hand_faces).astype(np.int32)
    of = _to_numpy(obj_faces).astype(np.int32)

    hand_mesh = _make_trimesh(hv, hf)
    obj_mesh = _make_trimesh(ov, of)

    hand_pts = trimesh.sample.sample_surface(hand_mesh, count=n_samples)[0]
    obj_pts = trimesh.sample.sample_surface(obj_mesh, count=n_samples)[0]

    tree_obj = cKDTree(obj_pts)
    dist_h2o, _ = tree_obj.query(hand_pts, workers=-1)
    tree_hand = cKDTree(hand_pts)
    dist_o2h, _ = tree_hand.query(obj_pts, workers=-1)

    return float((dist_h2o.mean() + dist_o2h.mean()) / 2.0)


def compute_chamfer_distance_batch(
    hand_verts_seq: np.ndarray,  # (T, V_hand, 3) cm-units
    hand_faces: np.ndarray,
    obj_verts_seq: np.ndarray,  # (T, V_obj, 3) cm-units
    obj_faces: np.ndarray,
    inter_frames: List[int],
    n_samples: int = 3_000,
    device: str = 'cuda',
) -> float:
    """
    Batched bidirectional Chamfer distance using pytorch3d.ops.knn_points.

    1. Sample surface points per interaction frame (CPU, trimesh).
    2. Stack into (N_inter, n_samples, 3) GPU tensors.
    3. One knn_points call for forward + one for backward direction.

    Returns mean Chamfer distance in cm over all interaction frames.
    """
    from pytorch3d.ops import knn_points

    if not inter_frames:
        return 0.0

    dev = torch.device(device)
    hand_pts_list: List[np.ndarray] = []
    obj_pts_list: List[np.ndarray] = []

    hf = hand_faces.astype(np.int32)
    of = obj_faces.astype(np.int32)

    for fi in inter_frames:
        hand_mesh = _make_trimesh(hand_verts_seq[fi], hf)
        obj_mesh = _make_trimesh(obj_verts_seq[fi], of)
        hand_pts_list.append(trimesh.sample.sample_surface(hand_mesh, count=n_samples)[0])
        obj_pts_list.append(trimesh.sample.sample_surface(obj_mesh, count=n_samples)[0])

    hand_t = torch.tensor(np.stack(hand_pts_list), dtype=torch.float32, device=dev)  # (N, P, 3)
    obj_t = torch.tensor(np.stack(obj_pts_list), dtype=torch.float32, device=dev)  # (N, P, 3)

    with torch.no_grad():
        # knn_points returns squared L2 distances → sqrt for actual cm distances
        nn_h2o = knn_points(hand_t, obj_t, K=1, return_nn=False)  # dists: (N, P, 1)
        nn_o2h = knn_points(obj_t, hand_t, K=1, return_nn=False)

        d_h2o = nn_h2o.dists[..., 0].sqrt().mean(dim=1)  # (N,) in cm
        d_o2h = nn_o2h.dists[..., 0].sqrt().mean(dim=1)  # (N,) in cm
        chamfer = ((d_h2o + d_o2h) / 2.0).mean().item()

    return float(chamfer)


# Worker for penetration volume only (Chamfer is now GPU-batched separately)
def _penetration_vol_worker(args) -> tuple:
    fi, hv, hf, ov, of, n_vol, unit_to_cm = args
    hv = hv * unit_to_cm
    ov = ov * unit_to_cm

    hand_mesh = _make_trimesh(hv, hf)
    obj_mesh = _make_trimesh(ov, of)

    bbox_min = np.minimum(hv.min(0), ov.min(0))
    bbox_max = np.maximum(hv.max(0), ov.max(0))
    bbox_vol_cm3 = float(np.prod(bbox_max - bbox_min))
    if bbox_vol_cm3 <= 0:
        return fi, 0.0

    samples = np.random.uniform(bbox_min, bbox_max, (n_vol, 3)).astype(np.float32)
    try:
        frac = (hand_mesh.contains(samples) & obj_mesh.contains(samples)).mean()
        vol = float(frac * bbox_vol_cm3)
    except Exception:
        vol = 0.0

    return fi, vol


# ---------------------------------------------------------------------------
# Metric 3: Mask IoU — batched GPU rasterization over all T frames
# ---------------------------------------------------------------------------


def compute_mask_iou_batch(
    obj_verts_seq: np.ndarray,  # (T, V, 3)
    obj_faces: np.ndarray,  # (F, 3)
    focal_lengths: np.ndarray,  # (T, 2)  [fx, fy]
    principal_points: np.ndarray,  # (T, 2)  [cx, cy]
    H: int,
    W: int,
    amodal_masks: np.ndarray,  # (T, H, W) float [0, 1]
    device: str = 'cuda',
    batch_size: int = 64,
) -> float:
    """
    Batched IoU: rasterise all T frames in GPU batches, no per-frame Python loop.

    Returns average IoU in [0, 1].
    """
    from pytorch3d.renderer import (MeshRasterizer, PerspectiveCameras, RasterizationSettings)
    from pytorch3d.structures import Meshes

    T = obj_verts_seq.shape[0]
    dev = torch.device(device)

    faces_t = torch.from_numpy(obj_faces.astype(np.int64)).long().to(dev)
    ious: List[float] = []

    for start in range(0, T, batch_size):
        end = min(start + batch_size, T)
        bs = end - start

        verts_list = [torch.from_numpy(obj_verts_seq[i].astype(np.float32)).to(dev) for i in range(start, end)]
        mesh_batch = Meshes(verts=verts_list, faces=[faces_t] * bs)

        fl_t = torch.tensor(focal_lengths[start:end], dtype=torch.float32, device=dev)
        pp_t = torch.tensor(principal_points[start:end], dtype=torch.float32, device=dev)
        cameras = PerspectiveCameras(
            focal_length=fl_t,
            principal_point=pp_t,
            image_size=[(H, W)] * bs,
            in_ndc=False,
            device=dev,
        )

        raster_settings = RasterizationSettings(
            image_size=(H, W),
            blur_radius=0.0,
            faces_per_pixel=1,
            bin_size=0,
        )
        rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)

        with torch.no_grad():
            frags = rasterizer(mesh_batch)
            rendered_bin = (frags.zbuf[..., 0] > 0).float()  # (bs, H, W)

        amodal_batch = torch.from_numpy(amodal_masks[start:end].astype(np.float32)).to(dev)
        amodal_bin = (amodal_batch > 0.5).float()

        inter = (rendered_bin * amodal_bin).sum(dim=(-2, -1))  # (bs,)
        union = (rendered_bin + amodal_bin).clamp(0, 1).sum(dim=(-2, -1))
        iou_b = (inter / (union + 1e-6)).cpu().numpy()
        ious.extend(iou_b.tolist())

        del mesh_batch, frags, rendered_bin, amodal_batch, amodal_bin
        if device != 'cpu':
            torch.cuda.empty_cache()

    return float(np.mean(ious)) if ious else 0.0


# Keep the single-frame version for external use / backward compatibility
def compute_mask_iou_single_frame(
    obj_verts_posed: np.ndarray,
    obj_faces: np.ndarray,
    focal_length_xy: np.ndarray,
    principal_point_xy: np.ndarray,
    H: int,
    W: int,
    amodal_mask: np.ndarray,
    device: str = 'cuda',
) -> float:
    return compute_mask_iou_batch(
        obj_verts_seq=obj_verts_posed[None],
        obj_faces=obj_faces,
        focal_lengths=focal_length_xy[None],
        principal_points=principal_point_xy[None],
        H=H,
        W=W,
        amodal_masks=amodal_mask[None],
        device=device,
        batch_size=1,
    )


# ---------------------------------------------------------------------------
# Metric 4: Vertex acceleration (already vectorised)
# ---------------------------------------------------------------------------


def compute_vertex_acceleration(
    verts_seq: np.ndarray,
    fps: float,
    unit_to_cm: float = 1.0,
) -> float:
    """
    Mean vertex acceleration over the sequence using second-order finite differences.

    Parameters
    ----------
    verts_seq  : (T, V, 3) vertex positions over time in original scene units.
    fps        : frames per second of the sequence.
    unit_to_cm : multiply to convert to cm.

    Returns
    -------
    Mean acceleration magnitude in cm/s².
    """
    pos = _to_numpy(verts_seq) * unit_to_cm  # (T, V, 3)
    T = pos.shape[0]
    if T < 3:
        return 0.0

    dt = 1.0 / fps
    accel = (pos[2:] - 2.0 * pos[1:-1] + pos[:-2]) / (dt**2)  # (T-2, V, 3)
    return float(np.linalg.norm(accel, axis=-1).mean())


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def compute_all_metrics(
    obj_verts_seq: np.ndarray,  # (T, V_obj, 3)  posed, world-space
    obj_faces: np.ndarray,  # (F_obj, 3)
    hand_verts_seq: np.ndarray,  # (T, V_hand, 3) posed, world-space
    hand_faces: np.ndarray,  # (F_hand, 3)
    amodal_masks: np.ndarray,  # (T, H, W)  float [0, 1]
    focal_lengths: np.ndarray,  # (T, 2)  [fx, fy]
    principal_points: np.ndarray,  # (T, 2)  [cx, cy]
    H: int,
    W: int,
    interaction_start: int,  # first interaction frame index (inclusive)
    interaction_end: int,  # last  interaction frame index (exclusive)
    fps: float = 30.0,
    unit_to_cm: float = 1.0,
    device: str = 'cuda',
    n_samples_vol: int = 10_000,
    n_samples_chamfer: int = 3_000,
    iou_batch_size: int = 64,
    num_workers: Optional[int] = None,
    verbose: bool = True,
) -> dict:
    """
    Compute all four in-the-wild metrics and return them in a dict.

    Returns
    -------
    {
      'avg_penetration_vol_cm3' : float,
      'avg_chamfer_dist_cm'     : float,
      'avg_mask_iou_pct'        : float,
      'obj_accel_cm_s2'         : float,
      'hand_accel_cm_s2'        : float,
    }
    """
    obj_verts_seq = _to_numpy(obj_verts_seq)
    hand_verts_seq = _to_numpy(hand_verts_seq)
    obj_faces = _to_numpy(obj_faces).astype(np.int32)
    hand_faces = _to_numpy(hand_faces).astype(np.int32)
    amodal_masks = _to_numpy(amodal_masks)
    focal_lengths = _to_numpy(focal_lengths)
    principal_points = _to_numpy(principal_points)

    T = obj_verts_seq.shape[0]
    inter_frames = list(range(interaction_start, interaction_end))
    n_workers = num_workers if num_workers is not None else max(1, os.cpu_count() or 1)

    # ------------------------------------------------------------------
    # 1: Penetration volume — parallel CPU (trimesh.contains, per-frame)
    # ------------------------------------------------------------------
    if verbose:
        print(f"  [metric 1] penetration vol: {len(inter_frames)} interaction frames, "
              f"{n_workers} workers ...", flush=True)

    penet_vols: List[float] = []

    if inter_frames:
        args_list = [(fi, hand_verts_seq[fi].copy(), hand_faces, obj_verts_seq[fi].copy(), obj_faces, n_samples_vol, unit_to_cm) for fi in inter_frames]
        try:
            with ThreadPool(processes=n_workers) as pool:
                pen_results = pool.map(_penetration_vol_worker, args_list)
        except Exception:
            pen_results = [_penetration_vol_worker(a) for a in args_list]

        pen_results.sort(key=lambda x: x[0])
        penet_vols = [v for _, v in pen_results]

    avg_penetration_vol = float(np.mean(penet_vols)) if penet_vols else 0.0

    # ------------------------------------------------------------------
    # 2: Chamfer distance — batched GPU via pytorch3d knn_points
    # ------------------------------------------------------------------
    if verbose:
        print(f"  [metric 2] Chamfer distance: {len(inter_frames)} frames on GPU ...", flush=True)

    # Convert to cm before passing (knn_points works in whatever unit we give it)
    obj_verts_cm = obj_verts_seq * unit_to_cm
    hand_verts_cm = hand_verts_seq * unit_to_cm

    avg_chamfer_dist = compute_chamfer_distance_batch(
        hand_verts_seq=hand_verts_cm,
        hand_faces=hand_faces,
        obj_verts_seq=obj_verts_cm,
        obj_faces=obj_faces,
        inter_frames=inter_frames,
        n_samples=n_samples_chamfer,
        device=device,
    )

    # ------------------------------------------------------------------
    # 3: Mask IoU — single batched GPU rasterization pass
    # ------------------------------------------------------------------
    if verbose:
        print(f"  [metric 3] IoU over {T} frames (batch={iou_batch_size}) ...", flush=True)

    avg_iou = compute_mask_iou_batch(
        obj_verts_seq=obj_verts_seq,
        obj_faces=obj_faces,
        focal_lengths=focal_lengths,
        principal_points=principal_points,
        H=H,
        W=W,
        amodal_masks=amodal_masks,
        device=device,
        batch_size=iou_batch_size,
    )
    avg_iou_pct = avg_iou * 100.0

    # ------------------------------------------------------------------
    # 4: Vertex acceleration (object + hand) — vectorised NumPy
    # ------------------------------------------------------------------
    if verbose:
        print("  [metric 4] vertex acceleration ...", flush=True)

    obj_accel = compute_vertex_acceleration(obj_verts_seq, fps, unit_to_cm)
    hand_accel = compute_vertex_acceleration(hand_verts_seq, fps, unit_to_cm)

    results_dict = {
        'avg_penetration_vol_cm3': avg_penetration_vol,
        'avg_chamfer_dist_cm': avg_chamfer_dist,
        'avg_mask_iou_pct': avg_iou_pct,
        'obj_accel_cm_s2': obj_accel,
        'hand_accel_cm_s2': hand_accel,
    }

    if verbose:
        print("\n===== In-the-Wild Metrics =====")
        print(f"  Avg penetration volume   : {avg_penetration_vol:.4f}  cm³")
        print(f"  Avg Chamfer distance     : {avg_chamfer_dist:.4f}  cm")
        print(f"  Avg mask IoU             : {avg_iou_pct:.2f}  %")
        print(f"  Obj  vertex acceleration : {obj_accel:.4f}  cm/s²")
        print(f"  Hand vertex acceleration : {hand_accel:.4f}  cm/s²")
        print("================================")

    return results_dict
