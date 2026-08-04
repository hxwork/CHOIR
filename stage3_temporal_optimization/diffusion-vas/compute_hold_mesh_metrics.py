from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np
import trimesh
from scipy.spatial import cKDTree

from in_the_wild_metric import (
    compute_chamfer_distance_batch,
    compute_chamfer_distance_cm,
    compute_mask_iou_batch,
    compute_vertex_acceleration,
)


METRIC_KEYS = ["mIoU", "Pen. Ratio", "H-O Dist.", "Acc_h", "Acc_o"]
DEFAULT_MESH_ROOT = "rendering_for_paper/ours_HO3D"
DEFAULT_ORIGINAL_ROOT = "HO3D_backup"
DEFAULT_OUTPUT_DIR = "rendering_for_paper/metrics_results_hold_new"


def frame_number(path: Path) -> int:
    nums = re.findall(r"\d+", path.stem)
    return int(nums[0]) if nums else -1


def load_mesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(path, force="mesh", process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.dump()))
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Unsupported mesh type at {path}: {type(mesh)!r}")
    return mesh


def make_contains_mesh(vertices: np.ndarray, faces: np.ndarray) -> trimesh.Trimesh:
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    if not mesh.is_watertight:
        trimesh.repair.fill_holes(mesh)
    return mesh


def mesh_frame(mesh_seq: MeshSeq, frame_idx: int) -> np.ndarray:
    return mesh_seq[frame_idx] if isinstance(mesh_seq, list) else mesh_seq[frame_idx]


def face_frame(face_seq: FaceSeq, frame_idx: int) -> np.ndarray:
    return face_seq[frame_idx] if isinstance(face_seq, list) else face_seq


def mesh_seq_len(mesh_seq: MeshSeq) -> int:
    return len(mesh_seq) if isinstance(mesh_seq, list) else int(mesh_seq.shape[0])


def object_bbox_centers(obj_verts_seq: MeshSeq, unit_to_cm: float) -> np.ndarray:
    centers = []
    for frame_idx in range(mesh_seq_len(obj_verts_seq)):
        verts = mesh_frame(obj_verts_seq, frame_idx) * unit_to_cm
        centers.append((verts.min(axis=0) + verts.max(axis=0)) * 0.5)
    return np.stack(centers, axis=0)


def compute_center_acceleration(centers: np.ndarray) -> float:
    if centers.shape[0] < 3:
        return 0.0
    accel = centers[2:] - 2.0 * centers[1:-1] + centers[:-2]
    return float(np.linalg.norm(accel, axis=-1).mean())


def list_mesh_files(seq_dir: Path) -> tuple[list[Path], list[Path]]:
    hand_dir = seq_dir / "hand"
    obj_dir = seq_dir / "object"
    if not hand_dir.is_dir() or not obj_dir.is_dir():
        raise FileNotFoundError(f"Expected hand/ and object/ mesh folders under {seq_dir}")

    hand_files = sorted([*hand_dir.glob("*.ply"), *hand_dir.glob("*.obj")], key=frame_number)
    obj_files = sorted([*obj_dir.glob("*.ply"), *obj_dir.glob("*.obj")], key=frame_number)
    if not hand_files or not obj_files:
        raise FileNotFoundError(f"No mesh files found under {seq_dir}")
    return hand_files, obj_files


MeshSeq = np.ndarray | list[np.ndarray]
FaceSeq = np.ndarray | list[np.ndarray]


def load_mesh_sequence(seq_dir: Path) -> tuple[np.ndarray, np.ndarray, MeshSeq, FaceSeq, int]:
    hand_files, obj_files = list_mesh_files(seq_dir)
    num_frames = min(len(hand_files), len(obj_files))
    if len(hand_files) != len(obj_files):
        print(f"[WARN] {seq_dir.name}: hand/object frame count mismatch; truncating to {num_frames}")

    hand_mesh0 = load_mesh(hand_files[0])
    hand_faces = np.asarray(hand_mesh0.faces, dtype=np.int32)

    hand_verts = []
    obj_verts = []
    obj_faces = []
    for hand_file, obj_file in zip(hand_files[:num_frames], obj_files[:num_frames]):
        hand_mesh = load_mesh(hand_file)
        obj_mesh = load_mesh(obj_file)
        if hand_mesh.vertices.shape[0] != hand_mesh0.vertices.shape[0]:
            raise ValueError(f"Hand vertex count changed at {hand_file}")
        hand_verts.append(np.asarray(hand_mesh.vertices, dtype=np.float32))
        obj_verts.append(np.asarray(obj_mesh.vertices, dtype=np.float32))
        obj_faces.append(np.asarray(obj_mesh.faces, dtype=np.int32))

    obj_vert_counts = {verts.shape[0] for verts in obj_verts}
    obj_face_counts = {faces.shape[0] for faces in obj_faces}
    if len(obj_vert_counts) == 1 and len(obj_face_counts) == 1:
        obj_verts_out: MeshSeq = np.stack(obj_verts, axis=0)
        obj_faces_out: FaceSeq = obj_faces[0]
    else:
        print(
            f"[WARN] {seq_dir.name}: object topology changes across frames; "
            "keeping per-frame object meshes and using center trajectory for Acc_o"
        )
        obj_verts_out = obj_verts
        obj_faces_out = obj_faces

    return (
        np.stack(hand_verts, axis=0),
        hand_faces,
        obj_verts_out,
        obj_faces_out,
        num_frames,
    )


def load_intrinsics(original_seq_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    intrinsics_path = original_seq_dir / "intrinsics.json"
    with intrinsics_path.open("r", encoding="utf-8") as f:
        intrinsics = np.asarray(json.load(f)["intrinsics"], dtype=np.float32)
    focal = np.asarray([intrinsics[0, 0], intrinsics[1, 1]], dtype=np.float32)
    principal = np.asarray([intrinsics[0, 2], intrinsics[1, 2]], dtype=np.float32)
    return focal, principal


def load_camera_for_mask(original_seq_dir: Path, mask_shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    warnings: list[str] = []
    focal, principal = load_intrinsics(original_seq_dir)
    mask_h, mask_w = mask_shape
    if (mask_h, mask_w) == (480, 640):
        return focal, principal, warnings

    bbox_path = original_seq_dir / "global_bbox.json"
    if not bbox_path.exists():
        warnings.append("global_bbox.json missing; using original intrinsics for non-original-size masks")
        return focal, principal, warnings

    with bbox_path.open("r", encoding="utf-8") as f:
        bboxes = json.load(f)
    if not bboxes:
        warnings.append("global_bbox.json is empty; using original intrinsics for non-original-size masks")
        return focal, principal, warnings

    x1, y1, x2, y2 = [float(v) for v in bboxes[0]]
    w_crop = max(x2 - x1, 1.0)
    h_crop = max(y2 - y1, 1.0)
    scale = min(float(mask_w) / w_crop, float(mask_h) / h_crop)
    new_w = int(w_crop * scale)
    new_h = int(h_crop * scale)
    pad_left = (int(mask_w) - new_w) // 2
    pad_top = (int(mask_h) - new_h) // 2

    crop_focal = np.asarray([focal[0] * scale, focal[1] * scale], dtype=np.float32)
    crop_principal = np.asarray(
        [(principal[0] - x1) * scale + pad_left, (principal[1] - y1) * scale + pad_top],
        dtype=np.float32,
    )
    return crop_focal, crop_principal, warnings


def load_mask(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Failed to read mask: {path}")
    if image.ndim == 3 and image.shape[2] == 4:
        mask = image[:, :, 3] > 0
    elif image.ndim == 3:
        mask = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) > 0
    else:
        mask = image > 0
    return mask.astype(np.float32)


def load_masks(original_seq_dir: Path, num_frames: int) -> tuple[np.ndarray | None, str | None, list[str]]:
    warnings: list[str] = []
    mask_dir = original_seq_dir / "amodal_masks"
    mask_source = "amodal_masks"
    if not any(mask_dir.glob("*")):
        mask_dir = original_seq_dir / "obj_masks"
        mask_source = "obj_masks"
        warnings.append("amodal_masks missing or empty; fell back to obj_masks")
    if not mask_dir.is_dir():
        warnings.append("no mask directory found; mIoU will be null")
        return None, None, warnings

    mask_files = sorted(
        [p for p in mask_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}],
        key=frame_number,
    )
    if not mask_files:
        warnings.append("no mask files found; mIoU will be null")
        return None, None, warnings
    if len(mask_files) < num_frames:
        warnings.append(f"only {len(mask_files)} masks for {num_frames} mesh frames; truncating metrics")

    masks = np.stack([load_mask(path) for path in mask_files[:num_frames]], axis=0)
    return masks, mask_source, warnings


def load_interaction_frames(original_seq_dir: Path, num_frames: int) -> tuple[list[int], list[str]]:
    return list(range(num_frames)), []


def compute_penetration_ratio(
    hand_verts_seq: np.ndarray,
    obj_verts_seq: MeshSeq,
    obj_faces: FaceSeq,
    inter_frames: Iterable[int],
) -> float:
    ratios = []
    for frame_idx in inter_frames:
        obj_mesh = make_contains_mesh(mesh_frame(obj_verts_seq, frame_idx), face_frame(obj_faces, frame_idx))
        try:
            inside = obj_mesh.contains(hand_verts_seq[frame_idx])
            ratios.append(float(np.mean(inside)))
        except Exception as exc:
            print(f"[WARN] penetration ratio failed at frame {frame_idx}: {exc}")
    return float(np.mean(ratios)) if ratios else 0.0


def compute_chamfer_metric(
    hand_verts_seq: np.ndarray,
    hand_faces: np.ndarray,
    obj_verts_seq: np.ndarray,
    obj_faces: np.ndarray,
    inter_frames: list[int],
    unit_to_cm: float,
    samples: int,
    device: str,
) -> float:
    if not inter_frames:
        return 0.0
    hand_cm = hand_verts_seq * unit_to_cm
    obj_cm = obj_verts_seq * unit_to_cm
    try:
        return compute_chamfer_distance_batch(
            hand_verts_seq=hand_cm,
            hand_faces=hand_faces,
            obj_verts_seq=obj_cm,
            obj_faces=obj_faces,
            inter_frames=inter_frames,
            n_samples=samples,
            device=device,
        )
    except Exception as exc:
        print(f"[WARN] batched Chamfer failed, falling back to CPU: {exc}")
        values = [
            compute_chamfer_distance_cm(
                hand_verts_seq[i],
                hand_faces,
                obj_verts_seq[i],
                obj_faces,
                n_samples=samples,
                unit_to_cm=unit_to_cm,
            )
            for i in inter_frames
        ]
        return float(np.mean(values)) if values else 0.0


def compute_nearest_surface_distance_metric(
    hand_verts_seq: np.ndarray,
    hand_faces: np.ndarray,
    obj_verts_seq: MeshSeq,
    obj_faces: FaceSeq,
    inter_frames: list[int],
    unit_to_cm: float,
    samples: int,
) -> float:
    """Mean per-frame closest hand-object surface distance in cm."""
    if not inter_frames:
        return 0.0
    values = []
    hand_faces_np = hand_faces.astype(np.int32)
    for frame_idx in inter_frames:
        obj_faces_np = face_frame(obj_faces, frame_idx).astype(np.int32)
        hand_mesh = trimesh.Trimesh(
            vertices=hand_verts_seq[frame_idx] * unit_to_cm,
            faces=hand_faces_np,
            process=False,
        )
        obj_mesh = trimesh.Trimesh(
            vertices=mesh_frame(obj_verts_seq, frame_idx) * unit_to_cm,
            faces=obj_faces_np,
            process=False,
        )
        hand_pts = trimesh.sample.sample_surface(hand_mesh, count=samples)[0]
        obj_pts = trimesh.sample.sample_surface(obj_mesh, count=samples)[0]
        dist_h2o, _ = cKDTree(obj_pts).query(hand_pts, workers=-1)
        dist_o2h, _ = cKDTree(hand_pts).query(obj_pts, workers=-1)
        values.append(float(min(dist_h2o.min(), dist_o2h.min())))
    return float(np.mean(values)) if values else 0.0


def compute_mask_iou_metric(
    obj_verts_seq: MeshSeq,
    obj_faces: FaceSeq,
    masks: np.ndarray | None,
    focal: np.ndarray,
    principal: np.ndarray,
    device: str,
    batch_size: int,
    flip_xy_for_iou: bool,
) -> float | None:
    if masks is None:
        return None
    num_frames = min(mesh_seq_len(obj_verts_seq), masks.shape[0])
    h, w = masks.shape[1:3]
    focal_lengths = np.repeat(focal[None], num_frames, axis=0)
    principal_points = np.repeat(principal[None], num_frames, axis=0)
    flip = np.asarray([-1.0, -1.0, 1.0], dtype=np.float32)
    if isinstance(obj_verts_seq, list) or isinstance(obj_faces, list):
        ious = []
        for frame_idx in range(num_frames):
            obj_verts_for_iou = mesh_frame(obj_verts_seq, frame_idx)
            if flip_xy_for_iou:
                obj_verts_for_iou = obj_verts_for_iou * flip
            ious.append(
                compute_mask_iou_batch(
                    obj_verts_seq=obj_verts_for_iou[None],
                    obj_faces=face_frame(obj_faces, frame_idx),
                    focal_lengths=focal_lengths[frame_idx : frame_idx + 1],
                    principal_points=principal_points[frame_idx : frame_idx + 1],
                    H=h,
                    W=w,
                    amodal_masks=masks[frame_idx : frame_idx + 1],
                    device=device,
                    batch_size=1,
                )
            )
        iou = float(np.mean(ious)) if ious else 0.0
    else:
        obj_verts_for_iou = obj_verts_seq[:num_frames]
        if flip_xy_for_iou:
            # HOLD paper meshes are saved in the flattened visualization frame.
            # Convert them back to the camera frame before rasterizing masks.
            obj_verts_for_iou = obj_verts_for_iou * flip
        iou = compute_mask_iou_batch(
            obj_verts_seq=obj_verts_for_iou,
            obj_faces=obj_faces,
            focal_lengths=focal_lengths,
            principal_points=principal_points,
            H=h,
            W=w,
            amodal_masks=masks[:num_frames],
            device=device,
            batch_size=batch_size,
        )
    return float(iou * 100.0)


def compute_sequence_metrics(
    seq_dir: Path,
    original_root: Path,
    output_dir: Path,
    *,
    device: str,
    unit_to_cm: float,
    chamfer_samples: int,
    iou_batch_size: int,
    render_iou: bool,
    flip_xy_for_iou: bool,
) -> dict:
    seq_name = seq_dir.name
    original_seq_dir = original_root / seq_name
    warnings: list[str] = []

    hand_verts, hand_faces, obj_verts, obj_faces, num_frames = load_mesh_sequence(seq_dir)
    inter_frames, range_warnings = load_interaction_frames(original_seq_dir, num_frames)
    warnings.extend(range_warnings)

    masks = None
    mask_source = None
    focal = principal = None
    if render_iou:
        try:
            masks, mask_source, mask_warnings = load_masks(original_seq_dir, num_frames)
            warnings.extend(mask_warnings)
            if masks is not None:
                focal, principal, camera_warnings = load_camera_for_mask(original_seq_dir, masks.shape[1:3])
                warnings.extend(camera_warnings)
        except Exception as exc:
            warnings.append(f"mIoU setup failed: {exc}")

    print(f"[{seq_name}] frames={num_frames}, interaction={len(inter_frames)}")
    pen_ratio = compute_penetration_ratio(hand_verts, obj_verts, obj_faces, inter_frames)
    ho_dist = compute_nearest_surface_distance_metric(
        hand_verts,
        hand_faces,
        obj_verts,
        obj_faces,
        inter_frames,
        unit_to_cm,
        chamfer_samples,
    )
    miou = None
    if render_iou and focal is not None and principal is not None:
        miou = compute_mask_iou_metric(
            obj_verts,
            obj_faces,
            masks,
            focal,
            principal,
            device,
            iou_batch_size,
            flip_xy_for_iou,
        )
    hand_acc = compute_vertex_acceleration(hand_verts, fps=1.0, unit_to_cm=unit_to_cm)
    obj_acc = compute_center_acceleration(object_bbox_centers(obj_verts, unit_to_cm=unit_to_cm))
    obj_acc_mode = "bbox_center"

    result = {
        "sequence_name": seq_name,
        "num_frames": int(num_frames),
        "interaction_start": int(inter_frames[0]) if inter_frames else None,
        "interaction_end": int(inter_frames[-1] + 1) if inter_frames else None,
        "mask_source": mask_source,
        "acceleration_modes": {
            "Acc_h": "vertex",
            "Acc_o": obj_acc_mode,
        },
        "metrics": {
            "mIoU": miou,
            "Pen. Ratio": pen_ratio,
            "H-O Dist.": ho_dist,
            "Acc_h": hand_acc,
            "Acc_o": obj_acc,
        },
        "units": {
            "mIoU": "%",
            "Pen. Ratio": "ratio",
            "H-O Dist.": "cm",
            "Acc_h": "cm/frame^2",
            "Acc_o": "cm/frame^2",
        },
        "warnings": warnings,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / f"{seq_name}_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return result


def summarize_results(results: list[dict]) -> dict:
    summary = {}
    for key in METRIC_KEYS:
        values = [r["metrics"].get(key) for r in results]
        values = [float(v) for v in values if v is not None and np.isfinite(v)]
        if not values:
            summary[key] = None
            continue
        arr = np.asarray(values, dtype=np.float64)
        summary[key] = {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "min": float(arr.min()),
            "max": float(arr.max()),
        }
    return summary


def write_csv(results: list[dict], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sequence_name", "num_frames", *METRIC_KEYS])
        for result in results:
            writer.writerow([
                result["sequence_name"],
                result["num_frames"],
                *(result["metrics"].get(key) for key in METRIC_KEYS),
            ])


def iter_sequences(mesh_root: Path, requested: list[str] | None) -> list[Path]:
    if requested:
        return [mesh_root / name for name in requested]
    return sorted([p for p in mesh_root.iterdir() if p.is_dir() and p.name.startswith("hold_")])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute HOLD metrics from saved hand/object meshes.")
    parser.add_argument("--mesh_root", type=Path, default=Path(DEFAULT_MESH_ROOT), help="Root containing hold_id/hand and hold_id/object mesh folders.")
    parser.add_argument("--original_root", type=Path, default=Path(DEFAULT_ORIGINAL_ROOT), help="Root containing original HOLD/HO3D data and masks.")
    parser.add_argument("--output_dir", type=Path, default=Path(DEFAULT_OUTPUT_DIR), help="Directory for per-sequence and aggregate metric outputs.")
    parser.add_argument("--device", type=str, default="cuda", help="Device for PyTorch3D mask IoU rendering.")
    parser.add_argument("--sequence", type=str, nargs="*", default=None, help="Optional sequence names to process.")
    parser.add_argument("--no_render_iou", action="store_true", help="Skip rendered object mask IoU.")
    parser.set_defaults(flip_xy_for_iou=True)
    parser.add_argument("--flip_xy_for_iou", dest="flip_xy_for_iou", action="store_true", help="Apply xy flip before mask IoU rendering.")
    parser.add_argument("--no_flip_xy_for_iou", dest="flip_xy_for_iou", action="store_false", help="Disable xy flip before mask IoU rendering.")
    parser.add_argument("--iou_batch_size", type=int, default=32, help="Batch size for rendered mask IoU.")
    parser.add_argument("--chamfer_samples", type=int, default=3000, help="Surface samples per mesh for nearest H-O distance.")
    parser.add_argument("--unit_to_cm", type=float, default=100.0, help="Scale factor from mesh units to centimeters.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for seq_dir in iter_sequences(args.mesh_root, args.sequence):
        if not seq_dir.exists():
            print(f"[WARN] missing sequence directory: {seq_dir}")
            continue
        try:
            results.append(
                compute_sequence_metrics(
                    seq_dir,
                    args.original_root,
                    args.output_dir,
                    device=args.device,
                    unit_to_cm=args.unit_to_cm,
                    chamfer_samples=args.chamfer_samples,
                    iou_batch_size=args.iou_batch_size,
                    render_iou=not args.no_render_iou,
                    flip_xy_for_iou=args.flip_xy_for_iou,
                )
            )
        except Exception as exc:
            print(f"[ERROR] failed sequence {seq_dir.name}: {exc}")

    summary = summarize_results(results)
    aggregate = {"summary": summary, "per_sequence_results": results}
    with (args.output_dir / "all_hold_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(aggregate, f, indent=2)
    write_csv(results, args.output_dir / "all_hold_metrics.csv")

    print("\n===== HOLD Metrics Summary =====")
    for key in METRIC_KEYS:
        stats = summary.get(key)
        if stats is None:
            print(f"{key}: n/a")
        else:
            print(f"{key}: {stats['mean']:.6f}")


if __name__ == "__main__":
    main()
