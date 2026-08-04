from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from compute_hold_mesh_metrics import (
    compute_center_acceleration,
    compute_nearest_surface_distance_metric,
    compute_penetration_ratio,
    load_intrinsics,
    load_masks,
    load_mesh_sequence,
    object_bbox_centers,
    summarize_results,
    write_csv,
)
from in_the_wild_metric import compute_mask_iou_batch, compute_vertex_acceleration


METRIC_KEYS = ["mIoU", "Pen. Ratio", "H-O Dist.", "Acc_h", "Acc_o"]
DEFAULT_ABLATION_ROOT = "outputs/ablations"
DEFAULT_MESH_ROOT = "rendering_for_paper/ours_SelfCaptured"
DEFAULT_DATA_ROOT = "input_data"


def load_crop_camera(run_dir: Path, data_seq_dir: Path, num_frames: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
    warnings: list[str] = []
    focal, principal = load_intrinsics(data_seq_dir)
    bbox_path = run_dir / "global_bbox.json"
    if not bbox_path.exists():
        warnings.append("global_bbox.json missing; using original intrinsics without crop adjustment")
        return (
            np.repeat(focal[None], num_frames, axis=0),
            np.repeat(principal[None], num_frames, axis=0),
            warnings,
        )

    with bbox_path.open("r", encoding="utf-8") as f:
        bboxes = json.load(f)
    if not bboxes:
        warnings.append("global_bbox.json is empty; using original intrinsics without crop adjustment")
        return (
            np.repeat(focal[None], num_frames, axis=0),
            np.repeat(principal[None], num_frames, axis=0),
            warnings,
        )

    x1, y1, x2, y2 = [float(v) for v in bboxes[0]]
    h_out, w_out = 256.0, 512.0
    w_crop = max(x2 - x1, 1.0)
    h_crop = max(y2 - y1, 1.0)
    scale = min(w_out / w_crop, h_out / h_crop)
    new_w = int(w_crop * scale)
    new_h = int(h_crop * scale)
    pad_left = (int(w_out) - new_w) // 2
    pad_top = (int(h_out) - new_h) // 2

    crop_focal = np.asarray([focal[0] * scale, focal[1] * scale], dtype=np.float32)
    crop_principal = np.asarray(
        [(principal[0] - x1) * scale + pad_left, (principal[1] - y1) * scale + pad_top],
        dtype=np.float32,
    )
    return (
        np.repeat(crop_focal[None], num_frames, axis=0),
        np.repeat(crop_principal[None], num_frames, axis=0),
        warnings,
    )


def load_interaction_frames(run_dir: Path, num_frames: int) -> tuple[list[int], list[str]]:
    warnings: list[str] = []
    ranges_path = run_dir / "stage_frame_ranges.json"
    if not ranges_path.exists():
        ckpt_path = run_dir / "stage3_checkpoint.pt"
        if ckpt_path.exists():
            try:
                ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                start = int(ckpt["approaching_end_idx"])
                end = int(ckpt["interaction_end_idx"])
                start = max(0, min(start, num_frames))
                end = max(start, min(end, num_frames))
                if start < end:
                    warnings.append("stage_frame_ranges.json missing; loaded interaction range from stage3_checkpoint.pt")
                    return list(range(start, end)), warnings
            except Exception as exc:
                warnings.append(f"failed to load interaction range from stage3_checkpoint.pt: {exc}")
        warnings.append("stage_frame_ranges.json and usable checkpoint range missing; using all frames as interaction")
        return list(range(num_frames)), warnings

    with ranges_path.open("r", encoding="utf-8") as f:
        ranges = json.load(f)
    interaction = next((v for k, v in ranges.items() if "interaction" in k), None)
    if interaction is None:
        warnings.append("interaction range missing; using all frames as interaction")
        return list(range(num_frames)), warnings

    start = int(interaction.get("original_start", interaction.get("sampled_start", 0)))
    end = int(interaction.get("original_end", interaction.get("sampled_end", num_frames)))
    start = max(0, min(start, num_frames))
    end = max(start, min(end, num_frames))
    if start >= end:
        warnings.append("empty interaction range; using all frames as interaction")
        return list(range(num_frames)), warnings
    return list(range(start, end)), warnings


def run_dirs(ablation_root: Path, ablations: list[str] | None, video_ids: list[str] | None) -> list[tuple[str, str, Path]]:
    ablation_names = ablations or sorted([p.name for p in ablation_root.iterdir() if p.is_dir()])
    runs: list[tuple[str, str, Path]] = []
    for ablation in ablation_names:
        ablation_dir = ablation_root / ablation
        if not ablation_dir.is_dir():
            print(f"[WARN] missing ablation directory: {ablation_dir}")
            continue
        selected_videos = video_ids or sorted([p.name for p in ablation_dir.iterdir() if p.is_dir()])
        for video_id in selected_videos:
            run_dir = ablation_dir / video_id
            if run_dir.is_dir():
                runs.append((ablation, video_id, run_dir))
            else:
                print(f"[WARN] missing run directory: {run_dir}")
    return runs


def compute_run_metrics(
    ablation: str,
    video_id: str,
    run_dir: Path,
    mesh_root: Path,
    data_root: Path,
    *,
    device: str,
    unit_to_cm: float,
    chamfer_samples: int,
    iou_batch_size: int,
    render_iou: bool,
    flip_xy_for_iou: bool,
) -> dict:
    warnings: list[str] = []
    mesh_seq_dir = mesh_root / f"{video_id}_{ablation}"
    data_seq_dir = data_root / video_id
    if not mesh_seq_dir.is_dir():
        raise FileNotFoundError(f"Missing saved mesh directory: {mesh_seq_dir}")

    hand_verts, hand_faces, obj_verts, obj_faces, num_frames = load_mesh_sequence(mesh_seq_dir)
    inter_frames, inter_warnings = load_interaction_frames(run_dir, num_frames)
    warnings.extend(inter_warnings)

    masks = None
    mask_source = None
    focal_lengths = principal_points = None
    if render_iou:
        focal_lengths, principal_points, camera_warnings = load_crop_camera(run_dir, data_seq_dir, num_frames)
        warnings.extend(camera_warnings)
        masks, mask_source, mask_warnings = load_masks(run_dir, num_frames)
        warnings.extend(mask_warnings)

    print(f"[{ablation}/{video_id}] frames={num_frames}, interaction={len(inter_frames)}")
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
    if render_iou and masks is not None and focal_lengths is not None and principal_points is not None:
        n = min(num_frames, masks.shape[0])
        h, w = masks.shape[1:3]
        obj_verts_for_iou = obj_verts[:n]
        if flip_xy_for_iou:
            obj_verts_for_iou = obj_verts_for_iou * np.asarray([-1.0, -1.0, 1.0], dtype=np.float32)
        miou = compute_mask_iou_batch(
            obj_verts_seq=obj_verts_for_iou,
            obj_faces=obj_faces,
            focal_lengths=focal_lengths[:n],
            principal_points=principal_points[:n],
            H=h,
            W=w,
            amodal_masks=masks[:n],
            device=device,
            batch_size=iou_batch_size,
        ) * 100.0
    result = {
        "ablation": ablation,
        "video_id": video_id,
        "num_frames": int(num_frames),
        "interaction_start": int(inter_frames[0]) if inter_frames else None,
        "interaction_end": int(inter_frames[-1] + 1) if inter_frames else None,
        "mesh_dir": str(mesh_seq_dir),
        "mask_source": mask_source,
        "acceleration_modes": {
            "Acc_h": "vertex",
            "Acc_o": "bbox_center",
        },
        "metrics": {
            "mIoU": miou,
            "Pen. Ratio": pen_ratio,
            "H-O Dist.": ho_dist,
            "Acc_h": float(compute_vertex_acceleration(hand_verts, fps=1.0, unit_to_cm=unit_to_cm)),
            "Acc_o": float(compute_center_acceleration(object_bbox_centers(obj_verts, unit_to_cm=unit_to_cm))),
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

    with (run_dir / "metric_in_the_wild.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return result


def write_ablation_csv(results: list[dict], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ablation", "video_id", "num_frames", *METRIC_KEYS])
        for result in results:
            writer.writerow([
                result["ablation"],
                result["video_id"],
                result["num_frames"],
                *(result["metrics"].get(key) for key in METRIC_KEYS),
            ])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute metrics for ablation outputs from saved meshes.")
    parser.add_argument("--ablation_root", type=Path, default=Path(DEFAULT_ABLATION_ROOT), help="Root containing <ablation>/<video_id> output folders.")
    parser.add_argument("--mesh_root", type=Path, default=Path(DEFAULT_MESH_ROOT), help="Root containing <video_id>_<ablation>/hand and object mesh folders.")
    parser.add_argument("--data_root", type=Path, default=Path(DEFAULT_DATA_ROOT), help="Original input_data root for intrinsics.")
    parser.add_argument("--output_dir", type=Path, default=None, help="Aggregate metric output directory; defaults to <ablation_root>/metrics_summary.")
    parser.add_argument("--ablation", type=str, nargs="*", default=None, help="Optional ablation names to process.")
    parser.add_argument("--video_id", type=str, nargs="*", default=None, help="Optional video IDs to process.")
    parser.add_argument("--device", type=str, default="cuda", help="Device for PyTorch3D mask IoU rendering.")
    parser.add_argument("--iou_batch_size", type=int, default=32, help="Batch size for rendered mask IoU.")
    parser.add_argument("--chamfer_samples", type=int, default=3000, help="Surface samples per mesh for nearest H-O distance.")
    parser.add_argument("--no_render_iou", action="store_true", help="Skip rendered object mask IoU.")
    parser.add_argument("--no_flip_xy_for_iou", action="store_true", help="Disable xy flip before mask IoU rendering.")
    parser.add_argument("--unit_to_cm", type=float, default=100.0, help="Scale factor from mesh units to centimeters.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or (args.ablation_root / "metrics_summary")
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for ablation, video_id, run_dir in run_dirs(args.ablation_root, args.ablation, args.video_id):
        try:
            results.append(
                compute_run_metrics(
                    ablation,
                    video_id,
                    run_dir,
                    args.mesh_root,
                    args.data_root,
                    device=args.device,
                    unit_to_cm=args.unit_to_cm,
                    chamfer_samples=args.chamfer_samples,
                    iou_batch_size=args.iou_batch_size,
                    render_iou=not args.no_render_iou,
                    flip_xy_for_iou=not args.no_flip_xy_for_iou,
                )
            )
        except Exception as exc:
            print(f"[ERROR] failed {ablation}/{video_id}: {exc}")

    aggregate = {"summary": summarize_results(results), "per_run_results": results}
    with (output_dir / "all_ablation_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(aggregate, f, indent=2)
    write_ablation_csv(results, output_dir / "all_ablation_metrics.csv")
    write_csv(
        [
            {
                "sequence_name": f"{result['ablation']}/{result['video_id']}",
                "num_frames": result["num_frames"],
                "metrics": result["metrics"],
            }
            for result in results
        ],
        output_dir / "all_ablation_metrics_compat.csv",
    )

    print("\n===== Ablation Metrics Summary =====")
    for key in METRIC_KEYS:
        stats = aggregate["summary"].get(key)
        if stats is None:
            print(f"{key}: n/a")
        else:
            print(f"{key}: {stats['mean']:.6f}")


if __name__ == "__main__":
    main()
