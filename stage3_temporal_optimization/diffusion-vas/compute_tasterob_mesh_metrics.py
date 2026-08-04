from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


METRIC_KEYS = ["mIoU", "Pen. Ratio", "H-O Dist.", "Acc_h", "Acc_o"]
DEFAULT_RENDER_ROOT = "rendering_for_paper/rendered_output/ours_TasteRob"
DEFAULT_MESH_ROOT = "rendering_for_paper/ours_SelfCaptured"
DEFAULT_DATA_ROOT = "input_data"
VIDEO_ID_RE = re.compile(r"^(?:\d+|IMG_\d+)$")


@dataclass(frozen=True)
class MetricRun:
    video_id: str
    render_seq_dir: Path
    mesh_seq_dir: Path
    data_seq_dir: Path


def is_metric_video_id(name: str) -> bool:
    return VIDEO_ID_RE.fullmatch(name) is not None


def video_sort_key(video_id: str) -> tuple[int, int | str]:
    if video_id.isdigit():
        return (0, int(video_id))
    return (1, video_id)


def mesh_dir_has_sequence(mesh_seq_dir: Path) -> bool:
    hand_dir = mesh_seq_dir / "hand"
    obj_dir = mesh_seq_dir / "object"
    if not hand_dir.is_dir() or not obj_dir.is_dir():
        return False
    hand_files = [*hand_dir.glob("*.ply"), *hand_dir.glob("*.obj")]
    obj_files = [*obj_dir.glob("*.ply"), *obj_dir.glob("*.obj")]
    return bool(hand_files and obj_files)


def discover_runs(
    render_root: Path,
    mesh_root: Path,
    video_ids: list[str] | None,
    data_root: Path = Path(DEFAULT_DATA_ROOT),
) -> tuple[list[MetricRun], list[str]]:
    warnings: list[str] = []
    if video_ids:
        candidate_ids = list(dict.fromkeys(video_ids))
    else:
        candidate_ids = sorted(
            [p.name for p in render_root.iterdir() if p.is_dir() and is_metric_video_id(p.name)],
            key=video_sort_key,
        )

    runs: list[MetricRun] = []
    for video_id in candidate_ids:
        if not is_metric_video_id(video_id):
            warnings.append(f"skipping unsupported video_id pattern: {video_id}")
            continue
        render_seq_dir = render_root / video_id
        if not render_seq_dir.is_dir():
            warnings.append(f"render directory missing: {render_seq_dir}")
            continue
        mesh_seq_dir = mesh_root / video_id
        if not mesh_dir_has_sequence(mesh_seq_dir):
            warnings.append(f"mesh sequence missing or empty: {mesh_seq_dir}")
            continue
        runs.append(
            MetricRun(
                video_id=video_id,
                render_seq_dir=render_seq_dir,
                mesh_seq_dir=mesh_seq_dir,
                data_seq_dir=data_root / video_id,
            )
        )
    return runs, warnings


def parse_stage_range(path: Path, num_frames: int) -> tuple[list[int] | None, str | None]:
    with path.open("r", encoding="utf-8") as f:
        ranges = json.load(f)
    interaction = next((v for k, v in ranges.items() if "interaction" in k), None)
    if interaction is None:
        return None, f"interaction range missing in {path}"
    start = int(interaction.get("original_start", interaction.get("sampled_start", 0)))
    end = int(interaction.get("original_end", interaction.get("sampled_end", num_frames)))
    start = max(0, min(start, num_frames))
    end = max(start, min(end, num_frames))
    if start >= end:
        return None, f"empty interaction range in {path}"
    return list(range(start, end)), None


def load_interaction_frames(render_seq_dir: Path, data_seq_dir: Path, num_frames: int) -> tuple[list[int], list[str]]:
    warnings: list[str] = []
    for base_dir in (render_seq_dir, data_seq_dir):
        ranges_path = base_dir / "stage_frame_ranges.json"
        if not ranges_path.exists():
            continue
        try:
            frames, warning = parse_stage_range(ranges_path, num_frames)
        except Exception as exc:
            warnings.append(f"failed to read interaction range from {ranges_path}: {exc}")
            continue
        if warning:
            warnings.append(warning)
            continue
        if frames is not None:
            return frames, warnings
    warnings.append("stage_frame_ranges.json missing or unusable; using all frames as interaction")
    return list(range(num_frames)), warnings


def load_crop_camera(
    render_seq_dir: Path,
    data_seq_dir: Path,
    num_frames: int,
    mask_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    from compute_hold_mesh_metrics import load_intrinsics

    warnings: list[str] = []
    focal, principal = load_intrinsics(data_seq_dir)
    mask_h, mask_w = mask_shape

    bbox_path = next(
        (path for path in (render_seq_dir / "global_bbox.json", data_seq_dir / "global_bbox.json") if path.exists()),
        None,
    )
    if bbox_path is None:
        warnings.append("global_bbox.json missing; using original intrinsics without crop adjustment")
        return (
            np.repeat(focal[None], num_frames, axis=0),
            np.repeat(principal[None], num_frames, axis=0),
            warnings,
        )

    with bbox_path.open("r", encoding="utf-8") as f:
        bboxes = json.load(f)
    if not bboxes:
        warnings.append(f"{bbox_path} is empty; using original intrinsics without crop adjustment")
        return (
            np.repeat(focal[None], num_frames, axis=0),
            np.repeat(principal[None], num_frames, axis=0),
            warnings,
        )

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
    return (
        np.repeat(crop_focal[None], num_frames, axis=0),
        np.repeat(crop_principal[None], num_frames, axis=0),
        warnings,
    )


def load_masks_with_fallback(render_seq_dir: Path, data_seq_dir: Path, num_frames: int) -> tuple[np.ndarray | None, str | None, list[str]]:
    from compute_hold_mesh_metrics import load_masks

    warnings: list[str] = []
    for source_name, seq_dir in (("render", render_seq_dir), ("data", data_seq_dir)):
        masks, mask_source, mask_warnings = load_masks(seq_dir, num_frames)
        if masks is not None:
            warnings.extend(f"{source_name}: {warning}" for warning in mask_warnings)
            return masks, f"{source_name}/{mask_source}", warnings
        warnings.extend(f"{source_name}: {warning}" for warning in mask_warnings)
    return None, None, warnings


def compute_run_metrics(
    run: MetricRun,
    *,
    device: str,
    unit_to_cm: float,
    chamfer_samples: int,
    iou_batch_size: int,
    render_iou: bool,
    flip_xy_for_iou: bool,
) -> dict:
    from compute_hold_mesh_metrics import (
        compute_center_acceleration,
        compute_mask_iou_metric,
        compute_nearest_surface_distance_metric,
        compute_penetration_ratio,
        load_mesh_sequence,
        object_bbox_centers,
    )
    from in_the_wild_metric import compute_vertex_acceleration

    warnings: list[str] = []
    hand_verts, hand_faces, obj_verts, obj_faces, num_frames = load_mesh_sequence(run.mesh_seq_dir)
    inter_frames, inter_warnings = load_interaction_frames(run.render_seq_dir, run.data_seq_dir, num_frames)
    warnings.extend(inter_warnings)

    masks = None
    mask_source = None
    focal_lengths = principal_points = None
    if render_iou:
        try:
            masks, mask_source, mask_warnings = load_masks_with_fallback(run.render_seq_dir, run.data_seq_dir, num_frames)
            warnings.extend(mask_warnings)
            if masks is not None:
                focal_lengths, principal_points, camera_warnings = load_crop_camera(
                    run.render_seq_dir,
                    run.data_seq_dir,
                    num_frames,
                    masks.shape[1:3],
                )
                warnings.extend(camera_warnings)
        except Exception as exc:
            warnings.append(f"mIoU setup failed: {exc}")

    print(f"[{run.video_id}] frames={num_frames}, interaction={len(inter_frames)}")
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
        miou = compute_mask_iou_metric(
            obj_verts,
            obj_faces,
            masks,
            focal_lengths[0],
            principal_points[0],
            device,
            iou_batch_size,
            flip_xy_for_iou,
        )

    result = {
        "video_id": run.video_id,
        "num_frames": int(num_frames),
        "interaction_start": int(inter_frames[0]) if inter_frames else None,
        "interaction_end": int(inter_frames[-1] + 1) if inter_frames else None,
        "render_dir": str(run.render_seq_dir),
        "mesh_dir": str(run.mesh_seq_dir),
        "data_dir": str(run.data_seq_dir),
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

    with (run.render_seq_dir / "metric_in_the_wild.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return result


def write_csv(results: list[dict], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["video_id", "num_frames", *METRIC_KEYS])
        for result in results:
            writer.writerow([
                result["video_id"],
                result["num_frames"],
                *(result["metrics"].get(key) for key in METRIC_KEYS),
            ])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute TasteRob metrics from saved hand/object meshes.")
    parser.add_argument("--render_root", type=Path, default=Path(DEFAULT_RENDER_ROOT), help="Root containing rendered_output/ours_TasteRob video folders.")
    parser.add_argument("--mesh_root", type=Path, default=Path(DEFAULT_MESH_ROOT), help="Root containing <video_id>/hand and object mesh folders.")
    parser.add_argument("--data_root", type=Path, default=Path(DEFAULT_DATA_ROOT), help="Original input_data root for intrinsics and fallback masks.")
    parser.add_argument("--output_dir", type=Path, default=None, help="Aggregate metric output directory; defaults to <render_root>/metrics_summary.")
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
    output_dir = args.output_dir or (args.render_root / "metrics_summary")
    output_dir.mkdir(parents=True, exist_ok=True)

    from compute_hold_mesh_metrics import summarize_results

    runs, discovery_warnings = discover_runs(args.render_root, args.mesh_root, args.video_id, args.data_root)
    for warning in discovery_warnings:
        print(f"[WARN] {warning}")

    results = []
    for run in runs:
        try:
            results.append(
                compute_run_metrics(
                    run,
                    device=args.device,
                    unit_to_cm=args.unit_to_cm,
                    chamfer_samples=args.chamfer_samples,
                    iou_batch_size=args.iou_batch_size,
                    render_iou=not args.no_render_iou,
                    flip_xy_for_iou=not args.no_flip_xy_for_iou,
                )
            )
        except Exception as exc:
            print(f"[ERROR] failed {run.video_id}: {exc}")

    aggregate = {
        "summary": summarize_results(results),
        "per_run_results": results,
        "discovery_warnings": discovery_warnings,
    }
    with (output_dir / "all_tasterob_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(aggregate, f, indent=2)
    write_csv(results, output_dir / "all_tasterob_metrics.csv")
    print(f"Wrote {len(results)} result(s) to {output_dir}")


if __name__ == "__main__":
    main()
