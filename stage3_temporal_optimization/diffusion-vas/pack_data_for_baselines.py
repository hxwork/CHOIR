#!/usr/bin/env python3
"""Pack RGB and label-mask videos from input_data/<video_id>/.

Outputs are written to pack_data_for_baselines/<video_id>/. Mask videos use
label frames at the matching RGB frame size where 0=background, 50=modal object,
150=right hand, 250=left hand.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
from PIL import Image

DEFAULT_INPUT_ROOT = "input_data"
DEFAULT_OUTPUT_ROOT = "pack_data_for_baselines"

_MASK_IMAGE_EXT = frozenset({".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"})
_MASK_THRESH = 127

try:
    _RESAMPLE_MASK = Image.Resampling.NEAREST
except AttributeError:  # Pillow < 9.1
    _RESAMPLE_MASK = Image.NEAREST  # type: ignore[attr-defined]


def _rgba_to_binary_l(arr: np.ndarray) -> Image.Image:
    """arr: HxW or HxWxC uint8 -> mode L binary {0,255}."""
    if arr.ndim == 2:
        m = arr > _MASK_THRESH
    else:
        alpha = arr[..., 3] if arr.shape[-1] >= 4 else None
        rgb_max = arr[..., :3].max(axis=-1) if arr.shape[-1] >= 3 else arr.max(axis=-1)
        if alpha is not None:
            m = (alpha > _MASK_THRESH) | (rgb_max > _MASK_THRESH)
        else:
            m = rgb_max > _MASK_THRESH
    out = (m.astype(np.uint8)) * 255
    return Image.fromarray(out, mode="L")


def _image_to_binary_l(im: Image.Image) -> Image.Image:
    """Convert mask image to binary L without inventing alpha for L images."""
    if im.mode in {"1", "L", "I", "I;16", "F"}:
        arr = np.asarray(im.convert("L"))
    elif im.mode in {"RGBA", "LA"} or "transparency" in im.info:
        arr = np.asarray(im.convert("RGBA"))
    else:
        arr = np.asarray(im.convert("RGB"))
    return _rgba_to_binary_l(arr)


def _image_files_by_output_rel(src: Path) -> dict[Path, Path]:
    files: dict[Path, Path] = {}
    if not src.is_dir():
        return files
    for path in sorted(src.rglob("*")):
        if path.is_dir() or path.suffix.lower() not in _MASK_IMAGE_EXT:
            continue
        files[path.relative_to(src).with_suffix(".png")] = path
    return files


def _image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as im:
        return im.size


def _numeric_frame_key(path: Path) -> tuple[int, str]:
    try:
        return int(path.stem), path.name
    except ValueError as exc:
        raise ValueError(f"frame name must have a numeric stem: {path}") from exc


def _list_frame_paths(frame_dir: Path) -> list[Path]:
    if not frame_dir.is_dir():
        return []
    frames = [
        path
        for path in frame_dir.iterdir()
        if path.is_file() and path.suffix.lower() in _MASK_IMAGE_EXT
    ]
    return sorted(frames, key=_numeric_frame_key)


def _read_video_frame(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        im.load()
        arr = np.asarray(im)
    if arr.ndim == 2:
        return np.repeat(arr[..., None], 3, axis=-1)
    if arr.shape[-1] == 4:
        return arr[..., :3]
    return arr


def _write_video_from_frames(frame_paths: list[Path], output_path: Path, fps: int) -> None:
    if not frame_paths:
        raise ValueError(f"no frames to write for {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with imageio.get_writer(
            str(output_path),
            format="ffmpeg",
            fps=fps,
            codec="libx264",
            pixelformat="yuv420p",
            macro_block_size=None,
            ffmpeg_params=["-crf", "18", "-preset", "veryfast"],
        ) as writer:
            for frame_path in frame_paths:
                writer.append_data(_read_video_frame(frame_path))
    except ImportError:
        _write_video_from_frames_cv2(frame_paths, output_path, fps)


def _write_video_from_frames_cv2(frame_paths: list[Path], output_path: Path, fps: int) -> None:
    first_frame = _read_video_frame(frame_paths[0])
    height, width = first_frame.shape[:2]
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
        True,
    )
    if not writer.isOpened():
        raise RuntimeError(f"could not open video writer for {output_path}")
    try:
        for frame_path in frame_paths:
            frame = _read_video_frame(frame_path)
            if frame.shape[:2] != (height, width):
                raise ValueError(
                    f"frame size mismatch for {frame_path}: "
                    f"expected {(width, height)}, got {(frame.shape[1], frame.shape[0])}"
                )
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def _load_binary_mask(path: Path, target_size: tuple[int, int]) -> np.ndarray:
    with Image.open(path) as im:
        im.load()
        bin_im = _image_to_binary_l(im).resize(target_size, _RESAMPLE_MASK)
    return np.asarray(bin_im) > 0


def _write_combined_label_masks(
    rgb_src: Path | None,
    obj_src: Path | None,
    rh_src: Path | None,
    lh_src: Path | None,
    dst: Path,
) -> int:
    """Write label masks: 0 background, 50 object, 150 right hand, 250 left hand."""
    rgb_files = _image_files_by_output_rel(rgb_src) if rgb_src else {}
    sources = [
        (150, _image_files_by_output_rel(rh_src) if rh_src else {}),
        (250, _image_files_by_output_rel(lh_src) if lh_src else {}),
        (50, _image_files_by_output_rel(obj_src) if obj_src else {}),
    ]
    rels = sorted({rel for _, files in sources for rel in files})

    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)

    overlap_pixels = 0
    for rel in rels:
        target_path = rgb_files.get(rel)
        if target_path is None:
            target_path = next(files[rel] for _, files in sources if rel in files)
        target_size = _image_size(target_path)
        width, height = target_size
        out = np.zeros((height, width), dtype=np.uint8)
        occupied = np.zeros((height, width), dtype=bool)
        for label, files in sources:
            path = files.get(rel)
            if path is None:
                continue
            mask = _load_binary_mask(path, target_size)
            overlap_pixels += int((occupied & mask).sum())
            out[mask] = label
            occupied |= mask
        out_path = dst / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(out, mode="L").save(out_path)
    return overlap_pixels


def resolve_video_ids(input_root: Path, video_ids: list[str], all_img: bool) -> list[str]:
    resolved: list[str] = []
    seen: set[str] = set()

    def add(video_id: str) -> None:
        if video_id not in seen:
            seen.add(video_id)
            resolved.append(video_id)

    if all_img:
        for path in sorted(input_root.glob("IMG*"), key=lambda p: p.name):
            if path.is_dir():
                add(path.name)
    for video_id in video_ids:
        add(video_id)
    return resolved


def pack_one(
    video_id: str,
    input_root: Path,
    output_root: Path,
    dry_run: bool,
    fps: int = 30,
) -> list[str]:
    src_dir = input_root / video_id
    dst_dir = output_root / video_id
    notes: list[str] = []

    if not src_dir.is_dir():
        notes.append(f"skip {video_id}: missing {src_dir}")
        return notes

    s_obj = src_dir / "obj_masks"
    if not s_obj.is_dir():
        notes.append(f"warn {video_id}: no obj_masks/ under {src_dir}")

    s_rgb = src_dir / "rgbs"
    if not s_rgb.is_dir():
        notes.append(f"warn {video_id}: no rgbs/ under {src_dir}")
    try:
        rgb_frames = _list_frame_paths(s_rgb)
    except ValueError as exc:
        notes.append(f"skip {video_id}: {exc}")
        return notes
    if s_rgb.is_dir() and not rgb_frames:
        notes.append(f"warn {video_id}: no RGB frames under {s_rgb}")

    has_rh = (src_dir / "rh_masks").is_dir()
    has_lh = (src_dir / "lh_masks").is_dir()
    has_hand = (src_dir / "hand_masks").is_dir()

    if not (has_rh or has_lh or has_hand):
        notes.append(
            f"warn {video_id}: no rh_masks/, lh_masks/, or hand_masks/ under {src_dir}"
        )

    label_obj = s_obj if s_obj.is_dir() else None
    label_rh = src_dir / "rh_masks" if has_rh else None
    label_lh = src_dir / "lh_masks" if has_lh else None
    if label_rh is None and label_lh is None and has_hand:
        notes.append(f"warn {video_id}: hand_masks/ has no side; writing it as right hand in masks/")
        label_rh = src_dir / "hand_masks"
    should_write_masks = label_obj is not None or label_rh is not None or label_lh is not None

    if dry_run:
        if rgb_frames:
            notes.append(
                f"dry-run {video_id}: would write {dst_dir / f'{video_id}.mp4'} "
                f"from {len(rgb_frames)} RGB frames"
            )
        if should_write_masks:
            notes.append(
                f"dry-run {video_id}: would build temporary label mask frames "
                "(0/50/150/250 at matching RGB frame size)"
            )
            notes.append(f"dry-run {video_id}: would write {dst_dir / f'{video_id}_mask.mp4'}")
        if not rgb_frames and not should_write_masks:
            notes.append(f"warn {video_id}: no RGB frames or mask sources; nothing to write")
        return notes

    if not rgb_frames and not should_write_masks:
        notes.append(f"warn {video_id}: no RGB frames or mask sources; nothing to write")
        return notes

    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    if rgb_frames:
        rgb_video_path = dst_dir / f"{video_id}.mp4"
        _write_video_from_frames(rgb_frames, rgb_video_path, fps)
        notes.append(f"ok {video_id}: wrote {rgb_video_path} ({len(rgb_frames)} frames)")
    if should_write_masks:
        mask_frames_dir = dst_dir / "_mask_frames"
        try:
            overlap_pixels = _write_combined_label_masks(
                s_rgb if s_rgb.is_dir() else None, label_obj, label_rh, label_lh, mask_frames_dir
            )
            if overlap_pixels:
                notes.append(f"warn {video_id}: masks had {overlap_pixels} overlapping pixels")
            try:
                mask_frames = _list_frame_paths(mask_frames_dir)
            except ValueError as exc:
                notes.append(f"skip {video_id}: {exc}")
                return notes
            if mask_frames:
                mask_video_path = dst_dir / f"{video_id}_mask.mp4"
                _write_video_from_frames(mask_frames, mask_video_path, fps)
                notes.append(f"ok {video_id}: wrote {mask_video_path} ({len(mask_frames)} frames)")
            else:
                notes.append(f"warn {video_id}: no generated mask frames under {mask_frames_dir}")
        finally:
            if mask_frames_dir.exists():
                shutil.rmtree(mask_frames_dir)

    notes.append(f"ok {video_id}: wrote {dst_dir}")
    return notes


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--video_id",
        nargs="+",
        default=[],
        metavar="ID",
        help="folder name(s) under input_data, e.g. --video_id 4092 or --video_id 4092 4902",
    )
    p.add_argument(
        "--all_img",
        action="store_true",
        help="process all IMG* folders under input_root",
    )
    p.add_argument(
        "--input_root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help=f"default: {DEFAULT_INPUT_ROOT}",
    )
    p.add_argument(
        "--output_root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"default: {DEFAULT_OUTPUT_ROOT}",
    )
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="print planned copies only",
    )
    p.add_argument(
        "--fps",
        type=int,
        default=30,
        help="output video frame rate. default: 30",
    )
    args = p.parse_args()

    video_ids = resolve_video_ids(args.input_root, args.video_id, args.all_img)
    if not video_ids:
        p.error("provide --video_id ID [ID ...] and/or --all_img")

    all_notes: list[str] = []
    for vid in video_ids:
        all_notes.extend(pack_one(vid, args.input_root, args.output_root, args.dry_run, args.fps))

    for line in all_notes:
        print(line)

    if any(line.startswith("skip ") for line in all_notes):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
