#!/usr/bin/env python3
"""Compose each hold*/rgbs image sequence into a video_id-named mp4."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence

import imageio.v2 as imageio


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compose input_data/hold*/rgbs frames into <video_id>.mp4 files."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("input_data"),
        help="Root directory containing hold* sequence folders. Default: input_data",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Output video frame rate. Default: 30",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing <video_id>.mp4 files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned outputs without writing videos.",
    )
    return parser.parse_args()


def numeric_frame_key(path: Path) -> tuple[int, str]:
    try:
        return int(path.stem), path.name
    except ValueError as exc:
        raise ValueError(f"Frame name must have a numeric stem: {path}") from exc


def list_frame_paths(rgb_dir: Path) -> list[Path]:
    frames = [
        path
        for path in rgb_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    return sorted(frames, key=numeric_frame_key)


def iter_hold_dirs(input_root: Path) -> Iterable[Path]:
    return sorted(
        (path for path in input_root.glob("hold*") if path.is_dir()),
        key=lambda path: path.name,
    )


def compose_video(frame_paths: Sequence[Path], output_path: Path, fps: int) -> None:
    with imageio.get_writer(
        output_path,
        fps=fps,
        codec="libx264",
        pixelformat="yuv420p",
        macro_block_size=None,
        ffmpeg_params=["-crf", "18", "-preset", "veryfast"],
    ) as writer:
        for frame_path in frame_paths:
            writer.append_data(imageio.imread(frame_path))


def main() -> int:
    args = parse_args()
    input_root = args.input_root.expanduser().resolve()

    if not input_root.is_dir():
        raise FileNotFoundError(f"Input root does not exist or is not a directory: {input_root}")

    hold_dirs = list(iter_hold_dirs(input_root))
    if not hold_dirs:
        print(f"No hold* directories found under {input_root}")
        return 0

    processed = 0
    skipped = 0

    for hold_dir in hold_dirs:
        video_id = hold_dir.name
        rgb_dir = hold_dir / "rgbs"
        output_path = hold_dir / f"{video_id}.mp4"

        if not rgb_dir.is_dir():
            print(f"[skip] {video_id}: missing rgbs directory")
            skipped += 1
            continue

        frame_paths = list_frame_paths(rgb_dir)
        if not frame_paths:
            print(f"[skip] {video_id}: no image frames in {rgb_dir}")
            skipped += 1
            continue

        if output_path.exists() and not args.overwrite:
            print(f"[skip] {video_id}: output exists ({output_path})")
            skipped += 1
            continue

        print(f"[write] {video_id}: {len(frame_paths)} frames -> {output_path}")
        if not args.dry_run:
            compose_video(frame_paths, output_path, args.fps)
        processed += 1

    action = "planned" if args.dry_run else "written"
    print(f"Done: {processed} videos {action}, {skipped} skipped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
