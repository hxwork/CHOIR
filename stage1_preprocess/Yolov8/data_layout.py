"""Shared paths for Stage 1 video inputs and manual annotations.

Delegates per-video path names to the repo-level ``output_layout.VideoLayout``.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = REPO_ROOT / "data"
DEFAULT_OUTPUT_DATA_DIR = REPO_ROOT / "output"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from output_layout import VideoLayout  # noqa: E402


def discover_input_videos(data_dir, video_ids=None):
    """Return flat input videos matching data/<video_id>.mp4."""
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        return []

    requested = set(video_ids) if video_ids else None
    videos = sorted(data_dir.glob("*.mp4"))
    if requested is not None:
        videos = [video for video in videos if video.stem in requested]
    return videos


def output_dir_for_video(output_root, video_path):
    """Return output/<video_id> for an input video."""
    return Path(output_root) / Path(video_path).stem


def layout_for_video(output_root, video_path) -> VideoLayout:
    return VideoLayout.from_output_root(output_root, Path(video_path).stem)


def annotation_dir_for_video(output_root, video_path):
    """Return output/<video_id>/inputs/annotations for an input video."""
    return layout_for_video(output_root, video_path).annotations_dir


def visualization_dir_for_video(output_root, video_path):
    """Return output/<video_id>/stage1/visualizations for inference previews."""
    return layout_for_video(output_root, video_path).visualizations_dir
