"""Shared paths for Stage 1 video inputs and manual annotations."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = REPO_ROOT / "data"
DEFAULT_OUTPUT_DATA_DIR = REPO_ROOT / "output"


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


def annotation_dir_for_video(output_root, video_path):
    """Return output/<video_id>/annotations for an input video."""
    return output_dir_for_video(output_root, video_path) / "annotations"


def visualization_dir_for_video(output_root, video_path):
    """Return output/<video_id>/visualizations for inference previews."""
    return output_dir_for_video(output_root, video_path) / "visualizations"
