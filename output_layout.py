"""Canonical per-video layout under CHOIR ``output/<VIDEO_ID>/``.

Only the new layout is supported. Stage writers and readers should resolve paths
through ``VideoLayout`` instead of hardcoding flat root filenames.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Union

PathLike = Union[str, Path]


@dataclass(frozen=True)
class VideoLayout:
    """Paths for one ``output/<video_id>/`` tree."""

    root: Path
    video_id: str

    @classmethod
    def from_root(cls, video_root: PathLike) -> "VideoLayout":
        root = Path(video_root).resolve()
        return cls(root=root, video_id=root.name)

    @classmethod
    def from_output_root(cls, output_root: PathLike, video_id: str) -> "VideoLayout":
        return cls.from_root(Path(output_root) / video_id)

    # --- inputs (Stage 1 Yolov8 + shared camera later filled by Stage 3) ---
    @property
    def inputs_dir(self) -> Path:
        return self.root / "inputs"

    @property
    def video_mp4(self) -> Path:
        return self.inputs_dir / "video.mp4"

    @property
    def annotations_dir(self) -> Path:
        return self.inputs_dir / "annotations"

    @property
    def frames_rgb_dir(self) -> Path:
        return self.inputs_dir / "frames" / "rgb"

    @property
    def object_masks_dir(self) -> Path:
        return self.inputs_dir / "masks" / "object"

    def hand_masks_dir(self, side: str) -> Path:
        side = side.lower()
        if side in ("left", "lh"):
            return self.inputs_dir / "masks" / "hand_left"
        if side in ("right", "rh"):
            return self.inputs_dir / "masks" / "hand_right"
        raise ValueError(f"unknown hand side: {side}")

    @property
    def lh_masks_dir(self) -> Path:
        return self.hand_masks_dir("left")

    @property
    def rh_masks_dir(self) -> Path:
        return self.hand_masks_dir("right")

    @property
    def bbox_dir(self) -> Path:
        return self.inputs_dir / "bbox"

    @property
    def keypoints_dir(self) -> Path:
        return self.inputs_dir / "keypoints"

    def bbox_json(self, side: str) -> Path:
        prefix = "lh" if side in ("left", "lh") else "rh"
        return self.bbox_dir / f"{prefix}_bbox.json"

    def keypoints_json(self, side: str) -> Path:
        prefix = "lh" if side in ("left", "lh") else "rh"
        return self.keypoints_dir / f"{prefix}_keypoints.json"

    @property
    def camera_dir(self) -> Path:
        return self.inputs_dir / "camera"

    @property
    def intrinsics_json(self) -> Path:
        # Sibling of inputs/video.mp4: Dyn-HaMR/VIPE looks up parent(video)/intrinsics.json.
        return self.inputs_dir / "intrinsics.json"

    @property
    def global_bbox_json(self) -> Path:
        return self.camera_dir / "global_bbox.json"

    # --- stage1 ---
    @property
    def stage1_dir(self) -> Path:
        return self.root / "stage1"

    @property
    def object_init_dir(self) -> Path:
        return self.stage1_dir / "object_init"

    @property
    def glb_path(self) -> Path:
        return self.object_init_dir / "glb_0.glb"

    @property
    def transform_json(self) -> Path:
        return self.object_init_dir / "transform_0.json"

    @property
    def rendered_on_image(self) -> Path:
        return self.object_init_dir / "rendered_on_image.png"

    @property
    def env_depths_dir(self) -> Path:
        return self.stage1_dir / "depth" / "env_depths"

    @property
    def mano_params_dir(self) -> Path:
        return self.stage1_dir / "hand" / "mano_params"

    @property
    def hand_meshes_dir(self) -> Path:
        return self.stage1_dir / "hand" / "hand_meshes"

    @property
    def visualizations_dir(self) -> Path:
        return self.stage1_dir / "visualizations"

    @property
    def dynhamr_dir(self) -> Path:
        """Dyn-HaMR working/resume directory; internal layout is unchanged."""
        return self.stage1_dir / "dynhamr"

    # --- stage2 ---
    @property
    def stage2_dir(self) -> Path:
        return self.root / "stage2"

    @property
    def grasp_correction_dir(self) -> Path:
        return self.stage2_dir / "grasp_correction"

    @property
    def camera_ray_depth_offset_json(self) -> Path:
        return self.grasp_correction_dir / "camera_ray_depth_offset.json"

    @property
    def gfm_input_hoi_seq_dir(self) -> Path:
        return self.grasp_correction_dir / "gfm_input_hoi_seq"

    # --- stage3 ---
    @property
    def stage3_dir(self) -> Path:
        return self.root / "stage3"

    @property
    def stage3_intermediates_dir(self) -> Path:
        return self.stage3_dir / "intermediates"

    @property
    def amodal_masks_dir(self) -> Path:
        return self.stage3_intermediates_dir / "amodal_masks"

    @property
    def cropped_depths_dir(self) -> Path:
        return self.stage3_intermediates_dir / "cropped_depths"

    @property
    def cropped_metric_depths_dir(self) -> Path:
        return self.stage3_intermediates_dir / "cropped_metric_depths"

    @property
    def optimized_hoi_seq_dir(self) -> Path:
        return self.stage3_intermediates_dir / "optimized_hoi_seq"

    @property
    def optimized_hoi_contact_seq_dir(self) -> Path:
        # Final dense MANO + object pose/geometry parameters (not an intermediate).
        return self.stage3_final_dir / "optimized_hoi_contact_seq"

    @property
    def optimized_object_meshes_dir(self) -> Path:
        return self.stage3_intermediates_dir / "optimized_object_meshes"

    @property
    def stage3_final_dir(self) -> Path:
        return self.stage3_dir / "final"

    @property
    def optimized_meshes_dir(self) -> Path:
        return self.stage3_final_dir / "optimized_meshes"

    @property
    def optimized_transform_sequence_json(self) -> Path:
        return self.stage3_final_dir / "optimized_transform_sequence.json"

    @property
    def stage3_logs_dir(self) -> Path:
        return self.stage3_dir / "logs"

    @property
    def stage3_diagnostics_dir(self) -> Path:
        return self.stage3_dir / "diagnostics"

    @property
    def stage3_checkpoint(self) -> Path:
        return self.stage3_diagnostics_dir / "stage3_checkpoint.pt"

    @property
    def stage3_previews_dir(self) -> Path:
        return self.stage3_dir / "previews"

    @property
    def stage3_preview_videos_dir(self) -> Path:
        return self.stage3_previews_dir / "videos"

    @property
    def stage3_preview_images_dir(self) -> Path:
        return self.stage3_previews_dir / "images"

    @property
    def stage3_preview_meshes_dir(self) -> Path:
        return self.stage3_previews_dir / "meshes"

    @property
    def visualization_3d_html(self) -> Path:
        return self.stage3_previews_dir / "visualization_3d.html"

    def ensure_stage_dirs(self) -> None:
        for path in (
            self.inputs_dir,
            self.stage1_dir,
            self.stage2_dir,
            self.stage3_intermediates_dir,
            self.stage3_final_dir,
            self.stage3_logs_dir,
            self.stage3_diagnostics_dir,
            self.stage3_preview_videos_dir,
            self.stage3_preview_images_dir,
            self.stage3_preview_meshes_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


def discover_video_roots(output_root: PathLike, video_ids: Optional[Iterable[str]] = None) -> list[Path]:
    """Discover ``output/<video_id>`` directories that look like CHOIR video trees."""
    output_root = Path(output_root)
    if not output_root.is_dir():
        return []
    requested = set(video_ids) if video_ids is not None else None
    roots: list[Path] = []
    for child in sorted(output_root.iterdir()):
        if not child.is_dir():
            continue
        if requested is not None and child.name not in requested:
            continue
        layout = VideoLayout.from_root(child)
        if layout.video_mp4.exists() or layout.frames_rgb_dir.is_dir():
            roots.append(child)
    return roots
