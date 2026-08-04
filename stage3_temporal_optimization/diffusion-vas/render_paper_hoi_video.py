"""High-quality PyTorch3D rendering for paper videos.

This script is a post-processing renderer for sequences produced by
``demo_fitting_5stages_new.py``.  It reads the optimized HOI sequence from
``optimized_hoi_contact_seq`` (or ``optimized_hoi_seq``) and writes clean,
high-resolution camera/side/top mesh renders plus a camera-view RGB overlay.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import imageio
import numpy as np
from tqdm import tqdm


DEFAULT_DATA_PATH = Path("input_data")
DEFAULT_MANO_ROOT = Path("../../stage1_preprocess/Dyn_HaMR_new/_DATA/data/mano")


@dataclass(frozen=True)
class CropCamera:
    image_size: tuple[int, int]
    focal_length: tuple[float, float]
    principal_point: tuple[float, float]


@dataclass(frozen=True)
class RenderConfig:
    data_path: Path
    video_ids: list[str] | None
    output_name: str
    height: int
    width: int
    fps: float
    crf: int
    preset: str
    batch_size: int
    device: str
    quality: str
    image_scale: int
    hand_color: tuple[float, float, float]
    object_color: tuple[float, float, float]
    background_color: tuple[float, float, float]
    overlay_alpha: float
    max_frames: int | None
    skip_existing: bool
    mano_root: Path


def parse_rgb_triplet(values: Sequence[str | float]) -> tuple[float, float, float]:
    if len(values) != 3:
        raise ValueError("RGB color must contain exactly 3 values.")
    color = tuple(float(v) for v in values)
    if any(v < 0.0 or v > 1.0 for v in color):
        raise ValueError("RGB color values must be in range [0, 1].")
    return color


def list_video_dirs(data_path: Path, video_ids: Sequence[str] | None) -> list[Path]:
    data_path = Path(data_path)
    if video_ids:
        missing = [vid for vid in video_ids if not (data_path / vid).is_dir()]
        if missing:
            raise FileNotFoundError(f"Missing requested video directories: {', '.join(missing)}")
        return [data_path / vid for vid in video_ids]

    return sorted([p for p in data_path.iterdir() if p.is_dir()])


def find_hoi_sequence_dir(seq_dir: Path) -> Path:
    for name in ("optimized_hoi_contact_seq", "optimized_hoi_seq"):
        candidate = seq_dir / name
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"No optimized HOI sequence found under {seq_dir}")


def compute_crop_camera(intrinsics: np.ndarray, bbox: Sequence[float], height: int, width: int) -> CropCamera:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    crop_w = x2 - x1
    crop_h = y2 - y1
    if crop_w <= 0 or crop_h <= 0:
        raise ValueError(f"Invalid crop bbox: {bbox}")

    scale = min(width / crop_w, height / crop_h)
    new_w = int(crop_w * scale)
    new_h = int(crop_h * scale)
    pad_left = (width - new_w) // 2
    pad_top = (height - new_h) // 2

    fx = float(intrinsics[0, 0]) * scale
    fy = float(intrinsics[1, 1]) * scale
    cx = (float(intrinsics[0, 2]) - x1) * scale + pad_left
    cy = (float(intrinsics[1, 2]) - y1) * scale + pad_top
    return CropCamera(image_size=(height, width), focal_length=(fx, fy), principal_point=(cx, cy))


def crop_and_resize_frame(frame: np.ndarray, bbox: Sequence[float], height: int, width: int, is_mask: bool = False) -> np.ndarray:
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    h, w = frame.shape[:2]
    x1 = max(0, min(w, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h, y1))
    y2 = max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid clipped crop bbox: {(x1, y1, x2, y2)}")

    cropped = frame[y1:y2, x1:x2]
    crop_h, crop_w = cropped.shape[:2]
    scale = min(width / crop_w, height / crop_h)
    new_w = int(crop_w * scale)
    new_h = int(crop_h * scale)
    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_AREA
    resized = cv2.resize(cropped, (new_w, new_h), interpolation=interp)

    if is_mask:
        canvas = np.zeros((height, width), dtype=resized.dtype)
    else:
        canvas = np.zeros((height, width, 3), dtype=resized.dtype)
    pad_left = (width - new_w) // 2
    pad_top = (height - new_h) // 2
    canvas[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized
    return canvas


def sorted_frame_files(folder: Path) -> list[Path]:
    suffixes = {".png", ".jpg", ".jpeg"}
    files = [p for p in Path(folder).iterdir() if p.suffix.lower() in suffixes]
    return sorted(files, key=lambda p: int(p.stem))


def load_rgb_frame(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to read RGB frame: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def load_mask_frame(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Failed to read mask frame: {path}")
    if image.ndim == 3 and image.shape[2] == 4:
        mask = image[:, :, 3]
    elif image.ndim == 3:
        mask = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        mask = image
    return (mask > 128).astype(np.uint8)


def load_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def load_intrinsics(seq_dir: Path) -> np.ndarray:
    data = load_json(seq_dir / "intrinsics.json")
    return np.asarray(data["intrinsics"], dtype=np.float32)


def load_global_bbox(seq_dir: Path) -> list[float]:
    bbox_path = seq_dir / "global_bbox.json"
    bboxes = load_json(bbox_path)
    if not bboxes:
        raise ValueError(f"Empty global_bbox.json: {bbox_path}")
    return bboxes[0]


def load_optimized_indices(hoi_dir: Path, max_frames: int | None) -> list[int]:
    obj_files = sorted(hoi_dir.glob("obj_*.json"), key=lambda p: int(p.stem.split("_")[-1]))
    mano_files = sorted(hoi_dir.glob("mano_*.json"), key=lambda p: int(p.stem.split("_")[-1]))
    obj_indices = [int(p.stem.split("_")[-1]) for p in obj_files]
    mano_indices = [int(p.stem.split("_")[-1]) for p in mano_files]
    indices = [idx for idx in obj_indices if idx in set(mano_indices)]
    if not indices:
        raise FileNotFoundError(f"No matching obj_*.json/mano_*.json files in {hoi_dir}")
    if max_frames is not None:
        indices = indices[:max_frames]
    return indices


def infer_frame_offset(seq_dir: Path) -> int:
    """Return the raw-frame index corresponding to optimized frame 0."""
    mano_dir = seq_dir / "mano_params"
    if not mano_dir.is_dir():
        return 0

    json_files = list(mano_dir.glob("*.json"))
    if not json_files:
        for child in mano_dir.iterdir():
            if child.is_dir():
                json_files.extend(child.glob("*.json"))
    if not json_files:
        return 0
    return min(int(path.stem) for path in json_files if path.stem.isdigit())


def load_sequence_frames(seq_dir: Path, indices: Sequence[int], bbox: Sequence[float], height: int, width: int, frame_offset: int = 0):
    rgb_files = sorted_frame_files(seq_dir / "rgbs")
    amodal_dir = seq_dir / "amodal_masks"
    hand_mask_dirs = [seq_dir / "lh_masks", seq_dir / "rh_masks"]
    amodal_files = sorted_frame_files(amodal_dir) if amodal_dir.is_dir() else []
    hand_files_by_dir = [sorted_frame_files(d) if d.is_dir() else [] for d in hand_mask_dirs]

    rgbs: list[np.ndarray] = []
    amodal_masks: list[np.ndarray] = []
    hand_masks: list[np.ndarray] = []

    for idx in indices:
        raw_idx = idx + frame_offset
        if raw_idx >= len(rgb_files):
            raise IndexError(f"RGB frame index {raw_idx} exceeds {len(rgb_files)} frames in {seq_dir / 'rgbs'}")
        rgb = crop_and_resize_frame(load_rgb_frame(rgb_files[raw_idx]), bbox, height, width, is_mask=False)
        rgbs.append(rgb.astype(np.float32) / 255.0)

        if raw_idx < len(amodal_files):
            mask = crop_and_resize_frame(load_mask_frame(amodal_files[raw_idx]), bbox, height, width, is_mask=True)
        else:
            mask = np.zeros((height, width), dtype=np.uint8)
        amodal_masks.append(mask.astype(np.uint8))

        combined_hand = np.zeros((height, width), dtype=np.uint8)
        for hand_files in hand_files_by_dir:
            if raw_idx < len(hand_files):
                hand_mask = crop_and_resize_frame(load_mask_frame(hand_files[raw_idx]), bbox, height, width, is_mask=True)
                combined_hand = np.logical_or(combined_hand, hand_mask).astype(np.uint8)
        hand_masks.append(combined_hand)

    return np.stack(rgbs, axis=0), np.stack(amodal_masks, axis=0), np.stack(hand_masks, axis=0)


def overlay_render(background: np.ndarray, render_rgb: np.ndarray, render_alpha: np.ndarray, alpha: float) -> np.ndarray:
    mask = np.clip(render_alpha[..., None], 0.0, 1.0)
    return np.clip(background * (1.0 - mask * alpha) + render_rgb * (mask * alpha), 0.0, 1.0)


def overlay_mask(rgb: np.ndarray, mask: np.ndarray, color: tuple[float, float, float], alpha: float = 0.55) -> np.ndarray:
    color_arr = np.asarray(color, dtype=np.float32).reshape(1, 1, 3)
    mask_arr = mask.astype(np.float32)[..., None]
    return np.clip(rgb * (1.0 - mask_arr * alpha) + color_arr * (mask_arr * alpha), 0.0, 1.0)


def _lazy_render_imports():
    import torch
    from manotorch.manolayer import ManoLayer as AMANOLayer
    from pytorch3d.io import load_obj
    from pytorch3d.renderer import (
        BlendParams,
        Materials,
        MeshRasterizer,
        MeshRenderer,
        PerspectiveCameras,
        PointLights,
        RasterizationSettings,
        TexturesVertex,
    )
    from pytorch3d.renderer.mesh.shader import HardPhongShader
    from pytorch3d.structures import Meshes, join_meshes_as_batch, join_meshes_as_scene

    from body_model import run_amano

    return {
        "torch": torch,
        "AMANOLayer": AMANOLayer,
        "load_obj": load_obj,
        "BlendParams": BlendParams,
        "Materials": Materials,
        "MeshRasterizer": MeshRasterizer,
        "MeshRenderer": MeshRenderer,
        "PerspectiveCameras": PerspectiveCameras,
        "PointLights": PointLights,
        "RasterizationSettings": RasterizationSettings,
        "TexturesVertex": TexturesVertex,
        "HardPhongShader": HardPhongShader,
        "Meshes": Meshes,
        "join_meshes_as_batch": join_meshes_as_batch,
        "join_meshes_as_scene": join_meshes_as_scene,
        "run_amano": run_amano,
    }


def make_mano_layer(imports: dict, mano_root: Path, device: str):
    return imports["AMANOLayer"](
        mano_assets_root=str(mano_root),
        flat_hand_mean=True,
        use_pca=False,
        side="right",
    ).to(device)


def load_object_mesh(imports: dict, hoi_dir: Path, device: str):
    torch = imports["torch"]
    verts, faces, _ = imports["load_obj"](str(hoi_dir / "obj_canonical.obj"), load_textures=False)
    return verts.to(device), faces.verts_idx.to(device).long()


def build_renderer(imports: dict, camera: CropCamera, config: RenderConfig, batch_size: int, image_scale: int = 1):
    torch = imports["torch"]
    height, width = camera.image_size
    image_size = (height * image_scale, width * image_scale)
    focal = torch.tensor([[camera.focal_length[0] * image_scale, camera.focal_length[1] * image_scale]],
                         dtype=torch.float32,
                         device=config.device).expand(batch_size, -1)
    principal = torch.tensor([[camera.principal_point[0] * image_scale, camera.principal_point[1] * image_scale]],
                             dtype=torch.float32,
                             device=config.device).expand(batch_size, -1)
    cameras = imports["PerspectiveCameras"](
        focal_length=focal,
        principal_point=principal,
        image_size=(image_size,) * batch_size,
        in_ndc=False,
        device=config.device,
    )
    raster_settings = imports["RasterizationSettings"](
        image_size=image_size,
        blur_radius=0.0,
        faces_per_pixel=1,
        bin_size=None,
    )
    blend_params = imports["BlendParams"](background_color=config.background_color)
    lights = imports["PointLights"](device=config.device, location=[[0.0, 0.0, -3.0]])
    materials = imports["Materials"](
        device=config.device,
        specular_color=[[1.0, 1.0, 1.0]],
        shininess=1.0,
    )
    renderer = imports["MeshRenderer"](
        rasterizer=imports["MeshRasterizer"](cameras=cameras, raster_settings=raster_settings),
        shader=imports["HardPhongShader"](
            device=config.device,
            cameras=cameras,
            lights=lights,
            materials=materials,
            blend_params=blend_params,
        ),
    )
    return renderer, cameras, lights, materials


def downsample_if_needed(images: np.ndarray, image_scale: int) -> np.ndarray:
    if image_scale == 1:
        return images
    out = []
    for image in images:
        h, w = image.shape[:2]
        out.append(cv2.resize(image, (w // image_scale, h // image_scale), interpolation=cv2.INTER_AREA))
    return np.stack(out, axis=0)


def make_textured_mesh(imports: dict, verts, faces, color: tuple[float, float, float]):
    torch = imports["torch"]
    TexturesVertex = imports["TexturesVertex"]
    Meshes = imports["Meshes"]
    if verts.ndim == 2:
        verts = verts.unsqueeze(0)
    if faces.ndim == 2:
        faces = faces.unsqueeze(0).expand(verts.shape[0], -1, -1)
    verts_rgb = torch.tensor(color, dtype=torch.float32, device=verts.device).view(1, 1, 3).expand(verts.shape[0], verts.shape[1], -1)
    return Meshes(verts=list(verts), faces=list(faces), textures=TexturesVertex(verts_features=verts_rgb))


def transform_object_vertices(imports: dict, verts, obj_records: Sequence[dict]):
    torch = imports["torch"]
    posed = []
    for record in obj_records:
        scale = torch.tensor(record["scale"], dtype=torch.float32, device=verts.device).view(1, 3)
        # Stored rotations are column-major in the HOI export.  The renderer path
        # uses row-vector multiplication: (verts * scale) @ R_row + T.
        rot = torch.tensor(record["rotation"], dtype=torch.float32, device=verts.device).t()
        trans = torch.tensor(record["translation"], dtype=torch.float32, device=verts.device).view(1, 3)
        posed.append((verts * scale) @ rot + trans)
    return torch.stack(posed, dim=0)


def build_hand_vertices(imports: dict, mano_layer, mano_records: Sequence[dict], device: str):
    torch = imports["torch"]
    run_amano = imports["run_amano"]
    root = torch.tensor([r["root_orient"] for r in mano_records], dtype=torch.float32, device=device)
    trans = torch.tensor([r["trans"] for r in mano_records], dtype=torch.float32, device=device)
    pose = torch.tensor([r["pose"] for r in mano_records], dtype=torch.float32, device=device)
    is_right = torch.tensor([[float(r["is_right"])] for r in mano_records], dtype=torch.float32, device=device)
    output = run_amano(mano_layer, trans[None], root[None], pose[None], is_right)

    flat_mat = torch.diag(torch.tensor([-1.0, -1.0, 1.0], dtype=torch.float32, device=device))
    verts = output["vertices"].squeeze(0) @ flat_mat
    faces = output["r_faces"] if is_right[0].item() > 0 else output["l_faces"]
    return verts, faces.long()


def rotate_view(imports: dict, obj_verts, hand_verts, view: str):
    torch = imports["torch"]
    if view == "camera":
        return obj_verts, hand_verts
    center = torch.cat([obj_verts, hand_verts], dim=1).reshape(-1, 3).mean(dim=0)

    def _rotate(v):
        w = v - center
        if view == "side":
            return torch.stack([w[..., 2], w[..., 1], -w[..., 0]], dim=-1) + center
        if view == "top":
            return torch.stack([w[..., 0], w[..., 2], -w[..., 1]], dim=-1) + center
        raise ValueError(f"Unknown view: {view}")

    return _rotate(obj_verts), _rotate(hand_verts)


def render_mesh_batch(imports: dict,
                      obj_verts,
                      obj_faces,
                      hand_verts,
                      hand_faces,
                      camera: CropCamera,
                      config: RenderConfig,
                      view: str,
                      image_scale: int):
    obj_view, hand_view = rotate_view(imports, obj_verts, hand_verts, view)
    obj_meshes = make_textured_mesh(imports, obj_view, obj_faces, config.object_color)
    hand_meshes = make_textured_mesh(imports, hand_view, hand_faces, config.hand_color)
    scenes = [
        imports["join_meshes_as_scene"]([obj_meshes[i], hand_meshes[i]])
        for i in range(obj_view.shape[0])
    ]
    batch_scene = imports["join_meshes_as_batch"](scenes)
    renderer, cameras, lights, materials = build_renderer(imports, camera, config, obj_view.shape[0], image_scale=image_scale)
    with imports["torch"].no_grad():
        rendered = renderer(batch_scene, cameras=cameras, lights=lights, materials=materials).detach().cpu().numpy()
    rendered = downsample_if_needed(rendered, image_scale)
    return np.clip(rendered[..., :3], 0.0, 1.0), np.clip(rendered[..., 3], 0.0, 1.0)


def open_writer(path: Path, fps: float, crf: int, preset: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    return imageio.get_writer(
        str(path),
        fps=fps,
        codec="libx264",
        pixelformat="yuv420p",
        ffmpeg_params=["-crf", str(crf), "-preset", preset],
        macro_block_size=None,
    )


def append_rgb(writer, frame: np.ndarray) -> None:
    writer.append_data((np.clip(frame, 0.0, 1.0) * 255).astype(np.uint8))


def render_sequence(seq_dir: Path, config: RenderConfig) -> None:
    imports = _lazy_render_imports()
    torch = imports["torch"]
    if config.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {config.device}, but CUDA is not available.")

    hoi_dir = find_hoi_sequence_dir(seq_dir)
    output_dir = seq_dir / config.output_name
    output_files = {
        "camera_overlay": output_dir / "paper_camera_overlay.mp4",
        "mesh_camera": output_dir / "paper_mesh_camera.mp4",
        "mesh_side": output_dir / "paper_mesh_side.mp4",
        "mesh_top": output_dir / "paper_mesh_top.mp4",
        "mask_overlay": output_dir / "paper_mask_overlay.mp4",
    }
    if config.skip_existing and all(path.exists() for path in output_files.values()):
        print(f"[{seq_dir.name}] all outputs exist, skipping.")
        return

    indices = load_optimized_indices(hoi_dir, config.max_frames)
    intrinsics = load_intrinsics(seq_dir)
    bbox = load_global_bbox(seq_dir)
    camera = compute_crop_camera(intrinsics, bbox, config.height, config.width)
    frame_offset = infer_frame_offset(seq_dir)
    rgbs, amodal_masks, hand_masks = load_sequence_frames(seq_dir, indices, bbox, config.height, config.width, frame_offset=frame_offset)

    obj_verts, obj_faces = load_object_mesh(imports, hoi_dir, config.device)
    mano_layer = make_mano_layer(imports, config.mano_root, config.device)

    writers = {
        key: open_writer(path, config.fps, config.crf, config.preset)
        for key, path in output_files.items()
    }
    try:
        for start in tqdm(range(0, len(indices), config.batch_size), desc=f"Render {seq_dir.name}"):
            batch_indices = indices[start:start + config.batch_size]
            obj_records = [load_json(hoi_dir / f"obj_{idx:05d}.json") for idx in batch_indices]
            mano_records = [load_json(hoi_dir / f"mano_{idx:05d}.json") for idx in batch_indices]

            batch_obj_verts = transform_object_vertices(imports, obj_verts, obj_records)
            batch_hand_verts, hand_faces = build_hand_vertices(imports, mano_layer, mano_records, config.device)

            camera_rgb, camera_alpha = render_mesh_batch(
                imports, batch_obj_verts, obj_faces, batch_hand_verts, hand_faces,
                camera, config, view="camera", image_scale=config.image_scale)
            side_rgb, _ = render_mesh_batch(
                imports, batch_obj_verts, obj_faces, batch_hand_verts, hand_faces,
                camera, config, view="side", image_scale=config.image_scale)
            top_rgb, _ = render_mesh_batch(
                imports, batch_obj_verts, obj_faces, batch_hand_verts, hand_faces,
                camera, config, view="top", image_scale=config.image_scale)

            for local_i in range(len(batch_indices)):
                frame_i = start + local_i
                append_rgb(writers["mesh_camera"], camera_rgb[local_i])
                append_rgb(writers["mesh_side"], side_rgb[local_i])
                append_rgb(writers["mesh_top"], top_rgb[local_i])
                overlay = overlay_render(rgbs[frame_i], camera_rgb[local_i], camera_alpha[local_i], config.overlay_alpha)
                append_rgb(writers["camera_overlay"], overlay)

                mask_frame = overlay_mask(rgbs[frame_i], amodal_masks[frame_i], color=(0.15, 0.55, 1.0), alpha=0.45)
                mask_frame = overlay_mask(mask_frame, hand_masks[frame_i], color=config.hand_color, alpha=0.45)
                append_rgb(writers["mask_overlay"], mask_frame)
    finally:
        for writer in writers.values():
            writer.close()

    print(f"[{seq_dir.name}] saved paper renders to {output_dir}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render high-quality PyTorch3D HOI videos for papers.")
    parser.add_argument("--data_path", type=Path, default=DEFAULT_DATA_PATH, help="Parent directory containing video sequence folders.")
    parser.add_argument("--video_id", nargs="+", default=None, help="One or more video IDs / sequence folder names to render.")
    parser.add_argument("--output_name", default="paper_renders", help="Output subdirectory name under each video folder.")
    parser.add_argument("--height", type=int, default=1080, help="Output render height.")
    parser.add_argument("--width", type=int, default=1920, help="Output render width.")
    parser.add_argument("--fps", type=float, default=30.0, help="Output video FPS.")
    parser.add_argument("--crf", type=int, default=18, help="libx264 CRF; lower is higher quality.")
    parser.add_argument("--preset", default="slow", help="libx264 preset.")
    parser.add_argument("--batch_size", type=int, default=4, help="Render batch size.")
    parser.add_argument("--device", default="cuda:0", help="Torch device, e.g. cuda:0 or cpu.")
    parser.add_argument("--quality", choices=("preview", "final"), default="final", help="Render quality preset. preview disables supersampling for quick checks.")
    parser.add_argument("--hand_color", nargs=3, default=("0.86", "0.63", "0.50"), help="Hand RGB values in [0, 1].")
    parser.add_argument("--object_color", nargs=3, default=("0.58", "0.70", "0.86"), help="Object RGB values in [0, 1].")
    parser.add_argument("--background_color", nargs=3, default=("1.0", "1.0", "1.0"), help="Clean-view background RGB values in [0, 1].")
    parser.add_argument("--overlay_alpha", type=float, default=0.85, help="Alpha for compositing camera render over RGB.")
    parser.add_argument("--max_frames", type=int, default=None, help="Optional frame limit for smoke tests.")
    parser.add_argument("--skip_existing", action="store_true", help="Skip a video if all expected outputs already exist.")
    parser.add_argument("--mano_root", type=Path, default=DEFAULT_MANO_ROOT, help="MANO assets directory.")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> RenderConfig:
    args = build_arg_parser().parse_args(argv)
    image_scale = 1 if args.quality == "preview" else 2
    return RenderConfig(
        data_path=args.data_path,
        video_ids=args.video_id,
        output_name=args.output_name,
        height=args.height,
        width=args.width,
        fps=args.fps,
        crf=args.crf,
        preset=args.preset,
        batch_size=args.batch_size,
        device=args.device,
        quality=args.quality,
        image_scale=image_scale,
        hand_color=parse_rgb_triplet(args.hand_color),
        object_color=parse_rgb_triplet(args.object_color),
        background_color=parse_rgb_triplet(args.background_color),
        overlay_alpha=float(args.overlay_alpha),
        max_frames=args.max_frames,
        skip_existing=bool(args.skip_existing),
        mano_root=args.mano_root,
    )


def main(argv: Sequence[str] | None = None) -> int:
    config = parse_args(argv)
    seq_dirs = list_video_dirs(config.data_path, config.video_ids)
    if not seq_dirs:
        print(f"No video directories found under {config.data_path}")
        return 0

    failed: list[tuple[str, str]] = []
    for seq_dir in seq_dirs:
        try:
            render_sequence(seq_dir, config)
        except Exception as exc:
            failed.append((seq_dir.name, str(exc)))
            print(f"[{seq_dir.name}] ERROR: {exc}")

    if failed:
        print("Some videos failed:")
        for video_id, error in failed:
            print(f"  - {video_id}: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
