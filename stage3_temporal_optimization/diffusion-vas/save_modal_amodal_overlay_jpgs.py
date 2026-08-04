import argparse
import json
import os
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np


DEFAULT_OUTPUT_SIZE = (256, 512)
# Slightly saturated Okabe-Ito style colors: still print-friendly, a bit clearer on RGB frames.
MODAL_COLOR = (0 / 255, 158 / 255, 142 / 255)      # teal, #009E8E
AMODAL_COLOR = (198 / 255, 90 / 255, 142 / 255)    # muted magenta, #C65A8E

ENV_INPUT_DATA = "DIFFUSION_VAS_INPUT_DATA"
HAND_SKELETON_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
)
OBJECT_BBOX_COLOR = (31 / 255, 92 / 255, 75 / 255)    # deep ink green, #1F5C4B
HAND_BBOX_COLORS = {
    "lh": (91 / 255, 85 / 255, 122 / 255),            # slate violet, #5B557A
    "rh": (109 / 255, 88 / 255, 120 / 255),           # muted plum, #6D5878
}

HAND_EDGE_COLORS = {
    (0, 1): (230 / 255, 70 / 255, 70 / 255),
    (1, 2): (238 / 255, 105 / 255, 55 / 255),
    (2, 3): (242 / 255, 135 / 255, 50 / 255),
    (3, 4): (245 / 255, 160 / 255, 55 / 255),

    (0, 5): (210 / 255, 170 / 255, 45 / 255),
    (5, 6): (170 / 255, 185 / 255, 45 / 255),
    (6, 7): (125 / 255, 190 / 255, 65 / 255),
    (7, 8): (80 / 255, 180 / 255, 95 / 255),

    (0, 9): (45 / 255, 170 / 255, 150 / 255),
    (9, 10): (35 / 255, 175 / 255, 180 / 255),
    (10, 11): (40 / 255, 165 / 255, 205 / 255),
    (11, 12): (55 / 255, 150 / 255, 220 / 255),

    (0, 13): (80 / 255, 125 / 255, 220 / 255),
    (13, 14): (90 / 255, 105 / 255, 220 / 255),
    (14, 15): (105 / 255, 90 / 255, 215 / 255),
    (15, 16): (120 / 255, 80 / 255, 210 / 255),

    (0, 17): (155 / 255, 85 / 255, 205 / 255),
    (17, 18): (180 / 255, 80 / 255, 190 / 255),
    (18, 19): (205 / 255, 80 / 255, 170 / 255),
    (19, 20): (220 / 255, 85 / 255, 145 / 255),
}


def default_data_root() -> Path:
    env = os.environ.get(ENV_INPUT_DATA)
    if env:
        return Path(env).expanduser().resolve()
    return (Path(__file__).resolve().parent / "input_data").resolve()


def load_json_dict(path):
    path = Path(path)
    if not path.exists():
        return {}
    with open(path, "r") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def load_hand_annotations(seq_path):
    seq_path = Path(seq_path)
    annotations = {}
    for side in ("lh", "rh"):
        bbox = load_json_dict(seq_path / f"{side}_bbox.json")
        keypoints = load_json_dict(seq_path / f"{side}_keypoints.json")
        if bbox or keypoints:
            annotations[side] = {"bbox": bbox, "keypoints": keypoints}
    return annotations


def select_hand_annotations_for_side(hand_annotations, hand_side="rh"):
    """When both lh and rh exist, keep only the chosen side; otherwise keep all loaded sides."""
    if hand_side not in ("lh", "rh"):
        hand_side = "rh"
    sides = set(hand_annotations.keys()) & {"lh", "rh"}
    if "lh" in sides and "rh" in sides:
        if hand_side in hand_annotations:
            return {hand_side: hand_annotations[hand_side]}
        return {}
    return dict(hand_annotations)


def auto_line_width(image_shape, base, ref_height=512, min_width=1):
    height = image_shape[0]
    return max(min_width, int(round(base * height / ref_height)))


def auto_radius(image_shape, base, ref_height=512, min_radius=2):
    height = image_shape[0]
    return max(min_radius, int(round(base * height / ref_height)))


def make_square_bbox(bbox, image_shape):
    if bbox is None or len(bbox) != 4:
        return None

    h, w = image_shape[:2]
    x1, y1, x2, y2 = [float(v) for v in bbox]
    if x2 <= x1 or y2 <= y1:
        return None

    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    side = max(x2 - x1, y2 - y1)

    sx1 = int(np.clip(round(cx - side / 2), 0, w - 1))
    sy1 = int(np.clip(round(cy - side / 2), 0, h - 1))
    sx2 = int(np.clip(round(cx + side / 2), 0, w - 1))
    sy2 = int(np.clip(round(cy + side / 2), 0, h - 1))

    if sx2 <= sx1 or sy2 <= sy1:
        return None
    return sx1, sy1, sx2, sy2


def hand_point_color(index):
    if index == 0:
        return (55 / 255, 55 / 255, 55 / 255)
    if 1 <= index <= 4:
        return HAND_EDGE_COLORS[(3, 4)]
    if 5 <= index <= 8:
        return HAND_EDGE_COLORS[(7, 8)]
    if 9 <= index <= 12:
        return HAND_EDGE_COLORS[(11, 12)]
    if 13 <= index <= 16:
        return HAND_EDGE_COLORS[(15, 16)]
    return HAND_EDGE_COLORS[(19, 20)]


def overlay_mask_with_color(
    rgb_img,
    mask,
    color,
    boundary_thickness=None,
    alpha_fill=0.18,
    alpha_edge=1.0,
    halo_thickness=None,
):
    """Publication-style overlay: light color fill + resolution-aware white halo + colored contour."""
    assert rgb_img.shape[-1] == 3, "Expected RGB image with 3 channels"

    rgb = rgb_img.astype(np.float32)
    if rgb.max() > 1.0:
        rgb = rgb / 255.0

    if boundary_thickness is None:
        boundary_thickness = auto_line_width(rgb.shape, base=2)
    if halo_thickness is None:
        halo_thickness = auto_line_width(rgb.shape, base=5)

    mask_bool = mask.astype(bool)
    color_arr = np.asarray(color, dtype=np.float32)

    out = rgb.copy()
    out[mask_bool] = out[mask_bool] * (1.0 - alpha_fill) + color_arr * alpha_fill

    mask_u8 = (mask_bool.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        out_u8 = (np.clip(out, 0.0, 1.0) * 255).astype(np.uint8)
        color_u8 = tuple(int(c * 255) for c in color_arr)

        cv2.drawContours(out_u8, contours, -1, (255, 255, 255), halo_thickness, cv2.LINE_AA)
        cv2.drawContours(out_u8, contours, -1, color_u8, boundary_thickness, cv2.LINE_AA)

        out = out_u8.astype(np.float32) / 255.0

    return np.clip(out, 0.0, 1.0)


def concat_with_gutter(left, right, gutter=12, gutter_color=1.0):
    images = (left, right)
    h = max(image.shape[0] for image in images)
    gutter_img = np.full((h, gutter, 3), gutter_color, dtype=np.float32)
    return np.hstack([left, gutter_img, right])


def concat_panels_with_gutter(panels, gutter=12, gutter_color=1.0):
    if not panels:
        raise ValueError("Expected at least one panel to concatenate")
    h = max(panel.shape[0] for panel in panels)
    prepared = []
    for panel in panels:
        if panel.shape[0] == h:
            prepared.append(panel)
            continue
        pad_h = h - panel.shape[0]
        pad = np.full((pad_h, panel.shape[1], 3), gutter_color, dtype=np.float32)
        prepared.append(np.vstack([panel, pad]))

    gutter_img = np.full((h, gutter, 3), gutter_color, dtype=np.float32)
    pieces = []
    for i, panel in enumerate(prepared):
        if i > 0:
            pieces.append(gutter_img)
        pieces.append(panel)
    return np.hstack(pieces)


def _to_uint8_rgb(img):
    if img.max() <= 1.0:
        return (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)
    return np.clip(img, 0, 255).astype(np.uint8)


def _rgb_color_to_u8(color):
    return tuple(int(np.clip(c, 0.0, 1.0) * 255) for c in color)


def compute_mask_bbox(mask):
    mask_u8 = (mask > 0).astype(np.uint8)
    if int(mask_u8.sum()) == 0:
        return None
    x, y, w, h = cv2.boundingRect(mask_u8)
    return (int(x), int(y), int(x + w), int(y + h))


def draw_bbox(img_u8, bbox, color, thickness=None, square=True):
    if bbox is None or len(bbox) != 4:
        return img_u8

    if square:
        bbox = make_square_bbox(bbox, img_u8.shape)
        if bbox is None:
            return img_u8

    h, w = img_u8.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    x1, x2 = int(np.clip(x1, 0, w - 1)), int(np.clip(x2, 0, w - 1))
    y1, y2 = int(np.clip(y1, 0, h - 1)), int(np.clip(y2, 0, h - 1))
    if x2 <= x1 or y2 <= y1:
        return img_u8

    if thickness is None:
        thickness = auto_line_width(img_u8.shape, base=3.2, min_width=2)

    halo_thickness = thickness + auto_line_width(img_u8.shape, base=1.4, min_width=2)

    cv2.rectangle(img_u8, (x1, y1), (x2, y2), (255, 255, 255), halo_thickness, cv2.LINE_AA)
    cv2.rectangle(img_u8, (x1, y1), (x2, y2), _rgb_color_to_u8(color), thickness, cv2.LINE_AA)
    return img_u8


def _valid_point(point):
    if point is None or len(point) < 2:
        return False
    x, y = point[:2]
    if not np.isfinite(x) or not np.isfinite(y):
        return False
    if len(point) >= 3 and point[2] <= 0:
        return False
    return True


def draw_hand_skeleton(img_u8, keypoints, color=None, line_thickness=None, point_radius=None):
    if not keypoints:
        return img_u8

    if line_thickness is None:
        line_thickness = auto_line_width(img_u8.shape, base=2.8, min_width=2)
    if point_radius is None:
        point_radius = auto_radius(img_u8.shape, base=4.2, min_radius=3)

    outer_radius = point_radius + max(2, point_radius // 3)

    points = []
    for point in keypoints:
        if _valid_point(point):
            points.append((int(round(point[0])), int(round(point[1]))))
        else:
            points.append(None)

    for i, j in HAND_SKELETON_EDGES:
        if i < len(points) and j < len(points) and points[i] is not None and points[j] is not None:
            edge_color = HAND_EDGE_COLORS.get((i, j), (120 / 255, 120 / 255, 120 / 255))
            cv2.line(img_u8, points[i], points[j], _rgb_color_to_u8(edge_color), line_thickness, cv2.LINE_AA)

    for idx, point in enumerate(points):
        if point is not None:
            point_color = hand_point_color(idx)
            cv2.circle(img_u8, point, outer_radius, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(img_u8, point, point_radius, _rgb_color_to_u8(point_color), -1, cv2.LINE_AA)

    return img_u8


def make_bbox_skeleton_panel(rgb, obj_mask, frame_idx, hand_annotations, hand_side="rh"):
    out_u8 = _to_uint8_rgb(rgb)
    hand_annotations = select_hand_annotations_for_side(hand_annotations, hand_side)

    obj_bbox = compute_mask_bbox(obj_mask)
    draw_bbox(out_u8, obj_bbox, OBJECT_BBOX_COLOR, square=True)

    frame_key = str(frame_idx)
    for side, annotation in hand_annotations.items():
        bbox = annotation.get("bbox", {}).get(frame_key)
        keypoints = annotation.get("keypoints", {}).get(frame_key)

        draw_bbox(
            out_u8,
            bbox,
            HAND_BBOX_COLORS.get(side, HAND_BBOX_COLORS["rh"]),
            square=True,
        )
        draw_hand_skeleton(out_u8, keypoints)

    return out_u8.astype(np.float32) / 255.0


def load_raw_frames(folder_path, frame_type="rgb"):
    """Self-contained image-sequence loader for RGB frames and binary masks."""
    folder = Path(folder_path)
    files = sorted(
        [p for p in folder.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}],
        key=lambda p: int(p.stem),
    )

    frames = []
    for path in files:
        if frame_type == "mask":
            img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if img is None:
                continue
            if img.ndim == 3 and img.shape[2] == 4:
                mask = img[:, :, 3]
            elif img.ndim == 3:
                mask = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                mask = img
            frames.append((mask > 128).astype(np.uint8))
        else:
            img = cv2.imread(str(path))
            if img is not None:
                frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return frames


def crop_and_resize_frame(frame, bbox, output_size=DEFAULT_OUTPUT_SIZE, is_mask=False):
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    crop_w = x2 - x1
    crop_h = y2 - y1
    if crop_w <= 0 or crop_h <= 0:
        raise ValueError(f"Invalid bbox with non-positive size: {bbox}")

    img_h, img_w = frame.shape[:2]
    src_x1, src_y1 = max(0, x1), max(0, y1)
    src_x2, src_y2 = min(img_w, x2), min(img_h, y2)
    dst_x1, dst_y1 = src_x1 - x1, src_y1 - y1
    dst_x2, dst_y2 = src_x2 - x1, src_y2 - y1

    if frame.ndim == 3:
        canvas = np.zeros((crop_h, crop_w, frame.shape[2]), dtype=frame.dtype)
    else:
        canvas = np.zeros((crop_h, crop_w), dtype=frame.dtype)

    if src_x2 > src_x1 and src_y2 > src_y1:
        canvas[dst_y1:dst_y2, dst_x1:dst_x2] = frame[src_y1:src_y2, src_x1:src_x2]

    if is_mask:
        canvas = (canvas > 0).astype(np.uint8) * 255

    h_out, w_out = output_size
    h_canvas, w_canvas = canvas.shape[:2]
    scale = min(w_out / w_canvas, h_out / h_canvas)
    new_w, new_h = int(w_canvas * scale), int(h_canvas * scale)
    interpolation = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    resized_canvas = cv2.resize(canvas, (new_w, new_h), interpolation=interpolation)

    if resized_canvas.ndim == 3:
        final_image = np.zeros((h_out, w_out, resized_canvas.shape[2]), dtype=resized_canvas.dtype)
    else:
        final_image = np.zeros((h_out, w_out), dtype=resized_canvas.dtype)

    pad_top = (h_out - new_h) // 2
    pad_left = (w_out - new_w) // 2
    final_image[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized_canvas

    if is_mask:
        return (final_image > 128).astype(np.uint8)
    return final_image.astype(np.uint8)


def project_crop_mask_to_original_frame(mask, bbox, image_shape):
    """Invert crop_and_resize_frame(..., is_mask=True) and paste the mask into original image coordinates."""
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    crop_w = x2 - x1
    crop_h = y2 - y1
    if crop_w <= 0 or crop_h <= 0:
        raise ValueError(f"Invalid bbox with non-positive size: {bbox}")

    img_h, img_w = image_shape[:2]
    mask_h, mask_w = mask.shape[:2]
    scale = min(mask_w / crop_w, mask_h / crop_h)
    new_w, new_h = int(crop_w * scale), int(crop_h * scale)
    pad_top = (mask_h - new_h) // 2
    pad_left = (mask_w - new_w) // 2

    resized_region = mask[pad_top:pad_top + new_h, pad_left:pad_left + new_w]
    crop_mask = cv2.resize(
        resized_region.astype(np.uint8),
        (crop_w, crop_h),
        interpolation=cv2.INTER_NEAREST,
    )
    crop_mask = (crop_mask > 0).astype(np.uint8)

    full_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    src_x1, src_y1 = max(0, x1), max(0, y1)
    src_x2, src_y2 = min(img_w, x2), min(img_h, y2)
    if src_x2 <= src_x1 or src_y2 <= src_y1:
        return full_mask

    dst_x1, dst_y1 = src_x1 - x1, src_y1 - y1
    dst_x2, dst_y2 = dst_x1 + (src_x2 - src_x1), dst_y1 + (src_y2 - src_y1)
    full_mask[src_y1:src_y2, src_x1:src_x2] = crop_mask[dst_y1:dst_y2, dst_x1:dst_x2]
    return full_mask


def load_bboxes(path):
    with open(path, "r") as f:
        return json.load(f)


def load_amodal_masks(amodal_dir, max_frames=None):
    amodal_dir = Path(amodal_dir)
    mask_files = sorted(
        [p for p in amodal_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}],
        key=lambda p: int(p.stem),
    )
    if max_frames is not None:
        mask_files = mask_files[:max_frames]

    masks = []
    for path in mask_files:
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Failed to read amodal mask: {path}")
        masks.append((mask > 128).astype(np.uint8))
    return np.asarray(masks, dtype=np.uint8)


def save_overlay_comparisons(
    rgbs,
    modal_masks,
    amodal_masks,
    output_dir,
    bbox_skeleton_panels=None,
    overwrite=False,
    modal_color=MODAL_COLOR,
    amodal_color=AMODAL_COLOR,
):
    num_frames = len(rgbs)
    if len(modal_masks) != num_frames or len(amodal_masks) != num_frames:
        raise ValueError(
            "RGB, modal mask, and amodal mask sequences must have the same length: "
            f"{num_frames}, {len(modal_masks)}, {len(amodal_masks)}"
        )
    if bbox_skeleton_panels is not None and len(bbox_skeleton_panels) != num_frames:
        raise ValueError(
            "BBox/skeleton panels must match RGB sequence length: "
            f"{len(bbox_skeleton_panels)} != {num_frames}"
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = list(output_dir.glob("*.jpg"))
    if existing and not overwrite:
        raise FileExistsError(f"{output_dir} already contains JPG files; pass --overwrite to replace them.")
    if overwrite:
        for path in existing:
            path.unlink()

    for frame_idx in range(num_frames):
        modal_overlay = overlay_mask_with_color(
            rgbs[frame_idx],
            modal_masks[frame_idx],
            modal_color,
            alpha_fill=0.18,
        )
        amodal_overlay = overlay_mask_with_color(
            rgbs[frame_idx],
            amodal_masks[frame_idx],
            amodal_color,
            alpha_fill=0.18,
        )

        panels = [modal_overlay, amodal_overlay]
        if bbox_skeleton_panels is not None:
            panels.append(bbox_skeleton_panels[frame_idx])
        combined = concat_panels_with_gutter(panels, gutter=18, gutter_color=1.0)
        imageio.imwrite(output_dir / f"{frame_idx:06d}.jpg", (combined * 255).astype(np.uint8), quality=98)

    return num_frames


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Save per-frame modal/amodal mask overlay comparison JPGs under {data_root}/{video_id}/."
    )
    parser.add_argument(
        "--video_id",
        type=str,
        nargs="+",
        required=True,
        help="One or more sequence folder names under the data root (each holds rgbs/, obj_masks/, global_bbox.json, amodal_masks/).",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help=f"Parent of sequence folders. Default: repo input_data/ or {ENV_INPUT_DATA} env.",
    )
    parser.add_argument("--output_dir_name", type=str, default="mask_overlay_compare_jpgs")
    parser.add_argument("--max_frames", type=int, default=None, help="Optional number of frames to save for a quick check.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing JPGs in the output directory.")
    parser.add_argument(
        "--hand_side",
        type=str,
        choices=("lh", "rh"),
        default="rh",
        help="When both left- and right-hand annotations exist, draw only this hand (default: rh).",
    )
    args = parser.parse_args(argv)
    root = args.data_root if args.data_root is not None else default_data_root()
    args.data_root = root.resolve()
    return args


def export_overlays_for_video_id(args, video_id: str) -> int:
    """Write JPGs for one sequence. Returns number of frames saved."""
    seq_path = args.data_root / video_id

    bbox_path = seq_path / "global_bbox.json"
    amodal_dir = seq_path / "amodal_masks"
    output_dir = seq_path / args.output_dir_name

    if not bbox_path.exists():
        raise FileNotFoundError(f"[{video_id}] Missing bbox file: {bbox_path}")
    if not amodal_dir.is_dir():
        raise FileNotFoundError(f"[{video_id}] Missing amodal mask directory: {amodal_dir}")

    raw_rgbs = load_raw_frames(str(seq_path / "rgbs"), frame_type="rgb")
    raw_modal_masks = load_raw_frames(str(seq_path / "obj_masks"), frame_type="mask")
    bboxes = load_bboxes(bbox_path)
    amodal_masks = load_amodal_masks(amodal_dir, max_frames=args.max_frames)
    hand_annotations = load_hand_annotations(seq_path)

    num_frames = min(len(raw_rgbs), len(raw_modal_masks), len(bboxes), len(amodal_masks))
    if args.max_frames is not None:
        num_frames = min(num_frames, args.max_frames)
    if num_frames == 0:
        raise ValueError(f"[{video_id}] No frames available for overlay export.")

    rgbs = []
    modal_masks = []
    amodal_original_masks = []
    bbox_skeleton_panels = []
    for frame_idx in range(num_frames):
        rgb = raw_rgbs[frame_idx].astype(np.float32) / 255.0
        modal_mask = raw_modal_masks[frame_idx]
        if modal_mask.shape[:2] != rgb.shape[:2]:
            modal_mask = cv2.resize(modal_mask, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)

        rgbs.append(rgb)
        modal_masks.append((modal_mask > 0).astype(np.uint8))
        amodal_original_masks.append(
            project_crop_mask_to_original_frame(amodal_masks[frame_idx], bboxes[frame_idx], rgb.shape[:2])
        )
        bbox_skeleton_panels.append(
            make_bbox_skeleton_panel(rgb, modal_mask, frame_idx, hand_annotations, hand_side=args.hand_side)
        )

    return save_overlay_comparisons(
        np.asarray(rgbs, dtype=np.float32),
        np.asarray(modal_masks, dtype=np.uint8),
        np.asarray(amodal_original_masks, dtype=np.uint8),
        output_dir,
        bbox_skeleton_panels=np.asarray(bbox_skeleton_panels, dtype=np.float32),
        overwrite=args.overwrite,
    )


def main(argv=None):
    args = parse_args(argv)

    unique_ids = list(dict.fromkeys(args.video_id))
    if len(unique_ids) != len(args.video_id):
        print("Warning: duplicate --video_id entries deduplicated while preserving order.")

    for video_id in unique_ids:
        saved_count = export_overlays_for_video_id(args, video_id)
        output_dir = args.data_root / video_id / args.output_dir_name
        print(f"[{video_id}] Saved {saved_count} modal/amodal overlay JPGs to {output_dir}")


if __name__ == "__main__":
    main()
