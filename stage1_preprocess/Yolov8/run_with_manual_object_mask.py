import json
import multiprocessing as mp
import os
import shutil
from pathlib import Path

import cv2
import imageio
import numpy as np
import torch
from tqdm import tqdm
from ultralytics import YOLO

from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor
from data_layout import (
    DEFAULT_DATA_DIR,
    DEFAULT_OUTPUT_DATA_DIR,
    annotation_dir_for_video,
    discover_input_videos,
    layout_for_video,
    visualization_dir_for_video,
)

MODULE_ROOT = Path(__file__).resolve().parent
DEFAULT_SAM2_CHECKPOINT = MODULE_ROOT / "sam2" / "checkpoints" / "sam2.1_hiera_large.pt"

# Target resolution (width, height) — all frames and outputs are enforced to this size.
TARGET_W, TARGET_H = 1920, 1080

# ---------------------------------------------------------------------------
# Data paths (overridable via CLI flags --data_dir / --output_dir).
# DATA_DIR        : flat data/{video_id}.mp4 inputs.
# OUTPUT_DATA_DIR : CHOIR output/ root; writes under output/<video_id>/inputs/
#                   (frames/rgb, masks, bbox, keypoints, annotations, video.mp4)
#                   and stage1/visualizations/.
# ---------------------------------------------------------------------------
# Global cache for models in worker processes
_WORKER_MODELS = {}

if torch.cuda.is_available():
    # no need to define autocast, directly use torch.amp.autocast
    from torch.amp import autocast
else:
    # fallback for CPU
    class autocast:

        def __init__(self, device_type='cpu', enabled=True):
            pass

        def __enter__(self):
            pass

        def __exit__(self, *args):
            pass


def read_video_frames(video_path):
    """
    Read all frames from a video file using imageio (ffmpeg backend).
    imageio/ffmpeg automatically applies rotation metadata (e.g. iPhone videos
    encoded with rotate=180), unlike cv2.VideoCapture which ignores it.

    Frames are cropped to the video's declared dimensions to remove any codec
    padding (e.g. H.264 pads height to multiples of 16: 1080 → 1088), then
    resized to TARGET_W × TARGET_H if necessary.

    Returns:
        list of BGR frames (numpy arrays), all at TARGET_W × TARGET_H.
    """
    # Get the true declared dimensions from container metadata
    cap = cv2.VideoCapture(str(video_path))
    true_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    true_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    reader = imageio.get_reader(str(video_path))
    frames = []
    for frame in reader:  # imageio returns RGB
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # Crop away any encoder padding
        if bgr.shape[0] != true_h or bgr.shape[1] != true_w:
            bgr = bgr[:true_h, :true_w]
        # Enforce target resolution
        if bgr.shape[1] != TARGET_W or bgr.shape[0] != TARGET_H:
            bgr = cv2.resize(bgr, (TARGET_W, TARGET_H), interpolation=cv2.INTER_LANCZOS4)
        frames.append(bgr)
    reader.close()
    return frames


def draw_detections(img, right_hand_info=None, left_hand_info=None, object_infos=None):
    """
    Draw hand and object detection results on image.

    Args:
        img: input image (numpy array)
        right_hand_info: [class_id, conf, x_center, y_center, w, h, kpt1_x, kpt1_y, kpt1_conf, ...]
        left_hand_info: [class_id, conf, x_center, y_center, w, h, kpt1_x, kpt1_y, kpt1_conf, ...]
        object_infos: list of object infos, each is [x_center, y_center, w, h, conf]
    
    Returns:
        img: image with detection results drawn
    """
    img_h, img_w = img.shape[:2]

    # Define colors (BGR format)
    RIGHT_HAND_COLOR = (0, 255, 0)  # Green for right hand
    LEFT_HAND_COLOR = (0, 0, 255)  # Red for left hand
    OBJECT_COLOR = (0, 255, 255)  # Yellow for object
    KEYPOINT_COLOR = (255, 255, 0)  # Cyan for keypoints

    def draw_hand(hand_info, color, hand_name):
        if hand_info is None or len(hand_info) < 6:
            return

        # --- Bbox from keypoints, with fallback to detector bbox ---
        use_kpt_bbox = False
        keypoints_data = hand_info[6:]
        num_keypoints = len(keypoints_data) // 3

        if num_keypoints > 0:
            kpts_x = [keypoints_data[i * 3] for i in range(num_keypoints) if keypoints_data[i * 3 + 2] > 0]
            kpts_y = [keypoints_data[i * 3 + 1] for i in range(num_keypoints) if keypoints_data[i * 3 + 2] > 0]

            if len(kpts_x) >= 2:
                min_x, max_x = min(kpts_x), max(kpts_x)
                min_y, max_y = min(kpts_y), max(kpts_y)

                x_center = (min_x + max_x) / 2
                y_center = (min_y + max_y) / 2
                w = (max_x - min_x) * 1.2
                h = (max_y - min_y) * 1.2
                use_kpt_bbox = True

        if not use_kpt_bbox:
            # Fallback to original detector bbox
            _, _, x_center, y_center, w, h = hand_info[:6]

        # Convert to pixel coordinates
        x_center_px = int(x_center * img_w)
        y_center_px = int(y_center * img_h)
        w_px = int(w * img_w)
        h_px = int(h * img_h)

        # Calculate bbox corners
        x1 = int(x_center_px - w_px / 2)
        y1 = int(y_center_px - h_px / 2)
        x2 = int(x_center_px + w_px / 2)
        y2 = int(y_center_px + h_px / 2)

        # Draw bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # Draw label
        label = hand_name
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        label_y = max(y1 - 10, label_size[1])
        cv2.rectangle(img, (x1, label_y - label_size[1] - 5), (x1 + label_size[0], label_y + 5), color, -1)
        cv2.putText(img, label, (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        for i in range(num_keypoints):
            kpt_x = keypoints_data[i * 3]
            kpt_y = keypoints_data[i * 3 + 1]
            kpt_conf = keypoints_data[i * 3 + 2]

            if kpt_conf > 0:  # Only draw if confident
                kpt_x_px = int(kpt_x * img_w)
                kpt_y_px = int(kpt_y * img_h)
                cv2.circle(img, (kpt_x_px, kpt_y_px), 3, KEYPOINT_COLOR, -1)
                cv2.circle(img, (kpt_x_px, kpt_y_px), 4, color, 1)

    def draw_object(object_info, color):
        if object_info is None or len(object_info) < 5:
            return

        # Extract bbox info (normalized)
        x_center, y_center, w, h, conf = object_info[:5]

        # Convert to pixel coordinates
        x_center_px = int(x_center * img_w)
        y_center_px = int(y_center * img_h)
        w_px = int(w * img_w)
        h_px = int(h * img_h)

        # Calculate bbox corners
        x1 = int(x_center_px - w_px / 2)
        y1 = int(y_center_px - h_px / 2)
        x2 = int(x_center_px + w_px / 2)
        y2 = int(y_center_px + h_px / 2)

        # Draw bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # Draw label
        label = f"Object {conf:.2f}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        label_y = max(y1 - 10, label_size[1])
        cv2.rectangle(img, (x1, label_y - label_size[1] - 5), (x1 + label_size[0], label_y + 5), color, -1)
        cv2.putText(img, label, (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Draw right hand (green)
    if right_hand_info is not None:
        draw_hand(right_hand_info, RIGHT_HAND_COLOR, "Right Hand")

    # Draw left hand (red)
    if left_hand_info is not None:
        draw_hand(left_hand_info, LEFT_HAND_COLOR, "Left Hand")

    # Draw objects (blue)
    if object_infos is not None:
        for obj_info in object_infos:
            draw_object(obj_info, OBJECT_COLOR)

    return img


def remove_small_regions(mask, min_area=2500):
    """
    Removes small connected components from a binary mask.
    """
    num_labels, labels_im, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    filtered_mask = np.zeros_like(mask, dtype=bool)
    if num_labels > 1:
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                filtered_mask[labels_im == i] = True
    return filtered_mask


def sam_video_tracking(video_predictor, frames, init_mask, init_frame_idx):
    """
    Tracks an object mask through a video sequence, both forwards and backwards
    from the initial frame.
    """
    if init_mask is None or not np.any(init_mask):
        return {}

    obj_id = 1
    init_mask_bool = init_mask.astype(bool) if init_mask.dtype != bool else init_mask

    # --- 1. Forward Tracking ---
    frames_forward = frames[init_frame_idx:]
    frames_forward_np = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_forward])

    video_segments_forward = {}
    if len(frames_forward_np) > 0:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float32):
            state_fwd = video_predictor.init_state(frames_forward_np)
            video_predictor.add_new_mask(inference_state=state_fwd, frame_idx=0, obj_id=obj_id, mask=init_mask_bool)
            for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(state_fwd):
                masks = (out_mask_logits > 0.0).cpu().numpy()
                original_frame_idx = init_frame_idx + out_frame_idx
                frame_segments = video_segments_forward.setdefault(original_frame_idx, {})
                for i, out_obj_id_i in enumerate(out_obj_ids):
                    if out_obj_id_i == obj_id:
                        frame_segments[out_obj_id_i] = masks[i]

    # --- 2. Backward Tracking ---
    frames_backward = frames[:init_frame_idx + 1][::-1]  # Reverse the frames
    frames_backward_np = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_backward])

    video_segments_backward = {}
    if len(frames_backward_np) > 1:  # Only track if there are frames before the init_frame
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float32):
            state_bwd = video_predictor.init_state(frames_backward_np)
            video_predictor.add_new_mask(inference_state=state_bwd, frame_idx=0, obj_id=obj_id, mask=init_mask_bool)
            for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(state_bwd):
                masks = (out_mask_logits > 0.0).cpu().numpy()
                original_frame_idx = init_frame_idx - out_frame_idx
                frame_segments = video_segments_backward.setdefault(original_frame_idx, {})
                for i, out_obj_id_i in enumerate(out_obj_ids):
                    if out_obj_id_i == obj_id:
                        frame_segments[out_obj_id_i] = masks[i]

    # --- 3. Combine results ---
    # The forward pass already includes the init_frame_idx, so we merge backward results into it
    video_segments_forward.update(video_segments_backward)

    return video_segments_forward


def visualize_sam_tracking(
    frames,
    video_segments,
    right_hand_tracks,
    left_hand_tracks,
    object_tracks,
    output_path,
    pause_frame_idx,
    right_hand_video_segments=None,
    left_hand_video_segments=None,
):
    """
    Visualizes detection and SAM tracking results, saving as a video.
    """
    fps = 30
    with imageio.get_writer(output_path, fps=fps, macro_block_size=1) as writer:
        for frame_idx, frame in enumerate(frames):
            img_to_write = frame.copy()
            alpha = 0.5
            mask_overlay = np.zeros_like(img_to_write, dtype=np.uint8)

            # Draw right hand mask (Green)
            if right_hand_video_segments and frame_idx in right_hand_video_segments:
                for obj_id, mask in right_hand_video_segments[frame_idx].items():
                    mask = mask.squeeze()
                    color = (0, 255, 0)  # Green
                    mask_overlay[mask] = color

            # Draw left hand mask (Red)
            if left_hand_video_segments and frame_idx in left_hand_video_segments:
                for obj_id, mask in left_hand_video_segments[frame_idx].items():
                    mask = mask.squeeze()
                    color = (0, 0, 255)  # Red
                    mask_overlay[mask] = color

            # 3. Draw object segmentation mask (yellow)
            has_mask_on_frame = frame_idx in video_segments
            if has_mask_on_frame:
                for obj_id, mask in video_segments[frame_idx].items():
                    mask = mask.squeeze()
                    color = (0, 255, 255)  # Yellow
                    mask_overlay[mask] = color

            # Apply masks only where overlay is non-zero to avoid dimming the rest
            mask_region = np.any(mask_overlay > 0, axis=-1)
            img_to_write[mask_region] = (
                img_to_write[mask_region].astype(np.float32) * (1 - alpha) +
                mask_overlay[mask_region].astype(np.float32) * alpha
            ).astype(np.uint8)

            # Get detection info for the current frame
            right_hand_info = right_hand_tracks.get(frame_idx)
            left_hand_info = left_hand_tracks.get(frame_idx)
            object_infos = object_tracks.get(frame_idx)

            # An "interaction frame" is one where at least one hand is detected.
            is_interaction_frame = right_hand_info is not None or left_hand_info is not None

            # 1, 2, 5: Draw hand kpts, hand bboxes (from kpts), and object bboxes.
            # Object bboxes are only drawn on interaction frames.
            object_infos_to_draw = object_infos if is_interaction_frame else None
            draw_detections(img_to_write, right_hand_info, left_hand_info, object_infos_to_draw)

            # Enforce target resolution before writing
            if img_to_write.shape[1] != TARGET_W or img_to_write.shape[0] != TARGET_H:
                img_to_write = cv2.resize(img_to_write, (TARGET_W, TARGET_H), interpolation=cv2.INTER_LANCZOS4)

            # Convert BGR to RGB for imageio
            frame_rgb = cv2.cvtColor(img_to_write, cv2.COLOR_BGR2RGB)
            writer.append_data(frame_rgb)

            # 4. Pause on the specific frame where the initial mask was generated.
            if frame_idx == pause_frame_idx:
                pause_duration_seconds = 2
                num_pause_frames = int(fps * pause_duration_seconds)
                for _ in range(num_pause_frames - 1):
                    writer.append_data(frame_rgb)

    print(f"SAM tracking visualization saved to {output_path}")


# ============================================================================
# Hand detection robustness: bbox cleaning utilities
# (adapted from Dyn_HaMR_new/third-party/hamer/run.py)
# ============================================================================


def compute_iou(bbox1, bbox2):
    """Compute IoU between two [x1,y1,x2,y2] bboxes."""
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    return inter / (area1 + area2 - inter)


def compute_containment_ratio(bbox_inner, bbox_outer):
    """Fraction of bbox_inner that is inside bbox_outer."""
    x1 = max(bbox_inner[0], bbox_outer[0])
    y1 = max(bbox_inner[1], bbox_outer[1])
    x2 = min(bbox_inner[2], bbox_outer[2])
    y2 = min(bbox_inner[3], bbox_outer[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0
    area_inner = (bbox_inner[2] - bbox_inner[0]) * (bbox_inner[3] - bbox_inner[1])
    return inter / area_inner if area_inner > 0 else 0.0


def detect_overlapping_bboxes(raw_data, iou_threshold=0.7, containment_threshold=0.7):
    """Remove overlapping bbox hallucinations; keep the higher-confidence detection."""
    print("\n" + "-" * 60)
    print("Step 1: Detecting overlapping bboxes")
    print("-" * 60)
    overlap_count = 0
    for frame_data in raw_data:
        frame_data['left_removed_due_to_overlap_with'] = None
        frame_data['right_removed_due_to_overlap_with'] = None
        if frame_data['left_bbox'] is not None and frame_data['right_bbox'] is not None:
            iou = compute_iou(frame_data['left_bbox'], frame_data['right_bbox'])
            left_in_right = compute_containment_ratio(frame_data['left_bbox'], frame_data['right_bbox'])
            right_in_left = compute_containment_ratio(frame_data['right_bbox'], frame_data['left_bbox'])
            max_containment = max(left_in_right, right_in_left)
            is_overlap = iou > iou_threshold or max_containment > containment_threshold
            if is_overlap:
                overlap_count += 1
                if frame_data['left_conf'] > frame_data['right_conf']:
                    frame_data['right_bbox'] = None
                    frame_data['right_keypoints'] = None
                    frame_data['right_conf'] = 0.0
                    frame_data['right_removed_due_to_overlap_with'] = 'left'
                else:
                    frame_data['left_bbox'] = None
                    frame_data['left_keypoints'] = None
                    frame_data['left_conf'] = 0.0
                    frame_data['left_removed_due_to_overlap_with'] = 'right'
    print(f"  Removed overlapping bboxes in {overlap_count} frames")
    return raw_data


def fix_handedness_swaps_by_trajectory(raw_data, position_threshold=200):
    """Fix handedness swaps detected via sudden trajectory jumps."""
    print("\n" + "-" * 60)
    print("Step 1.4a: Fixing handedness swaps via trajectory")
    print("-" * 60)
    swap_count = 0
    n = len(raw_data)
    for i in range(1, n):
        left_bbox = raw_data[i]['left_bbox']
        right_bbox = raw_data[i]['right_bbox']
        prev_left = raw_data[i - 1]['left_bbox']
        prev_right = raw_data[i - 1]['right_bbox']

        if right_bbox is not None and left_bbox is None and prev_left is not None and prev_right is not None:
            prev_left_cx = (prev_left[0] + prev_left[2]) / 2
            prev_right_cx = (prev_right[0] + prev_right[2]) / 2
            curr_right_cx = (right_bbox[0] + right_bbox[2]) / 2
            dist_to_prev_left = abs(curr_right_cx - prev_left_cx)
            dist_to_prev_right = abs(curr_right_cx - prev_right_cx)
            if dist_to_prev_left < dist_to_prev_right and dist_to_prev_right > position_threshold:
                raw_data[i]['left_bbox'] = right_bbox
                raw_data[i]['left_keypoints'] = raw_data[i]['right_keypoints']
                raw_data[i]['left_conf'] = raw_data[i]['right_conf']
                raw_data[i]['right_bbox'] = None
                raw_data[i]['right_keypoints'] = None
                raw_data[i]['right_conf'] = 0.0
                swap_count += 1

        elif left_bbox is not None and right_bbox is None and prev_right is not None and prev_left is not None:
            prev_left_cx = (prev_left[0] + prev_left[2]) / 2
            prev_right_cx = (prev_right[0] + prev_right[2]) / 2
            curr_left_cx = (left_bbox[0] + left_bbox[2]) / 2
            dist_to_prev_left = abs(curr_left_cx - prev_left_cx)
            dist_to_prev_right = abs(curr_left_cx - prev_right_cx)
            if dist_to_prev_right < dist_to_prev_left and dist_to_prev_left > position_threshold:
                raw_data[i]['right_bbox'] = left_bbox
                raw_data[i]['right_keypoints'] = raw_data[i]['left_keypoints']
                raw_data[i]['right_conf'] = raw_data[i]['left_conf']
                raw_data[i]['left_bbox'] = None
                raw_data[i]['left_keypoints'] = None
                raw_data[i]['left_conf'] = 0.0
                swap_count += 1

    print(f"  Fixed {swap_count} trajectory-based handedness swaps")
    return raw_data


def fix_handedness_inconsistencies(raw_data, original_data, context_window=5):
    """Fix handedness swaps using spatial-temporal consistency."""
    print("\n" + "-" * 60)
    print(f"Step 1.5: Fixing handedness inconsistencies (window={context_window})")
    print("-" * 60)
    swap_count = 0
    for i, frame_data in enumerate(raw_data):
        for hand_name in ['left', 'right']:
            bbox_key = f'{hand_name}_bbox'
            keyp_key = f'{hand_name}_keypoints'
            conf_key = f'{hand_name}_conf'
            other_hand = 'right' if hand_name == 'left' else 'left'
            other_bbox_key = f'{other_hand}_bbox'
            other_keyp_key = f'{other_hand}_keypoints'
            other_conf_key = f'{other_hand}_conf'

            current_bbox = frame_data[bbox_key]
            if current_bbox is None:
                continue

            same_hand_bboxes = []
            other_hand_bboxes = []
            for j in range(max(0, i - context_window), min(len(raw_data), i + context_window + 1)):
                if j == i:
                    continue
                if raw_data[j][bbox_key] is not None:
                    same_hand_bboxes.append(raw_data[j][bbox_key])
                if raw_data[j][other_bbox_key] is not None:
                    other_hand_bboxes.append(raw_data[j][other_bbox_key])

            if len(same_hand_bboxes) < 2 and len(other_hand_bboxes) < 2:
                continue

            avg_iou_same = np.mean([compute_iou(current_bbox, b) for b in same_hand_bboxes]) if same_hand_bboxes else 0.0
            avg_iou_other = np.mean([compute_iou(current_bbox, b) for b in other_hand_bboxes]) if other_hand_bboxes else 0.0

            if len(other_hand_bboxes) >= 3 and avg_iou_other > 0.5 and avg_iou_other > avg_iou_same * 1.5:
                print(f"  Frame {frame_data['frame_idx']:04d} ({hand_name}): swapping to {other_hand} "
                      f"(IoU same={avg_iou_same:.3f}, other={avg_iou_other:.3f})")
                frame_data[other_bbox_key] = current_bbox.copy()
                frame_data[other_keyp_key] = frame_data[keyp_key].copy() if frame_data[keyp_key] is not None else None
                frame_data[other_conf_key] = frame_data[conf_key]
                frame_data[bbox_key] = None
                frame_data[keyp_key] = None
                frame_data[conf_key] = 0.0
                swap_count += 1

    print(f"  Fixed {swap_count} spatial-temporal inconsistencies")
    return raw_data


def fix_handedness_swaps_frame_to_frame(raw_data, iou_threshold=0.7, max_gap=10):
    """Fix frame-to-frame handedness swaps: same bbox position but different label."""
    print("\n" + "-" * 60)
    print(f"Step 1.4: Fixing frame-to-frame handedness swaps (IoU>{iou_threshold}, gap<={max_gap})")
    print("-" * 60)
    swap_count = 0
    for i in range(1, len(raw_data)):
        curr = raw_data[i]
        for hand_name in ['left', 'right']:
            bbox_key = f'{hand_name}_bbox'
            keyp_key = f'{hand_name}_keypoints'
            conf_key = f'{hand_name}_conf'
            other_hand = 'right' if hand_name == 'left' else 'left'
            other_bbox_key = f'{other_hand}_bbox'
            other_keyp_key = f'{other_hand}_keypoints'
            other_conf_key = f'{other_hand}_conf'

            curr_bbox = curr[bbox_key]
            curr_other_bbox = curr[other_bbox_key]
            if curr_bbox is None or curr_other_bbox is not None:
                continue

            last_valid_other_bbox = None
            last_valid_other_idx = None
            for j in range(i - 1, max(0, i - max_gap - 1), -1):
                if raw_data[j][other_bbox_key] is not None:
                    last_valid_other_bbox = raw_data[j][other_bbox_key]
                    last_valid_other_idx = j
                    break

            if last_valid_other_bbox is not None:
                gap = i - last_valid_other_idx
                iou = compute_iou(curr_bbox, last_valid_other_bbox)
                if iou > iou_threshold:
                    print(f"  Frame {curr['frame_idx']:04d}: {hand_name}→{other_hand} "
                          f"(IoU={iou:.3f}, gap={gap})")
                    curr[other_bbox_key] = curr_bbox.copy()
                    curr[other_keyp_key] = curr[keyp_key].copy() if curr[keyp_key] is not None else None
                    curr[other_conf_key] = curr[conf_key]
                    curr[bbox_key] = None
                    curr[keyp_key] = None
                    curr[conf_key] = 0.0
                    swap_count += 1

    print(f"  Fixed {swap_count} frame-to-frame swaps")
    return raw_data


def clean_bbox_sequences(raw_data, img_h, img_w):
    """
    Clean hand detection bbox sequences for robustness:
      1. Remove oversized bboxes (> 50% of image area)
      2. Remove overlapping bbox hallucinations
      3. Fix handedness swaps via trajectory jump detection
      4. Fix handedness swaps via spatial-temporal consistency
      5. Fix frame-to-frame handedness swaps
      6. Interpolate short gaps (hand reappears within patience window)
      7. Remove spurious short motions
      8. Re-check overlaps after interpolation
    """
    print("\n" + "=" * 60)
    print("Cleaning hand bbox sequences for robustness")
    print("=" * 60)

    original_data = [{
        'left_bbox': f['left_bbox'].copy() if f['left_bbox'] is not None else None,
        'right_bbox': f['right_bbox'].copy() if f['right_bbox'] is not None else None,
        'left_keypoints': f['left_keypoints'].copy() if f['left_keypoints'] is not None else None,
        'right_keypoints': f['right_keypoints'].copy() if f['right_keypoints'] is not None else None,
        'left_conf': f['left_conf'],
        'right_conf': f['right_conf'],
    } for f in raw_data]

    # Step 1: Remove oversized bboxes
    print("\n" + "-" * 60)
    print("Step 1: Removing oversized bboxes (> 50% of image)")
    print("-" * 60)
    img_area = img_h * img_w
    removed_oversized = 0
    for frame_data in raw_data:
        for hand_name in ['left', 'right']:
            bbox = frame_data[f'{hand_name}_bbox']
            if bbox is not None:
                bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                if bbox_area / img_area > 0.5:
                    frame_data[f'{hand_name}_bbox'] = None
                    frame_data[f'{hand_name}_keypoints'] = None
                    frame_data[f'{hand_name}_conf'] = 0.0
                    removed_oversized += 1
    print(f"  Removed {removed_oversized} oversized bboxes")

    raw_data = detect_overlapping_bboxes(raw_data, iou_threshold=0.7)
    raw_data = fix_handedness_swaps_by_trajectory(raw_data, position_threshold=200)
    raw_data = fix_handedness_inconsistencies(raw_data, original_data, context_window=5)
    raw_data = fix_handedness_swaps_frame_to_frame(raw_data, iou_threshold=0.6)

    # Step 3: Patience mechanism with interpolation
    print("\n" + "-" * 60)
    print("Step 3: Patience mechanism with interpolation")
    print("-" * 60)
    PATIENCE_FRAMES = 25
    MAX_PATIENCE_WITHOUT_RETURN = 0
    interpolated_count = 0
    patience_applied_count = 0

    for hand_name in ['left', 'right']:
        bbox_key = f'{hand_name}_bbox'
        keyp_key = f'{hand_name}_keypoints'
        conf_key = f'{hand_name}_conf'
        i = 0
        while i < len(raw_data):
            if raw_data[i][bbox_key] is None:
                last_valid_idx = next((j for j in range(i - 1, -1, -1) if raw_data[j][bbox_key] is not None), None)
                if last_valid_idx is None:
                    i += 1
                    continue
                gap_start = i
                gap_end = next((j for j in range(i, min(i + PATIENCE_FRAMES, len(raw_data))) if raw_data[j][bbox_key] is not None), None)

                if gap_end is not None:
                    start_bbox = raw_data[last_valid_idx][bbox_key]
                    end_bbox = raw_data[gap_end][bbox_key]
                    start_cx = (start_bbox[0] + start_bbox[2]) / 2
                    start_cy = (start_bbox[1] + start_bbox[3]) / 2
                    end_cx = (end_bbox[0] + end_bbox[2]) / 2
                    end_cy = (end_bbox[1] + end_bbox[3]) / 2
                    center_dist = np.sqrt((end_cx - start_cx)**2 + (end_cy - start_cy)**2)
                    avg_width = ((start_bbox[2] - start_bbox[0]) + (end_bbox[2] - end_bbox[0])) / 2

                    if center_dist <= avg_width:
                        total_steps = gap_end - last_valid_idx
                        for j in range(gap_start, gap_end):
                            alpha = (j - last_valid_idx) / total_steps
                            raw_data[j][bbox_key] = (1 - alpha) * start_bbox + alpha * end_bbox
                            dummy_kpts = np.zeros((21, 3))
                            # Interpolated frames keep bbox continuity only; do not mark
                            # synthetic keypoints as valid detections.
                            dummy_kpts[:, 2] = 0.0
                            raw_data[j][keyp_key] = dummy_kpts
                            raw_data[j][conf_key] = 0.5
                            interpolated_count += 1
                        print(f"  {hand_name}: Interpolated frames {gap_start}-{gap_end-1} (dist={center_dist:.0f}px)")
                    i = gap_end
                else:
                    last_bbox = raw_data[last_valid_idx][bbox_key]
                    for j in range(gap_start, min(gap_start + MAX_PATIENCE_WITHOUT_RETURN, len(raw_data))):
                        raw_data[j][bbox_key] = last_bbox.copy()
                        dummy_kpts = np.zeros((21, 3))
                        dummy_kpts[:, 2] = 0.0
                        raw_data[j][keyp_key] = dummy_kpts
                        raw_data[j][conf_key] = 0.5
                        patience_applied_count += 1
                    next_valid = next((j for j in range(gap_start + MAX_PATIENCE_WITHOUT_RETURN, len(raw_data)) if raw_data[j][bbox_key] is not None),
                                      len(raw_data))
                    i = next_valid
            else:
                i += 1

    if interpolated_count:
        print(f"  Interpolated {interpolated_count} frames total")
    if patience_applied_count:
        print(f"  Patience applied to {patience_applied_count} frames")
    if not interpolated_count and not patience_applied_count:
        print("  No gaps needed filling")

    # Step 4: Remove spurious short motions
    print("\n" + "-" * 60)
    print("Step 4: Removing spurious short motions")
    print("-" * 60)
    MIN_DURATION = 30
    MIN_ABSENCE = 30

    for hand_name in ['left', 'right']:
        bbox_key = f'{hand_name}_bbox'
        keyp_key = f'{hand_name}_keypoints'
        conf_key = f'{hand_name}_conf'
        segments = []
        start_idx = None
        for i, fd in enumerate(raw_data):
            if fd[bbox_key] is not None:
                if start_idx is None:
                    start_idx = i
            else:
                if start_idx is not None:
                    segments.append((start_idx, i - 1))
                    start_idx = None
        if start_idx is not None:
            segments.append((start_idx, len(raw_data) - 1))

        removed_segs = 0
        for seg_start, seg_end in segments:
            duration = seg_end - seg_start + 1
            absence_before = seg_start
            absence_after = len(raw_data) - 1 - seg_end
            if duration < MIN_DURATION and absence_before >= MIN_ABSENCE and absence_after >= MIN_ABSENCE:
                for i in range(seg_start, seg_end + 1):
                    raw_data[i][bbox_key] = None
                    raw_data[i][keyp_key] = None
                    raw_data[i][conf_key] = 0.0
                removed_segs += 1
                print(f"  {hand_name}: Removed spurious segment frames {seg_start}-{seg_end} (duration={duration})")
        if not removed_segs:
            print(f"  {hand_name}: No spurious motions found")

    raw_data = detect_overlapping_bboxes(raw_data, iou_threshold=0.7)
    raw_data = fix_handedness_swaps_frame_to_frame(raw_data, iou_threshold=0.6)

    print("\n" + "=" * 60)
    print("Bbox cleaning complete")
    print("=" * 60)
    return raw_data


def hand_tracks_to_raw_data(right_hand_tracks, left_hand_tracks, frames, imgfiles):
    """
    Convert detect_track output dicts to the list-of-dicts format used by
    clean_bbox_sequences. Coordinates are converted from normalized to pixel space.
    Track format: [class_id, conf, xc_n, yc_n, w_n, h_n, kx0, ky0, kc0, ...]
    raw_data bbox format: np.array([x1, y1, x2, y2]) in pixels.
    raw_data keypoints format: np.array (N, 3) with [x_px, y_px, conf].
    """
    n = len(frames)
    raw_data = []
    for t in range(n):
        img_h, img_w = frames[t].shape[:2]

        def parse_track(track_info):
            if track_info is None:
                return None, None, 0.0
            conf = float(track_info[1])
            xc, yc, w, h = track_info[2], track_info[3], track_info[4], track_info[5]
            x1 = float((xc - w / 2) * img_w)
            y1 = float((yc - h / 2) * img_h)
            x2 = float((xc + w / 2) * img_w)
            y2 = float((yc + h / 2) * img_h)
            bbox = np.array([x1, y1, x2, y2], dtype=np.float32)
            kpts_data = track_info[6:]
            num_kpts = len(kpts_data) // 3
            keypoints = np.zeros((num_kpts, 3), dtype=np.float32)
            for i in range(num_kpts):
                keypoints[i, 0] = float(kpts_data[i * 3]) * img_w
                keypoints[i, 1] = float(kpts_data[i * 3 + 1]) * img_h
                keypoints[i, 2] = float(kpts_data[i * 3 + 2])
            return bbox, keypoints, conf

        rh_bbox, rh_kpts, rh_conf = parse_track(right_hand_tracks.get(t))
        lh_bbox, lh_kpts, lh_conf = parse_track(left_hand_tracks.get(t))

        raw_data.append({
            'frame_idx': t,
            'img_path': imgfiles[t] if imgfiles else None,
            'right_bbox': rh_bbox,
            'right_keypoints': rh_kpts,
            'right_conf': rh_conf,
            'left_bbox': lh_bbox,
            'left_keypoints': lh_kpts,
            'left_conf': lh_conf,
        })
    return raw_data


def raw_data_to_hand_tracks(raw_data, img_h, img_w):
    """
    Convert cleaned raw_data back to right_hand_tracks / left_hand_tracks dicts
    in the original normalized format expected by downstream code.
    """
    right_hand_tracks = {}
    left_hand_tracks = {}

    for fd in raw_data:
        t = fd['frame_idx']

        def make_track(class_id, bbox, keypoints, conf):
            xc = ((bbox[0] + bbox[2]) / 2) / img_w
            yc = ((bbox[1] + bbox[3]) / 2) / img_h
            w = (bbox[2] - bbox[0]) / img_w
            h = (bbox[3] - bbox[1]) / img_h
            kpts_flat = []
            if keypoints is not None:
                for kp in keypoints:
                    kpts_flat.extend([float(kp[0]) / img_w, float(kp[1]) / img_h, float(kp[2])])
            return [class_id, float(conf), float(xc), float(yc), float(w), float(h)] + kpts_flat

        if fd['right_bbox'] is not None:
            right_hand_tracks[t] = make_track(1, fd['right_bbox'], fd['right_keypoints'], fd['right_conf'])
        if fd['left_bbox'] is not None:
            left_hand_tracks[t] = make_track(0, fd['left_bbox'], fd['left_keypoints'], fd['left_conf'])

    return right_hand_tracks, left_hand_tracks


def get_video_fps(video_path):
    """Get FPS from a video file, falling back to 30 if unavailable."""
    try:
        reader = imageio.get_reader(str(video_path))
        fps = reader.get_meta_data().get('fps', 30)
        reader.close()
        return fps
    except Exception:
        return 30


def save_frames_as_video(frames, video_path, fps=30):
    """Save a list of BGR frames as a video file at TARGET_W × TARGET_H."""
    with imageio.get_writer(str(video_path), fps=fps, macro_block_size=1) as writer:
        for frame in frames:
            if frame.shape[1] != TARGET_W or frame.shape[0] != TARGET_H:
                frame = cv2.resize(frame, (TARGET_W, TARGET_H), interpolation=cv2.INTER_LANCZOS4)
            writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def find_hand_fully_in_frame(frames, right_hand_tracks, left_hand_tracks):
    """
    Find [crop_start, crop_end]: the first and last frames where every visible
    keypoint of every detected hand lies within normalised image bounds [0, 1].

    Frames at the start where the hand is gradually entering the scene and frames
    at the end where the hand is gradually leaving are excluded.

    Returns (0, len(frames)-1) unchanged if no qualifying frames are found.
    """
    n = len(frames)
    valid_frames = []

    for frame_idx in range(n):
        rh_info = right_hand_tracks.get(frame_idx)
        lh_info = left_hand_tracks.get(frame_idx)

        if rh_info is None and lh_info is None:
            continue

        all_in_frame = True
        for hand_info in [rh_info, lh_info]:
            if hand_info is None:
                continue
            keypoints_data = hand_info[6:]
            num_kpts = len(keypoints_data) // 3
            for i in range(num_kpts):
                kx = keypoints_data[i * 3]
                ky = keypoints_data[i * 3 + 1]
                kc = keypoints_data[i * 3 + 2]
                # Joint must be detected (kc > 0) and within normalised bounds
                if kc == 0 or not (0.0 <= kx <= 1.0 and 0.0 <= ky <= 1.0):
                    all_in_frame = False
                    break
            if not all_in_frame:
                break

        if all_in_frame:
            valid_frames.append(frame_idx)

    if not valid_frames:
        print("Warning: No frames found where all hand keypoints are within image bounds. Using full video.")
        return 0, n - 1

    return valid_frames[0], valid_frames[-1]


def build_continuous_hand_keypoints(hand_tracks, frames, num_joints=21):
    """
    Build temporally continuous keypoints for each tracked frame.
    For each joint:
      - Use detected keypoints when available (conf > 0 and in-bounds)
      - Fill missing values by linear interpolation over time
      - Use edge-value carry for start/end gaps (via np.interp behaviour)
      - If a joint has no valid observation at all, fall back to bbox center
    Returns: dict[frame_idx] -> list[[x_px, y_px], ...] (length=num_joints)
    """
    if not hand_tracks:
        return {}

    frame_ids = sorted(hand_tracks.keys())
    n = len(frame_ids)
    xs = np.full((n, num_joints), np.nan, dtype=np.float32)
    ys = np.full((n, num_joints), np.nan, dtype=np.float32)
    center_x = np.zeros(n, dtype=np.float32)
    center_y = np.zeros(n, dtype=np.float32)
    widths = np.zeros(n, dtype=np.float32)
    heights = np.zeros(n, dtype=np.float32)

    for idx, frame_idx in enumerate(frame_ids):
        track_info = hand_tracks[frame_idx]
        img_h, img_w = frames[frame_idx].shape[:2]
        xc, yc, w, h = track_info[2], track_info[3], track_info[4], track_info[5]
        center_x[idx] = float(xc * img_w)
        center_y[idx] = float(yc * img_h)
        widths[idx] = float(img_w)
        heights[idx] = float(img_h)

        keypoints_data = track_info[6:]
        parsed_joints = min(num_joints, len(keypoints_data) // 3)
        for j in range(parsed_joints):
            kx = float(keypoints_data[j * 3])
            ky = float(keypoints_data[j * 3 + 1])
            kc = float(keypoints_data[j * 3 + 2])
            if kc > 0 and 0.0 <= kx <= 1.0 and 0.0 <= ky <= 1.0:
                xs[idx, j] = kx * img_w
                ys[idx, j] = ky * img_h

    t = np.arange(n, dtype=np.float32)
    for j in range(num_joints):
        valid_x = ~np.isnan(xs[:, j])
        valid_y = ~np.isnan(ys[:, j])
        valid = valid_x & valid_y
        if np.any(valid):
            valid_idx = np.where(valid)[0].astype(np.float32)
            xs[:, j] = np.interp(t, valid_idx, xs[valid, j])
            ys[:, j] = np.interp(t, valid_idx, ys[valid, j])
        else:
            # No valid observation for this joint across the clip: use bbox center.
            xs[:, j] = center_x
            ys[:, j] = center_y

    # Clamp to image bounds per frame.
    xs = np.clip(xs, 0.0, (widths - 1.0)[:, None])
    ys = np.clip(ys, 0.0, (heights - 1.0)[:, None])

    continuous = {}
    for idx, frame_idx in enumerate(frame_ids):
        continuous[frame_idx] = [[float(xs[idx, j]), float(ys[idx, j])] for j in range(num_joints)]
    return continuous


def write_continuous_keypoints_back_to_tracks(hand_tracks, continuous_keypoints, frames):
    """Overwrite track keypoints with dense, continuous 21-joint trajectories."""
    for frame_idx, kpts in continuous_keypoints.items():
        if frame_idx not in hand_tracks:
            continue
        track_info = hand_tracks[frame_idx]
        if len(track_info) < 6:
            continue
        img_h, img_w = frames[frame_idx].shape[:2]
        kpts_flat = []
        for x_px, y_px in kpts:
            x_n = float(np.clip(x_px / img_w, 0.0, 1.0))
            y_n = float(np.clip(y_px / img_h, 0.0, 1.0))
            kpts_flat.extend([x_n, y_n, 1.0])
        hand_tracks[frame_idx] = track_info[:6] + kpts_flat


def detect_track(hand_det_model, imgfiles, conf=0.3, tracker='configs/bytetrack.yaml'):
    # Run
    right_hand_tracks = {}
    left_hand_tracks = {}
    object_tracks = {}

    for t, imgpath in enumerate(imgfiles):
        img_cv2 = cv2.imread(imgpath)

        # --- Detection ---
        with torch.no_grad():
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            with autocast(device_type=device):
                results = hand_det_model.track(img_cv2, conf=conf, tracker=tracker, persist=True, verbose=False)

                if results[0] is None:
                    continue

                boxes = results[0].boxes.xywhn.cpu().numpy()  # normalized to 0-1
                confs = results[0].boxes.conf.cpu().numpy()
                class_ids = results[0].boxes.cls.cpu().numpy()
                keypoints = results[0].keypoints.xyn.cpu().numpy() if results[0].keypoints is not None else None
                keypoints_conf = results[0].keypoints.conf.cpu().numpy() if results[0].keypoints is not None else None

                # Find best detection for each class
                best_right_hand = None
                best_left_hand = None
                current_objects = []
                max_conf_right = 0.0
                max_conf_left = 0.0

                for idx, box in enumerate(boxes):
                    class_id = int(class_ids[idx])
                    confidence = confs[idx]

                    if class_id == 0:  # Left Hand
                        if confidence > max_conf_left and keypoints is not None:
                            max_conf_left = confidence
                            best_left_hand = (box, keypoints[idx], keypoints_conf[idx])
                    elif class_id == 1:  # Right Hand
                        if confidence > max_conf_right and keypoints is not None:
                            max_conf_right = confidence
                            best_right_hand = (box, keypoints[idx], keypoints_conf[idx])
                    elif class_id == 2:  # Object
                        current_objects.append(box.tolist() + [confidence])

                # Process and store best hand detections for the frame
                if best_left_hand:
                    box, kpts, kpts_conf = best_left_hand
                    kpts_flat = [coord for kpt, c in zip(kpts, kpts_conf) for coord in [float(kpt[0]), float(kpt[1]), 1.0 if c > 0.5 else 0.0]]
                    left_hand_tracks[t] = [0, float(max_conf_left)] + box.tolist() + kpts_flat

                if best_right_hand:
                    box, kpts, kpts_conf = best_right_hand
                    kpts_flat = [coord for kpt, c in zip(kpts, kpts_conf) for coord in [float(kpt[0]), float(kpt[1]), 1.0 if c > 0.5 else 0.0]]
                    right_hand_tracks[t] = [1, float(max_conf_right)] + box.tolist() + kpts_flat

                if current_objects:
                    object_tracks[t] = current_objects

    hand_det_model.predictor.trackers[0].reset()

    return right_hand_tracks, left_hand_tracks, object_tracks


def generate_and_track_hand_mask(hand_tracks, frames, sam_predictor, sam_video_predictor, hand_name):
    """
    Generates an initial mask for a hand using keypoints and tracks it through the video.
    """
    if not hand_tracks:
        return {}, -1

    # --- 1. Find the best candidate frame to initialize SAM ---
    # Criteria: 1. Most visible keypoints, 2. Highest detection confidence.
    candidate_frames = []
    for frame_idx, track_info in hand_tracks.items():
        conf = track_info[1]  # Confidence is the second element
        keypoints_data = track_info[6:]
        num_keypoints = len(keypoints_data) // 3
        visible_kpts_count = sum(1 for i in range(num_keypoints) if keypoints_data[i * 3 + 2] > 0)
        candidate_frames.append({'frame_idx': frame_idx, 'conf': conf, 'visible_kpts': visible_kpts_count})

    if not candidate_frames:
        print(f"No valid hand detections found for {hand_name} to initialize tracking.")
        return {}, -1

    # Sort by visible keypoints (desc), then by confidence (desc)
    candidate_frames.sort(key=lambda x: (x['visible_kpts'], x['conf']), reverse=True)

    best_frame_idx = candidate_frames[0]['frame_idx']

    # --- 2. Get keypoints from the best frame ---
    init_hand_info = hand_tracks[best_frame_idx]
    keypoints_data = init_hand_info[6:]  # kpts start from index 6
    num_keypoints = len(keypoints_data) // 3

    img_h, img_w = frames[best_frame_idx].shape[:2]

    points = []
    for i in range(num_keypoints):
        kpt_x = keypoints_data[i * 3] * img_w
        kpt_y = keypoints_data[i * 3 + 1] * img_h
        kpt_conf = keypoints_data[i * 3 + 2]  # This is 0 or 1, not confidence

        if kpt_conf > 0:
            points.append([kpt_x, kpt_y])

    if not points:
        print(f"No confident keypoints found for {hand_name} in frame {best_frame_idx}.")
        return {}, -1

    points = np.array(points)
    point_labels = np.ones(len(points))  # All are positive points

    # --- 3. Generate initial mask with SAM using batch interface for a single image ---
    image_rgb = cv2.cvtColor(frames[best_frame_idx], cv2.COLOR_BGR2RGB)
    sam_predictor.set_image_batch([image_rgb])

    points_batch = [points]
    point_labels_batch = [point_labels]

    # Use the batch prediction with point inputs
    masks_batch, _, _ = sam_predictor.predict_batch(points_batch, point_labels_batch, box_batch=None, multimask_output=False)

    if masks_batch is None or len(masks_batch) == 0:
        print(f"SAM failed to generate a mask for {hand_name} at frame {best_frame_idx}.")
        return {}, -1

    initial_mask = masks_batch[0][0]  # first image, first mask
    initial_mask = remove_small_regions(initial_mask)

    if not np.any(initial_mask):
        print(f"SAM generated an empty mask for {hand_name} at frame {best_frame_idx}.")
        return {}, -1

    # --- 4. Track the mask with video predictor ---
    print(f"Generated initial mask for {hand_name} at frame {best_frame_idx}. Starting video tracking...")
    video_segments = sam_video_tracking(sam_video_predictor, frames, initial_mask, best_frame_idx)

    return video_segments, best_frame_idx


def process_video_with_visualization(video_path, hand_det_model, sam_predictor, sam_video_predictor, conf,
                                     output_data_dir, visualize=True):
    """
    Process a single video, generate an initial mask, and track it through the video.

    Args:
        output_data_dir: root for annotations, visualizations, and downstream artifacts.
    """
    output_data_dir = Path(output_data_dir)
    video_name = Path(video_path).stem

    # Keep temporary frames within this video's output directory.
    temp_frames_folder = output_data_dir / video_name / '.tmp_frames'
    temp_frames_folder.mkdir(parents=True, exist_ok=True)

    try:
        # Extract frames into memory (with automatic rotation correction)
        frames = read_video_frames(video_path)

        if not frames:
            return video_name, 0, False, None, False

        # Create a temporary list of image paths for detect_track
        imgfiles = []
        for i, frame in enumerate(frames):
            img_path = temp_frames_folder / f'{i:06d}.jpg'
            cv2.imwrite(str(img_path), frame)
            imgfiles.append(str(img_path))

        # Perform hand detection and tracking
        right_hand_tracks, left_hand_tracks, object_tracks = detect_track(
            hand_det_model,
            imgfiles,
            conf=conf,
            tracker='configs/bytetrack.yaml',
        )

        # --- Clean hand bbox sequences for robustness ---
        img_h_0, img_w_0 = frames[0].shape[:2]
        raw_data = hand_tracks_to_raw_data(right_hand_tracks, left_hand_tracks, frames, imgfiles)
        raw_data = clean_bbox_sequences(raw_data, img_h_0, img_w_0)
        right_hand_tracks, left_hand_tracks = raw_data_to_hand_tracks(raw_data, img_h_0, img_w_0)

        # --- Load initial object mask from first_mask.png (corresponds to ORIGINAL frame 0) ---
        # This MUST happen before cropping, because first_mask.png was authored against the
        # untrimmed video's frame 0. We track the object on the full sequence first, then
        # crop the resulting per-frame masks along with everything else further below.
        first_mask_path = annotation_dir_for_video(output_data_dir, video_path) / "first_mask.png"
        if not first_mask_path.exists():
            print(f"Error: first_mask.png not found at {first_mask_path}. Please create the mask first.")
            return video_name, 0, False, None, True

        first_mask_img = cv2.imread(str(first_mask_path), cv2.IMREAD_UNCHANGED)
        if first_mask_img is None:
            print(f"Error: Could not read {first_mask_path}.")
            return video_name, 0, False, None, True

        if first_mask_img.ndim == 3 and first_mask_img.shape[2] == 4:
            initial_mask = first_mask_img[:, :, 3] > 0
        elif first_mask_img.ndim == 3:
            initial_mask = np.any(first_mask_img > 0, axis=2)
        else:
            initial_mask = first_mask_img > 0

        # Resize mask to match original frame resolution if they differ
        mask_h, mask_w = initial_mask.shape[:2]
        frame_h, frame_w = frames[0].shape[:2]
        if mask_h != frame_h or mask_w != frame_w:
            print(f"Warning: first_mask.png is {mask_w}x{mask_h} but video frames are "
                  f"{frame_w}x{frame_h}. Resizing mask to match.")
            initial_mask = cv2.resize(initial_mask.astype(np.uint8), (frame_w, frame_h),
                                      interpolation=cv2.INTER_NEAREST) > 0

        initial_mask = remove_small_regions(initial_mask)
        if not np.any(initial_mask):
            print(f"Error: first_mask.png is empty for {video_name}.")
            return video_name, 0, False, None, True

        # --- Optionally save visualization of the loaded mask on frame 0 ---
        mask_vis_folder = None
        if visualize:
            mask_vis_folder = visualization_dir_for_video(output_data_dir, video_path)
            mask_vis_folder.mkdir(parents=True, exist_ok=True)
            vis_image1 = frames[0].copy()
            color1 = np.array([0, 0, 255])
            vis_image1[initial_mask] = (vis_image1[initial_mask] * 0.5 + color1 * 0.5).astype(np.uint8)
            mask_vis_path = mask_vis_folder / "initial_mask.jpg"
            cv2.imwrite(str(mask_vis_path), vis_image1)

        # --- Track object mask through the full (uncropped) video starting from frame 0 ---
        print(f"Loaded initial object mask from {first_mask_path}. "
              f"Starting video tracking from original frame 0 (before cropping)...")
        video_segments = sam_video_tracking(sam_video_predictor, frames, initial_mask, 0)

        # --- Crop to frames where hand is fully within image bounds ---
        original_fps = get_video_fps(video_path)
        sam3d_video_path = layout_for_video(output_data_dir, video_name).video_mp4
        crop_start, crop_end = find_hand_fully_in_frame(frames, right_hand_tracks, left_hand_tracks)
        if crop_start > 0 or crop_end < len(frames) - 1:
            print(f"\nCropping video '{video_name}': keeping frames {crop_start}–{crop_end} "
                  f"(out of {len(frames)} total, {crop_end - crop_start + 1} kept)")

            frames = frames[crop_start:crop_end + 1]

            # Re-index every track dict so indices start from 0 in the cropped video
            right_hand_tracks = {k - crop_start: v for k, v in right_hand_tracks.items() if crop_start <= k <= crop_end}
            left_hand_tracks = {k - crop_start: v for k, v in left_hand_tracks.items() if crop_start <= k <= crop_end}
            object_tracks = {k - crop_start: v for k, v in object_tracks.items() if crop_start <= k <= crop_end}
            # Re-index object SAM segments the same way so they stay aligned with frames
            video_segments = {k - crop_start: v for k, v in video_segments.items() if crop_start <= k <= crop_end}

        else:
            print(f"No cropping needed: hand is fully in frame throughout all {len(frames)} frames.")

        # Always encode the normalized frames so the video, masks, keypoints, and
        # RGB outputs share the same 1920 x 1080 coordinate system.
        save_frames_as_video(frames, sam3d_video_path, fps=original_fps)
        print(f"Wrote normalized output video: {sam3d_video_path}")
        print(f"Preserved original source video: {video_path}")

        # The "initial" frame for visualization purposes (post-crop coordinates).
        # Used by visualize_sam_tracking to pause on the frame where tracking was anchored.
        # If the original frame 0 is still in the cropped clip it sits at index -crop_start
        # (i.e. 0 when crop_start == 0, otherwise it's been removed and we just pause on 0).
        first_object_frame_idx = 0

        # --- Use refined hand bounding boxes directly from tracks ---
        rh_bboxes = {}
        for frame_idx, track_info in right_hand_tracks.items():
            if len(track_info) >= 6:
                img_h, img_w = frames[frame_idx].shape[:2]
                xc, yc, w, h = track_info[2], track_info[3], track_info[4], track_info[5]
                x1 = float((xc - w / 2) * img_w)
                y1 = float((yc - h / 2) * img_h)
                x2 = float((xc + w / 2) * img_w)
                y2 = float((yc + h / 2) * img_h)
                # Clamp to image bounds to avoid downstream out-of-range artifacts.
                x1 = max(0.0, min(x1, float(img_w - 1)))
                y1 = max(0.0, min(y1, float(img_h - 1)))
                x2 = max(0.0, min(x2, float(img_w - 1)))
                y2 = max(0.0, min(y2, float(img_h - 1)))
                if x2 > x1 and y2 > y1:
                    rh_bboxes[frame_idx] = [x1, y1, x2, y2]

        lh_bboxes = {}
        for frame_idx, track_info in left_hand_tracks.items():
            if len(track_info) >= 6:
                img_h, img_w = frames[frame_idx].shape[:2]
                xc, yc, w, h = track_info[2], track_info[3], track_info[4], track_info[5]
                x1 = float((xc - w / 2) * img_w)
                y1 = float((yc - h / 2) * img_h)
                x2 = float((xc + w / 2) * img_w)
                y2 = float((yc + h / 2) * img_h)
                x1 = max(0.0, min(x1, float(img_w - 1)))
                y1 = max(0.0, min(y1, float(img_h - 1)))
                x2 = max(0.0, min(x2, float(img_w - 1)))
                y2 = max(0.0, min(y2, float(img_h - 1)))
                if x2 > x1 and y2 > y1:
                    lh_bboxes[frame_idx] = [x1, y1, x2, y2]

        # --- Build dense, continuous 21-joint keypoint trajectories ---
        rh_keypoints = build_continuous_hand_keypoints(right_hand_tracks, frames, num_joints=21)
        lh_keypoints = build_continuous_hand_keypoints(left_hand_tracks, frames, num_joints=21)
        # Keep downstream modules consistent with exported trajectories.
        write_continuous_keypoints_back_to_tracks(right_hand_tracks, rh_keypoints, frames)
        write_continuous_keypoints_back_to_tracks(left_hand_tracks, lh_keypoints, frames)

        # --- Track Hands with SAM ---
        right_hand_video_segments, right_hand_pause_idx = generate_and_track_hand_mask(right_hand_tracks, frames, sam_predictor, sam_video_predictor,
                                                                                       "Right Hand")
        left_hand_video_segments, left_hand_pause_idx = generate_and_track_hand_mask(left_hand_tracks, frames, sam_predictor, sam_video_predictor, "Left Hand")

        # Save the tracked visualization only when requested.
        output_video_path = None
        if visualize:
            vis_output_folder = visualization_dir_for_video(output_data_dir, video_path)
            vis_output_folder.mkdir(parents=True, exist_ok=True)
            output_video_path = str(vis_output_folder / "track.mp4")
            visualize_sam_tracking(
                frames,
                video_segments,
                right_hand_tracks,
                left_hand_tracks,
                object_tracks,
                output_video_path,
                first_object_frame_idx,
                right_hand_video_segments=right_hand_video_segments,
                left_hand_video_segments=left_hand_video_segments,
            )

        # --- Save all RGB frames ---
        layout = layout_for_video(output_data_dir, video_name)
        output_rgb_folder = layout.frames_rgb_dir
        output_rgb_folder.mkdir(parents=True, exist_ok=True)
        for frame_idx, frame in enumerate(frames):
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            imageio.imwrite(output_rgb_folder / f'{frame_idx}.png', frame_rgb)

        # --- Save hand bounding boxes ---
        # JSON keys must be strings; use str(frame_idx) explicitly so downstream
        # code can rely on a consistent key type after json.load().
        output_folder = output_data_dir / video_name
        if rh_bboxes:
            layout.bbox_dir.mkdir(parents=True, exist_ok=True)
            with open(layout.bbox_json('right'), 'w') as f:
                json.dump({str(k): v for k, v in rh_bboxes.items()}, f, indent=4)
        if lh_bboxes:
            layout.bbox_dir.mkdir(parents=True, exist_ok=True)
            with open(layout.bbox_json('left'), 'w') as f:
                json.dump({str(k): v for k, v in lh_bboxes.items()}, f, indent=4)

        # --- Save hand keypoints ---
        if rh_keypoints:
            layout.keypoints_dir.mkdir(parents=True, exist_ok=True)
            with open(layout.keypoints_json('right'), 'w') as f:
                json.dump({str(k): v for k, v in rh_keypoints.items()}, f, indent=4)
        if lh_keypoints:
            layout.keypoints_dir.mkdir(parents=True, exist_ok=True)
            with open(layout.keypoints_json('left'), 'w') as f:
                json.dump({str(k): v for k, v in lh_keypoints.items()}, f, indent=4)

        # --- Save RGBA masks for OBJECT ---
        output_seg_mask_folder = layout.object_masks_dir
        output_seg_mask_folder.mkdir(parents=True, exist_ok=True)
        for frame_idx, segments in video_segments.items():
            if 1 in segments:
                frame_mask = segments[1]
                frame_rgba = cv2.cvtColor(cv2.cvtColor(frames[frame_idx], cv2.COLOR_BGR2RGB), cv2.COLOR_RGB2RGBA)
                rgba_cutout = np.zeros_like(frame_rgba, dtype=np.uint8)
                rgba_cutout = np.where(frame_mask.squeeze()[..., None], frame_rgba, rgba_cutout)
                imageio.imwrite(output_seg_mask_folder / f'{frame_idx}.png', rgba_cutout)

        # --- Save RGBA masks for RIGHT HAND ---
        if right_hand_video_segments:
            rh_mask_folder = layout.rh_masks_dir
            rh_mask_folder.mkdir(parents=True, exist_ok=True)
            for frame_idx, segments in right_hand_video_segments.items():
                if 1 in segments:
                    frame_mask = segments[1]
                    frame_rgba = cv2.cvtColor(cv2.cvtColor(frames[frame_idx], cv2.COLOR_BGR2RGB), cv2.COLOR_RGB2RGBA)
                    rgba_cutout = np.zeros_like(frame_rgba, dtype=np.uint8)
                    rgba_cutout = np.where(frame_mask.squeeze()[..., None], frame_rgba, rgba_cutout)
                    imageio.imwrite(rh_mask_folder / f'{frame_idx}.png', rgba_cutout)

        # --- Save RGBA masks for LEFT HAND ---
        if left_hand_video_segments:
            lh_mask_folder = layout.lh_masks_dir
            lh_mask_folder.mkdir(parents=True, exist_ok=True)
            for frame_idx, segments in left_hand_video_segments.items():
                if 1 in segments:
                    frame_mask = segments[1]
                    frame_rgba = cv2.cvtColor(cv2.cvtColor(frames[frame_idx], cv2.COLOR_BGR2RGB), cv2.COLOR_RGB2RGBA)
                    rgba_cutout = np.zeros_like(frame_rgba, dtype=np.uint8)
                    rgba_cutout = np.where(frame_mask.squeeze()[..., None], frame_rgba, rgba_cutout)
                    imageio.imwrite(lh_mask_folder / f'{frame_idx}.png', rgba_cutout)

        # Also, save the old visualization for reference on the first frame if it has a mask
        if visualize and 0 in video_segments and 1 in video_segments[0]:
            first_frame_mask = video_segments[0][1]
            vis_image_ff = frames[0].copy()
            color_ff = np.array([255, 0, 0])  # Blue
            vis_image_ff[first_frame_mask.squeeze()] = (vis_image_ff[first_frame_mask.squeeze()] * 0.5 + color_ff * 0.5).astype(np.uint8)
            mask_vis_path_ff = mask_vis_folder / "first_frame_tracked_mask.jpg"
            cv2.imwrite(str(mask_vis_path_ff), vis_image_ff)
        elif visualize:
            print(f"Warning: No mask propagated to the first frame for video {video_name}")

        return video_name, len(video_segments), True, output_video_path, True

    except Exception as e:
        print(f"Error processing video: {video_name} - {str(e)}")
        return video_name, 0, False, None, False

    finally:
        # Clean up temporary frames folder
        if temp_frames_folder.exists():
            shutil.rmtree(temp_frames_folder)


def process_video_worker(args):
    """
    Worker function for the multiprocessing pool. Loads model once per worker.
    """
    global _WORKER_MODELS
    video_path, hand_det_model_path, conf, output_data_dir, visualize = args

    # Load model if not already in the worker's cache
    if 'hand_det_model' not in _WORKER_MODELS:
        _WORKER_MODELS['hand_det_model'] = YOLO(hand_det_model_path)
    hand_det_model = _WORKER_MODELS['hand_det_model']

    if 'sam_predictor' not in _WORKER_MODELS:
        sam2_checkpoint = os.environ.get("CHOIR_SAM2_CHECKPOINT", str(DEFAULT_SAM2_CHECKPOINT))
        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=True)
        _WORKER_MODELS['sam_predictor'] = SAM2ImagePredictor(sam2, mask_threshold=0.2)
    sam_predictor = _WORKER_MODELS['sam_predictor']

    if 'sam_video_predictor' not in _WORKER_MODELS:
        sam2_checkpoint = os.environ.get("CHOIR_SAM2_CHECKPOINT", str(DEFAULT_SAM2_CHECKPOINT))
        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        _WORKER_MODELS['sam_video_predictor'] = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
    sam_video_predictor = _WORKER_MODELS['sam_video_predictor']

    video_name, num_frames, is_successful, output_video_path, success = process_video_with_visualization(video_path, hand_det_model, sam_predictor,
                                                                                                         sam_video_predictor, conf, output_data_dir,
                                                                                                         visualize)

    if success:
        return {'video_name': video_name, 'num_frames': num_frames, 'is_successful': is_successful, 'output_video_path': output_video_path, 'success': True}
    else:
        return {'video_name': video_name, 'num_frames': 0, 'is_successful': False, 'output_video_path': None, 'success': False}


def worker_main(gpu_id, video_chunk, args, shared_results):
    """
    Main function for a worker process assigned to a specific GPU.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print(f"Worker process for GPU {gpu_id} started, processing {len(video_chunk)} videos.")

    # Prepare arguments for each video in this GPU's chunk
    video_args_chunk = [
        (str(video), args.model, args.conf, args.output_dir, args.visualize)
        for video in video_chunk
    ]

    results = []
    # Use a multiprocessing pool to process videos on this GPU
    with mp.Pool(processes=args.num_workers) as pool:
        for result in tqdm(pool.imap_unordered(process_video_worker, video_args_chunk), total=len(video_args_chunk), desc=f"GPU {gpu_id}", position=gpu_id):
            results.append(result)

    shared_results.extend(results)
    print(f"Worker for GPU {gpu_id} finished.")


def load_input_videos(data_dir, output_data_dir, video_ids=None):
    """
    Loads all mp4 videos from `data_dir`.
    Pre-creates `output_data_dir/{video_id}/` for each video so downstream tools
    can find the final video at `{output_data_dir}/{video_id}/{video_id}.mp4`.
    The actual video file (original or cropped) is written there later during
    processing. Returns the original source paths for processing.
    """
    data_dir = Path(data_dir)
    output_data_dir = Path(output_data_dir)

    video_files = discover_input_videos(data_dir, video_ids)
    if not video_files:
        print(f"No mp4 files found in {data_dir}")
        return []

    print(f"Found {len(video_files)} selected video(s) in {data_dir}: {[v.name for v in video_files]}")

    for video_path in video_files:
        video_id = video_path.stem
        (output_data_dir / video_id).mkdir(parents=True, exist_ok=True)

    return video_files


def build_arg_parser():
    import argparse

    parser = argparse.ArgumentParser(description='Process custom videos with hand detection and SAM tracking')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of parallel workers per GPU (default: 4)')
    parser.add_argument('--model', type=str, default='models/wilor_hand_detector.pt', help='Path to the hand-only detection model')
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold for hand detection (default: 0.5)')
    parser.add_argument('--video_id', type=str, nargs='+', default=None,
                        help='One or more video IDs (stems) to process. If omitted, all videos in --data_dir are processed.')
    parser.add_argument('--data_dir', type=str, default=DEFAULT_DATA_DIR,
                        help=f'Directory containing flat {{video_id}}.mp4 inputs '
                             f'(default: {DEFAULT_DATA_DIR})')
    parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_DATA_DIR,
                        help=f'CHOIR output/ root for per-video inputs/ and '
                             f'stage1/visualizations/ (default: {DEFAULT_OUTPUT_DATA_DIR})')
    parser.add_argument('--no_visualize', dest='visualize', action='store_false',
                        help='Skip visualization images and videos without changing reconstruction outputs.')
    parser.set_defaults(visualize=True)
    return parser


def main():
    args = build_arg_parser().parse_args()

    # Set multiprocessing start method to spawn (required for CUDA)
    import multiprocessing as mp
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set, skip

    print("=" * 80)
    print(f"Data source: {args.data_dir}")
    print(f"Output dir : {args.output_dir}")
    print("=" * 80)
    requested = set(args.video_id) if args.video_id else None
    videos_to_process = load_input_videos(args.data_dir, args.output_dir, video_ids=requested)
    if requested:
        missing = requested - {v.stem for v in videos_to_process}
        if missing:
            print(f"Warning: the following requested video IDs were not found in {args.data_dir}: {sorted(missing)}")
        if not videos_to_process:
            print("No matching videos found. Exiting.")
            return
        print(f"Processing {len(videos_to_process)} selected video(s): {[v.name for v in videos_to_process]}")
    elif not videos_to_process:
        print(f"No videos found in {args.data_dir}. Please add mp4 files to that directory.")
        return

    hand_det_model_path = args.model
    print(f"Using model: {hand_det_model_path}")

    # --- Multi-GPU Processing Setup ---
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        print("Warning: No GPUs found. Running on CPU, which might be very slow.")
        num_gpus = 1  # Treat as 1 device for CPU-only execution

    print(f"Found {num_gpus} devices. Distributing {len(videos_to_process)} videos...")

    # Split videos into chunks for each GPU
    video_chunks = [[] for _ in range(num_gpus)]
    for i, video_file in enumerate(videos_to_process):
        video_chunks[i % num_gpus].append(video_file)

    with mp.Manager() as manager:
        shared_results = manager.list()
        processes = []
        for gpu_id in range(num_gpus):
            if not video_chunks[gpu_id]:
                continue  # Skip GPU if no videos are assigned
            p = mp.Process(target=worker_main, args=(gpu_id, video_chunks[gpu_id], args, shared_results))
            processes.append(p)
            p.start()

        for p in processes:
            p.join()

        print("\nAggregating results from all workers...")
        results = list(shared_results)

    # Filter successful results
    successful_results = [r for r in results if r['success']]

    # Print individual results
    print("\n" + "=" * 80)
    print("Test Results:")
    print("=" * 80)
    for result in successful_results:
        status = "OK" if result['is_successful'] else "Low detections"
        base_info = f"  Video: {result['video_name']}, detected frames: {result['num_frames']}, status: {status}"
        if 'output_video_path' in result and result['output_video_path']:
            print(f"{base_info}, visualization saved to: {result['output_video_path']}")
        else:
            print(base_info)

    # Statistics results
    print("\n" + "=" * 80)
    print("Statistics")
    print("=" * 80)

    total_videos = len(successful_results)
    success_count = sum(1 for r in successful_results if r['is_successful'])
    success_rate = (success_count / total_videos * 100) if total_videos > 0 else 0

    print(f"  Total videos processed: {total_videos}")
    print(f"  Videos with sufficient detections (>= 30 frames): {success_count}")
    print(f"  Videos with low detections (< 30 frames): {total_videos - success_count}")
    print(f"  Success rate: {success_rate:.2f}%")

    if args.visualize:
        print(f"Per-video visualizations saved under: {args.output_dir}/<video_id>/visualizations")

    print("=" * 80)


if __name__ == "__main__":
    main()
