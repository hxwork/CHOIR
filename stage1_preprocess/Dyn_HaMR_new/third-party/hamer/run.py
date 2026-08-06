"""
Improved HaMeR pipeline with 3-pass architecture using YOLO hand detector:
Pass 1: Extract all raw bboxes (YOLO → Hand bboxes directly)
Pass 2: Clean bbox sequences globally (detect anomalies, smooth, interpolate)
Pass 3: Run HaMeR on cleaned bboxes

This uses YOLO hand detector (like WiLoR) instead of Detectron2 + ViTPose.
"""

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import hamer
import imageio
import numpy as np
import torch
from hamer.configs import CACHE_DIR_HAMER
from hamer.datasets.vitdet_dataset import (DEFAULT_MEAN, DEFAULT_STD, ViTDetDataset)
from hamer.models import HAMER, download_models, load_hamer
from hamer.utils import recursive_to
from hamer.utils.geometry import aa_to_rotmat, perspective_projection
from hamer.utils.renderer import Renderer, cam_crop_to_full
from tqdm import tqdm
from ultralytics import YOLO

LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)

openpose_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
gt_indices = openpose_indices


def compute_bbox_from_keypoints(keypoints, img_shape, scale_factor=1.0, max_size_ratio=0.4):
    """Compute tight bbox from keypoints. Scaling is applied later when creating HaMeR input patches."""
    valid = keypoints[:, 2] > 0.5
    if valid.sum() < 10:
        return None

    valid_kp = keypoints[valid, :2]
    x_min, y_min = valid_kp.min(axis=0)
    x_max, y_max = valid_kp.max(axis=0)

    width = x_max - x_min
    height = y_max - y_min
    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2

    width *= scale_factor
    height *= scale_factor

    img_h, img_w = img_shape[:2]
    max_width = img_w * max_size_ratio
    max_height = img_h * max_size_ratio

    width = min(width, max_width)
    height = min(height, max_height)
    size = max(width, height)

    x_min = cx - size / 2
    y_min = cy - size / 2
    x_max = cx + size / 2
    y_max = cy + size / 2

    return np.array([x_min, y_min, x_max, y_max])


def compute_iou(bbox1, bbox2):
    """Compute IoU between two bboxes."""
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)
    bbox1_area = (x1_max - x1_min) * (y1_max - y1_min)
    bbox2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = bbox1_area + bbox2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0


def compute_containment_ratio(bbox1, bbox2):
    """
    Compute how much bbox1 is contained within bbox2.
    Returns the ratio of bbox1's area that overlaps with bbox2.
    """
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)
    bbox1_area = (x1_max - x1_min) * (y1_max - y1_min)

    return inter_area / bbox1_area if bbox1_area > 0 else 0


def create_video_from_images(image_folder, output_video_path, fps=30):
    """Create MP4 video from images in a folder using imageio (same as src_cam video)."""
    image_folder = Path(image_folder)
    if not image_folder.exists():
        print(f"Warning: {image_folder} does not exist, skipping video creation")
        return

    # Get all jpg images sorted by filename
    image_files = sorted(list(image_folder.glob('*.jpg')))
    if len(image_files) == 0:
        print(f"Warning: No images found in {image_folder}, skipping video creation")
        return

    print(f"Creating video: {output_video_path}")

    # Load all images
    frames = []
    for img_file in tqdm(image_files, desc=f"  Loading frames"):
        img = imageio.imread(str(img_file))
        if img is not None:
            frames.append(img)

    if len(frames) == 0:
        print(f"Warning: Could not load any images from {image_folder}")
        return

    # Write video using imageio (same method as src_cam video generation)
    imageio.mimwrite(str(output_video_path), frames, fps=fps)
    print(f"  Video saved: {output_video_path} ({len(frames)} frames @ {fps} fps)")


# ============================================================================
# PASS 1: Extract all raw bboxes using YOLO hand detector
# ============================================================================


def extract_raw_bboxes_from_keypoints_json(img_paths, sequence_name, hand_type='right', vis_dir=None):
    """
    Pass 1: Extract raw hand bboxes from a pre-existing keypoints JSON file.

    Args:
        img_paths: List of image paths under stage1/dynhamr/images/
        sequence_name: Sequence name (e.g., 'hold_ABF14_ho3d')
        hand_type: 'left' or 'right' - which hand to process
        vis_dir: Optional directory for visualizations

    Returns:
        raw_data: List of dicts for each frame with bbox data
    """
    print("\n" + "=" * 80)
    print(f"PASS 1: Extracting hand bboxes from keypoints JSON ({hand_type} hand)")
    print("=" * 80)

    # Images live at output/<id>/stage1/dynhamr/images/; keypoints at inputs/keypoints/.
    img_path0 = Path(img_paths[0]).resolve()
    if img_path0.parents[1].name == "dynhamr" and img_path0.parents[2].name == "stage1":
        video_dir = img_path0.parents[3]
    else:
        # Fallback for older trees where images sat under <id>/dynhamr/images/.
        video_dir = img_path0.parents[1]
    prefix = "lh" if hand_type == "left" else "rh"
    keypoints_file = video_dir / "inputs" / "keypoints" / f"{prefix}_keypoints.json"

    if not os.path.exists(keypoints_file):
        raise FileNotFoundError(f"Keypoints file not found: {keypoints_file}")

    print(f"Loading keypoints from: {keypoints_file}")
    with open(keypoints_file, 'r') as f:
        keypoints_data = json.load(f)

    if vis_dir is not None:
        pass1_dir = os.path.join(vis_dir, 'pass1_raw_bboxes')
        os.makedirs(pass1_dir, exist_ok=True)

    raw_data = []

    # Read the first image for height/width.
    first_img = cv2.imread(str(img_paths[0]))
    img_shape = first_img.shape

    for frame_idx, img_path in enumerate(tqdm(sorted(img_paths), desc="Pass 1: Computing bboxes from keypoints")):
        img_path = str(img_path)
        img_cv2 = cv2.imread(img_path)

        frame_data = {
            'frame_idx': frame_idx,
            'img_path': img_path,
            'left_bbox': None,
            'right_bbox': None,
            'left_keypoints': None,
            'right_keypoints': None,
            'left_conf': 0.0,
            'right_conf': 0.0
        }

        # Load keypoints for this frame from JSON.
        frame_key = str(frame_idx)
        if frame_key in keypoints_data:
            keypoints_2d = np.array(keypoints_data[frame_key])  # Shape: (21, 2)

            # Append a confidence column (all 1.0 = valid).
            keypoints_with_conf = np.zeros((21, 3))
            keypoints_with_conf[:, :2] = keypoints_2d
            keypoints_with_conf[:, 2] = 1.0

            # BBox from keypoints (1.2x padding).
            bbox = compute_bbox_from_keypoints(keypoints_with_conf, img_shape, scale_factor=1.2)

            if bbox is not None:
                # Assign to the selected hand side.
                if hand_type == 'right':
                    frame_data['right_bbox'] = bbox
                    frame_data['right_keypoints'] = keypoints_with_conf
                    frame_data['right_conf'] = 1.0
                else:  # left
                    frame_data['left_bbox'] = bbox
                    frame_data['left_keypoints'] = keypoints_with_conf
                    frame_data['left_conf'] = 1.0

        # Visualize bbox
        if vis_dir is not None:
            img_vis = img_cv2.copy()
            if frame_data['left_bbox'] is not None:
                x1, y1, x2, y2 = frame_data['left_bbox'].astype(int)
                cv2.rectangle(img_vis, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue for left
                cv2.putText(img_vis, f"L (kpts)", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                # Draw keypoints
                if frame_data['left_keypoints'] is not None:
                    for kp in frame_data['left_keypoints']:
                        x, y, conf = kp
                        if conf > 0.5:
                            cv2.circle(img_vis, (int(x), int(y)), 3, (255, 0, 0), -1)

            if frame_data['right_bbox'] is not None:
                x1, y1, x2, y2 = frame_data['right_bbox'].astype(int)
                cv2.rectangle(img_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green for right
                cv2.putText(img_vis, f"R (kpts)", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                # Draw keypoints
                if frame_data['right_keypoints'] is not None:
                    for kp in frame_data['right_keypoints']:
                        x, y, conf = kp
                        if conf > 0.5:
                            cv2.circle(img_vis, (int(x), int(y)), 3, (0, 255, 0), -1)

            img_fn = os.path.splitext(os.path.basename(img_path))[0]
            cv2.imwrite(os.path.join(pass1_dir, f'{img_fn}_raw.jpg'), img_vis)

        raw_data.append(frame_data)

    # Print stats
    left_count = sum(1 for f in raw_data if f['left_bbox'] is not None)
    right_count = sum(1 for f in raw_data if f['right_bbox'] is not None)
    missing_left = [i for i, f in enumerate(raw_data) if f['left_bbox'] is None]
    missing_right = [i for i, f in enumerate(raw_data) if f['right_bbox'] is None]

    print(f"\nComputed bboxes for {len(raw_data)} frames:")
    print(f"  Left hand detected in {left_count} frames ({100*left_count/len(raw_data):.1f}%)")
    print(f"  Right hand detected in {right_count} frames ({100*right_count/len(raw_data):.1f}%)")

    if len(missing_left) > 0:
        print(f"  Missing left hand in {len(missing_left)} frames")
        if len(missing_left) <= 20:
            print(f"    Frame indices: {missing_left}")
        else:
            print(f"    First 10: {missing_left[:10]}")
            print(f"    Last 10: {missing_left[-10:]}")

    if len(missing_right) > 0:
        print(f"  Missing right hand in {len(missing_right)} frames")
        if len(missing_right) <= 20:
            print(f"    Frame indices: {missing_right}")
        else:
            print(f"    First 10: {missing_right[:10]}")
            print(f"    Last 10: {missing_right[-10:]}")

    # Write video
    if vis_dir is not None:
        pass1_dir = os.path.join(vis_dir, 'pass1_raw_bboxes')
        video_path = os.path.join(vis_dir, 'pass1_raw_bboxes.mp4')
        create_video_from_images(pass1_dir, video_path, fps=30)

    return raw_data


def extract_raw_bboxes(img_paths, detector, vis_dir=None):
    """
    Pass 1: Extract raw hand bboxes using YOLO hand detector (like WiLoR).
    YOLO directly detects hand bboxes with left/right classification.
    No ViTPose needed - much simpler and more reliable!
    
    Args:
        img_paths: List of image paths
        detector: YOLO hand detector model
        vis_dir: Optional directory for visualizations
    
    Returns:
        raw_data: List of dicts for each frame with bbox data (no keypoints from YOLO)
    """
    print("\n" + "=" * 80)
    print("PASS 1: Extracting raw hand bboxes (YOLO hand detector)")
    print("=" * 80)

    if vis_dir is not None:
        pass1_dir = os.path.join(vis_dir, 'pass1_raw_bboxes')
        os.makedirs(pass1_dir, exist_ok=True)

    raw_data = []

    for frame_idx, img_path in enumerate(tqdm(sorted(img_paths), desc="Pass 1: Extracting hand bboxes")):
        img_path = str(img_path)
        img_cv2 = cv2.imread(img_path)

        frame_data = {
            'frame_idx': frame_idx,
            'img_path': img_path,
            'left_bbox': None,
            'right_bbox': None,
            'left_keypoints': None,  # Will be None for YOLO version
            'right_keypoints': None,  # Will be None for YOLO version
            'left_conf': 0.0,
            'right_conf': 0.0
        }

        # YOLO direct hand detection (following WiLoR demo.py)
        # Class 0 = left hand, Class 1 = right hand
        detections = detector(img_cv2, conf=0.5, verbose=False)[0]

        # Track best detection for each hand (in case of duplicates)
        left_detections = []
        right_detections = []

        # Helper function to expand bbox by a given scale (e.g., 1.2x) and make it SQUARE
        def enlarge_bbox(bbox, scale=1.2, img_shape=None):
            x1, y1, x2, y2 = bbox
            w = x2 - x1
            h = y2 - y1
            cx = x1 + w / 2.0
            cy = y1 + h / 2.0

            # Make it square by taking the max dimension
            size = max(w, h) * scale

            new_x1 = cx - size / 2.0
            new_y1 = cy - size / 2.0
            new_x2 = cx + size / 2.0
            new_y2 = cy + size / 2.0

            # Optionally clip to image boundaries
            if img_shape is not None:
                H, W = img_shape[:2]
                new_x1 = max(0, min(W - 1, new_x1))
                new_y1 = max(0, min(H - 1, new_y1))
                new_x2 = max(0, min(W - 1, new_x2))
                new_y2 = max(0, min(H - 1, new_y2))
            return np.array([new_x1, new_y1, new_x2, new_y2], dtype=bbox.dtype)

        img_shape = img_cv2.shape
        for det in detections:
            bbox = det.boxes.xyxy[0].cpu().numpy()  # x1, y1, x2, y2
            conf = float(det.boxes.conf[0])
            cls = int(det.boxes.cls[0])

            # Enlarge bbox by 1.2x, clamp to image
            bbox = enlarge_bbox(bbox, scale=1.2, img_shape=img_shape)

            if cls == 0:  # left hand
                left_detections.append({'bbox': bbox, 'conf': conf})
            elif cls == 1:  # right hand
                right_detections.append({'bbox': bbox, 'conf': conf})

        # Keep highest confidence detection for each hand
        if len(left_detections) > 0:
            best_left = max(left_detections, key=lambda x: x['conf'])
            frame_data['left_bbox'] = best_left['bbox']
            frame_data['left_conf'] = best_left['conf']

        if len(right_detections) > 0:
            best_right = max(right_detections, key=lambda x: x['conf'])
            frame_data['right_bbox'] = best_right['bbox']
            frame_data['right_conf'] = best_right['conf']

        # Visualize raw bboxes
        if vis_dir is not None:
            img_vis = img_cv2.copy()
            if frame_data['left_bbox'] is not None:
                x1, y1, x2, y2 = frame_data['left_bbox'].astype(int)
                cv2.rectangle(img_vis, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue for left
                cv2.putText(img_vis, f"L (conf={frame_data['left_conf']:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            if frame_data['right_bbox'] is not None:
                x1, y1, x2, y2 = frame_data['right_bbox'].astype(int)
                cv2.rectangle(img_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green for right
                cv2.putText(img_vis, f"R (conf={frame_data['right_conf']:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            img_fn = os.path.splitext(os.path.basename(img_path))[0]
            cv2.imwrite(os.path.join(pass1_dir, f'{img_fn}_raw.jpg'), img_vis)

        raw_data.append(frame_data)

    # Print statistics with missing frame details
    left_count = sum(1 for f in raw_data if f['left_bbox'] is not None)
    right_count = sum(1 for f in raw_data if f['right_bbox'] is not None)
    missing_left = [i for i, f in enumerate(raw_data) if f['left_bbox'] is None]
    missing_right = [i for i, f in enumerate(raw_data) if f['right_bbox'] is None]

    print(f"\nExtracted bboxes for {len(raw_data)} frames:")
    print(f"  Left hand detected in {left_count} frames ({100*left_count/len(raw_data):.1f}%)")
    print(f"  Right hand detected in {right_count} frames ({100*right_count/len(raw_data):.1f}%)")

    if len(missing_left) > 0:
        print(f"  Missing left hand in {len(missing_left)} frames")
        if len(missing_left) <= 20:
            print(f"    Frame indices: {missing_left}")
        else:
            print(f"    First 10: {missing_left[:10]}")
            print(f"    Last 10: {missing_left[-10:]}")

    if len(missing_right) > 0:
        print(f"  Missing right hand in {len(missing_right)} frames")
        if len(missing_right) <= 20:
            print(f"    Frame indices: {missing_right}")
        else:
            print(f"    First 10: {missing_right[:10]}")
            print(f"    Last 10: {missing_right[-10:]}")

    # Create video from Pass 1 visualizations
    if vis_dir is not None:
        pass1_dir = os.path.join(vis_dir, 'pass1_raw_bboxes')
        video_path = os.path.join(vis_dir, 'pass1_raw_bboxes.mp4')
        create_video_from_images(pass1_dir, video_path, fps=30)
    return raw_data


# ============================================================================
# PASS 2: Clean bbox sequences globally
# ============================================================================


def detect_overlapping_bboxes(raw_data, iou_threshold=0.7, containment_threshold=0.7):
    """
    Detect and remove overlapping bboxes (hallucinations) - YOLO version without keypoints.
    Uses simplified criteria:
    1. High bbox IoU (> 0.7) - very overlapping bboxes
    2. High containment (> 0.7) - one bbox is largely contained in another
    
    Since YOLO doesn't provide keypoints, we only use bbox-based overlap detection.
    This is actually more reliable since YOLO is specifically trained for hand detection.
    
    Also tracks which hand was removed due to overlap and which hand "caused" the removal,
    so we can potentially restore the removed hand if the winner is later invalidated.
    """
    print("\n" + "-" * 80)
    print("Step 1: Detecting overlapping bboxes (hallucinations)")
    print("-" * 80)

    overlap_count = 0

    for frame_data in raw_data:
        # Initialize removal tracking flags
        frame_data['left_removed_due_to_overlap_with'] = None
        frame_data['right_removed_due_to_overlap_with'] = None
        if frame_data['left_bbox'] is not None and frame_data['right_bbox'] is not None:
            # Criterion 1: Bbox IoU
            iou = compute_iou(frame_data['left_bbox'], frame_data['right_bbox'])

            # Criterion 2: Containment ratio (either bbox contains the other)
            left_in_right = compute_containment_ratio(frame_data['left_bbox'], frame_data['right_bbox'])
            right_in_left = compute_containment_ratio(frame_data['right_bbox'], frame_data['left_bbox'])
            max_containment = max(left_in_right, right_in_left)

            # Determine if this is an overlap/hallucination
            is_overlap = False
            reason = []

            if iou > iou_threshold:
                is_overlap = True
                reason.append(f"IoU={iou:.3f}")

            if max_containment > containment_threshold:
                is_overlap = True
                which_contained = "L_in_R" if left_in_right > right_in_left else "R_in_L"
                reason.append(f"containment={max_containment:.2f} ({which_contained})")

            if is_overlap:
                overlap_count += 1
                print(f"  Frame {frame_data['frame_idx']:04d}: {', '.join(reason)} - OVERLAP DETECTED")

                # Keep hand with higher confidence, track which hand was removed and why
                if frame_data['left_conf'] > frame_data['right_conf']:
                    print(f"    → Keeping LEFT (conf={frame_data['left_conf']:.3f}), removing RIGHT")
                    frame_data['right_bbox'] = None
                    frame_data['right_keypoints'] = None
                    frame_data['right_conf'] = 0.0
                    frame_data['right_removed_due_to_overlap_with'] = 'left'
                else:
                    print(f"    → Keeping RIGHT (conf={frame_data['right_conf']:.3f}), removing LEFT")
                    frame_data['left_bbox'] = None
                    frame_data['left_keypoints'] = None
                    frame_data['left_conf'] = 0.0
                    frame_data['left_removed_due_to_overlap_with'] = 'right'

    print(f"Removed overlapping bboxes in {overlap_count} frames")
    return raw_data


def fix_handedness_swaps_by_trajectory(raw_data, position_threshold=200):
    """
    Detect handedness swaps by analyzing trajectory jumps.
    If one hand suddenly jumps to where the other hand was, it's likely mislabeled.
    
    Example: Green curve (right) drops to blue curve's position while blue disappears
    → The "right" detection is actually the left hand with wrong label
    """
    print("\n" + "-" * 80)
    print(f"Step 1.4a: Detecting handedness swaps via trajectory analysis")
    print("-" * 80)

    swap_count = 0
    n = len(raw_data)

    for i in range(1, n):
        left_bbox = raw_data[i]['left_bbox']
        right_bbox = raw_data[i]['right_bbox']
        prev_left = raw_data[i - 1]['left_bbox']
        prev_right = raw_data[i - 1]['right_bbox']

        # Case 1: Right hand jumps to where left hand was (left disappears)
        if right_bbox is not None and left_bbox is None and prev_left is not None and prev_right is not None:
            # Calculate positions
            prev_left_cx = (prev_left[0] + prev_left[2]) / 2
            prev_right_cx = (prev_right[0] + prev_right[2]) / 2
            curr_right_cx = (right_bbox[0] + right_bbox[2]) / 2

            # Distance from current "right" to previous left/right
            dist_to_prev_left = abs(curr_right_cx - prev_left_cx)
            dist_to_prev_right = abs(curr_right_cx - prev_right_cx)

            # If current "right" is much closer to previous left than previous right, it's swapped
            if dist_to_prev_left < dist_to_prev_right and dist_to_prev_right > position_threshold:
                print(
                    f"  Frame {i:04d}: RIGHT jumped to LEFT position (dist to prev_left={dist_to_prev_left:.0f}, dist to prev_right={dist_to_prev_right:.0f}) - Swapping RIGHT → LEFT"
                )
                raw_data[i]['left_bbox'] = right_bbox
                raw_data[i]['left_keypoints'] = raw_data[i]['right_keypoints']
                raw_data[i]['left_conf'] = raw_data[i]['right_conf']
                raw_data[i]['right_bbox'] = None
                raw_data[i]['right_keypoints'] = None
                raw_data[i]['right_conf'] = 0.0
                swap_count += 1

        # Case 2: Left hand jumps to where right hand was (right disappears)
        elif left_bbox is not None and right_bbox is None and prev_right is not None and prev_left is not None:
            # Calculate positions
            prev_left_cx = (prev_left[0] + prev_left[2]) / 2
            prev_right_cx = (prev_right[0] + prev_right[2]) / 2
            curr_left_cx = (left_bbox[0] + left_bbox[2]) / 2

            # Distance from current "left" to previous left/right
            dist_to_prev_left = abs(curr_left_cx - prev_left_cx)
            dist_to_prev_right = abs(curr_left_cx - prev_right_cx)

            # If current "left" is much closer to previous right than previous left, it's swapped
            if dist_to_prev_right < dist_to_prev_left and dist_to_prev_left > position_threshold:
                print(
                    f"  Frame {i:04d}: LEFT jumped to RIGHT position (dist to prev_right={dist_to_prev_right:.0f}, dist to prev_left={dist_to_prev_left:.0f}) - Swapping LEFT → RIGHT"
                )
                raw_data[i]['right_bbox'] = left_bbox
                raw_data[i]['right_keypoints'] = raw_data[i]['left_keypoints']
                raw_data[i]['right_conf'] = raw_data[i]['left_conf']
                raw_data[i]['left_bbox'] = None
                raw_data[i]['left_keypoints'] = None
                raw_data[i]['left_conf'] = 0.0
                swap_count += 1

    if swap_count > 0:
        print(f"  Fixed {swap_count} trajectory-based handedness swaps")
    else:
        print(f"  No trajectory-based swaps detected")

    return raw_data


def fix_handedness_swaps(raw_data, context_window=10):
    """
    Detect and fix handedness swaps where YOLO misclassifies hand labels.
    For example, detecting left hand as right hand for a few frames.
    
    Strategy: If a hand appears/disappears while the other hand simultaneously 
    disappears/appears in a short sequence, it's likely a handedness swap.
    """
    print("\n" + "-" * 80)
    print(f"Step 1.4: Detecting handedness swaps (context window={context_window})")
    print("-" * 80)

    swap_count = 0
    n = len(raw_data)

    for i in range(n):
        left_bbox = raw_data[i]['left_bbox']
        right_bbox = raw_data[i]['right_bbox']

        # Case 1: Only right hand detected, but neighbors mostly have only left hand
        if right_bbox is not None and left_bbox is None:
            # Count neighbors with left/right
            left_count = 0
            right_count = 0

            for j in range(max(0, i - context_window), min(n, i + context_window + 1)):
                if j == i:
                    continue
                if raw_data[j]['left_bbox'] is not None and raw_data[j]['right_bbox'] is None:
                    left_count += 1
                elif raw_data[j]['right_bbox'] is not None and raw_data[j]['left_bbox'] is None:
                    right_count += 1

            # If neighbors are mostly left-only, this right is probably mislabeled left
            if left_count > 0 and left_count > right_count * 2:
                print(f"  Frame {i:04d}: Swapping RIGHT → LEFT (neighbors: {left_count} left-only, {right_count} right-only)")
                raw_data[i]['left_bbox'] = right_bbox
                raw_data[i]['left_keypoints'] = raw_data[i]['right_keypoints']
                raw_data[i]['left_conf'] = raw_data[i]['right_conf']
                raw_data[i]['right_bbox'] = None
                raw_data[i]['right_keypoints'] = None
                raw_data[i]['right_conf'] = 0.0
                swap_count += 1

        # Case 2: Only left hand detected, but neighbors mostly have only right hand
        elif left_bbox is not None and right_bbox is None:
            # Count neighbors with left/right
            left_count = 0
            right_count = 0

            for j in range(max(0, i - context_window), min(n, i + context_window + 1)):
                if j == i:
                    continue
                if raw_data[j]['left_bbox'] is not None and raw_data[j]['right_bbox'] is None:
                    left_count += 1
                elif raw_data[j]['right_bbox'] is not None and raw_data[j]['left_bbox'] is None:
                    right_count += 1

            # If neighbors are mostly right-only, this left is probably mislabeled right
            if right_count > 0 and right_count > left_count * 2:
                print(f"  Frame {i:04d}: Swapping LEFT → RIGHT (neighbors: {left_count} left-only, {right_count} right-only)")
                raw_data[i]['right_bbox'] = left_bbox
                raw_data[i]['right_keypoints'] = raw_data[i]['left_keypoints']
                raw_data[i]['right_conf'] = raw_data[i]['left_conf']
                raw_data[i]['left_bbox'] = None
                raw_data[i]['left_keypoints'] = None
                raw_data[i]['left_conf'] = 0.0
                swap_count += 1

    if swap_count > 0:
        print(f"  Fixed {swap_count} handedness swaps")
    else:
        print(f"  No handedness swaps detected")

    return raw_data


def fix_handedness_swaps_frame_to_frame(raw_data, iou_threshold=0.7, max_gap=10):
    """
    Detect frame-to-frame handedness swaps: same bbox position but different label.
    
    Example problem:
    - Frame N:   Right hand at position X (green box)
    - Frame N+1: Left hand at position X (blue box) - SAME BBOX, WRONG LABEL!
    
    This catches YOLO randomly mislabeling the same physical hand.
    
    Strategy:
    If a hand suddenly appears while the other hand recently disappeared (within max_gap frames),
    AND the new hand's bbox has high IoU with the disappeared hand's LAST VALID bbox,
    → It's the same hand with wrong label! Swap it back!
    
    Args:
        max_gap: Maximum frames to look back for last valid other hand (default 10)
    """
    print("\n" + "-" * 80)
    print(f"Step 1.4: Detecting frame-to-frame handedness swaps (IoU threshold={iou_threshold}, max_gap={max_gap})")
    print("-" * 80)

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

            # Case: Current hand exists and other hand is missing
            if curr_bbox is not None and curr_other_bbox is None:
                # Look back to find LAST VALID bbox of OTHER hand (within max_gap frames)
                last_valid_other_bbox = None
                last_valid_other_idx = None

                for j in range(i - 1, max(0, i - max_gap - 1), -1):
                    if raw_data[j][other_bbox_key] is not None:
                        last_valid_other_bbox = raw_data[j][other_bbox_key]
                        last_valid_other_idx = j
                        break

                # If found a recent OTHER hand detection, check IoU
                print(f"  Frame {curr['frame_idx']:04d}: {last_valid_other_bbox is not None}")
                if last_valid_other_bbox is not None:
                    gap = i - last_valid_other_idx
                    iou = compute_iou(curr_bbox, last_valid_other_bbox)
                    print(f"  Frame {curr['frame_idx']:04d}: IoU = {iou:.3f}")
                    if iou > iou_threshold:
                        print(f"  Frame {curr['frame_idx']:04d}: {hand_name} at same position as {other_hand} "
                              f"(last seen at frame {raw_data[last_valid_other_idx]['frame_idx']}, gap={gap}, IoU={iou:.3f}) "
                              f"- SWAPPING {hand_name} → {other_hand}")

                        # Swap: this is actually the OTHER hand
                        curr[other_bbox_key] = curr_bbox.copy()
                        curr[other_keyp_key] = curr[keyp_key].copy() if curr[keyp_key] is not None else None
                        curr[other_conf_key] = curr[conf_key]

                        # Remove from current hand
                        curr[bbox_key] = None
                        curr[keyp_key] = None
                        curr[conf_key] = 0.0

                        swap_count += 1

    if swap_count > 0:
        print(f"  Fixed {swap_count} frame-to-frame handedness swaps")
    else:
        print(f"  No frame-to-frame swaps detected")

    return raw_data


def fix_handedness_inconsistencies(raw_data, original_data, context_window=5):
    """
    Detect handedness swaps using SPATIAL-TEMPORAL consistency.
    
    Example problem: [L, L, L, _, _, R, R, R, L, L]
    - The 3 Rs are spatially in the LEFT hand's trajectory but mislabeled as RIGHT
    
    Strategy:
    1. For each detection, compute IoU with recent detections of SAME hand
    2. Also compute IoU with recent detections of OTHER hand
    3. If IoU with OTHER hand's trajectory is much higher → handedness is swapped!
    """
    print("\n" + "-" * 80)
    print(f"Step 1.5: Fixing handedness inconsistencies via spatial-temporal consistency (window={context_window})")
    print("-" * 80)

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

            # Collect recent bboxes for SAME hand and OTHER hand
            same_hand_bboxes = []
            other_hand_bboxes = []

            # Look at context window (both past and future)
            for j in range(max(0, i - context_window), min(len(raw_data), i + context_window + 1)):
                if j == i:
                    continue  # Skip current frame

                # Collect SAME hand bboxes
                if raw_data[j][bbox_key] is not None:
                    same_hand_bboxes.append(raw_data[j][bbox_key])

                # Collect OTHER hand bboxes
                if raw_data[j][other_bbox_key] is not None:
                    other_hand_bboxes.append(raw_data[j][other_bbox_key])

            # Need enough context to make a decision
            if len(same_hand_bboxes) < 2 and len(other_hand_bboxes) < 2:
                continue  # Not enough spatial context

            # Compute average IoU with SAME hand trajectory
            same_hand_ious = []
            for bbox in same_hand_bboxes:
                iou = compute_iou(current_bbox, bbox)
                same_hand_ious.append(iou)
            avg_iou_same = np.mean(same_hand_ious) if len(same_hand_ious) > 0 else 0.0

            # Compute average IoU with OTHER hand trajectory
            other_hand_ious = []
            for bbox in other_hand_bboxes:
                iou = compute_iou(current_bbox, bbox)
                other_hand_ious.append(iou)
            avg_iou_other = np.mean(other_hand_ious) if len(other_hand_ious) > 0 else 0.0

            # DETECTION LOGIC: If current bbox fits OTHER hand's trajectory much better, it's swapped!
            # Criteria:
            # 1. OTHER hand trajectory has high overlap (>0.5) with current bbox
            # 2. OTHER hand trajectory overlap is significantly higher than SAME hand (1.5x)
            # 3. We have enough OTHER hand detections nearby (at least 3)
            if (len(other_hand_bboxes) >= 3 and avg_iou_other > 0.5 and avg_iou_other > avg_iou_same * 1.5):

                print(f"  Frame {frame_data['frame_idx']:04d} ({hand_name}): "
                      f"Bbox fits {other_hand} trajectory better! "
                      f"(IoU with {hand_name}={avg_iou_same:.3f}, IoU with {other_hand}={avg_iou_other:.3f}, "
                      f"neighbors: {len(same_hand_bboxes)} {hand_name}, {len(other_hand_bboxes)} {other_hand}) "
                      f"- SWAPPING {hand_name} → {other_hand}")

                # Swap: move current bbox to OTHER hand
                frame_data[other_bbox_key] = current_bbox.copy()
                frame_data[other_keyp_key] = frame_data[keyp_key].copy() if frame_data[keyp_key] is not None else None
                frame_data[other_conf_key] = frame_data[conf_key]

                # Remove from current hand
                frame_data[bbox_key] = None
                frame_data[keyp_key] = None
                frame_data[conf_key] = 0.0

                swap_count += 1

    if swap_count > 0:
        print(f"  Fixed {swap_count} handedness swaps via spatial-temporal consistency")
    else:
        print(f"  No spatial-temporal inconsistencies found")

    return raw_data


def detect_bbox_jumps(raw_data, base_iou_threshold=0.5):
    """
    Detect and fix bbox jumps with ADAPTIVE threshold based on detection gaps.
    
    Adaptive threshold formula: adjusted_threshold = max(0.0, base_threshold - gap_frames * 0.15)
    - Gap 0 (consecutive): threshold = 0.5 (strict, catches hallucinations)
    - Gap 1: threshold = 0.35 (slightly lenient)
    - Gap 2: threshold = 0.2 (more lenient)
    - Gap 3+: threshold = 0.05/0.0 (very lenient, accepts genuine movement)
    
    This allows genuine hand movement after detection gaps while still protecting
    against spurious jumps in consecutive frames.
    """
    print("\n" + "-" * 80)
    print(f"Step 2: Detecting bbox jumps with adaptive threshold (base={base_iou_threshold})")
    print("-" * 80)

    for hand_name in ['left', 'right']:
        bbox_key = f'{hand_name}_bbox'
        hand_id = 0 if hand_name == 'left' else 1

        last_valid_bbox = None
        last_valid_idx = None
        frames_since_last_detection = 0
        jump_count = 0

        for i, frame_data in enumerate(raw_data):
            current_bbox = frame_data[bbox_key]

            if current_bbox is not None:
                if last_valid_bbox is not None:
                    # Calculate IoU with last valid bbox
                    iou = compute_iou(current_bbox, last_valid_bbox)

                    # Adaptive threshold based on gap
                    gap_frames = frames_since_last_detection
                    adjusted_threshold = max(0.0, base_iou_threshold - (gap_frames * 0.01))

                    if iou < adjusted_threshold:
                        # Jump detected!
                        jump_count += 1
                        print(f"  Frame {frame_data['frame_idx']:04d} ({hand_name}): "
                              f"IoU={iou:.3f} < threshold={adjusted_threshold:.3f} (gap={gap_frames}) - JUMP DETECTED")
                        print(f"    → Current bbox: [{current_bbox[0]:.1f}, {current_bbox[1]:.1f}, {current_bbox[2]:.1f}, {current_bbox[3]:.1f}]")
                        print(
                            f"    → Last valid bbox (frame {last_valid_idx}): [{last_valid_bbox[0]:.1f}, {last_valid_bbox[1]:.1f}, {last_valid_bbox[2]:.1f}, {last_valid_bbox[3]:.1f}]"
                        )
                        print(f"    → Replacing with last valid bbox from frame {last_valid_idx}")
                        frame_data[bbox_key] = last_valid_bbox.copy()
                    else:
                        # Valid detection
                        if gap_frames > 0:
                            print(f"  Frame {frame_data['frame_idx']:04d} ({hand_name}): "
                                  f"IoU={iou:.3f} > threshold={adjusted_threshold:.3f} (gap={gap_frames}) - ACCEPTING movement")
                        last_valid_bbox = current_bbox.copy()
                        last_valid_idx = i
                        frames_since_last_detection = 0
                else:
                    # First valid bbox
                    last_valid_bbox = current_bbox.copy()
                    last_valid_idx = i
                    frames_since_last_detection = 0
            else:
                # Hand is missing
                frames_since_last_detection += 1

        if jump_count > 0:
            print(f"  {hand_name.capitalize()}: Fixed {jump_count} jumps")

    return raw_data


def interpolate_missing_bboxes(raw_data, max_gap=5):
    """Interpolate missing bboxes for short gaps and create dummy keypoints."""
    print("\n" + "-" * 80)
    print(f"Step 3: Interpolating missing bboxes (max gap={max_gap} frames)")
    print("-" * 80)

    for hand_name in ['left', 'right']:
        bbox_key = f'{hand_name}_bbox'
        keyp_key = f'{hand_name}_keypoints'
        conf_key = f'{hand_name}_conf'

        # Find valid bbox indices
        valid_indices = [i for i, f in enumerate(raw_data) if f[bbox_key] is not None]

        if len(valid_indices) < 2:
            continue

        interpolated_count = 0

        for i in range(len(valid_indices) - 1):
            start_idx = valid_indices[i]
            end_idx = valid_indices[i + 1]
            gap = end_idx - start_idx - 1

            if 0 < gap <= max_gap:
                start_bbox = raw_data[start_idx][bbox_key]
                end_bbox = raw_data[end_idx][bbox_key]

                for j in range(1, gap + 1):
                    alpha = j / (gap + 1)
                    interp_bbox = (1 - alpha) * start_bbox + alpha * end_bbox
                    raw_data[start_idx + j][bbox_key] = interp_bbox

                    # Create dummy keypoints for interpolated frames
                    dummy_keypoints = np.zeros((21, 3))
                    dummy_keypoints[:, 2] = 0.5
                    raw_data[start_idx + j][keyp_key] = dummy_keypoints
                    raw_data[start_idx + j][conf_key] = 0.5

                    interpolated_count += 1

                print(f"  {hand_name.capitalize()}: Interpolated frames {start_idx+1}-{end_idx-1} (gap={gap})")

        if interpolated_count > 0:
            print(f"  {hand_name.capitalize()}: Interpolated {interpolated_count} frames total")

    return raw_data


def smooth_bbox_sequence(raw_data, window_size=5):
    """Apply temporal smoothing using moving average."""
    print("\n" + "-" * 80)
    print(f"Step 4: Smoothing bbox sequences (window={window_size})")
    print("-" * 80)

    for hand_name in ['left', 'right']:
        bbox_key = f'{hand_name}_bbox'

        bboxes = [f[bbox_key] for f in raw_data]
        smoothed_bboxes = []

        for i in range(len(bboxes)):
            if bboxes[i] is None:
                smoothed_bboxes.append(None)
                continue

            window_start = max(0, i - window_size // 2)
            window_end = min(len(bboxes), i + window_size // 2 + 1)
            window_bboxes = [bboxes[j] for j in range(window_start, window_end) if bboxes[j] is not None]

            if len(window_bboxes) > 0:
                smoothed_bbox = np.mean(window_bboxes, axis=0)
                smoothed_bboxes.append(smoothed_bbox)
            else:
                smoothed_bboxes.append(bboxes[i])

        for i, frame_data in enumerate(raw_data):
            if smoothed_bboxes[i] is not None:
                frame_data[bbox_key] = smoothed_bboxes[i]

    print("  Smoothing complete")
    return raw_data


def check_if_bboxes_overlap(bbox1,
                            bbox2,
                            keypoints1=None,
                            keypoints2=None,
                            iou_threshold=0.7,
                            containment_threshold=0.7,
                            wrist_dist_ratio=0.15,
                            keypoint_sim_ratio=0.10):
    """
    Check if two bboxes overlap using the same comprehensive criteria as detect_overlapping_bboxes.
    Returns True if the bboxes are considered overlapping (hallucination).
    """
    # Compute average bbox size for normalization
    size1 = max(bbox1[2] - bbox1[0], bbox1[3] - bbox1[1])
    size2 = max(bbox2[2] - bbox2[0], bbox2[3] - bbox2[1])
    avg_bbox_size = (size1 + size2) / 2

    # Criterion 1: Bbox IoU
    iou = compute_iou(bbox1, bbox2)
    if iou > iou_threshold:
        return True

    # Criterion 2: Containment ratio
    containment_1_in_2 = compute_containment_ratio(bbox1, bbox2)
    containment_2_in_1 = compute_containment_ratio(bbox2, bbox1)
    max_containment = max(containment_1_in_2, containment_2_in_1)
    if max_containment > containment_threshold:
        return True

    # Criterion 3 & 4: Keypoint-based checks (if keypoints available)
    if keypoints1 is not None and keypoints2 is not None:
        # Wrist distance
        wrist1 = keypoints1[0, :2]
        wrist2 = keypoints2[0, :2]
        wrist_dist = np.linalg.norm(wrist1 - wrist2)
        wrist_dist_normalized = wrist_dist / avg_bbox_size
        if wrist_dist_normalized < wrist_dist_ratio:
            return True

        # Average keypoint distance
        keypoint_dists = []
        for i in range(min(len(keypoints1), len(keypoints2))):
            kp1 = keypoints1[i, :2]
            kp2 = keypoints2[i, :2]
            dist = np.linalg.norm(kp1 - kp2)
            keypoint_dists.append(dist)
        avg_keypoint_dist = np.mean(keypoint_dists)
        avg_keypoint_dist_normalized = avg_keypoint_dist / avg_bbox_size
        if avg_keypoint_dist_normalized < keypoint_sim_ratio:
            return True

    return False


def plot_bbox_trajectories(raw_data, vis_dir, filename='bbox_trajectories.png'):
    """
    Plot left and right hand bbox center trajectories over time.
    This helps visualize handedness consistency and detect label swaps.
    """
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt

    frames = []
    left_centers_x = []
    left_centers_y = []
    right_centers_x = []
    right_centers_y = []

    for i, frame_data in enumerate(raw_data):
        frames.append(i)

        # Left hand
        if frame_data['left_bbox'] is not None:
            left_cx = (frame_data['left_bbox'][0] + frame_data['left_bbox'][2]) / 2
            left_cy = (frame_data['left_bbox'][1] + frame_data['left_bbox'][3]) / 2
            left_centers_x.append(left_cx)
            left_centers_y.append(left_cy)
        else:
            left_centers_x.append(None)
            left_centers_y.append(None)

        # Right hand
        if frame_data['right_bbox'] is not None:
            right_cx = (frame_data['right_bbox'][0] + frame_data['right_bbox'][2]) / 2
            right_cy = (frame_data['right_bbox'][1] + frame_data['right_bbox'][3]) / 2
            right_centers_x.append(right_cx)
            right_centers_y.append(right_cy)
        else:
            right_centers_x.append(None)
            right_centers_y.append(None)

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 10))

    # Plot X coordinates (horizontal position)
    ax1.plot(frames, left_centers_x, 'b-', label='Left hand', linewidth=1.5, alpha=0.8)
    ax1.plot(frames, right_centers_x, 'g-', label='Right hand', linewidth=1.5, alpha=0.8)
    ax1.set_xlabel('Frame', fontsize=12)
    ax1.set_ylabel('X coordinate (pixels)', fontsize=12)
    ax1.set_title('Hand Bbox Center X-Position Over Time\n(Left hand should be consistently < Right hand)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Highlight regions where left > right (potential swaps)
    swap_count = 0
    for i in range(len(frames)):
        if left_centers_x[i] is not None and right_centers_x[i] is not None:
            if left_centers_x[i] > right_centers_x[i]:
                ax1.axvspan(i - 0.5, i + 0.5, alpha=0.3, color='red')
                swap_count += 1

    # Plot Y coordinates (vertical position)
    ax2.plot(frames, left_centers_y, 'b-', label='Left hand', linewidth=1.5, alpha=0.8)
    ax2.plot(frames, right_centers_y, 'g-', label='Right hand', linewidth=1.5, alpha=0.8)
    ax2.set_xlabel('Frame', fontsize=12)
    ax2.set_ylabel('Y coordinate (pixels)', fontsize=12)
    ax2.set_title('Hand Bbox Center Y-Position Over Time', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.invert_yaxis()  # Invert Y axis since image coordinates go top-to-bottom

    plt.tight_layout()
    save_path = os.path.join(vis_dir, filename)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n{'='*80}")
    print(f"Bbox trajectory plot saved to: {save_path}")
    print(f"  Red regions indicate frames where left_x > right_x (potential label swaps)")
    print(f"  Total frames with left_x > right_x: {swap_count}")
    print(f"{'='*80}\n")


def clean_bbox_sequences(raw_data, vis_dir=None):
    """Pass 2: Clean bbox sequences using global temporal information."""
    print("\n" + "=" * 80)
    print("PASS 2: Cleaning bbox sequences")
    print("=" * 80)

    if vis_dir is not None:
        pass2_dir = os.path.join(vis_dir, 'pass2_cleaned_bboxes')
        os.makedirs(pass2_dir, exist_ok=True)

    # Store original bboxes and keypoints for comparison and restoration
    original_data = [{
        'left_bbox': f['left_bbox'].copy() if f['left_bbox'] is not None else None,
        'right_bbox': f['right_bbox'].copy() if f['right_bbox'] is not None else None,
        'left_keypoints': f['left_keypoints'].copy() if f['left_keypoints'] is not None else None,
        'right_keypoints': f['right_keypoints'].copy() if f['right_keypoints'] is not None else None,
        'left_conf': f['left_conf'] if f['left_conf'] is not None else None,
        'right_conf': f['right_conf'] if f['right_conf'] is not None else None,
    } for f in raw_data]

    # Plot trajectories BEFORE cleaning
    if vis_dir is not None:
        plot_bbox_trajectories(raw_data, vis_dir, filename='bbox_trajectories_before_cleaning.png')

    # Remove oversized bboxes (> 50% of image area)
    print("\n" + "-" * 80)
    print("Step 1.5: Removing oversized bboxes (> 50% of image area)")
    print("-" * 80)
    MAX_BBOX_AREA_RATIO = 0.5
    removed_oversized = 0

    # Load first image to get dimensions (all images should have same size)
    first_img = cv2.imread(raw_data[0]['img_path'])
    img_area = first_img.shape[0] * first_img.shape[1]

    for frame_data in raw_data:
        for hand_name in ['left', 'right']:
            bbox_key = f'{hand_name}_bbox'
            keyp_key = f'{hand_name}_keypoints'
            conf_key = f'{hand_name}_conf'

            bbox = frame_data[bbox_key]
            if bbox is not None:
                bbox_width = bbox[2] - bbox[0]
                bbox_height = bbox[3] - bbox[1]
                bbox_area = bbox_width * bbox_height
                area_ratio = bbox_area / img_area

                if area_ratio > MAX_BBOX_AREA_RATIO:
                    print(f"  Frame {frame_data['frame_idx']:04d} ({hand_name}): "
                          f"Bbox too large ({area_ratio:.1%} of image) - REMOVING")
                    frame_data[bbox_key] = None
                    frame_data[keyp_key] = None
                    frame_data[conf_key] = 0.0
                    removed_oversized += 1

    if removed_oversized > 0:
        print(f"  Removed {removed_oversized} oversized bboxes")
    else:
        print(f"  No oversized bboxes found")

    raw_data = detect_overlapping_bboxes(raw_data, iou_threshold=0.7)
    raw_data = fix_handedness_swaps_by_trajectory(raw_data, position_threshold=200)
    raw_data = fix_handedness_inconsistencies(raw_data, original_data, context_window=5)
    # raw_data = detect_bbox_jumps(raw_data, base_iou_threshold=0.4)
    raw_data = fix_handedness_swaps_frame_to_frame(raw_data, iou_threshold=0.6)

    # Apply patience mechanism with interpolation when hand reappears
    print("\n" + "-" * 80)
    print("Step 3: Applying patience mechanism with interpolation")
    print("-" * 80)
    PATIENCE_FRAMES = 25
    MAX_PATIENCE_WITHOUT_RETURN = 0  # No padding if hand never reappears
    patience_applied_count = 0
    interpolated_count = 0
    trimmed_count = 0

    for hand_id in [0, 1]:
        hand_name = 'left' if hand_id == 0 else 'right'
        bbox_key = f'{hand_name}_bbox'
        keyp_key = f'{hand_name}_keypoints'
        conf_key = f'{hand_name}_conf'

        i = 0
        while i < len(raw_data):
            # Find start of a gap (hand missing)
            if raw_data[i][bbox_key] is None:
                # Look back to find last valid bbox
                last_valid_idx = None
                for j in range(i - 1, -1, -1):
                    if raw_data[j][bbox_key] is not None:
                        last_valid_idx = j
                        break

                if last_valid_idx is None:
                    i += 1
                    continue

                # Find end of gap (hand reappears or sequence ends)
                gap_start = i
                gap_end = None
                for j in range(i, min(i + PATIENCE_FRAMES, len(raw_data))):
                    if raw_data[j][bbox_key] is not None:
                        gap_end = j
                        break

                if gap_end is not None:
                    # Hand reappeared within patience threshold - CHECK DISTANCE FIRST
                    gap_length = gap_end - gap_start
                    start_bbox = raw_data[last_valid_idx][bbox_key]
                    end_bbox = raw_data[gap_end][bbox_key]

                    # Compute center distance between start and end bbox
                    start_cx = (start_bbox[0] + start_bbox[2]) / 2
                    start_cy = (start_bbox[1] + start_bbox[3]) / 2
                    end_cx = (end_bbox[0] + end_bbox[2]) / 2
                    end_cy = (end_bbox[1] + end_bbox[3]) / 2
                    center_distance = np.sqrt((end_cx - start_cx)**2 + (end_cy - start_cy)**2)

                    # Compute bbox width (use average of start and end)
                    start_width = start_bbox[2] - start_bbox[0]
                    end_width = end_bbox[2] - end_bbox[0]
                    avg_width = (start_width + end_width) / 2

                    # Only interpolate if bboxes are close enough (within 2× bbox width)
                    max_distance = avg_width

                    if center_distance <= max_distance:
                        # Bboxes are close - INTERPOLATE
                        print(f"  {hand_name.capitalize()}: Interpolating frames {gap_start}-{gap_end-1} "
                              f"(gap={gap_length}, from frame {last_valid_idx} to {gap_end}, "
                              f"distance={center_distance:.1f}px < {max_distance:.1f}px)")

                        # Total distance includes the step from last_valid to gap_start
                        # and from gap_end-1 to gap_end
                        total_steps = gap_end - last_valid_idx

                        for j in range(gap_start, gap_end):
                            # Linear interpolation: evenly distribute motion across all steps
                            alpha = (j - last_valid_idx) / total_steps
                            interp_bbox = (1 - alpha) * start_bbox + alpha * end_bbox
                            raw_data[j][bbox_key] = interp_bbox

                            # Create dummy keypoints
                            dummy_keypoints = np.zeros((21, 3))
                            dummy_keypoints[:, 2] = 0.5
                            raw_data[j][keyp_key] = dummy_keypoints
                            raw_data[j][conf_key] = 0.5
                            interpolated_count += 1

                        i = gap_end
                    else:
                        # Bboxes are too far apart - DON'T interpolate, treat as separate instances
                        print(f"  {hand_name.capitalize()}: Skipping interpolation for frames {gap_start}-{gap_end-1} "
                              f"(distance={center_distance:.1f}px > {max_distance:.1f}px - likely different hand instances)")

                        # Move to the next detection without filling the gap
                        i = gap_end
                else:
                    # Hand never reappeared within patience threshold
                    # Apply limited patience (only MAX_PATIENCE_WITHOUT_RETURN frames), then skip the rest
                    last_bbox = raw_data[last_valid_idx][bbox_key]
                    frames_to_fill = min(MAX_PATIENCE_WITHOUT_RETURN, len(raw_data) - gap_start)

                    if frames_to_fill > 0:
                        print(f"  {hand_name.capitalize()}: Applying static patience for frames {gap_start}-{gap_start + frames_to_fill - 1} "
                              f"(hand never reappeared, using last bbox from frame {last_valid_idx})")

                    for j in range(gap_start, gap_start + frames_to_fill):
                        raw_data[j][bbox_key] = last_bbox.copy()

                        # Create dummy keypoints
                        dummy_keypoints = np.zeros((21, 3))
                        dummy_keypoints[:, 2] = 0.5
                        raw_data[j][keyp_key] = dummy_keypoints
                        raw_data[j][conf_key] = 0.5
                        patience_applied_count += 1

                    # Skip ahead to find where hand reappears (or end of sequence)
                    next_valid_idx = None
                    for j in range(gap_start + frames_to_fill, len(raw_data)):
                        if raw_data[j][bbox_key] is not None:
                            next_valid_idx = j
                            break

                    # Jump to next valid detection or end of sequence
                    i = next_valid_idx if next_valid_idx is not None else len(raw_data)
            else:
                i += 1

    if interpolated_count > 0:
        print(f"  Interpolated {interpolated_count} frames where hand reappeared within patience threshold")
    if patience_applied_count > 0:
        print(f"  Applied static patience to {patience_applied_count} frames where hand never reappeared")
    if interpolated_count == 0 and patience_applied_count == 0:
        print(f"  No patience needed")

    # Remove spurious short motions (brief appearances after long absence)
    # This runs AFTER patience to filter out hallucinations that patience might have extended
    print("\n" + "-" * 80)
    print("Step 4: Removing spurious short motions")
    print("-" * 80)
    MIN_MOTION_DURATION = 30  # Motion must last at least this many frames
    MIN_ABSENCE_BEFORE = 30  # Must be absent for this long before
    MIN_ABSENCE_AFTER = 30  # Must be absent for this long after

    for hand_name in ['left', 'right']:
        bbox_key = f'{hand_name}_bbox'
        keyp_key = f'{hand_name}_keypoints'
        conf_key = f'{hand_name}_conf'

        # Find all continuous segments where hand is present
        segments = []
        start_idx = None

        for i, frame_data in enumerate(raw_data):
            if frame_data[bbox_key] is not None:
                if start_idx is None:
                    start_idx = i  # Start of a new segment
            else:
                if start_idx is not None:
                    # End of a segment
                    segments.append((start_idx, i - 1))
                    start_idx = None

        # Don't forget the last segment if it extends to the end
        if start_idx is not None:
            segments.append((start_idx, len(raw_data) - 1))

        # Check each segment
        removed_segments = []
        for seg_start, seg_end in segments:
            seg_duration = seg_end - seg_start + 1

            # Check absence before
            absence_before = seg_start  # frames before this segment

            # Check absence after
            absence_after = len(raw_data) - 1 - seg_end  # frames after this segment

            # If this is a short motion after long absence and before long absence, remove it
            if (seg_duration < MIN_MOTION_DURATION and absence_before >= MIN_ABSENCE_BEFORE and absence_after >= MIN_ABSENCE_AFTER):

                print(f"  {hand_name.capitalize()}: Removing spurious short motion at frames {seg_start}-{seg_end} "
                      f"(duration={seg_duration}, absence_before={absence_before}, absence_after={absence_after})")

                # Remove this segment
                for i in range(seg_start, seg_end + 1):
                    raw_data[i][bbox_key] = None
                    raw_data[i][keyp_key] = None
                    raw_data[i][conf_key] = 0.0

                removed_segments.append((seg_start, seg_end))

        if len(removed_segments) > 0:
            print(f"  {hand_name.capitalize()}: Removed {len(removed_segments)} spurious short motion segments")
        else:
            print(f"  {hand_name.capitalize()}: No spurious short motions found")

    # raw_data = interpolate_missing_bboxes(raw_data, max_gap=10)
    # # Bbox smoothing disabled - it was causing drift (e.g., frames 326->330)
    # raw_data = smooth_bbox_sequence(raw_data, window_size=5)

    # Re-check for overlaps after interpolation (interpolation can create new overlaps)
    print("\n" + "-" * 80)
    print("Step 4: Re-checking overlaps after interpolation")
    print("-" * 80)
    raw_data = detect_overlapping_bboxes(raw_data, iou_threshold=0.7)

    # Restore hands that were removed due to overlap, but ONLY if the "winner" hand was
    # subsequently invalidated by other checks (handedness consistency).
    # Logic: If LEFT was removed because RIGHT had higher confidence, but then RIGHT itself
    # was removed by handedness check, we should restore LEFT since it was the original detection.
    print("\n" + "-" * 80)
    print("Step 5: Restoring hands removed by overlap if winner was invalidated")
    print("-" * 80)
    restored_count = 0
    for i, (frame_data, orig) in enumerate(zip(raw_data, original_data)):
        for hand_name in ['left', 'right']:
            bbox_key = f'{hand_name}_bbox'
            keyp_key = f'{hand_name}_keypoints'
            conf_key = f'{hand_name}_conf'
            removal_flag = f'{hand_name}_removed_due_to_overlap_with'

            # Check if this hand was removed due to overlap
            if frame_data.get(removal_flag) is not None:
                other_hand = frame_data[removal_flag]  # The hand that "won" the overlap check
                other_bbox_key = f'{other_hand}_bbox'
                other_handedness_flag = f'{other_hand}_removed_by_handedness_check'

                # Check if the "winner" hand was subsequently removed by handedness check
                if frame_data.get(other_handedness_flag, False):
                    print(f"  Frame {i}: Restoring {hand_name} hand (winner '{other_hand}' was invalidated)")
                    # Restore original keypoints and confidence
                    if orig[keyp_key] is not None:
                        frame_data[bbox_key] = orig[bbox_key].copy()
                        frame_data[keyp_key] = orig[keyp_key].copy()
                        frame_data[conf_key] = orig[conf_key]
                    else:
                        # Create dummy keypoints if original didn't have them
                        dummy_keypoints = np.zeros((21, 3))
                        dummy_keypoints[:, 2] = 0.5
                        frame_data[keyp_key] = dummy_keypoints
                        frame_data[conf_key] = 0.5
                    # Clear the removal flag since we've restored it
                    frame_data[removal_flag] = None
                    restored_count += 1

    raw_data = fix_handedness_swaps_frame_to_frame(raw_data, iou_threshold=0.6)

    if restored_count > 0:
        print(f"  Restored {restored_count} hand instances")
    else:
        print(f"  No hands needed restoration")

    # Visualize before/after comparison
    dash_length = 10
    if vis_dir is not None:
        for i, frame_data in enumerate(raw_data):
            img_path = frame_data['img_path']
            img_cv2 = cv2.imread(img_path)
            img_vis = img_cv2.copy()

            # Draw original bboxes (dashed)
            orig = original_data[i]
            if orig['left_bbox'] is not None:
                x1, y1, x2, y2 = orig['left_bbox'].astype(int)
                # Draw dashed rectangle

                for j in range(x1, x2, dash_length * 2):
                    cv2.line(img_vis, (j, y1), (min(j + dash_length, x2), y1), (173, 216, 230), 2)
                    cv2.line(img_vis, (j, y2), (min(j + dash_length, x2), y2), (173, 216, 230), 2)
                for j in range(y1, y2, dash_length * 2):
                    cv2.line(img_vis, (x1, j), (x1, min(j + dash_length, y2)), (173, 216, 230), 2)
                    cv2.line(img_vis, (x2, j), (x2, min(j + dash_length, y2)), (173, 216, 230), 2)
                cv2.putText(img_vis, "L_before", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (173, 216, 230), 2)

            if orig['right_bbox'] is not None:
                x1, y1, x2, y2 = orig['right_bbox'].astype(int)
                for j in range(x1, x2, dash_length * 2):
                    cv2.line(img_vis, (j, y1), (min(j + dash_length, x2), y1), (144, 238, 144), 2)
                    cv2.line(img_vis, (j, y2), (min(j + dash_length, x2), y2), (144, 238, 144), 2)
                for j in range(y1, y2, dash_length * 2):
                    cv2.line(img_vis, (x1, j), (x1, min(j + dash_length, y2)), (144, 238, 144), 2)
                    cv2.line(img_vis, (x2, j), (x2, min(j + dash_length, y2)), (144, 238, 144), 2)
                cv2.putText(img_vis, "R_before", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (144, 238, 144), 2)

            # Draw cleaned bboxes (solid) and keypoints
            if frame_data['left_bbox'] is not None:
                x1, y1, x2, y2 = frame_data['left_bbox'].astype(int)
                cv2.rectangle(img_vis, (x1, y1), (x2, y2), (255, 0, 0), 3)  # Blue for left
                cv2.putText(img_vis, "L_after", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

                # Draw left hand keypoints in blue
                if frame_data['left_keypoints'] is not None:
                    for kp in frame_data['left_keypoints']:
                        x, y, conf = kp
                        if conf > 0.1:  # Only draw confident keypoints
                            cv2.circle(img_vis, (int(x), int(y)), 3, (255, 0, 0), -1)  # Blue circles

            if frame_data['right_bbox'] is not None:
                x1, y1, x2, y2 = frame_data['right_bbox'].astype(int)
                cv2.rectangle(img_vis, (x1, y1), (x2, y2), (0, 255, 0), 3)  # Green for right
                cv2.putText(img_vis, "R_after", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Draw right hand keypoints in green
                if frame_data['right_keypoints'] is not None:
                    for kp in frame_data['right_keypoints']:
                        x, y, conf = kp
                        if conf > 0.1:  # Only draw confident keypoints
                            cv2.circle(img_vis, (int(x), int(y)), 3, (0, 255, 0), -1)  # Green circles

            img_fn = os.path.splitext(os.path.basename(img_path))[0]
            cv2.imwrite(os.path.join(pass2_dir, f'{img_fn}_cleaned.jpg'), img_vis)

    # Create video from Pass 2 visualizations
    if vis_dir is not None:
        pass2_dir = os.path.join(vis_dir, 'pass2_cleaned_bboxes')
        video_path = os.path.join(vis_dir, 'pass2_cleaned_bboxes.mp4')
        create_video_from_images(pass2_dir, video_path, fps=30)

        # Plot bbox trajectories to visualize handedness consistency
        plot_bbox_trajectories(raw_data, vis_dir, filename='bbox_trajectories_after_cleaning.png')

    print("\n" + "=" * 80)
    print("PASS 2 Complete: Bbox sequences cleaned")
    print("=" * 80)

    return raw_data


# ============================================================================
# PASS 3: Run HaMeR on cleaned bboxes
# ============================================================================


def render_openpose(img, hand_keypoints):
    """Render hand keypoints (simplified version)."""
    # Implementation from original code (lines 718-730)
    return img  # Placeholder


def convert_crop_coords_to_orig_img(bbox, keypoints, crop_size):
    """Convert cropped coordinates to original image coordinates."""
    cx, cy, h = bbox[:, 0], bbox[:, 1], bbox[:, 2]
    keypoints *= h[..., None, None] / crop_size
    keypoints[:, :, 0] = (cx - h / 2)[..., None] + keypoints[:, :, 0]
    keypoints[:, :, 1] = (cy - h / 2)[..., None] + keypoints[:, :, 1]
    return keypoints


def run_hamer_on_cleaned_bboxes(raw_data, model, model_cfg, renderer, args):
    """Pass 3: Run HaMeR on cleaned bboxes."""
    print("\n" + "=" * 80)
    print("PASS 3: Running HaMeR on cleaned bboxes")
    print("=" * 80)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    results_dict = {}

    for frame_data in tqdm(raw_data, desc="Pass 3: Running HaMeR"):
        img_path = frame_data['img_path']
        frame_idx = frame_data['frame_idx']
        img_cv2 = cv2.imread(img_path)
        img_fn = os.path.splitext(os.path.basename(img_path))[0]

        # Prepare bboxes
        bboxes = []
        is_right = []
        vit_keypoints_list = []

        if frame_data['left_bbox'] is not None:
            bboxes.append(frame_data['left_bbox'])
            is_right.append(0)
            vit_keypoints_list.append(frame_data['left_keypoints'])

        if frame_data['right_bbox'] is not None:
            bboxes.append(frame_data['right_bbox'])
            is_right.append(1)
            vit_keypoints_list.append(frame_data['right_keypoints'])

        if len(bboxes) == 0:
            results_dict[img_path] = {'mano': [], 'cam_trans': [], 'tracked_ids': [], 'tracked_time': [], 'extra_data': [], 'tid': np.array([]), 'shot': 0}
            continue

        boxes = np.array(bboxes)
        right = np.array(is_right)

        # Validate and fix keypoints (create dummy keypoints if None)
        fixed_keypoints_list = []
        for i, kp in enumerate(vit_keypoints_list):
            if kp is None:
                # Create dummy keypoints with 0.5 confidence
                dummy_keypoints = np.zeros((21, 3))
                dummy_keypoints[:, 2] = 0.5
                fixed_keypoints_list.append(dummy_keypoints)
            elif not isinstance(kp, np.ndarray):
                raise ValueError(f"Frame {frame_idx} ({img_fn}): keypoints are not numpy array for hand {i}, got {type(kp)}")
            elif kp.shape != (21, 3):
                raise ValueError(f"Frame {frame_idx} ({img_fn}): keypoints have wrong shape {kp.shape} for hand {i}, expected (21, 3)")
            else:
                fixed_keypoints_list.append(kp)

        vit_keypoints = np.stack(fixed_keypoints_list, axis=0)

        # Create dataset
        dataset = ViTDetDataset(model_cfg, img_cv2, boxes, right, vit_keypoints, rescale_factor=args.rescale_factor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)

        # # Save HaMeR input patches for debugging
        # if args.render and args.res_folder is not None:
        #     hamer_input_dir = os.path.join(os.path.dirname(args.res_folder), f'hamer_input_patches_{model_cfg.EXTRA.FOCAL_LENGTH}')
        #     os.makedirs(hamer_input_dir, exist_ok=True)

        #     for hand_idx, (bbox, is_right_val) in enumerate(zip(boxes, right)):
        #         # Get the cropped patch from dataset
        #         sample = dataset[hand_idx]
        #         img_patch = sample['img_patch'].astype(np.uint8)  # This is a tensor (C, H, W) normalized

        #         # Denormalize and convert back to image
        #         print('img_patch_bgr: ', img_patch.shape)
        #         img_patch_bgr = cv2.cvtColor(img_patch, cv2.COLOR_RGB2BGR)

        #         hand_name = 'right' if is_right_val == 1 else 'left'
        #         patch_filename = f'{img_fn}_{hand_name}.jpg'
        #         cv2.imwrite(os.path.join(hamer_input_dir, patch_filename), img_patch_bgr)

        # Run HaMeR
        all_verts = []
        all_cam_t = []
        all_right = []
        all_mano_params = []
        all_pred_2d = []
        all_bboxes = []

        for batch in dataloader:
            batch = recursive_to(batch, device)
            with torch.no_grad():
                out = model(batch)

            multiplier = (2 * batch['right'] - 1)
            pred_cam = out['pred_cam']
            pred_cam[:, 1] = multiplier * pred_cam[:, 1]

            box_center = batch["box_center"].float()
            box_size = batch["box_size"].float()
            img_size = batch["img_size"].float()
            scaled_focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()

            # Compute camera translation (SLAHMR-style)
            batch_size = batch['img'].shape[0]
            pred_cam_t_full = torch.zeros(batch_size, 3, device=pred_cam.device)

            for n in range(batch_size):
                cam = pred_cam[n]
                H, W = img_size[n, 1], img_size[n, 0]
                focal = scaled_focal_length[n] if scaled_focal_length.ndim > 0 else scaled_focal_length
                cx, cy = box_center[n]
                scale = box_size[n]

                tz = 2 * focal / (scale * cam[0] + 1e-6)
                tx = cam[1] + tz / focal * (cx - W / 2)
                ty = cam[2] + tz / focal * (cy - H / 2)

                pred_cam_t_full[n] = torch.tensor([tx, ty, tz], device=pred_cam.device)

            pred_cam_t_full = pred_cam_t_full.detach().cpu().numpy()

            for n in range(batch_size):
                verts = out['pred_vertices'][n].detach().cpu().numpy()
                pred_joints = out['pred_keypoints_2d'][n].detach().cpu().numpy()
                is_right_val = int(batch['right'][n].cpu().numpy())
                verts[:, 0] = (2 * is_right_val - 1) * verts[:, 0]
                pred_joints[:, 0] = (2 * is_right_val - 1) * pred_joints[:, 0]
                cam_t = pred_cam_t_full[n]

                all_verts.append(verts)
                all_cam_t.append(cam_t)
                all_right.append(is_right_val)
                all_pred_2d.append(pred_joints)
                all_bboxes.append(batch['bbox'][n].detach().cpu().numpy())

                mano_params = out['pred_mano_params'][n]
                mano_params['is_right'] = is_right_val
                all_mano_params.append(mano_params)

        # Convert 2D keypoints to original image coordinates
        if len(all_pred_2d) > 0:
            all_pred_2d_np = np.stack(all_pred_2d)
            all_bboxes_np = np.stack(all_bboxes)

            # Add confidence column
            v = np.ones((all_pred_2d_np.shape[0], all_pred_2d_np.shape[1], 1))
            all_pred_2d_np = np.concatenate((all_pred_2d_np, v), axis=-1)

            # Convert from crop coords to original image coords
            all_pred_2d_np = model_cfg.MODEL.IMAGE_SIZE * (all_pred_2d_np + 0.5)
            all_pred_2d_np = convert_crop_coords_to_orig_img(bbox=all_bboxes_np, keypoints=all_pred_2d_np, crop_size=model_cfg.MODEL.IMAGE_SIZE)
            all_pred_2d_np[:, :, -1] = 1  # Set all confidences to 1

            extra_data = [all_pred_2d_np[i].tolist() for i in range(len(all_pred_2d_np))]
        else:
            extra_data = []

        # Store results
        results_dict[img_path] = {
            'mano': all_mano_params,
            'cam_trans': all_cam_t,
            'tracked_ids': all_right,
            'tracked_time': [0] * len(all_right),
            'extra_data': extra_data,
            'tid': np.array(all_right),
            'shot': 0
        }

        # Render hand meshes if requested
        if args.render and len(all_verts) > 0 and renderer is not None:
            LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)
            misc_args = dict(
                mesh_base_color=LIGHT_BLUE,
                scene_bg_color=(1, 1, 1),
                focal_length=scaled_focal_length,
            )

            cam_view, _ = renderer.render_rgba_multiple(all_verts, cam_t=all_cam_t, render_res=img_size[0], is_right=all_right, **misc_args)

            # Overlay on original image
            input_img = img_cv2.astype(np.float32)[:, :, ::-1] / 255.0
            input_img = np.concatenate([input_img, np.ones_like(input_img[:, :, :1])], axis=2)
            input_img_overlay = input_img[:, :, :3] * (1 - cam_view[:, :, 3:]) + cam_view[:, :, :3] * cam_view[:, :, 3:]

            # Save rendered result
            render_path = os.path.join(os.path.dirname(args.res_folder), f'render_all_{model_cfg.EXTRA.FOCAL_LENGTH}')
            os.makedirs(render_path, exist_ok=True)
            cv2.imwrite(os.path.join(render_path, f'{img_fn}.jpg'), 255 * input_img_overlay[:, :, ::-1])

    # Create video from Pass 3 rendered results
    render_path = os.path.join(os.path.dirname(args.res_folder), f'render_all_{model_cfg.EXTRA.FOCAL_LENGTH}')
    if os.path.exists(render_path):
        video_path = os.path.join(os.path.dirname(args.res_folder), f'render_all_{model_cfg.EXTRA.FOCAL_LENGTH}.mp4')
        create_video_from_images(render_path, video_path, fps=30)

    print("\n" + "=" * 80)
    print("PASS 3 Complete: HaMeR reconstruction finished")
    print("=" * 80)

    return results_dict


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description='HaMeR with 3-pass architecture using YOLO hand detector')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to pretrained model checkpoint')
    parser.add_argument('--img_folder', type=str, default='images', help='Folder with input images')
    parser.add_argument('--out_folder', type=str, default='out_demo', help='Output folder to save rendered results')
    parser.add_argument('--res_folder', type=str, help='Output folder to save rendered results')
    parser.add_argument('--side_view', dest='side_view', action='store_true', default=False, help='If set, render side view also')
    parser.add_argument('--full_frame', dest='full_frame', action='store_true', default=True, help='If set, render all people together also')
    parser.add_argument('--save_mesh', dest='save_mesh', action='store_true', default=False, help='If set, save meshes to disk also')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inference/fitting')
    parser.add_argument('--rescale_factor', type=float, default=1.3, help='Factor for padding the bbox')
    parser.add_argument('--file_type', nargs='+', default=['*.jpg', '*.png'], help='List of file extensions to consider')
    parser.add_argument('--conf', type=float, default=2.0, help='Factor for padding the bbox')
    parser.add_argument('--type', type=str, default='EgoDexter', help='Path to pretrained model checkpoint')
    parser.add_argument('--render', dest='render', action='store_true', default=False, help='If set, render side view also')
    parser.add_argument('--yolo_model',
                        type=str,
                        # default='./pretrained_models/detector.withHOI.pt',
                        default='./pretrained_models/detector.pt',
                        help='Path to YOLO hand detector model (like WiLoR detector.pt)')

    args = parser.parse_args()

    # Setup
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    img_folder = Path(args.img_folder)

    # Get all images matching file_type patterns
    img_paths = sorted([img for end in args.file_type for img in img_folder.glob(end)])

    print(f"Found {len(img_paths)} images in {img_folder}")

    # Load models
    print("\nLoading models...")
    CACHE_DIR_HAMER = args.checkpoint
    download_models(CACHE_DIR_HAMER)
    model, model_cfg = load_hamer(args.checkpoint)
    model = model.to(device)
    model.eval()

    # Load YOLO hand detector (like WiLoR)
    print(f"\nLoading YOLO hand detector: {args.yolo_model}")
    yolo_detector = YOLO(args.yolo_model)
    yolo_detector.to(device)

    # Load renderer
    renderer = Renderer(model_cfg, faces=model.mano.faces)

    # Create visualization directory
    vis_dir = None
    if args.render and args.res_folder is not None:
        vis_dir = os.path.join(os.path.dirname(args.res_folder), f'bbox_vis_{model_cfg.EXTRA.FOCAL_LENGTH}')
        os.makedirs(vis_dir, exist_ok=True)
        print(f"\nBbox visualizations will be saved to: {vis_dir}")

    # PASS 1: Extract raw bboxes using YOLO
    raw_data = extract_raw_bboxes(img_paths, yolo_detector, vis_dir=vis_dir)

    # PASS 2: Clean bbox sequences
    cleaned_data = clean_bbox_sequences(raw_data, vis_dir=vis_dir)

    # cleaned_data = extract_raw_bboxes_from_keypoints_json(img_paths, img_folder.name, hand_type='right', vis_dir='debug')

    # PASS 3: Run HaMeR
    results = run_hamer_on_cleaned_bboxes(cleaned_data, model, model_cfg, renderer, args)

    # Save results
    output_path = Path(args.res_folder)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)

    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()
