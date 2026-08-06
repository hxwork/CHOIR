import argparse
import io
import multiprocessing as mp
import os
import random
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

import cv2
import imageio
import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt
from PIL import Image
from scipy.interpolate import CubicSpline
from scipy.signal import find_peaks
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from ultralytics import YOLO

from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Global cache for models in worker processes
_WORKER_MODELS = {}

if torch.cuda.is_available():
    # no need to define autocast, directly use torch.amp.autocast
    from torch.amp import autocast


def calculate_bbox_iou(boxA, boxB):
    """Calculates the Intersection over Union (IoU) for two bounding boxes."""
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle, ensuring it's non-negative
    interArea = max(0, xB - xA) * max(0, yB - yA)

    # compute the area of both bounding boxes
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the intersection area
    iou = interArea / float(boxAArea + boxBArea - interArea) if (boxAArea + boxBArea - interArea) > 0 else 0.0

    # return the intersection over union value
    return iou


def calculate_mask_iou(mask1, mask2):
    """Calculates the Intersection over Union (IoU) for two binary masks."""
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union > 0 else 0.0


def deduplicate_masks(masks):
    """
    Deduplicates a list of masks using a two-stage process:
    1. Fast Non-Maximum Suppression (NMS) on bounding boxes.
    2. Slower but more accurate IoU check on the actual mask pixels.
    """
    if len(masks) <= 1:
        return masks

    # --- Stage 1: Bounding Box NMS ---
    boxes_xywh = torch.tensor([m['bbox'] for m in masks], dtype=torch.float32)
    scores = torch.tensor([m['predicted_iou'] for m in masks], dtype=torch.float32)

    boxes_xyxy = boxes_xywh.clone()
    boxes_xyxy[:, 2] += boxes_xyxy[:, 0]
    boxes_xyxy[:, 3] += boxes_xyxy[:, 1]

    bbox_iou_thresh = 0.7  # Lowered threshold for more aggressive filtering
    indices_after_nms = torchvision.ops.nms(boxes_xyxy, scores, bbox_iou_thresh)

    remaining_masks = [masks[i] for i in indices_after_nms.cpu().numpy()]

    # --- Stage 2: Mask IoU Deduplication ---
    if len(remaining_masks) <= 1:
        return remaining_masks

    # Sort remaining masks by score to prioritize keeping the best ones
    remaining_masks.sort(key=lambda m: m['predicted_iou'], reverse=True)

    final_masks = []
    if not remaining_masks:
        return final_masks

    final_masks.append(remaining_masks[0])
    mask_iou_thresh = 0.9  # High threshold to remove only near-identical masks

    for mask_to_check in remaining_masks[1:]:
        is_duplicate = False
        for final_mask in final_masks:
            iou = calculate_mask_iou(mask_to_check['segmentation'], final_mask['segmentation'])
            if iou > mask_iou_thresh:
                is_duplicate = True
                break
        if not is_duplicate:
            final_masks.append(mask_to_check)

    return final_masks


def is_mask_valid(mask, min_area=1600, max_area=360000, max_components=2, max_holes=3):
    """
    Checks if a mask is valid based on area, number of components, and holes.
    A valid mask has:
    - Area greater than or equal to min_area.
    - Number of connected components less than or equal to max_components.
    - Number of holes less than or equal to max_holes.
    """
    # 1. Area check
    if np.sum(mask) < min_area or np.sum(mask) > max_area:
        return False

    mask_uint8 = mask.astype(np.uint8)

    # 2. Component check for disconnected regions
    num_labels, _, _, _ = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)
    if (num_labels - 1) > max_components:  # Subtract 1 for the background label
        return False

    # 3. Hole check
    contours, hierarchy = cv2.findContours(mask_uint8, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return True  # No contours, so no holes

    hole_count = 0
    # Hierarchy has shape (1, num_contours, 4)
    for i in range(hierarchy.shape[1]):
        # A contour is a hole if it has a parent.
        if hierarchy[0, i, 3] != -1:
            hole_count += 1

    if hole_count > max_holes:
        return False

    return True


def generate_masks_from_prompts(predictor, first_frame, right_hl_coords, left_hl_coords, points_per_side=14, spacing=20):
    """
    Generates tracking masks by prompting SAM with a grid of points around the highlight coordinates.
    """
    masks_to_track = []
    all_grid_points = []
    h, w, _ = first_frame.shape
    frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)

    # predictor.set_image(frame_rgb)

    def generate_and_select_for_hand(hl_coords, base_obj_id):
        if hl_coords is None:
            return [], []

        # 1. Generate the initial grid of points around the highlight coordinate.
        cx, cy = hl_coords
        offset = (points_per_side // 2) * spacing
        axis_offsets = np.arange(points_per_side) * spacing - offset
        x_coords = cx + axis_offsets
        y_coords = cy + axis_offsets
        xx, yy = np.meshgrid(x_coords, y_coords)
        grid_points = np.stack([xx.ravel(), yy.ravel()], axis=-1).astype(int)

        # 2. Calculate the necessary shift to move the entire grid inside the image boundaries.
        min_x, min_y = grid_points.min(axis=0)
        max_x, max_y = grid_points.max(axis=0)

        shift_x, shift_y = 0, 0
        if min_x < 0:
            shift_x = -min_x
        elif max_x >= w:
            shift_x = w - 1 - max_x

        if min_y < 0:
            shift_y = -min_y
        elif max_y >= h:
            shift_y = h - 1 - max_y

        # 3. Apply the shift to all points in the grid.
        if shift_x != 0 or shift_y != 0:
            grid_points += np.array([shift_x, shift_y])

        # 4. Predict masks in batches to manage memory
        batch_size = 16
        hand_masks = []
        num_points = grid_points.shape[0]

        for i in range(0, num_points, batch_size):
            batch_points = grid_points[i:i + batch_size]
            batch_num_points = batch_points.shape[0]

            # Reshape points and labels for batch prediction
            point_coords_batch = batch_points.reshape(batch_num_points, 1, 1, 2)
            point_coords_batch_list = [point_coords_batch[j] for j in range(batch_num_points)]
            point_labels_batch = np.ones((batch_num_points, 1, 1), dtype=np.int32)
            point_labels_batch_list = [point_labels_batch[j] for j in range(batch_num_points)]

            # The predictor expects a list of batches
            predictor.set_image_batch([frame_rgb] * batch_num_points)
            masks_batch, scores_batch, _ = predictor.predict_batch(point_coords_batch_list, point_labels_batch_list, box_batch=None, multimask_output=True)

            # Process the results for the current batch
            # Select the best single mask per object in the batch
            batch_best_masks = []
            for masks, scores in zip(masks_batch, scores_batch):
                best_idx = np.argmax(scores)
                batch_best_masks.append(masks[best_idx])

            best_masks = np.stack(batch_best_masks, axis=0)

            # Since we pick one mask per prompt, the scores are 1D
            best_scores = [np.max(s) for s in scores_batch]

            for best_mask, best_score in zip(best_masks, best_scores):
                if not is_mask_valid(best_mask):
                    continue

                y_indices, x_indices = np.where(best_mask)
                if y_indices.size == 0:
                    continue
                x_min, x_max = x_indices.min(), x_indices.max()
                y_min, y_max = y_indices.min(), y_indices.max()
                bbox_xywh = [int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)]

                mask_data = {'segmentation': best_mask, 'predicted_iou': float(best_score), 'bbox': bbox_xywh}
                hand_masks.append(mask_data)

        # 5. Deduplicate masks generated for this hand
        unique_hand_masks = deduplicate_masks(hand_masks)

        # 6. Format for output
        formatted_masks = []
        for i, mask_data in enumerate(unique_hand_masks):
            obj_id = base_obj_id * 100 + i
            formatted_masks.append({'mask_data': mask_data, 'obj_id': obj_id})
        return formatted_masks, grid_points.tolist()

    right_masks, right_grid_points = generate_and_select_for_hand(right_hl_coords, 1)
    masks_to_track.extend(right_masks)
    all_grid_points.extend(right_grid_points)

    left_masks, left_grid_points = generate_and_select_for_hand(left_hl_coords, 2)
    masks_to_track.extend(left_masks)
    all_grid_points.extend(left_grid_points)

    return masks_to_track, all_grid_points


def sam2_tracking(predictor, frames, masks_to_track):
    ann_frame_idx = 0

    if not masks_to_track:
        return {}

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float32):
        state = predictor.init_state(frames)
        predictor.reset_state(state)

        all_obj_ids = []
        for m_info in masks_to_track:
            mask_data = m_info['mask_data']
            obj_id = m_info['obj_id']
            all_obj_ids.append(obj_id)

            # Use the bounding box of the mask to initialize tracking
            bbox = mask_data['bbox']  # XYWH format
            x, y, w, h = bbox
            box_xyxy = np.array([x, y, x + w, y + h], dtype=np.float32)

            _, _, _ = predictor.add_new_points_or_box(
                inference_state=state,
                frame_idx=ann_frame_idx,
                obj_id=obj_id,
                box=box_xyxy,
            )

        video_segments = {}
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(state):
            masks = (out_mask_logits > 0.0).cpu().numpy()
            frame_segments = video_segments.setdefault(out_frame_idx, {})
            for i, out_obj_id in enumerate(out_obj_ids):
                if out_obj_id in all_obj_ids:
                    frame_segments[out_obj_id] = masks[i]

    return video_segments


def are_trajectories_similar(traj1, traj2, threshold, min_common_frames=10):
    """
    Checks if two trajectories are similar by calculating the average distance
    between their points on common frames.
    """
    common_frames = sorted(list(set(traj1.keys()) & set(traj2.keys())))

    if len(common_frames) < min_common_frames:
        return False  # Not enough overlap to compare reliably

    distances = [np.linalg.norm(traj1[frame] - traj2[frame]) for frame in common_frames]
    avg_distance = np.mean(distances)

    return avg_distance < threshold


def select_interactive_tracks(video_segments, right_hand_traj, left_hand_traj, right_hl_frame, left_hl_frame):
    """
    Selects and merges object tracks that are interacting with hands. This is a
    two-stage process:
    1. Filter for objects that are moving significantly.
    2. From the moving objects, select those closest to the hand in a time window.
    """
    if not video_segments:
        return {}, None, None

    # 1. Extract object trajectories
    sam_trajectories = {}
    image_shape = None
    for frame_idx, segments in sorted(video_segments.items()):
        for obj_id, mask in segments.items():
            if obj_id not in sam_trajectories:
                sam_trajectories[obj_id] = {}
            current_mask = mask.squeeze(0) if mask.ndim == 3 and mask.shape[0] == 1 else mask
            if image_shape is None and current_mask.ndim == 2:
                image_shape = current_mask.shape
            moments = cv2.moments(current_mask.astype(np.uint8))
            if moments["m00"] > 0:
                cx = int(moments["m10"] / moments["m00"])
                cy = int(moments["m01"] / moments["m00"])
                sam_trajectories[obj_id][frame_idx] = np.array([cx, cy])

    # 2. Pre-filter for significant movement independently for each hand
    right_object_trajs = {oid: traj for oid, traj in sam_trajectories.items() if oid // 100 == 1}
    left_object_trajs = {oid: traj for oid, traj in sam_trajectories.items() if oid // 100 == 2}

    def _find_significant_by_gap(trajs):
        track_displacements = {}
        for obj_id, trajectory in trajs.items():
            if len(trajectory) >= 10:
                points = np.array(list(trajectory.values()))
                from scipy.spatial.distance import pdist
                if len(points) > 1:
                    max_dist = np.max(pdist(points))
                    track_displacements[obj_id] = max_dist

        if not track_displacements:
            return set()

        sorted_displacements = sorted(track_displacements.items(), key=lambda item: item[1], reverse=True)
        displacements = [d for oid, d in sorted_displacements]
        gaps = [displacements[i - 1] - displacements[i] for i in range(1, len(displacements))]

        if gaps:
            max_gap_idx = np.argmax(gaps)
            return {oid for oid, d in sorted_displacements[:max_gap_idx + 1]}
        elif sorted_displacements:
            return {sorted_displacements[0][0]}
        return set()

    significant_right_ids = _find_significant_by_gap(right_object_trajs)
    significant_left_ids = _find_significant_by_gap(left_object_trajs)

    # 3. Find interactive tracks from the significantly moving ones
    def find_interactive_tracks_by_gap(hand_traj, object_ids, start_frame, window_size=60):
        if not hand_traj or not object_ids or start_frame is None:
            return [], None

        # Calculate avg min-distance for each object in the window
        avg_distances = {}
        end_frame = start_frame + window_size

        for obj_id in object_ids:
            frame_min_distances = []

            # Define the frames to analyze for this object
            frames_to_check = [f for f in hand_traj.keys() if start_frame <= f < end_frame and f in video_segments and obj_id in video_segments[f]]

            if len(frames_to_check) < 5:  # Need minimal overlap
                continue

            for frame in frames_to_check:
                hand_point = hand_traj[frame]
                mask = video_segments[frame][obj_id].squeeze()

                # Use distance transform for efficient distance calculation
                inverted_mask = (~mask.astype(bool)).astype(np.uint8)
                dist_transform = cv2.distanceTransform(inverted_mask, cv2.DIST_L2, 3)

                # Get the distance at the hand's coordinates
                hx, hy = int(hand_point[0]), int(hand_point[1])
                h_img, w_img = dist_transform.shape
                if 0 <= hy < h_img and 0 <= hx < w_img:
                    dist = dist_transform[hy, hx]
                    frame_min_distances.append(dist)

            if frame_min_distances:
                avg_distances[obj_id] = np.mean(frame_min_distances)

        if not avg_distances:
            return [], None

        # Sort objects by their average distance
        sorted_objects = sorted(avg_distances.items(), key=lambda item: item[1])

        # Find the largest gap between consecutive sorted distances
        gaps = [sorted_objects[i][1] - sorted_objects[i - 1][1] for i in range(1, len(sorted_objects))]

        interactive_ids = []
        if gaps:
            max_gap_idx = np.argmax(gaps)
            max_gap_value = gaps[max_gap_idx]

            # A gap is significant if it's larger than the distance to the closest object.
            # This helps differentiate a real gap from small variations within a cluster.
            if max_gap_value > sorted_objects[0][1]:
                # All objects before the largest gap are considered interactive
                interactive_ids = [obj_id for obj_id, dist in sorted_objects[:max_gap_idx + 1]]
            else:
                # No significant gap found, assume all objects form a single interactive cluster
                interactive_ids = [obj_id for obj_id, dist in sorted_objects]
        elif sorted_objects:
            # If no gaps (only one object), it is the interactive one
            interactive_ids = [sorted_objects[0][0]]

        # The closest one is still useful for highlighting
        closest_id = sorted_objects[0][0] if sorted_objects else None

        return interactive_ids, closest_id

    right_interactive_ids, best_right_track_id = find_interactive_tracks_by_gap(right_hand_traj, significant_right_ids, right_hl_frame)
    left_interactive_ids, best_left_track_id = find_interactive_tracks_by_gap(left_hand_traj, significant_left_ids, left_hl_frame)

    # 4. Filter and merge segments for all interactive tracks
    filtered_segments = {}
    for frame_idx, segments in sorted(video_segments.items()):
        frame_segs = {}

        # Process right hand
        right_merged_mask = None
        for obj_id in right_interactive_ids:
            if obj_id in segments:
                mask = segments[obj_id].squeeze()
                if right_merged_mask is None:
                    right_merged_mask = np.zeros_like(mask, dtype=bool)
                right_merged_mask = np.logical_or(right_merged_mask, mask)
        if right_merged_mask is not None and np.any(right_merged_mask):
            frame_segs[1] = right_merged_mask.reshape(1, *right_merged_mask.shape)

        # Process left hand
        left_merged_mask = None
        for obj_id in left_interactive_ids:
            if obj_id in segments:
                mask = segments[obj_id].squeeze()
                if left_merged_mask is None:
                    left_merged_mask = np.zeros_like(mask, dtype=bool)
                left_merged_mask = np.logical_or(left_merged_mask, mask)
        if left_merged_mask is not None and np.any(left_merged_mask):
            frame_segs[2] = left_merged_mask.reshape(1, *left_merged_mask.shape)

        if frame_segs:
            filtered_segments[frame_idx] = frame_segs

    return filtered_segments, best_right_track_id, best_left_track_id


def visualize_sam_tracking(frames, video_segments, output_path):
    """
    Visualizes SAM-2 tracking results and saves as a video.
    """
    with imageio.get_writer(output_path, fps=30) as writer:
        for frame_idx, frame in enumerate(tqdm(frames, desc="Visualizing SAM-2 Tracking")):
            if frame_idx in video_segments:
                # Create a blank image for the masks
                mask_overlay = np.zeros_like(frame, dtype=np.uint8)

                for obj_id, mask in video_segments[frame_idx].items():
                    mask = mask.squeeze()
                    if obj_id % 10 == 1:  # Right hand tracks (1, 11, 21...)
                        color = (0, 0, 255)  # Red
                    elif obj_id % 10 == 2:  # Left hand tracks (2, 12, 22...)
                        color = (0, 255, 0)  # Green
                    else:
                        color = (255, 255, 255)  # Default to white for any other case
                    mask_overlay[mask] = color

                # Blend the overlay with the original frame
                # Apply transparency to the mask overlay
                alpha = 0.5
                img = cv2.addWeighted(mask_overlay, alpha, frame, 1 - alpha, 0)
            else:
                img = frame

            # Handle macroblock size for video encoding
            h, w, _ = img.shape
            macro_block_size = 16
            padded_h = (h + macro_block_size - 1) // macro_block_size * macro_block_size
            padded_w = (w + macro_block_size - 1) // macro_block_size * macro_block_size

            if padded_h != h or padded_w != w:
                padded_img = np.zeros((padded_h, padded_w, 3), dtype=np.uint8)
                padded_img[:h, :w] = img
                img_to_write = padded_img
            else:
                img_to_write = img

            writer.append_data(cv2.cvtColor(img_to_write, cv2.COLOR_BGR2RGB))
    print(f"SAM-2 tracking visualization saved to {output_path}")


def visualize_all_sam_tracks_for_debug(frames, video_segments, best_right_id, best_left_id, output_path):
    """
    Visualizes all SAM-2 tracking results for debugging, highlighting the best-matching tracks.
    """

    # Using a simple hash function on obj_id to get a somewhat consistent color
    def get_color(obj_id):
        base_id = obj_id // 100
        point_id = obj_id % 100
        random.seed(point_id)  # Seed with the point id for consistency
        if base_id == 1:  # Right hand grid points (non-best) -> Shades of Blue/Magenta
            return (random.randint(100, 255), 0, random.randint(100, 255))
        elif base_id == 2:  # Left hand grid points (non-best) -> Shades of Yellow/Cyan
            return (random.randint(100, 255), random.randint(150, 255), 0)
        else:
            return (255, 255, 255)  # White

    with imageio.get_writer(output_path, fps=30) as writer:
        for frame_idx, frame in enumerate(tqdm(frames, desc="Visualizing All SAM-2 Tracks (Debug)")):
            if frame_idx in video_segments:
                mask_overlay = np.zeros_like(frame, dtype=np.uint8)

                for obj_id, mask in video_segments[frame_idx].items():
                    mask_squeezed = mask.squeeze(0)
                    if mask_squeezed.ndim != 2:
                        continue

                    if obj_id == best_right_id:
                        color = (0, 0, 255)  # Bright Red
                    elif obj_id == best_left_id:
                        color = (0, 255, 0)  # Bright Green
                    else:
                        color = get_color(obj_id)

                    mask_overlay[mask_squeezed] = color

                alpha = 0.6
                img = cv2.addWeighted(mask_overlay, alpha, frame, 1 - alpha, 0)
            else:
                img = frame

            # Handle macroblock size for video encoding
            h, w, _ = img.shape
            macro_block_size = 16
            padded_h = (h + macro_block_size - 1) // macro_block_size * macro_block_size
            padded_w = (w + macro_block_size - 1) // macro_block_size * macro_block_size

            if padded_h != h or padded_w != w:
                padded_img = np.zeros((padded_h, padded_w, 3), dtype=np.uint8)
                padded_img[:h, :w] = img
                img_to_write = padded_img
            else:
                img_to_write = img

            writer.append_data(cv2.cvtColor(img_to_write, cv2.COLOR_BGR2RGB))
    print(f"SAM-2 debug visualization saved to {output_path}")


def interpolate_tracks(tracks):
    """
    Fill missing frames in hand tracks using cubic-spline interpolation.

    Args:
        tracks: Mapping from frame index to bounding box [x1, y1, x2, y2].

    Returns:
        A copy of the tracks with interior gaps interpolated.
    """
    if len(tracks) == 0:
        return tracks

    # Collect observed frame indices.
    frame_ids = sorted(tracks.keys())

    if len(frame_ids) < 2:
        # At least two observations are required for interpolation.
        return tracks

    # Determine the complete observed frame range.
    min_frame = frame_ids[0]
    max_frame = frame_ids[-1]

    # Find missing frames within the observed range.
    all_frames = set(range(min_frame, max_frame + 1))
    existing_frames = set(frame_ids)
    missing_frames = sorted(all_frames - existing_frames)

    if len(missing_frames) == 0:
        # Return early when the trajectory is already complete.
        return tracks

    # Convert boxes to center coordinates, width, and height.
    frame_list = []
    cx_list = []
    cy_list = []
    w_list = []
    h_list = []

    for frame_id in frame_ids:
        bbox = tracks[frame_id]
        x1, y1, x2, y2 = bbox

        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1

        frame_list.append(frame_id)
        cx_list.append(cx)
        cy_list.append(cy)
        w_list.append(w)
        h_list.append(h)

    # Convert interpolation inputs to NumPy arrays.
    frame_array = np.array(frame_list)
    cx_array = np.array(cx_list)
    cy_array = np.array(cy_list)
    w_array = np.array(w_list)
    h_array = np.array(h_list)

    # Fit cubic splines for center, width, and height.
    cs_cx = CubicSpline(frame_array, cx_array)
    cs_cy = CubicSpline(frame_array, cy_array)
    cs_w = CubicSpline(frame_array, w_array)
    cs_h = CubicSpline(frame_array, h_array)

    # Interpolate each missing interior frame.
    interpolated_tracks = tracks.copy()
    for missing_frame in missing_frames:
        # Only interpolate gaps bounded by observed boxes.
        if missing_frame > min_frame and missing_frame < max_frame:
            cx_interp = cs_cx(missing_frame)
            cy_interp = cs_cy(missing_frame)
            w_interp = cs_w(missing_frame)
            h_interp = cs_h(missing_frame)

            # Keep interpolated box dimensions positive.
            w_interp = max(w_interp, 1.0)
            h_interp = max(h_interp, 1.0)

            # Convert center and size back to corner coordinates.
            x1_interp = cx_interp - w_interp / 2
            y1_interp = cy_interp - h_interp / 2
            x2_interp = cx_interp + w_interp / 2
            y2_interp = cy_interp + h_interp / 2

            interpolated_tracks[missing_frame] = np.array([x1_interp, y1_interp, x2_interp, y2_interp])

    return interpolated_tracks


def interpolate_tip_positions(original_tracks):
    """
    Calculates and interpolates the average position of fingertips for a dense trajectory.

    Args:
        original_tracks: The original detected tracks with keypoint data.

    Returns:
        A dictionary of {frame_id: (x, y)} for the dense trajectory.
    """
    tip_indices = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky fingertips

    frame_list = []
    avg_x_list = []
    avg_y_list = []

    sorted_frames = sorted(original_tracks.keys())

    for frame_id in sorted_frames:
        full_info = original_tracks[frame_id]
        keypoints_flat = full_info[5:]
        keypoints = np.array(keypoints_flat).reshape(-1, 3)

        tips = keypoints[tip_indices]
        valid_tips = tips[tips[:, 2] > 0.5]

        if len(valid_tips) > 0:
            avg_tip_norm = np.mean(valid_tips[:, :2], axis=0)
            frame_list.append(frame_id)
            avg_x_list.append(avg_tip_norm[0])
            avg_y_list.append(avg_tip_norm[1])

    if len(frame_list) < 2:
        return {}  # Not enough data to interpolate

    frame_array = np.array(frame_list)
    avg_x_array = np.array(avg_x_list)
    avg_y_array = np.array(avg_y_list)

    cs_x = CubicSpline(frame_array, avg_x_array)
    cs_y = CubicSpline(frame_array, avg_y_array)

    min_frame = frame_list[0]
    max_frame = frame_list[-1]

    interpolated_points = {}
    for frame_id in range(min_frame, max_frame + 1):
        x_interp = cs_x(frame_id)
        y_interp = cs_y(frame_id)
        interpolated_points[frame_id] = (int(x_interp), int(y_interp))

    return interpolated_points


def interpolate_root_joint_positions(original_tracks):
    """
    Calculates and interpolates the position of the root joint (wrist) for a dense trajectory.

    Args:
        original_tracks: The original detected tracks with keypoint data.

    Returns:
        A dictionary of {frame_id: (x, y)} for the dense root joint trajectory.
    """
    root_joint_index = 0

    frame_list = []
    x_list = []
    y_list = []

    sorted_frames = sorted(original_tracks.keys())

    for frame_id in sorted_frames:
        full_info = original_tracks[frame_id]
        keypoints_flat = full_info[5:]
        keypoints = np.array(keypoints_flat).reshape(-1, 3)

        root_joint = keypoints[root_joint_index]

        if root_joint[2] > 0.5:  # Check confidence
            frame_list.append(frame_id)
            x_list.append(root_joint[0])
            y_list.append(root_joint[1])

    if len(frame_list) < 2:
        return {}  # Not enough data to interpolate

    frame_array = np.array(frame_list)
    x_array = np.array(x_list)
    y_array = np.array(y_list)

    cs_x = CubicSpline(frame_array, x_array)
    cs_y = CubicSpline(frame_array, y_array)

    min_frame = frame_list[0]
    max_frame = frame_list[-1]

    interpolated_points = {}
    for frame_id in range(min_frame, max_frame + 1):
        x_interp = cs_x(frame_id)
        y_interp = cs_y(frame_id)
        interpolated_points[frame_id] = (int(x_interp), int(y_interp))

    return interpolated_points


def visualize_interpolated_tracks(
    frames,
    right_tracks,
    left_tracks,
    original_right_tracks,
    original_left_tracks,
    right_root_points,
    left_root_points,
    right_tip_points,
    left_tip_points,
    seq_folder,
    args,
):
    """
    Visualize interpolated hand tracks with a dynamic speed plot overlaid on the video.

    Args:
        frames: list of video frames (numpy arrays)
        right_tracks, left_tracks: dict, interpolated bbox tracks for hands
        original_right_tracks, original_left_tracks: dict, original tracks with keypoints
        right_root_points, left_root_points: dict, interpolated root joint positions for speed calculation
        right_tip_points, left_tip_points: dict, interpolated tip positions for path drawing
        seq_folder: output folder for the video sequence
    """

    # Pre-calculate highlight points, which are needed regardless of visualization
    right_hl_frame, right_hl_coords = find_speed_valley_point(right_root_points, right_tip_points)
    left_hl_frame, left_hl_coords = find_speed_valley_point(left_root_points, left_tip_points)

    if args.save_visualizations:
        output_video = f'{seq_folder}/interpolated_tracks.mp4'
        os.makedirs(os.path.dirname(output_video), exist_ok=True)

        # Pre-calculate speeds
        right_speeds = {}
        if len(right_root_points) > 1:
            frame_ids = sorted(right_root_points.keys())
            points = [right_root_points[fid] for fid in frame_ids]
            velocities = np.linalg.norm(np.array(points[1:]) - np.array(points[:-1]), axis=1)
            for i, fid in enumerate(frame_ids[1:]):
                right_speeds[fid] = velocities[i]

        left_speeds = {}
        if len(left_root_points) > 1:
            frame_ids = sorted(left_root_points.keys())
            points = [left_root_points[fid] for fid in frame_ids]
            velocities = np.linalg.norm(np.array(points[1:]) - np.array(points[:-1]), axis=1)
            for i, fid in enumerate(frame_ids[1:]):
                left_speeds[fid] = velocities[i]

        # Get max speed for y-axis limit
        all_speeds = list(right_speeds.values()) + list(left_speeds.values())
        max_speed = max(all_speeds) if all_speeds else 1.0

        # Create a persistent overlay for paths and highlights
        path_overlay = np.zeros_like(frames[0])
        prev_right_point = None
        prev_left_point = None

        with imageio.get_writer(output_video, fps=30) as writer:
            for frame_idx, frame in enumerate(tqdm(frames, desc="Creating video with speed plot")):
                # --- Update the persistent overlay ---
                # Draw new path segments
                if frame_idx in right_tip_points:
                    current_right_point = right_tip_points[frame_idx]
                    if prev_right_point is not None:
                        cv2.line(path_overlay, prev_right_point, current_right_point, (0, 0, 255), thickness=7)
                    prev_right_point = current_right_point

                if frame_idx in left_tip_points:
                    current_left_point = left_tip_points[frame_idx]
                    if prev_left_point is not None:
                        cv2.line(path_overlay, prev_left_point, current_left_point, (0, 255, 0), thickness=7)
                    prev_left_point = current_left_point

                # Draw highlight point once when its frame is reached
                if right_hl_frame is not None and frame_idx == right_hl_frame:
                    cv2.circle(path_overlay, right_hl_coords, 30, (0, 0, 255), -1)  # Red fill
                    cv2.circle(path_overlay, right_hl_coords, 30, (0, 0, 0), 5)  # Black outline

                if left_hl_frame is not None and frame_idx == left_hl_frame:
                    cv2.circle(path_overlay, left_hl_coords, 30, (0, 255, 0), -1)  # Green fill
                    cv2.circle(path_overlay, left_hl_coords, 30, (0, 0, 0), 5)  # Black outline

                # --- Combine frame with overlay ---
                img = cv2.add(frame.copy(), path_overlay)
                h, w, _ = img.shape

                # --- Draw Bbox and Keypoints (on the combined image) ---
                # Draw right hand track (Red)
                if frame_idx in right_tracks:
                    x1, y1, x2, y2 = right_tracks[frame_idx]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(img, "Right", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                    if frame_idx in original_right_tracks:
                        keypoints_flat = original_right_tracks[frame_idx][5:]
                        keypoints = np.array(keypoints_flat).reshape(-1, 3)
                        for x, y, conf in keypoints:
                            if conf > 0.5:
                                cv2.circle(img, (int(x * w), int(y * h)), 3, (0, 0, 255), -1)

                # Draw left hand track (Green)
                if frame_idx in left_tracks:
                    x1, y1, x2, y2 = left_tracks[frame_idx]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img, "Left", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    if frame_idx in original_left_tracks:
                        keypoints_flat = original_left_tracks[frame_idx][5:]
                        keypoints = np.array(keypoints_flat).reshape(-1, 3)
                        for x, y, conf in keypoints:
                            if conf > 0.5:
                                cv2.circle(img, (int(x * w), int(y * h)), 3, (0, 255, 0), -1)

                # --- Create Speed Plot ---
                fig, ax = plt.subplots(figsize=(w / 100, 2), dpi=100)

                # Plot right hand speed up to current frame
                if right_speeds:
                    frames_so_far = sorted([f for f in right_speeds.keys() if f <= frame_idx])
                    speeds_so_far = [right_speeds[f] for f in frames_so_far]
                    ax.plot(frames_so_far, speeds_so_far, color='red', label='Right Hand')

                # Plot left hand speed up to current frame
                if left_speeds:
                    frames_so_far = sorted([f for f in left_speeds.keys() if f <= frame_idx])
                    speeds_so_far = [left_speeds[f] for f in frames_so_far]
                    ax.plot(frames_so_far, speeds_so_far, color='green', label='Left Hand')

                ax.set_xlim(0, len(frames))
                ax.set_ylim(0, max_speed * 1.1)
                ax.set_title('Hand Speed')
                ax.set_xlabel('Frame Index')
                ax.set_ylabel('Speed (pixels/frame)')
                if right_speeds or left_speeds:
                    ax.legend()

                # Save plot to an in-memory buffer
                buf = io.BytesIO()
                fig.savefig(buf, format='png', dpi=100)
                buf.seek(0)
                plot_img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
                buf.close()
                plot_img = cv2.imdecode(plot_img_arr, cv2.IMREAD_COLOR)
                plt.close(fig)

                # --- Combine Plot and Frame ---
                plot_h, plot_w, _ = plot_img.shape
                # Resize plot to match video width, maintaining aspect ratio
                target_plot_h = int(plot_h * (w / plot_w))
                resized_plot = cv2.resize(plot_img, (w, target_plot_h))

                # Create a black canvas
                combined_h = h + target_plot_h
                combined_img = np.zeros((combined_h, w, 3), dtype=np.uint8)

                # Place plot on top, video frame at bottom
                combined_img[0:target_plot_h, 0:w] = resized_plot
                combined_img[target_plot_h:, 0:w] = img

                # Ensure dimensions are divisible by macro_block_size
                macro_block_size = 16
                padded_h = (combined_h + macro_block_size - 1) // macro_block_size * macro_block_size
                padded_w = (w + macro_block_size - 1) // macro_block_size * macro_block_size

                if padded_h != combined_h or padded_w != w:
                    padded_img = np.zeros((padded_h, padded_w, 3), dtype=np.uint8)
                    padded_img[:combined_h, :w] = combined_img
                    img_to_write = padded_img
                else:
                    img_to_write = combined_img

                # Convert BGR (OpenCV) to RGB (imageio)
                img_rgb = cv2.cvtColor(img_to_write, cv2.COLOR_BGR2RGB)
                writer.append_data(img_rgb)

        print(f"Visualization video saved to {output_video}")

    return right_hl_coords, left_hl_coords, right_hl_frame, left_hl_frame


def find_speed_valley_point(root_points, tip_points):
    """
    Finds the first significant speed valley point using peak prominence.

    Args:
        root_points: Interpolated root joint positions for speed calculation.
        tip_points: Interpolated tip positions for path location.

    Returns:
        A tuple (frame_index, (x, y)) of the highlight point, or (None, None).
    """
    if len(root_points) < 20:  # Need more points for robust peak detection
        return None, None

    frame_ids = sorted(root_points.keys())
    speed_points = [root_points[fid] for fid in frame_ids]

    velocities = np.array(speed_points[1:]) - np.array(speed_points[:-1])
    speeds = np.linalg.norm(velocities, axis=1)

    if len(speeds) < 10:
        return None, None

    # Find valleys by finding peaks in the inverted signal
    # A valley needs to be prominent to be considered significant
    prominence_threshold = np.std(speeds) * 0.5  # Must be at least 50% of std dev deep
    valleys, _ = find_peaks(-speeds, prominence=prominence_threshold, width=3)

    if valleys.size > 0:
        first_valley_idx = valleys[0]
        highlight_frame_idx = frame_ids[first_valley_idx + 1]

        if highlight_frame_idx in tip_points:
            highlight_coords = tip_points[highlight_frame_idx]
            return highlight_frame_idx, highlight_coords

    return None, None


def visualize_initial_masks(first_frame, masks_info_to_track, grid_points_to_draw):
    """
    Visualizes the initial masks generated from grid points, with a unique color for each mask.
    """
    vis_image = first_frame.copy()

    mask_contours_and_colors = []

    # Draw the initial masks if provided
    if masks_info_to_track:
        overlay = vis_image.copy()
        for m_info in masks_info_to_track:
            mask = m_info['mask_data']['segmentation']
            obj_id = m_info['obj_id']

            # Generate a pseudo-random color based on the object ID for uniqueness
            random.seed(obj_id)
            color = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))

            # Apply color to the mask area on the overlay
            overlay[mask.astype(bool)] = color

            # Find and store contours for drawing later
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            smoothed_contours = [cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True) for contour in contours]
            mask_contours_and_colors.append((smoothed_contours, color))

        # Blend the overlay with the original image
        alpha = 0.5
        vis_image = cv2.addWeighted(overlay, alpha, vis_image, 1 - alpha, 0)

        # Draw borders on top of the blended image
        for contours, color in mask_contours_and_colors:
            cv2.drawContours(vis_image, contours, -1, color, thickness=2)

    # Draw the grid points on top
    if grid_points_to_draw:
        for point in grid_points_to_draw:
            px, py = int(point[0]), int(point[1])
            # Draw a small, bright circle for each grid point
            cv2.circle(vis_image, (px, py), 3, (255, 0, 255), -1)  # Magenta for visibility

    return vis_image


def visualize_hand_trajectory(first_frame, right_tip_points, left_tip_points, right_root_points, left_root_points):
    """
    Visualizes hand trajectory.
    The trajectory path is based on the average of fingertips.
    The trajectory color and highlighted speed valley are based on the wrist (root joint) speed.

    Args:
        first_frame: The first frame of the video.
        right_tip_points: Interpolated tip trajectory for the right hand.
        left_tip_points: Interpolated tip trajectory for the left hand.
        right_root_points: Interpolated root joint trajectory for the right hand.
        left_root_points: Interpolated root joint trajectory for the left hand.
    
    Returns:
        The visualization image as a numpy array.
    """
    vis_image = first_frame.copy()

    def process_and_draw_trajectory(tip_points, root_points, base_color):
        if len(tip_points) < 10 or len(root_points) < 10:  # Need enough points for robust analysis
            return

        # Path is drawn using tip points
        tip_frame_ids = sorted(tip_points.keys())
        path_points = [tip_points[fid] for fid in tip_frame_ids]

        # Speed is calculated from root points
        root_frame_ids = sorted(root_points.keys())
        speed_points = [root_points[fid] for fid in root_frame_ids]

        velocities = np.array(speed_points[1:]) - np.array(speed_points[:-1])
        speeds = np.linalg.norm(velocities, axis=1)

        # --- Speed for coloring ---
        if speeds.size > 0:
            max_speed = np.max(speeds)
            min_speed = np.min(speeds)

            if max_speed == min_speed:
                normalized_speeds = np.ones_like(speeds)
            else:
                normalized_speeds = (speeds - min_speed) / (max_speed - min_speed)

            light_color = np.array(base_color)
            dark_color = light_color * 0.2

            # Ensure we don't go out of bounds if trajectories have slightly different interpolated lengths
            num_segments = min(len(path_points) - 1, len(speeds))

            for i in range(num_segments):
                speed_norm = normalized_speeds[i]
                # Slower (speed_norm=0) -> light, faster (speed_norm=1) -> dark
                line_color = tuple(int(c) for c in ((1 - speed_norm) * light_color + speed_norm * dark_color))
                cv2.line(vis_image, path_points[i], path_points[i + 1], line_color, 7)
        else:
            # Fallback to draw with solid color if no speed data
            for i in range(len(path_points) - 1):
                cv2.line(vis_image, path_points[i], path_points[i + 1], base_color, 7)

        # --- Find highlight point (first speed valley) based on root speed ---
        _, highlight_point = find_speed_valley_point(root_points, tip_points)

        if highlight_point is not None:
            cv2.circle(vis_image, highlight_point, 30, (255, 0, 255), -1)  # Magenta fill
            cv2.circle(vis_image, highlight_point, 30, (0, 0, 0), 5)  # Black outline

    process_and_draw_trajectory(right_tip_points, right_root_points, base_color=(0, 0, 255))
    process_and_draw_trajectory(left_tip_points, left_root_points, base_color=(0, 255, 0))

    return vis_image


def calculate_frame_difference_mask(frames):
    """
    Calculates a difference mask between the first and last frames to show changes.

    Args:
        frames: A list of video frames.

    Returns:
        The difference mask as a numpy array.
    """
    if len(frames) < 2:
        print("Not enough frames to calculate difference mask.")
        return None

    # Get first and last frames
    first_frame = frames[0]
    last_frame = frames[-1]

    # Convert frames to grayscale
    gray_first = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    gray_last = cv2.cvtColor(last_frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise and improve diff accuracy
    gray_first = cv2.GaussianBlur(gray_first, (21, 21), 0)
    gray_last = cv2.GaussianBlur(gray_last, (21, 21), 0)

    # Compute the absolute difference between the frames
    diff = cv2.absdiff(gray_first, gray_last)

    # Threshold the difference image to get a binary mask
    # Pixels with a difference value > 30 will be set to 255 (white)
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    # Dilate the thresholded image to fill in holes and make regions more solid
    thresh = cv2.dilate(thresh, None, iterations=2)

    return thresh


def visualize_tracks(imgfiles, tracks, seq_folder):
    # Create the visualization output directory.
    vis_folder = f'{seq_folder}/tracks_visualization'
    os.makedirs(vis_folder, exist_ok=True)

    # Restore the dictionary when tracks were saved as a NumPy object array.
    if isinstance(tracks, np.ndarray):
        tracks_dict = tracks.item()
    else:
        tracks_dict = tracks

    # Assign one color to each track ID.
    track_colors = {}
    color_palette = [
        (255, 0, 0),  # Blue (BGR)
        (0, 255, 0),  # Green (BGR)
        (0, 0, 255),  # Red (BGR)
        (255, 255, 0),  # Cyan (BGR)
        (255, 0, 255),  # Magenta (BGR)
        (0, 255, 255),  # Yellow (BGR)
    ]

    # Render every frame.
    for frame_idx, imgpath in enumerate(tqdm(imgfiles, desc="Visualizing tracks")):
        img = cv2.imread(imgpath)

        # Inspect all tracks for detections on this frame.
        for track_id, track_data in tracks_dict.items():
            # Find the detection for the current frame.
            for det in track_data:
                if det['frame'] == frame_idx:
                    # Read the detection box and handedness.
                    box = det['det_box'][0]  # [x1, y1, x2, y2, conf]
                    handedness = det['det_handedness'][0]

                    # Convert box coordinates to integers.
                    x1, y1, x2, y2, conf = box
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    # Assign a stable color to this track.
                    if track_id not in track_colors:
                        color_idx = len(track_colors) % len(color_palette)
                        track_colors[track_id] = color_palette[color_idx]
                    color = track_colors[track_id]

                    # Draw the detection box.
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

                    # Build the track label.
                    hand_type = "Right" if handedness > 0 else "Left"
                    label = f"ID:{int(track_id)} {hand_type} {conf:.2f}"

                    # Draw the label background.
                    (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(img, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)

                    # Draw the label text.
                    cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Save the rendered frame.
        output_path = os.path.join(vis_folder, os.path.basename(imgpath))
        cv2.imwrite(output_path, img)

    print(f"Visualization saved to {vis_folder}")

    # Optionally encode the rendered frames as a video.
    output_video = f'{seq_folder}/original_tracks_2d.mp4'
    command = ['ffmpeg', '-framerate', '30', '-i', f'{vis_folder}/%06d.jpg', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-y', output_video]
    try:
        subprocess.run(command, check=True, capture_output=True)
        print(f"Visualization video saved to {output_video}")
    except subprocess.CalledProcessError:
        print("Failed to create video, but images are saved")


def read_video_frames(video_path):
    """
    Reads all frames from a video file into a list of numpy arrays.
    
    Args:
        video_path: Path to the video file.
    
    Returns:
        A list of frames (numpy arrays).
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def detect_track(hand_det_model, frames, conf=0.5, conf_threshold=0.7, tracker='configs/bytetrack.yaml'):
    # Run
    boxes_ = []
    tracks = {}
    right_hand_tracks = {}
    left_hand_tracks = {}
    for t, img_cv2 in enumerate(frames):

        ### --- Detection ---
        with torch.no_grad():
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            with autocast(device_type=device):
                results = hand_det_model.track(img_cv2, conf=conf, tracker=tracker, persist=True, verbose=False)

                boxes = results[0].boxes.xywh.cpu().numpy()  # normalized to 0-1
                confs = results[0].boxes.conf.cpu().numpy()
                handedness = results[0].boxes.cls.cpu().numpy()
                keypoints = results[0].keypoints.xy.cpu().numpy()  # normalized to 0-1
                keypoints_conf = results[0].keypoints.conf.cpu().numpy()
                if not results[0].boxes.id is None:
                    track_id = results[0].boxes.id.cpu().numpy()
                else:
                    track_id = [-1] * len(boxes)

                # boxes = np.hstack([boxes, confs[:, None]])
                find_right = False
                find_left = False
                for idx, box in enumerate(boxes):
                    # only process high-confidence detections
                    if confs[[idx]][0] < conf_threshold:
                        continue
                    if track_id[idx] == -1:
                        if handedness[[idx]] > 0:
                            id = int(10000)
                        else:
                            id = int(5000)
                    else:
                        id = track_id[idx]
                    subj = dict()
                    subj['frame'] = t
                    subj['det'] = True
                    subj['det_box'] = boxes[[idx]]
                    subj['det_handedness'] = handedness[[idx]]

                    if (not find_right and handedness[[idx]] > 0) or (not find_left and handedness[[idx]] == 0):
                        cur_keypoints = keypoints[[idx]][0].tolist()
                        cur_keypoints_conf = (keypoints_conf[[idx]][0] > 0.5).astype(int).tolist()
                        cur_keypoints_flat = [coord for kpt, c in zip(cur_keypoints, cur_keypoints_conf) for coord in [kpt[0], kpt[1], c]]
                        x_coords = np.array(cur_keypoints)[:, 0]
                        y_coords = np.array(cur_keypoints)[:, 1]
                        x_min, x_max = x_coords.min(), x_coords.max()
                        y_min, y_max = y_coords.min(), y_coords.max()
                        x_center = (x_min + x_max) / 2
                        y_center = (y_min + y_max) / 2
                        width = x_max - x_min
                        height = y_max - y_min
                        scale = 1.2
                        cur_bbox = [x_center, y_center, width * scale, height * scale]
                        full_info = handedness[[idx]].tolist() + cur_bbox + cur_keypoints_flat
                        if (id in tracks) and (subj['det_handedness'][0] == tracks[id][-1]['det_handedness'][0]):
                            # make sure the handness is the same
                            tracks[id].append(subj)

                            if handedness[[idx]] > 0:
                                right_hand_tracks[t] = full_info
                                find_right = True
                            elif handedness[[idx]] == 0:
                                left_hand_tracks[t] = full_info
                                find_left = True
                        elif (id not in tracks):
                            tracks[id] = [subj]

                            if handedness[[idx]] > 0:
                                right_hand_tracks[t] = full_info
                                find_right = True
                            elif handedness[[idx]] == 0:
                                left_hand_tracks[t] = full_info
                                find_left = True
                        else:
                            continue
    hand_det_model.predictor.trackers[0].reset()

    return tracks, right_hand_tracks, left_hand_tracks


def save_detection_results(frames,
                           right_hand_tracks,
                           left_hand_tracks,
                           video_name,
                           output_path='./data_hoi',
                           fps=30,
                           train_val_ratio=0.9,
                           filtered_video_segments=None,
                           interpolated_right_hand_bboxes=None,
                           interpolated_left_hand_bboxes=None,
                           right_hl_frame=None,
                           left_hl_frame=None):
    # Get image dimensions for normalization
    if not frames:
        return f"Video {video_name} has no frames to process."
    h, w, _ = frames[0].shape
    dummy_keypoints = [0.0] * (21 * 3)

    # collect high-confidence frames
    high_conf_frames = {}

    for frame_idx in range(len(frames)):
        frame_data = []

        # check right hand
        if frame_idx in right_hand_tracks:
            full_info = right_hand_tracks[frame_idx]
            frame_data.append(('right', full_info))

        # check left hand
        if frame_idx in left_hand_tracks:
            full_info = left_hand_tracks[frame_idx]
            frame_data.append(('left', full_info))

        if frame_data:
            high_conf_frames[frame_idx] = frame_data

    # check if the video has low detection rate (< 30 frames)
    if len(high_conf_frames) < 30:

        # calculate the frame range to remove the first and last 1.5s
        skip_frames = int(1.5 * fps)  # 1.5 seconds corresponds to frames
        start_frame = skip_frames
        end_frame = len(frames) - skip_frames

        # save all frames (remove the first and last 1.5s) to test
        for frame_idx in range(start_frame, end_frame):
            frame = frames[frame_idx]
            # only copy images, no labels
            image_path = os.path.join(output_path, 'images', 'test', f'{video_name}_{frame_idx:06d}.jpg')
            cv2.imwrite(image_path, frame)

        return f"Video {video_name} has less than 30 high-confidence detections ({len(high_conf_frames)} frames), saving to test set..."

    # enough detections, split into train/val

    # get all frame indices and sort
    frame_indices = sorted(high_conf_frames.keys())

    # split by 9:1 ratio
    split_idx = int(len(frame_indices) * train_val_ratio)
    train_frames = set(frame_indices[:split_idx])
    val_frames = set(frame_indices[split_idx:])

    # save train results
    for frame_idx in train_frames:
        frame = frames[frame_idx]
        frame_data = high_conf_frames[frame_idx]
        labels = []

        for hand_type, full_info in frame_data:
            class_id = full_info[0]
            # Normalize bbox
            cx, cy, box_w, box_h = full_info[1:5]
            norm_cx = cx / w
            norm_cy = cy / h
            norm_box_w = box_w / w
            norm_box_h = box_h / h

            # Normalize keypoints
            keypoints = np.array(full_info[5:]).reshape(-1, 3)
            keypoints[:, 0] /= w  # Normalize x
            keypoints[:, 1] /= h  # Normalize y
            norm_keypoints_flat = keypoints.flatten().tolist()

            # Reconstruct the normalized label
            normalized_label = [class_id, norm_cx, norm_cy, norm_box_w, norm_box_h] + norm_keypoints_flat
            labels.append(normalized_label)

        # Add interacting object labels if they exist and overlap with hands
        if filtered_video_segments and frame_idx in filtered_video_segments:
            object_masks = filtered_video_segments[frame_idx]

            # Object interacting with right hand
            if 1 in object_masks and interpolated_right_hand_bboxes and frame_idx in interpolated_right_hand_bboxes:
                start_frame = right_hl_frame - 10 if right_hl_frame is not None else float('inf')
                if frame_idx >= start_frame:
                    mask = object_masks[1].squeeze()
                    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        all_points = np.concatenate(contours, axis=0)
                        obj_x, obj_y, obj_w, obj_h = cv2.boundingRect(all_points)
                        obj_box_xyxy = [obj_x, obj_y, obj_x + obj_w, obj_y + obj_h]
                        hand_box_xyxy = interpolated_right_hand_bboxes[frame_idx]

                        if calculate_bbox_iou(obj_box_xyxy, hand_box_xyxy) > 0:
                            cx = (obj_x + obj_w / 2) / w
                            cy = (obj_y + obj_h / 2) / h
                            norm_w = obj_w / w
                            norm_h = obj_h / h
                            labels.append([2, cx, cy, norm_w, norm_h] + dummy_keypoints)

            # Object interacting with left hand
            if 2 in object_masks and interpolated_left_hand_bboxes and frame_idx in interpolated_left_hand_bboxes:
                start_frame = left_hl_frame - 10 if left_hl_frame is not None else float('inf')
                if frame_idx >= start_frame:
                    mask = object_masks[2].squeeze()
                    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        all_points = np.concatenate(contours, axis=0)
                        obj_x, obj_y, obj_w, obj_h = cv2.boundingRect(all_points)
                        obj_box_xyxy = [obj_x, obj_y, obj_x + obj_w, obj_y + obj_h]
                        hand_box_xyxy = interpolated_left_hand_bboxes[frame_idx]

                        if calculate_bbox_iou(obj_box_xyxy, hand_box_xyxy) > 0:
                            cx = (obj_x + obj_w / 2) / w
                            cy = (obj_y + obj_h / 2) / h
                            norm_w = obj_w / w
                            norm_h = obj_h / h
                            labels.append([2, cx, cy, norm_w, norm_h] + dummy_keypoints)

        label_file = os.path.join(output_path, 'labels', 'train', f'{video_name}_{frame_idx:06d}.txt')

        with open(label_file, 'w') as f:
            for label in labels:
                f.write(f"{' '.join(str(x) for x in label)}\n")

        image_path = os.path.join(output_path, 'images', 'train', f'{video_name}_{frame_idx:06d}.jpg')
        cv2.imwrite(image_path, frame)

    # save val data
    for frame_idx in val_frames:
        frame = frames[frame_idx]
        frame_data = high_conf_frames[frame_idx]

        labels = []
        for hand_type, full_info in frame_data:
            class_id = full_info[0]
            # Normalize bbox
            cx, cy, box_w, box_h = full_info[1:5]
            norm_cx = cx / w
            norm_cy = cy / h
            norm_box_w = box_w / w
            norm_box_h = box_h / h

            # Normalize keypoints
            keypoints = np.array(full_info[5:]).reshape(-1, 3)
            keypoints[:, 0] /= w  # Normalize x
            keypoints[:, 1] /= h  # Normalize y
            norm_keypoints_flat = keypoints.flatten().tolist()

            # Reconstruct the normalized label
            normalized_label = [class_id, norm_cx, norm_cy, norm_box_w, norm_box_h] + norm_keypoints_flat
            labels.append(normalized_label)

        # Add interacting object labels if they exist and overlap with hands (for validation set)
        if filtered_video_segments and frame_idx in filtered_video_segments:
            object_masks = filtered_video_segments[frame_idx]

            # Object interacting with right hand
            if 1 in object_masks and interpolated_right_hand_bboxes and frame_idx in interpolated_right_hand_bboxes:
                start_frame = right_hl_frame - 10 if right_hl_frame is not None else float('inf')
                if frame_idx >= start_frame:
                    mask = object_masks[1].squeeze()
                    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        all_points = np.concatenate(contours, axis=0)
                        obj_x, obj_y, obj_w, obj_h = cv2.boundingRect(all_points)
                        obj_box_xyxy = [obj_x, obj_y, obj_x + obj_w, obj_y + obj_h]
                        hand_box_xyxy = interpolated_right_hand_bboxes[frame_idx]

                        if calculate_bbox_iou(obj_box_xyxy, hand_box_xyxy) > 0:
                            cx = (obj_x + obj_w / 2) / w
                            cy = (obj_y + obj_h / 2) / h
                            norm_w = obj_w / w
                            norm_h = obj_h / h
                            labels.append([2, cx, cy, norm_w, norm_h] + dummy_keypoints)

            # Object interacting with left hand
            if 2 in object_masks and interpolated_left_hand_bboxes and frame_idx in interpolated_left_hand_bboxes:
                start_frame = left_hl_frame - 10 if left_hl_frame is not None else float('inf')
                if frame_idx >= start_frame:
                    mask = object_masks[2].squeeze()
                    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        all_points = np.concatenate(contours, axis=0)
                        obj_x, obj_y, obj_w, obj_h = cv2.boundingRect(all_points)
                        obj_box_xyxy = [obj_x, obj_y, obj_x + obj_w, obj_y + obj_h]
                        hand_box_xyxy = interpolated_left_hand_bboxes[frame_idx]

                        if calculate_bbox_iou(obj_box_xyxy, hand_box_xyxy) > 0:
                            cx = (obj_x + obj_w / 2) / w
                            cy = (obj_y + obj_h / 2) / h
                            norm_w = obj_w / w
                            norm_h = obj_h / h
                            labels.append([2, cx, cy, norm_w, norm_h] + dummy_keypoints)

        label_file = os.path.join(output_path, 'labels', 'val', f'{video_name}_{frame_idx:06d}.txt')
        with open(label_file, 'w') as f:
            for label in labels:
                f.write(f"{' '.join(str(x) for x in label)}\n")

        image_path = os.path.join(output_path, 'images', 'val', f'{video_name}_{frame_idx:06d}.jpg')
        cv2.imwrite(image_path, frame)

    return f"Successfully saved {len(train_frames)} frames to train, {len(val_frames)} frames to val detection results for video: {video_name}!"


def sample_videos(num_samples=1000):
    """
    Randomly sample videos from the taste_rob dataset
    """
    # Define paths
    base_dir = Path("data/taste_rob")

    # Collect all video paths
    all_videos = []
    for hand_type in ['DoubleHand', 'SingleHand']:
        hand_dir = base_dir / hand_type
        if not hand_dir.exists():
            continue

        for scene_dir in hand_dir.iterdir():
            if not scene_dir.is_dir():
                continue

            video_files = list(scene_dir.glob("*.mp4"))
            for video_path in video_files:
                all_videos.append(video_path)

    print(f"Found {len(all_videos)} videos")

    # Randomly sample videos
    num_samples = min(num_samples, len(all_videos))
    random.seed(42)
    sampled_video_paths = random.sample(all_videos, num_samples)
    print(f"Successfully sampled {num_samples} videos from the taste_rob dataset")
    return sampled_video_paths


def filter_processed_videos(video_files, data_folder='./data_hoi'):
    """Checks for existing labels and filters the list of video files."""
    train_labels_dir = Path(data_folder) / 'labels' / 'train'
    val_labels_dir = Path(data_folder) / 'labels' / 'val'
    test_labels_dir = Path(data_folder) / 'labels' / 'test'

    if not train_labels_dir.exists() and not val_labels_dir.exists() and not test_labels_dir.exists():
        return video_files  # No labels exist yet, no need to filter.

    processed_video_names = set()

    def get_processed_names(label_dir):
        if not label_dir.exists():
            return set()
        names = set()
        for label_file in label_dir.glob('*.txt'):
            parts = label_file.stem.split('_')
            # Assuming format {video_name}_{frame_id}.txt
            if len(parts) > 1 and parts[-1].isdigit():
                video_name = '_'.join(parts[:-1])
                names.add(video_name)
        return names

    processed_video_names.update(get_processed_names(train_labels_dir))
    processed_video_names.update(get_processed_names(val_labels_dir))
    processed_video_names.update(get_processed_names(test_labels_dir))

    if processed_video_names:
        print(f"Found labels for {len(processed_video_names)} already processed videos. Filtering...")
        original_count = len(video_files)
        filtered_videos = [v for v in video_files if v.stem not in processed_video_names]
        print(f"Filtered out {original_count - len(filtered_videos)} videos. {len(filtered_videos)} videos remaining to process.")
        return filtered_videos
    else:
        return video_files


def process_single_video(
    video_path,
    hand_det_model_path,
    output_data_folder,
    output_base_folder,
    args,
    hand_det_model=None,
    sam2=None,
    video_predictor=None,
    predictor=None,
):
    """
    Process a single video
    """
    global _WORKER_MODELS
    # The device is now implicitly handled by CUDA_VISIBLE_DEVICES for multiprocessing
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load models if they are not provided (handles both single-thread and multi-process)
    if hand_det_model is None:
        if 'hand_det_model' not in _WORKER_MODELS:
            _WORKER_MODELS['hand_det_model'] = YOLO(hand_det_model_path)
        hand_det_model = _WORKER_MODELS['hand_det_model']

    sam2_checkpoint = "sam2/checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

    if sam2 is None:
        if 'sam2' not in _WORKER_MODELS:
            _WORKER_MODELS['sam2'] = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=True)
        sam2 = _WORKER_MODELS['sam2']

    if predictor is None:
        if 'predictor' not in _WORKER_MODELS:
            _WORKER_MODELS['predictor'] = SAM2ImagePredictor(sam2, mask_threshold=0.2)
        predictor = _WORKER_MODELS['predictor']

    if video_predictor is None:
        if 'video_predictor' not in _WORKER_MODELS:
            _WORKER_MODELS['video_predictor'] = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
        video_predictor = _WORKER_MODELS['video_predictor']

    video_name = video_path.stem

    # create output folder for this video
    video_output_folder = os.path.join(output_base_folder, video_name)
    os.makedirs(video_output_folder, exist_ok=True)

    # extract frames into memory
    frames = read_video_frames(str(video_path))
    if not frames:
        return f"Warning: Could not read any frames from video: {video_name}. File may be corrupted or in an unsupported format."

    # perform hand detection and tracking
    tracks, right_hand_tracks, left_hand_tracks = detect_track(
        hand_det_model,
        frames,
        conf=0.6,
        conf_threshold=0.7,
        tracker='configs/bytetrack.yaml',
    )

    if not right_hand_tracks and not left_hand_tracks:
        return f"Info: No hands were detected in video: {video_name}. Skipping visualization."

    # Get image dimensions from the first frame
    h, w, _ = frames[0].shape

    # Prepare tracks for interpolation (convert xywh to xyxy)
    def prepare_for_interpolation(hand_tracks):
        track_bboxes = {}
        for frame_idx, full_info in hand_tracks.items():
            cx, cy, bw, bh = full_info[1:5]  # cx,cy,w,h
            x1 = (cx - bw / 2)
            y1 = (cy - bh / 2)
            x2 = (cx + bw / 2)
            y2 = (cy + bh / 2)
            track_bboxes[frame_idx] = np.array([x1, y1, x2, y2])
        return track_bboxes

    right_hand_bboxes = prepare_for_interpolation(right_hand_tracks)
    left_hand_bboxes = prepare_for_interpolation(left_hand_tracks)

    # Interpolate tracks
    interpolated_right_hand = interpolate_tracks(right_hand_bboxes)
    interpolated_left_hand = interpolate_tracks(left_hand_bboxes)

    # Interpolate tip positions for trajectory
    interpolated_right_tips = interpolate_tip_positions(right_hand_tracks)
    interpolated_left_tips = interpolate_tip_positions(left_hand_tracks)

    # Interpolate root joint positions for speed calculation
    interpolated_right_root = interpolate_root_joint_positions(right_hand_tracks)
    interpolated_left_root = interpolate_root_joint_positions(left_hand_tracks)

    # Visualize interpolated tracks and create video
    right_hl_coords, left_hl_coords, right_hl_frame, left_hl_frame = visualize_interpolated_tracks(
        frames,
        interpolated_right_hand,
        interpolated_left_hand,
        right_hand_tracks,
        left_hand_tracks,
        interpolated_right_root,
        interpolated_left_root,
        interpolated_right_tips,
        interpolated_left_tips,
        video_output_folder,
        args,
    )

    # Save the highlight frame to the centralized folder
    if args.highlight_folder:
        save_highlight_frame(video_name, frames, right_hl_frame, left_hl_frame, args.highlight_folder)

    # Define grid parameters for SAM tracking
    masks_for_tracking = None
    grid_points_for_vis = None
    filtered_video_segments = {}

    # 1. SAM-2 Tracking
    if right_hl_coords or left_hl_coords:
        masks_for_tracking, grid_points_for_vis = generate_masks_from_prompts(
            predictor,
            frames[0],
            right_hl_coords,
            left_hl_coords,
        )

        # Visualize initial masks
        if masks_for_tracking and args.save_visualizations:
            if args.save_masks_separately:
                for i, m_info in enumerate(masks_for_tracking):
                    # Visualize and save each mask separately
                    single_mask_image = visualize_initial_masks(
                        frames[0],
                        [m_info],  # Pass a list containing only the current mask
                        [],  # Do not draw grid points on individual mask images
                    )
                    separate_mask_path = os.path.join(video_output_folder, f"initial_mask_{m_info['obj_id']}_{i}.jpg")
                    cv2.imwrite(separate_mask_path, single_mask_image)
                print(f"Saved {len(masks_for_tracking)} initial masks separately in {video_output_folder}")
            else:
                # Original behavior: save all masks on one image
                initial_masks_image = visualize_initial_masks(
                    frames[0],
                    masks_for_tracking,
                    grid_points_for_vis,
                )
                initial_masks_output_path = os.path.join(video_output_folder, 'initial_masks.jpg')
                cv2.imwrite(initial_masks_output_path, initial_masks_image)
                print(f"Initial masks visualization saved to {initial_masks_output_path}")

        video_segments = sam2_tracking(video_predictor, np.array(frames), masks_for_tracking)
        if video_segments:
            # Filter tracks based on interaction with hand trajectories
            (filtered_video_segments, best_right_track_id, best_left_track_id) = select_interactive_tracks(
                video_segments,
                interpolated_right_tips,
                interpolated_left_tips,
                right_hl_frame,
                left_hl_frame,
            )

            # Save a debug video with all tracks
            if args.save_visualizations:
                debug_sam_output_path = os.path.join(video_output_folder, 'sam2_tracking_debug.mp4')
                visualize_all_sam_tracks_for_debug(
                    frames,
                    video_segments,  # use original segments
                    best_right_track_id,
                    best_left_track_id,
                    debug_sam_output_path,
                )

            if filtered_video_segments and args.save_visualizations:
                sam_output_path = os.path.join(video_output_folder, 'sam2_tracking.mp4')
                visualize_sam_tracking(frames, filtered_video_segments, sam_output_path)

    # 2. Generate trajectory visualization
    if args.save_visualizations:
        trajectory_image = visualize_hand_trajectory(
            frames[0],
            interpolated_right_tips,
            interpolated_left_tips,
            interpolated_right_root,
            interpolated_left_root,
        )

        # 3. Save the images
        if trajectory_image is not None:
            trajectory_output_path = os.path.join(video_output_folder, 'trajectory.jpg')
            cv2.imwrite(trajectory_output_path, trajectory_image)
            print(f"Trajectory visualization saved to {trajectory_output_path}")

    # save results
    print_info = save_detection_results(
        frames,
        right_hand_tracks,
        left_hand_tracks,
        video_name,
        output_path=output_data_folder,
        fps=30,
        train_val_ratio=0.9,
        filtered_video_segments=filtered_video_segments,
        interpolated_right_hand_bboxes=interpolated_right_hand,
        interpolated_left_hand_bboxes=interpolated_left_hand,
        right_hl_frame=right_hl_frame,
        left_hl_frame=left_hl_frame,
    )

    # Add visualization of HOI results
    if args.save_visualizations:
        hoi_vis_output_path = os.path.join(video_output_folder, 'hoi_visualization.mp4')
        visualize_hoi_results(
            frames,
            right_hand_tracks,
            left_hand_tracks,
            interpolated_right_hand,
            interpolated_left_hand,
            filtered_video_segments,
            hoi_vis_output_path,
            right_hl_frame=right_hl_frame,
            left_hl_frame=left_hl_frame,
        )

    return f"Processed video {video_name}. {print_info}"


def visualize_hoi_results(
    frames,
    right_hand_tracks,
    left_hand_tracks,
    interpolated_right_hand,
    interpolated_left_hand,
    filtered_video_segments,
    output_path,
    right_hl_frame=None,
    left_hl_frame=None,
):
    """
    Visualizes hand-object interaction results: hand bboxes, keypoints, and object masks/bboxes.
    """
    with imageio.get_writer(output_path, fps=30) as writer:
        for frame_idx, frame in enumerate(tqdm(frames, desc="Visualizing HOI Results")):
            img = frame.copy()
            h, w, _ = img.shape

            # First, always draw hand bboxes and keypoints if they are present in the frame
            right_hand_bbox = None
            if frame_idx in interpolated_right_hand:
                x1, y1, x2, y2 = interpolated_right_hand[frame_idx]
                right_hand_bbox = [int(x1), int(y1), int(x2), int(y2)]  # Store for later IoU check
                cv2.rectangle(img, (right_hand_bbox[0], right_hand_bbox[1]), (right_hand_bbox[2], right_hand_bbox[3]), (0, 0, 255), 2)
                cv2.putText(img, "Right Hand", (right_hand_bbox[0], right_hand_bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                if frame_idx in right_hand_tracks:
                    keypoints_flat = right_hand_tracks[frame_idx][5:]
                    keypoints = np.array(keypoints_flat).reshape(-1, 3)
                    for x, y, conf in keypoints:
                        if conf > 0.5:
                            cv2.circle(img, (int(x), int(y)), 3, (0, 0, 255), -1)

            left_hand_bbox = None
            if frame_idx in interpolated_left_hand:
                x1, y1, x2, y2 = interpolated_left_hand[frame_idx]
                left_hand_bbox = [int(x1), int(y1), int(x2), int(y2)]
                cv2.rectangle(img, (left_hand_bbox[0], left_hand_bbox[1]), (left_hand_bbox[2], left_hand_bbox[3]), (0, 255, 0), 2)
                cv2.putText(img, "Left Hand", (left_hand_bbox[0], left_hand_bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                if frame_idx in left_hand_tracks:
                    keypoints_flat = left_hand_tracks[frame_idx][5:]
                    keypoints = np.array(keypoints_flat).reshape(-1, 3)
                    for x, y, conf in keypoints:
                        if conf > 0.5:
                            cv2.circle(img, (int(x), int(y)), 3, (0, 255, 0), -1)

            # Second, draw the object mask and bbox only when it overlaps with a hand present in the same frame
            if filtered_video_segments and frame_idx in filtered_video_segments:
                mask_overlay = np.zeros_like(img, dtype=np.uint8)
                object_segments = filtered_video_segments[frame_idx]
                object_drawn = False

                # Object interacting with right hand
                if 1 in object_segments and right_hand_bbox is not None:
                    start_frame = right_hl_frame - 10 if right_hl_frame is not None else float('inf')
                    if frame_idx >= start_frame:
                        mask = object_segments[1].squeeze()
                        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        if contours:
                            all_points = np.concatenate(contours, axis=0)
                            x, y, bw, bh = cv2.boundingRect(all_points)
                            obj_bbox = [x, y, x + bw, y + bh]
                            if calculate_bbox_iou(right_hand_bbox, obj_bbox) > 0:
                                object_drawn = True
                                color = (255, 0, 0)  # Blue for object
                                mask_overlay[mask] = color
                                cv2.rectangle(img, (obj_bbox[0], obj_bbox[1]), (obj_bbox[2], obj_bbox[3]), color, 2)
                                cv2.putText(img, "Object", (obj_bbox[0], obj_bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                # Object interacting with left hand
                if 2 in object_segments and left_hand_bbox is not None:
                    start_frame = left_hl_frame - 10 if left_hl_frame is not None else float('inf')
                    if frame_idx >= start_frame:
                        mask = object_segments[2].squeeze()
                        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        if contours:
                            all_points = np.concatenate(contours, axis=0)
                            x, y, bw, bh = cv2.boundingRect(all_points)
                            obj_bbox = [x, y, x + bw, y + bh]
                            if calculate_bbox_iou(left_hand_bbox, obj_bbox) > 0:
                                color = (255, 0, 0)  # Blue for object
                                mask_overlay[mask] = color
                                cv2.rectangle(img, (obj_bbox[0], obj_bbox[1]), (obj_bbox[2], obj_bbox[3]), color, 2)
                                if not object_drawn:  # Avoid drawing label twice if both hands interact
                                    cv2.putText(img, "Object", (obj_bbox[0], obj_bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                                object_drawn = True

                if object_drawn:
                    alpha = 0.4
                    img = cv2.addWeighted(mask_overlay, alpha, img, 1 - alpha, 0)

            # Handle macroblock size for video encoding
            h_img, w_img, _ = img.shape
            macro_block_size = 16
            padded_h = (h_img + macro_block_size - 1) // macro_block_size * macro_block_size
            padded_w = (w_img + macro_block_size - 1) // macro_block_size * macro_block_size

            if padded_h != h_img or padded_w != w_img:
                padded_img = np.zeros((padded_h, padded_w, 3), dtype=np.uint8)
                padded_img[:h_img, :w_img] = img
                img_to_write = padded_img
            else:
                img_to_write = img

            writer.append_data(cv2.cvtColor(img_to_write, cv2.COLOR_BGR2RGB))

    print(f"HOI visualization video saved to {output_path}")


def save_highlight_frame(video_name, frames, right_hl_frame, left_hl_frame, output_folder):
    """Saves the highlight frame of a video to a specified folder."""
    highlight_frame_index = None
    if right_hl_frame is not None:
        highlight_frame_index = right_hl_frame
    elif left_hl_frame is not None:
        highlight_frame_index = left_hl_frame
    else:
        highlight_frame_index = 0  # Default to the first frame if no highlight found

    if 0 <= highlight_frame_index + 20 < len(frames):
        frame_to_save = frames[highlight_frame_index + 20]
        output_path = os.path.join(output_folder, f"{video_name}.jpg")
        cv2.imwrite(output_path, frame_to_save)


def main():
    parser = argparse.ArgumentParser(description="Process videos to find hand interactions and track objects.")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode (e.g., disable multiprocessing for SAM-2).")
    parser.add_argument("--use-dense-segmentation", action="store_true", help="Use a denser set of parameters for SAM to find smaller objects.")
    parser.add_argument("--workers-per-gpu", type=int, default=4, help="Number of worker processes to spawn per GPU.")
    parser.add_argument("--save-masks-separately", action="store_true", help="Save each initial mask visualization separately instead of on a single image.")
    parser.add_argument("--save-visualizations", action="store_true", help="Save intermediate visualization videos and images.")
    parser.add_argument("--highlight-folder", type=str, default=None, help="Folder to save the highlight frame from each video.")
    parser.add_argument("--start-video-index", type=int, default=0, help="Start video index for processing.")
    parser.add_argument("--end-video-index", type=int, default=3000, help="End video index for processing.")
    args = parser.parse_args()

    # Create the centralized highlight frames folder
    if args.highlight_folder:
        os.makedirs(args.highlight_folder, exist_ok=True)

    # set multiprocessing start method to spawn (CUDA required)
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # if already set, skip

    output_base_folder = 'output_find_interact_frame'
    output_data_folder = './data_hoi'

    # Create output directories for YOLOv8 fine-tuning data
    os.makedirs(os.path.join(output_data_folder, 'images', 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_data_folder, 'images', 'val'), exist_ok=True)
    os.makedirs(os.path.join(output_data_folder, 'images', 'test'), exist_ok=True)
    os.makedirs(os.path.join(output_data_folder, 'labels', 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_data_folder, 'labels', 'val'), exist_ok=True)

    # sample videos
    video_files = sample_videos(num_samples=90000)[args.start_video_index:args.end_video_index]
    video_files = filter_processed_videos(video_files, output_data_folder)

    if not args.debug:
        # use multiprocessing to process videos
        num_gpus = torch.cuda.device_count()
        if num_gpus > 0:
            # Distribute videos among GPUs
            video_chunks = [[] for _ in range(num_gpus)]
            for i, video_file in enumerate(video_files):
                video_chunks[i % num_gpus].append(video_file)

            processes = []
            for gpu_id in range(num_gpus):
                if not video_chunks[gpu_id]:
                    continue  # Skip GPU if no videos are assigned
                p = mp.Process(target=worker_main, args=(gpu_id, video_chunks[gpu_id], args, output_data_folder, output_base_folder))
                processes.append(p)
                p.start()

            for p in processes:
                p.join()
        else:
            # Fallback to sequential processing if no GPU is available
            args.debug = True

    if args.debug:
        # process videos sequentially
        # Initialize models once for sequential processing
        device = "cuda" if torch.cuda.is_available() else "cpu"
        hand_det_model = YOLO('runs/fine_tune5/weights/last.pt')
        sam2_checkpoint = "sam2/checkpoints/sam2.1_hiera_large.pt"
        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)
        video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
        predictor = SAM2ImagePredictor(sam2)

        for video_file in tqdm(video_files, desc="Processing videos sequentially"):
            process_single_video(
                video_file,
                hand_det_model_path='runs/fine_tune5/weights/last.pt',
                output_data_folder=output_data_folder,
                output_base_folder=output_base_folder,
                args=args,
                hand_det_model=hand_det_model,
                sam2=sam2,
                video_predictor=video_predictor,
                predictor=predictor,
            )


def worker_main(gpu_id, video_chunk, args, output_data_folder, output_base_folder):
    """
    Worker process that manages a pool of subprocesses on a single GPU.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print(f"GPU manager process {os.getpid()} assigned to GPU {gpu_id}, processing {len(video_chunk)} videos.")

    process_func = partial(
        process_single_video,
        hand_det_model_path='runs/fine_tune5/weights/last.pt',
        output_data_folder=output_data_folder,
        output_base_folder=output_base_folder,
        args=args,
    )

    with mp.Pool(processes=args.workers_per_gpu) as pool:
        # Use a tqdm progress bar for this worker's chunk
        list(
            tqdm(
                pool.imap_unordered(process_func, video_chunk),
                total=len(video_chunk),
                desc=f"GPU {gpu_id}",
                position=gpu_id,  # position progress bars one under another
                leave=True))


if __name__ == "__main__":
    main()
