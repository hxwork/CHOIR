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
    discover_input_videos,
    layout_for_video,
    visualization_dir_for_video,
)

MODULE_ROOT = Path(__file__).resolve().parent
DEFAULT_SAM2_CHECKPOINT = MODULE_ROOT / "sam2" / "checkpoints" / "sam2.1_hiera_large.pt"

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


def draw_detections(img, right_hand_info=None, left_hand_info=None, object_infos=None):
    """
    Draw hand and object detection results on image.

    Args:
        img: input image (numpy array)
        right_hand_info: right hand info [class_id, x_center, y_center, w, h, kpt1_x, kpt1_y, kpt1_conf, ...]
        left_hand_info: left hand info [class_id, x_center, y_center, w, h, kpt1_x, kpt1_y, kpt1_conf, ...]
        object_infos: list of object infos, each is [x_center, y_center, w, h]
    
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

        # Draw keypoints (starting from index 5, each keypoint has [x, y, conf])
        keypoints_data = hand_info[6:]
        num_keypoints = len(keypoints_data) // 3

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

    # --- 1. Forward Tracking ---
    frames_forward = frames[init_frame_idx:]
    frames_forward_np = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_forward])

    video_segments_forward = {}
    if len(frames_forward_np) > 0:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float32):
            state_fwd = video_predictor.init_state(frames_forward_np)
            y_indices, x_indices = np.where(init_mask)
            box_xyxy = np.array([x_indices.min(), y_indices.min(), x_indices.max(), y_indices.max()], dtype=np.float32)
            obj_id = 1
            video_predictor.add_new_points_or_box(inference_state=state_fwd, frame_idx=0, obj_id=obj_id, box=box_xyxy)
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
            y_indices, x_indices = np.where(init_mask)
            box_xyxy = np.array([x_indices.min(), y_indices.min(), x_indices.max(), y_indices.max()], dtype=np.float32)
            obj_id = 1
            video_predictor.add_new_points_or_box(inference_state=state_bwd, frame_idx=0, obj_id=obj_id, box=box_xyxy)
            for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(state_bwd):
                masks = (out_mask_logits > 0.0).cpu().numpy()
                # Map back to original frame index
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

            # Apply all masks at once
            img_to_write = cv2.addWeighted(mask_overlay, alpha, img_to_write, 1 - alpha, 0)

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

            # Convert BGR to RGB for imageio
            writer.append_data(cv2.cvtColor(img_to_write, cv2.COLOR_BGR2RGB))

            # 4. Pause on the specific frame where the initial mask was generated.
            if frame_idx == pause_frame_idx:
                pause_duration_seconds = 2
                num_pause_frames = int(fps * pause_duration_seconds)
                # We've already written the frame once, so write it N-1 more times
                for _ in range(num_pause_frames - 1):
                    writer.append_data(cv2.cvtColor(img_to_write, cv2.COLOR_BGR2RGB))

    print(f"SAM tracking visualization saved to {output_path}")


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
                        if confidence > max_conf_left:
                            max_conf_left = confidence
                            best_left_hand = (box, keypoints[idx], keypoints_conf[idx])
                    elif class_id == 1:  # Right Hand
                        if confidence > max_conf_right:
                            max_conf_right = confidence
                            best_right_hand = (box, keypoints[idx], keypoints_conf[idx])
                    elif class_id == 2:  # Object
                        current_objects.append(box.tolist() + [confidence])

                # Process and store best hand detections for the frame
                if best_left_hand:
                    box, kpts, kpts_conf = best_left_hand
                    kpts_flat = [coord for kpt, c in zip(kpts, kpts_conf) for coord in [kpt[0], kpt[1], c > 0.5]]
                    left_hand_tracks[t] = [0, max_conf_left] + box.tolist() + kpts_flat

                if best_right_hand:
                    box, kpts, kpts_conf = best_right_hand
                    kpts_flat = [coord for kpt, c in zip(kpts, kpts_conf) for coord in [kpt[0], kpt[1], c > 0.5]]
                    right_hand_tracks[t] = [1, max_conf_right] + box.tolist() + kpts_flat

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


def process_video_with_visualization(video_path, hand_det_model, sam_predictor, sam_video_predictor, conf, output_data_dir, visualize=True):
    """
    Process a single video, generate an initial mask, and track it through the video.
    """
    video_name = Path(video_path).stem
    output_data_dir = Path(output_data_dir)

    # Keep temporary frames inside this video's output directory.
    temp_frames_folder = output_data_dir / video_name / '.tmp_frames'
    temp_frames_folder.mkdir(parents=True, exist_ok=True)

    try:
        # Extract frames into memory
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()

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

        # --- Calculate hand bounding boxes from keypoints ---
        rh_bboxes = {}
        for frame_idx, track_info in right_hand_tracks.items():
            keypoints_data = track_info[6:]
            num_keypoints = len(keypoints_data) // 3
            if num_keypoints > 0:
                all_kpts_x = [keypoints_data[i * 3] for i in range(num_keypoints)]
                all_kpts_y = [keypoints_data[i * 3 + 1] for i in range(num_keypoints)]
                if len(all_kpts_x) >= 2:
                    img_h, img_w = frames[frame_idx].shape[:2]
                    x1 = float(min(all_kpts_x) * img_w)
                    y1 = float(min(all_kpts_y) * img_h)
                    x2 = float(max(all_kpts_x) * img_w)
                    y2 = float(max(all_kpts_y) * img_h)
                    rh_bboxes[frame_idx] = [x1, y1, x2, y2]

        lh_bboxes = {}
        for frame_idx, track_info in left_hand_tracks.items():
            keypoints_data = track_info[6:]
            num_keypoints = len(keypoints_data) // 3
            if num_keypoints > 0:
                all_kpts_x = [keypoints_data[i * 3] for i in range(num_keypoints)]
                all_kpts_y = [keypoints_data[i * 3 + 1] for i in range(num_keypoints)]
                if len(all_kpts_x) >= 2:
                    img_h, img_w = frames[frame_idx].shape[:2]
                    x1 = float(min(all_kpts_x) * img_w)
                    y1 = float(min(all_kpts_y) * img_h)
                    x2 = float(max(all_kpts_x) * img_w)
                    y2 = float(max(all_kpts_y) * img_h)
                    lh_bboxes[frame_idx] = [x1, y1, x2, y2]

        # --- Extract hand keypoints ---
        rh_keypoints = {}
        for frame_idx, track_info in right_hand_tracks.items():
            keypoints_data = track_info[6:]
            num_keypoints = len(keypoints_data) // 3
            if num_keypoints > 0:
                img_h, img_w = frames[frame_idx].shape[:2]
                kpts_for_frame = []
                for i in range(num_keypoints):
                    kpt_x = float(keypoints_data[i * 3] * img_w)
                    kpt_y = float(keypoints_data[i * 3 + 1] * img_h)
                    kpts_for_frame.append([kpt_x, kpt_y])
                rh_keypoints[frame_idx] = kpts_for_frame

        lh_keypoints = {}
        for frame_idx, track_info in left_hand_tracks.items():
            keypoints_data = track_info[6:]
            num_keypoints = len(keypoints_data) // 3
            if num_keypoints > 0:
                img_h, img_w = frames[frame_idx].shape[:2]
                kpts_for_frame = []
                for i in range(num_keypoints):
                    kpt_x = float(keypoints_data[i * 3] * img_w)
                    kpt_y = float(keypoints_data[i * 3 + 1] * img_h)
                    kpts_for_frame.append([kpt_x, kpt_y])
                lh_keypoints[frame_idx] = kpts_for_frame

        # --- Track Hands with SAM ---
        right_hand_video_segments, right_hand_pause_idx = generate_and_track_hand_mask(right_hand_tracks, frames, sam_predictor, sam_video_predictor,
                                                                                       "Right Hand")
        left_hand_video_segments, left_hand_pause_idx = generate_and_track_hand_mask(left_hand_tracks, frames, sam_predictor, sam_video_predictor, "Left Hand")

        # --- Step 1: Collect ALL candidates from the video and sort by confidence ---
        all_candidates = []
        sorted_object_frames = sorted(object_tracks.keys())
        for frame_idx in sorted_object_frames:
            for object_info in object_tracks[frame_idx]:
                all_candidates.append({'frame_idx': frame_idx, 'conf': object_info[4], 'object_info': object_info})

        # Sort all potential candidates by confidence, descending
        all_candidates.sort(key=lambda x: x['conf'], reverse=True)

        # --- Step 2: Process sorted candidates in batches and stop when a valid mask is found ---
        best_candidate = None
        BATCH_SIZE = 8  # This is # of bboxes, can be larger than image batch size

        for i in range(0, len(all_candidates), BATCH_SIZE):
            if best_candidate:  # If we found a result in the previous batch, stop.
                break

            chunk_candidates = all_candidates[i:i + BATCH_SIZE]

            # Group candidates in this chunk by frame_idx to prepare for batch inference
            from collections import defaultdict
            chunk_grouped_by_frame = defaultdict(list)
            for cand in chunk_candidates:
                chunk_grouped_by_frame[cand['frame_idx']].append(cand)

            # Prepare inputs for sam_predictor.predict_batch
            images_sub_batch = [cv2.cvtColor(frames[idx], cv2.COLOR_BGR2RGB) for idx in chunk_grouped_by_frame.keys()]

            boxes_sub_batch = []
            flat_chunk_candidates_ordered = []
            for frame_idx in chunk_grouped_by_frame.keys():
                frame_candidates = chunk_grouped_by_frame[frame_idx]
                bboxes_for_this_image = []
                img_h, img_w = frames[frame_idx].shape[:2]
                for cand in frame_candidates:
                    xc, yc, w, h, _ = cand['object_info']
                    x1 = (xc - w * 1.0 / 2) * img_w
                    y1 = (yc - h * 1.0 / 2) * img_h
                    x2 = (xc + w * 1.0 / 2) * img_w
                    y2 = (yc + h * 1.0 / 2) * img_h
                    bboxes_for_this_image.append([x1, y1, x2, y2])
                    cand['bbox_xyxy'] = [x1, y1, x2, y2]
                    flat_chunk_candidates_ordered.append(cand)
                boxes_sub_batch.append(np.array(bboxes_for_this_image))

            # Run batch inference
            sam_predictor.set_image_batch(images_sub_batch)
            masks_batch, _, _ = sam_predictor.predict_batch(None, None, box_batch=boxes_sub_batch, multimask_output=True)

            # Process results from the chunk
            successful_candidates_in_batch = []
            candidate_idx_in_chunk = 0
            for masks_for_one_image in masks_batch:
                # This logic robustly handles cases where an image has one or multiple boxes.
                # If there's one box, the tensor might be 3D (num_masks, H, W).
                # If multiple, it's 4D (num_boxes, num_masks, H, W).
                # We ensure it's always 4D for consistent processing.
                if masks_for_one_image.ndim == 3:
                    masks_for_one_image = np.expand_dims(masks_for_one_image, axis=0)  # Add a dimension for the single box

                # Now, we can safely iterate through the boxes for this image.
                for box_masks_tensor in masks_for_one_image:
                    # box_masks_tensor has shape (num_masks, H, W). Take the last one.
                    best_mask_tensor = box_masks_tensor[-1, :, :]

                    original_info = flat_chunk_candidates_ordered[candidate_idx_in_chunk]

                    # Already numpy array
                    mask = best_mask_tensor.astype(bool)
                    mask = remove_small_regions(mask)

                    if np.any(mask):
                        successful_candidates_in_batch.append({
                            'score': original_info['conf'],
                            'mask': mask,
                            'frame': frames[original_info['frame_idx']],
                            'frame_idx': original_info['frame_idx'],
                            'bbox_xyxy': original_info['bbox_xyxy']
                        })
                    # This must be inside the inner loop to correctly map each box.
                    candidate_idx_in_chunk += 1

            # If we found any valid mask, select the best (earliest frame) and set up to stop.
            if successful_candidates_in_batch:
                successful_candidates_in_batch.sort(key=lambda x: (x['frame_idx'], -x['score']))
                best_candidate = successful_candidates_in_batch[0]

        # --- Step 3: Final selection and setup ---
        if not best_candidate:
            print(f"Could not generate any valid initial mask for any object in {video_name}.")
            return video_name, 0, False, None, True

        initial_mask = best_candidate['mask']
        object_init_frame = best_candidate['frame']
        first_object_frame_idx = best_candidate['frame_idx']
        best_bbox_xyxy = best_candidate['bbox_xyxy']

        # --- Step 1 (Visualization): Save visualization of the initial mask ---
        mask_vis_folder = None
        if visualize:
            vis_image1 = object_init_frame.copy()
            color1 = np.array([0, 0, 255])  # Red
            vis_image1[initial_mask] = (vis_image1[initial_mask] * 0.5 + color1 * 0.5).astype(np.uint8)

            # Use the bbox from the best candidate to draw on the visualization
            cv2.rectangle(vis_image1, (int(best_bbox_xyxy[0]), int(best_bbox_xyxy[1])), (int(best_bbox_xyxy[2]), int(best_bbox_xyxy[3])), (0, 255, 0), 2)

            mask_vis_folder = visualization_dir_for_video(output_data_dir, video_path)
            mask_vis_folder.mkdir(parents=True, exist_ok=True)
            mask_vis_path = mask_vis_folder / "initial_mask.jpg"
            cv2.imwrite(str(mask_vis_path), vis_image1)

        # --- Step 2: Propagate mask with Video Predictor ---
        video_segments = sam_video_tracking(sam_video_predictor, frames, initial_mask, first_object_frame_idx)

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
        if rh_bboxes:
            layout.bbox_dir.mkdir(parents=True, exist_ok=True)
            with open(layout.bbox_json("right"), 'w') as f:
                json.dump(rh_bboxes, f, indent=4)
        if lh_bboxes:
            layout.bbox_dir.mkdir(parents=True, exist_ok=True)
            with open(layout.bbox_json("left"), 'w') as f:
                json.dump(lh_bboxes, f, indent=4)

        # --- Save hand keypoints ---
        if rh_keypoints:
            layout.keypoints_dir.mkdir(parents=True, exist_ok=True)
            with open(layout.keypoints_json("right"), 'w') as f:
                json.dump(rh_keypoints, f, indent=4)
        if lh_keypoints:
            layout.keypoints_dir.mkdir(parents=True, exist_ok=True)
            with open(layout.keypoints_json("left"), 'w') as f:
                json.dump(lh_keypoints, f, indent=4)

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
        if visualize and 0 in video_segments:
            first_frame_mask = video_segments[0][1]  # obj_id is 1
            vis_image_ff = frames[0].copy()
            color_ff = np.array([255, 0, 0])  # Blue
            vis_image_ff[first_frame_mask.squeeze()] = (vis_image_ff[first_frame_mask.squeeze()] * 0.5 + color_ff * 0.5).astype(np.uint8)
            mask_vis_path_ff = mask_vis_folder / "first_frame_tracked_mask.jpg"
            cv2.imwrite(str(mask_vis_path_ff), vis_image_ff)
        elif visualize:
            print(f"Warning: No mask propagated to the first frame for video {video_name}")

        return video_name, len(sorted_object_frames), True, output_video_path, True

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

    video_name, num_frames, is_successful, output_video_path, success = process_video_with_visualization(
        video_path,
        hand_det_model,
        sam_predictor,
        sam_video_predictor,
        conf,
        output_data_dir,
        visualize,
    )

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

    # Use a multiprocessing pool to process videos on this GPU
    with mp.Pool(processes=args.num_workers) as pool:
        for result in tqdm(pool.imap_unordered(process_video_worker, video_args_chunk), total=len(video_args_chunk), desc=f"GPU {gpu_id}", position=gpu_id):
            shared_results.append(result)

    print(f"Worker for GPU {gpu_id} finished.")


def load_my_data_videos(data_dir=DEFAULT_DATA_DIR, output_dir=DEFAULT_OUTPUT_DATA_DIR, video_ids=None):
    """
    Loads flat input videos from the repository data directory and copies them to the
    per-video output folder for downstream processing.
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)

    video_files = discover_input_videos(data_dir, video_ids)
    if not video_files:
        print(f"No flat mp4 input videos found in {data_dir}")
        return []

    print(f"Found {len(video_files)} video(s) in {data_dir}: {[v.name for v in video_files]}")

    success_count = 0
    processing_paths = []
    for idx, video_path in enumerate(video_files, 1):
        video_id = video_path.stem
        output_path = output_dir / video_id / f"{video_id}.mp4"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"[{idx}/{len(video_files)}] Copying {video_path.name}...")
        try:
            shutil.copy2(video_path, output_path)
            print(f"  Saved to {output_path}")
            success_count += 1
            processing_paths.append(video_path)
        except Exception as e:
            print(f"  Failed to copy {video_path.name}: {e}")

    print(f"\n{'='*60}")
    print(f"Completed! Successfully copied {success_count}/{len(video_files)} videos")
    print(f"Output directory: {output_dir.absolute()}")
    print(f"{'='*60}")
    return processing_paths


def build_arg_parser():
    import argparse

    parser = argparse.ArgumentParser(description='Detect and segment hands and interacting objects in videos')
    parser.add_argument('--data_source',
                        type=str,
                        default='local',
                        choices=['local', 'my_data'],
                        help='"local" loads flat videos from --data_dir; "my_data" is a legacy alias '
                             '(default: local)')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of parallel workers PER GPU (default: 4)')
    parser.add_argument('--model', type=str, default='models/tasterob_hoi_detector.pt', help='Path to the hand and object detection model')
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold for detection (default: 0.5)')
    parser.add_argument('--video_id', type=str, nargs='+', default=None,
                        help='One or more flat input video IDs. Applies to local input data.')
    parser.add_argument('--data_dir', type=str, default=str(DEFAULT_DATA_DIR),
                        help=f'Directory containing flat {{video_id}}.mp4 inputs (default: {DEFAULT_DATA_DIR})')
    parser.add_argument('--output_dir', type=str, default=str(DEFAULT_OUTPUT_DATA_DIR),
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

    # Load videos based on selected data source
    print("=" * 80)
    print(f"Data source: local videos in {args.data_dir}")
    print("=" * 80)
    videos_to_process = load_my_data_videos(args.data_dir, args.output_dir, args.video_id)
    if not videos_to_process:
        print(f"No videos found in {args.data_dir}. Please add flat mp4 files to that directory.")
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
        if args.visualize and 'output_video_path' in result and result['output_video_path']:
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
