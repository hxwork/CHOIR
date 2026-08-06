import argparse
import json
import math
import os

import cv2
import imageio
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from tqdm import tqdm

# --- 1. Data loading helpers ---


def load_hand_data(seq_path, num_frames, hand_side=None):
    """Load hand bbox and keypoints JSON files."""
    from choir_stage3.io.video_layout import VideoLayout

    if hand_side not in (None, "left", "right"):
        raise ValueError("hand_side must be None, 'left', or 'right'")

    enabled_prefixes = {
        None: ("lh", "rh"),
        "left": ("lh",),
        "right": ("rh",),
    }[hand_side]
    layout = VideoLayout.from_root(seq_path)
    paths = {
        'lh_bbox': str(layout.bbox_json("left")),
        'rh_bbox': str(layout.bbox_json("right")),
        'lh_kpts': str(layout.keypoints_json("left")),
        'rh_kpts': str(layout.keypoints_json("right")),
    }

    data = {}
    for key, path in paths.items():
        if os.path.exists(path):
            with open(path, "r") as f:
                data[key] = json.load(f)
        else:
            data[key] = {}

    all_frames_data = []
    for i in range(num_frames):
        k = str(i)
        frame_hands = {}
        if "lh" in enabled_prefixes and k in data['lh_bbox']:
            frame_hands['lh'] = {'bbox': data['lh_bbox'].get(k)}
            if k in data['lh_kpts']:
                frame_hands['lh']['kpts'] = data['lh_kpts'].get(k)
        if "rh" in enabled_prefixes and k in data['rh_bbox']:
            frame_hands['rh'] = {'bbox': data['rh_bbox'].get(k)}
            if k in data['rh_kpts']:
                frame_hands['rh']['kpts'] = data['rh_kpts'].get(k)
        all_frames_data.append(frame_hands)
    return all_frames_data


def load_raw_frames(folder_path, frame_type='rgb'):
    """Load an image sequence; mode is 'rgb', 'mask', or 'depth'."""
    files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))], key=lambda f: int(os.path.splitext(f)[0]))

    frames = []
    for f in files:
        path = os.path.join(folder_path, f)
        if frame_type == 'mask':
            # Mask load: prefer alpha channel, else grayscale
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is not None:
                if len(img.shape) == 3 and img.shape[2] == 4:
                    # RGBA image, use alpha channel as mask
                    mask = img[:, :, 3]
                elif len(img.shape) == 3:
                    # BGR image, convert to grayscale
                    mask = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                else:
                    # Already grayscale
                    mask = img
                frames.append((mask > 128).astype(np.uint8))
        elif frame_type == 'depth':
            # Depth load: 16-bit -> float [0, 1]
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is not None:
                frames.append(img.astype(np.float32) / 65535.0)
        elif frame_type == 'metric_depth':
            # Metric depth load: 16-bit -> float [0, 1]
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is not None:
                frames.append(img.astype(np.float32) / 1000.0)
        else:  # 'rgb'
            # RGB load: BGR -> RGB
            img = cv2.imread(path)
            if img is not None:
                frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return frames


# --- 2. Core logic: smooth amodal bbox generation ---
def get_global_amodal_bbox(
        masks,
        hand_bboxes,
        padding_pixels=100,
        reliable_area_ratio=0.2,
        offset_ema_alpha=0.5,
        target_aspect_ratio=2.0  # Target aspect ratio W/H = 2.0 (2:1)
):
    num_frames = len(masks)
    if num_frames == 0:
        return []

    H_img, W_img = masks[0].shape

    # --- Step 1: extract basics and decide state ---
    areas = [cv2.countNonZero(m) for m in masks]
    max_area = np.max(areas) if any(areas) else 1
    frame_data = []

    for t in range(num_frames):
        data = {'status': 'lost', 'obj_cx': np.nan, 'obj_cy': np.nan, 'obj_w': np.nan, 'obj_h': np.nan, 'hand_cx': np.nan, 'hand_cy': np.nan}

        # A. Object Mask
        if areas[t] > 0:
            x, y, w, h = cv2.boundingRect(masks[t])
            cx, cy = x + w / 2, y + h / 2
            data.update({'obj_cx': cx, 'obj_cy': cy, 'obj_w': w, 'obj_h': h})

            if areas[t] >= max_area * reliable_area_ratio:
                data['status'] = 'reliable'
            else:
                data['status'] = 'partial'

        # B. Hand Data (Keypoints priority)
        hands = hand_bboxes[t]
        if hands:
            # (...keep existing nearest-hand selection logic...)
            # Simplified here for brevity; reuse the original Hand Parsing block.
            # Find best_hand_center and fill data['hand_cx'], data['hand_cy'].
            # Reuse the Hand Parsing block above.
            pass

        frame_data.append(data)

    df = pd.DataFrame(frame_data)

    # --- Step 2: fill missing-frame positions (hand-guided) ---
    # Need an approximate object location under occlusion so it stays in the global crop.

    inferred_cx = df['obj_cx'].copy()
    inferred_cy = df['obj_cy'].copy()
    current_offset = None

    # Simple size estimate for interaction-distance checks
    w_est = df['obj_w'].max() if not df['obj_w'].isnull().all() else 100
    h_est = df['obj_h'].max() if not df['obj_h'].isnull().all() else 100
    thresh = max(w_est, h_est) * 1.5

    for t in range(num_frames):
        status = df.loc[t, 'status']
        hand_cx, hand_cy = df.loc[t, 'hand_cx'], df.loc[t, 'hand_cy']
        obj_cx, obj_cy = df.loc[t, 'obj_cx'], df.loc[t, 'obj_cy']

        if not pd.isna(hand_cx):
            if status == 'reliable':
                raw_offset = np.array([obj_cx - hand_cx, obj_cy - hand_cy])
                if current_offset is None:
                    current_offset = raw_offset
                else:
                    current_offset = (1 - offset_ema_alpha) * current_offset + offset_ema_alpha * raw_offset

            elif status != 'reliable' and current_offset is not None:
                # Simple interaction flag: treat as interacting when an offset exists
                # Coarse estimate is enough to expand the bbox; fine damping is elsewhere.
                inferred_cx[t] = hand_cx + current_offset[0]
                inferred_cy[t] = hand_cy + current_offset[1]

    # Interpolate remaining NaNs
    df['cx'] = inferred_cx.interpolate(method='linear').ffill().bfill()
    df['cy'] = inferred_cy.interpolate(method='linear').ffill().bfill()

    # Also fill width/height (use max to keep room)
    df['w'] = df['obj_w'].max()  # take max extent
    df['h'] = df['obj_h'].max()

    # --- Step 3: compute global union bbox ---
    # Build the smallest rectangle covering inferred centers/sizes for all frames

    # Per-frame bbox (x1, y1, x2, y2)
    # Use inferred positions so occluded motion is still covered
    x1s = df['cx'] - df['w'] / 2
    y1s = df['cy'] - df['h'] / 2
    x2s = df['cx'] + df['w'] / 2
    y2s = df['cy'] + df['h'] / 2

    # Global extrema
    global_x1 = np.min(x1s)
    global_y1 = np.min(y1s)
    global_x2 = np.max(x2s)
    global_y2 = np.max(y2s)

    # Add padding
    global_x1 -= padding_pixels
    global_y1 -= padding_pixels
    global_x2 += padding_pixels
    global_y2 += padding_pixels

    # --- Step 4: enforce aspect ratio (2:1) ---
    curr_w = global_x2 - global_x1
    curr_h = global_y2 - global_y1
    curr_center_x = (global_x1 + global_x2) / 2
    curr_center_y = (global_y1 + global_y2) / 2

    target_w = 0
    target_h = 0

    # Keep the current box; only enlarge, never shrink
    if curr_w / curr_h > target_aspect_ratio:
        # Too wide -> keep width, grow height
        target_w = curr_w
        target_h = curr_w / target_aspect_ratio
    else:
        # Too tall -> keep height, grow width
        target_h = curr_h
        target_w = curr_h * target_aspect_ratio

    # Recompute coordinates
    final_x1 = int(curr_center_x - target_w / 2)
    final_y1 = int(curr_center_y - target_h / 2)
    final_x2 = int(curr_center_x + target_w / 2)
    final_y2 = int(curr_center_y + target_h / 2)

    # Optional boundary clamp: moving the center would break relative object motion
    # For amodal models, prefer keeping relative motion over strict in-image clipping.
    # No strict clip here; let the external crop handle padding.

    global_bbox = [final_x1, final_y1, final_x2, final_y2]

    # Return the same bbox for every frame
    return [global_bbox] * num_frames


def get_smooth_amodal_bbox(
        masks,
        hand_bboxes,
        padding_pixels=100,
        reliable_area_ratio=0.2,
        savgol_window=15,
        savgol_poly=2,
        offset_ema_alpha=0.5,  # Offset update rate when the mask is reliable
        guidance_damping=0.5  # Hand-follow rate when the mask is unreliable (0.0~1.0)
    # 0.1: heavy/stable, 0.5: medium, 1.0: no damping (hard follow)
):
    num_frames = len(masks)
    if num_frames == 0:
        return []

    # Precompute areas
    areas = [cv2.countNonZero(m) for m in masks]
    max_area = np.max(areas) if any(areas) else 1

    frame_data = []

    for t in range(num_frames):
        data = {
            'status': 'lost',
            'obj_cx': np.nan,
            'obj_cy': np.nan,
            'obj_w': np.nan,
            'obj_h': np.nan,
            'hand_cx': np.nan,
            'hand_cy': np.nan,
            'dist_hand_obj': np.inf
        }

        # A. Parse object mask
        area = areas[t]
        if area > 0:
            x, y, w, h = cv2.boundingRect(masks[t])
            cx, cy = x + w / 2, y + h / 2

            data.update({'obj_cx': cx, 'obj_cy': cy, 'obj_w': w, 'obj_h': h})

            if area >= max_area * reliable_area_ratio:
                data['status'] = 'reliable'
            else:
                data['status'] = 'partial'

        # B. Parse hand bbox (prefer keypoints)
        hands = hand_bboxes[t]
        if hands:
            target_cx = data['obj_cx'] if not np.isnan(data['obj_cx']) else (frame_data[-1]['obj_cx'] if t > 0 else 0)
            target_cy = data['obj_cy'] if not np.isnan(data['obj_cy']) else (frame_data[-1]['obj_cy'] if t > 0 else 0)

            best_hand_center = None
            min_dist = float('inf')

            TIP_INDICES = [4, 8, 12, 16, 20]

            for hand_info in hands.values():
                hcx, hcy = None, None

                # Prefer 1: mean of fingertip keypoints
                if 'kpts' in hand_info and hand_info['kpts']:
                    tips = [hand_info['kpts'][i] for i in TIP_INDICES if i < len(hand_info['kpts'])]
                    if tips:
                        tip_coords = np.array([[tip[0], tip[1]] for tip in tips if len(tip) >= 2])
                        if tip_coords.shape[0] > 0:
                            hcx, hcy = np.mean(tip_coords, axis=0)

                # Prefer 2: fall back to bbox center
                if hcx is None and 'bbox' in hand_info and hand_info['bbox']:
                    h_box = hand_info['bbox']
                    hcx = (h_box[0] + h_box[2]) / 2
                    hcy = (h_box[1] + h_box[3]) / 2

                if hcx is not None:
                    dist = (hcx - target_cx)**2 + (hcy - target_cy)**2
                    if dist < min_dist:
                        min_dist = dist
                        best_hand_center = (hcx, hcy)

            if best_hand_center:
                data['hand_cx'], data['hand_cy'] = best_hand_center
                data['dist_hand_obj'] = np.sqrt(min_dist)

        frame_data.append(data)

    df = pd.DataFrame(frame_data)

    # --- Step 1: lock size ---
    if not df[df['status'] == 'reliable'].empty:
        w_fix = df[df['status'] == 'reliable']['obj_w'].quantile(0.95)
        h_fix = df[df['status'] == 'reliable']['obj_h'].quantile(0.95)
    else:
        w_fix = df['obj_w'].max()
        h_fix = df['obj_h'].max()

    if pd.isna(w_fix) or w_fix == 0:
        w_fix = 100
    if pd.isna(h_fix) or h_fix == 0:
        h_fix = 100

    # --- Step 2: infer center (with damping) ---
    inferred_cx = df['obj_cx'].copy()
    inferred_cy = df['obj_cy'].copy()

    current_offset = None
    interaction_thresh = max(w_fix, h_fix) * 1.5

    for t in range(num_frames):
        status = df.loc[t, 'status']
        hand_cx, hand_cy = df.loc[t, 'hand_cx'], df.loc[t, 'hand_cy']
        obj_cx, obj_cy = df.loc[t, 'obj_cx'], df.loc[t, 'obj_cy']
        dist = df.loc[t, 'dist_hand_obj']

        if not pd.isna(hand_cx):

            # Case A: reliable mask -> update offset (truth)
            if status == 'reliable':
                raw_offset = np.array([obj_cx - hand_cx, obj_cy - hand_cy])
                if current_offset is None:
                    current_offset = raw_offset
                else:
                    current_offset = (1 - offset_ema_alpha) * current_offset + offset_ema_alpha * raw_offset

            # Case B: unreliable mask -> hand-guided inference
            elif (status != 'reliable') and (dist < interaction_thresh or current_offset is not None):
                if current_offset is not None:
                    # 1. Target position
                    target_cx = hand_cx + current_offset[0]
                    target_cy = hand_cy + current_offset[1]

                    # 2. Previous-frame position (history)
                    prev_cx, prev_cy = None, None
                    if t > 0:
                        prev_cx = inferred_cx[t - 1]
                        prev_cy = inferred_cy[t - 1]

                    # 3. Damped follow
                    if prev_cx is not None and not np.isnan(prev_cx):
                        # Apply damping when history exists
                        # Current = (1 - alpha) * Prev + alpha * Target
                        alpha = guidance_damping
                        pred_cx = (1 - alpha) * prev_cx + alpha * target_cx
                        pred_cy = (1 - alpha) * prev_cy + alpha * target_cy
                    else:
                        # First interaction frame or NaN history: jump to target
                        pred_cx, pred_cy = target_cx, target_cy

                    # 4. Assign
                    inferred_cx[t] = pred_cx
                    inferred_cy[t] = pred_cy

    # Fill gaps
    df['cx'] = inferred_cx.interpolate(method='spline', order=2).ffill().bfill()
    df['cy'] = inferred_cy.interpolate(method='spline', order=2).ffill().bfill()

    # --- Step 3: Savitzky–Golay post-filter ---
    win = min(savgol_window, num_frames)
    if win % 2 == 0:
        win -= 1
    if win > 3:
        df['cx'] = savgol_filter(df['cx'], win, savgol_poly)
        df['cy'] = savgol_filter(df['cy'], win, savgol_poly)

    # --- Step 4: emit final bboxes ---
    final_bboxes = []
    if w_fix / h_fix > 1:
        final_h = max(h_fix + padding_pixels * 2, (w_fix + padding_pixels * 2) / 2)
        final_w = final_h * 2
    else:
        final_w = max(w_fix + padding_pixels * 2, (h_fix + padding_pixels * 2) * 2)
        final_h = final_w / 2

    for t in range(num_frames):
        cx, cy = df.loc[t, 'cx'], df.loc[t, 'cy']
        x1 = int(cx - final_w / 2)
        y1 = int(cy - final_h / 2)
        x2 = int(cx + final_w / 2)
        y2 = int(cy + final_h / 2)
        final_bboxes.append([x1, y1, x2, y2])

    return final_bboxes
