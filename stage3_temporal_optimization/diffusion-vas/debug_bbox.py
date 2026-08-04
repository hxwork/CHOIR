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

# --- 1. 数据加载辅助函数 ---


def load_hand_data(seq_path, num_frames, hand_side=None):
    """加载手部 BBox 和 Keypoints json 文件"""
    if hand_side not in (None, "left", "right"):
        raise ValueError("hand_side must be None, 'left', or 'right'")

    enabled_prefixes = {
        None: ("lh", "rh"),
        "left": ("lh",),
        "right": ("rh",),
    }[hand_side]
    paths = {
        'lh_bbox': os.path.join(seq_path, 'lh_bbox.json'),
        'rh_bbox': os.path.join(seq_path, 'rh_bbox.json'),
        'lh_kpts': os.path.join(seq_path, 'lh_keypoints.json'),
        'rh_kpts': os.path.join(seq_path, 'rh_keypoints.json'),
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
    """快速加载图片序列，支持 'rgb', 'mask', 'depth'"""
    files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))], key=lambda f: int(os.path.splitext(f)[0]))

    frames = []
    for f in files:
        path = os.path.join(folder_path, f)
        if frame_type == 'mask':
            # Mask 读取: 优先 Alpha 通道, 否则回退到灰度
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
            # Depth 读取: 16-bit -> float [0, 1]
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is not None:
                frames.append(img.astype(np.float32) / 65535.0)
        elif frame_type == 'metric_depth':
            # Metric Depth 读取: 16-bit -> float [0, 1]
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is not None:
                frames.append(img.astype(np.float32) / 1000.0)
        else:  # 'rgb'
            # RGB 读取：BGR -> RGB
            img = cv2.imread(path)
            if img is not None:
                frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return frames


# --- 2. 核心逻辑：平滑 Amodal BBox 生成 ---
def get_global_amodal_bbox(
        masks,
        hand_bboxes,
        padding_pixels=100,
        reliable_area_ratio=0.2,
        offset_ema_alpha=0.5,
        target_aspect_ratio=2.0  # 目标宽高比 W/H = 2.0 (即 2:1)
):
    num_frames = len(masks)
    if num_frames == 0:
        return []

    H_img, W_img = masks[0].shape

    # --- Step 1: 基础数据提取 & 状态判定 ---
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
            # (...保持你原有的寻找最近手的逻辑...)
            # 为节省篇幅，这里简化，假设你用的是你原来那段逻辑
            # 找到 best_hand_center 并填入 data['hand_cx'], data['hand_cy']
            # 请直接复用你上面的 Hand Parsing 代码块
            pass

        frame_data.append(data)

    df = pd.DataFrame(frame_data)

    # --- Step 2: 补全丢失帧的位置 (利用 Hand-Guided) ---
    # 我们需要知道物体被遮挡时大概在哪，防止它跑出全局 Crop 框

    inferred_cx = df['obj_cx'].copy()
    inferred_cy = df['obj_cy'].copy()
    current_offset = None

    # 简单的尺寸估计，用于判定交互距离
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
                # 简单判定交互：如果有offset记录，就假设在交互
                # 这里不需要太精细的阻尼，只需要知道大概位置来撑开 BBox
                inferred_cx[t] = hand_cx + current_offset[0]
                inferred_cy[t] = hand_cy + current_offset[1]

    # 插值填补剩余 NaN
    df['cx'] = inferred_cx.interpolate(method='linear').ffill().bfill()
    df['cy'] = inferred_cy.interpolate(method='linear').ffill().bfill()

    # 同时也填补宽高 (用最大值撑开)
    df['w'] = df['obj_w'].max()  # 简单粗暴，取最大
    df['h'] = df['obj_h'].max()

    # --- Step 3: 计算全局包围盒 (Global Union BBox) ---
    # 我们利用每一帧推演出的中心点和尺寸，计算覆盖所有帧的最小矩形

    # 计算每一帧的 bbox (x1, y1, x2, y2)
    # 注意：这里用推演出的位置，能包含被遮挡时的运动范围
    x1s = df['cx'] - df['w'] / 2
    y1s = df['cy'] - df['h'] / 2
    x2s = df['cx'] + df['w'] / 2
    y2s = df['cy'] + df['h'] / 2

    # 取全局极值
    global_x1 = np.min(x1s)
    global_y1 = np.min(y1s)
    global_x2 = np.max(x2s)
    global_y2 = np.max(y2s)

    # 加上 Padding
    global_x1 -= padding_pixels
    global_y1 -= padding_pixels
    global_x2 += padding_pixels
    global_y2 += padding_pixels

    # --- Step 4: 强制宽高比 (Aspect Ratio 2:1) ---
    curr_w = global_x2 - global_x1
    curr_h = global_y2 - global_y1
    curr_center_x = (global_x1 + global_x2) / 2
    curr_center_y = (global_y1 + global_y2) / 2

    target_w = 0
    target_h = 0

    # 逻辑：我们要包含住当前的框，只能变大不能变小
    if curr_w / curr_h > target_aspect_ratio:
        # 当前太宽了 -> 宽度不动，增加高度
        target_w = curr_w
        target_h = curr_w / target_aspect_ratio
    else:
        # 当前太高了 -> 高度不动，增加宽度
        target_h = curr_h
        target_w = curr_h * target_aspect_ratio

    # 重新计算坐标
    final_x1 = int(curr_center_x - target_w / 2)
    final_y1 = int(curr_center_y - target_h / 2)
    final_x2 = int(curr_center_x + target_w / 2)
    final_y2 = int(curr_center_y + target_h / 2)

    # 边界检查 (可选：如果要保证不出图，可能需要移动中心，但这会破坏物体在画面中的相对位置)
    # 对于 amodal 模型，通常 padding 0 即可，保持物体相对运动更重要。
    # 这里不做 strict clip，由外部 crop 函数处理 padding。

    global_bbox = [final_x1, final_y1, final_x2, final_y2]

    # 返回每一帧都一样的 BBox
    return [global_bbox] * num_frames


def get_smooth_amodal_bbox(
        masks,
        hand_bboxes,
        padding_pixels=100,
        reliable_area_ratio=0.2,
        savgol_window=15,
        savgol_poly=2,
        offset_ema_alpha=0.5,  # 这一帧Mask可靠时，更新Offset的速率
        guidance_damping=0.5  # Mask不可靠时，物体跟随手的速率 (0.0~1.0)
    # 0.1: 非常稳/重，0.5: 适中，1.0: 无阻尼(硬跟随)
):
    num_frames = len(masks)
    if num_frames == 0:
        return []

    # 预计算面积
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

        # A. 解析 Object Mask
        area = areas[t]
        if area > 0:
            x, y, w, h = cv2.boundingRect(masks[t])
            cx, cy = x + w / 2, y + h / 2

            data.update({'obj_cx': cx, 'obj_cy': cy, 'obj_w': w, 'obj_h': h})

            if area >= max_area * reliable_area_ratio:
                data['status'] = 'reliable'
            else:
                data['status'] = 'partial'

        # B. 解析 Hand BBox (优先使用 Keypoints)
        hands = hand_bboxes[t]
        if hands:
            target_cx = data['obj_cx'] if not np.isnan(data['obj_cx']) else (frame_data[-1]['obj_cx'] if t > 0 else 0)
            target_cy = data['obj_cy'] if not np.isnan(data['obj_cy']) else (frame_data[-1]['obj_cy'] if t > 0 else 0)

            best_hand_center = None
            min_dist = float('inf')

            TIP_INDICES = [4, 8, 12, 16, 20]

            for hand_info in hands.values():
                hcx, hcy = None, None

                # 优先1: 使用指尖关键点均值
                if 'kpts' in hand_info and hand_info['kpts']:
                    tips = [hand_info['kpts'][i] for i in TIP_INDICES if i < len(hand_info['kpts'])]
                    if tips:
                        tip_coords = np.array([[tip[0], tip[1]] for tip in tips if len(tip) >= 2])
                        if tip_coords.shape[0] > 0:
                            hcx, hcy = np.mean(tip_coords, axis=0)

                # 优先2: 回退到 BBox 中心
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

    # --- Step 1: 尺寸锁定 ---
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

    # --- Step 2: 中心点推演 (带阻尼) ---
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

            # Case A: Mask 可靠 -> 更新 Offset (Truth)
            if status == 'reliable':
                raw_offset = np.array([obj_cx - hand_cx, obj_cy - hand_cy])
                if current_offset is None:
                    current_offset = raw_offset
                else:
                    current_offset = (1 - offset_ema_alpha) * current_offset + offset_ema_alpha * raw_offset

            # Case B: Mask 不可靠 -> Hand Guided Inference
            elif (status != 'reliable') and (dist < interaction_thresh or current_offset is not None):
                if current_offset is not None:
                    # 1. 计算目标位置 (Target)
                    target_cx = hand_cx + current_offset[0]
                    target_cy = hand_cy + current_offset[1]

                    # 2. 获取上一帧的位置 (History)
                    prev_cx, prev_cy = None, None
                    if t > 0:
                        prev_cx = inferred_cx[t - 1]
                        prev_cy = inferred_cy[t - 1]

                    # 3. 阻尼跟随逻辑 (Damping)
                    if prev_cx is not None and not np.isnan(prev_cx):
                        # 如果有上一帧记录，应用阻尼公式
                        # Current = (1 - alpha) * Prev + alpha * Target
                        alpha = guidance_damping
                        pred_cx = (1 - alpha) * prev_cx + alpha * target_cx
                        pred_cy = (1 - alpha) * prev_cy + alpha * target_cy
                    else:
                        # 第一帧交互，或者上一帧也是NaN，直接跳过去
                        pred_cx, pred_cy = target_cx, target_cy

                    # 4. 赋值
                    inferred_cx[t] = pred_cx
                    inferred_cy[t] = pred_cy

    # 填补空缺
    df['cx'] = inferred_cx.interpolate(method='spline', order=2).ffill().bfill()
    df['cy'] = inferred_cy.interpolate(method='spline', order=2).ffill().bfill()

    # --- Step 3: Savgol Filter 后处理 (保持) ---
    win = min(savgol_window, num_frames)
    if win % 2 == 0:
        win -= 1
    if win > 3:
        df['cx'] = savgol_filter(df['cx'], win, savgol_poly)
        df['cy'] = savgol_filter(df['cy'], win, savgol_poly)

    # --- Step 4: 生成最终 BBox ---
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
