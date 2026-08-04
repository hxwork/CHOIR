import json
import os
from pathlib import Path

import cv2
import numpy as np

# 手部关键点连接关系 (21个关键点的骨架结构)
HAND_CONNECTIONS = [
    # 手腕到各个手指的连接
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),  # 拇指
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),  # 食指
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),  # 中指
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),  # 无名指
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),  # 小指
]


def get_bbox_from_mask(mask):
    """从mask中计算外包围框"""
    if mask.sum() == 0:
        return None
    coords = np.argwhere(mask > 0)
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    return [int(x_min), int(y_min), int(x_max), int(y_max)]


def get_bbox_from_keypoints(keypoints):
    """从关键点计算外包围框"""
    keypoints = np.array(keypoints)
    x_coords = keypoints[:, 0]
    y_coords = keypoints[:, 1]
    x_min, y_min = x_coords.min(), y_coords.min()
    x_max, y_max = x_coords.max(), y_coords.max()
    return [int(x_min), int(y_min), int(x_max), int(y_max)]


def expand_bbox(bbox, scale=1.2, img_shape=None):
    """
    扩展bbox
    Args:
        bbox: [x_min, y_min, x_max, y_max]
        scale: 扩展比例，默认1.2倍
        img_shape: 图像尺寸(height, width)，用于边界检查
    """
    x_min, y_min, x_max, y_max = bbox

    # 计算中心点和当前尺寸
    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    w = x_max - x_min
    h = y_max - y_min

    # 扩展尺寸
    new_w = w * scale
    new_h = h * scale

    # 计算新的bbox
    new_x_min = int(cx - new_w / 2)
    new_y_min = int(cy - new_h / 2)
    new_x_max = int(cx + new_w / 2)
    new_y_max = int(cy + new_h / 2)

    # 边界检查
    if img_shape is not None:
        img_h, img_w = img_shape[:2]
        new_x_min = max(0, new_x_min)
        new_y_min = max(0, new_y_min)
        new_x_max = min(img_w, new_x_max)
        new_y_max = min(img_h, new_y_max)

    return [new_x_min, new_y_min, new_x_max, new_y_max]


def draw_bbox(image, bbox, color, thickness=2, label=None):
    """绘制边界框"""
    x_min, y_min, x_max, y_max = bbox
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness)
    if label:
        cv2.putText(image, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def draw_skeleton(image, keypoints, connections, color=(0, 255, 0), thickness=2):
    """绘制手部骨架"""
    keypoints = np.array(keypoints)

    # 绘制连接线
    for conn in connections:
        pt1 = tuple(keypoints[conn[0]].astype(int))
        pt2 = tuple(keypoints[conn[1]].astype(int))
        cv2.line(image, pt1, pt2, color, thickness)

    # 绘制关键点
    for kp in keypoints:
        cv2.circle(image, tuple(kp.astype(int)), 3, color, -1)


def visualize_bbox_and_skeleton(base_path, output_dir):
    """
    在原始尺寸的RGB图上显示物体和手的bbox以及手的skeleton
    """
    # 读取数据
    rgb_dir = os.path.join(base_path, "rgbs")
    mask_dir = os.path.join(base_path, "obj_masks")
    keypoints_file = os.path.join(base_path, "rh_keypoints.json")

    with open(keypoints_file, 'r') as f:
        keypoints_data = json.load(f)

    # 创建输出目录
    bbox_output_dir = os.path.join(output_dir, "bbox_skeleton")
    os.makedirs(bbox_output_dir, exist_ok=True)

    # 处理每一帧
    for frame_id in sorted(keypoints_data.keys(), key=int):
        # 读取RGB图像
        rgb_path = os.path.join(rgb_dir, f"{frame_id}.png")
        if not os.path.exists(rgb_path):
            continue

        image = cv2.imread(rgb_path)
        keypoints = keypoints_data[frame_id]

        # 读取modal mask计算物体bbox
        mask_path = os.path.join(mask_dir, f"{frame_id}.png")
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            obj_bbox = get_bbox_from_mask(mask)
            if obj_bbox:
                # 扩展物体bbox 1.2倍
                obj_bbox = expand_bbox(obj_bbox, scale=1.2, img_shape=image.shape)
                draw_bbox(image, obj_bbox, (0, 255, 255), thickness=3)  # SAM荧光青色

        # 从关键点计算手的bbox
        hand_bbox = get_bbox_from_keypoints(keypoints)
        # 扩展手部bbox 1.2倍
        hand_bbox = expand_bbox(hand_bbox, scale=1.2, img_shape=image.shape)
        draw_bbox(image, hand_bbox, (100, 150, 255), thickness=3)  # 浅红色

        # 绘制手部骨架
        draw_skeleton(image, keypoints, HAND_CONNECTIONS, color=(100, 150, 255), thickness=2)  # 浅红色(与手部bbox同色)

        # 保存结果
        output_path = os.path.join(bbox_output_dir, f"{frame_id}.png")
        cv2.imwrite(output_path, image)

    print(f"边界框和骨架可视化完成，保存在: {bbox_output_dir}")


def visualize_masks_on_cropped(base_path, global_bbox_file, output_dir):
    """
    在裁切后的图上显示物体的modal mask和amodal mask
    外边缘描边，中间填充颜色
    """
    # 读取数据
    rgb_dir = os.path.join(base_path, "rgbs")
    modal_mask_dir = os.path.join(base_path, "obj_masks")
    amodal_mask_dir = os.path.join(base_path, "amodal_masks")

    with open(global_bbox_file, 'r') as f:
        global_bboxes = json.load(f)

    # 创建输出目录
    modal_output_dir = os.path.join(output_dir, "cropped_modal_mask")
    amodal_output_dir = os.path.join(output_dir, "cropped_amodal_mask")
    os.makedirs(modal_output_dir, exist_ok=True)
    os.makedirs(amodal_output_dir, exist_ok=True)

    # 获取所有帧
    rgb_files = sorted([f for f in os.listdir(rgb_dir) if f.endswith('.png')], key=lambda x: int(x.split('.')[0]))

    for idx, rgb_file in enumerate(rgb_files):
        frame_id = rgb_file.split('.')[0]

        # 读取RGB图像
        rgb_path = os.path.join(rgb_dir, rgb_file)
        image = cv2.imread(rgb_path)

        # 获取全局bbox
        if idx >= len(global_bboxes):
            continue
        x_min, y_min, x_max, y_max = global_bboxes[idx]

        # 裁剪RGB图像
        cropped_rgb = image[y_min:y_max, x_min:x_max].copy()

        # 处理modal mask
        modal_mask_path = os.path.join(modal_mask_dir, rgb_file)
        if os.path.exists(modal_mask_path):
            modal_mask = cv2.imread(modal_mask_path, cv2.IMREAD_GRAYSCALE)
            # 裁剪modal mask
            cropped_modal_mask = modal_mask[y_min:y_max, x_min:x_max]

            # 创建可视化图像
            modal_vis = cropped_rgb.copy()

            # 填充颜色 (浅紫色)
            color_overlay = modal_vis.copy()
            color_overlay[cropped_modal_mask > 0] = [220, 150, 255]  # 浅紫色 BGR
            modal_vis = cv2.addWeighted(modal_vis, 0.5, color_overlay, 0.5, 0)

            # 绘制边缘 (深紫色)
            contours, _ = cv2.findContours(cropped_modal_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(modal_vis, contours, -1, (130, 0, 180), 3)  # 深紫色

            # 保存
            output_path = os.path.join(modal_output_dir, rgb_file)
            cv2.imwrite(output_path, modal_vis)

        # 处理amodal mask (已经是裁剪后的尺寸，不需要再次裁剪)
        amodal_mask_path = os.path.join(amodal_mask_dir, rgb_file)
        if os.path.exists(amodal_mask_path):
            amodal_mask = cv2.imread(amodal_mask_path, cv2.IMREAD_GRAYSCALE)

            # 确保amodal_mask和cropped_rgb尺寸匹配
            if amodal_mask.shape[:2] != cropped_rgb.shape[:2]:
                # 如果尺寸不匹配，调整amodal_mask的尺寸
                amodal_mask = cv2.resize(amodal_mask, (cropped_rgb.shape[1], cropped_rgb.shape[0]))

            # 创建可视化图像
            amodal_vis = cropped_rgb.copy()

            # 填充颜色 (半透明绿色)
            color_overlay = amodal_vis.copy()
            color_overlay[amodal_mask > 0] = [100, 255, 100]  # 浅绿色 BGR
            amodal_vis = cv2.addWeighted(amodal_vis, 0.5, color_overlay, 0.5, 0)

            # 绘制边缘 (深绿色)
            contours, _ = cv2.findContours(amodal_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(amodal_vis, contours, -1, (0, 200, 0), 3)  # 深绿色

            # 保存
            output_path = os.path.join(amodal_output_dir, rgb_file)
            cv2.imwrite(output_path, amodal_vis)

    print(f"Modal mask可视化完成，保存在: {modal_output_dir}")
    print(f"Amodal mask可视化完成，保存在: {amodal_output_dir}")


def main():
    # 数据路径
    base_path = "../../output/54362"
    global_bbox_file = os.path.join(base_path, "global_bbox.json")
    output_dir = "visualization_output_v2"

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 1. 可视化边界框和骨架
    print("开始生成边界框和骨架可视化...")
    visualize_bbox_and_skeleton(base_path, output_dir)

    # 2. 可视化裁切后的masks
    print("\n开始生成裁切后的mask可视化...")
    visualize_masks_on_cropped(base_path, global_bbox_file, output_dir)

    print("\n所有可视化完成!")


if __name__ == "__main__":
    main()
