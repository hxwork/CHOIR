#!/usr/bin/env python3
"""
处理 hold 开头的文件夹数据，将数据转换为指定格式：
1. 将 j2d.full.npy 中的关键点转换为 lh_keypoints.json 和 rh_keypoints.json
2. 将 rgba masks 复制到 obj_masks 文件夹，重命名为整数帧号
3. 将 rgb frames 复制到 rgbs 文件夹，重命名为整数帧号
"""

import json
import os
import shutil
from pathlib import Path

import cv2
import imageio
import numpy as np
from PIL import Image


def process_hold_folder(hold_folder_path):
    """
    处理单个 hold 文件夹
    
    Args:
        hold_folder_path: hold 文件夹的路径
    """
    hold_folder = Path(hold_folder_path)
    processed_folder = hold_folder / "processed"

    # 检查必要的文件是否存在
    j2d_path = processed_folder / "j2d.full.npy"
    rgbas_folder = processed_folder / "rgbas"
    images_folder = processed_folder / "images"
    seg_masks_folder = processed_folder / "masks"

    if not j2d_path.exists():
        print(f"Warning: {j2d_path} does not exist, skipping {hold_folder.name}")
        return

    if not rgbas_folder.exists():
        print(f"Warning: {rgbas_folder} does not exist, skipping {hold_folder.name}")
        return

    if not images_folder.exists():
        print(f"Warning: {images_folder} does not exist, skipping {hold_folder.name}")
        return

    print(f"Processing {hold_folder.name}...")

    # 1. 处理关键点数据
    print("  Processing keypoints...")
    data = np.load(str(j2d_path), allow_pickle=True)
    if isinstance(data, np.ndarray) and data.ndim == 0:
        data = data.item()

    j2d_right = data.get('j2d.right', None)
    j2d_left = data.get('j2d.left', None)
    im_paths = data.get('im_paths', [])

    # 从图片路径中提取帧号
    frame_indices = []
    for im_path in im_paths:
        # 路径格式类似: '../data/hold_GPMF12_ho3d/processed/images/0090.png'
        filename = os.path.basename(im_path)
        # 提取数字部分，去掉 .png 和前面的零
        frame_idx = int(filename.replace('.png', ''))
        frame_indices.append(frame_idx)

    # 如果没有 im_paths，从 images 文件夹中读取
    if not frame_indices:
        image_files = sorted(images_folder.glob("*.png"))
        for img_file in image_files:
            frame_idx = int(img_file.stem)
            frame_indices.append(frame_idx)

    # 对帧号进行排序，并创建从0开始的映射
    sorted_original_indices = sorted(frame_indices)
    # 创建原始帧号到新帧号(从0开始)的映射
    frame_mapping = {orig_idx: new_idx for new_idx, orig_idx in enumerate(sorted_original_indices)}

    # 创建输出字典（使用从0开始的新帧号）
    rh_keypoints = {}
    lh_keypoints = {}

    num_frames = len(frame_indices)
    # j2d数据的索引对应frame_indices的顺序，需要找到每个原始帧号在frame_indices中的位置
    if j2d_right is not None and len(j2d_right) > 0:
        for orig_frame_idx in sorted_original_indices:
            # 找到该原始帧号在frame_indices中的索引位置
            if orig_frame_idx in frame_indices:
                j2d_idx = frame_indices.index(orig_frame_idx)
                if j2d_idx < len(j2d_right):
                    # 将 (21, 2) 的数组转换为 [[x, y], [x, y], ...] 格式
                    joints = j2d_right[j2d_idx]  # (21, 2)
                    # 过滤掉 NaN 值，如果所有都是 NaN 则跳过
                    if not np.isnan(joints).all():
                        joints_list = [[float(joint[0]), float(joint[1])] for joint in joints]
                        # 使用新帧号(从0开始)作为key
                        new_frame_idx = frame_mapping[orig_frame_idx]
                        rh_keypoints[str(new_frame_idx)] = joints_list

    if j2d_left is not None and len(j2d_left) > 0:
        for orig_frame_idx in sorted_original_indices:
            # 找到该原始帧号在frame_indices中的索引位置
            if orig_frame_idx in frame_indices:
                j2d_idx = frame_indices.index(orig_frame_idx)
                if j2d_idx < len(j2d_left):
                    joints = j2d_left[j2d_idx]  # (21, 2)
                    # 过滤掉 NaN 值，如果所有都是 NaN 则跳过
                    if not np.isnan(joints).all():
                        joints_list = [[float(joint[0]), float(joint[1])] for joint in joints]
                        # 使用新帧号(从0开始)作为key
                        new_frame_idx = frame_mapping[orig_frame_idx]
                        lh_keypoints[str(new_frame_idx)] = joints_list

    # 保存关键点 JSON 文件
    lh_keypoints_path = hold_folder / "lh_keypoints.json"
    rh_keypoints_path = hold_folder / "rh_keypoints.json"

    if len(lh_keypoints) > 0:
        with open(lh_keypoints_path, 'w') as f:
            json.dump(lh_keypoints, f, indent=4)
        print(f"  Saved {lh_keypoints_path}")

    if len(rh_keypoints) > 0:
        with open(rh_keypoints_path, 'w') as f:
            json.dump(rh_keypoints, f, indent=4)
        print(f"  Saved {rh_keypoints_path}")

    # 2. 处理 rgba masks
    print("  Processing rgba masks...")
    obj_masks_folder = hold_folder / "obj_masks"
    obj_masks_folder.mkdir(exist_ok=True)

    rgba_files = sorted(rgbas_folder.glob("*.png"))
    for rgba_file in rgba_files:
        # 提取原始帧号
        orig_frame_idx = int(rgba_file.stem)
        # 如果该帧在映射中，使用新帧号(从0开始)
        if orig_frame_idx in frame_mapping:
            new_frame_idx = frame_mapping[orig_frame_idx]
            # 目标文件名使用新帧号
            target_name = f"{new_frame_idx}.png"
            target_path = obj_masks_folder / target_name
            # 复制文件
            shutil.copy2(rgba_file, target_path)

    print(f"  Copied {len(rgba_files)} rgba masks to {obj_masks_folder}")

    # 3. 处理 rgb frames
    print("  Processing rgb frames...")
    rgbs_folder = hold_folder / "rgbs"
    rgbs_folder.mkdir(exist_ok=True)

    image_files = sorted(images_folder.glob("*.png"))
    for img_file in image_files:
        # 提取原始帧号
        orig_frame_idx = int(img_file.stem)
        # 如果该帧在映射中，使用新帧号(从0开始)
        if orig_frame_idx in frame_mapping:
            new_frame_idx = frame_mapping[orig_frame_idx]
            # 目标文件名使用新帧号
            target_name = f"{new_frame_idx}.png"
            target_path = rgbs_folder / target_name
            # 复制文件
            shutil.copy2(img_file, target_path)

    print(f"  Copied {len(image_files)} rgb frames to {rgbs_folder}")

    # 3.5. 从 masks 提取手的 RGBA 并保存到 rh_masks
    print("  Creating hand RGBA from masks...")
    rh_masks_folder = hold_folder / "rh_masks"
    rh_masks_folder.mkdir(exist_ok=True)

    hand_rgba_count = 0
    mask_files = sorted(seg_masks_folder.glob("*.png"))
    for mask_file in mask_files:
        # 提取帧号
        orig_frame_idx = int(mask_file.stem)
        # 读取对应的 RGB 图像
        rgb_path = images_folder / f"{orig_frame_idx:04d}.png"

        # 读取 mask（HW3 格式）
        mask = cv2.imread(str(mask_file))  # BGR格式

        # 读取 RGB 图像
        rgb = cv2.imread(str(rgb_path))  # BGR格式
        # 提取手的 mask：值为 150 的位置
        # mask 是 HW3，检查每个像素的所有通道是否都为 150
        hand_mask = np.all(mask == 150, axis=2).astype(np.uint8) * 255

        # 创建 RGBA 图像：RGB 来自原图，A 来自手的 mask
        rgba = np.dstack([rgb, hand_mask])

        # 保存到 rh_masks 文件夹
        new_frame_idx = frame_mapping[orig_frame_idx]
        # 目标文件名使用新帧号
        target_name = f"{new_frame_idx}.png"
        target_path = rh_masks_folder / target_name
        cv2.imwrite(str(target_path), rgba)
        hand_rgba_count += 1

    print(f"  Created {hand_rgba_count} hand RGBA images in {rh_masks_folder}")

    # 3.6. 将所有 RGB 帧合成为 MP4 视频
    if image_files:
        print("  Creating MP4 video from RGB frames...")
        # 视频文件名与文件夹名称一致
        video_filename = f"{hold_folder.name}.mp4"
        video_path = hold_folder / video_filename

        # 按新帧号(从0开始)顺序读取所有帧
        frames = []
        for new_frame_idx in range(len(sorted_original_indices)):
            frame_path = rgbs_folder / f"{new_frame_idx}.png"
            if frame_path.exists():
                # 使用 imageio 读取图片
                frame = imageio.imread(str(frame_path))
                frames.append(frame)

        if frames:
            # 使用 imageio 创建视频，帧率 30
            fps = 30
            imageio.mimwrite(str(video_path), frames, fps=fps, codec='libx264', quality=8)
            print(f"  Created video: {video_path} ({len(frames)} frames, {fps} fps)")
        else:
            print(f"  Warning: No frames found to create video")

    # 4. 从 inpaint 文件夹复制手动挑选的帧
    inpaint_folder = processed_folder / "inpaint"
    print(f"  Looking for manually picked frame in {inpaint_folder}...")

    # 查找所有 *_rgba.png 文件
    rgba_files = list(inpaint_folder.glob("*_rgba.png"))

    # 假设只有一个手动挑选的帧，取第一个
    picked_rgba_file = rgba_files[0]
    # 从文件名中提取原始帧号，如 "0206_rgba.png" -> 206
    picked_orig_frame_idx = int(picked_rgba_file.stem.split('_')[0])

    print(f"  Found manually picked frame: {picked_rgba_file.name} (original frame index: {picked_orig_frame_idx})")

    # 检查该原始帧号是否在映射中
    picked_new_frame_idx = frame_mapping[picked_orig_frame_idx]
    print(f"  Mapped to new frame index: {picked_new_frame_idx}")

    # 复制对应的 RGB 到 image.png
    rgb_source = rgbs_folder / f"{picked_new_frame_idx}.png"
    image_png_path = hold_folder / "image.png"
    shutil.copy2(rgb_source, image_png_path)
    print(f"  Copied RGB frame to {image_png_path}")

    # 复制对应的 mask 到 0.png
    mask_0_path = hold_folder / "0.png"
    shutil.copy2(picked_rgba_file, mask_0_path)
    print(f"  Copied mask to {mask_0_path}")

    # 保存映射关系到 JSON
    mapping_data = {"new_frame_idx": picked_new_frame_idx, "original_frame_idx": picked_orig_frame_idx}
    mapping_json_path = hold_folder / "manually_picked_frame_idx.json"
    with open(mapping_json_path, 'w') as f:
        json.dump(mapping_data, f, indent=4)
    print(f"  Saved frame mapping to {mapping_json_path}")

    print(f"Completed processing {hold_folder.name}\n")


def main():
    """主函数：查找所有以 hold 开头的文件夹并处理"""
    base_dir = Path("../../output")

    if not base_dir.exists():
        print(f"Error: Base directory {base_dir} does not exist!")
        return

    # 查找所有以 hold 开头的文件夹
    hold_folders = [d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith("hold")]

    if not hold_folders:
        print(f"No folders starting with 'hold' found in {base_dir}")
        return

    print(f"Found {len(hold_folders)} folders starting with 'hold'")
    print("=" * 60)

    # 处理每个文件夹
    for hold_folder in sorted(hold_folders):
        process_hold_folder(hold_folder)

    print("=" * 60)
    print("All processing completed!")


if __name__ == "__main__":
    main()
