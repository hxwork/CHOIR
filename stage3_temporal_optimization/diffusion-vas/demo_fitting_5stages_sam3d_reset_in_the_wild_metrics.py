"""Variant of ``demo_fitting_5stages_new.py`` with extra diagnostics (e.g. PnP
health dry-run, interaction motion profile). Segmented SAM3D reset + bridge
that used to live here was removed; see ``backup/sam3d_reset_bridge_archive.txt``.
"""

import argparse
import glob
import json
import math
import os
import pickle
import shutil
import subprocess
import sys
import traceback
import warnings
from typing import Any, Dict, List, Optional

import cv2
import imageio
import k3d
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import trimesh
from manotorch.anatomy_loss import AnatomyConstraintLossEE
from manotorch.axislayer import AxisLayerFK
from manotorch.manolayer import ManoLayer as AMANOLayer
from PIL import Image
from pytorch3d.implicitron.tools.point_cloud_utils import get_rgbd_point_cloud
from pytorch3d.io import IO, load_ply, save_obj, save_ply
from pytorch3d.io.experimental_gltf_io import MeshGlbFormat
from pytorch3d.ops import iterative_closest_point, knn_points
from pytorch3d.renderer import (BlendParams, Materials, MeshRasterizer, MeshRenderer, PerspectiveCameras, PointLights, RasterizationSettings,
                                SoftSilhouetteShader, TexturesVertex)
from pytorch3d.renderer.blending import Device
from pytorch3d.renderer.mesh.shader import HardPhongShader
from pytorch3d.structures import (Meshes, join_meshes_as_batch, join_meshes_as_scene)
from pytorch3d.transforms import (
    Transform3d,
    axis_angle_to_matrix,
    matrix_to_axis_angle,
    matrix_to_quaternion,
    matrix_to_rotation_6d,
    quaternion_to_matrix,
    rotation_6d_to_matrix,
)
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from torchvision import transforms
from tqdm import tqdm

import torch_mesh_intersection.mesh_intersection.loss as collisions_loss
from body_model import MANO, run_amano, run_mano
from contact_constraint import (BUCKET_INSIDE, BUCKET_NEAR_CONTACT,
                                classify_and_build_correspondence,
                                compute_confident_gap_closing_loss,
                                compute_contact_patch_centroid_loss,
                                compute_grasp_template_anchor_loss,
                                compute_interaction_alignment_offsets,
                                compute_interaction_boundary_blend_offsets,
                                compute_neighbor_template_bridge_loss,
                                compute_penetration_loss,
                                compute_soft_contact_loss,
                                expand_reliable_contact_frontier,
                                propagate_dense_contact_memory,
                                propagate_grasp_template_from_reliable_frames,
                                temporal_lock_argmax)
from contact_map_estimator import estimate_contact_map_for_sampled_frames, load_contact_indices
from debug_bbox import get_global_amodal_bbox, load_hand_data, load_raw_frames
from frame_sampling import build_adaptive_sample_indices
from gfm_offset_application import apply_camera_ray_depth_offsets
from gfm_preprocess import export_graspflowmatching_sequence
from hand_input_selection import resolve_hand_inputs
from hand2d_anatomy_gate import compute_anatomy_hand2d_frame_weights
from models.diffusion_vas.pipeline_diffusion_vas import DiffusionVASPipeline
from motion_policy import FORCE_ROTATION_MODE, force_rotation_likely_profile
from hoi_io import save_hoi_sequence, save_k3d_visualization
from pnp import compute_pnp_health, flag_reset_candidates, generate_queries, run_pnp, run_pnp_1stage
from rendering import (get_render_params, render_side_view_rgb, vectorized_pose_guiding_loss, weighted_false_negative_loss, weighted_false_positive_loss)
from stage3_hold_policy import (
    hold_stage3_hand_refine_steps,
    hold_joint_target_indices,
    hold_mano_init_param_groups,
    hold_mano_init_path,
    hold_pose_from_fit,
    hold_pose_hand_mean,
    hold_reorder_valid_mask,
    hold_sam3d_rot_outlier_max_angle,
    is_hold_video_id,
    stage3_joint2d_relative_weight,
    stage3_object_smoothness_scale,
)
from stage3_rotation_protection import (
    rotation_drift_degrees,
    stage3_rotation_lr_for_step,
    stage3_rotation_smoothness_scale,
)
from stage5_mode import (
    STAGE5_FROZEN,
    STAGE5_FULL,
    STAGE5_MODES,
    STAGE5_OBJECT_LITE,
    STAGE5_POSE_ONLY,
    STAGE5_POSE_RAY,
    stage5_hand2d_grad_groups,
    stage5_hand2d_weight,
    stage5_trainable_params,
)
from stage5_object_lite import (
    DEFAULT_OBJECT_LITE_MAX_TRANS_DELTA,
    project_object_translation_delta_,
)
from stage5_ray_delta import (
    DEFAULT_STAGE5_RAY_MAX_DELTA,
    DEFAULT_STAGE5_RAY_MIN_DELTA,
    build_stage5_ray_delta,
    clamp_stage5_ray_delta_,
)
from stage_layout_detection import (
    LAYOUT_FIVE_STAGE,
    LAYOUT_INTERACTION_ONLY,
    apply_hold_interaction_only_override,
    classify_stage_layout_from_interaction_bounds,
)
from torch_mesh_intersection.mesh_intersection.bvh_search_tree import BVH
from utils import *
from sam3d_reset import calculate_iou
from sam3d_ref_masks import load_sam3d_ref_obj_mask
from sam3d_sparse_keyframes import (
    render_mesh_phong_overlay_world_vertices,
    render_mesh_soft_overlay_on_rgb,
    run_sam3d_rotation_dense_from_mask_onset,
    run_sam3d_sparse_keyframes_motion_window,
    sparse_stride_for_motion_profile,
    world_vertices_from_pnp_row,
)


def _draw_video_label(frame_uint8: np.ndarray, label: str, color=(255, 255, 255)) -> np.ndarray:
    """Draw a readable frame/stage label on a uint8 RGB video frame."""
    out = np.ascontiguousarray(frame_uint8)
    org = (10, 30)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(out, label, org, font, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(out, label, org, font, 0.8, color, 2, cv2.LINE_AA)
    return out


def find_static_end(masks_np, iou_threshold=0.9):
    """
    找到静止段的结束位置（第一个开始运动的帧）
    
    Args:
        masks_np: (N, H, W) numpy array of binary masks
        iou_threshold: IoU阈值，超过此值认为物体静止
    
    Returns:
        static_end: 静止段结束的帧索引（不包含），即第一个运动帧的索引
    """
    num_frames = len(masks_np)
    if num_frames <= 1:
        return 1

    # 计算连续帧之间的IoU
    ious = []
    for i in range(num_frames - 1):
        iou = calculate_iou(masks_np[i], masks_np[i + 1])
        ious.append(iou)
    ious = np.array(ious)

    is_static = (ious > iou_threshold)

    # 找到第一个非静止的帧
    static_end = 1  # 默认至少包含第一帧
    for i in range(len(is_static)):
        if not is_static[i]:
            static_end = i + 1  # +1因为is_static[i]比较的是帧i和i+1
            break

    # 如果全部都是静止的，返回最后一帧索引
    if static_end == 1 and len(is_static) > 0 and is_static.all():
        static_end = num_frames

    return static_end


def detect_static_segments(masks_np, iou_threshold=0.9):
    """
    检测视频序列中的静止段（0-1段和4-5段）
    默认第一帧和最后一帧是静止的
    
    Args:
        masks_np: (N, H, W) numpy array of binary masks
        iou_threshold: IoU阈值，超过此值认为物体静止
    
    Returns:
        start_static_end: 0-1段的结束帧索引（不包含）
        end_static_start: 4-5段的开始帧索引（包含）
    """
    num_frames = len(masks_np)

    # 检测开始静止段（0-1）：正序检测
    start_static_end = find_static_end(masks_np, iou_threshold)

    # 检测结束静止段（4-5）：逆序检测
    masks_reversed = masks_np[::-1]
    end_static_length = find_static_end(masks_reversed, iou_threshold)
    end_static_start = num_frames - end_static_length

    # 确保段不重叠
    if start_static_end >= end_static_start:
        # 如果检测失败，使用默认值：前1/4和后1/4
        quarter = max(1, num_frames // 4)
        start_static_end = quarter
        end_static_start = num_frames - quarter

    print(f"Detected static segments: [0:{start_static_end}] (start static), [{end_static_start}:{num_frames}] (end static)")

    return start_static_end, end_static_start


warnings.filterwarnings("ignore")


def init_amodal_segmentation_model(model_path_mask):
    device = f"cuda:{torch.cuda.current_device()}"
    pipeline_mask = DiffusionVASPipeline.from_pretrained(model_path_mask, torch_dtype=torch.float16).to(device)
    # pipeline_mask.enable_model_cpu_offload()
    pipeline_mask.set_progress_bar_config(disable=True)

    return pipeline_mask


def build_temporal_windows(num_frames, window_size=128, overlap=16):
    """Return inclusive-exclusive frame windows for memory-bounded video inference."""
    num_frames = int(num_frames)
    window_size = int(window_size)
    overlap = int(overlap)

    if num_frames <= 0:
        return []
    if window_size <= 0:
        raise ValueError(f"amodal_window_size must be positive, got {window_size}")
    if overlap < 0:
        raise ValueError(f"amodal_window_overlap must be non-negative, got {overlap}")
    if overlap >= window_size:
        raise ValueError(
            f"amodal_window_overlap ({overlap}) must be smaller than "
            f"amodal_window_size ({window_size})"
        )
    if num_frames <= window_size:
        return [(0, num_frames)]

    stride = window_size - overlap
    windows = []
    start = 0
    while start < num_frames:
        end = min(start + window_size, num_frames)
        windows.append((start, end))
        if end == num_frames:
            break
        start += stride
    return windows


def init_depth_model(model_path_depth, depth_encoder):
    device = f"cuda:{torch.cuda.current_device()}"

    from models.Depth_Anything_V2.depth_anything_v2.dpt import DepthAnythingV2

    depth_model_configs = {
        'vits': {
            'encoder': 'vits',
            'features': 64,
            'out_channels': [48, 96, 192, 384]
        },
        'vitb': {
            'encoder': 'vitb',
            'features': 128,
            'out_channels': [96, 192, 384, 768]
        },
        'vitl': {
            'encoder': 'vitl',
            'features': 256,
            'out_channels': [256, 512, 1024, 1024]
        },
        'vitg': {
            'encoder': 'vitg',
            'features': 384,
            'out_channels': [1536, 1536, 1536, 1536]
        }
    }

    depth_model = DepthAnythingV2(**depth_model_configs[depth_encoder]).to(device)
    depth_model.load_state_dict(torch.load(model_path_depth, map_location=device))
    depth_model.eval()

    return depth_model


def get_raw_depth_maps(raw_rgbs, depth_model):
    """
    Computes depth maps from raw RGB images.
    Returns a list of single-channel float numpy arrays, normalized to [0, 1].
    """
    depth_maps = []
    for rgb_image_np in tqdm(raw_rgbs, desc="Estimating Depth"):
        # depth_model expects a (H, W, 3) uint8 numpy array
        depth_map = depth_model.infer_image(rgb_image_np)  # returns a (H, W) float numpy array
        depth_maps.append(depth_map)

    # Normalize across the entire video sequence
    depth_maps_np = np.array(depth_maps)
    min_val, max_val = depth_maps_np.min(), depth_maps_np.max()
    depth_maps_np = (depth_maps_np - min_val) / (max_val - min_val)

    return depth_maps_np


def init_metric_depth_model():
    device = f"cuda:{torch.cuda.current_device()}"

    from moge.model.v2 import MoGeModel

    metric_depth_model = MoGeModel.from_pretrained("Ruicheng/moge-v2-vitl-normal/model.pt").to(device)

    return metric_depth_model


def get_raw_metric_depth_maps(raw_rgbs, metric_depth_model):
    """
    Computes metric depth maps from raw RGB images.
    Returns a list of single-channel float numpy arrays, normalized to [0, 1].
    """
    device = f"cuda:{torch.cuda.current_device()}"
    metric_depth_maps = []
    for rgb_image_np in tqdm(raw_rgbs, desc="Estimating Metric Depth"):
        # depth_model expects a (H, W, 3) uint8 numpy array

        # Read the input image and convert to tensor (3, H, W) with RGB values normalized to [0, 1]
        input_image = torch.tensor(rgb_image_np / 255, dtype=torch.float32, device=device).permute(2, 0, 1)

        metric_depth_map = metric_depth_model.infer(input_image)  # returns a (H, W) float numpy array
        metric_depth_map = metric_depth_map['depth'].cpu().numpy()  # (H, W) float numpy array
        metric_depth_maps.append(metric_depth_map)

    metric_depth_maps_np = np.array(metric_depth_maps)

    return metric_depth_maps_np


def crop_and_resize_frames(frames, bboxes, output_size, frame_type='rgb'):
    to_tensor = transforms.ToTensor()
    normalizer = transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)

    processed_frames = []
    for frame, bbox in zip(frames, bboxes):
        x1, y1, x2, y2 = bbox

        crop_w = x2 - x1
        crop_h = y2 - y1
        img_h, img_w = frame.shape[:2]

        src_x1, src_y1 = max(0, x1), max(0, y1)
        src_x2, src_y2 = min(img_w, x2), min(img_h, y2)

        dst_x1, dst_y1 = src_x1 - x1, src_y1 - y1
        dst_x2, dst_y2 = src_x2 - x1, src_y2 - y1

        if frame.ndim == 3:
            canvas = np.zeros((crop_h, crop_w, frame.shape[2]), dtype=frame.dtype)
        else:
            canvas = np.zeros((crop_h, crop_w), dtype=frame.dtype)

        if (src_x2 > src_x1) and (src_y2 > src_y1):
            canvas[dst_y1:dst_y2, dst_x1:dst_x2] = frame[src_y1:src_y2, src_x1:src_x2]

        # Pre-resize processing
        if frame_type == 'mask':
            # Binarized 0/1 mask to 0/255 for better resizing interpolation
            canvas = (canvas * 255).astype(np.uint8)

        # --- New resize logic to preserve aspect ratio ---
        h_canvas, w_canvas = canvas.shape[:2]
        h_out, w_out = output_size

        # Calculate scale to fit canvas into output_size while preserving aspect ratio
        scale = min(w_out / w_canvas, h_out / h_canvas)
        new_w, new_h = int(w_canvas * scale), int(h_canvas * scale)

        # Resize with aspect ratio preserved
        interpolation = cv2.INTER_LINEAR if frame_type != 'mask' else cv2.INTER_NEAREST
        resized_canvas = cv2.resize(canvas, (new_w, new_h), interpolation=interpolation)

        # Create a new canvas of the final output size and paste the resized image in the center
        if resized_canvas.ndim == 3:
            final_image = np.zeros((h_out, w_out, resized_canvas.shape[2]), dtype=canvas.dtype)
        else:
            final_image = np.zeros((h_out, w_out), dtype=canvas.dtype)

        pad_top = (h_out - new_h) // 2
        pad_left = (w_out - new_w) // 2

        final_image[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized_canvas

        # Post-resize processing & tensor conversion
        if frame_type in ['rgb']:
            pil_image = Image.fromarray(final_image.astype(np.uint8))
            tensor_frame = to_tensor(pil_image)
            tensor_frame = normalizer(tensor_frame)

        elif frame_type in ['mask']:
            pil_image = Image.fromarray(final_image.astype(np.uint8))
            tensor_frame = to_tensor(pil_image)
            tensor_frame = tensor_frame.repeat(3, 1, 1)
            tensor_frame = normalizer(tensor_frame)

        elif frame_type in ['depth']:
            # Add channel dimension for to_tensor. Input is float [0,1]
            tensor_frame = to_tensor(final_image[:, :, np.newaxis])
            tensor_frame = tensor_frame.repeat(3, 1, 1)
            tensor_frame = normalizer(tensor_frame)

        elif frame_type in ['metric_depth']:
            tensor_frame = torch.from_numpy(final_image[:, :, np.newaxis]).float().permute(2, 0, 1)
            tensor_frame = tensor_frame.repeat(3, 1, 1)

        processed_frames.append(tensor_frame)

    return torch.stack(processed_frames).unsqueeze(0)


def compute_smoothness_loss(seq):
    diff = torch.diff(seq.contiguous(), dim=0)

    return diff.pow(2).sum()


def _bbox_centers_np(vertices):
    vertices = np.asarray(vertices)
    return (vertices.min(axis=1) + vertices.max(axis=1)) / 2.0


def _default_eval_fnames(seq_path, num_frames):
    rgb_dir = os.path.join(seq_path, "rgbs")
    if os.path.isdir(rgb_dir):
        names = sorted(os.path.basename(p) for p in glob.glob(os.path.join(rgb_dir, "*")))
        if len(names) == num_frames:
            return np.asarray(names)
    return np.asarray([f"{i:05d}.png" for i in range(num_frames)])


def _openpose_to_mano_joint_order():
    mano_to_openpose = hold_joint_target_indices()
    return [mano_to_openpose.index(i) for i in range(len(mano_to_openpose))]


def save_ablation_eval_data(output_dir, seq_path, seq_name, obj_verts_seq, obj_faces, hand_verts_seq, hand_faces, hand_joints_seq):
    """Save the prediction tensor bundle consumed by eval_ours.py."""
    os.makedirs(output_dir, exist_ok=True)
    eval_data_path = os.path.join(output_dir, "eval_data.npy")
    source_eval_path = os.path.join(seq_path, "eval_data.npy")

    base = {}
    if os.path.exists(source_eval_path):
        try:
            base = np.load(source_eval_path, allow_pickle=True).item()
        except Exception as exc:
            print(f"  [eval_data] warning: failed to read {source_eval_path}: {exc}")

    obj_verts = torch.from_numpy(np.asarray(obj_verts_seq)).float()
    hand_verts = torch.from_numpy(np.asarray(hand_verts_seq)).float()
    hand_joints = torch.from_numpy(np.asarray(hand_joints_seq)).float()
    hand_joints = hand_joints[:, _openpose_to_mano_joint_order(), :]
    obj_faces_t = torch.from_numpy(np.asarray(obj_faces)).long()
    hand_faces_t = torch.from_numpy(np.asarray(hand_faces)).long()

    hand_root = hand_joints[:, 0, :]
    obj_root_np = _bbox_centers_np(obj_verts.numpy())
    obj_root = torch.from_numpy(obj_root_np).float()

    num_frames = int(obj_verts.shape[0])
    out_dict = {
        "fnames": base.get("fnames", _default_eval_fnames(seq_path, num_frames)),
        "K": base.get("K"),
        "full_seq_name": base.get("full_seq_name", seq_name),
        "verts.right": hand_verts,
        "jnts.right": hand_joints,
        "root.right": hand_root,
        "j3d_ra.right": hand_joints - hand_root[:, None, :],
        "verts.object": obj_verts,
        "v3d_c.object": obj_verts,
        "root.object": obj_root,
        "v3d_ra.object": obj_verts - obj_root[:, None, :],
        "v3d_right.object": obj_verts - hand_root[:, None, :],
        "faces": {
            "object": obj_faces_t,
            "right": hand_faces_t,
        },
    }
    if out_dict["K"] is None:
        intrinsics_path = os.path.join(seq_path, "intrinsics.json")
        with open(intrinsics_path, "r") as f:
            out_dict["K"] = np.asarray(json.load(f)["intrinsics"], dtype=np.float32)[None]

    np.save(eval_data_path, out_dict, allow_pickle=True)
    print(f"  -> Saved ablation evaluation data to {eval_data_path}")


def compute_segmented_smoothness_loss(seq, seg_start: int, seg_end: int) -> torch.Tensor:
    """
    Smoothness loss computed *within* each of three segments:
      [0, seg_start)  /  [seg_start, seg_end)  /  [seg_end, N)
    No gradient coupling across segment boundaries, so motion inside the
    interaction window cannot bleed into static segments.
    """
    N = seq.shape[0]
    total = seq.new_zeros(())
    for s, e in [(0, seg_start), (seg_start, seg_end), (seg_end, N)]:
        if e - s >= 2:
            chunk = seq[s:e].contiguous()
            total = total + torch.diff(chunk, dim=0).pow(2).sum()
    return total


def parse_contact_map(contact_map):
    """
    解析contact map: {hand_vertex_idx: {"face_id": int, "bary_coords": [w0, w1, w2]}}
    
    Returns:
        dict with 'correspondences', 'hand_indices', 'num_contacts'
    """
    if contact_map is None:
        return None

    # Convert string keys to int if needed
    correspondences = {int(k) if isinstance(k, str) else k: v for k, v in contact_map.items()}

    return {'correspondences': correspondences, 'hand_indices': list(correspondences.keys()), 'num_contacts': len(correspondences)}


def parse_per_frame_contact_map(contact_list, sampled_indices):
    """
    解析 per-frame contact map (新格式 JSON):
    [{"frame_index": int, "correspondences": {"hand_vtx": {"face_id": int, "bary_coords": [...]}}}, ...]

    frame_index 是视频绝对帧号，映射到 sampled_indices 中的位置。
    只保留 sampled_indices 中存在的帧。

    Returns:
        dict: {sampled_pos (int): {"hand_idx": Tensor, "face_vert_idx": Tensor, "bary": Tensor}}
        其中 sampled_pos 是该帧在 sampled_indices 列表中的位置索引。
    """
    if not contact_list:
        return {}
    frame_to_pos = {int(fi): pos for pos, fi in enumerate(sampled_indices)}
    per_frame = {}
    for entry in contact_list:
        fi = int(entry['frame_index'])
        if fi not in frame_to_pos:
            continue
        corr = {int(k): v for k, v in entry['correspondences'].items()}
        if not corr:
            continue
        per_frame[frame_to_pos[fi]] = corr
    return per_frame


def save_contact_debug_meshes(
    debug_dir,
    hand_verts_before,
    hand_verts_after,
    obj_verts_after,
    obj_faces,
    hand_faces,
    ray_end_points,
    sampled_indices,
    interaction_start,
    interaction_end,
):
    """
    Save per-frame debug meshes for contact-map estimation.
    Exports one combined PLY per frame in the interaction range:
    hand-before, hand-after, object, and camera-ray mesh.

    `ray_end_points`: (N, 3) tensor in mesh space. The debug ray is drawn from
    the camera origin (0,0,0) to this endpoint (typically the corrected hand
    root joint). This is for visualisation only, decoupled from the
    `mano_trans`-direction ray used for the actual depth correction.
    """
    def create_ray_mesh(start, end, radius=0.001, color=(255, 255, 0, 255), extend=1.5):
        """Build a thin cylinder from `start` through `end`, optionally extended past `end`.

        `extend` is a multiplier on (end - start); 1.0 means stop at end,
        1.5 means extend 50% past end so the ray visibly crosses the wrist.
        """
        start = np.asarray(start, dtype=np.float32)
        end = np.asarray(end, dtype=np.float32)
        seg = end - start
        length = float(np.linalg.norm(seg))
        if length < 1e-6:
            return trimesh.Trimesh()
        direction = seg / length
        total_len = length * extend
        midpoint = start + direction * (total_len / 2.0)
        mesh = trimesh.creation.cylinder(radius=radius, height=total_len, sections=8)
        rot_matrix = trimesh.geometry.align_vectors([0, 0, 1], direction)
        if rot_matrix is not None:
            mesh.apply_transform(rot_matrix)
        mesh.apply_translation(midpoint)
        # camera ray color: yellow
        mesh.visual.vertex_colors = list(color)
        return mesh

    os.makedirs(debug_dir, exist_ok=True)
    obj_faces_cpu = obj_faces.detach().cpu().numpy().astype(np.int64)
    hand_faces_cpu = hand_faces.detach().cpu().numpy().astype(np.int64)
    n_saved = 0

    for sp in range(interaction_start, interaction_end):
        frame_idx = int(sampled_indices[sp])
        tag = f"sp{sp:03d}_f{frame_idx:04d}"
        out_path = os.path.join(debug_dir, f"contact_debug_{tag}.ply")

        hand_before_mesh = trimesh.Trimesh(
            vertices=hand_verts_before[sp].detach().cpu().numpy(),
            faces=hand_faces_cpu,
            process=False,
        )
        # hand-before color: red
        hand_before_mesh.visual.vertex_colors = [255, 100, 100, 255]

        hand_after_mesh = trimesh.Trimesh(
            vertices=hand_verts_after[sp].detach().cpu().numpy(),
            faces=hand_faces_cpu,
            process=False,
        )
        # hand-after color: green
        hand_after_mesh.visual.vertex_colors = [100, 255, 120, 255]

        obj_after_mesh = trimesh.Trimesh(
            vertices=obj_verts_after[sp].detach().cpu().numpy(),
            faces=obj_faces_cpu,
            process=False,
        )
        # object color: gray
        obj_after_mesh.visual.vertex_colors = [128, 128, 128, 255]

        end_pt = ray_end_points[sp].detach().cpu().numpy()
        ray_mesh = create_ray_mesh(np.zeros(3, dtype=np.float32), end_pt)

        meshes = [hand_before_mesh, hand_after_mesh, obj_after_mesh]
        if ray_mesh.vertices.shape[0] > 0:
            meshes.append(ray_mesh)
        combined = trimesh.util.concatenate(meshes)
        combined.export(out_path, file_type='ply', encoding='ascii')
        n_saved += 1

    print(f"Saved contact combined debug meshes for {n_saved} interaction frames to: {debug_dir}")


def save_stage4_correction_debug_meshes(
    debug_dir,
    hand_verts_before,
    hand_verts_after,
    obj_verts,
    obj_faces,
    hand_faces,
    sampled_indices,
    interaction_start,
    interaction_end,
):
    """Save Stage 4 GFM offset before/after meshes for direct visual comparison."""
    os.makedirs(debug_dir, exist_ok=True)
    obj_faces_cpu = obj_faces.detach().cpu().numpy().astype(np.int64)
    hand_faces_cpu = hand_faces.detach().cpu().numpy().astype(np.int64)
    n_saved = 0

    def _colored_mesh(verts_t, faces_np, color):
        mesh = trimesh.Trimesh(
            vertices=verts_t.detach().cpu().numpy(),
            faces=faces_np,
            process=False,
        )
        mesh.visual.vertex_colors = color
        return mesh

    for sp in range(interaction_start, interaction_end):
        frame_idx = int(sampled_indices[sp])
        tag = f"sp{sp:03d}_f{frame_idx:04d}"

        before_hand = _colored_mesh(hand_verts_before[sp], hand_faces_cpu, [255, 80, 80, 255])
        after_hand = _colored_mesh(hand_verts_after[sp], hand_faces_cpu, [80, 255, 120, 255])
        obj_for_before = _colored_mesh(obj_verts[sp], obj_faces_cpu, [128, 128, 128, 255])
        obj_for_after = _colored_mesh(obj_verts[sp], obj_faces_cpu, [128, 128, 128, 255])
        obj_for_overlay = _colored_mesh(obj_verts[sp], obj_faces_cpu, [128, 128, 128, 255])

        trimesh.util.concatenate([before_hand, obj_for_before]).export(
            os.path.join(debug_dir, f"stage4_before_{tag}.ply"),
            file_type="ply",
            encoding="ascii",
        )
        trimesh.util.concatenate([after_hand, obj_for_after]).export(
            os.path.join(debug_dir, f"stage4_after_{tag}.ply"),
            file_type="ply",
            encoding="ascii",
        )
        trimesh.util.concatenate([before_hand.copy(), after_hand.copy(), obj_for_overlay]).export(
            os.path.join(debug_dir, f"stage4_overlay_{tag}.ply"),
            file_type="ply",
            encoding="ascii",
        )
        n_saved += 1

    print(f"Saved Stage 4 GFM correction before/after PLYs for {n_saved} interaction frames to: {debug_dir}")


def _dump_stage_debug_meshes(out_dir, multi_frame_model, obj_faces,
                             contact_cache, i0, i1, sampled_indices):
    """End-of-stage dump: one PLY per interaction frame containing
    obj (gray), hand (skin), inside hand verts (red), near-contact verts (yellow)."""
    os.makedirs(out_dir, exist_ok=True)
    with torch.no_grad():
        _om, _hm, _ = multi_frame_model()
        hv_all = _hm.verts_padded().detach().cpu().numpy()
        ov_all = _om.verts_padded().detach().cpu().numpy()
        hand_faces_np = _hm.faces_padded()[0].detach().cpu().numpy().astype(np.int64)
    obj_faces_np = obj_faces.detach().cpu().numpy().astype(np.int64)
    bucket_np = contact_cache.bucket.detach().cpu().numpy() if contact_cache is not None else None

    for f_local, f_global in enumerate(range(i0, i1)):
        try:
            sp_idx = int(sampled_indices[f_global])
        except Exception:
            sp_idx = f_global
        tag = f"f{f_global:04d}_sp{sp_idx:04d}"

        hv = hv_all[f_global]
        ov = ov_all[f_global]
        hand_mesh = trimesh.Trimesh(vertices=hv, faces=hand_faces_np, process=False)
        hand_colors = np.tile(np.array([220, 180, 160, 255], dtype=np.uint8), (hv.shape[0], 1))
        if bucket_np is not None:
            b = bucket_np[f_local]
            hand_colors[b == BUCKET_INSIDE]       = np.array([255,  60,  60, 255], dtype=np.uint8)
            hand_colors[b == BUCKET_NEAR_CONTACT] = np.array([255, 230,  60, 255], dtype=np.uint8)
        hand_mesh.visual.vertex_colors = hand_colors

        obj_mesh = trimesh.Trimesh(vertices=ov, faces=obj_faces_np, process=False)
        obj_mesh.visual.vertex_colors = [128, 128, 128, 255]

        combined = trimesh.util.concatenate([hand_mesh, obj_mesh])
        combined.export(os.path.join(out_dir, f"stage_debug_{tag}.ply"),
                        file_type='ply', encoding='ascii')
    print(f"Dumped {i1 - i0} stage debug PLYs to: {out_dir}")


def compute_collision_loss(object_verts, object_normals, hand_verts, ignore_indices=None):
    """
    计算hand和object之间的穿模损失
    Args:
        obj_verts: [B, N_obj, 3] object顶点
        hand_verts: [B, N_hand, 3] hand顶点
        hand_normals: [B, N_hand, 3] hand顶点法线
        ignore_indices: [K] 或 List, 不需要计算碰撞的hand顶点索引 (通常是接触点)
    """
    # 确保输入是batch形式
    if object_verts.ndim == 2:
        object_verts = object_verts.unsqueeze(0)
    if hand_verts.ndim == 2:
        hand_verts = hand_verts.unsqueeze(0)
    if object_normals.ndim == 2:
        object_normals = object_normals.unsqueeze(0)

    B, N_hand, _ = hand_verts.shape

    # 1. 计算hand顶点到object顶点的最近邻距离和索引
    # hand_nn_dist: [B, N_hand]
    hand_nn_dist, hand_nn_idx = get_NN(hand_verts, object_verts)

    # 2. 判断哪些hand顶点在object内部
    # hand_interior: [B, N_hand] (Bool)
    hand_interior = get_interior(object_normals, object_verts, hand_verts, hand_nn_idx).type(torch.bool)
    penetr_dists = hand_nn_dist[hand_interior]
    if penetr_dists.numel() == 0:
        return torch.tensor(0.0, device=hand_verts.device, requires_grad=True)
    # clamp per-vertex distance to suppress extremely large gradients from deep penetration
    hand_in_obj_penetr_dist = torch.clamp(penetr_dists, max=0.04).mean()

    return hand_in_obj_penetr_dist


def mano_pose_to_6d(pose):
    """Convert MANO finger axis-angle pose to 6D rotation representation.

    6D is continuous and free of the 2π wrap / zero-direction degeneracy
    issues of axis-angle, so MSE on 6D is the right form for anchor /
    smoothness / accel terms (see comments around the Stage 3 anchor in
    fit_and_visualize_pose).

    Args:
        pose: (B, J, 3) axis-angle (typically B = N_frames, J = 15).
    Returns:
        (B, J, 6) rotation6d.
    """
    B, J, _ = pose.shape
    R = axis_angle_to_matrix(pose.reshape(-1, 3))           # (B*J, 3, 3)
    return matrix_to_rotation_6d(R).reshape(B, J, 6)


def merge_sam3d_rotation_snapshots_into_stage3_object_pose_init(
    stage3_object_pose_init: torch.Tensor,
    snapshots: Optional[List[Dict[str, Any]]],
    device: torch.device,
) -> int:
    """Overwrite **rotation only** on rows of ``stage3_object_pose_init`` with SAM3D dense output (final, incl. outlier + interp).

    Tail translations stay from ``run_pnp_1stage`` (SAM3D translation is not copied into Stage 3
    ``TemporalHandObjectPose`` — tail ``initial_T`` uses ``init_relative_translation``).

    ``stage3_object_pose_init`` uses OpenCV-style 4×4 with **R column-major**; SAM3D quaternions
    match ``sam3d_sparse_keyframes.render_mesh_soft_overlay_on_rgb`` (``quaternion_to_matrix``).
    """
    if snapshots is None or len(snapshots) == 0:
        return 0
    n_merged = 0
    for snap in snapshots:
        si = int(snap["sampled_idx"])
        if si < 0 or si >= int(stage3_object_pose_init.shape[0]):
            continue
        pose = snap.get("pose") or {}
        q = pose.get("rotation")
        if q is None:
            continue
        q = q.float().to(device=device)
        if q.dim() == 3:
            q = q.squeeze(0)
        if q.dim() == 2 and q.shape[0] == 1:
            q = q.squeeze(0)
        q = q.reshape(-1)[:4]
        R_row = quaternion_to_matrix(q.unsqueeze(0))[0]
        R_col = R_row.mT
        stage3_object_pose_init[si, :3, :3] = R_col
        stage3_object_pose_init[si, 3, 3] = 1.0
        n_merged += 1
    if n_merged > 0:
        print(
            f"[SAM3D-ROT] merged {n_merged} SAM3D dense rotation(s) into stage3_object_pose_init "
            f"(translation unchanged: PnP tail + init_relative_translation for Stage 3)."
        )
    return n_merged


def infer_sam3d_ref_frame_from_inpaint(seq_path: str) -> Optional[int]:
    """Infer the SAM3D reference frame from a single processed/inpaint/*.png file."""
    inpaint_dir = os.path.join(seq_path, "processed", "inpaint")
    if not os.path.isdir(inpaint_dir):
        return None
    candidates = []
    for name in os.listdir(inpaint_dir):
        stem, ext = os.path.splitext(name)
        if ext.lower() not in (".png", ".jpg", ".jpeg"):
            continue
        if not stem.isdigit():
            continue
        candidates.append(int(stem))
    if len(candidates) == 0:
        return None
    if len(candidates) > 1:
        print(f"[SAM3D-REF] multiple inpaint frames in {inpaint_dir}; using earliest: {sorted(candidates)}")
    return int(sorted(candidates)[0])


def infer_processed_image_start_frame(seq_path: str) -> Optional[int]:
    """Return the first original frame id in processed/images, if filenames are numeric."""
    images_dir = os.path.join(seq_path, "processed", "images")
    if not os.path.isdir(images_dir):
        return None
    candidates = []
    for name in os.listdir(images_dir):
        stem, ext = os.path.splitext(name)
        if ext.lower() not in (".png", ".jpg", ".jpeg"):
            continue
        if stem.isdigit():
            candidates.append(int(stem))
    if not candidates:
        return None
    return int(min(candidates))


def prepare_sam3d_keyframe_input(
    raw_rgbs_np: np.ndarray,
    raw_masks_np: np.ndarray,
    frame_idx: int,
    *,
    min_mask_area: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """Return uint8 RGB and binary mask for one SAM3D keyframe."""
    idx = int(frame_idx)
    if idx < 0 or idx >= len(raw_rgbs_np) or idx >= len(raw_masks_np):
        raise IndexError(f"SAM3D keyframe frame_idx={idx} is outside rgb/mask arrays")
    img = np.ascontiguousarray(raw_rgbs_np[idx])
    mask = np.ascontiguousarray(raw_masks_np[idx])
    if img.dtype != np.uint8:
        if img.max() <= 1.0:
            img = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            img = np.clip(img, 0, 255).astype(np.uint8)
    mask_u8 = (mask > 0).astype(np.uint8)
    if int(mask_u8.sum()) < int(min_mask_area):
        raise ValueError(
            f"SAM3D keyframe mask too small at frame_idx={idx}: area={int(mask_u8.sum())}"
        )
    return img, mask_u8


def sam3d_keyframe_pose_to_stage2_init(
    sam3d_out: Dict[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert a SAM3D keyframe output into Stage2 row-major R, T, scale."""
    q = sam3d_out["rotation"]
    if q.dim() == 3:
        q = q.squeeze(1)
    q = q.float().to(device).reshape(-1)[:4]
    initial_R = quaternion_to_matrix(q.unsqueeze(0))[0]
    initial_T = sam3d_out["translation"].float().to(device).reshape(-1)[:3]
    initial_scale = sam3d_out["scale"].float().to(device).reshape(-1)
    if initial_scale.numel() == 1:
        initial_scale = initial_scale[0]
    else:
        initial_scale = initial_scale[:3].mean()
    return initial_R, initial_T, initial_scale


def run_sam3d_ref_keyframe_init(
    *,
    raw_rgbs_np: np.ndarray,
    raw_masks_np: np.ndarray,
    ref_clip_frame_idx: int,
    seed: int,
    quiet: bool,
    config_path: Optional[str],
    lambda_temp: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """Run a single SAM3D keyframe at the requested reference frame."""
    from sam3d_reset import (
        SAM3D_CONFIG_PATH,
        _SAM3D_AVAILABLE,
        _SAM3D_IMPORT_ERROR,
        Sam3dResetController,
    )

    if not _SAM3D_AVAILABLE:
        raise RuntimeError(f"SAM3D not importable: {_SAM3D_IMPORT_ERROR}")

    img, mask = prepare_sam3d_keyframe_input(raw_rgbs_np, raw_masks_np, ref_clip_frame_idx)
    ctrl = Sam3dResetController(
        config_path=config_path or SAM3D_CONFIG_PATH,
        lambda_temp=float(lambda_temp),
        quiet=quiet,
    )
    out = ctrl.keyframe(img, mask, frame_idx=int(ref_clip_frame_idx), seed=int(seed))
    initial_R, initial_T, initial_scale = sam3d_keyframe_pose_to_stage2_init(out, device)
    debug = {
        "enabled": True,
        "ref_clip_frame_idx": int(ref_clip_frame_idx),
        "seed": int(seed),
        "mask_area_px": int(mask.sum()),
        "translation": initial_T.detach().cpu().tolist(),
        "scale": float(initial_scale.detach().cpu().item()),
        "rotation_quat": out["rotation"].detach().cpu().reshape(-1)[:4].tolist(),
    }
    return initial_R, initial_T, initial_scale, debug


class TemporalHandObjectPose(nn.Module):

    def __init__(self,
                 initial_R,
                 initial_T,
                 initial_scale,
                 initial_verts,
                 faces,
                 mano_params=None,
                 mano_root_orient_init=None,
                 mano_trans_init=None,
                 mano_pose_init=None,
                 is_right_init=None):
        super().__init__()
        self.rot_6d = nn.Parameter(matrix_to_rotation_6d(initial_R), requires_grad=True)  # (N, 6)
        self.trans = nn.Parameter(initial_T, requires_grad=True)  # (N, 3)
        self.scale = nn.Parameter(torch.tensor([1.0], dtype=torch.float32, device=initial_scale.device), requires_grad=True)  # (1,)
        self.initial_scale = initial_scale

        self.register_buffer('initial_verts', initial_verts)  # (N, V, 3)
        self.register_buffer('faces', faces)  # (N, F, 3)

        mano_dir = '../../stage1_preprocess/Dyn_HaMR_new/_DATA/data'
        amano_cfg = {
            'mano_assets_root': os.path.join(mano_dir, 'mano'),
            'flat_hand_mean': True,
            'use_pca': False,
            'side': 'right',
        }
        self.hand_model_amano = AMANOLayer(**amano_cfg)

        self.axisFK = AxisLayerFK(side=amano_cfg['side'], mano_assets_root=amano_cfg['mano_assets_root'])
        self.anatomyLoss = AnatomyConstraintLossEE()
        self.anatomyLoss.setup()

        if mano_root_orient_init is not None:
            # This branch for rendering interpolated data
            self.mano_root_orient = nn.Parameter(mano_root_orient_init, requires_grad=False)
            self.mano_trans = nn.Parameter(mano_trans_init, requires_grad=False)
            self.mano_pose = nn.Parameter(mano_pose_init, requires_grad=False)
            self.is_right = is_right_init
        else:
            # This branch for optimization
            T_w2c = torch.stack([torch.tensor(mano_param['T_w2c'], dtype=torch.float32) for mano_param in mano_params])  # (N, 4, 4)
            root_orient = torch.stack([torch.tensor(mano_param['root_orient'], dtype=torch.float32) for mano_param in mano_params])  # (N, 3)
            trans = torch.stack([torch.tensor(mano_param['trans'], dtype=torch.float32) for mano_param in mano_params])  # (N, 3)

            # apply T_w2c to root_orient and trans
            root_orient = matrix_to_axis_angle(T_w2c[:, :3, :3] @ axis_angle_to_matrix(root_orient))
            trans = (T_w2c[:, :3, :3] @ trans.unsqueeze(-1)).squeeze(-1) + T_w2c[:, :3, 3]

            pose = torch.stack([torch.tensor(mano_param['pose'], dtype=torch.float32) for mano_param in mano_params])  # (N, 15, 3)
            betas = torch.stack([torch.tensor(mano_param['betas'], dtype=torch.float32) for mano_param in mano_params])  # (N, 10)
            self.is_right = torch.stack([torch.tensor(mano_param['is_right'], dtype=torch.float32) for mano_param in mano_params])  # (N, 1)

            self.mano_root_orient = nn.Parameter(root_orient, requires_grad=True)  # (N, 3)
            self.mano_trans = nn.Parameter(trans, requires_grad=True)  # (N, 3)
            self.mano_pose = nn.Parameter(pose, requires_grad=True)  # (N, 15, 3)

    def forward(self):
        # object
        N = self.rot_6d.shape[0]
        R = rotation_6d_to_matrix(self.rot_6d)  # (N, 3, 3)
        scale = (self.scale * self.initial_scale).unsqueeze(0).unsqueeze(0)  # (1, 1, 3)
        posed_verts = (self.initial_verts * scale) @ R + self.trans.unsqueeze(1)
        obj_textures = TexturesVertex(verts_features=torch.ones_like(posed_verts))
        obj_mesh = Meshes(verts=posed_verts, faces=self.faces, textures=obj_textures)

        # hand
        device = self.mano_trans.device
        flat_mat = torch.diag(torch.tensor([-1., -1., 1.], dtype=torch.float32, device=device))[None].repeat(N, 1, 1)  # (N, 3, 3)
        amano_output = run_amano(self.hand_model_amano, self.mano_trans[None], self.mano_root_orient[None], self.mano_pose[None], self.is_right.to(device))
        mano_joints = amano_output['joints'].squeeze(0) @ flat_mat  # (1, N, 21, 3) -> (N, 21, 3)
        mano_verts = amano_output['vertices'].squeeze(0) @ flat_mat  # (1, N, 778, 3) -> (N, 778, 3)
        mano_l_faces = amano_output['l_faces']  # (1538, 3)
        mano_r_faces = amano_output['r_faces']  # (1538, 3)
        mano_is_right = amano_output['is_right'].squeeze(0)  # (1, N) -> (N,)
        mano_hand_textures = TexturesVertex(verts_features=torch.ones_like(mano_verts))
        mano_faces = mano_r_faces if mano_is_right[0].item() > 0 else mano_l_faces
        mano_faces = mano_faces[None].repeat(N, 1, 1)
        mano_mesh = Meshes(verts=mano_verts, faces=mano_faces, textures=mano_hand_textures)
        self.transforms_abs = amano_output['transforms_abs']

        return obj_mesh, mano_mesh, mano_joints


class SingleObjectPose(nn.Module):

    def __init__(self, initial_R, initial_T, initial_scale, initial_verts, faces):
        super().__init__()
        self.rot_6d = nn.Parameter(matrix_to_rotation_6d(initial_R), requires_grad=True)  # (6,)
        self.trans = nn.Parameter(initial_T, requires_grad=True)  # (3,)
        self.scale = nn.Parameter(torch.tensor([1.0], dtype=torch.float32, device=initial_scale.device), requires_grad=True)  # (1,)
        self.initial_scale = initial_scale  # (3,)

        self.register_buffer('initial_verts', initial_verts)
        self.register_buffer('faces', faces)

    def forward(self):
        R = rotation_6d_to_matrix(self.rot_6d)  # (3, 3)
        scale = (self.scale * self.initial_scale).unsqueeze(0)  # (1, 3)
        transform = Transform3d(dtype=torch.float32, device=R.device).scale(scale).rotate(R.unsqueeze(0)).translate(self.trans.unsqueeze(0))
        posed_verts = transform.transform_points(self.initial_verts.unsqueeze(0))
        textures = TexturesVertex(verts_features=torch.ones_like(posed_verts))
        return Meshes(verts=posed_verts, faces=self.faces.unsqueeze(0), textures=textures)


def fit_and_visualize_pose(
    seq_path,
    output_path,
    device,
    pred_amodal_masks_np,
    bboxes,
    cropped_rgbs_np,
    cropped_modal_masks_np,
    cropped_hand_masks_np,
    cropped_depths_np,
    cropped_metric_depths_np,
    mano_params,
    hand_keypoints_np,
    hand_keypoints_valid_mask_np,
    num_total_frames,
    overwrite_stage1_3=False,
    overwrite_grasp_correction=False,
    overwrite_sam3d_dense_cache=False,
    smooth_gfm_depth_offset=True,
    lr=1e-2,
    num_steps=200,
    smoothness_weight=1.0,
    cotracker_model=None,
    contact_indices_path=None,
    contact_cone_angle_deg=60.0,
    contact_dist_thresh=0.02,
    contact_surface_samples=10000,
    save_contact_debug_meshes_flag=False,
    pnp_health_dump=False,
    sam3d_sparse_keyframes=False,
    sam3d_raw_rgbs_np=None,
    sam3d_raw_masks_np=None,
    sam3d_sparse_stride=8,
    sam3d_seed=42,
    sam3d_quiet=True,
    sam3d_config_path=None,
    sam3d_lambda_temp=0.5,
    sam3d_rot_outlier_filter=True,
    sam3d_rot_outlier_max_angle_deg=60.0,
    sam3d_rot_outlier_max_iters=3,
    sam3d_rot_retry_count=3,
    sam3d_rot_mesh_overlay_alpha=0.85,
    sam3d_global_frame_offset=0,
    sam3d_ref_frame_idx=None,
    sam3d_ref_from_inpaint=False,
    sam3d_ref_source="default_first_frame",
    sam3d_ref_keyframe_init=False,
    stage5_mode=STAGE5_OBJECT_LITE,
    stage5_presmooth_object=True,
    sample_target_stride=3,
    min_sampled_frames=64,
    max_sampled_frames=128,
    ablation="full",
    save_intermediates=False,
):
    # --- 0. Frame Sampling ---
    num_total_frames = len(bboxes)
    sampled_indices, sampling_policy = build_adaptive_sample_indices(
        num_total_frames,
        target_stride=sample_target_stride,
        min_sampled_frames=min_sampled_frames,
        max_sampled_frames=max_sampled_frames,
    )
    num_sampled_frames = len(sampled_indices)

    print(
        f"Total frames: {num_total_frames}, Sampled frames: {num_sampled_frames} "
        f"(mode={sampling_policy['mode']}, target_stride={sampling_policy['target_stride']}, "
        f"min={sampling_policy['min_sampled_frames']}, max={sampling_policy['max_sampled_frames']})"
    )
    os.makedirs(output_path, exist_ok=True)
    _valid_ablations = {
        "full",
        "no_fp",
        "no_vp",
        "no_stage4",
        "no_stage4_offset",
        "no_penetration",
        "no_stage5_pen",
        "no_contact",
        "no_dyn_contact",
        "no_stage5_contact",
        "no_stage5_smooth",
    }
    if ablation not in _valid_ablations:
        raise ValueError(f"Unknown ablation mode: {ablation}")
    print(f"[Ablation] mode={ablation}")
    disable_stage4 = ablation == "no_stage4"
    disable_contact_terms = ablation in {"no_contact", "no_stage5_contact"}
    disable_penetration_terms = ablation in {"no_penetration", "no_stage5_pen"}
    with open(os.path.join(output_path, "sampling_policy.json"), "w") as _spf:
        json.dump(sampling_policy, _spf, indent=2)

    phase_layout_mode = LAYOUT_FIVE_STAGE
    phase_layout_info = None

    def _stage_label_for_sampled_idx(sampled_idx: int):
        if start_static_end_idx is None or approaching_end_idx is None or interaction_end_idx is None or end_static_start_idx is None:
            return "stage: n/a", (255, 255, 255)
        if phase_layout_mode == LAYOUT_INTERACTION_ONLY:
            return "Interaction-Only", (255, 0, 0)
        if sampled_idx < start_static_end_idx:
            return "0-1: Start Static", (0, 255, 0)
        if sampled_idx < approaching_end_idx:
            return "1-2: Approaching", (255, 255, 0)
        if sampled_idx < interaction_end_idx:
            return "2-3: Interaction", (255, 0, 0)
        if sampled_idx < end_static_start_idx:
            return "3-4: Releasing", (255, 128, 0)
        return "4-5: End Static", (0, 128, 255)

    def _nearest_sampled_idx(full_idx: int) -> int:
        if num_sampled_frames <= 1:
            return 0
        pos = int(np.searchsorted(sampled_indices, full_idx))
        if pos <= 0:
            return 0
        if pos >= num_sampled_frames:
            return num_sampled_frames - 1
        prev_idx = int(sampled_indices[pos - 1])
        next_idx = int(sampled_indices[pos])
        return pos - 1 if abs(full_idx - prev_idx) <= abs(next_idx - full_idx) else pos

    sam3d_ref_clip_frame_idx = 0 if sam3d_ref_frame_idx is None else int(sam3d_ref_frame_idx) - int(sam3d_global_frame_offset)
    sam3d_ref_clip_frame_idx = int(np.clip(sam3d_ref_clip_frame_idx, 0, max(0, num_total_frames - 1)))
    sam3d_ref_sampled_idx = int(_nearest_sampled_idx(sam3d_ref_clip_frame_idx))
    sam3d_ref_actual_clip_frame_idx = int(sampled_indices[sam3d_ref_sampled_idx])
    sam3d_ref_dataset_frame_idx = sam3d_ref_actual_clip_frame_idx + int(sam3d_global_frame_offset)
    sam3d_ref_enabled = sam3d_ref_sampled_idx != 0
    sam3d_ref_obj_mask_path = None
    sam3d_ref_obj_mask_area = None
    if is_hold_video_id(os.path.basename(seq_path)) and sam3d_raw_masks_np is not None:
        _ref_mask, sam3d_ref_obj_mask_path = load_sam3d_ref_obj_mask(
            seq_path,
            sam3d_ref_actual_clip_frame_idx,
        )
        if _ref_mask is not None:
            sam3d_ref_obj_mask_area = int(_ref_mask.sum())
            _preloaded_ref_mask = np.asarray(sam3d_raw_masks_np[sam3d_ref_actual_clip_frame_idx])
            if tuple(_ref_mask.shape) == tuple(_preloaded_ref_mask.shape):
                sam3d_raw_masks_np = np.array(sam3d_raw_masks_np, copy=True)
                sam3d_raw_masks_np[sam3d_ref_actual_clip_frame_idx] = _ref_mask
            else:
                print(
                    "[SAM3D-REF] hold obj_masks ref mask shape mismatch: "
                    f"{_ref_mask.shape} vs {_preloaded_ref_mask.shape}; using preloaded obj_masks array"
                )
        else:
            print(
                "[SAM3D-REF] WARNING: hold ref object mask missing in obj_masks for "
                f"clip_frame={sam3d_ref_actual_clip_frame_idx}; SAM3D will use preloaded obj_masks array"
            )
    print(
        f"[SAM3D-REF] source={sam3d_ref_source} requested_dataset_frame="
        f"{sam3d_ref_frame_idx if sam3d_ref_frame_idx is not None else 0} "
        f"-> sampled_slot={sam3d_ref_sampled_idx}/{num_sampled_frames - 1} "
        f"clip_frame={sam3d_ref_actual_clip_frame_idx} dataset_frame={sam3d_ref_dataset_frame_idx}"
    )

    ref_policy_path = os.path.join(output_path, "sam3d_reference_frame.json")
    with open(ref_policy_path, "w") as _rpf:
        json.dump(
            {
                "source": sam3d_ref_source,
                "from_inpaint": bool(sam3d_ref_from_inpaint),
                "requested_dataset_frame_idx": int(sam3d_ref_frame_idx) if sam3d_ref_frame_idx is not None else 0,
                "requested_clip_frame_idx": int(sam3d_ref_clip_frame_idx),
                "sampled_idx": int(sam3d_ref_sampled_idx),
                "actual_clip_frame_idx": int(sam3d_ref_actual_clip_frame_idx),
                "actual_dataset_frame_idx": int(sam3d_ref_dataset_frame_idx),
                "obj_mask_source": "obj_masks" if sam3d_ref_obj_mask_path is not None else None,
                "obj_mask_path": sam3d_ref_obj_mask_path,
                "obj_mask_area_px": sam3d_ref_obj_mask_area,
                "enabled": bool(sam3d_ref_enabled),
            },
            _rpf,
            indent=2,
        )

    def _sampled_video_label(sampled_idx: int) -> tuple[str, tuple[int, int, int]]:
        stage_name, stage_color = _stage_label_for_sampled_idx(sampled_idx)
        full_idx = int(sampled_indices[sampled_idx]) if sampled_idx < num_sampled_frames else int(sampled_idx)
        return f"sampled={sampled_idx} full={full_idx} | {stage_name}", stage_color

    def _full_video_label(full_idx: int) -> tuple[str, tuple[int, int, int]]:
        sampled_idx = _nearest_sampled_idx(full_idx)
        stage_name, stage_color = _stage_label_for_sampled_idx(sampled_idx)
        return f"full={full_idx} sampled~{sampled_idx} | {stage_name}", stage_color

    # Sample all frame-related data
    sampled_pred_amodal_masks_np = pred_amodal_masks_np[sampled_indices]
    sampled_rgbs_np = cropped_rgbs_np[sampled_indices]
    sampled_modal_masks_np = cropped_modal_masks_np[sampled_indices]
    sampled_depths_np = cropped_depths_np[sampled_indices]
    sampled_metric_depths_np = cropped_metric_depths_np[sampled_indices]
    sampled_hand_masks_np = cropped_hand_masks_np[sampled_indices]
    sampled_mano_params = mano_params[sampled_indices]

    # --- 1. Data Loading & Pre-processing ---
    # Load Mesh
    io = IO()
    io.register_meshes_format(MeshGlbFormat())
    glb_path = os.path.join(seq_path, 'glb_0.glb')
    with open(glb_path, "rb") as f:
        mesh = io.load_mesh(f, include_textures=True).to(device)

    mesh = simplify_mesh(mesh, target_triangles=5000)

    verts, faces = mesh.verts_list()[0], mesh.faces_list()[0]
    verts = verts @ torch.tensor([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=torch.float32, device=device).T

    # Load and convert initial pose
    transform_path = os.path.join(seq_path, 'transform_0.json')
    with open(transform_path, 'r') as f:
        transform_matrix = torch.tensor(json.load(f)['transform'], dtype=torch.float32, device=device)  # (4, 4), row-major

    # Decompose the 3x3 part into rotation and scale
    M = transform_matrix[:3, :3]
    initial_scale = torch.linalg.norm(M, dim=0).mean()
    initial_R = M / initial_scale  # row major
    initial_T = transform_matrix[3, :3]
    sam3d_ref_keyframe_init_debug = {"enabled": False}
    if sam3d_ref_keyframe_init:
        print(
            "[SAM3D-REF-INIT] running single-frame SAM3D keyframe init at "
            f"clip_frame={sam3d_ref_actual_clip_frame_idx}"
        )
        initial_R, initial_T, initial_scale, sam3d_ref_keyframe_init_debug = run_sam3d_ref_keyframe_init(
            raw_rgbs_np=sam3d_raw_rgbs_np,
            raw_masks_np=sam3d_raw_masks_np,
            ref_clip_frame_idx=int(sam3d_ref_actual_clip_frame_idx),
            seed=int(sam3d_seed),
            quiet=bool(sam3d_quiet),
            config_path=sam3d_config_path,
            lambda_temp=float(sam3d_lambda_temp),
            device=device,
        )
        sam3d_ref_keyframe_init_debug.update(
            {
                "ref_sampled_idx": int(sam3d_ref_sampled_idx),
                "ref_dataset_frame_idx": int(sam3d_ref_dataset_frame_idx),
                "obj_mask_path": sam3d_ref_obj_mask_path,
            }
        )
        with open(os.path.join(output_path, "sam3d_ref_keyframe_init.json"), "w") as _skf:
            json.dump(sam3d_ref_keyframe_init_debug, _skf, indent=2)
        print(
            "[SAM3D-REF-INIT] replaced transform_0.json init with SAM3D keyframe pose "
            f"(mask_area={sam3d_ref_keyframe_init_debug['mask_area_px']})"
        )

    # Load camera intrinsics
    intrinsics_path = os.path.join(seq_path, 'intrinsics.json')
    with open(intrinsics_path, 'r') as f:
        intrinsics = torch.tensor(json.load(f)['intrinsics'], dtype=torch.float32, device=device)  # (3, 3)

    # --- 2. Build Global Crop-space Camera Model (Batched) ---
    H_ori, W_ori = 1080, 1920
    H_out, W_out = 256, 512

    # Since it's a global bbox, we compute intrinsics once
    global_bbox = bboxes[0]  # Assuming all bboxes are the same
    x1, y1, x2, y2 = global_bbox
    w_crop, h_crop = x2 - x1, y2 - y1

    scale = min(W_out / w_crop, H_out / h_crop)
    new_w, new_h = int(w_crop * scale), int(h_crop * scale)
    pad_left = (W_out - new_w) // 2
    pad_top = (H_out - new_h) // 2

    fx_new = intrinsics[0, 0] * scale
    fy_new = intrinsics[1, 1] * scale
    cx_new = (intrinsics[0, 2] - x1) * scale + pad_left
    cy_new = (intrinsics[1, 2] - y1) * scale + pad_top

    focal_length = torch.tensor([[fx_new, fy_new]], device=device).expand(num_sampled_frames, -1)
    principal_point = torch.tensor([[cx_new, cy_new]], device=device).expand(num_sampled_frames, -1)

    sampled_pred_amodal_masks = torch.from_numpy(sampled_pred_amodal_masks_np).float().to(device)
    sampled_depths = torch.from_numpy(sampled_depths_np).float().to(device)
    sampled_rgbs = torch.from_numpy(sampled_rgbs_np).float().to(device)
    sampled_modal_masks = torch.from_numpy(sampled_modal_masks_np).to(device)
    sampled_hand_masks = torch.from_numpy(sampled_hand_masks_np).to(device)

    # --- Process 2D hand keypoints ---
    gt_hand_joints_2d_np = hand_keypoints_np[..., :2].copy()  # (N, 21, 2)
    sampled_gt_hand_joints_2d_np = gt_hand_joints_2d_np[sampled_indices]
    sampled_hand_keypoints_valid_mask_np = hand_keypoints_valid_mask_np[sampled_indices]
    sampled_gt_hand_joints_2d_np[..., 0] = (sampled_gt_hand_joints_2d_np[..., 0] - x1) * scale + pad_left
    sampled_gt_hand_joints_2d_np[..., 1] = (sampled_gt_hand_joints_2d_np[..., 1] - y1) * scale + pad_top
    sampled_gt_hand_joints_2d = torch.from_numpy(sampled_gt_hand_joints_2d_np).float().to(device)
    sampled_gt_hand_joints_valid_mask = torch.from_numpy(sampled_hand_keypoints_valid_mask_np).to(device)

    # ========================================================================
    # Checkpoint: Stage 3 结果保存/加载，避免重复运行 Stages 1-3
    #   - 首次运行：执行 Stages 1-3 并在末尾保存 checkpoint
    #   - 有 contact map 后重新运行：加载 checkpoint，直接跳到 Stage 4+
    # ========================================================================
    checkpoint_path = os.path.join(output_path, "stage3_checkpoint.pt")
    _ckpt_loaded = False
    # Filled in Stage 1 for post–Stage 2 interaction typing (not in Stage 3 ckpt).
    cotracker_tracks_fwd_np = None
    cotracker_fwd_vis_np = None

    if overwrite_stage1_3 and os.path.exists(checkpoint_path):
        print(f"\n--overwrite_stage1_3: removing existing Stage 3 checkpoint at {checkpoint_path}")
        os.remove(checkpoint_path)

    checkpoint_is_usable = True
    if sam3d_ref_keyframe_init and os.path.exists(checkpoint_path):
        checkpoint_is_usable = False
        print(
            "  --sam3d_ref_keyframe_init: ignoring existing Stage 3 checkpoint so "
            "SAM3D keyframe init reruns at the requested reference frame."
        )
    if os.path.exists(checkpoint_path) and checkpoint_is_usable:
        try:
            _ckpt_probe = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            _ckpt_ref_sampled = int(_ckpt_probe.get("sam3d_ref_sampled_idx", 0))
            _ckpt_ref_actual = int(_ckpt_probe.get("sam3d_ref_actual_clip_frame_idx", int(sampled_indices[_ckpt_ref_sampled]) if _ckpt_ref_sampled < len(sampled_indices) else 0))
            if _ckpt_ref_sampled != int(sam3d_ref_sampled_idx) or _ckpt_ref_actual != int(sam3d_ref_actual_clip_frame_idx):
                checkpoint_is_usable = False
                print(
                    "  Existing Stage 3 checkpoint uses a different SAM3D/PnP reference frame "
                    f"(ckpt sampled={_ckpt_ref_sampled}, clip={_ckpt_ref_actual}; "
                    f"requested sampled={sam3d_ref_sampled_idx}, clip={sam3d_ref_actual_clip_frame_idx}). "
                    "Ignoring it for this run."
                )
            _ckpt_sparse = _ckpt_probe.get("sam3d_sparse_keyframes", None)
            _ckpt_ref_keyframe_init = _ckpt_probe.get("sam3d_ref_keyframe_init", None)
            if _ckpt_sparse is None or _ckpt_ref_keyframe_init is None:
                checkpoint_is_usable = False
                print(
                    "  Existing Stage 3 checkpoint has no SAM3D pose-source metadata. "
                    "Ignoring it for this run."
                )
            elif bool(_ckpt_sparse) != bool(sam3d_sparse_keyframes) or bool(_ckpt_ref_keyframe_init) != bool(sam3d_ref_keyframe_init):
                checkpoint_is_usable = False
                print(
                    "  Existing Stage 3 checkpoint uses a different object pose source "
                    f"(ckpt sparse={bool(_ckpt_sparse)}, ref_keyframe_init={bool(_ckpt_ref_keyframe_init)}; "
                    f"requested sparse={bool(sam3d_sparse_keyframes)}, ref_keyframe_init={bool(sam3d_ref_keyframe_init)}). "
                    "Ignoring it for this run."
                )
        except Exception as _e_probe:
            print(f"  WARNING: could not inspect Stage 3 checkpoint reference metadata ({_e_probe}); attempting normal load.")

    if os.path.exists(checkpoint_path) and checkpoint_is_usable:
        print("\n" + "=" * 80)
        print("Found Stage 3 checkpoint – loading and skipping Stages 1-3.")
        print("=" * 80)
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

        # Index / boundary variables
        requested_sampled_indices = list(sampled_indices)
        requested_sampling_policy = dict(sampling_policy)
        sampled_indices = ckpt['sampled_indices']
        num_sampled_frames = int(ckpt['num_sampled_frames'])
        sampling_policy = ckpt.get('sampling_policy', {
            "mode": "checkpoint_legacy",
            "num_total_frames": int(num_total_frames),
            "num_sampled_frames": int(num_sampled_frames),
            "sampled_indices": [int(i) for i in sampled_indices],
        })
        if [int(i) for i in sampled_indices] != [int(i) for i in requested_sampled_indices]:
            print(
                "  WARNING: existing Stage 3 checkpoint uses different sampled_indices "
                f"(ckpt={num_sampled_frames}, requested={len(requested_sampled_indices)}). "
                "Use --overwrite_stage1_3 to apply the current adaptive sampling policy."
            )
            print(f"  Requested sampling policy: {requested_sampling_policy}")
        start_static_end_idx = int(ckpt['start_static_end_idx'])
        approaching_end_idx = int(ckpt['approaching_end_idx'])
        interaction_end_idx = int(ckpt['interaction_end_idx'])
        end_static_start_idx = int(ckpt['end_static_start_idx'])
        phase_layout_mode = str(ckpt.get('phase_layout_mode', LAYOUT_FIVE_STAGE))
        phase_layout_info = ckpt.get('phase_layout_info')

        # Camera
        focal_length = ckpt['focal_length'].to(device)
        principal_point = ckpt['principal_point'].to(device)
        H_out = int(ckpt['H_out'])
        W_out = int(ckpt['W_out'])
        fx_new = float(ckpt['fx_new'])
        fy_new = float(ckpt['fy_new'])
        cx_new = float(ckpt['cx_new'])
        cy_new = float(ckpt['cy_new'])

        # Object mesh
        verts = ckpt['verts'].to(device)
        faces = ckpt['faces'].to(device)

        # Data arrays (numpy)
        sampled_pred_amodal_masks_np = ckpt['sampled_pred_amodal_masks_np']
        sampled_rgbs_np = ckpt['sampled_rgbs_np']
        sampled_modal_masks_np = ckpt['sampled_modal_masks_np']
        sampled_hand_masks_np = ckpt['sampled_hand_masks_np']
        sampled_metric_depths_np = ckpt['sampled_metric_depths_np']

        # Reconstruct tensors from numpy
        sampled_pred_amodal_masks = torch.from_numpy(sampled_pred_amodal_masks_np).float().to(device)
        sampled_rgbs = torch.from_numpy(sampled_rgbs_np).float().to(device)
        sampled_modal_masks = torch.from_numpy(sampled_modal_masks_np).to(device)
        sampled_hand_masks = torch.from_numpy(sampled_hand_masks_np).to(device)

        # 2D hand keypoints
        sampled_gt_hand_joints_2d = ckpt['sampled_gt_hand_joints_2d'].to(device)
        sampled_gt_hand_joints_valid_mask = ckpt['sampled_gt_hand_joints_valid_mask'].to(device)

        # Reconstruct multi_frame_model from checkpoint
        _ckpt_mano_params = ckpt['sampled_mano_params']
        _initial_scale_ckpt = ckpt['initial_scale'].to(device)
        verts_batch = verts.unsqueeze(0).repeat(num_sampled_frames, 1, 1)
        faces_batch = faces.unsqueeze(0).repeat(num_sampled_frames, 1, 1)
        dummy_R = torch.eye(3, device=device).unsqueeze(0).repeat(num_sampled_frames, 1, 1)
        dummy_T = torch.zeros(num_sampled_frames, 3, device=device)
        multi_frame_model = TemporalHandObjectPose(
            initial_R=dummy_R,
            initial_T=dummy_T,
            initial_scale=_initial_scale_ckpt,
            initial_verts=verts_batch,
            faces=faces_batch,
            mano_params=_ckpt_mano_params,
        ).to(device)
        # Snapshot raw HaMeR mano_pose BEFORE load_state_dict overwrites it
        # with the optimised Stage 3 result — Stages 5/6 anchor against the
        # original detector output, not the post-optim state.
        _hamer_mano_pose_init = multi_frame_model.mano_pose.detach().clone()
        _hamer_mano_pose_6d_init = mano_pose_to_6d(_hamer_mano_pose_init).detach()
        multi_frame_model.load_state_dict(ckpt['model_state_dict'])
        multi_frame_model.initial_scale = _initial_scale_ckpt  # not in state_dict

        _ckpt_loaded = True
        print(f"  Loaded: num_sampled_frames={num_sampled_frames}, "
              f"interaction_end={interaction_end_idx}, "
              f"end_static_start={end_static_start_idx}")

        if sam3d_sparse_keyframes:
            _gate_mode = FORCE_ROTATION_MODE
            _gate_original_mode = None
            for _fn in (
                "interaction_motion_profile_sam3d_gate.json",
                os.path.join("sam3d_policy_diag", "interaction_motion_profile_sam3d_gate.json"),
                "interaction_motion_profile_stage3.json",
            ):
                _jp = os.path.join(output_path, _fn)
                if not os.path.isfile(_jp):
                    continue
                try:
                    with open(_jp) as _jf:
                        _gate_profile_resume = force_rotation_likely_profile(
                            json.load(_jf),
                            reason="checkpoint_resume_force_rotation_likely",
                        )
                    _gate_mode = str(_gate_profile_resume["suggested_mode"])
                    _gate_original_mode = _gate_profile_resume.get("policy_original_suggested_mode")
                    print(f"[SAM3D-GATE] checkpoint resume: force suggested_mode={_gate_mode} "
                          f"(original={_gate_original_mode}, from {_jp})")
                    break
                except Exception:
                    continue
            _lo_ck = ckpt.get("stage1_interaction_lo_auto")
            _hi_ck = ckpt.get("stage1_interaction_hi_auto")
            _stride_ck = sparse_stride_for_motion_profile(_gate_mode, sam3d_sparse_stride)
            if _gate_mode == "translation_likely":
                run_sam3d_sparse_keyframes_motion_window(
                    enabled=True,
                    output_path=output_path,
                    sampled_indices=sampled_indices,
                    num_sampled_frames=num_sampled_frames,
                    stage1_interaction_lo_auto=int(_lo_ck) if _lo_ck is not None else None,
                    stage1_interaction_hi_auto=int(_hi_ck) if _hi_ck is not None else None,
                    approaching_end_idx_padded=int(approaching_end_idx),
                    interaction_end_idx_padded=int(interaction_end_idx),
                    raw_rgbs_np=sam3d_raw_rgbs_np,
                    raw_masks_np=sam3d_raw_masks_np,
                    stride=_stride_ck,
                    seed=sam3d_seed,
                    quiet=sam3d_quiet,
                    config_path=sam3d_config_path,
                    lambda_temp=sam3d_lambda_temp,
                    intrinsics_full=intrinsics,
                    mesh_verts=verts,
                    mesh_faces=faces,
                    device=device,
                )
            elif _gate_mode == "rotation_likely":
                print("[SAM3D-GATE] checkpoint resume: forced rotation_likely; dense SAM3D + PnP compare runs after Stage 2 — skipped here.")
            else:
                print(f"[SAM3D-GATE] checkpoint resume: skip milestone SAM3D (suggested_mode={_gate_mode}).")

    if not _ckpt_loaded:
        # ========================================================================
        # PIPELINE OVERVIEW (read this first)
        # ------------------------------------------------------------------------
        # The pipeline has 6 STAGEs.  Their relationship to the on-disk logs:
        #
        #   STAGE 1 (this block)  : Interaction-segment detection.  No optim,
        #                           no log file.
        #   STAGE 2 (~line 1090)  : 0-1 segment object SingleObjectPose opt.
        #                           Logs to optimize_stage1_3.log under header
        #                           "# Stage 2 — Start-static single-frame
        #                           optimisation".
        #   STAGE 3 (~line 1430)  : Multi-frame joint object+hand pose opt
        #                           (object: PnP chain + optional SAM3D dense merge
        #                           when rotation_likely → temporal optim).
        #                           Logs to optimize_stage1_3.log under header
        #                           "# Stage 3 — Multi-frame pose sequence
        #                           optimisation".
        #   STAGE 4 (~line 1870)  : Load contact correspondence + apply
        #                           camera_ray_depth_offset.json (hand z
        #                           rigid-shift correction).  No optim, no log.
        #   STAGE 5 (~line 2245)  : Penetration resolution (hand wrist+fingers
        #                           vs object).  Logs to optimize_stage5.log.
        #   STAGE 6 (~line 2485)  : Contact tightening.  Logs to
        #                           optimize_stage6.log.
        #
        # NOTE on hand global pose: the hand wrist/translation is initialised
        # from HaMeR and *only* refined by 2D-joint reprojection + smoothness
        # + anatomy in STAGE 3.  There is NO metric-depth supervision on the
        # hand at any stage; the only depth correction for the hand is the
        # post-hoc camera_ray_depth_offset.json applied in STAGE 4.  Cases
        # where HaMeR's z is badly off (e.g. occluded / out-of-frame hands)
        # can therefore exhibit perspective ambiguity (hand pushed far →
        # 2D joints collapse to a cluster) that STAGE 3 alone cannot fix.
        # ========================================================================

        # Cross-stage mutable state — assigned by nested stage functions via nonlocal
        start_static_end_idx = None
        end_static_start_idx = None
        approaching_end_idx  = None
        interaction_end_idx  = None
        single_frame_model   = None
        final_posed_mesh     = None
        _opt_log             = None
        stage3_object_pose_init = None  # (N,4,4) OpenCV col-major R; PnP tail + optional SAM3D overwrite
        multi_frame_model    = None
        _hamer_mano_pose_init    = None
        _hamer_mano_pose_6d_init = None
        # Interaction typing (Stage 3): CoTracker + mask + MANO; optional PnP enrich after initial PnP.
        interaction_profile_stage3 = None
        # Motion typing is kept for diagnostics, but policy is forced to rotation_likely.
        sam3d_gate_suggested_mode = FORCE_ROTATION_MODE
        # rotation_likely: SAM3D final ``pose_snapshots`` merged into ``stage3_object_pose_init`` (see ``merge_sam3d_*``).
        sam3d_rotation_pose_snapshots: Optional[List[Dict[str, Any]]] = None
        # Stage 1 auto interaction bounds (forward/backward + amodal IoU; pre-pad) for Stage 3 policy diag.
        stage1_interaction_lo_auto = None
        stage1_interaction_hi_auto = None
        stage1_ious_consecutive_np = None
        phase_layout_mode = LAYOUT_FIVE_STAGE
        phase_layout_info = None

        def _interaction_motion_typing_window():
            """Use the unpadded interaction core for motion typing, not boundary buffers."""
            opt_lo = int(approaching_end_idx)
            opt_hi = int(interaction_end_idx)
            type_lo = int(stage1_interaction_lo_auto) if stage1_interaction_lo_auto is not None else opt_lo
            type_hi = int(stage1_interaction_hi_auto) if stage1_interaction_hi_auto is not None else opt_hi
            if type_hi <= type_lo + 1:
                type_lo, type_hi = opt_lo, opt_hi
            return type_lo, type_hi, opt_lo, opt_hi

        def _run_stage1():
            nonlocal start_static_end_idx, end_static_start_idx, approaching_end_idx, interaction_end_idx
            nonlocal cotracker_tracks_fwd_np, cotracker_fwd_vis_np
            nonlocal stage1_interaction_lo_auto, stage1_interaction_hi_auto, stage1_ious_consecutive_np
            nonlocal phase_layout_mode, phase_layout_info
            # ========================================================================
            # STAGE 1: Detect Interaction Segments (0-1, 1-2, 2-3, 3-4, 4-5)
            # ========================================================================
            print("\n" + "=" * 80)
            print("STAGE 1: Detecting Interaction Segments (0-1, 1-2, 2-3, 3-4, 4-5)")
            print("=" * 80)
            # Step 1.1: Detect static segments using mask sequence
            start_static_end_idx, end_static_start_idx = detect_static_segments(sampled_modal_masks_np, iou_threshold=0.95)

            # ========================================================================
            # Auto-detect interaction segment (2-3) using forward and backward tracking
            # ========================================================================
            print("\n" + "=" * 80)
            print("Auto-detecting interaction segment using tracking displacement + mask IoU")
            print("=" * 80)

            displacement_threshold = 5.0  # pixels
            ratio_threshold = 0.5  # 50% of points
            iou_threshold = 0.8  # IoU threshold for mask change

            # Compute IoU between consecutive frames for amodal masks
            print("\nComputing IoU between consecutive frames...")
            ious_consecutive = []
            for t in range(num_sampled_frames - 1):
                mask_t = sampled_pred_amodal_masks_np[t].astype(bool)
                mask_t1 = sampled_pred_amodal_masks_np[t + 1].astype(bool)
                intersection = np.logical_and(mask_t, mask_t1).sum()
                union = np.logical_or(mask_t, mask_t1).sum()
                iou = intersection / (union + 1e-6)
                ious_consecutive.append(iou)
            ious_consecutive = np.array(ious_consecutive)  # (T-1,)
            print(f"  Mean IoU: {ious_consecutive.mean():.4f}, Min: {ious_consecutive.min():.4f}, Max: {ious_consecutive.max():.4f}")

            # Forward tracking: from first frame
            print("\nForward tracking from first frame...")

            # run CoTracker
            valid_points_y, valid_points_x = np.where(sampled_modal_masks_np[0])
            valid_points_xy = np.stack([valid_points_x, valid_points_y], axis=1)
            valid_points_xy_torch = torch.from_numpy(valid_points_xy).float().to(device)
            num_queries = 10 * 10
            if valid_points_xy_torch.shape[0] > num_queries:
                sampled_points_torch = farthest_point_sampling_torch(valid_points_xy_torch, num_queries)
            else:
                sampled_points_torch = valid_points_xy_torch

            N = sampled_points_torch.shape[0]
            t_col = torch.zeros((N, 1), device=device)  # frame index is always 0
            queries = torch.cat([t_col, sampled_points_torch], dim=1)[None]  # (1, N, 3)

            with torch.no_grad():
                video_forward = sampled_rgbs.permute(0, 3, 1, 2).unsqueeze(0).contiguous()  # (1, T, 3, H, W)
                pred_tracks_fwd, pred_vis_fwd = cotracker_model(
                    video_forward,
                    queries=queries,  # queries at t=0
                    backward_tracking=True,
                )
                tracks_2d_fwd = pred_tracks_fwd[0].cpu().numpy()  # (T, N, 2)
                cotracker_tracks_fwd_np = np.ascontiguousarray(tracks_2d_fwd)
                cotracker_fwd_vis_np = np.ascontiguousarray(
                    pred_vis_fwd[0].detach().cpu().numpy())

            # Compute displacement from first frame for all frames
            initial_positions = sampled_points_torch.cpu().numpy()  # (N, 2) - same as queries positions
            displacements_fwd = np.linalg.norm(tracks_2d_fwd - initial_positions[None, :, :], axis=2)  # (T, N)

            # Find first frame where (>50% points moved >5px) OR (IoU < 0.8 with prev).
            # OR (instead of AND): the previous AND form was too strict — a real
            # interaction frame would fail to trigger if only one of the two
            # signals was present, shrinking the detected window. Either signal
            # alone is sufficient evidence of motion.
            # Anti-noise: require the trigger condition to hold for `consec_required`
            # consecutive frames, to ignore one-off CoTracker glitches.
            moved_ratios_fwd = np.mean(displacements_fwd > displacement_threshold, axis=1)  # (T,)
            approaching_end_idx_auto = start_static_end_idx  # Default
            consec_required = 2

            def _per_frame_trigger_fwd(t):
                d_ok = moved_ratios_fwd[t] > ratio_threshold
                i_ok = (t > 0) and (ious_consecutive[t - 1] < iou_threshold)
                return d_ok or i_ok

            for t in range(start_static_end_idx, num_sampled_frames - consec_required + 1):
                if all(_per_frame_trigger_fwd(t + k) for k in range(consec_required)):
                    approaching_end_idx_auto = t
                    _disp_ok = moved_ratios_fwd[t] > ratio_threshold
                    _iou_ok = (t > 0) and (ious_consecutive[t - 1] < iou_threshold)
                    print(f"  Frame {t}: disp={moved_ratios_fwd[t]*100:.1f}% (>{ratio_threshold*100:.0f}%? {_disp_ok})  "
                          f"iou_prev={ious_consecutive[t-1] if t > 0 else float('nan'):.4f} (<{iou_threshold}? {_iou_ok})")
                    print(f"  → Interaction start (OR-rule, {consec_required}-frame consistent) at frame {t}")
                    break

            if approaching_end_idx_auto == start_static_end_idx:
                print(f"  No significant movement detected, using default: {start_static_end_idx}")

            # Backward tracking: from last frame
            print("\nBackward tracking from last frame...")

            # Get queries from last frame mask
            valid_points_y_last, valid_points_x_last = np.where(sampled_modal_masks_np[-1])
            valid_points_xy_last = np.stack([valid_points_x_last, valid_points_y_last], axis=1)
            valid_points_xy_torch_last = torch.from_numpy(valid_points_xy_last).float().to(device)

            if valid_points_xy_torch_last.shape[0] > num_queries:
                sampled_points_torch_last = farthest_point_sampling_torch(valid_points_xy_torch_last, num_queries)
            else:
                sampled_points_torch_last = valid_points_xy_torch_last

            N_last = sampled_points_torch_last.shape[0]
            # Queries for reversed video (first frame of reversed = last frame of original)
            t_col_reversed = torch.zeros((N_last, 1), device=device)
            queries_reversed = torch.cat([t_col_reversed, sampled_points_torch_last], dim=1)[None]  # (1, N, 3)

            with torch.no_grad():
                # Reverse the video for backward tracking
                video_backward = torch.flip(sampled_rgbs, dims=[0]).permute(0, 3, 1, 2).unsqueeze(0).contiguous()  # (1, T, 3, H, W)

                pred_tracks_bwd, pred_vis_bwd = cotracker_model(
                    video_backward,
                    queries=queries_reversed,
                    backward_tracking=True,
                )
                tracks_2d_bwd_reversed = pred_tracks_bwd[0].cpu().numpy()  # (T, N, 2) in reversed time
                # Flip back to original time order
                tracks_2d_bwd = tracks_2d_bwd_reversed[::-1]  # Now (T, N, 2) in original time order

            # Compute displacement from last frame for all frames
            last_positions = sampled_points_torch_last.cpu().numpy()  # (N, 2)
            displacements_bwd = np.linalg.norm(tracks_2d_bwd - last_positions[None, :, :], axis=2)  # (T, N)

            # Find last frame where (>50% moved >5px from last) OR (IoU < 0.8 with next).
            # Same OR + consecutive-frame consistency rule as forward pass.
            moved_ratios_bwd = np.mean(displacements_bwd > displacement_threshold, axis=1)  # (T,)
            interaction_end_idx_auto = end_static_start_idx  # Default

            def _per_frame_trigger_bwd(t):
                d_ok = moved_ratios_bwd[t] > ratio_threshold
                i_ok = (t < num_sampled_frames - 1) and (ious_consecutive[t] < iou_threshold)
                return d_ok or i_ok

            for t in range(end_static_start_idx - 1, approaching_end_idx_auto - 1 + (consec_required - 1), -1):
                if all(_per_frame_trigger_bwd(t - k) for k in range(consec_required)):
                    interaction_end_idx_auto = t + 1
                    _disp_ok = moved_ratios_bwd[t] > ratio_threshold
                    _iou_ok = (t < num_sampled_frames - 1) and (ious_consecutive[t] < iou_threshold)
                    print(f"  Frame {t}: disp={moved_ratios_bwd[t]*100:.1f}% (>{ratio_threshold*100:.0f}%? {_disp_ok})  "
                          f"iou_next={ious_consecutive[t] if t < num_sampled_frames - 1 else float('nan'):.4f} (<{iou_threshold}? {_iou_ok})")
                    print(f"  → Interaction end (OR-rule, {consec_required}-frame consistent) at frame {t + 1}")
                    break

            if interaction_end_idx_auto == end_static_start_idx:
                print(f"  No significant movement detected, using default: {end_static_start_idx}")

            stage1_ious_consecutive_np = np.asarray(ious_consecutive, dtype=np.float64).copy()
            stage1_interaction_lo_auto = int(approaching_end_idx_auto)
            stage1_interaction_hi_auto = int(interaction_end_idx_auto)

            phase_layout_info = classify_stage_layout_from_interaction_bounds(
                num_sampled_frames=num_sampled_frames,
                interaction_start=approaching_end_idx_auto,
                interaction_end=interaction_end_idx_auto,
                short_side_threshold=5,
            )
            phase_layout_info = apply_hold_interaction_only_override(
                phase_layout_info,
                video_id=os.path.basename(seq_path),
                num_sampled_frames=num_sampled_frames,
            )
            phase_layout_mode = str(phase_layout_info["layout_mode"])
            print(
                f"\nStage layout: {phase_layout_mode} "
                f"(pre={phase_layout_info['pre_interaction_len']}, "
                f"post={phase_layout_info['post_interaction_len']}, "
                f"reason={phase_layout_info['reason']})"
            )

            if phase_layout_mode == LAYOUT_INTERACTION_ONLY:
                print("  Interaction-only: use sampled frame 0 as Stage2 bootstrap; "
                      "treat sampled [0:N) as interaction.")
                start_static_end_idx = 1
                approaching_end_idx_auto = 0
                interaction_end_idx_auto = num_sampled_frames
                end_static_start_idx = num_sampled_frames
                stage1_interaction_lo_auto = 0
                stage1_interaction_hi_auto = int(num_sampled_frames)

            # Boundary buffer: detection is conservative (still tends to trigger
            # 1–2 frames after motion onset / before motion offset). Pad each side
            # by `BOUNDARY_PAD` frames, clipped to the static segments. Resulting
            # extra frames at the edges are mild outliers; the ray-scale alpha
            # uses a self-filtered subset (top-70% by hand-obj gap) so it is
            # robust to a few non-grasping frames included in the window.
            BOUNDARY_PAD = 2
            if phase_layout_mode == LAYOUT_INTERACTION_ONLY:
                approaching_end_idx_padded = 0
                interaction_end_idx_padded = num_sampled_frames
            else:
                approaching_end_idx_padded = max(start_static_end_idx, approaching_end_idx_auto - BOUNDARY_PAD)
                interaction_end_idx_padded = min(end_static_start_idx, interaction_end_idx_auto + BOUNDARY_PAD)
            if (approaching_end_idx_padded != approaching_end_idx_auto
                    or interaction_end_idx_padded != interaction_end_idx_auto):
                print(f"  Boundary pad ±{BOUNDARY_PAD}: "
                      f"[{approaching_end_idx_auto}:{interaction_end_idx_auto}) → "
                      f"[{approaching_end_idx_padded}:{interaction_end_idx_padded})")
            approaching_end_idx = approaching_end_idx_padded
            interaction_end_idx = interaction_end_idx_padded

            print(f"\nAuto-detected interaction segment: [{approaching_end_idx}:{interaction_end_idx}] ({interaction_end_idx - approaching_end_idx} frames)")

            # Per-frame diagnostic table for boundary detection (helps debug
            # cases where the auto-detected window looks wrong).
            print(f"\n  Per-frame motion signals (frame: disp_fwd% disp_bwd% iou_consec  in_window):")
            for t in range(num_sampled_frames):
                in_win = "*" if approaching_end_idx <= t < interaction_end_idx else " "
                iou_str = f"{ious_consecutive[t-1]:.3f}" if t > 0 else " ----"
                print(f"    f{t:3d}: fwd={moved_ratios_fwd[t]*100:5.1f}%  bwd={moved_ratios_bwd[t]*100:5.1f}%  iou_prev={iou_str}  {in_win}")

            # Store segment boundaries for later use
            if phase_layout_mode == LAYOUT_INTERACTION_ONLY:
                segment_boundaries = {
                    'bootstrap': (0, start_static_end_idx),
                    'approaching': (start_static_end_idx, start_static_end_idx),
                    'interaction': (approaching_end_idx, interaction_end_idx),
                    'releasing': (interaction_end_idx, interaction_end_idx),
                    'end_static': (num_sampled_frames, num_sampled_frames),
                }
            else:
                segment_boundaries = {
                    'start_static': (0, start_static_end_idx),
                    'approaching': (start_static_end_idx, approaching_end_idx),
                    'interaction': (approaching_end_idx, interaction_end_idx),
                    'releasing': (interaction_end_idx, end_static_start_idx),
                    'end_static': (end_static_start_idx, num_sampled_frames)
                }

            print(f"\nFinal segment boundaries:")
            for name, (start, end) in segment_boundaries.items():
                print(f"  {name}: [{start}:{end}] ({end - start} frames)")

            def _stage_range_payload(sampled_start, sampled_end):
                """Build safe sampled/original half-open range metadata."""
                s = max(0, min(int(sampled_start), num_sampled_frames))
                e = max(0, min(int(sampled_end), num_sampled_frames))
                start_lookup = min(s, num_sampled_frames - 1)
                end_lookup = min(e, num_sampled_frames - 1)
                return {
                    "sampled_start": int(s),
                    "sampled_end": int(e),
                    "original_start": int(sampled_indices[start_lookup]),
                    "original_end": int(sampled_indices[end_lookup]),
                }

            # Save stage frame ranges (both sampled and original indices), left close, right open.
            # For interaction-only videos, ``0-1_start_static`` is only a bootstrap
            # frame for Stage2 shape/pose init; the semantic interaction spans [0:N).
            if phase_layout_mode == LAYOUT_INTERACTION_ONLY:
                stage_ranges = {
                    "0-1_start_static": _stage_range_payload(0, start_static_end_idx),
                    "1-2_approaching": _stage_range_payload(start_static_end_idx, start_static_end_idx),
                    "2-3_interaction": _stage_range_payload(approaching_end_idx, interaction_end_idx),
                    "3-4_releasing": _stage_range_payload(interaction_end_idx, interaction_end_idx),
                    "4-5_end_static": _stage_range_payload(num_sampled_frames, num_sampled_frames),
                }
            else:
                stage_ranges = {
                    "0-1_start_static": _stage_range_payload(0, start_static_end_idx),
                    "1-2_approaching": _stage_range_payload(start_static_end_idx, approaching_end_idx),
                    "2-3_interaction": _stage_range_payload(approaching_end_idx, interaction_end_idx),
                    "3-4_releasing": _stage_range_payload(interaction_end_idx, end_static_start_idx),
                    "4-5_end_static": _stage_range_payload(end_static_start_idx, num_sampled_frames),
                }

            stage_ranges_path = os.path.join(output_path, 'stage_frame_ranges.json')
            with open(stage_ranges_path, 'w') as f:
                json.dump(stage_ranges, f, indent=4)
            print(f"Saved stage frame ranges to {stage_ranges_path}")

            stage_layout_debug_path = os.path.join(output_path, "stage_layout_debug.json")
            with open(stage_layout_debug_path, "w") as f:
                json.dump({
                    "layout_mode": phase_layout_mode,
                    "layout_info": phase_layout_info,
                    "stage2_bootstrap_sampled_frames": int(start_static_end_idx),
                    "semantic_interaction_half_open": [int(approaching_end_idx), int(interaction_end_idx)],
                    "note": (
                        "For interaction_only, 0-1_start_static is a one-frame bootstrap, "
                        "not a semantic static segment."
                    ),
                }, f, indent=4)
            print(f"Saved stage layout debug to {stage_layout_debug_path}")

        def _run_stage2():
            nonlocal single_frame_model, final_posed_mesh, _opt_log, initial_scale
            # ========================================================================
            # STAGE 2: Optimize 0-1 Shared ObjectPose
            # ========================================================================
            print("\n" + "=" * 80)
            print("STAGE 2: Optimizing Start Static Object Pose (0-1)")
            print("=" * 80)

            # Create batched camera for the bootstrap frames.
            # Default uses the old 0-1 static prefix; k-frame mode bootstraps on the reference frame.
            bootstrap_frame_indices = [int(sam3d_ref_sampled_idx)] if sam3d_ref_enabled else list(range(start_static_end_idx))
            if len(bootstrap_frame_indices) == 0:
                bootstrap_frame_indices = [0]
            bootstrap_frame_indices_np = np.asarray(bootstrap_frame_indices, dtype=np.int64)
            bootstrap_frame_indices_t = torch.tensor(bootstrap_frame_indices, dtype=torch.long, device=device)
            bootstrap_frame_idx = int(bootstrap_frame_indices[0])
            start_segment_size = len(bootstrap_frame_indices)
            print(f"Stage 2 bootstrap sampled slots: {bootstrap_frame_indices}")
            camera_start_segment = PerspectiveCameras(
                focal_length=focal_length[bootstrap_frame_indices_t],
                principal_point=principal_point[bootstrap_frame_indices_t],
                image_size=((H_out, W_out),),
                in_ndc=False,
                device=device,
            )

            # Prepare metric depth data for depth supervision
            sampled_metric_depths = torch.from_numpy(sampled_metric_depths_np[bootstrap_frame_indices_np, ..., 0]).float().to(device)  # (N, H, W)

            # ── Scale initialisation from 2D bounding box comparison ─────────────
            # Compare the 2D projected bounding box of the mesh at the initial pose
            # with the GT mask 2D bounding box. This avoids the 3D point cloud
            # diagonal approach, which is inflated by depth variation along Z and
            # gives unreliable scale estimates.
            try:
                _cam_f0 = PerspectiveCameras(
                    focal_length=focal_length[bootstrap_frame_idx:bootstrap_frame_idx + 1],
                    principal_point=principal_point[bootstrap_frame_idx:bootstrap_frame_idx + 1],
                    image_size=((H_out, W_out),),
                    in_ndc=False,
                    device=device,
                )
                _s0 = initial_scale  # 0-dim scalar tensor
                # Apply initial transform: scale → rotate (row-major) → translate
                with torch.no_grad():
                    _v_world = verts * _s0  # scale
                    _v_world = _v_world @ initial_R  # rotate (row-vector convention)
                    _v_world = _v_world + initial_T.unsqueeze(0)  # translate
                    # Keep only vertices in front of camera (Z > 0)
                    _in_front = _v_world[:, 2] > 0
                    if _in_front.sum() > 10:
                        _v_front = _v_world[_in_front]
                        _v_screen = _cam_f0.transform_points_screen(
                            _v_front.unsqueeze(0), image_size=((H_out, W_out),)
                        )[0, :, :2].cpu().numpy()
                        _v_screen = _v_screen[np.isfinite(_v_screen).all(axis=1)]
                        _mesh_w = float(_v_screen[:, 0].max() - _v_screen[:, 0].min())
                        _mesh_h = float(_v_screen[:, 1].max() - _v_screen[:, 1].min())
                        _mesh_bbox_diag = np.sqrt(_mesh_w ** 2 + _mesh_h ** 2)

                        _fm = sampled_modal_masks_np[bootstrap_frame_idx].astype(bool)
                        _gt_ys, _gt_xs = np.where(_fm)
                        if len(_gt_ys) > 10 and _mesh_bbox_diag > 0:
                            _gt_w = float(_gt_xs.max() - _gt_xs.min())
                            _gt_h = float(_gt_ys.max() - _gt_ys.min())
                            _gt_bbox_diag = np.sqrt(_gt_w ** 2 + _gt_h ** 2)
                            _scale_factor = float(_gt_bbox_diag / _mesh_bbox_diag)
                            _scale_factor = float(np.clip(_scale_factor, 0.2, 5.0))
                            initial_scale = torch.tensor(
                                _s0.item() * _scale_factor, dtype=torch.float32, device=device
                            ).repeat(3)
                            print(f"Scale init from 2D bbox: mesh_diag={_mesh_bbox_diag:.1f}px, "
                                  f"gt_diag={_gt_bbox_diag:.1f}px, scale_factor={_scale_factor:.3f}")
                        else:
                            initial_scale = torch.tensor(initial_scale, dtype=torch.float32, device=device).repeat(3)
                    else:
                        initial_scale = torch.tensor(initial_scale, dtype=torch.float32, device=device).repeat(3)
            except Exception as _e:
                print(f"Scale init from 2D bbox failed ({_e}), using SAM3D scale.")
                initial_scale = torch.tensor(initial_scale, dtype=torch.float32, device=device).repeat(3)

            # Optimize object scale and shared pose for 0-1 segment
            # initial_scale is now shape (3,), either from point cloud or SAM3D fallback
            single_frame_model = SingleObjectPose(initial_R, initial_T, initial_scale, verts, faces).to(device)
            optimizer = torch.optim.Adam([single_frame_model.rot_6d, single_frame_model.scale, single_frame_model.trans], lr=1e-3)

            # --- Differentiable Rendering Setup ---
            raster_settings = RasterizationSettings(image_size=(H_out, W_out), blur_radius=1e-4, faces_per_pixel=20)
            renderer = MeshRenderer(
                rasterizer=MeshRasterizer(cameras=camera_start_segment, raster_settings=raster_settings),
                shader=SoftSilhouetteShader(),
            )

            # ── Open optimize.log (created/overwritten here; Stage 8 appends later) ──
            _log_path = os.path.join(output_path, "optimize_stage1_3.log")
            _opt_log = open(_log_path, "w", buffering=1)
            # Header label matches code STAGE 2 (was historically "Stage 1" — kept
            # the file name optimize_stage1_3.log for backward-compat with old runs,
            # but the section labels now follow code STAGE numbering).
            _opt_log.write("# Stage 2 — Start-static single-frame object optimisation\n")
            _opt_log.write("step,total,l2,l2_raw,depth,depth_raw,scale\n")

            # EMA auto-balancing for Stage 1 (same mechanism as Stage 3)
            # _s1_ema_init stores the first-step value so the denominator never shrinks
            # below it, preventing the "loss decreases but normalized value inflates" artifact.
            _s1_ema_decay = 0.99
            _s1_ema = {}
            _s1_ema_init = {}
            _s1_eps = 1e-8
            _s1_w_rel = {'l2': 1.0, 'depth': 1.0}

            def _s1_ema_norm(name, raw_loss):
                val = raw_loss.detach().item()
                if name not in _s1_ema:
                    _s1_ema[name] = val if val > _s1_eps else 1.0
                    _s1_ema_init[name] = _s1_ema[name]
                else:
                    _s1_ema[name] = _s1_ema_decay * _s1_ema[name] + (1 - _s1_ema_decay) * val
                denom = max(_s1_ema[name], _s1_ema_init[name]) + _s1_eps
                return raw_loss / denom * _s1_w_rel.get(name, 1.0)

            # --- Optimization Loop for 0-1 Segment ---
            loop = tqdm(range(500), desc=f"Optimizing Stage2 bootstrap frames {bootstrap_frame_indices}")
            for step in loop:
                optimizer.zero_grad()

                # 同一个pose应用到0-1段的所有帧
                posed_mesh = single_frame_model()

                # 复制mesh到batch
                posed_verts = posed_mesh.verts_list()[0]  # (V, 3)
                posed_faces = posed_mesh.faces_list()[0]  # (F, 3)
                posed_verts_batch = posed_verts.unsqueeze(0).repeat(start_segment_size, 1, 1)  # (N, V, 3)
                posed_faces_batch = posed_faces.unsqueeze(0).repeat(start_segment_size, 1, 1)  # (N, F, 3)
                posed_textures = TexturesVertex(verts_features=torch.ones_like(posed_verts_batch))
                posed_mesh_batch = Meshes(verts=posed_verts_batch, faces=posed_faces_batch, textures=posed_textures)

                # Render all frames in 0-1 segment
                fragments = renderer.rasterizer(posed_mesh_batch, cameras=camera_start_segment)
                rendered_masks = renderer.shader(fragments, posed_mesh_batch, cameras=camera_start_segment)[..., 3]  # (N, H, W)

                rendered_depth = fragments.zbuf[..., 0]  # (N, H, W)

                bootstrap_modal_masks = sampled_modal_masks[bootstrap_frame_indices_t].float()
                depth_loss_mask = (rendered_depth > 0) & (rendered_masks > 0.5) & (bootstrap_modal_masks > 0.5)

                l2_loss_raw = torch.nn.functional.mse_loss(rendered_masks, bootstrap_modal_masks)

                depth_loss_raw = torch.tensor(0.0, device=device)
                if depth_loss_mask.sum() > 0:
                    rendered_depth_masked = rendered_depth[depth_loss_mask]
                    metric_depth_masked = sampled_metric_depths[depth_loss_mask]
                    depth_loss_raw = torch.nn.functional.mse_loss(rendered_depth_masked, metric_depth_masked)

                l2_loss = _s1_ema_norm('l2', l2_loss_raw)
                depth_loss = _s1_ema_norm('depth', depth_loss_raw)

                total_loss = l2_loss + depth_loss

                total_loss.backward()
                optimizer.step()
                loop.set_postfix(loss=total_loss.item(), l2=l2_loss.item(), l2_r=l2_loss_raw.item(), depth=depth_loss.item(), depth_r=depth_loss_raw.item(), scale=single_frame_model.scale.item())
                if step % 20 == 0 or step == 499:
                    _opt_log.write(f"{step},{total_loss.item():.6f},"
                                   f"{l2_loss.item():.6f},{l2_loss_raw.item():.6f},"
                                   f"{depth_loss.item():.6f},{depth_loss_raw.item():.6f},"
                                   f"{single_frame_model.scale.item():.6f}\n")

            # ICP alignment
            first_frame_idx = bootstrap_frame_idx
            camera_first_frame = PerspectiveCameras(
                focal_length=focal_length[first_frame_idx, None],
                principal_point=principal_point[first_frame_idx, None],
                image_size=((H_out, W_out),),
                in_ndc=False,
                device=device,
            )
            # Forward pass for first frame
            first_R = rotation_6d_to_matrix(single_frame_model.rot_6d)
            first_scale_full = single_frame_model.scale * single_frame_model.initial_scale
            first_transform = Transform3d(dtype=torch.float32, device=device).scale(first_scale_full.unsqueeze(0)).rotate(first_R.unsqueeze(0)).translate(
                single_frame_model.trans.unsqueeze(0))
            first_verts = first_transform.transform_points(verts.unsqueeze(0))
            first_mesh = Meshes(verts=first_verts, faces=faces.unsqueeze(0), textures=TexturesVertex(verts_features=torch.ones_like(first_verts)))

            first_fragments = renderer.rasterizer(first_mesh, cameras=camera_first_frame)
            first_rendered = renderer.shader(first_fragments, first_mesh, cameras=camera_first_frame)[0, ..., 3]

            # Metric depth loss for first
            first_depth = first_fragments.zbuf[0, ..., 0]  # (H, W)
            metric_depth_first = torch.from_numpy(sampled_metric_depths_np[first_frame_idx, ..., 0]).float().to(device)  # (H, W)

            first_target_mask = sampled_modal_masks[first_frame_idx].unsqueeze(0).unsqueeze(-1).permute(0, 3, 1, 2)
            first_target_points = get_rgbd_point_cloud(camera_first_frame, sampled_rgbs[first_frame_idx].unsqueeze(0).permute(0, 3, 1, 2),
                                                       metric_depth_first.unsqueeze(0).unsqueeze(-1).permute(0, 3, 1, 2), first_target_mask, 0.5)
            first_target_points = first_target_points.points_packed()
            first_target_points = first_target_points.squeeze(0)  # (N, 3)

            first_source_mask = first_rendered.unsqueeze(0).unsqueeze(-1).permute(0, 3, 1, 2)
            first_source_points = get_rgbd_point_cloud(camera_first_frame, sampled_rgbs[first_frame_idx].unsqueeze(0).permute(0, 3, 1, 2),
                                                       first_depth.unsqueeze(0).unsqueeze(-1).permute(0, 3, 1, 2), first_source_mask, 0.5)
            first_source_points = first_source_points.points_packed()
            first_source_points = first_source_points.squeeze(0)  # (N, 3)

            if first_target_points is not None and first_target_points.shape[0] > 10:
                estimate_scale = False
                # ICP needs batch dimension
                first_icp_result = iterative_closest_point(
                    first_source_points.unsqueeze(0),  # X
                    first_target_points.unsqueeze(0),  # Y
                    init_transform=None,  # init transform
                    max_iterations=50,
                    relative_rmse_thr=1e-6,
                    estimate_scale=estimate_scale,  # allow ICP to change scale?
                    allow_reflection=False,
                )
                # result.RT 是一个 (1, 4, 4) 矩阵，表示将 Source 对齐到 Target 的变换
                # P_target ~ R_icp * P_source + T_icp
                icp_transform = first_icp_result.RTs
                R_delta = icp_transform[0]  # (1, 3, 3)
                T_delta = icp_transform[1]  # (1, 3) translation part usually matches points translation
                if estimate_scale:
                    s_delta = icp_transform[2]  # (1,) scale part
                else:
                    s_delta = torch.tensor([1.0], device=device)

                if save_intermediates:
                    # Save pre-alignment point cloud with mesh
                    # Target = Green, Source = Red, Pre-aligned Mesh = Blue
                    pre_combined_points = torch.cat([first_target_points, first_source_points, first_verts[0]], dim=0).detach().cpu().numpy()

                    pre_target_colors = torch.tensor([[0.0, 1.0, 0.0]], device=device).repeat(first_target_points.shape[0], 1)
                    pre_source_colors = torch.tensor([[1.0, 0.0, 0.0]], device=device).repeat(first_source_points.shape[0], 1)
                    pre_mesh_colors = torch.tensor([[0.0, 0.0, 1.0]], device=device).repeat(first_verts[0].shape[0], 1)
                    pre_combined_colors = (torch.cat([pre_target_colors, pre_source_colors, pre_mesh_colors], dim=0).cpu().numpy() * 255).astype(np.uint8)

                    pre_pcd = trimesh.points.PointCloud(pre_combined_points, colors=pre_combined_colors)
                    pre_debug_ply_path = os.path.join(output_path, f"stage1_debug_icp_pre_alignment_with_mesh_{first_frame_idx}.ply")
                    pre_pcd.export(pre_debug_ply_path)
                    print(f"Saved ICP pre-alignment point cloud with mesh to {pre_debug_ply_path}")

                aligned_points = s_delta * (first_source_points @ R_delta[0]) + T_delta
                aliged_mesh_verts = s_delta * (first_verts[0] @ R_delta[0]) + T_delta
                if save_intermediates:
                    # Save post-alignment point cloud with mesh
                    # Target = Green, Aligned Source = Red, Aligned Mesh = Blue
                    combined_points = torch.cat([first_target_points, aligned_points, aliged_mesh_verts], dim=0).detach().cpu().numpy()

                    target_colors = torch.tensor([[0.0, 1.0, 0.0]], device=device).repeat(first_target_points.shape[0], 1)
                    source_colors = torch.tensor([[1.0, 0.0, 0.0]], device=device).repeat(first_source_points.shape[0], 1)
                    aliged_mesh_colors = torch.tensor([[0.0, 0.0, 1.0]], device=device).repeat(aliged_mesh_verts.shape[0], 1)
                    combined_colors = (torch.cat([target_colors, source_colors, aliged_mesh_colors], dim=0).cpu().numpy() * 255).astype(np.uint8)

                    pcd = trimesh.points.PointCloud(combined_points, colors=combined_colors)
                    debug_ply_path = os.path.join(output_path, f"stage1_debug_icp_post_alignment_{first_frame_idx}.ply")
                    pcd.export(debug_ply_path)
                    print(f"Saved ICP post-alignment point cloud to {debug_ply_path}")

                # Check if scale change is reasonable
                # If s_delta is too large or too small, metric depth is unreliable
                scale_threshold_min = 0.7
                scale_threshold_max = 1.3
                s_delta_value = s_delta.item() if isinstance(s_delta, torch.Tensor) else s_delta

                if scale_threshold_min <= s_delta_value <= scale_threshold_max:
                    # PyTorch3D ICP 返回的 RT
                    new_R_matrix = first_R @ R_delta.squeeze(0)
                    new_T_vector = s_delta * (single_frame_model.trans.unsqueeze(0) @ R_delta.squeeze(0)).squeeze(0) + T_delta

                    # Check IoU before applying ICP: render mask with new transform and compare with amodal mask
                    with torch.no_grad():
                        # Temporarily apply the new transform to test IoU
                        temp_scale_full = s_delta * single_frame_model.scale * single_frame_model.initial_scale
                        temp_transform = Transform3d(dtype=torch.float32,
                                                     device=device).scale(temp_scale_full.unsqueeze(0)).rotate(new_R_matrix.unsqueeze(0)).translate(new_T_vector)
                        temp_verts = temp_transform.transform_points(verts.unsqueeze(0))
                        temp_mesh = Meshes(verts=temp_verts, faces=faces.unsqueeze(0), textures=TexturesVertex(verts_features=torch.ones_like(temp_verts)))

                        # Render with new transform
                        temp_fragments = renderer.rasterizer(temp_mesh, cameras=camera_first_frame)
                        temp_rendered_mask = renderer.shader(temp_fragments, temp_mesh, cameras=camera_first_frame)[0, ..., 3]  # (H, W)

                        # Get amodal mask for first frame
                        amodal_mask_first = sampled_pred_amodal_masks[first_frame_idx]  # (H, W)

                        # Compute IoU
                        rendered_binary = (temp_rendered_mask > 0.5).float()
                        amodal_binary = (amodal_mask_first > 0.5).float()
                        intersection = (rendered_binary * amodal_binary).sum()
                        union = (rendered_binary + amodal_binary).clamp(0, 1).sum()
                        iou = (intersection / union) if union > 0 else torch.tensor(0.0, device=device)
                        iou_value = iou.item()

                    # Only apply ICP if IoU >= 0.8
                    if iou_value >= 0.8:
                        # assign back to parameter
                        with torch.no_grad():
                            single_frame_model.rot_6d.data = matrix_to_rotation_6d(new_R_matrix).squeeze(0)
                            single_frame_model.trans.data = new_T_vector.squeeze(0)
                            single_frame_model.scale.data = s_delta * single_frame_model.scale.data

                        print(f" ICP Alignment Applied to First frame! (scale delta: {s_delta_value:.4f}, IoU: {iou_value:.4f})")
                        # Recreate optimizer after ICP alignment to reset momentum state
                        optimizer = torch.optim.Adam([single_frame_model.rot_6d, single_frame_model.scale, single_frame_model.trans], lr=1e-3)
                    else:
                        print(f" Warning: ICP alignment rejected due to low IoU ({iou_value:.4f} < 0.8). Keeping original pose.")
                else:
                    print(f" Warning: ICP scale change too large ({s_delta_value:.4f}), metric depth may be unreliable.")
                    print(f" Skipping ICP alignment. Acceptable range: [{scale_threshold_min}, {scale_threshold_max}]")

            # --- Visualization for 0-1 Segment ---
            with torch.no_grad():
                final_posed_mesh = single_frame_model()
                posed_verts = final_posed_mesh.verts_list()[0]
                posed_faces = final_posed_mesh.faces_list()[0]
                posed_verts_batch = posed_verts.unsqueeze(0).repeat(start_segment_size, 1, 1)
                posed_faces_batch = posed_faces.unsqueeze(0).repeat(start_segment_size, 1, 1)
                posed_textures = TexturesVertex(verts_features=torch.ones_like(posed_verts_batch))
                posed_mesh_batch = Meshes(verts=posed_verts_batch, faces=posed_faces_batch, textures=posed_textures)

                # Render silhouette masks
                fragments = renderer.rasterizer(posed_mesh_batch, cameras=camera_start_segment)
                final_masks = renderer.shader(fragments, posed_mesh_batch, cameras=camera_start_segment)[..., 3].cpu().numpy()
                final_depths = fragments.zbuf[..., 0].cpu().numpy()  # (N, H, W)

                # Render RGB with lighting for better geometry visualization
                lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])
                rgb_textures = TexturesVertex(verts_features=torch.ones_like(posed_verts_batch) * torch.tensor([0.7, 0.7, 1.0], device=device))
                rgb_mesh_batch = Meshes(verts=posed_verts_batch, faces=posed_faces_batch, textures=rgb_textures)

                rgb_raster_settings = RasterizationSettings(image_size=(H_out, W_out), blur_radius=0.0, faces_per_pixel=1)
                rgb_renderer = MeshRenderer(rasterizer=MeshRasterizer(cameras=camera_start_segment, raster_settings=rgb_raster_settings),
                                            shader=HardPhongShader(device=device, cameras=camera_start_segment, lights=lights))
                rendered_rgb = rgb_renderer(rgb_mesh_batch).cpu().numpy()  # (N, H, W, 4)

            # Save visualization for first and last frame of 0-1 segment
            vis_indices = list(dict.fromkeys([bootstrap_frame_idx, bootstrap_frame_indices[-1]]))
            for vis_idx in vis_indices:
                vis_local_idx = bootstrap_frame_indices.index(vis_idx) if vis_idx in bootstrap_frame_indices else 0
                modal_frame = overlay_mask_on_image(sampled_rgbs_np[vis_idx], sampled_modal_masks_np[vis_idx])
                amodal_gt_frame = overlay_mask_on_image(sampled_rgbs_np[vis_idx], sampled_pred_amodal_masks_np[vis_idx])
                render_frame = overlay_mask_on_image(sampled_rgbs_np[vis_idx], (final_masks[vis_local_idx] > 0.5).astype(np.uint8))
                hand_frame = overlay_mask_on_image(sampled_rgbs_np[vis_idx], sampled_hand_masks_np[vis_idx], cmap_idx=1)

                # Overlay rendered RGB on original image for better geometry visualization
                rgb_render = rendered_rgb[vis_local_idx, ..., :3]  # (H, W, 3)
                alpha_render = rendered_rgb[vis_local_idx, ..., 3:4]  # (H, W, 1)
                rgb_overlay = sampled_rgbs_np[vis_idx] * (1 - alpha_render) + rgb_render * alpha_render

                panels = [modal_frame, amodal_gt_frame, render_frame, hand_frame, rgb_overlay]
                combined_image = (np.hstack(panels) * 255).astype(np.uint8)
                if save_intermediates:
                    save_path = os.path.join(output_path, f"stage1_start_static_frame_{vis_idx}.png")
                    imageio.imwrite(save_path, combined_image)
                    print(f"Saved stage 1 visualization to {save_path}")

        def _run_stage3():
            nonlocal stage3_object_pose_init, multi_frame_model, renderer
            nonlocal _hamer_mano_pose_init, _hamer_mano_pose_6d_init
            nonlocal interaction_profile_stage3
            # ========================================================================
            # STAGE 3: Multi-frame joint object + hand pose optimisation
            # ------------------------------------------------------------------------
            # Two sub-steps:
            #   3a. PnP init from last frame of 0-1 segment to seed object pose
            #       across the full sequence.
            #   3b. Joint optim of object (rot_6d + trans) and hand
            #       (mano_root_orient + mano_trans + mano_pose).
            # Hand losses here: 2D joint reproj + temporal smooth + anatomy.
            # No metric-depth supervision on hand — see PIPELINE OVERVIEW above.
            # ========================================================================
            print("\n" + "=" * 80)
            print(f"STAGE 3: Running PnP from frame {start_static_end_idx - 1}")
            print("=" * 80)

            # --- Interaction-only motion type (2-3 segment; not 1-2 approaching) ---
            try:
                from interaction_motion_profile import build_interaction_motion_profile

                if (approaching_end_idx is not None and interaction_end_idx is not None
                        and interaction_end_idx > approaching_end_idx + 1):
                    _type_lo, _type_hi, _opt_lo, _opt_hi = _interaction_motion_typing_window()
                    interaction_profile_stage3 = build_interaction_motion_profile(
                        _type_lo,
                        _type_hi,
                        cotracker_tracks_fwd_np,
                        cotracker_fwd_vis_np,
                        sampled_pred_amodal_masks_np,
                        pnp_R_col_major=None,
                        sampled_mano_params=sampled_mano_params,
                    )
                    interaction_profile_stage3 = force_rotation_likely_profile(
                        interaction_profile_stage3,
                        reason="stage3_profile_force_rotation_likely",
                    )
                    interaction_profile_stage3["source"] = "stage3_start_cotracker_mask_mano"
                    interaction_profile_stage3["motion_typing_core_half_open"] = [_type_lo, _type_hi]
                    interaction_profile_stage3["optimization_interaction_padded_half_open"] = [_opt_lo, _opt_hi]
                    _imp3s = os.path.join(output_path, "interaction_motion_profile_stage3.json")
                    with open(_imp3s, "w") as _imf:
                        json.dump(interaction_profile_stage3, _imf, indent=2)
                    print(f"\n[interaction-type] Stage 3 → wrote {_imp3s}  "
                          f"suggested_mode={interaction_profile_stage3.get('suggested_mode')}")
                else:
                    print("\n[interaction-type] Stage 3: skip profile (invalid 2-3 interaction range).")
            except Exception as _e_imp:
                print(f"\n[interaction-type] Stage 3 profile failed: {_e_imp}")
                traceback.print_exc()

            def _emit_interaction_policy_diagnostics():
                """Write ``sam3d_policy_diag/policy_plan.json`` (+ console lines)."""
                prof = interaction_profile_stage3
                if prof is None:
                    return
                raw_mode = str(prof.get("suggested_mode", FORCE_ROTATION_MODE))
                original_mode = prof.get("policy_original_suggested_mode")
                if raw_mode == "translation_likely":
                    policy = "TRANSLATION_LIKE"
                elif raw_mode == "rotation_likely":
                    policy = "ROTATION_LIKE"
                else:
                    policy = "ROTATION_LIKE"
                    raw_mode = FORCE_ROTATION_MODE
                ilo = int(approaching_end_idx)
                ihi = int(interaction_end_idx)
                type_lo, type_hi, _, _ = _interaction_motion_typing_window()
                _M = max(1, int(sam3d_sparse_stride))
                _kf = list(range(ilo, ihi, _M))
                _diag_dir = os.path.join(output_path, "sam3d_policy_diag")
                os.makedirs(_diag_dir, exist_ok=True)
                ext_lo, ext_hi = ilo, ihi
                if stage1_interaction_lo_auto is not None:
                    ext_lo = min(ext_lo, int(stage1_interaction_lo_auto))
                if stage1_interaction_hi_auto is not None:
                    ext_hi = max(ext_hi, int(stage1_interaction_hi_auto))
                _st1_line = (
                    f"Stage1 object-mask motion envelope (auto 2–3, pre-pad): [{stage1_interaction_lo_auto}, {stage1_interaction_hi_auto})"
                    if stage1_interaction_lo_auto is not None and stage1_interaction_hi_auto is not None
                    else "Stage1 object-mask motion envelope: [n/a, n/a)"
                )
                lines = [
                    f"policy_from_profile={policy}  (suggested_mode={raw_mode}, original={original_mode})",
                    "policy_override=force_rotation_likely",
                    f"motion_typing_core_half_open=[{type_lo}, {type_hi})  (sampled indices)",
                    f"optimization_interaction_padded_half_open=[{ilo}, {ihi})  (sampled indices)",
                    f"TRANSLATION_LIKE: milestone SAM3D every M={_M} (indices): {_kf}",
                    _st1_line,
                    f"ROTATION_LIKE: gap=1 SAM3D from Stage1 mask-onset → rotation_dense/ (after Stage 2); "
                    f"final poses merged into Stage 3 object init (see stage3_object_pose_init).",
                ]
                if sam3d_sparse_keyframes:
                    lines.append(
                        "SAM3D: translation_likely → sam3d_sparse_keyframes/ (milestones, after Stage 1); "
                        "rotation_likely → rotation_dense/ after Stage 2."
                    )
                else:
                    lines.append("SAM3D: pass --sam3d_sparse_keyframes for policy-gated SAM3D diagnostics.")
                payload = {
                    "policy": policy,
                    "suggested_mode": raw_mode,
                    "policy_override": prof.get("policy_override"),
                    "policy_original_suggested_mode": original_mode,
                    "motion_typing_core_half_open": [type_lo, type_hi],
                    "optimization_interaction_padded_half_open": [ilo, ihi],
                    "translation_spare_keyframes_stride_M": _M,
                    "translation_spare_keyframes_sampled_indices": _kf,
                    "stage1_mask_motion_auto_half_open": [
                        int(stage1_interaction_lo_auto) if stage1_interaction_lo_auto is not None else None,
                        int(stage1_interaction_hi_auto) if stage1_interaction_hi_auto is not None else None,
                    ],
                    "rotation_like_chain_hint_half_open": [int(ext_lo), int(ext_hi)],
                    "lines": lines,
                }
                _json_path = os.path.join(_diag_dir, "policy_plan.json")
                with open(_json_path, "w") as _jf:
                    json.dump(payload, _jf, indent=2)
                print(f"[interaction-type] policy diag → {_json_path}")
                for _ln in lines:
                    print(f"  {_ln}")

            _emit_interaction_policy_diagnostics()

            # Use the old static-prefix endpoint by default; k-frame mode anchors PnP at the SAM3D reference slot.
            pnp_start_frame_idx = int(sam3d_ref_sampled_idx) if sam3d_ref_enabled else start_static_end_idx - 1
            print(
                f"STAGE 3 PnP reference sampled_slot={pnp_start_frame_idx}, "
                f"clip_frame={int(sampled_indices[pnp_start_frame_idx])}"
            )
            verts_canonical_scaled = verts * (single_frame_model.scale.detach() * single_frame_model.initial_scale).unsqueeze(0)
            mesh_canonical_scaled = Meshes(
                verts=[verts_canonical_scaled],
                faces=[faces],
                textures=TexturesVertex(verts_features=torch.ones_like(verts)[None]),
            )
            queries_2d, queries_3d = generate_queries(final_posed_mesh,
                                                      mesh_canonical_scaled,
                                                      sampled_pred_amodal_masks_np[pnp_start_frame_idx],
                                                      focal_length[pnp_start_frame_idx, None],
                                                      principal_point[pnp_start_frame_idx, None],
                                                      grid_size=15,
                                                      device=device)

            K = np.array([[fx_new.item(), 0, cx_new.item()], [0, fy_new.item(), cy_new.item()], [0, 0, 1]])

            # Convert optimized 6D rotation to rotation matrix for PnP
            optimized_R = rotation_6d_to_matrix(single_frame_model.rot_6d.detach().unsqueeze(0))[0]  # (3, 3)
            optimized_T = single_frame_model.trans.detach()  # (3,)

            # Per-sampled-frame homogeneous object pose for Stage 3 ``TemporalHandObjectPose`` init.
            # Static prefix [0:approaching_end_idx): Stage 2 single-frame R,t; tail from ``run_pnp_1stage``.
            # When ``rotation_likely`` + ``--sam3d_sparse_keyframes``, SAM3D dense (outlier-filtered + interp)
            # overwrites **rotation** on tail rows before ``initial_R_mat``; tail **translation** stays PnP +
            # ``init_relative_translation`` for ``initial_T`` (see merge helper).
            stage3_object_pose_init = torch.zeros(num_sampled_frames, 4, 4, device=device)
            # NOTE: PnP returns column-major rotation (OpenCV), but PyTorch3D uses row-major
            # So we need to store the optimized rotation in column-major format to match PnP
            shared_rot = rotation_6d_to_matrix(single_frame_model.rot_6d.detach().unsqueeze(0))  # (1, 3, 3) row-major
            shared_rot_col_major = shared_rot.transpose(1, 2)  # Convert to column-major to match PnP format
            shared_trans = single_frame_model.trans.detach()  # (3,)
            static_fill_end = num_sampled_frames if sam3d_ref_enabled else approaching_end_idx
            for i in range(static_fill_end):
                stage3_object_pose_init[i, :3, :3] = shared_rot_col_major[0]
                stage3_object_pose_init[i, :3, 3] = shared_trans
                stage3_object_pose_init[i, 3, 3] = 1.0

            # Single-shot PnP. Segmented SAM3D reset/bridge removed; see backup/sam3d_reset_bridge_archive.txt.
            # Build the silhouette renderer for Stage 3 (_render_pose / diagnostics).
            camera = PerspectiveCameras(
                focal_length=focal_length,
                principal_point=principal_point,
                image_size=((H_out, W_out),),
                in_ndc=False,
                device=device,
            )
            raster_settings = RasterizationSettings(image_size=(H_out, W_out), blur_radius=1e-4, faces_per_pixel=20)
            renderer = MeshRenderer(
                rasterizer=MeshRasterizer(cameras=camera, raster_settings=raster_settings),
                shader=SoftSilhouetteShader(blend_params=BlendParams(sigma=1e-4, gamma=1e-4)),
            )

            pnp_forward = run_pnp_1stage(
                cotracker_model,
                sampled_rgbs[pnp_start_frame_idx:],
                queries_2d,
                queries_3d,
                sampled_pred_amodal_masks_np[pnp_start_frame_idx:],
                K,
                optimized_R,
                optimized_T,
                device,
                output_dir=os.path.join(output_path, "pnp_visualization_forward" if sam3d_ref_enabled else "pnp_visualization"),
                vis_threshold=0.0)

            pnp_forward = torch.from_numpy(np.stack(pnp_forward, axis=0)).float().to(device)
            if sam3d_ref_enabled:
                stage3_object_pose_init[pnp_start_frame_idx:] = pnp_forward
                if pnp_start_frame_idx > 0:
                    pnp_backward_rev = run_pnp_1stage(
                        cotracker_model,
                        torch.flip(sampled_rgbs[:pnp_start_frame_idx + 1], dims=[0]),
                        queries_2d,
                        queries_3d,
                        np.ascontiguousarray(sampled_pred_amodal_masks_np[:pnp_start_frame_idx + 1][::-1]),
                        K,
                        optimized_R,
                        optimized_T,
                        device,
                        output_dir=os.path.join(output_path, "pnp_visualization_backward"),
                        vis_threshold=0.0)
                    pnp_backward = torch.flip(torch.from_numpy(np.stack(pnp_backward_rev, axis=0)).float().to(device), dims=[0])
                    stage3_object_pose_init[:pnp_start_frame_idx + 1] = pnp_backward
            else:
                stage3_object_pose_init[approaching_end_idx:] = pnp_forward[(approaching_end_idx - pnp_start_frame_idx):]

            merge_sam3d_rotation_snapshots_into_stage3_object_pose_init(
                stage3_object_pose_init, sam3d_rotation_pose_snapshots, device
            )

            obj_init_R_col_major = stage3_object_pose_init[:, :3, :3]  # OpenCV column-major (matches run_pnp_1stage)
            obj_init_t = stage3_object_pose_init[:, :3, 3]

            try:
                from interaction_motion_profile import build_interaction_motion_profile

                if (approaching_end_idx is not None and interaction_end_idx is not None
                        and interaction_end_idx > approaching_end_idx + 1):
                    _R_np = obj_init_R_col_major.detach().cpu().numpy()
                    _type_lo, _type_hi, _opt_lo, _opt_hi = _interaction_motion_typing_window()
                    _imp3 = build_interaction_motion_profile(
                        _type_lo,
                        _type_hi,
                        cotracker_tracks_fwd_np,
                        cotracker_fwd_vis_np,
                        sampled_pred_amodal_masks_np,
                        pnp_R_col_major=_R_np,
                        sampled_mano_params=sampled_mano_params,
                    )
                    _imp3["source"] = "post_initial_pnp_cotracker_mask_mano_pnp_R"
                    _imp3["motion_typing_core_half_open"] = [_type_lo, _type_hi]
                    _imp3["optimization_interaction_padded_half_open"] = [_opt_lo, _opt_hi]
                    if interaction_profile_stage3 is not None:
                        _imp3["stage3_suggested_mode"] = interaction_profile_stage3.get("suggested_mode")
                    _imp3_path = os.path.join(output_path, "interaction_motion_profile_post_pnp.json")
                    with open(_imp3_path, "w") as _imf3:
                        json.dump(_imp3, _imf3, indent=2)
                    print(f"[interaction-type] wrote {_imp3_path}  suggested_mode={_imp3.get('suggested_mode')}")
            except Exception as _e_imp3:
                print(f"[interaction-type] post-PnP profile failed: {_e_imp3}")
                traceback.print_exc()

            # ====================================================================
            # PnP health dry-run (optional): per-frame IoU / angle-jump vs adaptive
            # reset thresholds; writes pnp_health.csv and pnp_health.json.
            # ====================================================================
            if pnp_health_dump:
                try:
                    _scale_full = (single_frame_model.scale.detach() * single_frame_model.initial_scale).to(device)
                    _eval_start = pnp_start_frame_idx + 1  # PnP[0] is the seed pose; first PnP-derived frame is +1
                    _eval_range = range(_eval_start, num_sampled_frames)
                    _metrics = compute_pnp_health(
                        pnp_rot_col_major=obj_init_R_col_major,
                        pnp_trans=obj_init_t,
                        verts=verts,
                        faces=faces,
                        scale_full=_scale_full,
                        amodal_masks_np=sampled_pred_amodal_masks_np,
                        focal_length=focal_length,
                        principal_point=principal_point,
                        H_out=H_out,
                        W_out=W_out,
                        device=device,
                        eval_frames=_eval_range,
                    )
                    _trig_cfg = dict(
                        iou_threshold=0.5,
                        iou_relative_margin=0.15,
                        iou_drop_threshold=0.2,
                        angle_jump_threshold_deg=30.0,
                        consecutive_required=3,
                        cooldown=3,
                    )
                    _candidates, _trig_summary = flag_reset_candidates(
                        _metrics,
                        eval_start=_eval_start,
                        **_trig_cfg,
                    )

                    _csv_path = os.path.join(output_path, "pnp_health.csv")
                    with open(_csv_path, "w") as _cf:
                        _cf.write("frame,iou,angle_jump_deg,mask_area_px\n")
                        for _i in range(num_sampled_frames):
                            _iou = _metrics["iou"][_i]
                            _aj = _metrics["angle_jump_deg"][_i]
                            _ar = _metrics["mask_area_px"][_i]
                            _cf.write(f"{_i},{_iou if np.isfinite(_iou) else ''},{_aj if np.isfinite(_aj) else ''},{int(_ar)}\n")
                    _json_path = os.path.join(output_path, "pnp_health.json")
                    with open(_json_path, "w") as _jf:
                        json.dump({
                            "thresholds": _trig_cfg,
                            "trigger_summary": _trig_summary,
                            "eval_start_frame": int(_eval_start),
                            "candidates": [{"frame": int(f), "reason": r} for f, r in _candidates],
                            "summary": {
                                "n_evaluated": int(np.sum(np.isfinite(_metrics["iou"]))),
                                "iou_mean": float(np.nanmean(_metrics["iou"])),
                                "iou_min": float(np.nanmin(_metrics["iou"])) if np.any(np.isfinite(_metrics["iou"])) else None,
                                "iou_below_05_count": int(np.sum(_metrics["iou"][np.isfinite(_metrics["iou"])] < 0.5)),
                            },
                        }, _jf, indent=2)

                    print("\n" + "-" * 80)
                    print("[PnP-HEALTH] dry-run summary")
                    print(f"[PnP-HEALTH]   PnP-evaluated frames: [{_eval_start}, {num_sampled_frames}) "
                          f"({num_sampled_frames - _eval_start} frames)")
                    _ious = _metrics["iou"][_eval_start:]
                    _ious_finite = _ious[np.isfinite(_ious)]
                    if len(_ious_finite) > 0:
                        print(f"[PnP-HEALTH]   IoU stats: mean={_ious_finite.mean():.3f} "
                              f"median={np.median(_ious_finite):.3f} "
                              f"min={_ious_finite.min():.3f} "
                              f"#<0.5={int((_ious_finite < 0.5).sum())} "
                              f"#<0.3={int((_ious_finite < 0.3).sum())}")
                    print(f"[PnP-HEALTH]   Adaptive trigger: baseline_iou={_trig_summary['baseline_iou_median']:.3f}  "
                          f"eff_iou_threshold={_trig_summary['effective_iou_threshold']:.3f}  "
                          f"(absolute={_trig_cfg['iou_threshold']}, margin={_trig_cfg['iou_relative_margin']})")
                    _ajs = _metrics["angle_jump_deg"][_eval_start:]
                    _ajs_finite = _ajs[np.isfinite(_ajs)]
                    if len(_ajs_finite) > 0:
                        print(f"[PnP-HEALTH]   Angle jump stats: mean={_ajs_finite.mean():.2f} deg "
                              f"max={_ajs_finite.max():.2f} deg "
                              f"#>=30={int((_ajs_finite >= 30.0).sum())}")
                    if len(_candidates) == 0:
                        print("[PnP-HEALTH]   No reset candidates flagged. (PnP appears healthy under adaptive threshold.)")
                    else:
                        print(f"[PnP-HEALTH]   {len(_candidates)} reset candidate(s) [DRY-RUN, not applied]:")
                        for _f, _r in _candidates:
                            print(f"[PnP-HEALTH]     frame {_f:4d}  iou={_metrics['iou'][_f]:.3f}  reason: {_r}")
                    print(f"[PnP-HEALTH]   Per-frame metrics dumped to:")
                    print(f"[PnP-HEALTH]     {_csv_path}")
                    print(f"[PnP-HEALTH]     {_json_path}")
                    print("-" * 80 + "\n")
                except Exception as _e:
                    print(f"[PnP-HEALTH] monitor failed: {_e}")
                    traceback.print_exc()

            

            # --- 5. Optimization Loop: Stage 3 (Global) ---
            # Initialize poses for the whole sequence
            initial_R_mat = single_frame_model.rot_6d.detach().unsqueeze(0)
            initial_R_mat = rotation_6d_to_matrix(initial_R_mat).repeat(num_sampled_frames, 1, 1)
            initial_R_mat = obj_init_R_col_major.mT  # row-major for verts @ R in TemporalHandObjectPose

            # calculate the initial translation for the whole sequence
            initial_T = init_relative_translation(single_frame_model.trans.cpu().detach().numpy(), sampled_pred_amodal_masks_np, fx_new.item(), fy_new.item())
            initial_T = torch.from_numpy(initial_T).float().to(device)

            # recover the first approaching_end_idx frames with the optimized static translation
            if sam3d_ref_enabled:
                initial_T[pnp_start_frame_idx] = single_frame_model.trans.detach().clone()
            else:
                initial_T[:approaching_end_idx] = single_frame_model.trans.detach().clone().unsqueeze(0).repeat(approaching_end_idx, 1)
            # Tail translation: keep ``init_relative_translation`` (do not take SAM3D t — SAM3D translation is often unreliable).

            initial_scale = single_frame_model.scale.detach() * single_frame_model.initial_scale

            # Prepare mesh data for the batch
            verts_batch = verts.unsqueeze(0).repeat(num_sampled_frames, 1, 1)
            faces_batch = faces.unsqueeze(0).repeat(num_sampled_frames, 1, 1)

            # --- Model and Optimizer ---
            _stage3_video_id = os.path.basename(seq_path)
            _hold_stage3 = is_hold_video_id(_stage3_video_id)
            multi_frame_model = TemporalHandObjectPose(initial_R_mat, initial_T, initial_scale, verts_batch, faces_batch, sampled_mano_params).to(device)
            if _hold_stage3:
                _hold_init_path = hold_mano_init_path(seq_path)
                if not os.path.exists(_hold_init_path):
                    raise FileNotFoundError(f"Missing hold MANO initialization: {_hold_init_path}")

                _hold_fit = np.load(_hold_init_path, allow_pickle=True).item()["right"]
                _hold_fit_len = int(np.asarray(_hold_fit["global_orient"]).shape[0])
                _sampled_indices_np = np.asarray(sampled_indices, dtype=np.int64)
                if _hold_fit_len == num_sampled_frames:
                    _hold_select = slice(None)
                elif _sampled_indices_np.size > 0 and int(_sampled_indices_np.max()) < _hold_fit_len:
                    _hold_select = _sampled_indices_np
                else:
                    raise ValueError(
                        f"Cannot align {_hold_init_path} length {_hold_fit_len} "
                        f"with {num_sampled_frames} sampled frames"
                    )

                _mano_mean_path = "../../stage1_preprocess/Dyn_HaMR_new/_DATA/data/mano/MANO_RIGHT.pkl"
                with open(_mano_mean_path, "rb") as _mano_file:
                    _mano_data = pickle.load(_mano_file, encoding="latin1")
                _flat_hand_mean = hold_pose_hand_mean(
                    torch.from_numpy(_mano_data["hands_mean"]).float().to(device)
                )

                with torch.no_grad():
                    _hold_pose_init = torch.from_numpy(
                        np.asarray(_hold_fit["hand_pose"])[_hold_select]
                    ).float().to(device)
                    multi_frame_model.mano_pose.data = hold_pose_from_fit(
                        _hold_pose_init,
                        _flat_hand_mean,
                    )
                print(
                    f"Stage 3 hold MANO init: loaded {_hold_init_path}; "
                    f"overrode {hold_mano_init_param_groups()} only"
                )

            # ── Stage 3 pre-optimisation snapshot (all sampled frames) ────────────
            # Rendered immediately after multi_frame_model is initialised (object R from
            # stage3_object_pose_init / SAM3D merge when rotation_likely; t from init_relative_translation; HaMeR hands),
            # before any gradient step in Stage 3.
            # Layout matches optimized_fitting_SPARSE_final.mp4:
            #   top-left: amodal | top-right: camera-view
            #   bottom-left: side  | bottom-right: top
            _s3pre_video_path = os.path.join(output_path, 'stage3_pre_optim.mp4')
            print(f"\nRendering Stage 3 pre-optimisation snapshot ({num_sampled_frames} frames) → {_s3pre_video_path}")
            with torch.no_grad():
                _s3pre_obj_m, _s3pre_hand_m, _ = multi_frame_model()

                _s3pre_obj_blue  = torch.tensor([0.65, 0.8, 1.0], device=device)
                _s3pre_hand_red  = torch.tensor([1.0,  0.0, 0.0],  device=device)
                N, V = _s3pre_obj_m.verts_padded().shape[:2]
                _s3pre_obj_m.textures  = TexturesVertex(verts_features=_s3pre_obj_blue.view(1, 1, 3).expand(N, V, -1))
                N, V = _s3pre_hand_m.verts_padded().shape[:2]
                _s3pre_hand_m.textures = TexturesVertex(verts_features=_s3pre_hand_red.view(1, 1, 3).expand(N, V, -1))

                _s3pre_sf_cam = PerspectiveCameras(
                    focal_length=focal_length,
                    principal_point=principal_point,
                    image_size=((H_out, W_out),) * num_sampled_frames,
                    in_ndc=False,
                    device=device,
                )
                _s3pre_lights_v = PointLights(device=device, location=[[0.0, 0.0, -3.0]])
                _s3pre_raster_v = RasterizationSettings(image_size=(H_out, W_out), blur_radius=0.0, faces_per_pixel=1)
                _s3pre_renderer_v = MeshRenderer(
                    rasterizer=MeshRasterizer(cameras=_s3pre_sf_cam, raster_settings=_s3pre_raster_v),
                    shader=HardPhongShader(device=device, cameras=_s3pre_sf_cam, lights=_s3pre_lights_v),
                )

                _s3pre_scenes = [join_meshes_as_scene([_s3pre_obj_m[j], _s3pre_hand_m[j]]) for j in range(num_sampled_frames)]
                _s3pre_cam_rgba = _s3pre_renderer_v(join_meshes_as_batch(_s3pre_scenes)).cpu().numpy()  # (N, H, W, 4)

                _s3pre_ov  = _s3pre_obj_m.verts_padded()
                _s3pre_hv  = _s3pre_hand_m.verts_padded()
                _s3pre_ctr = torch.cat([_s3pre_ov, _s3pre_hv], dim=1).reshape(-1, 3).mean(dim=0)

                def _s3pre_rot_side(v):
                    w = v - _s3pre_ctr
                    return torch.stack([w[..., 2], w[..., 1], -w[..., 0]], dim=-1) + _s3pre_ctr

                def _s3pre_rot_top(v):
                    w = v - _s3pre_ctr
                    return torch.stack([w[..., 0], w[..., 2], -w[..., 1]], dim=-1) + _s3pre_ctr

                _s3pre_side_obj  = Meshes(verts=list(_s3pre_rot_side(_s3pre_ov)),  faces=_s3pre_obj_m.faces_list())
                _s3pre_side_hand = Meshes(verts=list(_s3pre_rot_side(_s3pre_hv)),  faces=_s3pre_hand_m.faces_list())
                _s3pre_top_obj   = Meshes(verts=list(_s3pre_rot_top(_s3pre_ov)),   faces=_s3pre_obj_m.faces_list())
                _s3pre_top_hand  = Meshes(verts=list(_s3pre_rot_top(_s3pre_hv)),   faces=_s3pre_hand_m.faces_list())
                N, V = _s3pre_side_obj.verts_padded().shape[:2]
                _s3pre_side_obj.textures  = TexturesVertex(verts_features=_s3pre_obj_blue.view(1, 1, 3).expand(N, V, -1))
                N, V = _s3pre_side_hand.verts_padded().shape[:2]
                _s3pre_side_hand.textures = TexturesVertex(verts_features=_s3pre_hand_red.view(1, 1, 3).expand(N, V, -1))
                N, V = _s3pre_top_obj.verts_padded().shape[:2]
                _s3pre_top_obj.textures   = TexturesVertex(verts_features=_s3pre_obj_blue.view(1, 1, 3).expand(N, V, -1))
                N, V = _s3pre_top_hand.verts_padded().shape[:2]
                _s3pre_top_hand.textures  = TexturesVertex(verts_features=_s3pre_hand_red.view(1, 1, 3).expand(N, V, -1))

                _s3pre_side_scenes = [join_meshes_as_scene([_s3pre_side_obj[j], _s3pre_side_hand[j]]) for j in range(num_sampled_frames)]
                _s3pre_top_scenes  = [join_meshes_as_scene([_s3pre_top_obj[j],  _s3pre_top_hand[j]])  for j in range(num_sampled_frames)]
                _s3pre_side_rgba = _s3pre_renderer_v(join_meshes_as_batch(_s3pre_side_scenes)).cpu().numpy()
                _s3pre_top_rgba  = _s3pre_renderer_v(join_meshes_as_batch(_s3pre_top_scenes)).cpu().numpy()

            if save_intermediates:
                _s3pre_writer = imageio.get_writer(
                    _s3pre_video_path, fps=30, codec='libx264', pixelformat='yuv420p',
                    ffmpeg_params=['-crf', '28', '-preset', 'veryfast'], macro_block_size=None,
                )
                for _s3pre_i in range(num_sampled_frames):
                    _s3pre_amodal = overlay_mask_on_image(sampled_rgbs_np[_s3pre_i], sampled_pred_amodal_masks_np[_s3pre_i])
                    _s3pre_alpha  = _s3pre_cam_rgba[_s3pre_i, ..., 3:4]
                    _s3pre_cam_f  = np.clip(
                        sampled_rgbs_np[_s3pre_i] * (1 - _s3pre_alpha) + _s3pre_cam_rgba[_s3pre_i, ..., :3] * _s3pre_alpha,
                        0, 1,
                    )
                    _s3pre_side_f = np.clip(_s3pre_side_rgba[_s3pre_i, ..., :3], 0, 1)
                    _s3pre_top_f  = np.clip(_s3pre_top_rgba[_s3pre_i,  ..., :3], 0, 1)
                    _s3pre_frame = (np.vstack([np.hstack([_s3pre_amodal, _s3pre_cam_f]),
                                               np.hstack([_s3pre_side_f, _s3pre_top_f])]) * 255).astype(np.uint8)
                    _s3pre_label, _s3pre_color = _sampled_video_label(_s3pre_i)
                    _s3pre_writer.append_data(_draw_video_label(_s3pre_frame, _s3pre_label, _s3pre_color))
                _s3pre_writer.close()
                print(f"Saved Stage 3 pre-optimisation snapshot → {_s3pre_video_path}")

            # ── HaMeR mano_pose prior (used by Stages 3, 5, 6) ───────────────────
            # The 2D-keypoint detector is unreliable in heavy occlusion / motion
            # blur and can collapse fingers to a single point.  HaMeR's predicted
            # mano_pose, by contrast, is a 3D regression with built-in MANO prior
            # — generally robust on articulation even when 2D fails.  We snapshot
            # the initial mano_pose right after model creation (which copies from
            # `sampled_mano_params['pose']` = HaMeR output) and use it as a soft
            # anchor in all subsequent hand-optimisation stages.
            #
            # IMPORTANT: anchor / smoothness / accel on mano_pose are all
            # computed in 6D rotation space, NOT axis-angle.  Axis-angle MSE
            # has three subtle failure modes:
            #   (1) 2π multivaluedness: (θ, n̂) and (θ - 2π, n̂) are the same
            #       rotation but differ by 2π in r = θ·n̂; if any joint angle
            #       drifts past ±π (thumb base, tight grasp can hit ~150°),
            #       MSE explodes / sign-flips.
            #   (2) zero-rotation degeneracy: r ≈ 0 has undefined direction,
            #       MSE can give noisy gradients.
            #   (3) non-isometric: euclidean distance in r-space ≠ true
            #       angular distance.
            # 6D representation (R-matrix first two columns) is continuous,
            # rotation-invariant, and free of the 2π wrap problem — same fix
            # used for wrist_anchor.  The conversion helper `mano_pose_to_6d`
            # lives at module scope so it is reachable from both the ckpt-load
            # branch (Stages 5/6 only) and this branch (Stages 3/5/6).
            _hamer_mano_pose_init = multi_frame_model.mano_pose.detach().clone()
            _hamer_mano_pose_6d_init = mano_pose_to_6d(_hamer_mano_pose_init).detach()
            # Stage 3 weight: 6D MSE on small Δθ has roughly the same magnitude
            # as axis-angle MSE (both ~Δθ²/3 for small angles), so 1e0 stays a
            # reasonable starting point.  Inspect optimize_stage3.log
            # mano_pose_anchor column to retune if needed.
            _LAMBDA_MANO_POSE_ANCHOR_S3 = 1e0

            # No hard anchor on static frames in Stage 3: silhouette (fp/vp) +
            # static_tie + smoothness already keep static segments stable and
            # well-fit. Anchoring to Stage 1/2 z creates a discontinuity at the
            # release/end-static boundary (release drifts freely, end-static is
            # pinned to old z), causing visible z-jump. Since absolute z does
            # not matter (Stage 5 contact handles relative position; ray-scale
            # alignment slides the whole sequence afterwards), we let Stage 3
            # produce a globally smooth pose sequence without anchors.
            print(f"\nStage 3: no static-frame anchor (silhouette + static_tie + smoothness only)")

            _stage3_rot_protect_requested = (
                sam3d_sparse_keyframes and sam3d_gate_suggested_mode == "rotation_likely"
            )
            # Direct rot6d optimization: SAM3D remains the initialization, but
            # Stage 3 can repair bad per-frame rotations through silhouette and
            # pose-guiding losses. Rotation smoothness stays disabled below.
            _stage3_rot_protect = False
            _stage3_rot6d_anchor = multi_frame_model.rot_6d.detach().clone()
            _stage3_rotR_anchor = rotation_6d_to_matrix(_stage3_rot6d_anchor).detach()
            _STAGE3_ROT_FREEZE_FRAC = 0.65
            _STAGE3_ROT_UNLOCK_LR_SCALE = 0.05
            _LAMBDA_STAGE3_ROT_ANCHOR = 0.0
            _STAGE3_ROT_SMOOTH_SCALE = stage3_rotation_smoothness_scale()
            _STAGE3_OBJECT_SMOOTH_SCALE = stage3_object_smoothness_scale(_stage3_video_id)
            if _stage3_rot_protect_requested:
                _unlock_step = int(round(num_steps * _STAGE3_ROT_FREEZE_FRAC))
                print(
                    "\nStage 3 rotation protection requested "
                    f"(mode={sam3d_gate_suggested_mode}) but disabled: "
                    f"rot6d optimizes from step 0 at lr={lr:.3e}, "
                    f"rot smooth scale={_STAGE3_ROT_SMOOTH_SCALE:.3g}, rot anchor=0"
                )
            if _STAGE3_OBJECT_SMOOTH_SCALE <= 0.0:
                print("Stage 3 hold object policy: object rot/trans smoothness disabled")

            obj_optimizer = torch.optim.Adam([
                {
                    "params": [multi_frame_model.rot_6d],
                    "lr": lr,
                },
                {"params": [multi_frame_model.trans], "lr": lr},
            ])
            hand_optimizer = torch.optim.Adam([multi_frame_model.mano_root_orient, multi_frame_model.mano_trans, multi_frame_model.mano_pose], lr=lr)

            # Stage 3 section in log
            _opt_log.write("\n# Stage 3 — Multi-frame pose sequence optimisation\n")
            _opt_log.write("step,total,obj_total,fp,vp,fp_raw,vp_raw,sm_rot,sm_trans,static_tie,"
                           "rot_anchor,rot_lr,rot_drift_mean_deg,rot_drift_max_deg,"
                           "hand_total,joints_2d,joints_2d_raw,sm_hand,anatomy,hand_z,hand_z_raw,n_hz,"
                           "mano_pose_anchor\n")

            # ── Hand root-z anchor target (median metric depth in hand mask) ─────
            # Resolves the perspective ambiguity where HaMeR pushes the hand to a
            # bad z and STAGE 3's 2D-only joint loss cannot pull it back (joints
            # collapse to a 2D cluster, MSE plateaus).  Per frame we take the
            # median metric depth inside the hand modal mask (robust to mask edge
            # leakage / depth holes); during optim we MSE this scalar against the
            # MANO root-joint z (joint 0 = wrist) in camera frame.
            # Frames with too-few hand-mask pixels are marked invalid and skipped.
            _MIN_HAND_MASK_PIX = 50
            _hand_z_target_list = []
            _hand_z_valid_list = []
            for _t in range(num_sampled_frames):
                _hm = sampled_hand_masks_np[_t].astype(bool)            # (H, W)
                _md = sampled_metric_depths_np[_t, ..., 0]              # (H, W)
                # Only count pixels with positive metric depth (some depth maps
                # store zero / NaN where prediction is invalid).
                _vmask = _hm & (_md > 0) & np.isfinite(_md)
                if int(_vmask.sum()) >= _MIN_HAND_MASK_PIX:
                    _hand_z_target_list.append(float(np.median(_md[_vmask])))
                    _hand_z_valid_list.append(True)
                else:
                    _hand_z_target_list.append(0.0)
                    _hand_z_valid_list.append(False)
            _hand_z_target = torch.tensor(_hand_z_target_list, dtype=torch.float32, device=device)  # (N,)
            _hand_z_valid  = torch.tensor(_hand_z_valid_list,  dtype=torch.bool,    device=device)  # (N,)
            # Weight kept small — the heavy lifting is done by closed-form init
            # below; this loss only prevents joints_2d from drifting hand z back.
            # (Larger weights are useless inside the loop because clip_grad_norm_
            # caps the total hand-loss gradient regardless of per-term magnitude.)
            _LAMBDA_HAND_Z_S3 = 1e1
            _n_hz_valid = int(_hand_z_valid.sum().item())
            print(f"Stage 3 hand-z anchor: {_n_hz_valid}/{num_sampled_frames} valid frames "
                  f"(median target_z range "
                  f"{_hand_z_target[_hand_z_valid].min().item() if _n_hz_valid else 0:.3f} – "
                  f"{_hand_z_target[_hand_z_valid].max().item() if _n_hz_valid else 0:.3f} m), "
                  f"weight={_LAMBDA_HAND_Z_S3}")

            # ── Closed-form mano_trans z init correction ─────────────────────────
            # Inside the optim loop, clip_grad_norm_(max_norm=1.0) caps hand-loss
            # gradients regardless of how large hand_z is — so a 2-3 m HaMeR z
            # error would take tens of thousands of steps to fix via gradients.
            # Instead, do a one-shot closed-form shift of mano_trans[:, 2] before
            # the loop so the wrist starts at the right depth.  This costs 1
            # forward pass and immediately puts the hand in the right z range,
            # leaving the loop free to refine 2D fit + finger articulation.
            if _n_hz_valid > 0:
                with torch.no_grad():
                    _, _, _mano_joints_init = multi_frame_model()
                    _init_root_z = _mano_joints_init[:, 0, 2]                 # (N,)
                    _delta_z = torch.zeros_like(_init_root_z)
                    _delta_z[_hand_z_valid] = (_hand_z_target[_hand_z_valid]
                                               - _init_root_z[_hand_z_valid])
                    # For invalid frames, fall back to the median delta of valid
                    # frames so we don't leave a discontinuity.
                    if (~_hand_z_valid).any():
                        _delta_z[~_hand_z_valid] = _delta_z[_hand_z_valid].median()
                    # Sanity clamp: a single-frame correction > 5 m is almost
                    # certainly a degenerate metric_depth read, skip it.
                    _huge = _delta_z.abs() > 5.0
                    if _huge.any():
                        print(f"  [hand-z init] WARNING {int(_huge.sum())} frames with |Δz|>5m, "
                              f"clamping to 0 (likely bad metric_depth at those frames)")
                        _delta_z[_huge] = 0.0
                    multi_frame_model.mano_trans.data[:, 2] += _delta_z
                    # Diagnostic: report shift statistics.
                    _shift_v = _delta_z[_hand_z_valid]
                    print(f"  [hand-z init] shifted mano_trans z by "
                          f"mean={_shift_v.mean().item():+.3f} m, "
                          f"median={_shift_v.median().item():+.3f} m, "
                          f"min={_shift_v.min().item():+.3f}, max={_shift_v.max().item():+.3f}")
                    # Verify post-shift residual.
                    _, _, _mano_joints_post = multi_frame_model()
                    _post_root_z = _mano_joints_post[:, 0, 2]
                    _post_resid = (_post_root_z[_hand_z_valid]
                                   - _hand_z_target[_hand_z_valid]).abs()
                    print(f"  [hand-z init] post-shift |root_z - target_z| "
                          f"mean={_post_resid.mean().item()*1000:.1f} mm, "
                          f"max={_post_resid.max().item()*1000:.1f} mm")


            # EMA-based auto-balancing for losses with mismatched scales.
            # Each tracked loss is normalised by its running mean so all terms
            # contribute gradients at ~unit scale; _w_rel controls relative importance.
            # _ema_init fixes the denominator floor at the first-step value for losses
            # that can oscillate (fp, vp) so the normalised value does not inflate when
            # they temporarily dip.  Monotonically-decreasing losses (joints_2d) must
            # NOT use the floor: the floor would freeze the denominator at the large
            # initial value, causing the normalised contribution to decay to near-zero
            # and effectively zeroing out the joint-fitting signal over time.
            _ema_decay = 0.99
            _ema = {}
            _ema_init = {}
            _eps_ema = 1e-8
            _stage3_joint2d_w = stage3_joint2d_relative_weight(_stage3_video_id)
            _hold_refine_steps_s3 = hold_stage3_hand_refine_steps(_stage3_video_id)
            _w_rel = {'fp': 1.5, 'vp': 1.0, 'joints_2d': _stage3_joint2d_w}
            _no_floor = {'joints_2d'}  # losses that decrease monotonically — skip floor
            _target_hand_joints_2d_s3 = sampled_gt_hand_joints_2d
            _target_hand_joints_valid_mask_s3 = sampled_gt_hand_joints_valid_mask
            if _hold_stage3:
                _hold_joint_order_s3 = torch.as_tensor(hold_joint_target_indices(), device=device, dtype=torch.long)
                _target_hand_joints_2d_s3 = sampled_gt_hand_joints_2d.index_select(1, _hold_joint_order_s3)
                _target_hand_joints_valid_mask_s3 = hold_reorder_valid_mask(
                    sampled_gt_hand_joints_valid_mask,
                    _hold_joint_order_s3,
                )
                print(
                    "Stage 3 hold hand policy: "
                    f"joints_2d_rel_weight={_stage3_joint2d_w:g}, "
                    f"hand_refine_steps={_hold_refine_steps_s3}, "
                    "target_joint_order=mano_to_openpose"
                )

            def _ema_normalise(name, raw_loss):
                val = raw_loss.detach().item()
                if name not in _ema:
                    _ema[name] = val if val > _eps_ema else 1.0
                    _ema_init[name] = _ema[name]
                else:
                    _ema[name] = _ema_decay * _ema[name] + (1 - _ema_decay) * val
                if name in _no_floor:
                    denom = _ema[name] + _eps_ema
                else:
                    denom = max(_ema[name], _ema_init[name]) + _eps_ema
                return raw_loss / denom * _w_rel.get(name, 1.0)

            loop = tqdm(range(num_steps), desc="Optimizing Pose Sequence")
            for step in loop:
                _rot_lr_s3 = (
                    stage3_rotation_lr_for_step(
                        step,
                        num_steps,
                        lr,
                        freeze_fraction=_STAGE3_ROT_FREEZE_FRAC,
                        unlock_lr_scale=_STAGE3_ROT_UNLOCK_LR_SCALE,
                        protect_rotation=_stage3_rot_protect,
                    )
                )
                _rot_frozen_s3 = _stage3_rot_protect and _rot_lr_s3 <= 0.0
                obj_optimizer.param_groups[0]["lr"] = _rot_lr_s3
                multi_frame_model.rot_6d.requires_grad_(not _rot_frozen_s3)

                curr_sigma, curr_gamma, curr_fpp = get_render_params(step, num_steps)
                renderer.rasterizer.raster_settings.blur_radius = curr_sigma
                renderer.rasterizer.raster_settings.faces_per_pixel = curr_fpp
                new_blend_params = BlendParams(sigma=curr_sigma, gamma=curr_gamma)
                renderer.shader.blend_params = new_blend_params

                posed_obj_meshes_batch, posed_hand_meshes_batch, mano_joints_batch = multi_frame_model()

                obj_fragments = renderer.rasterizer(posed_obj_meshes_batch)
                rendered_obj_masks = renderer.shader(obj_fragments, posed_obj_meshes_batch)[..., 3]
                obj_zbuf = obj_fragments.zbuf[..., 0]

                # object loss (fp & vp auto-balanced via EMA)
                loss_fp_raw = weighted_false_positive_loss(rendered_obj_masks, sampled_pred_amodal_masks)
                loss_vp_raw = vectorized_pose_guiding_loss(posed_obj_meshes_batch, sampled_pred_amodal_masks, rendered_obj_masks, camera, num_samples=2000)
                loss_fp = (
                    torch.tensor(0.0, device=device)
                    if ablation == "no_fp"
                    else _ema_normalise('fp', loss_fp_raw)
                )
                loss_vp = (
                    torch.tensor(0.0, device=device)
                    if ablation == "no_vp"
                    else _ema_normalise('vp', loss_vp_raw)
                )

                # Stage 3 optimizes rot6d directly. A very light rotation
                # smoothness damps isolated bad-frame jitter without flattening
                # a good SAM3D rotation trajectory.
                loss_sm_rot = (
                    compute_smoothness_loss(multi_frame_model.rot_6d)
                    * _STAGE3_ROT_SMOOTH_SCALE
                    * _STAGE3_OBJECT_SMOOTH_SCALE
                )
                loss_sm_trans = compute_smoothness_loss(multi_frame_model.trans) * _STAGE3_OBJECT_SMOOTH_SCALE

                _w_st_s3 = 1e2
                loss_static_tie_s3 = torch.tensor(0.0, device=device)
                if approaching_end_idx > 1:
                    tr_s = multi_frame_model.trans[:approaching_end_idx]
                    loss_static_tie_s3 = loss_static_tie_s3 + (
                        tr_s - tr_s.mean(0, keepdim=True).detach()
                    ).pow(2).mean() * _w_st_s3
                if end_static_start_idx < num_sampled_frames - 1:
                    tr_e = multi_frame_model.trans[end_static_start_idx:]
                    loss_static_tie_s3 = loss_static_tie_s3 + (
                        tr_e - tr_e.mean(0, keepdim=True).detach()
                    ).pow(2).mean() * _w_st_s3

                # SAM3D is only an initialization here; do not anchor rot6d
                # back to it, so Stage 3 can repair bad rotation estimates.
                loss_rot_anchor_s3 = torch.tensor(0.0, device=device)
                with torch.no_grad():
                    _rot_drift_stats_s3 = rotation_drift_degrees(
                        rotation_6d_to_matrix(multi_frame_model.rot_6d.detach()),
                        _stage3_rotR_anchor,
                    )

                obj_loss = (loss_fp + loss_vp + loss_sm_rot * smoothness_weight + loss_sm_trans * smoothness_weight
                            + loss_static_tie_s3 + loss_rot_anchor_s3)

                # hand 2d joints loss (auto-balanced via EMA)
                projected_hand_joints = camera.transform_points_screen(mano_joints_batch, image_size=((H_out, W_out),))  # (N, 21, 3)
                pred_hand_joints_2d = projected_hand_joints[..., :2]
                num_valid_frames = _target_hand_joints_valid_mask_s3.sum()
                if num_valid_frames > 0:
                    loss_joints_2d_raw = torch.nn.functional.mse_loss(
                        pred_hand_joints_2d[_target_hand_joints_valid_mask_s3],
                        _target_hand_joints_2d_s3[_target_hand_joints_valid_mask_s3],
                    )
                else:
                    loss_joints_2d_raw = torch.tensor(0.0, device=device)
                loss_joints_2d = _ema_normalise('joints_2d', loss_joints_2d_raw)
                loss_sm_hand = compute_smoothness_loss(mano_joints_batch)

                # hand root-z anchor: pulls wrist depth toward median(metric_depth)
                # inside hand mask.  Single scalar per frame — resolves perspective
                # ambiguity without per-pixel hand depth supervision.
                if _n_hz_valid > 0:
                    _root_z_pred = mano_joints_batch[:, 0, 2]                    # (N,)
                    loss_hand_z_raw = torch.nn.functional.mse_loss(
                        _root_z_pred[_hand_z_valid], _hand_z_target[_hand_z_valid])
                else:
                    loss_hand_z_raw = torch.tensor(0.0, device=device)
                loss_hand_z = loss_hand_z_raw * _LAMBDA_HAND_Z_S3

                # hand anatomy loss — strengthened (1.0 → 5.0).  Stage 3 has no
                # other guard against unreasonable hand poses produced by noisy
                # 2D keypoints (e.g. fingers bent past joint limits, twisted
                # MCPs).  EMA-normalised joints_2d ≈ 1.0, so weight 5.0 makes
                # anatomy the dominant term whenever joints_2d tries to push
                # past anatomical limits, while still letting joints_2d drive
                # the fit when it is physically valid.
                T_g_p = multi_frame_model.transforms_abs  # (B, 16, 4, 4)
                T_g_a, _R, ee = multi_frame_model.axisFK(T_g_p)  # ee (B, 16, 3)
                _LAMBDA_ANATOMY_S3 = 5.0
                loss_anatomy = multi_frame_model.anatomyLoss(ee) * _LAMBDA_ANATOMY_S3

                # HaMeR mano_pose anchor — keeps fingers near HaMeR's 3D-regressed
                # articulation when joints_2d is noisy/missing.  Computed in 6D
                # rotation space (see mano_pose_to_6d note above) for continuity
                # and rotation-invariance.
                _mp_6d_s3 = mano_pose_to_6d(multi_frame_model.mano_pose)
                loss_mano_pose_anchor = torch.nn.functional.mse_loss(
                    _mp_6d_s3, _hamer_mano_pose_6d_init
                ) * _LAMBDA_MANO_POSE_ANCHOR_S3

                # hand_smooth_w is kept much smaller than smoothness_weight because
                # compute_smoothness_loss uses .sum() over all frames×joints×coords, so
                # it grows ~N×J×3 times larger than the per-element mse_loss used for
                # joints_2d.  A high shared smoothness_weight would cause the smoothness
                # gradient to suppress joint fitting, especially in fast-motion frames.
                hand_smooth_w = max(smoothness_weight * 0.1, 1.0)
                hand_loss = (loss_joints_2d + loss_sm_hand * hand_smooth_w
                             + loss_anatomy + loss_hand_z + loss_mano_pose_anchor)

                # ── Decoupled backward + grad-clip ───────────────────────────────────
                # Object and hand parameters are independent (no shared gradient path),
                # but a single global clip_grad_norm would let large object gradients
                # suppress the hand update.  Separate backward passes + per-group clips
                # give each side its own gradient budget.
                obj_optimizer.zero_grad()
                obj_loss.backward(retain_graph=True)  # keep graph alive for hand backward
                torch.nn.utils.clip_grad_norm_([multi_frame_model.rot_6d, multi_frame_model.trans], max_norm=1.0)
                obj_optimizer.step()

                hand_optimizer.zero_grad()
                hand_loss.backward()  # graph freed here
                torch.nn.utils.clip_grad_norm_([multi_frame_model.mano_root_orient, multi_frame_model.mano_trans, multi_frame_model.mano_pose], max_norm=1.0)
                hand_optimizer.step()

                total_loss = obj_loss + hand_loss  # for logging only, no backward
                loop.set_postfix(
                    loss=total_loss.item(),
                    fp=loss_fp.item(),
                    vp=loss_vp.item(),
                    sm_rot=loss_sm_rot.item(),
                    tie=loss_static_tie_s3.item(),
                    rot_lr=f"{_rot_lr_s3:.1e}",
                    rdeg=round(_rot_drift_stats_s3["mean_deg"], 2),
                    jts=loss_joints_2d.item(),
                    sm_h=loss_sm_hand.item(),
                    hz=round(loss_hand_z_raw.item(), 4),
                )
                if step % 50 == 0 or step == num_steps - 1:
                    _opt_log.write(f"{step},{total_loss.item():.6f},{obj_loss.item():.6f},"
                                   f"{loss_fp.item():.6f},{loss_vp.item():.6f},"
                                   f"{loss_fp_raw.item():.6f},{loss_vp_raw.item():.6f},"
                                   f"{loss_sm_rot.item():.6f},{loss_sm_trans.item():.6f},"
                                   f"{loss_static_tie_s3.item():.6f},"
                                   f"{loss_rot_anchor_s3.item():.6f},{_rot_lr_s3:.9g},"
                                   f"{_rot_drift_stats_s3['mean_deg']:.6f},"
                                   f"{_rot_drift_stats_s3['max_deg']:.6f},"
                                   f"{hand_loss.item():.6f},{loss_joints_2d.item():.6f},"
                                   f"{loss_joints_2d_raw.item():.6f},"
                                   f"{loss_sm_hand.item():.6f},{loss_anatomy.item():.6f},"
                                   f"{loss_hand_z.item():.6f},{loss_hand_z_raw.item():.6f},"
                                   f"{_n_hz_valid},"
                                   f"{loss_mano_pose_anchor.item():.6f}\n")

            if _hold_refine_steps_s3 > 0 and _target_hand_joints_valid_mask_s3.any():
                _opt_log.write("\n# Stage 3H — Hold hand 2D refinement\n")
                _opt_log.write("step,total,joints_2d,joints_2d_raw,sm_hand,anatomy,hand_z,hand_z_raw,mano_pose_anchor\n")
                print(
                    f"\nStage 3H hold-only hand refinement: {_hold_refine_steps_s3} steps "
                    "(object fixed, stronger 2D joints fit)"
                )

                multi_frame_model.rot_6d.requires_grad_(False)
                multi_frame_model.trans.requires_grad_(False)
                multi_frame_model.mano_root_orient.requires_grad_(True)
                multi_frame_model.mano_trans.requires_grad_(True)
                multi_frame_model.mano_pose.requires_grad_(True)

                _hold_hand_optimizer = torch.optim.Adam(
                    [
                        multi_frame_model.mano_root_orient,
                        multi_frame_model.mano_trans,
                        multi_frame_model.mano_pose,
                    ],
                    lr=lr,
                )
                _hold_j2d_scale = 100.0  # 10px RMS -> normalized loss ~= 1.
                _hold_smooth_w = max(float(smoothness_weight) * 0.02, 0.2)
                _hold_anatomy_w = 5.0
                _hold_pose_anchor_w = 2e-1

                loop_s3h = tqdm(range(_hold_refine_steps_s3), desc="Stage 3H: Hold hand 2D refinement")
                for _hstep in loop_s3h:
                    _hold_hand_optimizer.zero_grad()
                    _, _, _mano_joints_h = multi_frame_model()
                    _proj_h = camera.transform_points_screen(
                        _mano_joints_h, image_size=((H_out, W_out),))[..., :2]
                    _loss_j2d_raw_h = torch.nn.functional.mse_loss(
                        _proj_h[_target_hand_joints_valid_mask_s3],
                        _target_hand_joints_2d_s3[_target_hand_joints_valid_mask_s3],
                    )
                    _loss_j2d_h = _loss_j2d_raw_h / _hold_j2d_scale * _stage3_joint2d_w
                    _loss_sm_hand_h = compute_smoothness_loss(_mano_joints_h) * _hold_smooth_w

                    if _n_hz_valid > 0:
                        _root_z_h = _mano_joints_h[:, 0, 2]
                        _loss_hand_z_raw_h = torch.nn.functional.mse_loss(
                            _root_z_h[_hand_z_valid], _hand_z_target[_hand_z_valid])
                    else:
                        _loss_hand_z_raw_h = torch.tensor(0.0, device=device)
                    _loss_hand_z_h = _loss_hand_z_raw_h * _LAMBDA_HAND_Z_S3

                    _T_g_p_h = multi_frame_model.transforms_abs
                    _, _, _ee_h = multi_frame_model.axisFK(_T_g_p_h)
                    _loss_anatomy_h = multi_frame_model.anatomyLoss(_ee_h) * _hold_anatomy_w
                    _mp_6d_h = mano_pose_to_6d(multi_frame_model.mano_pose)
                    _loss_pose_anchor_h = torch.nn.functional.mse_loss(
                        _mp_6d_h, _hamer_mano_pose_6d_init
                    ) * _hold_pose_anchor_w

                    _hold_total_h = (
                        _loss_j2d_h
                        + _loss_sm_hand_h
                        + _loss_anatomy_h
                        + _loss_hand_z_h
                        + _loss_pose_anchor_h
                    )
                    _hold_total_h.backward()
                    torch.nn.utils.clip_grad_norm_(
                        [
                            multi_frame_model.mano_root_orient,
                            multi_frame_model.mano_trans,
                            multi_frame_model.mano_pose,
                        ],
                        max_norm=1.0,
                    )
                    _hold_hand_optimizer.step()

                    loop_s3h.set_postfix(
                        loss=round(float(_hold_total_h.item()), 4),
                        j2d=round(float(_loss_j2d_raw_h.item()), 2),
                        rms=round(float(torch.sqrt(_loss_j2d_raw_h.detach()).item()), 2),
                    )
                    if _hstep % 25 == 0 or _hstep == _hold_refine_steps_s3 - 1:
                        _opt_log.write(
                            f"{_hstep},{_hold_total_h.item():.6f},"
                            f"{_loss_j2d_h.item():.6f},{_loss_j2d_raw_h.item():.6f},"
                            f"{_loss_sm_hand_h.item():.6f},{_loss_anatomy_h.item():.6f},"
                            f"{_loss_hand_z_h.item():.6f},{_loss_hand_z_raw_h.item():.6f},"
                            f"{_loss_pose_anchor_h.item():.6f}\n"
                        )

                print("Stage 3H hold-only hand refinement completed.")

            multi_frame_model.rot_6d.requires_grad_(True)
            multi_frame_model.trans.requires_grad_(True)
            multi_frame_model.mano_root_orient.requires_grad_(True)
            multi_frame_model.mano_trans.requires_grad_(True)
            multi_frame_model.mano_pose.requires_grad_(True)

            # Close log after Stages 1-3; Stage 8 will append to the same file.
            _opt_log.close()

            # ----------------------------------------------------------------
            # Ray-scale alignment: shift the entire object sequence along the
            # camera ray so its average depth matches the hand's average depth.
            # Multiplying both `trans` and `scale` by the same factor `alpha`
            # leaves every per-frame projection EXACTLY unchanged (silhouette /
            # 2D fit preserved), but slides the object along the camera ray so
            # hand and object end up at roughly the same depth plane. This
            # gives Stage 4 contact-map estimation a usable initialization
            # without disturbing the Stage 1/2/3 silhouette fit.
            # ----------------------------------------------------------------
            with torch.no_grad():
                obj_meshes_align, hand_meshes_align, _ = multi_frame_model()
                obj_z_per_frame = obj_meshes_align.verts_padded()[..., 2].mean(dim=1)   # (N,)
                hand_z_per_frame = hand_meshes_align.verts_padded()[..., 2].mean(dim=1) # (N,)

                # Use interaction frames only for computing the alignment factor.
                if interaction_end_idx > approaching_end_idx:
                    obj_z_int = obj_z_per_frame[approaching_end_idx:interaction_end_idx]
                    hand_z_int = hand_z_per_frame[approaching_end_idx:interaction_end_idx]
                    gap_before_mm = (obj_z_int - hand_z_int).abs().mean().item() * 1000

                    # Self-filter: among interaction frames, the ones where
                    # |hand_z - obj_z| is smallest are most likely real-contact
                    # frames. False-positive boundary frames (hand still
                    # approaching / already retreating) tend to have larger gap
                    # and would bias the alpha toward a wrong target. Keep only
                    # the top-`KEEP_FRAC` closest frames for the median ratio.
                    # This decouples the alpha quality from the window-detection
                    # accuracy: even with ±5 frames of false interaction, alpha
                    # is determined by the truly-grasping subset.
                    KEEP_FRAC = 0.7
                    gaps = (obj_z_int - hand_z_int).abs()
                    n_total = obj_z_int.numel()
                    n_keep = max(3, int(n_total * KEEP_FRAC))
                    n_keep = min(n_keep, n_total)
                    keep_idx = torch.argsort(gaps)[:n_keep]
                    obj_z_keep = obj_z_int[keep_idx]
                    hand_z_keep = hand_z_int[keep_idx]
                    obj_z_med = obj_z_keep.median().item()
                    hand_z_med = hand_z_keep.median().item()
                    if obj_z_med > 1e-6:
                        alpha = hand_z_med / obj_z_med
                    else:
                        alpha = 1.0

                    # Sanity warning on absurd alpha (HaMeR or MoGe likely failed).
                    if not (0.5 <= alpha <= 2.0):
                        print(f"\n[Ray-scale alignment] WARNING alpha={alpha:.4f} outside [0.5, 2.0]. "
                              f"hand_z or obj_z initialization is likely wrong; clamping to 1.0 (skip alignment).")
                        alpha = 1.0

                    multi_frame_model.scale.data.mul_(alpha)
                    multi_frame_model.trans.data.mul_(alpha)

                    # Recompute after applying the correction for verification.
                    obj_meshes_after, hand_meshes_after, _ = multi_frame_model()
                    obj_z_after = obj_meshes_after.verts_padded()[..., 2].mean(dim=1)[approaching_end_idx:interaction_end_idx]
                    hand_z_after = hand_meshes_after.verts_padded()[..., 2].mean(dim=1)[approaching_end_idx:interaction_end_idx]
                    gap_after_mm = (obj_z_after - hand_z_after).abs().mean().item() * 1000

                    print(f"\n[Ray-scale alignment] "
                          f"interaction frames [{approaching_end_idx}:{interaction_end_idx})  "
                          f"alpha={alpha:.4f}  "
                          f"(top-{n_keep}/{n_total} closest frames used: "
                          f"obj_z med={obj_z_med:.4f} m, hand_z med={hand_z_med:.4f} m)")
                    print(f"[Ray-scale alignment] mean |obj_z - hand_z| over interaction: "
                          f"{gap_before_mm:.1f} mm  →  {gap_after_mm:.1f} mm  "
                          f"(silhouette preserved by construction)")
                else:
                    print("\n[Ray-scale alignment] no interaction frames detected — skipping.")

            # --- Quick Visualization of Sparse Optimization Results ---
            with torch.no_grad():
                sparse_posed_meshes, sparse_posed_hand_meshes, sparse_mano_joints = multi_frame_model()
                scenes = []
                for i in range(len(sparse_posed_meshes)):
                    scene_i = join_meshes_as_scene([sparse_posed_meshes[i]])
                    scenes.append(scene_i)
                sparse_scene = join_meshes_as_batch(scenes)
                final_rendered_masks_sparse = renderer(sparse_scene, cameras=camera)[..., 3].cpu().numpy()

                # Side-view with Phong shading (90° Y rotation, fresh HardPhong renderer)
                sparse_side_rgb = render_side_view_rgb(sparse_posed_meshes, sparse_posed_hand_meshes, camera, device, (H_out, W_out))

                projected_hand_joints = camera.transform_points_screen(sparse_mano_joints, image_size=((H_out, W_out),))  # (N, 21, 3)
                pred_joints_2d_np = projected_hand_joints[..., :2].cpu().numpy()
                gt_joints_2d_np = _target_hand_joints_2d_s3.cpu().numpy()

                # Camera-view object Phong for optimized_fitting_SPARSE.mp4 (row2 middle).
                # Match optimized_fitting_SPARSE_final: crop-space PerspectiveCameras + HardPhong +
                # overlay_rgb_render (not raw intrinsics + CPU overlay — that misaligns with H_out/W_out K).
                sparse_obj_cam_overlay_rgb01: List[np.ndarray] = []
                try:
                    _spar_blue = torch.tensor([0.65, 0.8, 1.0], device=device)
                    N, V = sparse_posed_meshes.verts_padded().shape[:2]
                    sparse_posed_meshes.textures = TexturesVertex(
                        verts_features=_spar_blue.view(1, 1, 3).expand(N, V, -1)
                    )
                    _spar_phong_lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])
                    _spar_phong_mats = Materials(device=device, specular_color=[[1.0, 1.0, 1.0]], shininess=1.0)
                    _spar_phong_blend = BlendParams(background_color=(0.0, 0.0, 0.0))
                    _spar_phong_rast = RasterizationSettings(
                        image_size=(H_out, W_out), blur_radius=0.0, faces_per_pixel=1
                    )
                    _spar_phong_renderer = MeshRenderer(
                        rasterizer=MeshRasterizer(raster_settings=_spar_phong_rast),
                        shader=HardPhongShader(
                            device=device, lights=_spar_phong_lights, blend_params=_spar_phong_blend
                        ),
                    )
                    _spar_phong_cam = PerspectiveCameras(
                        focal_length=focal_length,
                        principal_point=principal_point,
                        image_size=((H_out, W_out),) * num_sampled_frames,
                        in_ndc=False,
                        device=device,
                    )
                    _spar_scenes = [join_meshes_as_scene([sparse_posed_meshes[j]]) for j in range(num_sampled_frames)]
                    _spar_batch = join_meshes_as_batch(_spar_scenes)
                    _spar_rgb = _spar_phong_renderer(
                        _spar_batch,
                        cameras=_spar_phong_cam,
                        lights=_spar_phong_lights,
                        materials=_spar_phong_mats,
                    )[..., :3].cpu().numpy()
                    for _i in range(num_sampled_frames):
                        sparse_obj_cam_overlay_rgb01.append(
                            overlay_rgb_render(sampled_rgbs_np[_i], _spar_rgb[_i], alpha=1.0)
                        )
                except Exception as _e_ov_sp:
                    print(f"[Stage3 SPARSE video] object mesh Phong batch failed ({_e_ov_sp}); using hand mask panel.")
                    for _i in range(num_sampled_frames):
                        sparse_obj_cam_overlay_rgb01.append(
                            overlay_mask_on_image(sampled_rgbs_np[_i], sampled_hand_masks_np[_i], cmap_idx=1)
                        )

            if save_intermediates:
                sparse_video_path = os.path.join(output_path, 'optimized_fitting_SPARSE.mp4')

                writer_sparse = imageio.get_writer(sparse_video_path,
                                                   fps=30,
                                                   codec='libx264',
                                                   pixelformat='yuv420p',
                                                   ffmpeg_params=['-crf', '28', '-preset', 'veryfast'],
                                                   macro_block_size=None)

                for i in range(num_sampled_frames):
                    modal_frame = overlay_mask_on_image(sampled_rgbs_np[i], sampled_modal_masks_np[i])
                    amodal_gt_frame = overlay_mask_on_image(sampled_rgbs_np[i], sampled_pred_amodal_masks_np[i])
                    binary_rendered_mask = (final_rendered_masks_sparse[i] > 0.5).astype(np.uint8)
                    # Row1 col3: object silhouette (soft renderer mask on RGB).
                    render_frame = overlay_mask_on_image(sampled_rgbs_np[i], binary_rendered_mask)

                    # Row2: joints on silhouette | camera Phong object | side view.
                    joints_on_obj_silhouette = render_frame.copy()
                    joints_on_obj_silhouette = overlay_points_on_image(
                        joints_on_obj_silhouette, pred_joints_2d_np[i], color=(1.0, 0.0, 0.0)
                    )
                    joints_on_obj_silhouette = overlay_points_on_image(
                        joints_on_obj_silhouette, gt_joints_2d_np[i], color=(0.0, 1.0, 0.0)
                    )
                    obj_cam_overlay_frame = sparse_obj_cam_overlay_rgb01[i]

                    # Side-view: Phong-shaded RGB (object=blue, hand=red)
                    side_panel = np.clip(sparse_side_rgb[i], 0.0, 1.0)

                    # Layout: [modal | amodal | obj silhouette] / [joints@silhouette | Phong | side]
                    panels = [
                        modal_frame,
                        amodal_gt_frame,
                        render_frame,
                        joints_on_obj_silhouette,
                        obj_cam_overlay_frame,
                        side_panel,
                    ]

                    row1 = np.hstack(panels[:3])
                    row2 = np.hstack(panels[3:])
                    combined_frame = np.vstack([row1, row2])

                    # Convert to uint8 for cv2.putText
                    combined_frame_uint8 = (combined_frame * 255).astype(np.uint8)

                    # Add sampled/full frame and stage annotation.
                    _label, _color = _sampled_video_label(i)
                    combined_frame_uint8 = _draw_video_label(combined_frame_uint8, _label, _color)

                    writer_sparse.append_data(combined_frame_uint8)
                writer_sparse.close()
                print(f"Saved sparse fitting visualization to {sparse_video_path}")

        _run_stage1()

        # Motion-type gate for SAM3D (same signals as Stage 3 profile, without PnP R).
        try:
            from interaction_motion_profile import build_interaction_motion_profile

            if (approaching_end_idx is not None and interaction_end_idx is not None
                    and interaction_end_idx > approaching_end_idx + 1):
                _type_lo, _type_hi, _opt_lo, _opt_hi = _interaction_motion_typing_window()
                _prof_gate = build_interaction_motion_profile(
                    _type_lo,
                    _type_hi,
                    cotracker_tracks_fwd_np,
                    cotracker_fwd_vis_np,
                    sampled_pred_amodal_masks_np,
                    pnp_R_col_major=None,
                    sampled_mano_params=sampled_mano_params,
                )
                _prof_gate = force_rotation_likely_profile(
                    _prof_gate,
                    reason="sam3d_gate_force_rotation_likely",
                )
                _prof_gate["source"] = "post_stage1_sam3d_gate"
                _prof_gate["motion_typing_core_half_open"] = [_type_lo, _type_hi]
                _prof_gate["optimization_interaction_padded_half_open"] = [_opt_lo, _opt_hi]
                sam3d_gate_suggested_mode = str(_prof_gate.get("suggested_mode", FORCE_ROTATION_MODE))
                _gate_json = os.path.join(output_path, "interaction_motion_profile_sam3d_gate.json")
                with open(_gate_json, "w") as _gf:
                    json.dump(_prof_gate, _gf, indent=2)
                print(f"\n[SAM3D-GATE] post–Stage 1 → {_gate_json}  "
                      f"forced suggested_mode={sam3d_gate_suggested_mode} "
                      f"(original={_prof_gate.get('policy_original_suggested_mode')})")
            else:
                sam3d_gate_suggested_mode = FORCE_ROTATION_MODE
                print("\n[SAM3D-GATE] skip profile (invalid 2–3 interaction range); "
                      "suggested_mode forced to rotation_likely.")
        except Exception as _e_gate:
            print(f"\n[SAM3D-GATE] profile failed: {_e_gate}")
            traceback.print_exc()
            sam3d_gate_suggested_mode = FORCE_ROTATION_MODE

        # _stride_eff = sparse_stride_for_motion_profile(sam3d_gate_suggested_mode, sam3d_sparse_stride)
        # if sam3d_sparse_keyframes and sam3d_gate_suggested_mode == "translation_likely":
        #     run_sam3d_sparse_keyframes_motion_window(
        #         enabled=True,
        #         output_path=output_path,
        #         sampled_indices=sampled_indices,
        #         num_sampled_frames=num_sampled_frames,
        #         stage1_interaction_lo_auto=int(stage1_interaction_lo_auto) if stage1_interaction_lo_auto is not None else None,
        #         stage1_interaction_hi_auto=int(stage1_interaction_hi_auto) if stage1_interaction_hi_auto is not None else None,
        #         approaching_end_idx_padded=int(approaching_end_idx),
        #         interaction_end_idx_padded=int(interaction_end_idx),
        #         raw_rgbs_np=sam3d_raw_rgbs_np,
        #         raw_masks_np=sam3d_raw_masks_np,
        #         stride=_stride_eff,
        #         seed=sam3d_seed,
        #         quiet=sam3d_quiet,
        #         config_path=sam3d_config_path,
        #         lambda_temp=sam3d_lambda_temp,
        #         intrinsics_full=intrinsics,
        #         mesh_verts=verts,
        #         mesh_faces=faces,
        #         device=device,
        #     )
        # elif sam3d_sparse_keyframes and sam3d_gate_suggested_mode == "rotation_likely":
        #     print("[SAM3D-GATE] rotation_likely: skip translation-style milestone SAM3D; gap=1 dense + PnP compare after Stage 2.")
        # elif sam3d_sparse_keyframes:
        #     print(f"[SAM3D-GATE] skip milestone SAM3D (suggested_mode={sam3d_gate_suggested_mode}).")

        _run_stage2()

        if sam3d_sparse_keyframes and sam3d_gate_suggested_mode == "rotation_likely":
            _onset = int(stage1_interaction_lo_auto) if stage1_interaction_lo_auto is not None else int(approaching_end_idx)
            if _onset >= num_sampled_frames:
                print(f"[SAM3D-ROT] skip: mask-onset sampled_idx={_onset} out of range")
            else:
                _sam3d_rot_outlier_angle = hold_sam3d_rot_outlier_max_angle(
                    os.path.basename(seq_path),
                    default=float(sam3d_rot_outlier_max_angle_deg),
                )
                if abs(_sam3d_rot_outlier_angle - float(sam3d_rot_outlier_max_angle_deg)) > 1e-6:
                    print(
                        f"[SAM3D-ROT] hold video: overriding outlier threshold "
                        f"{float(sam3d_rot_outlier_max_angle_deg):.1f}deg -> "
                        f"{_sam3d_rot_outlier_angle:.1f}deg"
                    )
                _rot_summary, _pose_snaps = run_sam3d_rotation_dense_from_mask_onset(
                    enabled=True,
                    output_path=output_path,
                    onset_sampled_idx=_onset,
                    end_sampled_exclusive=int(num_sampled_frames),
                    sampled_indices=sampled_indices,
                    raw_rgbs_np=sam3d_raw_rgbs_np,
                    raw_masks_np=sam3d_raw_masks_np,
                    seed=sam3d_seed,
                    quiet=sam3d_quiet,
                    config_path=sam3d_config_path,
                    lambda_temp=sam3d_lambda_temp,
                    intrinsics_full=intrinsics,
                    mesh_verts=verts,
                    mesh_faces=faces,
                    device=device,
                    interaction_segment_lo=int(approaching_end_idx),
                    interaction_segment_hi=int(interaction_end_idx),
                    outlier_filter=bool(sam3d_rot_outlier_filter),
                    outlier_max_angle_deg=float(_sam3d_rot_outlier_angle),
                    outlier_max_iters=int(sam3d_rot_outlier_max_iters),
                    retry_count=int(sam3d_rot_retry_count),
                    mesh_overlay_alpha=float(sam3d_rot_mesh_overlay_alpha),
                    global_frame_offset=int(sam3d_global_frame_offset),
                    overwrite_dense_cache=bool(overwrite_sam3d_dense_cache),
                    reference_sampled_idx=int(sam3d_ref_sampled_idx),
                )
                if _pose_snaps:
                    sam3d_rotation_pose_snapshots = _pose_snaps
                _cmp_dir = os.path.join(output_path, "sam3d_sparse_keyframes", "rotation_dense", "pnp_vs_sam3d_vis")
                os.makedirs(_cmp_dir, exist_ok=True)
                try:
                    _cmp_anchor = int(sam3d_ref_sampled_idx) if sam3d_ref_enabled else int(_onset)
                    verts_canonical_scaled = verts * (single_frame_model.scale.detach() * single_frame_model.initial_scale).unsqueeze(0)
                    mesh_canonical_scaled = Meshes(
                        verts=[verts_canonical_scaled],
                        faces=[faces],
                        textures=TexturesVertex(verts_features=torch.ones_like(verts)[None]),
                    )
                    queries_2d, queries_3d = generate_queries(
                        final_posed_mesh,
                        mesh_canonical_scaled,
                        sampled_pred_amodal_masks_np[_cmp_anchor],
                        focal_length[_cmp_anchor : _cmp_anchor + 1],
                        principal_point[_cmp_anchor : _cmp_anchor + 1],
                        grid_size=15,
                        device=device,
                    )
                    _fxk = fx_new.item() if torch.is_tensor(fx_new) else float(fx_new)
                    _fyk = fy_new.item() if torch.is_tensor(fy_new) else float(fy_new)
                    _cxk = cx_new.item() if torch.is_tensor(cx_new) else float(cx_new)
                    _cyk = cy_new.item() if torch.is_tensor(cy_new) else float(cy_new)
                    K = np.array(
                        [[_fxk, 0.0, _cxk], [0.0, _fyk, _cyk], [0.0, 0.0, 1.0]],
                        dtype=np.float64,
                    )
                    optimized_R = rotation_6d_to_matrix(single_frame_model.rot_6d.detach().unsqueeze(0))[0]
                    optimized_T = single_frame_model.trans.detach()
                    pnp_poses_list = run_pnp_1stage(
                        cotracker_model,
                        sampled_rgbs[_cmp_anchor:],
                        queries_2d,
                        queries_3d,
                        sampled_pred_amodal_masks_np[_cmp_anchor:],
                        K,
                        optimized_R,
                        optimized_T,
                        device,
                        output_dir=_cmp_dir,
                        vis_threshold=0.0,
                    )
                    pnp_stack = torch.from_numpy(np.stack(pnp_poses_list, axis=0)).float().to(device)
                    _scale_full = (single_frame_model.scale.detach() * single_frame_model.initial_scale).to(device)
                    for _snap in _pose_snaps:
                        _sidx = int(_snap["sampled_idx"])
                        _k = _sidx - int(_cmp_anchor)
                        if _k < 0 or _k >= int(pnp_stack.shape[0]):
                            continue
                        _raw_i = int(_snap["raw_frame_idx"])
                        _img = np.ascontiguousarray(sam3d_raw_rgbs_np[_raw_i])
                        if _img.dtype != np.uint8:
                            _img = (np.clip(_img, 0.0, 1.0) * 255.0).astype(np.uint8) if _img.max() <= 1.0 else np.clip(_img, 0, 255).astype(np.uint8)
                        _out_s = {kk: vv.to(device) for kk, vv in _snap["pose"].items()}
                        ov_sam = render_mesh_soft_overlay_on_rgb(
                            _img, verts, faces, _out_s, intrinsics, device, mesh_color=(0.25, 0.75, 1.0), alpha=0.85
                        )
                        P = pnp_stack[_k]
                        vw = world_vertices_from_pnp_row(verts, P[:3, :3], P[:3, 3], _scale_full)
                        ov_pnp = render_mesh_phong_overlay_world_vertices(
                            _img,
                            vw,
                            faces,
                            intrinsics,
                            device,
                            mesh_color=(1.0, 0.45, 0.15),
                            alpha=0.85,
                        )
                        h = max(ov_sam.shape[0], ov_pnp.shape[0])
                        w1 = int(ov_sam.shape[1] * h / max(ov_sam.shape[0], 1))
                        w2 = int(ov_pnp.shape[1] * h / max(ov_pnp.shape[0], 1))
                        a = cv2.resize(ov_sam, (w1, h), interpolation=cv2.INTER_AREA)
                        b = cv2.resize(ov_pnp, (w2, h), interpolation=cv2.INTER_AREA)
                        cmp_bgr = cv2.cvtColor(np.hstack([a, b]), cv2.COLOR_RGB2BGR)
                        cv2.putText(
                            cmp_bgr,
                            "SAM3D (L) | PnP viz-only (R)",
                            (10, 28),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.75,
                            (255, 255, 255),
                            2,
                            cv2.LINE_AA,
                        )
                        cv2.imwrite(os.path.join(_cmp_dir, f"compare_sidx_{_sidx:04d}_raw_{_raw_i:05d}.png"), cmp_bgr)
                    print(f"[SAM3D-ROT] PnP compare panels → {_cmp_dir}  (n={len(_pose_snaps)} SAM3D frames aligned by sampled_idx)")
                except Exception as _e_cmp:
                    print(f"[SAM3D-ROT] PnP compare visualization failed: {_e_cmp}")
                    traceback.print_exc()

        _run_stage3()

        # ----------------------------------------------------------------
        # Save Stage 3 checkpoint (所有后续 Stage 7-8 可直接从此加载)
        # ----------------------------------------------------------------
        print(f"\nSaving Stage 3 checkpoint to: {checkpoint_path}")
        torch.save(
            {
                # Index / boundary
                'sampled_indices': sampled_indices,
                'num_sampled_frames': num_sampled_frames,
                'sampling_policy': sampling_policy,
                'start_static_end_idx': start_static_end_idx,
                'approaching_end_idx': approaching_end_idx,
                'interaction_end_idx': interaction_end_idx,
                'end_static_start_idx': end_static_start_idx,
                'stage1_interaction_lo_auto': stage1_interaction_lo_auto,
                'stage1_interaction_hi_auto': stage1_interaction_hi_auto,
                'phase_layout_mode': phase_layout_mode,
                'phase_layout_info': phase_layout_info,
                'sam3d_ref_frame_idx': int(sam3d_ref_frame_idx) if sam3d_ref_frame_idx is not None else 0,
                'sam3d_ref_sampled_idx': int(sam3d_ref_sampled_idx),
                'sam3d_ref_actual_clip_frame_idx': int(sam3d_ref_actual_clip_frame_idx),
                'sam3d_ref_actual_dataset_frame_idx': int(sam3d_ref_dataset_frame_idx),
                'sam3d_ref_source': sam3d_ref_source,
                'sam3d_sparse_keyframes': bool(sam3d_sparse_keyframes),
                'sam3d_ref_keyframe_init': bool(sam3d_ref_keyframe_init),
                'sam3d_ref_keyframe_init_debug': sam3d_ref_keyframe_init_debug,
                # Camera
                'focal_length': focal_length.cpu(),
                'principal_point': principal_point.cpu(),
                'H_out': H_out,
                'W_out': W_out,
                'fx_new': fx_new,
                'fy_new': fy_new,
                'cx_new': cx_new,
                'cy_new': cy_new,
                # Object mesh
                'verts': verts.detach().cpu(),
                'faces': faces.detach().cpu(),
                # Data arrays
                'sampled_pred_amodal_masks_np': sampled_pred_amodal_masks_np,
                'sampled_rgbs_np': sampled_rgbs_np,
                'sampled_modal_masks_np': sampled_modal_masks_np,
                'sampled_hand_masks_np': sampled_hand_masks_np,
                'sampled_metric_depths_np': sampled_metric_depths_np,
                # 2D keypoints
                'sampled_gt_hand_joints_2d': sampled_gt_hand_joints_2d.cpu(),
                'sampled_gt_hand_joints_valid_mask': sampled_gt_hand_joints_valid_mask.cpu(),
                # Model state
                'model_state_dict': {
                    k: v.cpu() for k, v in multi_frame_model.state_dict().items()
                },
                'initial_scale': multi_frame_model.initial_scale.detach().cpu(),
                'sampled_mano_params': sampled_mano_params,
            },
            checkpoint_path)
        print("Stage 3 checkpoint saved.")

    # Pre-declare Stage 4 outputs used by the downstream if-guard and shared setup
    parsed_contact_map = None
    contact_indices_override = None

    def _run_stage4():
        nonlocal parsed_contact_map, contact_indices_override
        # ========================================================================
        # STAGE 4: Load contact correspondence + apply depth-offset corrections
        # ------------------------------------------------------------------------
        # Two corrections applied in-place on multi_frame_model:
        #   - Ray-scale alignment (auto, computed at end of STAGE 3): scales obj
        #     trans + scale by alpha to align median obj_z with median hand_z over
        #     interaction frames.  Preserves silhouette by construction.
        #   - camera_ray_depth_offset.json (from external "Grasping Pose Correction"
        #     pre-pass): per-frame rigid shift of mano_trans along the camera ray
        #     to fix HaMeR z-bias.  Optional 2-pass workflow: 1st run stops here,
        #     2nd run after the JSON is generated applies the offset and continues.
        # No optimisation, no log file.  Loads contact_map_per_frame.json for
        # use by STAGE 5/6.
        # ========================================================================
        print("\n" + "=" * 80)
        print("STAGE 4: Loading Contact Correspondence")
        print("=" * 80)

        grasp_correction_dir = os.path.join(output_path, "grasp_correction")
        camera_ray_depth_offset_path = os.path.join(grasp_correction_dir, "camera_ray_depth_offset.json")

        if overwrite_grasp_correction and os.path.exists(camera_ray_depth_offset_path):
            print(f"\n--overwrite_grasp_correction: removing existing {camera_ray_depth_offset_path}")
            os.remove(camera_ray_depth_offset_path)

        contact_map_raw = None
        if not os.path.exists(camera_ray_depth_offset_path):
            # Auto-run the GraspFlowMatching pre-pass to generate the JSON, using
            # the same conda env as this script (the active Python environment) and the
            # currently-visible GPU count for DDP sampling.  Two commands, run
            # sequentially in stage2_grasp_correction/GraspFlowMatching:
            #   1. prepare_meshdata.py --video_id <vid>
            #   2. torchrun --nnodes=1 --nproc_per_node=<num_gpus> sample_cam_ray_ddp.py
            #      ODE --ckpt ... --output_dir samples_ddp --video_id <vid>
            # On failure or if the JSON still isn't produced we raise — this is a
            # hard prerequisite for Stages 4-6.
            print(f"No camera_ray_depth_offset.json at: {camera_ray_depth_offset_path}")
            print("Auto-running GraspFlowMatching pre-pass to generate it...")

            _gfm_dir = "../../stage2_grasp_correction/GraspFlowMatching"
            _hd_bin = os.environ.get("CHOIR_HD_BIN", "")
            _video_id = os.path.basename(seq_path)
            _gfm_seq_dir_name = "grasp_correction/gfm_input_hoi_seq"

            _gfm_input_dir = export_graspflowmatching_sequence(
                seq_path=output_path,
                sampled_indices=sampled_indices,
                canonical_verts=verts,
                canonical_faces=faces,
                obj_rot_mats=rotation_6d_to_matrix(multi_frame_model.rot_6d).detach().transpose(1, 2),
                obj_trans=multi_frame_model.trans.detach(),
                obj_scale=(multi_frame_model.scale * multi_frame_model.initial_scale).detach(),
                mano_root_orient=multi_frame_model.mano_root_orient.detach(),
                mano_pose=multi_frame_model.mano_pose.detach(),
                mano_trans=multi_frame_model.mano_trans.detach(),
                is_right=multi_frame_model.is_right.detach(),
                output_dir_name=_gfm_seq_dir_name,
                clean_output=True,
            )
            print(f"[GraspCorrection] Prepared GraspFlowMatching input at {_gfm_input_dir}")

            # Restrict the sub-process to the current worker's single GPU.
            # Why: the parent demo runs one worker per GPU via mp.spawn (each worker
            # owns one device id and processes its own slice of sequences).  If we
            # let the GraspFlowMatching DDP command grab `torch.cuda.device_count()`
            # GPUs it would step on the other workers and OOM.  So we (a) pick the
            # current worker's device id, mapped through the parent's
            # CUDA_VISIBLE_DEVICES if set, (b) pin CUDA_VISIBLE_DEVICES to that
            # single id in the child, and (c) force --nproc_per_node=1.
            # subprocess.run() already blocks until the child exits, so the parent
            # only proceeds to Stage 4 after both commands have finished.
            _local_dev = torch.cuda.current_device()
            _parent_cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
            if _parent_cvd is not None and _parent_cvd.strip() != "":
                _visible = [tok.strip() for tok in _parent_cvd.split(",") if tok.strip() != ""]
                _real_dev = _visible[_local_dev] if _local_dev < len(_visible) else str(_local_dev)
            else:
                _real_dev = str(_local_dev)

            _gfm_env = os.environ.copy()
            _gfm_env["PATH"] = _hd_bin + os.pathsep + _gfm_env.get("PATH", "")
            _gfm_env["CUDA_VISIBLE_DEVICES"] = _real_dev

            # Per-worker master port to avoid `torchrun` collisions when 8 demo
            # workers trigger `sample_cam_ray_ddp.py` at the same time (default
            # port 29500 ⇒ "Address already in use" on all but the first).
            # Offset by _local_dev (0..N-1) so each worker gets a unique port.
            _master_port = 29500 + int(_local_dev)

            _cmd_prep = [os.path.join(_hd_bin, "python"),
                         "prepare_meshdata.py",
                         "--video_id", _video_id,
                         "--seq_dir_name", _gfm_seq_dir_name]
            _cmd_sample = [os.path.join(_hd_bin, "torchrun"),
                           "--nnodes=1", "--nproc_per_node=1",
                           f"--master_port={_master_port}",
                           "sample_cam_ray_ddp.py", "ODE",
                           "--ckpt", "results/050-Linear-velocity-None/checkpoints/0040000.pt",
                           "--output_dir", "samples_ddp",
                           "--video_id", _video_id,
                           "--seq_dir_name", _gfm_seq_dir_name]

            for _cmd in (_cmd_prep, _cmd_sample):
                print(f"[GraspCorrection] cwd={_gfm_dir} CUDA_VISIBLE_DEVICES={_real_dev} master_port={_master_port}")
                print(f"[GraspCorrection] cmd={' '.join(_cmd)}")
                _ret = subprocess.run(_cmd, cwd=_gfm_dir, env=_gfm_env)
                if _ret.returncode != 0:
                    raise RuntimeError(
                        f"GraspFlowMatching command failed (exit={_ret.returncode}): "
                        f"{' '.join(_cmd)}"
                    )

            if not os.path.exists(camera_ray_depth_offset_path):
                _source_offset_path = os.path.join(seq_path, "grasp_correction", "camera_ray_depth_offset.json")
                if os.path.exists(_source_offset_path):
                    os.makedirs(grasp_correction_dir, exist_ok=True)
                    shutil.copy2(_source_offset_path, camera_ray_depth_offset_path)
                    print(
                        "[GraspCorrection] External GFM completed without writing to the ablation output; "
                        f"copied existing source offset from {_source_offset_path}"
                    )
                else:
                    _source_offset_path = None
            if not os.path.exists(camera_ray_depth_offset_path):
                raise FileNotFoundError(
                    f"GraspFlowMatching commands completed but did not produce "
                    f"camera_ray_depth_offset.json at {camera_ray_depth_offset_path}"
                )
            print(f"[GraspCorrection] Generated {camera_ray_depth_offset_path}, continuing.")

        # JSON now guaranteed to exist (was already there OR just generated above).
        print(f"Found camera_ray_depth_offset.json at: {camera_ray_depth_offset_path}")
        # Load contact-vertex subset (used by Stage 4 estimator and Stage 5/6 constraints).
        contact_indices_override = load_contact_indices(contact_indices_path)
        if contact_indices_override is not None:
            print(f"Loaded {len(contact_indices_override)} contact indices from {contact_indices_path}")

        print("Estimating per-frame contact map from Stage 3 hand/object meshes...")

        with torch.no_grad():
            obj_meshes_s4, hand_meshes_s4, _ = multi_frame_model()
            hand_verts_s4 = hand_meshes_s4.verts_padded()
            hand_normals_s4 = hand_meshes_s4.verts_normals_padded()
            hand_faces_s4 = hand_meshes_s4.faces_padded()[0]
            obj_verts_s4 = obj_meshes_s4.verts_padded()
            hand_verts_before_corr_s4 = hand_verts_s4.clone()
            mano_trans_s4 = multi_frame_model.mano_trans.detach()
            mano_root_orient_s4 = multi_frame_model.mano_root_orient.detach()
            mano_pose_s4 = multi_frame_model.mano_pose.detach()
            mano_is_right_s4 = multi_frame_model.is_right.detach().to(mano_trans_s4.device)
            # Camera ray for correction (pre-MANO space, matches training-time
            # depth_mag convention: hand shifts along normalize(-mano_trans)).
            camera_rays_s4 = -mano_trans_s4
            _flat_mat = torch.diag(torch.tensor([-1.0, -1.0, 1.0], dtype=mano_trans_s4.dtype, device=mano_trans_s4.device))

            # Default root joints for debug visualisation (mesh space, post-flat),
            # taken from the un-corrected forward pass. Overwritten below if a
            # correction is applied.
            with torch.no_grad():
                _amano_init = run_amano(
                    multi_frame_model.hand_model_amano,
                    mano_trans_s4[None],
                    mano_root_orient_s4[None],
                    mano_pose_s4[None],
                    mano_is_right_s4,
                )
                _N_init = mano_trans_s4.shape[0]
                _flat_mat_batch_init = _flat_mat.unsqueeze(0).expand(_N_init, -1, -1)
                root_joints_s4 = (_amano_init['joints'].squeeze(0) @ _flat_mat_batch_init)[:, 0, :]

            if ablation == "no_stage4_offset":
                print("[Ablation] Skipping Stage 4 camera-ray depth offset application.")
            else:
                # Apply hand translation correction before contact-map estimation
                # (camera_ray_depth_offset.json is guaranteed to exist at this point).
                with open(camera_ray_depth_offset_path, 'r') as _f:
                    depth_offset_raw = json.load(_f)

                if isinstance(depth_offset_raw, dict) and len(depth_offset_raw) > 0:
                    mano_trans_corr_s4, _gfm_offset_stats_s4 = apply_camera_ray_depth_offsets(
                        mano_trans_s4,
                        sampled_indices,
                        depth_offset_raw,
                        smooth_offsets=bool(smooth_gfm_depth_offset),
                        smooth_range=(int(approaching_end_idx), int(interaction_end_idx)),
                    )

                    if _gfm_offset_stats_s4["applied_count"] > 0:
                        # Persist the Stage 4 GFM correction into the model state.
                        # Stage 5 pose_only freezes mano_trans, so without this copy
                        # it would optimize/render from the pre-GFM hand position
                        # while contact was estimated from corrected hand vertices.
                        multi_frame_model.mano_trans.data.copy_(mano_trans_corr_s4)
                        amano_output_corr = run_amano(
                            multi_frame_model.hand_model_amano,
                            mano_trans_corr_s4[None],
                            mano_root_orient_s4[None],
                            mano_pose_s4[None],
                            mano_is_right_s4,
                        )
                        _N_s4 = mano_trans_corr_s4.shape[0]
                        _flat_mat_batch = _flat_mat.unsqueeze(0).expand(_N_s4, -1, -1)
                        mano_verts_corr = amano_output_corr['vertices'].squeeze(0) @ _flat_mat_batch
                        mano_joints_corr = amano_output_corr['joints'].squeeze(0) @ _flat_mat_batch
                        root_joints_s4 = mano_joints_corr[:, 0, :]
                        mano_l_faces = amano_output_corr['l_faces']
                        mano_r_faces = amano_output_corr['r_faces']
                        mano_is_right = amano_output_corr['is_right'].squeeze(0)
                        mano_faces = mano_r_faces if mano_is_right[0].item() > 0 else mano_l_faces
                        mano_faces = mano_faces[None].repeat(_N_s4, 1, 1)
                        mano_textures = TexturesVertex(verts_features=torch.ones_like(mano_verts_corr))
                        hand_meshes_corr_s4 = Meshes(verts=mano_verts_corr, faces=mano_faces, textures=mano_textures)
                        hand_verts_s4 = hand_meshes_corr_s4.verts_padded()
                        hand_normals_s4 = hand_meshes_corr_s4.verts_normals_padded()
                        hand_faces_s4 = hand_meshes_corr_s4.faces_padded()[0]

                    print(
                        f"Applied hand translation correction from {camera_ray_depth_offset_path}: "
                        f"{_gfm_offset_stats_s4['applied_count']}/{_gfm_offset_stats_s4['total_count']} sampled frames "
                        f"(frame_key={_gfm_offset_stats_s4['matched_by_frame']}, "
                        f"pos_key={_gfm_offset_stats_s4['matched_by_position']}, "
                        f"missing={_gfm_offset_stats_s4['missing_count']}, "
                        f"mean_abs={_gfm_offset_stats_s4['mean_abs_depth_offset'] * 1000:.1f}mm, "
                        f"max_abs={_gfm_offset_stats_s4['max_abs_depth_offset'] * 1000:.1f}mm, "
                        f"smoothed={_gfm_offset_stats_s4.get('smoothed', False)})"
                    )
                    with open(os.path.join(grasp_correction_dir, "camera_ray_depth_offset_apply_debug.json"), "w") as _dbg_f:
                        json.dump(_gfm_offset_stats_s4, _dbg_f, indent=2)
                    if _gfm_offset_stats_s4["skipped_invalid_ray_count"] > 0:
                        print(
                            f"Skipped {_gfm_offset_stats_s4['skipped_invalid_ray_count']} frames "
                            "due to near-zero mano_trans (undefined camera ray)"
                        )
                else:
                    print(f"camera_ray_depth_offset.json is empty or invalid at {camera_ray_depth_offset_path}")

        # Estimate contact only in interaction segment [approaching_end_idx, interaction_end_idx).
        interaction_start = int(approaching_end_idx)
        interaction_end = int(interaction_end_idx)
        if save_intermediates:
            save_stage4_correction_debug_meshes(
                debug_dir=os.path.join(grasp_correction_dir, "stage4_gfm_offset_debug_meshes"),
                hand_verts_before=hand_verts_before_corr_s4,
                hand_verts_after=hand_verts_s4,
                obj_verts=obj_verts_s4,
                obj_faces=faces,
                hand_faces=hand_faces_s4,
                sampled_indices=sampled_indices,
                interaction_start=interaction_start,
                interaction_end=interaction_end,
            )
        if disable_contact_terms:
            print("[Ablation] no_contact: skipping Stage 4 contact-map estimation.")
            parsed_contact_map = {"__empty__": []}
            return
        if save_intermediates and save_contact_debug_meshes_flag:
            debug_mesh_dir = os.path.join(grasp_correction_dir, "contact_map_debug_meshes")
            save_contact_debug_meshes(
                debug_dir=debug_mesh_dir,
                hand_verts_before=hand_verts_before_corr_s4,
                hand_verts_after=hand_verts_s4,
                obj_verts_after=obj_verts_s4,
                obj_faces=faces,
                hand_faces=hand_faces_s4,
                ray_end_points=root_joints_s4,
                sampled_indices=sampled_indices,
                interaction_start=interaction_start,
                interaction_end=interaction_end,
            )
        if interaction_end <= interaction_start:
            print("Interaction segment is empty; writing empty contact map.")
            contact_map_raw = []
        else:
            print(
                f"Estimating contact only on interaction sampled positions "
                f"[{interaction_start}, {interaction_end})"
            )
            sampled_indices_interaction = [int(v) for v in sampled_indices[interaction_start:interaction_end]]
            contact_map_raw = estimate_contact_map_for_sampled_frames(
                hand_verts_seq_world=hand_verts_s4[interaction_start:interaction_end],
                hand_normals_seq_world=hand_normals_s4[interaction_start:interaction_end],
                obj_verts_seq_world=obj_verts_s4[interaction_start:interaction_end],
                obj_faces=faces,
                sampled_indices=sampled_indices_interaction,
                contact_hand_indices_override=contact_indices_override,
                cone_angle_deg=float(contact_cone_angle_deg),
                dist_thresh=float(contact_dist_thresh),
                n_surface_samples=int(contact_surface_samples),
            )
        print(f"Computed per-frame contact map in-memory ({len(contact_map_raw)} frames).")

        per_frame_contact = parse_per_frame_contact_map(contact_map_raw, sampled_indices)
        total_corr = sum(len(v) for v in per_frame_contact.values())
        print(f"Loaded {len(per_frame_contact)} frames with contact, {total_corr} total correspondences")
        for pos, corr in list(per_frame_contact.items())[:3]:
            fi = sampled_indices[pos]
            print(f"  sampled pos {pos} (frame {fi}): {len(corr)} correspondences")
        if len(per_frame_contact) > 3:
            print(f"  ... and {len(per_frame_contact) - 3} more frames")

        parsed_contact_map = per_frame_contact  # unified name for downstream check


    _reclassify_log_file_s5 = None
    _prefix_diag_file_s5 = None
    _reclassify_log_path_s5 = os.path.join(output_path, "optimize_stage5_reclassify.log")
    _prefix_diag_path_s5 = os.path.join(output_path, "optimize_stage5_prefix_diag.csv")
    _current_reclassify_step_s5 = -1

    def _log_reclassify(msg: str):
        print(msg)
        if _reclassify_log_file_s5 is not None:
            _reclassify_log_file_s5.write(msg + "\n")


    def _run_stage5():
        nonlocal _reclassify_log_file_s5, _prefix_diag_file_s5, _current_reclassify_step_s5
        nonlocal _reclassify_log_path_s5, _prefix_diag_path_s5
        _stage5_mode = str(stage5_mode)
        _stage5_trainable = stage5_trainable_params(_stage5_mode)
        _stage5_frozen = _stage5_mode == STAGE5_FROZEN
        _stage5_object_lite = _stage5_mode == STAGE5_OBJECT_LITE
        _stage5_pose_only = _stage5_mode == STAGE5_POSE_ONLY
        _stage5_pose_ray = _stage5_mode == STAGE5_POSE_RAY
        _stage5_global_locked = _stage5_frozen or _stage5_pose_only or _stage5_pose_ray
        _stage5_hand_global_locked = _stage5_frozen or _stage5_pose_only or _stage5_pose_ray or _stage5_object_lite
        # ================================================================
        # STAGE 5 — Penetration Resolution (主穿模, 副 contact)
        # ================================================================
        # If Stage 3 + ray-scale alignment + camera_ray_depth_offset already
        # produce a satisfactory hand-object configuration, Stage 5 can do
        # more harm than good (pen loss pushes hand away from a thin object,
        # contact can't pull it back, equilibrium drifts).  Set this to False
        # to export the post-offset state as the final result.
        ENABLE_STAGE5 = not _stage5_frozen
        print("\n" + "=" * 80)
        print("STAGE 5: Penetration Resolution"
              + (" [SKIPPED]" if not ENABLE_STAGE5 else f" [{_stage5_mode}]"))
        print("=" * 80)
        if not ENABLE_STAGE5:
            print("Stage 5 disabled — using Stage 4/post-offset result as final.")
        else:
            print(f"Stage 5 trainable groups: {', '.join(_stage5_trainable)}")

        # Stage 5 optimization scope:
        #
        # Modes:
        # - full: current behavior. Object 6DoF, hand wrist/root, and fingers
        #   can all move under anchors/smoothness.
        # - object_lite: keep the Stage 4 smoothed hand root fixed while
        #   allowing small object 6DoF refinement under mask + anchor losses.
        # - pose_only: freeze the Stage 4 GFM-corrected object pose and global
        #   hand pose; only mano_pose changes to resolve penetration/contact.
        # - pose_ray: same global freeze as pose_only, but adds a bounded
        #   per-frame scalar hand-depth delta along the camera ray. This lets
        #   penetration/contact losses correct residual GFM depth errors without
        #   unlocking arbitrary wrist translation or object motion.
        #
        # - full keeps object 6DoF and hand wrist/root free with anchors.
        # - object_lite locks wrist/root but keeps object 6DoF live.
        # - pose_only/pose_ray lock object 6DoF and wrist/root; pose_ray's
        #   bounded ray_delta is the only global hand translation adjustment.
        # - Hand fingers (mano_pose): free.
        multi_frame_model.rot_6d.requires_grad_(not _stage5_global_locked)
        multi_frame_model.trans.requires_grad_(not _stage5_global_locked)
        multi_frame_model.mano_root_orient.requires_grad_(not _stage5_hand_global_locked)
        multi_frame_model.mano_trans.requires_grad_(not _stage5_hand_global_locked)
        multi_frame_model.mano_pose.requires_grad_(not _stage5_frozen)

        _stage5_ray_delta = None
        _stage5_ray_dirs = None
        _stage5_ray_inter_mask = None
        _stage5_ray_params: list[torch.Tensor] = []
        if _stage5_pose_ray:
            _stage5_ray_delta = nn.Parameter(torch.zeros(
                num_sampled_frames,
                dtype=multi_frame_model.mano_trans.dtype,
                device=device,
            ))
            _stage5_ray_dirs = torch.nn.functional.normalize(
                -_anchor_mano_trans.detach(), dim=-1, eps=1e-8)
            _stage5_ray_inter_mask = torch.zeros(num_sampled_frames, dtype=torch.bool, device=device)
            if _has_inter:
                _stage5_ray_inter_mask[i0:i1] = True
            _stage5_ray_params = [_stage5_ray_delta]

        _stage5_param_groups = [] if _stage5_frozen else [{'params': [multi_frame_model.mano_pose], 'lr': 5e-4}]
        if not _stage5_global_locked:
            _stage5_param_groups.insert(0, {
                'params': [multi_frame_model.rot_6d, multi_frame_model.trans],
                'lr': 5e-5 if _stage5_object_lite else 3e-4,
            })
        if not _stage5_hand_global_locked:
            # Wrist gets 10x smaller LR than fingers — moves only when contact
            # / pen really need it; smoothness + anchor do the rest.
            _stage5_param_groups.append({
                'params': [multi_frame_model.mano_root_orient, multi_frame_model.mano_trans],
                'lr': 5e-5,
            })
        if _stage5_pose_ray:
            _stage5_param_groups.append({'params': _stage5_ray_params, 'lr': 2e-4})
        optimizer_stage5 = torch.optim.Adam(_stage5_param_groups) if _stage5_param_groups else None

        num_steps_s5 = 800 if ENABLE_STAGE5 else 0
        # Bumped 2e2 → 5e2: with the new (0,0,1) obj_grad_scale, hand fingers
        # are pulled in xy toward the object but the object stays fixed in xy,
        # so fingers can slide into the object surface rather than around it.
        # Stronger pen guard counteracts that without re-introducing the old
        # 5e2 problems because the dead_zone now absorbs sub-mm noise.
        LAMBDA_PEN_S5 = 5e2
        # Dead-zone for penetration: depths < this (metres) contribute zero
        # loss/gradient.  Allows finger wrap-around for thin objects (handles,
        # pens) without sacrificing the deep-pen guard.  5mm covers MANO skin
        # thickness + typical handle-radius slack.
        PEN_DEAD_ZONE_S5 = 0.005
        # Contact bumped 5e1 → 1e2 → 1e3.  At 1e2, contact_loss only contributed
        # ~1.5% of total loss while hand_2d held 56%; mean_ct_mm stayed at
        # 34mm for 500 steps (active verts couldn't be pulled in).  10x
        # boost makes contact roughly comparable to hand_2d in magnitude.
        LAMBDA_CONTACT_S5 = 1e3
        # Conservative gap-closing pass for high-confidence temporal contacts.
        # This is separate from sticky contact: it only activates for visible
        # gaps and uses limited object gradients so RGB pose is not dragged far.
        LAMBDA_GAP_CLOSE_S5 = 8e2
        LAMBDA_PATCH_CLOSE_S5 = 5e2
        # Diagnostic: template loss is metre^2, so 1e3 only contributed ~0.05
        # with 12mm gaps.  Raise it to compete with silhouette/pose priors.
        LAMBDA_TEMPLATE_ANCHOR_S5 = 3e4
        # Experimental bridge for early/late interaction frames that are near
        # the frozen reliable template but failed frontier expansion.  It does
        # not alter reliable_frames; it only adds a decayed one-way pull.
        LAMBDA_NEIGHBOR_TEMPLATE_BRIDGE_S5 = 2.5e4
        LAMBDA_BOOTSTRAP_GAP_S5 = 6e2
        NEIGHBOR_TEMPLATE_BRIDGE_MAX_DIST_S5 = 12
        NEIGHBOR_TEMPLATE_BRIDGE_CONF_THRESH_S5 = 0.035
        # Ordinary dynamic correspondences can be wrong when object pose is off.
        # Let them refine finger pose only; object translation is driven by the
        # frozen reliable grasp template below.
        CONTACT_OBJ_GRAD_SCALE_S5 = (0.2, 0.2, 0.6) if _stage5_object_lite else (0.0, 0.0, 0.0)
        GAP_OBJ_GRAD_SCALE_S5 = (0.25, 0.25, 0.6) if _stage5_object_lite else (0.0, 0.0, 0.0)
        PATCH_OBJ_GRAD_SCALE_S5 = (0.3, 0.3, 0.8) if _stage5_object_lite else (0.0, 0.0, 0.0)
        OBJECT_LITE_MAX_TRANS_DELTA_S5 = DEFAULT_OBJECT_LITE_MAX_TRANS_DELTA
        TEMPLATE_OBJ_GRAD_SCALE_S5 = (1.0, 1.0, 1.0)
        NEIGHBOR_TEMPLATE_BRIDGE_OBJ_GRAD_SCALE_S5 = (0.05, 0.05, 0.6)
        # Keep hand projection close to 2D evidence in full mode.  pose_ray
        # disables this so depth is driven only by 3D geometry terms.
        LAMBDA_HAND2D_S5 = stage5_hand2d_weight(_stage5_mode)
        # HaMeR mano_pose anchor — keeps fingers near HaMeR's 3D-regressed
        # articulation when joints_2d is unreliable.  axis-angle MSE is
        # typically ~1e-2 per element; with 1e2 weight the anchor term
        # contributes ~1.0, comparable to hand_2d_loss (~5 raw × 5e-1 ≈ 2-3).
        LAMBDA_MANO_POSE_ANCHOR_S5 = 1e2
        LAMBDA_OBJ_SIL_S5 = 0.0 if _stage5_global_locked else 5e2
        LAMBDA_OBJ_ANCHOR_NI_S5 = 0.0 if _stage5_global_locked else (5e2 if _stage5_object_lite else 1e2)
        # Bumped 3e1 → 1e2: anchor to the post-offset object pose during
        # interaction so pen/contact tug-of-war can't drift the object far
        # from the validated Stage 3 + ray-scale + depth-offset state.
        LAMBDA_OBJ_ANCHOR_INTER_S5 = 0.0 if _stage5_global_locked else (5e2 if _stage5_object_lite else 1e2)
        # Bumped 1e1 → 3e1: 2D keypoint detector is unreliable in heavy
        # occlusion / motion-blur frames and can request anatomically
        # impossible joint angles.  Stronger anatomy keeps the hand in a
        # plausible configuration even when joints_2d demands otherwise.
        # Combined with the stronger HaMeR mano_pose anchor, this provides
        # a two-layer guard (HaMeR pose prior + biomechanical limits).
        LAMBDA_ANATOMY_S5 = 3e1
        # 1st-order (velocity) — penalises any motion. Reduced because we now
        # also have 2nd-order which is the more selective tool for jitter.
        # 2026-04 update: lowered 1e3 → 5e2 to let finger articulation
        # respond to contact attraction (was creating mid-interaction gap
        # because fingers couldn't deform fast enough to reach contact targets).
        LAMBDA_POSE_SMOOTH_S5 = 5e2
        LAMBDA_OBJ_SMOOTH_S5 = 0.0 if _stage5_global_locked else 5e2
        # Wrist guards.
        # 2026-04 update: lowered 8e2 → 2e2.  With contact_loss=1e3 and
        # detach_object=True, all the contact gradient flows into the hand
        # — but a heavy wrist anchor pinned the wrist so finger articulation
        # alone couldn't reach contact points 5-10mm away.  Loosening
        # wrist_anchor lets the wrist nudge alongside contact;
        # 2nd-order accel (also reduced) still kills jitter.
        LAMBDA_WRIST_ANCHOR_S5 = 0.0 if _stage5_hand_global_locked else 2e2
        LAMBDA_HAND_TR_SMOOTH_S5 = 0.0 if _stage5_hand_global_locked else 2e2
        LAMBDA_ROOT_R_SMOOTH_S5 = 0.0 if _stage5_hand_global_locked else 5e2         # was 1e3, rotation matrix Frobenius (wrap-around safe)
        # 2nd-order (acceleration) — penalises jitter, allows constant-velocity
        # motion. Generally more effective than 1st-order for de-jittering.
        # 2026-04 update: pose_accel kept at 1e3 (fingers should stay
        # responsive to contact); hand_tr_accel & root_R_accel bumped back
        # toward original values because wrist jitter returned after the
        # too-aggressive cut to 1e3.  wrist_anchor stays at 2e2 so wrist
        # can still drift to follow contact, but accel kills high-freq jitter.
        LAMBDA_POSE_ACCEL_S5 = 1e3
        LAMBDA_HAND_TR_ACCEL_S5 = 0.0 if _stage5_hand_global_locked else 5e3
        LAMBDA_ROOT_R_ACCEL_S5 = 0.0 if _stage5_hand_global_locked else 3e3
        LAMBDA_RAY_DELTA_ANCHOR_S5 = 2e3 if _stage5_pose_ray else 0.0
        LAMBDA_RAY_DELTA_SMOOTH_S5 = 5e3 if _stage5_pose_ray else 0.0
        LAMBDA_RAY_DELTA_ACCEL_S5 = 2e4 if _stage5_pose_ray else 0.0

        if disable_penetration_terms:
            LAMBDA_PEN_S5 = 0.0
        if disable_contact_terms:
            LAMBDA_CONTACT_S5 = 0.0
            LAMBDA_GAP_CLOSE_S5 = 0.0
            LAMBDA_PATCH_CLOSE_S5 = 0.0
            LAMBDA_TEMPLATE_ANCHOR_S5 = 0.0
            LAMBDA_NEIGHBOR_TEMPLATE_BRIDGE_S5 = 0.0
            LAMBDA_BOOTSTRAP_GAP_S5 = 0.0
        if ablation == "no_stage5_smooth":
            LAMBDA_POSE_SMOOTH_S5 = 0.0
            LAMBDA_OBJ_SMOOTH_S5 = 0.0
            LAMBDA_HAND_TR_SMOOTH_S5 = 0.0
            LAMBDA_ROOT_R_SMOOTH_S5 = 0.0
            LAMBDA_POSE_ACCEL_S5 = 0.0
            LAMBDA_HAND_TR_ACCEL_S5 = 0.0
            LAMBDA_ROOT_R_ACCEL_S5 = 0.0
            LAMBDA_RAY_DELTA_SMOOTH_S5 = 0.0
            LAMBDA_RAY_DELTA_ACCEL_S5 = 0.0

        _log_cols_s5 = ["step", "total", "pen", "contact", "template_anchor", "template_anchor_raw",
                        "neighbor_bridge", "neighbor_bridge_raw",
                        "bootstrap_gap",
                        "template_frozen",
                        "relax_mean", "hand2d_relaxed",
                        "hand2d_anatomy_gate_mean", "hand2d_anatomy_gate_min", "hand2d_anatomy_gate_n_low",
                        "obj_sil_relaxed", "obj_anchor_relaxed",
                        "hand_2d", "anatomy",
                        "mano_pose_anchor",
                        "obj_sil", "obj_anchor", "obj_anchor_rot_inter", "obj_anchor_tr_inter",
                        "gap_close", "patch_close",
                        "pose_sm", "obj_sm",
                        "wrist_anchor", "tr_sm", "root_R_sm",
                        "pose_acc", "tr_acc", "root_R_acc",
                        "ray_anchor", "ray_sm", "ray_acc",
                        "ray_delta_mean_mm", "ray_delta_min_mm", "ray_delta_max_mm",
                        "n_inside", "mean_pen_mm", "max_pen_mm",
                        "n_active", "mean_ct_mm",
                        "n_template_active", "mean_template_gap_mm", "mean_template_conf",
                        "n_bridge_active", "n_bridge_frames", "mean_bridge_gap_mm", "mean_bridge_conf",
                        "n_gap_active", "mean_gap_mm",
                        "n_patch_active", "mean_patch_mm", "mean_patch_contacts"]
        _log_path_s5 = os.path.join(output_path, "optimize_stage5.log")
        _log_file_s5 = open(_log_path_s5, "w", buffering=1)
        _log_file_s5.write("# Stage 5 — Penetration Resolution (one-sided push-out + sticky contact)\n")
        _log_file_s5.write(f"# mode={_stage5_mode}, trainable={','.join(_stage5_trainable)}\n")
        _log_file_s5.write(",".join(_log_cols_s5) + "\n")
        with open(os.path.join(output_path, "stage5_mode_debug.json"), "w") as _f:
            json.dump({
                "ablation": str(ablation),
                "stage5_mode": _stage5_mode,
                "stage5_presmooth_object": bool(stage5_presmooth_object),
                "stage5_presmooth_sigma": _stage5_presmooth_sigma,
                "trainable_groups": list(_stage5_trainable),
                "frozen": bool(_stage5_frozen),
                "object_lite": bool(_stage5_object_lite),
                "pose_only": bool(_stage5_pose_only),
                "pose_ray": bool(_stage5_pose_ray),
                "ray_delta_bounds_m": [
                    float(DEFAULT_STAGE5_RAY_MIN_DELTA),
                    float(DEFAULT_STAGE5_RAY_MAX_DELTA),
                ] if _stage5_pose_ray else None,
                "ray_delta_active_frames": int(_stage5_ray_inter_mask.sum().item()) if _stage5_ray_inter_mask is not None else 0,
                "hand2d_weight": float(LAMBDA_HAND2D_S5),
                "hand2d_grad_groups": list(stage5_hand2d_grad_groups(_stage5_mode)),
                "contact_obj_grad_scale": list(CONTACT_OBJ_GRAD_SCALE_S5),
                "gap_obj_grad_scale": list(GAP_OBJ_GRAD_SCALE_S5),
                "patch_obj_grad_scale": list(PATCH_OBJ_GRAD_SCALE_S5),
                "object_lite_max_trans_delta_m": (
                    float(OBJECT_LITE_MAX_TRANS_DELTA_S5) if _stage5_object_lite else None
                ),
                "disabled_losses_in_global_locked_mode": (
                    (["hand_2d"] if LAMBDA_HAND2D_S5 <= 0 else []) + [
                        "wrist_anchor",
                        "wrist_smooth",
                        "wrist_accel",
                    ] + ([] if not _stage5_global_locked else [
                        "object_silhouette",
                        "object_anchor",
                        "object_smooth",
                    ]
                    )
                ) if (_stage5_global_locked or _stage5_hand_global_locked) else [],
                "object_rotation_smoothness": "disabled",
            }, _f, indent=2)
        _reclassify_log_file_s5 = open(_reclassify_log_path_s5, "w", buffering=1)
        _reclassify_log_file_s5.write("# Stage 5 reclassify diagnostics\n")
        _prefix_diag_file_s5 = open(_prefix_diag_path_s5, "w", buffering=1)
        _prefix_diag_cols_s5 = [
            "step", "local", "sampled", "first_reliable_local", "strict", "weak",
            "n_conf035", "n_conf045", "mean_conf035", "max_conf",
            "template_valid", "overlap035", "overlap_ratio035", "face_agree035",
            "mean_template_gap_mm", "frontier_pass_overlap", "frontier_pass_face",
        ]
        _prefix_diag_file_s5.write(",".join(_prefix_diag_cols_s5) + "\n")

        contact_cache = None
        prev_argmax_face_full = None

        loop_s5 = tqdm(range(num_steps_s5), desc="Stage 5: Penetration Resolution")
        for step in loop_s5:
            _should_reclassify_s5 = (
                _has_inter
                and not disable_contact_terms
                and (
                    (ablation == "no_dyn_contact" and contact_cache is None)
                    or (ablation != "no_dyn_contact" and (step % _k_reclassify_s5(step) == 0))
                )
            )
            if _should_reclassify_s5:
                _current_reclassify_step_s5 = int(step)
                contact_cache, prev_argmax_face_full = _do_reclassify(prev_argmax_face_full)

            optimizer_stage5.zero_grad()

            obj_meshes_s5, hand_meshes_s5, hand_joints_s5 = multi_frame_model()
            hand_verts_s5 = hand_meshes_s5.verts_padded()
            obj_verts_s5  = obj_meshes_s5.verts_padded()
            _mano_trans_eff_s5 = multi_frame_model.mano_trans
            _stage5_ray_delta_scalar = torch.zeros(num_sampled_frames, dtype=hand_verts_s5.dtype, device=device)
            if _stage5_pose_ray and _stage5_ray_delta is not None:
                _ray_delta_vec_s5, _stage5_ray_delta_scalar = build_stage5_ray_delta(
                    _stage5_ray_delta,
                    _stage5_ray_dirs,
                    _stage5_ray_inter_mask,
                )
                _mano_trans_eff_s5 = _anchor_mano_trans + _ray_delta_vec_s5
                _flat_s5 = torch.diag(torch.tensor(
                    [-1.0, -1.0, 1.0],
                    dtype=hand_verts_s5.dtype,
                    device=device,
                )).unsqueeze(0).expand(num_sampled_frames, -1, -1)
                _amano_eff_s5 = run_amano(
                    multi_frame_model.hand_model_amano,
                    _mano_trans_eff_s5[None],
                    multi_frame_model.mano_root_orient[None],
                    multi_frame_model.mano_pose[None],
                    multi_frame_model.is_right.to(device),
                )
                hand_joints_s5 = _amano_eff_s5['joints'].squeeze(0) @ _flat_s5
                hand_verts_s5 = _amano_eff_s5['vertices'].squeeze(0) @ _flat_s5
                multi_frame_model.transforms_abs = _amano_eff_s5['transforms_abs']

            pen_info_s5 = {'n_inside': 0, 'mean_pen_depth': 0.0, 'max_pen_depth': 0.0}
            ct_info_s5  = {'n_active': 0, 'mean_contact_dist': 0.0}
            template_info_s5 = {'n_template_active': 0, 'mean_template_gap_mm': 0.0, 'mean_template_conf': 0.0}
            bridge_info_s5 = {'n_bridge_active': 0, 'n_bridge_frames': 0,
                              'mean_bridge_gap_mm': 0.0, 'mean_bridge_conf': 0.0}
            gap_info_s5 = {'n_gap_active': 0, 'mean_gap_dist': 0.0}
            patch_info_s5 = {'n_patch_active': 0, 'mean_patch_dist': 0.0, 'mean_patch_contacts': 0.0}
            template_anchor_s5_raw = torch.tensor(0.0, device=device)
            neighbor_bridge_s5_raw = torch.tensor(0.0, device=device)
            bootstrap_gap_s5 = torch.tensor(0.0, device=device)
            contact_relax_i = torch.zeros(max(N_inter, 1), dtype=torch.float32, device=device)
            contact_relax_full = torch.zeros(num_sampled_frames, dtype=torch.float32, device=device)
            hand2d_relaxed_s5 = torch.tensor(0.0, device=device)
            obj_sil_relaxed_s5 = torch.tensor(0.0, device=device)
            obj_anchor_relaxed_s5 = torch.tensor(0.0, device=device)
            obj_anchor_rot_inter_s5 = torch.tensor(0.0, device=device)
            obj_anchor_tr_inter_s5 = torch.tensor(0.0, device=device)

            # Penetration loss applied to ALL frames (not just [i0, i1)).
            # Approaching/releasing/static segments often suffer from real
            # 3D penetration due to Stage 3's per-frame independent
            # optimisation (no inter-frame geometric guard).  Pen loss is
            # purely repulsive (pushes hand vertices that are inside the
            # object back to its surface), so it can only help — frames
            # where the hand is not penetrating contribute zero gradient.
            _on_full = _compute_obj_normals_with_grad(obj_verts_s5)
            pen_loss_s5_raw, pen_info_s5 = compute_penetration_loss(
                hand_verts_s5, obj_verts_s5, _on_full, detach_object=True,
                dead_zone=PEN_DEAD_ZONE_S5)
            pen_loss_s5 = pen_loss_s5_raw * LAMBDA_PEN_S5

            # Contact loss stays restricted to interaction frames — contact
            # correspondences only exist there.  Per-axis obj gradient:
            # xy=0.0 (silhouette already constrains xy mask alignment, so
            # contact must NOT pull the object in xy or it drifts off
            # mask), z=1.0 (silhouette doesn't constrain depth at all, so
            # contact freely pulls the object in z to follow the hand).
            # This was the missing piece — when obj_grad_scale was a
            # single scalar (=0.3 isotropic), even a small xy pull
            # accumulated over 500 steps and drifted the object visibly
            # off mask.
            if _has_inter and contact_cache is not None and not disable_contact_terms:
                _hv_i = hand_verts_s5[i0:i1]
                _ov_i = obj_verts_s5[i0:i1]
                # Recompute hand verts for contact losses with global hand
                # translation/root rotation detached, but finger pose live.
                # This preserves the same values while routing contact gradients
                # to mano_pose (fine grasp shape), not global hand trajectory.
                _flat_i = torch.diag(torch.tensor(
                    [-1.0, -1.0, 1.0],
                    dtype=hand_verts_s5.dtype,
                    device=device)).unsqueeze(0).expand(i1 - i0, -1, -1)
                _contact_trans_i = (
                    _mano_trans_eff_s5[i0:i1]
                    if _stage5_pose_ray
                    else multi_frame_model.mano_trans[i0:i1].detach()
                )
                _amano_contact = run_amano(
                    multi_frame_model.hand_model_amano,
                    _contact_trans_i[None],
                    multi_frame_model.mano_root_orient[i0:i1].detach()[None],
                    multi_frame_model.mano_pose[i0:i1][None],
                    multi_frame_model.is_right[i0:i1].to(device))
                _hv_i_contact = _amano_contact['vertices'].squeeze(0) @ _flat_i
                contact_loss_s5_raw, ct_info_s5 = compute_soft_contact_loss(
                    _hv_i_contact, _ov_i, faces, contact_cache,
                    obj_grad_scale=CONTACT_OBJ_GRAD_SCALE_S5)
                contact_loss_s5 = contact_loss_s5_raw * LAMBDA_CONTACT_S5
                if frozen_template_conf is None:
                    # NO_TEMPLATE state: use temporary same-frame correspondences
                    # only to close the hand-object gap.  In full mode the hand
                    # wrist/root stay live; pose_only routes this through the
                    # detached-root contact vertices so only fingers can move.
                    _bootstrap_hv_i = _hv_i_contact if _stage5_global_locked else _hv_i
                    bootstrap_gap_raw_s5, gap_info_s5 = compute_confident_gap_closing_loss(
                        _bootstrap_hv_i, _ov_i.detach(), faces, contact_cache,
                        high_conf_thresh=BOOTSTRAP_GAP_CONF,
                        gap_start=BOOTSTRAP_GAP_START,
                        gap_clamp=BOOTSTRAP_GAP_CLAMP,
                        obj_grad_scale=(0.0, 0.0, 0.0))
                    bootstrap_gap_s5 = bootstrap_gap_raw_s5 * LAMBDA_BOOTSTRAP_GAP_S5
                if frozen_template_conf is not None and frozen_template_face is not None and frozen_template_bary is not None:
                    template_anchor_s5_raw, template_info_s5 = compute_grasp_template_anchor_loss(
                        _hv_i_contact, _ov_i, faces,
                        frozen_template_conf, frozen_template_face, frozen_template_bary,
                        high_conf_thresh=0.45,
                        gap_start=0.006,
                        gap_clamp=0.08,
                        obj_grad_scale=TEMPLATE_OBJ_GRAD_SCALE_S5)
                    template_anchor_s5 = template_anchor_s5_raw * LAMBDA_TEMPLATE_ANCHOR_S5
                    if frozen_template_reliable is not None and frozen_template_reliable.any():
                        neighbor_bridge_s5_raw, bridge_info_s5 = compute_neighbor_template_bridge_loss(
                            _hv_i_contact, _ov_i, faces,
                            frozen_template_conf, frozen_template_face, frozen_template_bary,
                            frozen_template_reliable,
                            max_bridge_distance=NEIGHBOR_TEMPLATE_BRIDGE_MAX_DIST_S5,
                            high_conf_thresh=NEIGHBOR_TEMPLATE_BRIDGE_CONF_THRESH_S5,
                            gap_start=0.006,
                            gap_clamp=0.06,
                            frame_weight_gamma=1.75,
                            obj_grad_scale=NEIGHBOR_TEMPLATE_BRIDGE_OBJ_GRAD_SCALE_S5)
                        neighbor_bridge_s5 = neighbor_bridge_s5_raw * LAMBDA_NEIGHBOR_TEMPLATE_BRIDGE_S5
                    else:
                        neighbor_bridge_s5 = torch.tensor(0.0, device=device)
                else:
                    template_anchor_s5 = torch.tensor(0.0, device=device)
                    neighbor_bridge_s5 = torch.tensor(0.0, device=device)
                if template_info_s5.get('template_active_count') is not None:
                    _tmpl_count = template_info_s5['template_active_count'].to(device=device, dtype=torch.float32)
                    _tmpl_gap = template_info_s5['template_gap_mean'].to(device=device, dtype=torch.float32)
                    _tmpl_conf = template_info_s5['template_conf_mean'].to(device=device, dtype=torch.float32)
                    _count_gate = ((_tmpl_count - 6.0) / 8.0).clamp(0.0, 1.0)
                    _gap_gate = ((_tmpl_gap - 0.006) / 0.008).clamp(0.0, 1.0)
                    _conf_gate = ((_tmpl_conf - 0.45) / 0.20).clamp(0.0, 1.0)
                    contact_relax_i = (_count_gate * _gap_gate * _conf_gate).detach()
                    contact_relax_full[i0:i1] = contact_relax_i
                # Anneal gap closing away as Stage 5 converges.  Early steps
                # can close large hand-object gaps; later steps rely on sticky
                # contact + object anchors to avoid over-pulling RGB pose.
                _gap_weight = LAMBDA_GAP_CLOSE_S5 * max(0.15, 1.0 - float(step) / max(float(num_steps_s5) * 0.9, 1.0))
                if _gap_weight > 0:
                    gap_close_s5_raw, gap_info_s5 = compute_confident_gap_closing_loss(
                        _hv_i_contact, _ov_i, faces, contact_cache,
                        high_conf_thresh=0.45,
                        gap_start=0.006,
                        gap_clamp=0.08,
                        obj_grad_scale=GAP_OBJ_GRAD_SCALE_S5)
                    gap_close_s5 = gap_close_s5_raw * _gap_weight
                    patch_close_s5_raw, patch_info_s5 = compute_contact_patch_centroid_loss(
                        _hv_i_contact, _ov_i, faces, contact_cache,
                        high_conf_thresh=0.45,
                        min_contacts=6,
                        gap_start=0.006,
                        gap_clamp=0.08,
                        obj_grad_scale=PATCH_OBJ_GRAD_SCALE_S5)
                    patch_close_s5 = patch_close_s5_raw * (_gap_weight / max(LAMBDA_GAP_CLOSE_S5, 1.0) * LAMBDA_PATCH_CLOSE_S5)
                else:
                    gap_close_s5 = torch.tensor(0.0, device=device)
                    patch_close_s5 = torch.tensor(0.0, device=device)
            else:
                contact_loss_s5 = torch.tensor(0.0, device=device)
                template_anchor_s5 = torch.tensor(0.0, device=device)
                neighbor_bridge_s5 = torch.tensor(0.0, device=device)
                gap_close_s5 = torch.tensor(0.0, device=device)
                patch_close_s5 = torch.tensor(0.0, device=device)

            # Hand 2D
            _hand_joints_for_2d_s5 = hand_joints_s5
            if LAMBDA_HAND2D_S5 > 0 and _stage5_pose_ray and _stage5_ray_delta is not None:
                _flat_2d_s5 = torch.diag(torch.tensor(
                    [-1.0, -1.0, 1.0],
                    dtype=hand_verts_s5.dtype,
                    device=device,
                )).unsqueeze(0).expand(num_sampled_frames, -1, -1)
                _amano_2d_s5 = run_amano(
                    multi_frame_model.hand_model_amano,
                    _mano_trans_eff_s5[None],
                    multi_frame_model.mano_root_orient.detach()[None],
                    multi_frame_model.mano_pose.detach()[None],
                    multi_frame_model.is_right.to(device),
                )
                _hand_joints_for_2d_s5 = _amano_2d_s5['joints'].squeeze(0) @ _flat_2d_s5
            if LAMBDA_HAND2D_S5 > 0 and sampled_gt_hand_joints_valid_mask.any():
                projected_joints_s5 = all_cameras_s56.transform_points_screen(
                    _hand_joints_for_2d_s5, image_size=((H_out, W_out),))[..., :2]
                _joint_sq_s5 = (projected_joints_s5 - sampled_gt_hand_joints_2d).pow(2).mean(dim=-1)
                _joint_valid_s5 = sampled_gt_hand_joints_valid_mask.float()
                _hand2d_frame_weight = hand2d_anatomy_frame_weight.to(
                    device=device, dtype=contact_relax_full.dtype)
                hand2d_relaxed_s5 = (1.0 - _hand2d_frame_weight).mean()
                if _joint_valid_s5.dim() == 1:
                    _frame_weight_s5 = _joint_valid_s5 * _hand2d_frame_weight
                    hand_2d_loss_s5 = (
                        (_joint_sq_s5.mean(dim=1) * _frame_weight_s5).sum() /
                        _frame_weight_s5.sum().clamp(min=1.0)
                    ) * LAMBDA_HAND2D_S5
                else:
                    _joint_weight_s5 = _joint_valid_s5 * _hand2d_frame_weight[:, None]
                    hand_2d_loss_s5 = (
                        (_joint_sq_s5 * _joint_weight_s5).sum() /
                        _joint_weight_s5.sum().clamp(min=1.0)
                    ) * LAMBDA_HAND2D_S5
            else:
                hand_2d_loss_s5 = torch.tensor(0.0, device=device)

            # Anatomy
            T_g_p_s5 = multi_frame_model.transforms_abs
            _, _R_s5, ee_s5 = multi_frame_model.axisFK(T_g_p_s5)
            loss_anatomy_s5 = multi_frame_model.anatomyLoss(ee_s5) * LAMBDA_ANATOMY_S5

            # mano_pose in 6D rotation space — used by anchor and the
            # temporal smoothness/accel terms below (computed once for
            # efficiency).
            _mp_6d_s5 = mano_pose_to_6d(multi_frame_model.mano_pose)

            # HaMeR mano_pose anchor — 3D articulation prior, more reliable
            # than 2D keypoints when the detector struggles.
            mano_pose_anchor_s5 = torch.nn.functional.mse_loss(
                _mp_6d_s5, _hamer_mano_pose_6d_init
            ) * LAMBDA_MANO_POSE_ANCHOR_S5

            # Object silhouette: SYMMETRIC penalty.
            #   - false-positive: rendered obj outside amodal mask (don't drift OUT)
            #   - false-negative: amodal mask not covered by rendered obj (don't drift IN/shrink)
            # Stage 3 uses fp+vp (vp = KNN pose-guiding, expensive).  In Stage 5
            # the object is already well-placed, so a cheap fn term is enough to
            # prevent contact_loss from sliding the object around inside the
            # mask without leaving it (which fp alone wouldn't catch).  Both
            # branches share LAMBDA_OBJ_SIL_S5 — same units (squared mask
            # residual averaged over pixels), symmetric weighting.
            obj_sil_loss_s5 = torch.tensor(0.0, device=device)
            if LAMBDA_OBJ_SIL_S5 > 0:
                _obj_frags_s5 = _sil_renderer.rasterizer(obj_meshes_s5)
                _rendered_obj_alpha_s5 = _sil_renderer.shader(_obj_frags_s5, obj_meshes_s5)[..., 3]
                _fp_frame_s5 = torch.relu(_rendered_obj_alpha_s5 - sampled_pred_amodal_masks).pow(2).flatten(1).mean(dim=1)
                _fn_frame_s5 = torch.relu(sampled_pred_amodal_masks - _rendered_obj_alpha_s5).pow(2).flatten(1).mean(dim=1)
                _obj_sil_frame_weight = (1.0 - 0.6 * contact_relax_full).clamp(0.4, 1.0)
                obj_sil_relaxed_s5 = (1.0 - _obj_sil_frame_weight).mean()
                obj_sil_loss_s5 = ((_fp_frame_s5 + _fn_frame_s5) * _obj_sil_frame_weight).mean() * LAMBDA_OBJ_SIL_S5

            # Temporal smoothness — fingers + obj.  Finger smoothness in
            # 6D space (matches anchor; avoids axis-angle 2π wrap).
            pose_smooth_s5 = (_mp_6d_s5[1:] - _mp_6d_s5[:-1]
                              ).pow(2).mean() * LAMBDA_POSE_SMOOTH_S5
            # Stage 5 object rotation smoothness is disabled: if Stage 3/SAM3D
            # already recovered a useful rotation trajectory, smoothing here can
            # flatten it. Full mode still allows translation smoothness.
            obj_smooth_s5 = (
                (multi_frame_model.trans[1:] - multi_frame_model.trans[:-1]).pow(2).mean()
            ) * LAMBDA_OBJ_SMOOTH_S5

            # Wrist guards — anchor + temporal smoothness.
            # CRITICAL: orientation diff must be in rotation-matrix space, not
            # axis-angle, otherwise projection (which canonicalises axis-angle
            # to |θ| ≤ π) can flip representation and explode the anchor loss.
            _R_root_s5 = axis_angle_to_matrix(multi_frame_model.mano_root_orient)  # (N, 3, 3)
            _wrist_frame_weight = torch.ones_like(contact_relax_full)
            _wrist_rot_frame = (_R_root_s5 - _R_root_anchor_s5).pow(2).mean(dim=(1, 2))
            _wrist_tr_frame = (multi_frame_model.mano_trans - _anchor_mano_trans).pow(2).mean(dim=1)
            wrist_anchor_s5 = ((_wrist_rot_frame + _wrist_tr_frame) * _wrist_frame_weight).mean() * LAMBDA_WRIST_ANCHOR_S5
            _tr_diff_s5 = multi_frame_model.mano_trans[1:] - multi_frame_model.mano_trans[:-1]
            _tr_relax_pair = 0.5 * (contact_relax_full[1:] + contact_relax_full[:-1])
            _tr_smooth_weight = torch.ones_like(_tr_relax_pair)
            tr_smooth_s5 = (_tr_diff_s5.pow(2).mean(dim=1) * _tr_smooth_weight).mean() * LAMBDA_HAND_TR_SMOOTH_S5
            root_R_smooth_s5 = (_R_root_s5[1:] - _R_root_s5[:-1]
                                ).pow(2).mean() * LAMBDA_ROOT_R_SMOOTH_S5

            # 2nd-order acceleration smoothness (needs N >= 3 frames).
            if multi_frame_model.mano_trans.shape[0] >= 3:
                _pose_acc = (_mp_6d_s5[2:] - 2 * _mp_6d_s5[1:-1] + _mp_6d_s5[:-2])
                pose_accel_s5 = _pose_acc.pow(2).mean() * LAMBDA_POSE_ACCEL_S5
                _tr_acc = (multi_frame_model.mano_trans[2:] - 2 * multi_frame_model.mano_trans[1:-1]
                           + multi_frame_model.mano_trans[:-2])
                _tr_accel_weight = torch.ones_like(contact_relax_full[1:-1])
                tr_accel_s5 = (_tr_acc.pow(2).mean(dim=1) * _tr_accel_weight).mean() * LAMBDA_HAND_TR_ACCEL_S5
                _R_acc = _R_root_s5[2:] - 2 * _R_root_s5[1:-1] + _R_root_s5[:-2]
                root_R_accel_s5 = _R_acc.pow(2).mean() * LAMBDA_ROOT_R_ACCEL_S5
            else:
                pose_accel_s5 = torch.tensor(0.0, device=device)
                tr_accel_s5 = torch.tensor(0.0, device=device)
                root_R_accel_s5 = torch.tensor(0.0, device=device)

            if _stage5_pose_ray:
                ray_delta_anchor_s5 = _stage5_ray_delta_scalar.pow(2).mean() * LAMBDA_RAY_DELTA_ANCHOR_S5
                if _stage5_ray_delta_scalar.shape[0] >= 2:
                    _ray_diff = _stage5_ray_delta_scalar[1:] - _stage5_ray_delta_scalar[:-1]
                    ray_delta_smooth_s5 = _ray_diff.pow(2).mean() * LAMBDA_RAY_DELTA_SMOOTH_S5
                else:
                    ray_delta_smooth_s5 = torch.tensor(0.0, device=device)
                if _stage5_ray_delta_scalar.shape[0] >= 3:
                    _ray_acc = (
                        _stage5_ray_delta_scalar[2:]
                        - 2 * _stage5_ray_delta_scalar[1:-1]
                        + _stage5_ray_delta_scalar[:-2]
                    )
                    ray_delta_accel_s5 = _ray_acc.pow(2).mean() * LAMBDA_RAY_DELTA_ACCEL_S5
                else:
                    ray_delta_accel_s5 = torch.tensor(0.0, device=device)
            else:
                ray_delta_anchor_s5 = torch.tensor(0.0, device=device)
                ray_delta_smooth_s5 = torch.tensor(0.0, device=device)
                ray_delta_accel_s5 = torch.tensor(0.0, device=device)

            # Object anchor (interaction vs non-interaction split)
            _ni_slices = []
            if i0 > 0:
                _ni_slices.append((0, i0))
            if i1 < num_sampled_frames:
                _ni_slices.append((i1, num_sampled_frames))
            if _ni_slices:
                _ni_rot = torch.cat([multi_frame_model.rot_6d[s:e] - _anchor_rot6d[s:e] for s, e in _ni_slices])
                _ni_tr  = torch.cat([multi_frame_model.trans[s:e]  - _anchor_trans[s:e]  for s, e in _ni_slices])
                obj_anchor_loss_s5 = (_ni_rot.pow(2).mean() + _ni_tr.pow(2).mean()) * LAMBDA_OBJ_ANCHOR_NI_S5
            else:
                obj_anchor_loss_s5 = torch.tensor(0.0, device=device)
            if _has_inter:
                _di_rot = multi_frame_model.rot_6d[i0:i1] - _anchor_rot6d[i0:i1]
                _di_tr  = multi_frame_model.trans[i0:i1]  - _anchor_trans[i0:i1]
                _obj_anchor_tr_weight = (1.0 - 0.8 * contact_relax_i[:_di_rot.shape[0]]).clamp(0.2, 1.0)
                obj_anchor_relaxed_s5 = (1.0 - _obj_anchor_tr_weight).mean()
                obj_anchor_rot_inter_s5 = _di_rot.pow(2).mean() * LAMBDA_OBJ_ANCHOR_INTER_S5
                obj_anchor_tr_inter_s5 = (
                    _di_tr.pow(2).mean(dim=1) * _obj_anchor_tr_weight
                ).mean() * LAMBDA_OBJ_ANCHOR_INTER_S5
                obj_anchor_loss_s5 = obj_anchor_loss_s5 + obj_anchor_rot_inter_s5 + obj_anchor_tr_inter_s5

            total_loss_s5 = (pen_loss_s5 + contact_loss_s5 + template_anchor_s5 + neighbor_bridge_s5
                             + bootstrap_gap_s5
                             + hand_2d_loss_s5
                             + loss_anatomy_s5 + mano_pose_anchor_s5
                             + obj_sil_loss_s5
                             + obj_anchor_loss_s5
                             + gap_close_s5
                             + patch_close_s5
                             + pose_smooth_s5 + obj_smooth_s5
                             + wrist_anchor_s5 + tr_smooth_s5 + root_R_smooth_s5
                             + pose_accel_s5 + tr_accel_s5 + root_R_accel_s5
                             + ray_delta_anchor_s5 + ray_delta_smooth_s5 + ray_delta_accel_s5)
            total_loss_s5.backward()
            _clip_params_s5 = list(multi_frame_model.parameters()) + _stage5_ray_params
            torch.nn.utils.clip_grad_norm_(_clip_params_s5, max_norm=1.0)
            optimizer_stage5.step()
            if _stage5_object_lite:
                project_object_translation_delta_(
                    multi_frame_model.trans,
                    _anchor_trans,
                    max_delta=OBJECT_LITE_MAX_TRANS_DELTA_S5,
                )
            if _stage5_pose_ray and _stage5_ray_delta is not None:
                clamp_stage5_ray_delta_(_stage5_ray_delta)

            # Periodic Gaussian projection on wrist trajectory — kills high-
            # frequency jitter that the contact-target lock can't catch.
            if multi_frame_model.mano_trans.requires_grad and (step + 1) % K_PROJECT_S5 == 0:
                _smooth_wrist_inplace(multi_frame_model)

            loop_s5.set_postfix(loss=total_loss_s5.item(), pen=pen_loss_s5.item(),
                                ct=contact_loss_s5.item(), sil=obj_sil_loss_s5.item(),
                                tmpl=template_anchor_s5.item(), gap=gap_close_s5.item(),
                                bridge=neighbor_bridge_s5.item(),
                                relax=round(contact_relax_full.mean().item(), 3),
                                patch=patch_close_s5.item(),
                                ray_mm=round(_stage5_ray_delta_scalar.abs().max().item() * 1000.0, 1),
                                pen_mm=round(pen_info_s5['mean_pen_depth'] * 1000, 2),
                                n_in=int(pen_info_s5['n_inside']))
            if step % 50 == 0 or step == num_steps_s5 - 1:
                _log_file_s5.write(",".join(str(round(v, 6)) for v in [
                    step, total_loss_s5.item(), pen_loss_s5.item(), contact_loss_s5.item(),
                    template_anchor_s5.item(),
                    template_anchor_s5_raw.item(),
                    neighbor_bridge_s5.item(),
                    neighbor_bridge_s5_raw.item(),
                    bootstrap_gap_s5.item(),
                    int(frozen_template_conf is not None),
                    contact_relax_full.mean().item(),
                    hand2d_relaxed_s5.item(),
                    hand2d_anatomy_frame_weight.mean().item(),
                    hand2d_anatomy_frame_weight.min().item(),
                    int(hand2d_anatomy_gate_info.get("n_low", 0)),
                    obj_sil_relaxed_s5.item(),
                    obj_anchor_relaxed_s5.item(),
                    hand_2d_loss_s5.item(), loss_anatomy_s5.item(),
                    mano_pose_anchor_s5.item(),
                    obj_sil_loss_s5.item(),
                    obj_anchor_loss_s5.item(),
                    obj_anchor_rot_inter_s5.item(),
                    obj_anchor_tr_inter_s5.item(),
                    gap_close_s5.item(),
                    patch_close_s5.item(),
                    pose_smooth_s5.item(), obj_smooth_s5.item(),
                    wrist_anchor_s5.item(), tr_smooth_s5.item(), root_R_smooth_s5.item(),
                    pose_accel_s5.item(), tr_accel_s5.item(), root_R_accel_s5.item(),
                    ray_delta_anchor_s5.item(), ray_delta_smooth_s5.item(), ray_delta_accel_s5.item(),
                    _stage5_ray_delta_scalar.mean().item() * 1000,
                    _stage5_ray_delta_scalar.min().item() * 1000,
                    _stage5_ray_delta_scalar.max().item() * 1000,
                    pen_info_s5['n_inside'],
                    pen_info_s5['mean_pen_depth'] * 1000,
                    pen_info_s5['max_pen_depth'] * 1000,
                    ct_info_s5['n_active'],
                    ct_info_s5['mean_contact_dist'] * 1000,
                    template_info_s5['n_template_active'],
                    template_info_s5['mean_template_gap_mm'],
                    template_info_s5['mean_template_conf'],
                    bridge_info_s5['n_bridge_active'],
                    bridge_info_s5['n_bridge_frames'],
                    bridge_info_s5['mean_bridge_gap_mm'],
                    bridge_info_s5['mean_bridge_conf'],
                    gap_info_s5['n_gap_active'],
                    gap_info_s5['mean_gap_dist'] * 1000,
                    patch_info_s5['n_patch_active'],
                    patch_info_s5['mean_patch_dist'] * 1000,
                    patch_info_s5['mean_patch_contacts'],
                ]) + "\n")

        if _stage5_pose_ray and _stage5_ray_delta is not None:
            with torch.no_grad():
                _ray_delta_vec_final, _ray_delta_scalar_final = build_stage5_ray_delta(
                    _stage5_ray_delta,
                    _stage5_ray_dirs,
                    _stage5_ray_inter_mask,
                )
                _mano_trans_final = _anchor_mano_trans + _ray_delta_vec_final
                multi_frame_model.mano_trans.data.copy_(_mano_trans_final)
                _active_ray = _ray_delta_scalar_final[_stage5_ray_inter_mask] if _stage5_ray_inter_mask is not None else _ray_delta_scalar_final
                _ray_stats = {
                    "applied": True,
                    "min_delta_m": float(_ray_delta_scalar_final.min().item()),
                    "max_delta_m": float(_ray_delta_scalar_final.max().item()),
                    "mean_abs_delta_m": float(_ray_delta_scalar_final.abs().mean().item()),
                    "active_mean_abs_delta_m": float(_active_ray.abs().mean().item()) if _active_ray.numel() > 0 else 0.0,
                    "active_count": int(_active_ray.numel()),
                    "bounds_m": [
                        float(DEFAULT_STAGE5_RAY_MIN_DELTA),
                        float(DEFAULT_STAGE5_RAY_MAX_DELTA),
                    ],
                }
            with open(os.path.join(output_path, "stage5_ray_delta_debug.json"), "w") as _f:
                json.dump(_ray_stats, _f, indent=2)
            print(
                "  [stage5 pose_ray] applied hand camera-ray delta: "
                f"active_mean_abs={_ray_stats['active_mean_abs_delta_m'] * 1000.0:.1f}mm, "
                f"min={_ray_stats['min_delta_m'] * 1000.0:.1f}mm, "
                f"max={_ray_stats['max_delta_m'] * 1000.0:.1f}mm"
            )

        _log_file_s5.close()
        if _reclassify_log_file_s5 is not None:
            _reclassify_log_file_s5.close()
            _reclassify_log_file_s5 = None
        if _prefix_diag_file_s5 is not None:
            _prefix_diag_file_s5.close()
            _prefix_diag_file_s5 = None
        print(f"\nSTAGE 5 completed: penetration resolved (sticky contact during).")
        multi_frame_model.rot_6d.requires_grad_(True)
        multi_frame_model.trans.requires_grad_(True)
        multi_frame_model.mano_root_orient.requires_grad_(True)
        multi_frame_model.mano_trans.requires_grad_(True)
        multi_frame_model.mano_pose.requires_grad_(True)

        # # Post-Stage-5 trajectory continuity: if the object jumps at the
        # # interaction/non-interaction boundary, blend that global translation
        # # jump back into the interaction segment.  Apply the same offset to
        # # hand and object so their relative grasp pose is preserved.
        # if _has_inter and i1 < num_sampled_frames and i1 > i0:
        #     with torch.no_grad():
        #         _boundary_jump = multi_frame_model.trans.data[i1] - multi_frame_model.trans.data[i1 - 1]
        #         _blend_offsets, _blend_info = compute_interaction_boundary_blend_offsets(
        #             num_frames=num_sampled_frames,
        #             i0=i0,
        #             i1=i1,
        #             jump=_boundary_jump,
        #             gamma=2.0,
        #             max_jump=0.05,
        #         )
        #         multi_frame_model.trans.data.add_(_blend_offsets)
        #         multi_frame_model.mano_trans.data.add_(_blend_offsets)
        #         print("  [post-s5] boundary translation blend: "
        #               f"jump={_blend_info['jump_norm'] * 1000.0:.2f}mm, "
        #               f"applied_max={_blend_info['applied_max_norm'] * 1000.0:.2f}mm, "
        #               f"clamped={_blend_info['clamped']}")

        # End-of-Stage 5 PLY dump for inspection.
        if save_intermediates and _has_inter:
            _dump_dir_s5 = os.path.join(output_path, "stage5_debug")
            _dump_stage_debug_meshes(
                _dump_dir_s5, multi_frame_model, faces,
                contact_cache, i0, i1, sampled_indices)


    def _run_stage6():
        # ================================================================
        # STAGE 6 — Contact Tightening (主 contact, 副 pen, bary 平滑)
        # ================================================================
        # Stage 6 — short "contact tightening" pass with high-frequency contact
        # map refresh. Wrist UNFROZEN with low LR + smoothness + anchor (same
        # philosophy as Stage 5). Contact dominates, penetration is a guard.
        ENABLE_STAGE6 = False
        if not ENABLE_STAGE6:
            print("\nSTAGE 6 skipped (ENABLE_STAGE6=False); using Stage 5 result.")

        print("\n" + "=" * 80)
        print("STAGE 6: Contact Tightening" + (" [SKIPPED]" if not ENABLE_STAGE6 else ""))
        print("=" * 80)

        # Stage 5 anchors for Stage 6.
        _anchor_rot6d_s6      = multi_frame_model.rot_6d.detach().clone()
        _anchor_trans_s6      = multi_frame_model.trans.detach().clone()
        _anchor_mano_root_s6  = multi_frame_model.mano_root_orient.detach().clone()
        _anchor_mano_trans_s6 = multi_frame_model.mano_trans.detach().clone()
        _anchor_mano_pose_s6  = multi_frame_model.mano_pose.detach().clone()
        _R_root_anchor_s6     = axis_angle_to_matrix(_anchor_mano_root_s6)

        # Wrist params remain trainable from Stage 5 — explicit re-enable just
        # in case future code freezes them.
        multi_frame_model.mano_root_orient.requires_grad_(True)
        multi_frame_model.mano_trans.requires_grad_(True)
        optimizer_stage6 = torch.optim.Adam([
            {'params': [multi_frame_model.rot_6d, multi_frame_model.trans], 'lr': 1e-4},
            {'params': [multi_frame_model.mano_pose],                        'lr': 2e-4},
            {'params': [multi_frame_model.mano_root_orient, multi_frame_model.mano_trans],
             'lr': 2e-5},
        ])

        num_steps_s6 = 150 if ENABLE_STAGE6 else 0
        LAMBDA_PEN_S6 = 5e2
        PEN_DEAD_ZONE_S6 = 0.003
        LAMBDA_CONTACT_S6 = 3e2
        # Mirror Stage 5 — relaxed since HaMeR mano_pose_anchor now provides
        # a more reliable 3D articulation prior.
        LAMBDA_HAND2D_S6 = 5e-1
        LAMBDA_MANO_POSE_ANCHOR_S6 = 1e2
        LAMBDA_OBJ_SIL_S6 = 5e2
        LAMBDA_OBJ_ANCHOR_S6 = 5e1
        # Mirror Stage 5: 1e1 → 3e1.  Same rationale — guard hand articulation
        # against unreliable 2D keypoints during contact tightening.
        LAMBDA_ANATOMY_S6 = 3e1
        # 1st-order — reduced (see Stage 5 note).
        LAMBDA_POSE_SMOOTH_S6 = 1.5e2          # was 3e2
        LAMBDA_OBJ_SMOOTH_S6 = 5e2
        # Wrist guards (anchor to Stage 5 end; smoothness mirrors Stage 5).
        LAMBDA_WRIST_ANCHOR_S6 = 5e2
        LAMBDA_HAND_TR_SMOOTH_S6 = 2e2         # was 5e2
        LAMBDA_ROOT_R_SMOOTH_S6 = 5e2          # was 1e3
        # 2nd-order acceleration (anti-jitter, allows constant-velocity motion).
        LAMBDA_POSE_ACCEL_S6 = 5e2
        LAMBDA_HAND_TR_ACCEL_S6 = 5e3      # bumped 5e2 → 5e3 (mirror of S5)
        LAMBDA_ROOT_R_ACCEL_S6 = 3e3       # bumped 1e3 → 3e3
        if disable_penetration_terms:
            LAMBDA_PEN_S6 = 0.0
        if disable_contact_terms:
            LAMBDA_CONTACT_S6 = 0.0

        _log_cols_s6 = ["step", "total", "pen", "contact",
                        "hand2d_anatomy_gate_mean", "hand2d_anatomy_gate_min", "hand2d_anatomy_gate_n_low",
                        "hand_2d", "anatomy",
                        "mano_pose_anchor",
                        "obj_sil", "obj_anchor",
                        "pose_sm", "obj_sm",
                        "wrist_anchor", "tr_sm", "root_R_sm",
                        "pose_acc", "tr_acc", "root_R_acc",
                        "n_inside", "mean_pen_mm", "max_pen_mm",
                        "n_active", "mean_ct_mm"]
        _log_path_s6 = os.path.join(output_path, "optimize_stage6.log")
        _log_file_s6 = open(_log_path_s6, "w", buffering=1)
        _log_file_s6.write("# Stage 6 — Contact tightening (sticky soft contact + light pen)\n")
        _log_file_s6.write(",".join(_log_cols_s6) + "\n")

        # Reset cache at the start of Stage 6 so the prev-face prior comes from the
        # geometry produced by Stage 5 rather than from Stage 5's initial state.
        contact_cache = None
        prev_argmax_face_full = None
        loop_s6 = tqdm(range(num_steps_s6), desc="Stage 6: Contact Tightening")
        for step in loop_s6:
            if _has_inter and not disable_contact_terms and (step % K_RECLASSIFY_S6 == 0):
                contact_cache, prev_argmax_face_full = _do_reclassify(prev_argmax_face_full)

            optimizer_stage6.zero_grad()

            obj_meshes_s6, hand_meshes_s6, hand_joints_s6 = multi_frame_model()
            hand_verts_s6 = hand_meshes_s6.verts_padded()
            obj_verts_s6  = obj_meshes_s6.verts_padded()

            pen_info_s6 = {'n_inside': 0, 'mean_pen_depth': 0.0, 'max_pen_depth': 0.0}
            ct_info_s6  = {'n_active': 0, 'mean_contact_dist': 0.0}

            if _has_inter:
                _hv_i = hand_verts_s6[i0:i1]
                _ov_i = obj_verts_s6[i0:i1]
                _on_i = _compute_obj_normals_with_grad(_ov_i)
                pen_loss_s6_raw, pen_info_s6 = compute_penetration_loss(
                    _hv_i, _ov_i, _on_i, detach_object=True,
                    dead_zone=PEN_DEAD_ZONE_S6)
                pen_loss_s6 = pen_loss_s6_raw * LAMBDA_PEN_S6
                if contact_cache is not None:
                    contact_loss_s6_raw, ct_info_s6 = compute_soft_contact_loss(
                        _hv_i, _ov_i, faces, contact_cache,
                        obj_grad_scale=(0.0, 0.0, 0.0))
                    contact_loss_s6 = contact_loss_s6_raw * LAMBDA_CONTACT_S6
                else:
                    contact_loss_s6 = torch.tensor(0.0, device=device)
            else:
                pen_loss_s6     = torch.tensor(0.0, device=device)
                contact_loss_s6 = torch.tensor(0.0, device=device)

            # Hand 2D (kept, lighter weight)
            projected_joints_s6 = all_cameras_s56.transform_points_screen(
                hand_joints_s6, image_size=((H_out, W_out),))[..., :2]
            if sampled_gt_hand_joints_valid_mask.any():
                _joint_sq_s6 = (projected_joints_s6 - sampled_gt_hand_joints_2d).pow(2).mean(dim=-1)
                _joint_valid_s6 = sampled_gt_hand_joints_valid_mask.float()
                _hand2d_frame_weight_s6 = hand2d_anatomy_frame_weight.to(
                    device=device, dtype=_joint_sq_s6.dtype)
                if _joint_valid_s6.dim() == 1:
                    _frame_weight_s6 = _joint_valid_s6 * _hand2d_frame_weight_s6
                    hand_2d_loss_s6 = (
                        (_joint_sq_s6.mean(dim=1) * _frame_weight_s6).sum() /
                        _frame_weight_s6.sum().clamp(min=1.0)
                    ) * LAMBDA_HAND2D_S6
                else:
                    _joint_weight_s6 = _joint_valid_s6 * _hand2d_frame_weight_s6[:, None]
                    hand_2d_loss_s6 = (
                        (_joint_sq_s6 * _joint_weight_s6).sum() /
                        _joint_weight_s6.sum().clamp(min=1.0)
                    ) * LAMBDA_HAND2D_S6
            else:
                hand_2d_loss_s6 = torch.tensor(0.0, device=device)

            # Anatomy
            T_g_p_s6 = multi_frame_model.transforms_abs
            _, _R_s6, ee_s6 = multi_frame_model.axisFK(T_g_p_s6)
            anatomy_loss_s6 = multi_frame_model.anatomyLoss(ee_s6) * LAMBDA_ANATOMY_S6

            # mano_pose in 6D (shared by anchor + smoothness/accel).
            _mp_6d_s6 = mano_pose_to_6d(multi_frame_model.mano_pose)

            # HaMeR mano_pose anchor.
            mano_pose_anchor_s6 = torch.nn.functional.mse_loss(
                _mp_6d_s6, _hamer_mano_pose_6d_init
            ) * LAMBDA_MANO_POSE_ANCHOR_S6

            # Object silhouette: SYMMETRIC (fp + fn), see Stage 5 for rationale.
            obj_sil_loss_s6 = torch.tensor(0.0, device=device)
            if LAMBDA_OBJ_SIL_S6 > 0:
                _obj_frags_s6 = _sil_renderer.rasterizer(obj_meshes_s6)
                _rendered_obj_alpha_s6 = _sil_renderer.shader(_obj_frags_s6, obj_meshes_s6)[..., 3]
                _fp_s6 = weighted_false_positive_loss(
                    _rendered_obj_alpha_s6, sampled_pred_amodal_masks)
                _fn_s6 = weighted_false_negative_loss(
                    _rendered_obj_alpha_s6, sampled_pred_amodal_masks, weight=1.0)
                obj_sil_loss_s6 = (_fp_s6 + _fn_s6) * LAMBDA_OBJ_SIL_S6

            # Object anchor.
            _all_rot_diff = multi_frame_model.rot_6d - _anchor_rot6d_s6
            _all_tr_diff  = multi_frame_model.trans  - _anchor_trans_s6
            obj_anchor_s6 = (_all_rot_diff.pow(2).mean() + _all_tr_diff.pow(2).mean()) * LAMBDA_OBJ_ANCHOR_S6

            # Smoothness — fingers + obj.  Finger smoothness in 6D space.
            pose_smooth_s6 = (_mp_6d_s6[1:] - _mp_6d_s6[:-1]
                              ).pow(2).mean() * LAMBDA_POSE_SMOOTH_S6
            obj_smooth_s6 = ((multi_frame_model.rot_6d[1:] - multi_frame_model.rot_6d[:-1]).pow(2).mean()
                            + (multi_frame_model.trans[1:] - multi_frame_model.trans[:-1]).pow(2).mean()
                            ) * LAMBDA_OBJ_SMOOTH_S6

            # Wrist guards (anchor to Stage 5 end + temporal smoothness).
            # Rotation-matrix space — invariant under axis-angle canonicalisation.
            _R_root_s6 = axis_angle_to_matrix(multi_frame_model.mano_root_orient)
            wrist_anchor_s6 = ((_R_root_s6 - _R_root_anchor_s6).pow(2).mean()
                              + (multi_frame_model.mano_trans - _anchor_mano_trans_s6).pow(2).mean()
                              ) * LAMBDA_WRIST_ANCHOR_S6
            tr_smooth_s6 = (multi_frame_model.mano_trans[1:] - multi_frame_model.mano_trans[:-1]
                            ).pow(2).mean() * LAMBDA_HAND_TR_SMOOTH_S6
            root_R_smooth_s6 = (_R_root_s6[1:] - _R_root_s6[:-1]
                                ).pow(2).mean() * LAMBDA_ROOT_R_SMOOTH_S6

            # 2nd-order acceleration smoothness (needs N >= 3 frames).
            if multi_frame_model.mano_trans.shape[0] >= 3:
                _pose_acc6 = (_mp_6d_s6[2:] - 2 * _mp_6d_s6[1:-1] + _mp_6d_s6[:-2])
                pose_accel_s6 = _pose_acc6.pow(2).mean() * LAMBDA_POSE_ACCEL_S6
                _tr_acc6 = (multi_frame_model.mano_trans[2:] - 2 * multi_frame_model.mano_trans[1:-1]
                            + multi_frame_model.mano_trans[:-2])
                tr_accel_s6 = _tr_acc6.pow(2).mean() * LAMBDA_HAND_TR_ACCEL_S6
                _R_acc6 = _R_root_s6[2:] - 2 * _R_root_s6[1:-1] + _R_root_s6[:-2]
                root_R_accel_s6 = _R_acc6.pow(2).mean() * LAMBDA_ROOT_R_ACCEL_S6
            else:
                pose_accel_s6 = torch.tensor(0.0, device=device)
                tr_accel_s6 = torch.tensor(0.0, device=device)
                root_R_accel_s6 = torch.tensor(0.0, device=device)

            total_loss_s6 = (pen_loss_s6 + contact_loss_s6 + hand_2d_loss_s6
                             + anatomy_loss_s6 + mano_pose_anchor_s6
                             + obj_sil_loss_s6
                             + obj_anchor_s6
                             + pose_smooth_s6 + obj_smooth_s6
                             + wrist_anchor_s6 + tr_smooth_s6 + root_R_smooth_s6
                             + pose_accel_s6 + tr_accel_s6 + root_R_accel_s6)
            total_loss_s6.backward()
            torch.nn.utils.clip_grad_norm_(multi_frame_model.parameters(), max_norm=1.0)
            optimizer_stage6.step()

            if multi_frame_model.mano_trans.requires_grad and (step + 1) % K_PROJECT_S6 == 0:
                _smooth_wrist_inplace(multi_frame_model)

            loop_s6.set_postfix(loss=total_loss_s6.item(), pen=pen_loss_s6.item(),
                                ct=contact_loss_s6.item(), sil=obj_sil_loss_s6.item(),
                                pen_mm=round(pen_info_s6['mean_pen_depth'] * 1000, 2),
                                ct_mm=round(ct_info_s6['mean_contact_dist'] * 1000, 2))
            if step % 10 == 0 or step == num_steps_s6 - 1:
                _log_file_s6.write(",".join(str(round(v, 6)) for v in [
                    step, total_loss_s6.item(), pen_loss_s6.item(), contact_loss_s6.item(),
                    hand2d_anatomy_frame_weight.mean().item(),
                    hand2d_anatomy_frame_weight.min().item(),
                    int(hand2d_anatomy_gate_info.get("n_low", 0)),
                    hand_2d_loss_s6.item(), anatomy_loss_s6.item(),
                    mano_pose_anchor_s6.item(),
                    obj_sil_loss_s6.item(),
                    obj_anchor_s6.item(),
                    pose_smooth_s6.item(), obj_smooth_s6.item(),
                    wrist_anchor_s6.item(), tr_smooth_s6.item(), root_R_smooth_s6.item(),
                    pose_accel_s6.item(), tr_accel_s6.item(), root_R_accel_s6.item(),
                    pen_info_s6['n_inside'],
                    pen_info_s6['mean_pen_depth'] * 1000,
                    pen_info_s6['max_pen_depth'] * 1000,
                    ct_info_s6['n_active'],
                    ct_info_s6['mean_contact_dist'] * 1000,
                ]) + "\n")

        _log_file_s6.close()
        if ENABLE_STAGE6:
            print(f"\nSTAGE 6 completed: contact tightened with low-residual penetration.")

        if save_intermediates and ENABLE_STAGE6 and _has_inter and contact_cache is not None:
            _dump_dir_s6 = os.path.join(output_path, "stage6_debug")
            _dump_stage_debug_meshes(
                _dump_dir_s6, multi_frame_model, faces,
                contact_cache, i0, i1, sampled_indices)

        # ─── Pin non-interaction frames to interaction boundaries ─────────
        # Outside [i0, i1) the object is physically stationary (hand hasn't
        # touched it / has released it).  Two issues to fix:
        #
        # (1) Object pose drift in static/approaching/releasing segments:
        #     Stage 3's per-frame obj_sil pulls the object towards noisy
        #     mask edges → object can drift from its true (interaction-
        #     consistent) pose.  Fix: replicate object 6DoF (rot_6d, trans)
        #     from frame i0 (pre-) and frame i1-1 (post-).
        #
        # (2) Hand depth (z) bias in non-interaction frames:
        #     Stage 3's hand_z anchor pulls hand mano_trans[:, 2] to the
        #     median metric depth inside the hand mask.  In approaching/
        #     releasing the mask is contaminated by background → median
        #     biased → hand z can land BEHIND the object → apparent "too
        #     deep" penetration in the render.  Fix: replicate ONLY hand
        #     mano_trans[:, 2] (the z component) from the boundary frames.
        #     Hand articulation (mano_pose), wrist orientation
        #     (mano_root_orient), and xy translation (mano_trans[:, :2])
        #     all stay as Stage 3 optimised — preserves natural approach
        #     animation, just clamps depth.
        #
        # Done after all optimisation so it cannot be undone by any later
        # loss.
        # NOTE: Both hand AND object Z (only Z, not full 6DoF) are pinned
        # in non-interaction frames to the interaction boundary frames.
        # Rationale (user-confirmed scenario):
        #   - Stage 5 contact pulls obj z toward hand z in [i0, i1) → obj
        #     ends up at z_inter (close to hand).
        #   - Stage 3 + obj_anchor leaves obj at z_static_obj in [0, i0) /
        #     [i1, N) (the original far position).
        #   - Result: at i0 the object visibly "snaps closer to the hand",
        #     and at i1 it "snaps away".  The hand also has no good z
        #     supervision outside [i0, i1).
        # Fix: pin both hand z AND object z (only the z component) in non-
        # interaction frames to frame i0 / i1-1.  Object xy & rotation stay
        # free so its silhouette remains accurate; only the depth jumps go
        # away.  Why only z (not full 6DoF)?  6DoF pinning replicated
        # rotation as well, but Stage 5 may have rotated the object slightly
        # at i0 → i0+1; CubicSpline interpolation between sparse frame i0-1
        # (pinned to i0) and i0+1 (free) then introduced a small rotation
        # blip.  Pinning only z avoids that — z varies smoothly across i0
        # by construction, and rotation interpolation stays untouched.
        # if _has_inter and i0 < i1:
        #     _N_sparse = multi_frame_model.rot_6d.shape[0]
        #     with torch.no_grad():
        #         if i0 > 0:
        #             multi_frame_model.mano_trans.data[:i0, 2] = multi_frame_model.mano_trans.data[i0, 2]
        #             multi_frame_model.trans.data[:i0, 2]      = multi_frame_model.trans.data[i0, 2]
        #             print(f"[Static-pose pin] hand z + obj z [0, {i0}) ← frame {i0} "
        #                   f"(hand_z={multi_frame_model.mano_trans.data[i0, 2].item():.4f} m, "
        #                   f"obj_z={multi_frame_model.trans.data[i0, 2].item():.4f} m); "
        #                   f"object xy & rotation kept free")
        #         if i1 < _N_sparse:
        #             multi_frame_model.mano_trans.data[i1:, 2] = multi_frame_model.mano_trans.data[i1-1, 2]
        #             multi_frame_model.trans.data[i1:, 2]      = multi_frame_model.trans.data[i1-1, 2]
        #             print(f"[Static-pose pin] hand z + obj z [{i1}, {_N_sparse}) ← frame {i1-1} "
        #                   f"(hand_z={multi_frame_model.mano_trans.data[i1-1, 2].item():.4f} m, "
        #                   f"obj_z={multi_frame_model.trans.data[i1-1, 2].item():.4f} m); "
        #                   f"object xy & rotation kept free")


    if disable_stage4:
        print("[Ablation] no_stage4: completely skipping Stage 4.")
        parsed_contact_map = {"__empty__": []}
    else:
        _run_stage4()

    if not parsed_contact_map:
        print(
            "WARNING: Stage 4 produced an empty contact map. Continuing anyway; "
            "Stage 5/6 will dynamically reclassify contacts and contact losses "
            "will be zero when no active contact candidates are found."
        )
        # parsed_contact_map is only used as a downstream gate. Keep the gate open
        # so qualitative ablations such as no_vp can still produce meshes.
        parsed_contact_map = {"__empty__": []}

    if parsed_contact_map:
        # ================================================================
        # Shared setup for Stage 5 / 6
        # ================================================================
        all_cameras_s56 = PerspectiveCameras(
            focal_length=focal_length,
            principal_point=principal_point,
            image_size=((H_out, W_out),) * num_sampled_frames,
            in_ndc=False,
            device=device,
        )

        # Subset of hand vertices allowed to be NEAR_CONTACT (drives soft contact).
        # Drop palm verts (dominant LBS joint == wrist) so the object is not
        # attracted to the palm. These 8 IDs are the dom_j==0 entries in the
        # DexGraspNet contact_indices.json (see chat).
        _PALM_VERTEX_IDS = {73, 96, 98, 99, 772, 774, 775, 777}
        if contact_indices_override is not None and len(contact_indices_override) > 0:
            _filtered = [v for v in contact_indices_override if v not in _PALM_VERTEX_IDS]
            n_dropped = len(contact_indices_override) - len(_filtered)
            print(f"contact subset: {len(_filtered)} verts (dropped {n_dropped} palm verts)")
            contact_subset_idx_t = torch.tensor(_filtered, dtype=torch.long, device=device)
        else:
            contact_subset_idx_t = None

        # Pre-smooth Stage 3 object trajectory (helps stability of subsequent fitting).
        # Frozen mode must preserve the exact Stage 4/post-offset state.
        _stage5_presmooth_allowed = bool(stage5_presmooth_object) and str(stage5_mode) != STAGE5_FROZEN
        _stage5_presmooth_sigma = 2.0 if _stage5_presmooth_allowed else None
        if _stage5_presmooth_allowed:
            from scipy.ndimage import gaussian_filter1d as _gf1d
            with torch.no_grad():
                for _param in [multi_frame_model.rot_6d, multi_frame_model.trans]:
                    _arr = _param.detach().cpu().numpy()
                    _arr = _gf1d(_arr, sigma=_stage5_presmooth_sigma, axis=0)
                    _param.data.copy_(torch.from_numpy(_arr).float().to(device))
            print(f"Pre-smoothed object trajectory (rot_6d, trans) sigma={_stage5_presmooth_sigma}")
        else:
            print("Skipped Stage5 object pre-smoothing")

        # Stage 3 anchors (used by Stage 5 anchor losses)
        _anchor_rot6d      = multi_frame_model.rot_6d.detach().clone()
        _anchor_trans      = multi_frame_model.trans.detach().clone()
        _anchor_mano_root  = multi_frame_model.mano_root_orient.detach().clone()
        # Pre-compute rotation-matrix form of the wrist anchor — anchor loss
        # compares in R-space (Frobenius), which is invariant under axis-angle
        # canonicalisation done by the periodic Gaussian projection.
        _R_root_anchor_s5 = axis_angle_to_matrix(_anchor_mano_root)
        _anchor_mano_trans = multi_frame_model.mano_trans.detach().clone()
        _anchor_mano_pose  = multi_frame_model.mano_pose.detach().clone()

        def _slerp_quaternion_tensor(q0: torch.Tensor, q1: torch.Tensor, alpha: float) -> torch.Tensor:
            q0 = torch.nn.functional.normalize(q0, dim=-1)
            q1 = torch.nn.functional.normalize(q1, dim=-1)
            dot = (q0 * q1).sum(dim=-1, keepdim=True)
            q1 = torch.where(dot < 0.0, -q1, q1)
            dot = (q0 * q1).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
            alpha_t = torch.as_tensor(float(alpha), dtype=q0.dtype, device=q0.device)
            linear = torch.nn.functional.normalize((1.0 - alpha_t) * q0 + alpha_t * q1, dim=-1)
            theta_0 = torch.acos(dot)
            sin_theta_0 = torch.sin(theta_0).clamp(min=1e-8)
            theta = theta_0 * alpha_t
            s0 = torch.sin(theta_0 - theta) / sin_theta_0
            s1 = torch.sin(theta) / sin_theta_0
            spherical = s0 * q0 + s1 * q1
            return torch.where(dot > 0.9995, linear, spherical)

        def _rotation_angle_delta_deg(before_aa: torch.Tensor, after_aa: torch.Tensor) -> torch.Tensor:
            before_R = axis_angle_to_matrix(before_aa.reshape(-1, 3))
            after_R = axis_angle_to_matrix(after_aa.reshape(-1, 3))
            rel_R = before_R.transpose(-1, -2) @ after_R
            trace = rel_R.diagonal(offset=0, dim1=-1, dim2=-2).sum(dim=-1)
            cos_angle = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
            return torch.rad2deg(torch.acos(cos_angle))

        def _repair_extreme_mano_pose_frames(_weights_t: torch.Tensor, _gate_info: dict) -> list[dict[str, Any]]:
            _extreme = [int(i) for i in _gate_info.get("extreme_indices", _gate_info.get("low_indices", []))]
            if not _extreme:
                return []
            _pose_orig = multi_frame_model.mano_pose.detach().clone()
            _pose_view = _pose_orig.reshape(_pose_orig.shape[0], -1, 3).clone()
            _num_frames = int(_pose_view.shape[0])
            _normal = torch.ones(_num_frames, dtype=torch.bool, device=_pose_view.device)
            _valid_extreme = [i for i in _extreme if 0 <= i < _num_frames]
            if not _valid_extreme:
                return []
            _normal[torch.tensor(_valid_extreme, dtype=torch.long, device=_pose_view.device)] = False
            _normal_indices = torch.nonzero(_normal, as_tuple=False).flatten().detach().cpu().tolist()
            if not _normal_indices:
                print("[hand2d-anatomy-gate] all frames are extreme; skip mano_pose interpolation")
                return []

            _sampled_list = [int(x) for x in np.asarray(sampled_indices).reshape(-1).tolist()]
            _records = []
            for _idx in _valid_extreme:
                _left_candidates = [int(j) for j in _normal_indices if int(j) < _idx]
                _right_candidates = [int(j) for j in _normal_indices if int(j) > _idx]
                _left = _left_candidates[-1] if _left_candidates else None
                _right = _right_candidates[0] if _right_candidates else None
                if _left is not None and _right is not None:
                    _alpha = float(_idx - _left) / float(max(_right - _left, 1))
                    _q0 = matrix_to_quaternion(axis_angle_to_matrix(_pose_view[_left].reshape(-1, 3)))
                    _q1 = matrix_to_quaternion(axis_angle_to_matrix(_pose_view[_right].reshape(-1, 3)))
                    _interp_q = _slerp_quaternion_tensor(_q0, _q1, _alpha)
                    _interp_pose = matrix_to_axis_angle(quaternion_to_matrix(_interp_q)).reshape_as(_pose_view[_idx])
                elif _left is not None:
                    _alpha = 0.0
                    _interp_pose = _pose_view[_left].clone()
                else:
                    _alpha = 1.0
                    _interp_pose = _pose_view[_right].clone()

                _before_pose = _pose_view[_idx].clone()
                _pose_view[_idx].copy_(_interp_pose)
                _angle_delta = _rotation_angle_delta_deg(_before_pose, _interp_pose)
                _sampled_idx = _sampled_list[_idx] if _idx < len(_sampled_list) else _idx
                _records.append({
                    "local_idx": int(_idx),
                    "sampled_idx": int(_sampled_idx),
                    "interp_source_left": None if _left is None else int(_left),
                    "interp_source_left_sampled": None if _left is None else int(_sampled_list[_left] if _left < len(_sampled_list) else _left),
                    "interp_source_right": None if _right is None else int(_right),
                    "interp_source_right_sampled": None if _right is None else int(_sampled_list[_right] if _right < len(_sampled_list) else _right),
                    "interp_alpha": float(_alpha),
                    "correction_angle_deg_mean": float(_angle_delta.mean().item()),
                    "correction_angle_deg_max": float(_angle_delta.max().item()),
                    "correction_l2_axis_angle": float(torch.linalg.norm(_interp_pose - _before_pose).item()),
                })

            with torch.no_grad():
                multi_frame_model.mano_pose.data.copy_(_pose_view.reshape_as(_pose_orig))
            return _records

        def _compute_stage3_anatomy_gate():
            with torch.no_grad():
                multi_frame_model()
                _, _, _ee_gate = multi_frame_model.axisFK(multi_frame_model.transforms_abs)
                _scores = []
                for _fi in range(_ee_gate.shape[0]):
                    _scores.append(multi_frame_model.anatomyLoss(_ee_gate[_fi:_fi + 1]).detach().float())
                _scores_t = torch.stack(_scores).reshape(-1).to(device)
                _weights_t, _gate_info = compute_anatomy_hand2d_frame_weights(
                    _scores_t,
                    min_weight=0.0,
                )
                _weights_t = _weights_t.to(device=device, dtype=torch.float32)

            _correction_records = _repair_extreme_mano_pose_frames(_weights_t, _gate_info)
            _csv_path = os.path.join(output_path, "stage3_hand_anatomy_gate.csv")
            _json_path = os.path.join(output_path, "stage3_hand_anatomy_gate.json")
            _corrections_path = os.path.join(output_path, "stage3_hand_anatomy_corrections.jsonl")
            _sampled_list = [int(x) for x in np.asarray(sampled_indices).reshape(-1).tolist()]
            _low_set = set(int(i) for i in _gate_info.get("extreme_indices", _gate_info.get("low_indices", [])))
            _correction_by_idx = {int(r["local_idx"]): r for r in _correction_records}
            _median = float(_gate_info.get("median", 0.0))
            _scale = float(_gate_info.get("robust_scale", 0.0))
            with open(_csv_path, "w") as _cf:
                _cf.write(
                    "local_idx,sampled_idx,anatomy_score,robust_z,hand2d_weight,is_extreme,"
                    "interp_source_left,interp_source_right,interp_alpha,"
                    "correction_angle_deg_mean,correction_angle_deg_max,correction_l2_axis_angle\n"
                )
                for _li, (_score, _weight) in enumerate(zip(_scores_t.detach().cpu().tolist(),
                                                           _weights_t.detach().cpu().tolist())):
                    _rz = 0.0 if _scale <= 1e-8 else max(0.0, (float(_score) - _median) / _scale)
                    _sampled = _sampled_list[_li] if _li < len(_sampled_list) else _li
                    _rec = _correction_by_idx.get(int(_li), {})
                    _alpha_s = "" if "interp_alpha" not in _rec else f"{float(_rec['interp_alpha']):.6f}"
                    _mean_s = "" if "correction_angle_deg_mean" not in _rec else f"{float(_rec['correction_angle_deg_mean']):.6f}"
                    _max_s = "" if "correction_angle_deg_max" not in _rec else f"{float(_rec['correction_angle_deg_max']):.6f}"
                    _l2_s = "" if "correction_l2_axis_angle" not in _rec else f"{float(_rec['correction_l2_axis_angle']):.6f}"
                    _cf.write(
                        f"{_li},{_sampled},{float(_score):.8f},{_rz:.6f},"
                        f"{float(_weight):.6f},{int(_li in _low_set)},"
                        f"{'' if _rec.get('interp_source_left') is None else _rec.get('interp_source_left')},"
                        f"{'' if _rec.get('interp_source_right') is None else _rec.get('interp_source_right')},"
                        f"{_alpha_s},{_mean_s},{_max_s},{_l2_s}\n"
                    )
            with open(_corrections_path, "w") as _corr_f:
                for _rec in _correction_records:
                    _corr_f.write(json.dumps(_rec) + "\n")
            with open(_json_path, "w") as _jf:
                json.dump({
                    "median": _gate_info.get("median", 0.0),
                    "mad": _gate_info.get("mad", 0.0),
                    "robust_scale": _gate_info.get("robust_scale", 0.0),
                    "extreme_z": _gate_info.get("extreme_z", 6.0),
                    "extreme_threshold": _gate_info.get("extreme_threshold", 0.0),
                    "min_weight": 0.0,
                    "n_low": _gate_info.get("n_low", 0),
                    "low_indices": _gate_info.get("low_indices", []),
                    "extreme_indices": _gate_info.get("extreme_indices", []),
                    "n_corrected": len(_correction_records),
                    "corrections_jsonl": os.path.basename(_corrections_path),
                    "csv": os.path.basename(_csv_path),
                }, _jf, indent=2)
            print(
                "[hand2d-anatomy-gate] "
                f"mean={_weights_t.mean().item():.3f}, min={_weights_t.min().item():.3f}, "
                f"n_extreme={int(_gate_info.get('n_low', 0))}, "
                f"n_corrected={len(_correction_records)}; wrote {os.path.basename(_csv_path)}"
            )
            return _scores_t.detach(), _weights_t.detach(), _gate_info

        hand2d_anatomy_scores_s3, hand2d_anatomy_frame_weight, hand2d_anatomy_gate_info = _compute_stage3_anatomy_gate()
        _anchor_mano_pose = multi_frame_model.mano_pose.detach().clone()

        # Interaction segment indices (contact / penetration losses operate here only).
        i0 = int(approaching_end_idx)
        i1 = int(interaction_end_idx)
        N_inter = max(0, i1 - i0)
        _has_inter = N_inter > 0
        if not _has_inter:
            print("Interaction segment is empty; Stage 5/6 will only run anchor losses.")

        # Hyper-params (kept inline as requested).
        K_RECLASSIFY = 50  # legacy default, used by Stage 6
        NEIGHBOR_TEMPLATE_BRIDGE_MAX_DIST_S5 = 12
        NEIGHBOR_TEMPLATE_BRIDGE_CONF_THRESH_S5 = 0.035
        # Annealed reclassify schedule for Stage 5: as hand/obj converge, refresh
        # contact targets more frequently. Early steps use long K so the optimizer
        # can settle toward an initial target before it shifts.
        def _k_reclassify_s5(_step):
            if _step < 200:
                return 100
            if _step < 400:
                return 50
            return 20
        # Stage 6 (contact tightening) uses fixed short K — geometry already
        # near-converged, frequent target refresh is safe and helps refine.
        K_RECLASSIFY_S6 = 20
        # 2cm was too tight: when STAGE 3 + ray-scale + depth-offset leaves a
        # ~1cm hand-object gap (common with closed-form hand-z init, where
        # metric depth bias on hand vs object differs), most fingertip verts
        # are 2-4cm from the surface and get filtered out of the active set.
        # Result: contact loss only sees the few already-touching verts and
        # cannot pull the rest in.  5cm catches all realistic grasp candidates
        # while still excluding clearly-non-contact verts.
        DIST_THRESH = 0.05
        CONE_DEG = 60.0
        N_SURFACE_SAMPLES = 8000
        TOPK = 8
        SIGMA = 0.01
        TEMPLATE_CLOSE_SEED_GAP_M = 0.012
        TEMPLATE_STRICT_SEED_CONF = 0.45
        TEMPLATE_STRICT_SEED_CONTACTS = 8
        BOOTSTRAP_GAP_CONF = 0.35
        BOOTSTRAP_GAP_START = 0.008
        BOOTSTRAP_GAP_CLAMP = 0.05
        PREV_FACE_BONUS = 2.0  # additive logit bonus for prev-frame argmax face
        contact_confidence_full = None  # dense (N_inter, 778) temporal contact memory
        contact_anchor_face_full = None  # dense (N_inter, 778) same-frame object anchor memory
        contact_anchor_bary_full = None  # dense (N_inter, 778, 3)
        frozen_template_conf = None
        frozen_template_face = None
        frozen_template_bary = None
        frozen_template_reliable = None
        template_phase_s5 = "NO_TEMPLATE"
        bootstrap_template_seen = False
        # Temporal contact-target locking (mode-filter on argmax_face). For
        # each (frame t, hand vert v): if a face id appears >= LOCK_MIN_CONSENSUS
        # times in the window [t-LOCK_WIN, t+LOCK_WIN] (and is in current
        # topk), force one-hot weight on that face.  Eliminates wrist jitter
        # caused by per-frame contact-target switching.
        LOCK_WIN = 3          # 7-frame window
        LOCK_MIN_CONSENSUS = 3  # 3 of 7 — true plurality of the window. Was
                                # min=3 in win=5 → only 3.8-12% locked, too
                                # strict. Wider window with same plurality
                                # threshold should bring lock rate to ~30-50%.

        # Wrist trajectory periodic Gaussian projection (anti-jitter).
        # Every K_PROJECT_* steps, pull mano_trans / mano_root_orient toward a
        # temporally-smoothed version (gaussian filter, σ = WRIST_PROJECT_SIGMA
        # frames). Rotation is smoothed in matrix space + SVD re-orthogonalised
        # to avoid axis-angle wrap-around. Acts as periodic projection onto a
        # smooth trajectory subspace; high-frequency wrist jitter that survives
        # the contact-target lock gets erased.
        from scipy.ndimage import gaussian_filter1d as _gauss1d_wrist
        from pytorch3d.transforms import matrix_to_axis_angle as _aa_from_R
        # Set K to a huge number to effectively disable. Periodic Gaussian
        # projection was tried but was non-selective: it smoothed real wrist
        # motion as well as jitter, making hand_2d explode (1.5 → 18.7).
        # Path forward instead: rely on stronger LAMBDA_HAND2D_S5/S6 to let the
        # per-frame stable 2D detector dominate wrist position naturally.
        K_PROJECT_S5 = 10**9
        K_PROJECT_S6 = 10**9
        WRIST_PROJECT_SIGMA = 1.0

        def _smooth_wrist_inplace(model, sigma=WRIST_PROJECT_SIGMA):
            """Replace mano_trans / mano_root_orient with a temporally-smoothed
            version (in place).  Detached + no_grad; preserves Adam state."""
            with torch.no_grad():
                _dev = model.mano_trans.device
                # Translation: direct gaussian.
                _tr_np = model.mano_trans.detach().cpu().numpy()
                _tr_sm = _gauss1d_wrist(_tr_np, sigma=sigma, axis=0, mode='nearest')
                model.mano_trans.data.copy_(torch.from_numpy(_tr_sm).to(_dev))
                # Rotation: smooth in matrix space, then SVD-orthogonalise.
                _R = axis_angle_to_matrix(model.mano_root_orient).detach().cpu().numpy()
                _R_sm = _gauss1d_wrist(_R, sigma=sigma, axis=0, mode='nearest')
                _U, _, _Vt = np.linalg.svd(_R_sm)
                _R_ortho = _U @ _Vt
                _det = np.linalg.det(_R_ortho)
                # Force det = +1 by flipping last column of U where det < 0.
                _flip = np.sign(_det)
                _U[..., :, -1] *= _flip[..., None]
                _R_ortho = _U @ _Vt
                _R_ortho_t = torch.from_numpy(_R_ortho).float().to(_dev)
                model.mano_root_orient.data.copy_(_aa_from_R(_R_ortho_t))

        # Silhouette renderer (object only) reused by Stage 5 / Stage 6.
        # SoftSilhouetteShader returns alpha in channel 3; we only need that.
        _sil_raster_settings = RasterizationSettings(
            image_size=(H_out, W_out), blur_radius=1e-4, faces_per_pixel=20)
        _sil_renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=all_cameras_s56,
                                      raster_settings=_sil_raster_settings),
            shader=SoftSilhouetteShader(
                blend_params=BlendParams(sigma=1e-4, gamma=1e-4)),
        )

        def _compute_obj_normals_with_grad(obj_verts_full):
            """obj_meshes from multi_frame_model() doesn't always cache normals
            cleanly; rebuild verts_normals_padded so gradients flow."""
            _meshes = Meshes(verts=list(obj_verts_full), faces=[faces] * obj_verts_full.shape[0])
            return _meshes.verts_normals_padded()

        def _dense_memory_from_cache(_cache, confidence_attr="contact_confidence"):
            """Convert padded active contact cache to dense per-hand-vertex memory."""
            _conf_full = torch.zeros((max(N_inter, 1), 778), dtype=torch.float32, device=device)
            _anchor_face_full = torch.full((max(N_inter, 1), 778), -1, dtype=torch.long, device=device)
            _anchor_bary_full = torch.zeros((max(N_inter, 1), 778, 3), dtype=torch.float32, device=device)
            _cache_conf = getattr(_cache, confidence_attr, None)
            if _cache.active_mask.any() and _cache_conf is not None:
                _conf_safe = torch.where(
                    _cache.active_mask,
                    _cache_conf,
                    torch.zeros_like(_cache_conf))
                _conf_full.scatter_(1, _cache.active_idx, _conf_safe)
                _amax_safe = torch.where(
                    _cache.active_mask,
                    _cache.argmax_face,
                    torch.full_like(_cache.argmax_face, -1))
                _anchor_face_full.scatter_(1, _cache.active_idx, _amax_safe)
                _ak = _cache.weight_topk.argmax(dim=-1, keepdim=True)
                _bary_argmax = _cache.bary_topk.gather(
                    2, _ak.unsqueeze(-1).expand(-1, -1, -1, 3)).squeeze(2)
                _anchor_bary_full.scatter_(
                    1, _cache.active_idx.unsqueeze(-1).expand(-1, -1, 3),
                    _bary_argmax)
            return _conf_full, _anchor_face_full, _anchor_bary_full

        def _do_reclassify(prev_argmax_face_full):
            """Run a no_grad forward pass and rebuild contact cache for the
            interaction segment. Returns (cache, new_prev_argmax_face_full)."""
            nonlocal contact_confidence_full, contact_anchor_face_full, contact_anchor_bary_full
            nonlocal frozen_template_conf, frozen_template_face, frozen_template_bary, frozen_template_reliable
            nonlocal bootstrap_template_seen, template_phase_s5
            with torch.no_grad():
                _om, _hm, _ = multi_frame_model()
                _hv_all = _hm.verts_padded()
                _hn_all = _hm.verts_normals_padded()
                _ov_all = _om.verts_padded()
                _on_all = _om.verts_normals_padded()
                _hv_i = _hv_all[i0:i1]
                _hn_i = _hn_all[i0:i1]
                _ov_i = _ov_all[i0:i1]
                _on_i = _on_all[i0:i1]

                def _build_locked_cache(_prev_conf, _prev_face, _prev_bary):
                    _cache_raw = classify_and_build_correspondence(
                        _hv_i, _hn_i, _ov_i, _on_i, faces,
                        contact_subset_idx=contact_subset_idx_t,
                        dist_thresh=DIST_THRESH,
                        cone_angle_deg=CONE_DEG,
                        n_surface_samples=N_SURFACE_SAMPLES,
                        topk=TOPK,
                        sigma=SIGMA,
                        prev_argmax_face=prev_argmax_face_full,
                        prev_face_logit_bonus=PREV_FACE_BONUS,
                        prev_contact_confidence=_prev_conf,
                        prev_anchor_face=_prev_face,
                        prev_anchor_bary=_prev_bary,
                    )
                    # Temporal lock: collapse weight_topk to one-hot on consensus
                    # face within a 5-frame window.  Removes per-frame contact
                    # target jitter that propagates into wrist jitter.
                    _cache_locked = temporal_lock_argmax(
                        _cache_raw, window=LOCK_WIN, min_consensus=LOCK_MIN_CONSENSUS)
                    if _cache_raw.active_mask.any():
                        _orig_argmax_k = _cache_raw.weight_topk.argmax(dim=-1)
                        _new_argmax_k = _cache_locked.weight_topk.argmax(dim=-1)
                        _locked_n = ((_orig_argmax_k != _new_argmax_k) | (
                            _cache_locked.weight_topk.max(dim=-1).values > 0.99
                        )).logical_and(_cache_raw.active_mask).sum().item()
                        _active_n = int(_cache_raw.active_mask.sum().item())
                        if _active_n > 0:
                            _log_reclassify(f"  [reclassify] locked {_locked_n}/{_active_n} "
                                            f"active verts ({100.0 * _locked_n / _active_n:.1f}%)")
                    return _cache_locked

                _cache = _build_locked_cache(
                    contact_confidence_full, contact_anchor_face_full, contact_anchor_bary_full)
                _conf_full, _anchor_face_full, _anchor_bary_full = _dense_memory_from_cache(_cache)
                _obs_conf_full, _, _ = _dense_memory_from_cache(
                    _cache, confidence_attr="observed_contact_confidence")

                def _close_template_seed_frames(_conf, _face, _bary):
                    _candidate = (_conf >= TEMPLATE_STRICT_SEED_CONF) & (_face >= 0)
                    _count = _candidate.sum(dim=1)
                    _gap_mean = torch.zeros(_candidate.shape[0], dtype=torch.float32, device=device)
                    for _fi in range(_candidate.shape[0]):
                        _mask = _candidate[_fi]
                        if not _mask.any():
                            continue
                        _face_ids = _face[_fi][_mask].clamp(min=0)
                        _fv = _ov_i[_fi][faces[_face_ids]]
                        _anchors = (_fv * _bary[_fi][_mask].unsqueeze(-1)).sum(dim=1)
                        _gap_mean[_fi] = torch.linalg.norm(_hv_i[_fi][_mask] - _anchors, dim=-1).mean()
                    _close = (_count >= TEMPLATE_STRICT_SEED_CONTACTS) & (_gap_mean <= TEMPLATE_CLOSE_SEED_GAP_M)
                    return _close, _count, _gap_mean

                _close_seed_frame, _strict_count, _strict_gap_mean = _close_template_seed_frames(
                    _obs_conf_full, _anchor_face_full, _anchor_bary_full)
                _near_strict_candidate = (
                    ((_obs_conf_full >= TEMPLATE_STRICT_SEED_CONF) & (_anchor_face_full >= 0)).sum(dim=1)
                    >= TEMPLATE_STRICT_SEED_CONTACTS)
                _strict_candidate = _near_strict_candidate & _close_seed_frame
                _weak_candidate = ((_obs_conf_full >= 0.35) & (_anchor_face_full >= 0)).sum(dim=1) >= 4

                def _propagate_template(_conf, _face, _bary, _obs_conf, *, allow_bootstrap):
                    return propagate_grasp_template_from_reliable_frames(
                        _conf, _face, _bary,
                        min_seed_conf=0.45,
                        min_seed_contacts=8,
                        min_reliable_run=3,
                        expand_reliable=True,
                        min_expand_conf=0.35,
                        min_expand_contacts=4,
                        min_expand_overlap=0.35,
                        min_expand_face_agree=0.6,
                        bootstrap_if_no_core=allow_bootstrap,
                        bootstrap_min_seed_conf=0.35,
                        bootstrap_min_seed_contacts=4,
                        bootstrap_min_reliable_run=2,
                        temporal_decay=0.86,
                        max_steps=12,
                        min_keep_conf=0.05,
                        seed_confidence=_obs_conf,
                        seed_frame_mask=_close_seed_frame,
                    )

                _template_state = "none"
                if frozen_template_conf is None:
                    _strict_conf, _strict_face, _strict_bary, _strict_reliable = _propagate_template(
                        _conf_full, _anchor_face_full, _anchor_bary_full, _obs_conf_full,
                        allow_bootstrap=False)
                    if _strict_reliable.any():
                        _prop_conf = _strict_conf
                        _prop_face = _strict_face
                        _prop_bary = _strict_bary
                        _reliable_frames = _strict_reliable
                        _template_state = "confirmed"
                    else:
                        _prop_conf = _conf_full
                        _prop_face = _anchor_face_full
                        _prop_bary = _anchor_bary_full
                        _reliable_frames = torch.zeros(_conf_full.shape[0], dtype=torch.bool, device=device)
                        _template_state = "no_template_bootstrap_gap"
                else:
                    _expanded_reliable = expand_reliable_contact_frontier(
                        _obs_conf_full,
                        _anchor_face_full,
                        frozen_template_face,
                        frozen_template_reliable,
                        min_expand_conf=0.35,
                        min_expand_contacts=4,
                        min_expand_overlap=0.45,
                        min_expand_face_agree=0.7,
                        max_new_frames_per_side=-1,
                    )
                    _expanded_reliable = frozen_template_reliable | (_expanded_reliable & _close_seed_frame)
                    _n_frontier_added = int((_expanded_reliable & (~frozen_template_reliable)).sum().item())
                    if _n_frontier_added > 0:
                        _seed_conf = torch.where(
                            frozen_template_reliable[:, None],
                            frozen_template_conf,
                            torch.zeros_like(frozen_template_conf))
                        _seed_face = torch.where(
                            frozen_template_reliable[:, None],
                            frozen_template_face,
                            torch.full_like(frozen_template_face, -1))
                        _seed_bary = torch.where(
                            frozen_template_reliable[:, None, None],
                            frozen_template_bary,
                            torch.zeros_like(frozen_template_bary))
                        _new_reliable = _expanded_reliable & (~frozen_template_reliable)
                        _new_contact = (_obs_conf_full >= 0.35) & (_anchor_face_full >= 0)
                        _new_seed = _new_reliable[:, None] & _new_contact
                        _seed_conf = torch.where(_new_seed, _obs_conf_full, _seed_conf)
                        _seed_face = torch.where(_new_seed, _anchor_face_full, _seed_face)
                        _seed_bary = torch.where(_new_seed.unsqueeze(-1), _anchor_bary_full, _seed_bary)
                        frozen_template_conf, frozen_template_face, frozen_template_bary = propagate_dense_contact_memory(
                            _seed_conf,
                            _seed_face,
                            _seed_bary,
                            temporal_decay=0.86,
                            max_steps=12,
                            min_seed_conf=0.05,
                            min_keep_conf=0.05,
                        )
                        frozen_template_reliable = _expanded_reliable.detach().clone()
                        _frontier_local = _new_reliable.nonzero(as_tuple=False).squeeze(-1).detach().cpu().tolist()
                        _frontier_sampled = [int(sampled_indices[i0 + int(_i)]) for _i in _frontier_local]
                        _log_reclassify(f"  [reclassify] grasp template frontier added local={_frontier_local}, "
                                        f"sampled={_frontier_sampled}")
                    _prop_conf = frozen_template_conf
                    _prop_face = frozen_template_face
                    _prop_bary = frozen_template_bary
                    _reliable_frames = frozen_template_reliable
                    _template_state = "frozen"
                _n_added = int(((_prop_conf > (_conf_full + 1e-6)) & (_prop_face >= 0)).sum().item())
                _n_template = int(((_prop_conf > 0) & (_prop_face >= 0)).sum().item())
                _n_reliable = int(_reliable_frames.sum().item())
                _reliable_local = _reliable_frames.nonzero(as_tuple=False).squeeze(-1).detach().cpu().tolist()
                _reliable_sampled = [int(sampled_indices[i0 + int(_i)]) for _i in _reliable_local]
                _strict_local = _strict_candidate.nonzero(as_tuple=False).squeeze(-1).detach().cpu().tolist()
                _weak_local = _weak_candidate.nonzero(as_tuple=False).squeeze(-1).detach().cpu().tolist()
                _near_strict_local = _near_strict_candidate.nonzero(as_tuple=False).squeeze(-1).detach().cpu().tolist()
                _close_seed_local = _close_seed_frame.nonzero(as_tuple=False).squeeze(-1).detach().cpu().tolist()
                _strict_sampled = [int(sampled_indices[i0 + int(_i)]) for _i in _strict_local]
                _weak_sampled = [int(sampled_indices[i0 + int(_i)]) for _i in _weak_local]
                _near_strict_sampled = [int(sampled_indices[i0 + int(_i)]) for _i in _near_strict_local]
                _close_seed_sampled = [int(sampled_indices[i0 + int(_i)]) for _i in _close_seed_local]
                if (_prefix_diag_file_s5 is not None and _n_reliable > 0
                        and _reliable_local and int(_reliable_local[0]) > 0):
                    _first_rel = int(_reliable_local[0])
                    _missing_prefix = list(range(_first_rel))
                    for _li in _missing_prefix:
                        _conf_row = _obs_conf_full[_li]
                        _cand035 = (_conf_row >= 0.35) & (_anchor_face_full[_li] >= 0)
                        _cand045 = (_conf_row >= 0.45) & (_anchor_face_full[_li] >= 0)
                        _template_valid = (_prop_face[_li] >= 0) & (_prop_conf[_li] > 0)
                        _overlap = _cand035 & _template_valid
                        _n035 = int(_cand035.sum().item())
                        _n045 = int(_cand045.sum().item())
                        _n_template_valid = int(_template_valid.sum().item())
                        _n_overlap = int(_overlap.sum().item())
                        _overlap_ratio = float(_n_overlap) / float(max(_n035, 1))
                        if _n035 > 0:
                            _mean_conf035 = float(_conf_row[_cand035].mean().item())
                        else:
                            _mean_conf035 = 0.0
                        _max_conf = float(_conf_row.max().item()) if _conf_row.numel() > 0 else 0.0
                        if _n_overlap > 0:
                            _face_agree = float((
                                _anchor_face_full[_li][_overlap] == _prop_face[_li][_overlap]
                            ).float().mean().item())
                        else:
                            _face_agree = -1.0
                        if _n_template_valid > 0:
                            _face_ids = _prop_face[_li][_template_valid].clamp(min=0)
                            _fv = _ov_i[_li][faces[_face_ids]]
                            _anchors = (_fv * _prop_bary[_li][_template_valid].unsqueeze(-1)).sum(dim=1)
                            _mean_template_gap_mm = float(
                                torch.linalg.norm(_hv_i[_li][_template_valid] - _anchors, dim=-1).mean().item() * 1000.0)
                        else:
                            _mean_template_gap_mm = -1.0
                        _frontier_pass_overlap = int(_overlap_ratio >= 0.45)
                        _frontier_pass_face = int(_face_agree >= 0.7)
                        _prefix_diag_file_s5.write(",".join(str(v) for v in [
                            int(_current_reclassify_step_s5),
                            int(_li),
                            int(sampled_indices[i0 + int(_li)]),
                            int(_first_rel),
                            int(bool(_strict_candidate[_li].item())),
                            int(bool(_weak_candidate[_li].item())),
                            _n035,
                            _n045,
                            round(_mean_conf035, 6),
                            round(_max_conf, 6),
                            _n_template_valid,
                            _n_overlap,
                            round(_overlap_ratio, 6),
                            round(_face_agree, 6),
                            round(_mean_template_gap_mm, 6),
                            _frontier_pass_overlap,
                            _frontier_pass_face,
                        ]) + "\n")
                    _strict_prefix = [int(_li) for _li in _missing_prefix if bool(_strict_candidate[_li].item())]
                    _weak_prefix = [int(_li) for _li in _missing_prefix if bool(_weak_candidate[_li].item())]
                    _log_reclassify(
                        f"  [prefix-diagnostic] first_reliable_local={_first_rel}, "
                        f"missing_prefix={_missing_prefix}, strict_prefix={_strict_prefix}, "
                        f"weak_prefix={_weak_prefix}, csv={os.path.basename(_prefix_diag_path_s5)}")
                _log_reclassify(f"  [reclassify] grasp candidates strict local={_strict_local}, "
                                f"sampled={_strict_sampled}; weak local={_weak_local}, sampled={_weak_sampled}")
                _log_reclassify(
                    f"  [reclassify] close seed gate: near_strict local={_near_strict_local}, "
                    f"sampled={_near_strict_sampled}; close local={_close_seed_local}, "
                    f"sampled={_close_seed_sampled}, max_gap_mm={TEMPLATE_CLOSE_SEED_GAP_M * 1000.0:.1f}")
                if _n_template > 0 and _template_state != "no_template_bootstrap_gap":
                    _log_reclassify(f"  [reclassify] grasp template ({_template_state}) reliable_frames={_n_reliable}, "
                                    f"local={_reliable_local}, sampled={_reliable_sampled}, "
                                    f"template anchors={_n_template}, increased {_n_added}")
                    if _n_reliable > 0:
                        _frame_idx = torch.arange(_reliable_frames.shape[0], device=device)
                        _rel_idx = torch.nonzero(_reliable_frames, as_tuple=False).flatten()
                        _nearest_rel_dist = (_frame_idx[:, None] - _rel_idx[None, :]).abs().min(dim=1).values
                        _bridge_frame = ((~_reliable_frames)
                                         & (_nearest_rel_dist <= NEIGHBOR_TEMPLATE_BRIDGE_MAX_DIST_S5)
                                         & (((_prop_conf > 0) & (_prop_face >= 0)).any(dim=1)))
                        _bridge_local = _bridge_frame.nonzero(as_tuple=False).squeeze(-1).detach().cpu().tolist()
                        _bridge_sampled = [int(sampled_indices[i0 + int(_i)]) for _i in _bridge_local]
                        _log_reclassify(
                            f"  [reclassify] neighbor bridge candidates local={_bridge_local}, "
                            f"sampled={_bridge_sampled}, max_dist={NEIGHBOR_TEMPLATE_BRIDGE_MAX_DIST_S5}, "
                            f"conf_thresh={NEIGHBOR_TEMPLATE_BRIDGE_CONF_THRESH_S5}")
                    _cache = _build_locked_cache(_prop_conf, _prop_face, _prop_bary)
                    if frozen_template_conf is None:
                        _conf_full, _anchor_face_full, _anchor_bary_full = _dense_memory_from_cache(_cache)
                        _obs_conf_full, _, _ = _dense_memory_from_cache(
                            _cache, confidence_attr="observed_contact_confidence")
                        if _template_state == "confirmed":
                            _prop_conf, _prop_face, _prop_bary, _reliable_frames = _propagate_template(
                                _conf_full, _anchor_face_full, _anchor_bary_full, _obs_conf_full,
                                allow_bootstrap=False)
                            frozen_template_conf = _prop_conf.detach().clone()
                            frozen_template_face = _prop_face.detach().clone()
                            frozen_template_bary = _prop_bary.detach().clone()
                            frozen_template_reliable = _reliable_frames.detach().clone()
                            if bootstrap_template_seen:
                                _log_reclassify("  [reclassify] grasp template upgraded bootstrap -> confirmed/frozen")
                            else:
                                _log_reclassify("  [reclassify] grasp template confirmed/frozen")
                elif _template_state == "no_template_bootstrap_gap":
                    _log_reclassify(
                        "  [reclassify] no close-contact template seed; "
                        "staying in NO_TEMPLATE and using dynamic gap bootstrap")
                elif _n_reliable > 0:
                    _log_reclassify(f"  [reclassify] grasp template reliable_frames={_n_reliable}, "
                                    f"local={_reliable_local}, sampled={_reliable_sampled}, no extra anchors")
                if frozen_template_conf is None:
                    template_phase_s5 = "NO_TEMPLATE"
                else:
                    template_phase_s5 = "TEMPLATE_FROZEN"
                if _cache.active_mask.any() and _cache.contact_confidence is not None:
                    _active_conf = _cache.contact_confidence[_cache.active_mask]
                    if _active_conf.numel() > 0:
                        _log_reclassify(f"  [reclassify] contact conf mean={_active_conf.mean().item():.3f}, "
                                        f"max={_active_conf.max().item():.3f}")
                contact_confidence_full = _prop_conf
                contact_anchor_face_full = _prop_face
                contact_anchor_bary_full = _prop_bary
            # Build dense (N_inter, 778) argmax map, then roll along time to
            # get "previous video frame" prior for next snapshot.
            _new_full = torch.full((max(N_inter, 1), 778), -1, dtype=torch.long, device=device)
            if _cache.active_mask.any():
                _amax_safe = torch.where(_cache.active_mask, _cache.argmax_face,
                                         torch.full_like(_cache.argmax_face, -1))
                _new_full.scatter_(1, _cache.active_idx, _amax_safe)
            _rolled = torch.roll(_new_full, shifts=1, dims=0)
            if _rolled.shape[0] > 0:
                _rolled[0] = -1
            return _cache, _rolled

        _run_stage5()
        _run_stage6()

        if _has_inter and i0 < i1:
            with torch.no_grad():
                _obj_align_offsets, _hand_align_offsets, _align_info = compute_interaction_alignment_offsets(
                    multi_frame_model.trans.data,
                    multi_frame_model.mano_trans.data,
                    i0=i0,
                    i1=i1,
                    pre_buffer=i0,
                    post_buffer=max(0, num_sampled_frames - i1),
                    max_offset=0.08,
                    gamma=2.0,
                )
                multi_frame_model.trans.data.add_(_obj_align_offsets)
                _flat_align = torch.diag(torch.tensor(
                    [-1.0, -1.0, 1.0],
                    dtype=_hand_align_offsets.dtype,
                    device=_hand_align_offsets.device))
                multi_frame_model.mano_trans.data.add_(_hand_align_offsets @ _flat_align)
                print("  [post-s5] interaction alignment offsets: "
                      f"start_jump={_align_info['start_jump_norm'] * 1000.0:.2f}mm, "
                      f"end_jump={_align_info['end_jump_norm'] * 1000.0:.2f}mm, "
                      f"applied_start={_align_info['applied_start_norm'] * 1000.0:.2f}mm, "
                      f"applied_end={_align_info['applied_end_norm'] * 1000.0:.2f}mm, "
                      f"hand_pre={_align_info['hand_pre_norm'] * 1000.0:.2f}mm, "
                      f"hand_post={_align_info['hand_post_norm'] * 1000.0:.2f}mm, "
                      f"hand_pre_buf={_align_info['pre_buffer']}, "
                      f"hand_post_buf={_align_info['post_buffer']}")

    # --- 5. Export & Visualization ---
    # prepare sparse data
    sparse_R_tensor = rotation_6d_to_matrix(multi_frame_model.rot_6d).detach().cpu()
    sparse_T_np = multi_frame_model.trans.detach().cpu().numpy()
    final_scale = (multi_frame_model.scale * multi_frame_model.initial_scale).detach().cpu().numpy()

    # timeline
    t_sparse = np.array(sampled_indices)
    t_full = np.arange(num_total_frames)

    # --- Interpolation Improvement Start ---

    sparse_rot_obj = R.from_matrix(sparse_R_tensor.numpy())

    # create Slerp interpolator
    slerp = Slerp(t_sparse, sparse_rot_obj)

    # generate all frames' rotation
    full_rot_obj = slerp(t_full)
    final_R_full_batch = full_rot_obj.as_matrix()  # (N_total, 3, 3)

    cs = CubicSpline(t_sparse, sparse_T_np, axis=0)
    final_T_full_batch = cs(t_full)  # (N_total, 3)

    R_full_tensor = torch.from_numpy(final_R_full_batch).float().to(device)
    T_full_tensor = torch.from_numpy(final_T_full_batch).float().to(device)

    # because the full_model needs rot_6d format, we convert the interpolated matrix back to 6D
    rot_6d_full = matrix_to_rotation_6d(R_full_tensor)
    trans_full = T_full_tensor

    # --- Interpolation Improvement End ---

    # --- Interpolate MANO parameters ---
    # Extract sparse optimized parameters
    sparse_mano_root_orient_np = multi_frame_model.mano_root_orient.detach().cpu().numpy()
    sparse_mano_trans_np = multi_frame_model.mano_trans.detach().cpu().numpy()
    sparse_mano_pose_np = multi_frame_model.mano_pose.detach().cpu().numpy()

    # Interpolate translation with CubicSpline
    cs_mano_trans = CubicSpline(t_sparse, sparse_mano_trans_np, axis=0)
    final_mano_T_full_batch = cs_mano_trans(t_full)

    # Interpolate root orientation with Slerp
    sparse_mano_root_rot = R.from_rotvec(sparse_mano_root_orient_np)
    slerp_mano_root = Slerp(t_sparse, sparse_mano_root_rot)
    full_mano_root_rot = slerp_mano_root(t_full)
    final_mano_root_full_batch = full_mano_root_rot.as_rotvec()

    # Interpolate hand pose (15 joints) with Slerp
    final_mano_pose_full_list = []
    for j in range(sparse_mano_pose_np.shape[1]):
        sparse_joint_pose = sparse_mano_pose_np[:, j, :]
        sparse_joint_rot = R.from_rotvec(sparse_joint_pose)
        slerp_joint = Slerp(t_sparse, sparse_joint_rot)
        full_joint_rot = slerp_joint(t_full)
        final_mano_pose_full_list.append(full_joint_rot.as_rotvec())
    final_mano_pose_full_batch = np.stack(final_mano_pose_full_list, axis=1)

    # Convert all to tensors for rendering
    mano_root_full_tensor = torch.from_numpy(final_mano_root_full_batch).float().to(device)
    mano_trans_full_tensor = torch.from_numpy(final_mano_T_full_batch).float().to(device)
    mano_pose_full_tensor = torch.from_numpy(final_mano_pose_full_batch).float().to(device)
    is_right_full_tensor = torch.stack([torch.tensor(mp['is_right'], dtype=torch.float32) for mp in mano_params]).to(device)
    # --- Interpolation for MANO End ---

    # Save the optimized transform sequence (full length)
    optimized_poses = []

    for i in range(num_total_frames):
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = final_R_full_batch[i] * final_scale
        transform_matrix[:3, 3] = final_T_full_batch[i]
        optimized_poses.append({'transform': transform_matrix.tolist()})

    save_path = os.path.join(output_path, 'optimized_transform_sequence.json')
    with open(save_path, 'w') as f:
        json.dump(optimized_poses, f, indent=4)

    render_batch_size = num_total_frames
    final_rendered_images_list = []
    final_rendered_images_side_list = []  # side-view renders (same loop)
    final_rendered_images_top_list  = []  # top-view renders (same loop)

    all_obj_verts_list = []
    all_obj_faces = None
    all_hand_verts_list = []
    all_hand_faces = None
    all_hand_joints_list = []

    print("Rendering full sequence (RGB + side-view) in batches...")

    lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])

    materials = Materials(device=device, specular_color=[[1.0, 1.0, 1.0]], shininess=1.0)

    blend_params = BlendParams(background_color=(0.0, 0.0, 0.0))

    raster_settings = RasterizationSettings(image_size=(H_out, W_out), blur_radius=0.0, faces_per_pixel=1)
    renderer = MeshRenderer(rasterizer=MeshRasterizer(raster_settings=raster_settings),
                            shader=HardPhongShader(device=device, lights=lights, blend_params=blend_params))

    with torch.no_grad():
        for i in range(0, num_total_frames, render_batch_size):
            end_idx = min(i + render_batch_size, num_total_frames)
            current_batch_size = end_idx - i
            current_R = rotation_6d_to_matrix(rot_6d_full[i:end_idx])
            current_T = trans_full[i:end_idx]
            current_scale = torch.from_numpy(final_scale).float().to(device)
            current_verts = verts[None, ...].repeat(current_batch_size, 1, 1)
            current_faces = faces[None, ...].repeat(current_batch_size, 1, 1)

            # Get interpolated MANO parameters for the current batch
            current_mano_root = mano_root_full_tensor[i:end_idx]
            current_mano_trans = mano_trans_full_tensor[i:end_idx]
            current_mano_pose = mano_pose_full_tensor[i:end_idx]
            current_is_right = is_right_full_tensor[i:end_idx]

            current_model = TemporalHandObjectPose(
                current_R,
                current_T,
                current_scale,
                current_verts,
                current_faces,
                mano_params=None,  # Not needed for this branch
                mano_root_orient_init=current_mano_root,
                mano_trans_init=current_mano_trans,
                mano_pose_init=current_mano_pose,
                is_right_init=current_is_right).to(device)

            current_obj_meshes, current_hand_meshes, current_mano_joints = current_model()

            obj_color = [0.65, 0.8, 1.0]  # blue color
            hand_color = [1.0, 0.0, 0.0]  # red color

            N, V = current_obj_meshes.verts_padded().shape[:2]
            current_verts_rgb = torch.tensor(obj_color, device=device).view(1, 1, 3).expand(N, V, -1)
            current_obj_meshes.textures = TexturesVertex(verts_features=current_verts_rgb)

            N, V = current_hand_meshes.verts_padded().shape[:2]
            current_verts_rgb = torch.tensor(hand_color, device=device).view(1, 1, 3).expand(N, V, -1)
            current_hand_meshes.textures = TexturesVertex(verts_features=current_verts_rgb)

            for j in range(current_batch_size):
                if save_intermediates:
                    current_mesh = current_obj_meshes[j]
                    os.makedirs(os.path.join(output_path, 'optimized_object_meshes'), exist_ok=True)
                    save_path = os.path.join(output_path, 'optimized_object_meshes', f"{i*render_batch_size+j:05d}.obj")
                    # x: left; y: up; z: forward -> x: right; y: down; z: forward
                    flat_mat = torch.diag(torch.tensor([-1., -1., 1.], dtype=torch.float32, device=device))
                    save_obj(save_path, current_mesh.verts_list()[0].squeeze() @ flat_mat, current_mesh.faces_list()[0].squeeze())

            # Collect mesh data for k3d visualization, replacing the file saving loop
            flat_mat = torch.diag(torch.tensor([-1., -1., 1.], dtype=torch.float32, device=device))
            all_obj_verts_list.append((current_obj_meshes.verts_padded() @ flat_mat).cpu().numpy())
            if all_obj_faces is None:
                all_obj_faces = current_obj_meshes.faces_padded()[0].cpu().numpy().astype(np.uint32)

            all_hand_verts_list.append((current_hand_meshes.verts_padded() @ flat_mat).cpu().numpy())
            all_hand_joints_list.append((current_mano_joints @ flat_mat).cpu().numpy())
            if all_hand_faces is None:
                all_hand_faces = current_hand_meshes.faces_padded()[0].cpu().numpy().astype(np.uint32)

            if save_intermediates:
                current_camera = PerspectiveCameras(
                    focal_length=torch.tensor([[fx_new, fy_new]], device=device).expand(current_batch_size, -1),
                    principal_point=torch.tensor([[cx_new, cy_new]], device=device).expand(current_batch_size, -1),
                    image_size=((H_out, W_out),),
                    in_ndc=False,
                    device=device,
                )

                scenes = []
                for j in range(current_batch_size):
                    scene_j = join_meshes_as_scene([current_obj_meshes[j], current_hand_meshes[j]])
                    scenes.append(scene_j)
                current_scene = join_meshes_as_batch(scenes)

                current_images = renderer(current_scene, cameras=current_camera, lights=lights, materials=materials)

                current_rgb = current_images[..., :3]

                final_rendered_images_list.append(current_rgb.cpu().numpy())

                # --- Side-view rendering ---
                # Rotate the scene 90° around the Y axis at the scene centroid.
                # After rotation: old depth (Z) → horizontal (X);
                #                 old horizontal (X) → depth (−Z).
                # The centroid is preserved so the scene remains in front of the camera.
                _ov = current_obj_meshes.verts_padded()  # (bs, V_obj,  3)
                _hv = current_hand_meshes.verts_padded()  # (bs, V_hand, 3)
                _ctr = torch.cat([_ov, _hv], dim=1).reshape(-1, 3).mean(dim=0)  # (3,)

                def _side_rot(v):
                    w = v - _ctr
                    # 90° Y rotation: (x,y,z) → (z, y, −x)
                    return torch.stack([w[..., 2], w[..., 1], -w[..., 0]], dim=-1) + _ctr

                def _top_rot(v):
                    w = v - _ctr
                    # −90° X rotation: (x,y,z) → (x, z, −y)  — looking down from above
                    return torch.stack([w[..., 0], w[..., 2], -w[..., 1]], dim=-1) + _ctr

                side_ov = _side_rot(_ov)
                side_hv = _side_rot(_hv)
                top_ov  = _top_rot(_ov)
                top_hv  = _top_rot(_hv)

                side_obj_meshes = Meshes(verts=list(side_ov), faces=current_obj_meshes.faces_list())
                side_hand_meshes = Meshes(verts=list(side_hv), faces=current_hand_meshes.faces_list())

                N, V = side_obj_meshes.verts_padded().shape[:2]
                side_obj_meshes.textures = TexturesVertex(verts_features=torch.tensor(obj_color, device=device).view(1, 1, 3).expand(N, V, -1))
                N, V = side_hand_meshes.verts_padded().shape[:2]
                side_hand_meshes.textures = TexturesVertex(verts_features=torch.tensor(hand_color, device=device).view(1, 1, 3).expand(N, V, -1))

                side_scenes = [join_meshes_as_scene([side_obj_meshes[j], side_hand_meshes[j]]) for j in range(current_batch_size)]
                side_scene_batch = join_meshes_as_batch(side_scenes)
                side_images = renderer(side_scene_batch, cameras=current_camera, lights=lights, materials=materials)
                final_rendered_images_side_list.append(side_images[..., :3].cpu().numpy())

                top_obj_meshes  = Meshes(verts=list(top_ov), faces=current_obj_meshes.faces_list())
                top_hand_meshes = Meshes(verts=list(top_hv), faces=current_hand_meshes.faces_list())
                N, V = top_obj_meshes.verts_padded().shape[:2]
                top_obj_meshes.textures  = TexturesVertex(verts_features=torch.tensor(obj_color,  device=device).view(1, 1, 3).expand(N, V, -1))
                N, V = top_hand_meshes.verts_padded().shape[:2]
                top_hand_meshes.textures = TexturesVertex(verts_features=torch.tensor(hand_color, device=device).view(1, 1, 3).expand(N, V, -1))
                top_scenes = [join_meshes_as_scene([top_obj_meshes[j], top_hand_meshes[j]]) for j in range(current_batch_size)]
                top_images = renderer(join_meshes_as_batch(top_scenes), cameras=current_camera, lights=lights, materials=materials)
                final_rendered_images_top_list.append(top_images[..., :3].cpu().numpy())

            # del current_meshes, current_camera, current_images, current_rgb, current_model, current_verts_rgb
            # torch.cuda.empty_cache()

    if save_intermediates:
        final_rendered_images_full = np.concatenate(final_rendered_images_list, axis=0)
        final_rendered_images_side = np.concatenate(final_rendered_images_side_list, axis=0)
        final_rendered_images_top  = np.concatenate(final_rendered_images_top_list,  axis=0)
        print(f"Finished rendering {len(final_rendered_images_full)} frames (camera + side + top view).")

        # ── Video 1: optimized_fitting.mp4
        #    Panels: modal | amodal | camera-render | hand-mask | side-render
        video_path = os.path.join(output_path, 'optimized_fitting.mp4')
        writer = imageio.get_writer(video_path,
                                    fps=30,
                                    codec='libx264',
                                    pixelformat='yuv420p',
                                    ffmpeg_params=['-crf', '28', '-preset', 'veryfast'],
                                    macro_block_size=None)

        for i in range(num_total_frames):
            modal_frame = overlay_mask_on_image(cropped_rgbs_np[i], cropped_modal_masks_np[i])
            amodal_gt_frame = overlay_mask_on_image(cropped_rgbs_np[i], pred_amodal_masks_np[i])
            render_frame = overlay_rgb_render(cropped_rgbs_np[i], final_rendered_images_full[i], alpha=1.0)
            hand_frame = overlay_mask_on_image(cropped_rgbs_np[i], cropped_hand_masks_np[i], cmap_idx=1)
            side_frame = np.clip(final_rendered_images_side[i], 0, 1)
            top_frame  = np.clip(final_rendered_images_top[i],  0, 1)
            panels = [modal_frame, amodal_gt_frame, render_frame, hand_frame, side_frame, top_frame]

            row1 = np.hstack(panels[:3])
            row2 = np.hstack(panels[3:])
            _full_frame = (np.vstack([row1, row2]) * 255).astype(np.uint8)
            _full_label, _full_color = _full_video_label(i)
            writer.append_data(_draw_video_label(_full_frame, _full_label, _full_color))

        writer.close()
        print(f"Saved optimized_fitting.mp4 → {video_path}")

    if save_intermediates:
        # ── Video 2: optimized_fitting_SPARSE_final.mp4
        #    Panels: amodal | camera-render | side-render  (sparse sampled frames)
        sparse_final_video_path = os.path.join(output_path, 'optimized_fitting_SPARSE_final.mp4')
        writer_sf = imageio.get_writer(sparse_final_video_path,
                                       fps=30,
                                       codec='libx264',
                                       pixelformat='yuv420p',
                                       ffmpeg_params=['-crf', '28', '-preset', 'veryfast'],
                                       macro_block_size=None)

        _sf_camera = PerspectiveCameras(
            focal_length=focal_length,
            principal_point=principal_point,
            image_size=((H_out, W_out),) * num_sampled_frames,
            in_ndc=False,
            device=device,
        )

        with torch.no_grad():
            _sf_obj_m, _sf_hand_m, _ = multi_frame_model()

            N, V = _sf_obj_m.verts_padded().shape[:2]
            _sf_obj_m.textures = TexturesVertex(verts_features=torch.tensor([0.65, 0.8, 1.0], device=device).view(1, 1, 3).expand(N, V, -1))
            N, V = _sf_hand_m.verts_padded().shape[:2]
            _sf_hand_m.textures = TexturesVertex(verts_features=torch.tensor([1.0, 0.0, 0.0], device=device).view(1, 1, 3).expand(N, V, -1))

            _sf_scenes = [join_meshes_as_scene([_sf_obj_m[j], _sf_hand_m[j]]) for j in range(num_sampled_frames)]
            _sf_scene_batch = join_meshes_as_batch(_sf_scenes)
            _sf_rgb = renderer(_sf_scene_batch, cameras=_sf_camera, lights=lights, materials=materials)[..., :3].cpu().numpy()

            # Also render side/top views for sparse frames.
            _sf_ov = _sf_obj_m.verts_padded()
            _sf_hv = _sf_hand_m.verts_padded()
            _sf_ctr = torch.cat([_sf_ov, _sf_hv], dim=1).reshape(-1, 3).mean(dim=0)

            def _sf_side_rot(v):
                w = v - _sf_ctr
                return torch.stack([w[..., 2], w[..., 1], -w[..., 0]], dim=-1) + _sf_ctr

            def _sf_top_rot(v):
                w = v - _sf_ctr
                return torch.stack([w[..., 0], w[..., 2], -w[..., 1]], dim=-1) + _sf_ctr

            _sf_side_obj = Meshes(verts=list(_sf_side_rot(_sf_ov)), faces=_sf_obj_m.faces_list())
            _sf_side_hand = Meshes(verts=list(_sf_side_rot(_sf_hv)), faces=_sf_hand_m.faces_list())
            N, V = _sf_side_obj.verts_padded().shape[:2]
            _sf_side_obj.textures = TexturesVertex(verts_features=torch.tensor([0.65, 0.8, 1.0], device=device).view(1, 1, 3).expand(N, V, -1))
            N, V = _sf_side_hand.verts_padded().shape[:2]
            _sf_side_hand.textures = TexturesVertex(verts_features=torch.tensor([1.0, 0.0, 0.0], device=device).view(1, 1, 3).expand(N, V, -1))
            _sf_side_scenes = [join_meshes_as_scene([_sf_side_obj[j], _sf_side_hand[j]]) for j in range(num_sampled_frames)]
            _sf_side_rgb = renderer(join_meshes_as_batch(_sf_side_scenes), cameras=_sf_camera, lights=lights, materials=materials)[..., :3].cpu().numpy()

            _sf_top_obj  = Meshes(verts=list(_sf_top_rot(_sf_ov)), faces=_sf_obj_m.faces_list())
            _sf_top_hand = Meshes(verts=list(_sf_top_rot(_sf_hv)), faces=_sf_hand_m.faces_list())
            N, V = _sf_top_obj.verts_padded().shape[:2]
            _sf_top_obj.textures  = TexturesVertex(verts_features=torch.tensor([0.65, 0.8, 1.0], device=device).view(1, 1, 3).expand(N, V, -1))
            N, V = _sf_top_hand.verts_padded().shape[:2]
            _sf_top_hand.textures = TexturesVertex(verts_features=torch.tensor([1.0, 0.0, 0.0], device=device).view(1, 1, 3).expand(N, V, -1))
            _sf_top_scenes = [join_meshes_as_scene([_sf_top_obj[j], _sf_top_hand[j]]) for j in range(num_sampled_frames)]
            _sf_top_rgb = renderer(join_meshes_as_batch(_sf_top_scenes), cameras=_sf_camera, lights=lights, materials=materials)[..., :3].cpu().numpy()

        for i in range(num_sampled_frames):
            amodal_gt = overlay_mask_on_image(sampled_rgbs_np[i], sampled_pred_amodal_masks_np[i])
            cam_render = overlay_rgb_render(sampled_rgbs_np[i], _sf_rgb[i], alpha=1.0)
            side_render = np.clip(_sf_side_rgb[i], 0, 1)
            top_render  = np.clip(_sf_top_rgb[i],  0, 1)
            panels_sf = [amodal_gt, cam_render, side_render, top_render]
            row1_sf = np.hstack(panels_sf[:2])
            row2_sf = np.hstack(panels_sf[2:])
            _sf_frame = (np.vstack([row1_sf, row2_sf]) * 255).astype(np.uint8)
            _sf_label, _sf_color = _sampled_video_label(i)
            writer_sf.append_data(_draw_video_label(_sf_frame, _sf_label, _sf_color))

        writer_sf.close()
        print(f"Saved sparse-final video → {sparse_final_video_path}")

    # Consolidate collected mesh data
    all_obj_verts = np.concatenate(all_obj_verts_list, axis=0)
    all_hand_verts = np.concatenate(all_hand_verts_list, axis=0)
    all_hand_joints = np.concatenate(all_hand_joints_list, axis=0)

    if save_intermediates:
        # save hand-object mesh sequences using k3d to html
        save_k3d_visualization(output_path, seq_path, all_obj_verts, all_obj_faces, all_hand_verts, all_hand_faces)

    # --- Save optimized HOI sequence to disk ---
    # Consolidate collected mesh data and prepare for saving
    final_obj_rot_mat = R_full_tensor.cpu().numpy()  # This is row-major
    final_obj_rot_mat_col_major = final_obj_rot_mat.transpose(0, 2, 1)  # Transpose each 3x3 matrix for column-major
    final_obj_trans = T_full_tensor.cpu().numpy()
    final_obj_scale = np.repeat(final_scale[np.newaxis, :], num_total_frames, axis=0)
    canonical_verts_np = verts.cpu()
    canonical_faces_np = faces.cpu()

    # Keep ablation final outputs isolated from the original input sequence tree.
    output_dir = os.path.join(output_path, "optimized_hoi_contact_seq")
    paper_seq_name = f"{os.path.basename(seq_path)}_{ablation}"

    save_hoi_sequence(output_dir, canonical_verts_np, canonical_faces_np, final_obj_scale, final_obj_rot_mat_col_major, final_obj_trans, mano_root_full_tensor,
                      mano_pose_full_tensor, mano_trans_full_tensor, is_right_full_tensor, all_obj_verts, all_obj_faces, all_hand_verts, all_hand_faces,
                      paper_seq_name=paper_seq_name)
    save_ablation_eval_data(output_path, seq_path, os.path.basename(seq_path), all_obj_verts, all_obj_faces, all_hand_verts, all_hand_faces, all_hand_joints)

    # # =========================================================================
    # # In-the-Wild Metrics
    # # =========================================================================
    # print("\n" + "=" * 80)
    # print("Computing in-the-wild metrics...")
    # print("=" * 80)

    # from in_the_wild_metric import compute_all_metrics as _compute_itw_metrics

    # # all_obj_verts / all_hand_verts have flat_mat (diag[-1,-1,1]) applied for
    # # visualization.  Applying flat_mat again undoes the flip (flat_mat^2 = I)
    # # and recovers original world-space coordinates suitable for 3-D metrics.
    # _flat_mat_np = np.diag(np.array([-1., -1., 1.]))
    # _obj_verts_world = all_obj_verts @ _flat_mat_np  # (T, V_obj,  3)
    # _hand_verts_world = all_hand_verts @ _flat_mat_np  # (T, V_hand, 3)

    # # Same intrinsics for all frames
    # _fl_full = np.tile(np.array([[float(fx_new), float(fy_new)]]), (num_total_frames, 1))  # (T, 2)
    # _pp_full = np.tile(np.array([[float(cx_new), float(cy_new)]]), (num_total_frames, 1))  # (T, 2)

    # # Map interaction segment from sampled-frame index space to full-frame index space.
    # # approaching_end_idx / interaction_end_idx index into sampled_indices.
    # _inter_start_full = (int(sampled_indices[approaching_end_idx]) if approaching_end_idx < num_sampled_frames else num_total_frames)
    # _last_inter_sidx = max(0, interaction_end_idx - 1)
    # _inter_end_full = (int(sampled_indices[min(_last_inter_sidx, num_sampled_frames - 1)]) +
    #                    1 if interaction_end_idx > approaching_end_idx else _inter_start_full)

    # try:
    #     itw_metrics = _compute_itw_metrics(
    #         obj_verts_seq=_obj_verts_world,
    #         obj_faces=all_obj_faces.astype(np.int32),
    #         hand_verts_seq=_hand_verts_world,
    #         hand_faces=all_hand_faces.astype(np.int32),
    #         amodal_masks=pred_amodal_masks_np,
    #         focal_lengths=_fl_full,
    #         principal_points=_pp_full,
    #         H=H_out,
    #         W=W_out,
    #         interaction_start=_inter_start_full,
    #         interaction_end=_inter_end_full,
    #         fps=30.0,
    #         unit_to_cm=1.0,
    #         device=str(device),
    #         verbose=True,
    #     )
    # except Exception as _e:
    #     import traceback as _tb
    #     print(f"[Warning] Metric computation failed: {_e}")
    #     _tb.print_exc()
    #     itw_metrics = {}

    # metric_save_path = os.path.join(output_path, 'metric_in_the_wild.json')
    # with open(metric_save_path, 'w') as _mf:
    #     json.dump({k: float(v) for k, v in itw_metrics.items()}, _mf, indent=4)
    # print(f"In-the-wild metrics saved -> {metric_save_path}")




def process_sequence(seq_path, data_output_path, worker_model_state, args, fps=30):
    # --- 1. Initialization and Raw Data Loading ---
    seq_name = os.path.basename(seq_path)
    output_seq_path = os.path.join(data_output_path, seq_name)
    os.makedirs(output_seq_path, exist_ok=True)
    hand_inputs = resolve_hand_inputs(seq_path, getattr(args, "hand_side", None))

    bbox_save_path = os.path.join(output_seq_path, 'global_bbox.json')
    amodal_masks_save_dir = os.path.join(output_seq_path, 'amodal_masks')
    cropped_depths_save_dir = os.path.join(output_seq_path, 'cropped_depths')
    cropped_metric_depths_save_dir = os.path.join(output_seq_path, 'cropped_metric_depths')

    # Load raw frames once at the beginning
    raw_rgbs_np = load_raw_frames(seq_path + "/rgbs", frame_type='rgb')
    raw_masks_np = load_raw_frames(seq_path + "/obj_masks", frame_type='mask')

    # Load hand masks if they exist. Dual inputs must use the selected hand only.
    if hand_inputs.selected_hand_side is not None:
        if hand_inputs.hand_masks_dir is not None:
            print(f"Found {hand_inputs.selected_hand_side} hand masks in {hand_inputs.hand_masks_dir}")
            raw_hand_masks_np = load_raw_frames(hand_inputs.hand_masks_dir, frame_type='mask')
        else:
            raw_hand_masks_np = np.zeros_like(raw_masks_np)
    else:
        raw_lh_masks_np = None
        lh_masks_path = os.path.join(seq_path, 'lh_masks')
        if os.path.exists(lh_masks_path):
            print(f"Found lh_masks in {seq_path}")
            raw_lh_masks_np = load_raw_frames(lh_masks_path, frame_type='mask')

        raw_rh_masks_np = None
        rh_masks_path = os.path.join(seq_path, 'rh_masks')
        if os.path.exists(rh_masks_path):
            print(f"Found rh_masks in {seq_path}")
            raw_rh_masks_np = load_raw_frames(rh_masks_path, frame_type='mask')

        raw_hand_masks_np = None
        if raw_lh_masks_np is not None and raw_rh_masks_np is not None:
            raw_hand_masks_np = np.logical_or(raw_lh_masks_np, raw_rh_masks_np).astype(np.uint8)
        elif raw_lh_masks_np is not None:
            raw_hand_masks_np = raw_lh_masks_np
        elif raw_rh_masks_np is not None:
            raw_hand_masks_np = raw_rh_masks_np
        else:
            raw_hand_masks_np = np.zeros_like(raw_masks_np)

    num_frames = len(raw_rgbs_np)
    pred_res = (256, 512)

    # --- 2. Get Global BBox for Cropping ---
    # This part runs every time as it's fast. Bbox is saved with amodal masks later.
    hand_bboxes = load_hand_data(seq_path, num_frames, hand_side=hand_inputs.selected_hand_side)
    global_bboxes = get_global_amodal_bbox(raw_masks_np, hand_bboxes)

    # --- 3. Cropped Depth Preparation (with Caching) ---
    if os.path.exists(cropped_depths_save_dir):
        print(f"--- Loading pre-computed cropped depths for {seq_name} ---")
        loaded_cropped_depths_list = load_raw_frames(cropped_depths_save_dir, frame_type='depth')

        # Manually convert loaded numpy [0,1] to tensor [-1,1] to match pipeline format
        processed_frames = []
        to_tensor = transforms.ToTensor()
        normalizer = transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
        for frame_np in loaded_cropped_depths_list:
            tensor_frame = to_tensor(frame_np[:, :, np.newaxis]).repeat(3, 1, 1)
            transformed_frame = normalizer(tensor_frame)
            processed_frames.append(transformed_frame)
        depth_pixels_tensor = torch.stack(processed_frames).unsqueeze(0)
    else:
        print(f"--- Running inference for full-frame depths for {seq_name} ---")
        if worker_model_state["depth_model"] is None:
            print(f"GPU {torch.cuda.current_device()}: Loading depth model...")
            worker_model_state["depth_model"] = init_depth_model(args.model_path_depth + f"/depth_anything_v2_{args.depth_encoder}.pth", args.depth_encoder)
        depth_model = worker_model_state["depth_model"]

        raw_depths_np = get_raw_depth_maps(raw_rgbs_np, depth_model)
        depth_pixels_tensor = crop_and_resize_frames(raw_depths_np, global_bboxes, pred_res, frame_type='depth')

        print(f"--- Saving cropped depths to {cropped_depths_save_dir} ---")
        os.makedirs(cropped_depths_save_dir, exist_ok=True)
        depth_to_save_float = (depth_pixels_tensor.squeeze(0).permute(0, 2, 3, 1).cpu().numpy() + 1) / 2.0
        for i, depth_img_float in enumerate(depth_to_save_float):
            depth_img_uint16 = (depth_img_float[:, :, 0] * 65535).astype(np.uint16)
            cv2.imwrite(os.path.join(cropped_depths_save_dir, f"{i}.png"), depth_img_uint16)

    # --- 4. Amodal Mask Preparation (with Caching) ---
    if os.path.exists(bbox_save_path) and os.path.exists(amodal_masks_save_dir):
        print(f"--- Loading pre-computed amodal masks for {seq_name} ---")
        mask_files = sorted(os.listdir(amodal_masks_save_dir), key=lambda x: int(os.path.splitext(x)[0]))
        pred_amodal_masks_np = np.array([(cv2.imread(os.path.join(amodal_masks_save_dir, f), cv2.IMREAD_GRAYSCALE) > 128).astype(np.uint8) for f in mask_files])
    else:
        print(f"--- Running inference for amodal masks for {seq_name} ---")
        if worker_model_state["pipeline_mask"] is None:
            print(f"GPU {torch.cuda.current_device()}: Loading amodal segmentation model...")
            worker_model_state["pipeline_mask"] = init_amodal_segmentation_model(args.model_path_mask)
            worker_model_state["generator"] = torch.manual_seed(23)
        pipeline_mask = worker_model_state["pipeline_mask"]
        generator = worker_model_state["generator"]

        # Prepare other input tensors
        rgb_pixels_tensor = crop_and_resize_frames(raw_rgbs_np, global_bboxes, pred_res, frame_type='rgb')
        modal_pixels_tensor = crop_and_resize_frames(raw_masks_np, global_bboxes, pred_res, frame_type='mask')

        amodal_windows = build_temporal_windows(
            num_frames,
            window_size=args.amodal_window_size,
            overlap=args.amodal_window_overlap,
        )
        print(
            "Amodal segmentation by diffusion-vas "
            f"({len(amodal_windows)} windows, size={args.amodal_window_size}, "
            f"overlap={args.amodal_window_overlap})..."
        )
        pred_amodal_masks_processed = np.zeros((num_frames, pred_res[0], pred_res[1]), dtype=np.uint8)
        for win_start, win_end in tqdm(amodal_windows, desc="Amodal windows"):
            window_len = win_end - win_start
            pred_amodal_masks_raw = pipeline_mask(
                modal_pixels_tensor[:, win_start:win_end],
                depth_pixels_tensor[:, win_start:win_end],
                height=pred_res[0],
                width=pred_res[1],
                num_frames=window_len,
                decode_chunk_size=min(8, window_len),
                motion_bucket_id=127,
                fps=8,
                noise_aug_strength=0.02,
                min_guidance_scale=1.5,
                max_guidance_scale=1.5,
                generator=generator,
            ).frames[0]

            window_masks = (
                np.array([np.array(img) for img in pred_amodal_masks_raw])
                .astype('uint8')
                .sum(axis=-1) > 600
            ).astype('uint8')
            pred_amodal_masks_processed[win_start:win_end] = window_masks
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        modal_mask_union_cropped = (modal_pixels_tensor[0, :, 0, :, :].cpu().numpy() > 0).astype('uint8')
        pred_amodal_masks_np = np.logical_or(pred_amodal_masks_processed, modal_mask_union_cropped).astype('uint8')

        print(f"--- Saving amodal masks and bbox to {output_seq_path} ---")
        os.makedirs(amodal_masks_save_dir, exist_ok=True)
        for i, mask in enumerate(pred_amodal_masks_np):
            cv2.imwrite(os.path.join(amodal_masks_save_dir, f"{i}.png"), mask * 255)
        with open(bbox_save_path, 'w') as f:
            json.dump(global_bboxes, f)

    # --- 5. Cropped Metric Depth Preparation (with Caching) ---
    if os.path.exists(cropped_metric_depths_save_dir):
        print(f"--- Loading pre-computed cropped metric depths for {seq_name} ---")
        loaded_cropped_metric_depths_list = load_raw_frames(cropped_metric_depths_save_dir, frame_type='metric_depth')

        # Manually convert loaded numpy [0,1] to tensor [-1,1] to match pipeline format
        processed_frames = []
        for frame_np in loaded_cropped_metric_depths_list:
            tensor_frame = torch.from_numpy(frame_np[:, :, np.newaxis]).float().permute(2, 0, 1).repeat(3, 1, 1)
            processed_frames.append(tensor_frame)
        device = f"cuda:{torch.cuda.current_device()}"
        metric_depth_pixels_tensor = torch.stack(processed_frames).unsqueeze(0).to(device)
    else:
        print(f"--- Running inference for full-frame metric depths for {seq_name} ---")
        if worker_model_state["metric_depth_model"] is None:
            print(f"GPU {torch.cuda.current_device()}: Loading metric depth model...")
            worker_model_state["metric_depth_model"] = init_metric_depth_model()
        metric_depth_model = worker_model_state["metric_depth_model"]

        raw_metric_depths_np = get_raw_metric_depth_maps(raw_rgbs_np, metric_depth_model)
        metric_depth_pixels_tensor = crop_and_resize_frames(raw_metric_depths_np, global_bboxes, pred_res, frame_type='metric_depth')

        print(f"--- Saving cropped metric depths to {cropped_metric_depths_save_dir} ---")
        os.makedirs(cropped_metric_depths_save_dir, exist_ok=True)
        metric_depth_to_save_float = metric_depth_pixels_tensor.squeeze(0).permute(0, 2, 3, 1).cpu().numpy()
        for i, metric_depth_img_float in enumerate(metric_depth_to_save_float):
            metric_depth_img_uint16 = (metric_depth_img_float[:, :, 0] * 1000).astype(np.uint16)  # metres → millimetres, uint16 max 65535 ≈ 65 m
            cv2.imwrite(os.path.join(cropped_metric_depths_save_dir, f"{i}.png"), metric_depth_img_uint16)

    # --- 5. Final Data Preparation for Visualization and Fitting ---
    # Create cropped numpy arrays from raw data using the global bbox
    # These are used for generating visualization videos.
    rgb_pixels_tensor = crop_and_resize_frames(raw_rgbs_np, global_bboxes, pred_res, frame_type='rgb')
    modal_pixels_tensor = crop_and_resize_frames(raw_masks_np, global_bboxes, pred_res, frame_type='mask')

    cropped_rgbs_np = (rgb_pixels_tensor.squeeze(0).permute(0, 2, 3, 1).cpu().numpy() + 1) / 2.0
    cropped_modal_masks_np = (modal_pixels_tensor.squeeze(0)[:, 0, :, :].cpu().numpy() > 0).astype(np.uint8)

    hand_pixels_tensor = crop_and_resize_frames(raw_hand_masks_np, global_bboxes, pred_res, frame_type='mask')
    cropped_hand_masks_np = (hand_pixels_tensor.squeeze(0)[:, 0, :, :].cpu().numpy() > 0).astype(np.uint8)

    # The depth tensor is already normalized, convert to numpy for fitting function
    cropped_depths_np = (depth_pixels_tensor.squeeze(0).permute(0, 2, 3, 1).cpu().numpy() + 1) / 2.0
    cropped_metric_depths_np = metric_depth_pixels_tensor.squeeze(0).permute(0, 2, 3, 1).cpu().numpy()
    # Load mano params
    mano_params_dir = hand_inputs.mano_params_dir
    mano_params_files = sorted(
        [f for f in os.listdir(mano_params_dir) if f.endswith('.json') and f != 'export_meta.json'],
        key=lambda x: int(os.path.splitext(x)[0]),
    )
    start_frame, end_frame = int(os.path.splitext(mano_params_files[0])[0]), int(os.path.splitext(mano_params_files[-1])[0]) + 1
    mano_params = np.array([json.load(open(os.path.join(mano_params_dir, f))) for f in mano_params_files])

    # Load 2D hand keypoints
    hand_keypoints_file = hand_inputs.keypoints_file
    if hand_keypoints_file is None:
        if hand_inputs.selected_hand_side is not None:
            raise FileNotFoundError(f"Missing keypoints file for {hand_inputs.selected_hand_side} hand in {seq_path}")
        hand_keypoints_candidates = glob.glob(os.path.join(seq_path, '*h_keypoints.json'))
        if not hand_keypoints_candidates:
            raise FileNotFoundError(f"Missing hand keypoints file in {seq_path}")
        hand_keypoints_file = sorted(hand_keypoints_candidates)[0]
    hand_keypoints_data = json.load(open(hand_keypoints_file))

    hand_keypoints_list = []
    hand_keypoints_valid_mask_list = []
    dummy_keypoints = np.zeros((21, 2))  # Dummy value for missing keypoints

    for i in range(start_frame, end_frame):
        key = str(i)
        if key in hand_keypoints_data and hand_keypoints_data[key] is not None:
            hand_keypoints_list.append(np.array(hand_keypoints_data[key]))
            hand_keypoints_valid_mask_list.append(True)
        else:
            hand_keypoints_list.append(dummy_keypoints)
            hand_keypoints_valid_mask_list.append(False)

    hand_keypoints_np = np.stack(hand_keypoints_list, axis=0)
    hand_keypoints_valid_mask_np = np.array(hand_keypoints_valid_mask_list, dtype=bool)

    # --- 6. Save Comparison Visualization Video ---
    tmp_cmap_idx = np.random.randint(0, plt.get_cmap("tab10").N)
    comparison_video_path = f"{output_seq_path}/comparison_modal_amodal.mp4"
    combined_frames = []
    for i in range(num_frames):
        modal_overlay = overlay_mask_on_image(cropped_rgbs_np[i], cropped_modal_masks_np[i], cmap_idx=tmp_cmap_idx)
        amodal_overlay = overlay_mask_on_image(cropped_rgbs_np[i], pred_amodal_masks_np[i].astype(np.uint8), cmap_idx=tmp_cmap_idx)

        hand_overlay = overlay_mask_on_image(cropped_rgbs_np[i], cropped_hand_masks_np[i], cmap_idx=1)
        panels = [modal_overlay, amodal_overlay, hand_overlay]
        combined_frame = np.hstack(panels)
        combined_frame_uint8 = (combined_frame * 255).astype(np.uint8)
        combined_frames.append(_draw_video_label(combined_frame_uint8, f"full={i}"))
    imageio.mimwrite(
        comparison_video_path,
        np.stack(combined_frames),
        format='ffmpeg',
        fps=float(fps),
        macro_block_size=None,
        ffmpeg_params=['-crf', '28', '-preset', 'veryfast'],
        codec='libx264',
        pixelformat='yuv420p',
    )
    print(f"Saved comparison video to {comparison_video_path}")

    # --- 7. Start Pose Fitting ---
    device = f"cuda:{torch.cuda.current_device()}"

    if worker_model_state["cotracker_model"] is None:
        print(f"GPU {torch.cuda.current_device()}: Loading CoTracker model...")
        from cotracker.predictor import CoTrackerPredictor
        worker_model_state["cotracker_model"] = CoTrackerPredictor(checkpoint=args.model_path_cotracker).to(device)
    cotracker_model = worker_model_state["cotracker_model"]

    seq_name = os.path.basename(seq_path)
    sam3d_ref_frame_idx = getattr(args, "sam3d_ref_frame_idx", None)
    sam3d_ref_source = "cli" if sam3d_ref_frame_idx is not None else "default_first_frame"
    sam3d_ref_from_inpaint = bool(getattr(args, "sam3d_ref_from_inpaint", False) or seq_name.startswith("hold"))
    if sam3d_ref_frame_idx is None and sam3d_ref_from_inpaint:
        inpaint_original_frame_idx = infer_sam3d_ref_frame_from_inpaint(seq_path)
        if inpaint_original_frame_idx is not None:
            processed_image_start = infer_processed_image_start_frame(seq_path)
            if processed_image_start is not None:
                sam3d_ref_frame_idx = int(start_frame) + max(0, int(inpaint_original_frame_idx) - int(processed_image_start))
                print(
                    f"[SAM3D-REF] inpaint original frame {inpaint_original_frame_idx} mapped via "
                    f"processed/images start {processed_image_start} and clip start {start_frame} -> dataset frame {sam3d_ref_frame_idx}"
                )
            else:
                sam3d_ref_frame_idx = int(inpaint_original_frame_idx)
            sam3d_ref_source = "processed_inpaint"
        else:
            sam3d_ref_source = "inpaint_missing_default_first_frame"

    fit_and_visualize_pose(
        seq_path=seq_path,
        output_path=output_seq_path,
        device=device,
        pred_amodal_masks_np=pred_amodal_masks_np[start_frame:end_frame],
        bboxes=global_bboxes[start_frame:end_frame],
        cropped_rgbs_np=cropped_rgbs_np[start_frame:end_frame],
        cropped_modal_masks_np=cropped_modal_masks_np[start_frame:end_frame],
        cropped_hand_masks_np=cropped_hand_masks_np[start_frame:end_frame],
        cropped_depths_np=cropped_depths_np[start_frame:end_frame],
        cropped_metric_depths_np=cropped_metric_depths_np[start_frame:end_frame],
        mano_params=mano_params,
        hand_keypoints_np=hand_keypoints_np,
        hand_keypoints_valid_mask_np=hand_keypoints_valid_mask_np,
        num_total_frames=end_frame - start_frame,
        lr=args.lr,
        num_steps=args.num_steps,
        smoothness_weight=args.smoothness_weight,
        cotracker_model=cotracker_model,
        contact_indices_path=args.contact_indices_path,
        contact_cone_angle_deg=args.contact_cone_angle_deg,
        contact_dist_thresh=args.contact_dist_thresh,
        contact_surface_samples=args.contact_surface_samples,
        save_contact_debug_meshes_flag=args.save_contact_debug_meshes,
        overwrite_stage1_3=args.overwrite_stage1_3,
        overwrite_grasp_correction=args.overwrite_grasp_correction,
        overwrite_sam3d_dense_cache=getattr(args, "overwrite_sam3d_dense_cache", False),
        smooth_gfm_depth_offset=not getattr(args, "no_smooth_gfm_depth_offset", False),
        pnp_health_dump=getattr(args, "sam3d_reset", False),
        sam3d_sparse_keyframes=getattr(args, "sam3d_sparse_keyframes", False),
        sam3d_raw_rgbs_np=raw_rgbs_np[start_frame:end_frame],
        sam3d_raw_masks_np=raw_masks_np[start_frame:end_frame],
        sam3d_sparse_stride=getattr(args, "sam3d_sparse_stride", 8),
        sam3d_seed=getattr(args, "sam3d_seed", 42),
        sam3d_quiet=getattr(args, "sam3d_quiet", True),
        sam3d_config_path=getattr(args, "sam3d_config_path", None),
        sam3d_lambda_temp=getattr(args, "sam3d_lambda_temp", 0.5),
        sam3d_rot_outlier_filter=not getattr(args, "sam3d_rot_no_outlier_filter", False),
        sam3d_rot_outlier_max_angle_deg=getattr(args, "sam3d_rot_outlier_max_angle_deg", 60.0),
        sam3d_rot_outlier_max_iters=getattr(args, "sam3d_rot_outlier_max_iters", 3),
        sam3d_rot_retry_count=getattr(args, "sam3d_rot_retry_count", 3),
        sam3d_rot_mesh_overlay_alpha=getattr(args, "sam3d_rot_mesh_overlay_alpha", 0.85),
        sam3d_global_frame_offset=int(start_frame),
        sam3d_ref_frame_idx=sam3d_ref_frame_idx,
        sam3d_ref_from_inpaint=sam3d_ref_from_inpaint,
        sam3d_ref_source=sam3d_ref_source,
        sam3d_ref_keyframe_init=getattr(args, "sam3d_ref_keyframe_init", False),
        stage5_mode=getattr(args, "stage5_mode", STAGE5_OBJECT_LITE),
        stage5_presmooth_object=not getattr(args, "no_stage5_presmooth_object", False),
        sample_target_stride=getattr(args, "sample_target_stride", 3),
        min_sampled_frames=getattr(args, "min_sampled_frames", 64),
        max_sampled_frames=getattr(args, "max_sampled_frames", 128),
        ablation=getattr(args, "ablation", "full"),
        save_intermediates=getattr(args, "save_intermediates", False),
    )


def worker_main_gpu(gpu_id, seq_path_chunks, args):
    torch.cuda.set_device(gpu_id)
    print(f"Worker on GPU {gpu_id} started, processing {len(seq_path_chunks[gpu_id])} sequences.")

    # LAZY LOADING: State will hold models, loaded only when needed.
    worker_model_state = {
        "pipeline_mask": None,
        "depth_model": None,  # Now fully lazy
        "metric_depth_model": None,
        "generator": None,
        "cotracker_model": None,
    }

    for seq_path in tqdm(seq_path_chunks[gpu_id], desc=f"GPU {gpu_id}", position=gpu_id):
        try:
            process_sequence(seq_path, args.data_output_path, worker_model_state, args, fps=30)
        except Exception as e:
            print(f"Error processing {os.path.basename(seq_path)} on GPU {gpu_id}: {e}")
            traceback.print_exc()


def main(args):
    data_path = args.data_path
    subfolders = [f.path for f in os.scandir(data_path) if f.is_dir()]

    if args.video_id:
        print(f"--- Processing only specified video IDs: {args.video_id} ---")
        target_ids = set(args.video_id)
        subfolders = [p for p in subfolders if os.path.basename(p) in target_ids]
        print(f"Found {len(subfolders)} matching video sequences to process.")

    # # 筛选有效的序列：需要有 mano_params 文件夹（含json文件）且没有 optimized 开头的文件夹
    # valid_subfolders = []
    # for subfolder in subfolders:
    #     mano_params_path = os.path.join(subfolder, 'mano_params')
    #     # 检查是否存在 mano_params 文件夹
    #     if not os.path.isdir(mano_params_path):
    #         continue
    #     # 检查 mano_params 文件夹中的 json 文件数量
    #     json_files = [f for f in os.listdir(mano_params_path) if f.endswith('.json')]
    #     if len(json_files) == 0:
    #         continue
    #     # 检查是否存在以 optmized 开头的文件夹
    #     has_optimized = any(name.startswith('optimized_hoi_contact_seq') for name in os.listdir(subfolder) if os.path.isdir(os.path.join(subfolder, name)))
    #     if has_optimized:
    #         continue
    #     valid_subfolders.append(subfolder)

    # print(f"Filtered to {len(valid_subfolders)} valid sequences (with mano_params/*.json and no optmized* folders).")
    # subfolders = valid_subfolders

    # 多机分布式处理：根据 total_parts 和 part_idx 分割序列
    if args.total_parts > 1:
        if args.part_idx < 0 or args.part_idx >= args.total_parts:
            print(f"Error: part_idx ({args.part_idx}) must be between 0 and {args.total_parts - 1}")
            return

        # 排序以确保不同机器的分割结果一致
        subfolders = sorted(subfolders)
        total_seqs = len(subfolders)

        # 计算当前部分的范围
        seqs_per_part = (total_seqs + args.total_parts - 1) // args.total_parts
        start_idx = args.part_idx * seqs_per_part
        end_idx = min(start_idx + seqs_per_part, total_seqs)

        subfolders = subfolders[start_idx:end_idx]
        print(f"--- Multi-machine mode: Processing part {args.part_idx + 1}/{args.total_parts} ---")
        print(f"--- Assigned sequences: {start_idx} to {end_idx - 1} (total {len(subfolders)} sequences) ---")

    if not subfolders:
        print("Warning: No video sequences found to process with the given criteria.")
        return

    if args.debug:
        print("--- RUNNING IN DEBUG MODE (single process, sequential) ---")
        torch.cuda.set_device(0)

        worker_model_state = {
            "pipeline_mask": None,
            "depth_model": None,  # Now fully lazy
            "metric_depth_model": None,
            "generator": None,
            "cotracker_model": None,
        }

        for seq_path in tqdm(subfolders, desc="Debug Processing"):
            process_sequence(seq_path, args.data_output_path, worker_model_state, args, fps=30)

    else:
        # --- MULTI-GPU MODE ---
        num_gpus = torch.cuda.device_count()
        if num_gpus == 0:
            print("Error: No GPUs found for multi-GPU mode. Use --debug to run on CPU.")
            return

        if num_gpus > len(subfolders):
            print(f"Warning: More GPUs ({num_gpus}) than directories ({len(subfolders)}). Using {len(subfolders)} GPUs.")
            num_gpus = len(subfolders)

        seq_path_chunks = [[] for _ in range(num_gpus)]
        for i, seq_path in enumerate(subfolders):
            seq_path_chunks[i % num_gpus].append(seq_path)

        spawn_args = (seq_path_chunks, args)
        mp.spawn(worker_main_gpu, args=spawn_args, nprocs=num_gpus, join=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video amodal segmentation and content completion using Diffusion-VAS.")

    parser.add_argument("--model_path_mask", type=str, default="checkpoints/diffusion-vas-amodal-segmentation", help="Path to diffusion-vas amodal segmentation checkpoint.")
    parser.add_argument("--depth_encoder", type=str, default="vitl", help="Depth encoder type.")
    parser.add_argument("--model_path_depth", type=str, default="checkpoints/", help="Path to depth anything v2's checkpoint's parent folder.")
    parser.add_argument("--model_path_cotracker", type=str, default="checkpoints/scaled_offline.pth", help="Path to cotracker checkpoint.")
    
    parser.add_argument("--data_path", type=str, default="../../output", help="Path to the parent directory containing sequence subfolders.")
    parser.add_argument("--data_output_path", type=str, default="../../output", help="Output path.")

    parser.add_argument('--video_id', type=str, nargs='+', default=None, help="One or more video IDs to process.")
    parser.add_argument('--hand_side', choices=('left', 'right'), default=None, help="Required for dual MANO exports; selects the hand to fit.")
    parser.add_argument('--debug', action='store_true', help='Run in single-process debug mode without multiprocessing.')
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate for pose optimization.")
    parser.add_argument('--num_steps', type=int, default=400, help="Number of optimization steps.")
    parser.add_argument('--smoothness_weight', type=float, default=10, help="Weight for the trajectory smoothness loss.")
    parser.add_argument('--amodal_window_size', type=int, default=128, help="Temporal window size for diffusion-vas amodal mask inference.")
    parser.add_argument('--amodal_window_overlap', type=int, default=16, help="Overlapping frames between amodal inference windows.")
    parser.add_argument('--sample_target_stride', type=int, default=3, help="Adaptive sampling target stride in original-frame units.")
    parser.add_argument('--min_sampled_frames', type=int, default=64, help="Minimum sampled frames for videos longer than this value.")
    parser.add_argument('--max_sampled_frames', type=int, default=64, help="Maximum sampled frames for adaptive sampling.")
    parser.add_argument('--total_parts', type=int, default=1, help="Total number of parts to split sequences into for multi-machine processing.")
    parser.add_argument('--part_idx', type=int, default=0, help="Index of the part to process (0-indexed). Must be < total_parts.")
    parser.add_argument(
        '--ablation',
        choices=[
            'full',
            'no_fp',
            'no_vp',
            'no_stage4',
            'no_stage4_offset',
            'no_penetration',
            'no_stage5_pen',
            'no_contact',
            'no_dyn_contact',
            'no_stage5_contact',
            'no_stage5_smooth',
        ],
        default='full',
        help=(
            "Ablation mode for the dedicated ablation script. "
            "full keeps the copied demo behavior; other modes disable one "
            "Stage 3/4/5 component for the in-the-wild ablation table."
        ),
    )

    parser.add_argument('--overwrite_stage1_3', action='store_true', help="Ignore existing Stage 3 checkpoint and re-run Stages 1-3 from scratch.")
    parser.add_argument('--overwrite_grasp_correction', action='store_true', help="Delete existing camera_ray_depth_offset.json and re-run GraspFlowMatching pre-pass.")
    parser.add_argument('--overwrite_sam3d_dense_cache', action='store_true', help="Ignore cached SAM3D rotation_dense pose_snapshots.pt and re-run dense SAM3D tracking.")
    parser.add_argument(
        '--stage5_mode',
        choices=STAGE5_MODES,
        default=STAGE5_POSE_ONLY,
        help=(
            "Stage 5 optimization mode: object_lite locks hand root, optimizes "
            "mano_pose and lightly refines object pose with mask/anchor losses; "
            "pose_ray freezes object/root and optimizes mano_pose plus a small "
            "hand camera-ray depth delta; pose_only optimizes mano_pose only; "
            "full keeps object+wrist+pose optimization."
        ),
    )
    parser.add_argument(
        '--no_smooth_gfm_depth_offset',
        action='store_true',
        help="Disable Stage 4 temporal smoothing of GFM camera-ray depth offsets inside the interaction window.",
    )
    parser.add_argument(
        '--no_stage5_presmooth_object',
        action='store_true',
        help="Disable Stage 5 Gaussian pre-smoothing of object rot_6d/trans before optimization.",
    )

    parser.add_argument('--contact_indices_path', type=str, default='../../stage2_grasp_correction/DexGraspNet_table/grasp_generation/mano/contact_indices.json', help="Optional path to contact hand-vertex indices JSON.")
    parser.add_argument('--contact_cone_angle_deg', type=float, default=60.0, help="Normal-cone angle threshold (degrees) for contact estimation.")
    parser.add_argument('--contact_dist_thresh', type=float, default=0.02, help="Distance threshold (meters) for contact estimation.")
    parser.add_argument('--contact_surface_samples', type=int, default=10000, help="Number of sampled object-surface points for contact estimation.")
    parser.add_argument('--save_contact_debug_meshes', action='store_true', help="Save hand/object meshes before and after contact correction for interaction frames.")
    parser.add_argument('--save_intermediates', action='store_true', help="Save intermediate debug videos, PLYs, HTML visualizations, and other bulky diagnostics.")

    parser.add_argument(
        '--sam3d_reset',
        action='store_true',
        help="After Stage 3 vanilla PnP, write pnp_health.csv/json (dry-run metrics only). "
             "Legacy flag name; segmented SAM3D reset/bridge was removed from this demo "
             "(see backup/sam3d_reset_bridge_archive.txt).",
    )
    parser.add_argument(
        '--sam3d_sparse_keyframes',
        action='store_true',
        help="After Stage 1, build interaction_motion_profile_sam3d_gate.json; then: "
             "translation_likely → milestone SAM3D (sparse) under sam3d_sparse_keyframes/; "
             "rotation_likely → gap=1 SAM3D dense after Stage 2: final poses (outlier + interp) merge into "
             "stage3_object_pose_init for TemporalHandObjectPose; rotation_dense/pnp_vs_sam3d_vis/ is an extra PnP compare panel. "
             "Checkpoint resume uses the gate JSON / interaction_motion_profile_stage3.json.",
    )
    parser.add_argument(
        '--sam3d_sparse_stride',
        type=int,
        default=8,
        help="Sampled-frame stride between SAM3D runs inside the Stage1 mask-motion window. "
             "Same temporal chain as sam-3d-objects/demo_proj_temporal.py (sequential follow). "
             "Use ~8 for translation-heavy motion, ~1 for rotation-heavy (cf. interaction_motion_profile suggested_mode).",
    )
    parser.add_argument('--sam3d_config_path', type=str, default=None, help="SAM3D pipeline.yaml; default from SAM3D_CONFIG_PATH / env SAM3D_REPO.")
    parser.add_argument('--sam3d_lambda_temp', type=float, default=0.5, help="Temporal guidance for SAM3D follow mode.")
    parser.add_argument('--sam3d_seed', type=int, default=42, help="RNG seed forwarded to SAM3D inference.")
    parser.add_argument('--sam3d_quiet', dest='sam3d_quiet', action='store_true', default=True, help="Suppress SAM3D third-party logs during inference.")
    parser.add_argument('--no_sam3d_quiet', dest='sam3d_quiet', action='store_false', help="Show SAM3D internal logs.")
    parser.add_argument('--sam3d_ref_frame_idx', type=int, default=None, help="Dataset-frame index used as SAM3D/PnP reference keyframe. Overrides processed/inpaint inference.")
    parser.add_argument('--sam3d_ref_from_inpaint', action='store_true', help="Infer SAM3D/PnP reference frame from processed/inpaint/*.png. hold* sequences enable this automatically.")
    parser.add_argument(
        '--sam3d_ref_keyframe_init',
        action='store_true',
        help="Run one SAM3D keyframe at the selected reference frame to initialize object pose, without enabling dense SAM3D tracking.",
    )
    parser.add_argument(
        '--sam3d_rot_no_outlier_filter',
        action='store_true',
        help="rotation_likely dense SAM3D: disable local angular reject, global SLERP peel, and gap interpolation.",
    )
    parser.add_argument(
        '--sam3d_rot_outlier_max_angle_deg',
        type=float,
        default=60.0,
        help="rotation_likely: local jump threshold vs last trusted quat and global SLERP residual (cf. demo_proj_temporal).",
    )
    parser.add_argument(
        '--sam3d_rot_outlier_max_iters',
        type=int,
        default=3,
        help="rotation_likely: max greedy global outlier peel iterations.",
    )
    parser.add_argument(
        '--sam3d_rot_retry_count',
        type=int,
        default=3,
        help="rotation_likely: extra SAM3D follow reruns after a local angular reject; 0 preserves single-attempt behavior.",
    )
    parser.add_argument(
        '--sam3d_rot_mesh_overlay_alpha',
        type=float,
        default=0.85,
        help="rotation_likely dense mesh overlay blend strength in [0,1] (higher = more opaque).",
    )

    args = parser.parse_args()

    main(args)
