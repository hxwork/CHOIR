import argparse
import glob
import json
import math
import os
import subprocess
import sys
import traceback
import warnings

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
from pytorch3d.transforms import (Transform3d, axis_angle_to_matrix, matrix_to_axis_angle, matrix_to_rotation_6d, rotation_6d_to_matrix)
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from torchvision import transforms
from tqdm import tqdm

import torch_mesh_intersection.mesh_intersection.loss as collisions_loss
from body_model import MANO, run_amano, run_mano
from contact_constraint import (BUCKET_INSIDE, BUCKET_NEAR_CONTACT,
                                classify_and_build_correspondence,
                                compute_penetration_loss,
                                compute_soft_contact_loss,
                                temporal_lock_argmax)
from contact_map_estimator import estimate_contact_map_for_sampled_frames, load_contact_indices
from debug_bbox import get_global_amodal_bbox, load_hand_data, load_raw_frames
from models.diffusion_vas.pipeline_diffusion_vas import DiffusionVASPipeline
from pnp import generate_queries, run_pnp, run_pnp_1stage
from rendering import (get_render_params, render_side_view_rgb, vectorized_pose_guiding_loss, weighted_false_negative_loss, weighted_false_positive_loss)
from torch_mesh_intersection.mesh_intersection.bvh_search_tree import BVH
from utils import *


def calculate_iou(mask1, mask2):
    """Calculates Intersection over Union for two binary masks."""
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union > 0 else 0.0


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
    overwrite=False,
    lr=1e-2,
    num_steps=200,
    smoothness_weight=1.0,
    cotracker_model=None,
    contact_indices_path=None,
    contact_cone_angle_deg=60.0,
    contact_dist_thresh=0.02,
    contact_surface_samples=10000,
    recompute_contact_map=False,
    save_contact_debug_meshes_flag=False,
):
    # --- 0. Frame Sampling ---
    num_total_frames = len(bboxes)
    max_sampled_frames = 64

    if num_total_frames > max_sampled_frames:
        # Sample frames evenly using linspace, ensuring the first and last frames are included.
        sampled_indices = np.linspace(0, num_total_frames - 1, max_sampled_frames, dtype=int)
        sampled_indices = np.unique(sampled_indices).tolist()  # Ensure uniqueness
    else:
        # If total frames are less than or equal to the max, use all of them
        sampled_indices = list(range(num_total_frames))

    num_sampled_frames = len(sampled_indices)

    print(f"Total frames: {num_total_frames}, Sampled frames: {num_sampled_frames}")

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

    if overwrite and os.path.exists(checkpoint_path):
        print(f"\n--overwrite: removing existing Stage 3 checkpoint at {checkpoint_path}")
        os.remove(checkpoint_path)

    if os.path.exists(checkpoint_path):
        print("\n" + "=" * 80)
        print("Found Stage 3 checkpoint – loading and skipping Stages 1-3.")
        print("=" * 80)
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

        # Index / boundary variables
        sampled_indices = ckpt['sampled_indices']
        num_sampled_frames = int(ckpt['num_sampled_frames'])
        start_static_end_idx = int(ckpt['start_static_end_idx'])
        approaching_end_idx = int(ckpt['approaching_end_idx'])
        interaction_end_idx = int(ckpt['interaction_end_idx'])
        end_static_start_idx = int(ckpt['end_static_start_idx'])

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
        #                           (PnP init → temporal optim).
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

        # Boundary buffer: detection is conservative (still tends to trigger
        # 1–2 frames after motion onset / before motion offset). Pad each side
        # by `BOUNDARY_PAD` frames, clipped to the static segments. Resulting
        # extra frames at the edges are mild outliers; the ray-scale alpha
        # uses a self-filtered subset (top-70% by hand-obj gap) so it is
        # robust to a few non-grasping frames included in the window.
        BOUNDARY_PAD = 2
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

        # Save stage frame ranges (both sampled and original indices), left close, right open
        stage_ranges = {
            "0-1_start_static": {
                "sampled_start": int(0),
                "sampled_end": int(start_static_end_idx),
                "original_start": int(sampled_indices[0]),
                "original_end": int(sampled_indices[start_static_end_idx]) if start_static_end_idx < num_sampled_frames else int(sampled_indices[-1])
            },
            "1-2_approaching": {
                "sampled_start": int(start_static_end_idx),
                "sampled_end": int(approaching_end_idx),
                "original_start": int(sampled_indices[start_static_end_idx]),
                "original_end": int(sampled_indices[approaching_end_idx]) if approaching_end_idx < num_sampled_frames else int(sampled_indices[-1])
            },
            "2-3_interaction": {
                "sampled_start": int(approaching_end_idx),
                "sampled_end": int(interaction_end_idx),
                "original_start": int(sampled_indices[approaching_end_idx]),
                "original_end": int(sampled_indices[interaction_end_idx]) if interaction_end_idx < num_sampled_frames else int(sampled_indices[-1])
            },
            "3-4_releasing": {
                "sampled_start": int(interaction_end_idx),
                "sampled_end": int(end_static_start_idx),
                "original_start": int(sampled_indices[interaction_end_idx]),
                "original_end": int(sampled_indices[end_static_start_idx]) if end_static_start_idx < num_sampled_frames else int(sampled_indices[-1])
            },
            "4-5_end_static": {
                "sampled_start": int(end_static_start_idx),
                "sampled_end": int(num_sampled_frames),
                "original_start": int(sampled_indices[end_static_start_idx]),
                "original_end": int(sampled_indices[-1])
            }
        }

        stage_ranges_path = os.path.join(output_path, 'stage_frame_ranges.json')
        with open(stage_ranges_path, 'w') as f:
            json.dump(stage_ranges, f, indent=4)
        print(f"Saved stage frame ranges to {stage_ranges_path}")

        # ========================================================================
        # STAGE 2: Optimize 0-1 Shared ObjectPose
        # ========================================================================
        print("\n" + "=" * 80)
        print("STAGE 2: Optimizing Start Static Object Pose (0-1)")
        print("=" * 80)

        # Create batched camera for all frames in 0-1 segment
        start_segment_size = start_static_end_idx
        camera_start_segment = PerspectiveCameras(
            focal_length=focal_length[:start_segment_size],
            principal_point=principal_point[:start_segment_size],
            image_size=((H_out, W_out),),
            in_ndc=False,
            device=device,
        )

        # Prepare metric depth data for depth supervision
        sampled_metric_depths = torch.from_numpy(sampled_metric_depths_np[:start_segment_size, ..., 0]).float().to(device)  # (N, H, W)

        # ── Scale initialisation from 2D bounding box comparison ─────────────
        # Compare the 2D projected bounding box of the mesh at the initial pose
        # with the GT mask 2D bounding box. This avoids the 3D point cloud
        # diagonal approach, which is inflated by depth variation along Z and
        # gives unreliable scale estimates.
        try:
            _cam_f0 = PerspectiveCameras(
                focal_length=focal_length[0:1],
                principal_point=principal_point[0:1],
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

                    _fm = sampled_modal_masks_np[0].astype(bool)
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
        loop = tqdm(range(500), desc=f"Optimizing Start Static Segment [0:{start_static_end_idx}]")
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

            depth_loss_mask = (rendered_depth > 0) & (rendered_masks > 0.5) & (sampled_modal_masks[:start_segment_size].float() > 0.5)

            l2_loss_raw = torch.nn.functional.mse_loss(rendered_masks, sampled_modal_masks[:start_segment_size].float())

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
        first_frame_idx = 0
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

            # Save post-alignment point cloud with mesh
            # Target = Green, Aligned Source = Red, Aligned Mesh = Blue
            aligned_points = s_delta * (first_source_points @ R_delta[0]) + T_delta
            aliged_mesh_verts = s_delta * (first_verts[0] @ R_delta[0]) + T_delta
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
        for vis_idx in [0, start_static_end_idx - 1]:
            modal_frame = overlay_mask_on_image(sampled_rgbs_np[vis_idx], sampled_modal_masks_np[vis_idx])
            amodal_gt_frame = overlay_mask_on_image(sampled_rgbs_np[vis_idx], sampled_pred_amodal_masks_np[vis_idx])
            render_frame = overlay_mask_on_image(sampled_rgbs_np[vis_idx], (final_masks[vis_idx] > 0.5).astype(np.uint8))
            hand_frame = overlay_mask_on_image(sampled_rgbs_np[vis_idx], sampled_hand_masks_np[vis_idx], cmap_idx=1)

            # Overlay rendered RGB on original image for better geometry visualization
            rgb_render = rendered_rgb[vis_idx, ..., :3]  # (H, W, 3)
            alpha_render = rendered_rgb[vis_idx, ..., 3:4]  # (H, W, 1)
            rgb_overlay = sampled_rgbs_np[vis_idx] * (1 - alpha_render) + rgb_render * alpha_render

            panels = [modal_frame, amodal_gt_frame, render_frame, hand_frame, rgb_overlay]
            combined_image = (np.hstack(panels) * 255).astype(np.uint8)
            save_path = os.path.join(output_path, f"stage1_start_static_frame_{vis_idx}.png")
            imageio.imwrite(save_path, combined_image)
            print(f"Saved stage 1 visualization to {save_path}")

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

        # Use the last frame of 0-1 segment to generate queries for PnP
        pnp_start_frame_idx = start_static_end_idx - 1
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

        # Run PnP starting from the anchor frame
        pnp_poses = run_pnp_1stage(
            cotracker_model,
            sampled_rgbs[pnp_start_frame_idx:],  # Start from last frame of 0-1
            queries_2d,
            queries_3d,
            sampled_pred_amodal_masks_np[pnp_start_frame_idx:],
            K,
            optimized_R,  # Use optimized rotation matrix as initial
            optimized_T,  # Use optimized translation as initial
            device,
            output_dir=os.path.join(output_path, "pnp_visualization"),
            vis_threshold=0.0)

        # Combine: 0-1 segment uses the shared pose, rest uses PnP results
        pnp_poses = torch.from_numpy(np.stack(pnp_poses, axis=0)).float().to(device)  # (M, 4, 4) where M = num_sampled_frames - pnp_start_frame_idx

        # Create full pose sequence
        full_pnp_poses = torch.zeros(num_sampled_frames, 4, 4, device=device)

        # Fill 0-1 and 1-2 segments with the shared optimized pose
        # NOTE: PnP returns column-major rotation (OpenCV), but PyTorch3D uses row-major
        # So we need to store the optimized rotation in column-major format to match PnP
        shared_rot = rotation_6d_to_matrix(single_frame_model.rot_6d.detach().unsqueeze(0))  # (1, 3, 3) row-major
        shared_rot_col_major = shared_rot.transpose(1, 2)  # Convert to column-major to match PnP format
        shared_trans = single_frame_model.trans.detach()  # (3,)

        for i in range(approaching_end_idx):
            full_pnp_poses[i, :3, :3] = shared_rot_col_major[0]
            full_pnp_poses[i, :3, 3] = shared_trans
            full_pnp_poses[i, 3, 3] = 1.0

        # Fill the rest with PnP results (already in column-major format)
        full_pnp_poses[approaching_end_idx:] = pnp_poses[(approaching_end_idx - pnp_start_frame_idx):]

        pnp_rot_mat = full_pnp_poses[:, :3, :3]  # Column-major format
        pnp_t = full_pnp_poses[:, :3, 3]

        # --- 5. Optimization Loop: Stage 3 (Global) ---
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

        # Initialize poses for the whole sequence
        initial_R_mat = single_frame_model.rot_6d.detach().unsqueeze(0)
        initial_R_mat = rotation_6d_to_matrix(initial_R_mat).repeat(num_sampled_frames, 1, 1)
        initial_R_mat = pnp_rot_mat.mT

        # calculate the initial translation for the whole sequence
        initial_T = init_relative_translation(single_frame_model.trans.cpu().detach().numpy(), sampled_pred_amodal_masks_np, fx_new.item(), fy_new.item())
        initial_T = torch.from_numpy(initial_T).float().to(device)

        # recover the first approaching_end_idx frames with the optimized static translation
        initial_T[:approaching_end_idx] = single_frame_model.trans.detach().clone().unsqueeze(0).repeat(approaching_end_idx, 1)

        initial_scale = single_frame_model.scale.detach() * single_frame_model.initial_scale

        # Prepare mesh data for the batch
        verts_batch = verts.unsqueeze(0).repeat(num_sampled_frames, 1, 1)
        faces_batch = faces.unsqueeze(0).repeat(num_sampled_frames, 1, 1)

        # --- Model and Optimizer ---
        multi_frame_model = TemporalHandObjectPose(initial_R_mat, initial_T, initial_scale, verts_batch, faces_batch, sampled_mano_params).to(device)

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

        obj_optimizer = torch.optim.Adam([multi_frame_model.rot_6d, multi_frame_model.trans], lr=lr)
        hand_optimizer = torch.optim.Adam([multi_frame_model.mano_root_orient, multi_frame_model.mano_trans, multi_frame_model.mano_pose], lr=lr)

        # Stage 3 section in log
        _opt_log.write("\n# Stage 3 — Multi-frame pose sequence optimisation\n")
        _opt_log.write("step,total,obj_total,fp,vp,fp_raw,vp_raw,sm_rot,sm_trans,static_tie,"
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
        _w_rel = {'fp': 1.5, 'vp': 1.0, 'joints_2d': 1.0}
        _no_floor = {'joints_2d'}  # losses that decrease monotonically — skip floor

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
            loss_fp = _ema_normalise('fp', loss_fp_raw)
            loss_vp = _ema_normalise('vp', loss_vp_raw)

            loss_sm_rot = compute_smoothness_loss(multi_frame_model.rot_6d)
            loss_sm_trans = compute_smoothness_loss(multi_frame_model.trans)

            _w_st_s3 = 1e2
            loss_static_tie_s3 = torch.tensor(0.0, device=device)
            if approaching_end_idx > 1:
                rot_s = multi_frame_model.rot_6d[:approaching_end_idx]
                tr_s = multi_frame_model.trans[:approaching_end_idx]
                loss_static_tie_s3 = loss_static_tie_s3 + ((rot_s - rot_s.mean(0, keepdim=True).detach()).pow(2).mean() +
                                                           (tr_s - tr_s.mean(0, keepdim=True).detach()).pow(2).mean()) * _w_st_s3
            if end_static_start_idx < num_sampled_frames - 1:
                rot_e = multi_frame_model.rot_6d[end_static_start_idx:]
                tr_e = multi_frame_model.trans[end_static_start_idx:]
                loss_static_tie_s3 = loss_static_tie_s3 + ((rot_e - rot_e.mean(0, keepdim=True).detach()).pow(2).mean() +
                                                           (tr_e - tr_e.mean(0, keepdim=True).detach()).pow(2).mean()) * _w_st_s3

            obj_loss = (loss_fp + loss_vp + loss_sm_rot * smoothness_weight + loss_sm_trans * smoothness_weight + loss_static_tie_s3)

            # hand 2d joints loss (auto-balanced via EMA)
            projected_hand_joints = camera.transform_points_screen(mano_joints_batch, image_size=((H_out, W_out),))  # (N, 21, 3)
            pred_hand_joints_2d = projected_hand_joints[..., :2]
            num_valid_frames = sampled_gt_hand_joints_valid_mask.sum()
            if num_valid_frames > 0:
                loss_joints_2d_raw = torch.nn.functional.mse_loss(pred_hand_joints_2d[sampled_gt_hand_joints_valid_mask],
                                                                  sampled_gt_hand_joints_2d[sampled_gt_hand_joints_valid_mask])
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
                               f"{hand_loss.item():.6f},{loss_joints_2d.item():.6f},"
                               f"{loss_joints_2d_raw.item():.6f},"
                               f"{loss_sm_hand.item():.6f},{loss_anatomy.item():.6f},"
                               f"{loss_hand_z.item():.6f},{loss_hand_z_raw.item():.6f},"
                               f"{_n_hz_valid},"
                               f"{loss_mano_pose_anchor.item():.6f}\n")

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
                scene_i = join_meshes_as_scene([sparse_posed_meshes[i], sparse_posed_hand_meshes[i]])
                scenes.append(scene_i)
            sparse_scene = join_meshes_as_batch(scenes)
            final_rendered_masks_sparse = renderer(sparse_scene, cameras=camera)[..., 3].cpu().numpy()

            # Side-view with Phong shading (90° Y rotation, fresh HardPhong renderer)
            sparse_side_rgb = render_side_view_rgb(sparse_posed_meshes, sparse_posed_hand_meshes, camera, device, (H_out, W_out))

            projected_hand_joints = camera.transform_points_screen(sparse_mano_joints, image_size=((H_out, W_out),))  # (N, 21, 3)
            pred_joints_2d_np = projected_hand_joints[..., :2].cpu().numpy()
            gt_joints_2d_np = sampled_gt_hand_joints_2d.cpu().numpy()

        sparse_video_path = os.path.join(output_path, 'optimized_fitting_SPARSE.mp4')

        writer_sparse = imageio.get_writer(sparse_video_path,
                                           fps=30,
                                           codec='libx264',
                                           pixelformat='yuv420p',
                                           ffmpeg_params=['-crf', '28', '-preset', 'veryfast'],
                                           macro_block_size=None)

        for i in range(num_sampled_frames):
            # Determine current stage for annotation
            if i < start_static_end_idx:
                stage_name = "0-1: Start Static"
                stage_color = (0, 255, 0)  # Green
            elif i < approaching_end_idx:
                stage_name = "1-2: Approaching"
                stage_color = (255, 255, 0)  # Yellow
            elif i < interaction_end_idx:
                stage_name = "2-3: Interaction"
                stage_color = (255, 0, 0)  # Red
            elif i < end_static_start_idx:
                stage_name = "3-4: Releasing"
                stage_color = (255, 128, 0)  # Orange
            else:
                stage_name = "4-5: End Static"
                stage_color = (0, 128, 255)  # Blue

            modal_frame = overlay_mask_on_image(sampled_rgbs_np[i], sampled_modal_masks_np[i])
            amodal_gt_frame = overlay_mask_on_image(sampled_rgbs_np[i], sampled_pred_amodal_masks_np[i])
            binary_rendered_mask = (final_rendered_masks_sparse[i] > 0.5).astype(np.uint8)
            render_frame = overlay_mask_on_image(sampled_rgbs_np[i], binary_rendered_mask)

            # Draw joints on a new frame
            joints_frame = sampled_rgbs_np[i].copy()
            # Draw predicted joints (red)
            joints_frame = overlay_points_on_image(joints_frame, pred_joints_2d_np[i], color=(1.0, 0.0, 0.0))
            # Draw ground truth joints (green)
            joints_frame = overlay_points_on_image(joints_frame, gt_joints_2d_np[i], color=(0.0, 1.0, 0.0))

            hand_frame = overlay_mask_on_image(sampled_rgbs_np[i], sampled_hand_masks_np[i], cmap_idx=1)

            # Side-view: Phong-shaded RGB (object=blue, hand=red)
            side_panel = np.clip(sparse_side_rgb[i], 0.0, 1.0)

            panels = [modal_frame, amodal_gt_frame, render_frame, hand_frame, joints_frame, side_panel]

            combined_frame = np.hstack(panels)

            # Convert to uint8 for cv2.putText
            combined_frame_uint8 = (combined_frame * 255).astype(np.uint8)

            # Add stage annotation on the frame
            cv2.putText(combined_frame_uint8, f"Frame {i}: {stage_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, stage_color, 2, cv2.LINE_AA)

            writer_sparse.append_data(combined_frame_uint8)
        writer_sparse.close()
        print(f"Saved sparse fitting visualization to {sparse_video_path}")
        # ----------------------------------------------------------------
        # Save Stage 3 checkpoint (所有后续 Stage 7-8 可直接从此加载)
        # ----------------------------------------------------------------
        print(f"\nSaving Stage 3 checkpoint to: {checkpoint_path}")
        torch.save(
            {
                # Index / boundary
                'sampled_indices': sampled_indices,
                'num_sampled_frames': num_sampled_frames,
                'start_static_end_idx': start_static_end_idx,
                'approaching_end_idx': approaching_end_idx,
                'interaction_end_idx': interaction_end_idx,
                'end_static_start_idx': end_static_start_idx,
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

    # raise ValueError("Stop here")
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

    # Two-pass workflow:
    #   1st run — no camera_ray_depth_offset.json yet: stop after Stage 3, user
    #             computes the offset externally from the Stage 3 result.
    #   2nd run — camera_ray_depth_offset.json present: apply the correction,
    #             estimate the per-frame contact map, then run Stages 5 & 6.
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
                     "--video_id", _video_id]
        _cmd_sample = [os.path.join(_hd_bin, "torchrun"),
                       "--nnodes=1", "--nproc_per_node=1",
                       f"--master_port={_master_port}",
                       "sample_cam_ray_ddp.py", "ODE",
                       "--ckpt", "results/050-Linear-velocity-None/checkpoints/0040000.pt",
                       "--output_dir", "samples_ddp",
                       "--video_id", _video_id]

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
            raise FileNotFoundError(
                f"GraspFlowMatching commands completed but did not produce "
                f"camera_ray_depth_offset.json at {camera_ray_depth_offset_path}"
            )
        print(f"[GraspCorrection] Generated {camera_ray_depth_offset_path}, continuing.")

    # JSON now guaranteed to exist (was already there OR just generated above).
    if True:
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

            # Apply hand translation correction before contact-map estimation
            # (camera_ray_depth_offset.json is guaranteed to exist at this point).
            with open(camera_ray_depth_offset_path, 'r') as _f:
                depth_offset_raw = json.load(_f)

            if isinstance(depth_offset_raw, dict) and len(depth_offset_raw) > 0:
                mano_trans_corr_s4 = mano_trans_s4.clone()
                ray_norm = camera_rays_s4.norm(dim=-1)
                ray_dirs = torch.nn.functional.normalize(camera_rays_s4, dim=-1, eps=1e-8)
                invalid_ray_mask = ray_norm < 1e-8
                applied_cnt = 0
                total_cnt = mano_trans_corr_s4.shape[0]
                skipped_invalid_ray_cnt = 0

                for _sp, _frame_idx in enumerate(sampled_indices):
                    _k_frame = str(int(_frame_idx))
                    _k_pos = str(int(_sp))
                    if _k_frame in depth_offset_raw:
                        _depth_mag = float(depth_offset_raw[_k_frame])
                    elif _k_pos in depth_offset_raw:
                        _depth_mag = float(depth_offset_raw[_k_pos])
                    else:
                        continue

                    if invalid_ray_mask[_sp]:
                        skipped_invalid_ray_cnt += 1
                        continue

                    _delta = ray_dirs[_sp] * _depth_mag
                    mano_trans_corr_s4[_sp] = mano_trans_corr_s4[_sp] + _delta
                    applied_cnt += 1

                if applied_cnt > 0:
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
                    f"{applied_cnt}/{total_cnt} sampled frames"
                )
                if skipped_invalid_ray_cnt > 0:
                    print(f"Skipped {skipped_invalid_ray_cnt} frames due to near-zero mano_trans (undefined camera ray)")
            else:
                print(f"camera_ray_depth_offset.json is empty or invalid at {camera_ray_depth_offset_path}")

        # Estimate contact only in interaction segment [approaching_end_idx, interaction_end_idx).
        interaction_start = int(approaching_end_idx)
        interaction_end = int(interaction_end_idx)
        if save_contact_debug_meshes_flag:
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

    if not parsed_contact_map:
        print("\nNo contact map available – skipping Stages 5 & 6, proceeding to export with Stage 3 results.")

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
        from scipy.ndimage import gaussian_filter1d as _gf1d
        _pre_smooth_sigma = 2.0
        with torch.no_grad():
            for _param in [multi_frame_model.rot_6d, multi_frame_model.trans]:
                _arr = _param.detach().cpu().numpy()
                _arr = _gf1d(_arr, sigma=_pre_smooth_sigma, axis=0)
                _param.data.copy_(torch.from_numpy(_arr).float().to(device))
        print(f"Pre-smoothed object trajectory (rot_6d, trans) sigma={_pre_smooth_sigma}")

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

        # Interaction segment indices (contact / penetration losses operate here only).
        i0 = int(approaching_end_idx)
        i1 = int(interaction_end_idx)
        N_inter = max(0, i1 - i0)
        _has_inter = N_inter > 0
        if not _has_inter:
            print("Interaction segment is empty; Stage 5/6 will only run anchor losses.")

        # Hyper-params (kept inline as requested).
        K_RECLASSIFY = 50  # legacy default, used by Stage 6
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
        PREV_FACE_BONUS = 2.0  # additive logit bonus for prev-frame argmax face
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

        def _do_reclassify(prev_argmax_face_full):
            """Run a no_grad forward pass and rebuild contact cache for the
            interaction segment. Returns (cache, new_prev_argmax_face_full)."""
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
                _cache = classify_and_build_correspondence(
                    _hv_i, _hn_i, _ov_i, _on_i, faces,
                    contact_subset_idx=contact_subset_idx_t,
                    dist_thresh=DIST_THRESH,
                    cone_angle_deg=CONE_DEG,
                    n_surface_samples=N_SURFACE_SAMPLES,
                    topk=TOPK,
                    sigma=SIGMA,
                    prev_argmax_face=prev_argmax_face_full,
                    prev_face_logit_bonus=PREV_FACE_BONUS,
                )
                # Temporal lock: collapse weight_topk to one-hot on consensus
                # face within a 5-frame window.  Removes per-frame contact
                # target jitter that propagates into wrist jitter.
                _cache_locked = temporal_lock_argmax(
                    _cache, window=LOCK_WIN, min_consensus=LOCK_MIN_CONSENSUS)
                # Diagnostic: how many active verts got locked.
                if _cache.active_mask.any():
                    _orig_argmax_k = _cache.weight_topk.argmax(dim=-1)
                    _new_argmax_k  = _cache_locked.weight_topk.argmax(dim=-1)
                    _locked_n = ((_orig_argmax_k != _new_argmax_k) | (
                        _cache_locked.weight_topk.max(dim=-1).values > 0.99
                    )).logical_and(_cache.active_mask).sum().item()
                    _active_n = int(_cache.active_mask.sum().item())
                    if _active_n > 0:
                        print(f"  [reclassify] locked {_locked_n}/{_active_n} "
                              f"active verts ({100.0 * _locked_n / _active_n:.1f}%)")
                _cache = _cache_locked
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

        # ================================================================
        # STAGE 5 — Penetration Resolution (主穿模, 副 contact)
        # ================================================================
        # If Stage 3 + ray-scale alignment + camera_ray_depth_offset already
        # produce a satisfactory hand-object configuration, Stage 5 can do
        # more harm than good (pen loss pushes hand away from a thin object,
        # contact can't pull it back, equilibrium drifts).  Set this to False
        # to export the post-offset state as the final result.
        ENABLE_STAGE5 = True
        print("\n" + "=" * 80)
        print("STAGE 5: Penetration Resolution" + (" [SKIPPED]" if not ENABLE_STAGE5 else ""))
        print("=" * 80)
        if not ENABLE_STAGE5:
            print("Stage 5 disabled — using Stage 3 + offset result as final.")

        # Stage 5 optimization scope:
        # - Object 6DoF: free.  (Earlier we tried freezing it to stop sil
        #   from exploding, but that breaks Stage 5's ability to nudge the
        #   object during contact.  Instead, we let the object stay free
        #   and pin non-interaction frames at the very end of the pipeline
        #   to avoid the snap-back at release.)
        # - Hand wrist (mano_root_orient + mano_trans): UNFROZEN with low LR
        #   + strong anchor to Stage 3 + temporal smoothness. Lets wrist nudge
        #   to resolve residual depth offset that finger articulation alone
        #   cannot fix, without re-introducing the wrist-jitter we saw before.
        # - Hand fingers (mano_pose): free.
        multi_frame_model.mano_root_orient.requires_grad_(True)
        multi_frame_model.mano_trans.requires_grad_(True)
        optimizer_stage5 = torch.optim.Adam([
            {'params': [multi_frame_model.rot_6d, multi_frame_model.trans],
             'lr': 3e-4},
            {'params': [multi_frame_model.mano_pose],
             'lr': 5e-4},
            # Wrist gets 10x smaller LR than fingers — moves only when contact
            # / pen really need it; smoothness + anchor do the rest.
            {'params': [multi_frame_model.mano_root_orient, multi_frame_model.mano_trans],
             'lr': 5e-5},
        ])

        num_steps_s5 = 500 if ENABLE_STAGE5 else 0
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
        # Lowered 1e0 → 5e-1: 2D keypoints can be unreliable (collapse,
        # heavy occlusion).  We now have HaMeR mano_pose_anchor as a 3D
        # articulation prior (more robust than 2D keypoints), so we relax
        # hand_2d so it no longer dominates when the detector is noisy.
        LAMBDA_HAND2D_S5 = 5e-1
        # HaMeR mano_pose anchor — keeps fingers near HaMeR's 3D-regressed
        # articulation when joints_2d is unreliable.  axis-angle MSE is
        # typically ~1e-2 per element; with 1e2 weight the anchor term
        # contributes ~1.0, comparable to hand_2d_loss (~5 raw × 5e-1 ≈ 2-3).
        LAMBDA_MANO_POSE_ANCHOR_S5 = 1e2
        LAMBDA_OBJ_SIL_S5 = 5e2
        LAMBDA_OBJ_ANCHOR_NI_S5 = 1e2
        # Bumped 3e1 → 1e2: anchor to the post-offset object pose during
        # interaction so pen/contact tug-of-war can't drift the object far
        # from the validated Stage 3 + ray-scale + depth-offset state.
        LAMBDA_OBJ_ANCHOR_INTER_S5 = 1e2
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
        LAMBDA_OBJ_SMOOTH_S5 = 5e2
        # Wrist guards.
        # 2026-04 update: lowered 8e2 → 2e2.  With contact_loss=1e3 and
        # detach_object=True, all the contact gradient flows into the hand
        # — but a heavy wrist anchor pinned the wrist so finger articulation
        # alone couldn't reach contact points 5-10mm away.  Loosening
        # wrist_anchor lets the wrist nudge alongside contact;
        # 2nd-order accel (also reduced) still kills jitter.
        LAMBDA_WRIST_ANCHOR_S5 = 2e2
        LAMBDA_HAND_TR_SMOOTH_S5 = 2e2        # was 5e2
        LAMBDA_ROOT_R_SMOOTH_S5 = 5e2         # was 1e3, rotation matrix Frobenius (wrap-around safe)
        # 2nd-order (acceleration) — penalises jitter, allows constant-velocity
        # motion. Generally more effective than 1st-order for de-jittering.
        # 2026-04 update: pose_accel kept at 1e3 (fingers should stay
        # responsive to contact); hand_tr_accel & root_R_accel bumped back
        # toward original values because wrist jitter returned after the
        # too-aggressive cut to 1e3.  wrist_anchor stays at 2e2 so wrist
        # can still drift to follow contact, but accel kills high-freq jitter.
        LAMBDA_POSE_ACCEL_S5 = 1e3
        LAMBDA_HAND_TR_ACCEL_S5 = 5e3
        LAMBDA_ROOT_R_ACCEL_S5 = 3e3

        _log_cols_s5 = ["step", "total", "pen", "contact", "hand_2d", "anatomy",
                        "mano_pose_anchor",
                        "obj_sil", "obj_anchor",
                        "pose_sm", "obj_sm",
                        "wrist_anchor", "tr_sm", "root_R_sm",
                        "pose_acc", "tr_acc", "root_R_acc",
                        "n_inside", "mean_pen_mm", "max_pen_mm",
                        "n_active", "mean_ct_mm"]
        _log_path_s5 = os.path.join(output_path, "optimize_stage5.log")
        _log_file_s5 = open(_log_path_s5, "w", buffering=1)
        _log_file_s5.write("# Stage 5 — Penetration Resolution (one-sided push-out + sticky contact)\n")
        _log_file_s5.write(",".join(_log_cols_s5) + "\n")

        contact_cache = None
        prev_argmax_face_full = None

        loop_s5 = tqdm(range(num_steps_s5), desc="Stage 5: Penetration Resolution")
        for step in loop_s5:
            if _has_inter and (step % _k_reclassify_s5(step) == 0):
                contact_cache, prev_argmax_face_full = _do_reclassify(prev_argmax_face_full)

            optimizer_stage5.zero_grad()

            obj_meshes_s5, hand_meshes_s5, hand_joints_s5 = multi_frame_model()
            hand_verts_s5 = hand_meshes_s5.verts_padded()
            obj_verts_s5  = obj_meshes_s5.verts_padded()

            pen_info_s5 = {'n_inside': 0, 'mean_pen_depth': 0.0, 'max_pen_depth': 0.0}
            ct_info_s5  = {'n_active': 0, 'mean_contact_dist': 0.0}

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
            if _has_inter and contact_cache is not None:
                _hv_i = hand_verts_s5[i0:i1]
                _ov_i = obj_verts_s5[i0:i1]
                contact_loss_s5_raw, ct_info_s5 = compute_soft_contact_loss(
                    _hv_i, _ov_i, faces, contact_cache,
                    obj_grad_scale=(0.0, 0.0, 1.0))
                contact_loss_s5 = contact_loss_s5_raw * LAMBDA_CONTACT_S5
            else:
                contact_loss_s5 = torch.tensor(0.0, device=device)

            # Hand 2D
            projected_joints_s5 = all_cameras_s56.transform_points_screen(
                hand_joints_s5, image_size=((H_out, W_out),))[..., :2]
            if sampled_gt_hand_joints_valid_mask.any():
                hand_2d_loss_s5 = torch.nn.functional.mse_loss(
                    projected_joints_s5[sampled_gt_hand_joints_valid_mask],
                    sampled_gt_hand_joints_2d[sampled_gt_hand_joints_valid_mask]) * LAMBDA_HAND2D_S5
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
                _fp_s5 = weighted_false_positive_loss(
                    _rendered_obj_alpha_s5, sampled_pred_amodal_masks)
                _fn_s5 = weighted_false_negative_loss(
                    _rendered_obj_alpha_s5, sampled_pred_amodal_masks, weight=1.0)
                obj_sil_loss_s5 = (_fp_s5 + _fn_s5) * LAMBDA_OBJ_SIL_S5

            # Temporal smoothness — fingers + obj.  Finger smoothness in
            # 6D space (matches anchor; avoids axis-angle 2π wrap).
            pose_smooth_s5 = (_mp_6d_s5[1:] - _mp_6d_s5[:-1]
                              ).pow(2).mean() * LAMBDA_POSE_SMOOTH_S5
            obj_smooth_s5 = ((multi_frame_model.rot_6d[1:] - multi_frame_model.rot_6d[:-1]).pow(2).mean()
                            + (multi_frame_model.trans[1:] - multi_frame_model.trans[:-1]).pow(2).mean()
                            ) * LAMBDA_OBJ_SMOOTH_S5

            # Wrist guards — anchor + temporal smoothness.
            # CRITICAL: orientation diff must be in rotation-matrix space, not
            # axis-angle, otherwise projection (which canonicalises axis-angle
            # to |θ| ≤ π) can flip representation and explode the anchor loss.
            _R_root_s5 = axis_angle_to_matrix(multi_frame_model.mano_root_orient)  # (N, 3, 3)
            wrist_anchor_s5 = ((_R_root_s5 - _R_root_anchor_s5).pow(2).mean()
                              + (multi_frame_model.mano_trans - _anchor_mano_trans).pow(2).mean()
                              ) * LAMBDA_WRIST_ANCHOR_S5
            tr_smooth_s5 = (multi_frame_model.mano_trans[1:] - multi_frame_model.mano_trans[:-1]
                            ).pow(2).mean() * LAMBDA_HAND_TR_SMOOTH_S5
            root_R_smooth_s5 = (_R_root_s5[1:] - _R_root_s5[:-1]
                                ).pow(2).mean() * LAMBDA_ROOT_R_SMOOTH_S5

            # 2nd-order acceleration smoothness (needs N >= 3 frames).
            if multi_frame_model.mano_trans.shape[0] >= 3:
                _pose_acc = (_mp_6d_s5[2:] - 2 * _mp_6d_s5[1:-1] + _mp_6d_s5[:-2])
                pose_accel_s5 = _pose_acc.pow(2).mean() * LAMBDA_POSE_ACCEL_S5
                _tr_acc = (multi_frame_model.mano_trans[2:] - 2 * multi_frame_model.mano_trans[1:-1]
                           + multi_frame_model.mano_trans[:-2])
                tr_accel_s5 = _tr_acc.pow(2).mean() * LAMBDA_HAND_TR_ACCEL_S5
                _R_acc = _R_root_s5[2:] - 2 * _R_root_s5[1:-1] + _R_root_s5[:-2]
                root_R_accel_s5 = _R_acc.pow(2).mean() * LAMBDA_ROOT_R_ACCEL_S5
            else:
                pose_accel_s5 = torch.tensor(0.0, device=device)
                tr_accel_s5 = torch.tensor(0.0, device=device)
                root_R_accel_s5 = torch.tensor(0.0, device=device)

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
                obj_anchor_loss_s5 = obj_anchor_loss_s5 + (_di_rot.pow(2).mean() + _di_tr.pow(2).mean()) * LAMBDA_OBJ_ANCHOR_INTER_S5

            total_loss_s5 = (pen_loss_s5 + contact_loss_s5 + hand_2d_loss_s5
                             + loss_anatomy_s5 + mano_pose_anchor_s5
                             + obj_sil_loss_s5
                             + obj_anchor_loss_s5
                             + pose_smooth_s5 + obj_smooth_s5
                             + wrist_anchor_s5 + tr_smooth_s5 + root_R_smooth_s5
                             + pose_accel_s5 + tr_accel_s5 + root_R_accel_s5)
            total_loss_s5.backward()
            torch.nn.utils.clip_grad_norm_(multi_frame_model.parameters(), max_norm=1.0)
            optimizer_stage5.step()

            # Periodic Gaussian projection on wrist trajectory — kills high-
            # frequency jitter that the contact-target lock can't catch.
            if multi_frame_model.mano_trans.requires_grad and (step + 1) % K_PROJECT_S5 == 0:
                _smooth_wrist_inplace(multi_frame_model)

            loop_s5.set_postfix(loss=total_loss_s5.item(), pen=pen_loss_s5.item(),
                                ct=contact_loss_s5.item(), sil=obj_sil_loss_s5.item(),
                                pen_mm=round(pen_info_s5['mean_pen_depth'] * 1000, 2),
                                n_in=int(pen_info_s5['n_inside']))
            if step % 50 == 0 or step == num_steps_s5 - 1:
                _log_file_s5.write(",".join(str(round(v, 6)) for v in [
                    step, total_loss_s5.item(), pen_loss_s5.item(), contact_loss_s5.item(),
                    hand_2d_loss_s5.item(), loss_anatomy_s5.item(),
                    mano_pose_anchor_s5.item(),
                    obj_sil_loss_s5.item(),
                    obj_anchor_loss_s5.item(),
                    pose_smooth_s5.item(), obj_smooth_s5.item(),
                    wrist_anchor_s5.item(), tr_smooth_s5.item(), root_R_smooth_s5.item(),
                    pose_accel_s5.item(), tr_accel_s5.item(), root_R_accel_s5.item(),
                    pen_info_s5['n_inside'],
                    pen_info_s5['mean_pen_depth'] * 1000,
                    pen_info_s5['max_pen_depth'] * 1000,
                    ct_info_s5['n_active'],
                    ct_info_s5['mean_contact_dist'] * 1000,
                ]) + "\n")

        _log_file_s5.close()
        print(f"\nSTAGE 5 completed: penetration resolved (sticky contact during).")

        # End-of-Stage 5 PLY dump for inspection.
        if _has_inter:
            _dump_dir_s5 = os.path.join(output_path, "stage5_debug")
            _dump_stage_debug_meshes(
                _dump_dir_s5, multi_frame_model, faces,
                contact_cache, i0, i1, sampled_indices)

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

        _log_cols_s6 = ["step", "total", "pen", "contact", "hand_2d", "anatomy",
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
            if _has_inter and (step % K_RECLASSIFY_S6 == 0):
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
                        obj_grad_scale=(0.0, 0.0, 1.0))
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
                hand_2d_loss_s6 = torch.nn.functional.mse_loss(
                    projected_joints_s6[sampled_gt_hand_joints_valid_mask],
                    sampled_gt_hand_joints_2d[sampled_gt_hand_joints_valid_mask]) * LAMBDA_HAND2D_S6
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

        if ENABLE_STAGE6 and _has_inter and contact_cache is not None:
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
        if _has_inter and i0 < i1:
            _N_sparse = multi_frame_model.rot_6d.shape[0]
            with torch.no_grad():
                if i0 > 0:
                    multi_frame_model.mano_trans.data[:i0, 2] = multi_frame_model.mano_trans.data[i0, 2]
                    multi_frame_model.trans.data[:i0, 2]      = multi_frame_model.trans.data[i0, 2]
                    print(f"[Static-pose pin] hand z + obj z [0, {i0}) ← frame {i0} "
                          f"(hand_z={multi_frame_model.mano_trans.data[i0, 2].item():.4f} m, "
                          f"obj_z={multi_frame_model.trans.data[i0, 2].item():.4f} m); "
                          f"object xy & rotation kept free")
                if i1 < _N_sparse:
                    multi_frame_model.mano_trans.data[i1:, 2] = multi_frame_model.mano_trans.data[i1-1, 2]
                    multi_frame_model.trans.data[i1:, 2]      = multi_frame_model.trans.data[i1-1, 2]
                    print(f"[Static-pose pin] hand z + obj z [{i1}, {_N_sparse}) ← frame {i1-1} "
                          f"(hand_z={multi_frame_model.mano_trans.data[i1-1, 2].item():.4f} m, "
                          f"obj_z={multi_frame_model.trans.data[i1-1, 2].item():.4f} m); "
                          f"object xy & rotation kept free")

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

            current_obj_meshes, current_hand_meshes, _ = current_model()

            obj_color = [0.65, 0.8, 1.0]  # blue color
            hand_color = [1.0, 0.0, 0.0]  # red color

            N, V = current_obj_meshes.verts_padded().shape[:2]
            current_verts_rgb = torch.tensor(obj_color, device=device).view(1, 1, 3).expand(N, V, -1)
            current_obj_meshes.textures = TexturesVertex(verts_features=current_verts_rgb)

            N, V = current_hand_meshes.verts_padded().shape[:2]
            current_verts_rgb = torch.tensor(hand_color, device=device).view(1, 1, 3).expand(N, V, -1)
            current_hand_meshes.textures = TexturesVertex(verts_features=current_verts_rgb)

            for j in range(current_batch_size):
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
            if all_hand_faces is None:
                all_hand_faces = current_hand_meshes.faces_padded()[0].cpu().numpy().astype(np.uint32)

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

        writer.append_data((np.hstack(panels) * 255).astype(np.uint8))

    writer.close()
    print(f"Saved optimized_fitting.mp4 → {video_path}")

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

        # Also render side view for sparse frames
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
        writer_sf.append_data((np.hstack(panels_sf) * 255).astype(np.uint8))

    writer_sf.close()
    print(f"Saved sparse-final video → {sparse_final_video_path}")

    # Consolidate collected mesh data
    all_obj_verts = np.concatenate(all_obj_verts_list, axis=0)
    all_hand_verts = np.concatenate(all_hand_verts_list, axis=0)

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

    # Determine output directory based on whether contact map was used
    if parsed_contact_map:
        output_dir = os.path.join(seq_path, "optimized_hoi_contact_seq")
    else:
        output_dir = os.path.join(seq_path, "optimized_hoi_seq")

    save_hoi_sequence(output_dir, canonical_verts_np, canonical_faces_np, final_obj_scale, final_obj_rot_mat_col_major, final_obj_trans, mano_root_full_tensor,
                      mano_pose_full_tensor, mano_trans_full_tensor, is_right_full_tensor, all_obj_verts, all_obj_faces, all_hand_verts, all_hand_faces)

    # =========================================================================
    # In-the-Wild Metrics
    # =========================================================================
    print("\n" + "=" * 80)
    print("Computing in-the-wild metrics...")
    print("=" * 80)

    from in_the_wild_metric import compute_all_metrics as _compute_itw_metrics

    # all_obj_verts / all_hand_verts have flat_mat (diag[-1,-1,1]) applied for
    # visualization.  Applying flat_mat again undoes the flip (flat_mat^2 = I)
    # and recovers original world-space coordinates suitable for 3-D metrics.
    _flat_mat_np = np.diag(np.array([-1., -1., 1.]))
    _obj_verts_world = all_obj_verts @ _flat_mat_np  # (T, V_obj,  3)
    _hand_verts_world = all_hand_verts @ _flat_mat_np  # (T, V_hand, 3)

    # Same intrinsics for all frames
    _fl_full = np.tile(np.array([[float(fx_new), float(fy_new)]]), (num_total_frames, 1))  # (T, 2)
    _pp_full = np.tile(np.array([[float(cx_new), float(cy_new)]]), (num_total_frames, 1))  # (T, 2)

    # Map interaction segment from sampled-frame index space to full-frame index space.
    # approaching_end_idx / interaction_end_idx index into sampled_indices.
    _inter_start_full = (int(sampled_indices[approaching_end_idx]) if approaching_end_idx < num_sampled_frames else num_total_frames)
    _last_inter_sidx = max(0, interaction_end_idx - 1)
    _inter_end_full = (int(sampled_indices[min(_last_inter_sidx, num_sampled_frames - 1)]) +
                       1 if interaction_end_idx > approaching_end_idx else _inter_start_full)

    try:
        itw_metrics = _compute_itw_metrics(
            obj_verts_seq=_obj_verts_world,
            obj_faces=all_obj_faces.astype(np.int32),
            hand_verts_seq=_hand_verts_world,
            hand_faces=all_hand_faces.astype(np.int32),
            amodal_masks=pred_amodal_masks_np,
            focal_lengths=_fl_full,
            principal_points=_pp_full,
            H=H_out,
            W=W_out,
            interaction_start=_inter_start_full,
            interaction_end=_inter_end_full,
            fps=30.0,
            unit_to_cm=1.0,
            device=str(device),
            verbose=True,
        )
    except Exception as _e:
        import traceback as _tb
        print(f"[Warning] Metric computation failed: {_e}")
        _tb.print_exc()
        itw_metrics = {}

    metric_save_path = os.path.join(output_path, 'metric_in_the_wild.json')
    with open(metric_save_path, 'w') as _mf:
        json.dump({k: float(v) for k, v in itw_metrics.items()}, _mf, indent=4)
    print(f"In-the-wild metrics saved -> {metric_save_path}")


def save_hoi_sequence(output_dir, canonical_verts, canonical_faces, obj_scales, obj_rots_col_major, obj_trans, mano_root_orient, mano_pose, mano_trans,
                      is_right, obj_verts_seq, obj_faces, hand_verts_seq, hand_faces):
    """
    Saves the optimized and interpolated hand-object sequence to disk.
    - Canonical object mesh
    - Per-frame object pose JSON
    - Per-frame MANO parameters JSON
    - Per-frame posed object mesh OBJ
    - Per-frame posed hand mesh OBJ
    """
    print("--- Saving Optimized HOI Sequence ---")
    os.makedirs(output_dir, exist_ok=True)
    seq_name = os.path.basename(os.path.dirname(output_dir))
    hand_mesh_dir = os.path.join('rendering_for_paper', "ours_SelfCaptured", f"{seq_name}", "hand")
    obj_mesh_dir = os.path.join('rendering_for_paper', "ours_SelfCaptured", f"{seq_name}", "object")
    os.makedirs(hand_mesh_dir, exist_ok=True)
    os.makedirs(obj_mesh_dir, exist_ok=True)

    # 1. Save canonical object mesh once
    obj_canonical_filename = os.path.join(output_dir, "obj_canonical.obj")
    save_obj(obj_canonical_filename, canonical_verts, canonical_faces)
    print(f"  -> Saved canonical object mesh to {obj_canonical_filename}")

    num_frames = obj_trans.shape[0]

    for i in tqdm(range(num_frames), desc="Saving HOI sequence"):
        # 2. Save per-frame object pose
        obj_pose_data = {'scale': obj_scales[i].tolist(), 'rotation': obj_rots_col_major[i].tolist(), 'translation': obj_trans[i].tolist()}
        obj_filename = os.path.join(output_dir, f"obj_{i:05d}.json")
        with open(obj_filename, 'w') as f:
            json.dump(obj_pose_data, f, indent=4)

        # 3. Save per-frame MANO parameters
        mano_data = {
            'root_orient': mano_root_orient[i].cpu().numpy().tolist(),
            'pose': mano_pose[i].cpu().numpy().tolist(),
            'trans': mano_trans[i].cpu().numpy().tolist(),
            'is_right': is_right[i].item()
        }
        mano_filename = os.path.join(output_dir, f"mano_{i:05d}.json")
        with open(mano_filename, 'w') as f:
            json.dump(mano_data, f, indent=4)

        # 4. Save per-frame posed meshes
        obj_mesh_path = os.path.join(obj_mesh_dir, f"{i:04d}_mesh.ply")
        save_ply(obj_mesh_path, torch.from_numpy(obj_verts_seq[i]).float(), torch.from_numpy(obj_faces).long())

        hand_mesh_path = os.path.join(hand_mesh_dir, f"{i:04d}_hand.ply")
        save_ply(hand_mesh_path, torch.from_numpy(hand_verts_seq[i]).float(), torch.from_numpy(hand_faces).long())

    print(f"  -> Saved {num_frames} frames of pose, MANO data, and meshes to {output_dir}")


def save_k3d_visualization(output_path, seq_path, obj_verts_seq, obj_faces, hand_verts_seq, hand_faces, target_fps=30.0):
    """Saves an interactive 3D visualization of hand-object interaction sequence to an HTML file."""
    print("--- Generating K3D Visualization ---")

    # ================= 相机参数配置 (Z-Forward, -Y Up) =================
    focal_length_mm = 25.0
    sensor_height_mm = 24.0
    fov_degrees = 2 * math.atan(sensor_height_mm / (2 * focal_length_mm)) * (180 / math.pi)
    camera_config = [0, 0, 0, 0, 0, 1, 0, -1, 0]  # [Pos, Target, Up]

    # ================= 辅助函数 =================
    def process_sequence_from_memory(all_verts_np, fps):
        """Converts a numpy vertex sequence to a k3d-compatible animation dictionary."""
        vertices_seq_dict = {}
        num_frames = all_verts_np.shape[0]
        for frame_idx in range(num_frames):
            time_stamp = frame_idx / fps
            vertices_seq_dict[str(time_stamp)] = all_verts_np[frame_idx].astype(np.float32)
        return vertices_seq_dict

    # ================= 主逻辑 =================
    # Assuming point cloud might exist at this conventional path
    bg_pc_path = os.path.join(seq_path, "env_depths", "sampled_pc_0.ply")
    output_filename = os.path.join(output_path, "visualization_3d.html")

    plot = k3d.plot(name=f"Vis: {os.path.basename(output_path)}", camera_auto_fit=False, camera_fov=fov_degrees, grid_visible=False, fps=target_fps)

    # # --- A. 加载背景点云 ---
    # if os.path.exists(bg_pc_path):
    #     bg_mesh = trimesh.load(bg_pc_path, process=False)
    #     if hasattr(bg_mesh, 'vertices'):
    #         bg_points = k3d.points(positions=bg_mesh.vertices.astype(np.float32), point_size=0.005, shader='3d', color=0xAAAAAA, name="Environment PC")
    #         plot += bg_points
    # else:
    #     print(f"  - info: background point cloud not found at {bg_pc_path}")

    # --- B. 加载手部序列 ---
    if hand_verts_seq is not None and hand_faces is not None:
        hand_k3d_verts = process_sequence_from_memory(hand_verts_seq, target_fps)
        if hand_k3d_verts:
            start_t = list(hand_k3d_verts.keys())[0]
            hand_mesh = k3d.mesh(hand_k3d_verts[start_t], hand_faces, color=0xffcccc, side='double', name="Hand")
            hand_mesh.vertices = hand_k3d_verts
            plot += hand_mesh
            print(f"  - Hand mesh loaded: {len(hand_verts_seq)} frames")

    # --- C. 加载物体序列 ---
    if obj_verts_seq is not None and obj_faces is not None:
        obj_k3d_verts = process_sequence_from_memory(obj_verts_seq, target_fps)
        if obj_k3d_verts:
            start_t = list(obj_k3d_verts.keys())[0]
            obj_mesh = k3d.mesh(obj_k3d_verts[start_t], obj_faces, color=0x00aaff, side='double', name="Object")
            obj_mesh.vertices = obj_k3d_verts
            plot += obj_mesh
            print(f"  - Object mesh loaded: {len(obj_verts_seq)} frames")

    # --- D. 设置相机与导出 ---
    plot.camera = camera_config
    plot.start_auto_play()

    snapshot_content = plot.get_snapshot()

    # Save to the original path
    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write(snapshot_content)
    print(f"  -> 3D visualization saved: {output_filename}")

    # Save to the additional requested path
    video_id = os.path.basename(seq_path)
    extra_save_dir = "../../stage1_preprocess/sam-3d-objects/vis_htmls"
    os.makedirs(extra_save_dir, exist_ok=True)
    extra_output_filename = os.path.join(extra_save_dir, f"{video_id}.html")
    with open(extra_output_filename, 'w', encoding='utf-8') as f:
        f.write(snapshot_content)
    print(f"  -> Also saved visualization to: {extra_output_filename}")


def process_sequence(seq_path, data_output_path, worker_model_state, args, fps=30):
    # --- 1. Initialization and Raw Data Loading ---
    seq_name = os.path.basename(seq_path)
    output_seq_path = os.path.join(data_output_path, seq_name)
    os.makedirs(output_seq_path, exist_ok=True)

    bbox_save_path = os.path.join(output_seq_path, 'global_bbox.json')
    amodal_masks_save_dir = os.path.join(output_seq_path, 'amodal_masks')
    cropped_depths_save_dir = os.path.join(output_seq_path, 'cropped_depths')
    cropped_metric_depths_save_dir = os.path.join(output_seq_path, 'cropped_metric_depths')

    # Load raw frames once at the beginning
    raw_rgbs_np = load_raw_frames(seq_path + "/rgbs", frame_type='rgb')
    raw_masks_np = load_raw_frames(seq_path + "/obj_masks", frame_type='mask')

    # Load hand masks if they exist
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
    hand_bboxes = load_hand_data(seq_path, num_frames)
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
        device = f"cuda:{torch.cuda.current_device()}"
        depth_pixels_tensor = torch.stack(processed_frames).unsqueeze(0).to(device)
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

        print("Amodal segmentation by diffusion-vas...")
        pred_amodal_masks_raw = pipeline_mask(
            modal_pixels_tensor,
            depth_pixels_tensor,
            height=pred_res[0],
            width=pred_res[1],
            num_frames=num_frames,
            decode_chunk_size=8,
            motion_bucket_id=127,
            fps=8,
            noise_aug_strength=0.02,
            min_guidance_scale=1.5,
            max_guidance_scale=1.5,
            generator=generator,
        ).frames[0]

        pred_amodal_masks_processed = (np.array([np.array(img) for img in pred_amodal_masks_raw]).astype('uint8').sum(axis=-1) > 600).astype('uint8')
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
    mano_params_dir = os.path.join(seq_path, 'mano_params')
    mano_params_files = sorted(os.listdir(mano_params_dir), key=lambda x: int(os.path.splitext(x)[0]))
    start_frame, end_frame = int(os.path.splitext(mano_params_files[0])[0]), int(os.path.splitext(mano_params_files[-1])[0]) + 1
    mano_params = np.array([json.load(open(os.path.join(mano_params_dir, f))) for f in mano_params_files])

    # Load 2D hand keypoints
    hand_keypoints_file = glob.glob(os.path.join(seq_path, '*h_keypoints.json'))
    hand_keypoints_data = json.load(open(hand_keypoints_file[0]))

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
        combined_frames.append(combined_frame)
    imageio.mimwrite(
        comparison_video_path,
        (np.stack(combined_frames) * 255).astype(np.uint8),
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
        recompute_contact_map=args.recompute_contact_map,
        save_contact_debug_meshes_flag=args.save_contact_debug_meshes,
        overwrite=args.overwrite,
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

    parser.add_argument(
        "--model_path_mask",
        type=str,
        default="checkpoints/diffusion-vas-amodal-segmentation",
        help="Path to diffusion-vas amodal segmentation checkpoint.",
    )

    parser.add_argument(
        "--depth_encoder",
        type=str,
        default="vitl",  # or 'vits', vitl, 'vitg'
        help="Depth encoder type.")

    parser.add_argument("--model_path_depth", type=str, default="checkpoints/", help="Path to depth anything v2's checkpoint's parent folder.")
    parser.add_argument("--model_path_cotracker", type=str, default="checkpoints/scaled_offline.pth", help="Path to cotracker checkpoint.")

    parser.add_argument("--data_path",
                        type=str,
                        default="../../output",
                        help="Path to the parent directory containing sequence subfolders.")

    parser.add_argument("--data_output_path", type=str, default="../../output", help="Output path.")

    parser.add_argument('--video_id', type=str, nargs='+', default=None, help="One or more video IDs to process.")
    parser.add_argument('--debug', action='store_true', help='Run in single-process debug mode without multiprocessing.')
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate for pose optimization.")
    parser.add_argument('--num_steps', type=int, default=400, help="Number of optimization steps.")
    parser.add_argument('--smoothness_weight', type=float, default=10, help="Weight for the trajectory smoothness loss.")
    parser.add_argument('--total_parts', type=int, default=1, help="Total number of parts to split sequences into for multi-machine processing.")
    parser.add_argument('--part_idx', type=int, default=0, help="Index of the part to process (0-indexed). Must be < total_parts.")
    parser.add_argument('--overwrite', action='store_true', help="Ignore existing Stage 3 checkpoint and re-run Stages 1-3 from scratch.")
    parser.add_argument('--contact_indices_path', type=str,
                        default='../../stage2_grasp_correction/DexGraspNet_table/grasp_generation/mano/contact_indices.json',
                        help="Optional path to contact hand-vertex indices JSON.")
    parser.add_argument('--contact_cone_angle_deg', type=float, default=60.0,
                        help="Normal-cone angle threshold (degrees) for contact estimation.")
    parser.add_argument('--contact_dist_thresh', type=float, default=0.02,
                        help="Distance threshold (meters) for contact estimation.")
    parser.add_argument('--contact_surface_samples', type=int, default=10000,
                        help="Number of sampled object-surface points for contact estimation.")
    parser.add_argument('--recompute_contact_map', action='store_true',
                        help="Force recomputing contact_map_per_frame.json even if file exists.")
    parser.add_argument('--save_contact_debug_meshes', action='store_true',
                        help="Save hand/object meshes before and after contact correction for interaction frames.")

    args = parser.parse_args()

    main(args)
