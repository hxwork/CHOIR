import argparse
import glob
import json
import math
import os
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
from debug_bbox import get_global_amodal_bbox, load_hand_data, load_raw_frames
from models.diffusion_vas.pipeline_diffusion_vas import DiffusionVASPipeline
from pnp import generate_queries, run_pnp, run_pnp_1stage
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


def lock_first_frame_hook(grad):
    # grad 的 shape 是 (N, 6) 或 (N, 3)
    new_grad = grad.clone()
    new_grad[0] = 0.0  # 强制把第一帧梯度置零
    return new_grad


def get_render_params(current_step, total_steps):
    # 起始值 (Coarse Stage)
    START_SIGMA = 1e-4  # 像素空间下，10个像素的模糊半径
    START_GAMMA = 1e-2  # 较高的透明度，平滑梯度

    # 结束值 (Fine Stage)
    END_SIGMA = 1e-4  # 最后的锐利程度
    END_GAMMA = 1e-4  # 接近硬遮挡

    # 计算进度 (0.0 -> 1.0)
    t = current_step / total_steps

    # 指数衰减公式: start * (end / start) ^ t
    # 这种衰减方式在前期下降快，后期平缓，适合 Pose 优化
    curr_sigma = START_SIGMA * (END_SIGMA / START_SIGMA)**t
    curr_gamma = START_GAMMA * (END_GAMMA / START_GAMMA)**t

    # 动态调整 faces_per_pixel
    # 如果模糊很大，必须看更多的面，否则会产生"锯齿状"的梯度
    if curr_sigma > 2.0:
        curr_faces_per_pixel = 80
    elif curr_sigma > 0.5:
        curr_faces_per_pixel = 50
    else:
        curr_faces_per_pixel = 20

    return curr_sigma, curr_gamma, curr_faces_per_pixel


def create_lock_frames_hook(indices_to_lock):

    def hook(grad):
        new_grad = grad.clone()
        new_grad[indices_to_lock] = 0.0
        return new_grad

    return hook


def compute_contact_points_from_correspondences(correspondences, obj_mesh, device='cuda'):
    """
    从face_id和barycentric coordinates重建object表面上的对应点和法线
    
    Args:
        correspondences: dict, {hand_vertex_idx: {"face_id": int, "bary_coords": [w0, w1, w2]}}
        obj_mesh: Meshes object (single mesh)
        device: torch device
    
    Returns:
        hand_indices: tensor of hand vertex indices, shape (N_contacts,)
        obj_surface_points: tensor of corresponding object surface points, shape (N_contacts, 3)
        obj_surface_normals: tensor of corresponding object surface normals, shape (N_contacts, 3)
    """
    if not correspondences:
        return None, None, None

    hand_indices = []
    obj_surface_points = []
    obj_surface_normals = []

    # Get mesh data
    verts = obj_mesh.verts_list()[0]  # (V, 3)
    faces = obj_mesh.faces_list()[0]  # (F, 3)

    # Compute vertex normals (average of adjacent face normals)
    # PyTorch3D provides a method to compute vertex normals
    verts_normals = obj_mesh.verts_normals_packed()  # (V, 3)

    for hand_idx, corr_data in correspondences.items():
        face_id = corr_data['face_id']
        bary_coords = torch.tensor(corr_data['bary_coords'], dtype=torch.float32, device=device)

        # Get the three vertices of the face
        face_verts = verts[faces[face_id]]  # (3, 3)
        face_vert_indices = faces[face_id]  # (3,)

        # Compute the point on the surface using barycentric coordinates
        # point = w0 * v0 + w1 * v1 + w2 * v2
        surface_point = torch.sum(bary_coords.unsqueeze(-1) * face_verts, dim=0)  # (3,)

        # Interpolate vertex normals using barycentric coordinates
        # normal = w0 * n0 + w1 * n1 + w2 * n2
        face_vert_normals = verts_normals[face_vert_indices]  # (3, 3)
        surface_normal = torch.sum(bary_coords.unsqueeze(-1) * face_vert_normals, dim=0)  # (3,)
        # Normalize the interpolated normal
        surface_normal = surface_normal / (torch.norm(surface_normal, dim=-1, keepdim=True) + 1e-8)

        hand_indices.append(hand_idx)
        obj_surface_points.append(surface_point)
        obj_surface_normals.append(surface_normal)

    hand_indices = torch.tensor(hand_indices, dtype=torch.long, device=device)
    obj_surface_points = torch.stack(obj_surface_points, dim=0)  # (N_contacts, 3)
    obj_surface_normals = torch.stack(obj_surface_normals, dim=0)  # (N_contacts, 3)

    return hand_indices, obj_surface_points, obj_surface_normals


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


# def compute_semantic_layering_loss(hand_mesh, obj_mesh, hand_mask_gt, obj_mask_gt, camera, raster_settings, device):
#     """
#     计算语义层次loss：基于语义着色渲染约束手和物体的前后关系

#     Args:
#         hand_mesh: Meshes object for hand
#         obj_mesh: Meshes object for object
#         hand_mask_gt: Ground truth hand mask, (H, W)
#         obj_mask_gt: Ground truth object mask, (H, W)
#         camera: PerspectiveCameras
#         raster_settings: RasterizationSettings
#         device: torch device

#     Returns:
#         semantic_loss: scalar tensor
#     """
#     # Step 1: 给手涂红色 (1, 0, 0)
#     hand_verts = hand_mesh.verts_packed()
#     hand_semantic_color = torch.tensor([1.0, 0.0, 0.0], device=device)
#     hand_colors = hand_semantic_color.unsqueeze(0).repeat(hand_verts.shape[0], 1)
#     hand_textures = TexturesVertex(verts_features=hand_colors.unsqueeze(0))

#     hand_mesh_colored = Meshes(verts=[hand_verts], faces=hand_mesh.faces_list(), textures=hand_textures)

#     # Step 2: 给物体涂蓝色 (0, 0, 1)
#     obj_verts = obj_mesh.verts_packed()
#     obj_semantic_color = torch.tensor([0.0, 0.0, 1.0], device=device)
#     obj_colors = obj_semantic_color.unsqueeze(0).repeat(obj_verts.shape[0], 1)
#     obj_textures = TexturesVertex(verts_features=obj_colors.unsqueeze(0))

#     obj_mesh_colored = Meshes(verts=[obj_verts], faces=obj_mesh.faces_list(), textures=obj_textures)

#     # Step 3: 合并场景
#     combined_mesh = join_meshes_as_scene([hand_mesh_colored, obj_mesh_colored])

#     # Step 4: 设置语义渲染器（只渲染纹理颜色，不受光照影响）
#     lights = PointLights(
#         device=device,
#         location=[[0.0, 0.0, -3.0]],
#         ambient_color=((1.0, 1.0, 1.0),),  # 纯环境光
#         diffuse_color=((0.0, 0.0, 0.0),),  # 关闭漫反射
#         specular_color=((0.0, 0.0, 0.0),)  # 关闭镜面反射
#     )

#     semantic_renderer = MeshRenderer(rasterizer=MeshRasterizer(cameras=camera, raster_settings=raster_settings),
#                                      shader=HardPhongShader(device=device, cameras=camera, lights=lights))

#     # Step 5: 渲染语义图
#     rendered_semantic = semantic_renderer(combined_mesh, cameras=camera)
#     rendered_semantic_rgb = rendered_semantic[0, ..., :3]  # (H, W, 3)

#     # Step 6: 构建GT语义图
#     H, W = hand_mask_gt.shape
#     gt_semantic = torch.zeros((H, W, 3), device=device)

#     # 手可见区域 → 红色
#     hand_visible_mask = hand_mask_gt > 0.5
#     gt_semantic[hand_visible_mask] = torch.tensor([1.0, 0.0, 0.0], device=device)

#     # 物体可见且手不可见区域 → 蓝色
#     obj_visible_mask = (obj_mask_gt > 0.5) & (~hand_visible_mask)
#     gt_semantic[obj_visible_mask] = torch.tensor([0.0, 0.0, 1.0], device=device)

#     # Step 7: 计算loss（仅在有效区域）
#     valid_region = (hand_mask_gt > 0.5) | (obj_mask_gt > 0.5)

#     if valid_region.sum() > 0:
#         semantic_loss = torch.nn.functional.mse_loss(rendered_semantic_rgb[valid_region], gt_semantic[valid_region])
#     else:
#         semantic_loss = torch.tensor(0.0, device=device)

#     return semantic_loss

# def compute_collision_loss(obj_verts, hand_verts, hand_normals):
#     """
#     计算hand和object之间的穿模损失
#     使用法线和最近邻判断穿透，只计算内部点的距离

#     Args:
#         obj_verts: [B, N_obj, 3] 或 [N_obj, 3] object顶点
#         hand_verts: [B, N_hand, 3] 或 [N_hand, 3] hand顶点
#         hand_normals: [B, N_hand, 3] 或 [N_hand, 3] hand顶点法线

#     Returns:
#         collision_loss: 穿模惩罚（只计算穿透点的距离之和）
#     """
#     # 确保输入是batch形式
#     if obj_verts.ndim == 2:
#         obj_verts = obj_verts.unsqueeze(0)
#     if hand_verts.ndim == 2:
#         hand_verts = hand_verts.unsqueeze(0)
#     if hand_normals.ndim == 2:
#         hand_normals = hand_normals.unsqueeze(0)

#     # 计算hand顶点到object顶点的最近邻距离和索引
#     hand_nn_dist, hand_nn_idx = get_NN(obj_verts, hand_verts)  # [B, N_hand]

#     # 判断哪些hand顶点在object内部
#     hand_interior = get_interior(hand_normals, hand_verts, obj_verts, hand_nn_idx).type(torch.bool)  # [B, N_hand]

#     # 只计算内部点的距离之和
#     hand_in_obj_penetr_dist = hand_nn_dist[hand_interior].sum()

#     return hand_in_obj_penetr_dist


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
    # # 3. --- 关键修改: 应用白名单 ---
    # if ignore_indices is not None:
    #     # 创建一个全True的mask
    #     calc_mask = torch.ones(N_hand, dtype=torch.bool, device=hand_verts.device)
    #     # 将白名单里的点设为False (不计算)
    #     calc_mask[ignore_indices] = False
    #     # 扩展到Batch维度
    #     calc_mask = calc_mask.unsqueeze(0).expand(B, -1)

    #     # 只有既在内部，又不在白名单里的点，才算穿模
    #     valid_penetration = hand_interior & calc_mask
    # else:
    #     valid_penetration = hand_interior

    # # 4. 只计算有效穿透点的距离之和
    # # 为了防止没有穿透时 sum() 导致梯度消失或报错，加一个判定
    # if valid_penetration.sum() > 0:
    #     hand_in_obj_penetr_dist = hand_nn_dist[valid_penetration].sum()
    # else:
    #     hand_in_obj_penetr_dist = torch.tensor(0.0, device=hand_verts.device, requires_grad=True)
    hand_in_obj_penetr_dist = hand_nn_dist[hand_interior].sum()

    return hand_in_obj_penetr_dist


def detect_interaction_segments(obj_meshes, hand_meshes, start_idx, end_idx, distance_percentile=10):
    """
    检测交互段（2-3段），基于hand-object之间的3D距离
    正序找第一个波谷（接触点），逆序找第一个波谷（释放点）
    
    Args:
        obj_meshes: Meshes object, (N, V_obj, 3) 
        hand_meshes: Meshes object, (N, V_hand, 3)
        start_idx: 开始检测的帧索引（通常是0-1段结束）
        end_idx: 结束检测的帧索引（通常是4-5段开始）
        distance_percentile: 使用第X百分位数的距离作为代表（避免outliers）
    
    Returns:
        approaching_end: 1-2段结束帧索引（2-3段开始，接触点）
        interaction_end: 2-3段结束帧索引（3-4段开始，释放点）
    """
    # 计算每一帧的hand-object距离
    distances = []
    for i in range(start_idx, end_idx):
        obj_verts = obj_meshes.verts_list()[i]  # (V_obj, 3)
        hand_verts = hand_meshes.verts_list()[i]  # (V_hand, 3)

        # 计算hand每个顶点到object最近点的距离
        # 使用knn_points来高效计算
        knn_result = knn_points(hand_verts.unsqueeze(0), obj_verts.unsqueeze(0), K=1)
        min_distances = torch.sqrt(knn_result.dists[0, :, 0])  # (V_hand,)

        # 使用百分位数作为该帧的代表距离（避免outliers影响）
        percentile_dist = torch.quantile(min_distances, distance_percentile / 100.0).item()
        distances.append(percentile_dist)

    distances = np.array(distances)

    # 找波谷（局部最小值）
    from scipy.signal import find_peaks

    # 找波谷 = 找负距离的波峰
    valleys, valley_properties = find_peaks(-distances, prominence=0.01)

    # 正序：找第一个波谷（接触点）
    if len(valleys) > 0:
        approaching_end = start_idx + valleys[0]
    else:
        # 如果没有找到波谷，使用最小值点
        approaching_end = start_idx + np.argmin(distances[:len(distances) // 2 + 1])

    # 逆序：找第一个波谷（释放点）
    # 反转距离数组
    distances_reversed = distances[::-1]
    valleys_rev, valley_properties_rev = find_peaks(-distances_reversed, prominence=0.01)

    if len(valleys_rev) > 0:
        interaction_end = end_idx - valleys_rev[0]
    else:
        # 如果没有找到波谷，使用最小值点
        interaction_end = start_idx + len(distances) // 2 + np.argmin(distances[len(distances) // 2:])

    # 确保顺序正确
    if approaching_end >= interaction_end:
        # 如果检测失败，使用距离最小的连续区域
        # 找到距离小于median的区间
        threshold = np.median(distances)
        is_close = distances < threshold

        # 找到最长的连续True区间
        diff = np.diff(np.concatenate(([False], is_close, [False])).astype(int))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]

        if len(starts) > 0 and len(ends) > 0:
            lengths = ends - starts
            longest_idx = np.argmax(lengths)
            approaching_end = start_idx + starts[longest_idx]
            interaction_end = start_idx + ends[longest_idx]
        else:
            # 最后的fallback：使用中间区域
            total_len = end_idx - start_idx
            approaching_end = start_idx + total_len // 3
            interaction_end = end_idx - total_len // 3

    # 确保至少有合理的交互段长度
    min_interaction_frames = max(3, (end_idx - start_idx) // 10)
    if interaction_end - approaching_end < min_interaction_frames:
        mid = (approaching_end + interaction_end) // 2
        half_len = min_interaction_frames // 2
        approaching_end = max(start_idx + 1, mid - half_len)
        interaction_end = min(end_idx - 1, mid + half_len + min_interaction_frames % 2)

    # 可视化信息
    print(f"\nInteraction Detection (based on 3D distance):")
    print(f"  Distance stats: mean={distances.mean():.4f}, min={distances.min():.4f}, max={distances.max():.4f}")
    print(f"  Found {len(valleys)} valleys in forward pass, {len(valleys_rev)} in reverse pass")
    print(f"  Detected segments:")
    print(f"    1-2 Approaching: [{start_idx}:{approaching_end}] ({approaching_end - start_idx} frames)")
    print(f"    2-3 Interaction: [{approaching_end}:{interaction_end}] ({interaction_end - approaching_end} frames)")
    print(f"    3-4 Releasing: [{interaction_end}:{end_idx}] ({end_idx - interaction_end} frames)")

    return approaching_end, interaction_end


def weighted_false_negative_loss(rendered_mask, gt_mask, weight=20.0):
    """
    rendered_mask: [B, H, W] (Soft 0~1)
    gt_mask: [B, H, W] (0 or 1)
    """
    # 1. 找到"漏报"区域 (False Negative)
    # 也就是 GT 是 1，但 Render 是 0 的地方
    # diff > 0 的地方就是 GT 有但 Render 没有的地方
    diff = gt_mask - rendered_mask

    # 只惩罚正值部分 (即漏掉的部分)
    # 使用 ReLU 截断，只取 diff > 0
    missed_area = torch.relu(diff)

    # 2. 计算加权 MSE
    # 这里的平方是为了让误差越大惩罚越狠
    loss = (missed_area**2).mean() * weight

    return loss


def weighted_false_positive_loss(pred_mask, gt_mask):
    # diff > 0 代表 Pred=1, GT=0 (溢出部分)
    diff = pred_mask - gt_mask
    return (torch.relu(diff)**2).mean()


def vectorized_pose_guiding_loss(posed_meshes, gt_mask, rendered_masks, cameras, num_samples=2000):
    """
    Args:
        gt_mask: (B, H, W) Ground Truth
        rendered_masks: (B, H, W) 当前渲染出来的 Mask (软/硬均可), 需要 detach
    """
    B, H, W = gt_mask.shape

    # ==========================================
    # 1. 计算重要性权重 (Importance Weights)
    # ==========================================
    flat_gt = gt_mask.view(B, -1)

    # 我们不需要对 rendered_masks 求导，它只用于决定"哪里重要"
    # 假设 rendered_masks 是软 mask (0~1)，我们取反得到 (1 - pred)
    # 当 pred 接近 0 时 (未覆盖)，(1-pred) 接近 1。
    flat_pred = rendered_masks.detach().view(B, -1)

    # 计算未覆盖程度 (False Negative Score)
    # 只有在 GT=1 的地方我们才关心覆盖情况
    uncovered_score = flat_gt * (1.0 - flat_pred)

    # 定义采样概率:
    # 基础权重: flat_gt (保证所有 GT 像素都有机会被采到)
    # 额外权重: uncovered_score * 10.0 (强行提升未覆盖区域的被选中概率)
    # + 1e-8: 防止全 0
    probs = flat_gt + (uncovered_score * 20.0) + 1e-8

    # ==========================================
    # 2. 基于权重进行采样 (Multinomial)
    # ==========================================
    # 此时，未覆盖的区域被选中的概率是已覆盖区域的 20 倍！
    # 这样能"保证"即使未覆盖区域很小，也能采到大量的点。
    flat_indices = torch.multinomial(probs, num_samples, replacement=True)

    # 3. 验证采样点有效性
    # 依然需要验证，因为我们加了 1e-8，且 GT 可能是空的
    sampled_val = torch.gather(flat_gt, 1, flat_indices)
    valid_mask = sampled_val > 0.5

    if valid_mask.sum() == 0:
        return torch.tensor(0.0, device=gt_mask.device, requires_grad=True)

    # 4. 坐标转换
    batch_y = torch.div(flat_indices, W, rounding_mode='floor')
    batch_x = flat_indices % W
    gt_points_batch = torch.stack([batch_x, batch_y], dim=2).float()

    # 5. 投影 & KNN
    verts_batch = posed_meshes.verts_padded()
    projected_batch = cameras.transform_points_screen(verts_batch, image_size=((H, W),))
    pred_points_batch = projected_batch[..., :2]

    num_verts_per_mesh = posed_meshes.num_verts_per_mesh()
    knn = knn_points(gt_points_batch, pred_points_batch, lengths2=num_verts_per_mesh, K=1)
    dists_sq = knn.dists.squeeze(-1)

    # 6. Loss 聚合
    valid_dists = dists_sq * valid_mask.float()
    num_valid_per_image = valid_mask.float().sum(dim=1)
    loss_per_image = valid_dists.sum(dim=1) / (num_valid_per_image + 1e-8)

    return loss_per_image.mean()


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
    lr=1e-2,
    num_steps=200,
    smoothness_weight=1.0,
    cotracker_model=None,
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

    # Find first frame where > 50% points moved > 5px AND IoU < 0.9 (compared to previous frame)
    moved_ratios_fwd = np.mean(displacements_fwd > displacement_threshold, axis=1)  # (T,)
    approaching_end_idx_auto = start_static_end_idx  # Default

    for t in range(start_static_end_idx, num_sampled_frames):
        # Check both displacement and IoU
        displacement_ok = moved_ratios_fwd[t] > ratio_threshold
        # For frame t, compare with frame t-1 (IoU between t-1 and t)
        iou_ok = False
        if t > 0:
            iou_ok = ious_consecutive[t - 1] < iou_threshold

        if displacement_ok and iou_ok:
            approaching_end_idx_auto = t
            print(f"  Frame {t}: {moved_ratios_fwd[t]*100:.1f}% points moved > {displacement_threshold}px from frame 0")
            print(f"         IoU with previous frame: {ious_consecutive[t-1]:.4f} < {iou_threshold}")
            print(f"  → Interaction start detected at frame {t}")
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

    # Find last frame where > 50% points moved > 5px AND IoU < 0.9 (searching backward from end_static_start_idx)
    moved_ratios_bwd = np.mean(displacements_bwd > displacement_threshold, axis=1)  # (T,)
    interaction_end_idx_auto = end_static_start_idx  # Default

    for t in range(end_static_start_idx - 1, approaching_end_idx_auto - 1, -1):
        # Check both displacement and IoU
        displacement_ok = moved_ratios_bwd[t] > ratio_threshold
        # For frame t, compare with frame t+1 (IoU between t and t+1)
        iou_ok = False
        if t < num_sampled_frames - 1:
            iou_ok = ious_consecutive[t] < iou_threshold

        if displacement_ok and iou_ok:
            interaction_end_idx_auto = t + 1
            print(f"  Frame {t}: {moved_ratios_bwd[t]*100:.1f}% points moved > {displacement_threshold}px from last frame")
            print(f"         IoU with next frame: {ious_consecutive[t]:.4f} < {iou_threshold}")
            print(f"  → Interaction end detected at frame {t + 1}")
            break

    if interaction_end_idx_auto == end_static_start_idx:
        print(f"  No significant movement detected, using default: {end_static_start_idx}")

    # Update the interaction boundaries
    approaching_end_idx = approaching_end_idx_auto
    interaction_end_idx = interaction_end_idx_auto

    print(f"\nAuto-detected interaction segment: [{approaching_end_idx}:{interaction_end_idx}] ({interaction_end_idx - approaching_end_idx} frames)")

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

    # # Save RGB frames between start_static_end_idx and end_static_start_idx
    # video_id = os.path.basename(seq_path)
    # rgb_save_dir = os.path.join(seq_path, video_id)
    # os.makedirs(rgb_save_dir, exist_ok=True)

    # # Convert sampled indices back to original frame indices
    # original_start_idx = sampled_indices[start_static_end_idx]
    # original_end_idx = sampled_indices[end_static_start_idx]

    # total_frames = original_end_idx - original_start_idx
    # target_frames = 30

    # # Uniformly sample 30 frames from the range
    # if total_frames > target_frames:
    #     sampled_frame_indices = np.linspace(original_start_idx, original_end_idx - 1, target_frames, dtype=int)
    #     sampled_frame_indices = np.unique(sampled_frame_indices).tolist()
    # else:
    #     sampled_frame_indices = list(range(original_start_idx, original_end_idx))

    # print(
    #     f"\nSaving {len(sampled_frame_indices)} uniformly sampled RGB frames from original frame {original_start_idx} to {original_end_idx} (total: {total_frames})"
    # )
    # print(f"  Save directory: {rgb_save_dir}")

    # for frame_counter, orig_frame_idx in enumerate(sampled_frame_indices):
    #     # Get RGB from original (non-sampled) data
    #     rgb_frame = cropped_rgbs_np[orig_frame_idx]  # (H, W, 3), values in [0, 1]

    #     # Convert to uint8 [0, 255]
    #     rgb_frame_uint8 = (rgb_frame * 255).astype(np.uint8)

    #     # Save as PNG
    #     save_filename = f"{frame_counter:04d}.png"
    #     save_path = os.path.join(rgb_save_dir, save_filename)
    #     imageio.imwrite(save_path, rgb_frame_uint8)

    # print(f"  Saved {len(sampled_frame_indices)} RGB frames to {rgb_save_dir}")

    # Optimize object scale and shared pose for 0-1 segment
    initial_scale = torch.tensor(initial_scale, dtype=torch.float32, device=device).repeat(3)  # (3,)
    single_frame_model = SingleObjectPose(initial_R, initial_T, initial_scale, verts, faces).to(device)
    optimizer = torch.optim.Adam([single_frame_model.rot_6d, single_frame_model.scale, single_frame_model.trans], lr=1e-4)

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

    # --- Differentiable Rendering Setup ---
    raster_settings = RasterizationSettings(image_size=(H_out, W_out), blur_radius=1e-4, faces_per_pixel=20)
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=camera_start_segment, raster_settings=raster_settings),
        shader=SoftSilhouetteShader(),
    )

    # --- Optimization Loop for 0-1 Segment ---
    loop = tqdm(range(200), desc=f"Optimizing Start Static Segment [0:{start_static_end_idx}]")
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

        # Extract rendered depth from fragments
        # zbuf shape: (N, H, W, K) where K is faces_per_pixel
        # Use the closest depth (zbuf[..., 0])
        rendered_depth = fragments.zbuf[..., 0]  # (N, H, W)

        # Combine valid depth and object mask
        depth_loss_mask = (rendered_depth > 0) & (rendered_masks > 0.5) & (sampled_modal_masks[:start_segment_size].float() > 0.5)  # (N, H, W)

        # Loss: MSE with all masks in 0-1 segment
        l2_loss = torch.nn.functional.mse_loss(rendered_masks, sampled_modal_masks[:start_segment_size].float())

        # Depth loss: constrain rendered depth to match metric depth in object region
        depth_loss = torch.tensor(0.0, device=device)
        if depth_loss_mask.sum() > 0:
            # Only compute depth loss where both masks are valid
            rendered_depth_masked = rendered_depth[depth_loss_mask]
            metric_depth_masked = sampled_metric_depths[depth_loss_mask]

            # L2 loss between rendered depth and metric depth
            depth_loss = torch.nn.functional.mse_loss(rendered_depth_masked, metric_depth_masked) * 1e2

        # total_loss = l2_loss + depth_loss
        total_loss = l2_loss

        total_loss.backward()
        optimizer.step()
        loop.set_postfix(loss=total_loss.item(), l2=l2_loss.item(), depth=depth_loss.item())

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
                optimizer = torch.optim.Adam([single_frame_model.rot_6d, single_frame_model.scale, single_frame_model.trans], lr=1e-4)
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
    # STAGE 3: Initialize PnP from the last frame of 0-1 segment
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

    # --- Frame Locking for object pose optimization (0-1, 1-2)---
    lock_indices = [i for i in range(0, approaching_end_idx)]
    lock_hook = create_lock_frames_hook(lock_indices)
    print(f"\nFrame locking: Locking object pose for frames 0-{approaching_end_idx-1} (0-1 and 1-2 segments)")
    print(f"  Locked frames: {lock_indices}")
    print(f"  Total locked: {len(lock_indices)} frames")
    print(f"  Unlocked frames: {list(range(approaching_end_idx, num_sampled_frames))}")

    # --- Model and Optimizer ---
    multi_frame_model = TemporalHandObjectPose(initial_R_mat, initial_T, initial_scale, verts_batch, faces_batch, sampled_mano_params).to(device)
    multi_frame_model.rot_6d.register_hook(lock_hook)
    multi_frame_model.trans.register_hook(lock_hook)

    obj_optimizer = torch.optim.Adam([multi_frame_model.rot_6d, multi_frame_model.trans], lr=lr)
    hand_optimizer = torch.optim.Adam([multi_frame_model.mano_root_orient, multi_frame_model.mano_trans, multi_frame_model.mano_pose], lr=lr)
    # num_steps = 0  # Uncomment to skip this stage
    loop = tqdm(range(num_steps), desc="Optimizing Pose Sequence")
    for step in loop:
        obj_optimizer.zero_grad()
        hand_optimizer.zero_grad()

        curr_sigma, curr_gamma, curr_fpp = get_render_params(step, num_steps)
        renderer.rasterizer.raster_settings.blur_radius = curr_sigma
        renderer.rasterizer.raster_settings.faces_per_pixel = curr_fpp
        new_blend_params = BlendParams(sigma=curr_sigma, gamma=curr_gamma)
        renderer.shader.blend_params = new_blend_params

        posed_obj_meshes_batch, posed_hand_meshes_batch, mano_joints_batch = multi_frame_model()

        obj_fragments = renderer.rasterizer(posed_obj_meshes_batch)
        rendered_obj_masks = renderer.shader(obj_fragments, posed_obj_meshes_batch)[..., 3]
        obj_zbuf = obj_fragments.zbuf[..., 0]

        # object loss
        loss_fp = weighted_false_positive_loss(rendered_obj_masks, sampled_pred_amodal_masks) * 1e2
        loss_vp = vectorized_pose_guiding_loss(posed_obj_meshes_batch, sampled_pred_amodal_masks, rendered_obj_masks, camera, num_samples=2000)
        loss_sm_rot = compute_smoothness_loss(multi_frame_model.rot_6d)
        loss_sm_trans = compute_smoothness_loss(multi_frame_model.trans)
        obj_loss = loss_fp + loss_vp + loss_sm_rot * smoothness_weight + loss_sm_trans * smoothness_weight

        # hand 2d joints loss
        projected_hand_joints = camera.transform_points_screen(mano_joints_batch, image_size=((H_out, W_out),))  # (N, 21, 3)
        pred_hand_joints_2d = projected_hand_joints[..., :2]
        num_valid_frames = sampled_gt_hand_joints_valid_mask.sum()
        if num_valid_frames > 0:
            loss_joints_2d = torch.nn.functional.mse_loss(pred_hand_joints_2d[sampled_gt_hand_joints_valid_mask],
                                                          sampled_gt_hand_joints_2d[sampled_gt_hand_joints_valid_mask])
        else:
            loss_joints_2d = torch.tensor(0.0, device=device)
        loss_sm_hand = compute_smoothness_loss(mano_joints_batch)

        # hand anatomy loss
        T_g_p = multi_frame_model.transforms_abs  # (B, 16, 4, 4)
        T_g_a, _R, ee = multi_frame_model.axisFK(T_g_p)  # ee (B, 16, 3)
        loss_anatomy = multi_frame_model.anatomyLoss(ee)

        hand_loss = loss_joints_2d + loss_sm_hand * smoothness_weight + loss_anatomy

        total_loss = obj_loss + hand_loss

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(multi_frame_model.parameters(), max_norm=1.0)
        obj_optimizer.step()
        hand_optimizer.step()
        loop.set_postfix(
            loss=total_loss.item(),
            loss_fp=loss_fp.item(),
            loss_vp=loss_vp.item(),
            loss_sm_rot=loss_sm_rot.item(),
            loss_sm_trans=loss_sm_trans.item(),
            loss_joints_2d=loss_joints_2d.item(),
            loss_sm_hand=loss_sm_hand.item(),
            loss_anatomy=loss_anatomy.item(),
        )

    # --- Quick Visualization of Sparse Optimization Results ---
    with torch.no_grad():
        sparse_posed_meshes, sparse_posed_hand_meshes, sparse_mano_joints = multi_frame_model()
        scenes = []
        for i in range(len(sparse_posed_meshes)):
            scene_i = join_meshes_as_scene([sparse_posed_meshes[i], sparse_posed_hand_meshes[i]])
            scenes.append(scene_i)
        sparse_scene = join_meshes_as_batch(scenes)
        final_rendered_masks_sparse = renderer(sparse_scene)[..., 3].cpu().numpy()

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
        panels = [modal_frame, amodal_gt_frame, render_frame, hand_frame, joints_frame]

        combined_frame = np.hstack(panels)

        # Convert to uint8 for cv2.putText
        combined_frame_uint8 = (combined_frame * 255).astype(np.uint8)

        # Add stage annotation on the frame
        cv2.putText(combined_frame_uint8, f"Frame {i}: {stage_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, stage_color, 2, cv2.LINE_AA)

        writer_sparse.append_data(combined_frame_uint8)
    writer_sparse.close()
    print(f"Saved sparse fitting visualization to {sparse_video_path}")

    # ========================================================================
    # STAGE 4: Load Contact Correspondence (from Grasping Pose Correction)
    # ========================================================================
    print("\n" + "=" * 80)
    print("STAGE 4: Loading Contact Correspondence")
    print("=" * 80)

    contact_map_path = os.path.join(output_path, "grasp_correction/contact_map.npy")

    if os.path.exists(contact_map_path):
        print(f"Found contact map at: {contact_map_path}")
        contact_map = np.load(contact_map_path, allow_pickle=True).item()

        # Parse: convert string keys to int if needed
        parsed_contact_map = parse_contact_map(contact_map)

        print(f"Loaded {parsed_contact_map['num_contacts']} contact correspondences")

        # Show some examples
        print(f"\nExample correspondences:")
        for i, (hand_idx, corr_data) in enumerate(list(parsed_contact_map['correspondences'].items())[:5]):
            face_id = corr_data['face_id']
            bary = corr_data['bary_coords']
            print(f"  Hand vertex {hand_idx} -> Object face {face_id}, bary=[{bary[0]:.3f}, {bary[1]:.3f}, {bary[2]:.3f}]")
        if parsed_contact_map['num_contacts'] > 5:
            print(f"  ... and {parsed_contact_map['num_contacts'] - 5} more")
    else:
        print(f"No contact map found at: {contact_map_path}")
        print("Skipping contact-based optimization")
        contact_map = None
        parsed_contact_map = None

    # ========================================================================
    # STAGE 5: Select Anchor Frame and Optimize HOI
    # ========================================================================
    if parsed_contact_map is not None:
        print("\n" + "=" * 80)
        print("STAGE 5: Anchor Frame Optimization")
        print("=" * 80)

        # Select anchor frame based on minimum contact correspondence distance
        # Search in last p1% of approaching (1-2) and first p2% of interaction (2-3) segments
        print("\nSelecting anchor frame based on contact correspondence distances...")

        # Calculate search range: last p1% of 1-2 segment + first p2% of 2-3 segment
        approaching_length = approaching_end_idx - start_static_end_idx
        interaction_length = interaction_end_idx - approaching_end_idx

        search_start = approaching_end_idx - int(approaching_length * 0.2)  # Last 20% of 1-2
        search_end = approaching_end_idx + int(interaction_length * 0.0)  # First 0% of 2-3

        # Ensure at least a few frames to search
        if search_end - search_start < 1:
            search_start = max(start_static_end_idx, approaching_end_idx - 1)

        min_distance = float('inf')
        anchor_frame_idx = approaching_end_idx  # Default fallback
        frame_distances = []

        for frame_idx in range(search_start, search_end):
            # Get object mesh for this frame (transformed)
            obj_R = rotation_6d_to_matrix(multi_frame_model.rot_6d[frame_idx])
            obj_scale_full = multi_frame_model.scale * multi_frame_model.initial_scale
            obj_trans = multi_frame_model.trans[frame_idx]
            obj_verts_transformed = (verts * obj_scale_full.unsqueeze(0)) @ obj_R.T + obj_trans
            obj_mesh_frame = Meshes(verts=[obj_verts_transformed],
                                    faces=[faces],
                                    textures=TexturesVertex(verts_features=torch.ones_like(obj_verts_transformed)[None]))

            # Get hand vertices for this frame
            flat_mat = torch.diag(torch.tensor([-1., -1., 1.], dtype=torch.float32, device=device))[None]
            with torch.no_grad():
                amano_output = run_amano(multi_frame_model.hand_model_amano, multi_frame_model.mano_trans[frame_idx:frame_idx + 1].unsqueeze(0),
                                         multi_frame_model.mano_root_orient[frame_idx:frame_idx + 1].unsqueeze(0),
                                         multi_frame_model.mano_pose[frame_idx:frame_idx + 1].unsqueeze(0),
                                         multi_frame_model.is_right[frame_idx:frame_idx + 1].to(device))
            hand_verts = amano_output['vertices'].squeeze(0).squeeze(0) @ flat_mat[0]

            # Use the existing function to compute contact correspondences
            hand_contact_indices, obj_contact_points, obj_contact_normals = compute_contact_points_from_correspondences(parsed_contact_map['correspondences'],
                                                                                                                        obj_mesh_frame,
                                                                                                                        device=device)

            # Get corresponding hand vertices
            hand_contact_verts = hand_verts[hand_contact_indices]

            # Compute average distance between corresponding points
            distances = torch.norm(hand_contact_verts - obj_contact_points, dim=1)
            avg_distance = distances.mean().item()
            frame_distances.append((frame_idx, avg_distance))

            # Update minimum
            if avg_distance < min_distance:
                min_distance = avg_distance
                anchor_frame_idx = frame_idx

        # Print results
        print(f"\nContact distance analysis (search range: [{search_start}:{search_end}]):")
        print(f"  Approaching segment (1-2): [{start_static_end_idx}:{approaching_end_idx}]")
        print(f"  Interaction segment (2-3): [{approaching_end_idx}:{interaction_end_idx}]")
        print(f"\nDistance statistics:")
        distances_array = np.array([d for _, d in frame_distances])
        print(f"  Min: {distances_array.min():.6f} m")
        print(f"  Max: {distances_array.max():.6f} m")
        print(f"  Mean: {distances_array.mean():.6f} m")
        print(f"  Std: {distances_array.std():.6f} m")

        # Show top 5 frames with smallest distances
        sorted_frames = sorted(frame_distances, key=lambda x: x[1])[:5]
        print(f"\nTop 5 frames with smallest contact distances:")
        for rank, (fid, dist) in enumerate(sorted_frames, 1):
            segment = "1-2 (approaching)" if fid < approaching_end_idx else "2-3 (interaction)"
            print(f"  {rank}. Frame {fid} ({segment}): {dist:.6f} m")

        print(f"\n✓ Selected anchor frame: {anchor_frame_idx} with distance {min_distance:.6f} m")

        # Step 5.2: Prepare data for anchor frame
        anchor_obj_amodal_mask = sampled_pred_amodal_masks[anchor_frame_idx]
        anchor_obj_modal_mask = sampled_modal_masks[anchor_frame_idx]
        anchor_hand_mask = sampled_hand_masks[anchor_frame_idx]
        anchor_hand_joints_2d = sampled_gt_hand_joints_2d[anchor_frame_idx]

        # Get canonical object mesh for correspondence computation
        verts_canonical_scaled = verts * multi_frame_model.scale.detach().unsqueeze(0) * multi_frame_model.initial_scale
        mesh_canonical_scaled = Meshes(
            verts=[verts_canonical_scaled],
            faces=[faces],
            textures=TexturesVertex(verts_features=torch.ones_like(verts)[None]),
        )

        # Create single-frame model for anchor optimization
        initial_anchor_R = rotation_6d_to_matrix(multi_frame_model.rot_6d[anchor_frame_idx].unsqueeze(0)).detach().clone()
        initial_anchor_T = multi_frame_model.trans[anchor_frame_idx].detach().clone()
        initial_anchor_scale = multi_frame_model.scale.detach().clone() * multi_frame_model.initial_scale

        anchor_obj_model = SingleObjectPose(initial_anchor_R[0], initial_anchor_T, initial_anchor_scale, verts, faces).to(device)

        # Extract anchor frame MANO parameters
        anchor_mano_root_orient = multi_frame_model.mano_root_orient[anchor_frame_idx].detach().clone()
        anchor_mano_trans = multi_frame_model.mano_trans[anchor_frame_idx].detach().clone()
        anchor_mano_pose = multi_frame_model.mano_pose[anchor_frame_idx].detach().clone()

        # Make MANO parameters optimizable
        anchor_mano_root_orient = nn.Parameter(anchor_mano_root_orient)
        anchor_mano_trans = nn.Parameter(anchor_mano_trans)
        anchor_mano_pose = nn.Parameter(anchor_mano_pose)

        # Optimizers
        # obj_optimizer_anchor = torch.optim.Adam([anchor_obj_model.rot_6d, anchor_obj_model.trans], lr=1e-4)
        hand_optimizer_anchor = torch.optim.Adam([
            {
                'params': anchor_mano_trans,
                'lr': 1e-3
            },
            {
                'params': anchor_mano_root_orient,
                'lr': 1e-3
            },
            {
                'params': anchor_mano_pose,
                'lr': 5e-4
            }  # Pose 学习率给大点，让手指能弯曲
        ])

        # Renderer for anchor frame
        anchor_camera = PerspectiveCameras(
            focal_length=focal_length[anchor_frame_idx, None],
            principal_point=principal_point[anchor_frame_idx, None],
            image_size=((H_out, W_out),),
            in_ndc=False,
            device=device,
        )

        anchor_raster_settings = RasterizationSettings(image_size=(H_out, W_out), blur_radius=1e-4, faces_per_pixel=20)
        anchor_renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=anchor_camera, raster_settings=anchor_raster_settings),
            shader=SoftSilhouetteShader(blend_params=BlendParams(sigma=1e-4, gamma=1e-4)),
        )

        # Step 5.3: Optimization loop
        num_anchor_steps = 3000
        loop = tqdm(range(num_anchor_steps), desc="Optimizing Anchor Frame HOI")

        for step in loop:
            # obj_optimizer_anchor.zero_grad()
            hand_optimizer_anchor.zero_grad()

            # Forward pass: get object and hand meshes
            anchor_obj_mesh = anchor_obj_model()

            # Get hand mesh
            flat_mat = torch.diag(torch.tensor([-1., -1., 1.], dtype=torch.float32, device=device))[None]
            amano_output = run_amano(multi_frame_model.hand_model_amano, anchor_mano_trans[None, None], anchor_mano_root_orient[None, None],
                                     anchor_mano_pose[None, None], multi_frame_model.is_right[anchor_frame_idx:anchor_frame_idx + 1].to(device))
            anchor_hand_verts = amano_output['vertices'].squeeze(0).squeeze(0) @ flat_mat[0]  # (778, 3)
            anchor_hand_joints = amano_output['joints'].squeeze(0).squeeze(0) @ flat_mat[0]  # (21, 3)
            mano_is_right = amano_output['is_right'].squeeze()
            mano_faces = amano_output['r_faces'] if mano_is_right.item() > 0 else amano_output['l_faces']

            anchor_hand_mesh = Meshes(
                verts=[anchor_hand_verts],
                faces=[mano_faces],
                textures=TexturesVertex(verts_features=torch.ones_like(anchor_hand_verts)[None]),
            )

            # Loss 1: Contact correspondence loss (with tangent and normal decomposition)
            if parsed_contact_map['num_contacts'] > 0:
                hand_contact_indices, obj_contact_points, obj_contact_normals = compute_contact_points_from_correspondences(
                    parsed_contact_map['correspondences'], anchor_obj_mesh, device=device)
                hand_contact_verts = anchor_hand_verts[hand_contact_indices]

                # # Get hand vertex normals
                # anchor_hand_normals = anchor_hand_mesh.verts_normals_packed()  # (778, 3)
                # hand_contact_normals = anchor_hand_normals[hand_contact_indices]  # (K, 3)

                # # Target points and normals
                # target_points = obj_contact_points  # (K, 3)
                # target_normals = obj_contact_normals  # (K, 3)

                # # 1. Compute difference vector
                # diff_vec = hand_contact_verts - target_points  # (K, 3)

                # # 2. Decompose into normal and tangent components
                # # Project onto normal direction (positive = outside, negative = inside)
                # # Note: target_normals are assumed to point outward
                # dist_normal = torch.sum(diff_vec * target_normals, dim=1, keepdim=True)  # (K, 1)

                # # Tangent vector (sliding component on surface)
                # vec_tangent = diff_vec - dist_normal * target_normals  # (K, 3)

                # # 3. Compute Loss
                # # Tangent Loss: must align (L2)
                # loss_tangent = torch.norm(vec_tangent, dim=1).mean()

                # # Normal attraction Loss: only pull when outside (ReLU)
                # # Allow 1mm tolerance (eps=0.001)
                # loss_normal_attract = torch.nn.functional.relu(dist_normal - 0.001).mean()

                # # Combined position Loss
                # contact_loss = loss_tangent + loss_normal_attract
                contact_loss = torch.nn.functional.mse_loss(hand_contact_verts, obj_contact_points)
            else:
                contact_loss = torch.tensor(0.0, device=device)

            # Loss 2: Object 2D mask loss
            # obj_fragments = anchor_renderer.rasterizer(anchor_obj_mesh, cameras=anchor_camera)
            # rendered_obj_mask = anchor_renderer.shader(obj_fragments, anchor_obj_mesh, cameras=anchor_camera)[0, ..., 3]
            # obj_mask_loss = torch.nn.functional.mse_loss(rendered_obj_mask, anchor_obj_amodel_mask)

            # loss_fp = weighted_false_positive_loss(rendered_obj_mask, anchor_obj_amodal_mask) * 1e2
            # loss_vp = vectorized_pose_guiding_loss(anchor_obj_mesh, anchor_obj_amodal_mask[None], rendered_obj_mask[None], anchor_camera, num_samples=2000)
            # obj_mask_loss = loss_fp + loss_vp

            # Loss 3: Hand 2D keypoints loss (exclude fingertips)
            # MANO has 21 joints: wrist (0) + 5 fingers x 4 joints each (1-4, 5-8, 9-12, 13-16, 17-20)
            # Fingertips are at indices: 4, 8, 12, 16, 20
            # Use all joints except fingertips: [0,1,2,3, 5,6,7, 9,10,11, 13,14,15, 17,18,19]
            # palm_joint_indices = torch.tensor([0, 1, 2, 5, 6, 9, 10, 13, 14, 17, 18], dtype=torch.long, device=device)
            palm_joint_indices = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], dtype=torch.long, device=device)

            projected_hand_joints = anchor_camera.transform_points_screen(anchor_hand_joints.unsqueeze(0), image_size=((H_out, W_out),))[0, :, :2]  # (21, 2)

            # Only compute loss on palm-related joints
            projected_palm_joints = projected_hand_joints[palm_joint_indices]  # (6, 2)
            gt_palm_joints = anchor_hand_joints_2d[palm_joint_indices]  # (6, 2)

            hand_2d_loss = torch.nn.functional.mse_loss(projected_palm_joints, gt_palm_joints)

            # Loss 4: Collision/penetration loss
            anchor_obj_verts = anchor_obj_mesh.verts_packed()  # (N_obj, 3)
            anchor_obj_normals = anchor_obj_mesh.verts_normals_packed()  # (N_obj, 3)
            anchor_hand_normals = anchor_hand_mesh.verts_normals_packed()  # (N_hand, 3)
            # collision_loss = compute_collision_loss(anchor_obj_verts, anchor_hand_verts, anchor_hand_normals)
            collision_hand_in_obj_loss = compute_collision_loss(anchor_obj_verts, anchor_obj_normals, anchor_hand_verts, ignore_indices=hand_contact_indices)
            collision_obj_in_hand_loss = compute_collision_loss(anchor_hand_verts, anchor_hand_normals, anchor_obj_verts, ignore_indices=hand_contact_indices)
            collision_loss = collision_hand_in_obj_loss + collision_obj_in_hand_loss
            # Loss 5: Hand anatomy loss
            T_g_p = amano_output['transforms_abs'].squeeze(0)  # (16, 4, 4)
            T_g_a, _R, ee = multi_frame_model.axisFK(T_g_p.unsqueeze(0))
            anatomy_loss = multi_frame_model.anatomyLoss(ee.squeeze(0).unsqueeze(0))

            # # Loss 6: Semantic layering loss (hand-object occlusion consistency)
            # semantic_loss = compute_semantic_layering_loss(anchor_hand_mesh, anchor_obj_mesh, anchor_hand_mask, anchor_obj_modal_mask, anchor_camera,
            #                                                anchor_raster_settings, device)

            # Total loss
            contact_loss_weight = 1e5
            # obj_mask_loss_weight = 1e0
            anatomy_loss_weight = 1e2
            # semantic_loss_weight = 1e4
            hand_2d_loss_weight = 1e0
            collision_loss_weight = 1e1
            total_loss = contact_loss_weight * contact_loss + \
                anatomy_loss_weight * anatomy_loss + \
                hand_2d_loss_weight * hand_2d_loss + \
                collision_loss_weight * collision_loss
            # obj_mask_loss_weight * obj_mask_loss + \
            # semantic_loss_weight * semantic_loss + \

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                # [anchor_obj_model.rot_6d, anchor_obj_model.trans, anchor_obj_model.scale, anchor_mano_root_orient, anchor_mano_trans, anchor_mano_pose],
                [anchor_mano_root_orient, anchor_mano_trans, anchor_mano_pose],
                max_norm=1.0)
            # obj_optimizer_anchor.step()
            hand_optimizer_anchor.step()

            loop.set_postfix(
                total=total_loss.item(),
                contact=(contact_loss * contact_loss_weight).item(),
                hand_2d=(hand_2d_loss * hand_2d_loss_weight).item(),
                collision=(collision_loss * collision_loss_weight).item(),
                anatomy=(anatomy_loss * anatomy_loss_weight).item(),
                # obj_mask=obj_mask_loss.item(),
                #  semantic=semantic_loss.item(),
            )

        # Step 5.4: Save optimized anchor frame results back to multi_frame_model
        with torch.no_grad():
            # multi_frame_model.rot_6d.data[anchor_frame_idx] = anchor_obj_model.rot_6d.data
            # multi_frame_model.trans.data[anchor_frame_idx] = anchor_obj_model.trans.data
            multi_frame_model.mano_root_orient.data[anchor_frame_idx] = anchor_mano_root_orient.data
            multi_frame_model.mano_trans.data[anchor_frame_idx] = anchor_mano_trans.data
            multi_frame_model.mano_pose.data[anchor_frame_idx] = anchor_mano_pose.data

        print(f"\nAnchor frame {anchor_frame_idx} optimized successfully")

        # Visualization: Render colored meshes
        with torch.no_grad():
            # Create colored renderer
            lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])
            materials = Materials(device=device, specular_color=[[1.0, 1.0, 1.0]], shininess=1.0)
            blend_params = BlendParams(background_color=(0.0, 0.0, 0.0))

            color_raster_settings = RasterizationSettings(image_size=(H_out, W_out), blur_radius=0.0, faces_per_pixel=1)
            color_renderer = MeshRenderer(rasterizer=MeshRasterizer(cameras=anchor_camera, raster_settings=color_raster_settings),
                                          shader=HardPhongShader(device=device, lights=lights, materials=materials, blend_params=blend_params))

            # Get final meshes with colors
            final_obj_mesh_raw = anchor_obj_model()
            obj_verts = final_obj_mesh_raw.verts_list()[0]
            obj_faces = final_obj_mesh_raw.faces_list()[0]

            # Object: blue color
            obj_verts_color = torch.tensor([0.3, 0.5, 0.9], device=device).repeat(obj_verts.shape[0], 1)

            # Hand: orange/skin color
            hand_verts_color = torch.tensor([0.9, 0.6, 0.4], device=device).repeat(anchor_hand_verts.shape[0], 1)

            # Create combined mesh with proper textures
            # Combine vertices and faces
            combined_verts = torch.cat([obj_verts, anchor_hand_verts.detach()], dim=0)
            # Offset hand faces by number of object vertices
            hand_faces_offset = mano_faces + obj_verts.shape[0]
            combined_faces = torch.cat([obj_faces, hand_faces_offset], dim=0)
            # Combine colors
            combined_colors = torch.cat([obj_verts_color, hand_verts_color], dim=0)

            # Create single mesh with combined geometry
            combined_mesh = Meshes(
                verts=[combined_verts],
                faces=[combined_faces],
                textures=TexturesVertex(verts_features=combined_colors[None]),
            )

            # Render combined scene with proper depth handling
            rendered_rgb = color_renderer(combined_mesh, cameras=anchor_camera)[0, ..., :3].cpu().numpy()

            # Composite rendered mesh on original image
            rendered_mask = (rendered_rgb.sum(axis=-1) > 0).astype(np.float32)
            original_img = sampled_rgbs_np[anchor_frame_idx]
            render_composite = original_img * (1 - rendered_mask[..., None]) + rendered_rgb * rendered_mask[..., None]

            # Save visualization
            modal_frame = overlay_mask_on_image(sampled_rgbs_np[anchor_frame_idx], sampled_modal_masks_np[anchor_frame_idx])
            amodal_frame = overlay_mask_on_image(sampled_rgbs_np[anchor_frame_idx], sampled_pred_amodal_masks_np[anchor_frame_idx])

            # Draw keypoints
            joints_frame = render_composite.copy()
            joints_frame = overlay_points_on_image(joints_frame, projected_hand_joints.cpu().numpy(), color=(1.0, 0.0, 0.0))
            joints_frame = overlay_points_on_image(joints_frame, anchor_hand_joints_2d.cpu().numpy(), color=(0.0, 1.0, 0.0))

            panels = [modal_frame, amodal_frame, render_composite, joints_frame]
            combined_image = (np.hstack(panels) * 255).astype(np.uint8)
            save_path = os.path.join(output_path, f"anchor_frame_{anchor_frame_idx}_optimized.png")
            imageio.imwrite(save_path, combined_image)
            print(f"Saved anchor frame visualization to {save_path}")

            # Save combined mesh using join_meshes_as_scene
            # Create colored meshes for object and hand
            obj_mesh_colored = Meshes(verts=[obj_verts], faces=[obj_faces], textures=TexturesVertex(verts_features=obj_verts_color.unsqueeze(0)))

            hand_mesh_colored = Meshes(verts=[anchor_hand_verts.detach()],
                                       faces=[mano_faces],
                                       textures=TexturesVertex(verts_features=hand_verts_color.unsqueeze(0)))

            # Join as scene
            combined_scene_mesh = join_meshes_as_scene([obj_mesh_colored, hand_mesh_colored])

            # Extract vertices, faces, and colors
            combined_verts_scene = combined_scene_mesh.verts_packed().cpu().numpy()
            combined_faces_scene = combined_scene_mesh.faces_packed().cpu().numpy()
            combined_colors_scene = combined_scene_mesh.textures.verts_features_packed().cpu().numpy()

            # Save as ply
            combined_mesh_save = trimesh.Trimesh(vertices=combined_verts_scene,
                                                 faces=combined_faces_scene,
                                                 vertex_colors=(combined_colors_scene * 255).astype(np.uint8))
            combined_ply_path = os.path.join(output_path, f"anchor_frame_{anchor_frame_idx}_optimized.ply")
            combined_mesh_save.export(combined_ply_path)
            print(f"Saved optimized combined mesh to {combined_ply_path}")
    else:
        print("\nSkipping STAGE 5: No contact correspondences available")

    # ============================================================================================================
    # STAGE 6: Propagate hand-object relative pose to interaction segment (2-3)
    # ============================================================================================================
    print("\n" + "=" * 80)
    print("STAGE 6: Propagate Hand-Object Relative Pose to Interaction Segment")
    print("=" * 80)

    if parsed_contact_map is not None and parsed_contact_map['num_contacts'] > 0:
        # Step 6.1: Copy MANO joint rotations (not root orient/trans) from anchor to remaining frames after anchor frame in 1-2 and all frames in 2-3
        print(f"\nStep 6.1: Copying MANO joint rotations from anchor frame {anchor_frame_idx} to frames {anchor_frame_idx+1}-{interaction_end_idx}")

        with torch.no_grad():
            # Get the optimized MANO pose from anchor frame (45 dims, 15 joints * 3)
            anchor_mano_pose_optimized = multi_frame_model.mano_pose.data[anchor_frame_idx].clone()

            # Copy to all frames in interaction segment
            for frame_idx in range(anchor_frame_idx + 1, interaction_end_idx):
                multi_frame_model.mano_pose.data[frame_idx] = anchor_mano_pose_optimized

        print(f"Copied MANO joint rotations to {interaction_end_idx - anchor_frame_idx - 1} frames")

        # Step 6.2: Compute hand-object relative transformation from anchor frame
        print(f"\nStep 6.2: Computing hand-object relative transformation from anchor frame")

        with torch.no_grad():
            # Get all meshes from multi_frame_model
            obj_meshes, hand_meshes, hand_joints_all = multi_frame_model()

            # Extract anchor frame data
            anchor_hand_joints = hand_joints_all[anchor_frame_idx]  # (21, 3)
            anchor_hand_wrist = anchor_hand_joints[0]  # (3,)

            # Build hand's local coordinate system from joints
            # Use wrist to middle finger base as forward direction
            hand_forward = anchor_hand_joints[9] - anchor_hand_joints[0]  # middle finger MCP - wrist
            hand_forward = hand_forward / torch.norm(hand_forward)

            # Use wrist to index finger base as side direction
            hand_side = anchor_hand_joints[5] - anchor_hand_joints[0]  # index finger MCP - wrist
            hand_side = hand_side / torch.norm(hand_side)

            # Compute hand up direction (perpendicular to forward and side)
            hand_up = torch.cross(hand_forward, hand_side)
            hand_up = hand_up / torch.norm(hand_up)

            # Recompute side to ensure orthogonality
            hand_side = torch.cross(hand_up, hand_forward)
            hand_side = hand_side / torch.norm(hand_side)

            # Build hand rotation matrix (columns are the local axes)
            anchor_hand_R = torch.stack([hand_side, hand_up, hand_forward], dim=1)  # (3, 3)
            anchor_hand_T = anchor_hand_wrist  # (3,)

            # Get anchor frame's object pose
            anchor_obj_R = rotation_6d_to_matrix(multi_frame_model.rot_6d[anchor_frame_idx])  # (3, 3)
            anchor_obj_T = multi_frame_model.trans[anchor_frame_idx]  # (3,)

            # Compute relative transformation: T_rel = T_hand^{-1} × T_obj
            # rel_R = hand_R^T @ obj_R
            # rel_T = hand_R^T @ (obj_T - hand_T)
            rel_R = anchor_hand_R.T @ anchor_obj_R
            rel_T = anchor_hand_R.T @ (anchor_obj_T - anchor_hand_T)

            print(f"  Anchor frame {anchor_frame_idx}:")
            print(f"    Hand wrist: {anchor_hand_wrist.cpu().numpy()}")
            print(f"    Hand forward: {hand_forward.cpu().numpy()}")
            print(f"    Object translation: {anchor_obj_T.cpu().numpy()}")
            print(f"    Relative translation (hand frame): {rel_T.cpu().numpy()}")
            print(f"    Distance: {torch.norm(rel_T).item():.4f}")

        print(f"Computed relative transformation")

        # Step 6.3: Apply relative transformation to remaining frames after anchor frame in 1-2 and all frames in 2-3
        print(f"\nStep 6.3: Aligning object poses to hand poses for frames {anchor_frame_idx+1}-{interaction_end_idx}")

        with torch.no_grad():
            for frame_idx in range(anchor_frame_idx + 1, interaction_end_idx):
                # Get current frame's hand joints
                hand_joints_frame = hand_joints_all[frame_idx]  # (21, 3)
                hand_wrist_frame = hand_joints_frame[0]  # (3,)

                # Build current frame's hand coordinate system (same method as anchor)
                hand_forward_frame = hand_joints_frame[9] - hand_joints_frame[0]
                hand_forward_frame = hand_forward_frame / torch.norm(hand_forward_frame)

                hand_side_frame = hand_joints_frame[5] - hand_joints_frame[0]
                hand_side_frame = hand_side_frame / torch.norm(hand_side_frame)

                hand_up_frame = torch.cross(hand_forward_frame, hand_side_frame)
                hand_up_frame = hand_up_frame / torch.norm(hand_up_frame)

                hand_side_frame = torch.cross(hand_up_frame, hand_forward_frame)
                hand_side_frame = hand_side_frame / torch.norm(hand_side_frame)

                hand_R_frame = torch.stack([hand_side_frame, hand_up_frame, hand_forward_frame], dim=1)  # (3, 3)
                hand_T_frame = hand_wrist_frame  # (3,)

                # Apply relative transformation: T_obj = T_hand × T_rel
                # obj_R = hand_R @ rel_R
                # obj_T = hand_R @ rel_T + hand_T
                obj_R_new = hand_R_frame @ rel_R
                obj_T_new = hand_R_frame @ rel_T + hand_T_frame

                # Update object pose
                multi_frame_model.rot_6d.data[frame_idx] = matrix_to_rotation_6d(obj_R_new)
                multi_frame_model.trans.data[frame_idx] = obj_T_new

            # Re-run forward pass to get updated meshes for verification
            obj_meshes_verify, hand_meshes_verify, hand_joints_verify = multi_frame_model()

            # Verify alignment for a few sample frames
            print(f"\n  Verification (sample frames):")
            sample_frames = [anchor_frame_idx + 1, anchor_frame_idx + 1 + (interaction_end_idx - anchor_frame_idx - 1) // 2, interaction_end_idx - 1]

            for frame_idx in sample_frames:
                if frame_idx < interaction_end_idx:
                    # Compute relative pose in current frame and compare with anchor
                    hand_joints_verify_frame = hand_joints_verify[frame_idx]
                    hand_wrist_verify = hand_joints_verify_frame[0]

                    # Build hand coordinate system
                    hand_forward_v = hand_joints_verify_frame[9] - hand_joints_verify_frame[0]
                    hand_forward_v = hand_forward_v / torch.norm(hand_forward_v)
                    hand_side_v = hand_joints_verify_frame[5] - hand_joints_verify_frame[0]
                    hand_side_v = hand_side_v / torch.norm(hand_side_v)
                    hand_up_v = torch.cross(hand_forward_v, hand_side_v)
                    hand_up_v = hand_up_v / torch.norm(hand_up_v)
                    hand_side_v = torch.cross(hand_up_v, hand_forward_v)
                    hand_side_v = hand_side_v / torch.norm(hand_side_v)
                    hand_R_verify = torch.stack([hand_side_v, hand_up_v, hand_forward_v], dim=1)

                    # Get object pose
                    obj_R_verify = rotation_6d_to_matrix(multi_frame_model.rot_6d[frame_idx])
                    obj_T_verify = multi_frame_model.trans[frame_idx]

                    # Compute relative pose in current frame
                    rel_R_verify = hand_R_verify.T @ obj_R_verify
                    rel_T_verify = hand_R_verify.T @ (obj_T_verify - hand_wrist_verify)

                    # Compare with anchor's relative pose
                    rel_R_diff = torch.norm(rel_R_verify - rel_R).item()
                    rel_T_diff = torch.norm(rel_T_verify - rel_T).item()

                    print(f"    Frame {frame_idx}: rel_R_diff={rel_R_diff:.6f}, rel_T_diff={rel_T_diff:.6f}")

        print(f"\nAligned object poses for {interaction_end_idx - anchor_frame_idx - 1} frames")

        # Save visualization for frames 0 to interaction_end_idx (0-3 segment)
        print(f"\nSaving K3D visualization for frames 0-{interaction_end_idx}...")
        with torch.no_grad():
            # Get all meshes for frames 0 to interaction_end_idx
            frames_to_save = list(range(0, min(interaction_end_idx, num_sampled_frames)))
            obj_meshes_all, hand_meshes_all, _ = multi_frame_model()

            # Extract vertices and faces for each frame, with coordinate transformation
            flat_mat = torch.diag(torch.tensor([-1., -1., 1.], dtype=torch.float32, device=device))
            obj_verts_list = []
            hand_verts_list = []

            for frame_idx in frames_to_save:
                # Object mesh - apply coordinate transformation
                obj_verts_frame = obj_meshes_all.verts_list()[frame_idx]  # (V_obj, 3)
                obj_verts_frame_transformed = obj_verts_frame @ flat_mat  # Apply flip transformation
                obj_verts_list.append(obj_verts_frame_transformed.cpu().numpy())

                # Hand mesh - apply coordinate transformation
                hand_verts_frame = hand_meshes_all.verts_list()[frame_idx]  # (V_hand, 3)
                hand_verts_frame_transformed = hand_verts_frame @ flat_mat  # Apply flip transformation
                hand_verts_list.append(hand_verts_frame_transformed.cpu().numpy())

            # Convert to numpy arrays: (num_frames, num_verts, 3)
            obj_verts_seq = np.stack(obj_verts_list, axis=0)  # (interaction_end_idx, V_obj, 3)
            hand_verts_seq = np.stack(hand_verts_list, axis=0)  # (interaction_end_idx, V_hand, 3)

            # Get faces (same for all frames)
            obj_faces = obj_meshes_all.faces_list()[0].cpu().numpy()  # (F_obj, 3)
            hand_faces = hand_meshes_all.faces_list()[0].cpu().numpy()  # (F_hand, 3)

            # Save to HTML
            save_k3d_visualization(output_path, seq_path, obj_verts_seq, obj_faces, hand_verts_seq, hand_faces, target_fps=30.0)

        print(f"K3D visualization saved for {interaction_end_idx} frames")

        # NOTE check no problem until here #####################################################################################
        # NOTE check no problem until here #####################################################################################
        # NOTE check no problem until here #####################################################################################
        # NOTE check no problem until here #####################################################################################
        # NOTE check no problem until here #####################################################################################
        # NOTE check no problem until here #####################################################################################
        # NOTE check no problem until here #####################################################################################
        # NOTE check no problem until here #####################################################################################
        # NOTE check no problem until here #####################################################################################
        # NOTE check no problem until here #####################################################################################
        # NOTE check no problem until here #####################################################################################
        # NOTE check no problem until here #####################################################################################

        # Step 6.4: Optimize hand poses for frames after anchor frame in 1-2 and all frames in 2-3 interaction segment (2-3)
        # Object poses will automatically follow due to fixed relative transformation
        print(f"\nStep 6.4: Optimizing hand poses for interaction segment {anchor_frame_idx+1}-{interaction_end_idx}")

        # Optimize hand root orientation and translation for each frame
        interaction_frames = list(range(anchor_frame_idx + 1, interaction_end_idx))
        num_interaction_frames = len(interaction_frames)

        # mano_root_orient is axis-angle (N, 3), keep as axis-angle for optimization
        interaction_hand_root_orient = nn.Parameter(multi_frame_model.mano_root_orient[interaction_frames].clone().detach())
        interaction_hand_trans = nn.Parameter(multi_frame_model.mano_trans[interaction_frames].clone().detach())

        # Optimizer
        hand_optimizer = torch.optim.Adam([interaction_hand_root_orient, interaction_hand_trans], lr=1e-3)

        # Setup for batch rendering
        interaction_cameras = PerspectiveCameras(
            focal_length=focal_length[interaction_frames],
            principal_point=principal_point[interaction_frames],
            image_size=((H_out, W_out),) * num_interaction_frames,
            in_ndc=False,
            device=device,
        )

        # Silhouette renderer for object mask loss
        raster_settings_interaction = RasterizationSettings(image_size=(H_out, W_out), blur_radius=1e-4, faces_per_pixel=20)
        silhouette_renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=interaction_cameras, raster_settings=raster_settings_interaction),
            shader=SoftSilhouetteShader(blend_params=BlendParams(sigma=1e-4, gamma=1e-4)),
        )

        # Pre-compute constants for batch computation
        # MANO parameters (constant for all frames in interaction segment)
        mano_pose_batch = multi_frame_model.mano_pose[interaction_frames].detach().clone()  # (N, 45)

        # Object mesh (canonical, will be transformed)
        obj_verts_canonical = multi_frame_model.initial_verts[0].detach().clone()  # (V, 3)
        obj_faces = multi_frame_model.faces[0].detach().clone()  # (F, 3)
        scale_mat = (multi_frame_model.scale * multi_frame_model.initial_scale).unsqueeze(0).detach().clone()  # (1, 3)

        # Transformation matrices for MANO
        flat_mat = torch.diag(torch.tensor([-1., -1., 1.], dtype=torch.float32, device=device))  # (3, 3)
        interaction_is_right = multi_frame_model.is_right[interaction_frames].detach().clone()

        rel_R = rel_R.detach().clone()
        rel_T = rel_T.detach().clone()

        # Pre-compute anchor frame state (anchor_frame_idx) for boundary loss
        print(f"  Pre-computing anchor frame {anchor_frame_idx} state for boundary loss")
        with torch.no_grad():
            # Get anchor frame's hand root orientation and translation
            anchor_hand_root_orient = multi_frame_model.mano_root_orient[anchor_frame_idx].clone().detach()  # (3,)
            anchor_hand_trans = multi_frame_model.mano_trans[anchor_frame_idx].clone().detach()  # (3,)
            anchor_mano_pose = multi_frame_model.mano_pose[anchor_frame_idx].clone().detach()  # (45,)
            anchor_is_right = multi_frame_model.is_right[anchor_frame_idx].clone().detach()

            # Run MANO to get anchor hand joints
            anchor_amano_output = run_amano(multi_frame_model.hand_model_amano, anchor_hand_trans[None, None], anchor_hand_root_orient[None, None],
                                            anchor_mano_pose[None, None],
                                            anchor_is_right.to(device)[None])
            anchor_hand_joints = anchor_amano_output['joints'].squeeze(0).squeeze(0) @ flat_mat  # (21, 3)

            # Build anchor hand coordinate system
            anchor_hand_wrist = anchor_hand_joints[0]  # (3,)
            anchor_hand_forward = anchor_hand_joints[9] - anchor_hand_joints[0]
            anchor_hand_forward = anchor_hand_forward / torch.norm(anchor_hand_forward)
            anchor_hand_side = anchor_hand_joints[5] - anchor_hand_joints[0]
            anchor_hand_side = anchor_hand_side / torch.norm(anchor_hand_side)
            anchor_hand_up = torch.cross(anchor_hand_forward, anchor_hand_side)
            anchor_hand_up = anchor_hand_up / torch.norm(anchor_hand_up)
            anchor_hand_side = torch.cross(anchor_hand_up, anchor_hand_forward)
            anchor_hand_side = anchor_hand_side / torch.norm(anchor_hand_side)
            anchor_hand_R = torch.stack([anchor_hand_side, anchor_hand_up, anchor_hand_forward], dim=1)  # (3, 3)

            # Get anchor object pose
            anchor_obj_R = rotation_6d_to_matrix(multi_frame_model.rot_6d[anchor_frame_idx])  # (3, 3)
            anchor_obj_T = multi_frame_model.trans[anchor_frame_idx]  # (3,)

        num_interaction_steps = 200
        loop = tqdm(range(num_interaction_steps), desc="Optimizing Interaction Segment")

        for step in loop:
            hand_optimizer.zero_grad()

            # Compute hand meshes and joints directly from optimized parameters (batch)
            # Call MANO forward pass
            # Need to unsqueeze to add batch dimension (B=1, T=N, ...)
            amano_output = run_amano(multi_frame_model.hand_model_amano, interaction_hand_trans[None], interaction_hand_root_orient[None],
                                     mano_pose_batch[None], interaction_is_right.to(device))

            # Apply transformation matrix to align with the camera coordinate system
            hand_verts_batch = amano_output['vertices'].squeeze(0) @ flat_mat
            hand_joints_batch = amano_output['joints'].squeeze(0) @ flat_mat

            # Compute object poses (batch, follows hand with fixed relative transformation)
            # FIX: Must use the same geometric hand coordinate system construction as in Step 6.2/6.3
            # instead of using MANO's global root orientation directly.
            hand_wrist = hand_joints_batch[:, 0]  # (N, 3)

            # Forward: Wrist -> Middle (idx 9)
            hand_forward = hand_joints_batch[:, 9] - hand_joints_batch[:, 0]
            hand_forward = torch.nn.functional.normalize(hand_forward, dim=1)

            # Side: Wrist -> Index (idx 5)
            hand_side = hand_joints_batch[:, 5] - hand_joints_batch[:, 0]
            hand_side = torch.nn.functional.normalize(hand_side, dim=1)

            # Up: Cross(Forward, Side)
            hand_up = torch.cross(hand_forward, hand_side, dim=1)
            hand_up = torch.nn.functional.normalize(hand_up, dim=1)

            # Recompute Side
            hand_side = torch.cross(hand_up, hand_forward, dim=1)
            hand_side = torch.nn.functional.normalize(hand_side, dim=1)

            # Stack to get Rotation Matrix (N, 3, 3) - Columns are [side, up, forward]
            hand_R_batch = torch.stack([hand_side, hand_up, hand_forward], dim=2)

            # Compute object pose based on geometric hand frame
            obj_R_batch = hand_R_batch @ rel_R  # (N, 3, 3)
            # obj_T was: hand_R @ rel_T + hand_T (where hand_T is wrist)
            obj_T_batch = (hand_R_batch @ rel_T.unsqueeze(-1)).squeeze(-1) + hand_wrist

            # Transform object vertices (batch)
            # Apply scale, rotation, translation: v' = (v * scale) @ R + T
            obj_verts_scaled = obj_verts_canonical.unsqueeze(0) * scale_mat  # (1, V, 3)
            obj_verts_batch = (obj_verts_scaled @ obj_R_batch) + obj_T_batch.unsqueeze(1)  # (1, V, 3) @ (N, 3, 3) + (N, 1, 3) -> (N, V, 3)

            # Create batch meshes for rendering
            obj_verts_list = [obj_verts_batch[i] for i in range(num_interaction_frames)]
            obj_faces_list = [obj_faces] * num_interaction_frames
            obj_meshes_batch = Meshes(
                verts=obj_verts_list,
                faces=obj_faces_list,
                textures=TexturesVertex(verts_features=torch.ones(num_interaction_frames, obj_verts_canonical.shape[0], 3, device=device)))

            # Loss 1: Object 2D mask loss
            obj_fragments = silhouette_renderer.rasterizer(obj_meshes_batch, cameras=interaction_cameras)
            rendered_obj_masks = silhouette_renderer.shader(obj_fragments, obj_meshes_batch, cameras=interaction_cameras)[..., 3]
            # obj_mask_loss = torch.nn.functional.mse_loss(rendered_obj_masks, gt_obj_masks) * 1e3
            loss_fp = weighted_false_positive_loss(rendered_obj_masks, sampled_pred_amodal_masks[interaction_frames]) * 1e2
            loss_vp = vectorized_pose_guiding_loss(obj_meshes_batch,
                                                   sampled_pred_amodal_masks[interaction_frames],
                                                   rendered_obj_masks,
                                                   interaction_cameras,
                                                   num_samples=2000)
            obj_mask_loss = loss_fp + loss_vp

            # Loss 2: Hand 2D keypoints loss
            projected_hand_joints = interaction_cameras.transform_points_screen(hand_joints_batch, image_size=((H_out, W_out),))[..., :2]
            gt_hand_joints_2d = sampled_gt_hand_joints_2d[interaction_frames]
            hand_2d_loss = torch.nn.functional.mse_loss(projected_hand_joints, gt_hand_joints_2d)

            # Loss 3: Smoothness losses
            # Rotation smoothness (on hand root orient)
            rot_diff = interaction_hand_root_orient[1:] - interaction_hand_root_orient[:-1]
            loss_sm_rot = (rot_diff**2).mean() * 1e3

            # Translation smoothness (on hand trans)
            trans_diff = interaction_hand_trans[1:] - interaction_hand_trans[:-1]
            loss_sm_trans = (trans_diff**2).mean() * 1e4

            # Loss 4: Boundary loss (first frame should match anchor frame)
            # Compute hand pose difference for first frame (index 0)
            first_frame_hand_root_orient = interaction_hand_root_orient[0]  # (3,)
            first_frame_hand_trans = interaction_hand_trans[0]  # (3,)

            # Hand root orientation loss (axis-angle difference)
            hand_orient_diff = first_frame_hand_root_orient - anchor_hand_root_orient
            loss_boundary_hand_orient = (hand_orient_diff**2).mean() * 1e3

            # Hand translation loss
            hand_trans_diff = first_frame_hand_trans - anchor_hand_trans
            loss_boundary_hand_trans = (hand_trans_diff**2).mean() * 1e4

            # Object pose loss (computed from first frame's hand frame)
            first_frame_hand_wrist = hand_wrist[0]  # (3,)
            first_frame_hand_R = hand_R_batch[0]  # (3, 3)
            first_frame_obj_R = obj_R_batch[0]  # (3, 3)
            first_frame_obj_T = obj_T_batch[0]  # (3,)

            # Rotation difference (Frobenius norm of difference matrix)
            obj_R_diff = first_frame_obj_R - anchor_obj_R
            loss_boundary_obj_rot = (obj_R_diff**2).mean() * 1e3

            # Translation difference
            obj_T_diff = first_frame_obj_T - anchor_obj_T
            loss_boundary_obj_trans = (obj_T_diff**2).mean() * 1e4

            # Total boundary loss
            loss_boundary = loss_boundary_hand_orient + loss_boundary_hand_trans + loss_boundary_obj_rot + loss_boundary_obj_trans

            # Total loss
            total_loss = obj_mask_loss + hand_2d_loss + loss_sm_rot + loss_sm_trans + loss_boundary

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_([interaction_hand_root_orient, interaction_hand_trans], max_norm=1.0)
            hand_optimizer.step()

            loop.set_postfix(total=total_loss.item(),
                             obj_mask=obj_mask_loss.item(),
                             hand_2d=hand_2d_loss.item(),
                             sm_rot=loss_sm_rot.item(),
                             sm_trans=loss_sm_trans.item(),
                             boundary=loss_boundary.item())

        # Step 6.5: Save optimized results back to multi_frame_model (batch)
        print(f"\nStep 6.5: Saving optimized interaction segment results")

        with torch.no_grad():
            # Re-compute final poses using the same geometric logic
            # Run MANO one last time with optimized parameters
            amano_output = run_amano(multi_frame_model.hand_model_amano, interaction_hand_trans[None], interaction_hand_root_orient[None],
                                     mano_pose_batch[None], interaction_is_right.to(device))
            hand_joints_batch = amano_output['joints'].squeeze(0) @ flat_mat

            # Build geometric hand frame
            hand_wrist = hand_joints_batch[:, 0]
            hand_forward = torch.nn.functional.normalize(hand_joints_batch[:, 9] - hand_joints_batch[:, 0], dim=1)
            hand_side = torch.nn.functional.normalize(hand_joints_batch[:, 5] - hand_joints_batch[:, 0], dim=1)
            hand_up = torch.nn.functional.normalize(torch.cross(hand_forward, hand_side, dim=1), dim=1)
            hand_side = torch.nn.functional.normalize(torch.cross(hand_up, hand_forward, dim=1), dim=1)
            hand_R_batch = torch.stack([hand_side, hand_up, hand_forward], dim=2)

            # Compute object poses
            obj_R_batch = hand_R_batch @ rel_R
            obj_T_batch = (hand_R_batch @ rel_T.unsqueeze(-1)).squeeze(-1) + hand_wrist

            # Batch update to model (using index assignment)
            multi_frame_model.mano_root_orient.data[interaction_frames] = interaction_hand_root_orient.data
            multi_frame_model.mano_trans.data[interaction_frames] = interaction_hand_trans.data
            multi_frame_model.rot_6d.data[interaction_frames] = matrix_to_rotation_6d(obj_R_batch)
            multi_frame_model.trans.data[interaction_frames] = obj_T_batch

        print(f"STAGE 6 completed: Interaction segment optimized")
    else:
        print("\nSkipping STAGE 6: No contact correspondences available")

    # ========================================================================
    # STAGE 7: Optimize 4-5 segments only, then propagate results
    #          - Optimize on last frame (4-5 segment)
    #          - Apply 4-5 result to entire 3-4 and 4-5 segments
    # ========================================================================
    if parsed_contact_map is not None and parsed_contact_map['num_contacts'] > 0:
        print("\n" + "=" * 80)
        print("STAGE 7: Optimize 4-5 segments only, propagate to 3-4 segments")
        print("=" * 80)

        # Step 7.1: Initialize both ends
        print(f"\nStep 7.1: Initializing 4-5 segments with interaction boundary poses")

        with torch.no_grad():
            # Initialize 4-5 segment (and 3-4 for now) with interaction ending pose
            interaction_end_frame_idx = interaction_end_idx - 1
            obj_R_end_init = rotation_6d_to_matrix(multi_frame_model.rot_6d[interaction_end_frame_idx])
            obj_T_end_init = multi_frame_model.trans[interaction_end_frame_idx]

            for frame_idx in range(interaction_end_idx, num_sampled_frames):
                multi_frame_model.rot_6d.data[frame_idx] = matrix_to_rotation_6d(obj_R_end_init.unsqueeze(0)).squeeze(0)
                multi_frame_model.trans.data[frame_idx] = obj_T_end_init

            print(f"  Initialized {num_sampled_frames - interaction_end_idx} frames (3-4 and 4-5) with interaction ending pose")

        # Step 7.2: Setup optimization for first and last frames only
        print(f"\nStep 7.2: Setting up optimization (last frame for 4-5, shared scale)")

        # Shared scale parameter
        shared_scale = nn.Parameter(multi_frame_model.scale.detach().clone(), requires_grad=True)
        shared_initial_scale = multi_frame_model.initial_scale

        # Back end pose parameters (optimize on last frame, will apply to 3-4 and 4-5)
        back_rot_6d = nn.Parameter(matrix_to_rotation_6d(obj_R_end_init), requires_grad=True)
        back_trans = nn.Parameter(obj_T_end_init.clone(), requires_grad=True)

        # Joint optimizer
        joint_optimizer = torch.optim.Adam([{'params': back_rot_6d, 'lr': 1e-3}, {'params': back_trans, 'lr': 1e-3}])

        # Prepare cameras and masks
        last_frame_idx = num_sampled_frames - 1

        last_camera = PerspectiveCameras(
            focal_length=focal_length[last_frame_idx, None],
            principal_point=principal_point[last_frame_idx, None],
            image_size=((H_out, W_out),),
            in_ndc=False,
            device=device,
        )

        target_mask_last = torch.tensor(sampled_pred_amodal_masks_np[last_frame_idx], dtype=torch.float32, device=device)

        # Prepare metric depth for last frame
        # sampled_metric_depths_np shape: (N, H, W, 3)
        metric_depth_last = torch.from_numpy(sampled_metric_depths_np[last_frame_idx, ..., 0]).float().to(device)  # (H, W)

        # Renderer
        raster_settings = RasterizationSettings(image_size=(H_out, W_out), blur_radius=1e-4, faces_per_pixel=20)
        renderer = MeshRenderer(rasterizer=MeshRasterizer(raster_settings=raster_settings), shader=SoftSilhouetteShader())

        # Step 7.3: Optimize only on last frame (4-5)
        print(f"\nStep 7.3: Optimizing last frame (4-5) with shared scale")

        # Save initial state for visualization
        with torch.no_grad():
            # Setup RGB renderer for visualization
            lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])
            rgb_raster_settings = RasterizationSettings(image_size=(H_out, W_out), blur_radius=0.0, faces_per_pixel=1)

            # --- Last Frame Initial ---
            back_R_init = rotation_6d_to_matrix(back_rot_6d)
            back_scale_full_init = shared_scale * shared_initial_scale
            back_transform_init = Transform3d(dtype=torch.float32,
                                              device=device).scale(back_scale_full_init.unsqueeze(0)).rotate(back_R_init.unsqueeze(0)).translate(
                                                  back_trans.unsqueeze(0))
            back_verts_init = back_transform_init.transform_points(verts.unsqueeze(0))
            back_mesh_init = Meshes(verts=back_verts_init, faces=faces.unsqueeze(0), textures=TexturesVertex(verts_features=torch.ones_like(back_verts_init)))

            back_fragments_init = renderer.rasterizer(back_mesh_init, cameras=last_camera)
            back_rendered_init = renderer.shader(back_fragments_init, back_mesh_init, cameras=last_camera)[0, ..., 3].cpu().numpy()

            # Render RGB
            rgb_textures_back_init = TexturesVertex(verts_features=torch.ones_like(back_verts_init) * torch.tensor([0.7, 0.7, 1.0], device=device))
            rgb_mesh_back_init = Meshes(verts=back_verts_init, faces=faces.unsqueeze(0), textures=rgb_textures_back_init)
            rgb_renderer_back_init = MeshRenderer(rasterizer=MeshRasterizer(cameras=last_camera, raster_settings=rgb_raster_settings),
                                                  shader=HardPhongShader(device=device, cameras=last_camera, lights=lights))
            rendered_rgb_back_init = rgb_renderer_back_init(rgb_mesh_back_init).cpu().numpy()[0]

            rgb_render_b_init = rendered_rgb_back_init[..., :3]
            alpha_render_b_init = rendered_rgb_back_init[..., 3:4]
            rgb_overlay_back_init = sampled_rgbs_np[last_frame_idx] * (1 - alpha_render_b_init) + rgb_render_b_init * alpha_render_b_init

            modal_back_init = overlay_mask_on_image(sampled_rgbs_np[last_frame_idx], sampled_modal_masks_np[last_frame_idx])
            amodal_back_init = overlay_mask_on_image(sampled_rgbs_np[last_frame_idx], sampled_pred_amodal_masks_np[last_frame_idx])
            render_back_init = overlay_mask_on_image(sampled_rgbs_np[last_frame_idx], (back_rendered_init > 0.5).astype(np.uint8))

            combined_back_init = (np.hstack([modal_back_init, amodal_back_init, render_back_init, rgb_overlay_back_init]) * 255).astype(np.uint8)
            imageio.imwrite(os.path.join(output_path, f"stage7_initial_back_frame_{last_frame_idx:04d}.png"), combined_back_init)

        num_steps = 300
        loop = tqdm(range(num_steps), desc="Optimizing last frame (4-5)")

        for step in loop:
            joint_optimizer.zero_grad()

            # Forward pass for last frame (4-5)
            back_R = rotation_6d_to_matrix(back_rot_6d)
            back_scale_full = shared_scale * shared_initial_scale
            back_transform = Transform3d(dtype=torch.float32,
                                         device=device).scale(back_scale_full.unsqueeze(0)).rotate(back_R.unsqueeze(0)).translate(back_trans.unsqueeze(0))
            back_verts = back_transform.transform_points(verts.unsqueeze(0))
            back_mesh = Meshes(verts=back_verts, faces=faces.unsqueeze(0), textures=TexturesVertex(verts_features=torch.ones_like(back_verts)))

            back_fragments = renderer.rasterizer(back_mesh, cameras=last_camera)
            back_rendered = renderer.shader(back_fragments, back_mesh, cameras=last_camera)[0, ..., 3]
            back_mse_loss = torch.nn.functional.mse_loss(back_rendered, target_mask_last) * 1e3

            # back_vp_loss = vectorized_pose_guiding_loss(back_mesh, target_mask_last.unsqueeze(0), back_rendered.unsqueeze(0), last_camera, num_samples=2000)
            # back_fp_loss = weighted_false_positive_loss(back_rendered, target_mask_last) * 1e2

            # # Metric depth loss for back
            # back_depth = back_fragments.zbuf[0, ..., 0]  # (H, W)
            # back_depth_loss_mask = (target_mask_last > 0.5) & (back_rendered > 0.5) & (back_depth > 0)

            # back_depth_loss = torch.tensor(0.0, device=device)
            # if back_depth_loss_mask.sum() > 0:
            #     back_depth_loss = torch.nn.functional.l1_loss(back_depth[back_depth_loss_mask], metric_depth_last[back_depth_loss_mask]) * 1e2

            # Total loss
            # total_loss = back_vp_loss + back_fp_loss
            total_loss = back_mse_loss
            total_loss.backward()
            joint_optimizer.step()

            loop.set_postfix(
                total=total_loss.item(),
                # loss_vp=back_vp_loss.item(),
                # loss_fp=back_fp_loss.item(),
                loss_mse=back_mse_loss.item(),
                # b_depth=back_depth_loss.item()
            )

        # ICP alignment
        # Forward pass for back end (last frame only, representing 4-5 segment)
        back_R = rotation_6d_to_matrix(back_rot_6d)
        back_scale_full = shared_scale * shared_initial_scale
        back_transform = Transform3d(dtype=torch.float32,
                                     device=device).scale(back_scale_full.unsqueeze(0)).rotate(back_R.unsqueeze(0)).translate(back_trans.unsqueeze(0))
        back_verts = back_transform.transform_points(verts.unsqueeze(0))
        back_mesh = Meshes(verts=back_verts, faces=faces.unsqueeze(0), textures=TexturesVertex(verts_features=torch.ones_like(back_verts)))

        back_fragments = renderer.rasterizer(back_mesh, cameras=last_camera)
        back_rendered = renderer.shader(back_fragments, back_mesh, cameras=last_camera)[0, ..., 3]

        # Metric depth loss for back
        back_depth = back_fragments.zbuf[0, ..., 0]  # (H, W)

        back_target_points = get_rgbd_point_cloud(last_camera, sampled_rgbs[last_frame_idx].unsqueeze(0).permute(0, 3, 1, 2),
                                                  metric_depth_last.unsqueeze(0).unsqueeze(-1).permute(0, 3, 1, 2),
                                                  target_mask_last.unsqueeze(0).unsqueeze(-1).permute(0, 3, 1, 2), 0.5)
        back_target_points = back_target_points.points_packed()
        back_target_points = back_target_points.squeeze(0)  # (N, 3)

        back_source_points = get_rgbd_point_cloud(last_camera, sampled_rgbs[last_frame_idx].unsqueeze(0).permute(0, 3, 1, 2),
                                                  back_depth.unsqueeze(0).unsqueeze(-1).permute(0, 3, 1, 2),
                                                  back_rendered.unsqueeze(0).unsqueeze(-1).permute(0, 3, 1, 2), 0.5)
        back_source_points = back_source_points.points_packed()
        back_source_points = back_source_points.squeeze(0)  # (N, 3)

        if back_target_points is not None and back_target_points.shape[0] > 10:
            # Save debug point cloud

            # Combine points: Target (Metric Depth) + Source (Rendered)
            # Target = Green, Source = Red
            combined_points = torch.cat([back_target_points, back_source_points], dim=0).detach().cpu().numpy()

            target_colors = torch.tensor([[0.0, 1.0, 0.0]], device=device).repeat(back_target_points.shape[0], 1)
            source_colors = torch.tensor([[1.0, 0.0, 0.0]], device=device).repeat(back_source_points.shape[0], 1)
            combined_colors = (torch.cat([target_colors, source_colors], dim=0).cpu().numpy() * 255).astype(np.uint8)

            # Create Trimesh PointCloud
            pcd = trimesh.points.PointCloud(combined_points, colors=combined_colors)

            debug_ply_path = os.path.join(output_path, f"stage7_debug_icp_pre_alignment_{last_frame_idx}.ply")
            pcd.export(debug_ply_path)
            print(f"Saved ICP debug point cloud to {debug_ply_path}")

            # ICP needs batch dimension
            back_icp_result = iterative_closest_point(
                back_source_points.unsqueeze(0),  # X
                back_target_points.unsqueeze(0),  # Y
                init_transform=None,  # init transform
                max_iterations=50,
                relative_rmse_thr=1e-6,
                estimate_scale=False,  # allow ICP to change scale?
                allow_reflection=False,
            )
            # result.RT 是一个 (1, 4, 4) 矩阵，表示将 Source 对齐到 Target 的变换
            # P_target ~ R_icp * P_source + T_icp
            icp_transform = back_icp_result.RTs
            R_delta = icp_transform[0]  # (1, 3, 3)
            T_delta = icp_transform[1]  # (1, 3) translation part usually matches points translation

            # Combine points: Target (Metric Depth) + Source (Rendered)
            # Target = Green, Source = Red

            aligned_points = (back_source_points @ R_delta[0]) + T_delta
            aliged_mesh_verts = back_verts[0] @ R_delta[0] + T_delta
            combined_points = torch.cat([back_target_points, aligned_points, aliged_mesh_verts], dim=0).detach().cpu().numpy()

            target_colors = torch.tensor([[0.0, 1.0, 0.0]], device=device).repeat(back_target_points.shape[0], 1)
            source_colors = torch.tensor([[1.0, 0.0, 0.0]], device=device).repeat(back_source_points.shape[0], 1)
            aliged_mesh_colors = torch.tensor([[0.0, 0.0, 1.0]], device=device).repeat(aliged_mesh_verts.shape[0], 1)
            combined_colors = (torch.cat([target_colors, source_colors, aliged_mesh_colors], dim=0).cpu().numpy() * 255).astype(np.uint8)

            # Create Trimesh PointCloud
            pcd = trimesh.points.PointCloud(combined_points, colors=combined_colors)

            debug_ply_path = os.path.join(output_path, f"stage7_debug_icp_post_alignment_{last_frame_idx}.ply")
            pcd.export(debug_ply_path)
            print(f"Saved ICP debug point cloud to {debug_ply_path}")

            # PyTorch3D ICP 返回的 RT
            new_R_matrix = back_R @ R_delta.squeeze(0)
            new_T_vector = (back_trans.unsqueeze(0) @ R_delta.squeeze(0)).squeeze(0) + T_delta

            # Check IoU before applying ICP: render mask with new transform and compare with amodal mask
            with torch.no_grad():
                # Temporarily apply the new transform to test IoU
                temp_scale_full = shared_scale * shared_initial_scale
                temp_transform = Transform3d(dtype=torch.float32,
                                             device=device).scale(temp_scale_full.unsqueeze(0)).rotate(new_R_matrix.unsqueeze(0)).translate(new_T_vector)
                temp_verts = temp_transform.transform_points(verts.unsqueeze(0))
                temp_mesh = Meshes(verts=temp_verts, faces=faces.unsqueeze(0), textures=TexturesVertex(verts_features=torch.ones_like(temp_verts)))

                # Render with new transform
                temp_fragments = renderer.rasterizer(temp_mesh, cameras=last_camera)
                temp_rendered_mask = renderer.shader(temp_fragments, temp_mesh, cameras=last_camera)[0, ..., 3]  # (H, W)

                # Get amodal mask for last frame
                amodal_mask_last = sampled_pred_amodal_masks[last_frame_idx]  # (H, W)

                # Compute IoU
                rendered_binary = (temp_rendered_mask > 0.5).float()
                amodal_binary = (amodal_mask_last > 0.5).float()
                intersection = (rendered_binary * amodal_binary).sum()
                union = (rendered_binary + amodal_binary).clamp(0, 1).sum()
                iou = (intersection / union) if union > 0 else torch.tensor(0.0, device=device)
                iou_value = iou.item()

            # Only apply ICP if IoU >= 0.8
            if iou_value >= 0.6:
                # assign back to parameter
                back_rot_6d.data = matrix_to_rotation_6d(new_R_matrix).squeeze(0)
                back_trans.data = new_T_vector.squeeze(0)

                print("ICP Alignment Applied to Back Frame!")
            else:
                print(f" Warning: ICP alignment rejected due to low IoU ({iou_value:.4f} < 0.8). Keeping original pose.")

        # Step 7.4: Propagate optimized results to adjacent segments
        print(f"\nStep 7.4: Propagating optimized poses to segments")

        with torch.no_grad():
            # Propagate 4-5 optimized pose to entire 3-4 and 4-5 segments
            opt_R_back = rotation_6d_to_matrix(back_rot_6d)
            for frame_idx in range(interaction_end_idx, num_sampled_frames):
                multi_frame_model.rot_6d.data[frame_idx] = matrix_to_rotation_6d(opt_R_back.unsqueeze(0)).squeeze(0)
                multi_frame_model.trans.data[frame_idx] = back_trans

            print(f"  4-5 optimized pose → {num_sampled_frames - interaction_end_idx} frames (3-4 and 4-5 segments)")
            print(f"  Shared scale: {shared_scale.item():.6f} (applied to all frames)")

        # Save visualization
        with torch.no_grad():
            # Re-render with current pose (which may have been updated by ICP)
            back_R_final = rotation_6d_to_matrix(back_rot_6d)
            back_scale_full_final = shared_scale * shared_initial_scale
            back_transform_final = Transform3d(dtype=torch.float32,
                                               device=device).scale(back_scale_full_final.unsqueeze(0)).rotate(back_R_final.unsqueeze(0)).translate(
                                                   back_trans.unsqueeze(0))
            back_verts_final = back_transform_final.transform_points(verts.unsqueeze(0))
            back_mesh_final = Meshes(verts=back_verts_final,
                                     faces=faces.unsqueeze(0),
                                     textures=TexturesVertex(verts_features=torch.ones_like(back_verts_final)))

            # Re-render mask with final pose
            back_fragments_final = renderer.rasterizer(back_mesh_final, cameras=last_camera)
            back_rendered_final = renderer.shader(back_fragments_final, back_mesh_final, cameras=last_camera)[0, ..., 3]

            # Setup RGB renderer for visualization
            lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])
            rgb_raster_settings = RasterizationSettings(image_size=(H_out, W_out), blur_radius=0.0, faces_per_pixel=1)

            # Visualize last frame
            back_mask_final = back_rendered_final.cpu().numpy()
            modal_back = overlay_mask_on_image(sampled_rgbs_np[last_frame_idx], sampled_modal_masks_np[last_frame_idx])
            amodal_back = overlay_mask_on_image(sampled_rgbs_np[last_frame_idx], sampled_pred_amodal_masks_np[last_frame_idx])
            render_back = overlay_mask_on_image(sampled_rgbs_np[last_frame_idx], (back_mask_final > 0.5).astype(np.uint8))

            # Render RGB for back frame
            rgb_textures_back = TexturesVertex(verts_features=torch.ones_like(back_verts_final) * torch.tensor([0.7, 0.7, 1.0], device=device))
            rgb_mesh_back = Meshes(verts=back_verts_final, faces=faces.unsqueeze(0), textures=rgb_textures_back)
            rgb_renderer_back = MeshRenderer(rasterizer=MeshRasterizer(cameras=last_camera, raster_settings=rgb_raster_settings),
                                             shader=HardPhongShader(device=device, cameras=last_camera, lights=lights))
            rendered_rgb_back = rgb_renderer_back(rgb_mesh_back).cpu().numpy()[0]  # (H, W, 4)

            rgb_render_b = rendered_rgb_back[..., :3]
            alpha_render_b = rendered_rgb_back[..., 3:4]
            rgb_overlay_back = sampled_rgbs_np[last_frame_idx] * (1 - alpha_render_b) + rgb_render_b * alpha_render_b

            combined_back = (np.hstack([modal_back, amodal_back, render_back, rgb_overlay_back]) * 255).astype(np.uint8)
            imageio.imwrite(os.path.join(output_path, f"stage7_back_frame_{last_frame_idx:04d}.png"), combined_back)

        print(f"\nSTAGE 7 completed: Optimized 0-1 and 4-5, propagated to adjacent segments")
    else:
        print("\nSkipping STAGE 7: No contact correspondences available")

    # ========================================================================
    # STAGE 8: Full Sequence Refinement (0-5)
    # ========================================================================
    if parsed_contact_map is not None and parsed_contact_map['num_contacts'] > 0:
        print("\n" + "=" * 80)
        print("STAGE 8: Full Sequence Refinement with Frozen Object Poses")
        print("=" * 80)

        # Prepare object mesh data
        obj_verts_canonical = multi_frame_model.initial_verts[0].detach()  # (V, 3)
        obj_faces = multi_frame_model.faces[0].detach()  # (F, 3)
        flat_mat = torch.diag(torch.tensor([-1., -1., 1.], dtype=torch.float32, device=device))  # (3, 3)

        # Step 8.1: Setup - define segments and parameters
        print(f"\nStep 8.1: Setting up full sequence optimization")
        print(f"  Segments:")
        print(f"    0-1 (start_static):  [0:{start_static_end_idx}]")
        print(f"    1-anchor_frame_idx (approaching):   [{start_static_end_idx}:{anchor_frame_idx}]")
        print(f"    anchor_frame_idx-3 (interaction):   [{anchor_frame_idx}:{interaction_end_idx}]")
        print(f"    3-4 (releasing):     [{interaction_end_idx}:{end_static_start_idx}]")
        print(f"    4-5 (end_static):    [{end_static_start_idx}:{num_sampled_frames}]")

        # Compute hand-object relative transforms for anchor_frame_idx-3 segment
        print(f"\nStep 8.1.1: Computing hand-object relative transforms for interaction segment")
        with torch.no_grad():
            interaction_frames_list = list(range(anchor_frame_idx, interaction_end_idx))
            num_interaction_frames = len(interaction_frames_list)

            # Get all hand joints from current model state
            _, _, hand_joints_all = multi_frame_model()

            # Since Stage 6 already ensured all frames in 2-3 share the same hand-object relative transform,
            # we only need to compute and store one relative transform (using the first frame as reference)
            reference_frame_idx = interaction_frames_list[0]

            # Get reference frame's hand joints
            hand_joints_ref = hand_joints_all[reference_frame_idx]  # (21, 3)
            hand_wrist_ref = hand_joints_ref[0]  # (3,)

            # Build reference frame's hand coordinate system (based on joint geometry)
            # Use wrist to middle finger base as forward direction
            hand_forward_ref = hand_joints_ref[9] - hand_joints_ref[0]  # middle finger MCP - wrist
            hand_forward_ref = hand_forward_ref / torch.norm(hand_forward_ref)

            # Use wrist to index finger base as side direction
            hand_side_ref = hand_joints_ref[5] - hand_joints_ref[0]  # index finger MCP - wrist
            hand_side_ref = hand_side_ref / torch.norm(hand_side_ref)

            # Compute hand up direction (perpendicular to forward and side)
            hand_up_ref = torch.cross(hand_forward_ref, hand_side_ref)
            hand_up_ref = hand_up_ref / torch.norm(hand_up_ref)

            # Recompute side to ensure orthogonality
            hand_side_ref = torch.cross(hand_up_ref, hand_forward_ref)
            hand_side_ref = hand_side_ref / torch.norm(hand_side_ref)

            # Build hand rotation matrix (columns are the local axes)
            hand_R_ref = torch.stack([hand_side_ref, hand_up_ref, hand_forward_ref], dim=1)  # (3, 3)
            hand_T_ref = hand_wrist_ref  # (3,)

            # Get reference object pose
            obj_R_ref = rotation_6d_to_matrix(multi_frame_model.rot_6d[reference_frame_idx])
            obj_T_ref = multi_frame_model.trans[reference_frame_idx]

            # Compute relative transform: T_rel = T_hand^{-1} * T_obj
            # R_rel = hand_R^T @ obj_R
            # T_rel = hand_R^T @ (obj_T - hand_T)
            relative_obj_R = (hand_R_ref.T @ obj_R_ref).detach()  # (3, 3)
            relative_obj_T = (hand_R_ref.T @ (obj_T_ref - hand_T_ref)).detach()  # (3,)

            print(f"  Stored single relative transform from frame {reference_frame_idx} (shared across {num_interaction_frames} interaction frames)")

        # Step 8.2: Create optimizable parameters
        print(f"\nStep 8.2: Creating optimizable parameters")

        # For segments 0-anchor_frame_idx, 3-5: optimize full MANO parameters
        static_transition_frames = (
            list(range(0, start_static_end_idx)) +  # 0-1
            list(range(start_static_end_idx, anchor_frame_idx)) +  # 1-anchor_frame_idx
            list(range(interaction_end_idx, end_static_start_idx)) +  # 3-4
            list(range(end_static_start_idx, num_sampled_frames))  # 4-5
        )
        num_static_transition = len(static_transition_frames)

        # Use multi_frame_model parameters directly, lock gradients with hooks
        # For hand parameters:
        # - static+transition segments: optimize root_orient, trans, pose (all frames)
        # - interaction segment: optimize root_orient, trans only (pose is frozen)
        # For object poses:
        # - static+transition segments: locked (0-anchor_frame_idx and interaction_end_idx-5)
        # - interaction segment: computed from hand pose (NOT optimized directly, gradients flow to hand)
        #   Object pose in interaction segment is calculated from hand coordinate frame and relative transform.
        #   Object loss gradients flow to hand root_orient and trans, not to object pose itself.

        # Define frames to lock
        # Object pose: lock ALL frames (object pose is either frozen or computed from hand)
        # - static+transition: frozen (locked)
        # - interaction: computed from hand (should not be optimized, gradients flow to hand)
        locked_obj_frames = list(range(num_sampled_frames))  # Lock all object pose frames

        # Hand pose: lock interaction segment (only root_orient and trans are optimized)
        locked_hand_pose_frames = interaction_frames_list

        print(f"  Static+Transition segments: {num_static_transition} frames (full MANO optimized)")
        print(f"  Interaction segment: {num_interaction_frames} frames (root orient+trans optimized, pose frozen)")
        print(f"  Object pose: all frames locked (static+transition frozen, interaction computed from hand)")
        print(f"  Hand pose locked frames: {locked_hand_pose_frames} (gradients will be zeroed)")

        # Step 8.3: Setup hooks to lock gradients
        print(f"\nStep 8.3: Setting up hooks to lock gradients")

        # Lock ALL object pose gradients (both static+transition and interaction)
        # Interaction object pose is computed from hand, so gradients should flow to hand, not object pose
        obj_rot_lock_hook = create_lock_frames_hook(locked_obj_frames)
        obj_trans_lock_hook = create_lock_frames_hook(locked_obj_frames)
        multi_frame_model.rot_6d.register_hook(obj_rot_lock_hook)
        multi_frame_model.trans.register_hook(obj_trans_lock_hook)

        # Lock hand pose gradients for interaction segment
        hand_pose_lock_hook = create_lock_frames_hook(locked_hand_pose_frames)
        multi_frame_model.mano_pose.register_hook(hand_pose_lock_hook)

        print(f"  Registered hooks to lock object pose gradients for all {len(locked_obj_frames)} frames")
        print(f"  Registered hooks to lock hand pose gradients for {len(locked_hand_pose_frames)} frames")

        # Step 8.4: Setup optimizer
        print(f"\nStep 8.4: Setting up optimizer")
        print(f"  Optimizing multi_frame_model parameters directly (hooks will lock specific frames)")

        optimizer = torch.optim.Adam([{
            'params': [
                multi_frame_model.mano_root_orient,  # All frames optimized (static+transition and interaction)
                multi_frame_model.mano_trans,  # All frames optimized (static+transition and interaction)
                multi_frame_model.mano_pose,  # Only static+transition frames optimized (interaction locked by hook)
                # Note: object pose (rot_6d, trans) NOT in optimizer
                # - static+transition: frozen (locked by hook)
                # - interaction: computed from hand pose, gradients flow to hand root_orient/trans
            ],
            'lr': 1e-3
        }])

        # Step 8.5: Prepare rendering components
        print(f"\nStep 8.4: Preparing cameras and renderer")

        all_cameras = PerspectiveCameras(
            focal_length=focal_length,
            principal_point=principal_point,
            image_size=((H_out, W_out),) * num_sampled_frames,
            in_ndc=False,
            device=device,
        )

        raster_settings_full = RasterizationSettings(
            image_size=(H_out, W_out),
            blur_radius=1e-4,
            faces_per_pixel=20,
        )
        silhouette_renderer_full = MeshRenderer(
            rasterizer=MeshRasterizer(raster_settings=raster_settings_full),
            shader=SoftSilhouetteShader(blend_params=BlendParams(sigma=1e-4, gamma=1e-4)),
        )

        # Get fixed data
        all_is_right = multi_frame_model.is_right.clone().detach()
        scale_mat = (multi_frame_model.scale * multi_frame_model.initial_scale).detach()  # Frozen scale

        # Pre-compute canonical object normals
        obj_mesh_canonical = Meshes(verts=[obj_verts_canonical],
                                    faces=[obj_faces],
                                    textures=TexturesVertex(verts_features=torch.ones(1, obj_verts_canonical.shape[0], 3, device=device)))
        obj_normals_canonical = obj_mesh_canonical.verts_normals_packed()  # (V, 3)

        # Step 8.6: Optimization loop
        print(f"\nStep 8.5: Optimizing full sequence")

        num_steps_full = 1000
        loop_full = tqdm(range(num_steps_full), desc="Full Sequence Refinement")

        # Learning rate scheduler: reduce lr when collision loss starts
        collision_start_step = 1000
        lr_reduced = False

        for step in loop_full:
            optimizer.zero_grad()

            # ===== Forward pass for static+transition segments (0-1, 1-2, 3-4, 4-5) =====
            amano_output_st = run_amano(multi_frame_model.hand_model_amano, multi_frame_model.mano_trans[static_transition_frames][None],
                                        multi_frame_model.mano_root_orient[static_transition_frames][None],
                                        multi_frame_model.mano_pose[static_transition_frames][None], all_is_right[static_transition_frames].to(device))
            st_hand_joints = amano_output_st['joints'].squeeze(0) @ flat_mat  # (N_st, 21, 3)
            st_hand_verts = amano_output_st['vertices'].squeeze(0) @ flat_mat  # (N_st, 778, 3)

            # Object verts for static+transition (use multi_frame_model directly, gradients locked by hook)
            st_obj_R = rotation_6d_to_matrix(multi_frame_model.rot_6d[static_transition_frames])  # (N_st, 3, 3)
            st_obj_verts_scaled = obj_verts_canonical.unsqueeze(0) * scale_mat  # (1, V, 3)
            st_obj_verts = (st_obj_verts_scaled @ st_obj_R) + multi_frame_model.trans[static_transition_frames].unsqueeze(1)  # (N_st, V, 3)
            # Transform normals: n' = n @ R.T (normals are covariant vectors)
            st_obj_normals = obj_normals_canonical.unsqueeze(0) @ st_obj_R.transpose(-2, -1)  # (1, V, 3) @ (N_st, 3, 3) -> (N_st, V, 3)

            # ===== Forward pass for interaction segment (2-3) =====
            # Use multi_frame_model directly (mano_pose is frozen for interaction segment via hook)
            amano_output_int = run_amano(multi_frame_model.hand_model_amano, multi_frame_model.mano_trans[interaction_frames_list][None],
                                         multi_frame_model.mano_root_orient[interaction_frames_list][None],
                                         multi_frame_model.mano_pose[interaction_frames_list][None], all_is_right[interaction_frames_list].to(device))
            int_hand_joints = amano_output_int['joints'].squeeze(0) @ flat_mat  # (N_int, 21, 3)
            int_hand_verts = amano_output_int['vertices'].squeeze(0) @ flat_mat  # (N_int, 778, 3)

            # Object follows hand for interaction segment
            # Build geometric hand coordinate frame for each interaction frame
            hand_wrist_int = int_hand_joints[:, 0]  # (N_int, 3)

            # Forward: Wrist -> Middle finger MCP (idx 9)
            hand_forward_int = int_hand_joints[:, 9] - int_hand_joints[:, 0]
            hand_forward_int = torch.nn.functional.normalize(hand_forward_int, dim=1)

            # Side: Wrist -> Index finger MCP (idx 5)
            hand_side_int = int_hand_joints[:, 5] - int_hand_joints[:, 0]
            hand_side_int = torch.nn.functional.normalize(hand_side_int, dim=1)

            # Up: Cross(Forward, Side)
            hand_up_int = torch.cross(hand_forward_int, hand_side_int, dim=1)
            hand_up_int = torch.nn.functional.normalize(hand_up_int, dim=1)

            # Recompute Side to ensure orthogonality
            hand_side_int = torch.cross(hand_up_int, hand_forward_int, dim=1)
            hand_side_int = torch.nn.functional.normalize(hand_side_int, dim=1)

            # Stack to get Rotation Matrix (N_int, 3, 3) - Columns are [side, up, forward]
            hand_R_int_batch = torch.stack([hand_side_int, hand_up_int, hand_forward_int], dim=2)

            # Compute object pose based on geometric hand frame
            # Apply relative transform: T_obj = T_hand * T_rel
            # R_obj = R_hand * R_rel (broadcast the single relative transform to all frames)
            # T_obj = R_hand * T_rel + T_hand
            int_obj_R = hand_R_int_batch @ relative_obj_R.unsqueeze(0)  # (N_int, 3, 3) @ (1, 3, 3) -> (N_int, 3, 3)
            int_obj_T = (hand_R_int_batch @ relative_obj_T.unsqueeze(-1)).squeeze(-1) + hand_wrist_int  # (N_int, 3)

            # Transform object vertices (batch)
            # Apply scale, rotation, translation: v' = (v * scale) @ R + T
            obj_verts_scaled_int = obj_verts_canonical.unsqueeze(0) * scale_mat  # (1, V, 3)
            int_obj_verts = (obj_verts_scaled_int @ int_obj_R) + int_obj_T.unsqueeze(1)  # (1, V, 3) @ (N_int, 3, 3) + (N_int, 1, 3) -> (N_int, V, 3)
            # Transform normals: n' = n @ R.T (normals are covariant vectors)
            int_obj_normals = obj_normals_canonical.unsqueeze(0) @ int_obj_R.transpose(-2, -1)  # (1, V, 3) @ (N_int, 3, 3) -> (N_int, V, 3)

            # Convert interaction object poses to 6D representation for smooth loss
            # Note: int_obj_R and int_obj_T are computed from hand pose, so gradients flow to hand
            int_obj_rot_6d = matrix_to_rotation_6d(int_obj_R)  # (N_int, 6)

            # Update multi_frame_model with interaction poses (for visualization/consistency, but gradients flow to hand)
            # Since object pose is locked by hook, this update doesn't affect gradients
            with torch.no_grad():
                multi_frame_model.rot_6d.data[interaction_frames_list] = int_obj_rot_6d.detach()
                multi_frame_model.trans.data[interaction_frames_list] = int_obj_T.detach()

            # For smooth loss calculation, use computed values for interaction segment
            # These values have gradients that flow to hand root_orient/trans
            all_obj_rot_6d_for_smooth = multi_frame_model.rot_6d.clone().detach()
            all_obj_trans_for_smooth = multi_frame_model.trans.clone().detach()
            all_obj_rot_6d_for_smooth[interaction_frames_list] = int_obj_rot_6d  # Has gradients, flows to hand
            all_obj_trans_for_smooth[interaction_frames_list] = int_obj_T  # Has gradients, flows to hand

            # ===== Reconstruct full sequence =====
            # Combine in correct order
            all_hand_joints = torch.zeros(num_sampled_frames, 21, 3, device=device)
            all_hand_verts = torch.zeros(num_sampled_frames, 778, 3, device=device)
            all_obj_verts = torch.zeros(num_sampled_frames, obj_verts_canonical.shape[0], 3, device=device)
            all_obj_normals = torch.zeros(num_sampled_frames, obj_verts_canonical.shape[0], 3, device=device)

            # Fill static+transition frames
            st_idx = 0
            for frame_idx in static_transition_frames:
                all_hand_joints[frame_idx] = st_hand_joints[st_idx]
                all_hand_verts[frame_idx] = st_hand_verts[st_idx]
                all_obj_verts[frame_idx] = st_obj_verts[st_idx]
                all_obj_normals[frame_idx] = st_obj_normals[st_idx]
                st_idx += 1

            # Fill interaction frames
            for i, frame_idx in enumerate(interaction_frames_list):
                all_hand_joints[frame_idx] = int_hand_joints[i]
                all_hand_verts[frame_idx] = int_hand_verts[i]
                all_obj_verts[frame_idx] = int_obj_verts[i]
                all_obj_normals[frame_idx] = int_obj_normals[i]

            # ===== Compute losses =====
            # Loss 1: Hand 2D keypoints (all frames) with tolerance
            projected_hand_joints = all_cameras.transform_points_screen(all_hand_joints, image_size=((H_out, W_out),))[..., :2]

            # # Compute per-joint 2D error
            # joint_2d_error = torch.abs(projected_hand_joints - sampled_gt_hand_joints_2d)  # (N, 21, 2)

            # # Apply tolerance: only penalize errors beyond threshold (in pixels)
            # tolerance_pixels = 5.0  # 5 pixel tolerance
            # joint_2d_error_clamped = torch.clamp(joint_2d_error - tolerance_pixels, min=0.0)  # Dead zone

            # # Compute loss only on errors exceeding tolerance
            # hand_2d_loss = torch.mean(joint_2d_error_clamped**2) * 1e-1
            hand_2d_loss = torch.nn.functional.mse_loss(projected_hand_joints, sampled_gt_hand_joints_2d)

            # Loss 2: Object mask (all frames)
            obj_meshes_all = Meshes(verts=all_obj_verts,
                                    faces=obj_faces[None].expand(num_sampled_frames, -1, -1),
                                    textures=TexturesVertex(verts_features=torch.ones_like(all_obj_verts)))
            obj_fragments = silhouette_renderer_full.rasterizer(obj_meshes_all, cameras=all_cameras)
            rendered_obj_masks = silhouette_renderer_full.shader(obj_fragments, obj_meshes_all, cameras=all_cameras)[..., 3]

            loss_fp = weighted_false_positive_loss(rendered_obj_masks, sampled_pred_amodal_masks) * 1e2
            loss_vp = vectorized_pose_guiding_loss(obj_meshes_all, sampled_pred_amodal_masks, rendered_obj_masks, all_cameras, num_samples=2000)
            loss_obj = loss_fp + loss_vp
            # loss_obj = torch.nn.functional.mse_loss(rendered_obj_masks, sampled_pred_amodal_masks)

            # # Loss 3: 3D Boundary Alignment Loss (The "Seamless" Enforcer)
            # # Force interaction segment (2-3) boundaries to align with frozen neighbor poses
            # loss_boundary_align = torch.tensor(0.0, device=device)

            # if approaching_end_idx > 0:
            #     # First frame of 2-3 should match last frame of 1-2 (frozen)
            #     frozen_before_rot = multi_frame_model.rot_6d[approaching_end_idx - 1].detach()
            #     frozen_before_trans = multi_frame_model.trans[approaching_end_idx - 1].detach()

            #     # Compare with first frame of interaction segment
            #     int_first_rot_6d = matrix_to_rotation_6d(int_obj_R[0:1]).squeeze(0)  # (6,)
            #     int_first_trans = int_obj_T[0]  # (3,)

            #     loss_boundary_align += torch.nn.functional.mse_loss(int_first_rot_6d, frozen_before_rot)
            #     loss_boundary_align += torch.nn.functional.mse_loss(int_first_trans, frozen_before_trans)

            # if interaction_end_idx < num_sampled_frames:
            #     # Last frame of 2-3 should match first frame of 3-4 (frozen)
            #     frozen_after_rot = multi_frame_model.rot_6d[interaction_end_idx].detach()
            #     frozen_after_trans = multi_frame_model.trans[interaction_end_idx].detach()

            #     # Compare with last frame of interaction segment
            #     int_last_rot_6d = matrix_to_rotation_6d(int_obj_R[-1:]).squeeze(0)  # (6,)
            #     int_last_trans = int_obj_T[-1]  # (3,)

            #     loss_boundary_align += torch.nn.functional.mse_loss(int_last_rot_6d, frozen_after_rot)
            #     loss_boundary_align += torch.nn.functional.mse_loss(int_last_trans, frozen_after_trans)

            # loss_boundary_align = loss_boundary_align * 1e3  # High weight for strong enforcement

            # Loss 4: Smoothness loss for object poses (simplified - use combined tensor)
            # Hook ensures gradients are zero for locked frames (static+transition)
            # Interaction frames use computed values with gradients
            loss_sm_obj = compute_smoothness_loss(all_obj_rot_6d_for_smooth) * 1e1 + compute_smoothness_loss(all_obj_trans_for_smooth) * 1e1

            # # Boundary alignment loss is no longer needed since smooth loss handles boundaries naturally
            # loss_boundary_align = torch.tensor(0.0, device=device)

            # # Loss 5: Hand Boundary Alignment (using object pose relative transform)
            # # Apply object pose changes to hand verts/joints to constrain boundary frames
            # loss_hand_boundary = torch.tensor(0.0, device=device)

            # if approaching_end_idx > 0:
            #     # Compute relative transform from 2-3 first frame to 0-2 last frame (via objects)
            #     # Object at 2-3 first frame (approaching_end_idx)
            #     obj_rot_23_start = int_obj_R[0]  # First frame of interaction segment
            #     obj_trans_23_start = int_obj_T[0]

            #     # Object at 0-2 last frame (approaching_end_idx-1) - frozen
            #     obj_rot_02_end = rotation_6d_to_matrix(multi_frame_model.rot_6d[approaching_end_idx - 1].detach())
            #     obj_trans_02_end = multi_frame_model.trans[approaching_end_idx - 1].detach()

            #     # Compute relative transform: T_02_end = T_rel * T_23_start
            #     # So T_rel = T_02_end * inv(T_23_start)
            #     T_23_start_inv_R = obj_rot_23_start.T  # Transpose for rotation matrix inverse
            #     T_23_start_inv_T = -T_23_start_inv_R @ obj_trans_23_start

            #     # T_rel = T_02_end * T_23_start_inv
            #     T_rel_R = obj_rot_02_end @ T_23_start_inv_R
            #     T_rel_T = obj_rot_02_end @ T_23_start_inv_T + obj_trans_02_end

            #     # Apply this transform to 2-3 first frame hand to get target hand at 0-2 end
            #     hand_verts_23_start = int_hand_verts[0]  # (778, 3)
            #     hand_joints_23_start = int_hand_joints[0]  # (21, 3)

            #     target_hand_verts_02_end = (hand_verts_23_start @ T_rel_R.T) + T_rel_T  # (778, 3)
            #     target_hand_joints_02_end = (hand_joints_23_start @ T_rel_R.T) + T_rel_T  # (21, 3)

            #     # Get current hand at 0-2 end
            #     st_frame_idx = static_transition_frames.index(approaching_end_idx - 1)
            #     current_hand_verts_02_end = st_hand_verts[st_frame_idx]  # (778, 3)
            #     current_hand_joints_02_end = st_hand_joints[st_frame_idx]  # (21, 3)

            #     # Constrain distance between current and target
            #     # loss_hand_boundary += torch.nn.functional.mse_loss(current_hand_verts_02_end, target_hand_verts_02_end.detach())
            #     loss_hand_boundary += torch.nn.functional.mse_loss(current_hand_joints_02_end, target_hand_joints_02_end.detach())

            # if interaction_end_idx < num_sampled_frames:
            #     # Compute relative transform from 2-3 last frame to 3-5 first frame (via objects)
            #     # Object at 2-3 last frame
            #     obj_rot_23_end = int_obj_R[-1]  # Last frame of interaction segment
            #     obj_trans_23_end = int_obj_T[-1]

            #     # Object at 3-5 first frame (interaction_end_idx) - frozen
            #     obj_rot_35_start = rotation_6d_to_matrix(multi_frame_model.rot_6d[interaction_end_idx].detach())
            #     obj_trans_35_start = multi_frame_model.trans[interaction_end_idx].detach()

            #     # Compute relative transform: T_35_start = T_rel * T_23_end
            #     # So T_rel = T_35_start * inv(T_23_end)
            #     T_23_end_inv_R = obj_rot_23_end.T
            #     T_23_end_inv_T = -T_23_end_inv_R @ obj_trans_23_end

            #     # T_rel = T_35_start * T_23_end_inv
            #     T_rel_R = obj_rot_35_start @ T_23_end_inv_R
            #     T_rel_T = obj_rot_35_start @ T_23_end_inv_T + obj_trans_35_start

            #     # Apply this transform to 2-3 last frame hand to get target hand at 3-5 start
            #     hand_verts_23_end = int_hand_verts[-1]  # (778, 3)
            #     hand_joints_23_end = int_hand_joints[-1]  # (21, 3)

            #     target_hand_verts_35_start = (hand_verts_23_end @ T_rel_R.T) + T_rel_T  # (778, 3)
            #     target_hand_joints_35_start = (hand_joints_23_end @ T_rel_R.T) + T_rel_T  # (21, 3)

            #     # Get current hand at 3-5 start
            #     st_frame_idx = static_transition_frames.index(interaction_end_idx)
            #     current_hand_verts_35_start = st_hand_verts[st_frame_idx]  # (778, 3)
            #     current_hand_joints_35_start = st_hand_joints[st_frame_idx]  # (21, 3)

            #     # Constrain distance between current and target
            #     # loss_hand_boundary += torch.nn.functional.mse_loss(current_hand_verts_35_start, target_hand_verts_35_start.detach())
            #     loss_hand_boundary += torch.nn.functional.mse_loss(current_hand_joints_35_start, target_hand_joints_35_start.detach())

            # loss_hand_boundary = loss_hand_boundary * 1e3  # High weight for boundary alignment

            # Loss 5b: Hand smoothness (simplified - directly use multi_frame_model)
            # Hook ensures gradients are zero for locked frames, so we can compute smoothness across all frames
            # Convert root_orient to 6D for smoothness
            hand_root_orient_mat = axis_angle_to_matrix(multi_frame_model.mano_root_orient)  # (N, 3, 3)
            hand_root_orient_6d = matrix_to_rotation_6d(hand_root_orient_mat)  # (N, 6)

            # Compute smoothness for all frames (hooks handle locking)
            loss_sm_hand = (compute_smoothness_loss(hand_root_orient_6d) * 1e1 + compute_smoothness_loss(multi_frame_model.mano_trans) * 1e1 +
                            compute_smoothness_loss(multi_frame_model.mano_pose) * 1e1)

            loss_sm_int = loss_sm_obj + loss_sm_hand

            # Loss 6: Collision loss (only for static+transition segments, not interaction 2-3)
            # if step > 10000000000:
            # Get hand faces
            hand_faces = amano_output_st['r_faces'] if all_is_right[0].item() > 0 else amano_output_st['l_faces']

            # Select hand and object verts for collision
            st_obj_verts_for_collision = all_obj_verts[static_transition_frames]  # (N_st, V, 3)
            st_obj_normals_for_collision = all_obj_normals[static_transition_frames]  # (N_st, V, 3)
            # Create hand meshes for normal computation
            hand_meshes_st = Meshes(verts=st_hand_verts,
                                    faces=hand_faces[None].expand(num_static_transition, -1, -1),
                                    textures=TexturesVertex(verts_features=torch.ones_like(st_hand_verts)))

            # Get hand normals
            hand_normals_st = hand_meshes_st.verts_normals_packed().view(num_static_transition, 778, 3)

            # Compute bidirectional collision loss (gradients flow to all hand parameters)
            # Hand vertices penetrating into object
            collision_hand_in_obj_loss = compute_collision_loss(st_obj_verts_for_collision, st_obj_normals_for_collision, st_hand_verts, ignore_indices=None)
            # Object vertices penetrating into hand
            collision_obj_in_hand_loss = compute_collision_loss(st_hand_verts, hand_normals_st, st_obj_verts_for_collision, ignore_indices=None)
            collision_loss = (collision_hand_in_obj_loss + collision_obj_in_hand_loss) * 1e2
            # else:
            #     collision_loss = torch.tensor(0.0, device=device)
            # Total loss
            # total_loss = hand_2d_loss + loss_obj + loss_boundary_align + loss_sm_int + collision_loss

            # Loss 7: Hand anatomy loss
            T_g_p = multi_frame_model.transforms_abs  # (B, 16, 4, 4)
            T_g_a, _R, ee = multi_frame_model.axisFK(T_g_p)  # ee (B, 16, 3)
            loss_anatomy = multi_frame_model.anatomyLoss(ee)

            total_loss = loss_sm_int + hand_2d_loss + collision_loss + loss_anatomy

            total_loss.backward()
            optimizer.step()

            loop_full.set_postfix(
                loss=total_loss.item(),
                hand_2d=hand_2d_loss.item(),
                #   obj_mask=loss_obj.item(),
                collision=collision_loss.item(),
                smooth_obj=loss_sm_obj.item(),
                smooth_hand=loss_sm_hand.item(),
                anatomy=loss_anatomy.item())

        # Step 8.7: Apply optimized results
        print(f"\nStep 8.7: Applying optimized results")
        print(f"  All parameters are already in multi_frame_model (optimized directly)")

        # No need to copy parameters back - they were optimized in-place
        # Just re-compute final object poses for interaction segment to ensure consistency
        with torch.no_grad():
            # Re-compute final object poses for interaction segment using the same geometric logic
            # Run MANO one last time with optimized parameters
            amano_output_final = run_amano(multi_frame_model.hand_model_amano, multi_frame_model.mano_trans[interaction_frames_list][None],
                                           multi_frame_model.mano_root_orient[interaction_frames_list][None],
                                           multi_frame_model.mano_pose[interaction_frames_list][None], all_is_right[interaction_frames_list].to(device))
            final_hand_joints = amano_output_final['joints'].squeeze(0) @ flat_mat  # (N_int, 21, 3)

            # Build geometric hand frame
            final_hand_wrist = final_hand_joints[:, 0]
            final_hand_forward = torch.nn.functional.normalize(final_hand_joints[:, 9] - final_hand_joints[:, 0], dim=1)
            final_hand_side = torch.nn.functional.normalize(final_hand_joints[:, 5] - final_hand_joints[:, 0], dim=1)
            final_hand_up = torch.nn.functional.normalize(torch.cross(final_hand_forward, final_hand_side, dim=1), dim=1)
            final_hand_side = torch.nn.functional.normalize(torch.cross(final_hand_up, final_hand_forward, dim=1), dim=1)
            final_hand_R_batch = torch.stack([final_hand_side, final_hand_up, final_hand_forward], dim=2)

            # Compute object poses (broadcast the single relative transform to all frames)
            final_obj_R_batch = final_hand_R_batch @ relative_obj_R.unsqueeze(0)  # (N_int, 3, 3) @ (1, 3, 3) -> (N_int, 3, 3)
            final_obj_T_batch = (final_hand_R_batch @ relative_obj_T.unsqueeze(-1)).squeeze(-1) + final_hand_wrist

            # Update object poses for interaction segment (hand poses are already updated in-place during optimization)
            multi_frame_model.rot_6d.data[interaction_frames_list] = matrix_to_rotation_6d(final_obj_R_batch)
            multi_frame_model.trans.data[interaction_frames_list] = final_obj_T_batch

        print(f"\nSTAGE 8 completed: Full sequence updated")
        print(f"  Segments 0-1,1-anchor_frame_idx,anchor_frame_idx-3,3-4,4-5: Hand optimized, Object frozen")
        print(f"  Segment anchor_frame_idx-3: Hand optimized (root only), Object follows hand")
    else:
        print("\nSkipping STAGE 8: No contact correspondences available")
    # NOTE modify until here #####################################################################################
    # NOTE modify until here #####################################################################################
    # NOTE modify until here #####################################################################################
    # NOTE modify until here #####################################################################################
    # NOTE modify until here #####################################################################################
    # NOTE modify until here #####################################################################################
    # NOTE modify until here #####################################################################################
    # NOTE modify until here #####################################################################################
    # NOTE modify until here #####################################################################################
    # NOTE modify until here #####################################################################################
    # NOTE modify until here #####################################################################################
    # NOTE modify until here #####################################################################################
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

    all_obj_verts_list = []
    all_obj_faces = None
    all_hand_verts_list = []
    all_hand_faces = None

    print("Rendering full sequence (RGB) in batches...")

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

            # del current_meshes, current_camera, current_images, current_rgb, current_model, current_verts_rgb
            # torch.cuda.empty_cache()

    final_rendered_images_full = np.concatenate(final_rendered_images_list, axis=0)
    print(f"Finished rendering {len(final_rendered_images_full)} frames.")

    # Create 3-way comparison video (full length)
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
        render_img_rgb = final_rendered_images_full[i]
        render_frame = overlay_rgb_render(cropped_rgbs_np[i], render_img_rgb, alpha=1.0)

        hand_frame = overlay_mask_on_image(cropped_rgbs_np[i], cropped_hand_masks_np[i], cmap_idx=1)
        panels = [modal_frame, amodal_gt_frame, render_frame, hand_frame]

        combined_frame = np.hstack(panels)
        writer.append_data((combined_frame * 255).astype(np.uint8))

    writer.close()
    print(f"Saved fitting visualization to {video_path}")

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
    if parsed_contact_map is not None:
        output_dir = os.path.join(seq_path, "optimized_hoi_contact_seq")
    else:
        output_dir = os.path.join(seq_path, "optimized_hoi_seq")

    save_hoi_sequence(output_dir, canonical_verts_np, canonical_faces_np, final_obj_scale, final_obj_rot_mat_col_major, final_obj_trans, mano_root_full_tensor,
                      mano_pose_full_tensor, mano_trans_full_tensor, is_right_full_tensor, all_obj_verts, all_obj_faces, all_hand_verts, all_hand_faces)


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
    hand_mesh_dir = os.path.join('rendering_for_paper', "ours_TasteRob", f"{seq_name}", "hand")
    obj_mesh_dir = os.path.join('rendering_for_paper', "ours_TasteRob", f"{seq_name}", "object")
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


def fit_single_frame_pose(seq_path,
                          output_path,
                          device,
                          frame_idx,
                          amodal_mask_0_np,
                          amodal_mask_np,
                          modal_mask_np,
                          hand_mask_np,
                          depth_np,
                          rgb_np,
                          bbox,
                          lr=1e-2,
                          num_steps=300,
                          l2_weight=1e3,
                          depth_weight=1e2):
    """
    Optimizes the pose for a single frame and saves a visualization.
    """
    print(f"\n--- Fitting single frame {frame_idx} ---")
    # --- 1. Data Loading (Mesh, Intrinsics, Initial Pose) ---
    io = IO()
    io.register_meshes_format(MeshGlbFormat())
    glb_path = os.path.join(seq_path, 'glb_0.glb')
    with open(glb_path, "rb") as f:
        mesh = io.load_mesh(f, include_textures=True).to(device)
    mesh = simplify_mesh(mesh, target_triangles=5000)
    verts, faces = mesh.verts_list()[0], mesh.faces_list()[0]
    verts = verts @ torch.tensor([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=torch.float32, device=device).T

    transform_path = os.path.join(seq_path, 'transform_0.json')
    with open(transform_path, 'r') as f:
        transform_matrix = torch.tensor(json.load(f)['transform'], dtype=torch.float32, device=device)
    M = transform_matrix[:3, :3]
    initial_scale = torch.linalg.norm(M, dim=0).mean()
    initial_R = M / initial_scale
    initial_T_0 = transform_matrix[3, :3]

    intrinsics_path = os.path.join(seq_path, 'intrinsics.json')
    with open(intrinsics_path, 'r') as f:
        intrinsics = torch.tensor(json.load(f)['intrinsics'], dtype=torch.float32, device=device)

    # --- 2. Camera Model Setup ---
    H_out, W_out = 256, 512
    x1, y1, x2, y2 = bbox
    w_crop, h_crop = x2 - x1, y2 - y1
    scale = min(W_out / w_crop, H_out / h_crop)
    new_w, new_h = int(w_crop * scale), int(h_crop * scale)
    pad_left = (W_out - new_w) // 2
    pad_top = (H_out - new_h) // 2
    fx_new = intrinsics[0, 0] * scale
    fy_new = intrinsics[1, 1] * scale
    cx_new = (intrinsics[0, 2] - x1) * scale + pad_left
    cy_new = (intrinsics[1, 2] - y1) * scale + pad_top

    camera = PerspectiveCameras(
        focal_length=((fx_new, fy_new),),
        principal_point=((cx_new, cy_new),),
        image_size=((H_out, W_out),),
        in_ndc=False,
        device=device,
    )

    # --- 3. Initial Translation Calculation (relative to frame 0) ---
    T0_np = initial_T_0.cpu().numpy()
    Z0 = T0_np[2]
    x0, y0, w0, h0 = cv2.boundingRect(amodal_mask_0_np)
    u0 = x0 + w0 / 2.0
    v0 = y0 + h0 / 2.0
    xt, yt, wt, ht = cv2.boundingRect(amodal_mask_np)
    ut = xt + wt / 2.0
    vt = yt + ht / 2.0
    delta_u, delta_v = ut - u0, vt - v0
    delta_X = -1 * (delta_u * Z0 / fx_new.item())
    delta_Y = -1 * (delta_v * Z0 / fy_new.item())
    initial_T_for_frame = torch.tensor([T0_np[0] + delta_X, T0_np[1] + delta_Y, Z0], dtype=torch.float32, device=device)

    # --- 4. Model and Optimizer ---
    initial_scale = torch.tensor(initial_scale.expand(3), dtype=torch.float32, device=device)
    model = SingleObjectPose(initial_R, initial_T_for_frame, initial_scale, verts, faces).to(device)
    # optimizer = torch.optim.Adam([model.rot_6d, model.scale, model.trans], lr=lr)
    optimizer = torch.optim.Adam([model.rot_6d, model.trans], lr=lr)

    # --- 5. Prepare Target Tensors ---
    target_amodal = torch.from_numpy(amodal_mask_np).float().to(device)
    target_modal = torch.from_numpy(modal_mask_np).to(device)
    target_hand = torch.from_numpy(hand_mask_np).to(device)
    target_depth = torch.from_numpy(depth_np[..., 0]).float().to(device)

    # --- 6. Differentiable Rendering Setup ---
    raster_settings = RasterizationSettings(image_size=(H_out, W_out), blur_radius=1e-4, faces_per_pixel=20)
    renderer = MeshRenderer(rasterizer=MeshRasterizer(raster_settings=raster_settings), shader=SoftSilhouetteShader())

    # --- 7. Optimization Loop ---
    loop = tqdm(range(num_steps), desc=f"Optimizing Frame {frame_idx}")
    for step in loop:

        optimizer.zero_grad()
        posed_mesh = model()
        fragments = renderer.rasterizer(posed_mesh, cameras=camera)
        rendered_mask = renderer.shader(fragments, posed_mesh, cameras=camera)[0, ..., 3]
        rendered_zbuf = fragments.zbuf[0, ..., 0]

        if (rendered_zbuf > 0).sum() == 0:
            rendered_zbuf = torch.zeros_like(rendered_zbuf)

        # Create a weight map for the loss.
        # Where the hand is present and the amodal mask is false (background),
        # we reduce the penalty for rendering object pixels, as the hand might be occluding the object.
        # weights = torch.where(target_hand.bool() & target_amodal.bool(), torch.ones_like(target_amodal) * 0.1, torch.ones_like(target_amodal))
        # l2_loss = (weights * torch.nn.functional.mse_loss(rendered_mask, target_amodal, reduction='none')).mean() * l2_weight
        l2_loss = torch.nn.functional.mse_loss(rendered_mask, target_amodal, reduction='mean') * l2_weight
        # import pdb
        # pdb.set_trace()
        rendered_depth = torch.where((rendered_zbuf > 0), rendered_zbuf, torch.zeros_like(rendered_zbuf))
        target_depth = torch.where(target_modal.bool(), target_depth, torch.zeros_like(target_depth))
        # depth_loss = compute_affine_invariant_loss(rendered_depth, target_depth) * depth_weight
        # depth_loss = compute_scale_invariant_depth_loss(rendered_depth.unsqueeze(0), target_depth.unsqueeze(0)) * depth_weight
        # depth_loss = torch.tensor(0.0, device=device)
        total_loss = l2_loss

        total_loss.backward()
        optimizer.step()
        loop.set_postfix(loss=total_loss.item(), l2=l2_loss.item())

    # --- 8. Visualization ---
    with torch.no_grad():
        final_posed_mesh = model()
        fragments = renderer.rasterizer(final_posed_mesh, cameras=camera)
        final_mask = renderer.shader(fragments, final_posed_mesh, cameras=camera)[0, ..., 3].cpu().numpy()

    # Create 4-panel comparison image
    modal_frame = overlay_mask_on_image(rgb_np, modal_mask_np)
    amodal_gt_frame = overlay_mask_on_image(rgb_np, amodal_mask_np)
    hand_frame = overlay_mask_on_image(rgb_np, hand_mask_np, cmap_idx=1)
    render_frame = overlay_mask_on_image(rgb_np, (final_mask > 0.5).astype(np.uint8))

    depth_map_uint8 = (depth_np.squeeze() * 255).astype(np.uint8)
    depth_colormap = cv2.applyColorMap(depth_map_uint8, cv2.COLORMAP_VIRIDIS)
    black_bg = np.zeros_like(depth_colormap)
    depth_viz_masked = np.where(modal_mask_np[..., np.newaxis], depth_colormap, black_bg)
    depth_frame = cv2.cvtColor(depth_viz_masked, cv2.COLOR_BGR2RGB) / 255.0

    combined_image = (np.hstack([modal_frame, amodal_gt_frame, render_frame, hand_frame, depth_frame]) * 255).astype(np.uint8)
    save_path = os.path.join(output_path, f"optimized_frame_{frame_idx:04d}.png")
    imageio.imwrite(save_path, combined_image)
    print(f"Saved single frame visualization to {save_path}")


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
            metric_depth_img_uint16 = (metric_depth_img_float[:, :, 0] * 65535).astype(np.uint16)
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
    )

    # # # --- Debug single frame fitting ---
    # frame_to_debug = 126
    # fit_single_frame_pose(
    #     seq_path=seq_path,
    #     output_path=output_seq_path,
    #     device=device,
    #     frame_idx=frame_to_debug,
    #     amodal_mask_0_np=pred_amodal_masks_np[0],
    #     amodal_mask_np=pred_amodal_masks_np[frame_to_debug],
    #     modal_mask_np=cropped_modal_masks_np[frame_to_debug],
    #     hand_mask_np=cropped_hand_masks_np[frame_to_debug],
    #     depth_np=cropped_depths_np[frame_to_debug],
    #     rgb_np=cropped_rgbs_np[frame_to_debug],
    #     bbox=global_bboxes[frame_to_debug],
    #     lr=args.lr,
    #     num_steps=args.num_steps,
    # )


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

    args = parser.parse_args()

    main(args)
