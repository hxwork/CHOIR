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
from pytorch3d.renderer import (AmbientLights, BlendParams, Materials, MeshRasterizer, MeshRenderer, PerspectiveCameras, PointLights, RasterizationSettings,
                                SoftSilhouetteShader, TexturesUV, TexturesVertex)
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
from pnp import generate_queries_1stage, run_pnp_1stage
from torch_mesh_intersection.mesh_intersection.bvh_search_tree import BVH
from utils import *


def calculate_iou(mask1, mask2):
    """Calculates Intersection over Union for two binary masks."""
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union > 0 else 0.0


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


def init_sam_video_predictor():
    device = f"cuda:{torch.cuda.current_device()}"

    from sam2.build_sam import build_sam2_video_predictor
    sam2_checkpoint = "sam2/checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

    return build_sam2_video_predictor(model_cfg, sam2_checkpoint)


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


def sam_video_tracking(video_predictor, frames, init_mask, init_frame_idx):
    """
    Tracks an object mask through a video sequence, both forwards and backwards
    from the initial frame.
    """
    if init_mask is None or not np.any(init_mask):
        return {}

    # --- 1. Forward Tracking ---
    frames_forward = frames[init_frame_idx:]
    # frames_forward_np = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_forward])
    frames_forward_np = np.stack([f for f in frames_forward])

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


def compute_density_balanced_loss(hand_contact_verts, obj_contact_verts, k=5):
    """
    基于局部空间密度的自适应权重
    接触点越密集的区域，单个点的权重越小
    
    Args:
        hand_contact_verts: (N, 3) 手部接触点的3D坐标
        obj_contact_verts: (N, 3) 物体接触点的3D坐标
        k: 用于密度估计的最近邻数量
    """
    N = len(hand_contact_verts)
    if N == 0:
        return torch.tensor(0.0, device=hand_contact_verts.device)

    if N == 1:
        # 只有一个接触点，直接返回loss
        return torch.nn.functional.l1_loss(hand_contact_verts, obj_contact_verts)

    # 计算所有接触点之间的两两距离矩阵
    # hand_contact_verts: (N, 3)
    # 扩展维度: (N, 1, 3) - (1, N, 3) = (N, N, 3)
    diff = hand_contact_verts.unsqueeze(1) - hand_contact_verts.unsqueeze(0)  # (N, N, 3)
    distances = torch.norm(diff, dim=-1)  # (N, N)

    # 对角线是自己到自己的距离（0），需要排除
    # 将对角线设为一个很大的值，这样在取最小值时不会被选中
    distances = distances + torch.eye(N, device=distances.device) * 1e10

    # 对每个点，找到最近的k个邻居的平均距离
    k_actual = min(k, N - 1)  # 如果总点数不够k个，就用全部
    if k_actual > 0:
        nearest_k_distances, _ = torch.topk(distances, k_actual, dim=1, largest=False)  # (N, k)
        local_density = nearest_k_distances.mean(dim=1)  # (N,) 平均距离越小，密度越大
    else:
        local_density = torch.ones(N, device=distances.device)

    # 权重与密度成反比：密度大（距离小）-> 权重小
    # 使用倒数作为权重，加一个小的epsilon防止除零
    weights = local_density / (local_density.sum() + 1e-8)  # 归一化到和为1
    weights = weights * N  # 乘以N，使总权重和等于点数（保持loss scale）

    # 计算加权loss
    per_point_loss = torch.abs(hand_contact_verts - obj_contact_verts).sum(dim=-1)  # (N,)
    weighted_loss = (per_point_loss * weights).mean()

    return weighted_loss


def parse_contact_map(contact_map):
    """
    解析contact map: {hand_vertex_idx: object_vertex_idx}
    
    Returns:
        dict with 'correspondences', 'hand_indices', 'num_contacts'
    """
    if contact_map is None:
        return None

    # Convert string keys and values to int if needed
    # Format: {hand_vertex_idx: object_vertex_idx}
    correspondences = {}
    for k, v in contact_map.items():
        hand_idx = int(k) if isinstance(k, str) else k
        obj_idx = int(v) if isinstance(v, str) else v
        correspondences[hand_idx] = obj_idx

    return {'correspondences': correspondences, 'hand_indices': list(correspondences.keys()), 'num_contacts': len(correspondences)}


def compute_collision(object_verts, object_normals, hand_verts, ignore_indices=None):
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
    hand_in_obj_penetr_dist = torch.zeros_like(hand_nn_dist)
    hand_in_obj_penetr_dist = torch.where(hand_interior, hand_nn_dist, hand_in_obj_penetr_dist)  # [B, N_hand]

    return hand_in_obj_penetr_dist


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
    hand_in_obj_penetr_dist = hand_nn_dist[hand_interior].sum()

    return hand_in_obj_penetr_dist


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
    # probs = flat_gt + (uncovered_score * 3.0) + 1e-8  original setting
    probs = flat_gt + (uncovered_score * 1.0) + 1e-8

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
                 mano_betas_init=None,
                 is_right_init=None,
                 obj_textures=None):
        super().__init__()
        self.rot_6d = nn.Parameter(matrix_to_rotation_6d(initial_R), requires_grad=True)  # (N, 6)
        self.trans = nn.Parameter(initial_T, requires_grad=True)  # (N, 3)
        self.scale = nn.Parameter(torch.tensor([1.0], dtype=torch.float32, device=initial_scale.device), requires_grad=True)  # (1,)
        self.initial_scale = initial_scale

        self.register_buffer('initial_verts', initial_verts)  # (N, V, 3)
        self.register_buffer('faces', faces)  # (N, F, 3)
        self.obj_textures = obj_textures  # Store original textures

        mano_dir = '../../stage1_preprocess/Dyn_HaMR_new/_DATA/data'
        amano_cfg = {
            'mano_assets_root': os.path.join(mano_dir, 'mano'),
            'flat_hand_mean': True,  # NOTE original setting
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
            self.mano_betas = nn.Parameter(mano_betas_init, requires_grad=False)
            self.is_right = is_right_init
        else:
            # This branch for optimization
            T_w2c = torch.stack([torch.tensor(mano_param['T_w2c'], dtype=torch.float32) for mano_param in mano_params])  # (N, 4, 4)
            root_orient = torch.stack([torch.tensor(mano_param['root_orient'], dtype=torch.float32) for mano_param in mano_params])  # (N, 3)
            trans = torch.stack([torch.tensor(mano_param['trans'], dtype=torch.float32) for mano_param in mano_params])  # (N, 3)

            # apply T_w2c to root_orient and trans
            root_orient = matrix_to_axis_angle(T_w2c[:, :3, :3] @ axis_angle_to_matrix(root_orient))  # (N, 3)
            trans = (T_w2c[:, :3, :3] @ trans.unsqueeze(-1)).squeeze(-1) + T_w2c[:, :3, 3]  # (N, 3)

            pose = torch.stack([torch.tensor(mano_param['pose'], dtype=torch.float32) for mano_param in mano_params])  # (N, 15, 3)
            betas = torch.stack([torch.tensor(mano_param['betas'], dtype=torch.float32) for mano_param in mano_params])  # (N, 10)
            self.is_right = torch.stack([torch.tensor(mano_param['is_right'], dtype=torch.float32) for mano_param in mano_params])  # (N, 1)

            self.mano_root_orient = nn.Parameter(root_orient, requires_grad=True)  # (N, 3)
            self.mano_trans = nn.Parameter(trans, requires_grad=True)  # (N, 3)
            self.mano_pose = nn.Parameter(pose, requires_grad=True)  # (N, 15, 3)
            self.mano_betas = nn.Parameter(betas, requires_grad=True)  # (N, 10)

    def forward(self, use_original_texture=False):
        # object
        N = self.rot_6d.shape[0]
        R = rotation_6d_to_matrix(self.rot_6d)  # (N, 3, 3)
        scale = (self.scale * self.initial_scale).unsqueeze(0).unsqueeze(0)  # (1, 1, 3)
        posed_verts = (self.initial_verts * scale) @ R + self.trans.unsqueeze(1)

        # Use original texture if available and requested, otherwise use white
        if use_original_texture and self.obj_textures is not None:
            # Extend texture to batch dimension
            obj_textures = self.obj_textures.extend(N)
        else:
            obj_textures = TexturesVertex(verts_features=torch.ones_like(posed_verts))

        obj_mesh = Meshes(verts=posed_verts, faces=self.faces, textures=obj_textures)

        # hand
        device = self.mano_trans.device
        flat_mat = torch.diag(torch.tensor([-1., -1., 1.], dtype=torch.float32, device=device))[None].repeat(N, 1, 1)  # (N, 3, 3)
        amano_output = run_amano(self.hand_model_amano, self.mano_trans[None], self.mano_root_orient[None], self.mano_pose[None], self.is_right.to(device),
                                 self.mano_betas[None])
        mano_joints = amano_output['joints'].squeeze(0) @ flat_mat  # (1, N, 21, 3) -> (N, 21, 3)
        mano_verts = amano_output['vertices'].squeeze(0) @ flat_mat  # (1, N, 778, 3) -> (N, 778, 3)
        mano_l_faces = amano_output['l_faces']  # (1538, 3)
        mano_r_faces = amano_output['r_faces']  # (1538, 3)
        mano_is_right = amano_output['is_right'].squeeze(0).view(-1)  # (1, N) -> (N,)
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
    overwrite=False,
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

    # Save original texture from mesh
    original_textures = None

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
    H_ori, W_ori = 480, 640
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
    # Check if checkpoint exists to skip STAGE 1-3
    # ========================================================================
    checkpoint_init_path = os.path.join(output_path, 'stage1_3_checkpoint.pth')
    checkpoint_scale_path = os.path.join(output_path, 'stage5_checkpoint.pth')

    if os.path.exists(checkpoint_init_path) and not overwrite:
        print("\n" + "=" * 80)
        print("Loading checkpoint from STAGE 1-3...")
        print("=" * 80)
        checkpoint = torch.load(checkpoint_init_path, map_location=device)

        # Load stage ranges and segment boundaries
        stage_ranges = checkpoint['stage_ranges']
        segment_boundaries = checkpoint['segment_boundaries']
        picked_frame_idx = checkpoint['picked_frame_idx']
        start_static_end_idx = checkpoint['start_static_end_idx']
        approaching_end_idx = checkpoint['approaching_end_idx']
        interaction_end_idx = checkpoint['interaction_end_idx']
        end_static_start_idx = checkpoint['end_static_start_idx']

        # Load camera parameters
        fx_new = checkpoint['fx_new']
        fy_new = checkpoint['fy_new']
        cx_new = checkpoint['cx_new']
        cy_new = checkpoint['cy_new']

        # Recreate focal_length and principal_point (already in device)
        focal_length = torch.tensor([[fx_new, fy_new]], device=device).expand(num_sampled_frames, -1)
        principal_point = torch.tensor([[cx_new, cy_new]], device=device).expand(num_sampled_frames, -1)

        # Recreate verts_batch and faces_batch
        verts_batch = verts.unsqueeze(0).repeat(num_sampled_frames, 1, 1)
        faces_batch = faces.unsqueeze(0).repeat(num_sampled_frames, 1, 1)

        # Recreate multi_frame_model with loaded parameters
        multi_frame_model = TemporalHandObjectPose(checkpoint['initial_R_mat'].to(device),
                                                   checkpoint['initial_T'].to(device),
                                                   checkpoint['initial_scale'].to(device),
                                                   verts_batch,
                                                   faces_batch,
                                                   sampled_mano_params,
                                                   obj_textures=original_textures).to(device)

        # Load optimized parameters
        multi_frame_model.rot_6d.data = checkpoint['rot_6d'].to(device)
        multi_frame_model.trans.data = checkpoint['trans'].to(device)
        multi_frame_model.scale.data = checkpoint['scale'].to(device)
        multi_frame_model.mano_root_orient.data = checkpoint['mano_root_orient'].to(device)
        multi_frame_model.mano_trans.data = checkpoint['mano_trans'].to(device)
        multi_frame_model.mano_pose.data = checkpoint['mano_pose'].to(device)
        if 'mano_betas' in checkpoint:
            multi_frame_model.mano_betas.data = checkpoint['mano_betas'].to(device)
        else:
            multi_frame_model.mano_betas.data = torch.zeros([num_sampled_frames, 10], dtype=torch.float32, device=device)

        print(f"Checkpoint loaded successfully!")
        print(f"  Picked frame: {picked_frame_idx}")
        print(f"  Segment boundaries: {segment_boundaries}")
        print("Skipping STAGE 1-3, continuing to STAGE 4...")

    else:
        print("\n" + "=" * 80)
        print("No checkpoint found, running STAGE 1-3...")
        print("=" * 80)

        # ========================================================================
        # STAGE 1: Detect Interaction Segments (0-1, 1-2, 2-3, 3-4, 4-5)
        # ========================================================================
        print("\n" + "=" * 80)
        print("STAGE 1: Detecting Interaction Segments (0-1, 1-2, 2-3, 3-4, 4-5)")
        print("=" * 80)
        # Step 1.1: Detect static segments using mask sequence
        start_static_end_idx = 1
        approaching_end_idx = 1
        interaction_end_idx = num_sampled_frames - 1
        end_static_start_idx = num_sampled_frames - 1

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
        # STAGE 2: Optimize the picked frame's object pose
        # ========================================================================
        print("\n" + "=" * 80)
        print("STAGE 2: Optimizing the picked frame's object pose")
        print("=" * 80)
        manually_picked_frame_json = os.path.join(seq_path, 'manually_picked_frame_idx.json')
        with open(manually_picked_frame_json, 'r') as f:
            picked_frame_idx = json.load(f)['new_frame_idx']

        # # Optimize object scale and shared pose for 0-1 segment
        # initial_scale = torch.tensor(initial_scale, dtype=torch.float32, device=device).repeat(3)  # (3,)
        # single_frame_model = SingleObjectPose(initial_R, initial_T, initial_scale, verts, faces).to(device)
        # optimizer = torch.optim.Adam([single_frame_model.rot_6d, single_frame_model.scale, single_frame_model.trans], lr=1e-4)

        # # Create batched camera for the picked frame's segment
        # camera_picked_frame = PerspectiveCameras(
        #     focal_length=focal_length[picked_frame_idx, None],
        #     principal_point=principal_point[picked_frame_idx, None],
        #     image_size=((H_out, W_out),),
        #     in_ndc=False,
        #     device=device,
        # )

        # # Prepare metric depth data for depth supervision
        # sampled_metric_depths = torch.from_numpy(sampled_metric_depths_np[picked_frame_idx:picked_frame_idx + 1, ..., 0]).float().to(device)  # (1, H, W)

        # # --- Differentiable Rendering Setup ---
        # raster_settings = RasterizationSettings(image_size=(H_out, W_out), blur_radius=1e-4, faces_per_pixel=20)
        # renderer = MeshRenderer(
        #     rasterizer=MeshRasterizer(cameras=camera_picked_frame, raster_settings=raster_settings),
        #     shader=SoftSilhouetteShader(),
        # )

        # # --- Optimization Loop for 0-1 Segment ---
        # loop = tqdm(range(200), desc=f"Optimizing the picked frame's object pose")
        # for step in loop:
        #     optimizer.zero_grad()

        #     # 同一个pose应用到picked frame
        #     posed_mesh = single_frame_model()

        #     # Render the picked frame
        #     fragments = renderer.rasterizer(posed_mesh, cameras=camera_picked_frame)
        #     rendered_masks = renderer.shader(fragments, posed_mesh, cameras=camera_picked_frame)[..., 3]  # (1, H, W)

        #     # Extract rendered depth from fragments
        #     # zbuf shape: (N, H, W, K) where K is faces_per_pixel
        #     # Use the closest depth (zbuf[..., 0])
        #     rendered_depth = fragments.zbuf[..., 0]  # (1, H, W)

        #     # Combine valid depth and object mask
        #     depth_loss_mask = (rendered_depth > 0) & (rendered_masks > 0.5) & (sampled_modal_masks[picked_frame_idx:picked_frame_idx + 1].float() > 0.5
        #                                                                       )  # (1, H, W)

        #     # Loss: MSE with all masks in 0-1 segment
        #     loss_mse = torch.nn.functional.mse_loss(rendered_masks, sampled_modal_masks[picked_frame_idx:picked_frame_idx + 1].float()) * 1e2

        #     # Depth loss: constrain rendered depth to match metric depth in object region
        #     depth_loss = torch.tensor(0.0, device=device)
        #     if depth_loss_mask.sum() > 0:
        #         # Only compute depth loss where both masks are valid
        #         rendered_depth_masked = rendered_depth[depth_loss_mask]
        #         metric_depth_masked = sampled_metric_depths[depth_loss_mask]

        #         # L2 loss between rendered depth and metric depth
        #         loss_depth = torch.nn.functional.mse_loss(rendered_depth_masked, metric_depth_masked) * 1e2

        #     total_loss = loss_mse + depth_loss
        #     # total_loss = loss_mse

        #     total_loss.backward()
        #     optimizer.step()
        #     loop.set_postfix(loss=total_loss.item(), loss_mse=loss_mse.item(), loss_depth=loss_depth.item())

        # Optimize object scale and shared pose for 0-1 segment
        initial_scale = torch.tensor(initial_scale, dtype=torch.float32, device=device).repeat(3)  # (3,)
        verts_batch = verts.unsqueeze(0)
        faces_batch = faces.unsqueeze(0)
        single_frame_model = TemporalHandObjectPose(initial_R.unsqueeze(0),
                                                    initial_T.unsqueeze(0),
                                                    initial_scale,
                                                    verts_batch,
                                                    faces_batch,
                                                    sampled_mano_params[picked_frame_idx:picked_frame_idx + 1],
                                                    obj_textures=original_textures).to(device)

        # Create batched camera for the picked frame's segment
        camera_picked_frame = PerspectiveCameras(
            focal_length=focal_length[picked_frame_idx, None],
            principal_point=principal_point[picked_frame_idx, None],
            image_size=((H_out, W_out),),
            in_ndc=False,
            device=device,
        )

        # Prepare metric depth data for depth supervision
        sampled_metric_depths = torch.from_numpy(sampled_metric_depths_np[picked_frame_idx:picked_frame_idx + 1, ..., 0]).float().to(device)  # (1, H, W)

        # --- Differentiable Rendering Setup ---
        raster_settings = RasterizationSettings(image_size=(H_out, W_out), blur_radius=1e-4, faces_per_pixel=20)
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=camera_picked_frame, raster_settings=raster_settings),
            shader=SoftSilhouetteShader(),
        )

        # NOTE ICP alignment
        # Forward pass for the picked frame
        picked_frame_R = rotation_6d_to_matrix(single_frame_model.rot_6d)
        picked_frame_scale_full = single_frame_model.scale * single_frame_model.initial_scale
        picked_frame_transform = Transform3d(dtype=torch.float32, device=device).scale(picked_frame_scale_full.unsqueeze(0)).rotate(
            picked_frame_R.unsqueeze(0)).translate(single_frame_model.trans)
        picked_frame_verts = picked_frame_transform.transform_points(verts.unsqueeze(0))
        picked_frame_mesh = Meshes(verts=picked_frame_verts,
                                   faces=faces.unsqueeze(0),
                                   textures=TexturesVertex(verts_features=torch.ones_like(picked_frame_verts)))

        picked_frame_fragments = renderer.rasterizer(picked_frame_mesh, cameras=camera_picked_frame)
        picked_frame_rendered = renderer.shader(picked_frame_fragments, picked_frame_mesh, cameras=camera_picked_frame)[0, ..., 3]

        # Metric depth loss for first
        picked_frame_depth = picked_frame_fragments.zbuf[0, ..., 0]  # (H, W)
        metric_depth_picked_frame = torch.from_numpy(sampled_metric_depths_np[picked_frame_idx, ..., 0]).float().to(device)  # (H, W)

        picked_frame_target_mask = sampled_modal_masks[picked_frame_idx].unsqueeze(0).unsqueeze(-1).permute(0, 3, 1, 2)
        picked_frame_target_points = get_rgbd_point_cloud(camera_picked_frame, sampled_rgbs[picked_frame_idx].unsqueeze(0).permute(0, 3, 1, 2),
                                                          metric_depth_picked_frame.unsqueeze(0).unsqueeze(-1).permute(0, 3, 1, 2), picked_frame_target_mask,
                                                          0.5)
        picked_frame_target_points = picked_frame_target_points.points_packed()
        picked_frame_target_points = picked_frame_target_points.squeeze(0)  # (N, 3)

        # Filter out points with abnormally large depth values (likely from mask edges)
        if picked_frame_target_points.shape[0] > 0:
            z_coords = picked_frame_target_points[:, 2]  # (N,)
            depth_threshold = torch.quantile(z_coords, 0.8)
            valid_mask = z_coords <= depth_threshold
            picked_frame_target_points = picked_frame_target_points[valid_mask]
            print(f"  Filtered first_target_points: {valid_mask.sum().item()}/{len(valid_mask)} points kept (depth threshold: {depth_threshold.item():.4f})")

        picked_frame_source_mask = picked_frame_rendered.unsqueeze(0).unsqueeze(-1).permute(0, 3, 1, 2)
        picked_frame_source_points = get_rgbd_point_cloud(camera_picked_frame, sampled_rgbs[picked_frame_idx].unsqueeze(0).permute(0, 3, 1, 2),
                                                          picked_frame_depth.unsqueeze(0).unsqueeze(-1).permute(0, 3, 1, 2), picked_frame_source_mask, 0.5)
        picked_frame_source_points = picked_frame_source_points.points_packed()
        picked_frame_source_points = picked_frame_source_points.squeeze(0)  # (N, 3)

        if picked_frame_target_points is not None and picked_frame_target_points.shape[0] > 10:
            estimate_scale = True
            # ICP needs batch dimension
            first_icp_result = iterative_closest_point(
                picked_frame_source_points.unsqueeze(0),  # X
                picked_frame_target_points.unsqueeze(0),  # Y
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
            pre_combined_points = torch.cat([picked_frame_target_points, picked_frame_source_points, picked_frame_verts[0]], dim=0).detach().cpu().numpy()

            pre_target_colors = torch.tensor([[0.0, 1.0, 0.0]], device=device).repeat(picked_frame_target_points.shape[0], 1)
            pre_source_colors = torch.tensor([[1.0, 0.0, 0.0]], device=device).repeat(picked_frame_source_points.shape[0], 1)
            pre_mesh_colors = torch.tensor([[0.0, 0.0, 1.0]], device=device).repeat(picked_frame_verts[0].shape[0], 1)
            pre_combined_colors = (torch.cat([pre_target_colors, pre_source_colors, pre_mesh_colors], dim=0).cpu().numpy() * 255).astype(np.uint8)

            pre_pcd = trimesh.points.PointCloud(pre_combined_points, colors=pre_combined_colors)
            pre_debug_ply_path = os.path.join(output_path, f"stage1_debug_icp_pre_alignment_with_mesh_{picked_frame_idx}.ply")
            pre_pcd.export(pre_debug_ply_path)
            print(f"Saved ICP pre-alignment point cloud with mesh to {pre_debug_ply_path}")

            # Save post-alignment point cloud with mesh
            # Target = Green, Aligned Source = Red, Aligned Mesh = Blue
            aligned_points = s_delta * (picked_frame_source_points @ R_delta[0]) + T_delta
            aliged_mesh_verts = s_delta * (picked_frame_verts[0] @ R_delta[0]) + T_delta
            combined_points = torch.cat([picked_frame_target_points, aligned_points, aliged_mesh_verts], dim=0).detach().cpu().numpy()

            target_colors = torch.tensor([[0.0, 1.0, 0.0]], device=device).repeat(picked_frame_target_points.shape[0], 1)
            source_colors = torch.tensor([[1.0, 0.0, 0.0]], device=device).repeat(picked_frame_source_points.shape[0], 1)
            aliged_mesh_colors = torch.tensor([[0.0, 0.0, 1.0]], device=device).repeat(aliged_mesh_verts.shape[0], 1)
            combined_colors = (torch.cat([target_colors, source_colors, aliged_mesh_colors], dim=0).cpu().numpy() * 255).astype(np.uint8)

            pcd = trimesh.points.PointCloud(combined_points, colors=combined_colors)
            debug_ply_path = os.path.join(output_path, f"stage1_debug_icp_post_alignment_{picked_frame_idx}.ply")
            pcd.export(debug_ply_path)
            print(f"Saved ICP post-alignment point cloud to {debug_ply_path}")

            # Check if scale change is reasonable
            # If s_delta is too large or too small, metric depth is unreliable
            scale_threshold_min = 0.6
            scale_threshold_max = 1.4
            s_delta_value = s_delta.item() if isinstance(s_delta, torch.Tensor) else s_delta

            if scale_threshold_min <= s_delta_value <= scale_threshold_max:
                # PyTorch3D ICP 返回的 RT
                new_R_matrix = picked_frame_R @ R_delta.squeeze(0)
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
                    temp_fragments = renderer.rasterizer(temp_mesh, cameras=camera_picked_frame)
                    temp_rendered_mask = renderer.shader(temp_fragments, temp_mesh, cameras=camera_picked_frame)[0, ..., 3]  # (H, W)

                    # Get amodal mask for first frame
                    amodal_mask_picked_frame = sampled_pred_amodal_masks[picked_frame_idx]  # (H, W)

                    # Compute IoU
                    rendered_binary = (temp_rendered_mask > 0.5).float()
                    amodal_binary = (amodal_mask_picked_frame > 0.5).float()
                    intersection = (rendered_binary * amodal_binary).sum()
                    union = (rendered_binary + amodal_binary).clamp(0, 1).sum()
                    iou = (intersection / union) if union > 0 else torch.tensor(0.0, device=device)
                    iou_value = iou.item()

                # Only apply ICP if IoU >= 0.8
                if iou_value >= 0.8:
                    # assign back to parameter
                    with torch.no_grad():
                        single_frame_model.rot_6d.data = matrix_to_rotation_6d(new_R_matrix)  # (1, 6)
                        single_frame_model.trans.data = new_T_vector  # (1, 3)
                        single_frame_model.scale.data = s_delta * single_frame_model.scale.data  # (1,)

                    print(f" ICP Alignment Applied to First frame! (scale delta: {s_delta_value:.4f}, IoU: {iou_value:.4f})")
                    # Recreate optimizer after ICP alignment to reset momentum state
                    optimizer = torch.optim.Adam([single_frame_model.rot_6d, single_frame_model.scale, single_frame_model.trans], lr=1e-4)
                else:
                    print(f" Warning: ICP alignment rejected due to low IoU ({iou_value:.4f} < 0.8). Keeping original pose.")
            else:
                print(f" Warning: ICP scale change too large ({s_delta_value:.4f}), metric depth may be unreliable.")
                print(f" Skipping ICP alignment. Acceptable range: [{scale_threshold_min}, {scale_threshold_max}]")

        optimizer = torch.optim.Adam(
            [
                single_frame_model.rot_6d,
                single_frame_model.scale,
                single_frame_model.trans,
                single_frame_model.mano_root_orient,
                single_frame_model.mano_trans,
                single_frame_model.mano_pose,
            ],
            lr=1e-3,
        )

        # --- Optimization Loop for 0-1 Segment ---
        loop = tqdm(range(3000), desc=f"Optimizing the picked frame's object and hand pose")
        for step in loop:
            optimizer.zero_grad()

            posed_obj_meshes_batch, posed_hand_meshes_batch, mano_joints_batch = single_frame_model()

            obj_fragments = renderer.rasterizer(posed_obj_meshes_batch)
            rendered_obj_masks = renderer.shader(obj_fragments, posed_obj_meshes_batch)[..., 3]
            rendered_depth = obj_fragments.zbuf[..., 0]  # (1, H, W)

            # Combine valid depth and object mask
            depth_loss_mask = (rendered_depth > 0) & (rendered_obj_masks > 0.5) & (sampled_modal_masks[picked_frame_idx:picked_frame_idx + 1].float() > 0.5
                                                                                  )  # (1, H, W)
            loss_depth = torch.nn.functional.mse_loss(rendered_depth[depth_loss_mask], sampled_metric_depths[depth_loss_mask]) * 1e1

            # object loss
            # loss_mse = torch.nn.functional.mse_loss(rendered_obj_masks, sampled_pred_amodal_masks[picked_frame_idx:picked_frame_idx + 1]) * 1e3
            loss_fp = weighted_false_positive_loss(rendered_obj_masks, sampled_pred_amodal_masks[picked_frame_idx:picked_frame_idx + 1]) * 1e2
            loss_vp = vectorized_pose_guiding_loss(
                posed_obj_meshes_batch,
                sampled_pred_amodal_masks[picked_frame_idx:picked_frame_idx + 1],
                rendered_obj_masks,
                camera_picked_frame,
                num_samples=2000,
            )
            loss_mse = loss_fp + loss_vp
            # obj_loss = loss_mse
            obj_loss = loss_mse + loss_depth

            # hand 2d joints loss
            projected_hand_joints = camera_picked_frame.transform_points_screen(mano_joints_batch, image_size=((H_out, W_out),))  # (N, 21, 3)
            pred_hand_joints_2d = projected_hand_joints[..., :2]
            target_hand_joints_2d = sampled_gt_hand_joints_2d[picked_frame_idx:picked_frame_idx + 1]
            mano_to_openpose = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]
            target_hand_joints_2d = target_hand_joints_2d[:, mano_to_openpose, :]
            loss_joints_2d = torch.nn.functional.l1_loss(pred_hand_joints_2d, target_hand_joints_2d)

            # hand anatomy loss
            T_g_p = single_frame_model.transforms_abs  # (B, 16, 4, 4)
            T_g_a, _R, ee = single_frame_model.axisFK(T_g_p)  # ee (B, 16, 3)
            loss_anatomy = single_frame_model.anatomyLoss(ee)

            hand_loss = loss_joints_2d
            total_loss = obj_loss + hand_loss

            total_loss.backward()
            # torch.nn.utils.clip_grad_norm_([single_frame_model.mano_root_orient, single_frame_model.mano_trans, single_frame_model.mano_pose], max_norm=1.0)
            torch.nn.utils.clip_grad_norm_([single_frame_model.mano_trans], max_norm=1.0)
            torch.nn.utils.clip_grad_norm_([single_frame_model.rot_6d, single_frame_model.scale, single_frame_model.trans], max_norm=1.0)
            optimizer.step()
            loop.set_postfix(
                loss=total_loss.item(),
                loss_mse=loss_mse.item(),
                loss_depth=loss_depth.item(),
                loss_joints_2d=loss_joints_2d.item(),
                loss_anatomy=loss_anatomy.item(),
            )

        # NOTE compute the relative scale between the MANO hand and the metric depth
        # NOTE compute the relative scale between the MANO hand and the metric depth
        # NOTE compute the relative scale between the MANO hand and the metric depth

        with torch.no_grad():

            posed_obj_meshes_batch, posed_hand_meshes_batch, mano_joints_batch = single_frame_model()

            # Render hand depth
            hand_fragments = renderer.rasterizer(posed_hand_meshes_batch)

            rendered_hand_depth = hand_fragments.zbuf[0, ..., 0]  # (H, W)
            valid_rendered_hand_mask = (rendered_hand_depth > 0)  # (H, W)
            valid_metric_hand_mask = (sampled_hand_masks[picked_frame_idx] > 0)  # (H, W)
            valid_mask = valid_rendered_hand_mask & valid_metric_hand_mask  # (H, W)
            valid_mask = valid_mask.unsqueeze(0).unsqueeze(1)  # (1, 1, H, W)
            dummy_rgb = sampled_rgbs[picked_frame_idx].unsqueeze(0).permute(0, 3, 1, 2)  # (1, 3, H, W)

            # Metric hand depth
            metric_hand_depth = torch.from_numpy(sampled_metric_depths_np[picked_frame_idx, ..., 0]).float().to(device)  # (H, W)
            metric_hand_depth = metric_hand_depth.unsqueeze(0).unsqueeze(1)  # (1, 1, H, W)
            metric_hand_points = get_rgbd_point_cloud(camera_picked_frame, dummy_rgb, metric_hand_depth, valid_mask, 0.5)
            metric_hand_points = metric_hand_points.points_packed()
            metric_hand_points = metric_hand_points.squeeze(0)  # (N, 3)

            rendered_hand_depth = rendered_hand_depth.unsqueeze(0).unsqueeze(1)  # (1, 1, H, W)
            rendered_hand_points = get_rgbd_point_cloud(camera_picked_frame, dummy_rgb, rendered_hand_depth, valid_mask, 0.5)
            rendered_hand_points = rendered_hand_points.points_packed()
            rendered_hand_points = rendered_hand_points.squeeze(0)  # (N, 3)

            # NOTE new logic
            # NOTE new logic
            # NOTE new logic

            relative_scale = torch.median(rendered_hand_points[:, 2] / metric_hand_points[:, 2])

            print(f"\n[Scale Calibration]")
            print(f"  Valid Hand Points: {metric_hand_points.shape[0]}")
            print(f"  => Scale Factor:        {relative_scale.item():.4f}")

            # object scale = metric depth * relative scale
            print(f"Original object scale: {single_frame_model.scale * single_frame_model.initial_scale}")
            single_frame_model.scale.data = single_frame_model.scale.data * relative_scale
            print(f"New object scale: {single_frame_model.scale * single_frame_model.initial_scale}")

            # NOTE new logic
            # NOTE new logic
            # NOTE new logic

            # metric_z = metric_hand_points[:, 2]  # (N,)
            # depth_threshold = torch.quantile(metric_z, 0.85)
            # foreground_mask = metric_z <= depth_threshold
            # metric_hand_points = metric_hand_points[foreground_mask]

            # rendered_z = rendered_hand_points[:, 2]  # (N,)
            # depth_threshold = torch.quantile(rendered_z, 0.85)
            # foreground_mask = rendered_z <= depth_threshold
            # rendered_hand_points = rendered_hand_points[foreground_mask]

            # # Save both point clouds to a single PLY file with different colors
            # # Convert to numpy if they are tensors
            # metric_points_np = metric_hand_points.detach().cpu().numpy() if isinstance(metric_hand_points, torch.Tensor) else metric_hand_points
            # rendered_points_np = rendered_hand_points.detach().cpu().numpy() if isinstance(rendered_hand_points, torch.Tensor) else rendered_hand_points

            # # Create colors: metric_hand_points = Red, rendered_hand_points = Green
            # metric_colors = np.tile([1.0, 0.0, 0.0], (metric_points_np.shape[0], 1))  # Red
            # rendered_colors = np.tile([0.0, 1.0, 0.0], (rendered_points_np.shape[0], 1))  # Green

            # # Combine points and colors
            # combined_points = np.vstack([metric_points_np, rendered_points_np])
            # combined_colors = np.vstack([metric_colors, rendered_colors])
            # combined_colors_uint8 = (combined_colors * 255).astype(np.uint8)

            # # Create and save point cloud
            # pcd = trimesh.points.PointCloud(combined_points, colors=combined_colors_uint8)
            # ply_path = os.path.join(output_path, f"hand_points_comparison_frame_{picked_frame_idx}.ply")
            # pcd.export(ply_path)
            # print(f"Saved combined point cloud (metric=red, rendered=green) to {ply_path}")

            # if metric_hand_points.shape[0] < 50 or rendered_hand_points.shape[0] < 50:
            #     print(f"Warning: Not enough valid points (N={metric_hand_points.shape[0]}) to compute scale. Skipping.")
            # else:
            #     src_points = metric_hand_points.detach().cpu().numpy()
            #     tgt_points = rendered_hand_points.detach().cpu().numpy()

            #     # 2. 去中心化 (Centering)
            #     src_center = np.mean(src_points, axis=0)
            #     tgt_center = np.mean(tgt_points, axis=0)

            #     src_centered = src_points - src_center
            #     tgt_centered = tgt_points - tgt_center

            #     src_pcd = o3d.geometry.PointCloud()
            #     tgt_pcd = o3d.geometry.PointCloud()
            #     src_pcd.points = o3d.utility.Vector3dVector(src_centered)
            #     tgt_pcd.points = o3d.utility.Vector3dVector(tgt_centered)

            #     estimation_method = o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=True)
            #     icp_output = o3d.pipelines.registration.registration_icp(src_pcd,
            #                                                              tgt_pcd,
            #                                                              max_correspondence_distance=0.01,
            #                                                              init=np.eye(4),
            #                                                              estimation_method=estimation_method,
            #                                                              criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50))

            #     # 计算行列式的立方根得到 relative_scale
            #     det = np.linalg.det(icp_output.transformation[:3, :3])
            #     relative_scale = np.power(det, 1 / 3.0)

            #     src_centered_homogeneous = np.hstack([src_centered, np.ones((src_centered.shape[0], 1))])  # (N, 4)
            #     src_aligned_centered = (icp_output.transformation @ src_centered_homogeneous.T).T[:, :3]  # (N, 3)
            #     src_aligned = src_aligned_centered + tgt_center  # Transform back to original coordinate system

            #     # Save three point clouds: src (red), tgt (green), aligned (blue)
            #     src_colors = np.tile([1.0, 0.0, 0.0], (src_points.shape[0], 1))  # Red
            #     tgt_colors = np.tile([0.0, 1.0, 0.0], (tgt_points.shape[0], 1))  # Green
            #     aligned_colors = np.tile([0.0, 0.0, 1.0], (src_aligned.shape[0], 1))  # Blue

            #     # Combine points and colors
            #     combined_points = np.vstack([src_points, tgt_points, src_aligned])
            #     combined_colors = np.vstack([src_colors, tgt_colors, aligned_colors])
            #     combined_colors_uint8 = (combined_colors * 255).astype(np.uint8)

            #     # Create and save point cloud
            #     pcd = trimesh.points.PointCloud(combined_points, colors=combined_colors_uint8)
            #     ply_path = os.path.join(output_path, f"hand_points_icp_registration_frame_{picked_frame_idx}.ply")
            #     pcd.export(ply_path)
            #     print(f"Saved ICP registration point clouds (src=red, tgt=green, aligned=blue) to {ply_path}")

            #     print(f"\n[Scale Calibration]")
            #     print(f"  Valid Hand Points: {metric_hand_points.shape[0]}")
            #     print(f"  => Scale Factor:        {relative_scale.item():.4f}")

            #     # object scale = metric depth * relative scale
            #     print(f"Original object scale: {single_frame_model.scale * single_frame_model.initial_scale}")
            #     single_frame_model.scale.data = single_frame_model.scale.data * relative_scale
            #     print(f"New object scale: {single_frame_model.scale * single_frame_model.initial_scale}")

        # NOTE compute the relative scale between the MANO hand and the metric depth
        # NOTE compute the relative scale between the MANO hand and the metric depth
        # NOTE compute the relative scale between the MANO hand and the metric depth

        # --- Optimization Loop for 0-1 Segment ---
        optimizer = torch.optim.Adam(
            [
                # single_frame_model.rot_6d,
                # single_frame_model.scale,
                single_frame_model.trans,
                # single_frame_model.mano_root_orient,
                # single_frame_model.mano_trans,
                # single_frame_model.mano_pose,
            ],
            lr=1e-3,
        )

        loop = tqdm(range(1000), desc=f"Optimizing the picked frame's object and hand pose")
        for step in loop:
            optimizer.zero_grad()

            posed_obj_meshes_batch, posed_hand_meshes_batch, mano_joints_batch = single_frame_model()

            obj_fragments = renderer.rasterizer(posed_obj_meshes_batch)
            rendered_obj_masks = renderer.shader(obj_fragments, posed_obj_meshes_batch)[..., 3]

            # object loss
            # loss_mse = torch.nn.functional.mse_loss(rendered_obj_masks, sampled_pred_amodal_masks[picked_frame_idx:picked_frame_idx + 1]) * 1e3
            loss_fp = weighted_false_positive_loss(rendered_obj_masks, sampled_pred_amodal_masks[picked_frame_idx:picked_frame_idx + 1]) * 1e2
            loss_vp = vectorized_pose_guiding_loss(
                posed_obj_meshes_batch,
                sampled_pred_amodal_masks[picked_frame_idx:picked_frame_idx + 1],
                rendered_obj_masks,
                camera_picked_frame,
                num_samples=2000,
            )
            # loss_mse = torch.nn.functional.mse_loss(rendered_obj_masks, sampled_pred_amodal_masks) * 1e2
            loss_mse = loss_fp + loss_vp
            obj_loss = loss_mse

            total_loss = obj_loss

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_([single_frame_model.trans], max_norm=1.0)
            optimizer.step()
            loop.set_postfix(
                loss=total_loss.item(),
                loss_mse=loss_mse.item(),
            )

        # --- Visualization for the picked frame's segment ---
        with torch.no_grad():
            # Get object mesh with original texture
            posed_mesh_textured, posed_hand_mesh, mano_joints = single_frame_model(use_original_texture=True)
            posed_verts = posed_mesh_textured.verts_list()[0]
            posed_faces = posed_mesh_textured.faces_list()[0]
            posed_verts_batch = posed_verts.unsqueeze(0)
            posed_faces_batch = posed_faces.unsqueeze(0)

            # For mask rendering, use simple white texture
            posed_textures = TexturesVertex(verts_features=torch.ones_like(posed_verts_batch))
            posed_mesh_batch = Meshes(verts=posed_verts_batch, faces=posed_faces_batch, textures=posed_textures)

            # Create posed_mesh for PnP (geometry only, texture doesn't matter for PnP)
            posed_mesh = posed_mesh_textured

            # Get hand mesh vertices and faces
            posed_hand_verts = posed_hand_mesh.verts_list()[0]
            posed_hand_faces = posed_hand_mesh.faces_list()[0]

            # Create TexturesVertex for hand (orange/skin color)
            hand_color = torch.tensor([0.9, 0.6, 0.4], device=device)  # orange/skin color
            hand_verts_features = hand_color.view(1, 1, 3).expand(1, posed_hand_verts.shape[0], 3)  # (1, V, 3)
            hand_textures = TexturesVertex(verts_features=hand_verts_features)
            hand_mesh_batch = Meshes(verts=posed_hand_verts.unsqueeze(0), faces=posed_hand_faces.unsqueeze(0), textures=hand_textures)

            # Render silhouette masks (object only for mask comparison)
            fragments = renderer.rasterizer(posed_mesh_batch, cameras=camera_picked_frame)
            final_masks = renderer.shader(fragments, posed_mesh_batch, cameras=camera_picked_frame)[..., 3].cpu().numpy()

            # Render combined RGB with both object (original texture) and hand (dummy TexturesUV)
            # Now both use TexturesUV, so we can use join_meshes_as_scene
            combined_mesh_batch = join_meshes_as_scene([posed_mesh_textured, hand_mesh_batch])

            lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])
            rgb_raster_settings = RasterizationSettings(image_size=(H_out, W_out), blur_radius=0.0, faces_per_pixel=1)
            rgb_renderer = MeshRenderer(rasterizer=MeshRasterizer(cameras=camera_picked_frame, raster_settings=rgb_raster_settings),
                                        shader=HardPhongShader(device=device, cameras=camera_picked_frame, lights=lights))

            # Render combined mesh (object + hand, both with TexturesUV)
            rendered_rgb = rgb_renderer(combined_mesh_batch).cpu().numpy()  # (1, H, W, 4)

            # Save visualization for first and last frame of 0-1 segment
            modal_frame = overlay_mask_on_image(sampled_rgbs_np[picked_frame_idx], sampled_modal_masks_np[picked_frame_idx])
            amodal_gt_frame = overlay_mask_on_image(sampled_rgbs_np[picked_frame_idx], sampled_pred_amodal_masks_np[picked_frame_idx])
            render_frame = overlay_mask_on_image(sampled_rgbs_np[picked_frame_idx], (final_masks[0] > 0.5).astype(np.uint8))
            hand_frame = overlay_mask_on_image(sampled_rgbs_np[picked_frame_idx], sampled_hand_masks_np[picked_frame_idx], cmap_idx=1)

            # Overlay rendered RGB (with both object and hand) on original image for better geometry visualization
            rgb_render = rendered_rgb[0, ..., :3]  # (H, W, 3)
            alpha_render = rendered_rgb[0, ..., 3:4]  # (H, W, 1)
            rgb_overlay = sampled_rgbs_np[picked_frame_idx] * (1 - alpha_render) + rgb_render * alpha_render

            panels = [modal_frame, amodal_gt_frame, render_frame, hand_frame, rgb_overlay]
            combined_image = (np.hstack(panels) * 255).astype(np.uint8)
            save_path = os.path.join(output_path, f"stage1_start_static_frame_{picked_frame_idx}.png")
            imageio.imwrite(save_path, combined_image)
            print(f"Saved stage 1 visualization to {save_path}")

        # NOTE no problem until here ####################################################################
        # NOTE no problem until here ####################################################################
        # NOTE no problem until here ####################################################################
        # NOTE no problem until here ####################################################################
        # NOTE no problem until here ####################################################################
        # NOTE no problem until here ####################################################################
        # NOTE no problem until here ####################################################################
        # NOTE no problem until here ####################################################################
        # NOTE no problem until here ####################################################################
        # NOTE no problem until here ####################################################################
        # NOTE no problem until here ####################################################################
        # NOTE no problem until here ####################################################################

        # raise Exception("Stop here")

        # ========================================================================
        # STAGE 3: Initialize PnP from the last frame of 0-1 segment
        # ========================================================================
        print("\n" + "=" * 80)
        print(f"STAGE 3: Running PnP from picked frame {picked_frame_idx} back and forth")
        print("=" * 80)

        verts_canonical_scaled = verts * (single_frame_model.scale.detach() * single_frame_model.initial_scale).unsqueeze(0)
        mesh_canonical_scaled = Meshes(
            verts=[verts_canonical_scaled],
            faces=[faces],
            textures=TexturesVertex(verts_features=torch.ones_like(verts)[None]),
        )

        K = np.array([[fx_new.item(), 0, cx_new.item()], [0, fy_new.item(), cy_new.item()], [0, 0, 1]])

        # Convert optimized 6D rotation to rotation matrix for PnP
        optimized_R = rotation_6d_to_matrix(single_frame_model.rot_6d.detach().unsqueeze(0)).squeeze(0)  # (3, 3)
        optimized_T = single_frame_model.trans.squeeze(0).detach()  # (3,)

        # ========================================================================
        # PnP forward pass start
        # ========================================================================
        # Use the picked frame to generate queries for PnP
        queries_2d, queries_3d = generate_queries_1stage(posed_mesh,
                                                         mesh_canonical_scaled,
                                                         sampled_modal_masks_np[picked_frame_idx],
                                                         focal_length[picked_frame_idx, None],
                                                         principal_point[picked_frame_idx, None],
                                                         grid_size=15,
                                                         device=device)

        # Run PnP starting from the anchor frame
        pnp_poses = run_pnp_1stage(
            cotracker_model,
            sampled_rgbs[picked_frame_idx:],  # picked frame to the end
            queries_2d,
            queries_3d,
            sampled_modal_masks_np[picked_frame_idx:],
            K,
            optimized_R,  # Use optimized rotation matrix as initial
            optimized_T,  # Use optimized translation as initial
            device,
            output_dir=os.path.join(output_path, "pnp_forward_visualization"),
            vis_threshold=0.0)

        forward_pnp_poses = torch.from_numpy(np.stack(pnp_poses, axis=0)).float().to(device)  # (M, 4, 4) where M = num_sampled_frames - picked_frame_idx

        # ========================================================================
        # PnP backward pass start
        # ========================================================================
        pnp_poses = run_pnp_1stage(
            cotracker_model,
            sampled_rgbs[:picked_frame_idx].flip(dims=[0]),  # picked frame to the start
            queries_2d,
            queries_3d,
            sampled_pred_amodal_masks_np[:picked_frame_idx][::-1],
            K,
            optimized_R,  # Use optimized rotation matrix as initial
            optimized_T,  # Use optimized translation as initial
            device,
            output_dir=os.path.join(output_path, "pnp_backward_visualization"),
            vis_threshold=0.0)

        backward_pnp_poses = torch.from_numpy(np.stack(pnp_poses, axis=0)).float().to(device).flip(dims=[0])  # (M, 4, 4) where M = picked_frame_idx

        pnp_pose = torch.cat([backward_pnp_poses, forward_pnp_poses], dim=0)  # (num_sampled_frames, 4, 4)
        # ========================================================================

        # Fill picked frame with the optimized pose
        # NOTE: PnP returns column-major rotation (OpenCV), but PyTorch3D uses row-major
        # So we need to store the optimized rotation in column-major format to match PnP
        picked_frame_rot = rotation_6d_to_matrix(single_frame_model.rot_6d.detach())  # (1, 3, 3) row-major
        picked_frame_rot_col_major = picked_frame_rot.transpose(1, 2)  # Convert to column-major to match PnP format
        picked_frame_trans = single_frame_model.trans.detach()  # (1, 3)

        pnp_pose[picked_frame_idx, :3, :3] = picked_frame_rot_col_major[0]
        pnp_pose[picked_frame_idx, :3, 3] = picked_frame_trans[0]
        pnp_pose[picked_frame_idx, 3, 3] = 1.0

        pnp_rot_mat = pnp_pose[:, :3, :3]  # Column-major format
        pnp_t = pnp_pose[:, :3, 3]

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
        initial_R_mat = single_frame_model.rot_6d.detach()
        initial_R_mat = rotation_6d_to_matrix(initial_R_mat).repeat(num_sampled_frames, 1, 1)
        initial_R_mat = pnp_rot_mat.mT

        # calculate the initial translation for the whole sequence
        initial_T = init_relative_translation(single_frame_model.trans.squeeze(0).cpu().detach().numpy(),
                                              sampled_pred_amodal_masks_np,
                                              fx_new.item(),
                                              fy_new.item(),
                                              init_frame_idx=picked_frame_idx)
        initial_T = torch.from_numpy(initial_T).float().to(device)

        initial_scale = single_frame_model.scale.detach() * single_frame_model.initial_scale

        # Prepare mesh data for the batch
        verts_batch = verts.unsqueeze(0).repeat(num_sampled_frames, 1, 1)
        faces_batch = faces.unsqueeze(0).repeat(num_sampled_frames, 1, 1)

        # --- Frame Locking for object pose optimization (picked frame)---
        lock_indices = [picked_frame_idx]
        lock_hook = create_lock_frames_hook(lock_indices)
        print(f"\nFrame locking: Locking object pose for picked frame {picked_frame_idx}")
        print(f"  Locked frames: {lock_indices}")
        print(f"  Total locked: {len(lock_indices)} frames")
        print(f"  Unlocked frames: {list(range(0, picked_frame_idx)) + list(range(picked_frame_idx + 1, num_sampled_frames))}")

        # --- Model and Optimizer ---
        sampled_metric_depths = torch.from_numpy(sampled_metric_depths_np[..., 0]).float().to(device)  # (N, H, W)

        multi_frame_model = TemporalHandObjectPose(initial_R_mat,
                                                   initial_T,
                                                   initial_scale,
                                                   verts_batch,
                                                   faces_batch,
                                                   sampled_mano_params,
                                                   obj_textures=original_textures).to(device)
        # multi_frame_model.rot_6d.register_hook(lock_hook)
        # multi_frame_model.trans.register_hook(lock_hook)

        # NOTE use hamer output for metric computation
        # NOTE use hamer output for metric computation
        # NOTE use hamer output for metric computation

        print('Using hamer output for metric computation')
        seq_name_short = os.path.basename(seq_path)
        hamer_output = f'../../output/{seq_name_short}/processed/hold_fit.slerp.npy'
        hamer_output = np.load(hamer_output, allow_pickle=True).item()
        hamer_output = hamer_output['right']
        hamer_mano_global_orient = torch.from_numpy(hamer_output['global_orient'])  # (N, 3)
        hamer_mano_trans = torch.from_numpy(hamer_output['transl'])  # (N, 3)
        hamer_mano_pose = torch.from_numpy(hamer_output['hand_pose'])  # (N, 45)
        hamer_mano_betas = torch.from_numpy(hamer_output['betas'])  # (N, 10)

        import pickle
        with open('../../stage1_preprocess/Dyn_HaMR_new/_DATA/data/mano/MANO_RIGHT.pkl', "rb") as mano_file:
            mano_data = pickle.load(mano_file, encoding='latin1')
        flat_hand_mean = torch.from_numpy(mano_data['hands_mean']).float()  # (15, 3)
        # # convert from flat_hand_mean=False to flat_hand_mean=True
        # hamer_mano_pose = hamer_mano_pose.reshape(-1, 15, 3)  # (N, 45) -> (N, 15, 3)
        # hamer_mano_pose = hamer_mano_pose - flat_hand_mean[None, :]  # (N, 15, 3)

        # from magichoi_mano.body_models import MANO
        # hamer_mano_layer = MANO(model_path='../../stage1_preprocess/Dyn_HaMR_new/_DATA/data/mano',
        #                         is_rhand=True,
        #                         flat_hand_mean=False,
        #                         use_pca=False)
        # hamer_mano_output = hamer_mano_layer(global_orient=hamer_mano_global_orient,
        #                                      transl=hamer_mano_trans,
        #                                      hand_pose=hamer_mano_pose,
        #                                      betas=hamer_mano_betas)

        # hamer_mano_joints = hamer_mano_output.joints
        # mano_dir = '../../stage1_preprocess/Dyn_HaMR_new/_DATA/data'
        # amano_cfg = {
        #     'mano_assets_root': os.path.join(mano_dir, 'mano'),
        #     'flat_hand_mean': True,
        #     'use_pca': False,
        #     'side': 'right',
        # }
        # amano_layer = AMANOLayer(**amano_cfg)
        hamer_mano_pose = hamer_mano_pose + flat_hand_mean[None, :]
        with torch.no_grad():
            multi_frame_model.mano_root_orient.data = hamer_mano_global_orient.to(device)
            multi_frame_model.mano_trans.data = hamer_mano_trans.to(device)
            multi_frame_model.mano_pose.data = hamer_mano_pose.to(device)
            multi_frame_model.mano_betas.data = hamer_mano_betas.to(device)

        # amano_output = run_amano(amano_layer, hamer_mano_trans[None], hamer_mano_global_orient, hamer_mano_pose, self.is_right, hamer_mano_betas)
        # amano_joints = amano_output['joints'].squeeze(0)
        # mano_to_openpose = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]
        # openpose_to_mano = [mano_to_openpose.index(i) for i in range(21)]
        # amano_joints = amano_joints.clone().detach()[:, openpose_to_mano]

        # Detach all parameters to clear any lingering computation graphs from previous stages
        # This ensures we start with a clean computation graph
        with torch.no_grad():

            multi_frame_model.rot_6d.data = multi_frame_model.rot_6d.data.detach().clone()
            multi_frame_model.trans.data = multi_frame_model.trans.data.detach().clone()
            multi_frame_model.mano_root_orient.data = multi_frame_model.mano_root_orient.data.detach().clone()
            multi_frame_model.mano_trans.data = multi_frame_model.mano_trans.data.detach().clone()
            multi_frame_model.mano_pose.data = multi_frame_model.mano_pose.data.detach().clone()
            multi_frame_model.mano_betas.data = multi_frame_model.mano_betas.data.detach().clone()
            multi_frame_model.scale.data = multi_frame_model.scale.data.detach().clone()

        # --- Model and Optimizer ---
        hand_optimizer = torch.optim.Adam([multi_frame_model.mano_trans], lr=1e-2)
        num_steps = 10000
        loop = tqdm(range(num_steps), desc="Optimizing Pose Sequence")
        for step in loop:
            hand_optimizer.zero_grad()

            posed_obj_meshes_batch, posed_hand_meshes_batch, mano_joints_batch = multi_frame_model()

            # mano_joints_batch = hamer_mano_joints + multi_frame_model.mano_trans[:, None]

            # Recompute loss_joints_2d in the loop to avoid reusing old computation graph
            projected_hand_joints = camera.transform_points_screen(mano_joints_batch, image_size=((H_out, W_out),))  # (N, 21, 3)
            pred_hand_joints_2d = projected_hand_joints[..., :2]  # (N, 21, 2)
            target_hand_joints_2d = sampled_gt_hand_joints_2d  # (N, 21, 2)
            mano_to_openpose = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]
            target_hand_joints_2d = target_hand_joints_2d[:, mano_to_openpose, :]  # (N, 21, 2)
            loss_joints_2d = torch.nn.functional.mse_loss(pred_hand_joints_2d, target_hand_joints_2d)

            loss_joints_2d_weight = 1e0
            total_loss = loss_joints_2d * loss_joints_2d_weight

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_([multi_frame_model.mano_trans], max_norm=1.0)
            hand_optimizer.step()

            loop.set_postfix(
                loss=total_loss.item(),
                loss_joints_2d=(loss_joints_2d * loss_joints_2d_weight).item(),
            )

        # NOTE use hamer output for metric computation
        # NOTE use hamer output for metric computation
        # NOTE use hamer output for metric computation

        # Detach all parameters to clear any lingering computation graphs from previous stages
        # This ensures we start with a clean computation graph
        with torch.no_grad():
            multi_frame_model.rot_6d.data = multi_frame_model.rot_6d.data.detach().clone()
            multi_frame_model.trans.data = multi_frame_model.trans.data.detach().clone()
            multi_frame_model.mano_root_orient.data = multi_frame_model.mano_root_orient.data.detach().clone()
            multi_frame_model.mano_trans.data = multi_frame_model.mano_trans.data.detach().clone()
            multi_frame_model.mano_pose.data = multi_frame_model.mano_pose.data.detach().clone()
            multi_frame_model.mano_betas.data = multi_frame_model.mano_betas.data.detach().clone()
            multi_frame_model.scale.data = multi_frame_model.scale.data.detach().clone()

        obj_optimizer = torch.optim.Adam([multi_frame_model.rot_6d, multi_frame_model.trans], lr=1e-3)
        # hand_optimizer = torch.optim.Adam([{
        #     'params': [multi_frame_model.mano_root_orient, multi_frame_model.mano_pose],
        #     'lr': 1e-3
        # }, {
        #     'params': [multi_frame_model.mano_trans],
        #     'lr': 1e-2
        # }])
        hand_optimizer = torch.optim.Adam([{'params': [multi_frame_model.mano_trans], 'lr': 1e-3}])
        loop = tqdm(range(2000), desc="Optimizing Pose Sequence")
        for step in loop:
            obj_optimizer.zero_grad()
            hand_optimizer.zero_grad()

            curr_sigma, curr_gamma, curr_fpp = get_render_params(step, 2000)
            renderer.rasterizer.raster_settings.blur_radius = curr_sigma
            renderer.rasterizer.raster_settings.faces_per_pixel = curr_fpp
            new_blend_params = BlendParams(sigma=curr_sigma, gamma=curr_gamma)
            renderer.shader.blend_params = new_blend_params

            posed_obj_meshes_batch, posed_hand_meshes_batch, mano_joints_batch = multi_frame_model()

            obj_fragments = renderer.rasterizer(posed_obj_meshes_batch)
            rendered_obj_masks = renderer.shader(obj_fragments, posed_obj_meshes_batch)[..., 3]

            # object loss
            loss_fp = weighted_false_positive_loss(rendered_obj_masks, sampled_pred_amodal_masks) * 1e2
            loss_vp = vectorized_pose_guiding_loss(posed_obj_meshes_batch, sampled_pred_amodal_masks, rendered_obj_masks, camera, num_samples=2000)
            # loss_mse = torch.nn.functional.mse_loss(rendered_obj_masks, sampled_pred_amodal_masks) * 1e2
            loss_mse = loss_fp + loss_vp
            loss_sm_rot = compute_smoothness_loss(multi_frame_model.rot_6d)
            if loss_sm_rot < 1:
                loss_sm_rot = torch.tensor(0.0, device=device)
            loss_sm_trans = compute_smoothness_loss(multi_frame_model.trans)
            # if loss_sm_trans < 0.05 and step > 500:
            # if loss_sm_trans < 0.05:
            #     loss_sm_trans = torch.tensor(0.0, device=device)

            obj_loss = loss_mse + loss_sm_rot + loss_sm_trans

            # hand 2d joints loss
            projected_hand_joints = camera.transform_points_screen(mano_joints_batch, image_size=((H_out, W_out),))  # (N, 21, 3)
            pred_hand_joints_2d = projected_hand_joints[..., :2]

            target_hand_joints_2d = sampled_gt_hand_joints_2d
            mano_to_openpose = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]
            target_hand_joints_2d = target_hand_joints_2d[:, mano_to_openpose, :]
            loss_joints_2d = torch.nn.functional.l1_loss(pred_hand_joints_2d, target_hand_joints_2d)

            # # hand depth loss
            # hand_fragments = renderer.rasterizer(posed_hand_meshes_batch)  # (N, H, W, 1)
            # hand_depth = hand_fragments.zbuf[..., 0]  # (N, H, W)
            # hand_valid_mask = hand_depth > 0  # (N, H, W)

            # loss_hand_depth = torch.nn.functional.l1_loss(hand_depth[hand_valid_mask], sampled_metric_depths[hand_valid_mask])

            loss_sm_hand = compute_smoothness_loss(mano_joints_batch)

            # hand anatomy loss
            T_g_p = multi_frame_model.transforms_abs  # (B, 16, 4, 4)
            T_g_a, _R, ee = multi_frame_model.axisFK(T_g_p)  # ee (B, 16, 3)
            loss_anatomy = multi_frame_model.anatomyLoss(ee) * 1e2

            # hand_loss = loss_joints_2d + loss_sm_hand * smoothness_weight + loss_anatomy
            hand_loss = loss_joints_2d
            total_loss = obj_loss + hand_loss + loss_sm_hand

            total_loss.backward()
            # Clip gradients separately for each optimizer to avoid interference
            torch.nn.utils.clip_grad_norm_([multi_frame_model.rot_6d, multi_frame_model.trans], max_norm=1.0)
            # torch.nn.utils.clip_grad_norm_([multi_frame_model.mano_root_orient, multi_frame_model.mano_trans, multi_frame_model.mano_pose], max_norm=1.0)
            torch.nn.utils.clip_grad_norm_([multi_frame_model.mano_trans], max_norm=1.0)
            obj_optimizer.step()
            hand_optimizer.step()
            loop.set_postfix(
                loss=total_loss.item(),
                loss_mse=loss_mse.item(),
                loss_sm_rot=loss_sm_rot.item(),
                loss_sm_trans=loss_sm_trans.item(),
                loss_joints_2d=loss_joints_2d.item(),
                # loss_hand_depth=loss_hand_depth.item(),
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
        # Save checkpoint for STAGE 1-3
        # ========================================================================
        print("\n" + "=" * 80)
        print("Saving checkpoint for STAGE 1-3...")
        print("=" * 80)

        # Convert rot_6d to rotation matrices for initial_R_mat
        saved_initial_R_mat = rotation_6d_to_matrix(multi_frame_model.rot_6d.data.detach())

        checkpoint = {
            # Segment boundaries and stage ranges
            'stage_ranges': stage_ranges,
            'segment_boundaries': segment_boundaries,
            'picked_frame_idx': picked_frame_idx,
            'start_static_end_idx': start_static_end_idx,
            'approaching_end_idx': approaching_end_idx,
            'interaction_end_idx': interaction_end_idx,
            'end_static_start_idx': end_static_start_idx,

            # Camera parameters
            'fx_new': fx_new.item() if isinstance(fx_new, torch.Tensor) else fx_new,
            'fy_new': fy_new.item() if isinstance(fy_new, torch.Tensor) else fy_new,
            'cx_new': cx_new.item() if isinstance(cx_new, torch.Tensor) else cx_new,
            'cy_new': cy_new.item() if isinstance(cy_new, torch.Tensor) else cy_new,

            # Model parameters (for reconstruction)
            'initial_R_mat': saved_initial_R_mat.cpu(),  # (N, 3, 3) rotation matrices
            'initial_T': multi_frame_model.trans.data.detach().cpu(),
            'initial_scale': multi_frame_model.initial_scale.detach().cpu(),

            # Optimized parameters
            'rot_6d': multi_frame_model.rot_6d.data.detach().cpu(),
            'trans': multi_frame_model.trans.data.detach().cpu(),
            'scale': multi_frame_model.scale.data.detach().cpu(),
            'mano_root_orient': multi_frame_model.mano_root_orient.data.detach().cpu(),
            'mano_trans': multi_frame_model.mano_trans.data.detach().cpu(),
            'mano_pose': multi_frame_model.mano_pose.data.detach().cpu(),
            'mano_betas': multi_frame_model.mano_betas.data.detach().cpu(),
        }

        torch.save(checkpoint, checkpoint_init_path)
        print(f"Checkpoint saved to {checkpoint_init_path}")

    if os.path.exists(checkpoint_scale_path) and not overwrite:
        print("\n" + "=" * 80)
        print("Loading checkpoint from STAGE 5...")
        print("=" * 80)
        checkpoint = torch.load(checkpoint_scale_path, map_location=device)

        # Load stage ranges and segment boundaries
        stage_ranges = checkpoint['stage_ranges']
        segment_boundaries = checkpoint['segment_boundaries']
        picked_frame_idx = checkpoint['picked_frame_idx']
        start_static_end_idx = checkpoint['start_static_end_idx']
        approaching_end_idx = checkpoint['approaching_end_idx']
        interaction_end_idx = checkpoint['interaction_end_idx']
        end_static_start_idx = checkpoint['end_static_start_idx']

        # Load camera parameters
        fx_new = checkpoint['fx_new']
        fy_new = checkpoint['fy_new']
        cx_new = checkpoint['cx_new']
        cy_new = checkpoint['cy_new']

        # Recreate focal_length and principal_point (already in device)
        focal_length = torch.tensor([[fx_new, fy_new]], device=device).expand(num_sampled_frames, -1)
        principal_point = torch.tensor([[cx_new, cy_new]], device=device).expand(num_sampled_frames, -1)

        # Recreate verts_batch and faces_batch
        verts_batch = verts.unsqueeze(0).repeat(num_sampled_frames, 1, 1)
        faces_batch = faces.unsqueeze(0).repeat(num_sampled_frames, 1, 1)

        # Recreate multi_frame_model with loaded parameters
        multi_frame_model = TemporalHandObjectPose(checkpoint['initial_R_mat'].to(device),
                                                   checkpoint['initial_T'].to(device),
                                                   checkpoint['initial_scale'].to(device),
                                                   verts_batch,
                                                   faces_batch,
                                                   sampled_mano_params,
                                                   obj_textures=original_textures).to(device)

        # Load optimized parameters
        multi_frame_model.rot_6d.data = checkpoint['rot_6d'].to(device)
        multi_frame_model.trans.data = checkpoint['trans'].to(device)
        multi_frame_model.scale.data = checkpoint['scale'].to(device)
        multi_frame_model.mano_root_orient.data = checkpoint['mano_root_orient'].to(device)
        multi_frame_model.mano_trans.data = checkpoint['mano_trans'].to(device)
        multi_frame_model.mano_pose.data = checkpoint['mano_pose'].to(device)
        if 'mano_betas' in checkpoint:
            multi_frame_model.mano_betas.data = checkpoint['mano_betas'].to(device)
        else:
            multi_frame_model.mano_betas.data = torch.zeros([num_sampled_frames, 10], dtype=torch.float32, device=device)

        print(f"Checkpoint loaded successfully!")
        print(f"  Picked frame: {picked_frame_idx}")
        print(f"  Segment boundaries: {segment_boundaries}")
        print("Skipping STAGE 5, continuing to STAGE 6...")
    else:
        print("Doing scale optimization")

        # Detach all parameters to clear any lingering computation graphs from previous stages
        # This ensures we start with a clean computation graph
        with torch.no_grad():

            multi_frame_model.rot_6d.data = multi_frame_model.rot_6d.data.detach().clone()
            multi_frame_model.trans.data = multi_frame_model.trans.data.detach().clone() + 0.5
            multi_frame_model.scale.data = multi_frame_model.scale.data.detach().clone()
            multi_frame_model.mano_root_orient.data = multi_frame_model.mano_root_orient.data.detach().clone()
            multi_frame_model.mano_trans.data = multi_frame_model.mano_trans.data.detach().clone()
            multi_frame_model.mano_pose.data = multi_frame_model.mano_pose.data.detach().clone()
            multi_frame_model.mano_betas.data = multi_frame_model.mano_betas.data.detach().clone()

        # Find picked_frame_idx in sampled_indices to get batch_idx
        picked_batch_idx = sampled_indices.index(picked_frame_idx)
        print(f"Optimizing only picked frame {picked_frame_idx} (batch_idx={picked_batch_idx})")

        camera = PerspectiveCameras(
            focal_length=focal_length[picked_batch_idx:picked_batch_idx + 1],
            principal_point=principal_point[picked_batch_idx:picked_batch_idx + 1],
            image_size=((H_out, W_out),),
            in_ndc=False,
            device=device,
        )
        # Use bin_size=0 for naive rasterization to avoid overflow issues during optimization
        # This is more stable when mesh scale changes during optimization
        raster_settings = RasterizationSettings(
            image_size=(H_out, W_out),
            blur_radius=1e-4,
            faces_per_pixel=20,
        )
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=camera, raster_settings=raster_settings),
            shader=SoftSilhouetteShader(blend_params=BlendParams(sigma=1e-4, gamma=1e-4)),
        )

        # --- Model and Optimizer ---
        # obj_optimizer = torch.optim.Adam([multi_frame_model.rot_6d, multi_frame_model.trans], lr=1e-3)
        obj_optimizer = torch.optim.Adam([multi_frame_model.trans, multi_frame_model.scale], lr=1e-2)
        num_steps = 500
        loop = tqdm(range(num_steps), desc=f"Optimizing object scale (frame {picked_frame_idx})")
        for step in loop:
            obj_optimizer.zero_grad()
            # hand_optimizer.zero_grad()

            curr_sigma, curr_gamma, curr_fpp = get_render_params(step, num_steps)
            # curr_sigma, curr_gamma, curr_fpp = 1e-4, 1e-4, 20
            renderer.rasterizer.raster_settings.blur_radius = curr_sigma
            renderer.rasterizer.raster_settings.faces_per_pixel = curr_fpp
            new_blend_params = BlendParams(sigma=curr_sigma, gamma=curr_gamma)
            renderer.shader.blend_params = new_blend_params

            # Get all frames but only use picked frame for optimization
            posed_obj_meshes_batch, posed_hand_meshes_batch, mano_joints_batch = multi_frame_model()

            # Extract only the picked frame
            posed_obj_mesh_picked = posed_obj_meshes_batch[picked_batch_idx:picked_batch_idx + 1]
            posed_hand_mesh_picked = posed_hand_meshes_batch[picked_batch_idx:picked_batch_idx + 1]

            obj_fragments = renderer.rasterizer(posed_obj_mesh_picked)
            rendered_obj_masks = renderer.shader(obj_fragments, posed_obj_mesh_picked)[..., 3]
            # obj_zbuf = obj_fragments.zbuf[..., 0]

            # object loss - only for picked frame
            picked_frame_mask = sampled_pred_amodal_masks[picked_batch_idx:picked_batch_idx + 1]
            # loss_mse = torch.nn.functional.mse_loss(rendered_obj_masks, picked_frame_mask)
            loss_fp = weighted_false_positive_loss(rendered_obj_masks, picked_frame_mask) * 1e2
            loss_vp = vectorized_pose_guiding_loss(posed_obj_mesh_picked, picked_frame_mask, rendered_obj_masks, camera, num_samples=2000)
            # loss_mse = torch.nn.functional.mse_loss(rendered_obj_masks, picked_frame_mask) * 1e2
            loss_mse = loss_fp + loss_vp
            loss_sm_rot = compute_smoothness_loss(multi_frame_model.rot_6d)
            # if loss_sm_rot < 1:
            #     loss_sm_rot = torch.tensor(0.0, device=device)
            loss_sm_trans = compute_smoothness_loss(multi_frame_model.trans)

            # Get batch vertices (needed for both contact loss and collision loss) - only picked frame
            obj_verts_picked = posed_obj_mesh_picked.verts_padded()  # (1, N_obj, 3)
            hand_verts_picked = posed_hand_mesh_picked.verts_padded()  # (1, N_hand, 3)

            # Get batch normals: all frames have same canonical object, so just reshape packed normals
            obj_normals_packed = posed_obj_mesh_picked.verts_normals_packed()  # (1*N_obj, 3)
            num_obj_verts = obj_verts_picked.shape[1]
            obj_normals_picked = obj_normals_packed.view(1, num_obj_verts, 3)  # (1, N_obj, 3)

            # Get hand normals
            hand_normals_packed = posed_hand_mesh_picked.verts_normals_packed()  # (1*N_hand, 3)
            num_hand_verts = hand_verts_picked.shape[1]
            hand_normals_picked = hand_normals_packed.view(1, num_hand_verts, 3)  # (1, N_hand, 3)

            # Compute bidirectional collision loss
            # 1. Hand vertices penetrating object
            loss_hand_in_obj = compute_collision_loss(
                obj_verts_picked,  # (1, N_obj, 3)
                obj_normals_picked,  # (1, N_obj, 3)
                hand_verts_picked.detach(),  # (1, N_hand, 3) NOTE detach hand here
                ignore_indices=None  # Note: batch ignore_indices not supported yet
            )

            # 2. Object vertices penetrating hand
            loss_obj_in_hand = compute_collision_loss(
                hand_verts_picked.detach(),  # (1, N_hand, 3) NOTE detach hand here
                hand_normals_picked.detach(),  # (1, N_hand, 3) NOTE detach hand here
                obj_verts_picked,  # (1, N_obj, 3)
                ignore_indices=None)

            # Total bidirectional collision loss
            if step > (num_steps * 0.0):
                loss_collision = (loss_hand_in_obj + loss_obj_in_hand)
            else:
                loss_collision = torch.tensor(0.0, device=device)

            # nearest distance loss
            hand_nn_dist, hand_nn_idx = get_NN(obj_verts_picked, hand_verts_picked)
            hand_in_obj_penetr_dist = hand_nn_dist.sum()
            loss_nearest_distance = hand_in_obj_penetr_dist

            loss_mse_weight = 1e0
            loss_sm_rot_weight = 1e1
            loss_sm_trans_weight = 1e1
            loss_nearest_distance_weight = 1e0
            loss_collision_weight = 0

            total_loss = \
            loss_mse * loss_mse_weight + \
            loss_sm_rot * loss_sm_rot_weight + \
            loss_sm_trans * loss_sm_trans_weight + \
            loss_nearest_distance * loss_nearest_distance_weight + \
            loss_collision * loss_collision_weight

            total_loss.backward()
            # Clip gradients separately for each optimizer to avoid interference
            torch.nn.utils.clip_grad_norm_([multi_frame_model.trans, multi_frame_model.scale], max_norm=1.0)
            obj_optimizer.step()

            loop.set_postfix(
                loss=total_loss.item(),
                loss_mse=(loss_mse * loss_mse_weight).item(),
                loss_sm_rot=(loss_sm_rot * loss_sm_rot_weight).item(),
                loss_sm_trans=(loss_sm_trans * loss_sm_trans_weight).item(),
                loss_nearest_distance=(loss_nearest_distance * loss_nearest_distance_weight).item(),
                loss_collision=(loss_collision * loss_collision_weight).item(),
                scale=(multi_frame_model.scale * multi_frame_model.initial_scale)[0].item(),
            )

        # Detach all parameters to clear any lingering computation graphs from previous stages
        # This ensures we start with a clean computation graph
        with torch.no_grad():

            multi_frame_model.rot_6d.data = multi_frame_model.rot_6d.data.detach().clone()
            multi_frame_model.trans.data = multi_frame_model.trans.data.detach().clone()
            multi_frame_model.scale.data = multi_frame_model.scale.data.detach().clone()
            multi_frame_model.mano_root_orient.data = multi_frame_model.mano_root_orient.data.detach().clone()
            multi_frame_model.mano_trans.data = multi_frame_model.mano_trans.data.detach().clone()
            multi_frame_model.mano_pose.data = multi_frame_model.mano_pose.data.detach().clone()
            multi_frame_model.mano_betas.data = multi_frame_model.mano_betas.data.detach().clone()

        camera = PerspectiveCameras(
            focal_length=focal_length,
            principal_point=principal_point,
            image_size=((H_out, W_out),),
            in_ndc=False,
            device=device,
        )
        # Use bin_size=0 for naive rasterization to avoid overflow issues during optimization
        # This is more stable when mesh scale changes during optimization
        raster_settings = RasterizationSettings(
            image_size=(H_out, W_out),
            blur_radius=1e-4,
            faces_per_pixel=20,
        )
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=camera, raster_settings=raster_settings),
            shader=SoftSilhouetteShader(blend_params=BlendParams(sigma=1e-4, gamma=1e-4)),
        )

        obj_optimizer = torch.optim.Adam([multi_frame_model.rot_6d, multi_frame_model.trans], lr=1e-3)
        num_steps = 1000
        loop = tqdm(range(num_steps), desc="Optimizing Pose Sequence After Scale")
        for step in loop:
            obj_optimizer.zero_grad()

            curr_sigma, curr_gamma, curr_fpp = get_render_params(num_steps - 1, num_steps)
            renderer.rasterizer.raster_settings.blur_radius = curr_sigma
            renderer.rasterizer.raster_settings.faces_per_pixel = curr_fpp
            new_blend_params = BlendParams(sigma=curr_sigma, gamma=curr_gamma)
            renderer.shader.blend_params = new_blend_params

            posed_obj_meshes_batch, posed_hand_meshes_batch, mano_joints_batch = multi_frame_model()

            obj_fragments = renderer.rasterizer(posed_obj_meshes_batch)
            rendered_obj_masks = renderer.shader(obj_fragments, posed_obj_meshes_batch)[..., 3]

            # object loss
            loss_fp = weighted_false_positive_loss(rendered_obj_masks, sampled_pred_amodal_masks) * 1e3
            loss_vp = vectorized_pose_guiding_loss(posed_obj_meshes_batch, sampled_pred_amodal_masks, rendered_obj_masks, camera, num_samples=1000)
            # loss_mse = torch.nn.functional.mse_loss(rendered_obj_masks, sampled_pred_amodal_masks) * 1e2
            loss_mse = loss_fp + loss_vp
            loss_sm_rot = compute_smoothness_loss(multi_frame_model.rot_6d)
            if loss_sm_rot < 1:
                loss_sm_rot = torch.tensor(0.0, device=device)
            loss_sm_trans = compute_smoothness_loss(multi_frame_model.trans)
            # if loss_sm_trans < 0.05 and step > 500:
            # if loss_sm_trans < 0.05:
            #     loss_sm_trans = torch.tensor(0.0, device=device)

            obj_loss = loss_mse + loss_sm_rot + loss_sm_trans

            total_loss = obj_loss

            total_loss.backward()
            # Clip gradients separately for each optimizer to avoid interference
            torch.nn.utils.clip_grad_norm_([multi_frame_model.rot_6d, multi_frame_model.trans], max_norm=1.0)
            obj_optimizer.step()
            loop.set_postfix(
                loss=total_loss.item(),
                loss_mse=loss_mse.item(),
                loss_sm_rot=loss_sm_rot.item(),
                loss_sm_trans=loss_sm_trans.item(),
            )
        scale_json = os.path.join(output_path, "optimized_scale.json")
        with open(scale_json, 'w') as f:
            json.dump({'scale': (multi_frame_model.scale.data * multi_frame_model.initial_scale)[0].detach().cpu().item()}, f)
        print(f"Saved optimized scale to {scale_json}")

        # ========================================================================
        # Save checkpoint for STAGE 5
        # ========================================================================
        print("\n" + "=" * 80)
        print("Saving checkpoint for STAGE 5...")
        print("=" * 80)

        # Convert rot_6d to rotation matrices for initial_R_mat
        saved_initial_R_mat = rotation_6d_to_matrix(multi_frame_model.rot_6d.data.detach())

        checkpoint = {
            # Segment boundaries and stage ranges
            'stage_ranges': stage_ranges,
            'segment_boundaries': segment_boundaries,
            'picked_frame_idx': picked_frame_idx,
            'start_static_end_idx': start_static_end_idx,
            'approaching_end_idx': approaching_end_idx,
            'interaction_end_idx': interaction_end_idx,
            'end_static_start_idx': end_static_start_idx,

            # Camera parameters
            'fx_new': fx_new.item() if isinstance(fx_new, torch.Tensor) else fx_new,
            'fy_new': fy_new.item() if isinstance(fy_new, torch.Tensor) else fy_new,
            'cx_new': cx_new.item() if isinstance(cx_new, torch.Tensor) else cx_new,
            'cy_new': cy_new.item() if isinstance(cy_new, torch.Tensor) else cy_new,

            # Model parameters (for reconstruction)
            'initial_R_mat': saved_initial_R_mat.cpu(),  # (N, 3, 3) rotation matrices
            'initial_T': multi_frame_model.trans.data.detach().cpu(),
            'initial_scale': multi_frame_model.initial_scale.detach().cpu(),

            # Optimized parameters
            'rot_6d': multi_frame_model.rot_6d.data.detach().cpu(),
            'trans': multi_frame_model.trans.data.detach().cpu(),
            'scale': multi_frame_model.scale.data.detach().cpu(),
            'mano_root_orient': multi_frame_model.mano_root_orient.data.detach().cpu(),
            'mano_trans': multi_frame_model.mano_trans.data.detach().cpu(),
            'mano_pose': multi_frame_model.mano_pose.data.detach().cpu(),
            'mano_betas': multi_frame_model.mano_betas.data.detach().cpu(),
        }

        torch.save(checkpoint, checkpoint_scale_path)
        print(f"Checkpoint saved to {checkpoint_scale_path}")

    # ========================================================================
    # STAGE 4: Load Contact Correspondence (from Grasping Pose Correction)
    # ========================================================================
    print("\n" + "=" * 80)
    print("STAGE 4: Loading Contact Correspondence")
    print("=" * 80)

    contact_map_path = os.path.join(output_path, "grasp_correction/contact_map.npy")

    if os.path.exists(contact_map_path):
        print(f"Found contact map at: {contact_map_path}")
        contact_map_all_frames = np.load(contact_map_path, allow_pickle=True).item()

        # contact_map_all_frames is now a dict: {frame_idx: contact_map}
        # Parse each frame's contact map
        parsed_contact_maps = {}
        total_contacts = 0

        for frame_idx_str, frame_contact_map in contact_map_all_frames.items():
            # Convert frame key to int if needed
            frame_idx = int(frame_idx_str) if isinstance(frame_idx_str, str) else frame_idx_str
            parsed_contact_maps[frame_idx] = parse_contact_map(frame_contact_map)
            total_contacts += parsed_contact_maps[frame_idx]['num_contacts']

        print(f"Loaded contact maps for {len(parsed_contact_maps)} frames")
        print(f"Total contact correspondences: {total_contacts}")

        # Show some examples from first available frame
        if len(parsed_contact_maps) > 0:
            first_frame_idx = min(parsed_contact_maps.keys())
            first_parsed = parsed_contact_maps[first_frame_idx]
            print(f"\nExample correspondences from frame {first_frame_idx}:")
            for i, (hand_idx, obj_vertex_idx) in enumerate(list(first_parsed['correspondences'].items())[:5]):
                print(f"  Hand vertex {hand_idx} -> Object vertex {obj_vertex_idx}")
            if first_parsed['num_contacts'] > 5:
                print(f"  ... and {first_parsed['num_contacts'] - 5} more")

        # For backward compatibility, keep contact_map and parsed_contact_map
        # Use the first available frame's contact map as default
        parsed_contact_map = parsed_contact_maps.get(min(parsed_contact_maps.keys())) if parsed_contact_maps else None
    else:
        print(f"No contact map found at: {contact_map_path}")
        print("Skipping contact-based optimization")
        contact_map = None
        parsed_contact_map = None
        parsed_contact_maps = None

    # ========================================================================
    # STAGE 5: Full sequence optimization with contact map
    # ========================================================================
    if parsed_contact_map is not None:
        print("\n" + "=" * 80)
        print("STAGE 5: Full sequence optimization with contact map")
        print("=" * 80)
        # Collect all contact point indices across all batches
        batch_indices_list = []  # which batch each contact point belongs to
        hand_indices_list = []  # hand vertex index for each contact point
        obj_indices_list = []  # object vertex index for each contact point

        for batch_idx in range(num_sampled_frames):
            frame_idx = sampled_indices[batch_idx]
            frame_contact_map = parsed_contact_maps.get(frame_idx)

            if frame_contact_map is not None and frame_contact_map['num_contacts'] > 0:
                correspondences = frame_contact_map['correspondences']  # {hand_vertex_idx: object_vertex_idx}

                num_contacts = len(correspondences)
                batch_indices_list.extend([batch_idx] * num_contacts)
                hand_indices_list.extend(list(correspondences.keys()))
                obj_indices_list.extend(list(correspondences.values()))

        # Vectorized computation: get all contact vertices at once
        batch_indices = torch.tensor(batch_indices_list, dtype=torch.long, device=device)  # (Total_N,)
        hand_indices = torch.tensor(hand_indices_list, dtype=torch.long, device=device)  # (Total_N,)
        obj_indices = torch.tensor(obj_indices_list, dtype=torch.long, device=device)  # (Total_N,)

        # ========================================================================
        # DEBUG: Visualize contact correspondences for each frame
        # ========================================================================
        print("\n" + "=" * 80)
        print("DEBUG: Saving contact correspondence visualizations")
        print("=" * 80)

        contact_vis_dir = os.path.join(output_path, "contact_correspondence_debug")
        os.makedirs(contact_vis_dir, exist_ok=True)

        with torch.no_grad():
            # Get current posed meshes
            posed_obj_meshes_batch, posed_hand_meshes_batch, _ = multi_frame_model()
            obj_verts_batch = posed_obj_meshes_batch.verts_padded()  # (B, N_obj, 3)
            hand_verts_batch = posed_hand_meshes_batch.verts_padded()  # (B, N_hand, 3)

            # Iterate through each frame that has contact correspondences
            all_contact_distances = []
            for batch_idx in range(num_sampled_frames):
                frame_idx = sampled_indices[batch_idx]
                frame_contact_map = parsed_contact_maps.get(frame_idx)

                if frame_contact_map is not None and frame_contact_map['num_contacts'] > 0:
                    correspondences = frame_contact_map['correspondences']  # {hand_vertex_idx: object_vertex_idx}

                    # Get hand and object vertices for this frame
                    hand_verts = hand_verts_batch[batch_idx].cpu().numpy()  # (N_hand, 3)
                    obj_verts = obj_verts_batch[batch_idx].cpu().numpy()  # (N_obj, 3)

                    # Extract contact points
                    hand_contact_indices = list(correspondences.keys())
                    obj_contact_indices = list(correspondences.values())

                    hand_contact_pts = hand_verts[hand_contact_indices]  # (N_contacts, 3)
                    obj_contact_pts = obj_verts[obj_contact_indices]  # (N_contacts, 3)
                    contact_distances = np.linalg.norm(hand_contact_pts - obj_contact_pts, axis=-1).sum().item()  # (N_contacts,)
                    all_contact_distances.append(contact_distances)

                    # Generate unique colors for each correspondence pair using HSV colormap
                    num_contacts = len(hand_contact_indices)
                    import matplotlib.cm as cm
                    colormap = cm.get_cmap('hsv')

                    # Create colors for contact points - each pair gets the same unique color
                    hand_contact_colors = []
                    obj_contact_colors = []
                    for i in range(num_contacts):
                        # Generate a unique color for this correspondence pair
                        color = colormap(i / max(num_contacts, 1))[:3]  # RGB only
                        hand_contact_colors.append(color)
                        obj_contact_colors.append(color)

                    hand_contact_colors = np.array(hand_contact_colors)
                    obj_contact_colors = np.array(obj_contact_colors)

                    # Combine all points: hand mesh (gray), object mesh (blue), hand contacts (unique colors), obj contacts (same unique colors)
                    all_points = np.vstack([hand_verts, obj_verts, hand_contact_pts, obj_contact_pts])

                    # Create colors
                    hand_colors = np.tile([0.5, 0.5, 0.5], (hand_verts.shape[0], 1))  # Gray for hand mesh
                    obj_colors = np.tile([0.3, 0.3, 1.0], (obj_verts.shape[0], 1))  # Blue for object mesh

                    all_colors = np.vstack([hand_colors, obj_colors, hand_contact_colors, obj_contact_colors])
                    all_colors_uint8 = (all_colors * 255).astype(np.uint8)

                    # Create point cloud
                    pcd = trimesh.points.PointCloud(all_points, colors=all_colors_uint8)

                    # Save the point cloud
                    ply_path = os.path.join(contact_vis_dir, f"contact_correspondence_frame_{frame_idx:04d}.ply")
                    pcd.export(ply_path)
                    print(f"  Frame {frame_idx}: Saved {len(correspondences)} correspondences to {ply_path}")

        print(f"Saved contact correspondence visualizations to {contact_vis_dir}")
        print("=" * 80)

        # Detach all parameters to clear any lingering computation graphs from previous stages
        # This ensures we start with a clean computation graph
        with torch.no_grad():

            multi_frame_model.rot_6d.data = multi_frame_model.rot_6d.data.detach().clone()
            multi_frame_model.trans.data = multi_frame_model.trans.data.detach().clone()
            multi_frame_model.mano_root_orient.data = multi_frame_model.mano_root_orient.data.detach().clone()
            multi_frame_model.mano_trans.data = multi_frame_model.mano_trans.data.detach().clone()
            multi_frame_model.mano_pose.data = multi_frame_model.mano_pose.data.detach().clone()
            multi_frame_model.mano_betas.data = multi_frame_model.mano_betas.data.detach().clone()
            multi_frame_model.scale.data = multi_frame_model.scale.data.detach().clone()

        camera = PerspectiveCameras(
            focal_length=focal_length,
            principal_point=principal_point,
            image_size=((H_out, W_out),),
            in_ndc=False,
            device=device,
        )
        # Use bin_size=0 for naive rasterization to avoid overflow issues during optimization
        # This is more stable when mesh scale changes during optimization
        raster_settings = RasterizationSettings(
            image_size=(H_out, W_out),
            blur_radius=1e-4,
            faces_per_pixel=20,
        )
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=camera, raster_settings=raster_settings),
            shader=SoftSilhouetteShader(blend_params=BlendParams(sigma=1e-4, gamma=1e-4)),
        )

        # --- Model and Optimizer ---
        # obj_optimizer = torch.optim.Adam([multi_frame_model.rot_6d, multi_frame_model.trans], lr=1e-3)
        obj_optimizer = torch.optim.Adam([
            {
                "params": [multi_frame_model.trans],
                "lr": 1e-3
            },
            {
                "params": [multi_frame_model.rot_6d],
                "lr": 1e-3
            },
            # {
            #     "params": [multi_frame_model.scale],
            #     "lr": 1e-3
            # },
        ])
        hand_optimizer = torch.optim.Adam([multi_frame_model.mano_trans, multi_frame_model.mano_pose], lr=1e-3)
        num_steps = 2000
        loop = tqdm(range(num_steps), desc="Optimizing Contact Pose Sequence")
        for step in loop:
            obj_optimizer.zero_grad()
            hand_optimizer.zero_grad()

            curr_sigma, curr_gamma, curr_fpp = get_render_params(num_steps - 1, num_steps)
            # curr_sigma, curr_gamma, curr_fpp = 1e-4, 1e-4, 20
            renderer.rasterizer.raster_settings.blur_radius = curr_sigma
            renderer.rasterizer.raster_settings.faces_per_pixel = curr_fpp
            new_blend_params = BlendParams(sigma=curr_sigma, gamma=curr_gamma)
            renderer.shader.blend_params = new_blend_params

            posed_obj_meshes_batch, posed_hand_meshes_batch, mano_joints_batch = multi_frame_model()

            obj_fragments = renderer.rasterizer(posed_obj_meshes_batch)
            rendered_obj_masks = renderer.shader(obj_fragments, posed_obj_meshes_batch)[..., 3]
            # obj_zbuf = obj_fragments.zbuf[..., 0]

            # object loss
            # loss_mse = torch.nn.functional.mse_loss(rendered_obj_masks, sampled_pred_amodal_masks)
            loss_fp = weighted_false_positive_loss(rendered_obj_masks, sampled_pred_amodal_masks) * 1e2
            loss_vp = vectorized_pose_guiding_loss(posed_obj_meshes_batch, sampled_pred_amodal_masks, rendered_obj_masks, camera, num_samples=2000)
            loss_mse = loss_fp + loss_vp
            # loss_mse = torch.nn.functional.mse_loss(rendered_obj_masks, sampled_pred_amodal_masks) * 1e2
            loss_sm_rot = compute_smoothness_loss(multi_frame_model.rot_6d)
            if loss_sm_rot < 1:
                loss_sm_rot = torch.tensor(0.0, device=device)
            loss_sm_trans = compute_smoothness_loss(multi_frame_model.trans)

            # hand 2d joints loss
            projected_hand_joints = camera.transform_points_screen(mano_joints_batch, image_size=((H_out, W_out),))  # (N, 21, 3)
            pred_hand_joints_2d = projected_hand_joints[..., :2]  # (N, 21, 2)
            target_hand_joints_2d = sampled_gt_hand_joints_2d  # (N, 21, 2)
            mano_to_openpose = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]
            target_hand_joints_2d = target_hand_joints_2d[:, mano_to_openpose, :]  # (N, 21, 2)

            loss_joints_2d = torch.nn.functional.l1_loss(pred_hand_joints_2d, target_hand_joints_2d)
            # loss_sm_hand = compute_smoothness_loss(mano_joints_batch)

            # hand anatomy loss
            T_g_p = multi_frame_model.transforms_abs  # (B, 16, 4, 4)
            T_g_a, _R, ee = multi_frame_model.axisFK(T_g_p)  # ee (B, 16, 3)
            loss_anatomy = multi_frame_model.anatomyLoss(ee)

            # Get batch vertices (needed for both contact loss and collision loss)
            obj_verts_batch = posed_obj_meshes_batch.verts_padded()  # (B, N_obj, 3)
            hand_verts_batch = posed_hand_meshes_batch.verts_padded()  # (B, N_hand, 3)

            # contact map loss
            # Get all contact vertices using advanced indexing
            hand_contact_verts = hand_verts_batch[batch_indices, hand_indices]  # (Total_N, 3)
            obj_contact_verts = obj_verts_batch[batch_indices, obj_indices]  # (Total_N, 3)
            loss_contact = torch.nn.functional.l1_loss(hand_contact_verts, obj_contact_verts)
            # loss_contact = torch.nn.functional.mse_loss(hand_contact_verts, obj_contact_verts)
            # 计算每对接触点之间的距离
            # per_point_distances = torch.norm(hand_contact_verts - obj_contact_verts, dim=-1)  # (N,)

            # # 只惩罚超过阈值的部分
            # threshold = 0.005  # 0.5cm
            # exceeded_distances = torch.relu(per_point_distances - threshold)  # 小于threshold的变为0

            # # 计算loss（使用L2范数）
            # loss_contact = (exceeded_distances**2).mean()
            # loss_contact = compute_density_balanced_loss(hand_contact_verts, obj_contact_verts)

            # Get batch normals: all frames have same canonical object, so just reshape packed normals
            obj_normals_packed = posed_obj_meshes_batch.verts_normals_packed()  # (B*N_obj, 3)
            num_obj_verts = obj_verts_batch.shape[1]
            obj_normals_batch = obj_normals_packed.view(num_sampled_frames, num_obj_verts, 3)  # (B, N_obj, 3)

            # Get hand normals
            hand_normals_packed = posed_hand_meshes_batch.verts_normals_packed()  # (B*N_hand, 3)
            num_hand_verts = hand_verts_batch.shape[1]
            hand_normals_batch = hand_normals_packed.view(num_sampled_frames, num_hand_verts, 3)  # (B, N_hand, 3)

            # Compute bidirectional collision loss
            # 1. Hand vertices penetrating object
            loss_hand_in_obj = compute_collision_loss(
                obj_verts_batch,  # (B, N_obj, 3)
                obj_normals_batch,  # (B, N_obj, 3)
                hand_verts_batch,  # (B, N_hand, 3)
                ignore_indices=None  # Note: batch ignore_indices not supported yet
            )

            # 2. Object vertices penetrating hand
            loss_obj_in_hand = compute_collision_loss(
                hand_verts_batch,  # (B, N_hand, 3)
                hand_normals_batch,  # (B, N_hand, 3)
                obj_verts_batch,  # (B, N_obj, 3)
                ignore_indices=None)

            # Total bidirectional collision loss
            if step > (num_steps * 0.2):
                loss_collision = (loss_hand_in_obj + loss_obj_in_hand)
            else:
                loss_collision = torch.tensor(0.0, device=device)

            # hand_loss = loss_joints_2d + loss_sm_hand * smoothness_weight + loss_anatomy
            loss_joints_2d_weight = 1e1
            loss_anatomy_weight = 1e2
            # loss_anatomy_weight = 0  # NOTE disable anatomy loss
            loss_mse_weight = 1e0
            loss_sm_rot_weight = 1e1
            loss_sm_trans_weight = 1e1
            loss_contact_weight = 1e4
            loss_collision_weight = 1e2

            total_loss = loss_joints_2d * loss_joints_2d_weight + \
            loss_mse * loss_mse_weight + \
            loss_contact * loss_contact_weight + \
            loss_sm_rot * loss_sm_rot_weight + \
            loss_sm_trans * loss_sm_trans_weight + \
            loss_collision * loss_collision_weight + \
            loss_anatomy * loss_anatomy_weight

            total_loss.backward()
            # Clip gradients separately for each optimizer to avoid interference
            # torch.nn.utils.clip_grad_norm_([multi_frame_model.scale, multi_frame_model.trans], max_norm=1.0)
            torch.nn.utils.clip_grad_norm_([multi_frame_model.rot_6d, multi_frame_model.trans], max_norm=1.0)
            torch.nn.utils.clip_grad_norm_([multi_frame_model.mano_trans, multi_frame_model.mano_pose, multi_frame_model.mano_root_orient], max_norm=1.0)
            obj_optimizer.step()
            hand_optimizer.step()

            loop.set_postfix(
                loss=total_loss.item(),
                loss_mse=(loss_mse * loss_mse_weight).item(),
                loss_sm_rot=(loss_sm_rot * loss_sm_rot_weight).item(),
                loss_sm_trans=(loss_sm_trans * loss_sm_trans_weight).item(),
                loss_joints_2d=(loss_joints_2d * loss_joints_2d_weight).item(),
                # loss_sm_hand=loss_sm_hand.item(),
                loss_anatomy=(loss_anatomy * loss_anatomy_weight).item(),
                loss_contact=(loss_contact * loss_contact_weight).item(),
                loss_collision=(loss_collision * loss_collision_weight).item(),
                scale=(multi_frame_model.scale * multi_frame_model.initial_scale)[0].item(),
            )

        # NOTE modify until here #####################################################################################
        # NOTE modify until here #####################################################################################
        # NOTE modify until here #####################################################################################
        # NOTE modify until here #####################################################################################
        # NOTE modify until here #####################################################################################
        # NOTE modify until here #####################################################################################

        # # ========================================================================
        # # Save checkpoint for STAGE 6
        # # ========================================================================
        # print("\n" + "=" * 80)
        # print("Saving checkpoint for STAGE 6...")
        # print("=" * 80)

        # # Convert rot_6d to rotation matrices for initial_R_mat
        # saved_initial_R_mat = rotation_6d_to_matrix(multi_frame_model.rot_6d.data.detach())

        # checkpoint = {
        #     # Segment boundaries and stage ranges
        #     'stage_ranges': stage_ranges,
        #     'segment_boundaries': segment_boundaries,
        #     'picked_frame_idx': picked_frame_idx,
        #     'start_static_end_idx': start_static_end_idx,
        #     'approaching_end_idx': approaching_end_idx,
        #     'interaction_end_idx': interaction_end_idx,
        #     'end_static_start_idx': end_static_start_idx,

        #     # Camera parameters
        #     'fx_new': fx_new.item() if isinstance(fx_new, torch.Tensor) else fx_new,
        #     'fy_new': fy_new.item() if isinstance(fy_new, torch.Tensor) else fy_new,
        #     'cx_new': cx_new.item() if isinstance(cx_new, torch.Tensor) else cx_new,
        #     'cy_new': cy_new.item() if isinstance(cy_new, torch.Tensor) else cy_new,

        #     # Model parameters (for reconstruction)
        #     'initial_R_mat': saved_initial_R_mat.cpu(),  # (N, 3, 3) rotation matrices
        #     'initial_T': multi_frame_model.trans.data.detach().cpu(),
        #     'initial_scale': multi_frame_model.initial_scale.detach().cpu(),

        #     # Optimized parameters
        #     'rot_6d': multi_frame_model.rot_6d.data.detach().cpu(),
        #     'trans': multi_frame_model.trans.data.detach().cpu(),
        #     'scale': multi_frame_model.scale.data.detach().cpu(),
        #     'mano_root_orient': multi_frame_model.mano_root_orient.data.detach().cpu(),
        #     'mano_trans': multi_frame_model.mano_trans.data.detach().cpu(),
        #     'mano_pose': multi_frame_model.mano_pose.data.detach().cpu(),
        #     'mano_betas': multi_frame_model.mano_betas.data.detach().cpu(),
        # }

        # torch.save(checkpoint, checkpoint_final_path)
        # print(f"Checkpoint saved to {checkpoint_final_path}")

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

    # Check if interpolation is needed
    need_interpolation = (num_sampled_frames < num_total_frames)

    if need_interpolation:
        print(f"Interpolating from {num_sampled_frames} frames to {num_total_frames} frames...")

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

        # --- Interpolate MANO parameters ---
        # Extract sparse optimized parameters
        sparse_mano_root_orient_np = multi_frame_model.mano_root_orient.detach().cpu().numpy()
        sparse_mano_trans_np = multi_frame_model.mano_trans.detach().cpu().numpy()
        sparse_mano_pose_np = multi_frame_model.mano_pose.detach().cpu().numpy()
        sparse_mano_betas_np = multi_frame_model.mano_betas.detach().cpu().numpy()

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

        # Interpolate betas (shape parameters) with CubicSpline
        # Betas represent hand shape and should vary smoothly across frames
        cs_mano_betas = CubicSpline(t_sparse, sparse_mano_betas_np, axis=0)
        final_mano_betas_full_batch = cs_mano_betas(t_full)  # (num_total_frames, 10)

    else:
        print(f"No interpolation needed: already have {num_total_frames} frames")

        # Directly use sparse parameters (no interpolation)
        final_R_full_batch = sparse_R_tensor.numpy()
        final_T_full_batch = sparse_T_np

        R_full_tensor = sparse_R_tensor
        T_full_tensor = torch.from_numpy(sparse_T_np).float().to(device)

        rot_6d_full = multi_frame_model.rot_6d.detach()
        trans_full = multi_frame_model.trans.detach()

        # Use sparse MANO parameters directly
        final_mano_root_full_batch = multi_frame_model.mano_root_orient.detach().cpu().numpy()
        final_mano_T_full_batch = multi_frame_model.mano_trans.detach().cpu().numpy()
        final_mano_pose_full_batch = multi_frame_model.mano_pose.detach().cpu().numpy()
        final_mano_betas_full_batch = multi_frame_model.mano_betas.detach().cpu().numpy()

    # Convert all to tensors for rendering
    mano_root_full_tensor = torch.from_numpy(final_mano_root_full_batch).float().to(device)
    mano_trans_full_tensor = torch.from_numpy(final_mano_T_full_batch).float().to(device)
    mano_pose_full_tensor = torch.from_numpy(final_mano_pose_full_batch).float().to(device)
    mano_betas_full_tensor = torch.from_numpy(final_mano_betas_full_batch).float().to(device)
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
    all_hand_joints_list = []

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
            current_mano_betas = mano_betas_full_tensor[i:end_idx]
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
                mano_betas_init=current_mano_betas,
                is_right_init=current_is_right,
                obj_textures=original_textures).to(device)

            # Use original texture for object visualization
            current_obj_meshes, current_hand_meshes, current_hand_joints = current_model(use_original_texture=True)

            # Create TexturesVertex for hand (red color for visualization)
            hand_color = torch.tensor([1.0, 0.0, 0.0], device=device)  # red color
            N, V = current_hand_meshes.verts_padded().shape[:2]
            hand_verts_features = hand_color.view(1, 1, 3).expand(N, V, 3)  # (N, V, 3)
            hand_textures = TexturesVertex(verts_features=hand_verts_features)

            # Replace hand textures with TexturesVertex
            current_hand_meshes.textures = hand_textures

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

            all_hand_joints_list.append((current_hand_joints @ flat_mat).cpu().numpy())

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
    all_obj_verts = np.concatenate(all_obj_verts_list, axis=0)  # (N_total, V_obj, 3)
    all_hand_verts = np.concatenate(all_hand_verts_list, axis=0)  # (N_total, V_hand, 3)
    all_hand_joints = np.concatenate(all_hand_joints_list, axis=0)  # (N_total, 21, 3)

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

    # Determine output directory
    if parsed_contact_map is not None:
        output_dir = os.path.join(seq_path, "optimized_hoi_contact_seq")
    else:
        output_dir = os.path.join(seq_path, "optimized_hoi_init_seq")

    save_hoi_sequence(output_dir, canonical_verts_np, canonical_faces_np, final_obj_scale, final_obj_rot_mat_col_major, final_obj_trans, mano_root_full_tensor,
                      mano_pose_full_tensor, mano_trans_full_tensor, is_right_full_tensor, all_obj_verts, all_obj_faces, all_hand_verts, all_hand_faces)

    # --- Save evaluation-ready data ---
    save_eval_data(seq_path, seq_path,
                   intrinsics.cpu().numpy(), num_total_frames, all_hand_verts, all_hand_joints, all_hand_faces, all_obj_verts, all_obj_faces)


def save_eval_data(output_dir, seq_path, intrinsics, num_frames, all_hand_verts, all_hand_joints, all_hand_faces, all_obj_verts, all_obj_faces):
    """
    保存评估所需的数据为单个字典文件
    
    Args:
        all_hand_verts: (N, 778, 3) numpy array, 已在OpenCV坐标系
        all_hand_joints: (N, 21, 3) numpy array, 已在OpenCV坐标系
        all_hand_faces: (F_hand, 3) numpy array
        all_obj_verts: (N, M, 3) numpy array, 已在OpenCV坐标系
        all_obj_faces: (F_obj, 3) numpy array
    """
    print("--- Saving Evaluation Data ---")

    # 获取序列名称
    seq_name = os.path.basename(seq_path)

    # 1. 构建图像路径列表
    im_paths = []
    for i in range(num_frames):
        im_path = f'rgb/{i:04d}.png'
        im_paths.append(im_path)

    # 2. 确保intrinsics是3x3矩阵，扩展为 (1, 3, 3)
    if intrinsics.shape == (3, 3):
        K = intrinsics[np.newaxis, ...]  # (1, 3, 3)
    else:
        K = intrinsics[:3, :3][np.newaxis, ...]  # (1, 3, 3)

    # 3. 转换为torch.Tensor (数据已经是OpenCV坐标系)
    hand_verts = torch.from_numpy(all_hand_verts).float()  # (N, 778, 3)
    hand_joints_openpose = torch.from_numpy(all_hand_joints).float()  # (N, 21, 3) OpenPose顺序

    # 将关节点从OpenPose顺序还原回MANO顺序
    # mano_to_openpose = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]
    # 构建逆映射: openpose_to_mano[i] 表示OpenPose索引i应该放在MANO的哪个位置
    mano_to_openpose = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]
    openpose_to_mano = [mano_to_openpose.index(i) for i in range(21)]
    hand_joints = hand_joints_openpose[:, openpose_to_mano, :]  # (N, 21, 3) MANO顺序

    obj_verts = torch.from_numpy(all_obj_verts).float()  # (N, M, 3)

    obj_faces = torch.from_numpy(all_obj_faces).long()
    hand_faces = torch.from_numpy(all_hand_faces).long()

    # 4. 计算手部相关数据
    hand_root = hand_joints[:, 0, :]  # (N, 3) 手腕位置 (MANO顺序中索引0是手腕)
    j3d_ra_right = hand_joints - hand_root[:, None, :]  # (N, 21, 3) 关节点相对手腕

    # 5. 计算物体相关数据
    obj_root = obj_verts.mean(dim=1)  # (N, 3) bbox中心
    v3d_ra_object = obj_verts - obj_root[:, None, :]  # (N, M, 3) 相对bbox中心
    v3d_right_object = obj_verts - hand_root[:, None, :]  # (N, M, 3) 相对手腕

    # 6. 组装输出字典
    out_dict = {
        # ========== 基础信息 ==========
        "fnames": np.array(im_paths),  # (N_frames,)
        "K": K,  # (1, 3, 3)
        "full_seq_name": seq_name,

        # ========== 手部数据 ==========
        "verts.right": hand_verts,  # (N, 778, 3)
        "jnts.right": hand_joints,  # (N, 21, 3)
        "root.right": hand_root,  # (N, 3)
        "j3d_ra.right": j3d_ra_right,  # (N, 21, 3)

        # ========== 物体数据 ==========
        "verts.object": obj_verts,  # (N, M, 3)
        "v3d_c.object": obj_verts,  # (N, M, 3) 同 verts.object
        "root.object": obj_root,  # (N, 3)
        "v3d_ra.object": v3d_ra_object,  # (N, M, 3)
        "v3d_right.object": v3d_right_object,  # (N, M, 3)

        # ========== 拓扑结构 ==========
        "faces": {
            'object': obj_faces,  # (F_obj, 3)
            'right': hand_faces,  # (F_hand, 3)
        }
    }

    # 7. 保存字典
    save_path = os.path.join(output_dir, 'eval_data.npy')
    np.save(save_path, out_dict, allow_pickle=True)
    print(f"  -> Saved evaluation data to {save_path}")

    print(f"--- Evaluation Data Summary ---")
    print(f"  Sequence: {seq_name}")
    print(f"  Frames: {num_frames}")
    print(f"  Hand vertices: {hand_verts.shape}")
    print(f"  Hand joints: {hand_joints.shape}")
    print(f"  Object vertices: {obj_verts.shape}")
    print(f"  Coordinate system: OpenCV (x: right, y: down, z: forward)")


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
    hand_mesh_dir = os.path.join('rendering_for_paper', "ours_HO3D", f"{seq_name}", "hand")
    obj_mesh_dir = os.path.join('rendering_for_paper', "ours_HO3D", f"{seq_name}", "object")
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
    replaced_masks_save_dir = os.path.join(output_seq_path, 'replaced_obj_masks')
    cropped_depths_save_dir = os.path.join(output_seq_path, 'cropped_depths')
    cropped_metric_depths_save_dir = os.path.join(output_seq_path, 'cropped_metric_depths')

    # Load raw frames once at the beginning
    raw_rgbs_np = load_raw_frames(seq_path + "/rgbs", frame_type='rgb')

    # NOTE use video tracking to replace original mask #############
    # NOTE use video tracking to replace original mask #############
    # NOTE use video tracking to replace original mask #############
    # NOTE use video tracking to replace original mask #############

    # --- 1. Video Tracking to replace original mask ---
    if os.path.exists(replaced_masks_save_dir):
        print(f"--- Loading pre-computed replaced masks for {seq_name} ---")
    else:
        print(f"--- Running video tracking to replace original masks for {seq_name} ---")
        if worker_model_state["sam_video_predictor"] is None:
            print(f"GPU {torch.cuda.current_device()}: Loading video tracking model...")
            worker_model_state["sam_video_predictor"] = init_sam_video_predictor()
        sam_video_predictor = worker_model_state["sam_video_predictor"]

        manually_picked_frame_json = os.path.join(seq_path, 'manually_picked_frame_idx.json')
        with open(manually_picked_frame_json, 'r') as f:
            picked_frame_idx = json.load(f)['new_frame_idx']

        initial_mask = load_raw_frames(seq_path + "/obj_masks", frame_type='mask')[picked_frame_idx]

        video_segments = sam_video_tracking(sam_video_predictor, raw_rgbs_np, initial_mask, picked_frame_idx)
        os.makedirs(seq_path + "/replaced_obj_masks", exist_ok=True)
        for frame_idx, segments in video_segments.items():
            if 1 in segments:
                frame_mask = segments[1]
                frame_rgba = cv2.cvtColor(raw_rgbs_np[frame_idx], cv2.COLOR_RGB2RGBA)
                rgba_cutout = np.zeros_like(frame_rgba, dtype=np.uint8)
                rgba_cutout = np.where(frame_mask.squeeze()[..., None], frame_rgba, rgba_cutout)
                imageio.imwrite(seq_path + "/replaced_obj_masks/" + f'{frame_idx}.png', rgba_cutout)

    # NOTE use video tracking to replace original mask #############
    # NOTE use video tracking to replace original mask #############
    # NOTE use video tracking to replace original mask #############
    # NOTE use video tracking to replace original mask #############

    raw_masks_np = load_raw_frames(seq_path + "/replaced_obj_masks", frame_type='mask')

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
            metric_depth_img_uint16 = (metric_depth_img_float[:, :, 0] * 1000).astype(np.uint16)
            cv2.imwrite(os.path.join(cropped_metric_depths_save_dir, f"{i}.png"), metric_depth_img_uint16)

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
        "sam_video_predictor": None,
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
            "sam_video_predictor": None,
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
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing output.')

    args = parser.parse_args()

    main(args)
