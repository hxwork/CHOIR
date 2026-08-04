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
from pytorch3d.io import IO, load_ply, save_obj, save_ply
from pytorch3d.io.experimental_gltf_io import MeshGlbFormat
from pytorch3d.ops import knn_points
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
from pnp import generate_queries, run_pnp
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
        if frame_type in ['rgb', 'mask']:
            pil_image = Image.fromarray(final_image.astype(np.uint8))
            tensor_frame = to_tensor(pil_image)
        elif frame_type == 'depth':
            # Add channel dimension for to_tensor. Input is float [0,1]
            tensor_frame = to_tensor(final_image[:, :, np.newaxis])

        if frame_type != 'rgb':
            tensor_frame = tensor_frame.repeat(3, 1, 1)

        # Normalize from [0, 1] to [-1, 1]
        transformed_frame = normalizer(tensor_frame)

        processed_frames.append(transformed_frame)

    return torch.stack(processed_frames).unsqueeze(0)


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
        # mano_cfg = {
        #     'model_path': os.path.join(mano_dir, 'mano'),
        #     'gender': 'neutral',
        #     'num_hand_joints': 15,
        #     'mean_params': os.path.join(mano_dir, 'mano_mean_params.npz'),
        #     'create_body_pose': False
        # }

        # batch_size = initial_R.shape[0] if initial_R is not None else 1
        # print('initializing MANO model with cfgs:', mano_cfg, 'B*T', batch_size)
        # self.hand_model = MANO(batch_size=batch_size, pose2rot=True, **mano_cfg)

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
        # mano_output = run_mano(self.hand_model, self.mano_trans[None], self.mano_root_orient[None], self.mano_pose[None], self.is_right.to(device))
        amano_output = run_amano(self.hand_model_amano, self.mano_trans[None], self.mano_root_orient[None], self.mano_pose[None], self.is_right.to(device))
        mano_joints = amano_output['joints'].squeeze(0) @ flat_mat  # (1, N, 21, 3) -> (N, 21, 3)
        mano_verts = amano_output['vertices'].squeeze(0) @ flat_mat  # (1, N, 778, 3) -> (N, 778, 3)
        mano_l_faces = amano_output['l_faces']  # (1538, 3)
        mano_r_faces = amano_output['r_faces']  # (1538, 3)
        mano_is_right = amano_output['is_right'].squeeze(0)  # (1, N) -> (N,)
        # mano_hand_pose = mano_output['body_pose'].squeeze(0)  # (1, N, 15, 3) -> (N, 15, 3)
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
        self.scale = nn.Parameter(initial_scale, requires_grad=True)  # (3,)

        self.register_buffer('initial_verts', initial_verts)
        self.register_buffer('faces', faces)

    def forward(self):
        R = rotation_6d_to_matrix(self.rot_6d)  # (3, 3)
        transform = Transform3d(dtype=torch.float32, device=R.device).scale(self.scale).rotate(R.unsqueeze(0)).translate(self.trans.unsqueeze(0))
        posed_verts = transform.transform_points(self.initial_verts.unsqueeze(0))
        textures = TexturesVertex(verts_features=torch.ones_like(posed_verts))
        return Meshes(verts=posed_verts, faces=self.faces.unsqueeze(0), textures=textures)


def compute_smoothness_loss(seq):
    # Simple temporal difference using torch.diff for conciseness
    diff = torch.diff(seq.contiguous(), dim=0)

    return diff.pow(2).sum()


def lock_first_frame_hook(grad):
    # grad 的 shape 是 (N, 6) 或 (N, 3)
    # 我们 clone 一个 grad 以免原地修改导致副作用（虽然通常原地改也没事）
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
    sampled_hand_masks_np = cropped_hand_masks_np[sampled_indices]
    sampled_mano_params = mano_params[sampled_indices]
    # --- 1. Data Loading & Pre-processing ---

    # ply_path = os.path.join(seq_path, 'mesh_0.ply')
    # verts, faces = load_ply(ply_path)
    # mesh = Meshes(verts=[verts], faces=[faces]).to(device)

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

    # --- 4. Optimization Loop ---
    initial_scale = torch.tensor(initial_scale, dtype=torch.float32, device=device).repeat(3)  # (3,)
    single_frame_model = SingleObjectPose(initial_R, initial_T, initial_scale, verts, faces).to(device)
    optimizer = torch.optim.Adam([single_frame_model.rot_6d, single_frame_model.scale, single_frame_model.trans], lr=lr)

    camera = PerspectiveCameras(
        focal_length=focal_length[0, None],
        principal_point=principal_point[0, None],
        image_size=((H_out, W_out),),  # Correct image size for the camera
        in_ndc=False,
        device=device,
    )

    # --- 3. Differentiable Rendering Setup ---
    raster_settings = RasterizationSettings(image_size=(H_out, W_out), blur_radius=1e-4, faces_per_pixel=20)
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=camera, raster_settings=raster_settings),
        shader=SoftSilhouetteShader(),
    )
    # --- 7. Optimization Loop ---
    loop = tqdm(range(200), desc="Optimizing First Frame")
    for step in loop:

        optimizer.zero_grad()
        posed_mesh = single_frame_model()
        fragments = renderer.rasterizer(posed_mesh, cameras=camera)
        rendered_mask = renderer.shader(fragments, posed_mesh, cameras=camera)[0, ..., 3]
        rendered_zbuf = fragments.zbuf[0, ..., 0]

        if (rendered_zbuf > 0).sum() == 0:
            rendered_zbuf = torch.zeros_like(rendered_zbuf)

        l2_loss = torch.nn.functional.mse_loss(rendered_mask, sampled_pred_amodal_masks[0, None]) * 1e3
        total_loss = l2_loss

        total_loss.backward()
        optimizer.step()
        loop.set_postfix(loss=total_loss.item(), l2=l2_loss.item())

    # --- 8. Visualization ---
    with torch.no_grad():
        final_posed_mesh = single_frame_model()
        fragments = renderer.rasterizer(final_posed_mesh, cameras=camera)
        final_mask = renderer.shader(fragments, final_posed_mesh, cameras=camera)[0, ..., 3].cpu().numpy()

    # Create 4-panel comparison image
    modal_frame = overlay_mask_on_image(sampled_rgbs_np[0], sampled_modal_masks_np[0])
    amodal_gt_frame = overlay_mask_on_image(sampled_rgbs_np[0], sampled_pred_amodal_masks_np[0])
    render_frame = overlay_mask_on_image(sampled_rgbs_np[0], (final_mask > 0.5).astype(np.uint8))

    hand_frame = overlay_mask_on_image(sampled_rgbs_np[0], sampled_hand_masks_np[0], cmap_idx=1)
    panels = [modal_frame, amodal_gt_frame, render_frame, hand_frame]
    combined_image = (np.hstack(panels) * 255).astype(np.uint8)
    save_path = os.path.join(output_path, f"optimized_frame_0.png")
    imageio.imwrite(save_path, combined_image)
    print(f"Saved single frame visualization to {save_path}")

    # use the optimized first frame to run PnP
    verts_canonical_scaled = verts * single_frame_model.scale.detach().unsqueeze(0)
    mesh_canonical_scaled = Meshes(
        verts=[verts_canonical_scaled],
        faces=[faces],
        textures=TexturesVertex(verts_features=torch.ones_like(verts)[None]),
    )
    queries_2d, queries_3d = generate_queries(final_posed_mesh,
                                              mesh_canonical_scaled,
                                              sampled_pred_amodal_masks_np[0],
                                              focal_length[0, None],
                                              principal_point[0, None],
                                              grid_size=15,
                                              device=device)

    K = np.array([[fx_new.item(), 0, cx_new.item()], [0, fy_new.item(), cy_new.item()], [0, 0, 1]])
    pnp_poses = run_pnp(cotracker_model,
                        sampled_rgbs,
                        queries_2d,
                        queries_3d,
                        sampled_pred_amodal_masks_np,
                        K,
                        initial_R,
                        initial_T,
                        device,
                        output_dir=os.path.join(output_path, "pnp_visualization"),
                        vis_threshold=0.0)

    pnp_poses = torch.from_numpy(np.stack(pnp_poses, axis=0)).float().to(device)
    pnp_rot_mat = pnp_poses[:, :3, :3]
    pnp_t = pnp_poses[:, :3, 3]
    # --- 5. Optimization Loop: Stage 2 (Global) ---

    camera = PerspectiveCameras(
        focal_length=focal_length,
        principal_point=principal_point,
        image_size=((H_out, W_out),),
        in_ndc=False,
        device=device,
    )
    raster_settings = RasterizationSettings(image_size=(H_out, W_out), blur_radius=1e-4, faces_per_pixel=20)
    # raster_settings = RasterizationSettings(image_size=(H_out, W_out), blur_radius=0.1, faces_per_pixel=20)
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=camera, raster_settings=raster_settings),
        shader=SoftSilhouetteShader(blend_params=BlendParams(sigma=1e-4, gamma=1e-4)),
        # shader=SoftSilhouetteShader(blend_params=BlendParams(sigma=0.1, gamma=1e-2)),
    )

    # Initialize poses for the whole sequence
    initial_R_mat = single_frame_model.rot_6d.detach().unsqueeze(0)
    initial_R_mat = rotation_6d_to_matrix(initial_R_mat).repeat(num_sampled_frames, 1, 1)
    initial_R_mat = pnp_rot_mat.mT

    initial_T = init_relative_translation(single_frame_model.trans.cpu().detach().numpy(), sampled_pred_amodal_masks_np, fx_new.item(), fy_new.item())
    initial_T = torch.from_numpy(initial_T).float().to(device)
    # initial_T = pnp_t

    initial_scale = single_frame_model.scale.detach()

    # Prepare mesh data for the batch
    verts_batch = verts.unsqueeze(0).repeat(num_sampled_frames, 1, 1)
    faces_batch = faces.unsqueeze(0).repeat(num_sampled_frames, 1, 1)

    # --- Frame Locking ---
    # first_mask = sampled_pred_amodal_masks_np[0]
    # lock_indices = [i for i, mask in enumerate(sampled_pred_amodal_masks_np) if calculate_iou(first_mask, mask) > 0.95]
    lock_indices = [0]
    if 0 not in lock_indices:
        lock_indices.insert(0, 0)
    print(f"Locking frames with IoU > 0.95 with first frame: {lock_indices}")

    def create_lock_frames_hook(indices_to_lock):

        def hook(grad):
            new_grad = grad.clone()
            new_grad[indices_to_lock] = 0.0
            return new_grad

        return hook

    lock_hook = create_lock_frames_hook(lock_indices)

    # --- Model and Optimizer ---
    multi_frame_model = TemporalHandObjectPose(initial_R_mat, initial_T, initial_scale, verts_batch, faces_batch, sampled_mano_params).to(device)
    multi_frame_model.rot_6d.register_hook(lock_hook)
    multi_frame_model.trans.register_hook(lock_hook)

    obj_optimizer = torch.optim.Adam([multi_frame_model.rot_6d, multi_frame_model.trans], lr=lr)
    hand_optimizer = torch.optim.Adam([multi_frame_model.mano_root_orient, multi_frame_model.mano_trans, multi_frame_model.mano_pose], lr=lr)

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

    # 如果contact map存在，则使用contact map进行优化
    contact_map_path = os.path.join(output_path, "grasp_correction/contact_map.npy")
    if os.path.exists(contact_map_path):
        contact_map = np.load(contact_map_path, allow_pickle=True)
        if contact_map.shape == ():  # it's a 0-d array wrapping an object
            contact_map = contact_map.item()

    else:
        contact_map = None

    # --- Detect interaction window based on both IoU and translation changes ---
    interaction_start_idx = 0
    interaction_end_idx = num_sampled_frames - 1  # Default to all frames
    if num_sampled_frames > 1:
        # 1. IoU based detection (looser threshold)
        ious = [calculate_iou(sampled_pred_amodal_masks_np[i - 1], sampled_pred_amodal_masks_np[i]) for i in range(1, num_sampled_frames)]
        # A looser threshold means we require a larger change in mask to be considered motion.
        interaction_frames_iou = np.where(np.array(ious) < 0.9)[0] + 1  # Add 1 to align indices

        # 2. Translation based detection
        translation_diff = torch.linalg.norm(torch.diff(multi_frame_model.trans.detach(), dim=0), dim=1)
        # Threshold can be tuned. Let's assume units are meters and set a 5mm threshold.
        interaction_frames_trans_tensor = torch.where(translation_diff > 0.005)[0] + 1  # Add 1 to align indices
        interaction_frames_trans = interaction_frames_trans_tensor.cpu().numpy()

        # 3. Combine: only frames detected by BOTH methods are considered interaction frames
        interaction_frames = np.intersect1d(interaction_frames_iou, interaction_frames_trans)

        if len(interaction_frames) > 0:
            interaction_start_idx = max(0, interaction_frames[0] - 1)  # Include one frame before motion starts
            interaction_end_idx = min(num_sampled_frames - 1, interaction_frames[-1] + 1)  # Include one frame after motion ends
            print(f"Interaction window detected (IoU & Translation): frames {interaction_start_idx} to {interaction_end_idx}")
        else:
            print("No significant object motion detected by both IoU and translation, will apply contact loss to all frames if contact map exists.")

    if contact_map is not None and isinstance(contact_map, dict) and len(contact_map) > 0:
        hand_indices = list(contact_map.keys())
        obj_indices = list(contact_map.values())
        hand_indices_tensor = torch.tensor(hand_indices, dtype=torch.long, device=device)
        obj_indices_tensor = torch.tensor(obj_indices, dtype=torch.long, device=device)
        contact_loss_weight = 1e3
        joint_loss_weight = 0.2
        weight_inter_penetr_max = 1e2
        start_penetr_step = int(num_steps * 0.7)  # 例如在前70%的步数不计算穿模
        loop = tqdm(range(num_steps), desc="Optimizing Pose Sequence using contact map")
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

            # hand loss
            projected_hand_joints = camera.transform_points_screen(mano_joints_batch, image_size=((H_out, W_out),))  # (N, 21, 3)
            pred_hand_joints_2d = projected_hand_joints[..., :2]
            # Masked 2D joint loss
            num_valid_frames = sampled_gt_hand_joints_valid_mask.sum()
            if num_valid_frames > 0:
                loss_joints_2d = torch.nn.functional.mse_loss(pred_hand_joints_2d[sampled_gt_hand_joints_valid_mask],
                                                              sampled_gt_hand_joints_2d[sampled_gt_hand_joints_valid_mask])
            else:
                loss_joints_2d = torch.tensor(0.0, device=device)
            loss_sm_hand = compute_smoothness_loss(mano_joints_batch)

            T_g_p = multi_frame_model.transforms_abs  # (B, 16, 4, 4)
            T_g_a, _R, ee = multi_frame_model.axisFK(T_g_p)  # ee (B, 16, 3)
            loss_anatomy = multi_frame_model.anatomyLoss(ee)

            hand_loss = loss_joints_2d + loss_sm_hand * smoothness_weight + loss_anatomy

            # contact map loss
            obj_verts = posed_obj_meshes_batch.verts_padded()
            hand_verts = posed_hand_meshes_batch.verts_padded()

            obj_verts = posed_obj_meshes_batch.verts_padded()
            hand_verts = posed_hand_meshes_batch.verts_padded()

            contact_hand_verts = torch.index_select(hand_verts, 1, hand_indices_tensor)
            contact_obj_verts = torch.index_select(obj_verts, 1, obj_indices_tensor)

            # Calculate per-frame loss
            per_frame_loss_contact = torch.nn.functional.mse_loss(contact_hand_verts, contact_obj_verts, reduction='none').mean(dim=[1, 2])

            # Create a mask for the interaction window
            frame_indices = torch.arange(num_sampled_frames, device=device)
            interaction_mask = (frame_indices >= interaction_start_idx) & (frame_indices <= interaction_end_idx)

            # Apply mask to the loss
            loss_contact = (per_frame_loss_contact * interaction_mask.float()).sum() / (interaction_mask.float().sum() + 1e-8) * contact_loss_weight

            # collision loss
            # === 动态调整穿模权重 ===
            if step < start_penetr_step:
                # 第一阶段：完全关闭，允许自由穿梭寻找最佳接触点
                current_penetr_weight = 0.0
            else:
                # 第二阶段：线性增加权重 (Warm-up)
                progress = (step - start_penetr_step) / (num_steps - start_penetr_step)
                current_penetr_weight = weight_inter_penetr_max * progress
            hand_normal = posed_hand_meshes_batch.verts_normals_packed().view(-1, 778, 3)
            hand_nn_dist, hand_nn_idx = get_NN(obj_verts, hand_verts)
            hand_interior = get_interior(hand_normal, hand_verts, obj_verts, hand_nn_idx).type(torch.bool)
            hand_in_obj_penetr_dist = hand_nn_dist[hand_interior].sum()
            loss_inter_penetr = hand_in_obj_penetr_dist * current_penetr_weight

            total_loss = obj_loss + hand_loss + loss_contact + loss_inter_penetr

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
                loss_contact=loss_contact.item(),
                loss_inter_penetr=loss_inter_penetr.item(),
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
        writer_sparse.append_data((combined_frame * 255).astype(np.uint8))
    writer_sparse.close()
    print(f"Saved sparse fitting visualization to {sparse_video_path}")

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
    if contact_map is not None and isinstance(contact_map, dict) and len(contact_map) > 0:
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
    # obj_mesh_dir = os.path.join(output_dir, "obj_meshes")
    # os.makedirs(obj_mesh_dir, exist_ok=True)
    # hand_mesh_dir = os.path.join(output_dir, "hand_meshes")
    # os.makedirs(hand_mesh_dir, exist_ok=True)

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

        # # 4. Save per-frame posed meshes
        # obj_mesh_path = os.path.join(obj_mesh_dir, f"obj_{i:05d}.obj")
        # save_obj(obj_mesh_path, torch.from_numpy(obj_verts_seq[i]).float(), torch.from_numpy(obj_faces).long())

        # hand_mesh_path = os.path.join(hand_mesh_dir, f"hand_{i:05d}.obj")
        # save_obj(hand_mesh_path, torch.from_numpy(hand_verts_seq[i]).float(), torch.from_numpy(hand_faces).long())

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

    # --- A. 加载背景点云 ---
    if os.path.exists(bg_pc_path):
        bg_mesh = trimesh.load(bg_pc_path, process=False)
        if hasattr(bg_mesh, 'vertices'):
            bg_points = k3d.points(positions=bg_mesh.vertices.astype(np.float32), point_size=0.005, shader='3d', color=0xAAAAAA, name="Environment PC")
            plot += bg_points
    else:
        print(f"  - info: background point cloud not found at {bg_pc_path}")

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

    if not subfolders:
        print("Warning: No video sequences found to process with the given criteria.")
        return

    if args.debug:
        print("--- RUNNING IN DEBUG MODE (single process, sequential) ---")
        torch.cuda.set_device(0)

        worker_model_state = {
            "pipeline_mask": None,
            "depth_model": None,  # Now fully lazy
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

    args = parser.parse_args()

    main(args)
