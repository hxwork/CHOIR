import argparse
import json
import os
import traceback
import warnings

import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import torch
import torch.multiprocessing as mp
import torch.nn as nn
from PIL import Image
from pytorch3d.io import IO
from pytorch3d.io.experimental_gltf_io import MeshGlbFormat
from pytorch3d.renderer import (BlendParams, Materials, MeshRasterizer, MeshRenderer, PerspectiveCameras, PointLights, RasterizationSettings,
                                SoftSilhouetteShader, TexturesVertex)
from pytorch3d.renderer.mesh.shader import HardPhongShader
from pytorch3d.structures import Meshes
from pytorch3d.transforms import (Transform3d, matrix_to_rotation_6d, rotation_6d_to_matrix)
from scipy.interpolate import CubicSpline
from scipy.ndimage import binary_dilation
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from sympy.sets.sets import true
from torchvision import transforms
from tqdm import tqdm

from debug_bbox import get_global_amodal_bbox, load_hand_data, load_raw_frames
from models.diffusion_vas.pipeline_diffusion_vas import DiffusionVASPipeline

warnings.filterwarnings("ignore")


def simplify_mesh(mesh, target_triangles=5000):
    vertices = mesh.verts_list()[0].cpu().numpy()
    faces = mesh.faces_list()[0].cpu().numpy()
    mesh_o3d = o3d.geometry.TriangleMesh()
    mesh_o3d.vertices = o3d.utility.Vector3dVector(vertices)
    mesh_o3d.triangles = o3d.utility.Vector3iVector(faces)
    mesh_o3d.remove_duplicated_vertices()
    mesh_o3d.remove_degenerate_triangles()
    mesh_o3d.remove_duplicated_triangles()
    mesh_o3d.remove_non_manifold_edges()

    if len(mesh_o3d.triangles) > target_triangles:
        mesh_simplified = mesh_o3d.simplify_quadric_decimation(target_triangles)
    else:
        mesh_simplified = mesh_o3d
    device = mesh.device
    verts = torch.tensor(np.asarray(mesh_simplified.vertices), dtype=torch.float32, device=device)
    faces = torch.tensor(np.asarray(mesh_simplified.triangles), dtype=torch.int64, device=device)
    textures = TexturesVertex(verts_features=torch.ones_like(verts)[None])  # (1, V, 3)
    mesh_simplified = Meshes(verts=[verts], faces=[faces], textures=textures)

    return mesh_simplified


class TemporalObjectPose(nn.Module):

    def __init__(self, initial_R, initial_T, initial_scale, initial_verts, faces):
        super().__init__()
        self.rot_6d = nn.Parameter(matrix_to_rotation_6d(initial_R), requires_grad=True)  # (N, 6)
        self.trans = nn.Parameter(initial_T, requires_grad=True)  # (N, 3)
        self.scale = nn.Parameter(initial_scale, requires_grad=True)  # (3,)
        # self.scale = torch.tensor(initial_scale)  # (N, 3)

        self.register_buffer('initial_verts', initial_verts)  # (N, V, 3)
        self.register_buffer('faces', faces)  # (N, F, 3)

    def forward(self):
        N = self.rot_6d.shape[0]
        R = rotation_6d_to_matrix(self.rot_6d)  # (N, 3, 3)

        scale = self.scale.unsqueeze(0).unsqueeze(0)  # (1, 1, 3)
        # posed_verts = (self.initial_verts * self.scale.unsqueeze(1)) @ R + self.trans.unsqueeze(1)
        posed_verts = (self.initial_verts * scale) @ R + self.trans.unsqueeze(1)
        # transform = Transform3d(dtype=torch.float32, device=R.device).scale(self.scale).rotate(R).translate(self.trans)  # (N, 4, 4)
        # posed_verts = transform.transform_points(self.initial_verts)
        textures = TexturesVertex(verts_features=torch.ones_like(posed_verts))
        return Meshes(verts=posed_verts, faces=self.faces, textures=textures)


class SingleObjectPose(nn.Module):

    def __init__(self, initial_R, initial_T, initial_scale, initial_verts, faces):
        super().__init__()
        self.rot_6d = nn.Parameter(matrix_to_rotation_6d(initial_R), requires_grad=True)  # (6,)
        self.trans = nn.Parameter(initial_T, requires_grad=True)  # (3,)
        self.scale = nn.Parameter(initial_scale, requires_grad=True)

        self.register_buffer('initial_verts', initial_verts)
        self.register_buffer('faces', faces)

    def forward(self):
        R = rotation_6d_to_matrix(self.rot_6d)  # (3, 3)
        transform = Transform3d(dtype=torch.float32, device=R.device).scale(self.scale).rotate(R.unsqueeze(0)).translate(self.trans.unsqueeze(0))
        posed_verts = transform.transform_points(self.initial_verts.unsqueeze(0))
        textures = TexturesVertex(verts_features=torch.ones_like(posed_verts))
        return Meshes(verts=posed_verts, faces=self.faces.unsqueeze(0), textures=textures)


def init_relative_translation(
        init_pose_T,  # (3,) 第一帧 GT Pose
        masks,  # (T, H, W) Amodal Mask 序列
        fx,
        fy  # (2,) 内参
):
    num_frames = len(masks)

    # 1. 提取基础参数
    T0 = init_pose_T  # [X0, Y0, Z0]
    Z0 = T0[2]

    # 2. 计算 Frame 0 的基准信息 (Reference)
    # 使用 BBox 中心和面积
    x, y, w, h = cv2.boundingRect(masks[0].astype(np.uint8))
    if w == 0 or h == 0:
        raise ValueError("first frame mask is empty!")

    u0 = x + w / 2.0
    v0 = y + h / 2.0

    initial_translations = []

    print("Initializing via Relative Motion (XY + Z-Scaling)...")

    for t in range(num_frames):
        mask = masks[t]

        # A. 计算当前帧状态
        x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))

        if w == 0 or h == 0:
            # 异常：沿用上一帧
            if t == 0:
                current_T = T0
            else:
                current_T = initial_translations[-1]
            initial_translations.append(current_T)
            continue

        ut = x + w / 2.0
        vt = y + h / 2.0

        Z_new = Z0

        # C. 计算 2D 位移 delta
        delta_u = ut - u0
        delta_v = vt - v0

        # D. 将 2D 位移映射为 3D 位移 (Back-projection)
        # 注意：这里用新的 Z_new 会更准，也可以用 Z0 做近似
        delta_X = -1 * (delta_u * Z_new / fx)
        delta_Y = -1 * (delta_v * Z_new / fy)

        # E. 叠加到初始 Pose 上
        X_new = T0[0] + delta_X
        Y_new = T0[1] + delta_Y

        current_T = np.array([X_new, Y_new, Z_new], dtype=np.float32)
        initial_translations.append(current_T)

    return np.stack(initial_translations)  # (T, 3)


def compute_smoothness_loss(rot_6d, trans):
    # Simple temporal difference using torch.diff for conciseness
    diff_rot = torch.diff(rot_6d.contiguous(), dim=0)
    diff_trans = torch.diff(trans.contiguous(), dim=0)

    return diff_rot.pow(2).sum() + diff_trans.pow(2).sum()


def compute_affine_invariant_loss(rendered_depth, gt_depth):
    """
    Affine Invariant Loss (Shift + Scale Invariant).
    专门用于对齐 Depth Anything / MiDaS 这类单目估计深度。
    
    公式: Loss = MSE( Normalize(Pred), Normalize(GT) )
    """
    # 1. Masking: 只在重叠区域计算
    import ipdb
    ipdb.set_trace()
    valid_mask = (gt_depth > 0) & (rendered_depth > 0)

    if valid_mask.sum() < 2:  # 至少需要2个点才能算方差
        return torch.tensor(0.0, device=rendered_depth.device)

    pred_valid = rendered_depth[valid_mask]
    gt_valid = gt_depth[valid_mask]

    # 2. 计算统计量 (Mean & Std)
    pred_mean = pred_valid.mean()
    pred_std = pred_valid.std() + 1e-8  # 防止除零

    gt_mean = gt_valid.mean()
    gt_std = gt_valid.std() + 1e-8

    # 3. Z-Score 归一化 (Normalization)
    # 将两者都拉到 N(0, 1) 分布
    pred_norm = (pred_valid - pred_mean) / pred_std
    gt_norm = (gt_valid - gt_mean) / gt_std

    # 4. 计算 MSE
    # 这里不需要 detach mean/std，因为我们希望优化器为了匹配分布
    # 而去调整 Pose (比如旋转物体会改变深度的方差分布)
    loss = torch.nn.functional.mse_loss(pred_norm, gt_norm)

    return loss


def compute_scale_invariant_depth_loss(rendered_depths, gt_depths):
    """
    Computes a scale-and-shift-invariant depth loss.
    This is done by z-score normalizing both depth maps over the valid mask
    before comparing them with an MSE loss.
    """
    # --- Compute Mean (for shift invariance) ---
    pred_valid = rendered_depths > 0
    gt_valid = gt_depths > 0
    mean_pred = (rendered_depths.sum(dim=[1, 2]) / pred_valid.sum(dim=[1, 2])).view(-1, 1, 1)
    mean_gt = (gt_depths.sum(dim=[1, 2]) / gt_valid.sum(dim=[1, 2])).view(-1, 1, 1)

    # --- Compute Std Dev (for scale invariance) ---
    pred_centered = (rendered_depths - mean_pred) * pred_valid
    gt_centered = (gt_depths - mean_gt) * gt_valid

    std_pred = torch.sqrt((pred_centered**2).sum(dim=[1, 2]) / pred_valid.sum(dim=[1, 2])).view(-1, 1, 1).clamp(min=1e-8)
    std_gt = torch.sqrt((gt_centered**2).sum(dim=[1, 2]) / gt_valid.sum(dim=[1, 2])).view(-1, 1, 1).clamp(min=1e-8)

    # --- Normalize (Z-Score) ---
    pred_norm = pred_centered / std_pred
    gt_norm = gt_centered / std_gt

    # --- Compute Loss ---
    diff_sq_masked = torch.nn.functional.mse_loss(pred_norm, gt_norm, reduction='mean')
    return diff_sq_masked


def init_amodal_segmentation_model(model_path_mask):
    device = f"cuda:{torch.cuda.current_device()}"
    pipeline_mask = DiffusionVASPipeline.from_pretrained(model_path_mask, torch_dtype=torch.float16).to(device)
    # pipeline_mask.enable_model_cpu_offload()
    pipeline_mask.set_progress_bar_config(disable=True)

    return pipeline_mask


def init_rgb_model(model_path_rgb):
    device = f"cuda:{torch.cuda.current_device()}"
    pipeline_rgb = DiffusionVASPipeline.from_pretrained(model_path_rgb, torch_dtype=torch.float16).to(device)
    # pipeline_rgb.enable_model_cpu_offload()
    pipeline_rgb.set_progress_bar_config(disable=True)

    return pipeline_rgb


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


def overlay_mask_on_image(rgb_img, mask, cmap_idx=None, random_color=False, boundary_thickness=3, darken_factor=2):
    # Ensure the input image is RGB and in the range [0, 1]
    assert rgb_img.shape[-1] == 3, "Expected RGB image with 3 channels"
    # assert rgb_img.min() >= 0 and rgb_img.max() <= 1, "Expected rgb_img values in the range [0, 1]"

    # Select a color for the mask overlay
    cmap = plt.get_cmap("tab10")

    cmap_idx = 4
    if cmap_idx is None and random_color:
        cmap_idx = np.random.randint(0, cmap.N)  # Randomly choose a colormap index if not provided

    color = np.array([*cmap(cmap_idx)[:3], 0.6])
    boundary_color = color[:3] * darken_factor  # Darken the color by the darken_factor
    boundary_color = np.concatenate([boundary_color, [1.0]])  # Make boundary fully opaque

    # Create a boundary mask
    dilated_mask = binary_dilation(mask, iterations=boundary_thickness)
    boundary_mask = dilated_mask & ~mask

    # Create a colored mask in the range [0, 1]
    mask_image = np.zeros_like(rgb_img, dtype=np.float32)
    boundary_image = np.zeros_like(rgb_img, dtype=np.float32)

    for i in range(3):  # Apply the mask and boundary to each channel
        mask_image[..., i] = mask * color[i]
        boundary_image[..., i] = boundary_mask * boundary_color[i]

    # Combine the RGB image with the colored mask and boundary
    overlayed_image = np.clip(rgb_img * 0.5 + mask_image + boundary_image, 0, 1)

    return overlayed_image


def overlay_rgb_render(background_img, render_img, alpha=1.0):
    # Ensure inputs are floats in [0, 1]
    bg = background_img.astype(np.float32)
    fg = render_img.astype(np.float32)

    mask = (fg.sum(axis=-1) > 0.0).astype(np.float32)
    mask = mask[..., None]
    composited = bg * (1 - mask * alpha) + fg * alpha

    return np.clip(composited, 0, 1)


def calculate_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 1.0
    return intersection / union


def lock_first_frame_hook(grad):
    # grad 的 shape 是 (N, 6) 或 (N, 3)
    # 我们 clone 一个 grad 以免原地修改导致副作用（虽然通常原地改也没事）
    new_grad = grad.clone()
    new_grad[0] = 0.0  # 强制把第一帧梯度置零
    return new_grad


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
    num_total_frames,
    lr=1e-2,
    num_steps=200,
    smoothness_weight=1.0,
):
    # --- 0. Frame Sampling ---
    num_total_frames = len(bboxes)
    max_sampled_frames = 128

    if num_total_frames > max_sampled_frames:
        # Calculate stride to get approximately max_sampled_frames
        stride = num_total_frames // max_sampled_frames
        sampled_indices = list(range(stride, num_total_frames, stride))
    else:
        # If total frames are less than the max, use all of them
        sampled_indices = list(range(num_total_frames))

    # Ensure the first and last frames are always included
    if sampled_indices[0] != 0:
        sampled_indices.append(0)
    if sampled_indices[-1] != num_total_frames - 1:
        sampled_indices.append(num_total_frames - 1)

    # Remove duplicates and sort the indices
    sampled_indices = sorted(list(set(sampled_indices)))
    num_sampled_frames = len(sampled_indices)

    print(f"Total frames: {num_total_frames}, Sampled frames: {num_sampled_frames}")

    # Sample all frame-related data
    sampled_pred_amodal_masks_np = pred_amodal_masks_np[sampled_indices]
    sampled_rgbs_np = cropped_rgbs_np[sampled_indices]
    sampled_modal_masks_np = cropped_modal_masks_np[sampled_indices]
    sampled_depths_np = cropped_depths_np[sampled_indices]

    sampled_hand_masks_np = None
    sampled_hand_masks_np = cropped_hand_masks_np[sampled_indices]

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
    initial_R = M / initial_scale
    initial_T = transform_matrix[3, :3]

    # Load camera intrinsics
    intrinsics_path = os.path.join(seq_path, 'intrinsics.json')
    with open(intrinsics_path, 'r') as f:
        intrinsics = torch.tensor(json.load(f)['intrinsics'], dtype=torch.float32, device=device)  # (3, 3)

    # --- 2. Build Global Crop-space Camera Model (Batched) ---
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
    sampled_modal_masks = torch.from_numpy(sampled_modal_masks_np).to(device)
    sampled_hand_masks = torch.from_numpy(sampled_hand_masks_np).to(device)

    # --- 4. Optimization Loop ---
    initial_scale = torch.tensor(initial_scale, dtype=torch.float32, device=device).repeat(3)
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

    # --- 5. Optimization Loop: Stage 2 (Chunk-wise) ---

    raster_settings = RasterizationSettings(image_size=(H_out, W_out), blur_radius=1e-4, faces_per_pixel=20)
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(raster_settings=raster_settings),
        shader=SoftSilhouetteShader(),
    )

    optimized_first_frame_rot_6d = single_frame_model.rot_6d.detach()
    optimized_first_frame_trans = single_frame_model.trans.detach()
    optimized_scale = single_frame_model.scale.detach()

    chunk_size = 16
    num_chunks = int(np.ceil(num_sampled_frames / chunk_size))
    print(f"Starting chunk-wise optimization: {num_sampled_frames} frames in {num_chunks} chunks of size {chunk_size}.")

    list_of_optimized_rot_6d = []
    list_of_optimized_trans = []

    last_pose_rot_6d = optimized_first_frame_rot_6d
    last_pose_trans = optimized_first_frame_trans

    base_verts = verts
    base_faces = faces

    for i in range(num_chunks):
        loop_desc = f"Optimizing Chunk {i+1}/{num_chunks}"
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, num_sampled_frames)
        chunk_slice = slice(start_idx, end_idx)
        num_frames_in_chunk = end_idx - start_idx

        if num_frames_in_chunk == 0:
            continue

        chunk_pred_amodal_masks_np = sampled_pred_amodal_masks_np[chunk_slice]
        chunk_pred_amodal_masks = sampled_pred_amodal_masks[chunk_slice]
        chunk_hand_masks = sampled_hand_masks[chunk_slice]

        initial_rot_6d_chunk = last_pose_rot_6d.unsqueeze(0).repeat(num_frames_in_chunk, 1)

        initial_T_np_chunk = init_relative_translation(last_pose_trans.cpu().numpy(), chunk_pred_amodal_masks_np, fx_new.item(), fy_new.item())
        initial_T_chunk = torch.from_numpy(initial_T_np_chunk).float().to(device)

        chunk_verts = base_verts.unsqueeze(0).repeat(num_frames_in_chunk, 1, 1)
        chunk_faces = base_faces.unsqueeze(0).repeat(num_frames_in_chunk, 1, 1)

        chunk_model = TemporalObjectPose(rotation_6d_to_matrix(initial_rot_6d_chunk), initial_T_chunk, optimized_scale, chunk_verts, chunk_faces).to(device)

        # --- Visualization of initial pose for the chunk ---
        with torch.no_grad():
            initial_posed_mesh = chunk_model()[0:1]
            camera_for_viz = PerspectiveCameras(
                focal_length=focal_length[0, None],
                principal_point=principal_point[0, None],
                image_size=((H_out, W_out),),
                in_ndc=False,
                device=device,
            )
            fragments = renderer.rasterizer(initial_posed_mesh, cameras=camera_for_viz)
            rendered_mask = renderer.shader(fragments, initial_posed_mesh, cameras=camera_for_viz)[0, ..., 3].cpu().numpy()
            rgb_np = sampled_rgbs_np[start_idx]
            render_frame = overlay_mask_on_image(rgb_np, (rendered_mask > 0.5).astype(np.uint8))
            amodal_gt_frame = overlay_mask_on_image(rgb_np, sampled_pred_amodal_masks_np[start_idx])
            combined_image = (np.hstack([amodal_gt_frame, render_frame]) * 255).astype(np.uint8)
            frame_global_idx = sampled_indices[start_idx]
            save_path = os.path.join(output_path, f"chunk_{i:02d}_initial_frame_{frame_global_idx:04d}.png")
            imageio.imwrite(save_path, combined_image)

        # # Frame locking should only apply to the first chunk if at all
        # if i == 0:
        #     first_mask = sampled_pred_amodal_masks_np[0]
        #     lock_indices = [
        #         idx for idx, i in enumerate(range(*chunk_slice.indices(num_sampled_frames)))
        #         if calculate_iou(first_mask, sampled_pred_amodal_masks_np[i]) > 0.95
        #     ]
        #     if 0 not in lock_indices:
        #         lock_indices.insert(0, 0)

        #     if lock_indices:
        #         print(f"Chunk {i+1}: Locking frame indices (relative to chunk): {lock_indices}")

        #         def create_lock_frames_hook(indices_to_lock):

        #             def hook(grad):
        #                 new_grad = grad.clone()
        #                 new_grad[indices_to_lock] = 0.0
        #                 return new_grad

        #             return hook

        #         lock_hook = create_lock_frames_hook(lock_indices)
        #         chunk_model.rot_6d.register_hook(lock_hook)
        #         chunk_model.trans.register_hook(lock_hook)

        optimizer = torch.optim.Adam([chunk_model.rot_6d, chunk_model.trans], lr=lr)

        camera = PerspectiveCameras(
            focal_length=focal_length[chunk_slice],
            principal_point=principal_point[chunk_slice],
            image_size=((H_out, W_out),),
            in_ndc=False,
            device=device,
        )

        loop = tqdm(range(num_steps), desc=loop_desc)
        for step in loop:
            optimizer.zero_grad()
            posed_meshes_batch = chunk_model()
            fragments = renderer.rasterizer(posed_meshes_batch, cameras=camera)
            rendered_masks = renderer.shader(fragments, posed_meshes_batch, cameras=camera)[..., 3]

            # weights = torch.where(chunk_hand_masks.bool() & chunk_pred_amodal_masks.bool(),
            #                       torch.ones_like(chunk_pred_amodal_masks) * 0.1, torch.ones_like(chunk_pred_amodal_masks))
            # l2_loss = (weights * torch.nn.functional.mse_loss(rendered_masks, chunk_pred_amodal_masks, reduction='none')).mean() * 1e3
            l2_loss = torch.nn.functional.mse_loss(rendered_masks, chunk_pred_amodal_masks, reduction='mean') * 1e3

            # if step > 300:
            #     smoothness_loss = compute_smoothness_loss(chunk_model.rot_6d, chunk_model.trans) * 1e-2
            # else:
            #     smoothness_loss = compute_smoothness_loss(chunk_model.rot_6d, chunk_model.trans) * 1e-2
            smoothness_loss = torch.tensor(0.0, device=device)

            total_loss = l2_loss + smoothness_loss
            total_loss.backward()
            optimizer.step()
            loop.set_postfix(loss=total_loss.item(), l2=l2_loss.item(), smoothness=smoothness_loss.item())

        optimized_rot_6d_chunk = chunk_model.rot_6d.detach()
        optimized_trans_chunk = chunk_model.trans.detach()

        list_of_optimized_rot_6d.append(optimized_rot_6d_chunk)
        list_of_optimized_trans.append(optimized_trans_chunk)

        last_pose_rot_6d = optimized_rot_6d_chunk[-1]
        last_pose_trans = optimized_trans_chunk[-1]

    multi_frame_model_rot_6d = torch.cat(list_of_optimized_rot_6d, dim=0)
    multi_frame_model_trans = torch.cat(list_of_optimized_trans, dim=0)
    multi_frame_model_scale = optimized_scale

    # --- Quick Visualization of Sparse Optimization Results ---
    with torch.no_grad():
        # Temporarily create a model for visualization of all sparse frames
        sparse_verts = base_verts.unsqueeze(0).repeat(num_sampled_frames, 1, 1)
        sparse_faces = base_faces.unsqueeze(0).repeat(num_sampled_frames, 1, 1)

        sparse_model_for_viz = TemporalObjectPose(rotation_6d_to_matrix(multi_frame_model_rot_6d), multi_frame_model_trans, multi_frame_model_scale,
                                                  sparse_verts, sparse_faces).to(device)
        camera = PerspectiveCameras(
            focal_length=focal_length,
            principal_point=principal_point,
            image_size=((H_out, W_out),),
            in_ndc=False,
            device=device,
        )

        sparse_posed_meshes = sparse_model_for_viz()
        final_rendered_masks_sparse = renderer(sparse_posed_meshes, cameras=camera)[..., 3].cpu().numpy()

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

        hand_frame = overlay_mask_on_image(sampled_rgbs_np[i], sampled_hand_masks_np[i], cmap_idx=1)
        panels = [modal_frame, amodal_gt_frame, render_frame, hand_frame]

        combined_frame = np.hstack(panels)
        writer_sparse.append_data((combined_frame * 255).astype(np.uint8))
    writer_sparse.close()
    print(f"Saved sparse fitting visualization to {sparse_video_path}")

    # --- 5. Export & Visualization ---

    # prepare sparse data
    sparse_R_tensor = rotation_6d_to_matrix(multi_frame_model_rot_6d).detach().cpu()
    sparse_T_np = multi_frame_model_trans.detach().cpu().numpy()
    final_scale = multi_frame_model_scale.detach().cpu().numpy()

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
            # current_scale = torch.tensor([final_scale], dtype=torch.float32, device=device).repeat(current_batch_size, 3)
            current_scale = torch.from_numpy(final_scale).float().to(device)
            current_verts = verts[None, ...].repeat(current_batch_size, 1, 1)
            current_faces = faces[None, ...].repeat(current_batch_size, 1, 1)
            current_model = TemporalObjectPose(current_R, current_T, current_scale, current_verts, current_faces).to(device)

            current_meshes = current_model()

            model_color = [0.65, 0.8, 1.0]  # blue color

            N, V = current_meshes.verts_padded().shape[:2]

            current_verts_rgb = torch.tensor(model_color, device=device).view(1, 1, 3).expand(N, V, -1)

            current_meshes.textures = TexturesVertex(verts_features=current_verts_rgb)

            current_camera = PerspectiveCameras(
                focal_length=torch.tensor([[fx_new, fy_new]], device=device).expand(current_batch_size, -1),
                principal_point=torch.tensor([[cx_new, cy_new]], device=device).expand(current_batch_size, -1),
                image_size=((H_out, W_out),),
                in_ndc=False,
                device=device,
            )

            current_images = renderer(current_meshes, cameras=current_camera, lights=lights, materials=materials)

            current_rgb = current_images[..., :3]

            final_rendered_images_list.append(current_rgb.cpu().numpy())

            del current_meshes, current_camera, current_images, current_rgb, current_model, current_verts_rgb
            torch.cuda.empty_cache()

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
        depth_loss = compute_affine_invariant_loss(rendered_depth, target_depth) * depth_weight
        # depth_loss = compute_scale_invariant_depth_loss(rendered_depth.unsqueeze(0), target_depth.unsqueeze(0)) * depth_weight
        # depth_loss = torch.tensor(0.0, device=device)
        total_loss = l2_loss + depth_loss

        total_loss.backward()
        optimizer.step()
        loop.set_postfix(loss=total_loss.item(), l2=l2_loss.item(), depth=depth_loss.item())

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
    # fit_and_visualize_pose(
    #     seq_path=seq_path,
    #     output_path=output_seq_path,
    #     device=device,
    #     pred_amodal_masks_np=pred_amodal_masks_np,
    #     bboxes=global_bboxes,
    #     cropped_rgbs_np=cropped_rgbs_np,
    #     cropped_modal_masks_np=cropped_modal_masks_np,
    #     cropped_hand_masks_np=cropped_hand_masks_np,
    #     cropped_depths_np=cropped_depths_np,
    #     num_total_frames=num_frames,
    #     lr=args.lr,
    #     num_steps=args.num_steps,
    #     smoothness_weight=args.smoothness_weight,
    # )

    # # --- Debug single frame fitting ---
    frame_to_debug = 126
    fit_single_frame_pose(
        seq_path=seq_path,
        output_path=output_seq_path,
        device=device,
        frame_idx=frame_to_debug,
        amodal_mask_0_np=pred_amodal_masks_np[0],
        amodal_mask_np=pred_amodal_masks_np[frame_to_debug],
        modal_mask_np=cropped_modal_masks_np[frame_to_debug],
        hand_mask_np=cropped_hand_masks_np[frame_to_debug],
        depth_np=cropped_depths_np[frame_to_debug],
        rgb_np=cropped_rgbs_np[frame_to_debug],
        bbox=global_bboxes[frame_to_debug],
        lr=args.lr,
        num_steps=args.num_steps,
    )


def worker_main_gpu(gpu_id, seq_path_chunks, args):
    torch.cuda.set_device(gpu_id)
    print(f"Worker on GPU {gpu_id} started, processing {len(seq_path_chunks[gpu_id])} sequences.")

    # LAZY LOADING: State will hold models, loaded only when needed.
    worker_model_state = {
        "pipeline_mask": None,
        "depth_model": None,  # Now fully lazy
        "generator": None
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
        target_ids = set(args.video_id.split(','))
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
            "generator": None
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
        "--model_path_rgb",
        type=str,
        default="checkpoints/diffusion-vas-content-completion",
        help="Path to diffusion-vas content completion checkpoint.",
    )

    parser.add_argument(
        "--depth_encoder",
        type=str,
        default="vitl",  # or 'vits', vitl, 'vitg'
        help="Depth encoder type.")

    parser.add_argument("--model_path_depth", type=str, default="checkpoints/", help="Path to depth anything v2's checkpoint's parent folder.")

    parser.add_argument("--data_path",
                        type=str,
                        default="../../output",
                        help="Path to the parent directory containing sequence subfolders.")

    parser.add_argument("--data_output_path", type=str, default="outputs", help="Output path.")

    parser.add_argument('--video_id', type=str, default=None, help="Comma-separated video IDs to process.")
    parser.add_argument('--debug', action='store_true', help='Run in single-process debug mode without multiprocessing.')
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate for pose optimization.")
    parser.add_argument('--num_steps', type=int, default=300, help="Number of optimization steps.")
    parser.add_argument('--smoothness_weight', type=float, default=1, help="Weight for the trajectory smoothness loss.")

    args = parser.parse_args()

    main(args)
