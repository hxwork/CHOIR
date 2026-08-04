import math
import os

import cv2
import numpy as np
import torch
import trimesh
from PIL import Image
from pytorch3d.renderer import (MeshRasterizer, MeshRenderer, MeshRendererWithFragments, PerspectiveCameras,
                                 RasterizationSettings, SoftPhongShader, SoftSilhouetteShader, TexturesVertex)
from pytorch3d.structures import Meshes

from cotracker.utils.visualizer import Visualizer
from utils import farthest_point_sampling_torch


def generate_queries(mesh_for_sampling: Meshes, mesh_canonical_scaled: Meshes, segm_mask: np.ndarray, focal_length: torch.Tensor, principal_point: torch.Tensor,
                     grid_size: int, device: torch.device):
    """
    Generate query points by rendering a depth map and combining it with a segmentation mask.
    """
    image_size = (segm_mask.shape[0], segm_mask.shape[1])

    cameras_frag = PerspectiveCameras(device=device, focal_length=focal_length, principal_point=principal_point, in_ndc=False, image_size=[image_size])
    raster_settings_frag = RasterizationSettings(image_size=image_size, faces_per_pixel=1, max_faces_per_bin=100000)
    renderer_frag = MeshRendererWithFragments(
        rasterizer=MeshRasterizer(cameras=cameras_frag, raster_settings=raster_settings_frag),
        shader=SoftPhongShader(device=device, cameras=cameras_frag)  # Add cameras here
    )
    _, fragments = renderer_frag(mesh_for_sampling)
    depth_mask = (fragments.zbuf[0, ..., 0].cpu().numpy() > 0).astype(np.uint8)

    # 4. Determine sampling area and sample 2D points (same as before)
    combined_mask = (segm_mask > 0) & (depth_mask > 0)
    valid_points_y, valid_points_x = np.where(combined_mask)
    if len(valid_points_x) == 0:
        return None, None

    valid_points_xy = np.stack([valid_points_x, valid_points_y], axis=1)
    valid_points_xy_torch = torch.from_numpy(valid_points_xy).float().to(device)
    num_queries = grid_size * grid_size

    if valid_points_xy_torch.shape[0] > num_queries:
        sampled_points_torch = farthest_point_sampling_torch(valid_points_xy_torch, num_queries)
    else:
        sampled_points_torch = valid_points_xy_torch

    if sampled_points_torch.shape[0] == 0:
        return None, None

    sampled_points_np = sampled_points_torch.cpu().numpy().astype(int)
    sampled_x, sampled_y = sampled_points_np[:, 0], sampled_points_np[:, 1]

    # 5. Get 3D coordinates in the SCALED CANONICAL frame
    pix_to_face = fragments.pix_to_face[0, sampled_y, sampled_x].squeeze(-1)
    bary_coords = fragments.bary_coords[0, sampled_y, sampled_x].squeeze(1)

    valid_mask = pix_to_face != -1
    if not torch.any(valid_mask):
        return None, None

    # Create vertices for the SCALED canonical mesh
    verts_scaled_canonical = mesh_canonical_scaled.verts_list()[0]

    # Interpolate using the barycentric coordinates on the SCALED canonical mesh
    # to get the 3D points PnP needs.
    face_verts = mesh_canonical_scaled.faces_list()[0][pix_to_face[valid_mask]]
    queries_3d = (verts_scaled_canonical[face_verts] * bary_coords[valid_mask].unsqueeze(-1)).sum(dim=-2)

    if queries_3d.shape[0] == 0:
        return None, None

    queries_3d = queries_3d.cpu().numpy()

    # Filter 2D points to match the valid 3D points
    sampled_x = sampled_x[valid_mask.cpu().numpy()]
    sampled_y = sampled_y[valid_mask.cpu().numpy()]

    # --- Debug Saving Section ---
    debug_dir = "./debug_output"
    os.makedirs(debug_dir, exist_ok=True)

    seg_mask_img = Image.fromarray(((segm_mask > 0) * 255).astype(np.uint8))
    seg_mask_img.save(os.path.join(debug_dir, "seg_mask.png"))

    depth_mask_img = Image.fromarray((depth_mask * 255).astype(np.uint8))
    depth_mask_img.save(os.path.join(debug_dir, "depth_mask.png"))

    combined_mask_img = Image.fromarray((combined_mask * 255).astype(np.uint8))
    combined_mask_img.save(os.path.join(debug_dir, "combined_mask.png"))

    # Create a visualization of the sampled points on the combined_mask
    points_vis = np.stack([combined_mask.astype(np.uint8) * 255] * 3, axis=-1)
    points_vis[sampled_y, sampled_x] = [255, 0, 0]  # Draw points in red
    points_vis_img = Image.fromarray(points_vis)
    points_vis_img.save(os.path.join(debug_dir, "sampled_points.png"))

    # Save the 3D points as spheres along with the mesh for debugging
    # Create a Trimesh object for the SCALED simplified mesh
    # to match the coordinate system of queries_3d.
    mesh_for_vis = trimesh.Trimesh(vertices=mesh_canonical_scaled.verts_list()[0].cpu().numpy(), faces=mesh_canonical_scaled.faces_list()[0].cpu().numpy())

    # Create a list of sphere meshes for each 3D query point
    sphere_radius = np.mean(mesh_for_vis.bounding_box.extents) / 100.0  # Heuristic for sphere size
    spheres = []
    for point in queries_3d:
        sphere = trimesh.primitives.Sphere(radius=sphere_radius, center=point)
        sphere.visual.face_colors = [255, 0, 0, 255]  # Red
        spheres.append(sphere)

    # Concatenate the main mesh and all sphere meshes into a single mesh
    combined_mesh = trimesh.util.concatenate([mesh_for_vis] + spheres)

    # Export the combined mesh to a PLY file
    combined_mesh.export(os.path.join(debug_dir, "sampled_points_3d.ply"))

    # 6. Format outputs
    queries_2d = np.zeros((len(sampled_x), 2), dtype=np.float32)
    queries_2d[:, 0] = sampled_x
    queries_2d[:, 1] = sampled_y

    return queries_2d, queries_3d


def run_pnp(model, video, queries_2d, queries_3d, masks, K, R, T, device, output_dir=None, vis_threshold=0.5):
    # convert between tensors and numpy
    queries_2d = torch.from_numpy(queries_2d).float().to(device)
    queries_3d = torch.from_numpy(queries_3d).float().to(device)
    R = R.cpu().numpy()
    T = T.cpu().numpy()

    # convert row major to column major
    R = R.T

    # 1. run CoTracker
    N = queries_2d.shape[0]
    t_col = torch.zeros((N, 1), device=device)  # frame index is always 0
    queries = torch.cat([t_col, queries_2d], dim=1)[None]  # (1, N, 3)

    with torch.no_grad():
        video = video.permute(0, 3, 1, 2).unsqueeze(0).contiguous()
        pred_tracks, pred_vis = model(
            video,
            queries=queries,
            backward_tracking=True,
        )

    # Save tracking visualization if an output directory is provided
    if output_dir:
        # The visualizer expects the video as a (B, T, C, H, W) tensor on CPU.
        # video is (1, T, C, H, W), so we just move it to CPU.
        vis = Visualizer(save_dir=output_dir, pad_value=0, linewidth=2)
        vis.visualize(video, pred_tracks, pred_vis, filename="tracking_visualization.mp4")

    tracks_2d = pred_tracks[0].cpu().numpy()  # (T, N, 2)
    vis = pred_vis[0].cpu().numpy()  # (T, N)
    queries_3d = queries_3d.cpu().numpy()  # (N, 3)

    T_frames = tracks_2d.shape[0]
    pnp_poses = [None] * T_frames  # Initialize as a list of Nones

    # 2. run PnP
    # Convert pose from PyTorch3D to OpenCV coordinate system
    flip_mat = np.diag([-1, -1, 1]).astype(np.float32)
    prev_R_cv = flip_mat @ R  # column major
    prev_T_cv = flip_mat @ T
    prev_rvec, prev_tvec = cv2.Rodrigues(prev_R_cv)[0], prev_T_cv
    for t in range(T_frames):
        cur_2d = tracks_2d[t]  # (N, 2)
        cur_mask = masks[t]  # (H, W)
        cur_mask = cv2.dilate(cur_mask, np.ones((9, 9), np.uint8), iterations=1)

        # --- Filter points to be inside the current mask ---
        # Round tracked points to integer coordinates for indexing
        cur_2d_int = np.round(cur_2d).astype(np.int64)
        x_coords = cur_2d_int[:, 0]
        y_coords = cur_2d_int[:, 1]

        # Clip coordinates to valid mask bounds
        mask_h, mask_w = cur_mask.shape
        x_coords = np.clip(x_coords, 0, mask_w - 1)
        y_coords = np.clip(y_coords, 0, mask_h - 1)

        # Check the mask values at these coordinates
        mask_values = cur_mask[y_coords, x_coords]
        # Update the mask for in-bounds points based on the object mask
        is_in_mask = (mask_values > 0)

        # Final filtering
        valid_2d = cur_2d[is_in_mask]  # (N, 2)
        valid_3d = queries_3d[is_in_mask]  # (N, 3)

        if len(valid_2d) < 8:
            print(f"Frame {t}: Not enough points ({len(valid_2d)}) for PnP after masking. Using previous pose.")
            rvec, tvec = prev_rvec, prev_tvec
        else:
            # first use RANSAC to find inliers
            success_ransac, rvec_ransac, tvec_ransac, inliers = cv2.solvePnPRansac(valid_3d,
                                                                                   valid_2d,
                                                                                   K,
                                                                                   None,
                                                                                   iterationsCount=500,
                                                                                   reprojectionError=2.0,
                                                                                   confidence=0.99,
                                                                                   flags=cv2.USAC_MAGSAC)

            if success_ransac:
                success, rvec, tvec = cv2.solvePnP(valid_3d[inliers],
                                                   valid_2d[inliers],
                                                   K,
                                                   None,
                                                   rvec=prev_rvec,
                                                   tvec=prev_tvec,
                                                   useExtrinsicGuess=True,
                                                   flags=cv2.SOLVEPNP_ITERATIVE)

                if success:
                    prev_rvec = rvec
                    prev_tvec = tvec

            else:
                rvec, tvec = prev_rvec, prev_tvec

        # Convert rotation vector to matrix
        R_cv, T_cv = cv2.Rodrigues(rvec)[0], tvec.flatten()
        # Convert pose from OpenCV to PyTorch3D coordinate system
        flip_mat = np.diag([-1, -1, 1]).astype(np.float32)
        R_p3d = flip_mat @ R_cv
        T_p3d = flip_mat @ T_cv

        # Assemble the 4x4 pose matrix in PyTorch3D's convention
        pose_mat = np.eye(4)
        pose_mat[:3, :3] = R_p3d
        pose_mat[:3, 3] = T_p3d

        pnp_poses[t] = pose_mat

        # if inliers is not None and t % 10 == 0:
        #     inlier_ratio = len(inliers) / len(visible_2d)
        #     print(f"Frame {t}: PnP Success. Inliers: {len(inliers)}/{len(visible_2d)} ({inlier_ratio:.2f})")

    return pnp_poses


def generate_queries_1stage(mesh_for_sampling: Meshes, mesh_canonical_scaled: Meshes, segm_mask: np.ndarray, focal_length: torch.Tensor,
                            principal_point: torch.Tensor, grid_size: int, device: torch.device):
    """
    Generate query points by rendering a depth map and combining it with a segmentation mask.
    """
    image_size = (segm_mask.shape[0], segm_mask.shape[1])

    cameras_frag = PerspectiveCameras(device=device, focal_length=focal_length, principal_point=principal_point, in_ndc=False, image_size=[image_size])
    raster_settings_frag = RasterizationSettings(image_size=image_size, faces_per_pixel=1, max_faces_per_bin=100000)
    renderer_frag = MeshRendererWithFragments(
        rasterizer=MeshRasterizer(cameras=cameras_frag, raster_settings=raster_settings_frag),
        shader=SoftPhongShader(device=device, cameras=cameras_frag)  # Add cameras here
    )
    _, fragments = renderer_frag(mesh_for_sampling)
    depth_mask = (fragments.zbuf[0, ..., 0].cpu().numpy() > 0).astype(np.uint8)

    # 4. Determine sampling area and sample 2D points (same as before)
    combined_mask = (segm_mask > 0) & (depth_mask > 0)
    valid_points_y, valid_points_x = np.where(combined_mask)
    if len(valid_points_x) == 0:
        return None, None

    valid_points_xy = np.stack([valid_points_x, valid_points_y], axis=1)
    valid_points_xy_torch = torch.from_numpy(valid_points_xy).float().to(device)
    num_queries = grid_size * grid_size

    if valid_points_xy_torch.shape[0] > num_queries:
        # Use random sampling instead of FPS to avoid edge points
        random_indices = torch.randperm(valid_points_xy_torch.shape[0], device=device)[:num_queries]
        sampled_points_torch = valid_points_xy_torch[random_indices]
    else:
        sampled_points_torch = valid_points_xy_torch

    if sampled_points_torch.shape[0] == 0:
        return None, None

    sampled_points_np = sampled_points_torch.cpu().numpy().astype(int)
    sampled_x, sampled_y = sampled_points_np[:, 0], sampled_points_np[:, 1]

    # 5. Get 3D coordinates in the SCALED CANONICAL frame
    pix_to_face = fragments.pix_to_face[0, sampled_y, sampled_x].squeeze(-1)
    bary_coords = fragments.bary_coords[0, sampled_y, sampled_x].squeeze(1)

    valid_mask = pix_to_face != -1
    if not torch.any(valid_mask):
        return None, None

    # Create vertices for the SCALED canonical mesh
    verts_scaled_canonical = mesh_canonical_scaled.verts_list()[0]

    # Interpolate using the barycentric coordinates on the SCALED canonical mesh
    # to get the 3D points PnP needs.
    face_verts = mesh_canonical_scaled.faces_list()[0][pix_to_face[valid_mask]]
    queries_3d = (verts_scaled_canonical[face_verts] * bary_coords[valid_mask].unsqueeze(-1)).sum(dim=-2)

    if queries_3d.shape[0] == 0:
        return None, None

    queries_3d = queries_3d.cpu().numpy()

    # Filter 2D points to match the valid 3D points
    sampled_x = sampled_x[valid_mask.cpu().numpy()]
    sampled_y = sampled_y[valid_mask.cpu().numpy()]

    # --- Debug Saving Section ---
    debug_dir = "./debug_output"
    os.makedirs(debug_dir, exist_ok=True)

    seg_mask_img = Image.fromarray(((segm_mask > 0) * 255).astype(np.uint8))
    seg_mask_img.save(os.path.join(debug_dir, "seg_mask.png"))

    depth_mask_img = Image.fromarray((depth_mask * 255).astype(np.uint8))
    depth_mask_img.save(os.path.join(debug_dir, "depth_mask.png"))

    combined_mask_img = Image.fromarray((combined_mask * 255).astype(np.uint8))
    combined_mask_img.save(os.path.join(debug_dir, "combined_mask.png"))

    # Create a visualization of the sampled points on the combined_mask
    points_vis = np.stack([combined_mask.astype(np.uint8) * 255] * 3, axis=-1)
    points_vis[sampled_y, sampled_x] = [255, 0, 0]  # Draw points in red
    points_vis_img = Image.fromarray(points_vis)
    points_vis_img.save(os.path.join(debug_dir, "sampled_points.png"))

    # Save the 3D points as spheres along with the mesh for debugging
    # Create a Trimesh object for the SCALED simplified mesh
    # to match the coordinate system of queries_3d.
    mesh_for_vis = trimesh.Trimesh(vertices=mesh_canonical_scaled.verts_list()[0].cpu().numpy(), faces=mesh_canonical_scaled.faces_list()[0].cpu().numpy())

    # Create a list of sphere meshes for each 3D query point
    sphere_radius = np.mean(mesh_for_vis.bounding_box.extents) / 100.0  # Heuristic for sphere size
    spheres = []
    for point in queries_3d:
        sphere = trimesh.primitives.Sphere(radius=sphere_radius, center=point)
        sphere.visual.face_colors = [255, 0, 0, 255]  # Red
        spheres.append(sphere)

    # Concatenate the main mesh and all sphere meshes into a single mesh
    combined_mesh = trimesh.util.concatenate([mesh_for_vis] + spheres)

    # Export the combined mesh to a PLY file
    combined_mesh.export(os.path.join(debug_dir, "sampled_points_3d.ply"))

    # 6. Format outputs
    queries_2d = np.zeros((len(sampled_x), 2), dtype=np.float32)
    queries_2d[:, 0] = sampled_x
    queries_2d[:, 1] = sampled_y

    return queries_2d, queries_3d


def angular_distance(rvec1, rvec2):
    """计算两个旋转向量之间的角度差（弧度）"""
    R1, _ = cv2.Rodrigues(rvec1)
    R2, _ = cv2.Rodrigues(rvec2)
    R_diff = np.dot(R1, R2.T)
    trace = np.trace(R_diff)
    # trace = 1 + 2cos(theta), so theta = arccos((tr-1)/2)
    # Clamp value to range [-1, 1] to avoid numerical errors
    val = (trace - 1) / 2
    val = np.clip(val, -1.0, 1.0)
    return np.arccos(val)


def run_pnp_1stage(model, video, queries_2d, queries_3d, masks, K, R, T, device, output_dir=None, vis_threshold=0.5):
    # convert between tensors and numpy
    queries_2d = torch.from_numpy(queries_2d).float().to(device)
    queries_3d = torch.from_numpy(queries_3d).float().to(device)
    R = R.cpu().numpy()
    T = T.cpu().numpy()

    # convert row major to column major
    R = R.T

    # 1. run CoTracker
    N = queries_2d.shape[0]
    t_col = torch.zeros((N, 1), device=device)  # frame index is always 0
    queries = torch.cat([t_col, queries_2d], dim=1)[None]  # (1, N, 3)

    with torch.no_grad():
        video = video.permute(0, 3, 1, 2).unsqueeze(0).contiguous()
        pred_tracks, pred_vis = model(
            video,
            queries=queries,
            backward_tracking=True,
        )

    # Save tracking visualization if an output directory is provided
    if output_dir:
        # The visualizer expects the video as a (B, T, C, H, W) tensor on CPU.
        # video is (1, T, C, H, W), so we just move it to CPU.
        vis = Visualizer(save_dir=output_dir, pad_value=0, linewidth=2)
        vis.visualize(video, pred_tracks, pred_vis, filename="tracking_visualization")

    tracks_2d = pred_tracks[0].cpu().numpy()  # (T, N, 2)
    vis = pred_vis[0].cpu().numpy()  # (T, N)
    queries_3d = queries_3d.cpu().numpy()  # (N, 3)

    T_frames = tracks_2d.shape[0]
    pnp_poses = [None] * T_frames  # Initialize as a list of Nones

    # 2. run PnP
    # Convert pose from PyTorch3D to OpenCV coordinate system
    flip_mat = np.diag([-1, -1, 1]).astype(np.float32)
    prev_R_cv = flip_mat @ R  # column major
    prev_T_cv = flip_mat @ T
    prev_rvec, prev_tvec = cv2.Rodrigues(prev_R_cv)[0], prev_T_cv
    for t in range(T_frames):
        cur_2d = tracks_2d[t]  # (N, 2)
        cur_mask = masks[t]  # (H, W)
        cur_mask = cv2.dilate(cur_mask, np.ones((9, 9), np.uint8), iterations=1)

        # --- Filter points to be inside the current mask ---
        # Round tracked points to integer coordinates for indexing
        cur_2d_int = np.round(cur_2d).astype(np.int64)
        x_coords = cur_2d_int[:, 0]
        y_coords = cur_2d_int[:, 1]

        # Clip coordinates to valid mask bounds
        mask_h, mask_w = cur_mask.shape
        x_coords = np.clip(x_coords, 0, mask_w - 1)
        y_coords = np.clip(y_coords, 0, mask_h - 1)

        # Check the mask values at these coordinates
        mask_values = cur_mask[y_coords, x_coords]
        # Update the mask for in-bounds points based on the object mask
        is_in_mask = (mask_values > 0)

        # Filter points by visibility using vis_threshold
        cur_vis = vis[t]  # (N,)
        is_visible = (cur_vis >= vis_threshold)

        # Final filtering: combine mask and visibility filters
        is_valid = is_in_mask & is_visible
        valid_2d = cur_2d[is_valid]  # (M, 2) where M <= N
        valid_3d = queries_3d[is_valid]  # (M, 3)

        if len(valid_2d) < 8:
            print(f"Frame {t}: Not enough points ({len(valid_2d)}). Using previous pose.")
            rvec, tvec = prev_rvec, prev_tvec
        else:
            # -----------------------------------------------------------
            # 1. RANSAC (先获取内点，排除噪声干扰)
            # -----------------------------------------------------------
            success_ransac, rvec_ransac, tvec_ransac, inliers = cv2.solvePnPRansac(valid_3d,
                                                                                   valid_2d,
                                                                                   K,
                                                                                   None,
                                                                                   iterationsCount=500,
                                                                                   reprojectionError=2.0,
                                                                                   confidence=0.99,
                                                                                   flags=cv2.USAC_MAGSAC)

            success_refine = False  # 标记 Refinement 是否成功

            if success_ransac:
                inliers = inliers.ravel()
                pts3d_in = valid_3d[inliers]
                pts2d_in = valid_2d[inliers]

                # -----------------------------------------------------------
                # 2. 动态检测是否共面 (Planarity Check) - 移到 RANSAC 之后
                # -----------------------------------------------------------
                # 对“内点”做中心化
                pts3d_centered = pts3d_in - np.mean(pts3d_in, axis=0)
                u, s, vh = np.linalg.svd(pts3d_centered)

                # 增加防除零保护，虽然 RANSAC 保证了点数，但防止极端情况
                if s[0] < 1e-6:
                    is_planar = True  # 几乎聚成一点，视为退化/平面
                else:
                    is_planar = (s[2] / s[0]) < 0.05

                # -------------------------------------------------------
                # 3. 分支精炼求解 (Refinement)
                # -------------------------------------------------------
                if is_planar:
                    # 【策略 A：平面专用 - IPPE】
                    # 【策略 A：平面专用 - IPPE】
                    try:
                        # 1. 接收所有返回值，兼容不同 OpenCV 版本
                        # solvePnPGeneric 可能返回 3个 (count, rvecs, tvecs) 或 4个值 (..., error)
                        ippe_results = cv2.solvePnPGeneric(pts3d_in, pts2d_in, K, None, flags=cv2.SOLVEPNP_IPPE)

                        # 2. 按索引提取 (rvecs 是第2个，tvecs 是第3个)
                        # index 0: int (解的数量)
                        # index 1: tuple of rvecs
                        # index 2: tuple of tvecs
                        rvecs_cand = ippe_results[1]
                        tvecs_cand = ippe_results[2]

                        # 3. 消除歧义：找与上一帧角度最近的解
                        best_idx = 0
                        min_ang_dist = float('inf')

                        # 确保我们确实有解
                        if len(rvecs_cand) > 0:
                            for i in range(len(rvecs_cand)):
                                dist = angular_distance(rvecs_cand[i], prev_rvec)
                                if dist < min_ang_dist:
                                    min_ang_dist = dist
                                    best_idx = i

                            rvec = rvecs_cand[best_idx]
                            tvec = tvecs_cand[best_idx]
                            success_refine = True
                        else:
                            success_refine = False

                    except Exception as e:
                        print(f"IPPE failed: {e}, fallback to RANSAC result")
                        success_refine = False

                else:
                    # 【策略 B：通用 3D - SQPNP】
                    # SQPNP 不需要 useExtrinsicGuess，它是全局最优解
                    solve_flag = cv2.SOLVEPNP_SQPNP if hasattr(cv2, 'SOLVEPNP_SQPNP') else cv2.SOLVEPNP_ITERATIVE

                    # 注意：如果是 ITERATIVE 才真正需要 useExtrinsicGuess，SQPNP 会忽略它但也不报错
                    success_refine, rvec, tvec = cv2.solvePnP(pts3d_in,
                                                              pts2d_in,
                                                              K,
                                                              None,
                                                              rvec=prev_rvec,
                                                              tvec=prev_tvec,
                                                              useExtrinsicGuess=True,
                                                              flags=solve_flag)

            # -------------------------------------------------------
            # 4. 结果更新决策
            # -------------------------------------------------------
            if success_ransac and success_refine:
                # Refinement 成功，更新历史
                prev_rvec = rvec
                prev_tvec = tvec
            elif success_ransac:
                # Refinement 失败 (比如 IPPE 挂了)，但 RANSAC 成功，用 RANSAC 的结果保底
                rvec = rvec_ransac
                tvec = tvec_ransac
                prev_rvec = rvec
                prev_tvec = tvec
            else:
                # RANSAC 都失败了，完全回退到上一帧
                rvec, tvec = prev_rvec, prev_tvec

        # -------------------------------------------------------
        # 5. 坐标转换 (这一块必须在 if/else 之外！)
        # -------------------------------------------------------
        # Convert rotation vector to matrix
        R_cv, _ = cv2.Rodrigues(rvec)
        T_cv = tvec.flatten()

        # Convert pose from OpenCV to PyTorch3D coordinate system
        # OpenCV: Right-handed (X Right, Y Down, Z Forward)
        # PyTorch3D: Right-handed (X Left, Y Up, Z Forward) -> This flip mat is correct for that.
        flip_mat = np.diag([-1, -1, 1]).astype(np.float32)
        R_p3d = flip_mat @ R_cv
        T_p3d = flip_mat @ T_cv

        # Assemble the 4x4 pose matrix
        pose_mat = np.eye(4)
        pose_mat[:3, :3] = R_p3d
        pose_mat[:3, 3] = T_p3d

        pnp_poses[t] = pose_mat

    # if inliers is not None and t % 10 == 0:
    #     inlier_ratio = len(inliers) / len(visible_2d)
    #     print(f"Frame {t}: PnP Success. Inliers: {len(inliers)}/{len(visible_2d)} ({inlier_ratio:.2f})")

    return pnp_poses


# def run_pnp_1stage(model, video, queries_2d, queries_3d, masks, K, R, T, device, output_dir=None, vis_threshold=0.5):
#     # convert between tensors and numpy
#     queries_2d = torch.from_numpy(queries_2d).float().to(device)
#     queries_3d = torch.from_numpy(queries_3d).float().to(device)
#     R = R.cpu().numpy()
#     T = T.cpu().numpy()

#     # convert row major to column major
#     R = R.T

#     # 1. run CoTracker
#     N = queries_2d.shape[0]
#     t_col = torch.zeros((N, 1), device=device)  # frame index is always 0
#     queries = torch.cat([t_col, queries_2d], dim=1)[None]  # (1, N, 3)

#     with torch.no_grad():
#         video = video.permute(0, 3, 1, 2).unsqueeze(0).contiguous()
#         pred_tracks, pred_vis = model(
#             video,
#             queries=queries,
#             backward_tracking=True,
#         )

#     # Save tracking visualization if an output directory is provided
#     if output_dir:
#         # The visualizer expects the video as a (B, T, C, H, W) tensor on CPU.
#         # video is (1, T, C, H, W), so we just move it to CPU.
#         vis = Visualizer(save_dir=output_dir, pad_value=0, linewidth=2)
#         vis.visualize(video, pred_tracks, pred_vis, filename="tracking_visualization")

#     tracks_2d = pred_tracks[0].cpu().numpy()  # (T, N, 2)
#     vis = pred_vis[0].cpu().numpy()  # (T, N)
#     queries_3d = queries_3d.cpu().numpy()  # (N, 3)

#     T_frames = tracks_2d.shape[0]
#     pnp_poses = [None] * T_frames  # Initialize as a list of Nones

#     # 2. run PnP
#     # Convert pose from PyTorch3D to OpenCV coordinate system
#     flip_mat = np.diag([-1, -1, 1]).astype(np.float32)
#     prev_R_cv = flip_mat @ R  # column major
#     prev_T_cv = flip_mat @ T
#     prev_rvec, prev_tvec = cv2.Rodrigues(prev_R_cv)[0], prev_T_cv
#     for t in range(T_frames):
#         cur_2d = tracks_2d[t]  # (N, 2)
#         cur_mask = masks[t]  # (H, W)
#         cur_mask = cv2.dilate(cur_mask, np.ones((9, 9), np.uint8), iterations=1)

#         # --- Filter points to be inside the current mask ---
#         # Round tracked points to integer coordinates for indexing
#         cur_2d_int = np.round(cur_2d).astype(np.int64)
#         x_coords = cur_2d_int[:, 0]
#         y_coords = cur_2d_int[:, 1]

#         # Clip coordinates to valid mask bounds
#         mask_h, mask_w = cur_mask.shape
#         x_coords = np.clip(x_coords, 0, mask_w - 1)
#         y_coords = np.clip(y_coords, 0, mask_h - 1)

#         # Check the mask values at these coordinates
#         mask_values = cur_mask[y_coords, x_coords]
#         # Update the mask for in-bounds points based on the object mask
#         is_in_mask = (mask_values > 0)

#         # Filter points by visibility using vis_threshold
#         cur_vis = vis[t]  # (N,)
#         is_visible = (cur_vis >= vis_threshold)

#         # Final filtering: combine mask and visibility filters
#         is_valid = is_in_mask & is_visible
#         valid_2d = cur_2d[is_valid]  # (M, 2) where M <= N
#         valid_3d = queries_3d[is_valid]  # (M, 3)

#         if len(valid_2d) < 8:
#             print(f"Frame {t}: Not enough points ({len(valid_2d)}) for PnP after masking. Using previous pose.")
#             rvec, tvec = prev_rvec, prev_tvec
#         else:
#             # first use RANSAC to find inliers
#             success_ransac, rvec_ransac, tvec_ransac, inliers = cv2.solvePnPRansac(valid_3d,
#                                                                                    valid_2d,
#                                                                                    K,
#                                                                                    None,
#                                                                                    iterationsCount=500,
#                                                                                    reprojectionError=2.0,
#                                                                                    confidence=0.99,
#                                                                                    flags=cv2.USAC_MAGSAC)

#             if success_ransac:
#                 success, rvec, tvec = cv2.solvePnP(valid_3d[inliers],
#                                                    valid_2d[inliers],
#                                                    K,
#                                                    None,
#                                                    rvec=prev_rvec,
#                                                    tvec=prev_tvec,
#                                                    useExtrinsicGuess=True,
#                                                    flags=cv2.SOLVEPNP_ITERATIVE)

#                 if success:
#                     prev_rvec = rvec
#                     prev_tvec = tvec

#             else:
#                 rvec, tvec = prev_rvec, prev_tvec

#         # Convert rotation vector to matrix
#         R_cv, T_cv = cv2.Rodrigues(rvec)[0], tvec.flatten()
#         # Convert pose from OpenCV to PyTorch3D coordinate system
#         flip_mat = np.diag([-1, -1, 1]).astype(np.float32)
#         R_p3d = flip_mat @ R_cv
#         T_p3d = flip_mat @ T_cv

#         # Assemble the 4x4 pose matrix in PyTorch3D's convention
#         pose_mat = np.eye(4)
#         pose_mat[:3, :3] = R_p3d
#         pose_mat[:3, 3] = T_p3d

#         pnp_poses[t] = pose_mat

#         # if inliers is not None and t % 10 == 0:
#         #     inlier_ratio = len(inliers) / len(visible_2d)
#         #     print(f"Frame {t}: PnP Success. Inliers: {len(inliers)}/{len(visible_2d)} ({inlier_ratio:.2f})")

#     return pnp_poses

# def run_pnp(model, video, queries_2d, queries_3d, K, R, T, device, output_dir=None, vis_threshold=0.5):
#     # Data prep
#     queries_2d = torch.from_numpy(queries_2d).float().to(device)
#     queries_3d = torch.from_numpy(queries_3d).float().to(device)
#     R = R.cpu().numpy().T  # P3D format (Row major) -> Col major for calculation
#     T = T.cpu().numpy()

#     # 1. run CoTracker
#     N = queries_2d.shape[0]
#     t_col = torch.zeros((N, 1), device=device)  # frame index is always 0
#     queries = torch.cat([t_col, queries_2d], dim=1)[None]  # (1, N, 3)

#     with torch.no_grad():
#         video = video.permute(0, 3, 1, 2).unsqueeze(0).contiguous()
#         pred_tracks, pred_vis = model(
#             video,
#             queries=queries,
#             backward_tracking=True,
#         )

#     # Save tracking visualization if an output directory is provided
#     if output_dir:
#         # The visualizer expects the video as a (B, T, C, H, W) tensor on CPU.
#         # video is (1, T, C, H, W), so we just move it to CPU.
#         vis = Visualizer(save_dir=output_dir, pad_value=0, linewidth=2)
#         vis.visualize(video, pred_tracks, pred_vis, filename="tracking_visualization.mp4")

#     tracks_2d = pred_tracks[0].cpu().numpy()
#     queries_3d = queries_3d.cpu().numpy()

#     T_frames = tracks_2d.shape[0]
#     pnp_poses = [None] * T_frames

#     # ================= 步骤 1: 定义转换矩阵 F =================
#     # F 用于在 (X-Left, Y-Up) 和 (X-Right, Y-Down) 之间转换
#     # 也就是 diag(-1, -1, 1)
#     F = np.diag([-1, -1, 1]).astype(np.float32)

#     # ================= 步骤 2: 将输入数据转为 OpenCV 格式 =================
#     # 2.1 转换 3D 点: P_cv = F @ P_p3d
#     # 也就是把 queries_3d 的 X 和 Y 坐标取反，Z 不变
#     queries_3d_cv = queries_3d.copy()
#     queries_3d_cv = (F @ queries_3d_cv.T).T  # 矩阵乘法转置技巧，或者直接 queries_3d_cv[:, :2] *= -1

#     # 2.2 转换初始 Pose Guess
#     # R_cv = F @ R_p3d @ F
#     # T_cv = F @ T_p3d
#     prev_R_cv = F @ R @ F
#     prev_T_cv = F @ T

#     prev_rvec = cv2.Rodrigues(prev_R_cv)[0]
#     prev_tvec = prev_T_cv
#     # ===================================================================

#     for t in range(T_frames):
#         cur_2d = tracks_2d[t]
#         visible_2d = cur_2d
#         # 注意：这里我们要用转换后的 OpenCV 格式的点
#         visible_3d_cv = queries_3d_cv

#         if len(visible_2d) < 10:
#             rvec, tvec = prev_rvec, prev_tvec
#         else:
#             # 使用标准的 OpenCV K (正焦距)
#             success_ransac, rvec_ransac, tvec_ransac, inliers = cv2.solvePnPRansac(
#                 visible_3d_cv,  # <--- 传入转换后的点
#                 visible_2d,
#                 K,  # <--- 传入原始的正 K
#                 None,
#                 iterationsCount=500,
#                 reprojectionError=2.0,
#                 confidence=0.99,
#                 flags=cv2.USAC_MAGSAC)

#             if success_ransac:
#                 success, rvec, tvec = cv2.solvePnP(
#                     visible_3d_cv[inliers],  # <--- 传入转换后的点
#                     visible_2d[inliers],
#                     K,  # <--- 传入原始的正 K
#                     None,
#                     rvec=rvec_ransac,
#                     tvec=tvec_ransac,
#                     useExtrinsicGuess=True,
#                     flags=cv2.SOLVEPNP_ITERATIVE)
#                 if success:
#                     prev_rvec = rvec
#                     prev_tvec = tvec
#             else:
#                 rvec, tvec = prev_rvec, prev_tvec

#         # ================= 步骤 3: 将结果转回 PyTorch3D 格式 =================
#         R_cv_result, _ = cv2.Rodrigues(rvec)
#         T_cv_result = tvec.flatten()

#         # 公式逆推: R_p3d = F @ R_cv @ F
#         R_p3d = F @ R_cv_result @ F
#         # 公式逆推: T_p3d = F @ T_cv
#         T_p3d = F @ T_cv_result

#         pose_mat = np.eye(4)
#         pose_mat[:3, :3] = R_p3d
#         pose_mat[:3, 3] = T_p3d

#         pnp_poses[t] = pose_mat
#         # ===================================================================

#     return pnp_poses


# ---------------------------------------------------------------------------
# PnP health monitor
# ---------------------------------------------------------------------------

def compute_pnp_health(
    pnp_rot_col_major: torch.Tensor,    # (N, 3, 3) PyTorch3D column-major (P3D-flipped from OpenCV)
    pnp_trans: torch.Tensor,            # (N, 3)
    verts: torch.Tensor,                # (V, 3) canonical (after twist)
    faces: torch.Tensor,                # (F, 3)
    scale_full: torch.Tensor,           # (3,)  = single_frame_model.scale * initial_scale
    amodal_masks_np: np.ndarray,        # (N, H, W) {0,1}
    focal_length: torch.Tensor,         # (N, 2)
    principal_point: torch.Tensor,      # (N, 2)
    H_out: int,
    W_out: int,
    device: torch.device,
    eval_frames: range = None,          # frames to evaluate (default: all)
):
    """Render the PnP pose at each frame and compute IoU vs amodal mask.

    Returns
    -------
    metrics : dict with numpy arrays of length N
        - 'iou'         : silhouette IoU per frame, NaN where not evaluated.
        - 'angle_jump_deg' : angle( R_pnp(i)^T R_pnp(i-1) ) in degrees, NaN @ i=0.
        - 'mask_area_px'   : amodal mask area in pixels per frame.
    """
    N = pnp_rot_col_major.shape[0]
    if eval_frames is None:
        eval_frames = range(N)

    iou_arr = np.full(N, np.nan, dtype=np.float32)
    angle_arr = np.full(N, np.nan, dtype=np.float32)
    area_arr = np.zeros(N, dtype=np.float32)

    cameras_all = PerspectiveCameras(
        focal_length=focal_length,
        principal_point=principal_point,
        image_size=((H_out, W_out),),
        in_ndc=False,
        device=device,
    )
    raster_settings = RasterizationSettings(image_size=(H_out, W_out), blur_radius=1e-4, faces_per_pixel=20)
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras_all, raster_settings=raster_settings),
        shader=SoftSilhouetteShader(),
    )

    eval_idx = list(eval_frames)
    if len(eval_idx) == 0:
        return {"iou": iou_arr, "angle_jump_deg": angle_arr, "mask_area_px": area_arr}

    # mT to convert PnP column-major -> row-major for `verts @ R + T` convention.
    R_row = pnp_rot_col_major[eval_idx].mT.contiguous()        # (M, 3, 3)
    T_row = pnp_trans[eval_idx].contiguous()                   # (M, 3)
    M = len(eval_idx)
    posed_verts = (verts * scale_full).unsqueeze(0).expand(M, -1, -1)
    posed_verts = posed_verts @ R_row + T_row.unsqueeze(1)
    posed_faces = faces.unsqueeze(0).expand(M, -1, -1).contiguous()
    posed_textures = TexturesVertex(verts_features=torch.ones_like(posed_verts))
    mesh_batch = Meshes(verts=posed_verts, faces=posed_faces, textures=posed_textures)

    cameras_eval = PerspectiveCameras(
        focal_length=focal_length[eval_idx],
        principal_point=principal_point[eval_idx],
        image_size=((H_out, W_out),),
        in_ndc=False,
        device=device,
    )
    with torch.no_grad():
        fragments = renderer.rasterizer(mesh_batch, cameras=cameras_eval)
        rendered = renderer.shader(fragments, mesh_batch, cameras=cameras_eval)[..., 3]  # (M, H, W)
        rendered_bin = (rendered > 0.5).cpu().numpy()
    for ii, fi in enumerate(eval_idx):
        gt = amodal_masks_np[fi].astype(bool)
        pr = rendered_bin[ii]
        inter = np.logical_and(gt, pr).sum()
        union = np.logical_or(gt, pr).sum()
        iou_arr[fi] = (inter / union) if union > 0 else 0.0
        area_arr[fi] = float(gt.sum())

    R_world = pnp_rot_col_major.mT  # (N, 3, 3) row-major for angle arithmetic
    for fi in eval_idx:
        if fi == 0:
            continue
        rel = R_world[fi - 1].T @ R_world[fi]
        cos = ((rel.diagonal().sum() - 1.0) * 0.5).clamp(-1.0, 1.0)
        angle_arr[fi] = float(torch.acos(cos).item() * 180.0 / math.pi)

    return {"iou": iou_arr, "angle_jump_deg": angle_arr, "mask_area_px": area_arr}


def flag_reset_candidates(
    metrics: dict,
    iou_threshold: float = 0.5,
    iou_relative_margin: float = 0.15,
    iou_drop_threshold: float = 0.2,
    angle_jump_threshold_deg: float = 30.0,
    consecutive_required: int = 3,
    cooldown: int = 3,
    eval_start: int = 0,
):
    """Return list[(frame_idx, reason_str)] of dry-run reset candidates.

    The effective IoU threshold is computed adaptively:

        eff_iou = min(iou_threshold, max(0.0, median(iou) - iou_relative_margin))

    so that low-baseline sequences (e.g. amodal masks systematically off,
    typical IoU ~0.5) do not trigger constantly, and high-baseline sequences
    can still catch sub-0.5 dips early.

    Trigger rules (logical OR):
      A. IoU < ``eff_iou`` for ``consecutive_required`` consecutive frames.
         Trigger fires at the first frame of the streak.
      B. Single-frame IoU drop >= ``iou_drop_threshold`` AND new IoU below
         ``eff_iou + 0.1``.
      C. Single-frame angle jump >= ``angle_jump_threshold_deg`` AND IoU also
         below ``eff_iou + 0.1`` (combined to avoid firing on legitimate
         fast rotations that PnP still tracks).

    A ``cooldown`` of ``cooldown`` frames is applied after each trigger to
    avoid a cluster of redundant resets.
    """
    iou = metrics["iou"]
    djump = metrics["angle_jump_deg"]
    N = len(iou)
    iou_eval = iou[eval_start:N]
    iou_eval_finite = iou_eval[np.isfinite(iou_eval)]
    if len(iou_eval_finite) > 0:
        baseline = float(np.median(iou_eval_finite))
        eff_iou = min(iou_threshold, max(0.0, baseline - iou_relative_margin))
    else:
        baseline = float("nan")
        eff_iou = iou_threshold

    out = []
    last_trig = -10**9
    streak = 0
    prev_iou = None
    for fi in range(eval_start, N):
        if not np.isfinite(iou[fi]):
            streak = 0
            prev_iou = None
            continue
        if iou[fi] < eff_iou:
            streak += 1
        else:
            streak = 0

        reasons = []
        if streak >= consecutive_required:
            reasons.append(f"iou<{eff_iou:.3f} for {streak} consecutive frames (cur={iou[fi]:.3f}, baseline={baseline:.3f})")
        if prev_iou is not None and (prev_iou - iou[fi]) >= iou_drop_threshold and iou[fi] < (eff_iou + 0.1):
            reasons.append(f"iou dropped {prev_iou - iou[fi]:.3f} (prev={prev_iou:.3f} cur={iou[fi]:.3f})")
        if (np.isfinite(djump[fi]) and djump[fi] >= angle_jump_threshold_deg and iou[fi] < (eff_iou + 0.1)):
            reasons.append(f"angle_jump {djump[fi]:.1f} deg with iou={iou[fi]:.3f}")

        if reasons and (fi - last_trig) >= cooldown:
            trig_frame = max(eval_start, fi - max(0, streak - 1)) if "consecutive frames" in reasons[0] else fi
            out.append((int(trig_frame), "; ".join(reasons)))
            last_trig = fi
            streak = 0
        prev_iou = iou[fi]

    summary = {"baseline_iou_median": baseline, "effective_iou_threshold": eff_iou}
    return out, summary
