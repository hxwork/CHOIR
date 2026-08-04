"""HOI sequence I/O and visualisation helpers.

Contains:
  - save_hoi_sequence  : persist per-frame pose / MANO params and posed meshes
  - save_k3d_visualization : write an interactive K3D HTML viewer
"""

import json
import math
import os

import k3d
import numpy as np
import torch
import trimesh
from pytorch3d.io import save_obj, save_ply
from tqdm import tqdm


def _as_numpy(array):
    if isinstance(array, torch.Tensor):
        return array.detach().cpu().numpy()
    return np.asarray(array)


def _confidence_to_vertex_colors(confidence):
    """Map [0, 1] confidence to a dark-blue -> red RGBA ramp."""
    conf = np.clip(np.asarray(confidence, dtype=np.float32), 0.0, 1.0)
    low = np.array([35, 45, 70], dtype=np.float32)
    high = np.array([255, 45, 35], dtype=np.float32)
    rgb = low[None, :] * (1.0 - conf[:, None]) + high[None, :] * conf[:, None]
    alpha = np.full((conf.shape[0], 1), 255.0, dtype=np.float32)
    return np.concatenate([rgb, alpha], axis=1).round().astype(np.uint8)


def contact_cache_to_dense_hand_confidence(
    cache,
    num_hand_verts=778,
    confidence_attr="contact_confidence",
):
    """Convert padded active contact confidence to dense hand-vertex values."""
    conf_full = np.zeros((int(cache.active_idx.shape[0]), int(num_hand_verts)), dtype=np.float32)
    cache_conf = getattr(cache, confidence_attr, None)
    if cache_conf is None:
        return conf_full
    cache_conf_np = _as_numpy(cache_conf).astype(np.float32)
    if cache_conf_np.shape == conf_full.shape:
        return np.clip(cache_conf_np, 0.0, 1.0)

    active_idx = _as_numpy(cache.active_idx).astype(np.int64)
    active_mask = _as_numpy(cache.active_mask).astype(bool)
    active_conf = cache_conf_np
    active_conf = np.where(active_mask, active_conf, 0.0)
    for frame_idx in range(conf_full.shape[0]):
        valid = active_mask[frame_idx]
        if valid.any():
            conf_full[frame_idx, active_idx[frame_idx, valid]] = active_conf[frame_idx, valid]
    return np.clip(conf_full, 0.0, 1.0)


def contact_cache_to_object_vertex_confidence(
    cache,
    obj_faces,
    num_obj_verts,
    confidence_attr="contact_confidence",
    hand_verts=None,
    obj_verts=None,
):
    """Project active hand contact confidence to object vertices through anchors."""
    obj_faces_np = _as_numpy(obj_faces).astype(np.int64)
    obj_conf = np.zeros((int(cache.active_idx.shape[0]), int(num_obj_verts)), dtype=np.float32)
    cache_conf = getattr(cache, confidence_attr, None)
    if cache_conf is None:
        return obj_conf
    cache_conf_np = _as_numpy(cache_conf).astype(np.float32)
    if cache_conf_np.shape == obj_conf.shape[:1] + (cache_conf_np.shape[1],) and hand_verts is not None and obj_verts is not None:
        hand_verts_np = _as_numpy(hand_verts).astype(np.float32)
        obj_verts_np = _as_numpy(obj_verts).astype(np.float32)
        if hand_verts_np.shape[:2] != cache_conf_np.shape:
            raise ValueError(
                "hand_verts must have shape matching dense confidence "
                f"{cache_conf_np.shape}, got {hand_verts_np.shape[:2]}"
            )
        if obj_verts_np.shape[0] != cache_conf_np.shape[0] or obj_verts_np.shape[1] != int(num_obj_verts):
            raise ValueError(
                "obj_verts must have shape (N_frames, num_obj_verts, 3), got "
                f"{obj_verts_np.shape}"
            )
        for frame_idx in range(obj_conf.shape[0]):
            positive_idx = np.nonzero(cache_conf_np[frame_idx] > 0.0)[0]
            if positive_idx.size == 0:
                continue
            hand_pts = hand_verts_np[frame_idx, positive_idx]
            diff = hand_pts[:, None, :] - obj_verts_np[frame_idx][None, :, :]
            nearest_obj = np.sum(diff * diff, axis=-1).argmin(axis=1)
            for hand_col, obj_v in zip(positive_idx, nearest_obj):
                obj_conf[frame_idx, int(obj_v)] = max(
                    obj_conf[frame_idx, int(obj_v)],
                    float(cache_conf_np[frame_idx, hand_col]),
                )
        return np.clip(obj_conf, 0.0, 1.0)

    active_mask = _as_numpy(cache.active_mask).astype(bool)
    active_conf = cache_conf_np
    face_id_topk = _as_numpy(cache.face_id_topk).astype(np.int64)
    bary_topk = _as_numpy(cache.bary_topk).astype(np.float32)
    weight_topk = _as_numpy(cache.weight_topk).astype(np.float32)
    best_k = weight_topk.argmax(axis=-1)

    for frame_idx in range(obj_conf.shape[0]):
        for active_col in np.nonzero(active_mask[frame_idx])[0]:
            face_id = int(face_id_topk[frame_idx, active_col, best_k[frame_idx, active_col]])
            if face_id < 0 or face_id >= obj_faces_np.shape[0]:
                continue
            conf = float(active_conf[frame_idx, active_col])
            if conf <= 0.0:
                continue
            bary = np.clip(bary_topk[frame_idx, active_col, best_k[frame_idx, active_col]], 0.0, 1.0)
            for obj_v, weight in zip(obj_faces_np[face_id], bary):
                if 0 <= int(obj_v) < int(num_obj_verts):
                    obj_conf[frame_idx, int(obj_v)] = max(obj_conf[frame_idx, int(obj_v)], conf * float(weight))
    return np.clip(obj_conf, 0.0, 1.0)


def compute_distance_contact_confidence(hand_verts_seq, obj_verts_seq, distance_scale=0.02):
    """Compute per-frame distance-only contact confidence for hand and object vertices."""
    hand_verts_np = _as_numpy(hand_verts_seq).astype(np.float32)
    obj_verts_np = _as_numpy(obj_verts_seq).astype(np.float32)
    if hand_verts_np.ndim != 3 or obj_verts_np.ndim != 3:
        raise ValueError("hand_verts_seq and obj_verts_seq must have shape (N_frames, N_verts, 3)")
    if hand_verts_np.shape[0] != obj_verts_np.shape[0]:
        raise ValueError(
            "hand_verts_seq and obj_verts_seq must have the same number of frames, "
            f"got {hand_verts_np.shape[0]} and {obj_verts_np.shape[0]}"
        )

    scale = max(float(distance_scale), 1e-12)
    hand_conf = np.zeros(hand_verts_np.shape[:2], dtype=np.float32)
    obj_conf = np.zeros(obj_verts_np.shape[:2], dtype=np.float32)
    for frame_idx in range(hand_verts_np.shape[0]):
        diff = hand_verts_np[frame_idx][:, None, :] - obj_verts_np[frame_idx][None, :, :]
        dist_sq = np.sum(diff * diff, axis=-1)
        nearest_obj = dist_sq.argmin(axis=1)
        nearest_dist = np.sqrt(dist_sq[np.arange(dist_sq.shape[0]), nearest_obj])
        conf = np.exp(-((nearest_dist / scale) ** 2)).astype(np.float32)
        hand_conf[frame_idx] = conf
        for hand_idx, obj_idx in enumerate(nearest_obj):
            obj_conf[frame_idx, int(obj_idx)] = max(obj_conf[frame_idx, int(obj_idx)], float(conf[hand_idx]))
    return np.clip(hand_conf, 0.0, 1.0), np.clip(obj_conf, 0.0, 1.0)


def interpolate_vertex_confidence(sampled_indices, sampled_confidence, num_frames):
    """Linearly expand sampled per-vertex confidence to the full frame timeline."""
    sampled_t = np.asarray(sampled_indices, dtype=np.float32)
    sampled_conf = np.asarray(sampled_confidence, dtype=np.float32)
    full_t = np.arange(int(num_frames), dtype=np.float32)
    full_conf = np.zeros((int(num_frames), sampled_conf.shape[1]), dtype=np.float32)
    if sampled_t.size == 0:
        return full_conf

    order = np.argsort(sampled_t)
    sampled_t = sampled_t[order]
    sampled_conf = sampled_conf[order]
    for vertex_idx in range(sampled_conf.shape[1]):
        full_conf[:, vertex_idx] = np.interp(
            full_t,
            sampled_t,
            sampled_conf[:, vertex_idx],
            left=sampled_conf[0, vertex_idx],
            right=sampled_conf[-1, vertex_idx],
        )
    return np.clip(full_conf, 0.0, 1.0)


def save_contactmap_sequence(
    video_id,
    obj_verts_seq,
    obj_faces,
    hand_verts_seq,
    hand_faces,
    hand_confidence_seq,
    obj_confidence_seq,
    rendering_root=os.path.join("rendering_for_paper", "ours_SelfCaptured"),
    sequence_suffix="contactmap",
):
    """Save colored contactmap meshes using the paper-rendering sequence layout."""
    seq_dir = os.path.join(rendering_root, f"{video_id}_{sequence_suffix}")
    hand_mesh_dir = os.path.join(seq_dir, "hand")
    obj_mesh_dir = os.path.join(seq_dir, "object")
    os.makedirs(hand_mesh_dir, exist_ok=True)
    os.makedirs(obj_mesh_dir, exist_ok=True)

    obj_faces_np = _as_numpy(obj_faces).astype(np.int64)
    hand_faces_np = _as_numpy(hand_faces).astype(np.int64)
    obj_verts_np = _as_numpy(obj_verts_seq).astype(np.float32)
    hand_verts_np = _as_numpy(hand_verts_seq).astype(np.float32)
    hand_conf_np = np.asarray(hand_confidence_seq, dtype=np.float32)
    obj_conf_np = np.asarray(obj_confidence_seq, dtype=np.float32)

    for i in tqdm(range(obj_verts_np.shape[0]), desc="Saving contactmap sequence"):
        obj_mesh = trimesh.Trimesh(vertices=obj_verts_np[i], faces=obj_faces_np, process=False)
        obj_mesh.visual.vertex_colors = _confidence_to_vertex_colors(obj_conf_np[i])
        obj_mesh.export(os.path.join(obj_mesh_dir, f"{i:04d}_mesh.ply"), file_type="ply")

        hand_mesh = trimesh.Trimesh(vertices=hand_verts_np[i], faces=hand_faces_np, process=False)
        hand_mesh.visual.vertex_colors = _confidence_to_vertex_colors(hand_conf_np[i])
        hand_mesh.export(os.path.join(hand_mesh_dir, f"{i:04d}_hand.ply"), file_type="ply")

    print(f"  -> Saved contactmap meshes to {seq_dir}")
    return seq_dir


def save_hoi_sequence(output_dir, canonical_verts, canonical_faces, obj_scales, obj_rots_col_major, obj_trans, mano_root_orient, mano_pose, mano_trans,
                      is_right, obj_verts_seq, obj_faces, hand_verts_seq, hand_faces, paper_seq_name=None):
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
    seq_name = paper_seq_name or os.path.basename(os.path.dirname(output_dir))
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
