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
from pytorch3d.io import save_obj, save_ply
from tqdm import tqdm


def save_hoi_sequence(output_dir, canonical_verts, canonical_faces, obj_scales, obj_rots_col_major, obj_trans, mano_root_orient, mano_pose, mano_trans,
                      is_right, obj_verts_seq, obj_faces, hand_verts_seq, hand_faces, paper_seq_name=None, mesh_root=None):
    """
    Saves the optimized and interpolated hand-object sequence to disk.
    - Canonical object mesh
    - Per-frame object pose JSON
    - Per-frame MANO parameters JSON
    - Per-frame posed hand/object PLY meshes under ``mesh_root/{hand,object}/``
    """
    print("--- Saving Optimized HOI Sequence ---")
    os.makedirs(output_dir, exist_ok=True)
    if mesh_root is None:
        raise ValueError("mesh_root is required (expected stage3/final/optimized_meshes)")
    hand_mesh_dir = os.path.join(mesh_root, "hand")
    obj_mesh_dir = os.path.join(mesh_root, "object")
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

    print(f"  -> Saved {num_frames} frames of pose/MANO to {output_dir}")
    print(f"  -> Saved posed meshes to {mesh_root}")


def save_k3d_visualization(output_path, seq_path, obj_verts_seq, obj_faces, hand_verts_seq, hand_faces, target_fps=30.0, html_path=None, env_depths_dir=None):
    """Saves an interactive 3D visualization of hand-object interaction sequence to an HTML file."""
    print("--- Generating K3D Visualization ---")

    # ================= Camera setup (Z-Forward, -Y Up) =================
    focal_length_mm = 25.0
    sensor_height_mm = 24.0
    fov_degrees = 2 * math.atan(sensor_height_mm / (2 * focal_length_mm)) * (180 / math.pi)
    camera_config = [0, 0, 0, 0, 0, 1, 0, -1, 0]  # [Pos, Target, Up]

    # ================= Helpers =================
    def process_sequence_from_memory(all_verts_np, fps):
        """Converts a numpy vertex sequence to a k3d-compatible animation dictionary."""
        vertices_seq_dict = {}
        num_frames = all_verts_np.shape[0]
        for frame_idx in range(num_frames):
            time_stamp = frame_idx / fps
            vertices_seq_dict[str(time_stamp)] = all_verts_np[frame_idx].astype(np.float32)
        return vertices_seq_dict

    # ================= Main =================
    if html_path is None:
        html_path = os.path.join(output_path, "visualization_3d.html")
    output_filename = html_path
    os.makedirs(os.path.dirname(os.path.abspath(output_filename)), exist_ok=True)

    plot = k3d.plot(name=f"Vis: {os.path.basename(output_path)}", camera_auto_fit=False, camera_fov=fov_degrees, grid_visible=False, fps=target_fps)

    # --- Load hand sequence ---
    if hand_verts_seq is not None and hand_faces is not None:
        hand_k3d_verts = process_sequence_from_memory(hand_verts_seq, target_fps)
        if hand_k3d_verts:
            start_t = list(hand_k3d_verts.keys())[0]
            hand_mesh = k3d.mesh(hand_k3d_verts[start_t], hand_faces, color=0xffcccc, side='double', name="Hand")
            hand_mesh.vertices = hand_k3d_verts
            plot += hand_mesh
            print(f"  - Hand mesh loaded: {len(hand_verts_seq)} frames")

    # --- Load object sequence ---
    if obj_verts_seq is not None and obj_faces is not None:
        obj_k3d_verts = process_sequence_from_memory(obj_verts_seq, target_fps)
        if obj_k3d_verts:
            start_t = list(obj_k3d_verts.keys())[0]
            obj_mesh = k3d.mesh(obj_k3d_verts[start_t], obj_faces, color=0x00aaff, side='double', name="Object")
            obj_mesh.vertices = obj_k3d_verts
            plot += obj_mesh
            print(f"  - Object mesh loaded: {len(obj_verts_seq)} frames")

    # --- D. Set camera and export ---
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
