import argparse
from pathlib import Path

import numpy as np
import torch
import trimesh
from pytorch3d.transforms import matrix_to_axis_angle, rotation_6d_to_matrix

from manotorch.manolayer import ManoLayer
from data_layout import DEFAULT_CONTACT_INDICES, DEFAULT_MANO_ASSETS, DEFAULT_SOURCE_DIR


def decode_mano_params(hand_tensor):
    """
    Decode a (17, 6) hand representation into MANO params (translation, global rotation, pose).
    """
    if not isinstance(hand_tensor, torch.Tensor):
        hand_tensor = torch.from_numpy(hand_tensor)

    trans = hand_tensor[0, :3].unsqueeze(0)  # (1, 3)
    root_rot_6d = hand_tensor[1, :].unsqueeze(0)  # (1, 6)
    pose_6d = hand_tensor[2:, :]  # (15, 6)

    # 6D -> axis-angle
    root_rot_mat = rotation_6d_to_matrix(root_rot_6d)
    root_rot_aa = matrix_to_axis_angle(root_rot_mat)  # (1, 3)

    pose_mat = rotation_6d_to_matrix(pose_6d)
    pose_aa = matrix_to_axis_angle(pose_mat)  # (15, 3)

    # Concatenate into the format expected by ManoLayer
    pose_coeffs = torch.cat([root_rot_aa, pose_aa.view(1, 45)], dim=1)  # (1, 48)

    return trans, pose_coeffs


def get_hand_pointcloud(trans, pose_coeffs, mano_layer):
    """
    Generate hand vertices with ManoLayer.
    """
    mano_output = mano_layer(pose_coeffs)
    verts = mano_output.verts + trans
    return verts[0].detach().cpu().numpy()


def main(args):
    # Load data
    data_path = Path(args.file_path)
    if not data_path.exists():
        print(f"Error: File not found at {data_path}")
        return

    print(f"Loading data from {data_path}...")
    saved_data = torch.load(data_path, map_location='cpu', weights_only=False)

    # Visualize one sample (default index 0)
    for sample_idx in range(saved_data['pred_hand'].shape[0]):
        pred_hand = saved_data['pred_hand'][sample_idx]
        gt_hand = saved_data['gt_hand'][sample_idx]
        cond_hand = saved_data['cond_hand'][sample_idx]
        obj_pc = saved_data['cond_obj_pointcloud'][sample_idx]
        obj_scale = saved_data['cond_obj_scale'][sample_idx]

        # Initialize ManoLayer
        print("Initializing MANO layer...")
        mano_layer = ManoLayer(
            mano_assets_root=str(DEFAULT_MANO_ASSETS),
            side='right',
            use_pca=False,
            flat_hand_mean=True,
        )

        # --- Hands ---
        print("Processing hand poses...")
        # Faces from ManoLayer
        faces = mano_layer.th_faces.detach().cpu().numpy()

        # GT hand (green)
        gt_trans, gt_pose_coeffs = decode_mano_params(gt_hand)
        gt_trans = gt_trans * obj_scale
        gt_hand_verts = get_hand_pointcloud(gt_trans, gt_pose_coeffs, mano_layer)
        gt_mesh = trimesh.Trimesh(vertices=gt_hand_verts, faces=faces)
        gt_mesh.visual.vertex_colors = [0, 255, 0, 200]

        # Pred hand (red)
        pred_trans, pred_pose_coeffs = decode_mano_params(pred_hand)
        pred_trans = pred_trans * obj_scale
        pred_hand_verts = get_hand_pointcloud(pred_trans, pred_pose_coeffs, mano_layer)
        pred_mesh = trimesh.Trimesh(vertices=pred_hand_verts, faces=faces)
        pred_mesh.visual.vertex_colors = [255, 0, 0, 200]

        # Cond hand (blue)
        cond_trans, cond_pose_coeffs = decode_mano_params(cond_hand)
        cond_trans = cond_trans * obj_scale
        cond_hand_verts = get_hand_pointcloud(cond_trans, cond_pose_coeffs, mano_layer)
        cond_mesh = trimesh.Trimesh(vertices=cond_hand_verts, faces=faces)
        cond_mesh.visual.vertex_colors = [0, 0, 255, 200]

        # --- Object ---
        print("Processing object point cloud...")
        # Restore scale
        obj_pc_rescaled = obj_pc.numpy() * obj_scale.numpy()

        # Represent each point as a small sphere for clearer visualization
        # Build one merged mesh manually instead of thousands of trimesh objects
        if obj_pc_rescaled.shape[0] > 0:
            # Base low-poly sphere for performance
            sphere = trimesh.creation.icosphere(radius=0.001, subdivisions=1)
            base_vertices = sphere.vertices
            base_faces = sphere.faces

            # Collect vertices and faces
            all_vertices = []
            all_faces = []
            vertex_offset = 0

            for point in obj_pc_rescaled:
                all_vertices.append(base_vertices + point)
                all_faces.append(base_faces + vertex_offset)
                vertex_offset += len(base_vertices)

            # Concatenate into single arrays
            final_vertices = np.concatenate(all_vertices, axis=0)
            final_faces = np.concatenate(all_faces, axis=0)

            # Final object mesh
            object_mesh = trimesh.Trimesh(vertices=final_vertices, faces=final_faces)
            object_mesh.visual.vertex_colors = [128, 128, 128, 255]
        else:
            # Empty mesh if there are no points
            object_mesh = trimesh.Trimesh()

        # --- Merge and save ---
        # Combine clouds/meshes with trimesh.Scene
        scene = trimesh.Scene()
        scene.add_geometry(object_mesh)
        scene.add_geometry(gt_mesh)
        scene.add_geometry(pred_mesh)
        scene.add_geometry(cond_mesh)

        output_filename = data_path.stem + f"_sample_{sample_idx}.ply"
        output_path = Path("vis_results") / output_filename
        output_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"Saving visualization to {output_path}...")
        scene.export(output_path)
    print("Done.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize grasp results saved during training.")
    parser.add_argument("--file_path", type=str, help="Path to the .pt file containing saved samples.")
    args = parser.parse_args()
    main(args)
