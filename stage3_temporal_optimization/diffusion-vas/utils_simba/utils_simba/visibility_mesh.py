import argparse
import numpy as np
import trimesh
import open3d as o3d
import os

import sys
sys.path.append("./third_party/utils_simba")
from utils_simba.visibility_test import mesh_to_voxel_grid, voxel_centers, determine_visibility
from utils_simba.visibility_test import create_colored_mesh_from_voxels

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh_file", type=str, help="mesh file")
    parser.add_argument("--c2w", type=str, help="camera poses from camera to world")
    parser.add_argument("--out_dir", type=str, help="output directory")
    parser.add_argument("--mute", action="store_true")


    args = parser.parse_args()

    return args


def get_visibility_mesh(mesh_file, c2w_np_type, out_dir, idx):

    voxel_size = 0.02  # Adjust voxel size as needed
    num_views = 5  # Number of camera views
    max_distance = 1000  # Maximum distance for visibility checks
    # Load mesh
    mesh = trimesh.load(mesh_file)
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = mesh.dump().sum()  # In case the mesh is a scene with multiple meshes
    # Voxelization

    voxels, voxel_origin, grid_size = mesh_to_voxel_grid(mesh, voxel_size)
    print(f"Voxel grid size: {grid_size}")

    camera_poses = c2w_np_type

    # Compute voxel centers
    voxel_centers_np, occupied_indices = voxel_centers(voxels, voxel_origin, voxel_size)
    print(f"Number of occupied voxels: {voxel_centers_np.shape[0]}")

    # Determine visibility
    try:
        # Try to use PyEmbree for faster ray tracing
        visibility = determine_visibility(mesh, voxel_centers_np, camera_poses, max_distance)
    except ImportError:
        print("PyEmbree not installed, falling back to slower ray tracing.")
        trimesh.ray.ray_pyembree.RayMeshIntersector = None
        visibility = determine_visibility(mesh, voxel_centers_np, camera_poses, max_distance)

    # # Example: Print visibility of first 10 voxels
    # for i in range(min(10, visibility.shape[0])):
    #     print(f"Voxel {i}: Visible from views {np.where(visibility[i])[0]}")
    
    # Create colored mesh from voxels
    mesh_with_visibility = create_colored_mesh_from_voxels(
        voxels, voxel_origin, voxel_size, visibility
    )
    # Optional: Save the mesh to a file
    o3d.io.write_triangle_mesh(f"{out_dir}/mesh_with_visibility_{idx}.ply", mesh_with_visibility)
    
    return visibility

def combile_visibility_mesh(mesh_file, visibilities, out_dir):

    voxel_size = 0.02  # Adjust voxel size as needed
    # Load mesh
    mesh = trimesh.load(mesh_file)
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = mesh.dump().sum()  # In case the mesh is a scene with multiple meshes
    # Voxelization

    
    voxels, voxel_origin, grid_size = mesh_to_voxel_grid(mesh, voxel_size)
    visibility = np.zeros_like((visibilities[0])).astype(bool)
    for vis in visibilities:
        visibility = np.logical_or(visibility, vis)

    # Create colored mesh from voxels
    mesh_with_visibility = create_colored_mesh_from_voxels(
        voxels, voxel_origin, voxel_size, visibility
    )
    # Optional: Save the mesh to a file
    o3d.io.write_triangle_mesh(f"{out_dir}/mesh_with_visibility.ply", mesh_with_visibility)

    return visibility

if __name__ == "__main__":
    args = parse_args()

    mesh_file = args.mesh_file
    c2w = args.c2w
    out_dir = args.out_dir

    os.makedirs(out_dir, exist_ok=True)
    get_visibility_mesh(mesh_file, c2w, out_dir)