"""
Farthest point sampling on object meshes.
"""
import argparse
import math
import os

import numpy as np
import open3d as o3d
import pytorch3d.ops
import pytorch3d.structures
import torch
import torch.multiprocessing as mp
import trimesh as tm
from tqdm import tqdm


def export_ply_with_normals(path: str, points: np.ndarray, normals: np.ndarray):
    """Export point cloud with normals to a PLY file via open3d."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.normals = o3d.utility.Vector3dVector(normals.astype(np.float64))
    o3d.io.write_point_cloud(path, pcd)


def worker(rank, world_size, args):
    """
    Worker process for point cloud sampling.
    """
    device = torch.device(f'cuda:{rank}')
    print(f"--> Starting worker on rank {rank} (GPU {device})")

    # Distribute object codes
    all_object_codes = args.object_code_list
    chunk_size = math.ceil(len(all_object_codes) / world_size)
    start_index = rank * chunk_size
    end_index = min((rank + 1) * chunk_size, len(all_object_codes))
    object_code_subset = all_object_codes[start_index:end_index]

    if not object_code_subset:
        print(f"Rank {rank} has no object codes to process. Exiting.")
        return

    print(f"Rank {rank} processing {len(object_code_subset)} object codes: from index {start_index} to {end_index-1}")

    # Use tqdm for rank 0 only
    iterator = object_code_subset
    if rank == 0:
        iterator = tqdm(object_code_subset, desc='Processing objects')

    num_samples_dexgraspnet = 2000
    num_samples_grasp_generation = 10000

    for object_dir in iterator:
        obj_path = os.path.join(object_dir, "decomposed.obj")
        if not os.path.exists(obj_path):
            if rank == 0:
                print(f"Skipping {object_dir}, decomposed.obj not found.")
            continue

        try:
            obj_mesh = tm.load(obj_path, force="mesh", process=False)
            vertices = torch.tensor(obj_mesh.vertices, dtype=torch.float, device=device)
            faces = torch.tensor(obj_mesh.faces, dtype=torch.long, device=device)
            mesh = pytorch3d.structures.Meshes(vertices.unsqueeze(0), faces.unsqueeze(0))

            # Densely sample points and normals for FPS
            dense_point_cloud, dense_normals = pytorch3d.ops.sample_points_from_meshes(mesh, num_samples=100 * num_samples_dexgraspnet, return_normals=True)
            # dense_point_cloud: (1, N, 3), dense_normals: (1, N, 3)

            # Sample for DexGraspNet
            fps_pts_dex, fps_idx_dex = pytorch3d.ops.sample_farthest_points(dense_point_cloud, K=num_samples_dexgraspnet)
            surface_points_dexgraspnet = fps_pts_dex[0]  # (K, 3)
            surface_normals_dexgraspnet = dense_normals[0][fps_idx_dex[0]]  # (K, 3)
            export_ply_with_normals(
                os.path.join(object_dir, f"obj_points_{num_samples_dexgraspnet}.ply"),
                surface_points_dexgraspnet.cpu().numpy(),
                surface_normals_dexgraspnet.cpu().numpy(),
            )

            # Sample for Grasp Generation
            fps_pts_grasp, fps_idx_grasp = pytorch3d.ops.sample_farthest_points(dense_point_cloud, K=num_samples_grasp_generation)
            surface_points_grasp_generation = fps_pts_grasp[0]  # (K, 3)
            surface_normals_grasp_generation = dense_normals[0][fps_idx_grasp[0]]  # (K, 3)
            export_ply_with_normals(
                os.path.join(object_dir, f"obj_points_{num_samples_grasp_generation}.ply"),
                surface_points_grasp_generation.cpu().numpy(),
                surface_normals_grasp_generation.cpu().numpy(),
            )
        except Exception as e:
            if rank == 0:
                print(f"Error processing {object_dir}: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--object_code_list', nargs='+', default=[], help="List of object codes to process.")
    parser.add_argument('--data_root',
                        type=str,
                        default='/vepfs_default/chanxueyan/lhp/xh/code/DexGraspNet_table/meshdata',
                        help="Root directory for mesh data.")
    args = parser.parse_args()

    # Discover and filter object_codes in the main process
    if not args.object_code_list:
        print("object_code_list not specified, discovering objects in data_root")
        all_object_dirs = []
        for source in ['sam3d', 'dexgraspnet']:
            source_path = os.path.join(args.data_root, source)
            if os.path.isdir(source_path):
                for obj_code in os.listdir(source_path):
                    obj_dir = os.path.join(source_path, obj_code)
                    if os.path.isdir(obj_dir):
                        all_object_dirs.append(obj_dir)
        args.object_code_list = all_object_dirs
        print(f"Found {len(args.object_code_list)} object directories")
    else:
        # If object codes are provided, construct full paths
        all_object_dirs = []
        for code in args.object_code_list:
            found = False
            for source in ['sam3d', 'dexgraspnet']:
                obj_dir = os.path.join(args.data_root, source, code)
                if os.path.isdir(obj_dir):
                    all_object_dirs.append(obj_dir)
                    found = True
                    break
            if not found:
                print(f"Warning: object code {code} not found in sam3d or dexgraspnet directories.")
        args.object_code_list = all_object_dirs

    # Spawn worker processes
    world_size = torch.cuda.device_count()
    if world_size > 1:
        print(f"Found {world_size} GPUs. Spawning worker processes.")
        mp.spawn(worker, args=(world_size, args), nprocs=world_size, join=True)
    else:
        print("Found 1 or 0 GPUs. Running in single-process mode.")
        worker(0, 1, args)
