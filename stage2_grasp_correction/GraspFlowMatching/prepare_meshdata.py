"""
Prepare object meshes for GraspFlowMatching inference (packaging into meshdata/).

Prefer stage2_grasp_correction/DexGraspNet_table/prepare_meshdata.py when preparing training meshdata;
this copy remains for GFM inference / diffusion-vas callers.
"""
import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import open3d as o3d
import pytorch3d.ops
import pytorch3d.structures
import torch
import torch.multiprocessing as mp
import trimesh
import trimesh as tm
from tqdm import tqdm

from data_layout import DEFAULT_MESHDATA, DEFAULT_SOURCE_DIR

# ---------------------------------------------------------------------------
# Step 1 – prepare sam3d meshes
# ---------------------------------------------------------------------------


def prepare_sam3d(source_base_dir: Path, dest_base_dir: Path, video_ids=None, seq_dir_name="stage3/intermediates/optimized_hoi_seq"):
    """
    Read HOI mesh outputs and write meshdata/sam3d/<video_id>/:
      - obj_canonical.obj -> flip x,y -> decomposed.obj
      - obj_00000.json    -> extract "scale" -> scale.json
    """
    exclude_dirs = {"dynhamr", "images"}

    if not source_base_dir.is_dir():
        print(f"[Step 1] error: source directory {source_base_dir} not found, skipping.")
        return

    print(f"[Step 1] scanning {source_base_dir} ...")

    if video_ids:
        video_dirs = []
        for vid in video_ids:
            d = source_base_dir / vid
            if d.is_dir():
                video_dirs.append(d)
            else:
                print(f"[Step 1] warning: video_id '{vid}' not found, skipping.")
    else:
        video_dirs = [d for d in source_base_dir.iterdir() if d.is_dir() and d.name not in exclude_dirs]

    processed_files = 0
    processed_dirs = 0
    skipped_dirs = []

    for video_dir in tqdm(video_dirs, desc="[Step 1] preparing sam3d meshes"):
        video_id = video_dir.name
        seq_dir = video_dir / seq_dir_name

        source_obj = seq_dir / "obj_canonical.obj"
        source_json = seq_dir / "obj_00000.json"

        if not source_obj.is_file() and not source_json.is_file():
            skipped_dirs.append(video_id)
            continue

        dest_dir = dest_base_dir / video_id
        dest_dir.mkdir(parents=True, exist_ok=True)
        processed_dirs += 1

        if source_obj.is_file():
            dest_obj = dest_dir / "decomposed.obj"
            try:
                mesh = trimesh.load(source_obj, force="mesh", process=False)
                mesh.vertices = mesh.vertices @ np.diag([-1, -1, 1])
                mesh.export(dest_obj)
                processed_files += 1
            except Exception as e:
                print(f"[Step 1] error processing {source_obj}: {e}")

        if source_json.is_file():
            try:
                with source_json.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                if "scale" in data:
                    dest_scale = dest_dir / "scale.json"
                    with dest_scale.open("w", encoding="utf-8") as f:
                        json.dump({"scale": data["scale"]}, f, indent=4)
                    processed_files += 1
                else:
                    print(f"[Step 1] warning: 'scale' key not found in {source_json}")
            except Exception as e:
                print(f"[Step 1] error processing {source_json}: {e}")

    print(f"[Step 1] done. {processed_files} files from {processed_dirs} directories.")
    if skipped_dirs:
        print(f"[Step 1] {len(skipped_dirs)} directories had no target files.")


# ---------------------------------------------------------------------------
# Step 2 – FPS sampling with normals
# ---------------------------------------------------------------------------


def export_ply_with_normals(path: str, points: np.ndarray, normals: np.ndarray):
    """Export point cloud with normals to a PLY file via open3d."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.normals = o3d.utility.Vector3dVector(normals.astype(np.float64))
    o3d.io.write_point_cloud(path, pcd)


def fps_worker(rank, world_size, object_dirs):
    """Worker process: FPS-sample point clouds from a subset of object directories."""
    device = torch.device(f"cuda:{rank}")
    print(f"[Step 2] rank {rank} starting on {device}")

    chunk_size = math.ceil(len(object_dirs) / world_size)
    subset = object_dirs[rank * chunk_size:(rank + 1) * chunk_size]
    if not subset:
        return

    num_samples_dexgraspnet = 2000
    num_samples_grasp_generation = 10000

    iterator = tqdm(subset, desc=f"[Step 2] rank {rank} FPS sampling") if rank == 0 else subset

    for object_dir in iterator:
        obj_path = os.path.join(object_dir, "decomposed.obj")
        if not os.path.exists(obj_path):
            if rank == 0:
                print(f"[Step 2] skipping {object_dir}: decomposed.obj not found.")
            continue

        try:
            obj_mesh = tm.load(obj_path, force="mesh", process=False)
            vertices = torch.tensor(obj_mesh.vertices, dtype=torch.float, device=device)
            faces = torch.tensor(obj_mesh.faces, dtype=torch.long, device=device)
            mesh = pytorch3d.structures.Meshes(vertices.unsqueeze(0), faces.unsqueeze(0))

            # Densely sample points + normals for subsequent FPS
            dense_pts, dense_nrm = pytorch3d.ops.sample_points_from_meshes(
                mesh,
                num_samples=100 * num_samples_dexgraspnet,
                return_normals=True,
            )
            # dense_pts / dense_nrm: (1, N, 3)

            # --- 2000-point cloud (DexGraspNet) ---
            fps_pts_dex, fps_idx_dex = pytorch3d.ops.sample_farthest_points(dense_pts, K=num_samples_dexgraspnet)
            export_ply_with_normals(
                os.path.join(object_dir, f"obj_points_{num_samples_dexgraspnet}.ply"),
                fps_pts_dex[0].cpu().numpy(),
                dense_nrm[0][fps_idx_dex[0]].cpu().numpy(),
            )

            # --- 10000-point cloud (grasp generation) ---
            fps_pts_grasp, fps_idx_grasp = pytorch3d.ops.sample_farthest_points(dense_pts, K=num_samples_grasp_generation)
            export_ply_with_normals(
                os.path.join(object_dir, f"obj_points_{num_samples_grasp_generation}.ply"),
                fps_pts_grasp[0].cpu().numpy(),
                dense_nrm[0][fps_idx_grasp[0]].cpu().numpy(),
            )

        except Exception as e:
            if rank == 0:
                print(f"[Step 2] error processing {object_dir}: {e}")


def fps_sampling(data_root: str, video_ids=None):
    """Discover object directories and launch multi-GPU FPS sampling workers."""
    if not video_ids:
        print("[Step 2] no video_id given, discovering all objects in data_root ...")
        object_dirs = []
        for source in ["sam3d", "dexgraspnet"]:
            source_path = os.path.join(data_root, source)
            if os.path.isdir(source_path):
                for name in os.listdir(source_path):
                    d = os.path.join(source_path, name)
                    if os.path.isdir(d):
                        object_dirs.append(d)
        print(f"[Step 2] found {len(object_dirs)} object directories.")
    else:
        object_dirs = []
        for vid in video_ids:
            found = False
            for source in ["sam3d", "dexgraspnet"]:
                d = os.path.join(data_root, source, vid)
                if os.path.isdir(d):
                    object_dirs.append(d)
                    found = True
                    break
            if not found:
                print(f"[Step 2] warning: video_id '{vid}' not found in data_root.")

    if not object_dirs:
        print("[Step 2] no object directories found, skipping FPS sampling.")
        return

    world_size = torch.cuda.device_count()
    if world_size > 1:
        print(f"[Step 2] found {world_size} GPUs, spawning workers ...")
        mp.spawn(fps_worker, args=(world_size, object_dirs), nprocs=world_size, join=True)
    else:
        print("[Step 2] running in single-process mode.")
        fps_worker(0, 1, object_dirs)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare sam3d meshdata and run FPS point-cloud sampling.")

    # Step 1 args
    parser.add_argument(
        "--source_dir",
        type=str,
        default=str(DEFAULT_SOURCE_DIR),
        help="Source root with <video_id>/HOI seq (default: CHOIR output/).",
    )
    parser.add_argument(
        "--video_id",
        nargs="+",
        metavar="VIDEO_ID",
        default=None,
        help="Video IDs to process in both steps (default: all).",
    )
    parser.add_argument(
        "--seq_dir_name",
        type=str,
        default="stage3/intermediates/optimized_hoi_seq",
        help="HOI sequence directory under each video_id to prepare meshes from.",
    )

    # Step 2 args
    parser.add_argument(
        "--data_root",
        type=str,
        default=str(DEFAULT_MESHDATA),
        help="Root directory for meshdata (default: stage2_grasp_correction/.../meshdata).",
    )

    # Skip flags
    parser.add_argument("--skip_prepare", action="store_true", help="Skip Step 1 (sam3d mesh preparation).")
    parser.add_argument("--skip_sampling", action="store_true", help="Skip Step 2 (FPS sampling).")

    args = parser.parse_args()

    if not args.skip_prepare:
        prepare_sam3d(
            source_base_dir=Path(args.source_dir),
            dest_base_dir=Path(args.data_root) / "sam3d",
            video_ids=args.video_id,
            seq_dir_name=args.seq_dir_name,
        )

    if not args.skip_sampling:
        fps_sampling(
            data_root=args.data_root,
            video_ids=args.video_id,
        )
