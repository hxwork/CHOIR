# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
Samples a large number of grasps from a pre-trained GraspSiT model using DDP.
"""
import argparse
import json
import os
import sys
from time import time

import numpy as np
import torch
import torch.distributed as dist
import trimesh
from pytorch3d.transforms import matrix_to_axis_angle, rotation_6d_to_matrix
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from data_layout import DEFAULT_MANO_ASSETS, DEFAULT_SOURCE_DIR
from data_loader.GraspPair import GraspPair, GraspTest
from gfm_dataset_checks import assert_nonempty_dataset
from manotorch.manolayer import ManoLayer
from model.model import GraspDepthMagSiT
from train_utils import parse_ode_args, parse_sde_args, parse_transport_args
from transport import Sampler, create_transport


def points_to_spheres(points, radius=0.002, subdivision=1):
    """
    Convert a point cloud to a union of small sphere meshes (vectorized, no Python loop).
    """
    sphere = trimesh.creation.icosphere(subdivisions=subdivision, radius=radius)
    v_template = sphere.vertices
    f_template = sphere.faces

    n_points = len(points)
    n_v = len(v_template)

    new_vertices = (points[:, np.newaxis, :] + v_template[np.newaxis, :, :]).reshape(-1, 3)
    offsets = np.arange(n_points) * n_v
    new_faces = (f_template[np.newaxis, :, :] + offsets[:, np.newaxis, np.newaxis]).reshape(-1, 3)
    mesh = trimesh.Trimesh(vertices=new_vertices, faces=new_faces)

    return mesh


def decode_mano_params(hand_tensor, depth_mag=None, camera_ray=None):
    """
    Decode a (17, 6) hand representation into MANO params (translation, global rotation, pose).
    """
    if not isinstance(hand_tensor, torch.Tensor):
        hand_tensor = torch.from_numpy(hand_tensor)

    trans = hand_tensor[0, :3].unsqueeze(0)  # (1, 3)
    if depth_mag is not None and camera_ray is not None:
        trans = trans + depth_mag * camera_ray
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


def get_hand_mesh(trans, pose_coeffs, mano_layer):
    """
    Generate a hand mesh with ManoLayer.
    """
    with torch.no_grad():
        mano_output = mano_layer(pose_coeffs)
    verts = mano_output.verts + trans
    verts = verts[0].cpu().numpy()
    faces = mano_layer.th_faces.cpu().numpy()
    return verts, faces


def grasp_collate_fn(batch):
    """
    Custom collate_fn for variable-length mesh data.
    """
    variable_keys = ['cond_obj_verts', 'cond_obj_faces']
    variable_data = {key: [d.pop(key) for d in batch] for key in variable_keys}
    batch = default_collate(batch)
    batch.update(variable_data)
    return batch


def create_ray_mesh(vec, radius=0.001, color=[255, 255, 0, 255], length_scale=5.0):
    """
    Create a cylinder mesh along vector vec.
    """
    length = np.linalg.norm(vec)

    # 1. Create a canonical cylinder along Z, centered at the origin
    # sections=8 is enough; no need for high tessellation
    mesh = trimesh.creation.cylinder(radius=radius, height=length * length_scale, sections=8)

    # 2. Rotate so Z aligns with vec
    # trimesh.geometry.align_vectors returns a 4x4 transform
    direction = vec / length
    # Canonical Z axis is [0, 0, 1]
    rot_matrix = trimesh.geometry.align_vectors([0, 0, 1], direction)
    mesh.apply_transform(rot_matrix)

    # 3. Set color
    mesh.visual.vertex_colors = color

    return mesh


def main(mode, args):
    """
    Run sampling.
    """
    # =========================================================================
    # [Step 1] Must run first: pin the device before any CUDA ops
    # =========================================================================

    # 1. Local rank
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # 2. Immediately bind this process to its GPU
    # Must happen before touching torch.backends.cuda / torch.cuda.is_available
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # =========================================================================
    # [Step 2] Initialize DDP
    # =========================================================================
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # =========================================================================
    # [Step 3] Safe to configure CUDA now
    # =========================================================================
    # Query hardware props on the device set above, not default GPU 0
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    assert torch.cuda.is_available(), "Sampling with DDP requires at least one GPU."
    torch.set_grad_enabled(False)

    seed = args.global_seed * world_size + rank
    # torch.manual_seed(seed)

    print(f"[Rank {rank}] Initialized. Local Rank: {local_rank}, Device: {device}")
    # -------------------------------------------------------------------------
    # Load model:
    assert args.ckpt is not None, "A checkpoint path must be provided."
    model = GraspDepthMagSiT().to(device)
    state_dict = torch.load(args.ckpt, map_location="cpu", weights_only=False)

    if "ema" in state_dict and not args.no_ema:
        if rank == 0:
            print("Loading EMA weights from checkpoint.")
        state_dict = state_dict["ema"]
    else:
        if rank == 0:
            print("Loading original weights from checkpoint.")
        state_dict = state_dict["model"]

    if next(iter(state_dict)).startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model = DDP(model, device_ids=[device])
    model.eval()

    # Setup MANO layer
    mano_layer = ManoLayer(
        mano_assets_root=str(DEFAULT_MANO_ASSETS),
        side='right',
        use_pca=False,
        flat_hand_mean=True,
    ).to(device)

    # Setup transport and sampler
    transport = create_transport(args.path_type, args.prediction, args.loss_weight, args.train_eps, args.sample_eps)
    sampler = Sampler(transport)
    if mode == "ODE":
        sample_fn = sampler.sample_ode(sampling_method=args.sampling_method,
                                       num_steps=args.num_sampling_steps,
                                       atol=args.atol,
                                       rtol=args.rtol,
                                       reverse=args.reverse)
    elif mode == "SDE":
        sample_fn = sampler.sample_sde(
            sampling_method=args.sampling_method,
            diffusion_form=args.diffusion_form,
            diffusion_norm=args.diffusion_norm,
            last_step=args.last_step,
            last_step_size=args.last_step_size,
            num_steps=args.num_sampling_steps,
        )

    # Setup data loader
    if args.data_split == 'test':
        video_id = [v for v in (args.video_id or []) if v]
        video_id = video_id if video_id else None
        dataset = GraspTest(
            debug=False,
            video_id=video_id,
            seq_dir_name=args.seq_dir_name,
            source_dir=args.source_dir,
        )
        assert_nonempty_dataset(len(dataset), video_id, args.source_dir)
    else:
        dataset = GraspPair(data_split=args.data_split, debug=False)

    dist_sampler = DistributedSampler(dataset, num_replicas=dist.get_world_size(), rank=rank, shuffle=False)
    loader = DataLoader(dataset,
                        batch_size=args.per_proc_batch_size,
                        shuffle=False,
                        sampler=dist_sampler,
                        num_workers=24,
                        collate_fn=grasp_collate_fn if args.data_split == 'test' else default_collate,
                        pin_memory=True,
                        drop_last=False)

    # Create output directory:
    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Saving samples to {args.output_dir}")
    dist.barrier()

    pbar = loader
    if rank == 0:
        pbar = tqdm(pbar)

    # Per-rank local records: {video_id: {frame_id(str): depth_offset(float)}}
    local_depth_offsets = {}

    for i, input_dict in enumerate(pbar):
        gt_hand = input_dict['noise_depth_mag']
        cond_x = input_dict['cond_x']
        cond_hand = input_dict['cond_hand']
        cond_obj_scale = input_dict['cond_obj_scale']
        cond_obj_pointcloud = input_dict['cond_obj_pointcloud']
        cond_camera_ray = input_dict['cond_camera_ray']
        video_ids = input_dict['video_id']
        frame_ids = input_dict['frame_id']

        # These are lists of tensors/arrays due to collate_fn
        cond_obj_verts = input_dict.get('cond_obj_verts', [])
        cond_obj_faces = input_dict.get('cond_obj_faces', [])

        current_batch_size = cond_hand.shape[0]

        # Prepare data
        gt_hand_available = len(gt_hand) > 0
        if gt_hand_available:
            gt_hand = gt_hand.to(device)
        cond_x = cond_x.to(device)
        cond_hand = cond_hand.to(device)
        cond_obj_scale = cond_obj_scale.to(device)
        cond_obj_pointcloud = cond_obj_pointcloud.to(device)
        cond_camera_ray = cond_camera_ray.to(device)

        with torch.no_grad():
            cond_obj_pointcloud_emb = model.module.encode_condition_features(cond_obj_pointcloud)

        # Setup classifier-free guidance:
        use_cfg = args.cfg_scale > 1.0
        if use_cfg:
            model_kwargs = dict(cond_x=cond_hand,
                                cond_obj_scale=cond_obj_scale,
                                cond_obj_pointcloud_emb=cond_obj_pointcloud_emb,
                                cond_camera_ray=cond_camera_ray,
                                cfg_scale=args.cfg_scale)
            model_fn = model.module.forward_inference_with_cfg
        else:
            model_kwargs = dict(cond_x=cond_hand,
                                cond_obj_scale=cond_obj_scale,
                                cond_obj_pointcloud_emb=cond_obj_pointcloud_emb,
                                cond_camera_ray=cond_camera_ray)
            model_fn = model.module.forward_inference

        # Perform N inference iterations and take median
        all_samples = []
        for inference_iter in range(args.num_inference):
            z = torch.randn(current_batch_size, 1, 1, device=device)
            samples_iter = sample_fn(z, model_fn, **model_kwargs)[-1]
            all_samples.append(samples_iter)

        # Compute median across N inference iterations
        if args.num_inference > 1:
            all_samples = torch.stack(all_samples, dim=0)  # (N, batch_size, 1, 1)
            samples = torch.median(all_samples, dim=0).values  # (batch_size, 1, 1)
        else:
            samples = all_samples[0]

        # Each process saves its own samples locally and in parallel
        for j in range(current_batch_size):
            pred_hand_j = samples[j].cpu()
            cond_hand_j = cond_x[j].cpu()
            obj_pc_j = cond_obj_pointcloud[j].cpu()
            obj_scale_j = cond_obj_scale[j].cpu()
            camera_ray_j = cond_camera_ray[j].cpu()

            cond_trans, cond_pose_coeffs = decode_mano_params(cond_hand_j)
            cond_trans = cond_trans * obj_scale_j
            cond_hand_verts, cond_faces = get_hand_mesh(cond_trans.to(device), cond_pose_coeffs.to(device), mano_layer)
            cond_mesh = trimesh.Trimesh(vertices=cond_hand_verts, faces=cond_faces)
            cond_mesh.visual.vertex_colors = [0, 0, 255, 200]

            # --- Process hands ---
            pred_trans, pred_pose_coeffs = decode_mano_params(cond_hand_j, pred_hand_j, camera_ray_j)
            pred_trans = pred_trans * obj_scale_j
            pred_hand_verts, pred_faces = get_hand_mesh(pred_trans.to(device), pred_pose_coeffs.to(device), mano_layer)
            pred_mesh = trimesh.Trimesh(vertices=pred_hand_verts, faces=pred_faces)
            pred_mesh.visual.vertex_colors = [255, 0, 0, 200]

            camera_ray_mesh = create_ray_mesh(camera_ray_j, radius=0.005, color=[255, 255, 0, 200])

            scene = trimesh.Scene()
            scene.add_geometry(pred_mesh)
            scene.add_geometry(cond_mesh)
            scene.add_geometry(camera_ray_mesh)

            if gt_hand_available:
                gt_hand_j = gt_hand[j].cpu()
                gt_trans, gt_pose_coeffs = decode_mano_params(gt_hand_j)
                gt_trans = gt_trans * obj_scale_j
                gt_hand_verts, gt_faces = get_hand_mesh(gt_trans.to(device), gt_pose_coeffs.to(device), mano_layer)
                gt_mesh = trimesh.Trimesh(vertices=gt_hand_verts, faces=gt_faces)
                gt_mesh.visual.vertex_colors = [0, 255, 0, 200]
                scene.add_geometry(gt_mesh)

            # --- Process object ---
            if args.data_split == 'test':
                obj_verts_j = cond_obj_verts[j]
                obj_faces_j = cond_obj_faces[j]
                obj_mesh_j = trimesh.Trimesh(vertices=obj_verts_j, faces=obj_faces_j)
                obj_mesh_j.vertices = obj_mesh_j.vertices * obj_scale_j.numpy()
                obj_mesh_j.visual.vertex_colors = [128, 128, 128, 255]
                scene.add_geometry(obj_mesh_j)
            else:
                obj_pc_rescaled = obj_pc_j.numpy() * obj_scale_j.numpy()
                if obj_pc_rescaled.shape[0] > 0:
                    if obj_pc_rescaled.shape[0] > 2048:
                        indices = np.random.choice(obj_pc_rescaled.shape[0], 2048, replace=False)
                        obj_pc_rescaled = obj_pc_rescaled[indices]
                    object_pc_mesh = points_to_spheres(obj_pc_rescaled)
                    object_pc_mesh.visual.vertex_colors = [128, 128, 128, 255]
                    scene.add_geometry(object_pc_mesh)

            # --- Combine and save ---
            video_id = video_ids[j]
            frame_id = frame_ids[j].item()
            frame_id_str = str(frame_id)
            # pred_hand_j is depth mag in normalized units; multiply by obj_scale for world translation
            depth_offset = float((pred_hand_j.reshape(-1)[0] * obj_scale_j.reshape(-1)[0]).item())
            video_id_str = str(video_id)
            if video_id_str not in local_depth_offsets:
                local_depth_offsets[video_id_str] = {}
            local_depth_offsets[video_id_str][frame_id_str] = depth_offset
            output_filename = f"{frame_id:05d}.ply"
            output_path = os.path.join(args.output_dir, str(video_id), output_filename)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            scene.export(output_path)

            # --- Save separately ---
            output_path = os.path.join(f"{args.output_dir}_contact_map", str(video_id), f"hand_{output_filename}")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            pred_mesh.export(output_path)

            output_path = os.path.join(f"{args.output_dir}_contact_map", str(video_id), f"obj_{output_filename}")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            obj_mesh_j.export(output_path)

            # --- Save separate files for rendering ---
            render_dir = os.path.join(f"{args.output_dir}_render", str(video_id), f"{frame_id:05d}")
            os.makedirs(render_dir, exist_ok=True)

            # Frame transform: [X-fwd, Z-up] -> [Z-fwd, -Y-up] (Blender-style)
            # Source frame: X=forward, Y=right, Z=up
            # Target frame: X=left, Y=up, Z=back (then Blender Z-fwd / -Y-up)
            # Includes 180-deg about X: new_X=-old_Y, new_Y=old_Z, new_Z=-old_X
            transform_matrix = np.array([[0, -1, 0, 0], [0, 0, 1, 0], [-1, 0, 0, 0], [0, 0, 0, 1]])

            # 1. Save predicted hand mesh
            pred_mesh_copy = pred_mesh.copy()
            pred_mesh_copy.apply_transform(transform_matrix)
            pred_mesh_copy.export(os.path.join(render_dir, "pred_hand_mesh.ply"))

            # 2. Save condition hand mesh (source)
            cond_mesh_copy = cond_mesh.copy()
            cond_mesh_copy.apply_transform(transform_matrix)
            cond_mesh_copy.export(os.path.join(render_dir, "source_hand_mesh.ply"))

            # 3. Save GT hand mesh (target) if available
            if gt_hand_available:
                gt_mesh_copy = gt_mesh.copy()
                gt_mesh_copy.apply_transform(transform_matrix)
                gt_mesh_copy.export(os.path.join(render_dir, "target_hand_mesh.ply"))

            # 4. Save camera ray
            camera_ray_mesh_copy = camera_ray_mesh.copy()
            camera_ray_mesh_copy.apply_transform(transform_matrix)
            camera_ray_mesh_copy.export(os.path.join(render_dir, "camera_ray.ply"))

            # 5. Save object mesh and point cloud
            obj_mesh_j_copy = obj_mesh_j.copy()
            obj_mesh_j_copy.apply_transform(transform_matrix)
            obj_mesh_j_copy.export(os.path.join(render_dir, "object_mesh.ply"))

            obj_pc_rescaled_render = obj_pc_j.numpy() * obj_scale_j.numpy()
            if obj_pc_rescaled_render.shape[0] > 0:
                # Keep XYZ only
                if obj_pc_rescaled_render.shape[1] > 3:
                    obj_pc_rescaled_render = obj_pc_rescaled_render[:, :3]
                if obj_pc_rescaled_render.shape[0] > 4096:
                    indices = np.random.choice(obj_pc_rescaled_render.shape[0], 4096, replace=False)
                    obj_pc_rescaled_render = obj_pc_rescaled_render[indices]
                # Apply frame transform to the point cloud
                obj_pc_transformed = obj_pc_rescaled_render @ transform_matrix[:3, :3].T
                # Save raw points; do not convert to sphere meshes
                object_pc = trimesh.PointCloud(vertices=obj_pc_transformed)
                object_pc.export(os.path.join(render_dir, "object_point_cloud.ply"))

        # Wait for all processes to finish saving their files before starting the next batch
        dist.barrier()

    dist.barrier()
    gathered_depth_offsets = [None for _ in range(world_size)] if rank == 0 else None
    dist.gather_object(local_depth_offsets, gathered_depth_offsets, dst=0)

    if rank == 0:
        merged_depth_offsets = {}
        for rank_offsets in gathered_depth_offsets:
            if rank_offsets is None:
                continue
            for video_id, frame_offset_map in rank_offsets.items():
                if video_id not in merged_depth_offsets:
                    merged_depth_offsets[video_id] = {}
                merged_depth_offsets[video_id].update(frame_offset_map)

        grasp_correction_root = args.source_dir or str(DEFAULT_SOURCE_DIR)
        for video_id, frame_offset_map in merged_depth_offsets.items():
            sorted_frame_offset_map = dict(sorted(frame_offset_map.items(), key=lambda kv: int(kv[0])))
            output_json_path = os.path.join(
                grasp_correction_root,
                video_id,
                "grasp_correction",
                "camera_ray_depth_offset.json",
            )
            os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
            with open(output_json_path, "w", encoding="utf-8") as f:
                json.dump(sorted_frame_offset_map, f, ensure_ascii=False, indent=2)

    if rank == 0:
        print("Done.")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    if len(sys.argv) < 2:
        print("Usage: program.py <mode> [options]")
        sys.exit(1)

    mode = sys.argv[1]

    assert mode[:2] != "--", "Usage: program.py <mode> [options]"
    assert mode in ["ODE", "SDE"], "Invalid mode. Please choose 'ODE' or 'SDE'"

    parser.add_argument("--ckpt", type=str, required=True, help="Path to a GraspSiT checkpoint.")
    parser.add_argument("--output_dir", type=str, default="samples_ddp", help="Directory to save the generated samples.")
    parser.add_argument("--per_proc_batch_size", type=int, default=4)
    parser.add_argument("--cfg_scale", type=float, default=4.0)
    parser.add_argument("--no_ema", action="store_true", help="Do not use the EMA model.")
    parser.add_argument("--num_sampling_steps", type=int, default=50)
    parser.add_argument("--data_split", type=str, default="test", choices=['train', 'val', 'test'], help="Data split to use for sampling.")
    parser.add_argument("--video_id",
                        type=str,
                        nargs='+',
                        default=None,
                        help="Optional video_id list for data_split=test. Omit to use GraspTest defaults.")
    parser.add_argument(
        "--seq_dir_name",
        type=str,
        default=None,
        help="Optional HOI sequence subdir under each video_id for the test split.",
    )
    parser.add_argument(
        "--source_dir",
        type=str,
        default=None,
        help="Test input root (default: CHOIR output/).",
    )
    parser.add_argument("--global_seed", type=int, default=0)
    parser.add_argument("--num_inference", type=int, default=1, help="Number of inference iterations per sample to compute median.")
    parser.add_argument("--tf32",
                        action=argparse.BooleanOptionalAction,
                        default=True,
                        help="By default, use TF32 matmuls. This massively accelerates sampling on Ampere GPUs.")

    parse_transport_args(parser)
    if mode == "ODE":
        parse_ode_args(parser)
    elif mode == "SDE":
        parse_sde_args(parser)

    args = parser.parse_known_args()[0]
    main(mode, args)
