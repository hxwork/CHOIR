"""
CHOIR Stage 2: generate table-top grasp training candidates.

Entry for grasp-data generation. Expects each object under
`--data_root/{sam3d,dexgraspnet}/<id>/` to contain at least:
  - decomposed.obj
  - init_obj_poses.npy

Writes grasp results to `<object_dir>/grasp_data/` and `grasp_data.npy`.
Optional downstream filter: validate_grasping_pose.py (PyBullet).
"""

import os

os.chdir(os.path.dirname(__file__))
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import argparse
import json
import math
import shutil
from pathlib import Path

import numpy as np
import PIL.Image
import pytorch3d.ops
import pytorch3d.structures
import torch
import torch.multiprocessing as mp
import trimesh as tm
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle
from tests.visualize_result import plane2pose
from torchsdf import index_vertices_by_faces
from tqdm import tqdm
from utils.energy import cal_energy_mine
from utils.hand_model import HandMineModel, HandModel
from utils.initializations import initialize_table_top_mine
from utils.logger import Logger
from utils.object_model import ObjectMineModel
from utils.optimizer import Annealing

MODULE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MESHDATA = str(MODULE_ROOT / "meshdata")
DEFAULT_EXP_ROOT = str(MODULE_ROOT / "data" / "experiments")


def main_worker(rank, world_size, args):
    """
    Main worker function for a single process.
    """
    # 1. Set up device and distribute video_·ids
    device = torch.device(f'cuda:{rank}')
    print(f"--> Starting worker on rank {rank} (GPU {device})")

    all_object_codes = args.object_code_list
    chunk_size = math.ceil(len(all_object_codes) / world_size)
    start_index = rank * chunk_size
    end_index = min((rank + 1) * chunk_size, len(all_object_codes))
    object_code_subset = all_object_codes[start_index:end_index]

    if not object_code_subset:
        print(f"Rank {rank} has no object codes to process. Exiting.")
        return

    args.object_code_list = object_code_subset
    print(f"Rank {rank} processing {len(args.object_code_list)} object codes: from index {start_index} to {end_index-1}")

    # Rank-specific experiment logs under the module (not an external absolute path).
    log_dir = os.path.join(args.exp_root, args.name, 'logs', f'rank_{rank}')

    os.makedirs(log_dir, exist_ok=True)

    # 3. Core processing logic
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    np.seterr(all='raise')
    # Use a different seed for each process
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)

    total_batch_size = len(args.object_code_list) * args.batch_size

    hand_model = HandMineModel(mano_root='mano', contact_indices_path='mano/contact_indices.json', pose_distrib_path='mano/pose_distrib.pt', device=device)
    object_model = ObjectMineModel(data_root_path=args.data_root, batch_size_each=args.batch_size, num_samples=2000, device=device)
    object_model.initialize(args.object_code_list)

    initialize_table_top_mine(hand_model, object_model, args)

    print(f'[Rank {rank}] Total batch size: {total_batch_size}')
    hand_pose_st = hand_model.hand_pose.detach()

    optim_config = {
        'switch_possibility': args.switch_possibility,
        'starting_temperature': args.starting_temperature,
        'temperature_decay': args.temperature_decay,
        'annealing_period': args.annealing_period,
        'step_size': args.step_size,
        'stepsize_period': args.stepsize_period,
        'mu': args.mu,
        'device': device
    }
    optimizer = Annealing(hand_model, **optim_config)

    logger_config = {'thres_fc': args.thres_fc, 'thres_dis': args.thres_dis, 'thres_pen': args.thres_pen}
    logger = Logger(log_dir=log_dir, **logger_config)

    weight_dict = dict(
        w_dis=args.w_dis,
        w_pen=args.w_pen,
        w_prior=args.w_prior,
        w_spen=args.w_spen,
        w_tpen=args.w_tpen,
        w_anatomy=args.w_anatomy,
    )
    energy, E_fc, E_dis, E_pen, E_prior, E_spen, E_tpen, E_anatomy = cal_energy_mine(hand_model, object_model, verbose=True, **weight_dict)

    energy.sum().backward(retain_graph=True)
    logger.log(energy, E_fc, E_dis, E_pen, E_prior, E_spen, E_tpen, E_anatomy, 0, show=False)

    # Use tqdm only for rank 0 to avoid messy output
    if rank == 0:
        pbar = tqdm(total=args.n_iter, desc='optimizing')

    for step in range(1, args.n_iter + 1):
        s = optimizer.try_step()
        optimizer.zero_grad()
        new_energy, new_E_fc, new_E_dis, new_E_pen, new_E_prior, new_E_spen, new_E_tpen, new_E_anatomy = cal_energy_mine(hand_model,
                                                                                                                         object_model,
                                                                                                                         verbose=True,
                                                                                                                         **weight_dict)
        new_energy.sum().backward(retain_graph=True)

        with torch.no_grad():
            accept, t = optimizer.accept_step(energy, new_energy)
            energy[accept] = new_energy[accept]
            E_dis[accept] = new_E_dis[accept]
            E_fc[accept] = new_E_fc[accept]
            E_pen[accept] = new_E_pen[accept]
            E_prior[accept] = new_E_prior[accept]
            E_spen[accept] = new_E_spen[accept]
            E_tpen[accept] = new_E_tpen[accept]
            E_anatomy[accept] = new_E_anatomy[accept]
            logger.log(energy, E_fc, E_dis, E_pen, E_prior, E_spen, E_tpen, E_anatomy, step, show=False)

        if rank == 0:
            pbar.update(1)

    if rank == 0:
        pbar.close()

    # Save results
    for i in range(len(args.object_code_list)):
        object_dir = args.object_code_list[i]
        object_code = os.path.basename(object_dir)
        result_path = os.path.join(object_dir, 'grasp_data')
        os.makedirs(result_path, exist_ok=True)
        data_list = []
        for j in range(args.batch_size):
            idx = i * args.batch_size + j
            scale = object_model.object_scale_tensor[i][j].cpu().numpy().tolist()
            hand_pose = hand_model.hand_pose[idx].detach().cpu()
            qpos = dict(
                trans=hand_pose[:3].tolist(),
                rot=hand_pose[3:6].tolist(),
                thetas=hand_pose[6:].tolist(),
            )
            hand_pose = hand_pose_st[idx].detach().cpu()
            qpos_st = dict(
                trans=hand_pose[:3].tolist(),
                rot=hand_pose[3:6].tolist(),
                thetas=hand_pose[6:].tolist(),
            )
            data_list.append(
                dict(
                    scale=scale,
                    plane=object_model.plane_parameters[idx].tolist(),
                    qpos=qpos,
                    contact_point_indices=hand_model.contact_point_indices[idx].detach().cpu().tolist(),
                    qpos_st=qpos_st,
                    energy=energy[idx].item(),
                    E_fc=E_fc[idx].item(),
                    E_dis=E_dis[idx].item(),
                    E_pen=E_pen[idx].item(),
                    E_prior=E_prior[idx].item(),
                    E_spen=E_spen[idx].item(),
                    E_tpen=E_tpen[idx].item(),
                    E_anatomy=E_anatomy[idx].item(),
                ))

            # Get object mesh
            object_mesh_to_save = object_model.object_mesh_list[i].copy()
            scale = object_model.object_scale_tensor[i][j].cpu().numpy().tolist()
            object_mesh_to_save.apply_scale(scale)

            # Apply object pose
            pose = plane2pose(object_model.plane_parameters[idx]).cpu().numpy()
            object_mesh_to_save.apply_transform(pose)
            object_mesh_to_save.visual.vertex_colors = np.array([144, 238, 144, 255], dtype=np.uint8)

            vis_hand_model = HandMineModel(mano_root='mano',
                                           contact_indices_path='mano/contact_indices.json',
                                           pose_distrib_path='mano/pose_distrib.pt',
                                           device='cpu')

            # Get initial hand mesh
            hand_pose_st_individual = hand_pose_st[idx].detach().cpu()
            vis_hand_model.set_parameters(hand_pose_st_individual.unsqueeze(0))
            hand_mesh_st = vis_hand_model.get_trimesh_data(i=0)
            hand_mesh_st.apply_transform(pose)
            hand_mesh_st.visual.vertex_colors = np.array([173, 216, 230, 150], dtype=np.uint8)

            # Get final hand mesh
            hand_pose_final = hand_model.hand_pose[idx].detach().cpu()
            contact_point_indices = hand_model.contact_point_indices[idx].detach().cpu()
            vis_hand_model.set_parameters(hand_pose_final.unsqueeze(0), contact_point_indices.unsqueeze(0))
            hand_mesh_en = vis_hand_model.get_trimesh_data(i=0)
            hand_mesh_en.apply_transform(pose)
            hand_mesh_en.visual.vertex_colors = np.array([100, 149, 237, 255], dtype=np.uint8)

            combined_mesh = tm.util.concatenate([object_mesh_to_save, hand_mesh_st, hand_mesh_en])

            vis_filename = f"{j:05d}.obj"
            vis_path = os.path.join(result_path, vis_filename)

            hand_mesh_en.export(os.path.join(result_path, f"hand_mesh_{j:05d}.obj"), file_type='obj')
            obj_data = {
                'scale': scale,
                'pose': pose.tolist(),
            }
            with open(os.path.join(result_path, f"obj_data_{j:05d}.json"), 'w') as f:
                json.dump(obj_data, f, indent=4)

            combined_mesh.export(vis_path, file_type='obj')

            # Save parameters to JSON
            grasp_data = {
                "initial_mano_params": {
                    "root_orient": qpos_st['rot'],
                    "trans": qpos_st['trans'],
                    "pose": qpos_st['thetas']
                },
                "final_mano_params": {
                    "root_orient": qpos['rot'],
                    "trans": qpos['trans'],
                    "pose": qpos['thetas']
                },
                "object_scale": scale,
                "plane_pose": {
                    "rotation": pose[:3, :3].tolist(),
                    "translation": pose[:3, 3].tolist()
                },
                "contact_point_indices": contact_point_indices.tolist(),
            }
            json_filename = f"{j:05d}.json"
            json_path = os.path.join(result_path, json_filename)
            with open(json_path, 'w') as f:
                json.dump(grasp_data, f, indent=4)

        np.save(os.path.join(object_dir, 'grasp_data.npy'), data_list, allow_pickle=True)
        print(f"[Rank {rank}] Saved {args.batch_size} visualization meshes for {object_code} to {result_path}")


if __name__ == '__main__':
    # prepare arguments
    parser = argparse.ArgumentParser()
    # experiment settings
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--object_code_list', nargs='+', default=[], help="List of object codes to process.")
    parser.add_argument(
        '--data_root',
        type=str,
        default=DEFAULT_MESHDATA,
        help="Root directory for mesh data (default: stage2_grasp_correction/DexGraspNet_table/meshdata).",
    )
    parser.add_argument(
        '--exp_root',
        type=str,
        default=DEFAULT_EXP_ROOT,
        help="Experiment/log root (default: stage2_grasp_correction/DexGraspNet_table/data/experiments).",
    )
    parser.add_argument('--name', default='mine', type=str)
    parser.add_argument('--n_contact', default=4, type=int)
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--n_iter', default=6000, type=int)
    # hyper parameters (** Magic, don't touch! **)
    parser.add_argument('--switch_possibility', default=0.5, type=float)
    parser.add_argument('--mu', default=0.98, type=float)
    parser.add_argument('--step_size', default=0.005, type=float)
    parser.add_argument('--stepsize_period', default=50, type=int)
    parser.add_argument('--starting_temperature', default=18, type=float)
    parser.add_argument('--annealing_period', default=30, type=int)
    parser.add_argument('--temperature_decay', default=0.95, type=float)
    parser.add_argument('--w_dis', default=100.0, type=float)
    parser.add_argument('--w_pen', default=100.0, type=float)
    parser.add_argument('--w_prior', default=0.5, type=float)
    parser.add_argument('--w_spen', default=10.0, type=float)
    parser.add_argument('--w_tpen', default=40.0, type=float)
    # parser.add_argument('--w_tpen', default=0.0, type=float)
    parser.add_argument('--w_anatomy', default=1.0, type=float)
    # initialization settings
    parser.add_argument('--jitter_strength', default=0., type=float)
    parser.add_argument('--distance_lower', default=0.1, type=float)
    parser.add_argument('--distance_upper', default=0.1, type=float)
    parser.add_argument('--theta_lower', default=0, type=float)
    parser.add_argument('--theta_upper', default=0, type=float)
    parser.add_argument('--angle_upper', default=math.pi / 2, type=float)
    # energy thresholds
    parser.add_argument('--thres_fc', default=0.3, type=float)
    parser.add_argument('--thres_dis', default=0.005, type=float)
    parser.add_argument('--thres_pen', default=0.001, type=float)
    # distributed settings
    parser.add_argument('--num_parts', default=1, type=int, help="Number of parts to split the dataset into.")
    parser.add_argument('--part_id', default=0, type=int, help="ID of the part to process (0-indexed).")
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

    print("Checking for data completeness...")
    complete_object_dirs = []
    for obj_dir in args.object_code_list:
        mesh_path = os.path.join(obj_dir, "decomposed.obj")
        poses_path = os.path.join(obj_dir, "init_obj_poses.npy")

        # scale.json is optional, so we don't check for its existence here for completeness
        if os.path.exists(mesh_path) and os.path.exists(poses_path):
            complete_object_dirs.append(obj_dir)
        else:
            print(f"Skipping incomplete object directory: {obj_dir}")
            if not os.path.exists(mesh_path):
                print(f"  - Missing: {mesh_path}")
            if not os.path.exists(poses_path):
                print(f"  - Missing: {poses_path}")

    original_count = len(args.object_code_list)
    args.object_code_list = complete_object_dirs
    print(f"Filtered object directories. Kept {len(args.object_code_list)} out of {original_count}.")

    # Handle dataset splitting for distributed processing
    if args.num_parts > 1:
        print(f"Splitting dataset into {args.num_parts} parts. Processing part {args.part_id}.")
        total_objects = len(args.object_code_list)
        part_size = math.ceil(total_objects / args.num_parts)
        start_index = args.part_id * part_size
        end_index = min((args.part_id + 1) * part_size, total_objects)

        if start_index >= total_objects:
            print(f"Part ID {args.part_id} is out of range. Nothing to process.")
            args.object_code_list = []
        else:
            args.object_code_list = args.object_code_list[start_index:end_index]
            print(f"Processing {len(args.object_code_list)} objects in this part (indices {start_index} to {end_index - 1}).")

    # Spawn worker processes
    world_size = torch.cuda.device_count()
    if world_size > 1:
        print(f"Found {world_size} GPUs. Spawning worker processes.")
        mp.spawn(main_worker, args=(world_size, args), nprocs=world_size, join=True)
    else:
        print("Found 1 or 0 GPUs. Running in single-process mode.")
        main_worker(0, 1, args)
