"""
Last modified date: 2023.02.23
Author: Jialiang Zhang, Ruicheng Wang
Description: Entry of the program, generate small-scale experiments
"""

import os

os.chdir(os.path.dirname(__file__))
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import argparse
import json
import math
import shutil

import numpy as np
import pytorch3d.ops
import pytorch3d.structures
import torch
import trimesh as tm
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle
from torchsdf import index_vertices_by_faces
from tqdm import tqdm
from utils.energy import cal_energy
from utils.hand_model import HandModel
from utils.initializations import initialize_table_top
from utils.logger import Logger
from utils.object_model import ObjectModel
from utils.optimizer import Annealing

# prepare arguments

parser = argparse.ArgumentParser()
# experiment settings
parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--gpu', default="0", type=str)
parser.add_argument('--object_code_list', default=['87141'], type=list, help="List of video_ids to process.")
parser.add_argument('--data_root', type=str, default='/mlp_vepfs/share/hpl/project/code/sam-3d-objects/input_data')
parser.add_argument('--name', default='exp_32', type=str)
parser.add_argument('--n_contact', default=4, type=int)
parser.add_argument('--batch_size', default=128, type=int)
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

parser.add_argument('--hand_init_mode',
                    default='random',
                    type=str,
                    choices=['custom', 'random'],
                    help='Hand initialization mode: "custom" for user-defined paths, "random" for random initialization.')

args = parser.parse_args()

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

np.seterr(all='raise')
np.random.seed(args.seed)
torch.manual_seed(args.seed)

# prepare models

total_batch_size = len(args.object_code_list) * args.batch_size

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('running on', device)

hand_model = HandModel(mano_root='mano', contact_indices_path='mano/contact_indices.json', pose_distrib_path='mano/pose_distrib.pt', device=device)

# ================== Object and Hand Initialization ==================
# 1. 在这里填入您的文件路径
# ----------------------------------------------------
user_obj_path = "/mlp_vepfs/share/hpl/project/code/sam-3d-objects/input_data/87141/optimized_hoi_seq/obj_canonical.obj"
user_obj_pose_path = "/mlp_vepfs/share/hpl/project/code/sam-3d-objects/input_data/87141/optimized_hoi_seq/obj_00037.json"
user_mano_path = "/mlp_vepfs/share/hpl/project/code/sam-3d-objects/input_data/87141/optimized_hoi_seq/mano_00037.json"
# ----------------------------------------------------

# 2. 从文件中加载物体数据
# ----------------------------------------------------
with open(user_obj_pose_path, 'r') as f:
    obj_pose_data = json.load(f)
user_object_scale = np.mean(obj_pose_data['scale'])
user_object_translation = np.array(obj_pose_data['translation'])
user_object_rotation_matrix = np.array(obj_pose_data['rotation'])
# ----------------------------------------------------

# 3. 根据手部初始化模式，设置不同的参数
# ----------------------------------------------------
if args.hand_init_mode == 'custom':
    # 自定义模式下，batch_size 为 1
    args.batch_size = 1
elif args.hand_init_mode == 'random':
    # 随机模式下，使用命令行传入的 batch_size
    pass

# 4. 初始化 ObjectModel
# ----------------------------------------------------
# 更新 args 以适应自定义物体加载
args.object_code_list = ['custom_object']

object_model = ObjectModel(data_root_path='../data/meshdata', batch_size_each=args.batch_size, num_samples=2000, device=device)

# ----------------------------------------------------

# 5. 根据选择的模式初始化手部
# ----------------------------------------------------
if args.hand_init_mode == 'custom':
    # 手动加载您的物体
    custom_mesh = tm.load(user_obj_path, force="mesh", process=False)
    custom_mesh.vertices = custom_mesh.vertices @ np.diag([-1, -1, 1])
    object_model.object_mesh_list = [custom_mesh]
    object_model.object_code_list = args.object_code_list

    # 设置物体的 scale
    object_model.object_scale_tensor = torch.tensor([[user_object_scale] * args.batch_size], dtype=torch.float, device=device)

    # 准备物体表面点云等后续计算所需的数据
    object_verts = custom_mesh.vertices
    object_verts = torch.Tensor(object_verts).to(device)
    object_faces = torch.Tensor(custom_mesh.faces).long().to(device)
    object_model.object_face_verts_list = [index_vertices_by_faces(object_verts, object_faces)]
    if object_model.num_samples != 0:
        mesh_pytorch3d = pytorch3d.structures.Meshes(object_verts.unsqueeze(0), object_faces.unsqueeze(0))
        dense_point_cloud = pytorch3d.ops.sample_points_from_meshes(mesh_pytorch3d, num_samples=100 * object_model.num_samples)
        surface_points = pytorch3d.ops.sample_farthest_points(dense_point_cloud, K=object_model.num_samples)[0][0]
        surface_points.to(dtype=float, device=device)
        object_model.surface_points_tensor = surface_points.unsqueeze(0).repeat_interleave(args.batch_size, dim=0)
    # 从文件加载 MANO 参数
    with open(user_mano_path, 'r') as f:
        mano_data = json.load(f)
    mano_trans = torch.tensor(mano_data['trans'], dtype=torch.float, device=device)
    mano_root_orient = torch.tensor(mano_data['root_orient'], dtype=torch.float, device=device)
    mano_pose = torch.tensor(mano_data['pose'], dtype=torch.float, device=device).reshape(45)
    is_right = mano_data['is_right'] > 0.
    user_mano_params = torch.cat([mano_trans, mano_root_orient, mano_pose], dim=0).unsqueeze(0)

    # 设置手部姿态
    hand_translation_world = user_mano_params[:, :3]
    hand_rotation_world_aa = user_mano_params[:, 3:6]

    diag_mat = np.diag([-1, -1, 1])
    new_translation = user_object_translation @ diag_mat
    new_rotation_matrix = diag_mat @ user_object_rotation_matrix @ diag_mat

    hand_translation_world = (hand_translation_world -
                              torch.from_numpy(new_translation).float().to(device).unsqueeze(0)) @ torch.from_numpy(new_rotation_matrix).float().to(device)
    hand_rotation_world_aa = matrix_to_axis_angle(torch.from_numpy(new_rotation_matrix).float().to(device).T @ axis_angle_to_matrix(hand_rotation_world_aa))

    final_mano_params = torch.cat([hand_translation_world, hand_rotation_world_aa, user_mano_params[:, 6:]], dim=1).requires_grad_(True)

    # 初始化手部模型
    contact_point_indices = torch.randint(hand_model.n_contact_candidates, (args.batch_size, args.n_contact), device=device)
    hand_model.set_parameters(final_mano_params, contact_point_indices)

elif args.hand_init_mode == 'random':
    # 手动加载您的物体
    custom_mesh = tm.load(user_obj_path, force="mesh", process=False)
    custom_mesh.vertices = custom_mesh.vertices @ np.diag([-1, -1, 1])
    custom_mesh.vertices[:, 2] = custom_mesh.vertices[:, 2] - np.min(custom_mesh.vertices[:, 2], keepdims=True)

    object_model.object_mesh_list = [custom_mesh]
    object_model.object_code_list = args.object_code_list

    # 设置物体的 scale
    object_model.object_scale_tensor = torch.tensor([[user_object_scale] * args.batch_size], dtype=torch.float, device=device)

    # 准备物体表面点云等后续计算所需的数据
    object_verts = custom_mesh.vertices
    object_verts = torch.Tensor(object_verts).to(device)
    object_faces = torch.Tensor(custom_mesh.faces).long().to(device)
    object_model.object_face_verts_list = [index_vertices_by_faces(object_verts, object_faces)]
    if object_model.num_samples != 0:
        mesh_pytorch3d = pytorch3d.structures.Meshes(object_verts.unsqueeze(0), object_faces.unsqueeze(0))
        dense_point_cloud = pytorch3d.ops.sample_points_from_meshes(mesh_pytorch3d, num_samples=100 * object_model.num_samples)
        surface_points = pytorch3d.ops.sample_farthest_points(dense_point_cloud, K=object_model.num_samples)[0][0]
        surface_points.to(dtype=float, device=device)
        object_model.surface_points_tensor = surface_points.unsqueeze(0).repeat_interleave(args.batch_size, dim=0)
    # 随机初始化手部模型
    initialize_table_top(hand_model, object_model, args)
# ----------------------------------------------------

# 6. 保存初始化状态的 Mesh (可选，用于调试)
# ----------------------------------------------------
# 创建保存目录
initial_state_path = os.path.join('../data/experiments', args.name, 'initial_state')
os.makedirs(initial_state_path, exist_ok=True)

# 获取手部模型 (只保存 batch 中的第一个)
hand_mesh = hand_model.get_trimesh_data()[0]

# 获取并变换物体模型
object_mesh_to_save = custom_mesh.copy()
object_mesh_to_save.apply_scale(user_object_scale)

# 将手和物体合并到一个场景中并导出为单个文件
scene_mesh = tm.util.concatenate(hand_mesh, object_mesh_to_save)
scene_path = os.path.join(initial_state_path, 'scene_initial.obj')
scene_mesh.export(scene_path)

print(f"Initial scene mesh saved to: {scene_path}")
# ----------------------------------------------------

total_batch_size = len(args.object_code_list) * args.batch_size
print('total batch size', total_batch_size)
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

try:
    shutil.rmtree(os.path.join('../data/experiments', args.name, 'logs'))
except FileNotFoundError:
    pass
os.makedirs(os.path.join('../data/experiments', args.name, 'logs'), exist_ok=True)
logger_config = {'thres_fc': args.thres_fc, 'thres_dis': args.thres_dis, 'thres_pen': args.thres_pen}
logger = Logger(log_dir=os.path.join('../data/experiments', args.name, 'logs'), **logger_config)

# log settings

with open(os.path.join('../data/experiments', args.name, 'output.txt'), 'w') as f:
    f.write(str(args) + '\n')

# optimize

weight_dict = dict(
    w_dis=args.w_dis,
    w_pen=args.w_pen,
    w_prior=args.w_prior,
    w_spen=args.w_spen,
    w_tpen=args.w_tpen,
)
energy, E_fc, E_dis, E_pen, E_prior, E_spen, E_tpen = cal_energy(hand_model, object_model, verbose=True, **weight_dict)

energy.sum().backward(retain_graph=True)
logger.log(energy, E_fc, E_dis, E_pen, E_prior, E_spen, E_tpen, 0, show=False)

for step in tqdm(range(1, args.n_iter + 1), desc='optimizing'):
    s = optimizer.try_step()

    optimizer.zero_grad()
    new_energy, new_E_fc, new_E_dis, new_E_pen, new_E_prior, new_E_spen, new_E_tpen = cal_energy(hand_model, object_model, verbose=True, **weight_dict)

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

        logger.log(energy, E_fc, E_dis, E_pen, E_prior, E_spen, E_tpen, step, show=False)

# save results
try:
    shutil.rmtree(os.path.join('../data/experiments', args.name, 'results'))
except FileNotFoundError:
    pass
os.makedirs(os.path.join('../data/experiments', args.name, 'results'), exist_ok=True)
result_path = os.path.join('../data/experiments', args.name, 'results')
os.makedirs(result_path, exist_ok=True)
for i in range(len(args.object_code_list)):
    data_list = []
    for j in range(args.batch_size):
        idx = i * args.batch_size + j
        scale = object_model.object_scale_tensor[i][j].item()
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
            ))
    np.save(os.path.join(result_path, args.object_code_list[i] + '.npy'), data_list, allow_pickle=True)

# ================== Save results as OBJ meshes ==================

print("\nSaving visualization meshes...")

# Hand model for visualization
vis_hand_model = HandModel(mano_root='mano', contact_indices_path='mano/contact_indices.json', pose_distrib_path='mano/pose_distrib.pt', device='cpu')

for i in range(len(args.object_code_list)):
    for j in range(args.batch_size):
        grasp_idx_global = i * args.batch_size + j

        # Get object mesh
        object_mesh_to_save = custom_mesh.copy() if 'custom_mesh' in locals() else object_model.object_mesh_list[i].copy()
        scale = object_model.object_scale_tensor[i][j].item()
        object_mesh_to_save.apply_scale(scale)
        object_mesh_to_save.visual.vertex_colors = [144, 238, 144, 255]  # Light green

        # Get initial and final hand meshes
        start_pose = hand_pose_st[grasp_idx_global].detach().cpu()
        end_pose = hand_model.hand_pose[grasp_idx_global].detach().cpu()
        hand_poses_for_vis = torch.stack([start_pose, end_pose])
        vis_hand_model.set_parameters(hand_poses_for_vis)
        hand_mesh_st, hand_mesh_en = vis_hand_model.get_trimesh_data()
        hand_mesh_st.visual.vertex_colors = [173, 216, 230, 150]  # Light blue with transparency
        hand_mesh_en.visual.vertex_colors = [100, 149, 237, 255]  # Cornflower blue

        # Combine meshes
        scene_mesh = tm.util.concatenate([object_mesh_to_save, hand_mesh_st, hand_mesh_en])

        # Define file path and save
        vis_filename = f"{args.object_code_list[i]}_{j:05d}.obj"
        vis_path = os.path.join(result_path, vis_filename)
        scene_mesh.export(vis_path)

    print(f"Saved {args.batch_size} visualization meshes for {args.object_code_list[i]} to {result_path}")
