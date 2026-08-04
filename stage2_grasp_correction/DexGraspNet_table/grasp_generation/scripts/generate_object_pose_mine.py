"""
Generate init_obj_poses.npy for each object under meshdata/{sam3d,dexgraspnet}/<id>/.

Required before running grasp_generation/main_prep_data.py when poses are missing.
Default --data_root_path is stage2_grasp_correction/DexGraspNet_table/meshdata.
"""
import argparse
import glob
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import sapien.core as sapien
import sapien.physx as physx
import transforms3d
import trimesh
from tqdm import tqdm

DEFAULT_MESHDATA = str(Path(__file__).resolve().parents[2] / "meshdata")


def align_pose_to_ground(mesh, pose_matrix, margin=0.002):
    """
    计算精准的贴地 Pose。
    原理：应用旋转后，找到 Mesh 最低点，通过平移让最低点位于 z=margin 处。
    """
    # 1. 提取旋转部分
    rotation_matrix = pose_matrix[:3, :3]

    # 2. 将旋转应用到 Mesh 的顶点上 (临时计算，不修改原 Mesh)
    # 这一步极其快速，只是矩阵乘法
    rotated_vertices = mesh.vertices.dot(rotation_matrix.T)

    # 3. 找到最低的 Z 值
    min_z = np.min(rotated_vertices[:, 2])

    # 4. 计算需要的平移量
    # 我们希望新的 min_z 变成 0 + margin
    # 所以 translation_z = margin - current_min_z
    target_z_translation = -min_z + margin

    # 5. 构建新的平移向量
    # X, Y 保持原来的（或者随机），Z 使用计算出的值
    translation = np.zeros(3)
    translation[:2] = pose_matrix[:3, 3][:2]
    translation[2] = target_z_translation

    # 6. 转换为 Sapien Pose
    quat = transforms3d.quaternions.mat2quat(rotation_matrix)
    return sapien.Pose(translation, quat)


def matrix_to_sapien_pose(matrix):
    return sapien.Pose(matrix[:3, 3], transforms3d.quaternions.mat2quat(matrix[:3, :3]))


def generate_object_pose(_):
    args, object_code = _

    mesh_path = glob.glob(os.path.join(args.data_root_path, '*', object_code, 'decomposed.obj'))[0]
    mesh_dir = os.path.dirname(mesh_path)
    if not os.path.exists(mesh_path):
        return
    output_filepath = os.path.join(mesh_dir, 'init_obj_poses.npy')
    posed_mesh_dir = os.path.join(mesh_dir, 'init_posed_meshes')
    os.makedirs(posed_mesh_dir, exist_ok=True)
    if not args.overwrite and os.path.exists(output_filepath):
        return

    # --- Scene Setup ---
    scene = sapien.Scene(systems=[physx.PhysxCpuSystem()])
    scene.timestep = args.time_step
    # 高摩擦力，高阻尼，保证物体“放”下去之后不动
    default_material = scene.create_physical_material(static_friction=1.0, dynamic_friction=1.0, restitution=0.0)
    scene.default_physical_material = default_material

    ground_builder = scene.create_actor_builder()
    ground_builder.add_plane_collision(pose=sapien.Pose(p=[0, 0, 0], q=[0.7071068, 0, -0.7071068, 0]), material=default_material)
    ground_builder.build_static(name='ground')

    # --- Load Object ---
    mesh = trimesh.load(mesh_path, force='mesh', process=False)

    # Pre-calculate stable poses
    try:
        stable_transforms, probs = trimesh.poses.compute_stable_poses(mesh, n_samples=5, threshold=0.01)
        if len(stable_transforms) > 0:
            probs = probs / probs.sum()
        else:
            stable_transforms, probs = [], []
    except Exception:
        stable_transforms, probs = [], []

    builder = scene.create_actor_builder()
    builder.add_convex_collision_from_file(mesh_path)
    object_actor = builder.build(name='object')
    physx_body = object_actor.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)

    pose_matrices = []

    for i in range(args.n_samples):
        # Reset velocities
        physx_body.linear_velocity = [0, 0, 0]
        physx_body.angular_velocity = [0, 0, 0]

        # 统一使用高阻尼，因为我们现在都是“放”在地上，而不是“扔”
        physx_body.linear_damping = 2.0
        physx_body.angular_damping = 2.0

        # --- 策略选择 ---
        rand_val = np.random.rand()

        # 策略 1: Stable Poses (物理稳定姿态) - 50%
        if len(stable_transforms) > 0 and rand_val < 0.5:
            idx = np.random.choice(len(stable_transforms), p=probs)
            base_transform = stable_transforms[idx].copy()

        # 策略 2: Canonical Upright (强制保留原始正向) - 30%
        elif rand_val < 0.8:
            base_transform = np.eye(4)

        # 策略 3: Random Rotate (全向随机旋转) - 20%
        # 替代了原本的 Random Drop。这里只生成旋转矩阵，后面会统一做贴地计算。
        else:
            # 生成随机的 Euler 角 (3轴全随机，覆盖 360 度)
            rotation_euler = 2 * np.pi * np.random.rand(3)
            rot_mat = transforms3d.euler.euler2mat(*rotation_euler, axes='sxyz')

            base_transform = np.eye(4)
            base_transform[:3, :3] = rot_mat  # 填入随机旋转

        # --- 统一处理逻辑 (All Strategies) ---

        # 此时 base_transform 包含了物体的“设计朝向”。
        # 下面我们对其进行微调和贴地计算。

        # 1. [Augmentation] 叠加随机 Z 轴旋转 (Yaw)
        # 无论是哪种策略，绕着世界坐标系 Z 轴再转一圈都不影响原本的“朝向性质”
        # (比如躺着的还是躺着，站着的还是站着，随机的还是随机的)
        z_angle = np.random.rand() * 2 * np.pi
        z_rot = transforms3d.euler.euler2mat(0, 0, z_angle)
        # Apply global Z rotation: New_Rot = Z_Rot * Old_Rot
        base_transform[:3, :3] = np.dot(z_rot, base_transform[:3, :3])

        # 2. [Grounding] 精准贴地计算
        # 计算需要平移多少 Z 才能让 Mesh 最低点位于 z=margin
        # 这一步彻底替代了物理 Drop 的过程
        pose = align_pose_to_ground(mesh, base_transform, margin=0.002)

        # 3. [Jitter] 叠加随机 XY 平移
        mat = pose.to_transformation_matrix()
        mat[0, 3] += (np.random.rand() - 0.5) * 0.2
        mat[1, 3] += (np.random.rand() - 0.5) * 0.2

        # 4. 设置 Pose
        object_actor.set_pose(matrix_to_sapien_pose(mat))

        # 5. Settle (极短的物理微调)
        # 因为我们已经通过几何计算(align_pose_to_ground)把位置放得很准了
        # 这里只需要很少的步数来处理 margin 和微小的穿模
        sim_steps = 50

        # --- 运行模拟 ---
        for _ in range(sim_steps):
            scene.step()

        # ... (保存结果) ...
        pose_matrix = object_actor.get_pose().to_transformation_matrix()
        pose_matrices.append(pose_matrix)

        # Export visualized mesh (Optional)
        posed_mesh = mesh.copy()
        posed_mesh.apply_transform(pose_matrix)
        ground_mesh = trimesh.creation.box(extents=[1, 1, 0.00001])
        combined_mesh = trimesh.util.concatenate([posed_mesh, ground_mesh])
        combined_mesh.export(os.path.join(posed_mesh_dir, f'{i:05d}.obj'))

    pose_matrices = np.stack(pose_matrices)
    np.save(output_filepath, pose_matrices)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # experiment settings
    parser.add_argument('--data_root_path', type=str, default=DEFAULT_MESHDATA)
    parser.add_argument('--object_code_list', type=str, default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_samples', type=int, default=1000)
    parser.add_argument('--overwrite', type=bool, default=True)
    parser.add_argument('--n_cpu', type=int, default=64)
    # simulator settings
    # 现在的 sim_steps 只用于 Settle，50-100 步足矣，之前的 1000 步已不再需要
    parser.add_argument('--sim_steps', type=int, default=100)
    parser.add_argument('--time_step', type=float, default=1 / 100)
    parser.add_argument('--restitution', type=float, default=0.01)

    args = parser.parse_args()

    np.random.seed(args.seed)

    if args.object_code_list is None:
        object_code_list = os.listdir(os.path.join(args.data_root_path, 'sam3d')) + os.listdir(os.path.join(args.data_root_path, 'dexgraspnet'))
    else:
        object_code_list = [args.object_code_list]

    with Pool(args.n_cpu) as p:
        param_list = []
        for object_code in object_code_list:
            param_list.append((args, object_code))
        list(tqdm(p.imap(generate_object_pose, param_list), desc='generating', total=len(param_list), miniters=1))
# """
# Last modified date: 2023.02.23
# Author: Jialiang Zhang
# Description: Generate object pose, random free-fall, use SAPIEN
# """
# import os

# os.environ["CUDA_VISIBLE_DEVICES"] = ""
# os.chdir(os.path.dirname(os.path.dirname(__file__)))

# import argparse
# from multiprocessing import Pool

# import numpy as np
# import sapien.core as sapien
# import sapien.physx as physx
# import transforms3d
# import trimesh
# from tqdm import tqdm

# def generate_object_pose(_):
#     args, object_code = _

#     output_dir = os.path.join(args.data_root_path, object_code, 'grasp_correction')
#     os.makedirs(output_dir, exist_ok=True)
#     output_filepath = os.path.join(output_dir, 'init_poses.npy')

#     if not args.overwrite and os.path.exists(output_filepath):
#         return

#     # 1. 显式指定 systems 列表，只包含 PhysxCpuSystem
#     scene = sapien.Scene(systems=[physx.PhysxCpuSystem()])
#     scene.timestep = args.time_step

#     # 2. 设置物理材质
#     default_material = scene.create_physical_material(1.0, 1.0, args.restitution)
#     scene.default_physical_material = default_material

#     # 3. [关键修改] 手动创建纯物理地面，替代 scene.add_ground
#     ground_builder = scene.create_actor_builder()
#     ground_builder.add_plane_collision(pose=sapien.Pose(p=[0, 0, 0], q=[0.7071068, 0, -0.7071068, 0]), material=default_material)
#     ground_builder.build_static(name='ground')

#     # load object

#     mesh_path = os.path.join(args.data_root_path, object_code, 'optimized_hoi_seq', 'obj_canonical.obj')
#     if not os.path.exists(mesh_path):
#         return

#     mesh = trimesh.load(mesh_path, force='mesh', process=False)

#     posed_mesh_dir = os.path.join(output_dir, 'init_posed_meshes')
#     os.makedirs(posed_mesh_dir, exist_ok=True)

#     builder = scene.create_actor_builder()
#     builder.add_convex_collision_from_file(mesh_path)
#     object_actor = builder.build(name='object')

#     # generate object pose
#     ground_mesh = trimesh.creation.box(extents=[1, 1, 0.00001])
#     pose_matrices = []
#     for i in range(args.n_samples):
#         # random pose
#         translation = np.zeros(3)
#         if i >= args.n_samples // 2:
#             translation[2] = 1 + np.random.rand()
#             rotation_euler = 2 * np.pi * np.random.rand(3)
#         else:
#             translation[2] = -np.min(mesh.bounds[:, 2])
#             rotation_euler = np.zeros(3)
#         rotation_quaternion = transforms3d.euler.euler2quat(*rotation_euler, axes='sxyz')
#         try:
#             object_actor.set_root_pose(sapien.Pose(translation, rotation_quaternion))
#         except AttributeError:
#             object_actor.set_pose(sapien.Pose(translation, rotation_quaternion))
#         # simulate
#         for t in range(args.sim_steps):
#             scene.step()

#         pose_matrix = object_actor.get_pose().to_transformation_matrix()
#         pose_matrices.append(pose_matrix)

#         # --- Base simulated pose ---
#         posed_mesh = mesh.copy()
#         posed_mesh.apply_transform(pose_matrix)

#         combined_mesh = trimesh.util.concatenate([posed_mesh, ground_mesh])
#         posed_mesh_filepath = os.path.join(posed_mesh_dir, f'obj_init_{i:05d}.obj')
#         combined_mesh.export(posed_mesh_filepath)

#     pose_matrices = np.stack(pose_matrices)

#     # save results

#     np.save(output_filepath, pose_matrices)
