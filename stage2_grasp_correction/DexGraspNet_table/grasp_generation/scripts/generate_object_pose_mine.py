"""
Generate init_obj_poses.npy for each object under meshdata/dexgraspnet/<id>/.

Required before running grasp_generation/main_prep_data.py when poses are missing.
Default --data_root_path is stage2_grasp_correction/DexGraspNet_table/meshdata.
"""
import argparse
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
    Compute a precise ground-aligned pose.
    After applying rotation, translate so the mesh minimum Z sits at z=margin.
    """
    # 1. Extract rotation
    rotation_matrix = pose_matrix[:3, :3]

    # 2. Apply rotation to mesh vertices (temporary; do not mutate mesh)
    # Cheap matrix multiply only
    rotated_vertices = mesh.vertices.dot(rotation_matrix.T)

    # 3. Find the lowest Z
    min_z = np.min(rotated_vertices[:, 2])

    # 4. Required translation
    # Target: new min_z == margin
    # translation_z = margin - current_min_z
    target_z_translation = -min_z + margin

    # 5. Build translation vector
    # Keep X/Y (or random); set Z from the computation above
    translation = np.zeros(3)
    translation[:2] = pose_matrix[:3, 3][:2]
    translation[2] = target_z_translation

    # 6. Convert to Sapien Pose
    quat = transforms3d.quaternions.mat2quat(rotation_matrix)
    return sapien.Pose(translation, quat)


def matrix_to_sapien_pose(matrix):
    return sapien.Pose(matrix[:3, 3], transforms3d.quaternions.mat2quat(matrix[:3, :3]))


def generate_object_pose(_):
    args, object_code = _

    mesh_path = os.path.join(args.data_root_path, 'dexgraspnet', object_code, 'decomposed.obj')
    mesh_dir = os.path.dirname(mesh_path)
    if not os.path.exists(mesh_path):
        print(f"Skipping {object_code}: {mesh_path} not found.")
        return
    output_filepath = os.path.join(mesh_dir, 'init_obj_poses.npy')
    posed_mesh_dir = os.path.join(mesh_dir, 'init_posed_meshes')
    os.makedirs(posed_mesh_dir, exist_ok=True)
    if not args.overwrite and os.path.exists(output_filepath):
        return

    # --- Scene Setup ---
    scene = sapien.Scene(systems=[physx.PhysxCpuSystem()])
    scene.timestep = args.time_step
    # High friction/damping so the object stays after placement
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

        # Always high damping: we place objects, not drop them
        physx_body.linear_damping = 2.0
        physx_body.angular_damping = 2.0

        # --- Strategy selection ---
        rand_val = np.random.rand()

        # Strategy 1: stable poses - 50%
        if len(stable_transforms) > 0 and rand_val < 0.5:
            idx = np.random.choice(len(stable_transforms), p=probs)
            base_transform = stable_transforms[idx].copy()

        # Strategy 2: canonical upright - 30%
        elif rand_val < 0.8:
            base_transform = np.eye(4)

        # Strategy 3: random rotate - 20%
        # Replaces the old random-drop path; only a rotation is sampled here.
        else:
            # Random Euler angles (all 3 axes, full 360 deg)
            rotation_euler = 2 * np.pi * np.random.rand(3)
            rot_mat = transforms3d.euler.euler2mat(*rotation_euler, axes='sxyz')

            base_transform = np.eye(4)
            base_transform[:3, :3] = rot_mat  # random rotation

        # --- Shared post-processing (all strategies) ---

        # base_transform now holds the desired orientation.
        # Next: small yaw jitter + ground alignment.

        # 1. [Augmentation] random world-Z yaw
        # Extra yaw preserves the strategy class (upright/lying/random)
        # (lying stays lying, upright stays upright, random stays random)
        z_angle = np.random.rand() * 2 * np.pi
        z_rot = transforms3d.euler.euler2mat(0, 0, z_angle)
        # Apply global Z rotation: New_Rot = Z_Rot * Old_Rot
        base_transform[:3, :3] = np.dot(z_rot, base_transform[:3, :3])

        # 2. [Grounding] precise ground alignment
        # Translate Z so mesh min Z sits at z=margin
        # This fully replaces a physics drop
        pose = align_pose_to_ground(mesh, base_transform, margin=0.002)

        # 3. [Jitter] random XY translation
        mat = pose.to_transformation_matrix()
        mat[0, 3] += (np.random.rand() - 0.5) * 0.2
        mat[1, 3] += (np.random.rand() - 0.5) * 0.2

        # 4. Set pose
        object_actor.set_pose(matrix_to_sapien_pose(mat))

        # 5. Short physics settle
        # Geometry (align_pose_to_ground) already places the object accurately
        # Only a few steps are needed for margin / tiny penetration
        sim_steps = 50

        # --- Run simulation ---
        for _ in range(sim_steps):
            scene.step()

        # ... (save results) ...
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
    # sim_steps is only for settle now; 50-100 is enough (was 1000)
    parser.add_argument('--sim_steps', type=int, default=100)
    parser.add_argument('--time_step', type=float, default=1 / 100)
    parser.add_argument('--restitution', type=float, default=0.01)

    args = parser.parse_args()

    np.random.seed(args.seed)

    if args.object_code_list is None:
        dexgraspnet_root = os.path.join(args.data_root_path, 'dexgraspnet')
        object_code_list = os.listdir(dexgraspnet_root) if os.path.isdir(dexgraspnet_root) else []
    else:
        object_code_list = [args.object_code_list]

    with Pool(args.n_cpu) as p:
        param_list = []
        for object_code in object_code_list:
            param_list.append((args, object_code))
        list(tqdm(p.imap(generate_object_pose, param_list), desc='generating', total=len(param_list), miniters=1))
