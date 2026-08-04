import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
import trimesh
from pytorch3d.transforms import (axis_angle_to_matrix, matrix_to_axis_angle, matrix_to_rotation_6d)
from scipy.spatial.transform import Rotation as R

from manotorch.axislayer import AxisLayerFK
from manotorch.manolayer import ManoLayer

_GFM_ROOT = Path(__file__).resolve().parents[1]
if str(_GFM_ROOT) not in sys.path:
    sys.path.insert(0, str(_GFM_ROOT))
from data_layout import DEFAULT_MANO_ASSETS, DEFAULT_MESHDATA, DEFAULT_SOURCE_DIR

logger = logging.getLogger(__name__)


def parse_anatomy_limits(cfg_list):
    """
    Parse a manotorch-style limit string into (Min, Max) radian tensors.
    Input: cfg_list (list of str), e.g., ["+-:0", "+-:10", "+:90,-:0"]
    Output: limits (Tensor), shape (1, 3, 2) -> last dim is [min, max]
    """
    limits = []
    for axis_cfg in cfg_list:
        if "+-" in axis_cfg:
            _, val = axis_cfg.split(":")
            rad = np.deg2rad(float(val))
            limits.append([-rad, rad])
        else:
            # e.g., "+:90,-:0"
            parts = axis_cfg.split(",")
            pos_val = float(parts[0].split(":")[1])
            neg_val = float(parts[1].split(":")[1])
            limits.append([-np.deg2rad(neg_val), np.deg2rad(pos_val)])

    return torch.tensor(limits, dtype=torch.float32).unsqueeze(0)  # (1, 3, 2)


def create_ray_mesh(vec, radius=0.001, color=[255, 255, 0, 255], length_scale=5.0):
    """
    Create a cylinder mesh along vector vec.
    """
    length = np.linalg.norm(vec)

    if length < 1e-6:
        return trimesh.Trimesh()

    # 1. Create a canonical cylinder along Z, centered at the origin
    # sections=8 is enough; no need for high tessellation
    mesh = trimesh.creation.cylinder(radius=radius, height=length * length_scale, sections=8)

    # 2. Rotate so Z aligns with vec
    # trimesh.geometry.align_vectors returns a 4x4 transform
    direction = vec / length
    # Canonical Z axis is [0, 0, 1]
    rot_matrix = trimesh.geometry.align_vectors([0, 0, 1], direction)
    mesh.apply_transform(rot_matrix)

    # 3. Translate to the midpoint of p1 and p2
    midpoint = np.array([0, 0, 0])
    mesh.apply_translation(midpoint)

    # 4. Set color
    mesh.visual.vertex_colors = color

    return mesh


class GraspPair(torch.utils.data.Dataset):

    def __init__(self, data_split='train', debug=False):
        self.data_split = data_split
        self.debug = debug  # debug mode saves visualization files

        # Noise hyperparameters
        # Depth-axis noise (along view ray): large, mimics depth error
        # In-plane noise (perp. to view): small, mimics 2D detection jitter
        self.noise_depth_range = 0.3  # 30cm
        self.noise_plane_range = 0.02  # 2cm
        # self.noise_plane_range = 0.  # NOTE for debug

        # Keep rotation / PCA pose noise modest,
        # since accurate rotation is the key that guides translation recovery
        self.noise_rot_std = 0.1  # slightly smaller (was 0.1)
        self.noise_pose_std = 0.1
        # self.noise_rot_std = 0.  # NOTE for debug
        # self.noise_pose_std = 0.  # NOTE for debug

        # Load sample list
        self.datalist = self.load_data()
        logger.info(f"Loaded {len(self.datalist)} samples")

        # Initialize ManoLayer
        self.manolayer = ManoLayer(
            mano_assets_root=str(DEFAULT_MANO_ASSETS),
            flat_hand_mean=True,
            use_pca=False,  # we inject noise ourselves; use full pose params
            side='right')

        # PCA components (45, 45)
        self.pca_comps = self.manolayer.th_comps.numpy()

        # MANO root-joint location at zero pose (J)
        zeros_pose = torch.zeros(1, 48)

        # Joint locations
        mano_output = self.manolayer(zeros_pose)
        joints = mano_output.joints

        # joints are in meters
        self.root_joint_J = joints[0, 0, :].detach().numpy()

        # 1. AxisLayerFK (anatomical basis)
        # Requires manotorch assets path
        self.axis_layer = AxisLayerFK(side='right', mano_assets_root=str(DEFAULT_MANO_ASSETS))

        # 2. Precompute joint limits (AnatomyConstraintLossEE.setup defaults)
        # Layout: [Twist(X), Spread(Y), Bend(Z)]

        # Thumb limits
        self.limit_thumb_cmc = parse_anatomy_limits(["+-:45", "+:45,-:15", "+:45,-:0"])  # Index 13
        self.limit_thumb_mcp = parse_anatomy_limits(["+-:0", "+-:10", "+:90,-:0"])  # Index 14
        self.limit_thumb_pip = parse_anatomy_limits(["+-:0", "+-:0", "+:90,-:0"])  # Index 15

        # Finger limits (MCP, PIP, DIP)
        self.limit_finger_mcp = parse_anatomy_limits(["+-:0", "+-:5", "+:90,-:0"])  # Indices [1, 4, 7, 10]
        self.limit_finger_pip = parse_anatomy_limits(["+-:0", "+-:0", "+:90,-:0"])  # Indices [2, 5, 8, 11]
        self.limit_finger_dip = parse_anatomy_limits(["+-:0", "+-:0", "+:90,-:0"])  # Indices [3, 6, 9, 12]

    def load_data(self):
        datalist = []
        data_root = DEFAULT_MESHDATA
        # Avoid hard-failing when meshdata has not been prepared yet.
        if not data_root.exists():
            print(f"Warning: Path {data_root} not found.")
            return []

        for data_item in data_root.glob('*/*/grasp_data/[0-9][0-9][0-9][0-9][0-9].json'):
            datalist.append(data_item.as_posix())
        # for data_item in data_root.glob('*/*/grasp_data/[0-9][0-9][0-9][0-9][0-9].json'):
        #     if '87141' in data_item.as_posix():
        #         datalist.append(data_item.as_posix())
        # datalist = datalist[:1]
        # for data_item in data_root.glob('*/*.json'):
        #     if '32661' in data_item.as_posix():
        #         datalist.append(data_item.as_posix())
        # datalist = datalist[:2]

        # return datalist
        if self.data_split == 'train':
            return datalist[:int(len(datalist) * 0.9)]
        else:
            return datalist[int(len(datalist) * 0.9):]

    def __len__(self):
        return len(self.datalist)

    def _process_pointcloud(self, obj_points):
        """
        Input: 
            obj_points: (N, 3) numpy array (World Frame)
        Output: 
            points_norm: (num_points, 3) normalized points
            centroid: (3,) centroid used to translate the hand
            scale: (1,) scale used to resize the hand and as a condition
        """
        # Convert to numpy for consistent ops
        points = np.array(obj_points, dtype=np.float32)
        centroid = np.mean(points, axis=0)
        points_centered = points - centroid

        scale = np.max(np.linalg.norm(points_centered, axis=1))
        if scale < 1e-6:
            scale = 1.0

        points_norm = points_centered / scale
        return points_norm.astype(np.float32), centroid.astype(np.float32), np.array([scale], dtype=np.float32)

    def _sanitize_pose_manotorch(self, pose_15x3):
        """
        Hard-clamp pose via manotorch anatomical frame transforms.
        Input: pose_15x3 (numpy, 15x3) -> Noisy Axis-Angle
        Output: clean_pose (numpy, 15x3)
        """
        # 0. Prepare tensors / batch dim
        # AxisLayerFK needs global transforms, so run ManoLayer first
        # Indirect, but the correct path to anatomically aligned angles

        noisy_pose = torch.from_numpy(pose_15x3).float().flatten().unsqueeze(0)  # (1, 45)
        # Pad zero global rot / shape just to run the layer
        # Real trans/rot are unnecessary; only local joint rotations matter
        global_rot = torch.zeros(1, 3)
        shape = torch.zeros(1, 10)
        full_pose = torch.cat([global_rot, noisy_pose], dim=1)  # (1, 48)

        # 1. FK via ManoLayer -> global transforms
        # Reuse self.manolayer
        mano_out = self.manolayer(full_pose, shape)
        transf_abs = mano_out.transforms_abs  # (1, 16, 4, 4)

        # 2. AxisLayerFK forward -> anatomical Euler angles
        #
        # T_g_a: Anatomy aligned transforms
        # ee: Anatomical Euler Angles (1, 16, 3) -> Order: Twist, Spread, Bend (XYZ)
        _, _, ee_anat = self.axis_layer(transf_abs)

        # 3. Hard clamp with physiological limits
        # Mutate ee_anat in place

        # Indices (manotorch 16-joint order; index 0 is wrist, usually untouched)
        # Helper clamp
        # MCP: [1, 4, 7, 10], PIP: [2, 5, 8, 11], DIP: [3, 6, 9, 12], Thumb CMC: [13], Thumb MCP: [14], Thumb PIP: [15]
        for indices, limits_tensor in zip(
            [[1, 4, 7, 10], [2, 5, 8, 11], [3, 6, 9, 12], [13], [14], [15]],
            [self.limit_finger_mcp, self.limit_finger_pip, self.limit_finger_dip, self.limit_thumb_cmc, self.limit_thumb_mcp, self.limit_thumb_pip]):
            # limits_tensor: (1, 3, 2) -> min, max
            min_val = limits_tensor[:, :, 0].to(ee_anat.device)
            max_val = limits_tensor[:, :, 1].to(ee_anat.device)

            # Euler angles for selected joints (1, N, 3)
            current_ee = ee_anat[:, indices, :]

            # Clip
            clamped_ee = torch.max(torch.min(current_ee, max_val), min_val)

            # Write back
            ee_anat[:, indices, :] = clamped_ee

        # 4. AxisLayerFK compose -> back to MANO pose
        # Rebuild MANO axis-angle from anatomical Euler angles
        clean_pose_aa = self.axis_layer.compose(ee_anat)  # (1, 16, 3)

        # Drop wrist (index 0) -> (15, 3)
        clean_pose_15x3 = clean_pose_aa[0, 1:, :].detach().cpu().numpy()

        return clean_pose_15x3

    def _sample_perspective_noise(self, target_trans):
        """
        Core augmentation: anisotropic (needle-like) noise
        Mimics monocular RGB: accurate in-plane projection, inaccurate depth.
        """
        # 1. Random virtual camera distance in normalized space
        # Object at origin; place camera uniformly in [0.5, 2.0]
        # Covers close-up to typical capture distances
        radius = np.random.uniform(0.5, 2.0)

        # 2. Random camera direction on the sphere
        raw_dir = np.random.normal(0, 1, 3)
        camera_dir = raw_dir / np.linalg.norm(raw_dir)  # Normalize

        # Virtual camera position
        camera_pos = camera_dir * radius

        # 3. View-ray direction
        # Ray = camera position - true hand position
        # Most error occurs along this ray
        ray_vec = camera_pos - target_trans
        # True hand-to-camera distance
        dist_hand_to_cam = np.linalg.norm(ray_vec)
        view_dir = ray_vec / (dist_hand_to_cam + 1e-6)

        # 4. Anisotropic noise

        # A. Depth error along view_dir: Uniform
        noise_depth_mag = np.random.uniform(-self.noise_depth_range, self.noise_depth_range)

        # Prevent the hand from going behind the camera
        # Danger case: noise_depth_mag > 0 (moves toward the camera along view_dir)

        min_safe_dist = 0.1  # Minimum near distance to camera

        # Max forward move = current distance - safe margin
        # If already closer than the safe margin, max_move_forward becomes negative,
        # and the following min forces noise_depth_mag negative (push away), which is fine.
        max_move_forward = dist_hand_to_cam - min_safe_dist

        # Clamp forward (toward-camera) noise
        if noise_depth_mag > 0:
            noise_depth_mag = min(noise_depth_mag, max_move_forward)
        noise_depth = noise_depth_mag * view_dir

        # B. In-plane error (perp. to view_dir): Gaussian
        # Detection jitter is well modeled by a Normal distribution
        random_vec = np.random.normal(0, 1, 3)
        proj = np.dot(random_vec, view_dir) * view_dir
        perp_vec = random_vec - proj
        if np.linalg.norm(perp_vec) > 1e-6:
            perp_vec = perp_vec / np.linalg.norm(perp_vec)

        noise_plane_mag = np.random.normal(0, self.noise_plane_range)
        noise_plane = noise_plane_mag * perp_vec

        # 5. Compose final noise vector
        total_noise = noise_depth + noise_plane
        return total_noise, camera_pos, noise_depth_mag

    def _add_noise_to_mano_pca(self, trans, rot_vec, pose_15x3):
        """
        Input:
            trans: (3,) numpy
            rot_vec: (3,) numpy (Axis-Angle)
            pose_15x3: (45,) or (15, 3) numpy
        Output:
            Noisy params in same shape
        """
        # 1. Perspective noise instead of isotropic Gaussian ---
        noise_t, camera_pos, noise_depth_mag = self._sample_perspective_noise(trans)
        noisy_trans = trans + noise_t

        # 2. Rotation noise (axis-angle add, small perturbation)
        noise_r = np.random.normal(0, self.noise_rot_std, size=3)
        noisy_rot = rot_vec + noise_r

        # 3. Pose Noise via PCA
        pose_flat = pose_15x3.flatten()
        n_comps = self.pca_comps.shape[0]

        # 3. Geometric (exponential) decay instead of linear
        # Better matches PCA eigenvalue decay
        start_std = self.noise_pose_std
        if start_std < 1e-6:
            # If zero, skip geomspace and use an all-zero vector
            pca_std = np.zeros(n_comps)
        else:
            end_std = self.noise_pose_std * 0.01
            pca_std = np.geomspace(start_std, end_std, n_comps)
        noise_pca = np.random.normal(0, pca_std, size=(n_comps,))

        # Project: (N,) @ (N, 45) -> (45,)
        delta_pose_flat = noise_pca @ self.pca_comps

        # Add onto pose
        noisy_pose_flat = pose_flat + delta_pose_flat

        noisy_pose_reshaped = noisy_pose_flat.reshape(15, 3)

        # Sanitize via manotorch FK/IK transforms
        final_pose = self._sanitize_pose_manotorch(noisy_pose_reshaped)

        return noisy_trans, noisy_rot, final_pose, camera_pos, noise_depth_mag

    def _preprocess_mano(self, mano_params):
        trans, root_rot, pose = mano_params[:3], mano_params[3:6], mano_params[6:]
        trans = torch.from_numpy(trans).float()
        root_rot = torch.from_numpy(root_rot).float()
        pose = torch.from_numpy(pose).float()

        # 1. Convert
        root_rot_6d = matrix_to_rotation_6d(axis_angle_to_matrix(root_rot))  # (6,)
        pose_6d = matrix_to_rotation_6d(axis_angle_to_matrix(pose.reshape(15, 3)))  # (15, 6)

        # 3. Pad translation
        trans_padded = torch.cat([trans, trans], dim=0)  # (6,)

        # 4. Concatenate tokens
        # Layout: [Trans(6), Root(6), Joint1(6), ..., Joint15(6)] -> (17, 6)
        x_6d = torch.cat([trans_padded.unsqueeze(0), root_rot_6d.unsqueeze(0), pose_6d], dim=0)  # (17, 6)

        return x_6d

    def save_debug_viz(self, video_id, target_params, source_params, camera_ray, obj_mesh, obj_points, obj_scale):
        """
        Debug helper: save GT/noisy hand and object as separate PLY files.
        params format: (trans, rot, pose)
        """
        # Per-sample output subdirectory
        save_dir = Path("debug_viz") / f"{video_id}"
        save_dir.mkdir(parents=True, exist_ok=True)

        # --- 4. Ray mesh (cylinder) ---
        ray_mesh = create_ray_mesh(camera_ray, radius=0.005, color=[255, 255, 0, 255])

        # 1. Tensors for ManoLayer
        def prepare_input(params, scale):
            trans, rot, pose = params
            # ManoLayer expects a batch dim
            th_trans = torch.from_numpy(trans * scale).float().unsqueeze(0)  # (1, 3)
            th_rot = torch.from_numpy(rot).float().unsqueeze(0)  # (1, 3)
            th_pose = torch.from_numpy(pose.flatten()).float().unsqueeze(0)  # (1, 45)
            # Concat rot and pose -> (1, 48)
            th_pose_coeffs = torch.cat([th_rot, th_pose], dim=1)
            return th_pose_coeffs, th_trans

        def points_to_spheres(points, radius=0.002, subdivision=1, num_samples=None):
            """
            Convert a point cloud to a union of small sphere meshes (vectorized, no Python loop).
            
            Args:
                points: (n, 3) float point coordinates
                radius: float sphere radius
                subdivision: int sphere tessellation (lower is faster)
                
            Returns:
                trimesh.Trimesh: merged mesh
            """
            if num_samples is not None and num_samples < len(points):
                indices = np.random.permutation(len(points))[:num_samples]
                points = points[indices]

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

        # 2. GT hand mesh (target)
        gt_pose_coeffs, gt_trans = prepare_input(target_params, obj_scale)
        mano_output = self.manolayer(pose_coeffs=gt_pose_coeffs)
        gt_verts = mano_output.verts + gt_trans

        gt_verts = gt_verts[0].detach().cpu().numpy()
        faces = self.manolayer.th_faces.detach().cpu().numpy()
        gt_mesh = trimesh.Trimesh(vertices=gt_verts, faces=faces)
        gt_mesh.visual.vertex_colors = np.array([0, 255, 0, 200])  # GT hand green

        # 3. Noisy hand mesh (source)
        src_pose_coeffs, src_trans = prepare_input(source_params, obj_scale)
        mano_output = self.manolayer(pose_coeffs=src_pose_coeffs)
        src_verts = mano_output.verts + src_trans

        src_verts = src_verts[0].detach().cpu().numpy()
        src_mesh = trimesh.Trimesh(vertices=src_verts, faces=faces)
        src_mesh.visual.vertex_colors = np.array([255, 0, 0, 200])  # noisy hand red

        # 4. Object mesh
        obj_mesh_copy = obj_mesh.copy()
        obj_mesh_copy.visual.vertex_colors = np.array([128, 128, 128, 255])  # object gray
        obj_mesh_copy.apply_scale(obj_scale)

        # 5. Object point-cloud mesh
        obj_points_mesh = points_to_spheres(obj_points)
        obj_points_mesh.visual.vertex_colors = np.array([128, 128, 128, 255])  # object points gray
        obj_points_mesh.apply_scale(obj_scale)

        # Save five separate files
        src_mesh.export(save_dir / "source_hand_mesh.ply", file_type='ply')
        gt_mesh.export(save_dir / "target_hand_mesh.ply", file_type='ply')
        ray_mesh.export(save_dir / "camera_ray.ply", file_type='ply')
        obj_mesh_copy.export(save_dir / "object_mesh.ply", file_type='ply')
        obj_points_mesh.export(save_dir / "object_point_cloud.ply", file_type='ply')

        print(f"[DEBUG] Saved visualization files to {save_dir}/")

    def __getitem__(self, index):
        data_path = self.datalist[index]
        with open(data_path, 'r') as f:
            data = json.load(f)

        video_id = Path(data_path).parent.name

        # --- Load raw data (world frame) ---
        mano_params = data['final_mano_params']
        # Ensure float32 numpy
        gt_root_orient = np.array(mano_params['root_orient'], dtype=np.float32)
        gt_trans = np.array(mano_params['trans'], dtype=np.float32)
        gt_pose = np.array(mano_params['pose'], dtype=np.float32)  # (45,)

        obj_scale = np.array(data['object_scale'], dtype=np.float32)
        plane_pose = data['plane_pose']
        plane_rot = np.array(plane_pose['rotation'], dtype=np.float32)
        plane_trans = np.array(plane_pose['translation'], dtype=np.float32)

        # --- Process points and poses ---
        # 1. Transform object and hand into world frame
        base_path = Path(data_path).parent.parent
        canonical_object_path = base_path / 'decomposed.obj'
        point_cloud_path = base_path / 'obj_points_10000.ply'

        # Load and transform point cloud
        point_cloud = trimesh.load(point_cloud_path, process=False)
        obj_mesh = trimesh.load(canonical_object_path, force='mesh', process=False)
        point_cloud.apply_scale(obj_scale)
        obj_mesh.apply_scale(obj_scale)
        transform = np.eye(4)
        transform[:3, :3] = plane_rot
        transform[:3, 3] = plane_trans
        point_cloud.apply_transform(transform)
        obj_mesh.apply_transform(transform)
        obj_points = point_cloud.vertices
        # obj_normals = point_cloud.vertex_normals
        # Try the standard attribute path
        obj_normals = getattr(point_cloud, 'vertex_normals', None)

        # If missing, fall back to reading normals from PLY metadata
        if obj_normals is None or obj_normals.shape[0] == 0:
            try:
                # Read structured array from raw PLY data
                raw_data = point_cloud.metadata['_ply_raw']['vertex']['data']

                # Stack nx, ny, nz
                # raw_data['nx'] is 1D; stack to (N, 3)
                obj_normals = np.column_stack((raw_data['nx'], raw_data['ny'], raw_data['nz']))
                # Ensure float32
                obj_normals = obj_normals.astype(np.float32)

            except KeyError:
                raise RuntimeError(f"Fatal: Could not find normals in {point_cloud_path} metadata. Please checking generation.")

        # Randomly downsample to 4096
        num_points = 4096  # or 2048
        if len(obj_points) >= num_points:
            choice = np.random.choice(len(obj_points), num_points, replace=False)
        else:
            # Guard against meshes with too few points
            choice = np.random.choice(len(obj_points), num_points, replace=True)

        obj_points = obj_points[choice]
        obj_normals = obj_normals[choice]

        # Transform hand pose
        gt_root_orient = R.from_matrix(plane_rot @ R.from_rotvec(gt_root_orient).as_matrix()).as_rotvec()
        j_correction = (plane_rot @ self.root_joint_J.reshape(-1, 1)).reshape(-1) - self.root_joint_J
        gt_trans = (plane_rot @ gt_trans.reshape(-1, 1)).reshape(-1) + plane_trans + j_correction

        # 2. (train) random rotation augmentation
        if self.data_split == 'train':
            random_rot = R.random().as_matrix().astype(np.float32)

            # Apply to points, normals, mesh, and hand in world frame
            obj_points = obj_points @ random_rot.T
            obj_normals = obj_normals @ random_rot.T
            obj_mesh.vertices = obj_mesh.vertices @ random_rot.T
            gt_root_orient = R.from_matrix(random_rot @ R.from_rotvec(gt_root_orient).as_matrix()).as_rotvec()
            j_correction = (random_rot @ self.root_joint_J.reshape(-1, 1)).reshape(-1) - self.root_joint_J
            gt_trans = (random_rot @ gt_trans.reshape(-1, 1)).reshape(-1) + j_correction

        target_trans = gt_trans
        target_root_orient = gt_root_orient
        target_pose = gt_pose

        # --- Add noise to build the source input ---
        source_trans, source_root_orient, source_pose, camera_pos, noise_depth_mag = self._add_noise_to_mano_pca(target_trans, target_root_orient, target_pose)
        noise_depth_mag = -noise_depth_mag  # from source to target

        camera_ray = camera_pos - source_trans
        camera_ray = camera_ray / np.linalg.norm(camera_ray)

        # get hand joints and vertices
        # ManoLayer expects a batch dim
        th_trans = torch.from_numpy(source_trans).float().unsqueeze(0)  # (1, 3)
        th_rot = torch.from_numpy(source_root_orient).float().unsqueeze(0)  # (1, 3)
        th_pose = torch.from_numpy(source_pose.flatten()).float().unsqueeze(0)  # (1, 45)
        # Concat rot and pose -> (1, 48)
        th_pose_coeffs = torch.cat([th_rot, th_pose], dim=1)
        mano_output = self.manolayer(pose_coeffs=th_pose_coeffs)
        source_verts = mano_output.verts + th_trans
        source_joints = mano_output.joints + th_trans
        source_verts = source_verts[0].detach().cpu().numpy()
        source_joints = source_joints[0].detach().cpu().numpy()

        th_trans = torch.from_numpy(target_trans).float().unsqueeze(0)  # (1, 3)
        th_rot = torch.from_numpy(target_root_orient).float().unsqueeze(0)  # (1, 3)
        th_pose = torch.from_numpy(target_pose.flatten()).float().unsqueeze(0)  # (1, 45)
        # Concat rot and pose -> (1, 48)
        th_pose_coeffs = torch.cat([th_rot, th_pose], dim=1)
        mano_output = self.manolayer(pose_coeffs=th_pose_coeffs)
        target_verts = mano_output.verts + th_trans
        target_joints = mano_output.joints + th_trans
        target_verts = target_verts[0].detach().cpu().numpy()
        target_joints = target_joints[0].detach().cpu().numpy()

        # 3. Normalize
        normalized_obj_points, obj_centroid, normalized_obj_scale = self._process_pointcloud(obj_points)
        normalized_obj_mesh = trimesh.Trimesh(vertices=(obj_mesh.vertices - obj_centroid) / normalized_obj_scale, faces=obj_mesh.faces)

        target_trans = (target_trans - obj_centroid) / normalized_obj_scale  # add the same shift and scale to the hand as object's
        source_trans = (source_trans - obj_centroid) / normalized_obj_scale  # add the same shift and scale to the hand as object's
        noise_depth_mag = noise_depth_mag / normalized_obj_scale
        noise_depth_mag = np.array([noise_depth_mag], dtype=np.float32)

        source_verts = (source_verts - obj_centroid) / normalized_obj_scale
        target_verts = (target_verts - obj_centroid) / normalized_obj_scale
        source_joints = (source_joints - obj_centroid) / normalized_obj_scale
        target_joints = (target_joints - obj_centroid) / normalized_obj_scale

        # --- [DEBUG] visualization ---
        # Save for random/selected indices to avoid heavy IO
        if self.debug:
            print(f"[DEBUG] Saving debug visualization for sample {data_path}")
            self.save_debug_viz(
                video_id=f"{video_id}_{index}",
                target_params=(target_trans, target_root_orient, target_pose),
                source_params=(source_trans, source_root_orient, source_pose),
                camera_ray=camera_ray,
                obj_mesh=normalized_obj_mesh,
                obj_points=normalized_obj_points,
                obj_scale=normalized_obj_scale,
            )

        # --- 6. Convert to tensors and return ---
        # Pack hand state: [Trans(3), Rot(3), Pose(45)] -> (51,)
        target_hand = np.concatenate([target_trans, target_root_orient, target_pose.flatten()])
        source_hand = np.concatenate([source_trans, source_root_orient, source_pose.flatten()])

        normalized_obj_points = np.concatenate([normalized_obj_points, obj_normals], axis=1)

        x = self._preprocess_mano(target_hand)
        cond_x = self._preprocess_mano(source_hand)
        gt_hand = self._preprocess_mano(target_hand)
        cond_hand = torch.from_numpy(source_joints).float()
        cond_obj_scale = torch.from_numpy(normalized_obj_scale).float()
        cond_obj_pointcloud = torch.from_numpy(normalized_obj_points).float()
        cond_camera_ray = torch.from_numpy(camera_ray).float()
        noise_depth_mag = torch.from_numpy(noise_depth_mag).float()

        input_dict = {
            'x': x,
            'cond_x': cond_x,
            'gt_hand': gt_hand,
            'cond_hand': cond_hand,
            'cond_obj_scale': cond_obj_scale,
            'cond_obj_pointcloud': cond_obj_pointcloud,
            'cond_camera_ray': cond_camera_ray,
            'noise_depth_mag': noise_depth_mag,
        }

        return input_dict


DEFAULT_TEST_SOURCE_DIR = DEFAULT_SOURCE_DIR


class GraspTest(torch.utils.data.Dataset):

    def __init__(self, debug=False, video_id=None, seq_dir_name=None, source_dir=None):
        self.debug = debug  # debug mode writes visualization dumps
        # Optional video_id filter (None/empty -> all discovered ids)
        self.selected_video_id = video_id
        self.seq_dir_name = seq_dir_name
        self.source_dir = Path(source_dir) if source_dir is not None else DEFAULT_TEST_SOURCE_DIR

        self.datalist = self.load_data()
        logger.info(f"Loaded {len(self.datalist)} samples")

        self.manolayer = ManoLayer(
            mano_assets_root=str(DEFAULT_MANO_ASSETS),
            flat_hand_mean=True,
            use_pca=False,
            side='right')

        # [New] MANO root joint at zero pose (J)
        # Only one forward pass needed
        zeros_pose = torch.zeros(1, 48)

        # One forward pass for joint locations
        mano_output = self.manolayer(zeros_pose)
        joints = mano_output.joints

        # joints are in meters
        self.root_joint_J = joints[0, 0, :].detach().numpy()

    def load_data(self):
        datalist = []
        base_input_path = self.source_dir
        excluded_dirs = ['dynhamr', 'images']

        if not base_input_path.exists():
            logger.warning(f"Fatal: Source directory {base_input_path} not found. Please checking generation.")
            return datalist

        for video_id_path in base_input_path.iterdir():
            if not video_id_path.is_dir() or video_id_path.name in excluded_dirs:
                continue
            video_id = video_id_path.name
            if self.selected_video_id:
                selected_video_ids = list(self.selected_video_id)
            else:
                selected_video_ids = [
                    'IMG_5044',
                    # 'IMG_5050',
                    # 'IMG_5051',
                    # 'IMG_5052',
                    # 'IMG_5053',
                    # 'IMG_5055',
                ]
            if str(video_id) not in selected_video_ids:
                continue

            # # Skip if optimized_hoi_contact_seq already exists
            # contact_seq_path = video_id_path / 'optimized_hoi_contact_seq'
            # if contact_seq_path.exists():
            #     logger.info(f"Skipping {video_id}: optimized_hoi_contact_seq already exists")
            #     continue

            if self.seq_dir_name:
                data_root = video_id_path / self.seq_dir_name
                if not data_root.exists():
                    logger.warning(f"Fatal: Data root {data_root} not found. Please checking generation.")
                    continue
            else:
                data_root = video_id_path / 'optimized_hoi_init_seq'
                if not data_root.exists():
                    data_root = video_id_path / 'optimized_hoi_seq'
                    if not data_root.exists():
                        logger.warning(f"Fatal: Data root {data_root} not found. Please checking generation.")
                        continue
                        # raise RuntimeError(f"Fatal: Data root {data_root} not found. Please checking generation.")

            mano_files = sorted(data_root.glob("mano_*.json"))
            for mano_file in mano_files:
                frame_id = mano_file.stem.split('_')[-1]
                obj_file = data_root / f"obj_{frame_id}.json"

                if obj_file.exists():
                    datalist.append(str(mano_file))
                else:
                    logger.warning(f"Missing obj file for frame {frame_id} in {video_id}.")

        if not datalist:
            logger.error("No valid data found across all video directories.")

        return datalist

    def __len__(self):
        return len(self.datalist)

    def _process_pointcloud(self, obj_points):
        """
        Input: 
            obj_points: (N, 3) numpy array (World Frame)
        Output: 
            points_norm: (num_points, 3) normalized points
            centroid: (3,) centroid used to translate the hand
            scale: (1,) scale used to resize the hand and as a condition
        """
        # Convert to numpy for consistent ops
        points = np.array(obj_points, dtype=np.float32)
        centroid = np.mean(points, axis=0)
        points_centered = points - centroid

        scale = np.max(np.linalg.norm(points_centered, axis=1))
        if scale < 1e-6:
            scale = 1.0

        points_norm = points_centered / scale
        return points_norm.astype(np.float32), centroid.astype(np.float32), np.array([scale], dtype=np.float32)

    def _preprocess_mano(self, mano_params):
        trans, root_rot, pose = mano_params[:3], mano_params[3:6], mano_params[6:]
        trans = torch.from_numpy(trans).float()
        root_rot = torch.from_numpy(root_rot).float()
        pose = torch.from_numpy(pose).float()

        # 1. Convert
        root_rot_6d = matrix_to_rotation_6d(axis_angle_to_matrix(root_rot))  # (6,)
        pose_6d = matrix_to_rotation_6d(axis_angle_to_matrix(pose.reshape(15, 3)))  # (15, 6)

        # 3. Pad translation
        trans_padded = torch.cat([trans, trans], dim=0)  # (6,)

        # 4. Concatenate tokens
        # Layout: [Trans(6), Root(6), Joint1(6), ..., Joint15(6)] -> (17, 6)
        x_6d = torch.cat([trans_padded.unsqueeze(0), root_rot_6d.unsqueeze(0), pose_6d], dim=0)  # (17, 6)

        return x_6d

    def save_debug_viz(self, video_id, source_params, camera_ray, obj_mesh, obj_points, obj_scale, side):
        """
        Debug helper: save GT/noisy hand and object into one OBJ.
        params format: (trans, rot, pose)
        """
        save_dir = Path("debug_viz")
        save_dir.mkdir(exist_ok=True)
        filename = save_dir / f"{video_id}_debug.ply"

        # --- 4. Ray mesh (cylinder) ---
        ray_mesh = create_ray_mesh(camera_ray, radius=0.005, color=[255, 255, 0, 255])

        # 1. Tensors for ManoLayer
        def prepare_input(params, scale):
            trans, rot, pose = params
            # ManoLayer expects a batch dim
            th_trans = torch.from_numpy(trans * scale).float().unsqueeze(0)  # (1, 3)
            th_rot = torch.from_numpy(rot).float().unsqueeze(0)  # (1, 3)
            th_pose = torch.from_numpy(pose.flatten()).float().unsqueeze(0)  # (1, 45)
            # Concat rot and pose -> (1, 48)
            th_pose_coeffs = torch.cat([th_rot, th_pose], dim=1)
            return th_pose_coeffs, th_trans

        def points_to_spheres(points, radius=0.01, subdivision=1, num_samples=None):
            """
            Convert a point cloud to a union of small sphere meshes (vectorized, no Python loop).
            """
            if num_samples is not None and num_samples < len(points):
                indices = np.random.permutation(len(points))[:num_samples]
                points = points[indices]

            sphere = trimesh.creation.icosphere(subdivisions=subdivision, radius=radius)
            v_template = sphere.vertices
            f_template = sphere.faces

            n_points = len(points)
            n_v = len(v_template)
            n_f = len(f_template)

            new_vertices = (points[:, np.newaxis, :] + v_template[np.newaxis, :, :]).reshape(-1, 3)
            offsets = np.arange(n_points) * n_v
            new_faces = (f_template[np.newaxis, :, :] + offsets[:, np.newaxis, np.newaxis]).reshape(-1, 3)
            mesh = trimesh.Trimesh(vertices=new_vertices, faces=new_faces)

            return mesh

        hand_faces = self.manolayer.th_faces.detach().cpu().numpy()

        # 3. Noisy hand mesh (source)
        src_pose_coeffs, src_trans = prepare_input(source_params, obj_scale)
        mano_output = self.manolayer(src_pose_coeffs)
        src_verts = mano_output.verts + src_trans

        src_verts = src_verts[0].detach().cpu().numpy()
        src_mesh = trimesh.Trimesh(vertices=src_verts, faces=hand_faces)
        src_mesh.visual.vertex_colors = np.array([255, 0, 0, 200])  # noisy hand red

        # Export
        obj_mesh.apply_scale(obj_scale)
        obj_mesh.visual.vertex_colors = np.array([128, 128, 128, 255])  # object gray

        obj_points_mesh = points_to_spheres(obj_points, num_samples=100)
        obj_points_mesh.visual.vertex_colors = np.array([128, 128, 128, 255])  # object gray
        obj_points_mesh.apply_scale(obj_scale)

        scene_mesh = trimesh.util.concatenate([obj_points_mesh, obj_mesh, src_mesh, ray_mesh])
        # scene_mesh = trimesh.util.concatenate([obj_mesh, src_mesh, ray_mesh])
        scene_mesh.export(filename, file_type='ply')
        print(f"[DEBUG] Saved visualization to {filename}")

    def __getitem__(self, index):
        mano_path_str = self.datalist[index]
        mano_path = Path(mano_path_str)

        frame_id = mano_path.stem.split('_')[-1]
        if self.seq_dir_name:
            # seq_dir_name can be nested, e.g. grasp_correction/gfm_input_hoi_seq.
            # In that case parent.parent is grasp_correction, not the video id.
            video_id = mano_path.parents[len(Path(self.seq_dir_name).parts)].name
        else:
            video_id = mano_path.parent.parent.name
        base_path = mano_path.parent

        obj_path = base_path / f"obj_{frame_id}.json"
        canonical_obj_path = str(DEFAULT_MESHDATA / "sam3d" / video_id / "decomposed.obj")

        with open(mano_path, 'r') as f:
            mano_data = json.load(f)
        with open(obj_path, 'r') as f:
            obj_data = json.load(f)

        side = 'right' if mano_data['is_right'] else 'left'

        source_root_orient = np.array(mano_data['root_orient'], dtype=np.float32)
        source_pose = np.array(mano_data['pose'], dtype=np.float32)
        source_trans = np.array(mano_data['trans'], dtype=np.float32)

        obj_scale = np.array(obj_data['scale'], dtype=np.float32)
        obj_rot = np.array(obj_data['rotation'], dtype=np.float32)
        obj_trans = np.array(obj_data['translation'], dtype=np.float32)

        camera_pos_world = np.array([0, 0, 0], dtype=np.float32)

        obj_mesh = trimesh.load(canonical_obj_path, force='mesh', process=False)
        obj_mesh.apply_scale(obj_scale)
        diag_mat = np.diag([-1, -1, 1])  # hand/object frames differ by 180 about Z; flip object pose
        obj_trans = obj_trans @ diag_mat
        obj_rot = diag_mat @ obj_rot @ diag_mat
        obj_transform = np.eye(4)
        obj_transform[:3, :3] = obj_rot
        obj_transform[:3, 3] = obj_trans
        if side == 'left':
            # Flip-X matrix
            flip_x_mat = np.diag([-1, 1, 1, 1])
            obj_transform = flip_x_mat @ obj_transform @ flip_x_mat
            obj_mesh.faces = obj_mesh.faces[:, [0, 2, 1]]
        # Rotation from [Z-fwd, Y-up] to [X-fwd, Z-up]
        coord_transfer_transform = np.array([[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
        obj_transform = coord_transfer_transform @ obj_transform
        obj_mesh.apply_transform(obj_transform)

        # Transform hand pose into the training [X-fwd, Z-up] frame
        source_root_orient = matrix_to_axis_angle(
            torch.from_numpy(coord_transfer_transform[:3, :3]).float().unsqueeze(0) @ axis_angle_to_matrix(
                torch.from_numpy(source_root_orient).float().unsqueeze(0)))
        source_root_orient = source_root_orient.squeeze(0).detach().cpu().numpy()

        j_correction = (coord_transfer_transform[:3, :3] @ self.root_joint_J.reshape(-1, 1)).reshape(-1) - self.root_joint_J
        source_trans = (torch.from_numpy(source_trans).float().unsqueeze(0) - torch.from_numpy(
            coord_transfer_transform[:3, 3]).float().unsqueeze(0)) @ torch.from_numpy(coord_transfer_transform[:3, :3]).float().T + j_correction
        source_trans = source_trans.squeeze(0).detach().cpu().numpy()

        camera_ray = camera_pos_world - source_trans
        camera_ray = camera_ray / np.linalg.norm(camera_ray)

        # get hand joints and vertices
        # ManoLayer expects a batch dim
        th_trans = torch.from_numpy(source_trans).float().unsqueeze(0)  # (1, 3)
        th_rot = torch.from_numpy(source_root_orient).float().unsqueeze(0)  # (1, 3)
        th_pose = torch.from_numpy(source_pose.flatten()).float().unsqueeze(0)  # (1, 45)
        # Concat rot and pose -> (1, 48)
        th_pose_coeffs = torch.cat([th_rot, th_pose], dim=1)
        mano_output = self.manolayer(pose_coeffs=th_pose_coeffs)
        source_verts = mano_output.verts + th_trans
        source_joints = mano_output.joints + th_trans
        source_verts = source_verts[0].detach().cpu().numpy()
        source_joints = source_joints[0].detach().cpu().numpy()

        obj_points_path = str(DEFAULT_MESHDATA / "sam3d" / video_id / "obj_points_10000.ply")
        point_cloud = trimesh.load(obj_points_path, process=False)
        point_cloud.apply_scale(obj_scale)
        point_cloud.apply_transform(obj_transform)
        obj_points = point_cloud.vertices
        # Try the standard attribute path
        obj_normals = getattr(point_cloud, 'vertex_normals', None)

        # If missing, fall back to reading normals from PLY metadata
        if obj_normals is None or obj_normals.shape[0] == 0:
            try:
                # Read structured array from raw PLY data
                raw_data = point_cloud.metadata['_ply_raw']['vertex']['data']

                # Stack nx, ny, nz
                # raw_data['nx'] is 1D; stack to (N, 3)
                obj_normals = np.column_stack((raw_data['nx'], raw_data['ny'], raw_data['nz']))

                # Ensure float32
                obj_normals = obj_normals.astype(np.float32)

            except KeyError:
                raise RuntimeError(f"Fatal: Could not find normals in {obj_points_path} metadata. Please checking generation.")

        # Randomly downsample to 4096
        num_points = 4096  # or 2048
        if len(obj_points) >= num_points:
            choice = np.random.choice(len(obj_points), num_points, replace=False)
        else:
            # Guard against meshes with too few points
            choice = np.random.choice(len(obj_points), num_points, replace=True)

        obj_points = obj_points[choice]
        obj_normals = obj_normals[choice]

        normalized_obj_points, obj_centroid, normalized_obj_scale = self._process_pointcloud(obj_points)

        normalized_obj_mesh = trimesh.Trimesh(vertices=(obj_mesh.vertices - obj_centroid) / normalized_obj_scale[0], faces=obj_mesh.faces)
        source_trans = (source_trans - obj_centroid) / normalized_obj_scale[0]

        source_verts = (source_verts - obj_centroid) / normalized_obj_scale
        source_joints = (source_joints - obj_centroid) / normalized_obj_scale

        if self.debug:
            print(f"[DEBUG] Saving debug visualization for sample {mano_path_str}")
            self.save_debug_viz(video_id=f"{video_id}_{frame_id}_{index}",
                                source_params=(source_trans, source_root_orient, source_pose),
                                camera_ray=camera_ray,
                                obj_mesh=normalized_obj_mesh,
                                obj_points=normalized_obj_points,
                                obj_scale=normalized_obj_scale,
                                side=side)

        source_hand = np.concatenate([source_trans, source_root_orient, source_pose.flatten()])

        normalized_obj_points = np.concatenate([normalized_obj_points, obj_normals], axis=1)

        x = []
        noise_depth_mag = []
        cond_x = self._preprocess_mano(source_hand)
        cond_hand = torch.from_numpy(source_joints).float()
        cond_obj_scale = torch.from_numpy(normalized_obj_scale).float()
        cond_obj_pointcloud = torch.from_numpy(normalized_obj_points).float()
        cond_camera_ray = torch.from_numpy(camera_ray).float()

        cond_obj_verts = normalized_obj_mesh.vertices
        cond_obj_faces = normalized_obj_mesh.faces

        input_dict = {
            'x': x,
            'noise_depth_mag': noise_depth_mag,
            'cond_x': cond_x,
            'cond_hand': cond_hand,
            'cond_obj_scale': cond_obj_scale,
            'cond_obj_pointcloud': cond_obj_pointcloud,
            'cond_camera_ray': cond_camera_ray,
            'cond_obj_verts': cond_obj_verts,
            'cond_obj_faces': cond_obj_faces,
            'video_id': video_id,
            'frame_id': int(frame_id),
        }

        return input_dict


if __name__ == '__main__':
    # Fix seed for reproducible debug dumps
    np.random.seed(42)
    torch.manual_seed(42)

    print("=== start to load Dataset ===")
    # debug=True makes __getitem__ dump OBJ/PLY files
    dataset = GraspPair(data_split='train', debug=True)
    # dataset = GraspTest(debug=True)
    print(f"Dataset loaded, total samples: {len(dataset)}")

    if len(dataset) > 0:
        # --- Test 3: trigger visualization dumps ---
        print("\n=== try to trigger Debug visualization save ===")
        # Outputs under ./debug_viz/
        import tqdm
        print("fastly traverse 1 samples to trigger debug save...")

        count = 0
        for i in tqdm.tqdm(range(min(10000, len(dataset)))):
            if i % 500 == 0:
                _ = dataset[i]

        # Check that files were written
        viz_dir = Path("debug_viz")
        if viz_dir.exists():
            files = list(viz_dir.glob("*.ply"))
            if len(files) > 0:
                print(f"\n[Success] {len(files)} visualization files generated in {viz_dir.absolute()}")
                print(f"Please use MeshLab or Blender to open {files[0]} to view:")
                print("  - green: GT Hand")
                print("  - red: Noisy Input Hand")
                print("  - gray: Canonical Object")
            else:
                print("\n[Warning] directory created but no files generated, maybe not lucky enough or path problem.")
        else:
            print("\n[Error] debug_viz directory not created, please check dataset code permission.")

    else:
        print("\n[Error] no data found, please check data_root path is correct.")

            # # 106 1555 4161 5120 5573 6206 6784 7785 10573 11407 15603 16053 25606 26304 26784 29010 35519 36362 37522 37941 42062 53436 53817 53847 54362 80413 80772 81827 87574 93025
            # selected_video_ids = [
            #     # '106',
            #     # '1555',
            #     # '4161',
            #     # '5120',
            #     # '5573',
            #     # '6206',
            #     '6784',
            #     # '7785',
            #     # '10573',
            #     # '11407',
            #     '15603',
            #     '16053',
            #     # '25606',
            #     # '26304',
            #     # '26784',
            #     # '29010',
            #     '35519',
            #     # '36362',
            #     # '37522',
            #     '37941',
            #     '42062',
            #     # '53436',
            #     '53817',
            #     # '53847',
            #     '54362',
            #     '80413',
            #     # '80772',
            #     '81827',
            #     '87574',
            #     '93025',
            # ]
            # # selected_video_ids = ['hold_ABF12_ho3d']
            # # selected_video_ids = ['hold_ABF14_ho3d']
            # # selected_video_ids = ['hold_GPMF12_ho3d']
            # selected_video_ids += [
            #     'hold_ABF12_ho3d',
            #     'hold_ABF14_ho3d',
            #     'hold_GPMF12_ho3d',
            #     'hold_GPMF14_ho3d',
            #     'hold_MC1_ho3d',
            #     'hold_MC4_ho3d',
            #     'hold_MDF12_ho3d',
            #     'hold_MDF14_ho3d',
            #     'hold_ShSu10_ho3d',
            #     'hold_ShSu12_ho3d',
            #     'hold_SM2_ho3d',
            #     'hold_SM4_ho3d',
            #     'hold_SMu1_ho3d',
            #     'hold_SMu40_ho3d',
            # ]
            # selected_video_ids = ['96468']