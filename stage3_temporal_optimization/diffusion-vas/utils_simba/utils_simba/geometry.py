import numpy as np
import math
import torch
from plyfile import PlyData, PlyElement
import trimesh
from PIL import Image

import os

def fast_depthmap_to_pts3d(depth, pixel_grid, focal, pp):
    """
    Converts a depth map to 3D points in camera coordinates.

    Args:
        depth (torch.Tensor): Depth map with shape (N, P).
        pixel_grid (torch.Tensor): Pixel grid coordinates with shape (N, P, 2).
        focal (torch.Tensor): Focal length with shape (N, 1).
        pp (torch.Tensor): Principal point with shape (N, 2).

    Returns:
        torch.Tensor: 3D points in camera coordinates with shape (N, P, 3).
    """
    pp = pp.unsqueeze(1)
    focal = focal.unsqueeze(1)
    assert focal.shape == (len(depth), 1, 1)
    assert pp.shape == (len(depth), 1, 2)
    assert pixel_grid.shape == depth.shape + (2,)
    depth = depth.unsqueeze(-1)

    pts3d = torch.cat((depth * (pixel_grid - pp) / focal, depth), dim=-1)
    mask = depth > 0
    valid_pts3d = pts3d[mask.expand_as(pts3d)].view(pts3d.shape[0], -1, 3)

    return valid_pts3d

def xy_grid(W, H, device=None, origin=(0, 0), unsqueeze=None, cat_dim=-1, homogeneous=False, **arange_kw):
    """ Output a (H,W,2) array of int32 
        with output[j,i,0] = i + origin[0]
             output[j,i,1] = j + origin[1]
    """
    if device is None:
        # numpy
        arange, meshgrid, stack, ones = np.arange, np.meshgrid, np.stack, np.ones
    else:
        # torch
        arange = lambda *a, **kw: torch.arange(*a, device=device, **kw)
        meshgrid, stack = torch.meshgrid, torch.stack
        ones = lambda *a: torch.ones(*a, device=device)

    tw, th = [arange(o, o + s, **arange_kw) for s, o in zip((W, H), origin)]
    grid = meshgrid(tw, th, indexing='xy')
    if homogeneous:
        grid = grid + (ones((H, W)),)
    if unsqueeze is not None:
        grid = (grid[0].unsqueeze(unsqueeze), grid[1].unsqueeze(unsqueeze))
    if cat_dim is not None:
        grid = stack(grid, cat_dim)
    return grid

def geotrf(Trf, pts, ncol=None, norm=False):
    """ Apply a geometric transformation to a list of 3-D points.

    H: 3x3 or 4x4 projection matrix (typically a Homography)
    p: numpy/torch/tuple of coordinates. Shape must be (...,2) or (...,3)

    ncol: int. number of columns of the result (2 or 3)
    norm: float. if != 0, the resut is projected on the z=norm plane.

    Returns an array of projected 2d points.
    """
    assert Trf.ndim >= 2
    if isinstance(Trf, np.ndarray):
        pts = np.asarray(pts)
    elif isinstance(Trf, torch.Tensor):
        pts = torch.as_tensor(pts, dtype=Trf.dtype)

    # adapt shape if necessary
    output_reshape = pts.shape[:-1]
    ncol = ncol or pts.shape[-1]

    # optimized code
    if (isinstance(Trf, torch.Tensor) and isinstance(pts, torch.Tensor) and
            Trf.ndim == 3 and pts.ndim == 4):
        d = pts.shape[3]
        if Trf.shape[-1] == d:
            pts = torch.einsum("bij, bhwj -> bhwi", Trf, pts)
        elif Trf.shape[-1] == d + 1:
            pts = torch.einsum("bij, bhwj -> bhwi", Trf[:, :d, :d], pts) + Trf[:, None, None, :d, d]
        else:
            raise ValueError(f'bad shape, not ending with 3 or 4, for {pts.shape=}')
    else:
        if Trf.ndim >= 3:
            n = Trf.ndim - 2
            assert Trf.shape[:n] == pts.shape[:n], 'batch size does not match'
            Trf = Trf.reshape(-1, Trf.shape[-2], Trf.shape[-1])

            if pts.ndim > Trf.ndim:
                # Trf == (B,d,d) & pts == (B,H,W,d) --> (B, H*W, d)
                pts = pts.reshape(Trf.shape[0], -1, pts.shape[-1])
            elif pts.ndim == 2:
                # Trf == (B,d,d) & pts == (B,d) --> (B, 1, d)
                pts = pts[:, None, :]

        if pts.shape[-1] + 1 == Trf.shape[-1]:
            Trf = Trf.swapaxes(-1, -2)  # transpose Trf
            pts = pts @ Trf[..., :-1, :] + Trf[..., -1:, :]
        elif pts.shape[-1] == Trf.shape[-1]:
            Trf = Trf.swapaxes(-1, -2)  # transpose Trf
            pts = pts @ Trf
        else:
            pts = Trf @ pts.T
            if pts.ndim >= 2:
                pts = pts.swapaxes(-1, -2)

    if norm:
        pts = pts / pts[..., -1:]  # DONT DO /= BECAUSE OF WEIRD PYTORCH BUG
        if norm != 1:
            pts *= norm

    res = pts[..., :ncol].reshape(*output_reshape, ncol)
    return res

def depth_to_pts3d(focals, principal_points, c2w4x4, depth):
    """
    Convert depth map to 3D points in world frame.

    Args:
        focals (tensor): Focal lengths of the camera (N, 1).
        principal_points (tensor): Principal points of the camera (N, 2) with (cx, cy).
        c2w4x4 (tensor): Camera-to-world transformation matrix (N, 4, 4).
        depth (tensor): Depth map (N, H, W).

    Returns:
        tensor: 3D points in world frame with shape of (N, H, W, 3).
    """

    # Get grid
    grid = xy_grid(depth.shape[-1], depth.shape[-2], device=depth.device, unsqueeze=0, cat_dim=-1)
    grid = grid.view(grid.shape[0], -1, 2)
    depth = depth.view(grid.shape[0], -1)
    # Get pointmaps in camera frame
    ptmaps_in_c = fast_depthmap_to_pts3d(depth, grid, focals, pp=principal_points) # (N, P, 3)
    # Project to world frame
    return geotrf(c2w4x4, ptmaps_in_c) # (N, P, 3)

def depth_to_pts3d_without_transform(depth, K):
    """
    Convert depth map to 3D points in world frame.

    Args:
        depth (tensor): Depth map (N, H, W).
        K (tensor): Intrinsic matrix of the camera (N, 3, 3).

    Returns:
        tensor: 3D points in world frame with shape of (N, H, W, 3).
    """
    focals = K[:, 0, 0].unsqueeze(1)  # (N, 1)
    principal_points = torch.stack((K[:, 0, 2], K[:, 1, 2]), dim=-1)  # (N, 2)
    c2w4x4 = torch.eye(4, device=depth.device).unsqueeze(0)  # (N, 4, 4)
    pts_3D = depth_to_pts3d(focals, principal_points, c2w4x4, depth)
    return pts_3D



def save_point_cloud_to_ply(pts_3d, filepath, colors=None):
    # Create a structured array for the points
    # check if pts_3d is a numpy array
    if isinstance(pts_3d, torch.Tensor):
        pts_3d = pts_3d.cpu().numpy()

    if colors is not None:
        colors = colors.astype(np.uint8)
        dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

        points_array = np.array(
            [(*point, *color) for point, color in zip(pts_3d, colors)],
            dtype=dtype
        )                

    else:
        dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
        points_array = np.array([(point[0], point[1], point[2]) for point in pts_3d],
                            dtype=dtype)

    ply_element = PlyElement.describe(points_array, 'vertex')
    # Write to a PLY file
    PlyData([ply_element]).write(filepath)
    # print(f"Point cloud saved to {filepath}")

# Function to read PLY file and extract point clouds
def read_point_cloud_from_ply(filepath):
    # Read the PLY file
    ply_data = PlyData.read(filepath)

    # Extract the vertex data (points) from the PLY file
    vertex_data = ply_data['vertex']
    try:
        red = np.array(vertex_data['red'])
        green = np.array(vertex_data['green'])
        blue = np.array(vertex_data['blue'])
        colors = np.vstack((red, green, blue)).T
    except:
        colors = None

    # Convert the data into a NumPy array
    positions = np.vstack([vertex_data['x'], vertex_data['y'], vertex_data['z']]).T

    return positions, colors    

# Function to read PLY file and extract point clouds
def read_point_cloud_from_obj(obj_file, texture_file=None):
    """
    Transforms an OBJ mesh to a point cloud with colors.

    Parameters:
        obj_file (str): Path to the OBJ file.
        texture_file (str, optional): Path to the texture image. If None, assigns default colors.

    Returns:
        tuple: (positions, colors)
            - positions (np.ndarray): Vertex positions with shape (N, 3).
            - colors (np.ndarray): Vertex colors with shape (N, 3).
    """
    # Step 1: Load the OBJ model using trimesh
    mesh = trimesh.load(obj_file, force='mesh')

    # Step 2: Get vertex positions and face indices
    positions = mesh.vertices            # numpy array of shape (n_vertices, 3)
    faces = mesh.faces                   # numpy array of shape (n_faces, 3)

    if texture_file is not None and os.path.exists(texture_file):
        # Step 3: Load the texture image using PIL
        texture = Image.open(texture_file).convert('RGB')  # Ensure it's in RGB format
        texture_data = np.asarray(texture)    # Convert image to numpy array
        texture_width, texture_height = texture.size

        # Step 4: Get UV coordinates from the mesh
        if mesh.visual.uv is None:
            print(f"No UV coordinates found for {obj_file}. Assigning default color.")
            colors = np.ones((positions.shape[0], 3), dtype=np.float32)  # White color
        else:
            uv_coords = mesh.visual.uv            # numpy array of shape (n_vertices, 2)

            # Ensure UV coordinates are within [0, 1]
            uv_coords = np.clip(uv_coords, 0, 1)

            # Step 5: Map UV coordinates to texture pixel indices
            u_indices = (uv_coords[:, 0] * (texture_width - 1)).astype(np.int32)
            v_indices = ((1 - uv_coords[:, 1]) * (texture_height - 1)).astype(np.int32)  # Invert V-axis

            # Step 6: Get colors from the texture image
            colors = texture_data[v_indices, u_indices] / 255.0  # Normalize RGB values to [0, 1]
    else:
        # Assign default color, e.g., white
        colors = np.ones((positions.shape[0], 3), dtype=np.float32)

    return positions, colors

def convert_point_cloud(pc_f, pc_normalized_f, transformation_matrix):
    import open3d as o3d
    pcd = o3d.io.read_point_cloud(pc_f)
    if not pcd.has_points():
        raise ValueError(f"The point cloud at {pc_f} has no points.")
    
    print(f"Loaded point cloud with {len(pcd.points)} points.")

    # Check if point cloud has colors
    has_colors = pcd.has_colors()
    if has_colors:
        print("Point cloud has color information.")
    else:
        print("Point cloud does NOT have color information.")

    # Convert point cloud to numpy array for processing
    points = np.asarray(pcd.points)
    points_normalized = points @ transformation_matrix[:3, :3].T + transformation_matrix[:3, 3]
    pcd.points = o3d.utility.Vector3dVector(points_normalized)

    # If colors exist, ensure they are preserved
    if has_colors:
        colors = np.asarray(pcd.colors)
        # Normalize colors if they are in [0, 255]
        if colors.max() > 1.0:
            colors = colors / 255.0
            print("Normalized color values to [0, 1].")
        pcd.colors = o3d.utility.Vector3dVector(colors)
        print("Preserved color information in the transformed point cloud.")    

    success = o3d.io.write_point_cloud(pc_normalized_f, pcd, write_ascii=True)
    if success:
        print(f"Transformed point cloud saved to {pc_normalized_f}")
    else:
        raise IOError(f"Failed to save the transformed point cloud to {pc_normalized_f}")    

def get_revert_tvec(tvec, quat_xyzw):
    from scipy.spatial.transform import Rotation as R
    rotation = R.from_quat(quat_xyzw)
    rot_matrix = rotation.as_matrix()
    inverse_rot_matrix = rot_matrix.T
    reverse_tvec = -inverse_rot_matrix @ tvec
    return reverse_tvec

def transform_points(pts_c1, c1Toc2_all):
    """
    Transforms 3D points from coordinate system 1 to coordinate system 2.

    Parameters:
        pts_c1 (np.ndarray or torch.Tensor): Points in coordinate system 1 with shape (batch_size, num_points, 3).
        c1Toc2_all (np.ndarray or torch.Tensor): Transformation matrices from coordinate system 1 to coordinate system 2 with shape (batch_size, 4, 4).

    Returns:
        np.ndarray or torch.Tensor: Transformed points in coordinate system 2 with shape (batch_size, num_points, 3).
                                   The type matches the input type.
    """
    # Determine the module and functions to use based on the input type
    if isinstance(pts_c1, torch.Tensor):
        module = torch
        cat_func = torch.cat
        expand_dims = torch.unsqueeze
        squeeze = torch.squeeze
        newaxis = None
        device = pts_c1.device
        ones_kwargs = {'device': device}
    elif isinstance(pts_c1, np.ndarray):
        module = np
        cat_func = np.concatenate
        expand_dims = np.expand_dims
        squeeze = np.squeeze
        newaxis = np.newaxis
        ones_kwargs = {}
    else:
        raise TypeError("Input must be either a numpy array or a torch tensor.")

    # Convert to homogeneous coordinates
    ones_shape = list(pts_c1.shape[:-1]) + [1]  # Shape: (batch_size, num_points, 1)
    ones = module.ones(ones_shape, dtype=pts_c1.dtype, **ones_kwargs)
    pts_c1_homo = cat_func([pts_c1, ones], axis=-1)  # Shape: (batch_size, num_points, 4)

    # Expand dimensions for matrix multiplication
    pts_c1_homo_expanded = expand_dims(pts_c1_homo, axis=-1)  # Shape: (batch_size, num_points, 4, 1)

    # Expand transformation matrices for batch matrix multiplication
    if module == np:
        c1Toc2_expanded = c1Toc2_all[:, np.newaxis, :, :]  # Shape: (batch_size, 1, 4, 4)
    else:
        c1Toc2_expanded = c1Toc2_all.unsqueeze(1)  # Shape: (batch_size, 1, 4, 4)

    # Perform batch matrix multiplication
    pts_c2_homo = module.matmul(c1Toc2_expanded, pts_c1_homo_expanded)  # Shape: (batch_size, num_points, 4, 1)

    # Squeeze the last dimension and extract Cartesian coordinates
    pts_c2 = squeeze(pts_c2_homo, axis=-1)[..., :3]  # Shape: (batch_size, num_points, 3)

    return pts_c2

def project_points(pts_c, intrinsic):
    """
    Projects 3D points to 2D image coordinates using camera intrinsics.

    Parameters:
        pts_c (np.ndarray or torch.Tensor): 3D points in camera coordinates with shape (batch_size, N, 3)
        intrinsic (np.ndarray or torch.Tensor): Camera intrinsic matrix with shape (batch_size, 3, 3)

    Returns:
        np.ndarray or torch.Tensor: Projected 2D points in image coordinates with shape (batch_size, N, 3).
                                   The first two columns are x, y pixel coordinates,
                                   the third column contains the depth values.
    """

    # Check if pts_c and intrinsic are numpy arrays or torch tensors
    if isinstance(pts_c, np.ndarray):
        matmul_fn = np.matmul
        is_numpy = True
    elif isinstance(pts_c, torch.Tensor):
        matmul_fn = torch.matmul
        is_numpy = False
    else:
        raise TypeError("pts_c should be either a numpy array or a torch tensor.")

    # Project the 3D points (multiplying with the intrinsic matrix)
    pts_2d = matmul_fn(intrinsic, pts_c.transpose(0, 2, 1))  # (batch_size, 3, N)

    # Divide by depth (the third row of the resulting matrix corresponds to the z coordinate)
    pts_2d /= pts_2d[:, 2:3, :]  # Broadcasting to divide by the depth

    # Reorganize to have the shape (batch_size, N, 3), where the last column is depth
    projected_points = np.concatenate([pts_2d[:, :2, :].transpose(0, 2, 1), pts_c[:, :, 2:3]], axis=-1) if is_numpy else torch.cat([pts_2d[:, :2, :].transpose(1, 2), pts_c[:, :, 2:3]], dim=-1)

    return projected_points


def rodrigues_to_rotation_matrix(rvec):
    """
    Convert Rodrigues vector (angle-axis) to a 3x3 rotation matrix.
    """
    theta = np.linalg.norm(rvec)
    if theta < 1e-12:
        # No rotation
        return np.eye(3)
    
    k = rvec / theta
    K = np.array([[    0, -k[2],  k[1]],
                  [ k[2],     0, -k[0]],
                  [-k[1],  k[0],    0]], dtype=float)
    R = np.eye(3) + np.sin(theta)*K + (1-np.cos(theta))*(K @ K)
    return R

def rotation_matrix_to_rodrigues(R):
    """
    Convert a 3x3 rotation matrix to a Rodrigues (angle-axis) vector (3,).
    """
    # Ensure R is 3x3
    R = R[:3, :3]

    # Compute the angle from the trace
    # Numerical safety for trace precision
    eps = 1e-12
    trace_val = np.trace(R)
    # Clip the value in case of slight numerical overshoot
    trace_val = max(min(trace_val, 3.0), -1.0)
    
    theta = np.arccos((trace_val - 1.0) / 2.0)

    # If theta is very close to 0 => no rotation
    if abs(theta) < eps:
        return np.zeros(3)

    # If theta is close to pi or for certain numeric cases, the standard formula
    # still works but must be handled carefully with numerical issues.

    # Compute axis
    rx = (R[2, 1] - R[1, 2]) / (2.0 * np.sin(theta))
    ry = (R[0, 2] - R[2, 0]) / (2.0 * np.sin(theta))
    rz = (R[1, 0] - R[0, 1]) / (2.0 * np.sin(theta))
    r_axis = np.array([rx, ry, rz], dtype=float)

    # Construct Rodrigues vector = angle * unit_axis
    rvec = theta * r_axis
    return rvec

def transform4x4_to_rodrigues_and_translation(T):
    """
    Given a 4x4 homogeneous transformation matrix, 
    return:
      - rvec: 3D Rodrigues rotation vector
      - tvec: 3D translation vector
    """
    # Extract rotation (3x3) and translation (3x1)
    R = T[:3, :3]
    t = T[:3, 3]

    # Convert rotation matrix to Rodrigues vector
    rvec = rotation_matrix_to_rodrigues(R)

    return rvec, t

def rodrigues_and_translation_to_transform4x4(rvec, tvec):
    """
    Convert Rodrigues vector and translation vector to a 4x4 transformation matrix.
    Parameters:
        - rvec: 3D Rodrigues rotation vector
        - tvec: 3D translation vector
    Returns:
        - T: 4x4 transformation matrix
    """
    R = rodrigues_to_rotation_matrix(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec
    return T

def save_mesh(vertices, faces, out_f):
    if torch.is_tensor(vertices):
        vertices = vertices.detach().cpu().numpy()
    if torch.is_tensor(faces):
        faces = faces.detach().cpu().numpy()

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)    

    mesh.export(out_f)
    print(f"save mesh in {out_f}")

def save_point_cloud(x_c, filename, colors=None):
    """
    Saves a point cloud to a PLY file using trimesh.

    Parameters:
        x_c (np.ndarray): Point cloud data with shape (N, 3).
        filename (str): Output PLY file path.
        colors (optional, np.ndarray): Color data with shape (N, 3), values in [0, 255].
    """
    if not isinstance(x_c, np.ndarray):
        raise TypeError("x_c must be a NumPy array.")
    if x_c.shape[1] != 3:
        raise ValueError("x_c must have shape (N, 3).")
    if colors is not None:
        if not isinstance(colors, np.ndarray):
            raise TypeError("colors must be a NumPy array.")
        if colors.shape[1] != 3:
            raise ValueError("colors must have shape (N, 3).")
        if colors.shape[0] != x_c.shape[0]:
            raise ValueError("colors and x_c must have the same number of points.")
    
    # Create trimesh PointCloud object
    cloud = trimesh.points.PointCloud(vertices=x_c, colors=colors)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    # Export to PLY
    cloud.export(filename)
    print(f"Point cloud saved to {filename}")    

def get_incident_angle(normal, light_dir):
    """
    Computes the incident angle between the surface normal and the light direction.

    Parameters:
        normal (np.ndarray): Surface normal with shape (3,).
        light_dir (np.ndarray): Light direction with shape (3,).

    Returns:
        float: Incident angle in degrees.
    """
    # Normalize the vectors
    normal = normal / np.linalg.norm(normal)
    light_dir = light_dir / np.linalg.norm(light_dir)

    # Compute the cosine of the angle
    cos_theta = np.dot(normal, light_dir)

    # Clamp the value to avoid numerical errors
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    # Compute the angle in degrees
    angle_rad = np.arccos(cos_theta)
    angle_deg = np.degrees(angle_rad)

    return angle_rad, angle_deg

def get_intrinsic_matrix_from_projection(P, w, h):
    '''
    obtain the intrinsic matrix from the projection matrix
    args:
        P: projection matrix (4x4)
        w: image width
        h: image height
    return:
        K: intrinsic matrix (3x3)
    '''
    fx = P[0][0] * w / 2
    fy = P[1][1] * h / 2
    cx = (1.0 - P[0][2]) * w / 2
    cy = (1.0 + P[1][2]) * h / 2
    
    K = torch.eye(3, dtype=torch.float32)
    K[0,0] = fx
    K[1,1] = fy
    K[0,2] = cx 
    K[1,2] = cy
    
    return K

def matrix_to_axis_angle_t(mat: torch.Tensor):
    from pytorch3d.transforms.rotation_conversions import matrix_to_axis_angle
    r, t, s = homo_to_rt(mat)
    return matrix_to_axis_angle(r), t, s

def homo_to_rt(mat):
    """
    :param (N, 4, 4) [R, t; 0, 1]
    :return: rot: (N, 3, 3), t: (N, 3), s: (N, 1)
    """
    mat, _ = torch.split(mat, [3, mat.size(-2) - 3], dim=-2)
    rot_scale, trans = torch.split(mat, [3, 1], dim=-1)
    rot, scale = mat_to_scale_rot(rot_scale)

    trans = trans.squeeze(-1)
    return rot, trans, scale

def mat_to_scale_rot(mat):
    """s*R to s, R
    
    Args:
        mat ( ): (..., 3, 3)
    Returns:
        scale: (..., 3)
        rot: (..., 3, 3)
    """
    sq = torch.matmul(mat, mat.transpose(-1, -2))
    scale_flat = torch.sqrt(torch.diagonal(sq, dim1=-1, dim2=-2))  # (..., 3)
    scale_inv = scale_matrix(1/scale_flat, homo=False)
    rot = torch.matmul(mat, scale_inv)
    return rot, scale_flat

def scale_matrix(scale, homo=True):
    """
    :param scale: (..., 3)
    :return: scale matrix (..., 4, 4)
    """
    dims = scale.size()[0:-1]
    one_dims = [1,] * len(dims)
    device = scale.device
    if scale.size(-1) == 1:
        scale = scale.expand(*dims, 3)
    mat = torch.diag_embed(scale, dim1=-2, dim2=-1)
    if homo:
        mat = rt_to_homo(mat)
    return mat

def rt_to_homo(rot, t=None, s=None):
    """
    :param rot: (..., 3, 3)
    :param t: (..., 3 ,(1))
    :param s: (..., 1)
    :return: (N, 4, 4) [R, t; 0, 1] sRX + t
    """
    rest_dim = list(rot.size())[:-2]
    if t is None:
        t = torch.zeros(rest_dim + [3]).to(rot)
    if t.size(-1) != 1:
        t = t.unsqueeze(-1)  # ..., 3, 1
    mat = torch.cat([rot, t], dim=-1)
    zeros = torch.zeros(rest_dim + [1, 4], device=t.device)
    zeros[..., -1] += 1
    mat = torch.cat([mat, zeros], dim=-2)
    if s is not None:
        s = scale_matrix(s)
        mat = torch.matmul(mat, s)

    return mat

def axis_angle_t_to_matrix(axisang=None, t=None, s=None, homo=True):
    """
    :param axisang: (N, 3)
    :param t: (N, 3)
    :return: (N, 4, 4)
    """
    import pytorch3d.transforms.rotation_conversions as rot_cvt    
    if axisang is None:
        axisang = torch.zeros_like(t)
    if t is None:
        t = torch.zeros_like(axisang)
    rot = rot_cvt.axis_angle_to_matrix(axisang)
    if homo:
        return rt_to_homo(rot, t, s)
    else:
        return rot