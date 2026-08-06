import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import torch
from pytorch3d.ops.knn import knn_points
from pytorch3d.renderer import TexturesVertex
from pytorch3d.structures import Meshes
from scipy.ndimage import binary_dilation


def set_seed(seed):
    random.seed(seed)  # Python's built-in random module
    np.random.seed(seed)  # NumPy's random module
    torch.manual_seed(seed)  # PyTorch

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def simplify_mesh(mesh, target_triangles=5000):
    vertices = mesh.verts_list()[0].cpu().numpy()
    faces = mesh.faces_list()[0].cpu().numpy()
    mesh_o3d = o3d.geometry.TriangleMesh()
    mesh_o3d.vertices = o3d.utility.Vector3dVector(vertices)
    mesh_o3d.triangles = o3d.utility.Vector3iVector(faces)
    mesh_o3d.remove_duplicated_vertices()
    mesh_o3d.remove_degenerate_triangles()
    mesh_o3d.remove_duplicated_triangles()
    mesh_o3d.remove_non_manifold_edges()

    if len(mesh_o3d.triangles) > target_triangles:
        mesh_simplified = mesh_o3d.simplify_quadric_decimation(target_triangles)
    else:
        mesh_simplified = mesh_o3d
    device = mesh.device
    verts = torch.tensor(np.asarray(mesh_simplified.vertices), dtype=torch.float32, device=device)
    faces = torch.tensor(np.asarray(mesh_simplified.triangles), dtype=torch.int64, device=device)
    textures = TexturesVertex(verts_features=torch.ones_like(verts)[None])  # (1, V, 3)
    mesh_simplified = Meshes(verts=[verts], faces=[faces], textures=textures)

    return mesh_simplified


def init_relative_translation(
        init_pose_T,  # (3,) GT pose of the init frame
        masks,  # (T, H, W) amodal mask sequence
        fx,
        fy,  # (2,) intrinsics
        init_frame_idx=0  # init frame index (default 0)
):
    """
    Initialize translations from relative mask motion.
    
    Args:
        init_pose_T: GT pose of the init frame (3,)
        masks: amodal mask sequence (T, H, W)
        fx, fy: camera intrinsics
        init_frame_idx: init frame index (default 0)
                       - if 0: propagate forward from frame 0
                       - if nonzero: propagate both forward and backward from that frame
    
    Returns:
        initial_translations: (T, 3) translation sequence from frame 0 to the end
    """
    num_frames = len(masks)

    # Validate init frame index
    if init_frame_idx < 0 or init_frame_idx >= num_frames:
        raise ValueError(f"init_frame_idx={init_frame_idx} out of range [0, {num_frames-1}]")

    # Ensure the init-frame mask is valid
    x_init, y_init, w_init, h_init = cv2.boundingRect(masks[init_frame_idx].astype(np.uint8))
    if w_init == 0 or h_init == 0:
        raise ValueError(f"init frame {init_frame_idx} mask is empty!")

    # 1. Extract reference info from the init frame
    T_init = init_pose_T  # [X_init, Y_init, Z_init]
    Z_init = T_init[2]
    u_init = x_init + w_init / 2.0
    v_init = y_init + h_init / 2.0

    # 2. Allocate the result array
    initial_translations = np.zeros((num_frames, 3), dtype=np.float32)
    initial_translations[init_frame_idx] = T_init

    print(f"Initializing via Relative Motion (XY + Z-Scaling) from frame {init_frame_idx}...")

    def compute_translation_from_reference(ref_T, ref_u, ref_v, ref_Z, target_mask):
        """Compute target-frame translation from a reference frame."""
        x, y, w, h = cv2.boundingRect(target_mask.astype(np.uint8))

        if w == 0 or h == 0:
            # Empty mask -> None
            return None

        ut = x + w / 2.0
        vt = y + h / 2.0

        # Use reference-frame depth
        Z_new = ref_Z

        # 2D displacement
        delta_u = ut - ref_u
        delta_v = vt - ref_v

        # Map 2D displacement to 3D (back-projection)
        delta_X = -1 * (delta_u * Z_new / fx)
        delta_Y = -1 * (delta_v * Z_new / fy)

        # Add onto the reference pose
        X_new = ref_T[0] + delta_X
        Y_new = ref_T[1] + delta_Y

        return np.array([X_new, Y_new, Z_new], dtype=np.float32)

    # 3. Forward propagation: init_frame_idx -> last frame
    for t in range(init_frame_idx + 1, num_frames):
        result = compute_translation_from_reference(T_init, u_init, v_init, Z_init, masks[t])

        if result is not None:
            initial_translations[t] = result
        else:
            # Empty mask: reuse previous frame
            initial_translations[t] = initial_translations[t - 1]

    # 4. Backward propagation: init_frame_idx -> frame 0
    for t in range(init_frame_idx - 1, -1, -1):
        result = compute_translation_from_reference(T_init, u_init, v_init, Z_init, masks[t])

        if result is not None:
            initial_translations[t] = result
        else:
            # Empty mask: reuse next frame
            initial_translations[t] = initial_translations[t + 1]

    return initial_translations  # (T, 3)


def overlay_mask_on_image(rgb_img, mask, cmap_idx=None, random_color=False, boundary_thickness=3, darken_factor=2):
    # Ensure the input image is RGB and in the range [0, 1]
    assert rgb_img.shape[-1] == 3, "Expected RGB image with 3 channels"

    # Select a color for the mask overlay
    cmap = plt.get_cmap("tab10")

    cmap_idx = 4
    if cmap_idx is None and random_color:
        cmap_idx = np.random.randint(0, cmap.N)  # Randomly choose a colormap index if not provided

    color = np.array([*cmap(cmap_idx)[:3], 0.6])
    boundary_color = color[:3] * darken_factor  # Darken the color by the darken_factor
    boundary_color = np.concatenate([boundary_color, [1.0]])  # Make boundary fully opaque

    # Create a boundary mask
    dilated_mask = binary_dilation(mask, iterations=boundary_thickness)
    boundary_mask = dilated_mask & ~mask

    # Create a colored mask in the range [0, 1]
    mask_image = np.zeros_like(rgb_img, dtype=np.float32)
    boundary_image = np.zeros_like(rgb_img, dtype=np.float32)

    for i in range(3):  # Apply the mask and boundary to each channel
        mask_image[..., i] = mask * color[i]
        boundary_image[..., i] = boundary_mask * boundary_color[i]

    # Combine the RGB image with the colored mask and boundary
    overlayed_image = np.clip(rgb_img * 0.5 + mask_image + boundary_image, 0, 1)

    return overlayed_image


def overlay_rgb_render(background_img, render_img, alpha=1.0):
    # Ensure inputs are floats in [0, 1]
    bg = background_img.astype(np.float32)
    fg = render_img.astype(np.float32)

    mask = (fg.sum(axis=-1) > 0.0).astype(np.float32)
    mask = mask[..., None]
    composited = bg * (1 - mask * alpha) + fg * alpha

    return np.clip(composited, 0, 1)


def overlay_points_on_image(image_np, points_np, color, radius=3):
    """
    Draws 2D points on an image.
    Args:
        image_np: (H, W, 3) float numpy array [0, 1]
        points_np: (N, 2) int or float numpy array
        color: tuple of floats (R, G, B) for color, values in [0, 1]
    Returns:
        (H, W, 3) float numpy array [0, 1] with points overlaid.
    """
    import cv2
    image_out = image_np.copy()
    for i in range(points_np.shape[0]):
        center = tuple(points_np[i].astype(int))
        cv2.circle(image_out, center, radius, color, -1)
    return image_out


def calculate_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 1.0
    return intersection / union


def farthest_point_sampling_torch(points, n_samples):
    """
    Selects a subset of points using the farthest point sampling algorithm with PyTorch.
    points: (N, D) tensor
    n_samples: int
    """
    device = points.device
    N, D = points.shape

    if N == 0:
        return torch.zeros((0, D), dtype=torch.float, device=device)

    sampled_indices = torch.zeros(n_samples, dtype=torch.long, device=device)

    # Randomly select the first point
    start_idx = torch.randint(0, N, (1,), device=device)[0]
    sampled_indices[0] = start_idx

    dists = torch.sum((points - points[start_idx])**2, dim=1)

    for i in range(1, n_samples):
        farthest_idx = torch.argmax(dists)
        sampled_indices[i] = farthest_idx

        new_dists = torch.sum((points - points[farthest_idx])**2, dim=1)
        dists = torch.minimum(dists, new_dists)

    return points[sampled_indices]


def get_NN(src_xyz, trg_xyz, k=1):
    '''
    :param src_xyz: [B, N1, 3]
    :param trg_xyz: [B, N2, 3]
    :return: nn_dists, nn_dix: all [B, 3000] tensor for NN distance and index in N2
    '''
    B = src_xyz.size(0)
    src_lengths = torch.full((src_xyz.shape[0],), src_xyz.shape[1], dtype=torch.int64, device=src_xyz.device)  # [B], N for each num
    trg_lengths = torch.full((trg_xyz.shape[0],), trg_xyz.shape[1], dtype=torch.int64, device=trg_xyz.device)
    src_nn = knn_points(src_xyz, trg_xyz, lengths1=src_lengths, lengths2=trg_lengths, K=k)  # [dists, idx]
    nn_dists = src_nn.dists[..., 0]
    nn_idx = src_nn.idx[..., 0]
    return nn_dists, nn_idx


def get_interior(src_face_normal, src_xyz, trg_xyz, trg_NN_idx):
    '''
    :param src_face_normal: [B, 778, 3], surface normal of every vert in the source mesh
    :param src_xyz: [B, 778, 3], source mesh vertices xyz
    :param trg_xyz: [B, 3000, 3], target mesh vertices xyz
    :param trg_NN_idx: [B, 3000], index of NN in source vertices from target vertices
    :return: interior [B, 3000], inter-penetrated trg vertices as 1, instead 0 (bool)
    '''
    N1, N2 = src_xyz.size(1), trg_xyz.size(1)

    # get vector from trg xyz to NN in src, should be a [B, 3000, 3] vector
    NN_src_xyz = batched_index_select(src_xyz, trg_NN_idx)  # [B, 3000, 3]
    NN_vector = NN_src_xyz - trg_xyz  # [B, 3000, 3]

    # get surface normal of NN src xyz for every trg xyz, should be a [B, 3000, 3] vector
    NN_src_normal = batched_index_select(src_face_normal, trg_NN_idx)

    interior = (NN_vector * NN_src_normal).sum(dim=-1) > 0  # interior as true, exterior as false
    return interior


def batched_index_select(input, index, dim=1):
    '''
    :param input: [B, N1, *]
    :param dim: the dim to be selected
    :param index: [B, N2]
    :return: [B, N2, *] selected result
    '''
    views = [input.size(0)] + [1 if i != dim else -1 for i in range(1, len(input.shape))]
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.view(views).expand(expanse)
    return torch.gather(input, dim=dim, index=index)
