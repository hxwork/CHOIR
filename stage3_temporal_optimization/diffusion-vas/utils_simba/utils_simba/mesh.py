import numpy as np

def transform_vertices(src_verts, src_K, target_K):
    """
    Transforms vertices from one camera coordinate system to another based on different intrinsic matrices.
    
    Args:
        vertices (np.ndarray): Input vertices with shape (N, M, 3) in source camera coordinates.
        ours_K (np.ndarray): Source camera intrinsic matrix with shape (3, 3).
        gt_K (np.ndarray): Target camera intrinsic matrix with shape (3, 3).
        
    Returns:
        np.ndarray: Transformed vertices with shape (N, M, 3) in target camera coordinates.
    """
    # Scale transformation matrix from gt K to ours K
    scale_mat = np.eye(3)
    scale_mat[0,0] = src_K[0,0] / target_K[0,0]
    scale_mat[1,1] = src_K[1,1] / target_K[1,1]

    # Account for different center pixel locations
    center_offset = np.array([
        [src_K[0,2] - target_K[0,2]],
        [src_K[1,2] - target_K[1,2]],
        [0]
    ])

    # Reshape center_offset for proper broadcasting with vertices
    center_offset = center_offset.T[None, :, :]
    # Scale offset by depth and focal length
    center_offset = center_offset * src_verts[:,:,2:3] / np.array([target_K[0,0], target_K[1,1], 1])[None,None,:]
    
    # Apply transformations
    src_verts = src_verts @ scale_mat.T
    src_verts += center_offset
    return src_verts