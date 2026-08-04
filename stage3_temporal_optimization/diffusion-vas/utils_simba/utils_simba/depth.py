import numpy as np
import cv2
import torch
import torch.nn.functional as F

def save_depth(depth, fname, scale= 0.00012498664727900177):
    depth_scale = 1 / scale
    max_depth = 2**24 / (1/ scale)     # max 8.19112491607666 = 2**16 / (1/ 0.00012498664727900177)
    exceed_depth = depth[depth > max_depth]
    if len(exceed_depth) > 0:
        print(f"Warning: {len(exceed_depth)} depth values exceed the maximum depth of {max_depth}. Clipping to {max_depth}.")
    depth = depth.clip(0, max_depth)

    depth_scaled = (depth * depth_scale).astype(np.uint32)
    depth_lsb = np.bitwise_and(depth_scaled, 0xFF)  # Least significant byte
    depth_msb = np.bitwise_and(np.right_shift(depth_scaled, 8), 0xFF)  # Most significant byte
    depth_msb2 = np.bitwise_and(np.right_shift(depth_scaled, 16), 0xFF)  # Most significant byte
    depth_encoded = np.zeros((depth.shape[0], depth.shape[1], 3), dtype=np.uint8)
    depth_encoded[..., 2] = depth_lsb
    depth_encoded[..., 1] = depth_msb
    depth_encoded[..., 0] = depth_msb2
    cv2.imwrite(fname, depth_encoded)
    # print(f"Saved depth to {fname}")

def get_depth(depth_file, zfar=np.inf, depth_scale = 0.00012498664727900177):
    # depth = cv2.imread(self.color_files[i].replace('.jpg','.png').replace('rgb','depth'), -1)
    depth = cv2.imread(depth_file, -1)
    depth = (depth[...,0]*256.0*256.0 + depth[...,1]*256.0 + depth[...,2])*depth_scale
    depth[(depth<0.01) | (depth>=zfar)] = 0
    return depth


def depth2xyzmap_cuda(depth, K, uvs=None):
    H, W = depth.shape[-2:]  # assume (H, W) or (1, H, W)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    device = depth.device
    if uvs is None:
        us, vs = torch.meshgrid(
            torch.arange(W, device=device),
            torch.arange(H, device=device),
            indexing='xy'
        )
        us = us.flatten()
        vs = vs.flatten()
    else:
        uvs = uvs.round().long()
        us = uvs[:, 0]
        vs = uvs[:, 1]

    zs = depth[vs, us]
    valid = zs >= 0.01
    xs = (us[valid].float() - cx) * zs[valid] / fx
    ys = (vs[valid].float() - cy) * zs[valid] / fy
    pts = torch.stack((xs, ys, zs[valid]), dim=1)

    xyz_map = torch.zeros((H, W, 3), dtype=torch.float32, device=device)
    xyz_map[vs[valid], us[valid]] = pts
    return xyz_map

def depth2xyzmap(depth, K, uvs=None):
    invalid_mask = (depth<0.01)
    H,W = depth.shape[:2]
    if uvs is None:
        vs,us = np.meshgrid(np.arange(0,H),np.arange(0,W), sparse=False, indexing='ij')
        vs = vs.reshape(-1)
        us = us.reshape(-1)
    else:
        uvs = uvs.round().astype(int)
        us = uvs[:,0]
        vs = uvs[:,1]
    zs = depth[vs,us]
    xs = (us-K[0,2])*zs/K[0,0]
    ys = (vs-K[1,2])*zs/K[1,1]
    pts = np.stack((xs.reshape(-1),ys.reshape(-1),zs.reshape(-1)), 1)  #(N,3)
    xyz_map = np.zeros((H,W,3), dtype=np.float32)
    xyz_map[vs,us] = pts
    xyz_map[invalid_mask] = 0
    return xyz_map

def xyz2depthmap(xyz, K, image_size):
    """
    Convert 3D points to a depth map.

    Parameters:
    - xyz: (N, 3) array of 3D points in camera coordinates.
    - K: (3, 3) camera intrinsic matrix.
    - image_size: Tuple (H, W) specifying the height and width of the depth map.

    Returns:
    - depth_map: (H, W) array representing the depth map.
    """
    H, W = image_size
    depth_map = np.zeros((H, W), dtype=np.float32)

    # Extract intrinsic parameters
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    # Extract X, Y, Z
    X = xyz[:, 0]
    Y = xyz[:, 1]
    Z = xyz[:, 2]

    # Filter out points with non-positive depth
    valid = Z > 0
    X = X[valid]
    Y = Y[valid]
    Z = Z[valid]

    # Project to pixel coordinates
    u = (fx * X / Z + cx).round().astype(np.int32)
    v = (fy * Y / Z + cy).round().astype(np.int32)

    # Filter points that fall inside the image boundaries
    valid_idx = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u = u[valid_idx]
    v = v[valid_idx]
    Z = Z[valid_idx]

    # Iterate through points and populate the depth map
    for idx in range(len(Z)):
        pixel_u = u[idx]
        pixel_v = v[idx]
        depth = Z[idx]

        # If the pixel is empty or the current depth is closer, update the depth map
        if depth_map[pixel_v, pixel_u] == 0 or depth < depth_map[pixel_v, pixel_u]:
            depth_map[pixel_v, pixel_u] = depth

    return depth_map

def erode_depth_map(depth, structure_size=1, d_thresh=0.05, frac_req=0.5, zfar=np.inf):
    """drop pixels whose neighbors disagree too much.

    Args:
        depth: 2D depth array.
        structure_size: half-window size for the neighborhood.
        d_thresh: allowed depth difference to treat a neighbor as consistent.
        frac_req: fraction of inconsistent neighbors to trigger removal.
        zfar: maximum valid depth.
    """
    h, w = depth.shape
    out = np.zeros_like(depth, dtype=np.float32)
    win = range(-structure_size, structure_size + 1)
    for y in range(h):
        for x in range(w):
            center = depth[y, x]
            if center <= 0.1 or center > zfar:
                out[y, x] = 0
                continue
            count = 0
            for dy in win:
                ny = y + dy
                if ny < 0 or ny >= h:
                    continue
                for dx in win:
                    nx = x + dx
                    if nx < 0 or nx >= w:
                        continue
                    val = depth[ny, nx]
                    if val <= 0.1 or val > zfar or abs(val - center) > d_thresh:
                        count += 1
            total = (2 * structure_size + 1) ** 2
            out[y, x] = 0 if (count / total) >= frac_req else center
    return out

def gauss_filter_depth_map(depth, radius=2, sigma_d=1.0, sigma_r=0.05, zfar=np.inf):
    """bilateral-like smoothing on valid, consistent depths.

    Args:
        depth: 2D depth array.
        radius: half-window size for filtering.
        sigma_d: Penalizes distance in image space determining how far from the center pixel the filter is willing to gather information.
        sigma_r: Penalizes difference in depth values controling how much depth difference is allowed before a neighbor is rejected. Small sigma_r → strict, even small depth differences → strongly down-weighted. → preserves edges sharply → BUT may oversuppress smoothing (too selective)
        zfar: maximum valid depth.
    """
    h, w = depth.shape
    out = np.zeros_like(depth, dtype=np.float32)
    win = range(-radius, radius + 1)
    total_neighbors = (2 * radius + 1) ** 2
    for y in range(h):
        for x in range(w):
            mean_depth = 0.0
            valid_count = 0
            for dy in win:
                ny = y + dy
                if ny < 0 or ny >= h:
                    continue
                for dx in win:
                    nx = x + dx
                    if nx < 0 or nx >= w:
                        continue
                    dval = depth[ny, nx]
                    if 0.1 <= dval <= zfar:
                        valid_count += 1
                        mean_depth += dval
            if valid_count == 0:
                continue
            mean_depth /= valid_count

            center_depth = depth[y, x]
            sum_w = 0.0
            sum_val = 0.0
            for dy in win:
                ny = y + dy
                if ny < 0 or ny >= h:
                    continue
                for dx in win:
                    nx = x + dx
                    if nx < 0 or nx >= w:
                        continue
                    dval = depth[ny, nx]
                    if 0.1 <= dval <= zfar and abs(dval - mean_depth) < 0.01:
                        spatial = (dx * dx + dy * dy) / (2.0 * sigma_d * sigma_d)
                        range_term = ((center_depth - dval) ** 2) / (2.0 * sigma_r * sigma_r)
                        weight = np.exp(-(spatial + range_term))
                        sum_w += weight
                        sum_val += weight * dval
            if sum_w > 0.0 and valid_count / total_neighbors > 0:
                out[y, x] = sum_val / sum_w
    return out


def erode_depth_map_torch(depth: torch.Tensor, structure_size: int = 1, d_thresh: float = 0.05, frac_req: float = 0.5, zfar: float = float("inf")) -> torch.Tensor:
    """CUDA-friendly version of erodeDepthMapDevice.

    Args:
        depth: (H, W) torch tensor (supports CUDA) with depths.
        structure_size: half-window size.
        d_thresh: allowed difference between center and neighbors.
        frac_req: fraction of inconsistent neighbors needed to drop the center.
        zfar: maximum valid depth.
    """
    if depth.dim() != 2:
        raise ValueError("depth must be HxW tensor")
    device = depth.device
    k = 2 * structure_size + 1
    padded = F.pad(depth.unsqueeze(0).unsqueeze(0), (structure_size,) * 4, mode="constant", value=0)
    patches = F.unfold(padded, kernel_size=k)  # [1, k*k, H*W]
    h, w = depth.shape
    center = depth.reshape(1, 1, h * w)
    valid_neighbors = (patches > 0.1) & (patches <= zfar)
    diff_mask = torch.abs(patches - center) > d_thresh
    inconsistent = (~valid_neighbors) | diff_mask
    count = inconsistent.sum(dim=1)
    total = float(k * k)
    keep = (count.float() / total) < frac_req
    center_valid = (center > 0.1) & (center <= zfar)
    keep = keep & center_valid.squeeze(1)
    out = torch.where(keep, center.squeeze(1), torch.zeros_like(center.squeeze(1)))
    return out.view(h, w).to(device)


def gauss_filter_depth_map_torch(depth: torch.Tensor, radius: int = 2, sigma_d: float = 1.0, sigma_r: float = 0.05, zfar: float = float("inf")) -> torch.Tensor:
    """CUDA-friendly version of gaussFilterDepthMapDevice using a bilateral-like filter.

    Args:
        depth: 2D depth array.
        radius: half-window size for filtering.
        sigma_d: Penalizes distance in image space determining how far from the center pixel the filter is willing to gather information.
        sigma_r: Penalizes difference in depth values controling how much depth difference is allowed before a neighbor is rejected. Small sigma_r → strict, even small depth differences → strongly down-weighted. → preserves edges sharply → BUT may oversuppress smoothing (too selective)
        zfar: maximum valid depth.
    """
    if depth.dim() != 2:
        raise ValueError("depth must be HxW tensor")
    device = depth.device
    depth = depth.float()
    k = 2 * radius + 1
    h, w = depth.shape
    padded = F.pad(depth.unsqueeze(0).unsqueeze(0), (radius,) * 4, mode="constant", value=0)
    patches = F.unfold(padded, kernel_size=k)  # [1, k*k, H*W] flattened row-major
    center = depth.reshape(1, 1, h * w)

    valid_mask = (patches >= 0.1) & (patches <= zfar)
    valid_count = valid_mask.sum(dim=1).float()  # [1, H*W]
    sum_valid = (patches * valid_mask).sum(dim=1)
    mean_depth = torch.zeros_like(sum_valid)
    nonzero = valid_count > 0
    mean_depth[nonzero] = sum_valid[nonzero] / valid_count[nonzero]

    # Spatial term aligned with unfold order (y first, then x).
    offsets_y = torch.arange(k, device=device).view(k, 1).expand(k, k) - radius
    offsets_x = torch.arange(k, device=device).view(1, k).expand(k, k) - radius
    spatial = (offsets_x ** 2 + offsets_y ** 2).reshape(-1).float() / (2.0 * sigma_d * sigma_d)  # [k*k]

    within_mean = torch.abs(patches - mean_depth.unsqueeze(1)) < 0.01
    range_term = (center - patches) ** 2 / (2.0 * sigma_r * sigma_r)
    weights = torch.exp(-(spatial.view(1, -1, 1) + range_term))
    weights = weights * valid_mask * within_mean

    sum_w = weights.sum(dim=1)
    sum_val = (weights * patches).sum(dim=1)
    mask_out = (sum_w > 0) & (valid_count > 0)
    out = torch.zeros_like(sum_w)
    out[mask_out] = sum_val[mask_out] / sum_w[mask_out]
    return out.view(h, w)
