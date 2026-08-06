"""
rendering.py — Pure rendering utilities for the HOI 6DoF pipeline.

Mesh rendering helpers used by Stage 3 fitting and diagnostics.
Importing with `from choir_stage3.geometry.rendering import *` exposes all
public symbols.
"""

import numpy as np
import torch
from pytorch3d.ops import knn_points
from pytorch3d.renderer import (Materials, MeshRasterizer, MeshRenderer,
                                PointLights, RasterizationSettings,
                                TexturesVertex)
from pytorch3d.renderer.mesh.shader import HardPhongShader
from pytorch3d.structures import (Meshes, join_meshes_as_batch,
                                  join_meshes_as_scene)

__all__ = [
    "get_render_params",
    "weighted_false_negative_loss",
    "weighted_false_positive_loss",
    "vectorized_pose_guiding_loss",
    "render_side_view_silhouette",
    "render_side_view_rgb",
]

# ---------------------------------------------------------------------------
# Render-parameter schedule
# ---------------------------------------------------------------------------


def get_render_params(current_step, total_steps):
    """Return (blur_radius, gamma, faces_per_pixel) for the given optimisation step."""
    START_SIGMA = 1e-4
    START_GAMMA = 1e-2
    END_SIGMA = 1e-4
    END_GAMMA = 1e-4

    t = current_step / total_steps
    curr_sigma = START_SIGMA * (END_SIGMA / START_SIGMA)**t
    curr_gamma = START_GAMMA * (END_GAMMA / START_GAMMA)**t

    if curr_sigma > 2.0:
        curr_faces_per_pixel = 80
    elif curr_sigma > 0.5:
        curr_faces_per_pixel = 50
    else:
        curr_faces_per_pixel = 20

    return curr_sigma, curr_gamma, curr_faces_per_pixel


# ---------------------------------------------------------------------------
# Mask losses
# ---------------------------------------------------------------------------


def weighted_false_negative_loss(rendered_mask, gt_mask, weight=20.0):
    """Penalise GT pixels that are not covered by the rendered mask.

    Args:
        rendered_mask: (B, H, W) soft values in [0, 1]
        gt_mask:       (B, H, W) binary 0/1
    """
    diff = gt_mask - rendered_mask
    missed_area = torch.relu(diff)
    return (missed_area**2).mean() * weight


def weighted_false_positive_loss(pred_mask, gt_mask):
    """Penalise rendered pixels that fall outside the GT mask.

    Args:
        pred_mask: (B, H, W) soft values in [0, 1]
        gt_mask:   (B, H, W) binary 0/1
    """
    diff = pred_mask - gt_mask
    return (torch.relu(diff)**2).mean()


# ---------------------------------------------------------------------------
# Vectorised pose-guiding loss
# ---------------------------------------------------------------------------


def vectorized_pose_guiding_loss(posed_meshes, gt_mask, rendered_masks, cameras, num_samples=2000):
    """Pull mesh vertices towards uncovered GT-mask pixels via KNN in screen space.

    Args:
        posed_meshes:   pytorch3d Meshes, batch size B
        gt_mask:        (B, H, W) binary GT mask
        rendered_masks: (B, H, W) soft rendered mask (detached internally)
        cameras:        pytorch3d PerspectiveCameras
        num_samples:    number of GT pixels to sample per image
    """
    B, H, W = gt_mask.shape

    flat_gt = gt_mask.view(B, -1)
    flat_pred = rendered_masks.detach().view(B, -1)

    uncovered_score = flat_gt * (1.0 - flat_pred)
    probs = flat_gt + (uncovered_score * 20.0) + 1e-8

    flat_indices = torch.multinomial(probs, num_samples, replacement=True)

    sampled_val = torch.gather(flat_gt, 1, flat_indices)
    valid_mask = sampled_val > 0.5

    if valid_mask.sum() == 0:
        return torch.tensor(0.0, device=gt_mask.device, requires_grad=True)

    batch_y = torch.div(flat_indices, W, rounding_mode='floor')
    batch_x = flat_indices % W
    gt_points_batch = torch.stack([batch_x, batch_y], dim=2).float()

    verts_batch = posed_meshes.verts_padded()
    projected_batch = cameras.transform_points_screen(verts_batch, image_size=((H, W),))
    pred_points_batch = projected_batch[..., :2]

    num_verts_per_mesh = posed_meshes.num_verts_per_mesh()
    knn = knn_points(gt_points_batch, pred_points_batch, lengths2=num_verts_per_mesh, K=1)
    dists_sq = knn.dists.squeeze(-1)

    valid_dists = dists_sq * valid_mask.float()
    num_valid_per_image = valid_mask.float().sum(dim=1)
    loss_per_image = valid_dists.sum(dim=1) / (num_valid_per_image + 1e-8)

    return loss_per_image.mean()


# ---------------------------------------------------------------------------
# Side-view silhouette helper
# ---------------------------------------------------------------------------


def render_side_view_silhouette(obj_meshes: Meshes, hand_meshes: Meshes, cameras, renderer, device) -> np.ndarray:
    """Render a pseudo side-view silhouette by rotating mesh vertices 90° around Y.

    The scene centroid is preserved so that the object stays roughly centred in the
    image.  The same camera / renderer that is used for the front view is reused,
    which gives a consistent image-space scale.

    Args:
        obj_meshes:  Meshes batch, shape (N, V_obj, 3)
        hand_meshes: Meshes batch, shape (N, V_hand, 3)
        cameras:     PerspectiveCameras with N cameras
        renderer:    MeshRenderer (silhouette or phong)
        device:      torch device

    Returns:
        numpy array (N, H, W) of silhouette alpha values in [0, 1]
    """
    with torch.no_grad():
        ov = obj_meshes.verts_padded()  # (N, V_obj, 3)
        hv = hand_meshes.verts_padded()  # (N, V_hand, 3)

        # Scene centroid (across all frames and both meshes)
        ctr = torch.cat([ov, hv], dim=1).reshape(-1, 3).mean(dim=0)  # (3,)

        def _rot90_y(v):
            """Rotate 90° around Y axis: [x,y,z] → [z,y,-x] relative to centroid."""
            w = v - ctr
            return torch.stack([w[..., 2], w[..., 1], -w[..., 0]], dim=-1) + ctr

        side_obj_verts = list(_rot90_y(ov))  # list of N tensors (V_obj, 3)
        side_hand_verts = list(_rot90_y(hv))  # list of N tensors (V_hand, 3)

        N = ov.shape[0]
        side_obj = Meshes(verts=side_obj_verts, faces=obj_meshes.faces_list())
        side_hand = Meshes(verts=side_hand_verts, faces=hand_meshes.faces_list())

        # White textures for silhouette renderer
        N_ov, V_ov = ov.shape[:2]
        N_hv, V_hv = hv.shape[:2]
        side_obj.textures = TexturesVertex(verts_features=torch.ones(N_ov, V_ov, 3, device=device))
        side_hand.textures = TexturesVertex(verts_features=torch.ones(N_hv, V_hv, 3, device=device))

        side_scenes = [join_meshes_as_scene([side_obj[i], side_hand[i]]) for i in range(N)]
        side_batch = join_meshes_as_batch(side_scenes)

        result = renderer(side_batch, cameras=cameras)[..., 3]  # (N, H, W)
        return result.cpu().numpy()


# ---------------------------------------------------------------------------
# Side-view RGB (Phong-shaded) helper
# ---------------------------------------------------------------------------

def render_side_view_rgb(obj_meshes: Meshes,
                         hand_meshes: Meshes,
                         cameras,
                         device,
                         image_size) -> np.ndarray:
    """Render a Phong-shaded side view by rotating meshes 90° around Y.

    Uses a fresh HardPhongShader renderer so the caller does not need to
    provide an external renderer or lights.

    Args:
        obj_meshes:  Meshes batch (N, V_obj, 3)
        hand_meshes: Meshes batch (N, V_hand, 3)
        cameras:     PerspectiveCameras with N cameras
        device:      torch device
        image_size:  (H, W) tuple

    Returns:
        numpy array (N, H, W, 3) RGB in [0, 1]
    """
    with torch.no_grad():
        ov = obj_meshes.verts_padded()   # (N, V_obj, 3)
        hv = hand_meshes.verts_padded()  # (N, V_hand, 3)

        ctr = torch.cat([ov, hv], dim=1).reshape(-1, 3).mean(dim=0)

        def _rot90_y(v):
            w = v - ctr
            return torch.stack([w[..., 2], w[..., 1], -w[..., 0]], dim=-1) + ctr

        N = ov.shape[0]
        N_ov, V_ov = ov.shape[:2]
        N_hv, V_hv = hv.shape[:2]

        side_obj = Meshes(verts=list(_rot90_y(ov)), faces=obj_meshes.faces_list())
        side_hand = Meshes(verts=list(_rot90_y(hv)), faces=hand_meshes.faces_list())

        # Blue-ish object, red-ish hand — consistent with the full-sequence video
        side_obj.textures = TexturesVertex(
            verts_features=torch.tensor([0.65, 0.8, 1.0], device=device)
                           .view(1, 1, 3).expand(N_ov, V_ov, -1))
        side_hand.textures = TexturesVertex(
            verts_features=torch.tensor([1.0, 0.4, 0.4], device=device)
                           .view(1, 1, 3).expand(N_hv, V_hv, -1))

        side_scenes = [join_meshes_as_scene([side_obj[i], side_hand[i]]) for i in range(N)]
        side_batch = join_meshes_as_batch(side_scenes)

        lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])
        materials = Materials(device=device, shininess=50.0)
        raster_settings = RasterizationSettings(image_size=image_size, blur_radius=0.0, faces_per_pixel=1)
        phong_renderer = MeshRenderer(
            rasterizer=MeshRasterizer(raster_settings=raster_settings),
            shader=HardPhongShader(device=device, cameras=cameras, lights=lights),
        )

        rgb = phong_renderer(side_batch, cameras=cameras, lights=lights, materials=materials)
        return rgb[..., :3].cpu().numpy()  # (N, H, W, 3)
