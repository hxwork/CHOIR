import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from pytorch3d.ops import knn_points


def load_contact_indices(contact_indices_path: Optional[str]) -> Optional[List[int]]:
    if not contact_indices_path:
        return None
    path = Path(contact_indices_path)
    if not path.exists():
        print(f"Contact indices file does not exist: {path}")
        return None
    with path.open("r") as f:
        indices = json.load(f)
    if not isinstance(indices, list):
        raise ValueError(f"Expected a list in contact indices file: {path}")
    return [int(v) for v in indices]


def compute_per_frame_correspondences(
    hand_verts_seq: torch.Tensor,        # (N_frames, N_hand, 3)
    hand_normals_seq: torch.Tensor,      # (N_frames, N_hand, 3)
    obj_surface_points: torch.Tensor,    # (N_obj_samples, 3)
    canonical_obj_mesh: trimesh.Trimesh,
    contact_hand_indices_override: Optional[List[int]] = None,
    cone_angle_deg: float = 60.0,
    dist_thresh: float = 0.02,
    k_neighbors: int = 50,
) -> List[Dict[int, Dict[str, object]]]:
    device = hand_verts_seq.device

    original_hand_indices_map = None
    if contact_hand_indices_override is not None:
        original_hand_indices_map = torch.tensor(
            contact_hand_indices_override, device=device, dtype=torch.long)
        hand_verts_seq = hand_verts_seq[:, original_hand_indices_map, :]
        hand_normals_seq = hand_normals_seq[:, original_hand_indices_map, :]

    n_frames, n_hand, _ = hand_verts_seq.shape
    print(f"Computing per-frame correspondences: {n_frames} frames x {n_hand} hand vertices")

    obj_points_expanded = obj_surface_points.unsqueeze(0).expand(n_frames, -1, -1)
    knn = knn_points(hand_verts_seq, obj_points_expanded, K=k_neighbors, return_nn=True)

    neighbor_points = knn.knn
    dists_raw = torch.sqrt(knn.dists)

    direction_vec = neighbor_points - hand_verts_seq.unsqueeze(2)
    direction_vec_norm = F.normalize(direction_vec, p=2, dim=-1)
    cos_sim = torch.sum(hand_normals_seq.unsqueeze(2) * direction_vec_norm, dim=-1)
    cos_thresh = torch.cos(torch.tensor(np.radians(cone_angle_deg), device=device))

    valid_mask = (cos_sim > cos_thresh) & (dists_raw < dist_thresh)
    masked_dists = dists_raw.clone()
    masked_dists[~valid_mask] = float("inf")

    min_dists, min_indices_local = torch.min(masked_dists, dim=2)
    idx_expanded = min_indices_local.unsqueeze(2).unsqueeze(3).expand(-1, -1, 1, 3)
    best_obj_points = torch.gather(neighbor_points, 2, idx_expanded).squeeze(2)

    proximity_query = trimesh.proximity.ProximityQuery(canonical_obj_mesh)
    per_frame_results: List[Dict[int, Dict[str, object]]] = []

    for frame_idx in range(n_frames):
        frame_dict: Dict[int, Dict[str, object]] = {}
        frame_dists = min_dists[frame_idx]
        frame_best_pts = best_obj_points[frame_idx]

        valid_hand_mask = ~torch.isinf(frame_dists)
        valid_hand_indices = torch.where(valid_hand_mask)[0]
        if len(valid_hand_indices) == 0:
            per_frame_results.append(frame_dict)
            continue

        valid_pts = frame_best_pts[valid_hand_indices].detach().cpu().numpy()
        closest_points, _, face_indices = proximity_query.on_surface(valid_pts)

        for j, h_sub_idx in enumerate(valid_hand_indices):
            face_idx = int(face_indices[j])
            bary_coords = trimesh.triangles.points_to_barycentric(
                triangles=canonical_obj_mesh.triangles[face_idx:face_idx + 1],
                points=closest_points[j:j + 1],
            ).reshape(-1).tolist()
            h_idx = (
                int(original_hand_indices_map[h_sub_idx].item())
                if original_hand_indices_map is not None
                else int(h_sub_idx.item())
            )
            frame_dict[h_idx] = {"face_id": face_idx, "bary_coords": bary_coords}

        per_frame_results.append(frame_dict)

    contact_counts = [len(v) for v in per_frame_results]
    if contact_counts:
        print(
            f"Per-frame contact counts: min={min(contact_counts)}, "
            f"max={max(contact_counts)}, mean={np.mean(contact_counts):.1f}"
        )
    return per_frame_results


def estimate_contact_map_for_sampled_frames(
    hand_verts_seq_world: torch.Tensor,      # (N, 778, 3)
    hand_normals_seq_world: torch.Tensor,    # (N, 778, 3)
    obj_verts_seq_world: torch.Tensor,       # (N, V_obj, 3)
    obj_faces: torch.Tensor,                 # (F, 3)
    sampled_indices: List[int],
    contact_hand_indices_override: Optional[List[int]] = None,
    cone_angle_deg: float = 60.0,
    dist_thresh: float = 0.02,
    n_surface_samples: int = 10000,
) -> List[Dict[str, object]]:
    if hand_verts_seq_world.ndim != 3 or obj_verts_seq_world.ndim != 3:
        raise ValueError("Expected batched hand/object vertices of shape (N, V, 3)")
    if len(sampled_indices) != hand_verts_seq_world.shape[0]:
        raise ValueError("sampled_indices length must match number of sampled frames")

    device = hand_verts_seq_world.device
    faces_np = obj_faces.detach().cpu().numpy()
    canonical_obj_verts = obj_verts_seq_world[0].detach().cpu().numpy()
    canonical_obj_mesh = trimesh.Trimesh(vertices=canonical_obj_verts, faces=faces_np, process=False)

    sampled_points_np, _ = trimesh.sample.sample_surface(
        canonical_obj_mesh, n_surface_samples, seed=42)
    obj_surface_points = torch.from_numpy(sampled_points_np).float().to(device)

    aligned_hand_verts: List[torch.Tensor] = []
    aligned_hand_normals: List[torch.Tensor] = []

    for frame_idx in range(hand_verts_seq_world.shape[0]):
        obj_verts_np = obj_verts_seq_world[frame_idx].detach().cpu().numpy()
        transform_matrix, _, _ = trimesh.registration.procrustes(obj_verts_np, canonical_obj_verts)

        hand_verts_np = hand_verts_seq_world[frame_idx].detach().cpu().numpy()
        hand_normals_np = hand_normals_seq_world[frame_idx].detach().cpu().numpy()

        transformed_verts = trimesh.transform_points(hand_verts_np, transform_matrix)
        transformed_normals = trimesh.transform_points(
            hand_normals_np, transform_matrix, translate=False)

        aligned_hand_verts.append(torch.from_numpy(transformed_verts).float())
        aligned_hand_normals.append(torch.from_numpy(transformed_normals).float())

    hand_verts_aligned = torch.stack(aligned_hand_verts).to(device)
    hand_normals_aligned = torch.stack(aligned_hand_normals).to(device)

    per_frame_results = compute_per_frame_correspondences(
        hand_verts_seq=hand_verts_aligned,
        hand_normals_seq=hand_normals_aligned,
        obj_surface_points=obj_surface_points,
        canonical_obj_mesh=canonical_obj_mesh,
        contact_hand_indices_override=contact_hand_indices_override,
        cone_angle_deg=cone_angle_deg,
        dist_thresh=dist_thresh,
    )

    output_data: List[Dict[str, object]] = []
    for local_idx, frame_dict in enumerate(per_frame_results):
        output_data.append(
            {
                "frame_index": int(sampled_indices[local_idx]),
                "correspondences": {str(h_idx): corr for h_idx, corr in frame_dict.items()},
            }
        )
    return output_data
