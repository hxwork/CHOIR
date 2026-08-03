import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from pytorch3d.ops import knn_points
from scipy.spatial import cKDTree
from data_layout import DEFAULT_CONTACT_INDICES, DEFAULT_MANO_ASSETS, DEFAULT_SOURCE_DIR


def compute_per_frame_correspondences(
        hand_verts_seq,  # (N_frames, 778, 3)
        hand_normals_seq,  # (N_frames, 778, 3)
        obj_surface_points,  # (N_obj_samples, 3)
        obj_surface_normals,  # (N_obj_samples, 3)
        canonical_obj_mesh,
        contact_hand_indices_override=None,
        cone_angle_deg=60.0,
        dist_thresh=0.02,
):
    """
    Per-frame contact correspondence computation.

    For each frame independently, finds hand vertices in contact with the object
    surface (within cone constraint + distance threshold), then snaps each contact
    point to the canonical object mesh surface via barycentric coordinates.

    Because the object is rigid and every frame's object mesh shares the same face
    topology as the canonical mesh (only vertex positions differ due to pose), the
    returned face_id + bary_coords are directly usable on any frame's object mesh
    to reconstruct the 3-D contact location in that frame's world space.

    Returns:
        list[dict]: Length N_frames. Each entry maps
            hand_vertex_index (int) -> {"face_id": int, "bary_coords": [w0, w1, w2]}
    """
    device = hand_verts_seq.device

    # --- 1. Filter hand indices ---
    original_hand_indices_map = None
    if contact_hand_indices_override is not None:
        print(f"Filtering hand vertices using {len(contact_hand_indices_override)} provided indices.")
        original_hand_indices_map = torch.tensor(contact_hand_indices_override, device=device, dtype=torch.long)
        hand_verts_seq = hand_verts_seq[:, original_hand_indices_map, :]
        hand_normals_seq = hand_normals_seq[:, original_hand_indices_map, :]

    N_frames, N_hand, _ = hand_verts_seq.shape
    print(f"Computing per-frame correspondences: {N_frames} frames × {N_hand} hand vertices.")

    # --- 2. Batched KNN search across all frames ---
    # obj_surface_points are fixed (canonical space); hand vertices are aligned to canonical space.
    obj_points_expanded = obj_surface_points.unsqueeze(0).expand(N_frames, -1, -1)
    K_NEIGHBORS = 50
    knn = knn_points(hand_verts_seq, obj_points_expanded, K=K_NEIGHBORS, return_nn=True)

    neighbor_points = knn.knn          # (N_frames, N_hand, K, 3)
    neighbor_indices = knn.idx         # (N_frames, N_hand, K)
    dists_raw = torch.sqrt(knn.dists)  # (N_frames, N_hand, K)

    # --- 3. Normal-cone + distance filtering ---
    # Direction: object_point -> hand_point should align with the outward hand normal.
    direction_vec = neighbor_points - hand_verts_seq.unsqueeze(2)  # (N_frames, N_hand, K, 3)
    direction_vec_norm = F.normalize(direction_vec, p=2, dim=-1)
    cos_sim = torch.sum(hand_normals_seq.unsqueeze(2) * direction_vec_norm, dim=-1)  # (N_frames, N_hand, K)
    cos_thresh = torch.cos(torch.tensor(np.radians(cone_angle_deg), device=device))

    valid_mask = (cos_sim > cos_thresh) & (dists_raw < dist_thresh)  # (N_frames, N_hand, K)

    masked_dists = dists_raw.clone()
    masked_dists[~valid_mask] = float('inf')

    # Best match per (frame, hand_vertex): closest valid object point
    min_dists, min_indices_local = torch.min(masked_dists, dim=2)  # (N_frames, N_hand)

    # Coordinates of best matching object surface point  (N_frames, N_hand, 3)
    idx_expanded = min_indices_local.unsqueeze(2).unsqueeze(3).expand(-1, -1, 1, 3)
    best_obj_points = torch.gather(neighbor_points, 2, idx_expanded).squeeze(2)

    # --- 4. Per-frame: snap to canonical mesh surface (get face_id + bary_coords) ---
    # best_obj_points are already sampled surface points; on_surface call gets the precise
    # face_id and barycentric coordinates needed for topology-consistent representation.
    proximity_query = trimesh.proximity.ProximityQuery(canonical_obj_mesh)
    per_frame_results = []

    for frame_idx in range(N_frames):
        frame_dict = {}

        frame_dists = min_dists[frame_idx]           # (N_hand,)
        frame_best_pts = best_obj_points[frame_idx]  # (N_hand, 3)

        valid_hand_mask = ~torch.isinf(frame_dists)
        valid_hand_indices = torch.where(valid_hand_mask)[0]

        if len(valid_hand_indices) > 0:
            valid_pts = frame_best_pts[valid_hand_indices].cpu().numpy()  # (M, 3)
            closest_points, _, face_indices = proximity_query.on_surface(valid_pts)

            for j, h_sub_idx in enumerate(valid_hand_indices):
                face_idx = int(face_indices[j])
                bary_coords = trimesh.triangles.points_to_barycentric(
                    triangles=canonical_obj_mesh.triangles[face_idx:face_idx + 1],
                    points=closest_points[j:j + 1],
                )
                h_idx = (original_hand_indices_map[h_sub_idx].item()
                         if original_hand_indices_map is not None
                         else h_sub_idx.item())
                frame_dict[h_idx] = {"face_id": face_idx, "bary_coords": bary_coords.flatten().tolist()}

        per_frame_results.append(frame_dict)

    contact_counts = [len(d) for d in per_frame_results]
    print(f"Per-frame contact counts: min={min(contact_counts)}, max={max(contact_counts)}, "
          f"mean={np.mean(contact_counts):.1f}")

    return per_frame_results


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def _make_palette(n: int) -> np.ndarray:
    """Return (n, 4) uint8 RGBA colours from the viridis colormap."""
    return (plt.cm.viridis(np.linspace(0, 1, max(n, 1))) * 255).astype(np.uint8)


def save_per_frame_visualization(
        hand_files,
        obj_files,
        per_frame_results,
        vis_output_dir: Path,
):
    """
    For each frame, save two visualisation artefacts into *vis_output_dir*:

    1. ``corr_lines_<frame_id>.ply``  – hand mesh + object mesh + coloured
       cylinders connecting every (hand vertex, object contact point) pair.

    2. ``hand_subd_<frame_id>.ply`` / ``obj_subd_<frame_id>.ply``  – subdivided
       meshes with contact vertices coloured by correspondence ID.

    The object contact point for a given (face_id, bary_coords) is reconstructed
    directly from the *current frame's* object mesh triangles, which is valid
    because the object is rigid and all frames share the same face topology.
    """
    vis_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nGenerating per-frame visualisations into {vis_output_dir} ...")

    for local_idx, (hand_file, obj_file, frame_dict) in enumerate(
            zip(hand_files, obj_files, per_frame_results)):

        # Extract the numeric frame id from the filename for output naming
        frame_id_str = hand_file.stem.split('_')[-1]

        if not frame_dict:
            continue

        hand_mesh = trimesh.load(hand_file, process=False)
        obj_mesh = trimesh.load(obj_file, process=False)
        obj_triangles = obj_mesh.triangles
        num_contacts = len(frame_dict)
        palette = _make_palette(num_contacts)

        # ---- 1. Correspondence-lines visualisation -------------------------
        scale_ref = np.linalg.norm(hand_mesh.extents)
        line_radius = max(scale_ref * 0.005, 0.002)
        cylinders = []

        for i, (h_idx, o_data) in enumerate(frame_dict.items()):
            if h_idx >= len(hand_mesh.vertices):
                continue
            face_id = o_data['face_id']
            bary = np.array(o_data['bary_coords'])
            if face_id >= len(obj_triangles):
                continue

            p_hand = hand_mesh.vertices[h_idx]
            # Reconstruct contact point on the *current frame's* object surface
            p_obj = bary @ obj_triangles[face_id]  # (3,)

            try:
                cyl = trimesh.creation.cylinder(radius=line_radius, sections=6,
                                                segment=[p_hand, p_obj])
                cyl.visual.face_colors = palette[i]
                cylinders.append(cyl)
            except Exception:
                continue

        if cylinders:
            lines_mesh = trimesh.util.concatenate(cylinders)
            scene = trimesh.Scene([hand_mesh, obj_mesh, lines_mesh])
            corr_path = vis_output_dir / f"corr_lines_{frame_id_str}.ply"
            scene.export(corr_path, file_type='ply', encoding='ascii')

        # ---- 2. Subdivided coloured-mesh visualisation ----------------------
        sub_hand_v, sub_hand_f = trimesh.remesh.subdivide(hand_mesh.vertices, hand_mesh.faces)
        vis_hand = trimesh.Trimesh(vertices=sub_hand_v, faces=sub_hand_f)
        sub_obj_v, sub_obj_f = trimesh.remesh.subdivide(obj_mesh.vertices, obj_mesh.faces)
        vis_obj = trimesh.Trimesh(vertices=sub_obj_v, faces=sub_obj_f)

        hand_colors = np.full((len(vis_hand.vertices), 4), [128, 128, 128, 255], dtype=np.uint8)
        obj_colors = np.full((len(vis_obj.vertices), 4), [128, 128, 128, 255], dtype=np.uint8)

        # Object: colour subdivided vertices nearest to each contact point
        contact_pts_3d = []
        contact_colors_list = []
        for i, (h_idx, o_data) in enumerate(frame_dict.items()):
            face_id = o_data['face_id']
            bary = np.array(o_data['bary_coords'])
            if face_id < len(obj_triangles):
                contact_pts_3d.append(bary @ obj_triangles[face_id])
                contact_colors_list.append(palette[i])

        if contact_pts_3d:
            kdtree_obj = cKDTree(vis_obj.vertices)
            _, nn_indices = kdtree_obj.query(contact_pts_3d, k=5)
            for i, color in enumerate(contact_colors_list):
                obj_colors[nn_indices[i]] = color

        # Hand: colour subdivided vertices nearest to each original contact vertex
        kdtree_hand = cKDTree(vis_hand.vertices)
        for i, (h_idx, _) in enumerate(frame_dict.items()):
            if h_idx < len(hand_mesh.vertices):
                p_hand = hand_mesh.vertices[h_idx]
                _, nn_idx = kdtree_hand.query(p_hand, k=20)
                hand_colors[nn_idx] = palette[i]

        vis_hand.visual.vertex_colors = hand_colors
        vis_obj.visual.vertex_colors = obj_colors

        vis_hand.export(vis_output_dir / f"hand_subd_{frame_id_str}.ply", encoding='ascii')
        vis_obj.export(vis_output_dir / f"obj_subd_{frame_id_str}.ply", encoding='ascii')

    print(f"Saved per-frame visualisations to {vis_output_dir}")


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def process_video(video_id, contact_indices_path_str, stage_ranges_dir_template,
                  approaching_last_percent, interaction_first_percent,
                  samples_dir="samples_ddp"):
    """
    Processes a single video sequence: compute per-frame contact correspondences,
    save results as JSON, and generate per-frame visualisations.

    Expects hand/obj meshes under ``{samples_dir}_contact_map/<video_id>/``,
    matching ``sample_cam_ray_ddp.py --output_dir <samples_dir>``.
    """
    base_dir = Path(f"{samples_dir}_contact_map")
    data_dir = base_dir / video_id

    print(f"\n{'='*20} Processing video: {video_id} {'='*20}")

    def sort_key(p):
        match = re.search(r'(\d+)', p.name)
        return int(match.group(1)) if match else -1

    all_hand_files = sorted(data_dir.glob("hand_*.ply"), key=sort_key)
    all_obj_files = sorted(data_dir.glob("obj_*.ply"), key=sort_key)
    print(f"Found {len(all_hand_files)} total frames in directory.")

    # --- Determine interaction frames via stage ranges ---
    stage_ranges_dir = Path(stage_ranges_dir_template.format(video_id=video_id))
    stage_ranges_path = stage_ranges_dir / 'stage_frame_ranges.json'

    final_hand_files = all_hand_files
    final_obj_files = all_obj_files

    if stage_ranges_path.exists():
        print(f"Filtering frames using stage ranges from {stage_ranges_path}...")
        with open(stage_ranges_path, 'r') as f:
            stage_ranges = json.load(f)

        valid_frame_indices = set()

        app_range = stage_ranges.get("1-2_approaching")
        if app_range and "original_start" in app_range and "original_end" in app_range:
            app_start, app_end = app_range["original_start"], app_range["original_end"]
            app_len = app_end - app_start
            app_frames_start_idx = app_start + int(app_len * (100.0 - approaching_last_percent) / 100.0)
            valid_frame_indices.update(range(app_frames_start_idx, app_end + 1))
        else:
            print("Warning: '1-2_approaching' stage data is incomplete or not found.")

        int_range = stage_ranges.get("2-3_interaction")
        if int_range and "original_start" in int_range and "original_end" in int_range:
            int_start, int_end = int_range["original_start"], int_range["original_end"]
            int_len = int_end - int_start
            int_frames_end_idx = int_start + int(int_len * interaction_first_percent / 100.0)
            valid_frame_indices.update(range(int_start, int_frames_end_idx + 1))
        else:
            print("Warning: '2-3_interaction' stage data is incomplete or not found.")

        if valid_frame_indices:
            hand_filtered = sorted([f for f in all_hand_files if sort_key(f) in valid_frame_indices], key=sort_key)
            obj_filtered = sorted([f for f in all_obj_files if sort_key(f) in valid_frame_indices], key=sort_key)

            if hand_filtered and len(hand_filtered) == len(obj_filtered):
                final_hand_files = hand_filtered
                final_obj_files = obj_filtered
                print(f"Using {len(final_hand_files)} filtered frames.")
            else:
                print("Warning: Filtering resulted in empty or mismatched files. Using all frames.")
        else:
            print("Warning: No valid frames found for specified stage percentages. Using all frames.")
    else:
        print(f"Warning: Stage ranges file not found at {stage_ranges_path}. Using all frames.")

    contact_indices_override = None
    if contact_indices_path_str:
        print(f"Loading contact indices from {contact_indices_path_str}...")
        with open(contact_indices_path_str, 'r') as f:
            contact_indices_override = json.load(f)
        print(f"Loaded {len(contact_indices_override)} indices.")

    if not final_hand_files or not final_obj_files:
        print(f"Warning: No hand or object .ply files found in {data_dir}. Skipping.")
        return

    # --- Load mesh sequences ---
    print(f"Loading {len(final_hand_files)} frames...")
    all_hand_meshes = [trimesh.load(f, process=False) for f in final_hand_files]
    all_obj_meshes = [trimesh.load(f, process=False) for f in final_obj_files]

    # --- Canonical object and surface sampling ---
    canonical_obj_mesh = all_obj_meshes[0]
    print(f"Canonical object: {len(canonical_obj_mesh.vertices)} verts, "
          f"{len(canonical_obj_mesh.faces)} faces.")

    N_SURFACE_SAMPLES = 10000
    obj_surface_points, face_indices_sampled = trimesh.sample.sample_surface(
        canonical_obj_mesh, N_SURFACE_SAMPLES, seed=42)
    obj_surface_normals = canonical_obj_mesh.face_normals[face_indices_sampled]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    obj_surface_points_tensor = torch.from_numpy(obj_surface_points).float().to(device)
    obj_surface_normals_tensor = torch.from_numpy(obj_surface_normals).float().to(device)

    # --- Align hand poses to canonical object space ---
    # Because the object is rigid, we recover the pose transform from each frame's
    # object mesh to the canonical mesh via Procrustes, then apply it to the hand.
    # This puts all hand vertices in a common coordinate frame aligned with the
    # canonical object — required for consistent KNN and bary_coords computation.
    print("Aligning hand meshes to canonical object space...")
    hand_verts_list = []
    hand_normals_list = []
    canonical_obj_verts_np = np.array(canonical_obj_mesh.vertices)

    for hand_mesh, obj_mesh in zip(all_hand_meshes, all_obj_meshes):
        transform_matrix, _, _ = trimesh.registration.procrustes(
            np.array(obj_mesh.vertices), canonical_obj_verts_np)
        transformed_verts = trimesh.transform_points(np.array(hand_mesh.vertices), transform_matrix)
        transformed_normals = trimesh.transform_points(
            hand_mesh.vertex_normals, transform_matrix, translate=False)
        hand_verts_list.append(torch.from_numpy(transformed_verts).float())
        hand_normals_list.append(torch.from_numpy(transformed_normals).float())

    hand_verts_seq = torch.stack(hand_verts_list).to(device)     # (N_frames, 778, 3)
    hand_normals_seq = torch.stack(hand_normals_list).to(device)  # (N_frames, 778, 3)

    # --- Compute per-frame correspondences ---
    per_frame_results = compute_per_frame_correspondences(
        hand_verts_seq,
        hand_normals_seq,
        obj_surface_points_tensor,
        obj_surface_normals_tensor,
        canonical_obj_mesh,
        contact_hand_indices_override=contact_indices_override,
    )

    # --- Save per-frame results ---
    # frame_index is extracted directly from the filename to avoid index mismatch
    # when some frame files are missing from the filtered set.
    output_dir = DEFAULT_SOURCE_DIR / video_id / "grasp_correction"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_data = []
    for local_idx, frame_dict in enumerate(per_frame_results):
        frame_index = sort_key(final_hand_files[local_idx])  # reliable: taken from filename
        output_data.append({
            "frame_index": frame_index,
            "correspondences": {str(h_idx): corr for h_idx, corr in frame_dict.items()},
        })

    output_path = output_dir / "contact_map_per_frame.json"
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"\nSaved per-frame contact map ({len(output_data)} frames) to {output_path}")

    # --- Per-frame visualisation ---
    vis_output_dir = data_dir / "visualization_per_frame"
    save_per_frame_visualization(
        final_hand_files,
        final_obj_files,
        per_frame_results,
        vis_output_dir,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute per-frame contact correspondences for interaction frames and save results.")
    parser.add_argument(
        "--video_id",
        type=str,
        nargs='+',
        default=None,
        help="One or more video IDs to process. If not provided, all videos in the root directory are processed.")
    parser.add_argument(
        "--contact_indices_path",
        type=str,
        default=str(DEFAULT_CONTACT_INDICES),
        help="Path to a JSON file containing a list of hand vertex indices to use for contact.")
    parser.add_argument(
        "--approaching_last_percent",
        type=float,
        default=50.0,
        help="Percentage of the last part of the 'approaching' stage to include (e.g. 50 = last 50%%).")
    parser.add_argument(
        "--interaction_first_percent",
        type=float,
        default=20.0,
        help="Percentage of the first part of the 'interaction' stage to include (e.g. 20 = first 20%%).")
    parser.add_argument(
        "--stage_ranges_dir",
        type=str,
        default=str(DEFAULT_SOURCE_DIR / '{video_id}'),
        help="Directory template for 'stage_frame_ranges.json'. Must contain '{video_id}'.")
    parser.add_argument(
        "--samples_dir",
        type=str,
        default="samples_ddp",
        help="Same as sample_cam_ray_ddp.py --output_dir. Reads meshes from {samples_dir}_contact_map/.")

    args = parser.parse_args()

    contact_map_root = Path(f"{args.samples_dir}_contact_map")
    if args.video_id:
        video_ids_to_process = args.video_id
        print(f"Processing specified video IDs: {video_ids_to_process}")
    else:
        print(f"No video_id specified, processing all subdirectories in {contact_map_root}...")
        video_ids_to_process = [d.name for d in contact_map_root.iterdir() if d.is_dir()]
        print(f"Found {len(video_ids_to_process)} videos to process.")

    for video_id in video_ids_to_process:
        process_video(
            video_id,
            args.contact_indices_path,
            stage_ranges_dir_template=args.stage_ranges_dir,
            approaching_last_percent=args.approaching_last_percent,
            interaction_first_percent=args.interaction_first_percent,
            samples_dir=args.samples_dir,
        )

    print("\nAll processing finished.")

# Usage:
# python compute_contact_map_per_frame.py --video_id <video_id>
# python compute_contact_map_per_frame.py --samples_dir samples_ddp --video_id <video_id>
#
# Output files:
#   output/<video_id>/grasp_correction/contact_map_per_frame.json
#     [{"frame_index": 42,
#       "correspondences": {"123": {"face_id": 7, "bary_coords": [0.2, 0.5, 0.3]}, ...}}, ...]
#
#   {samples_dir}_contact_map/<video_id>/visualization_per_frame/
#     corr_lines_<frame_id>.ply    — hand + object + colored correspondence lines
#     hand_subd_<frame_id>.ply     — subdivided hand mesh with contact vertices colored
#     obj_subd_<frame_id>.ply      — subdivided object mesh with contact vertices colored
