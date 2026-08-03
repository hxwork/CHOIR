import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from pytorch3d.ops import ball_query, knn_points
from scipy.spatial import cKDTree
from data_layout import DEFAULT_CONTACT_INDICES, DEFAULT_MANO_ASSETS, DEFAULT_SOURCE_DIR


def compute_frame_correspondences(
        hand_verts,  # (N_hand, 3)
        hand_normals,  # (N_hand, 3)
        obj_surface_points,  # (N_obj_samples, 3)
        obj_surface_normals,  # (N_obj_samples, 3)
        canonical_obj_mesh,
        contact_hand_indices_override=None,
        top_k_percent=0.3,  # keep only the most confident contacts
        cone_angle_deg=60.0,  # cone angle threshold; directions within this angle are valid
        dist_thresh=0.04  # absolute distance threshold; farther points are not treated as contact
):
    """
    Per-frame contact search based on a normal cone.
    Only object points in front of the hand normal are considered.
    Each frame is computed independently without temporal aggregation.
    """
    device = hand_verts.device

    # --- 1. Preprocess hand indices ---
    original_hand_indices_map = None
    if contact_hand_indices_override is not None:
        original_hand_indices_map = torch.tensor(contact_hand_indices_override, device=device, dtype=torch.long)
        hand_verts = hand_verts[original_hand_indices_map, :]
        hand_normals = hand_normals[original_hand_indices_map, :]

    N_hand, _ = hand_verts.shape

    # --- 2. KNN search (expanded neighborhood) ---
    # Find K=50 neighbors, then keep those inside the normal cone
    K_NEIGHBORS = 50
    # Add batch dims: (1, N_hand, 3) and (1, N_obj_samples, 3)
    hand_verts_batch = hand_verts.unsqueeze(0)  # (1, N_hand, 3)
    obj_points_batch = obj_surface_points.unsqueeze(0)  # (1, N_obj_samples, 3)

    knn = knn_points(hand_verts_batch, obj_points_batch, K=K_NEIGHBORS, return_nn=True)

    # Candidate neighbor data
    # neighbors: (1, N_hand, K, 3)
    neighbor_points = knn.knn.squeeze(0)  # (N_hand, K, 3)
    # neighbor_indices: (1, N_hand, K)
    neighbor_indices = knn.idx.squeeze(0)  # (N_hand, K)
    # dists_raw: (1, N_hand, K)
    dists_raw = torch.sqrt(knn.dists).squeeze(0)  # (N_hand, K)

    # --- 3. Cone direction filtering ---
    # Direction vector: Object_Point - Hand_Point
    # hand_verts: (N_hand, 3) -> (N_hand, 1, 3)
    # direction_vec: (N_hand, K, 3)
    direction_vec = neighbor_points - hand_verts.unsqueeze(1)

    # Normalize direction vectors
    direction_vec_norm = F.normalize(direction_vec, p=2, dim=-1)

    # Expand hand normals: (N_hand, 1, 3)
    hand_normals_exp = hand_normals.unsqueeze(1)

    # Cosine similarity: Hand_Normal · Direction
    # cos_sim: (N_hand, K)
    cos_sim = torch.sum(hand_normals_exp * direction_vec_norm, dim=-1)

    # Cosine of the cone angle threshold
    # cos_sim > threshold means inside the cone
    cos_thresh = torch.cos(torch.tensor(np.radians(cone_angle_deg), device=device))

    # --- 4. Validity mask ---
    # Condition 1: inside the normal cone
    mask_angle = cos_sim > cos_thresh

    # Condition 2: absolute distance within a reasonable range
    mask_dist = dists_raw < dist_thresh

    # Combined mask
    valid_mask = mask_angle & mask_dist  # (N_hand, K)

    # --- 5. Pick best matches ---
    # Set invalid distances to +inf
    masked_dists = dists_raw.clone()
    masked_dists[~valid_mask] = float('inf')

    # Take the nearest valid neighbor among K
    # min_dists: (N_hand,) shortest valid distance per hand vertex
    # min_indices_local: (N_hand,) neighbor index in 0..K-1
    min_dists, min_indices_local = torch.min(masked_dists, dim=1)

    # 1. Gather best indices from (N, K)
    # Input: (N, K) -> 2D
    # Index: (N, 1) -> 2D
    best_obj_indices = torch.gather(neighbor_indices, 1, min_indices_local.unsqueeze(-1)).squeeze(-1)

    # 2. Gather best points from (N, K, 3)
    # Input: (N, K, 3) -> 3D
    # Index must be shaped as (N, 1, 3)
    idx_expanded_for_points = min_indices_local.unsqueeze(1).unsqueeze(2).expand(-1, 1, 3)
    # gather returns (N, 1, 3); squeeze to (N, 3)
    best_obj_points = torch.gather(neighbor_points, 1, idx_expanded_for_points).squeeze(1)

    # --- 6. Filter valid contacts (per-frame, no temporal cue) ---
    # Drop points whose distance is inf
    valid_points_mask = ~torch.isinf(min_dists)

    if not valid_points_mask.any():
        print("Warning: No valid contact points found within cone constraint.")
        return {}

    # Rank among valid points
    valid_indices = torch.where(valid_points_mask)[0]
    valid_scores = min_dists[valid_indices]

    # Top K selection
    k_count = max(1, int(len(valid_indices) * top_k_percent))
    _, top_k_local_indices = torch.topk(valid_scores, k=k_count, largest=False)

    top_indices = valid_indices[top_k_local_indices]  # indices within the filtered subset

    # --- 7. Build result dict ---
    # KDTree for nearest mesh vertices
    obj_vertices = canonical_obj_mesh.vertices
    obj_vertex_tree = cKDTree(obj_vertices)

    correspondence_dict = {}

    for h_sub_idx in top_indices:
        # Map back to original hand vertex indices
        h_idx = original_hand_indices_map[h_sub_idx].item() if original_hand_indices_map is not None else h_sub_idx.item()

        # 3D coords of the best matched sample points
        best_point = best_obj_points[h_sub_idx].cpu().numpy()

        # Nearest mesh vertex indices
        _, nearest_vertex_idx = obj_vertex_tree.query(best_point, k=1)

        correspondence_dict[h_idx] = int(nearest_vertex_idx)

    return correspondence_dict


def save_correspondence_lines_visualization(hand_files, obj_files, all_frame_mappings, vis_output_dir):
    """
    For each frame, creates a visualization of lines connecting hand and object correspondence points.
    Saves the combined mesh (hand + object + lines) to a file.
    
    Uses 3D Cylinders (tubes) and assigns the specific correspondence color to each cylinder.
    all_frame_mappings: dict with frame_number as key and frame_mapping as value
    """
    print("\nGenerating correspondence line visualizations (colored 3D cylinders)...")
    vis_output_dir.mkdir(parents=True, exist_ok=True)

    def get_frame_idx(file_path):
        match = re.search(r'(\d+)', file_path.name)
        return int(match.group(1)) if match else -1

    for frame_idx, (hand_file, obj_file) in enumerate(zip(hand_files, obj_files)):
        frame_number = get_frame_idx(hand_file)
        mapping = all_frame_mappings.get(frame_number, {})

        if not mapping:
            continue

        hand_mesh = trimesh.load(hand_file, process=False)
        obj_mesh = trimesh.load(obj_file, process=False)

        # Determine line thickness based on scene scale
        scale_reference = np.linalg.norm(hand_mesh.extents)
        # Cylinder radius ~0.5% of hand size
        line_radius = scale_reference * 0.005
        if line_radius == 0:
            line_radius = 0.002  # Fallback

        # Generate colors for this frame's contacts
        num_contacts = len(mapping)
        palette = (plt.cm.viridis(np.linspace(0, 1, num_contacts)) * 255).astype(np.uint8)

        cylinders = []

        for i, (h_idx, obj_vertex_idx) in enumerate(mapping.items()):
            if h_idx >= len(hand_mesh.vertices):
                continue

            # Get coordinates
            p_hand = hand_mesh.vertices[h_idx]

            # obj_vertex_idx is now a vertex index
            if obj_vertex_idx >= len(obj_mesh.vertices):
                continue
            p_obj = obj_mesh.vertices[obj_vertex_idx]

            # Create a cylinder connecting the two points
            try:
                # sections=6 ensures it looks round but keeps polygon count low
                cyl = trimesh.creation.cylinder(radius=line_radius, sections=6, segment=[p_hand, p_obj])

                # --- Color the cylinder faces directly ---
                # palette[i] is an [R, G, B, A] array
                # Assigning face_colors broadcasts to all faces of this cylinder
                cyl.visual.face_colors = palette[i]

                cylinders.append(cyl)
            except Exception:
                continue

        if not cylinders:
            continue

        # Combine all colored cylinders into one mesh
        # concatenate merges face_colors into one multi-colored mesh
        lines_mesh = trimesh.util.concatenate(cylinders)

        # Combine meshes (Hand + Object + Colored Lines)
        scene = trimesh.Scene([hand_mesh, obj_mesh, lines_mesh])

        # Export
        file_name = f"correspondence_lines_{hand_file.stem.split('_')[-1]}.ply"
        output_path = vis_output_dir / file_name

        # Export PLY with colors written
        scene.export(output_path, file_type='ply', encoding='ascii')

    print(f"Saved correspondence line visualizations to {vis_output_dir}")


def process_video(video_id, contact_indices_path_str, stage_ranges_dir_template, approaching_last_percent,
                  interaction_first_percent, samples_dir="samples_ddp"):
    """
    Processes a single video sequence to compute and save the contact map.

    Reads meshes from ``{samples_dir}_contact_map/<video_id>/`` and writes per-frame
    ``contact_map.json`` under ``{samples_dir}_render/<video_id>/<frame>/``, matching
    ``sample_cam_ray_ddp.py --output_dir <samples_dir>``.
    """
    base_dir = Path(f"{samples_dir}_contact_map")
    data_dir = base_dir / video_id

    print(f"\n{'='*20} Processing video: {video_id} {'='*20}")

    def sort_key(p):
        match = re.search(r'(\d+)', p.name)
        return int(match.group(1)) if match else -1

    all_hand_files = sorted(data_dir.glob("hand_*.ply"), key=sort_key)
    all_obj_files = sorted(data_dir.glob("obj_*.ply"), key=sort_key)

    final_hand_files = all_hand_files
    final_obj_files = all_obj_files

    print(f"Found {len(all_hand_files)} total frames in directory.")

    contact_indices_override = None
    if contact_indices_path_str:
        print(f"Loading contact indices from {contact_indices_path_str}...")
        with open(contact_indices_path_str, 'r') as f:
            contact_indices_override = json.load(f)
        print(f"Loaded {len(contact_indices_override)} indices.")

    if not final_hand_files or not final_obj_files:
        print(f"Warning: No hand or object .ply files found in {data_dir} after potential filtering. Skipping.")
        return

    # --- 1. Load all mesh sequences ---
    print(f"Loading data from {len(final_hand_files)} frames...")
    all_hand_meshes = [trimesh.load(f, process=False) for f in final_hand_files]
    all_obj_meshes = [trimesh.load(f, process=False) for f in final_obj_files]

    if not all_obj_meshes:
        print("Error: No object meshes loaded. Skipping.")
        return

    # --- 2. Create canonical object and sample surface points ---
    canonical_obj_mesh = all_obj_meshes[0]
    print(f"Using first object mesh as canonical. Vertices: {len(canonical_obj_mesh.vertices)}, Faces: {len(canonical_obj_mesh.faces)}")

    N_SURFACE_SAMPLES = 10000
    obj_surface_points, face_indices = trimesh.sample.sample_surface(canonical_obj_mesh, N_SURFACE_SAMPLES)
    obj_surface_normals = canonical_obj_mesh.face_normals[face_indices]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    obj_surface_points_tensor = torch.from_numpy(obj_surface_points).float().to(device)
    obj_surface_normals_tensor = torch.from_numpy(obj_surface_normals).float().to(device)

    # --- 3. Align all hand poses to the canonical object space ---
    print("Aligning hand meshes to canonical object space...")
    hand_verts_seq_canonical = []
    hand_normals_seq_canonical = []
    canonical_obj_verts_np = np.array(canonical_obj_mesh.vertices)

    for i in range(len(all_hand_meshes)):
        hand_mesh = all_hand_meshes[i]
        obj_mesh = all_obj_meshes[i]

        # Find transform from current obj pose to canonical obj pose
        transform_matrix, _, _ = trimesh.registration.procrustes(np.array(obj_mesh.vertices), canonical_obj_verts_np)

        # Apply transform to hand vertices
        transformed_verts = trimesh.transform_points(np.array(hand_mesh.vertices), transform_matrix)
        hand_verts_seq_canonical.append(torch.from_numpy(transformed_verts).float())

        # Apply transform to hand normals (using inverse transpose of rotation part)
        transform_matrix_3x3 = transform_matrix[:3, :3]
        transformed_normals = trimesh.transform_points(hand_mesh.vertex_normals, transform_matrix, translate=False)
        hand_normals_seq_canonical.append(torch.from_numpy(transformed_normals).float())

    hand_verts_seq = torch.stack(hand_verts_seq_canonical).to(device)
    hand_normals_seq = torch.stack(hand_normals_seq_canonical).to(device)

    # --- 4. Compute contact map for each frame independently ---
    print("\nComputing contact map for each frame independently...")
    all_frame_mappings = {}

    def get_frame_idx(file_path):
        match = re.search(r'(\d+)', file_path.name)
        return int(match.group(1)) if match else -1

    for frame_idx, (hand_file, obj_file) in enumerate(zip(final_hand_files, final_obj_files)):
        # Current frame index
        frame_number = get_frame_idx(hand_file)

        # Hand vertices/normals for this frame
        hand_verts_frame = hand_verts_seq[frame_idx]  # (N_hand, 3)
        hand_normals_frame = hand_normals_seq[frame_idx]  # (N_hand, 3)

        # Compute this frame's contact map
        frame_mapping = compute_frame_correspondences(hand_verts_frame,
                                                      hand_normals_frame,
                                                      obj_surface_points_tensor,
                                                      obj_surface_normals_tensor,
                                                      canonical_obj_mesh,
                                                      contact_hand_indices_override=contact_indices_override)

        all_frame_mappings[frame_number] = frame_mapping

        if (frame_idx + 1) % 10 == 0 or (frame_idx + 1) == len(final_hand_files):
            print(f"Processed {frame_idx + 1}/{len(final_hand_files)} frames...")

    print(f"\nComputed mappings for {len(all_frame_mappings)} frames.")
    print(f"Total contact points across all frames: {sum(len(m) for m in all_frame_mappings.values())}")

    # --- Save each frame's contact map to its corresponding frame folder ---
    base_render_dir = Path(f"{samples_dir}_render") / video_id

    for frame_number, frame_mapping in all_frame_mappings.items():
        # Create frame-specific folder
        frame_folder = base_render_dir / f"{frame_number:05d}"
        frame_folder.mkdir(parents=True, exist_ok=True)

        # Save contact map to the frame folder
        frame_output_path = frame_folder / "contact_map.json"
        with open(frame_output_path, 'w') as f:
            # Convert keys to strings for JSON serialization
            json_mapping = {str(k): int(v) for k, v in frame_mapping.items()}
            json.dump(json_mapping, f, indent=2)

    print(f"\nSaved {len(all_frame_mappings)} frame contact maps to {base_render_dir}/[frame_folders]/")

    # --- 5. Generate and save subdivided visualization meshes for each frame ---
    print("\nGenerating and saving subdivided visualization meshes for each frame...")
    vis_output_dir = data_dir / "visualization_subdivided"
    vis_output_dir.mkdir(parents=True, exist_ok=True)

    # --- Also save correspondence lines visualization using the *original* non-subdivided meshes ---
    save_correspondence_lines_visualization(final_hand_files, final_obj_files, all_frame_mappings, vis_output_dir)

    def get_frame_idx(file_path):
        match = re.search(r'(\d+)', file_path.name)
        return int(match.group(1)) if match else -1

    for hand_file, obj_file in zip(final_hand_files, final_obj_files):
        frame_number = get_frame_idx(hand_file)
        mapping = all_frame_mappings.get(frame_number, {})

        # 1. Load original meshes
        original_hand_mesh = trimesh.load(hand_file, process=False)
        original_obj_mesh = trimesh.load(obj_file, process=False)

        # 2. Subdivide meshes for a high-resolution canvas
        sub_hand_v, sub_hand_f = trimesh.remesh.subdivide(original_hand_mesh.vertices, original_hand_mesh.faces)
        vis_hand_mesh = trimesh.Trimesh(vertices=sub_hand_v, faces=sub_hand_f)

        sub_obj_v, sub_obj_f = trimesh.remesh.subdivide(original_obj_mesh.vertices, original_obj_mesh.faces)
        vis_obj_mesh = trimesh.Trimesh(vertices=sub_obj_v, faces=sub_obj_f)

        # 3. Initialize vertex color arrays for the subdivided meshes
        hand_colors = np.full((len(vis_hand_mesh.vertices), 4), [128, 128, 128, 255], dtype=np.uint8)
        obj_vertex_colors = np.full((len(vis_obj_mesh.vertices), 4), [128, 128, 128, 255], dtype=np.uint8)

        if mapping:
            # Generate unique colors for this frame's correspondences
            num_contacts = len(mapping)
            palette = (plt.cm.viridis(np.linspace(0, 1, num_contacts)) * 255).astype(np.uint8)

            # 4. Get the 3D locations of contact points from vertex indices
            contact_points_3d = []
            contact_colors = []

            for i, (h_idx, obj_vertex_idx) in enumerate(mapping.items()):
                # obj_vertex_idx is now a vertex index
                if obj_vertex_idx >= len(original_obj_mesh.vertices):
                    continue
                point_on_surface = original_obj_mesh.vertices[obj_vertex_idx]
                contact_points_3d.append(point_on_surface)
                contact_colors.append(palette[i])

            # 5. Find the nearest vertices on the *subdivided* mesh to these contact points and color them
            if contact_points_3d:
                # Build a KD-Tree for efficient nearest neighbor search on the dense mesh
                kdtree = cKDTree(vis_obj_mesh.vertices)
                # Query the tree to find the 5 nearest vertices for each contact point
                distances, indices = kdtree.query(contact_points_3d, k=5)

                for i, color in enumerate(contact_colors):
                    # Apply the color to the found vertices
                    obj_vertex_colors[indices[i]] = color

            # 6. Color hand vertices using a KD-Tree on the subdivided mesh for correct positioning and visibility
            hand_kdtree = cKDTree(vis_hand_mesh.vertices)
            for i, (h_idx, _) in enumerate(mapping.items()):
                if h_idx < len(original_hand_mesh.vertices):
                    # Get the 3D coordinate of the original hand contact vertex
                    p_hand = original_hand_mesh.vertices[h_idx]
                    # Find the k nearest vertices in the *subdivided* mesh
                    _, indices = hand_kdtree.query(p_hand, k=20)
                    # Color the found vertices to create a visible spot
                    hand_colors[indices] = palette[i]

        # 7. Assign vertex colors and export
        vis_hand_mesh.visual.vertex_colors = hand_colors
        vis_obj_mesh.visual.vertex_colors = obj_vertex_colors

        hand_out_path = vis_output_dir / f"{hand_file.stem}_subd.ply"
        obj_out_path = vis_output_dir / f"{obj_file.stem}_subd.ply"
        vis_hand_mesh.export(hand_out_path, encoding='ascii')
        vis_obj_mesh.export(obj_out_path, encoding='ascii')

    print(f"Saved subdivided colored meshes for all {len(final_hand_files)} frames to {vis_output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute robust correspondences, save the map, and export colored meshes for visualization.")
    parser.add_argument(
        "--video_id",
        type=str,
        nargs='+',  # Accept one or more video IDs
        default=None,
        help="One or more video IDs to process. If not provided, all videos in the root directory will be processed.")
    parser.add_argument("--contact_indices_path",
                        type=str,
                        default=str(DEFAULT_CONTACT_INDICES),
                        help="Path to a JSON file containing a list of hand vertex indices to use for contact.")
    parser.add_argument("--approaching_last_percent",
                        type=float,
                        default=50.0,
                        help="Percentage of the last part of the 'approaching' stage to use (e.g., 50 means the last 50%%).")
    parser.add_argument("--interaction_first_percent",
                        type=float,
                        default=20.0,
                        help="Percentage of the first part of the 'interaction' stage to use (e.g., 80 means the first 80%%).")
    parser.add_argument("--stage_ranges_dir",
                        type=str,
                        default=str(DEFAULT_SOURCE_DIR / '{video_id}'),
                        help="Directory template to find 'stage_frame_ranges.json'. Should contain '{video_id}'.")
    parser.add_argument(
        "--samples_dir",
        type=str,
        default="samples_ddp",
        help="Same as sample_cam_ray_ddp.py --output_dir. Reads {samples_dir}_contact_map/, writes {samples_dir}_render/.")

    args = parser.parse_args()

    contact_map_root = Path(f"{args.samples_dir}_contact_map")
    if args.video_id:
        # If video_id is provided, it will be a list because of nargs='+'
        video_ids_to_process = args.video_id
        print(f"Processing specified video IDs: {video_ids_to_process}")
    else:
        print(f"No video_id specified, processing all subdirectories in {contact_map_root}...")
        video_ids_to_process = [d.name for d in contact_map_root.iterdir() if d.is_dir()]
        print(f"Found {len(video_ids_to_process)} videos to process.")

    for video_id in video_ids_to_process:
        process_video(video_id,
                      args.contact_indices_path,
                      stage_ranges_dir_template=args.stage_ranges_dir,
                      approaching_last_percent=args.approaching_last_percent,
                      interaction_first_percent=args.interaction_first_percent,
                      samples_dir=args.samples_dir)

    print("\nAll processing finished.")

# Usage:
# python compute_contact_map_render.py --video_id <video_id>
# python compute_contact_map_render.py --samples_dir samples_ddp --video_id <video_id>
# mapping layout: { frame_number: { hand_vertex_index: obj_vertex_index } }
