import argparse
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import k3d
import numpy as np
import trimesh
from ipywidgets.embed import embed_minimal_html
from tqdm import tqdm


def process_single_video(video_id, input_root, output_root, compression_level, frame_step, use_float16):
    """
    Process one video; intended for multiprocessing workers.
    """
    video_dir = os.path.join(input_root, video_id)
    output_html_path = os.path.join(output_root, f"{video_id}.html")

    # 1. List input files
    try:
        all_files = sorted([f for f in os.listdir(video_dir) if f.endswith('.ply')])
    except FileNotFoundError:
        return f"Error: {video_dir} not found."

    ply_files = all_files[::frame_step]

    if not ply_files:
        return f"Skipped {video_id}: No .ply files"

    vertices_seq = {}
    faces = None
    is_mesh = False
    dtype = np.float16 if use_float16 else np.float32

    # 2. Load data
    try:
        for i, ply_file in enumerate(ply_files):
            key_frame = str(i)
            file_path = os.path.join(video_dir, ply_file)

            mesh = trimesh.load(file_path, process=False)
            vertices_seq[key_frame] = mesh.vertices.astype(dtype)

            if i == 0 and hasattr(mesh, 'faces') and len(mesh.faces) > 0:
                faces = mesh.faces
                is_mesh = True
    except Exception as e:
        return f"Error loading {video_id}: {str(e)}"

    if not vertices_seq:
        return f"Skipped {video_id}: Empty data"

    # 3. Create K3D plot
    plot = k3d.plot(name='Mesh Sequence')
    plot.camera_auto_fit = True
    plot.grid_visible = False  # optional: hide grid for a cleaner view

    if is_mesh:
        mesh_object = k3d.mesh(vertices_seq['0'], faces, color=0x00aaff, side='double', compression_level=compression_level)
        mesh_object.vertices = vertices_seq
        plot += mesh_object
    else:
        points_object = k3d.points(positions=vertices_seq, point_size=0.01, compression_level=compression_level)
        plot += points_object

    plot.start_auto_play()

    # 4. Save HTML (often the slowest step, especially with compression)
    try:
        embed_minimal_html(output_html_path, views=[plot], title=f'{video_id}')
        return None  # None means success
    except Exception as e:
        return f"Error saving HTML for {video_id}: {e}"


def main():
    parser = argparse.ArgumentParser(description="Multi-process K3D visualization generator.")
    parser.add_argument('--input_dir', type=str, default='samples_ddp', help='Input directory.')
    parser.add_argument('--output_dir', type=str, default='samples_ddp_html', help='Output directory.')
    parser.add_argument('--compression_level', type=int, default=9, help='0-9 (9=slowest but smallest).')
    parser.add_argument('--step', type=int, default=1, help='Frame skip step.')
    parser.add_argument('--no_float16', action='store_true', help='Disable float16 optimization.')

    # workers argument
    default_workers = max(1, multiprocessing.cpu_count() - 2)  # leave 2 cores for the OS by default
    parser.add_argument('--workers', type=int, default=default_workers, help=f'Number of parallel processes (Default: {default_workers})')

    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory not found at {args.input_dir}")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    # Collect all video IDs
    video_ids = [d for d in os.listdir(args.input_dir) if os.path.isdir(os.path.join(args.input_dir, d))]
    print(f"Found {len(video_ids)} videos. Starting processing with {args.workers} workers...")

    # ProcessPoolExecutor for parallel work
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        # Submit all jobs
        futures = {
            executor.submit(process_single_video, vid, args.input_dir, args.output_dir, args.compression_level, args.step, not args.no_float16): vid
            for vid in video_ids
        }

        # Progress with tqdm
        for future in tqdm(as_completed(futures), total=len(video_ids), desc="Processing"):
            vid = futures[future]
            try:
                result = future.result()
                if result:   # non-None result means error/skip info
                    # tqdm.write avoids breaking the progress bar
                    tqdm.write(f"[{vid}] {result}")
            except Exception as e:
                tqdm.write(f"[{vid}] CRITICAL ERROR: {e}")


if __name__ == '__main__':
    # Usually unnecessary on Linux; required on some envs / Windows
    multiprocessing.set_start_method('spawn', force=True)
    main()
