import os
import subprocess

import cv2
import numpy as np
import preproc.launch_hamer as hamer
from preproc.extract_frames import split_frame, video_to_frames
from preproc.launch_slam import check_intrins, get_command, split_frames_shots


def is_nonempty(d):
    return os.path.isdir(d) and len(os.listdir(d)) > 0


def preprocess_frames(img_dir, src_path, overwrite=False, **kwargs):
    if not overwrite and is_nonempty(img_dir):
        print(f"FOUND {len(os.listdir(img_dir))} FRAMES in {img_dir}")
        return
    print(f"EXTRACTING FRAMES FROM {src_path} TO {img_dir}")
    print(kwargs)

    # out = video_to_frames(src_path, img_dir, overwrite=overwrite, **kwargs)
    out = split_frame(src_path, img_dir, overwrite=overwrite, **kwargs)
    assert out == 0, "FAILED FRAME EXTRACTION"


def preprocess_tracks(datatype, img_dir, track_dir, shot_dir, gpu, overwrite=False):
    """
    :param img_dir
    :param track_dir, expected format: res_root/track_name/sequence
    :param shot_dir, expected format: res_root/shot_name/sequence
    """
    if not overwrite and is_nonempty(track_dir):
        print(f"FOUND TRACKS IN {track_dir}")
        return

    print(f"RUNNING HAMER ON {img_dir}")
    track_root, seq = os.path.split(track_dir.rstrip("/"))
    res_root, track_name = os.path.split(track_root)
    shot_name = shot_dir.rstrip("/").split("/")[-2]

    hamer.process_seq(
        [gpu],
        res_root,
        seq,
        img_dir,
        track_name=track_name,
        shot_name=shot_name,
        datatype=datatype,
        overwrite=overwrite,
    )


def load_vipe_cameras(vipe_dir, seq_name, img_dir, start=0, end=-1):
    """
    Load VIPE camera outputs and convert to DROID-SLAM format
    
    Args:
        vipe_dir: Path to VIPE results directory
        seq_name: Sequence name
        img_dir: Image directory to get image size
        start: Start frame index
        end: End frame index
        
    Returns:
        w2c: (N, 4, 4) world-to-camera matrices
        intrins_full: (N, 6) intrinsics [fx, fy, cx, cy, W, H]
    """
    # Load VIPE pose and intrinsics
    vipe_pose_path = os.path.join(vipe_dir, "pose", f"{seq_name}.npz")
    vipe_intrins_path = os.path.join(vipe_dir, "intrinsics", f"{seq_name}.npz")

    if not os.path.exists(vipe_pose_path):
        raise FileNotFoundError(f"VIPE pose file not found: {vipe_pose_path}")
    if not os.path.exists(vipe_intrins_path):
        raise FileNotFoundError(f"VIPE intrinsics file not found: {vipe_intrins_path}")

    print(f"Loading VIPE cameras from {vipe_dir}")
    pose_data = np.load(vipe_pose_path)
    c2w = pose_data['data']  # (N, 4, 4) camera-to-world
    pose_inds = pose_data['inds']  # (N,) frame indices

    intrins_data = np.load(vipe_intrins_path)
    intrins = intrins_data['data']  # (N, 4) [fx, fy, cx, cy]
    intrins_inds = intrins_data['inds']  # (N,) frame indices

    # Verify indices match
    assert np.array_equal(pose_inds, intrins_inds), "Pose and intrinsics indices don't match!"

    # Get image size from first image
    image_files = sorted([f for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    if not image_files:
        # Infer from intrinsics (cx, cy should be roughly at center)
        img_width = int(intrins[0, 2] * 2)
        img_height = int(intrins[0, 3] * 2)
        print(f"Inferred image size from intrinsics: {img_width}x{img_height}")
    else:
        img_path = os.path.join(img_dir, image_files[0])
        img = cv2.imread(img_path)
        img_height, img_width = img.shape[:2]
        print(f"Got image size from images: {img_width}x{img_height}")

    # Select frames based on start/end
    if end < 0:
        end = len(c2w)
    c2w = c2w[start:end]
    intrins = intrins[start:end]

    # Convert camera-to-world to world-to-camera
    w2c = np.linalg.inv(c2w)

    # Add width and height to intrinsics
    N = len(w2c)
    intrins_full = np.zeros((N, 6), dtype=np.float32)
    intrins_full[:, :4] = intrins
    intrins_full[:, 4] = img_width
    intrins_full[:, 5] = img_height

    print(f"Loaded {N} VIPE camera poses")
    return w2c, intrins_full


def save_vipe_cameras_as_droid(output_dir, w2c, intrins_full):
    """
    Save VIPE cameras in DROID-SLAM format
    
    Args:
        output_dir: Output directory
        w2c: (N, 4, 4) world-to-camera matrices
        intrins_full: (N, 6) intrinsics [fx, fy, cx, cy, W, H]
    """
    os.makedirs(output_dir, exist_ok=True)

    # Extract parameters
    W, H = intrins_full[0, 4], intrins_full[0, 5]
    focal = intrins_full[:, :2].mean()

    print(f"Saving VIPE cameras to {output_dir}")
    print(f"Image size: {int(W)}x{int(H)}, focal: {focal:.2f}")

    # Save cameras.npz (main format used by Dyn-HaMR)
    np.savez(
        f"{output_dir}/cameras.npz",
        height=H,
        width=W,
        focal=focal,
        intrins=intrins_full[:, :4],
        w2c=w2c,
    )
    print(f"Saved cameras.npz with {len(w2c)} frames")


def run_vipe(video_path, vipe_dir, vipe_root):
    """
    Run VIPE camera estimation on a video
    
    Args:
        video_path: Path to input video
        vipe_dir: Directory where VIPE results will be saved
        vipe_root: Root directory of VIPE installation
    
    Returns:
        0 if successful, non-zero otherwise
    """
    print(f"Running VIPE on {video_path}")
    print(f"VIPE root: {vipe_root}")
    print(f"Results will be saved to: {vipe_dir}")

    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', None)
    if cuda_visible is not None:
        print(f"Using CUDA_VISIBLE_DEVICES={cuda_visible} for VIPE")

    # Prefer an explicit env path; otherwise activate a conda env named "vipe".
    conda_sh = os.path.expanduser("~/miniconda3/etc/profile.d/conda.sh")
    if not os.path.exists(conda_sh):
        conda_sh = os.path.expanduser("~/anaconda3/etc/profile.d/conda.sh")

    vipe_env = os.environ.get("CHOIR_VIPE_ENV", "vipe")

    # Write VIPE results directly into the per-video output folder.
    os.makedirs(vipe_dir, exist_ok=True)
    # Use in-repo torch/hf caches. Do NOT set TRANSFORMERS_CACHE — it breaks the
    # HF hub layout under HF_HOME/hub (bert-base-uncased etc. become invisible).
    # Offline + no proxy: use local weights when the network/proxy is unavailable.
    cache_exports = (
        "export TORCH_HOME=$(pwd)/torch_cache && "
        "export HF_HOME=$(pwd)/hf_cache && "
        "export HF_HUB_OFFLINE=1 && "
        "export TRANSFORMERS_OFFLINE=1 && "
        "unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy"
    )
    if cuda_visible is not None:
        cmd = (
            f"source {conda_sh} && conda activate {vipe_env} && cd {vipe_root} && "
            f"export CUDA_VISIBLE_DEVICES={cuda_visible} && "
            f"{cache_exports} && "
            f"vipe infer {video_path} --output {vipe_dir}"
        )
    else:
        cmd = (
            f"source {conda_sh} && conda activate {vipe_env} && cd {vipe_root} && "
            f"{cache_exports} && "
            f"vipe infer {video_path} --output {vipe_dir}"
        )

    print(f"Executing: {cmd}")

    env = os.environ.copy()
    out = subprocess.call(cmd, shell=True, executable="/bin/bash", env=env)

    if out != 0:
        print(f"WARNING: VIPE failed with exit code {out}")
    else:
        print("VIPE completed successfully")

    return out


def _copy_vipe_results(src_vipe_dir, dst_vipe_dir, seq, alt_stems=()):
    """Copy VIPE results from src to dst for the given sequence.

    VIPE names outputs after the input video stem (e.g. inputs/video.mp4 -> video.npz).
    Dyn-HaMR expects {seq}.npz, so also try alt_stems and normalize into {seq}.*.
    """
    import shutil

    stems = (seq,) + tuple(s for s in alt_stems if s and s != seq)
    for sub in ("pose", "intrinsics"):
        dst = os.path.join(dst_vipe_dir, sub, f"{seq}.npz")
        if os.path.exists(dst):
            continue
        for stem in stems:
            src = os.path.join(src_vipe_dir, sub, f"{stem}.npz")
            if os.path.exists(src):
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(src, dst)
                print(f"Copied VIPE result: {src} -> {dst}")
                break


def _normalize_vipe_seq_names(vipe_dir, seq, alt_stems=()):
    """Ensure pose/intrinsics (and sibling artifacts) exist under {seq}.* names."""
    import shutil

    if not os.path.isdir(vipe_dir):
        return
    stems = tuple(s for s in alt_stems if s and s != seq)
    if not stems:
        return

    for sub in ("pose", "intrinsics", "depth", "rgb", "mask", "vipe"):
        sub_dir = os.path.join(vipe_dir, sub)
        if not os.path.isdir(sub_dir):
            continue
        for name in os.listdir(sub_dir):
            stem, ext = os.path.splitext(name)
            # Handle names like video_camera.txt / video_info.pkl
            base = stem
            suffix = ""
            for alt in stems:
                if stem == alt:
                    break
                if stem.startswith(alt + "_"):
                    suffix = stem[len(alt) :]
                    base = alt
                    break
            else:
                continue
            if base not in stems:
                continue
            dst_name = f"{seq}{suffix}{ext}"
            src_path = os.path.join(sub_dir, name)
            dst_path = os.path.join(sub_dir, dst_name)
            if os.path.exists(dst_path):
                continue
            shutil.copy2(src_path, dst_path)
            print(f"Normalized VIPE name: {src_path} -> {dst_path}")


def preprocess_cameras(cfg, overwrite=False):
    if not overwrite and is_nonempty(cfg.sources.cameras):
        print(f"FOUND CAMERAS IN {cfg.sources.cameras}")
        return

    # Check if we should use VIPE instead of DROID-SLAM
    use_vipe = cfg.get("use_vipe", False)
    vipe_dir = cfg.get("vipe_dir", None)

    if use_vipe and vipe_dir is not None:
        # vipe_root: VIPE installation directory (separate from where results are stored)
        # Falls back to parent of vipe_dir for backward compatibility
        vipe_root = cfg.get("vipe_root", os.path.dirname(vipe_dir))
        vipe_root = os.path.abspath(vipe_root)
        vipe_dir = os.path.abspath(vipe_dir)
        default_vipe_results = os.path.join(vipe_root, "vipe_results")
        video_path = cfg.get("src_path", None)
        video_stem = (
            os.path.splitext(os.path.basename(video_path))[0]
            if video_path
            else "video"
        )
        alt_stems = (video_stem, "video")

        # Check if VIPE results exist for this sequence
        vipe_pose_path = os.path.join(vipe_dir, "pose", f"{cfg.seq}.npz")
        vipe_intrins_path = os.path.join(vipe_dir, "intrinsics", f"{cfg.seq}.npz")

        # Prefer already-written per-video outputs named after the video stem.
        _normalize_vipe_seq_names(vipe_dir, cfg.seq, alt_stems=alt_stems)

        # If not at per-video location, check the VIPE installation's default output dir
        if not (os.path.exists(vipe_pose_path) and os.path.exists(vipe_intrins_path)):
            _copy_vipe_results(default_vipe_results, vipe_dir, cfg.seq, alt_stems=alt_stems)
            _normalize_vipe_seq_names(vipe_dir, cfg.seq, alt_stems=alt_stems)

        # If still not found, run VIPE
        if not (os.path.exists(vipe_pose_path) and os.path.exists(vipe_intrins_path)):
            print(f"VIPE results not found for sequence '{cfg.seq}', running VIPE...")

            if video_path is None or not os.path.exists(video_path):
                raise FileNotFoundError(f"Video path not found: {video_path}\n"
                                        f"Cannot run VIPE. Please provide a valid 'src_path' in your config.")

            out = run_vipe(video_path, vipe_dir, vipe_root)
            if out != 0:
                raise RuntimeError(
                    f"VIPE failed with exit code {out}\n"
                    f"Please check VIPE installation and try running manually:\n"
                    f"  conda activate ${{CHOIR_VIPE_ENV:-vipe}}\n"
                    f"  cd {vipe_root}\n"
                    f"  vipe infer {video_path} --output {vipe_dir}"
                )

            # Copy from VIPE's default output location if it still wrote there
            _copy_vipe_results(default_vipe_results, vipe_dir, cfg.seq, alt_stems=alt_stems)
            _normalize_vipe_seq_names(vipe_dir, cfg.seq, alt_stems=alt_stems)

        # Load VIPE results (after potentially running VIPE)
        if not (os.path.exists(vipe_pose_path) and os.path.exists(vipe_intrins_path)):
            raise FileNotFoundError(f"VIPE results not found after execution:\n"
                                    f"  Pose: {vipe_pose_path}\n"
                                    f"  Intrinsics: {vipe_intrins_path}\n"
                                    f"VIPE may have failed silently. Please check VIPE logs.")

        print(f"USING VIPE CAMERAS FROM {vipe_dir}")
        img_dir = cfg.sources.images
        map_dir = cfg.sources.cameras

        # Get frame range
        subseqs, shot_idcs = split_frames_shots(cfg.sources.images, cfg.sources.shots)
        shot_idx = np.where(shot_idcs == cfg.shot_idx)[0][0]
        start, end = subseqs[shot_idx]

        if not cfg.split_cameras:
            # only run on specified segment within shot
            end = start + cfg.end_idx
            start = start + cfg.start_idx

        # Load and convert VIPE cameras
        w2c, intrins_full = load_vipe_cameras(vipe_dir, cfg.seq, img_dir, start, end)
        save_vipe_cameras_as_droid(map_dir, w2c, intrins_full)
        return

    # Default: use DROID-SLAM
    print(f"RUNNING SLAM ON {cfg.seq}")
    img_dir = cfg.sources.images
    map_dir = cfg.sources.cameras
    subseqs, shot_idcs = split_frames_shots(cfg.sources.images, cfg.sources.shots)
    print(shot_idcs, cfg.shot_idx, np.where(shot_idcs == cfg.shot_idx), cfg.sources.images, cfg.sources.shots)
    print(subseqs)
    shot_idx = np.where(shot_idcs == cfg.shot_idx)[0][0]
    # run on selected shot
    start, end = subseqs[shot_idx]
    if not cfg.split_cameras:
        # only run on specified segment within shot
        end = start + cfg.end_idx
        start = start + cfg.start_idx
    intrins_path = cfg.sources.get("intrins", None)
    if intrins_path is not None:
        intrins_path = check_intrins(cfg.type, cfg.root, intrins_path, cfg.seq, cfg.split)

    print('img_dir, map_dir, start, end, intrins_path', img_dir, map_dir, start, end, intrins_path)
    # raise ValueERRROR
    cmd = get_command(
        img_dir,
        map_dir,
        start=start,
        end=end,
        intrins_path=intrins_path,
        overwrite=overwrite,
    )
    print(cmd)
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", 0)
    out = subprocess.call(f"CUDA_VISIBLE_DEVICES={gpu} {cmd}", shell=True)
    assert out == 0, "SLAM FAILED"
