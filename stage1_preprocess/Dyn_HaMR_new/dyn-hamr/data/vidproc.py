import os
import numpy as np
import subprocess
import cv2

import preproc.launch_hamer as hamer
from preproc.launch_slam import split_frames_shots, get_command, check_intrins
from preproc.extract_frames import video_to_frames, split_frame


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
    image_files = sorted([f for f in os.listdir(img_dir) 
                         if f.endswith(('.png', '.jpg', '.jpeg'))])
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
    
    # Build VIPE command with proper conda activation
    # We need to source conda.sh first to make conda activate work in subprocess
    conda_sh = os.path.expanduser("~/miniconda3/etc/profile.d/conda.sh")
    if not os.path.exists(conda_sh):
        conda_sh = os.path.expanduser("~/anaconda3/etc/profile.d/conda.sh")
    
    cmd = f"source {conda_sh} && conda activate vipe && cd {vipe_root} && vipe infer {video_path}"
    
    print(f"Executing: {cmd}")
    out = subprocess.call(cmd, shell=True, executable="/bin/bash")
    
    if out != 0:
        print(f"WARNING: VIPE failed with exit code {out}")
    else:
        print(f"✓ VIPE completed successfully")
    
    return out


def preprocess_cameras(cfg, overwrite=False):
    if not overwrite and is_nonempty(cfg.sources.cameras):
        print(f"FOUND CAMERAS IN {cfg.sources.cameras}")
        return

    # Check if we should use VIPE instead of DROID-SLAM
    use_vipe = cfg.get("use_vipe", False)
    vipe_dir = cfg.get("vipe_dir", None)
    
    if use_vipe and vipe_dir is not None:
        # Check if VIPE results exist for this sequence
        vipe_pose_path = os.path.join(vipe_dir, "pose", f"{cfg.seq}.npz")
        vipe_intrins_path = os.path.join(vipe_dir, "intrinsics", f"{cfg.seq}.npz")
        
        # If VIPE results don't exist, run VIPE
        if not (os.path.exists(vipe_pose_path) and os.path.exists(vipe_intrins_path)):
            print(f"VIPE results not found for sequence '{cfg.seq}', running VIPE...")
            
            # Get VIPE root directory (parent of vipe_dir) and video path
            vipe_root = os.path.dirname(vipe_dir)
            video_path = cfg.get("src_path", None)
            
            if video_path is None or not os.path.exists(video_path):
                raise FileNotFoundError(
                    f"Video path not found: {video_path}\n"
                    f"Cannot run VIPE. Please provide a valid 'src_path' in your config."
                )
            
            # Run VIPE
            out = run_vipe(video_path, vipe_dir, vipe_root)
            if out != 0:
                raise RuntimeError(
                    f"VIPE failed with exit code {out}\n"
                    f"Please check VIPE installation and try running manually:\n"
                    f"  conda activate vipe\n"
                    f"  cd {vipe_root}\n"
                    f"  vipe infer {video_path}"
                )
        
        # Load VIPE results (after potentially running VIPE)
        if not (os.path.exists(vipe_pose_path) and os.path.exists(vipe_intrins_path)):
            raise FileNotFoundError(
                f"VIPE results not found after execution:\n"
                f"  Pose: {vipe_pose_path}\n"
                f"  Intrinsics: {vipe_intrins_path}\n"
                f"VIPE may have failed silently. Please check VIPE logs."
            )
        
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
