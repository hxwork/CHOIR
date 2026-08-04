# CHOIR Pipeline

This document describes the release workflow at the module boundary level. Each
module keeps its own internal structure; run commands from the module root unless
noted otherwise.

## Stage 1: Preprocess and Initial Reconstruction

### 1. Optional Manual Object Prompt

```bash
cd stage1_preprocess/Yolov8
python labeling.py --data ../../data --output ../../output --video-id <VIDEO_ID>
```

This creates:

```text
output/<VIDEO_ID>/annotations/first_mask.png
```

### 2. Hand/Object Detection and Mask Propagation

Manual object mask path:

```bash
cd stage1_preprocess/Yolov8
python run_with_manual_object_mask.py --video_id <VIDEO_ID>
```

HOI detector path:

```bash
cd stage1_preprocess/Yolov8
python run_with_hoi_detector.py --video_id <VIDEO_ID>
```

### 3. Object Geometry and Initial Pose

```bash
cd stage1_preprocess/sam-3d-objects
python run_reconstruction.py --video_id <VIDEO_ID>
```

### 4. MANO Sequence

```bash
conda activate /path/to/env/dynhamr
cd stage1_preprocess/Dyn_HaMR_new/dyn-hamr
python batch_test_videos.py --video_id <VIDEO_ID> --gpus 0
```

## Stage 2: GraspFlowMatching

### 1. Prepare Meshdata

```bash
cd stage2_grasp_correction/DexGraspNet_table
python prepare_meshdata.py --video_id <VIDEO_ID>
```

If `init_obj_poses.npy` is missing:

```bash
cd stage2_grasp_correction/DexGraspNet_table/grasp_generation
python scripts/generate_object_pose_mine.py --data_root_path ../meshdata
```

### 2. Generate Training Grasps

```bash
cd stage2_grasp_correction/DexGraspNet_table/grasp_generation
python main_prep_data.py --object_code_list <VIDEO_ID>
```

### 3. Run Depth-Offset Correction

```bash
cd stage2_grasp_correction/GraspFlowMatching
torchrun --nproc_per_node=<N> sample_cam_ray_ddp.py ODE \
  --ckpt results/050-Linear-velocity-None/checkpoints/0040000.pt \
  --output_dir samples_ddp \
  --video_id <VIDEO_ID>
```

Optional contact maps:

```bash
python compute_contact_map_per_frame.py --samples_dir samples_ddp --video_id <VIDEO_ID>
python compute_contact_map_render.py --samples_dir samples_ddp --video_id <VIDEO_ID>
```

## Stage 3: Temporal Optimization

```bash
cd stage3_temporal_optimization/diffusion-vas
python demo_fitting_5stages_sam3d_reset.py \
  --data_path ../../output \
  --data_output_path ../../output \
  --debug_index 0
```

When `camera_ray_depth_offset.json` is missing, the Stage 3 entry can auto-run
the GraspFlowMatching pre-pass using the repository-relative Stage 2 path.

## Final Output

The final HOI sequence is expected under:

```text
output/<VIDEO_ID>/optimized_hoi_seq/
```
