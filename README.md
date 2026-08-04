# CHOIR

CHOIR reconstructs 4D hand-object interaction (HOI) from a monocular RGB
video: hand motion, object geometry, object pose trajectory, and a physically
plausible hand-object interaction sequence.

The code is organized by the three stages used in the paper. Local input data
and generated results are intentionally kept out of Git.

## Repository Layout

```text
CHOIR-upload/
├── data/                         # local input videos, gitignored
├── output/                       # all derived results, gitignored
├── stage1_preprocess/
│   ├── Yolov8/                   # hand/object detection + SAM2 masks
│   ├── sam-3d-objects/           # object geometry and single-frame pose init
│   └── Dyn_HaMR_new/             # MANO pose sequence reconstruction
├── stage2_grasp_correction/
│   ├── DexGraspNet_table/        # grasp training data generation
│   └── GraspFlowMatching/        # hand depth-offset correction
├── stage3_temporal_optimization/
│   └── diffusion-vas/            # tracking, fitting, GFM interleave, final HOI
└── docs/
    ├── DATA_LAYOUT.md
    ├── PIPELINE.md
    └── CHECKPOINTS.md
```

## Data Contract

Put raw input videos in `data/`:

```text
data/<VIDEO_ID>.mp4
```

All intermediate and final outputs should live under:

```text
output/<VIDEO_ID>/
```

Typical per-video layout:

```text
output/<VIDEO_ID>/
├── <VIDEO_ID>.mp4
├── annotations/
├── rgbs/
├── obj_masks/
├── rh_masks/
├── lh_masks/
├── dynhamr/
├── mano_params/
├── hand_meshes/
├── optimized_hoi_init_seq/
├── grasp_correction/
└── optimized_hoi_seq/
```

See `docs/DATA_LAYOUT.md` for the full contract between modules.

## Environments

Most modules use the same `hd` environment:

```bash
conda activate /path/to/env/hd
```

This covers:

- `stage1_preprocess/Yolov8`
- `stage1_preprocess/sam-3d-objects`
- `stage2_grasp_correction/DexGraspNet_table`
- `stage2_grasp_correction/GraspFlowMatching`
- `stage3_temporal_optimization/diffusion-vas`

`stage1_preprocess/Dyn_HaMR_new` is the exception. It uses its own `dynhamr`
environment, and internally calls VIPE with a separate `vipe` environment.

## Stage 1: Preprocess and Initial Reconstruction

Stage 1 prepares object masks, object geometry, object pose initialization, and
the hand MANO sequence.

### 1. Hand/Object Detection and Masks

Module:

```text
stage1_preprocess/Yolov8/
```

Main entries:

```text
run_with_manual_object_mask.py
run_with_hoi_detector.py
labeling.py
```

The two inference entries both write masks, RGB frames, boxes, and keypoints to
`output/<VIDEO_ID>/`. Use `labeling.py` first when you want to provide a manual
first-frame object mask.

Example:

```bash
cd stage1_preprocess/Yolov8
python labeling.py --data ../../data --output ../../output --video-id <VIDEO_ID>
python run_with_manual_object_mask.py --video_id <VIDEO_ID>
```

SAM2.1 Large is expected at:

```text
stage1_preprocess/Yolov8/sam2/checkpoints/sam2.1_hiera_large.pt
```

or set:

```bash
export CHOIR_SAM2_CHECKPOINT=/path/to/sam2.1_hiera_large.pt
```

### 2. Object Geometry and Pose Initialization

Module:

```text
stage1_preprocess/sam-3d-objects/
```

Entry:

```text
run_reconstruction.py
```

Example:

```bash
cd stage1_preprocess/sam-3d-objects
python run_reconstruction.py --video_id <VIDEO_ID>
```

This stage consumes masks/RGB from `output/<VIDEO_ID>/` and writes the object
geometry and initial object sequence used by Stage 3.

### 3. Hand MANO Sequence

Module:

```text
stage1_preprocess/Dyn_HaMR_new/
```

Entry:

```text
dyn-hamr/batch_test_videos.py
```

Example:

```bash
conda activate /path/to/env/dynhamr
cd stage1_preprocess/Dyn_HaMR_new/dyn-hamr
python batch_test_videos.py --video_id <VIDEO_ID> --gpus 0
```

Important outputs:

```text
output/<VIDEO_ID>/dynhamr/
output/<VIDEO_ID>/mano_params/
output/<VIDEO_ID>/hand_meshes/
```

Dyn-HaMR assets, BMC checkpoints, VIPE checkpoints, and MANO files are local
assets and should not be committed.

## Stage 2: Grasp-Aware Hand Depth Correction

Stage 2 trains and runs GraspFlowMatching. DexGraspNet_table is used to generate
training grasps for the correction model.

### Training Data Generation

Module:

```text
stage2_grasp_correction/DexGraspNet_table/
```

Important entries:

```text
prepare_meshdata.py
grasp_generation/main_prep_data.py
grasp_generation/validate_grasping_pose.py
```

Example:

```bash
cd stage2_grasp_correction/DexGraspNet_table
python prepare_meshdata.py --video_id <VIDEO_ID>
cd grasp_generation
python main_prep_data.py --object_code_list <VIDEO_ID>
```

Local generated data is gitignored:

```text
stage2_grasp_correction/DexGraspNet_table/meshdata/
stage2_grasp_correction/DexGraspNet_table/data/
```

### GraspFlowMatching

Module:

```text
stage2_grasp_correction/GraspFlowMatching/
```

Main entries:

```text
train_cam_ray.py
sample_cam_ray_ddp.py
compute_contact_map_per_frame.py
compute_contact_map_render.py
```

Expected local pretrained checkpoint:

```text
stage2_grasp_correction/GraspFlowMatching/results/050-Linear-velocity-None/checkpoints/0040000.pt
```

Inference example:

```bash
cd stage2_grasp_correction/GraspFlowMatching
torchrun --nproc_per_node=<N> sample_cam_ray_ddp.py ODE \
  --ckpt results/050-Linear-velocity-None/checkpoints/0040000.pt \
  --output_dir samples_ddp \
  --video_id <VIDEO_ID>
```

Important output:

```text
output/<VIDEO_ID>/grasp_correction/camera_ray_depth_offset.json
```

Optional contact maps:

```bash
python compute_contact_map_per_frame.py --samples_dir samples_ddp --video_id <VIDEO_ID>
python compute_contact_map_render.py --samples_dir samples_ddp --video_id <VIDEO_ID>
```

## Stage 3: Temporal Optimization and Final HOI

Module:

```text
stage3_temporal_optimization/diffusion-vas/
```

Primary release entry:

```text
demo_fitting_5stages_sam3d_reset.py
```

Stage 3 performs:

1. object pose tracking from the Stage 1 single-frame object pose,
2. initial per-sequence hand/object optimization,
3. GraspFlowMatching pre-pass and depth-offset application,
4. final joint optimization of hand and object pose sequence.

Example:

```bash
cd stage3_temporal_optimization/diffusion-vas
python demo_fitting_5stages_sam3d_reset.py \
  --data_path ../../output \
  --data_output_path ../../output \
  --debug_index 0
```

The script writes final HOI outputs under:

```text
output/<VIDEO_ID>/optimized_hoi_seq/
```

See `docs/PIPELINE.md` for the full stage-by-stage run order.

## Checkpoints and Local Assets

Large assets are intentionally excluded from Git. See `docs/CHECKPOINTS.md` for
expected local paths and environment variables.

At minimum, prepare:

- SAM2.1 Large checkpoint for Stage 1 masks.
- Dyn-HaMR / HaMeR / VIPE local assets.
- MANO assets for Dyn-HaMR, DexGraspNet_table, and GraspFlowMatching.
- GraspFlowMatching checkpoint under `stage2_grasp_correction/GraspFlowMatching/results/`.
- diffusion-vas checkpoints under `stage3_temporal_optimization/diffusion-vas/checkpoints/`.

## Git-Ignored Runtime Content

The following are local only:

```text
data/
output/
stage1_preprocess/*/checkpoints/
stage1_preprocess/Dyn_HaMR_new/_DATA/
stage1_preprocess/Dyn_HaMR_new/assets/
stage2_grasp_correction/DexGraspNet_table/meshdata/
stage2_grasp_correction/DexGraspNet_table/data/
stage2_grasp_correction/GraspFlowMatching/results/
stage2_grasp_correction/GraspFlowMatching/samples*/
stage3_temporal_optimization/diffusion-vas/checkpoints/
stage3_temporal_optimization/diffusion-vas/input_data/
stage3_temporal_optimization/diffusion-vas/outputs/
```

Please follow the licenses of CHOIR, Dyn-HaMR, HaMeR, VIPE, SAM2,
sam-3d-objects, MANO, DexGraspNet, GraspFlowMatching dependencies, and
diffusion-vas dependencies.
