# CHOIR Data Layout

This repository uses two root-level runtime directories:

```text
data/       # raw input videos only
output/     # every derived intermediate and final result
```

Both directories are gitignored.

## Input

Use flat video IDs:

```text
data/<VIDEO_ID>.mp4
```

`<VIDEO_ID>` should be stable across all stages. All stage outputs are grouped
under `output/<VIDEO_ID>/`.

## Per-Video Output

The intended cross-module contract is:

```text
output/<VIDEO_ID>/
├── <VIDEO_ID>.mp4
├── annotations/
├── rgbs/
├── obj_masks/
├── rh_masks/
├── lh_masks/
├── *_bbox.json
├── *h_keypoints.json
├── dynhamr/
├── mano_params/
├── hand_meshes/
├── optimized_hoi_init_seq/
├── grasp_correction/
└── optimized_hoi_seq/
```

## Stage Ownership

`stage1_preprocess/Yolov8` writes:

```text
output/<VIDEO_ID>/rgbs/
output/<VIDEO_ID>/obj_masks/
output/<VIDEO_ID>/rh_masks/
output/<VIDEO_ID>/lh_masks/
output/<VIDEO_ID>/*_bbox.json
output/<VIDEO_ID>/*h_keypoints.json
```

`stage1_preprocess/sam-3d-objects` writes object geometry and pose initialization
used by Stage 3, including the files consumed later as an HOI sequence.

`stage1_preprocess/Dyn_HaMR_new` writes:

```text
output/<VIDEO_ID>/dynhamr/
output/<VIDEO_ID>/mano_params/
output/<VIDEO_ID>/hand_meshes/
```

`stage2_grasp_correction/GraspFlowMatching` writes:

```text
output/<VIDEO_ID>/grasp_correction/camera_ray_depth_offset.json
output/<VIDEO_ID>/grasp_correction/contact_map_per_frame.json   # optional
```

`stage3_temporal_optimization/diffusion-vas` writes optimization logs, debug
meshes, videos, and the final HOI sequence under:

```text
output/<VIDEO_ID>/optimized_hoi_seq/
```

## Module-Local Runtime Data

Training data, checkpoints, local caches, and debug renders should stay inside
their module and remain gitignored:

```text
stage2_grasp_correction/DexGraspNet_table/meshdata/
stage2_grasp_correction/DexGraspNet_table/data/
stage2_grasp_correction/GraspFlowMatching/results/
stage2_grasp_correction/GraspFlowMatching/samples*/
stage3_temporal_optimization/diffusion-vas/checkpoints/
stage3_temporal_optimization/diffusion-vas/outputs/
```
