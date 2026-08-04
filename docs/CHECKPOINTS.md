# CHOIR Checkpoints and Local Assets

Large checkpoints and licensed assets are not committed to Git. This document
lists the expected local locations and common overrides.

## Stage 1: Yolov8 and SAM2

The two YOLO detector checkpoints are kept in:

```text
stage1_preprocess/Yolov8/models/
├── wilor_hand_detector.pt
└── tasterob_hoi_detector.pt
```

SAM2.1 Large should be placed at:

```text
stage1_preprocess/Yolov8/sam2/checkpoints/sam2.1_hiera_large.pt
```

or provided with:

```bash
export CHOIR_SAM2_CHECKPOINT=/path/to/sam2.1_hiera_large.pt
```

## Stage 1: Dyn-HaMR, HaMeR, VIPE

Dyn-HaMR local assets are expected under:

```text
stage1_preprocess/Dyn_HaMR_new/_DATA/
stage1_preprocess/Dyn_HaMR_new/assets/
```

VIPE local checkpoints/caches are expected under:

```text
stage1_preprocess/Dyn_HaMR_new/third-party/vipe/checkpoints/
stage1_preprocess/Dyn_HaMR_new/third-party/vipe/torch_cache/
stage1_preprocess/Dyn_HaMR_new/third-party/vipe/hf_cache/
```

These paths are gitignored.

## Stage 2: DexGraspNet_table

DexGraspNet_table needs MANO/contact assets under:

```text
stage2_grasp_correction/DexGraspNet_table/grasp_generation/mano/
```

Generated meshdata and logs are local:

```text
stage2_grasp_correction/DexGraspNet_table/meshdata/
stage2_grasp_correction/DexGraspNet_table/data/
```

## Stage 2: GraspFlowMatching

Expected pretrained checkpoint:

```text
stage2_grasp_correction/GraspFlowMatching/results/050-Linear-velocity-None/checkpoints/0040000.pt
```

`results/` and sampling outputs are gitignored.

## Stage 3: diffusion-vas

Place diffusion-vas checkpoints under:

```text
stage3_temporal_optimization/diffusion-vas/checkpoints/
```

Typical checkpoint names used by the release entry:

```text
checkpoints/diffusion-vas-amodal-segmentation
checkpoints/diffusion-vas-content-completion
```

Stage 3 uses the Stage 1 MANO assets through repository-relative defaults:

```text
stage1_preprocess/Dyn_HaMR_new/_DATA/data/mano/
```

## Git Policy

Do not commit:

```text
*.pt
*.pth
*.ckpt
*.safetensors
*.pkl
data/
output/
```

Small metadata files that are required for code execution may remain in Git only
when their license permits redistribution.
