# Third-Party Notices

This repository includes CHOIR-authored code together with vendored third-party
components and interfaces to external assets. CHOIR-authored code is released
under the MIT License in `LICENSE`. Third-party components, model weights,
datasets, and MANO assets remain subject to their own licenses and usage terms.

Important third-party components include, but are not limited to:

- Dyn-HaMR, HaMeR, ViTPose, DROID-SLAM, VIPE, and their bundled dependencies
  under `stage1_preprocess/Dyn_HaMR_new/`.
- SAM2 / Ultralytics-based detection and segmentation code under
  `stage1_preprocess/Yolov8/`.
- SAM-3D Objects, MoGe, DINOv2, mip-splatting, and gaussian-splatting-related
  code under `stage1_preprocess/sam-3d-objects/`.
- DexGraspNet, GraspFlowMatching, and manotorch-related code under
  `stage2_grasp_correction/`.
- Depth Anything V2, torch-mesh-intersection, and related geometry/vision
  dependencies under `stage3_temporal_optimization/diffusion-vas/`.

Some vendored components have licenses that are more restrictive than MIT. For
example, the vendored manotorch copy is GPL-3.0, and gaussian-splatting-related
code is licensed for non-commercial research/evaluation use. Review the license
files in the corresponding subdirectories before redistribution or commercial
use.

Manual assets such as MANO files, DexGraspNet/manotorch hand assets, pretrained
checkpoints, and datasets are not covered by the CHOIR MIT License. Obtain them
from their official sources and follow their respective terms.
