#!/usr/bin/env python3
"""Validate CHOIR checkpoint and asset placement."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import NamedTuple


class Asset(NamedTuple):
    key: str
    label: str
    path: str
    category: str
    source: str
    required: bool = True


class MissingAsset(NamedTuple):
    asset: Asset
    target: str
    hint: str


ASSETS = (
    Asset(
        "tasterob_hoi_detector",
        "TasteRob hand-object detector",
        "stage1_preprocess/Yolov8/models/tasterob_hoi_detector.pt",
        "internal",
        "Install from choir_release_assets.zip.",
    ),
    Asset(
        "gfm_checkpoint",
        "GraspFlowMatching CHOIR checkpoint",
        "stage2_grasp_correction/GraspFlowMatching/results/050-Linear-velocity-None/checkpoints/0040000.pt",
        "internal",
        "Install from choir_release_assets.zip.",
    ),
    Asset(
        "wilor_hand_detector",
        "WiLoR hand detector for Stage 1 manual-mask mode",
        "stage1_preprocess/Yolov8/models/wilor_hand_detector.pt",
        "external",
        "https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/detector.pt",
    ),
    Asset(
        "hamer_yolo_detector",
        "WiLoR/HaMeR detector copy for Dyn-HaMR",
        "stage1_preprocess/Dyn_HaMR_new/third-party/hamer/pretrained_models/detector.pt",
        "external",
        "https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/detector.pt",
    ),
    Asset(
        "sam2_hiera_large",
        "SAM2.1 Hiera-L checkpoint",
        "stage1_preprocess/Yolov8/sam2/checkpoints/sam2.1_hiera_large.pt",
        "external",
        "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt",
    ),
    Asset(
        "dynhamr_hamer_checkpoint",
        "HaMeR checkpoint used by Dyn-HaMR",
        "stage1_preprocess/Dyn_HaMR_new/_DATA/hamer_ckpts/checkpoints/hamer.ckpt",
        "external",
        "Run stage1_preprocess/Dyn_HaMR_new/scripts/prepare.sh or tools/download_external_assets.sh.",
    ),
    Asset(
        "dynhamr_vitpose_checkpoint",
        "ViTPose whole-body checkpoint used by HaMeR",
        "stage1_preprocess/Dyn_HaMR_new/_DATA/vitpose_ckpts/vitpose+_huge/wholebody.pth",
        "external",
        "Run stage1_preprocess/Dyn_HaMR_new/scripts/prepare.sh or tools/download_external_assets.sh.",
    ),
    Asset(
        "dynhamr_mano_mean_params",
        "HaMeR MANO mean parameters",
        "stage1_preprocess/Dyn_HaMR_new/_DATA/data/mano_mean_params.npz",
        "external",
        "Run stage1_preprocess/Dyn_HaMR_new/scripts/prepare.sh or tools/download_external_assets.sh.",
    ),
    Asset(
        "dynhamr_hmp_model",
        "HMP motion-prior assets",
        "stage1_preprocess/Dyn_HaMR_new/_DATA/hmp_model",
        "external",
        "Run stage1_preprocess/Dyn_HaMR_new/scripts/prepare.sh or tools/download_external_assets.sh.",
    ),
    Asset(
        "dynhamr_droid_checkpoint",
        "DROID-SLAM checkpoint retained for Dyn-HaMR compatibility",
        "stage1_preprocess/Dyn_HaMR_new/_DATA/droid.pth",
        "external",
        "Run stage1_preprocess/Dyn_HaMR_new/scripts/prepare.sh or tools/download_external_assets.sh.",
    ),
    Asset(
        "mano_right_dynhamr_models",
        "MANO right hand model for Dyn-HaMR and Stage 3",
        "stage1_preprocess/Dyn_HaMR_new/_DATA/data/mano/MANO_RIGHT.pkl",
        "manual",
        "Register and download from https://mano.is.tue.mpg.de/.",
    ),
    Asset(
        "mano_right_dexgrasp",
        "MANO right hand model for DexGraspNet/GFM",
        "stage2_grasp_correction/DexGraspNet_table/grasp_generation/mano/MANO_RIGHT.pkl",
        "manual",
        "Copy from the MANO download after accepting the MANO license.",
    ),
    Asset(
        "mano_right_dexgrasp_models",
        "MANO right hand model compatibility copy",
        "stage2_grasp_correction/DexGraspNet_table/grasp_generation/mano/models/MANO_RIGHT.pkl",
        "manual",
        "Copy from the MANO download after accepting the MANO license.",
    ),
    Asset(
        "dex_contact_indices",
        "DexGraspNet contact vertex indices",
        "stage2_grasp_correction/DexGraspNet_table/grasp_generation/mano/contact_indices.json",
        "manual",
        "Use the DexGraspNet/manotorch asset bundled with that project, if its license permits.",
    ),
    Asset(
        "dex_pose_distrib",
        "DexGraspNet hand-pose prior distribution",
        "stage2_grasp_correction/DexGraspNet_table/grasp_generation/mano/pose_distrib.pt",
        "manual",
        "Use the DexGraspNet/manotorch asset bundled with that project, if its license permits.",
    ),
    Asset(
        "bmc_convex_hulls",
        "BMC biomechanical constraint arrays",
        "stage1_preprocess/Dyn_HaMR_new/_DATA/BMC/CONVEX_HULLS.npy",
        "manual",
        "Generate or copy BMC arrays following Hand-BMC-pytorch instructions.",
    ),
    Asset(
        "bmc_bone_len_max",
        "BMC bone length maximums",
        "stage1_preprocess/Dyn_HaMR_new/_DATA/BMC/bone_len_max.npy",
        "manual",
        "Generate or copy BMC arrays following Hand-BMC-pytorch instructions.",
    ),
    Asset(
        "bmc_bone_len_min",
        "BMC bone length minimums",
        "stage1_preprocess/Dyn_HaMR_new/_DATA/BMC/bone_len_min.npy",
        "manual",
        "Generate or copy BMC arrays following Hand-BMC-pytorch instructions.",
    ),
    Asset(
        "bmc_curvatures_max",
        "BMC curvature maximums",
        "stage1_preprocess/Dyn_HaMR_new/_DATA/BMC/curvatures_max.npy",
        "manual",
        "Generate or copy BMC arrays following Hand-BMC-pytorch instructions.",
    ),
    Asset(
        "bmc_curvatures_min",
        "BMC curvature minimums",
        "stage1_preprocess/Dyn_HaMR_new/_DATA/BMC/curvatures_min.npy",
        "manual",
        "Generate or copy BMC arrays following Hand-BMC-pytorch instructions.",
    ),
    Asset(
        "bmc_joint_angles",
        "BMC joint angle constraints",
        "stage1_preprocess/Dyn_HaMR_new/_DATA/BMC/joint_angles.npy",
        "manual",
        "Generate or copy BMC arrays following Hand-BMC-pytorch instructions.",
    ),
    Asset(
        "bmc_phi_max",
        "BMC PHI maximums",
        "stage1_preprocess/Dyn_HaMR_new/_DATA/BMC/PHI_max.npy",
        "manual",
        "Generate or copy BMC arrays following Hand-BMC-pytorch instructions.",
    ),
    Asset(
        "bmc_phi_min",
        "BMC PHI minimums",
        "stage1_preprocess/Dyn_HaMR_new/_DATA/BMC/PHI_min.npy",
        "manual",
        "Generate or copy BMC arrays following Hand-BMC-pytorch instructions.",
    ),
    Asset(
        "diffusion_vas_amodal",
        "Diffusion-VAS amodal segmentation checkpoint",
        "stage3_temporal_optimization/diffusion-vas/checkpoints/diffusion-vas-amodal-segmentation/model_index.json",
        "external",
        "https://huggingface.co/kaihuac/diffusion-vas-amodal-segmentation",
    ),
    Asset(
        "depth_anything_v2_vitl",
        "Depth Anything V2 ViT-L checkpoint",
        "stage3_temporal_optimization/diffusion-vas/checkpoints/depth_anything_v2_vitl.pth",
        "external",
        "https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth",
    ),
    Asset(
        "depth_anything_metric_hypersim_vitl",
        "Depth Anything V2 metric Hypersim ViT-L checkpoint",
        "stage3_temporal_optimization/diffusion-vas/checkpoints/depth_anything_v2_metric_hypersim_vitl.pth",
        "external",
        "https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Large/resolve/main/depth_anything_v2_metric_hypersim_vitl.pth",
        required=False,
    ),
    Asset(
        "cotracker_scaled_offline",
        "CoTracker3 scaled offline checkpoint",
        "stage3_temporal_optimization/diffusion-vas/checkpoints/scaled_offline.pth",
        "external",
        "https://huggingface.co/facebook/cotracker3/resolve/main/scaled_offline.pth",
    ),
    Asset(
        "moge_v2_vitl_normal",
        "MoGe-2 ViT-L normal checkpoint",
        "stage3_temporal_optimization/diffusion-vas/checkpoints/moge-v2-vitl-normal/model.pt",
        "external",
        "https://huggingface.co/Ruicheng/moge-2-vitl-normal",
    ),
    Asset(
        "sam3d_pipeline",
        "SAM-3D Objects Hugging Face pipeline",
        "stage1_preprocess/sam-3d-objects/checkpoints/hf/pipeline.yaml",
        "external",
        "https://huggingface.co/facebook/sam-3d-objects",
    ),
)


def hint_for(asset: Asset) -> str:
    if asset.category == "internal":
        return (
            "Install from choir_release_assets.zip with "
            "`bash tools/install_choir_assets.sh /path/to/choir_release_assets.zip`."
        )
    if asset.category == "manual":
        return f"Manual step required. {asset.source}"
    return f"Download with `bash tools/download_external_assets.sh`. Source: {asset.source}"


def collect_asset_status(repo_root: Path) -> list[MissingAsset]:
    repo_root = repo_root.resolve()
    missing = []
    for asset in ASSETS:
        target = repo_root / asset.path
        if not target.exists() and asset.required:
            missing.append(MissingAsset(asset=asset, target=asset.path, hint=hint_for(asset)))
    return missing


def print_manifest() -> None:
    for category in ("internal", "external", "manual"):
        print(f"{category}:")
        for asset in ASSETS:
            if asset.category != category:
                continue
            marker = "required" if asset.required else "optional"
            print(f"  - {asset.key} ({marker}): {asset.path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--list", action="store_true", help="Print the expected asset manifest.")
    args = parser.parse_args(argv)

    if args.list:
        print_manifest()
        return 0

    missing = collect_asset_status(args.repo_root)
    if not missing:
        print("All required CHOIR assets are present.")
        return 0

    print("Missing required CHOIR assets:")
    for item in missing:
        print(f"- {item.asset.key}: {item.asset.label}")
        print(f"  target: {item.target}")
        print(f"  category: {item.asset.category}")
        print(f"  hint: {item.hint}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
