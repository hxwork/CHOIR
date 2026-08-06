#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
FORCE=0

usage() {
  cat <<'EOF'
Usage:
  bash tools/download_external_assets.sh [--force]

Downloads public third-party CHOIR assets to the exact paths used by the code.
Manual/license-gated assets such as MANO, BMC arrays, and gated SAM-3D access
are reported with instructions when they cannot be fetched automatically.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --force)
      FORCE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

download_url() {
  local url=$1
  local target=$2

  if [[ -f "$target" && "$FORCE" -eq 0 ]]; then
    echo "Exists: ${target#$REPO_ROOT/}"
    return
  fi

  mkdir -p "$(dirname "$target")"
  echo "Downloading ${target#$REPO_ROOT/}"
  if command -v curl >/dev/null 2>&1; then
    curl -L --fail --retry 3 -o "$target" "$url"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "$target" "$url"
  else
    echo "Need curl or wget to download $url" >&2
    exit 1
  fi
}

hf_download_file() {
  local repo=$1
  local filename=$2
  local target_dir=$3

  mkdir -p "$target_dir"
  if command -v hf >/dev/null 2>&1; then
    hf download --repo-type model --local-dir "$target_dir" "$repo" "$filename"
  elif command -v huggingface-cli >/dev/null 2>&1; then
    huggingface-cli download "$repo" "$filename" --local-dir "$target_dir" --local-dir-use-symlinks False
  else
    return 1
  fi
}

hf_download_repo() {
  local repo=$1
  local target_dir=$2

  mkdir -p "$target_dir"
  if command -v hf >/dev/null 2>&1; then
    hf download --repo-type model --local-dir "$target_dir" --max-workers 1 "$repo"
  elif command -v huggingface-cli >/dev/null 2>&1; then
    huggingface-cli download "$repo" --local-dir "$target_dir" --local-dir-use-symlinks False
  else
    return 1
  fi
}

download_wilor_detector() {
  local url="https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/detector.pt"
  local yolo_target="$REPO_ROOT/stage1_preprocess/Yolov8/models/wilor_hand_detector.pt"
  local hamer_target="$REPO_ROOT/stage1_preprocess/Dyn_HaMR_new/third-party/hamer/pretrained_models/detector.pt"

  download_url "$url" "$yolo_target"
  mkdir -p "$(dirname "$hamer_target")"
  cp -f "$yolo_target" "$hamer_target"
  echo "Copied WiLoR detector to ${hamer_target#$REPO_ROOT/}"
}

run_dynhamr_prepare() {
  local dyn_root="$REPO_ROOT/stage1_preprocess/Dyn_HaMR_new"
  local ckpt="$dyn_root/_DATA/hamer_ckpts/checkpoints/hamer.ckpt"
  local vitpose="$dyn_root/_DATA/vitpose_ckpts/vitpose+_huge/wholebody.pth"
  local mean_params="$dyn_root/_DATA/data/mano_mean_params.npz"
  local hmp="$dyn_root/_DATA/hmp_model"

  if [[ "$FORCE" -eq 0 && -f "$ckpt" && -f "$vitpose" && -f "$mean_params" && -d "$hmp" ]]; then
    echo "Exists: Dyn-HaMR upstream assets"
    return
  fi

  if [[ ! -x "$dyn_root/scripts/prepare.sh" && ! -f "$dyn_root/scripts/prepare.sh" ]]; then
    echo "Dyn-HaMR prepare script not found: $dyn_root/scripts/prepare.sh" >&2
    return 1
  fi

  if ! command -v gdown >/dev/null 2>&1; then
    echo "Skipping Dyn-HaMR prepare.sh because gdown is not installed."
    echo "Install gdown, then run: (cd stage1_preprocess/Dyn_HaMR_new && bash scripts/prepare.sh)"
    return 0
  fi

  echo "Running Dyn-HaMR prepare.sh"
  (
    cd "$dyn_root"
    bash scripts/prepare.sh
  )
}

download_diffusion_vas() {
  local target="$REPO_ROOT/stage3_temporal_optimization/diffusion-vas/checkpoints/diffusion-vas-amodal-segmentation"
  if [[ -f "$target/model_index.json" && "$FORCE" -eq 0 ]]; then
    echo "Exists: ${target#$REPO_ROOT/}"
    return
  fi
  if ! hf_download_repo "kaihuac/diffusion-vas-amodal-segmentation" "$target"; then
    echo "Install huggingface-hub CLI, then run:"
    echo "  huggingface-cli download kaihuac/diffusion-vas-amodal-segmentation --local-dir $target"
  fi
}

download_moge() {
  local target_dir="$REPO_ROOT/stage3_temporal_optimization/diffusion-vas/checkpoints/moge-v2-vitl-normal"
  local target="$target_dir/model.pt"
  if [[ -f "$target" && "$FORCE" -eq 0 ]]; then
    echo "Exists: ${target#$REPO_ROOT/}"
    return
  fi
  if ! hf_download_file "Ruicheng/moge-2-vitl-normal" "model.pt" "$target_dir"; then
    download_url "https://huggingface.co/Ruicheng/moge-2-vitl-normal/resolve/main/model.pt" "$target"
  fi
}

download_sam3d() {
  local sam3d_root="$REPO_ROOT/stage1_preprocess/sam-3d-objects"
  local target="$sam3d_root/checkpoints/hf"
  local tmp="$sam3d_root/checkpoints/hf-download"

  if [[ -f "$target/pipeline.yaml" && "$FORCE" -eq 0 ]]; then
    echo "Exists: ${target#$REPO_ROOT/}/pipeline.yaml"
    return
  fi

  echo "Downloading SAM-3D Objects checkpoints. This requires approved Hugging Face access."
  rm -rf "$tmp"
  if hf_download_repo "facebook/sam-3d-objects" "$tmp"; then
    mkdir -p "$(dirname "$target")"
    rm -rf "$target"
    if [[ -d "$tmp/checkpoints" ]]; then
      mv "$tmp/checkpoints" "$target"
    else
      mv "$tmp" "$target"
    fi
    rm -rf "$tmp"
    echo "Installed ${target#$REPO_ROOT/}"
  else
    echo "SAM-3D automatic download skipped or failed."
    echo "Request access at https://huggingface.co/facebook/sam-3d-objects, run 'hf auth login', then retry."
  fi
}

mkdir -p "$REPO_ROOT/stage1_preprocess/Dyn_HaMR_new/third-party/vipe/checkpoints"
mkdir -p "$REPO_ROOT/stage1_preprocess/Dyn_HaMR_new/third-party/vipe/torch_cache"
mkdir -p "$REPO_ROOT/stage1_preprocess/Dyn_HaMR_new/third-party/vipe/hf_cache"

download_wilor_detector
download_url "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt" \
  "$REPO_ROOT/stage1_preprocess/Yolov8/sam2/checkpoints/sam2.1_hiera_large.pt"
run_dynhamr_prepare

download_diffusion_vas
download_url "https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth?download=true" \
  "$REPO_ROOT/stage3_temporal_optimization/diffusion-vas/checkpoints/depth_anything_v2_vitl.pth"
download_url "https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Large/resolve/main/depth_anything_v2_metric_hypersim_vitl.pth?download=true" \
  "$REPO_ROOT/stage3_temporal_optimization/diffusion-vas/checkpoints/depth_anything_v2_metric_hypersim_vitl.pth"
download_url "https://huggingface.co/facebook/cotracker3/resolve/main/scaled_offline.pth" \
  "$REPO_ROOT/stage3_temporal_optimization/diffusion-vas/checkpoints/scaled_offline.pth"
download_moge
download_sam3d

cat <<'EOF'

Manual assets still required:
  1. MANO: download MANO_RIGHT.pkl from https://mano.is.tue.mpg.de/ after accepting the license.
  2. BMC arrays: generate/copy Hand-BMC-pytorch arrays into stage1_preprocess/Dyn_HaMR_new/_DATA/BMC/.
  3. DexGraspNet/manotorch assets: provide contact_indices.json and pose_distrib.pt under
     stage2_grasp_correction/DexGraspNet_table/grasp_generation/mano/.
  4. VIPE may download additional cache files on first run depending on its configured priors.

Run validation:
  python tools/check_assets.py
EOF
