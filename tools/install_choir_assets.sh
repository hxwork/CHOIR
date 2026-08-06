#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash tools/install_choir_assets.sh /path/to/choir_release_assets.zip

The zip should contain CHOIR-owned/internal weights at repository-relative paths:
  stage1_preprocess/Yolov8/models/tasterob_hoi_detector.pt
  stage2_grasp_correction/GraspFlowMatching/results/050-Linear-velocity-None/checkpoints/0040000.pt
EOF
}

if [[ $# -ne 1 || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 1
fi

ZIP_PATH=$1
if [[ ! -f "$ZIP_PATH" ]]; then
  echo "Asset zip not found: $ZIP_PATH" >&2
  exit 1
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
TMP_DIR=$(mktemp -d)
trap 'rm -rf "$TMP_DIR"' EXIT

unzip -q "$ZIP_PATH" -d "$TMP_DIR"

copy_required_asset() {
  local rel_path=$1
  local src=""

  if [[ -f "$TMP_DIR/$rel_path" ]]; then
    src="$TMP_DIR/$rel_path"
  elif [[ -f "$TMP_DIR/CHOIR/$rel_path" ]]; then
    src="$TMP_DIR/CHOIR/$rel_path"
  fi

  if [[ -z "$src" ]]; then
    echo "Missing required internal asset in zip: $rel_path" >&2
    exit 1
  fi

  mkdir -p "$REPO_ROOT/$(dirname "$rel_path")"
  cp -f "$src" "$REPO_ROOT/$rel_path"
  echo "Installed $rel_path"
}

copy_required_asset "stage1_preprocess/Yolov8/models/tasterob_hoi_detector.pt"
copy_required_asset "stage2_grasp_correction/GraspFlowMatching/results/050-Linear-velocity-None/checkpoints/0040000.pt"

echo
python "$SCRIPT_DIR/check_assets.py" --repo-root "$REPO_ROOT" || true
echo
echo "Internal CHOIR assets installed. Install external/manual assets next if any are still missing."
