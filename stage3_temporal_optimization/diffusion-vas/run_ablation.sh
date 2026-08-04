#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT="demo_fitting_5stages_sam3d_reset_ablation.py"
DATA_PATH="${DATA_PATH:-input_data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/ablations}"
CACHE_SOURCE_ROOT="${CACHE_SOURCE_ROOT:-$DATA_PATH}"
SEED_CACHE="${SEED_CACHE:-1}"
SAVE_INTERMEDIATES="${SAVE_INTERMEDIATES:-0}"
OVERWRITE_STAGE1_3="${OVERWRITE_STAGE1_3:-0}"
OVERWRITE_GRASP_CORRECTION="${OVERWRITE_GRASP_CORRECTION:-0}"
OVERWRITE_SAM3D_DENSE_CACHE="${OVERWRITE_SAM3D_DENSE_CACHE:-0}"
SAM3D_REF_KEYFRAME_INIT="${SAM3D_REF_KEYFRAME_INIT:-0}"
SAM3D_SPARSE_KEYFRAMES="${SAM3D_SPARSE_KEYFRAMES:-0}"
STAGE5_MODE="${STAGE5_MODE:-}"

ABLATION="${1:-full}"
if [[ $# -gt 0 ]]; then
  shift
fi

if [[ $# -gt 0 ]]; then
  VIDEO_IDS=("$@")
else
  # n=1 smoke test: one TasteRob numeric video_id + one self-captured IMG_ video_id.
  VIDEO_IDS=("594" "IMG_5233")
fi

case "$ABLATION" in
  full|no_fp|no_vp|no_stage4|no_stage4_offset|no_penetration|no_stage5_pen|no_contact|no_dyn_contact|no_stage5_contact|no_stage5_smooth|stage3_raw_metric|post_gfm_metric)
    ;;
  *)
    echo "Unknown ablation: $ABLATION" >&2
    echo "Choices: full no_fp no_vp no_stage4 no_penetration no_contact no_stage4_offset no_stage5_pen no_dyn_contact no_stage5_contact no_stage5_smooth stage3_raw_metric post_gfm_metric" >&2
    exit 2
    ;;
esac

echo "Ablation: $ABLATION"
echo "Videos: ${VIDEO_IDS[*]}"
echo "Output: $OUTPUT_ROOT/$ABLATION"
echo "Cache source: $CACHE_SOURCE_ROOT (SEED_CACHE=$SEED_CACHE)"
echo "Save intermediates: $SAVE_INTERMEDIATES"
echo "Overwrite stage1_3: $OVERWRITE_STAGE1_3"
echo "Overwrite grasp correction: $OVERWRITE_GRASP_CORRECTION"
echo "Overwrite SAM3D dense cache: $OVERWRITE_SAM3D_DENSE_CACHE"
echo "SAM3D ref keyframe init: $SAM3D_REF_KEYFRAME_INIT"
echo "SAM3D sparse keyframes: $SAM3D_SPARSE_KEYFRAMES"
echo "Stage5 mode: ${STAGE5_MODE:-<script default>}"

copy_cache_path() {
  local src="$1"
  local dst="$2"
  if [[ ! -e "$src" ]]; then
    return
  fi
  if [[ -e "$dst" ]]; then
    return
  fi
  mkdir -p "$(dirname "$dst")"
  cp -a "$src" "$dst"
  echo "Seeded cache: $dst"
}

seed_video_cache() {
  local video_id="$1"
  local src_dir="$CACHE_SOURCE_ROOT/$video_id"
  local dst_dir="$OUTPUT_ROOT/$ABLATION/$video_id"

  if [[ "$SEED_CACHE" != "1" || ! -d "$src_dir" ]]; then
    return
  fi

  mkdir -p "$dst_dir"

  # Reusable preprocessing caches. These do not depend on the ablation mode.
  copy_cache_path "$src_dir/global_bbox.json" "$dst_dir/global_bbox.json"
  copy_cache_path "$src_dir/amodal_masks" "$dst_dir/amodal_masks"
  copy_cache_path "$src_dir/cropped_depths" "$dst_dir/cropped_depths"
  copy_cache_path "$src_dir/cropped_metric_depths" "$dst_dir/cropped_metric_depths"
  copy_cache_path "$src_dir/sam3d_sparse_keyframes" "$dst_dir/sam3d_sparse_keyframes"
  copy_cache_path "$src_dir/grasp_correction/camera_ray_depth_offset.json" "$dst_dir/grasp_correction/camera_ray_depth_offset.json"

  # Stage 3 loss ablations must rerun Stage 3, so do not seed their checkpoint.
  case "$ABLATION" in
    no_fp|no_vp)
      echo "Skipping stage3_checkpoint.pt seed for Stage 3-dependent ablation: $ABLATION"
      ;;
    *)
      copy_cache_path "$src_dir/stage3_checkpoint.pt" "$dst_dir/stage3_checkpoint.pt"
      ;;
  esac
}

for video_id in "${VIDEO_IDS[@]}"; do
  seed_video_cache "$video_id"
done

ARGS=(
  --data_path "$DATA_PATH" \
  --data_output_path "$OUTPUT_ROOT/$ABLATION" \
  --video_id "${VIDEO_IDS[@]}" \
  --ablation "$ABLATION" \
  --debug
)

if [[ "$SAVE_INTERMEDIATES" == "1" ]]; then
  ARGS+=(--save_intermediates)
fi
if [[ "$OVERWRITE_STAGE1_3" == "1" ]]; then
  ARGS+=(--overwrite_stage1_3)
fi
if [[ "$OVERWRITE_GRASP_CORRECTION" == "1" ]]; then
  ARGS+=(--overwrite_grasp_correction)
fi
if [[ "$OVERWRITE_SAM3D_DENSE_CACHE" == "1" ]]; then
  ARGS+=(--overwrite_sam3d_dense_cache)
fi
if [[ "$SAM3D_REF_KEYFRAME_INIT" == "1" ]]; then
  ARGS+=(--sam3d_ref_keyframe_init)
fi
if [[ "$SAM3D_SPARSE_KEYFRAMES" == "1" ]]; then
  ARGS+=(--sam3d_sparse_keyframes)
fi
if [[ -n "$STAGE5_MODE" ]]; then
  ARGS+=(--stage5_mode "$STAGE5_MODE")
fi

"$PYTHON_BIN" "$SCRIPT" "${ARGS[@]}"
