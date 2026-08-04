#!/bin/sh
set -eu

cd "$(dirname "$0")"

VIDEO_ID="${1:-${VIDEO_ID:-hold_ABF12_ho3d}}"
_safe_video_id=$(printf "%s" "$VIDEO_ID" | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9_' '_')
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/subjective_ablation_${_safe_video_id}}"
SAVE_INTERMEDIATES="${SAVE_INTERMEDIATES:-0}"
OVERWRITE_STAGE1_3="${OVERWRITE_STAGE1_3:-0}"
OVERWRITE_GRASP_CORRECTION="${OVERWRITE_GRASP_CORRECTION:-0}"
OVERWRITE_SAM3D_DENSE_CACHE="${OVERWRITE_SAM3D_DENSE_CACHE:-0}"
SAM3D_REF_KEYFRAME_INIT="${SAM3D_REF_KEYFRAME_INIT:-0}"
SAM3D_SPARSE_KEYFRAMES="${SAM3D_SPARSE_KEYFRAMES:-0}"
STAGE5_MODE="${STAGE5_MODE:-object_lite}"

ABLATIONS="full no_fp no_vp no_stage4 no_penetration no_contact"

echo "Qualitative ablation video: $VIDEO_ID"
echo "Output root: $OUTPUT_ROOT"
echo "Save intermediates: $SAVE_INTERMEDIATES"
echo "Overwrite stage1_3: $OVERWRITE_STAGE1_3"
echo "Overwrite grasp correction: $OVERWRITE_GRASP_CORRECTION"
echo "Overwrite SAM3D dense cache: $OVERWRITE_SAM3D_DENSE_CACHE"
echo "SAM3D ref keyframe init: $SAM3D_REF_KEYFRAME_INIT"
echo "SAM3D sparse keyframes: $SAM3D_SPARSE_KEYFRAMES"
echo "Stage5 mode: $STAGE5_MODE"
echo "Ablations: $ABLATIONS"

for ablation in $ABLATIONS; do
  echo
  echo "================================================================================"
  echo "Running qualitative ablation: $ablation / $VIDEO_ID"
  echo "================================================================================"

  OUTPUT_ROOT="$OUTPUT_ROOT" \
  SAVE_INTERMEDIATES="$SAVE_INTERMEDIATES" \
  OVERWRITE_STAGE1_3="$OVERWRITE_STAGE1_3" \
  OVERWRITE_GRASP_CORRECTION="$OVERWRITE_GRASP_CORRECTION" \
  OVERWRITE_SAM3D_DENSE_CACHE="$OVERWRITE_SAM3D_DENSE_CACHE" \
  SAM3D_REF_KEYFRAME_INIT="$SAM3D_REF_KEYFRAME_INIT" \
  SAM3D_SPARSE_KEYFRAMES="$SAM3D_SPARSE_KEYFRAMES" \
  STAGE5_MODE="$STAGE5_MODE" \
  ./run_ablation.sh "$ablation" "$VIDEO_ID"
done

echo
echo "Qualitative ablation finished."
echo "Final fitting outputs:"
echo "  $OUTPUT_ROOT/<ablation>/$VIDEO_ID/optimized_hoi_contact_seq/"
echo "Paper meshes:"
echo "  rendering_for_paper/ours_SelfCaptured/${VIDEO_ID}_<ablation>/{hand,object}/"
