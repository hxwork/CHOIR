#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT="compute_in_the_wild_metrics.py"
RENDER_ROOT="${RENDER_ROOT:-../../output}"
MESH_ROOT="${MESH_ROOT:-../../output}"
DATA_ROOT="${DATA_ROOT:-../../output}"
OUTPUT_DIR="${OUTPUT_DIR:-../../metrics/in_the_wild}"
DEVICE="${DEVICE:-cuda}"
IOU_BATCH_SIZE="${IOU_BATCH_SIZE:-32}"
CHAMFER_SAMPLES="${CHAMFER_SAMPLES:-3000}"
NO_RENDER_IOU="${NO_RENDER_IOU:-0}"
NO_FLIP_XY_FOR_IOU="${NO_FLIP_XY_FOR_IOU:-0}"
UNIT_TO_CM="${UNIT_TO_CM:-100.0}"

ARGS=(
  --render_root "$RENDER_ROOT"
  --mesh_root "$MESH_ROOT"
  --data_root "$DATA_ROOT"
  --output_dir "$OUTPUT_DIR"
  --device "$DEVICE"
  --iou_batch_size "$IOU_BATCH_SIZE"
  --chamfer_samples "$CHAMFER_SAMPLES"
  --unit_to_cm "$UNIT_TO_CM"
)

if [[ "$NO_RENDER_IOU" == "1" ]]; then
  ARGS+=(--no_render_iou)
fi
if [[ "$NO_FLIP_XY_FOR_IOU" == "1" ]]; then
  ARGS+=(--no_flip_xy_for_iou)
fi
if [[ $# -gt 0 ]]; then
  ARGS+=(--video_id "$@")
fi

echo "Render root: $RENDER_ROOT"
echo "Mesh root: $MESH_ROOT"
echo "Data root: $DATA_ROOT"
echo "Output dir: $OUTPUT_DIR"
echo "Device: $DEVICE"
echo "No render IoU: $NO_RENDER_IOU"
echo "No flip xy for IoU: $NO_FLIP_XY_FOR_IOU"

"$PYTHON_BIN" "$SCRIPT" "${ARGS[@]}"
