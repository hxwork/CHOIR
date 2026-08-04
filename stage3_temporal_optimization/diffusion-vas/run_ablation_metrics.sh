#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT="compute_ablation_metrics.py"
ABLATION_ROOT="${ABLATION_ROOT:-outputs/ablations}"
MESH_ROOT="${MESH_ROOT:-rendering_for_paper/ours_SelfCaptured}"
DATA_ROOT="${DATA_ROOT:-input_data}"
OUTPUT_DIR="${OUTPUT_DIR:-$ABLATION_ROOT/metrics_summary}"
DEVICE="${DEVICE:-cuda}"
IOU_BATCH_SIZE="${IOU_BATCH_SIZE:-32}"
CHAMFER_SAMPLES="${CHAMFER_SAMPLES:-3000}"
NO_RENDER_IOU="${NO_RENDER_IOU:-0}"
NO_FLIP_XY_FOR_IOU="${NO_FLIP_XY_FOR_IOU:-0}"

ARGS=(
  --ablation_root "$ABLATION_ROOT"
  --mesh_root "$MESH_ROOT"
  --data_root "$DATA_ROOT"
  --output_dir "$OUTPUT_DIR"
  --device "$DEVICE"
  --iou_batch_size "$IOU_BATCH_SIZE"
  --chamfer_samples "$CHAMFER_SAMPLES"
)

if [[ "$NO_RENDER_IOU" == "1" ]]; then
  ARGS+=(--no_render_iou)
fi
if [[ "$NO_FLIP_XY_FOR_IOU" == "1" ]]; then
  ARGS+=(--no_flip_xy_for_iou)
fi

if [[ $# -gt 0 ]]; then
  ABLATION="$1"
  shift
  case "$ABLATION" in
    all)
      ;;
    full|no_fp|no_vp|no_stage4_offset|no_stage5_pen|no_dyn_contact|no_stage5_contact|no_stage5_smooth)
      ARGS+=(--ablation "$ABLATION")
      ;;
    *)
      echo "Unknown ablation: $ABLATION" >&2
      echo "Choices: all full no_fp no_vp no_stage4_offset no_stage5_pen no_dyn_contact no_stage5_contact no_stage5_smooth" >&2
      exit 2
      ;;
  esac
fi

if [[ $# -gt 0 ]]; then
  ARGS+=(--video_id "$@")
fi

echo "Ablation root: $ABLATION_ROOT"
echo "Mesh root: $MESH_ROOT"
echo "Data root: $DATA_ROOT"
echo "Output dir: $OUTPUT_DIR"
echo "Device: $DEVICE"
echo "No render IoU: $NO_RENDER_IOU"
echo "No flip xy for IoU: $NO_FLIP_XY_FOR_IOU"

"$PYTHON_BIN" "$SCRIPT" "${ARGS[@]}"
