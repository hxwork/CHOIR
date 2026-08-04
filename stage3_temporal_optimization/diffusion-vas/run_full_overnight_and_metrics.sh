#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT="${SCRIPT:-demo_fitting_5stages_sam3d_reset_ablation.py}"
METRIC_SCRIPT="${METRIC_SCRIPT:-compute_ablation_metrics.py}"

DATA_PATH="${DATA_PATH:-../../output}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/ablations}"
MESH_ROOT="${MESH_ROOT:-rendering_for_paper/ours_SelfCaptured}"

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-logs/full_overnight_${RUN_TAG}}"
METRIC_OUTPUT_DIR="${METRIC_OUTPUT_DIR:-${OUTPUT_ROOT}/metrics_summary_full_${RUN_TAG}}"

SAVE_INTERMEDIATES="${SAVE_INTERMEDIATES:-0}"
OVERWRITE_STAGE1_3="${OVERWRITE_STAGE1_3:-0}"
OVERWRITE_GRASP_CORRECTION="${OVERWRITE_GRASP_CORRECTION:-0}"
OVERWRITE_SAM3D_DENSE_CACHE="${OVERWRITE_SAM3D_DENSE_CACHE:-0}"
SAM3D_REF_KEYFRAME_INIT="${SAM3D_REF_KEYFRAME_INIT:-0}"
SAM3D_SPARSE_KEYFRAMES="${SAM3D_SPARSE_KEYFRAMES:-0}"
STAGE5_MODE="${STAGE5_MODE:-}"

DEVICE="${DEVICE:-cuda}"
IOU_BATCH_SIZE="${IOU_BATCH_SIZE:-32}"
CHAMFER_SAMPLES="${CHAMFER_SAMPLES:-3000}"
NO_RENDER_IOU="${NO_RENDER_IOU:-0}"
NO_FLIP_XY_FOR_IOU="${NO_FLIP_XY_FOR_IOU:-0}"

if [[ $# -gt 0 ]]; then
  VIDEO_IDS=("$@")
else
  mapfile -t VIDEO_IDS < <("$PYTHON_BIN" - "$DATA_PATH" <<'PY'
from pathlib import Path
import sys

data_path = Path(sys.argv[1])
excluded = {"dynhamr", "images"}
for child in sorted(data_path.iterdir(), key=lambda p: p.name):
    if child.is_dir() and child.name not in excluded:
        print(child.name)
PY
)
fi

mkdir -p "$LOG_DIR" "$METRIC_OUTPUT_DIR"

FIT_LOG="$LOG_DIR/full_fit.log"
METRIC_LOG="$LOG_DIR/full_metric.log"

echo "Run tag: $RUN_TAG"
echo "Visible CUDA devices: ${CUDA_VISIBLE_DEVICES:-<all visible>}"
echo "Videos (${#VIDEO_IDS[@]}): ${VIDEO_IDS[*]}"
echo "Data path: $DATA_PATH"
echo "Output root: $OUTPUT_ROOT"
echo "Metric output: $METRIC_OUTPUT_DIR"
echo "Fit log: $FIT_LOG"
echo "Metric log: $METRIC_LOG"

FIT_ARGS=(
  --data_path "$DATA_PATH"
  --data_output_path "$OUTPUT_ROOT/full"
  --video_id "${VIDEO_IDS[@]}"
  --ablation full
)

if [[ "$SAVE_INTERMEDIATES" == "1" ]]; then
  FIT_ARGS+=(--save_intermediates)
fi
if [[ "$OVERWRITE_STAGE1_3" == "1" ]]; then
  FIT_ARGS+=(--overwrite_stage1_3)
fi
if [[ "$OVERWRITE_GRASP_CORRECTION" == "1" ]]; then
  FIT_ARGS+=(--overwrite_grasp_correction)
fi
if [[ "$OVERWRITE_SAM3D_DENSE_CACHE" == "1" ]]; then
  FIT_ARGS+=(--overwrite_sam3d_dense_cache)
fi
if [[ "$SAM3D_REF_KEYFRAME_INIT" == "1" ]]; then
  FIT_ARGS+=(--sam3d_ref_keyframe_init)
fi
if [[ "$SAM3D_SPARSE_KEYFRAMES" == "1" ]]; then
  FIT_ARGS+=(--sam3d_sparse_keyframes)
fi
if [[ -n "$STAGE5_MODE" ]]; then
  FIT_ARGS+=(--stage5_mode "$STAGE5_MODE")
fi

echo "===== Running full fitting ====="
"$PYTHON_BIN" "$SCRIPT" "${FIT_ARGS[@]}" 2>&1 | tee "$FIT_LOG"

METRIC_ARGS=(
  --ablation_root "$OUTPUT_ROOT"
  --mesh_root "$MESH_ROOT"
  --data_root "$DATA_PATH"
  --output_dir "$METRIC_OUTPUT_DIR"
  --ablation full
  --video_id "${VIDEO_IDS[@]}"
  --device "$DEVICE"
  --iou_batch_size "$IOU_BATCH_SIZE"
  --chamfer_samples "$CHAMFER_SAMPLES"
)

if [[ "$NO_RENDER_IOU" == "1" ]]; then
  METRIC_ARGS+=(--no_render_iou)
fi
if [[ "$NO_FLIP_XY_FOR_IOU" == "1" ]]; then
  METRIC_ARGS+=(--no_flip_xy_for_iou)
fi

echo "===== Running full metrics ====="
"$PYTHON_BIN" "$METRIC_SCRIPT" "${METRIC_ARGS[@]}" 2>&1 | tee "$METRIC_LOG"

echo "===== Done ====="
echo "Fit log: $FIT_LOG"
echo "Metric log: $METRIC_LOG"
echo "Metric CSV: $METRIC_OUTPUT_DIR/all_ablation_metrics.csv"
echo "Metric JSON: $METRIC_OUTPUT_DIR/all_ablation_metrics.json"
