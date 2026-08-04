#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")"

N="${N:-25}"
SEED="${SEED:-20260504}"
DATA_PATH="${DATA_PATH:-input_data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/ablations}"
MESH_ROOT="${MESH_ROOT:-rendering_for_paper/ours_SelfCaptured}"
PYTHON_BIN="${PYTHON_BIN:-python}"

RUN_METRICS="${RUN_METRICS:-1}"
STOP_ON_ERROR="${STOP_ON_ERROR:-0}"
STRICT_FAILURE="${STRICT_FAILURE:-0}"
FORCE_RERUN="${FORCE_RERUN:-0}"
FORCE_RESAMPLE="${FORCE_RESAMPLE:-0}"
DRY_RUN="${DRY_RUN:-0}"
SEED_CACHE="${SEED_CACHE:-1}"
DEVICE="${DEVICE:-cuda}"
GPU_IDS="${GPU_IDS:-}"
IOU_BATCH_SIZE="${IOU_BATCH_SIZE:-32}"
CHAMFER_SAMPLES="${CHAMFER_SAMPLES:-3000}"

ABLATIONS=(
  full
  no_fp
  no_vp
  no_stage4_offset
  no_stage5_pen
  no_dyn_contact
  no_stage5_contact
  no_stage5_smooth
)

RUN_NAME="n${N}_seed${SEED}"
RUN_DIR="$OUTPUT_ROOT/overnight_${RUN_NAME}"
LOG_DIR="$RUN_DIR/logs"
STATUS_DIR="$RUN_DIR/status"
SAMPLE_JSON="$RUN_DIR/video_sample.json"
SAMPLE_TXT="$RUN_DIR/video_ids.txt"
SUMMARY_LOG="$RUN_DIR/overnight_summary.log"

mkdir -p "$RUN_DIR" "$LOG_DIR" "$STATUS_DIR"

log_msg() {
  local msg="$1"
  echo "[$(date -Iseconds)] $msg" | tee -a "$SUMMARY_LOG"
}

if [[ "$FORCE_RESAMPLE" == "1" || ! -f "$SAMPLE_TXT" || ! -f "$SAMPLE_JSON" ]]; then
  log_msg "Sampling N=$N TasteRob + N=$N IMG videos with seed=$SEED"
  "$PYTHON_BIN" - "$DATA_PATH" "$N" "$SEED" "$SAMPLE_JSON" "$SAMPLE_TXT" <<'PY'
import json
import random
import sys
from pathlib import Path

data_path = Path(sys.argv[1])
n = int(sys.argv[2])
seed = int(sys.argv[3])
sample_json = Path(sys.argv[4])
sample_txt = Path(sys.argv[5])

numeric = sorted(
    p.name for p in data_path.iterdir()
    if p.is_dir() and p.name.isdigit()
)
img = sorted(
    p.name for p in data_path.iterdir()
    if p.is_dir() and p.name.startswith("IMG_") and p.name != "IMG_5189"
)
if len(numeric) < n:
    raise SystemExit(f"Need {n} numeric TasteRob videos, found {len(numeric)}")
if len(img) < n:
    raise SystemExit(f"Need {n} IMG videos excluding IMG_5189, found {len(img)}")

rng = random.Random(seed)
selected_numeric = rng.sample(numeric, n)
selected_img = rng.sample(img, n)
video_ids = selected_numeric + selected_img

payload = {
    "seed": seed,
    "n_per_source": n,
    "data_path": str(data_path),
    "excluded": ["IMG_5189"],
    "num_numeric_candidates": len(numeric),
    "num_img_candidates": len(img),
    "selected_tasterob_numeric": selected_numeric,
    "selected_img": selected_img,
    "video_ids": video_ids,
}
sample_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
sample_txt.write_text("\n".join(video_ids) + "\n", encoding="utf-8")
PY
else
  log_msg "Reusing existing sample: $SAMPLE_JSON"
fi

mapfile -t VIDEO_IDS < "$SAMPLE_TXT"

log_msg "Run directory: $RUN_DIR"
log_msg "Videos ($((${#VIDEO_IDS[@]}))): ${VIDEO_IDS[*]}"
log_msg "Ablations: ${ABLATIONS[*]}"
log_msg "FORCE_RERUN=$FORCE_RERUN STOP_ON_ERROR=$STOP_ON_ERROR STRICT_FAILURE=$STRICT_FAILURE RUN_METRICS=$RUN_METRICS DRY_RUN=$DRY_RUN"

if [[ "$DRY_RUN" == "1" ]]; then
  log_msg "DRY_RUN=1, stopping after sample generation"
  exit 0
fi

if [[ -z "$GPU_IDS" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_IDS="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | tr '\n' ' ' | xargs || true)"
  fi
fi

if [[ -z "$GPU_IDS" ]]; then
  log_msg "No GPU detected. Set GPU_IDS manually, e.g. GPU_IDS='0 1 2 3'."
  exit 2
fi

read -r -a GPU_ID_LIST <<< "$GPU_IDS"
NUM_GPUS="${#GPU_ID_LIST[@]}"
if [[ "$NUM_GPUS" -lt 1 ]]; then
  log_msg "GPU_IDS is empty after parsing: '$GPU_IDS'"
  exit 2
fi

log_msg "Detected/selected GPUs: ${GPU_ID_LIST[*]} (one fitting task per GPU)"

ABORT_FILE="$RUN_DIR/abort.flag"
rm -f "$ABORT_FILE"

TASK_ABLATIONS=()
TASK_VIDEO_IDS=()
TASK_NUMBERS=()
TOTAL_TASKS=$((${#ABLATIONS[@]} * ${#VIDEO_IDS[@]}))
TASK_IDX=0

for ablation in "${ABLATIONS[@]}"; do
  for video_id in "${VIDEO_IDS[@]}"; do
    TASK_IDX=$((TASK_IDX + 1))
    safe_video="${video_id//\//_}"
    done_file="$STATUS_DIR/${ablation}__${safe_video}.done"

    if [[ "$FORCE_RERUN" != "1" && -f "$done_file" ]]; then
      log_msg "[$TASK_IDX/$TOTAL_TASKS] skip done $ablation/$video_id"
      continue
    fi

    TASK_ABLATIONS+=("$ablation")
    TASK_VIDEO_IDS+=("$video_id")
    TASK_NUMBERS+=("$TASK_IDX")
  done
done

run_one_task() {
  local gpu_id="$1"
  local task_number="$2"
  local ablation="$3"
  local video_id="$4"
  local safe_video="${video_id//\//_}"
  local done_file="$STATUS_DIR/${ablation}__${safe_video}.done"
  local fail_file="$STATUS_DIR/${ablation}__${safe_video}.failed"
  local log_file="$LOG_DIR/${ablation}__${safe_video}.log"

  if [[ -f "$ABORT_FILE" ]]; then
    log_msg "[$task_number/$TOTAL_TASKS][gpu=$gpu_id] skip after abort $ablation/$video_id"
    return 0
  fi

  rm -f "$done_file" "$fail_file"
  log_msg "[$task_number/$TOTAL_TASKS][gpu=$gpu_id] start $ablation/$video_id -> $log_file"

  {
    echo "started_at=$(date -Iseconds)"
    echo "ablation=$ablation"
    echo "video_id=$video_id"
    echo "gpu_id=$gpu_id"
    echo "cuda_visible_devices=$gpu_id"
    echo "data_path=$DATA_PATH"
    echo "output_root=$OUTPUT_ROOT"
    echo "seed_cache=$SEED_CACHE"
    echo
  } > "$log_file"

  CUDA_VISIBLE_DEVICES="$gpu_id" \
  SEED_CACHE="$SEED_CACHE" \
  DATA_PATH="$DATA_PATH" \
  OUTPUT_ROOT="$OUTPUT_ROOT" \
  ./run_ablation.sh "$ablation" "$video_id" >> "$log_file" 2>&1
  local exit_code=$?

  {
    echo
    echo "finished_at=$(date -Iseconds)"
    echo "exit_code=$exit_code"
  } >> "$log_file"

  if [[ "$exit_code" -eq 0 ]]; then
    touch "$done_file"
    log_msg "[$task_number/$TOTAL_TASKS][gpu=$gpu_id] done $ablation/$video_id"
  else
    touch "$fail_file"
    echo "$ablation/$video_id exit=$exit_code log=$log_file" >> "$RUN_DIR/failed_runs.live.txt"
    log_msg "[$task_number/$TOTAL_TASKS][gpu=$gpu_id] FAILED $ablation/$video_id exit=$exit_code"
    if [[ "$STOP_ON_ERROR" == "1" ]]; then
      touch "$ABORT_FILE"
      return "$exit_code"
    fi
  fi
}

gpu_worker() {
  local gpu_slot="$1"
  local gpu_id="$2"
  local task_count="${#TASK_ABLATIONS[@]}"
  local idx

  for ((idx = gpu_slot; idx < task_count; idx += NUM_GPUS)); do
    run_one_task "$gpu_id" "${TASK_NUMBERS[$idx]}" "${TASK_ABLATIONS[$idx]}" "${TASK_VIDEO_IDS[$idx]}" || return $?
  done
}

PIDS=()
for gpu_slot in "${!GPU_ID_LIST[@]}"; do
  gpu_worker "$gpu_slot" "${GPU_ID_LIST[$gpu_slot]}" &
  PIDS+=("$!")
done

worker_exit=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then
    worker_exit=1
  fi
done

FAILED=()
for fail_file in "$STATUS_DIR"/*.failed; do
  [[ -e "$fail_file" ]] || continue
  failed_name="$(basename "$fail_file" .failed)"
  FAILED+=("${failed_name//__//}")
done

if [[ "${#FAILED[@]}" -gt 0 ]]; then
  printf "%s\n" "${FAILED[@]}" > "$RUN_DIR/failed_runs.txt"
  log_msg "Failed runs saved to $RUN_DIR/failed_runs.txt"
else
  rm -f "$RUN_DIR/failed_runs.txt"
  log_msg "All fitting runs completed successfully"
fi

if [[ "$worker_exit" -ne 0 && "$STOP_ON_ERROR" == "1" ]]; then
  log_msg "STOP_ON_ERROR=1 and at least one GPU worker failed"
  exit "$worker_exit"
fi

if [[ "$RUN_METRICS" == "1" ]]; then
  METRIC_OUT="$RUN_DIR/metrics_summary"
  METRIC_LOG="$LOG_DIR/metrics_all.log"
  log_msg "Starting ablation metrics -> $METRIC_LOG"
  {
    echo "started_at=$(date -Iseconds)"
    echo "metric_output=$METRIC_OUT"
    echo
  } > "$METRIC_LOG"

  DEVICE="$DEVICE" \
  IOU_BATCH_SIZE="$IOU_BATCH_SIZE" \
  CHAMFER_SAMPLES="$CHAMFER_SAMPLES" \
  ABLATION_ROOT="$OUTPUT_ROOT" \
  MESH_ROOT="$MESH_ROOT" \
  DATA_ROOT="$DATA_PATH" \
  OUTPUT_DIR="$METRIC_OUT" \
  ./run_ablation_metrics.sh all "${VIDEO_IDS[@]}" >> "$METRIC_LOG" 2>&1
  metric_exit=$?

  {
    echo
    echo "finished_at=$(date -Iseconds)"
    echo "exit_code=$metric_exit"
  } >> "$METRIC_LOG"

  if [[ "$metric_exit" -eq 0 ]]; then
    log_msg "Metrics completed: $METRIC_OUT/all_ablation_metrics.json"
  else
    log_msg "Metrics FAILED exit=$metric_exit; see $METRIC_LOG"
    exit "$metric_exit"
  fi
fi

if [[ "${#FAILED[@]}" -gt 0 ]]; then
  log_msg "Finished with ${#FAILED[@]} failed fitting runs"
  if [[ "$STRICT_FAILURE" == "1" ]]; then
    exit 1
  fi
  exit 0
fi

log_msg "Overnight ablation benchmark finished successfully"
