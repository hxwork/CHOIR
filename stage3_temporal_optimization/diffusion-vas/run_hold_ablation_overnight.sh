#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_PATH="${DATA_PATH:-input_data}"
MVS_DATA_ROOT="${MVS_DATA_ROOT:-$DATA_PATH}"
HO3DV3_ROOT="${HO3DV3_ROOT:-HO3Dv3}"
HO3DV3_ZIP="${HO3DV3_ZIP:-HO3Dv3.zip}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/ablations_hold}"
METRIC_ROOT="${METRIC_ROOT:-$OUTPUT_ROOT/metrics}"
SEED_CACHE="${SEED_CACHE:-1}"
SAVE_INTERMEDIATES="${SAVE_INTERMEDIATES:-0}"
RUN_EVAL="${RUN_EVAL:-1}"
FORCE_RERUN="${FORCE_RERUN:-0}"
FORCE_EVAL="${FORCE_EVAL:-0}"
STOP_ON_ERROR="${STOP_ON_ERROR:-0}"
STRICT_FAILURE="${STRICT_FAILURE:-0}"
DRY_RUN="${DRY_RUN:-0}"
DEBUG_EVAL="${DEBUG_EVAL:-0}"
ONLY_EVAL_HAND="${ONLY_EVAL_HAND:-0}"
GPU_IDS="${GPU_IDS:-}"

ABLATIONS=(
  full
  no_fp
  no_vp
  no_stage4_offset
  no_stage5_pen
  no_dyn_contact
  no_stage5_contact
  no_stage5_smooth
  stage3_raw_metric
  post_gfm_metric
)

HOLD_SEQUENCES=(
  hold_ABF12_ho3d.180
  hold_ABF14_ho3d.180
  hold_GPMF12_ho3d.90
  hold_GPMF14_ho3d.90
  hold_MC1_ho3d.0
  hold_MC4_ho3d.0
  hold_MDF12_ho3d.60
  hold_MDF14_ho3d.300
  hold_ShSu10_ho3d.30
  hold_ShSu12_ho3d.30
  hold_SM2_ho3d.90
  hold_SM4_ho3d.0
  hold_SMu1_ho3d.0
  hold_SMu40_ho3d.0
)

RUN_DIR="$OUTPUT_ROOT/hold14_run"
LOG_DIR="$RUN_DIR/logs"
STATUS_DIR="$RUN_DIR/status"
SUMMARY_LOG="$RUN_DIR/hold_ablation_summary.log"
ABORT_FILE="$RUN_DIR/abort.flag"

mkdir -p "$RUN_DIR" "$LOG_DIR" "$STATUS_DIR" "$METRIC_ROOT"
rm -f "$ABORT_FILE"

log_msg() {
  local msg="$1"
  echo "[$(date -Iseconds)] $msg" | tee -a "$SUMMARY_LOG"
}

seq_base() {
  local seq_name="$1"
  echo "${seq_name%%.*}"
}

mvs_root_for_seq() {
  local seq_name="$1"
  local base
  base="$(seq_base "$seq_name")"
  echo "$MVS_DATA_ROOT/$base/processed/colmap_$seq_name/sfm_superpoint+superglue/mvs/"
}

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
TOTAL_TASKS=$((${#ABLATIONS[@]} * ${#HOLD_SEQUENCES[@]}))

log_msg "Run directory: $RUN_DIR"
log_msg "Output root: $OUTPUT_ROOT"
log_msg "Metric root: $METRIC_ROOT"
log_msg "MVS data root: $MVS_DATA_ROOT"
log_msg "HO3Dv3 root: $HO3DV3_ROOT"
log_msg "Detected/selected GPUs: ${GPU_ID_LIST[*]} (one fitting task per GPU)"
log_msg "Tasks: $TOTAL_TASKS (${#ABLATIONS[@]} ablations x ${#HOLD_SEQUENCES[@]} HOLD sequences)"
log_msg "RUN_EVAL=$RUN_EVAL FORCE_RERUN=$FORCE_RERUN FORCE_EVAL=$FORCE_EVAL STOP_ON_ERROR=$STOP_ON_ERROR STRICT_FAILURE=$STRICT_FAILURE DRY_RUN=$DRY_RUN"

if [[ "$DRY_RUN" == "1" ]]; then
  printf "%s\n" "${HOLD_SEQUENCES[@]}" > "$RUN_DIR/hold_sequences.txt"
  printf "%s\n" "${ABLATIONS[@]}" > "$RUN_DIR/ablations.txt"
  log_msg "DRY_RUN=1, wrote sequence and ablation lists only"
  exit 0
fi

ensure_ho3dv3_root() {
  if [[ -f "$HO3DV3_ROOT/processed/ABF12.pt" ]]; then
    return 0
  fi
  if [[ ! -f "$HO3DV3_ZIP" ]]; then
    log_msg "Missing HO3Dv3 root ($HO3DV3_ROOT) and zip ($HO3DV3_ZIP). Set HO3DV3_ROOT or HO3DV3_ZIP."
    return 1
  fi
  log_msg "Extracting HO3Dv3 assets from $HO3DV3_ZIP"
  "$PYTHON_BIN" - "$HO3DV3_ZIP" "$(dirname "$HO3DV3_ROOT")" <<'PY'
import sys
import zipfile
from pathlib import Path

zip_path = Path(sys.argv[1])
target_parent = Path(sys.argv[2])
target_parent.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(zip_path) as zf:
    zf.extractall(target_parent)
print(f"Extracted {zip_path} -> {target_parent}")
PY
}

if ! ensure_ho3dv3_root; then
  exit 2
fi

TASK_ABLATIONS=()
TASK_SEQUENCES=()
TASK_NUMBERS=()
task_idx=0

for ablation in "${ABLATIONS[@]}"; do
  for seq_name in "${HOLD_SEQUENCES[@]}"; do
    task_idx=$((task_idx + 1))
    base="$(seq_base "$seq_name")"
    done_file="$STATUS_DIR/${ablation}__${base}.done"
    if [[ "$FORCE_RERUN" != "1" && "$FORCE_EVAL" != "1" && -f "$done_file" ]]; then
      log_msg "[$task_idx/$TOTAL_TASKS] skip done $ablation/$seq_name"
      continue
    fi
    TASK_ABLATIONS+=("$ablation")
    TASK_SEQUENCES+=("$seq_name")
    TASK_NUMBERS+=("$task_idx")
  done
done

run_eval_for_task() {
  local gpu_id="$1"
  local ablation="$2"
  local seq_name="$3"
  local base pred_data_path out_dir mvs_root eval_args
  base="$(seq_base "$seq_name")"
  pred_data_path="$OUTPUT_ROOT/$ablation/$base/eval_data.npy"
  out_dir="$METRIC_ROOT/$ablation/$base"
  mvs_root="$(mvs_root_for_seq "$seq_name")"

  if [[ ! -f "$pred_data_path" ]]; then
    echo "Missing prediction data: $pred_data_path" >&2
    return 1
  fi

  eval_args=(
    --seq_name "$seq_name"
    --mvs_root "$mvs_root"
    --out_dir "$out_dir"
    --pred_data_path "$pred_data_path"
  )
  if [[ "$DEBUG_EVAL" == "1" ]]; then
    eval_args+=(--debug)
  fi
  if [[ "$ONLY_EVAL_HAND" == "1" ]]; then
    eval_args+=(--only_eval_hand)
  fi

  CUDA_VISIBLE_DEVICES="$gpu_id" HO3DV3_ROOT="$HO3DV3_ROOT" "$PYTHON_BIN" eval_ours.py "${eval_args[@]}"
}

run_one_task() {
  local gpu_id="$1"
  local task_number="$2"
  local ablation="$3"
  local seq_name="$4"
  local base done_file fail_file log_file fit_exit eval_exit
  base="$(seq_base "$seq_name")"
  done_file="$STATUS_DIR/${ablation}__${base}.done"
  fail_file="$STATUS_DIR/${ablation}__${base}.failed"
  log_file="$LOG_DIR/${ablation}__${base}.log"

  if [[ -f "$ABORT_FILE" ]]; then
    log_msg "[$task_number/$TOTAL_TASKS][gpu=$gpu_id] skip after abort $ablation/$seq_name"
    return 0
  fi

  rm -f "$done_file" "$fail_file"
  log_msg "[$task_number/$TOTAL_TASKS][gpu=$gpu_id] start $ablation/$seq_name -> $log_file"
  {
    echo "started_at=$(date -Iseconds)"
    echo "ablation=$ablation"
    echo "seq_name=$seq_name"
    echo "video_id=$base"
    echo "gpu_id=$gpu_id"
    echo "data_path=$DATA_PATH"
    echo "output_root=$OUTPUT_ROOT"
    echo
  } > "$log_file"

  fit_exit=0
  if [[ "$FORCE_EVAL" != "1" ]]; then
    CUDA_VISIBLE_DEVICES="$gpu_id" \
    DATA_PATH="$DATA_PATH" \
    OUTPUT_ROOT="$OUTPUT_ROOT" \
    SEED_CACHE="$SEED_CACHE" \
    SAVE_INTERMEDIATES="$SAVE_INTERMEDIATES" \
    ./run_ablation.sh "$ablation" "$base" >> "$log_file" 2>&1
    fit_exit=$?
  else
    echo "FORCE_EVAL=1, skipping fitting" >> "$log_file"
  fi

  eval_exit=0
  if [[ "$fit_exit" -eq 0 && "$RUN_EVAL" == "1" ]]; then
    run_eval_for_task "$gpu_id" "$ablation" "$seq_name" >> "$log_file" 2>&1
    eval_exit=$?
  fi

  {
    echo
    echo "finished_at=$(date -Iseconds)"
    echo "fit_exit=$fit_exit"
    echo "eval_exit=$eval_exit"
  } >> "$log_file"

  if [[ "$fit_exit" -eq 0 && "$eval_exit" -eq 0 ]]; then
    touch "$done_file"
    log_msg "[$task_number/$TOTAL_TASKS][gpu=$gpu_id] done $ablation/$seq_name"
  else
    touch "$fail_file"
    echo "$ablation/$seq_name fit_exit=$fit_exit eval_exit=$eval_exit log=$log_file" >> "$RUN_DIR/failed_runs.live.txt"
    log_msg "[$task_number/$TOTAL_TASKS][gpu=$gpu_id] FAILED $ablation/$seq_name fit_exit=$fit_exit eval_exit=$eval_exit"
    if [[ "$STOP_ON_ERROR" == "1" ]]; then
      touch "$ABORT_FILE"
      return 1
    fi
  fi
}

gpu_worker() {
  local gpu_slot="$1"
  local gpu_id="$2"
  local task_count="${#TASK_ABLATIONS[@]}"
  local idx
  for ((idx = gpu_slot; idx < task_count; idx += NUM_GPUS)); do
    run_one_task "$gpu_id" "${TASK_NUMBERS[$idx]}" "${TASK_ABLATIONS[$idx]}" "${TASK_SEQUENCES[$idx]}" || return $?
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
  log_msg "All HOLD ablation runs completed successfully"
fi

log_msg "Aggregating metrics"
"$PYTHON_BIN" - "$METRIC_ROOT" "$RUN_DIR/hold_ablation_metrics.json" "$RUN_DIR/hold_ablation_metrics.csv" "${ABLATIONS[@]}" -- "${HOLD_SEQUENCES[@]}" <<'PY'
import csv
import json
import sys
from pathlib import Path

sep = sys.argv.index("--")
metric_root = Path(sys.argv[1])
json_out = Path(sys.argv[2])
csv_out = Path(sys.argv[3])
ablations = sys.argv[4:sep]
sequences = sys.argv[sep + 1:]

rows = []
summary = {}
metric_keys = set()
for ablation in ablations:
    summary[ablation] = {}
    for seq_name in sequences:
        base = seq_name.split(".")[0]
        metric_path = metric_root / ablation / base / "metric.json"
        if metric_path.exists():
            metrics = json.loads(metric_path.read_text(encoding="utf-8"))
            metric_keys.update(k for k in metrics if k not in {"timestamp", "seq_name"})
            row = {"ablation": ablation, "seq_name": seq_name, "status": "ok", **metrics}
            summary[ablation][seq_name] = metrics
        else:
            row = {"ablation": ablation, "seq_name": seq_name, "status": "missing"}
            summary[ablation][seq_name] = {"error": f"missing {metric_path}"}
        rows.append(row)

json_out.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
fieldnames = ["ablation", "seq_name", "status"] + sorted(metric_keys) + ["timestamp"]
with csv_out.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
print(f"Saved {json_out}")
print(f"Saved {csv_out}")
PY
aggregate_exit=$?
if [[ "$aggregate_exit" -ne 0 ]]; then
  log_msg "Metric aggregation FAILED exit=$aggregate_exit"
  exit "$aggregate_exit"
fi

if [[ "$worker_exit" -ne 0 && "$STOP_ON_ERROR" == "1" ]]; then
  log_msg "STOP_ON_ERROR=1 and at least one worker failed"
  exit "$worker_exit"
fi

if [[ "${#FAILED[@]}" -gt 0 && "$STRICT_FAILURE" == "1" ]]; then
  log_msg "STRICT_FAILURE=1 and ${#FAILED[@]} runs failed"
  exit 1
fi

log_msg "HOLD ablation benchmark finished"
