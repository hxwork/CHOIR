#!/usr/bin/env bash
set -euo pipefail

VIDEO_ID="${1:-107407}"
REPO_ROOT="${REPO_ROOT:-../..}"
DATA_PATH="${DATA_PATH:-$REPO_ROOT/output}"
DATA_OUTPUT_PATH="${DATA_OUTPUT_PATH:-$REPO_ROOT/output}"

python run_temporal_optimization.py \
  --video_id "$VIDEO_ID" \
  --data_path "$DATA_PATH" \
  --data_output_path "$DATA_OUTPUT_PATH" \
  --debug
