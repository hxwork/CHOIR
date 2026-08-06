#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
OUT_ZIP="${1:-$REPO_ROOT/choir_demo_data.zip}"
STAGING=$(mktemp -d)
trap 'rm -rf "$STAGING"' EXIT

DATA_SRC="$REPO_ROOT/data"
OUT_SRC="$REPO_ROOT/output"

if [[ ! -d "$DATA_SRC" ]]; then
  echo "Missing data directory: $DATA_SRC" >&2
  exit 1
fi
if [[ ! -d "$OUT_SRC" ]]; then
  echo "Missing output directory: $OUT_SRC" >&2
  exit 1
fi

mkdir -p "$STAGING/data" "$STAGING/output"

mp4_count=0
shopt -s nullglob
for mp4 in "$DATA_SRC"/*.mp4; do
  cp -a "$mp4" "$STAGING/data/"
  mp4_count=$((mp4_count + 1))
done
shopt -u nullglob

if [[ "$mp4_count" -eq 0 ]]; then
  echo "No .mp4 files found under $DATA_SRC" >&2
  exit 1
fi

ann_count=0
missing=0
for mp4 in "$STAGING/data"/*.mp4; do
  vid=$(basename "$mp4" .mp4)
  ann_src="$OUT_SRC/$vid/inputs/annotations"
  if [[ ! -d "$ann_src" ]]; then
    echo "WARNING: missing annotations for $vid" >&2
    missing=$((missing + 1))
    continue
  fi
  dest="$STAGING/output/$vid/inputs/annotations"
  mkdir -p "$dest"
  # Copy annotation files only; skip AppleDouble / DS_Store noise.
  find "$ann_src" -type f \
    ! -name '.DS_Store' \
    ! -name '._*' \
    -exec cp -a {} "$dest/" \;
  ann_count=$((ann_count + 1))
done

if [[ "$missing" -ne 0 ]]; then
  echo "Refusing to pack: $missing videos lack output/<id>/inputs/annotations" >&2
  exit 1
fi

rm -f "$OUT_ZIP"
(
  cd "$STAGING"
  zip -rq "$OUT_ZIP" data output
)

size_bytes=$(stat -c%s "$OUT_ZIP" 2>/dev/null || stat -f%z "$OUT_ZIP")
size_h=$(python - <<PY
n=int("$size_bytes")
for u in ["B","KB","MB","GB","TB"]:
    if n < 1024 or u == "TB":
        print(f"{n:.1f}{u}" if u != "B" else f"{n}B")
        break
    n /= 1024
PY
)

echo "Wrote $OUT_ZIP"
echo "  mp4 videos: $mp4_count"
echo "  annotation trees: $ann_count"
echo "  size: $size_h ($size_bytes bytes)"
