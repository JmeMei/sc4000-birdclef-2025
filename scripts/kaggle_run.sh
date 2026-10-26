#!/usr/bin/env bash
# Push a notebook folder to Kaggle, wait for it to finish, and download its outputs.
#
# Usage:   bash scripts/kaggle_run.sh notebooks/eda
#
# The folder must contain the notebook and a kernel-metadata.json
# (copy one from an existing folder). Keep "is_private": true —
# the course forbids releasing code publicly.
# Outputs land in outputs/<folder name>/ (gitignored).
set -euo pipefail

cd "$(dirname "$0")/.."
DIR="${1:?usage: bash scripts/kaggle_run.sh <notebook-folder>}"
KAGGLE=".venv/bin/kaggle"

if ! grep -q '"is_private": *true' "$DIR/kernel-metadata.json"; then
  echo "Refusing to push: $DIR/kernel-metadata.json must set \"is_private\": true" >&2
  exit 1
fi

KERNEL_ID=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["id"])' "$DIR/kernel-metadata.json")
OUT="outputs/$(basename "$DIR")"

echo "Pushing $DIR -> https://www.kaggle.com/code/$KERNEL_ID"
"$KAGGLE" kernels push -p "$DIR"

echo "Waiting for the run to finish (Ctrl+C to stop waiting; the run continues on Kaggle)..."
while true; do
  STATUS=$("$KAGGLE" kernels status "$KERNEL_ID" 2>&1 || true)
  case "$STATUS" in
    *COMPLETE*) echo "Run complete."; break ;;
    *ERROR*|*CANCEL*) echo "$STATUS"; echo "Run failed — downloading the log." ;
      mkdir -p "$OUT"; "$KAGGLE" kernels output "$KERNEL_ID" -p "$OUT" -o >/dev/null || true
      exit 1 ;;
  esac
  sleep 30
done

mkdir -p "$OUT"
"$KAGGLE" kernels output "$KERNEL_ID" -p "$OUT" -o
echo "Outputs saved to $OUT/"
