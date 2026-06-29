#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DEFAULT_MOTION_FILE="data/motions/go2/go2_apex_jump_clean.pkl"
MOTION_FILE="${MOTION_FILE:-$DEFAULT_MOTION_FILE}"

if [[ "$MOTION_FILE" == "$DEFAULT_MOTION_FILE" && ! -f "$MOTION_FILE" ]]; then
  python tools/gmr_to_mimickit/apex_go2_csv_to_mimickit.py \
    --input_file data/motions/go2/apex_csv/go2_retarget_jump.csv \
    --output_file "$DEFAULT_MOTION_FILE" \
    --fps 50 \
    --loop_mode 0 \
    --drop_final_wrap_threshold 1.0
fi

export GAIT=jump
export MOTION_FILE

exec "$ROOT_DIR/scripts/train_go2_trot_track_policy.sh" "$@"
