#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

STAMP="$(date +%m%d%H%M%S)"
OUT_DIR="${OUT_DIR:-output/smp_prior_go2_trot_track_${STAMP}}"
DEVICE="${DEVICE:-cuda}"

echo "Training Go2 trot track prior"
echo "Output: ${OUT_DIR}"

python tools/diffusion_model/train_tinymdm.py \
  --cfg_path tools/diffusion_model/config/tinymdm_go2_trot_track.yaml \
  --out_dir "$OUT_DIR" \
  --device "$DEVICE" \
  "$@"
