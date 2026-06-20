#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

STAMP="$(date +%m%d%H%M%S)"
GAIT="${GAIT:-trot}"
case "$GAIT" in
  trot|pace|canter|jump) ;;
  *)
    echo "Unsupported GAIT=${GAIT}. Use one of: trot, pace, canter, jump." >&2
    exit 1
    ;;
esac

MOTION_FILE="data/motions/go2/go2_apex_${GAIT}.pkl"
OUT_DIR="${OUT_DIR:-output/smp_prior_go2_${GAIT}_track_${STAMP}}"
DEVICE="${DEVICE:-cuda}"

if [[ ! -f "$MOTION_FILE" ]]; then
  echo "Missing motion file: ${MOTION_FILE}" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"
RUN_ENV_CONFIG="${OUT_DIR}/run_env_config.yaml"
RUN_PRIOR_CONFIG="${OUT_DIR}/run_tinymdm_config.yaml"
cp data/envs/smp_go2_trot_track_env.yaml "$RUN_ENV_CONFIG"
cp tools/diffusion_model/config/tinymdm_go2_trot_track.yaml "$RUN_PRIOR_CONFIG"
sed -i \
  -e "s#^motion_file:.*#motion_file: \"${MOTION_FILE}\"#" \
  "$RUN_ENV_CONFIG"
sed -i \
  -e "s#^env_config:.*#env_config: \"${RUN_ENV_CONFIG}\"#" \
  -e "s#^motion_file:.*#motion_file: \"${MOTION_FILE}\"#" \
  "$RUN_PRIOR_CONFIG"

if [[ -n "${PRIOR_ITERS:-}" ]]; then
  sed -i \
    -e "s#^num_iterations:.*#num_iterations: ${PRIOR_ITERS}#" \
    "$RUN_PRIOR_CONFIG"
fi

echo "Training Go2 ${GAIT} track prior"
echo "Output: ${OUT_DIR}"
echo "Motion: ${MOTION_FILE}"

python tools/diffusion_model/train_tinymdm.py \
  --cfg_path "$RUN_PRIOR_CONFIG" \
  --out_dir "$OUT_DIR" \
  --device "$DEVICE" \
  "$@"

sed -i \
  -e "s#^env_config:.*#env_config: ${OUT_DIR}/env_config.yaml#" \
  "${OUT_DIR}/diffusion_config.yaml"
