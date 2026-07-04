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

MOTION_FILE="${MOTION_FILE:-data/motions/go2/go2_apex_${GAIT}.pkl}"
OUT_DIR="${OUT_DIR:-output/smp_go2_${GAIT}_track_${STAMP}}"
MAX_SAMPLES="${MAX_SAMPLES:-1310720000}"

if [[ ! -f "$MOTION_FILE" ]]; then
  echo "Missing motion file: ${MOTION_FILE}" >&2
  exit 1
fi

if [[ -z "${PRIOR_DIR:-}" ]]; then
  LATEST_PRIOR="$(find output -maxdepth 1 -type d -name "smp_prior_go2_${GAIT}_track_*" | sort | tail -n 1)"
  if [[ -n "$LATEST_PRIOR" ]]; then
    PRIOR_DIR="$LATEST_PRIOR"
  elif [[ "$GAIT" == "trot" ]]; then
    PRIOR_DIR="output/smp_go2_trot_track_reward/prior"
  else
    echo "No prior found for GAIT=${GAIT}." >&2
    echo "Run: GAIT=${GAIT} scripts/train_go2_trot_track_prior.sh" >&2
    exit 1
  fi
fi

if [[ ! -f "${PRIOR_DIR}/diffusion_config.yaml" || ! -f "${PRIOR_DIR}/model.pt" ]]; then
  echo "Missing prior files in ${PRIOR_DIR}" >&2
  echo "Set PRIOR_DIR=/path/to/prior_dir or run scripts/train_go2_trot_track_prior.sh first." >&2
  exit 1
fi

PRIOR_MOTION_FILE="$(sed -n 's/^motion_file:[[:space:]]*//p' "${PRIOR_DIR}/diffusion_config.yaml" | head -n 1 | tr -d '"' | xargs)"
if [[ -n "$PRIOR_MOTION_FILE" && "${ALLOW_PRIOR_MOTION_MISMATCH:-False}" != "True" ]]; then
  MOTION_ABS="$(readlink -f "$MOTION_FILE")"
  PRIOR_MOTION_ABS="$(readlink -f "$PRIOR_MOTION_FILE" 2>/dev/null || true)"
  if [[ -n "$PRIOR_MOTION_ABS" && "$MOTION_ABS" != "$PRIOR_MOTION_ABS" ]]; then
    echo "Prior motion mismatch:" >&2
    echo "  policy motion: ${MOTION_FILE}" >&2
    echo "  prior motion:  ${PRIOR_MOTION_FILE}" >&2
    echo "Retrain the prior with the same MOTION_FILE, or set ALLOW_PRIOR_MOTION_MISMATCH=True." >&2
    exit 1
  fi
fi

mkdir -p "$OUT_DIR"
RUN_ENV_CONFIG="${OUT_DIR}/smp_go2_${GAIT}_track_env.yaml"
RUN_AGENT_CONFIG="${OUT_DIR}/smp_go2_trot_track_agent.yaml"
cp data/envs/smp_go2_trot_track_env.yaml "$RUN_ENV_CONFIG"
cp data/agents/smp_go2_trot_track_agent.yaml "$RUN_AGENT_CONFIG"
sed -i \
  -e "s#^motion_file:.*#motion_file: \"${MOTION_FILE}\"#" \
  "$RUN_ENV_CONFIG"
sed -i \
  -e "s#^smp_prior_cfg:.*#smp_prior_cfg: \"${PRIOR_DIR}/diffusion_config.yaml\"#" \
  -e "s#^smp_prior_model:.*#smp_prior_model: \"${PRIOR_DIR}/model.pt\"#" \
  "$RUN_AGENT_CONFIG"

echo "Training Go2 ${GAIT} track policy"
echo "Output: ${OUT_DIR}"
echo "Prior:  ${PRIOR_DIR}"
echo "Motion: ${MOTION_FILE}"
echo "TensorBoard: tensorboard --logdir=${OUT_DIR} --port=6006"

python mimickit/run.py \
  --arg_file args/smp_go2_trot_track_args.txt \
  --env_config "$RUN_ENV_CONFIG" \
  --agent_config "$RUN_AGENT_CONFIG" \
  --out_dir "$OUT_DIR" \
  --max_samples "$MAX_SAMPLES" \
  "$@"
