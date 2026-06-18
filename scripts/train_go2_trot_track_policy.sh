#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

STAMP="$(date +%m%d%H%M%S)"
OUT_DIR="${OUT_DIR:-output/smp_go2_trot_track_${STAMP}}"
MAX_SAMPLES="${MAX_SAMPLES:-1310720000}"

if [[ -z "${PRIOR_DIR:-}" ]]; then
  LATEST_PRIOR="$(find output -maxdepth 1 -type d -name 'smp_prior_go2_trot_track_*' | sort | tail -n 1)"
  if [[ -n "$LATEST_PRIOR" ]]; then
    PRIOR_DIR="$LATEST_PRIOR"
  else
    PRIOR_DIR="output/smp_go2_trot_track_reward/prior"
  fi
fi

if [[ ! -f "${PRIOR_DIR}/diffusion_config.yaml" || ! -f "${PRIOR_DIR}/model.pt" ]]; then
  echo "Missing prior files in ${PRIOR_DIR}" >&2
  echo "Set PRIOR_DIR=/path/to/prior_dir or run scripts/train_go2_trot_track_prior.sh first." >&2
  exit 1
fi

mkdir -p "$OUT_DIR"
RUN_AGENT_CONFIG="${OUT_DIR}/smp_go2_trot_track_agent.yaml"
cp data/agents/smp_go2_trot_track_agent.yaml "$RUN_AGENT_CONFIG"
sed -i \
  -e "s#^smp_prior_cfg:.*#smp_prior_cfg: \"${PRIOR_DIR}/diffusion_config.yaml\"#" \
  -e "s#^smp_prior_model:.*#smp_prior_model: \"${PRIOR_DIR}/model.pt\"#" \
  "$RUN_AGENT_CONFIG"

echo "Training Go2 trot track policy"
echo "Output: ${OUT_DIR}"
echo "Prior:  ${PRIOR_DIR}"

python mimickit/run.py \
  --arg_file args/smp_go2_trot_track_args.txt \
  --agent_config "$RUN_AGENT_CONFIG" \
  --out_dir "$OUT_DIR" \
  --max_samples "$MAX_SAMPLES" \
  "$@"
