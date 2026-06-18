#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -z "${MODEL_FILE:-}" ]]; then
  if [[ $# -gt 0 ]]; then
    MODEL_FILE="$1"
  else
    LATEST_POLICY="$(find output -maxdepth 1 -type d -name 'smp_go2_trot_track_[0-9]*' | sort | tail -n 1)"
    if [[ -n "$LATEST_POLICY" && -f "${LATEST_POLICY}/model.pt" ]]; then
      MODEL_FILE="${LATEST_POLICY}/model.pt"
    else
      MODEL_FILE="output/smp_go2_trot_track_reward/model.pt"
    fi
  fi
fi
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-6006}"
ENV_CONFIG="${ENV_CONFIG:-data/envs/smp_go2_trot_track_env.yaml}"

MODEL_DIR="$(dirname "$MODEL_FILE")"
if [[ -f "${MODEL_DIR}/agent_config.yaml" ]]; then
  AGENT_CONFIG="${AGENT_CONFIG:-${MODEL_DIR}/agent_config.yaml}"
elif [[ -f "${MODEL_DIR}/smp_go2_trot_track_agent.yaml" ]]; then
  AGENT_CONFIG="${AGENT_CONFIG:-${MODEL_DIR}/smp_go2_trot_track_agent.yaml}"
else
  AGENT_CONFIG="${AGENT_CONFIG:-data/agents/smp_go2_trot_track_agent.yaml}"
fi

echo "Testing Go2 trot track policy"
echo "Model:  ${MODEL_FILE}"
echo "Agent:  ${AGENT_CONFIG}"
echo "URL:    http://${HOST}:${PORT}"

python tools/visualize_go2_policy_viser.py \
  --model_file "$MODEL_FILE" \
  --env_config "$ENV_CONFIG" \
  --agent_config "$AGENT_CONFIG" \
  --host "$HOST" \
  --port "$PORT"
