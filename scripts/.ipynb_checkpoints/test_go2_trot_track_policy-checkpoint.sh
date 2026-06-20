#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

GAIT="${GAIT:-trot}"
case "$GAIT" in
  trot|pace|canter|jump) ;;
  *)
    echo "Unsupported GAIT=${GAIT}. Use one of: trot, pace, canter, jump." >&2
    exit 1
    ;;
esac

if [[ -z "${MODEL_FILE:-}" ]]; then
  if [[ $# -gt 0 ]]; then
    MODEL_FILE="$1"
  else
    LATEST_POLICY="$(find output -maxdepth 1 -type d -name "smp_go2_${GAIT}_track_[0-9]*" | sort | tail -n 1)"
    if [[ -n "$LATEST_POLICY" && -f "${LATEST_POLICY}/model.pt" ]]; then
      MODEL_FILE="${LATEST_POLICY}/model.pt"
    elif [[ "$GAIT" == "trot" ]]; then
      MODEL_FILE="output/smp_go2_trot_track_reward/model.pt"
    else
      echo "No policy found for GAIT=${GAIT}." >&2
      echo "Run: GAIT=${GAIT} scripts/train_go2_trot_track_policy.sh" >&2
      exit 1
    fi
  fi
fi
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-6006}"

MODEL_DIR="$(dirname "$MODEL_FILE")"
if [[ -z "${ENV_CONFIG:-}" ]]; then
  if [[ -f "${MODEL_DIR}/env_config.yaml" ]]; then
    ENV_CONFIG="${MODEL_DIR}/env_config.yaml"
  elif [[ -f "${MODEL_DIR}/smp_go2_${GAIT}_track_env.yaml" ]]; then
    ENV_CONFIG="${MODEL_DIR}/smp_go2_${GAIT}_track_env.yaml"
  else
    ENV_CONFIG="data/envs/smp_go2_trot_track_env.yaml"
  fi
fi

if [[ -f "${MODEL_DIR}/agent_config.yaml" ]]; then
  AGENT_CONFIG="${AGENT_CONFIG:-${MODEL_DIR}/agent_config.yaml}"
elif [[ -f "${MODEL_DIR}/smp_go2_trot_track_agent.yaml" ]]; then
  AGENT_CONFIG="${AGENT_CONFIG:-${MODEL_DIR}/smp_go2_trot_track_agent.yaml}"
else
  AGENT_CONFIG="${AGENT_CONFIG:-data/agents/smp_go2_trot_track_agent.yaml}"
fi

echo "Testing Go2 trot track policy"
echo "Gait:   ${GAIT}"
echo "Model:  ${MODEL_FILE}"
echo "Env:    ${ENV_CONFIG}"
echo "Agent:  ${AGENT_CONFIG}"
echo "URL:    http://${HOST}:${PORT}"

python tools/visualize_go2_policy_viser.py \
  --model_file "$MODEL_FILE" \
  --env_config "$ENV_CONFIG" \
  --agent_config "$AGENT_CONFIG" \
  --host "$HOST" \
  --port "$PORT"
