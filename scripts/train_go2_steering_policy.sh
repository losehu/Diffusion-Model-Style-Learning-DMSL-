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

PRESET_CONFIG="${PRESET_CONFIG:-data/agents/smp_go2_steering_presets.yaml}"
if [[ -f "$PRESET_CONFIG" ]]; then
  eval "$(
    GAIT="$GAIT" PRESET_CONFIG="$PRESET_CONFIG" python - <<'PY'
import os
import shlex
import yaml


def merge_dict(base, override):
    merged = dict(base or {})
    merged.update(override or {})
    return merged


def shell_value(value):
    if isinstance(value, bool):
        value = "True" if value else "False"
    return shlex.quote(str(value))


with open(os.environ["PRESET_CONFIG"], "r") as f:
    presets = yaml.safe_load(f) or {}

gait = os.environ["GAIT"]
default = presets.get("default", {})
gait_preset = presets.get(gait, {})
env_preset = merge_dict(default.get("env", {}), gait_preset.get("env", {}))
agent_preset = merge_dict(default.get("agent", {}), gait_preset.get("agent", {}))

items = {
    "MOTION_FILE": env_preset.get("motion_file"),
    "RAND_TAR_DIR": env_preset.get("rand_tar_dir"),
    "RAND_FACE_DIR": env_preset.get("rand_face_dir"),
    "SPEED_MIN": env_preset.get("tar_speed_min"),
    "SPEED_MAX": env_preset.get("tar_speed_max"),
    "REWARD_TAR_W": env_preset.get("reward_steering_tar_w"),
    "REWARD_FACE_W": env_preset.get("reward_steering_face_w"),
    "REWARD_VEL_SCALE": env_preset.get("reward_steering_vel_scale"),
    "TASK_REWARD_WEIGHT": agent_preset.get("task_reward_weight"),
    "SMP_REWARD_WEIGHT": agent_preset.get("smp_reward_weight"),
}

for name, value in items.items():
    if os.environ.get(name, ""):
        value = os.environ[name]
    if value is not None:
        print(f"{name}={shell_value(value)}")
PY
  )"
fi

if [[ -z "${MOTION_FILE:-}" ]]; then
  CLEAN_MOTION_FILE="data/motions/go2/go2_apex_${GAIT}_clean.pkl"
  if [[ -f "$CLEAN_MOTION_FILE" ]]; then
    MOTION_FILE="$CLEAN_MOTION_FILE"
  else
    MOTION_FILE="data/motions/go2/go2_apex_${GAIT}.pkl"
  fi
fi
OUT_DIR="${OUT_DIR:-output/smp_go2_${GAIT}_steering_${STAMP}}"
MAX_SAMPLES="${MAX_SAMPLES:-1310720000}"

SPEED_MIN="${SPEED_MIN:-0.5}"
SPEED_MAX="${SPEED_MAX:-5.0}"
RAND_TAR_DIR="${RAND_TAR_DIR:-True}"
RAND_FACE_DIR="${RAND_FACE_DIR:-False}"
REWARD_TAR_W="${REWARD_TAR_W:-0.7}"
REWARD_FACE_W="${REWARD_FACE_W:-0.3}"
REWARD_VEL_SCALE="${REWARD_VEL_SCALE:-0.5}"
TASK_REWARD_WEIGHT="${TASK_REWARD_WEIGHT:-0.5}"
SMP_REWARD_WEIGHT="${SMP_REWARD_WEIGHT:-0.5}"

if [[ ! -f "$MOTION_FILE" ]]; then
  echo "Missing motion file: ${MOTION_FILE}" >&2
  exit 1
fi

if [[ -z "${PRIOR_DIR:-}" ]]; then
  LATEST_PRIOR="$(find output -maxdepth 1 -type d -name "smp_prior_go2_${GAIT}_steering_*" | sort | tail -n 1)"
  if [[ -n "$LATEST_PRIOR" ]]; then
    PRIOR_DIR="$LATEST_PRIOR"
  else
    echo "No steering prior found for GAIT=${GAIT}." >&2
    echo "Run: GAIT=${GAIT} scripts/train_go2_steering_prior.sh" >&2
    exit 1
  fi
fi

if [[ ! -f "${PRIOR_DIR}/diffusion_config.yaml" || ! -f "${PRIOR_DIR}/model.pt" ]]; then
  echo "Missing prior files in ${PRIOR_DIR}" >&2
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
RUN_ENV_CONFIG="${OUT_DIR}/smp_go2_${GAIT}_steering_env.yaml"
RUN_AGENT_CONFIG="${OUT_DIR}/smp_go2_steering_agent.yaml"
cp data/envs/smp_go2_steering_env.yaml "$RUN_ENV_CONFIG"
cp data/agents/smp_go2_steering_agent.yaml "$RUN_AGENT_CONFIG"
sed -i \
  -e "s#^motion_file:.*#motion_file: \"${MOTION_FILE}\"#" \
  -e "s#^rand_tar_dir:.*#rand_tar_dir: ${RAND_TAR_DIR}#" \
  -e "s#^rand_face_dir:.*#rand_face_dir: ${RAND_FACE_DIR}#" \
  -e "s#^tar_speed_min:.*#tar_speed_min: ${SPEED_MIN}#" \
  -e "s#^tar_speed_max:.*#tar_speed_max: ${SPEED_MAX}#" \
  -e "s#^reward_steering_tar_w:.*#reward_steering_tar_w: ${REWARD_TAR_W}#" \
  -e "s#^reward_steering_face_w:.*#reward_steering_face_w: ${REWARD_FACE_W}#" \
  -e "s#^reward_steering_vel_scale:.*#reward_steering_vel_scale: ${REWARD_VEL_SCALE}#" \
  "$RUN_ENV_CONFIG"
sed -i \
  -e "s#^smp_prior_cfg:.*#smp_prior_cfg: \"${PRIOR_DIR}/diffusion_config.yaml\"#" \
  -e "s#^smp_prior_model:.*#smp_prior_model: \"${PRIOR_DIR}/model.pt\"#" \
  -e "s#^task_reward_weight:.*#task_reward_weight: ${TASK_REWARD_WEIGHT}#" \
  -e "s#^smp_reward_weight:.*#smp_reward_weight: ${SMP_REWARD_WEIGHT}#" \
  "$RUN_AGENT_CONFIG"

echo "Training Go2 ${GAIT} steering policy"
echo "Output: ${OUT_DIR}"
echo "Prior:  ${PRIOR_DIR}"
echo "Motion: ${MOTION_FILE}"
echo "Preset: ${PRESET_CONFIG}"
echo "Speed:  ${SPEED_MIN}..${SPEED_MAX} m/s"
echo "Random target direction: ${RAND_TAR_DIR}"
echo "Steering reward: tar=${REWARD_TAR_W}, face=${REWARD_FACE_W}, vel_scale=${REWARD_VEL_SCALE}"
echo "Reward weights: task=${TASK_REWARD_WEIGHT}, smp=${SMP_REWARD_WEIGHT}"

python mimickit/run.py \
  --arg_file args/smp_go2_steering_args.txt \
  --env_config "$RUN_ENV_CONFIG" \
  --agent_config "$RUN_AGENT_CONFIG" \
  --out_dir "$OUT_DIR" \
  --max_samples "$MAX_SAMPLES" \
  "$@"
