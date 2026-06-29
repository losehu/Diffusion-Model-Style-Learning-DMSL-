#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export GAIT=jump

exec "$ROOT_DIR/scripts/test_go2_trot_track_policy.sh" "$@"
