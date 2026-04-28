#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export PYTHONPATH="$ROOT/legacy_v3b_restore/external/tdmpc2/tdmpc2:$ROOT/legacy_v3b_restore:$ROOT/external/tdmpc2/tdmpc2:$ROOT:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-glfw}"

python -u "$ROOT/legacy_v3b_restore/external/tdmpc2/tdmpc2/train.py" "$@"
