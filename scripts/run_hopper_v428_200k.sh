#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "[deprecated] scripts/run_hopper_v428_200k.sh now forwards to v429 mainline." >&2
exec "$ROOT/run_hopper_v429_200k.sh" "$@"
