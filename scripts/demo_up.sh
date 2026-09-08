#!/usr/bin/env bash
# One-command local demo: venv + deps + Flask dashboard.
# Usage:
#   ./scripts/demo_up.sh           # start dashboard (foreground)
#   ./scripts/demo_up.sh --smoke   # start, GET /health, then stop
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python3}"
HOST="${APP_HOST:-127.0.0.1}"
PORT="${APP_PORT:-5000}"
SMOKE=0

if [[ "${1:-}" == "--smoke" ]]; then
  SMOKE=1
  shift
fi

if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "error: ${PYTHON} not found. Set PYTHON=... or install Python 3.10+." >&2
  exit 1
fi

if [[ ! -d .venv ]]; then
  echo "[demo] creating virtualenv at .venv"
  "$PYTHON" -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate
echo "[demo] installing requirements (quiet)"
python -m pip install -q --upgrade pip
python -m pip install -q -r requirements.txt

export APP_HOST="$HOST"
export APP_PORT="$PORT"
export APP_DEBUG="${APP_DEBUG:-0}"

echo "[demo] dashboard: http://127.0.0.1:${PORT}"
echo "[demo] health:    GET http://127.0.0.1:${PORT}/health"
echo "[demo] CNN mode requires models/artifact_detector.pt (optional; heuristic mode is the default)"
echo "[demo] optional CNN path: python scripts/generate_demo_dataset.py && python training/train.py"

if [[ "$SMOKE" -eq 1 ]]; then
  python app/server.py &
  server_pid=$!
  cleanup() {
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  }
  trap cleanup EXIT

  echo "[demo] waiting for /health (pid ${server_pid})"
  for _ in $(seq 1 90); do
    if curl -fsS "http://127.0.0.1:${PORT}/health"; then
      echo
      echo "[demo] smoke check passed"
      exit 0
    fi
    sleep 1
  done
  echo "[demo] health check timed out after 90s" >&2
  exit 1
fi

exec python app/server.py
