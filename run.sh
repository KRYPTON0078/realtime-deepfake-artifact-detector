#!/usr/bin/env bash
# Unix twin of run.bat — start the Flask demo using .venv when present.
set -euo pipefail
cd "$(dirname "$0")"
if [[ -x .venv/bin/python ]]; then
  exec .venv/bin/python app/server.py
fi
exec "${PYTHON:-python3}" app/server.py
