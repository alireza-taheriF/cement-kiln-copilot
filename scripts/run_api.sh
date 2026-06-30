#!/usr/bin/env bash
# Run the FastAPI server with uvicorn.
#
# Usage: ./scripts/run_api.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

API_HOST="${API_HOST:-0.0.0.0}"
API_PORT="${API_PORT:-8000}"

echo "[run_api] Starting API on ${API_HOST}:${API_PORT}"
exec uvicorn src.api.main:app --host "${API_HOST}" --port "${API_PORT}" --reload
