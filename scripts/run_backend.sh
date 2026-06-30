#!/usr/bin/env bash
# Start the FastAPI backend (prediction + recommendation APIs).
#
# Requires a trained model artifact referenced by MODEL_PATH.
#
# Usage: ./scripts/run_backend.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# shellcheck source=scripts/_lib.sh
source "${SCRIPT_DIR}/_lib.sh"

# Load .env WITHOUT overriding variables already exported in this shell.
load_env_file .env

API_HOST="${API_HOST:-0.0.0.0}"
API_PORT="${API_PORT:-8000}"
export MODEL_PATH="${MODEL_PATH:-artifacts/models/kiln_zone1_temp_model.joblib}"

if [[ ! -f "${MODEL_PATH}" ]]; then
  echo "[run_backend] ERROR: model artifact not found at MODEL_PATH=${MODEL_PATH}" >&2
  echo "[run_backend] Train one first:  ./scripts/train_demo_model.sh" >&2
  exit 1
fi

echo "[run_backend] MODEL_PATH=${MODEL_PATH}"
echo "[run_backend] API:    http://localhost:${API_PORT}"
echo "[run_backend] Docs:   http://localhost:${API_PORT}/docs"
echo "[run_backend] Health: http://localhost:${API_PORT}/health"
exec uvicorn src.api.main:app --host "${API_HOST}" --port "${API_PORT}" --reload
