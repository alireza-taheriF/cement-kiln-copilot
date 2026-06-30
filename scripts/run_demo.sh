#!/usr/bin/env bash
# Orchestrate the local demo happy path.
#
# This intentionally does NOT manage multiple processes. It validates the
# environment, ensures a model exists (training one if needed), prints how to
# launch the dashboard, then runs the backend in the foreground.
#
# Usage: ./scripts/run_demo.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# shellcheck source=scripts/_lib.sh
source "${SCRIPT_DIR}/_lib.sh"

# Demo defaults set BEFORE loading .env so an explicit `export` wins, the demo
# falls back to local SQLite, and the (PostgreSQL) DATABASE_URL in .env does not
# force the demo onto a database server that may not be running.
export DATABASE_URL="${DATABASE_URL:-sqlite:///./cement_copilot_dev.db}"
export MODEL_PATH="${MODEL_PATH:-artifacts/models/kiln_zone1_temp_model.joblib}"
export API_BASE_URL="${API_BASE_URL:-http://localhost:8000}"

# Load remaining .env values WITHOUT overriding anything already set above.
load_env_file .env

# 1. Validate environment ----------------------------------------------------
if ! command -v python >/dev/null 2>&1; then
  echo "[run_demo] ERROR: 'python' not found on PATH." >&2
  exit 1
fi

# 2. Ensure a model artifact exists ------------------------------------------
if [[ ! -f "${MODEL_PATH}" ]]; then
  echo "[run_demo] No model at ${MODEL_PATH}; training a demo model ..."
  ./scripts/train_demo_model.sh
else
  echo "[run_demo] Using existing model: ${MODEL_PATH}"
fi

# 3. Print dashboard instructions (run in a second terminal) -----------------
cat <<EOF

============================================================
 Cement Kiln Copilot — local demo
============================================================
 Backend (this terminal): ${API_BASE_URL}
 API docs:                ${API_BASE_URL}/docs

 In a SECOND terminal, start the dashboard:
   ./scripts/run_dashboard.sh

 Then open the dashboard (default): http://localhost:8501
============================================================

EOF

# 4. Start the backend in the foreground -------------------------------------
exec ./scripts/run_backend.sh
