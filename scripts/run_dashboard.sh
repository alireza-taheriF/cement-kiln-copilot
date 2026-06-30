#!/usr/bin/env bash
# Start the Streamlit operator dashboard.
#
# The dashboard talks to the backend at API_BASE_URL.
#
# Usage: ./scripts/run_dashboard.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# shellcheck source=scripts/_lib.sh
source "${SCRIPT_DIR}/_lib.sh"

# Load .env WITHOUT overriding variables already exported in this shell.
load_env_file .env

export API_BASE_URL="${API_BASE_URL:-http://localhost:8000}"
DASHBOARD_PORT="${DASHBOARD_PORT:-8501}"

echo "[run_dashboard] Backend API: ${API_BASE_URL}"
echo "[run_dashboard] Dashboard:   http://localhost:${DASHBOARD_PORT}"
exec streamlit run src/dashboard/app_streamlit.py --server.port "${DASHBOARD_PORT}"
