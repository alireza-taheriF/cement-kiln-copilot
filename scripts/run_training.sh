#!/usr/bin/env bash
# Run the model training pipeline.
#
# Usage: ./scripts/run_training.sh [path/to/model_config.yaml]
set -euo pipefail

# Resolve repo root (parent of this script's directory).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# Load environment if present.
if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

CONFIG_PATH="${1:-config/model_config.yaml}"

echo "[run_training] Using config: ${CONFIG_PATH}"
python -m src.ml.train --config "${CONFIG_PATH}"
