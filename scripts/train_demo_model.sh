#!/usr/bin/env bash
# Train every demo model end-to-end on the simulated plant.
#
# Steps:
#   1. simulate 14 days of a 3,000 t/d line and seed it into the database,
#   2. train the burning-zone temperature model (config/model_config.yaml),
#   3. train the CO model (config/models/co.yaml),
#   4. train the free-lime soft sensor from the lab results.
#
# Usage: ./scripts/train_demo_model.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# shellcheck source=scripts/_lib.sh
source "${SCRIPT_DIR}/_lib.sh"

# Demo default: local SQLite unless DATABASE_URL is explicitly exported. Set
# BEFORE loading .env so its PostgreSQL URL does not force a server.
export DATABASE_URL="${DATABASE_URL:-sqlite:///./cement_copilot_dev.db}"
load_env_file .env

DEMO_DIR="${DEMO_DIR:-data/demo}"
DEMO_DAYS="${DEMO_DAYS:-14}"

echo "[train_demo_model] Database: ${DATABASE_URL}"
echo "[train_demo_model] Simulating ${DEMO_DAYS} days into ${DEMO_DIR} and seeding the database ..."
python -m src.sim.generate --out "${DEMO_DIR}" --days "${DEMO_DAYS}" --seed-db

echo "[train_demo_model] Training the burning-zone model (config/model_config.yaml) ..."
python -m src.ml.train --config config/model_config.yaml

echo "[train_demo_model] Training the CO model (config/models/co.yaml) ..."
python -m src.ml.train --config config/models/co.yaml

echo "[train_demo_model] Training the free-lime soft sensor ..."
python -m src.softsensor.train

cat <<MSG

[train_demo_model] Done. Artifacts in artifacts/models/.

Next steps:
  1) ./scripts/run_backend.sh        # start the API
  2) ./scripts/run_dashboard.sh      # start the dashboard (new terminal)
MSG
