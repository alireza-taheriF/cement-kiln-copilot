#!/usr/bin/env bash
# Train a demo model end-to-end on the bundled demo dataset.
#
# Steps:
#   1. seed data/demo_signals.csv into the database (fresh tables),
#   2. train the model defined in config/model_config.yaml,
#   3. print clear next steps.
#
# Usage: ./scripts/train_demo_model.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# shellcheck source=scripts/_lib.sh
source "${SCRIPT_DIR}/_lib.sh"

# Demo default: use a local SQLite DB unless DATABASE_URL is explicitly
# exported. This is set BEFORE loading .env so the (PostgreSQL) DATABASE_URL in
# .env does not force the demo onto a database server that may not be running.
export DATABASE_URL="${DATABASE_URL:-sqlite:///./cement_copilot_dev.db}"

# Load remaining .env values WITHOUT overriding anything already set above.
load_env_file .env

CONFIG_PATH="config/model_config.yaml"
export DEMO_CSV="${DEMO_CSV:-data/demo_signals.csv}"

if [[ ! -f "${DEMO_CSV}" ]]; then
  echo "[train_demo_model] ERROR: demo dataset not found at ${DEMO_CSV}" >&2
  exit 1
fi

echo "[train_demo_model] Database: ${DATABASE_URL}"
echo "[train_demo_model] Seeding demo signals from ${DEMO_CSV} ..."
python - <<'PY'
import os

from src.data_ingest.loader import load_signals_from_csv, save_signals_to_db
from src.db.models import Base, get_engine, get_session_factory

csv_path = os.environ["DEMO_CSV"]
engine = get_engine()

# Fresh tables so re-running the demo is idempotent.
Base.metadata.drop_all(engine)
Base.metadata.create_all(engine)

session = get_session_factory(engine)()
try:
    df = load_signals_from_csv(csv_path)
    n = save_signals_to_db(df, session)
    print(f"[train_demo_model] Seeded {n} signal rows.")
finally:
    session.close()
PY

echo "[train_demo_model] Training model with ${CONFIG_PATH} ..."
echo "[train_demo_model] MLflow file store: ${MLFLOW_TRACKING_URI:-./mlruns} (registry model: cement-kiln-kpi)"
python -m src.ml.train --config "${CONFIG_PATH}"

MODEL_PATH_DEFAULT="artifacts/models/kiln_zone1_temp_model.joblib"
echo ""
echo "[train_demo_model] Done. Model artifact written to: ${MODEL_PATH_DEFAULT}"
echo ""
echo "Next steps:"
echo "  1) export MODEL_PATH=${MODEL_PATH_DEFAULT}"
echo "  2) ./scripts/run_backend.sh        # start the API"
echo "  3) ./scripts/run_dashboard.sh      # start the dashboard (new terminal)"
