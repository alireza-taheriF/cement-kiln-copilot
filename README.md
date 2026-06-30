# Cement Kiln Copilot

A production-grade AI advisory system for **cement kiln and mill optimization**. It
ingests plant time-series and lab data, trains ML models to predict key process
quality/efficiency targets, and serves real-time advisory **recommendations**
(setpoint nudges) through a REST API and an operator-facing dashboard.

---

## Project Goal

Cement pyroprocessing (the kiln) and grinding (the mills) are energy-intensive,
non-linear processes. Operators must continuously balance:

- **Clinker quality** (e.g. free lime, C3S, Blaine fineness)
- **Energy efficiency** (specific heat/electrical consumption, kWh/t)
- **Throughput and stability** (avoiding kiln upsets, ring formation)
- **Emissions** (NOx, CO)

The goal of `cement-kiln-copilot` is to act as a **decision-support copilot**:
given the current sensor/lab state of the plant, it predicts process outcomes and
recommends control adjustments that improve efficiency and quality while staying
within safe operating envelopes.

---

## Architecture

The system follows a **modular clean architecture**, separating ingestion,
persistence, ML, and serving layers so each can evolve and be tested
independently.

```
                 ┌──────────────────────────────────────────────┐
                 │                  Data Sources                  │
                 │   DCS / historian time-series  +  lab QC data  │
                 └───────────────────────┬──────────────────────┘
                                         │
                          ┌──────────────▼──────────────┐
                          │        data_ingest          │
                          │  schemas → loader → cleaner │
                          └──────────────┬──────────────┘
                                         │
                          ┌──────────────▼──────────────┐
                          │             db              │
                          │   SQLAlchemy ORM models     │
                          └──────────────┬──────────────┘
                                         │
                          ┌──────────────▼──────────────┐
                          │             ml              │
                          │ features → train → evaluate │
                          │           → recommend       │
                          └──────────────┬──────────────┘
                                         │
              ┌──────────────────────────┴──────────────────────────┐
              │                                                      │
   ┌──────────▼──────────┐                            ┌─────────────▼────────────┐
   │         api         │                            │        dashboard         │
   │  FastAPI routers:   │                            │   Streamlit operator UI  │
   │ health / prediction │◄───────────────────────────┤  (calls the API layer)   │
   │  / recommendation   │                            │                          │
   └─────────────────────┘                            └──────────────────────────┘
```

**Layer responsibilities**

| Layer          | Package            | Responsibility                                             |
| -------------- | ------------------ | ---------------------------------------------------------- |
| Ingestion      | `src/data_ingest`  | Validate, load, and clean raw plant data                   |
| Persistence    | `src/db`           | SQLAlchemy models & session management                     |
| ML             | `src/ml`           | Feature engineering, training, evaluation, recommendations |
| Serving (API)  | `src/api`          | FastAPI app + versioned routers                            |
| Serving (UI)   | `src/dashboard`    | Streamlit advisory dashboard                               |

---

## Requirements

- **Python 3.11**
- A PostgreSQL database (for persistence; SQLite works for local dev)
- See [`requirements.txt`](./requirements.txt) for the full dependency list

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then edit values
```

---

## How to Run Training

Training reads configuration from `config/model_config.yaml` and persists model
artifacts (via `joblib`) for the API/dashboard to load.

```bash
# convenience wrapper
./scripts/run_training.sh

# or directly
python -m src.ml.train --config config/model_config.yaml
```

---

## How to Run the API

The FastAPI app exposes health, prediction, and recommendation endpoints.

```bash
# convenience wrapper
./scripts/run_api.sh

# or directly
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

Interactive docs are then available at <http://localhost:8000/docs>.

---

## How to Run the Dashboard

The Streamlit dashboard is an operator-facing UI that calls the API.

```bash
streamlit run src/dashboard/app_streamlit.py
```

By default it opens at <http://localhost:8501>.

---

## 5-minute demo setup

The fastest reliable path to a working local demo. It uses the bundled
synthetic dataset (`data/demo_signals.csv`) and a local SQLite database, so no
real plant data or PostgreSQL is required.

```bash
# 1. Create and activate a virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set demo environment variables (SQLite + local artifact + local API)
export DATABASE_URL=sqlite:///./cement_copilot_dev.db
export MODEL_PATH=artifacts/models/kiln_zone1_temp_model.joblib
export API_BASE_URL=http://localhost:8000

# 4. Train the demo model (seeds demo data, then trains)
./scripts/train_demo_model.sh

# 5. Start the backend API (this terminal)
./scripts/run_backend.sh

# 6. Start the dashboard (a SECOND terminal)
./scripts/run_dashboard.sh
```

Then open the dashboard at <http://localhost:8501>.

> Shortcut: `./scripts/run_demo.sh` validates the environment, trains a model if
> one is missing, prints the dashboard command, and then runs the backend in the
> foreground.

### What the operator can demonstrate in the UI

- **Backend status** badges (health + service banner).
- **One-click demo scenario** that loads a realistic multi-tag signal window.
- **KPI prediction**: forecast of the target tag with mean/min/max and a trend
  chart.
- **Setpoint recommendations**: a ranked table of candidate setpoint
  adjustments, the best candidate highlighted, and a score bar chart.

### What the recommendation engine actually does

For a chosen `desired_target`, the engine perturbs the **adjustable tags**
around their current setpoints (by the configured `step_fractions`, optionally
clipped to `absolute_bounds`), and for **each candidate** it rebuilds the exact
model features and runs the **real trained model** to predict the KPI. Each
candidate is scored by mean absolute error to the desired target — **lower is
better** — and the best `top_k` candidates are returned. There is no heuristic
shortcut; every recommendation is model-scored.

> **Advisory only.** The system proposes setpoint adjustments for operators to
> review; it is **not** a closed-loop controller and never actuates equipment.

### Troubleshooting

| Symptom | Cause & fix |
| ------- | ----------- |
| `500` / `MODEL_PATH ... not set` or `not found` | The API needs a trained artifact. Run `./scripts/train_demo_model.sh`, then `export MODEL_PATH=artifacts/models/kiln_zone1_temp_model.joblib` before starting the backend. |
| Dashboard shows **backend unreachable** | The API is not running or `API_BASE_URL` is wrong. Start it with `./scripts/run_backend.sh` and confirm `curl http://localhost:8000/health`. |
| `400` "No rows remain after building ... features" | The input window is shorter than the largest lag/rolling window. Increase the number of rows (use the demo scenario, or raise **Rows** in the sidebar). |
| `400` "missing required base tag" | The request is missing a tag the model needs. Include all of `KILN_ZONE1_TEMP`, `KILN_ZONE2_TEMP`, `KILN_FUEL_FLOW` (the demo scenario does this). |
| `422` validation error | Malformed payload: need ≥ 2 timestamps, every signal array the same length as `timestamps`, non-empty `current_setpoints`, and each adjustable tag present in `current_setpoints`. |

---

## Factory outreach kit

Sales-ready materials for contacting cement plants, industrial partners, and
accelerators. All documents are advisory-only in positioning and designed to
be customized per plant.

| Document | Purpose |
| -------- | ------- |
| [`docs/factory_one_pager.md`](docs/factory_one_pager.md) | One-page product summary for email attachments and intro meetings |
| [`docs/factory_outreach_email.md`](docs/factory_outreach_email.md) | Cold, warm, and accelerator email templates |
| [`docs/factory_meeting_script.md`](docs/factory_meeting_script.md) | 10-minute demo meeting script (English + Persian summary) |
| [`docs/factory_discovery_questions.md`](docs/factory_discovery_questions.md) | First-meeting discovery checklist |
| [`docs/factory_poc_proposal.md`](docs/factory_poc_proposal.md) | Editable 4–8 week PoC proposal template |

---

## Factory demo script

A simple flow for a live meeting (≈ 3 minutes):

1. **Show backend health** — click *Check backend status* in the sidebar; the
   service banner and `Health: OK` confirm the system is live.
2. **Load sample payload** — click *Load demo scenario (all tags)* to populate a
   realistic signal window.
3. **Run prediction** — click *Run KPI Prediction*; point out the forecast trend
   and the mean/min/max summary for `KILN_ZONE1_TEMP`.
4. **Change desired target** — set a `desired_target` slightly away from the
   current operating point.
5. **Run recommendation** — click *Run Recommendation*; the table ranks
   candidate setpoints.
6. **Explain best candidate & score** — the top row is the recommended setpoint
   change; its **score** is the predicted distance to the desired target (lower
   is better). Emphasize this is **advisory** — operators decide whether to act.

---

## Running the dashboard end-to-end

This walks through the full advisory loop: train a model, serve it, and drive
it from the Streamlit dashboard.

### 1. Train a model

The trainer reads long-format signals from the database and writes a model
artifact to the path configured in `config/model_config.yaml`
(`model_output_path`).

```bash
export DATABASE_URL=postgresql+psycopg2://copilot:copilot@localhost:5432/cement_copilot
python -m src.ml.train --config config/model_config.yaml
```

> For a quick local run you can point `DATABASE_URL` at SQLite, e.g.
> `export DATABASE_URL=sqlite:///./cement_copilot_dev.db` (after seeding signals).

### 2. Point the API at the trained artifact

The prediction and recommendation endpoints load the model from `MODEL_PATH`.

```bash
export MODEL_PATH=artifacts/models/kiln_zone1_temp_model.joblib
```

### 3. Start the FastAPI backend

```bash
uvicorn src.api.main:app --reload
```

### 4. Run the Streamlit dashboard

```bash
export API_BASE_URL=http://localhost:8000
streamlit run src/dashboard/app_streamlit.py
```

### Endpoints used by the dashboard

| Method & path                  | Purpose                                  |
| ------------------------------ | ---------------------------------------- |
| `GET /health`                  | Liveness probe                           |
| `GET /`                        | Service/version banner                   |
| `POST /v1/predict/energy-kpi`  | Forecast the target KPI over a window    |
| `POST /v1/recommend/setpoints` | Rank model-scored setpoint adjustments   |

### Dashboard capabilities

- **Backend status panel** with health/root badges and graceful failures.
- **Input builder** in two modes: a deterministic synthetic sample generator
  (`KILN_ZONE1_TEMP`, `KILN_FUEL_FLOW`, optional `KILN_ZONE2_TEMP`) and a
  manual JSON editor with validation.
- **Prediction panel**: summary metrics, a predictions line chart, and the raw
  response.
- **Recommendation panel**: desired-target / adjustable-tags / step-fractions /
  top-k / optional absolute-bounds controls, a ranked recommendations table with
  the best candidate highlighted, a score bar chart, and the raw response.
- Clear error surfaces for `400` / `422` / `500` responses.

> **Advisory only.** The system proposes setpoint adjustments for operators to
> review; it is **not** a closed-loop controller and never actuates equipment.

---

## Testing

```bash
pytest -q
```

---

## Repository Layout

```
cement-kiln-copilot/
  data/            sample CSVs for local development
  notebooks/       exploratory analysis (kiln & mill)
  config/          YAML configuration (model, db)
  src/
    data_ingest/   schemas, loader, cleaner
    db/            SQLAlchemy models
    ml/            features, models, train, evaluate, recommend
    api/           FastAPI app + routers
    dashboard/     Streamlit app
  tests/           unit/integration tests
  scripts/         shell entrypoints
```

---

## Status

The core advisory loop is implemented end-to-end: data ingestion, the
industrial DB schema, feature engineering, model training/persistence, the
prediction and recommendation APIs, and the Streamlit dashboard are all wired
together. The system is **advisory only** (open-loop) and is intended as an
MVP/demo for cement plants rather than a closed-loop controller.
