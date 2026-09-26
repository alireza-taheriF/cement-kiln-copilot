# Cement Kiln Copilot

A production-grade AI advisory system for **cement kiln and mill optimization**. It
ingests plant time-series and lab data, trains ML models to predict key process
quality/efficiency targets, and serves real-time advisory **recommendations**
(setpoint nudges) through a REST API and an operator-facing dashboard.

> **مستند فارسی** (معماری، راه‌اندازی و محدودیت‌ها):
> [`docs/DOCUMENTATION_FA.md`](docs/DOCUMENTATION_FA.md)

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
# convenience wrapper (logs the run to the local MLflow file store ./mlruns)
./scripts/run_training.sh

# demo data, SQLite, no plant historian and no cloud account
export DATABASE_URL=sqlite:///./cement_copilot_dev.db
./scripts/train_demo_model.sh

# or directly
python -m src.ml.train --config config/model_config.yaml
```

Each training run records the YAML config as MLflow params, the trainer's
evaluation metrics, and the joblib artifact, and registers `cement-kiln-kpi`
in the local registry. See [MLOps](#mlops) for the tracking URI, the CI gate,
and drift monitoring.

## Demo baseline comparison

The demo trainer scores two naive forecasts on the **same chronological
validation split** as the energy model (`train_valid_split_time_series` in
`src/ml/evaluate.py`; the rows are not shuffled). Both use only the training
portion to fit. Persistence predicts the target value already observed at each
forecast origin (not the future label). Ordinary linear regression uses the
same feature columns as the model.

- Target: `KILN_ZONE1_TEMP`, horizon `3`
- Config: `config/model_config.yaml`
- Data: `data/demo_signals.csv` (bundled synthetic demo signals, not a plant historian)
- Model artifact path (unchanged): `artifacts/models/kiln_zone1_temp_model.joblib`

Command that produced the numbers below (local SQLite, no cloud credentials):

```bash
export DATABASE_URL=sqlite:///./cement_copilot_dev.db
./scripts/train_demo_model.sh
```

The trainer printed:

```text
Validation comparison n_samples=16 model_valid_rmse=2.3600270120946925 persistence_rmse=2.861225567829288 linear_regression_rmse=0.022998466648510584
```

The same run's JSON summary included this `baseline_comparison` object (16 validation rows out of 81 supervised rows; 65 rows were used for training):

```json
{
  "n_samples": 16,
  "model_valid_rmse": 2.3600270120946925,
  "persistence": {
    "rmse": 2.861225567829288,
    "mae": 2.3388750000000016,
    "n_samples": 16
  },
  "linear_regression": {
    "rmse": 0.022998466648510584,
    "mae": 0.020028990999449547,
    "n_samples": 16
  }
}
```

On these 16 rows the model RMSE is lower than persistence and higher than ordinary linear regression. The model does not beat both baselines.

The CI gate is still [`tests/baselines/demo_metrics.json`](tests/baselines/demo_metrics.json) (`valid_rmse` = `2.3600270120946925`, lower is better). This training run did not move that model RMSE, so the gate file was not rewritten. The naive scores are not a replacement for that gate.

These figures are forecast error on synthetic demo data. They are not a plant energy saving. The product is advisory only and does not actuate equipment.

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
| `GET /v1/monitor/drift`        | Feature z-scores versus training stats   |

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

## MLOps

Training, CI, and monitoring stay on the bundled demo CSV and a local SQLite
database. Nothing here calls a plant historian or a cloud account. The product
remains **advisory only**: it recommends setpoints and never actuates equipment.
`POST /v1/predict/energy-kpi` and `POST /v1/recommend/setpoints` keep their
existing JSON schemas. The dashboard still calls those endpoints, and shows a
small drift-status panel from `GET /v1/monitor/drift`.

### Train (MLflow)

```bash
export DATABASE_URL=sqlite:///./cement_copilot_dev.db
# optional; this directory is the default file store
export MLFLOW_TRACKING_URI=./mlruns
./scripts/train_demo_model.sh
```

Equivalent direct command, after the demo CSV has been seeded:

```bash
python -m src.ml.train --config config/model_config.yaml
```

Params come from `config/model_config.yaml`. Metrics are the trainer's
train/validation scores, plus `model_valid_rmse`, `persistence_rmse`, and
`linear_regression_rmse` from the [demo baseline comparison](#demo-baseline-comparison).
The joblib file is logged as an artifact and
registered as `cement-kiln-kpi`, with the `Production` alias updated to that
version. Feature means and standard deviations are written next to the joblib
(`*.feature_stats.json`).

The API still loads `MODEL_PATH` when `MODEL_URI` is unset. To load the
registered model instead:

```bash
export MODEL_URI=models:/cement-kiln-kpi/Production
```

The file store cannot host MLflow's model registry, so the registry is a
SQLite file at `./mlruns/registry.db` (override with `MLFLOW_REGISTRY_URI`).

### Local train, image, cluster, cloud dry-run

Do these in order. Local training stays the default. Nothing in this path
actuates equipment.

**1. Local train**

```bash
export DATABASE_URL=sqlite:///./cement_copilot_dev.db
python -m src.ml.train --config config/model_config.yaml
```

**2. Docker build**

The API image runs `uvicorn src.api.main:app` and reads `MODEL_PATH` and
`DATABASE_URL` from the environment (they are not baked into the image). The
Streamlit image is a separate, smaller Dockerfile.

```bash
docker build -t cement-kiln-copilot-api:local .
docker build -f Dockerfile.streamlit -t cement-kiln-copilot-dashboard:local .
```

**3. Kubernetes apply**

Manifests live in [`deploy/k8s/`](deploy/k8s/README.md). For kind or minikube,
load the local images, then:

```bash
kubectl apply -k deploy/k8s
kubectl port-forward svc/cement-copilot-api 8000:8000
kubectl port-forward svc/cement-copilot-dashboard 8501:8501
```

`secret.example.yaml` is a placeholder (`CHANGE_ME`). Do not commit real
passwords. The MLflow PVC is for a local kind/minikube cluster only, not a
production artifact store.

**4. Cloud dry-run**

Each launcher prints the same training command inside its job spec and exits 0.
No AWS, GCP, or Azure credentials are required, and CI runs only these dry-runs.

```bash
python -m src.cloud.sagemaker_job --dry-run
python -m src.cloud.vertex_job --dry-run
python -m src.cloud.azureml_job --dry-run
```

`--submit` sends that job only when the expected account environment is set
(`AWS_*` / `SAGEMAKER_*`, `GOOGLE_*` / `VERTEX_TRAINING_IMAGE`, or `AZURE_*` /
`AZUREML_*`). Submit needs a real cloud account and the optional SDK
(`boto3`, `google-cloud-aiplatform`, or `azure-ai-ml` + `azure-identity`).
It is not exercised in CI.

### How CI judges the baseline

[`.github/workflows/model-ci.yml`](.github/workflows/model-ci.yml) installs
`requirements.txt`, runs `pytest -q`, trains on `data/demo_signals.csv` (no
external dataset), then runs:

```bash
python -m src.ml.check_baseline
```

That compares held-out RMSE (`valid_rmse`) to
[`tests/baselines/demo_metrics.json`](tests/baselines/demo_metrics.json). The
job fails when the new score is worse than the committed `value` by more than
the `tolerance` in that file (relative or absolute, `direction` is
lower-is-better). A better score passes. The checked-in baseline uses a
relative tolerance of `0.15` so small machine-to-machine differences on the
16-row demo validation window do not fail the job.

Refresh the committed value after an intentional demo-metric change, from the
repository root and after a demo training run:

```bash
python -m src.ml.check_baseline --write
```

The first write creates the file and records a small relative tolerance of
`0.05`. Later writes keep the tolerance already in the file and replace
`value`.

### Monitor

```bash
curl -s "http://localhost:8000/v1/monitor/drift?n=20"
curl -s http://localhost:8000/metrics
```

`GET /v1/monitor/drift` returns a per-feature z-score of the last N prediction
inputs against the training means and standard deviations, plus `max_abs_z`.
`GET /metrics` is Prometheus text (request count and a latency histogram).
Successful predict and recommend calls also append a row to the `inference_events`
table: timestamp, route, latency, model version, and a hash of the numeric input.

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
  deploy/k8s/      local kind/minikube manifests
  src/
    data_ingest/   schemas, loader, cleaner
    db/            SQLAlchemy models
    ml/            features, models, train, evaluate, recommend
    api/           FastAPI app + routers
    dashboard/     Streamlit app
    cloud/         SageMaker, Vertex AI, and Azure ML job launchers
  tests/           unit/integration tests
  scripts/         shell entrypoints
  Dockerfile       FastAPI image
  Dockerfile.streamlit
```

---

## Status

The core advisory loop is implemented end-to-end: data ingestion, the
industrial DB schema, feature engineering, model training/persistence, the
prediction and recommendation APIs, and the Streamlit dashboard are all wired
together. The system is **advisory only** (open-loop) and is intended as an
MVP/demo for cement plants rather than a closed-loop controller.
