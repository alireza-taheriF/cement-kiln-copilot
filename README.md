# Cement Kiln Copilot

Open-loop decision support that lowers **fuel and electricity cost** in cement
plants: kiln firing, gas ↔ mazut transfers, mill scheduling under the power
quota, clinker quality between lab samples, and early equipment warnings. Every
saving is measured with an agreed M&V method so it can be invoiced.

> **مستند فارسی:** [`docs/DOCUMENTATION_FA.md`](docs/DOCUMENTATION_FA.md) ·
> **Safety case:** [`docs/SAFETY.md`](docs/SAFETY.md)

**Status: pre-pilot.** Everything below runs and is tested on a *simulated*
3,000 t/d line with a known ground truth (`src/sim`). No real plant data is in
this repository and no plant pilot has been run yet. The copilot is advisory
only: it never writes to the DCS.

---

## What it does

| # | Capability | Module | Endpoint |
| - | ---------- | ------ | -------- |
| 1 | **Mill scheduling under the power quota.** MILP (HiGHS via SciPy) that decides which mill runs in which hour given the quota, regulated vs free-market tariff, silo limits and dispatch; compared like-for-like with the plant's silo-level rule. | `src/energy/mill_scheduler.py` | `POST /v1/energy/mill-schedule` |
| 2 | **Measured savings (M&V).** Baseline regression of energy on drivers, ASHRAE Guideline 14 checks (CV(RMSE), NMBE), adjusted baseline, savings with fractional-savings uncertainty and autocorrelation correction, valued in money. | `src/mv/baseline.py` | `POST /v1/energy/mv-report` |
| 3 | **Gas ↔ mazut transfer advisor.** Detects past transfers, measures the upset each caused, learns the least-upsetting ramp from the plant's own history, plans burner flows at constant heat input with an ID-fan bias for the O2 dip. | `src/fuel/switch.py` | `POST /v1/fuel/transfer-plan` |
| 4 | **Free-lime soft sensor.** Searches the lab delay, fits a ridge model on process-window aggregates, gives a conformal interval, and corrects its bias with every new lab result. | `src/softsensor/free_lime.py` | `POST /v1/quality/free-lime` |
| 5 | **Business KPIs as targets.** Specific heat (kcal/kg clinker), gas heat share, mill kWh/t, computed from raw tags with explicit fuel heating values. | `src/domain/kpis.py` | used everywhere |
| 6 | **Read-only plant connectors.** Historian CSV exports, Excel lab sheets (Jalali dates, Persian digits), OPC UA history reads; tag mapping with unit conversion and validity ranges; idempotent sync. A test fails if any connector calls a write service. | `src/connectors/` | `python -m src.connectors.sync` |
| 7 | **Safety envelope, explanations, operator feedback.** Lever ranges and rate limits, process limits tested on the pessimistic end of the prediction interval, a redundant final gate, explanations in Persian and English, accept/reject with reason codes and acceptance statistics. | `src/safety/`, `src/advisory/` | `POST /v1/advise/kiln`, `/v1/advise/feedback`, `GET /v1/advise/stats` |
| 8 | **Robust models.** Transport-delay estimation for lags, delta targets, ridge for deterministic constraint models, split-conformal intervals; the advisor checks each model's local response against process knowledge and uses **bounded gains** (from step tests) instead of trusting a confounded learnt gain. | `src/ml/`, `src/advisory/advisor.py` | — |
| 9 | **Condition monitoring.** Kiln-shell hot spots and mill vibration/bearing temperature: robust z-score against the asset's own healthy reference, trend, hours to alarm. | `src/maintenance/condition.py` | `POST /v1/maintenance/condition` |
| 10 | **Shift handover report.** Every number computed from data; Persian and English templates; optional rewrite by a *local* LLM that is rejected if it changes or invents a number. Alarm episodes and bad actors rebuilt from signals. | `src/reports/` | `POST /v1/reports/shift` |
| 11 | **Shadow mode and closed-loop readiness.** Replays the advice log against what operators and the plant did: agreement, violations after followed advice, interval coverage; explicit go/no-go criteria. There is deliberately no DCS writer. | `src/control/shadow.py` | `GET /v1/control/readiness` |

How the kiln advisor decides (details in [`src/advisory/advisor.py`](src/advisory/advisor.py)):

1. The objective (kcal/kg) **is** a lever, so it is computed exactly, never predicted.
2. Constraint models forecast burning-zone temperature (75 min ahead) and CO with 90 % intervals.
3. A lever is used only if each model's local response has the physically expected sign.
4. A lever's effect is `forecast + gain × change`, with the gain clipped into step-test bounds; the interval uses the *worst* gain.
5. Candidates are generated inside the lever range and max step; a candidate is feasible only if the pessimistic interval end is inside every process limit.
6. A change is proposed only if it saves at least `min_improvement`; otherwise "keep current".
7. The chosen change passes the safety envelope a second time before it is returned.

`tests/test_advisor_and_safety.py` applies recommendations **in the simulator**
(same random draws, only the setpoint changed) and checks the real burning zone
and CO stay inside their limits over the next three hours.

---

## Quick start (simulated plant, SQLite)

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

./scripts/train_demo_model.sh     # simulate 14 days, seed SQLite, train all models
./scripts/run_backend.sh          # API on http://localhost:8000 (docs at /docs)
./scripts/run_dashboard.sh        # second terminal: dashboard on http://localhost:8501
```

The dashboard is Persian by default (RTL) with an English switch. Each tab can
load a window of the simulated plant (`GET /v1/demo/window`).

### Configuration

| File | Purpose |
| ---- | ------- |
| `config/plant.yaml` | Fuel heating values, clinker factor, mills, tariffs (Toman) |
| `config/safety_limits.yaml` | **Every limit** the copilot respects (lever ranges/steps, process limits) |
| `config/advisor.yaml` | Objective, levers, constraint models, expected effects, gain bounds |
| `config/model_config.yaml` | Burning-zone constraint model (ridge, delta target, auto lags) |
| `config/models/co.yaml` | CO constraint model |
| `config/connectors.yaml` | Plant data sources and tag mapping |

Defaults describe the simulated plant. Before a pilot the plant's process
engineer replaces the limits and gain bounds and signs them off (`docs/SAFETY.md`).

### Training individual models

```bash
python -m src.ml.train --config config/model_config.yaml   # burning zone
python -m src.ml.train --config config/models/co.yaml      # CO
python -m src.softsensor.train                             # free lime
```

Runs are tracked in a local MLflow file store (`./mlruns`), each target under its
own registered model. Artifacts land in `artifacts/models/`.

### Plant data

```bash
python -m src.connectors.sync --config config/connectors.yaml --since 2026-11-01 --dry-run
```

prints a data-quality report (unmapped tags, rejected values) without writing.
Without `--dry-run` it stores signals and lab results, skipping rows already stored.

---

## Repository layout

```
config/          plant, safety, advisor, model and connector configuration
src/
  domain/        canonical tags, KPI formulas
  sim/           plant simulator with ground truth (demo + tests)
  connectors/    read-only sources, Jalali dates, tag mapping, sync
  data_ingest/   CSV loaders, DB persisters
  db/            SQLAlchemy models (signals, lab, inference, advice, feedback)
  ml/            features, delays, uncertainty, pipeline, training, inference
  safety/        safety envelope
  advisory/      kiln advisor, operator feedback
  energy/        mill scheduler
  mv/            measurement & verification
  fuel/          gas/mazut transfer advisor
  softsensor/    free-lime soft sensor
  maintenance/   condition monitoring
  reports/       alarms, shift report
  control/       shadow mode, readiness (no DCS writes)
  api/           FastAPI app and routers
  dashboard/     Streamlit operator UI
deploy/k8s/      local kind/minikube manifests
docs/            Persian documentation, safety case, factory outreach kit
tests/           unit, simulator-validation and API tests
```

## Testing and CI

```bash
ruff check src tests
pytest
```

CI (`.github/workflows/model-ci.yml`) lints, runs the tests, trains every demo
model with `scripts/train_demo_model.sh`, and fails if the burning-zone model's
validation RMSE regresses more than 5 % past `tests/baselines/demo_metrics.json`
(the ridge model is deterministic, so the tolerance is tight). Refresh it after
an intentional change with `python -m src.ml.check_baseline --write`.

## Limitations (read before selling)

- All validation is on a simulator whose coefficients are plausible, not measured.
  The first pilot must produce the first real numbers.
- 75 minutes ahead the burning zone is barely more predictable than "stays where it
  is"; the advisor relies on honest intervals and bounded gains, not on forecast skill.
  Gain bounds must come from plant step tests.
- The advisor is a quasi-steady-state what-if, not a dynamic optimizer (that is APC).
- M&V needs a baseline period of normal operation (weeks) before savings can be billed.
- Closed-loop control is out of scope; `src/control` produces the evidence needed to
  propose it, together with HAZOP and management of change.

## Factory outreach kit

[`docs/factory_one_pager.md`](docs/factory_one_pager.md) ·
[`docs/factory_outreach_email.md`](docs/factory_outreach_email.md) ·
[`docs/factory_meeting_script.md`](docs/factory_meeting_script.md) ·
[`docs/factory_discovery_questions.md`](docs/factory_discovery_questions.md) ·
[`docs/factory_poc_proposal.md`](docs/factory_poc_proposal.md)
