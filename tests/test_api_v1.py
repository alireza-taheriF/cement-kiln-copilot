"""HTTP tests for the advisor, energy and operations endpoints."""

from __future__ import annotations

import json
import math

import joblib
import numpy as np
import pandas as pd
import pytest
import yaml

pytest.importorskip("fastapi")
pytest.importorskip("sqlalchemy")

from fastapi.testclient import TestClient  # noqa: E402

from src.api import services  # noqa: E402
from src.api.main import app  # noqa: E402
from src.domain import tags as T  # noqa: E402
from src.softsensor.free_lime import FreeLimeSoftSensor  # noqa: E402

client = TestClient(app, raise_server_exceptions=False)

DERIVED = {T.SPECIFIC_HEAT_KCAL_KG, T.GAS_HEAT_SHARE, T.HEAT_INPUT_GCALH}


def _window(frame: pd.DataFrame) -> dict:
    raw = frame[[c for c in frame.columns if c not in DERIVED and "_KWH_T" not in c]]
    return {
        "timestamps": [ts.isoformat() for ts in raw.index],
        "signals": {
            col: [None if (isinstance(v, float) and math.isnan(v)) else float(v) for v in raw[col]]
            for col in raw.columns
        },
    }


@pytest.fixture()
def configured(tmp_path, monkeypatch, sim, constraint_models):
    """Advisor and soft sensor artifacts in a temp dir, wired through env vars."""
    paths = {}
    for role, model in constraint_models.items():
        path = tmp_path / f"{role}.joblib"
        model.save(str(path))
        paths[role] = str(path)
    advisor_cfg = yaml.safe_load(open("config/advisor.yaml", encoding="utf-8"))
    advisor_cfg["constraint_models"] = paths
    advisor_path = tmp_path / "advisor.yaml"
    advisor_path.write_text(yaml.safe_dump(advisor_cfg), encoding="utf-8")
    cut = sim.wide.index[3200]
    sensor = FreeLimeSoftSensor.fit(sim.wide.loc[:cut], sim.lab[sim.lab["timestamp"] < cut])
    sensor_path = tmp_path / "fl.joblib"
    joblib.dump(sensor, sensor_path)
    monkeypatch.setenv("ADVISOR_CONFIG", str(advisor_path))
    monkeypatch.setenv("FREE_LIME_MODEL", str(sensor_path))
    services.reset_cache()
    yield
    services.reset_cache()


def test_advise_feedback_stats_log_round_trip(configured, sim):
    response = client.post("/v1/advise/kiln", json=_window(sim.wide.iloc[3390:3450]))
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] in {"recommend", "keep_current"}
    assert body["advice_id"] >= 1
    assert body["explanation_fa"]
    json.dumps(body, allow_nan=False)

    rejected = client.post(
        "/v1/advise/feedback",
        json={"advice_id": body["advice_id"], "decision": "rejected", "operator": "ali"},
    )
    assert rejected.status_code == 422  # a rejection needs a reason
    ok = client.post(
        "/v1/advise/feedback",
        json={
            "advice_id": body["advice_id"],
            "decision": "rejected",
            "operator": "ali",
            "reason": "quality_risk",
            "shift": "A",
        },
    )
    assert ok.status_code == 200, ok.text
    missing = client.post(
        "/v1/advise/feedback",
        json={"advice_id": 99999, "decision": "accepted", "operator": "ali"},
    )
    assert missing.status_code == 404

    stats = client.get("/v1/advise/stats").json()
    assert stats["n_answered"] == 1 and stats["reasons"] == {"quality_risk": 1}
    log = client.get("/v1/advise/log?limit=5").json()
    assert log[0]["feedback"][0]["reason"] == "quality_risk"
    assert "quality_risk" in client.get("/v1/advise/reasons").json()


def test_advise_reports_missing_models(monkeypatch, tmp_path):
    cfg = yaml.safe_load(open("config/advisor.yaml", encoding="utf-8"))
    cfg["constraint_models"] = {"bz_temp": str(tmp_path / "missing.joblib")}
    path = tmp_path / "advisor.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    monkeypatch.setenv("ADVISOR_CONFIG", str(path))
    services.reset_cache()
    idx = pd.date_range("2026-01-01", periods=3, freq="5min")
    response = client.post(
        "/v1/advise/kiln",
        json={"timestamps": [t.isoformat() for t in idx], "signals": {"X": [1.0, 2.0, 3.0]}},
    )
    services.reset_cache()
    assert response.status_code == 503
    assert "missing.joblib" in response.json()["detail"]


def test_mill_schedule_endpoint():
    n = 48
    hours = np.arange(n) % 24
    payload = {
        "mills": [
            {"name": "M1", "rated_kw": 4500, "rated_tph": 105},
            {"name": "M2", "rated_kw": 4600, "rated_tph": 96},
        ],
        "slot_hours": 1,
        "start": "2026-11-24T00:00:00",
        "quota_kw": np.where((hours >= 19) & (hours < 23), 0, 9500).tolist(),
        "grid_price": np.where((hours >= 17) & (hours < 23), 1500, 700).tolist(),
        "free_market_price": [4000] * n,
        "free_market_cap_kw": [4600] * n,
        "dispatch_tph": np.where((hours >= 7) & (hours < 19), 170, 40).tolist(),
        "silo_initial_t": 6000,
        "silo_min_t": 2000,
        "silo_max_t": 12000,
        "baseline_start_below": [0.55, 0.45],
        "baseline_stop_above": [0.7, 0.65],
    }
    response = client.post("/v1/energy/mill-schedule", json=payload)
    assert response.status_code == 200, response.text
    body = response.json()
    assert len(body["optimized"]["slots"]) == n
    assert body["comparison"]["cost_saving"] >= 0
    bad = client.post("/v1/energy/mill-schedule", json={**payload, "grid_price": [1.0]})
    assert bad.status_code == 422


def test_mv_report_endpoint():
    rng = np.random.default_rng(0)
    clinker = rng.uniform(2800, 3100, 40)
    heat = 0.75 * clinker + rng.normal(0, 5, 40)
    heat[30:] -= 0.01 * clinker[30:]  # 10 kcal/kg saved
    rows = [{"heat_gcal": float(h), "clinker_t": float(c)} for h, c in zip(heat, clinker)]
    response = client.post(
        "/v1/energy/mv-report",
        json={
            "baseline": rows[:30],
            "reporting": rows[30:],
            "energy_column": "heat_gcal",
            "drivers": ["clinker_t"],
            "fit_intercept": False,
            "unit_price": 1.0,
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["baseline_valid"] and body["savings_significant"]
    assert body["savings"] > 0


def test_fuel_transfer_plan_learns_from_history(sim):
    response = client.post(
        "/v1/fuel/transfer-plan",
        json={
            "window": _window(sim.wide.iloc[200:240]),
            "history": _window(sim.wide.iloc[:2200]),
            "direction": "gas_to_mazut",
            "o2_per_fan_pct": 0.12,
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["ramp_source"] == "history"
    assert body["plan"]["steps"][-1]["gas_share"] == 0


def test_free_lime_endpoint(configured, sim):
    response = client.post("/v1/quality/free-lime", json=_window(sim.wide.iloc[3500:3560]))
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["lower"] < body["free_lime_pct"] < body["upper"]


def test_condition_endpoint(sim):
    response = client.post(
        "/v1/maintenance/condition",
        json={
            "window": _window(sim.wide.iloc[::3]),
            "reference_end": (sim.wide.index[0] + pd.Timedelta(days=7)).isoformat(),
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["n_alarm"] >= 1
    assert body["findings"][0]["severity"] == "alarm"


def test_shift_report_endpoint(sim):
    window = sim.wide.loc["2026-01-06 06:00":"2026-01-06 16:00"]
    lab = sim.lab[(sim.lab["timestamp"] >= window.index[0]) & (sim.lab["timestamp"] <= window.index[-1])]
    response = client.post(
        "/v1/reports/shift",
        json={
            "window": _window(window),
            "at": "2026-01-06T10:00:00",
            "lab": [
                {"timestamp": r.timestamp.isoformat(), "parameter": r.parameter, "value": r.value}
                for r in lab.itertuples()
            ],
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["facts"]["clinker_t"] > 0
    assert "گزارش شیفت" in body["report_fa"]


def test_readiness_endpoint(configured, sim):
    from src.data_ingest.loader import save_signals_to_db
    from src.db.models import get_session_factory, init_db

    init_db()
    session = get_session_factory()()
    try:
        save_signals_to_db(sim.to_long(include_derived=False).iloc[: 34 * 600], session)
    finally:
        session.close()
    client.post("/v1/advise/kiln", json=_window(sim.wide.iloc[300:360]))
    response = client.get("/v1/control/readiness")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["readiness"]["ready_for_closed_loop_trial"] is False
    assert body["evidence"]["n_records"] == 1
