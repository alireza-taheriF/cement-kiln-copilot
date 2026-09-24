"""MLflow tracking, Prometheus metrics, drift, and the demo baseline gate."""

from __future__ import annotations

import json
import math
from pathlib import Path

import mlflow
import pandas as pd
from fastapi.testclient import TestClient
from mlflow.tracking import MlflowClient

import src.api.routers.prediction as prediction_module
from src.api.main import app
from src.data_ingest.loader import load_signals_from_csv, save_signals_to_db
from src.db.models import InferenceEvent, get_engine, get_session_factory, init_db
from src.ml.check_baseline import is_regression, main as check_baseline_main
from src.ml.feature_stats import feature_stats_path
from src.ml.tracking import REGISTERED_MODEL_NAME, resolve_tracking_uri
from src.ml.train import load_training_config, train_energy_model

REPO_ROOT = Path(__file__).resolve().parents[1]
client = TestClient(app, raise_server_exceptions=False)


def _session():
    engine = get_engine("sqlite:///:memory:")
    init_db(engine)
    return get_session_factory(engine)()


def _seed(session, periods: int = 40) -> None:
    index = pd.date_range("2026-01-01", periods=periods, freq="min")
    frames = []
    for offset, tag in enumerate(("KILN_ZONE1_TEMP", "KILN_FUEL_FLOW")):
        frames.append(
            pd.DataFrame(
                {
                    "timestamp": index,
                    "tag_name": tag,
                    "value": pd.Series(range(periods), dtype=float) + offset,
                }
            )
        )
    save_signals_to_db(pd.concat(frames, ignore_index=True), session)


def test_default_tracking_uri_is_local_mlruns(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    assert resolve_tracking_uri() == "./mlruns"


def test_tiny_train_creates_mlflow_run(tmp_path, monkeypatch):
    session = _session()
    try:
        _seed(session)
        model_path = tmp_path / "tiny.joblib"
        summary = train_energy_model(
            session,
            target_tag="KILN_ZONE1_TEMP",
            input_tags=["KILN_ZONE1_TEMP", "KILN_FUEL_FLOW"],
            lags=[1],
            rolling_windows=[3],
            horizon=1,
            model_output_path=str(model_path),
        )
    finally:
        session.close()

    assert summary["mlflow_run_id"]
    assert feature_stats_path(str(model_path)).exists()

    tracking_client = MlflowClient()
    experiment = tracking_client.get_experiment_by_name("cement-kiln-kpi")
    assert experiment is not None
    runs = tracking_client.search_runs(experiment_ids=[experiment.experiment_id])
    scored = [run for run in runs if "valid_rmse" in run.data.metrics]
    assert scored, "expected an MLflow run with validation metrics"
    latest = scored[0]
    assert "target_tag" in latest.data.params
    assert "valid_rmse" in latest.data.metrics
    assert "train_rmse" in latest.data.metrics
    artifact_names = [
        item.path for item in tracking_client.list_artifacts(latest.info.run_id)
    ]
    assert any(name.endswith(".joblib") for name in artifact_names)

    production = tracking_client.get_model_version_by_alias(
        REGISTERED_MODEL_NAME, "Production"
    )
    assert int(production.version) >= 1

    # MODEL_URI loads the registered Production model; MODEL_PATH is not required.
    monkeypatch.setenv("MODEL_URI", f"models:/{REGISTERED_MODEL_NAME}/Production")
    monkeypatch.delenv("MODEL_PATH", raising=False)
    prediction_module._CACHED_MODEL = None
    prediction_module._CACHED_PATH = None
    loaded = prediction_module.load_prediction_model()
    assert loaded.is_fitted
    assert loaded.target_name == "KILN_ZONE1_TEMP"
    assert "mlruns" in mlflow.get_tracking_uri()


def test_metrics_endpoint_returns_200():
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
    body = response.text
    assert "cement_kiln_copilot_requests_total" in body
    assert "cement_kiln_copilot_request_latency_ms_bucket" in body
    assert 'le="+Inf"' in body


def _demo_window(n_rows: int = 40) -> dict:
    frame = pd.read_csv(REPO_ROOT / "data" / "demo_signals.csv")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    wide = (
        frame.pivot_table(
            index="timestamp",
            columns="tag_name",
            values="value",
            aggfunc="last",
        )
        .sort_index()
        .tail(n_rows)
    )
    return {
        "timestamps": [ts.isoformat() for ts in wide.index],
        "signals": {column: wide[column].astype(float).tolist() for column in wide.columns},
    }


def test_drift_endpoint_returns_number_for_demo_window(tmp_path, monkeypatch):
    csv_path = REPO_ROOT / "data" / "demo_signals.csv"
    session = _session()
    try:
        save_signals_to_db(load_signals_from_csv(str(csv_path)), session)
        model_path = tmp_path / "demo.joblib"
        config = load_training_config(str(REPO_ROOT / "config" / "model_config.yaml"))
        config["model_output_path"] = str(model_path)
        train_energy_model(
            session,
            target_tag=config["target_tag"],
            input_tags=list(config["input_tags"]),
            lags=list(config["lags"]),
            rolling_windows=list(config["rolling_windows"]),
            horizon=int(config["horizon"]),
            model_output_path=str(model_path),
            config=config,
        )
    finally:
        session.close()

    monkeypatch.setenv("MODEL_PATH", str(model_path))
    monkeypatch.delenv("MODEL_URI", raising=False)
    prediction_module._CACHED_MODEL = None
    prediction_module._CACHED_PATH = None

    predicted = client.post("/v1/predict/energy-kpi", json=_demo_window())
    assert predicted.status_code == 200, predicted.text

    drift = client.get("/v1/monitor/drift?n=20")
    assert drift.status_code == 200, drift.text
    body = drift.json()
    assert body["n"] >= 1
    assert isinstance(body["max_abs_z"], (int, float))
    assert math.isfinite(body["max_abs_z"])
    assert body["features"]
    assert all(isinstance(value, (int, float)) for value in body["features"].values())

    db_session = get_session_factory()()
    try:
        rows = (
            db_session.query(InferenceEvent)
            .filter(InferenceEvent.route == "/v1/predict/energy-kpi")
            .all()
        )
    finally:
        db_session.close()
    assert len(rows) >= 1
    assert len(rows[0].input_hash) == 64
    assert rows[0].latency_ms >= 0
    assert rows[0].model_version


def test_baseline_tolerance_flags_only_real_regressions():
    assert is_regression(1.04, 1.0, 0.05, "relative") is False
    assert is_regression(1.06, 1.0, 0.05, "relative") is True
    assert is_regression(1.2, 1.0, 0.25, "absolute") is False
    assert is_regression(1.3, 1.0, 0.25, "absolute") is True


def test_check_baseline_cli_write_and_compare(tmp_path):
    summary = tmp_path / "training_summary.json"
    baseline = tmp_path / "demo_metrics.json"
    summary.write_text(json.dumps({"valid_metrics": {"rmse": 1.0}}), encoding="utf-8")

    assert check_baseline_main(
        ["--write", "--summary", str(summary), "--baseline", str(baseline)]
    ) == 0
    document = json.loads(baseline.read_text(encoding="utf-8"))
    assert document["metric"] == "valid_rmse"
    assert document["value"] == 1.0
    assert document["tolerance"] == 0.05
    assert document["tolerance_type"] == "relative"

    summary.write_text(json.dumps({"valid_metrics": {"rmse": 1.04}}), encoding="utf-8")
    assert check_baseline_main(
        ["--summary", str(summary), "--baseline", str(baseline)]
    ) == 0

    summary.write_text(json.dumps({"valid_metrics": {"rmse": 1.2}}), encoding="utf-8")
    assert check_baseline_main(
        ["--summary", str(summary), "--baseline", str(baseline)]
    ) == 1


def test_committed_demo_baseline_declares_tolerance():
    document = json.loads(
        (REPO_ROOT / "tests" / "baselines" / "demo_metrics.json").read_text(
            encoding="utf-8"
        )
    )
    assert document["metric"] == "valid_rmse"
    assert isinstance(document["value"], (int, float))
    assert isinstance(document["tolerance"], (int, float))
    assert document["tolerance"] > 0
    assert document["tolerance_type"] in {"relative", "absolute"}
    assert document["direction"] == "lower_is_better"
