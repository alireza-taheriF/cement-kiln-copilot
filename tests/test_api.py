"""Tests for the FastAPI application, including real model-backed inference."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import src.api.routers.prediction as prediction_module
from src.api.main import app
from src.ml.features import make_supervised_dataset
from src.ml.models import EnergyKPIModel

client = TestClient(app, raise_server_exceptions=False)

# Model / request shape shared across inference tests.
_TARGET_TAG = "KILN_ZONE1_TEMP"
_INPUT_TAGS = ["KILN_ZONE1_TEMP", "KILN_FUEL_FLOW"]
_LAGS = [1, 2]
_ROLLING_WINDOWS = [3]
_HORIZON = 1


@pytest.fixture(autouse=True)
def _reset_model_cache(monkeypatch):
    """Ensure each test starts with a clean module-level model cache."""
    prediction_module._CACHED_MODEL = None
    prediction_module._CACHED_PATH = None
    monkeypatch.delenv("MODEL_URI", raising=False)
    yield
    prediction_module._CACHED_MODEL = None
    prediction_module._CACHED_PATH = None


def _train_tiny_model(path: Path) -> EnergyKPIModel:
    """Train and save a small real EnergyKPIModel artifact at ``path``."""
    periods = 60
    index = pd.date_range("2026-01-01 00:00", periods=periods, freq="min")
    frames = []
    for offset, tag in enumerate(_INPUT_TAGS):
        frames.append(
            pd.DataFrame(
                {
                    "timestamp": index,
                    "tag_name": tag,
                    "value": np.linspace(0, 5, periods)
                    + offset
                    + np.sin(np.arange(periods) / 4.0),
                }
            )
        )
    signals = pd.concat(frames, ignore_index=True)

    X, y = make_supervised_dataset(
        signals,
        target_tag=_TARGET_TAG,
        input_tags=_INPUT_TAGS,
        lags=_LAGS,
        rolling_windows=_ROLLING_WINDOWS,
        horizon=_HORIZON,
    )
    model = EnergyKPIModel(target_name=_TARGET_TAG).fit(X, y)
    model.save(str(path))
    return model


def _valid_payload(n_rows: int = 12) -> dict:
    index = pd.date_range("2026-02-01 00:00", periods=n_rows, freq="min")
    timestamps = [ts.isoformat() for ts in index]
    rng = np.random.default_rng(0)
    return {
        "timestamps": timestamps,
        "signals": {
            tag: (rng.normal(size=n_rows) + 1.0).tolist() for tag in _INPUT_TAGS
        },
    }


# --------------------------------------------------------------------------- #
# Smoke endpoints
# --------------------------------------------------------------------------- #
def test_root():
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.json()["service"] == "cement-kiln-copilot"


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def _recommend_payload(n_rows: int = 12, **overrides) -> dict:
    """Valid recommendation payload over the trained model's tags."""
    index = pd.date_range("2026-02-01 00:00", periods=n_rows, freq="min")
    timestamps = [ts.isoformat() for ts in index]
    rng = np.random.default_rng(1)
    payload: dict = {
        "timestamps": timestamps,
        "signals": {
            tag: (rng.normal(size=n_rows) + 1.0).tolist() for tag in _INPUT_TAGS
        },
        "current_setpoints": {tag: 1.0 for tag in _INPUT_TAGS},
        "desired_target": 0.0,
        "adjustable_tags": ["KILN_FUEL_FLOW"],
        "step_fractions": [-0.10, -0.05, 0.0, 0.05, 0.10],
        "top_k": 3,
    }
    payload.update(overrides)
    return payload


# --------------------------------------------------------------------------- #
# Energy-KPI prediction
# --------------------------------------------------------------------------- #
def test_predict_energy_kpi_success(tmp_path, monkeypatch):
    model_path = tmp_path / "model.joblib"
    _train_tiny_model(model_path)
    monkeypatch.setenv("MODEL_PATH", str(model_path))

    resp = client.post("/v1/predict/energy-kpi", json=_valid_payload())
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["predictions"]) > 0
    assert body["n_rows"] == len(body["predictions"])
    assert body["target_name"] == _TARGET_TAG
    assert body["model_backend"]
    assert body["model_version"] is not None


def test_predict_missing_model_path_returns_500(monkeypatch):
    monkeypatch.delenv("MODEL_PATH", raising=False)
    resp = client.post("/v1/predict/energy-kpi", json=_valid_payload())
    assert resp.status_code == 500
    assert "MODEL_PATH" in resp.json()["detail"]


def test_predict_nonexistent_model_file_returns_500(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "does_not_exist.joblib"))
    resp = client.post("/v1/predict/energy-kpi", json=_valid_payload())
    assert resp.status_code == 500
    assert "not found" in resp.json()["detail"].lower()


def test_predict_mismatched_signal_lengths_returns_422(tmp_path, monkeypatch):
    model_path = tmp_path / "model.joblib"
    _train_tiny_model(model_path)
    monkeypatch.setenv("MODEL_PATH", str(model_path))

    payload = _valid_payload(n_rows=12)
    payload["signals"][_INPUT_TAGS[0]] = payload["signals"][_INPUT_TAGS[0]][:-1]
    resp = client.post("/v1/predict/energy-kpi", json=payload)
    assert resp.status_code == 422


def test_predict_too_short_window_returns_400(tmp_path, monkeypatch):
    model_path = tmp_path / "model.joblib"
    _train_tiny_model(model_path)
    monkeypatch.setenv("MODEL_PATH", str(model_path))

    # Only 2 rows but the rolling window (3) needs more -> all rows dropped.
    resp = client.post("/v1/predict/energy-kpi", json=_valid_payload(n_rows=2))
    assert resp.status_code == 400
    assert "No rows remain" in resp.json()["detail"]


def test_predict_missing_base_tag_returns_400(tmp_path, monkeypatch):
    model_path = tmp_path / "model.joblib"
    _train_tiny_model(model_path)
    monkeypatch.setenv("MODEL_PATH", str(model_path))

    payload = _valid_payload(n_rows=12)
    # Drop a required base tag from the request payload.
    del payload["signals"]["KILN_FUEL_FLOW"]
    resp = client.post("/v1/predict/energy-kpi", json=payload)
    assert resp.status_code == 400
    assert "missing required base tag" in resp.json()["detail"].lower()


# --------------------------------------------------------------------------- #
# Setpoint recommendation
# --------------------------------------------------------------------------- #
def test_recommend_setpoints_success(tmp_path, monkeypatch):
    model_path = tmp_path / "model.joblib"
    _train_tiny_model(model_path)
    monkeypatch.setenv("MODEL_PATH", str(model_path))

    resp = client.post("/v1/recommend/setpoints", json=_recommend_payload())
    assert resp.status_code == 200
    body = resp.json()
    assert body["target_name"] == _TARGET_TAG
    assert body["desired_target"] == 0.0
    assert "baseline_prediction_mean" in body
    assert body["model_backend"]
    assert body["n_candidates_evaluated"] >= 1
    assert len(body["recommendations"]) > 0


def test_recommend_setpoints_sorted_by_score(tmp_path, monkeypatch):
    model_path = tmp_path / "model.joblib"
    _train_tiny_model(model_path)
    monkeypatch.setenv("MODEL_PATH", str(model_path))

    resp = client.post("/v1/recommend/setpoints", json=_recommend_payload())
    assert resp.status_code == 200
    scores = [item["score"] for item in resp.json()["recommendations"]]
    assert scores == sorted(scores)


def test_recommend_setpoints_top_k_limits_output(tmp_path, monkeypatch):
    model_path = tmp_path / "model.joblib"
    _train_tiny_model(model_path)
    monkeypatch.setenv("MODEL_PATH", str(model_path))

    resp = client.post(
        "/v1/recommend/setpoints", json=_recommend_payload(top_k=2)
    )
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["recommendations"]) <= 2
    # The baseline plus 4 non-zero perturbations are evaluated for one tag.
    assert body["n_candidates_evaluated"] >= len(body["recommendations"])


def test_recommend_setpoints_respects_absolute_bounds(tmp_path, monkeypatch):
    model_path = tmp_path / "model.joblib"
    _train_tiny_model(model_path)
    monkeypatch.setenv("MODEL_PATH", str(model_path))

    bounds_hi = 1.02
    payload = _recommend_payload(
        top_k=10,
        absolute_bounds={"KILN_FUEL_FLOW": [0.98, bounds_hi]},
    )
    resp = client.post("/v1/recommend/setpoints", json=payload)
    assert resp.status_code == 200
    for item in resp.json()["recommendations"]:
        value = item["candidate_setpoints"]["KILN_FUEL_FLOW"]
        assert 0.98 <= value <= bounds_hi


def test_recommend_setpoints_missing_model_path_returns_500(monkeypatch):
    monkeypatch.delenv("MODEL_PATH", raising=False)
    resp = client.post("/v1/recommend/setpoints", json=_recommend_payload())
    assert resp.status_code == 500
    assert "MODEL_PATH" in resp.json()["detail"]


def test_recommend_setpoints_missing_adjustable_in_setpoints_returns_422(
    tmp_path, monkeypatch
):
    model_path = tmp_path / "model.joblib"
    _train_tiny_model(model_path)
    monkeypatch.setenv("MODEL_PATH", str(model_path))

    payload = _recommend_payload()
    payload["adjustable_tags"] = ["NOT_IN_SETPOINTS"]
    resp = client.post("/v1/recommend/setpoints", json=payload)
    assert resp.status_code == 422


def test_recommend_setpoints_too_short_window_returns_400(tmp_path, monkeypatch):
    model_path = tmp_path / "model.joblib"
    _train_tiny_model(model_path)
    monkeypatch.setenv("MODEL_PATH", str(model_path))

    # Only 2 rows but the rolling window (3) needs more -> all rows dropped.
    resp = client.post(
        "/v1/recommend/setpoints", json=_recommend_payload(n_rows=2)
    )
    assert resp.status_code == 400
