"""Keep tests off the developer's SQLite file and MLflow directory."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolate_local_stores(tmp_path_factory, monkeypatch):
    """Point DATABASE_URL and MLflow at a throwaway directory for every test."""
    root = tmp_path_factory.mktemp("stores")
    mlruns = root / "mlruns"
    mlruns.mkdir()
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{root / 'copilot.db'}")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", str(mlruns))
    monkeypatch.delenv("MODEL_URI", raising=False)
    monkeypatch.delenv("MLFLOW_REGISTRY_URI", raising=False)
