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


# ---------------------------------------------------------------------------
# Simulated plant and trained constraint models, shared across test modules.
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def sim():
    """14 days of the simulated 3,000 t/d line (deterministic)."""
    from src.sim.plant import simulate

    return simulate()


@pytest.fixture(scope="session")
def constraint_models(sim):
    """Burning-zone and CO models trained on the first ~10 days only.

    The last days stay unseen so advisor tests run out of sample.
    """
    from pathlib import Path

    import yaml

    from src.ml.pipeline import fit_kpi_model, prepare_wide

    root = Path(__file__).resolve().parents[1]
    train = sim.wide.iloc[:3000]
    models = {}
    for role, path in (("bz_temp", "config/model_config.yaml"), ("co", "config/models/co.yaml")):
        cfg = yaml.safe_load((root / path).read_text(encoding="utf-8"))
        wide = prepare_wide(train, cfg["input_tags"] + [cfg["target_tag"]])
        models[role] = fit_kpi_model(
            wide,
            cfg["target_tag"],
            cfg["input_tags"],
            cfg["lags"],
            cfg["rolling_windows"],
            cfg["horizon"],
            cfg,
        ).model
    return models
