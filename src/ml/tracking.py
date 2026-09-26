"""Local MLflow tracking for cement-kiln KPI training runs.

The default tracking URI is a file store at ``./mlruns``. Set
``MLFLOW_TRACKING_URI`` to point somewhere else. The file store cannot host
MLflow's model registry, so when tracking uses a directory the registry is a
SQLite database inside that same directory (``registry.db``). No cloud account
is required. Override the registry with ``MLFLOW_REGISTRY_URI``.
"""

from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import joblib
import mlflow
import pandas as pd

from src.ml.models import EnergyKPIModel

logger = logging.getLogger(__name__)

REGISTERED_MODEL_NAME = "cement-kiln-kpi"
DEFAULT_TRACKING_URI = "./mlruns"
EXPERIMENT_NAME = "cement-kiln-kpi"
PRODUCTION_ALIAS = "Production"

_DB_PREFIXES = ("sqlite:", "postgresql:", "postgresql+", "mysql:", "mysql+")
_REMOTE_PREFIXES = ("http:", "https:", "databricks")


def resolve_tracking_uri() -> str:
    """Return ``MLFLOW_TRACKING_URI`` or the local ``./mlruns`` file store."""
    return os.getenv("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)


def _is_file_store(uri: str) -> bool:
    lowered = uri.lower()
    if lowered.startswith(_DB_PREFIXES) or lowered.startswith(_REMOTE_PREFIXES):
        return False
    return True


def file_store_directory(uri: str) -> Path:
    """Directory that holds a file-store tracking URI."""
    if uri.startswith("file:"):
        parsed = urlparse(uri)
        raw = unquote(parsed.path or uri[len("file:") :])
        return Path(raw)
    return Path(uri)


def configure_mlflow() -> str:
    """Apply the tracking URI and, for a file store, a local SQLite registry."""
    tracking_uri = resolve_tracking_uri()
    os.environ.setdefault("MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING", "false")
    if _is_file_store(tracking_uri):
        file_store_directory(tracking_uri).mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(tracking_uri)

    registry_override = os.getenv("MLFLOW_REGISTRY_URI")
    if registry_override:
        mlflow.set_registry_uri(registry_override)
    elif _is_file_store(tracking_uri):
        registry_db = (file_store_directory(tracking_uri) / "registry.db").resolve()
        registry_db.parent.mkdir(parents=True, exist_ok=True)
        mlflow.set_registry_uri(f"sqlite:///{registry_db}")
    return tracking_uri


def _param_value(value: Any) -> str:
    if isinstance(value, (list, tuple, dict)):
        text = json.dumps(value)
    elif value is None:
        text = ""
    else:
        text = str(value)
    return text[:500]


def _log_baseline_rmses(comparison: dict[str, Any]) -> None:
    """Log the model RMSE and the two naive validation RMSEs."""
    persistence = comparison.get("persistence") or {}
    linear = comparison.get("linear_regression") or {}
    named = {
        "model_valid_rmse": comparison.get("model_valid_rmse"),
        "persistence_rmse": persistence.get("rmse"),
        "linear_regression_rmse": linear.get("rmse"),
    }
    for key, value in named.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        numeric = float(value)
        if math.isfinite(numeric):
            mlflow.log_metric(key, numeric)


def _log_split_metrics(split: str, metrics: dict[str, Any]) -> None:
    for key, value in metrics.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        numeric = float(value)
        if not math.isfinite(numeric):
            continue
        mlflow.log_metric(f"{split}_{key}", numeric)


class _EnergyKPIPyfunc(mlflow.pyfunc.PythonModel):
    """Pyfunc wrapper so the joblib artifact can be registered."""

    def load_context(self, context):  # type: ignore[no-untyped-def]
        self._model = joblib.load(context.artifacts["model"])

    def predict(self, context, model_input, params=None):  # type: ignore[no-untyped-def]
        frame = model_input
        if not isinstance(frame, pd.DataFrame):
            frame = pd.DataFrame(frame)
        return self._model.predict(frame)


def _promote_to_production(version: str) -> None:
    client = mlflow.MlflowClient()
    alias_ok = False
    stage_ok = False
    try:
        client.set_registered_model_alias(
            REGISTERED_MODEL_NAME, PRODUCTION_ALIAS, version
        )
        alias_ok = True
    except Exception:
        logger.warning(
            "Could not set the %s alias on %s version %s",
            PRODUCTION_ALIAS,
            REGISTERED_MODEL_NAME,
            version,
            exc_info=True,
        )
    try:
        client.transition_model_version_stage(
            name=REGISTERED_MODEL_NAME,
            version=version,
            stage=PRODUCTION_ALIAS,
            archive_existing_versions=True,
        )
        stage_ok = True
    except Exception:
        logger.warning(
            "Could not transition %s version %s to %s",
            REGISTERED_MODEL_NAME,
            version,
            PRODUCTION_ALIAS,
            exc_info=True,
        )
    if not (alias_ok or stage_ok):
        raise RuntimeError(
            f"Failed to mark {REGISTERED_MODEL_NAME} version {version} as Production."
        )


def log_training_run(
    config: dict[str, Any],
    summary: dict[str, Any],
    model_path: str,
    stats_path: str | None = None,
) -> str:
    """Log config params, trainer metrics, and the joblib artifact.

    Registers ``cement-kiln-kpi`` and points the ``Production`` alias (and
    stage) at this version. Returns the MLflow run id.
    """
    configure_mlflow()
    try:
        mlflow.autolog(disable=True)
    except Exception:
        logger.debug("mlflow.autolog(disable=True) was not applied", exc_info=True)

    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name="train-energy-kpi") as run:
        mlflow.log_params({key: _param_value(value) for key, value in config.items()})
        mlflow.set_tag("registered_model", REGISTERED_MODEL_NAME)
        _log_split_metrics("train", summary.get("train_metrics") or {})
        _log_split_metrics("valid", summary.get("valid_metrics") or {})
        _log_baseline_rmses(summary.get("baseline_comparison") or {})

        # Raw joblib artifact, as produced by EnergyKPIModel.save.
        mlflow.log_artifact(model_path)
        if stats_path and Path(stats_path).exists():
            mlflow.log_artifact(stats_path)

        log_kwargs = dict(
            python_model=_EnergyKPIPyfunc(),
            artifacts={"model": model_path},
            pip_requirements=[
                "joblib",
                "pandas",
                "numpy",
                "scikit-learn",
                "lightgbm",
            ],
            registered_model_name=REGISTERED_MODEL_NAME,
        )
        try:
            info = mlflow.pyfunc.log_model(artifact_path="model", **log_kwargs)
        except TypeError:
            info = mlflow.pyfunc.log_model(name="model", **log_kwargs)

        version = getattr(info, "registered_model_version", None)
        if not version:
            client = mlflow.MlflowClient()
            versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
            if not versions:
                raise RuntimeError(
                    f"MLflow did not register a version of {REGISTERED_MODEL_NAME}."
                )
            version = str(max(int(item.version) for item in versions))
        _promote_to_production(str(version))
        logger.info(
            "Logged MLflow run %s and registered %s version %s as Production",
            run.info.run_id,
            REGISTERED_MODEL_NAME,
            version,
        )
        return run.info.run_id


def load_energy_model_from_uri(model_uri: str) -> EnergyKPIModel:
    """Download ``model_uri`` (for example ``models:/cement-kiln-kpi/Production``)."""
    configure_mlflow()
    local = Path(mlflow.artifacts.download_artifacts(artifact_uri=model_uri))
    if local.is_file():
        return EnergyKPIModel.load(str(local))
    matches = sorted(local.rglob("*.joblib"))
    if not matches:
        raise FileNotFoundError(f"No joblib artifact under downloaded model {local}.")
    return EnergyKPIModel.load(str(matches[0]))
