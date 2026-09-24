"""Production-grade prediction endpoint.

Exposes ``POST /v1/predict/energy-kpi`` which accepts a batch of long-format
process signals, reconstructs exactly the feature columns expected by a
trained :class:`~src.ml.models.EnergyKPIModel`, runs real inference, and
returns the resulting KPI forecasts.

The trained model artifact is located via the ``MODEL_PATH`` environment
variable and cached in module memory after first load. When ``MODEL_URI`` is
set (for example ``models:/cement-kiln-kpi/Production``), that registry URI is
loaded through MLflow instead and ``MODEL_PATH`` is not consulted.
"""

from __future__ import annotations

import logging
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.api.monitoring import PREDICT_ROUTE, record_inference
from src.ml.features import create_lag_features, create_rolling_features, pivot_signals
from src.ml.models import EnergyKPIModel

logger = logging.getLogger(__name__)

router = APIRouter()

# Long-format column names produced by build_long_dataframe.
_TIMESTAMP_COL = "timestamp"
_TAG_COL = "tag_name"
_VALUE_COL = "value"

# Supported trained-feature naming patterns.
_LAG_RE = re.compile(r"^(?P<tag>.+)_lag_(?P<n>\d+)$")
_ROLL_RE = re.compile(r"^(?P<tag>.+)_roll_(?P<agg>mean|std|min|max)_(?P<n>\d+)$")

# Module-level model cache (keyed by resolved artifact path).
_CACHED_MODEL: Optional[EnergyKPIModel] = None
_CACHED_PATH: Optional[str] = None


# --------------------------------------------------------------------------- #
# Request / response schemas
# --------------------------------------------------------------------------- #
class PredictionRequest(BaseModel):
    """A batch of long-format process signals for inference.

    Attributes
    ----------
    timestamps:
        Ordered list of sample timestamps (at least two are required to make
        lag/rolling features meaningful).
    signals:
        Mapping of ``tag_name -> values``. Every value array must have the same
        length as ``timestamps``.
    """

    timestamps: List[datetime] = Field(..., min_length=2)
    signals: Dict[str, List[float]] = Field(...)

    @model_validator(mode="after")
    def _validate_shapes(self) -> "PredictionRequest":
        """Validate the signal dictionary against the timestamp axis."""
        if not self.signals:
            raise ValueError("signals must contain at least one tag.")

        n = len(self.timestamps)
        for tag_name, values in self.signals.items():
            if not tag_name or not tag_name.strip():
                raise ValueError("signal tag names must be non-empty.")
            if len(values) != n:
                raise ValueError(
                    f"signal {tag_name!r} has {len(values)} values but there "
                    f"are {n} timestamps; lengths must match."
                )
        return self


class PredictionResponse(BaseModel):
    """KPI forecasts plus model provenance metadata."""

    model_config = ConfigDict(protected_namespaces=())

    predictions: List[float]
    target_name: str
    model_backend: str
    model_version: Optional[str] = None
    n_rows: int


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #
def build_long_dataframe(request: PredictionRequest) -> pd.DataFrame:
    """Convert a request payload into a long-format DataFrame.

    Parameters
    ----------
    request:
        A validated prediction request.

    Returns
    -------
    pandas.DataFrame
        Columns ``timestamp``, ``tag_name``, ``value`` with one row per
        timestamp/tag pair, preserving the request's timestamp order.
    """
    timestamps = pd.to_datetime(pd.Series(request.timestamps), utc=False)
    records: List[dict] = []
    for tag_name, values in request.signals.items():
        for ts, value in zip(timestamps, values):
            records.append(
                {_TIMESTAMP_COL: ts, _TAG_COL: tag_name, _VALUE_COL: float(value)}
            )
    return pd.DataFrame(records, columns=[_TIMESTAMP_COL, _TAG_COL, _VALUE_COL])


def load_prediction_model() -> EnergyKPIModel:
    """Load (and cache) the model from ``MODEL_URI`` or ``MODEL_PATH``.

    ``MODEL_URI`` is optional. When it is unset or blank, the artifact is
    loaded from ``MODEL_PATH`` exactly as before.

    Returns
    -------
    EnergyKPIModel
        The cached or freshly loaded model wrapper.

    Raises
    ------
    RuntimeError
        If ``MODEL_URI`` is set but cannot be loaded, or if it is unset and
        ``MODEL_PATH`` is missing or the referenced file does not exist.
    """
    global _CACHED_MODEL, _CACHED_PATH

    model_uri = (os.getenv("MODEL_URI") or "").strip()
    if model_uri:
        cache_key = f"mlflow:{model_uri}"
        if _CACHED_MODEL is not None and _CACHED_PATH == cache_key:
            return _CACHED_MODEL
        from src.ml.tracking import load_energy_model_from_uri

        try:
            model = load_energy_model_from_uri(model_uri)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load model from MODEL_URI={model_uri!r}: {exc}"
            ) from exc
        _CACHED_MODEL = model
        _CACHED_PATH = cache_key
        logger.info("Loaded prediction model from MODEL_URI=%s", model_uri)
        return model

    model_path = os.getenv("MODEL_PATH")
    if not model_path:
        raise RuntimeError("MODEL_PATH environment variable is not set.")

    if _CACHED_MODEL is not None and _CACHED_PATH == model_path:
        return _CACHED_MODEL

    if not Path(model_path).exists():
        raise RuntimeError(
            f"Model artifact not found at MODEL_PATH={model_path!r}."
        )

    model = EnergyKPIModel.load(model_path)
    _CACHED_MODEL = model
    _CACHED_PATH = model_path
    logger.info("Loaded prediction model from %s", model_path)
    return model


def _parse_feature_spec(
    feature_cols: List[str],
) -> tuple[set[str], set[int], set[str], set[int]]:
    """Infer base tags, lags and rolling windows from feature column names.

    Parameters
    ----------
    feature_cols:
        The trained model's ordered feature column names.

    Returns
    -------
    tuple
        ``(lag_tags, lag_values, roll_tags, roll_windows)``.

    Raises
    ------
    ValueError
        If a feature column does not match any supported naming pattern.
    """
    lag_tags: set[str] = set()
    lag_values: set[int] = set()
    roll_tags: set[str] = set()
    roll_windows: set[int] = set()

    for col in feature_cols:
        lag_match = _LAG_RE.match(col)
        if lag_match:
            lag_tags.add(lag_match.group("tag"))
            lag_values.add(int(lag_match.group("n")))
            continue

        roll_match = _ROLL_RE.match(col)
        if roll_match:
            roll_tags.add(roll_match.group("tag"))
            roll_windows.add(int(roll_match.group("n")))
            continue

        raise ValueError(
            f"Unsupported feature column {col!r}; only "
            "{tag}_lag_{n} and {tag}_roll_(mean|std|min|max)_{n} are supported."
        )

    return lag_tags, lag_values, roll_tags, roll_windows


def make_prediction_features(
    request_df: pd.DataFrame,
    model: EnergyKPIModel,
) -> pd.DataFrame:
    """Reconstruct the exact feature matrix expected by ``model``.

    The trained feature column names are parsed to infer which base tags, lag
    offsets and rolling windows are required; those features are rebuilt from
    the request data and returned in the model's exact column order.

    Parameters
    ----------
    request_df:
        Long-format request data (``timestamp``, ``tag_name``, ``value``).
    model:
        A fitted model whose ``feature_cols`` drive feature reconstruction.

    Returns
    -------
    pandas.DataFrame
        Feature matrix with columns exactly equal to ``model.feature_cols`` and
        rows containing lag/rolling-window NaNs dropped.

    Raises
    ------
    ValueError
        If the model is not fitted, a feature name is unsupported, a required
        base tag is absent from the request, or no rows survive feature
        construction.
    """
    if not model.is_fitted or not model.feature_cols:
        raise ValueError("Model is not fitted; cannot build features.")

    feature_cols = list(model.feature_cols)
    lag_tags, lag_values, roll_tags, roll_windows = _parse_feature_spec(feature_cols)

    wide = pivot_signals(request_df)

    required_tags = sorted(lag_tags | roll_tags)
    missing = [tag for tag in required_tags if tag not in wide.columns]
    if missing:
        raise ValueError(
            f"Request is missing required base tag(s): {missing}. "
            f"Provided tags: {list(wide.columns)}."
        )

    feature_frames: List[pd.DataFrame] = []
    if lag_tags:
        feature_frames.append(
            create_lag_features(wide, sorted(lag_tags), sorted(lag_values))
        )
    if roll_tags:
        feature_frames.append(
            create_rolling_features(wide, sorted(roll_tags), sorted(roll_windows))
        )

    all_features = pd.concat(feature_frames, axis=1)

    built_missing = [c for c in feature_cols if c not in all_features.columns]
    if built_missing:
        raise ValueError(
            f"Could not reconstruct required feature column(s): {built_missing}."
        )

    X = all_features[feature_cols].dropna(axis=0, how="any")
    if X.empty:
        raise ValueError(
            "No rows remain after building lag/rolling features; provide a "
            "longer input window than the largest lag/rolling window."
        )
    return X


# --------------------------------------------------------------------------- #
# Endpoint
# --------------------------------------------------------------------------- #
@router.post("/predict/energy-kpi", response_model=PredictionResponse)
async def predict_energy_kpi(request: PredictionRequest) -> PredictionResponse:
    """Run real inference for an energy/KPI forecasting model.

    A successful call is logged for monitoring. The JSON body is unchanged.

    Returns
    -------
    PredictionResponse
        Forecasts plus model provenance.

    Raises
    ------
    fastapi.HTTPException
        ``400`` for feature-building/validation errors, ``500`` for model
        artifact/environment misconfiguration. (``422`` is raised
        automatically by FastAPI for malformed request bodies.)
    """
    started = time.perf_counter()
    try:
        model = load_prediction_model()
    except RuntimeError as exc:
        logger.error("Model load failure: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    request_df = build_long_dataframe(request)

    try:
        X = make_prediction_features(request_df, model)
    except ValueError as exc:
        logger.warning("Feature-building failure: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    predictions = model.predict(X)

    backend = model.metadata.get("model_backend") or model.model_name
    version = model.metadata.get("trained_at")
    response = PredictionResponse(
        predictions=[float(p) for p in predictions],
        target_name=str(model.target_name),
        model_backend=str(backend),
        model_version=version,
        n_rows=len(predictions),
    )
    record_inference(
        route=PREDICT_ROUTE,
        started=started,
        model_version=version,
        signals=request.signals,
        feature_frame=X,
    )
    return response
