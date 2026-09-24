"""Prometheus metrics and feature-drift endpoints.

``GET /metrics`` is plain Prometheus text. ``GET /v1/monitor/drift`` compares
the last N successful prediction inputs to the training feature distribution.
Neither endpoint changes the predict or recommend JSON schemas.
"""

from __future__ import annotations

from typing import Dict, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, ConfigDict

from src.api.monitoring import build_drift_report, render_prometheus
from src.api.routers.prediction import load_prediction_model

metrics_router = APIRouter()
monitor_router = APIRouter()


class DriftResponse(BaseModel):
    """Per-feature z-scores of recent predictions versus training stats."""

    model_config = ConfigDict(protected_namespaces=())

    n: int
    features: Dict[str, float]
    max_abs_z: float
    model_version: Optional[str] = None


@metrics_router.get("/metrics", response_class=PlainTextResponse)
async def metrics() -> PlainTextResponse:
    """Request count and latency histogram in Prometheus text format."""
    return PlainTextResponse(
        render_prometheus(),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )


@monitor_router.get("/monitor/drift", response_model=DriftResponse)
async def monitor_drift(
    n: int = Query(default=50, ge=1, le=5000),
) -> DriftResponse:
    """Z-score the last ``n`` prediction inputs against training feature stats."""
    try:
        model = load_prediction_model()
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    report = build_drift_report(model, n)
    if report is None:
        raise HTTPException(
            status_code=404,
            detail=(
                "Training feature stats were not found next to the model "
                "artifact. Retrain so the sidecar is written."
            ),
        )
    return DriftResponse(**report)
