"""Production-grade setpoint recommendation endpoint.

Exposes ``POST /v1/recommend/setpoints`` which uses the trained
:class:`~src.ml.models.EnergyKPIModel` to score perturbed setpoint candidates
and return those that move the predicted KPI closest to a desired target.

Model loading and feature building are shared with the prediction endpoint to
guarantee identical inference semantics.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.api.routers.prediction import (
    build_long_dataframe,
    load_prediction_model,
    make_prediction_features,
)
from src.ml.recommend import (
    apply_candidate_to_signals,
    generate_candidates,
    rank_candidates,
)

logger = logging.getLogger(__name__)

router = APIRouter()


# --------------------------------------------------------------------------- #
# Request / response schemas
# --------------------------------------------------------------------------- #
class RecommendationRequest(BaseModel):
    """A setpoint-recommendation request over a recent signal window."""

    timestamps: List[datetime] = Field(..., min_length=2)
    signals: Dict[str, List[float]] = Field(...)
    current_setpoints: Dict[str, float] = Field(...)
    desired_target: float
    adjustable_tags: List[str] = Field(..., min_length=1)
    step_fractions: List[float] = Field(
        default_factory=lambda: [-0.10, -0.05, 0.0, 0.05, 0.10]
    )
    top_k: int = Field(default=3, ge=1)
    absolute_bounds: Optional[Dict[str, Tuple[float, float]]] = None

    @model_validator(mode="after")
    def _validate(self) -> "RecommendationRequest":
        """Validate cross-field invariants of the request."""
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

        if not self.current_setpoints:
            raise ValueError("current_setpoints must be non-empty.")

        for tag in self.adjustable_tags:
            if not tag or not tag.strip():
                raise ValueError("adjustable_tags entries must be non-empty.")
            if tag not in self.current_setpoints:
                raise ValueError(
                    f"adjustable tag {tag!r} is not present in current_setpoints."
                )

        if self.absolute_bounds:
            for tag, bound in self.absolute_bounds.items():
                if len(bound) != 2:
                    raise ValueError(
                        f"absolute_bounds[{tag!r}] must have exactly 2 elements."
                    )
                lo, hi = bound
                if lo > hi:
                    raise ValueError(
                        f"absolute_bounds[{tag!r}] has min > max ({lo} > {hi})."
                    )
        return self


class RecommendationItem(BaseModel):
    """A single scored setpoint candidate."""

    candidate_setpoints: Dict[str, float]
    predicted_mean: float
    predicted_min: float
    predicted_max: float
    score: float


class RecommendationResponse(BaseModel):
    """Ranked setpoint recommendations plus model provenance."""

    model_config = ConfigDict(protected_namespaces=())

    target_name: str
    desired_target: float
    baseline_prediction_mean: float
    recommendations: List[RecommendationItem]
    model_backend: str
    model_version: Optional[str] = None
    n_candidates_evaluated: int


# --------------------------------------------------------------------------- #
# Endpoint
# --------------------------------------------------------------------------- #
@router.post("/recommend/setpoints", response_model=RecommendationResponse)
async def recommend_setpoints_endpoint(
    request: RecommendationRequest,
) -> RecommendationResponse:
    """Recommend setpoint adjustments that move the KPI toward a target.

    Raises
    ------
    fastapi.HTTPException
        ``400`` for feature-building/value errors, ``500`` for model
        artifact/environment misconfiguration. (``422`` is raised
        automatically for malformed request bodies.)
    """
    try:
        model = load_prediction_model()
    except RuntimeError as exc:
        logger.error("Model load failure: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    # build_long_dataframe only reads .timestamps and .signals, both present here.
    request_df = build_long_dataframe(request)  # type: ignore[arg-type]

    baseline_setpoints = {
        tag: float(request.current_setpoints[tag]) for tag in request.adjustable_tags
    }

    try:
        # Evaluate the unchanged baseline first.
        baseline_df = apply_candidate_to_signals(request_df, baseline_setpoints)
        baseline_X = make_prediction_features(baseline_df, model)
        baseline_predictions = model.predict(baseline_X)
        baseline_mean = float(np.mean(baseline_predictions))

        # Total candidate count (deterministic; all share the same feature
        # viability as the baseline, which just succeeded).
        n_candidates = len(
            generate_candidates(
                request.current_setpoints,
                request.adjustable_tags,
                request.step_fractions,
                absolute_bounds=request.absolute_bounds,
            )
        )

        ranked = rank_candidates(
            request_df,
            request.current_setpoints,
            request.desired_target,
            model,
            adjustable_tags=request.adjustable_tags,
            step_fractions=request.step_fractions,
            absolute_bounds=request.absolute_bounds,
            top_k=request.top_k,
        )
    except ValueError as exc:
        logger.warning("Recommendation failure: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    items = [RecommendationItem(**item) for item in ranked]
    backend = model.metadata.get("model_backend") or model.model_name
    version = model.metadata.get("trained_at")
    return RecommendationResponse(
        target_name=str(model.target_name),
        desired_target=request.desired_target,
        baseline_prediction_mean=baseline_mean,
        recommendations=items,
        model_backend=str(backend),
        model_version=version,
        n_candidates_evaluated=n_candidates,
    )
