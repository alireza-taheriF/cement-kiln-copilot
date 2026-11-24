"""Energy endpoints: mill scheduling under the power quota, and M&V savings.

* ``POST /v1/energy/mill-schedule``  optimal mill schedule vs today's rule
* ``POST /v1/energy/mv-report``      baseline-adjusted savings (ASHRAE 14)
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, model_validator

from src.energy.mill_scheduler import Mill, ScheduleProblem, like_for_like, optimize_schedule
from src.mv.baseline import fit_baseline, savings_report

router = APIRouter()


class MillSpec(BaseModel):
    name: str
    rated_kw: float = Field(..., gt=0)
    rated_tph: float = Field(..., gt=0)
    start_cost: float = Field(default=0.0, ge=0)


class ScheduleRequest(BaseModel):
    mills: List[MillSpec] = Field(..., min_length=1)
    slot_hours: float = Field(default=1.0, gt=0)
    start: Optional[str] = None  # ISO time of the first slot
    quota_kw: List[float]
    grid_price: List[float]
    free_market_price: List[float]
    free_market_cap_kw: List[float]
    dispatch_tph: List[float]
    silo_initial_t: float
    silo_min_t: float
    silo_max_t: float
    silo_end_min_t: Optional[float] = None
    initially_running: Optional[List[bool]] = None
    # Today's silo-level rule, to report the saving against (fractions of silo_max_t).
    baseline_start_below: Optional[List[float]] = None
    baseline_stop_above: Optional[List[float]] = None

    @model_validator(mode="after")
    def _lengths(self) -> "ScheduleRequest":
        n = len(self.quota_kw)
        for name in ("grid_price", "free_market_price", "free_market_cap_kw", "dispatch_tph"):
            if len(getattr(self, name)) != n:
                raise ValueError(f"{name} must have {n} slots like quota_kw.")
        if (self.baseline_start_below is None) != (self.baseline_stop_above is None):
            raise ValueError("give both baseline_start_below and baseline_stop_above, or neither.")
        return self


def _clean(obj):
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean(v) for v in obj]
    return obj


def _schedule_json(schedule) -> dict:
    frame = schedule.to_frame()
    rows = frame.reset_index().to_dict(orient="records") if schedule.index is not None else frame.to_dict(orient="records")
    for row in rows:
        for key, value in list(row.items()):
            if isinstance(value, pd.Timestamp):
                row[key] = value.isoformat()
            elif isinstance(value, (np.integer, np.floating)):
                row[key] = value.item()
    return {"summary": _clean(schedule.summary()), "slots": rows}


@router.post("/energy/mill-schedule")
def mill_schedule(request: ScheduleRequest) -> dict:
    mills = [Mill(m.name, m.rated_kw, m.rated_tph, m.start_cost) for m in request.mills]
    index = (
        pd.date_range(request.start, periods=len(request.quota_kw), freq=pd.Timedelta(hours=request.slot_hours))
        if request.start
        else None
    )
    try:
        problem = ScheduleProblem(
            mills=mills,
            slot_hours=request.slot_hours,
            quota_kw=request.quota_kw,
            grid_price=request.grid_price,
            free_market_price=request.free_market_price,
            free_market_cap_kw=request.free_market_cap_kw,
            dispatch_tph=request.dispatch_tph,
            silo_initial_t=request.silo_initial_t,
            silo_min_t=request.silo_min_t,
            silo_max_t=request.silo_max_t,
            silo_end_min_t=request.silo_end_min_t,
            start_index=index,
        )
        if request.baseline_start_below is not None:
            result = like_for_like(
                problem,
                request.baseline_start_below,
                request.baseline_stop_above,
                request.initially_running,
            )
            return {
                "optimized": _schedule_json(result["optimized"]),
                "baseline": _schedule_json(result["baseline"]),
                "comparison": _clean(
                    {k: v for k, v in result["comparison"].items() if k not in ("optimized", "baseline")}
                ),
            }
        return {"optimized": _schedule_json(optimize_schedule(problem, request.initially_running))}
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


class MVRequest(BaseModel):
    """Periods (usually days) of energy use and drivers, before and after."""

    baseline: List[Dict[str, float]] = Field(..., min_length=5)
    reporting: List[Dict[str, float]] = Field(..., min_length=1)
    energy_column: str
    drivers: List[str] = Field(..., min_length=1)
    unit: str = "Gcal"
    granularity: str = "daily"
    fit_intercept: bool = True
    confidence: float = Field(default=0.9, gt=0, lt=1)
    unit_price: Optional[float] = None
    currency: Optional[str] = None


@router.post("/energy/mv-report")
def mv_report(request: MVRequest) -> dict:
    try:
        model = fit_baseline(
            pd.DataFrame(request.baseline),
            request.energy_column,
            request.drivers,
            granularity=request.granularity,
            fit_intercept=request.fit_intercept,
        )
        report = savings_report(
            model,
            pd.DataFrame(request.reporting),
            unit=request.unit,
            confidence=request.confidence,
            unit_price=request.unit_price,
            currency=request.currency,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _clean(report.to_dict())
