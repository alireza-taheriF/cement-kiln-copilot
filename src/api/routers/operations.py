"""Operations endpoints: fuel transfers, free lime, equipment condition, shift report.

* ``POST /v1/fuel/transfer-plan``      gas ↔ mazut transfer plan (+ learnt ramp)
* ``POST /v1/quality/free-lime``       soft-sensor free lime with interval
* ``POST /v1/maintenance/condition``   shell hot spots, mill bearings
* ``POST /v1/reports/shift``           shift handover report (fa/en)
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import List, Optional

import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.api import services
from src.api.schemas import LabRow, SignalWindow, lab_frame
from src.fuel.switch import (
    GAS_TO_MAZUT,
    MAZUT_TO_GAS,
    detect_transfers,
    estimate_o2_dip_per_share_rate,
    learn_ramp_policy,
    measure_impact,
    plan_transfer,
)
from src.maintenance.condition import assess_mills, assess_shell, worst_first
from src.reports.alarms import rules_from_limits
from src.reports.shift_report import render_en, render_fa, shift_facts, shift_window

router = APIRouter()


def _finite(obj):
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _finite(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_finite(v) for v in obj]
    return obj


def _fuels():
    plant = services.plant_config()
    return plant.fuels if plant else None


# --------------------------------------------------------------------------- #
class TransferRequest(BaseModel):
    window: SignalWindow  # recent data: sets heat input and fan speed
    direction: str = Field(..., pattern=f"^({GAS_TO_MAZUT}|{MAZUT_TO_GAS})$")
    history: Optional[SignalWindow] = None  # past transfers to learn the ramp from
    ramp_minutes: Optional[float] = Field(default=None, gt=0)
    default_ramp_minutes: float = Field(default=90.0, gt=0)
    n_steps: int = Field(default=6, ge=1, le=60)
    o2_per_fan_pct: Optional[float] = Field(default=None, gt=0)


@router.post("/fuel/transfer-plan")
def fuel_transfer_plan(request: TransferRequest) -> dict:
    fuels = _fuels()
    policy = None
    dip = 0.0
    try:
        if request.history is not None:
            history = request.history.to_wide()
            impacts = []
            for transfer in detect_transfers(history, fuels=fuels):
                try:
                    impacts.append(measure_impact(history, transfer, fuels=fuels))
                except ValueError:
                    continue
            policy = learn_ramp_policy(impacts, default_ramp_minutes=request.default_ramp_minutes)
            dip = estimate_o2_dip_per_share_rate(history, fuels=fuels)
        ramp = request.ramp_minutes or (
            policy.ramp_minutes[request.direction] if policy else request.default_ramp_minutes
        )
        plan = plan_transfer(
            request.window.to_wide(),
            request.direction,
            ramp,
            n_steps=request.n_steps,
            o2_dip_per_rate=dip,
            o2_per_fan_pct=request.o2_per_fan_pct,
            fuels=fuels,
        )
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _finite(
        {
            "plan": plan.to_dict(),
            "ramp_source": (policy.source[request.direction] if policy else "request" if request.ramp_minutes else "default"),
            "ramp_evidence": policy.evidence[request.direction] if policy else [],
            "o2_dip_per_share_rate": dip,
        }
    )


# --------------------------------------------------------------------------- #
@router.post("/quality/free-lime")
def free_lime(window: SignalWindow) -> dict:
    try:
        sensor = services.free_lime_sensor()
    except services.ServiceUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    from src.domain.kpis import FuelProperties, add_derived_kpis

    wide = add_derived_kpis(window.to_wide(), _fuels() or FuelProperties())
    try:
        estimate = sensor.estimate(wide)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    last = estimate.iloc[-1]
    return _finite(
        {
            "timestamp": estimate.index[-1].isoformat(),
            "free_lime_pct": float(last["free_lime"]),
            "lower": float(last["lower"]),
            "upper": float(last["upper"]),
            "lab_due_at": pd.Timestamp(last["lab_due_at"]).isoformat(),
            "series": [
                {"timestamp": ts.isoformat(), "free_lime_pct": float(v)}
                for ts, v in estimate["free_lime"].items()
            ],
            "sensor": sensor.summary(),
        }
    )


# --------------------------------------------------------------------------- #
class ConditionRequest(BaseModel):
    window: SignalWindow  # several days of history, so trends and references exist
    reference_end: Optional[datetime] = None
    shell_positions: Optional[List[int]] = None
    mill_ids: Optional[List[int]] = None


@router.post("/maintenance/condition")
def maintenance_condition(request: ConditionRequest) -> dict:
    wide = request.window.to_wide()
    plant = services.plant_config()
    positions = request.shell_positions or list(range(plant.shell_positions if plant else 10))
    mills = request.mill_ids or list(plant.mill_ids if plant else (1, 2))
    ref_end = pd.Timestamp(request.reference_end) if request.reference_end else None
    try:
        findings = worst_first(
            assess_shell(wide, positions, reference_end=ref_end)
            + assess_mills(wide, mills, reference_end=ref_end)
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _finite(
        {
            "n_alarm": sum(f.severity == "alarm" for f in findings),
            "n_warning": sum(f.severity == "warning" for f in findings),
            "findings": [f.to_dict() for f in findings],
        }
    )


# --------------------------------------------------------------------------- #
class ShiftReportRequest(BaseModel):
    window: SignalWindow
    lab: Optional[List[LabRow]] = None
    at: Optional[datetime] = None  # any time inside the shift; default: last sample


@router.post("/reports/shift")
def shift_report(request: ShiftReportRequest) -> dict:
    wide = request.window.to_wide()
    at = pd.Timestamp(request.at) if request.at else wide.index[-1]
    start, end, name = shift_window(at)
    try:
        facts = shift_facts(
            wide,
            start,
            end,
            lab=lab_frame(request.lab),
            alarm_rules=rules_from_limits(services.envelope().process_limits),
            fuels=_fuels(),
            shift_name=name,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _finite({"facts": facts, "report_fa": render_fa(facts), "report_en": render_en(facts)})


# --------------------------------------------------------------------------- #
@router.get("/demo/window")
def demo_window(
    rows: int = 60,
    end_row: int = 3450,
    step: int = 1,
) -> dict:
    """A window of the simulated demo plant in the request format of the other endpoints.

    ``end_row`` indexes the 5-minute grid (0-4031); ``step`` thins long
    windows (e.g. 3 for 15-minute data). Raw tags only: derived KPIs are
    computed by the endpoints, exactly as with plant data. Simulated data:
    never use it for engineering decisions.
    """
    if rows < 2 or step < 1:
        raise HTTPException(status_code=422, detail="rows must be >= 2 and step >= 1")
    plant = services.demo_plant()
    end_row = max(rows * step, min(end_row, len(plant.wide)))
    frame = plant.wide.iloc[end_row - rows * step : end_row : step]
    derived = {"HEAT_INPUT_GCALH", "GAS_HEAT_SHARE", "SPECIFIC_HEAT_KCAL_KG"}
    frame = frame[[c for c in frame.columns if c not in derived and not c.endswith("_KWH_T")]]
    lab = plant.lab[(plant.lab["timestamp"] >= frame.index[0]) & (plant.lab["timestamp"] <= frame.index[-1])]
    return {
        "timestamps": [ts.isoformat() for ts in frame.index],
        "signals": {c: [round(float(v), 4) for v in frame[c]] for c in frame.columns},
        "lab": [
            {"timestamp": r.timestamp.isoformat(), "parameter": r.parameter, "value": float(r.value)}
            for r in lab.itertuples()
        ],
        "simulated": True,
    }
