"""Kiln advisor endpoints: advice, operator feedback, acceptance statistics.

* ``POST /v1/advise/kiln``      constrained, uncertainty-aware recommendation
* ``POST /v1/advise/feedback``  operator decision on an advice
* ``GET  /v1/advise/stats``     acceptance and rejection reasons
* ``GET  /v1/advise/log``       recent advice, newest first
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from src.advisory.feedback import REASONS, FeedbackRecord, acceptance_stats
from src.api import services
from src.api.schemas import SignalWindow
from src.db.models import AdviceFeedback, AdviceLog, get_engine, get_session_factory, init_db
from src.safety.envelope import UnsafeRecommendationError

logger = logging.getLogger(__name__)
router = APIRouter()


class FeedbackRequest(BaseModel):
    advice_id: int
    decision: str = Field(..., pattern="^(accepted|rejected|modified)$")
    operator: str = Field(..., min_length=1)
    reason: Optional[str] = None
    shift: Optional[str] = None
    applied_values: Optional[Dict[str, float]] = None
    note: Optional[str] = None


_factories: Dict[str, object] = {}


def _session():
    """Session on ``DATABASE_URL``; tables are created once per URL."""
    url = os.getenv("DATABASE_URL", "sqlite:///./cement_copilot_dev.db")
    if url not in _factories:
        engine = get_engine(url)
        init_db(engine)
        _factories[url] = get_session_factory(engine)
    return _factories[url]()


@router.post("/advise/kiln")
def advise_kiln(window: SignalWindow) -> dict:
    """Recommend specific-heat and ID-fan settings for the latest state.

    The response is :meth:`src.advisory.advisor.Advice.to_dict` plus the
    ``advice_id`` operators answer with. Every advice is logged.
    """
    try:
        advisor = services.advisor()
    except services.ServiceUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    try:
        advice = advisor.advise(window.to_wide())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except UnsafeRecommendationError as exc:  # final gate tripped: a bug, never show it
        logger.error("Final safety gate rejected a recommendation: %s", exc)
        raise HTTPException(status_code=500, detail="internal safety gate rejected the result") from exc

    payload = advice.to_dict()
    session = _session()
    try:
        row = AdviceLog(
            ts=datetime.now(timezone.utc),
            data_ts=max(window.timestamps),
            status=advice.status,
            improvement=advice.improvement,
            payload_json=json.dumps(payload),
        )
        session.add(row)
        session.commit()
        payload["advice_id"] = row.id
    finally:
        session.close()
    return payload


@router.post("/advise/feedback")
def advise_feedback(request: FeedbackRequest) -> dict:
    """Store an operator's decision on an advice."""
    try:
        record = FeedbackRecord(
            advice_id=request.advice_id,
            decision=request.decision,
            operator=request.operator,
            timestamp=datetime.now(timezone.utc),
            reason=request.reason,
            shift=request.shift,
            applied_values=request.applied_values,
            note=request.note,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    session = _session()
    try:
        if session.get(AdviceLog, request.advice_id) is None:
            raise HTTPException(status_code=404, detail=f"advice {request.advice_id} not found")
        row = AdviceFeedback(
            advice_id=record.advice_id,
            ts=record.timestamp,
            decision=record.decision,
            reason=record.reason,
            operator=record.operator,
            shift=record.shift,
            applied_json=json.dumps(record.applied_values) if record.applied_values else None,
            note=record.note,
        )
        session.add(row)
        session.commit()
        return {"feedback_id": row.id, "advice_id": row.advice_id, "decision": row.decision}
    finally:
        session.close()


@router.get("/advise/reasons")
def advise_reasons() -> dict:
    """Rejection reason codes with English and Persian labels."""
    return REASONS


@router.get("/advise/stats")
def advise_stats() -> dict:
    """Acceptance rate, response rate and rejection reasons over all advice."""
    session = _session()
    try:
        n = session.query(AdviceLog).filter(AdviceLog.status.in_(["recommend", "restore_limits"])).count()
        rows = session.query(AdviceFeedback).all()
        records = [
            FeedbackRecord(
                advice_id=r.advice_id,
                decision=r.decision,
                operator=r.operator,
                timestamp=r.ts,
                reason=r.reason,
                shift=r.shift,
            )
            for r in rows
        ]
    finally:
        session.close()
    return acceptance_stats(n, records).to_dict()


@router.get("/advise/log")
def advise_log(limit: int = Query(default=20, ge=1, le=500)) -> List[dict]:
    session = _session()
    try:
        rows = session.query(AdviceLog).order_by(AdviceLog.id.desc()).limit(limit).all()
        return [
            {
                "advice_id": r.id,
                "ts": r.ts.isoformat() if r.ts else None,
                "status": r.status,
                "improvement": r.improvement,
                "advice": json.loads(r.payload_json),
                "feedback": [
                    {"decision": f.decision, "reason": f.reason, "operator": f.operator}
                    for f in r.feedback
                ],
            }
            for r in rows
        ]
    finally:
        session.close()


@router.get("/control/readiness")
def control_readiness() -> dict:
    """Shadow-mode evidence and go/no-go criteria for a closed-loop trial.

    Replays the advice log against the process data stored in the database
    (:mod:`src.control.shadow`). Nothing is ever written to the DCS.
    """
    import pandas as pd

    from src.control.shadow import AdviceRecord, evaluate_shadow, readiness
    from src.ml.features import pivot_signals
    from src.ml.train import load_signals_from_db

    try:
        advisor = services.advisor()
    except services.ServiceUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    session = _session()
    try:
        logs = session.query(AdviceLog).order_by(AdviceLog.id.asc()).all()
        if not logs:
            raise HTTPException(status_code=404, detail="no advice has been logged yet")
        records = []
        for row in logs:
            payload = json.loads(row.payload_json)
            latest = max(row.feedback, key=lambda f: f.ts).decision if row.feedback else None
            stamp = row.data_ts or row.ts
            records.append(
                AdviceRecord(
                    timestamp=pd.Timestamp(stamp).tz_localize(None) if pd.Timestamp(stamp).tzinfo else pd.Timestamp(stamp),
                    status=row.status,
                    current=payload["current"],
                    proposed=payload.get("proposed"),
                    baseline_forecast={
                        o["tag"]: (o["point"], o["lower"], o["upper"])
                        for o in payload.get("baseline_outcomes", [])
                        if o.get("lower") is not None and o.get("upper") is not None
                    },
                    decision=latest,
                )
            )
        signals = load_signals_from_db(session)
    finally:
        session.close()
    if signals.empty:
        raise HTTPException(status_code=404, detail="no process data in the database")
    signals["timestamp"] = pd.to_datetime(signals["timestamp"], utc=True).dt.tz_localize(None)
    wide = pivot_signals(signals)
    from src.domain.kpis import add_derived_kpis

    wide = add_derived_kpis(wide, advisor.fuels)
    step_min = float(pd.Series(wide.index).diff().median() / pd.Timedelta(minutes=1))
    horizons = {
        str(m.target_name): float(m.metadata.get("horizon_rows", 1)) * step_min
        for m in advisor.models.values()
    }
    report = evaluate_shadow(
        records,
        wide,
        advisor.envelope,
        advisor.config.objective_tag,
        horizons,
    )
    out = report.to_dict()
    out.pop("details", None)
    return {"evidence": out, "readiness": readiness(report)}
