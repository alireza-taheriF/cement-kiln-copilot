"""Shadow mode and the evidence needed before any closed-loop control.

Closing the loop on a kiln (the copilot writing setpoints to the DCS) is an
APC project, not a software switch. It needs a HAZOP, a management-of-change
approval, DCS-side interlocks, operator training and, first, months of
evidence that the recommendations are right. This module produces that
evidence; it does **not** write to the DCS.

* :class:`DryRunWriter` is the only :class:`SetpointWriter` in this code
  base. It records what *would* have been written. There is intentionally
  no implementation that talks to a DCS; that belongs to the APC/DCS vendor
  integration and its own safety case (see ``docs/SAFETY.md``).
* :func:`evaluate_shadow` replays the recommendation log against what
  actually happened: did operators move the same way, did the plant stay
  inside its limits afterwards, were the prediction intervals honest?
* :func:`readiness` compares that evidence with explicit go/no-go
  criteria, so "are we ready for closed loop?" has a written answer.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Mapping, Optional, Protocol, Sequence

import numpy as np
import pandas as pd

from src.safety.envelope import SafetyEnvelope


class SetpointWriter(Protocol):
    """Interface a future, separately certified DCS integration would implement."""

    def write(self, setpoints: Mapping[str, float], reason: str) -> None: ...


@dataclass
class DryRunWriter:
    """Records intended writes; never contacts any device."""

    log: List[Dict] = field(default_factory=list)

    def write(self, setpoints: Mapping[str, float], reason: str) -> None:
        self.log.append(
            {"timestamp": datetime.now(timezone.utc).isoformat(), "setpoints": dict(setpoints), "reason": reason}
        )


@dataclass(frozen=True)
class AdviceRecord:
    """One logged advisor output (see :class:`src.advisory.advisor.Advice`)."""

    timestamp: pd.Timestamp
    status: str
    current: Dict[str, float]
    proposed: Optional[Dict[str, float]]
    # Predicted guarded outcomes for the baseline (keep current) case:
    # {tag: (point, lower, upper)} at that tag's horizon.
    baseline_forecast: Dict[str, tuple] = field(default_factory=dict)
    decision: Optional[str] = None  # operator feedback, if any


@dataclass
class ShadowReport:
    n_records: int
    n_recommendations: int
    days_covered: float
    agreement_rate: Optional[float]  # operator moved the same way, unprompted or not
    acceptance_rate: Optional[float]  # from operator feedback
    violations_after_followed: int
    interval_coverage: Dict[str, float]
    realized_objective_change: Optional[float]
    details: List[Dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate_shadow(
    records: Sequence[AdviceRecord],
    actual: pd.DataFrame,
    envelope: SafetyEnvelope,
    objective_tag: str,
    horizons_minutes: Mapping[str, float],
    follow_minutes: float = 60.0,
    outcome_minutes: float = 120.0,
    follow_fraction: float = 0.5,
) -> ShadowReport:
    """Replay the recommendation log against what the plant actually did."""
    if not records:
        raise ValueError("no advice records to evaluate.")
    actual = actual.sort_index()
    details: List[Dict] = []
    agreements: List[bool] = []
    violations = 0
    objective_changes: List[float] = []
    hits: Dict[str, List[bool]] = {}

    for rec in records:
        ts = pd.Timestamp(rec.timestamp)
        # Interval coverage of the baseline forecast (valid whatever the operator did
        # only when the operator did not move the levers; checked below).
        after_follow = actual.loc[ts : ts + pd.Timedelta(minutes=follow_minutes)]
        moved_any = False
        if not after_follow.empty:
            for lever, value in rec.current.items():
                if lever not in after_follow:
                    continue
                limit = envelope.levers.get(lever)
                tolerance = 0.25 * limit.max_step if limit else 1e-6
                moved_any |= abs(float(after_follow[lever].iloc[-1]) - value) > tolerance
        if not moved_any:
            for tag, (point, lower, upper) in rec.baseline_forecast.items():
                horizon = horizons_minutes.get(tag)
                if horizon is None or tag not in actual:
                    continue
                target_time = ts + pd.Timedelta(minutes=horizon)
                near = actual[tag].loc[target_time - pd.Timedelta(minutes=5) : target_time + pd.Timedelta(minutes=5)]
                if not near.empty:
                    hits.setdefault(tag, []).append(bool(lower <= float(near.iloc[0]) <= upper))

        if rec.status not in ("recommend", "restore_limits") or not rec.proposed:
            continue
        followed = True
        for lever, target in rec.proposed.items():
            delta = target - rec.current[lever]
            if abs(delta) < 1e-9 or lever not in after_follow or after_follow.empty:
                continue
            moved = float(after_follow[lever].iloc[-1]) - rec.current[lever]
            followed &= bool(np.sign(moved) == np.sign(delta) and abs(moved) >= follow_fraction * abs(delta))
        agreements.append(followed)

        outcome = actual.loc[ts : ts + pd.Timedelta(minutes=outcome_minutes)]
        broken: List[str] = []
        if followed and not outcome.empty:
            for tag, limit in envelope.process_limits.items():
                if tag not in outcome:
                    continue
                lowest, highest = float(outcome[tag].min()), float(outcome[tag].max())
                if limit.min is not None and lowest < limit.min:
                    broken.append(f"{tag} fell to {lowest:.1f} (min {limit.min})")
                if limit.max is not None and highest > limit.max:
                    broken.append(f"{tag} rose to {highest:.1f} (max {limit.max})")
            violations += int(bool(broken))
            if objective_tag in outcome:
                objective_changes.append(float(outcome[objective_tag].mean()) - rec.current[objective_tag])
        details.append(
            {
                "timestamp": ts.isoformat(),
                "followed": followed,
                "decision": rec.decision,
                "violations_after": broken,
            }
        )

    answered = [r for r in records if r.decision is not None]
    accepted = [r for r in answered if r.decision in ("accepted", "modified")]
    stamps = [pd.Timestamp(r.timestamp) for r in records]
    return ShadowReport(
        n_records=len(records),
        n_recommendations=len(agreements),
        days_covered=float((max(stamps) - min(stamps)) / pd.Timedelta(days=1)),
        agreement_rate=float(np.mean(agreements)) if agreements else None,
        acceptance_rate=len(accepted) / len(answered) if answered else None,
        violations_after_followed=violations,
        interval_coverage={tag: float(np.mean(v)) for tag, v in hits.items() if v},
        realized_objective_change=float(np.mean(objective_changes)) if objective_changes else None,
        details=details,
    )


@dataclass(frozen=True)
class ReadinessCriteria:
    """Go/no-go thresholds for proposing a closed-loop trial (all must pass)."""

    min_days: float = 60.0
    min_recommendations: int = 200
    min_acceptance_rate: float = 0.7
    min_agreement_rate: float = 0.6
    max_violations_after_followed: int = 0
    nominal_coverage: float = 0.9
    coverage_tolerance: float = 0.05


def readiness(report: ShadowReport, criteria: ReadinessCriteria = ReadinessCriteria()) -> Dict:
    """Pass/fail per criterion plus an overall verdict."""
    checks = {
        "days_covered": (report.days_covered >= criteria.min_days, report.days_covered, criteria.min_days),
        "n_recommendations": (
            report.n_recommendations >= criteria.min_recommendations,
            report.n_recommendations,
            criteria.min_recommendations,
        ),
        "acceptance_rate": (
            report.acceptance_rate is not None and report.acceptance_rate >= criteria.min_acceptance_rate,
            report.acceptance_rate,
            criteria.min_acceptance_rate,
        ),
        "agreement_rate": (
            report.agreement_rate is not None and report.agreement_rate >= criteria.min_agreement_rate,
            report.agreement_rate,
            criteria.min_agreement_rate,
        ),
        "violations_after_followed": (
            report.violations_after_followed <= criteria.max_violations_after_followed,
            report.violations_after_followed,
            criteria.max_violations_after_followed,
        ),
    }
    for tag, coverage in report.interval_coverage.items():
        ok = abs(coverage - criteria.nominal_coverage) <= criteria.coverage_tolerance
        checks[f"coverage_{tag}"] = (ok, coverage, criteria.nominal_coverage)
    if not report.interval_coverage:
        checks["coverage"] = (False, None, criteria.nominal_coverage)
    result = {
        name: {"passed": bool(ok), "value": value, "threshold": threshold}
        for name, (ok, value, threshold) in checks.items()
    }
    return {
        "ready_for_closed_loop_trial": all(c["passed"] for c in result.values()),
        "criteria": result,
        "note": (
            "Passing these criteria justifies proposing a supervised closed-loop trial; "
            "it does not replace HAZOP, management of change, or DCS-side interlocks."
        ),
    }
