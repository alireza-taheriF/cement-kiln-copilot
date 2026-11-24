"""Operator feedback on recommendations, and the acceptance statistics.

Every recommendation shown to an operator gets an id; the operator answers
*accepted*, *rejected* or *modified*, with a reason code. That record is the
most valuable data the product collects:

* the acceptance rate is the first KPI a plant manager asks for,
* rejection reasons tell the team what the models miss,
* accepted-and-applied recommendations are the closest thing to designed
  experiments routine data offers, and feed the model-gain estimates.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Optional

DECISIONS = ("accepted", "rejected", "modified")

#: Rejection reason codes with operator-facing labels.
REASONS: Dict[str, Dict[str, str]] = {
    "process_unstable": {"en": "Process unstable right now", "fa": "فرایند الان پایدار نیست"},
    "quality_risk": {"en": "Clinker quality risk", "fa": "ریسک کیفیت کلینکر"},
    "equipment_limit": {"en": "Equipment limitation", "fa": "محدودیت تجهیزات"},
    "fuel_supply": {"en": "Fuel supply issue", "fa": "مشکل تأمین سوخت"},
    "lab_disagrees": {"en": "Lab results disagree", "fa": "نتایج آزمایشگاه همخوانی ندارد"},
    "too_large_step": {"en": "Step too large", "fa": "گام بیش از حد بزرگ است"},
    "instruction_from_supervisor": {"en": "Supervisor instruction", "fa": "دستور سرشیفت/مدیر"},
    "other": {"en": "Other", "fa": "سایر"},
}


@dataclass(frozen=True)
class FeedbackRecord:
    """One operator answer to one recommendation."""

    advice_id: int
    decision: str
    operator: str
    timestamp: datetime
    reason: Optional[str] = None
    shift: Optional[str] = None
    applied_values: Optional[Dict[str, float]] = None  # what was actually set, if modified
    note: Optional[str] = None

    def __post_init__(self) -> None:
        if self.decision not in DECISIONS:
            raise ValueError(f"decision must be one of {DECISIONS}.")
        if self.decision != "accepted" and self.reason is None:
            raise ValueError("a rejected or modified recommendation needs a reason code.")
        if self.reason is not None and self.reason not in REASONS:
            raise ValueError(f"unknown reason {self.reason!r}; use one of {sorted(REASONS)}.")
        if not self.operator.strip():
            raise ValueError("operator must be given.")


@dataclass
class AcceptanceStats:
    n_recommendations: int
    n_answered: int
    accepted: int
    rejected: int
    modified: int
    acceptance_rate: Optional[float]  # (accepted + modified) / answered
    response_rate: Optional[float]  # answered / recommendations
    reasons: Dict[str, int]
    by_shift: Dict[str, Dict[str, int]]

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def acceptance_stats(n_recommendations: int, feedback: Iterable[FeedbackRecord]) -> AcceptanceStats:
    """Acceptance, response rate and rejection reasons.

    Only the latest answer per recommendation counts (an operator may
    correct a mis-click).
    """
    latest: Dict[int, FeedbackRecord] = {}
    for record in sorted(feedback, key=lambda r: r.timestamp):
        latest[record.advice_id] = record
    records: List[FeedbackRecord] = list(latest.values())
    counts = Counter(r.decision for r in records)
    reasons = Counter(r.reason for r in records if r.reason)
    by_shift: Dict[str, Dict[str, int]] = {}
    for r in records:
        shift = r.shift or "unknown"
        by_shift.setdefault(shift, {d: 0 for d in DECISIONS})[r.decision] += 1
    answered = len(records)
    positive = counts["accepted"] + counts["modified"]
    return AcceptanceStats(
        n_recommendations=n_recommendations,
        n_answered=answered,
        accepted=counts["accepted"],
        rejected=counts["rejected"],
        modified=counts["modified"],
        acceptance_rate=positive / answered if answered else None,
        response_rate=answered / n_recommendations if n_recommendations else None,
        reasons=dict(reasons.most_common()),
        by_shift=by_shift,
    )
