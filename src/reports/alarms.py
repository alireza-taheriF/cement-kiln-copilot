"""Alarm events from process signals, and the bad-actor summary.

Many DCS alarm logs are not exported, but the limits are known, so alarm
episodes can be rebuilt from the signals themselves with a deadband
(hysteresis), the same way the DCS raises and clears them. The summary
follows alarm-management practice (EEMUA 191 / ISA-18.2): counts, time in
alarm, *chattering* alarms that toggle repeatedly, and the few bad actors
that produce most of the load.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Mapping, Optional

import pandas as pd


@dataclass(frozen=True)
class AlarmRule:
    tag: str
    kind: str  # "HI" | "LO"
    limit: float
    deadband: float = 0.0

    def __post_init__(self) -> None:
        if self.kind not in ("HI", "LO"):
            raise ValueError("kind must be 'HI' or 'LO'.")
        if self.deadband < 0:
            raise ValueError("deadband must be >= 0.")


@dataclass(frozen=True)
class AlarmEvent:
    tag: str
    kind: str
    limit: float
    start: pd.Timestamp
    end: pd.Timestamp
    peak: float

    @property
    def minutes(self) -> float:
        return (self.end - self.start) / pd.Timedelta(minutes=1)


def rules_from_limits(process_limits: Mapping, deadband_fraction: float = 0.01) -> List[AlarmRule]:
    """HI/LO rules from :class:`~src.safety.envelope.SafetyEnvelope` process limits."""
    rules: List[AlarmRule] = []
    for tag, limit in process_limits.items():
        lo = getattr(limit, "min", None)
        hi = getattr(limit, "max", None)
        if hi is not None:
            rules.append(AlarmRule(tag, "HI", float(hi), abs(float(hi)) * deadband_fraction))
        if lo is not None:
            rules.append(AlarmRule(tag, "LO", float(lo), abs(float(lo)) * deadband_fraction))
    return rules


def derive_alarms(wide: pd.DataFrame, rules: List[AlarmRule]) -> List[AlarmEvent]:
    """Alarm episodes: raised beyond the limit, cleared back past the deadband."""
    events: List[AlarmEvent] = []
    for rule in rules:
        if rule.tag not in wide:
            continue
        series = wide[rule.tag].dropna()
        active = False
        start: Optional[pd.Timestamp] = None
        peak = 0.0
        last_ts = None
        for ts, value in series.items():
            if not active:
                raised = value > rule.limit if rule.kind == "HI" else value < rule.limit
                if raised:
                    active, start, peak = True, ts, value
            else:
                peak = max(peak, value) if rule.kind == "HI" else min(peak, value)
                cleared = (
                    value < rule.limit - rule.deadband
                    if rule.kind == "HI"
                    else value > rule.limit + rule.deadband
                )
                if cleared:
                    events.append(AlarmEvent(rule.tag, rule.kind, rule.limit, start, ts, float(peak)))
                    active = False
            last_ts = ts
        if active and start is not None:
            events.append(AlarmEvent(rule.tag, rule.kind, rule.limit, start, last_ts, float(peak)))
    return sorted(events, key=lambda e: e.start)


@dataclass
class AlarmSummary:
    n_events: int
    per_tag: Dict[str, Dict[str, float]]
    chattering: List[str]
    bad_actors: List[str]  # tags producing >= 80 % of events, worst first
    events_per_hour: float

    def to_dict(self) -> dict:
        return asdict(self)


def summarize(
    events: List[AlarmEvent],
    hours: float,
    chatter_window_minutes: float = 10.0,
    chatter_count: int = 3,
) -> AlarmSummary:
    """Counts, time in alarm, chattering tags and bad actors."""
    per_tag: Dict[str, Dict[str, float]] = {}
    by_key: Dict[str, List[AlarmEvent]] = {}
    for e in events:
        key = f"{e.tag} {e.kind}"
        by_key.setdefault(key, []).append(e)
    for key, items in by_key.items():
        per_tag[key] = {
            "count": float(len(items)),
            "minutes_in_alarm": float(sum(e.minutes for e in items)),
            "peak": float(max(e.peak for e in items) if items[0].kind == "HI" else min(e.peak for e in items)),
        }
    chattering = []
    window = pd.Timedelta(minutes=chatter_window_minutes)
    for key, items in by_key.items():
        starts = [e.start for e in items]
        for i in range(len(starts)):
            j = i
            while j < len(starts) and starts[j] - starts[i] <= window:
                j += 1
            if j - i >= chatter_count:
                chattering.append(key)
                break
    ranked = sorted(per_tag, key=lambda k: per_tag[k]["count"], reverse=True)
    total = sum(per_tag[k]["count"] for k in ranked)
    bad, running = [], 0.0
    for key in ranked:
        if total and running / total >= 0.8:
            break
        bad.append(key)
        running += per_tag[key]["count"]
    return AlarmSummary(
        n_events=len(events),
        per_tag=per_tag,
        chattering=sorted(chattering),
        bad_actors=bad,
        events_per_hour=len(events) / hours if hours > 0 else 0.0,
    )
