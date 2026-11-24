"""Safety envelope: hard limits every recommendation must satisfy.

Two kinds of limit, both loaded from ``config/safety_limits.yaml``:

* **lever limits** on what the copilot may *propose*: an absolute range and
  a maximum change per recommendation (operators move a kiln in small
  steps; a model that wants a 40 kcal/kg cut in one go is wrong or
  extrapolating);
* **process limits** on what the *predicted outcome* must stay within, e.g.
  burning-zone temperature and CO. With ``use_interval`` the pessimistic end
  of the prediction interval is tested, not the point forecast.

The envelope is applied twice: while generating candidates (so nothing
outside it is ever scored) and as a final gate on the recommendation that
leaves the system (:meth:`SafetyEnvelope.assert_safe`). The second check is
deliberately redundant: a bug in candidate generation must not reach an
operator screen.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional

import yaml


class UnsafeRecommendationError(RuntimeError):
    """Raised when a recommendation about to be returned violates the envelope."""


@dataclass(frozen=True)
class LeverLimit:
    """Range and rate limit of an operator-adjustable tag."""

    min: float
    max: float
    max_step: float

    def __post_init__(self) -> None:
        if self.min > self.max:
            raise ValueError(f"lever min {self.min} > max {self.max}.")
        if self.max_step <= 0:
            raise ValueError("lever max_step must be positive.")


@dataclass(frozen=True)
class ProcessLimit:
    """Allowed range of a predicted process variable."""

    min: Optional[float] = None
    max: Optional[float] = None
    use_interval: bool = False

    def __post_init__(self) -> None:
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError(f"process limit min {self.min} > max {self.max}.")


@dataclass(frozen=True)
class Violation:
    """One broken limit, with a message an operator can read."""

    tag: str
    kind: str  # "lever_range" | "lever_step" | "process_min" | "process_max" | "unknown_lever"
    value: float
    limit: float
    message: str
    message_fa: str


@dataclass(frozen=True)
class ProcessCheck:
    """Outcome of testing one predicted variable against its limit."""

    tag: str
    point: float
    lower: float
    upper: float
    limit: ProcessLimit
    violations: tuple[Violation, ...]

    @property
    def ok(self) -> bool:
        return not self.violations

    @property
    def margin(self) -> float:
        """Distance of the tested bound to the nearest limit (negative = violated)."""
        tested_low = self.lower if self.limit.use_interval else self.point
        tested_high = self.upper if self.limit.use_interval else self.point
        margins = []
        if self.limit.min is not None:
            margins.append(tested_low - self.limit.min)
        if self.limit.max is not None:
            margins.append(self.limit.max - tested_high)
        return min(margins) if margins else math.inf


class SafetyEnvelope:
    """Lever and process limits with check helpers."""

    def __init__(
        self,
        levers: Mapping[str, LeverLimit],
        process_limits: Mapping[str, ProcessLimit],
    ) -> None:
        self.levers: Dict[str, LeverLimit] = dict(levers)
        self.process_limits: Dict[str, ProcessLimit] = dict(process_limits)

    # ------------------------------------------------------------------ #
    @classmethod
    def from_yaml(cls, path: str | Path = "config/safety_limits.yaml") -> "SafetyEnvelope":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: Mapping) -> "SafetyEnvelope":
        levers = {
            str(tag): LeverLimit(float(v["min"]), float(v["max"]), float(v["max_step"]))
            for tag, v in (raw.get("levers") or {}).items()
        }
        process = {
            str(tag): ProcessLimit(
                min=None if v.get("min") is None else float(v["min"]),
                max=None if v.get("max") is None else float(v["max"]),
                use_interval=bool(v.get("use_interval", False)),
            )
            for tag, v in (raw.get("process_limits") or {}).items()
        }
        return cls(levers, process)

    # ------------------------------------------------------------------ #
    def lever_candidates(self, tag: str, current: float, deltas: List[float]) -> List[float]:
        """Values ``current + delta`` that satisfy the lever's range and step.

        The current value is always first, even when it is outside the range:
        "keep as is" is never something the copilot proposes as a change.
        """
        limit = self._lever(tag)
        values = [float(current)]
        for delta in deltas:
            value = float(current) + float(delta)
            if abs(delta) <= limit.max_step + 1e-9 and limit.min <= value <= limit.max:
                if all(abs(value - v) > 1e-9 for v in values):
                    values.append(value)
        return values

    def check_levers(
        self, current: Mapping[str, float], proposed: Mapping[str, float]
    ) -> List[Violation]:
        """Lever range and rate-limit violations of a proposed change."""
        out: List[Violation] = []
        for tag, value in proposed.items():
            if tag not in self.levers:
                out.append(
                    Violation(
                        tag,
                        "unknown_lever",
                        float(value),
                        math.nan,
                        f"{tag} is not an approved lever",
                        f"{tag} در فهرست اهرم‌های مجاز نیست",
                    )
                )
                continue
            limit = self.levers[tag]
            step = float(value) - float(current[tag])
            if abs(step) < 1e-9:
                continue  # unchanged levers are never a violation
            if not limit.min <= value <= limit.max:
                bound = limit.min if value < limit.min else limit.max
                out.append(
                    Violation(
                        tag,
                        "lever_range",
                        float(value),
                        bound,
                        f"{tag}={value:.2f} outside [{limit.min}, {limit.max}]",
                        f"{tag}={value:.2f} خارج از بازهٔ مجاز [{limit.min}، {limit.max}]",
                    )
                )
            if abs(step) > limit.max_step + 1e-9:
                out.append(
                    Violation(
                        tag,
                        "lever_step",
                        step,
                        limit.max_step,
                        f"{tag} step {step:+.2f} exceeds max step {limit.max_step}",
                        f"گام {step:+.2f} برای {tag} بیشتر از حد مجاز {limit.max_step} است",
                    )
                )
        return out

    def check_process(self, tag: str, point: float, lower: float, upper: float) -> ProcessCheck:
        """Test a predicted value (and its interval) against the process limit."""
        limit = self.process_limits.get(tag, ProcessLimit())
        low = lower if limit.use_interval else point
        high = upper if limit.use_interval else point
        violations: List[Violation] = []
        if limit.min is not None and not low >= limit.min:
            violations.append(
                Violation(
                    tag,
                    "process_min",
                    low,
                    limit.min,
                    f"predicted {tag} {low:.1f} below minimum {limit.min}",
                    f"{tag} پیش‌بینی‌شده ({low:.1f}) کمتر از حداقل {limit.min} است",
                )
            )
        if limit.max is not None and not high <= limit.max:
            violations.append(
                Violation(
                    tag,
                    "process_max",
                    high,
                    limit.max,
                    f"predicted {tag} {high:.1f} above maximum {limit.max}",
                    f"{tag} پیش‌بینی‌شده ({high:.1f}) بیشتر از حداکثر {limit.max} است",
                )
            )
        return ProcessCheck(tag, point, lower, upper, limit, tuple(violations))

    def assert_safe(
        self,
        current: Mapping[str, float],
        proposed: Mapping[str, float],
        process_checks: List[ProcessCheck],
    ) -> None:
        """Final gate. Raise :class:`UnsafeRecommendationError` on any violation."""
        problems = self.check_levers(current, proposed)
        for check in process_checks:
            problems.extend(check.violations)
        if problems:
            raise UnsafeRecommendationError("; ".join(p.message for p in problems))

    def _lever(self, tag: str) -> LeverLimit:
        if tag not in self.levers:
            raise KeyError(f"{tag!r} is not an approved lever in the safety envelope.")
        return self.levers[tag]
