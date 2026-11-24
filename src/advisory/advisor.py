"""Constrained, uncertainty-aware kiln advisor.

What it answers
---------------
"How far can fuel per kg clinker come down right now, and which ID-fan
setting goes with it, without the burning zone getting too cold or CO
getting too high?"

How
---
1. **Objective is not learnt.** The objective (specific heat consumption,
   kcal/kg) *is* a lever, so its value for a candidate is exact. Models are
   only used for what cannot be computed: the consequences.
2. **Constraint models** forecast the guarded outcomes (burning-zone
   temperature, CO) with conformal prediction intervals. A candidate is
   feasible only if the *pessimistic* end of every interval is inside the
   safety envelope (``config/safety_limits.yaml``).
3. **Direction check.** Before a lever is used, each model's local response
   to it is probed and compared with process knowledge
   (``expected_effects``). If a model says more fuel makes the kiln colder,
   or that fuel has no effect at all, the lever is held: a model that cannot
   reproduce the basic physics locally must not be trusted to trade it off.
4. **Bounded gains.** Models learnt from routine operation see little
   independent variation of a setpoint and are confounded by operator
   behaviour, so their *gain* (how much BZT drops per kcal/kg less fuel)
   is the least reliable thing they produce. Where ``gain_bounds`` are
   configured (from plant step tests), a lever's effect is computed as
   ``baseline forecast + gain x change``: the point estimate uses the
   model's local gain clipped into the bounds, and the interval uses the
   *worst* gain in the bounds. The model is still used for what it is good
   at: the baseline forecast from the current state and disturbances.
5. **Rate-limited candidates.** Only changes inside each lever's range and
   max step are generated.
6. **Keep-current bias.** A change is recommended only when it improves the
   objective by at least ``min_improvement``; otherwise the answer is "keep
   current", which is the right answer most of the time.
7. **Final gate.** The chosen recommendation passes the envelope check a
   second time before it is returned.

What-if semantics
-----------------
A candidate is scored as "the lever had been at this value over the model's
whole input window" (quasi-steady state), and the last forecast row is read.
That matches how operators reason about a setpoint move held for the next
half hour; it is not a dynamic trajectory optimizer (that is what an APC
does, see ``docs/ROADMAP.md``).
"""

from __future__ import annotations

import itertools
import logging
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional

import numpy as np
import pandas as pd
import yaml

from src.domain import tags as T
from src.domain.kpis import FuelProperties, PlantConfig, add_derived_kpis
from src.ml.inference import features_from_wide, min_window_rows
from src.ml.models import EnergyKPIModel
from src.safety.envelope import ProcessCheck, SafetyEnvelope

logger = logging.getLogger(__name__)

#: Rows averaged to read the "current" lever value (removes sensor jitter).
CURRENT_VALUE_ROWS = 3


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AdvisorConfig:
    """Parsed ``config/advisor.yaml``."""

    objective_tag: str
    direction: str  # "minimize" | "maximize"
    min_improvement: float
    lever_deltas: Dict[str, List[float]]
    constraint_models: Dict[str, str]
    expected_effects: Dict[str, Dict[str, int]]
    direction_probe: Dict[str, float]
    direction_tolerance: Dict[str, float]
    gain_bounds: Dict[str, Dict[str, tuple[float, float]]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.direction not in ("minimize", "maximize"):
            raise ValueError("objective.direction must be 'minimize' or 'maximize'.")
        if self.objective_tag not in self.lever_deltas:
            raise ValueError(
                f"objective tag {self.objective_tag!r} must be one of the levers: "
                "the advisor computes the objective exactly from the candidate."
            )
        if self.min_improvement < 0:
            raise ValueError("min_improvement must be >= 0.")
        for lever, targets in self.gain_bounds.items():
            for target, (lo, hi) in targets.items():
                if lo > hi:
                    raise ValueError(f"gain_bounds[{lever}][{target}] has min > max.")
                expected = self.expected_effects.get(lever, {}).get(target)
                if expected and (lo * expected < 0 or hi * expected < 0):
                    raise ValueError(
                        f"gain_bounds[{lever}][{target}] = [{lo}, {hi}] disagrees with "
                        f"expected_effects sign {expected:+d}."
                    )

    @classmethod
    def from_yaml(cls, path: str | Path = "config/advisor.yaml") -> "AdvisorConfig":
        return cls.from_dict(yaml.safe_load(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def from_dict(cls, raw: Mapping) -> "AdvisorConfig":
        objective = raw["objective"]
        return cls(
            objective_tag=str(objective["tag"]),
            direction=str(objective.get("direction", "minimize")),
            min_improvement=float(objective.get("min_improvement", 0.0)),
            lever_deltas={str(k): [float(x) for x in v] for k, v in raw["levers"].items()},
            constraint_models={str(k): str(v) for k, v in raw.get("constraint_models", {}).items()},
            expected_effects={
                str(lever): {str(t): int(np.sign(s)) for t, s in effects.items()}
                for lever, effects in (raw.get("expected_effects") or {}).items()
            },
            direction_probe={str(k): float(v) for k, v in (raw.get("direction_probe") or {}).items()},
            direction_tolerance={
                str(k): float(v) for k, v in (raw.get("direction_tolerance") or {}).items()
            },
            gain_bounds={
                str(lever): {str(t): (float(b[0]), float(b[1])) for t, b in targets.items()}
                for lever, targets in (raw.get("gain_bounds") or {}).items()
            },
        )


# --------------------------------------------------------------------------- #
# Result types
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Outcome:
    """Forecast of one guarded variable for one candidate."""

    role: str
    tag: str
    point: float
    lower: float
    upper: float
    ok: bool
    margin: float
    violations: List[str]
    violations_fa: List[str]

    @classmethod
    def from_check(cls, role: str, check: ProcessCheck) -> "Outcome":
        return cls(
            role=role,
            tag=check.tag,
            point=float(check.point),
            lower=float(check.lower),
            upper=float(check.upper),
            ok=check.ok,
            margin=float(check.margin),
            violations=[v.message for v in check.violations],
            violations_fa=[v.message_fa for v in check.violations],
        )


@dataclass(frozen=True)
class DirectionCheck:
    """Local response of a model to a lever, against process knowledge."""

    lever: str
    target: str
    expected_sign: int
    response: float  # change in target for +2*probe on the lever
    status: str  # "confirmed" | "contradicted" | "no_effect"
    model_gain: float = 0.0  # response / (2 * probe), target units per lever unit
    gain_used: Optional[float] = None  # model gain clipped into gain_bounds, if bounded

    @property
    def trusted(self) -> bool:
        return self.status == "confirmed"


@dataclass
class Candidate:
    """A scored lever combination."""

    levers: Dict[str, float]
    objective: float
    outcomes: List[Outcome]

    @property
    def feasible(self) -> bool:
        return all(o.ok for o in self.outcomes)

    @property
    def min_margin(self) -> float:
        return min((o.margin for o in self.outcomes), default=math.inf)


@dataclass
class Advice:
    """What the advisor returns (serializable with :meth:`to_dict`)."""

    status: str  # "recommend" | "keep_current" | "restore_limits" | "no_safe_candidate"
    current: Dict[str, float]
    proposed: Optional[Dict[str, float]]
    objective_tag: str
    objective_current: float
    objective_proposed: Optional[float]
    improvement: float
    baseline_outcomes: List[Outcome]
    proposed_outcomes: List[Outcome]
    direction_checks: List[DirectionCheck]
    held_levers: List[str]
    n_candidates: int
    n_feasible: int
    binding_constraint: Optional[str]
    operator_setpoints: Dict[str, float] = field(default_factory=dict)
    savings: Dict[str, float] = field(default_factory=dict)
    explanation: List[str] = field(default_factory=list)
    explanation_fa: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        out = asdict(self)
        # asdict drops properties; keep the trust flag explicit for clients.
        for check, raw in zip(self.direction_checks, out["direction_checks"]):
            raw["trusted"] = check.trusted
        return _finite(out)


def _finite(obj):
    """Replace non-finite floats (JSON has no inf/nan) with ``None``."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _finite(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_finite(v) for v in obj]
    return obj


# --------------------------------------------------------------------------- #
# Advisor
# --------------------------------------------------------------------------- #
class KilnAdvisor:
    """Search lever combinations under learnt, interval-aware constraints."""

    def __init__(
        self,
        config: AdvisorConfig,
        envelope: SafetyEnvelope,
        models: Mapping[str, EnergyKPIModel],
        fuels: Optional[FuelProperties] = None,
        plant: Optional[PlantConfig] = None,
    ) -> None:
        missing_limits = [
            str(m.target_name) for m in models.values() if m.target_name not in envelope.process_limits
        ]
        if missing_limits:
            raise ValueError(
                f"No process limit for model target(s) {missing_limits}; "
                "add them to safety_limits.yaml."
            )
        unknown = [lever for lever in config.lever_deltas if lever not in envelope.levers]
        if unknown:
            raise ValueError(f"Lever(s) {unknown} have no limits in the safety envelope.")
        self.config = config
        self.envelope = envelope
        self.models = dict(models)
        self.fuels = plant.fuels if plant else (fuels or FuelProperties())
        self.plant = plant
        self.hold_rows = max(min_window_rows(m) for m in self.models.values()) if self.models else 1

    @classmethod
    def from_paths(
        cls,
        advisor_config: str | Path = "config/advisor.yaml",
        safety_config: str | Path = "config/safety_limits.yaml",
        plant: Optional[PlantConfig] = None,
    ) -> "KilnAdvisor":
        cfg = AdvisorConfig.from_yaml(advisor_config)
        models = {role: EnergyKPIModel.load(path) for role, path in cfg.constraint_models.items()}
        return cls(cfg, SafetyEnvelope.from_yaml(safety_config), models, plant=plant)

    # ------------------------------------------------------------------ #
    def advise(self, wide: pd.DataFrame) -> Advice:
        """Recommend lever settings for the latest state in ``wide``."""
        wide = self._prepare(wide)
        current = {
            lever: float(wide[lever].tail(CURRENT_VALUE_ROWS).mean())
            for lever in self.config.lever_deltas
        }
        checks = self._direction_checks(wide, current)
        held = sorted(
            {c.lever for c in checks if not c.trusted}
            | {lever for lever in current if lever not in self.config.expected_effects}
        )

        self._gains = {
            (c.lever, c.target): c.gain_used for c in checks if c.trusted and c.gain_used is not None
        }
        self._base_cache: Dict[str, tuple] = {}
        baseline = self._score(wide, current, current)
        grids = []
        for lever, deltas in self.config.lever_deltas.items():
            if lever in held:
                grids.append([current[lever]])
            else:
                grids.append(self.envelope.lever_candidates(lever, current[lever], deltas))
        levers = list(self.config.lever_deltas)
        candidates = [baseline]
        for combo in itertools.product(*grids):
            values = dict(zip(levers, combo))
            if all(abs(values[k] - current[k]) < 1e-9 for k in levers):
                continue
            candidates.append(self._score(wide, values, current))

        feasible = [c for c in candidates if c.feasible]
        sign = 1.0 if self.config.direction == "minimize" else -1.0
        advice = Advice(
            status="keep_current",
            current=current,
            proposed=None,
            objective_tag=self.config.objective_tag,
            objective_current=baseline.objective,
            objective_proposed=None,
            improvement=0.0,
            baseline_outcomes=baseline.outcomes,
            proposed_outcomes=[],
            direction_checks=checks,
            held_levers=held,
            n_candidates=len(candidates),
            n_feasible=len(feasible),
            binding_constraint=None,
        )
        if not feasible:
            advice.status = "no_safe_candidate"
            self._explain(advice, None, baseline)
            return advice

        best = min(feasible, key=lambda c: (sign * c.objective, -c.min_margin))
        improvement = sign * (baseline.objective - best.objective)
        if baseline.feasible and (best is baseline or improvement < self.config.min_improvement):
            self._explain(advice, None, baseline)
            return advice

        advice.status = "recommend" if baseline.feasible else "restore_limits"
        advice.proposed = best.levers
        advice.objective_proposed = best.objective
        advice.improvement = float(improvement)
        advice.proposed_outcomes = best.outcomes
        binding = min(best.outcomes, key=lambda o: o.margin, default=None)
        advice.binding_constraint = binding.tag if binding else None
        advice.operator_setpoints = self._operator_setpoints(wide, best.levers)
        advice.savings = self._savings(wide, current, best.levers)

        # Final, deliberately redundant gate.
        self.envelope.assert_safe(
            current,
            best.levers,
            [self.envelope.check_process(o.tag, o.point, o.lower, o.upper) for o in best.outcomes],
        )
        self._explain(advice, best, baseline)
        return advice

    # ------------------------------------------------------------------ #
    def _prepare(self, wide: pd.DataFrame) -> pd.DataFrame:
        missing = [lever for lever in self.config.lever_deltas if lever not in wide]
        if missing:
            wide = add_derived_kpis(wide, self.fuels)
        missing = [lever for lever in self.config.lever_deltas if lever not in wide]
        if missing:
            raise ValueError(f"Signals are missing lever tag(s): {missing}.")
        if len(wide) < self.hold_rows:
            raise ValueError(
                f"Need at least {self.hold_rows} rows of history, got {len(wide)}."
            )
        return wide

    def _apply(self, wide: pd.DataFrame, values: Mapping[str, float]) -> pd.DataFrame:
        frame = wide.copy()
        rows = frame.index[-self.hold_rows :]
        for lever, value in values.items():
            frame.loc[rows, lever] = float(value)
        return frame

    def _predict(self, wide: pd.DataFrame, values: Mapping[str, float]) -> Dict[str, tuple]:
        out: Dict[str, tuple] = {}
        for role, model in self.models.items():
            point, lower, upper = self._predict_one(wide, values, model)
            out[role] = (str(model.target_name), point, lower, upper)
        return out

    def _score(
        self,
        wide: pd.DataFrame,
        values: Mapping[str, float],
        current: Mapping[str, float],
    ) -> Candidate:
        """Forecast every guarded outcome for ``values``.

        Levers with configured gain bounds for a target are reset to their
        current value in the model what-if and added back as
        ``gain x change`` (point: clipped model gain; interval: worst gain in
        the bounds). Other levers are scored by the model what-if directly.
        """
        bounded = self.config.gain_bounds
        outcomes: List[Outcome] = []
        for role, model in self.models.items():
            target = str(model.target_name)
            whatif = dict(values)
            add_point = add_low = add_high = 0.0
            for lever, value in values.items():
                delta = float(value) - float(current[lever])
                gain = self._gains.get((lever, target))
                if lever in bounded and target in bounded[lever] and gain is not None:
                    whatif[lever] = current[lever]
                    lo, hi = bounded[lever][target]
                    add_point += gain * delta
                    add_low += min(lo * delta, hi * delta)
                    add_high += max(lo * delta, hi * delta)
            key = repr(sorted(whatif.items()))
            cache_key = f"{role}|{key}"
            if cache_key not in self._base_cache:
                self._base_cache[cache_key] = self._predict_one(wide, whatif, model)
            point, lower, upper = self._base_cache[cache_key]
            check = self.envelope.check_process(
                target, point + add_point, lower + add_low, upper + add_high
            )
            outcomes.append(Outcome.from_check(role, check))
        return Candidate(dict(values), float(values[self.config.objective_tag]), outcomes)

    def _predict_one(
        self, wide: pd.DataFrame, values: Mapping[str, float], model: EnergyKPIModel
    ) -> tuple[float, float, float]:
        frame = self._apply(wide, values)
        X = features_from_wide(frame, model, fuels=self.fuels).tail(1)
        point, lower, upper = model.predict_interval(X)
        return float(point[0]), float(lower[0]), float(upper[0])

    def _direction_checks(self, wide: pd.DataFrame, current: Dict[str, float]) -> List[DirectionCheck]:
        checks: List[DirectionCheck] = []
        targets = {str(m.target_name): role for role, m in self.models.items()}
        for lever, effects in self.config.expected_effects.items():
            if lever not in current:
                continue
            probe = self.config.direction_probe.get(lever, 1.0)
            up = self._predict(wide, {**current, lever: current[lever] + probe})
            down = self._predict(wide, {**current, lever: current[lever] - probe})
            for target, expected in effects.items():
                role = targets.get(target)
                if role is None:
                    continue
                response = up[role][1] - down[role][1]
                tol = self.config.direction_tolerance.get(target, 0.0)
                if abs(response) < tol:
                    status = "no_effect"
                elif np.sign(response) == expected:
                    status = "confirmed"
                else:
                    status = "contradicted"
                model_gain = float(response) / (2.0 * probe)
                gain_used = None
                bounds = self.config.gain_bounds.get(lever, {}).get(target)
                if bounds is not None and status == "confirmed":
                    gain_used = float(np.clip(model_gain, bounds[0], bounds[1]))
                checks.append(
                    DirectionCheck(
                        lever, target, expected, float(response), status, model_gain, gain_used
                    )
                )
                if status != "confirmed":
                    logger.warning(
                        "Lever %s held: model for %s %s (response %+.3f)",
                        lever,
                        target,
                        status,
                        response,
                    )
        return checks

    def _operator_setpoints(self, wide: pd.DataFrame, levers: Mapping[str, float]) -> Dict[str, float]:
        """Translate the specific-heat lever into burner flows at the current fuel mix."""
        out: Dict[str, float] = {}
        if T.SPECIFIC_HEAT_KCAL_KG in levers and T.CLINKER_TPH in wide:
            clinker = float(wide[T.CLINKER_TPH].tail(CURRENT_VALUE_ROWS).mean())
            share = float(wide.get(T.GAS_HEAT_SHARE, pd.Series([1.0])).tail(CURRENT_VALUE_ROWS).mean())
            heat = levers[T.SPECIFIC_HEAT_KCAL_KG] * clinker / 1e3  # Gcal/h
            out[T.HEAT_INPUT_GCALH] = heat
            out[T.GAS_FLOW_NM3H] = heat * share * 1e6 / self.fuels.gas_lhv_kcal_per_nm3
            out[T.MAZUT_FLOW_KGH] = heat * (1 - share) * 1e6 / self.fuels.mazut_lhv_kcal_per_kg
        for lever, value in levers.items():
            if lever != T.SPECIFIC_HEAT_KCAL_KG:
                out[lever] = float(value)
        return out

    def _savings(
        self, wide: pd.DataFrame, current: Mapping[str, float], proposed: Mapping[str, float]
    ) -> Dict[str, float]:
        """Heat, fuel and money saved per hour and per day at the current mix."""
        if T.SPECIFIC_HEAT_KCAL_KG not in proposed or T.CLINKER_TPH not in wide:
            return {}
        clinker = float(wide[T.CLINKER_TPH].tail(CURRENT_VALUE_ROWS).mean())
        share = float(wide.get(T.GAS_HEAT_SHARE, pd.Series([1.0])).tail(CURRENT_VALUE_ROWS).mean())
        delta = current[T.SPECIFIC_HEAT_KCAL_KG] - proposed[T.SPECIFIC_HEAT_KCAL_KG]
        heat = delta * clinker / 1e3  # Gcal/h saved
        gas_nm3 = heat * share * 1e6 / self.fuels.gas_lhv_kcal_per_nm3
        mazut_kg = heat * (1 - share) * 1e6 / self.fuels.mazut_lhv_kcal_per_kg
        mazut_l = mazut_kg / self.fuels.mazut_density_kg_per_l
        out = {
            "heat_gcal_per_h": heat,
            "gas_nm3_per_h": gas_nm3,
            "mazut_l_per_h": mazut_l,
            "heat_gcal_per_day": heat * 24,
            "gas_nm3_per_day": gas_nm3 * 24,
            "mazut_l_per_day": mazut_l * 24,
        }
        if self.plant is not None:
            cost_h = (
                gas_nm3 * self.plant.gas_price_per_nm3
                + mazut_l * (self.plant.mazut_price_per_l + self.plant.mazut_transport_per_l)
            )
            out["cost_per_h"] = cost_h
            out["cost_per_day"] = cost_h * 24
        return out

    def _explain(self, advice: Advice, best: Optional[Candidate], baseline: Candidate) -> None:
        en: List[str] = []
        fa: List[str] = []
        for check in advice.direction_checks:
            if not check.trusted:
                en.append(
                    f"{check.lever} held: the {check.target} model's response "
                    f"({check.response:+.2f}) is {check.status.replace('_', ' ')}."
                )
                fa.append(
                    f"{check.lever} ثابت نگه داشته شد: پاسخ مدل {check.target} "
                    f"({check.response:+.2f}) با دانش فرایند سازگار نیست."
                )
        if advice.status == "no_safe_candidate":
            for outcome in baseline.outcomes:
                en.extend(outcome.violations)
                fa.extend(outcome.violations_fa)
            en.append("No candidate keeps every guarded variable inside its limits; escalate to the shift lead.")
            fa.append("هیچ گزینه‌ای همهٔ متغیرهای حفاظتی را در محدوده نگه نمی‌دارد؛ به سرشیفت اطلاع دهید.")
        elif best is None:
            en.append(
                f"Keep current settings: no safe change improves {advice.objective_tag} "
                f"by at least {self.config.min_improvement:g}."
            )
            fa.append(
                f"تنظیمات فعلی را نگه دارید: هیچ تغییر ایمنی {advice.objective_tag} را "
                f"حداقل {self.config.min_improvement:g} واحد بهتر نمی‌کند."
            )
        else:
            for lever, value in best.levers.items():
                delta = value - advice.current[lever]
                if abs(delta) > 1e-9:
                    en.append(f"Move {lever} from {advice.current[lever]:.1f} to {value:.1f} ({delta:+.1f}).")
                    fa.append(f"{lever} را از {advice.current[lever]:.1f} به {value:.1f} ببرید ({delta:+.1f}).")
            for outcome in best.outcomes:
                en.append(
                    f"Predicted {outcome.tag}: {outcome.point:.1f} "
                    f"(90% range {outcome.lower:.1f}–{outcome.upper:.1f}), margin {outcome.margin:.1f}."
                )
                fa.append(
                    f"{outcome.tag} پیش‌بینی‌شده: {outcome.point:.1f} "
                    f"(بازهٔ ۹۰٪: {outcome.lower:.1f} تا {outcome.upper:.1f})، فاصله تا حد مجاز {outcome.margin:.1f}."
                )
            if advice.binding_constraint:
                en.append(f"The limiting constraint is {advice.binding_constraint}.")
                fa.append(f"قید تعیین‌کننده {advice.binding_constraint} است.")
            if advice.savings:
                en.append(
                    f"Saves about {advice.savings['heat_gcal_per_day']:.1f} Gcal/day "
                    f"({advice.savings['mazut_l_per_day']:.0f} L mazut, "
                    f"{advice.savings['gas_nm3_per_day']:.0f} Nm3 gas per day at the current fuel mix)."
                )
                fa.append(
                    f"صرفه‌جویی حدود {advice.savings['heat_gcal_per_day']:.1f} گیگاکالری در روز "
                    f"({advice.savings['mazut_l_per_day']:.0f} لیتر مازوت و "
                    f"{advice.savings['gas_nm3_per_day']:.0f} مترمکعب گاز در روز با ترکیب سوخت فعلی)."
                )
            if advice.status == "restore_limits":
                en.insert(0, "Current settings are predicted to break a limit; this change restores it.")
                fa.insert(0, "تنظیمات فعلی پیش‌بینی می‌شود از حد مجاز خارج شود؛ این تغییر آن را برمی‌گرداند.")
        en.append("Advisory only: the operator decides; nothing is written to the DCS.")
        fa.append("فقط مشورتی: تصمیم با اپراتور است و هیچ فرمانی به DCS ارسال نمی‌شود.")
        advice.explanation = en
        advice.explanation_fa = fa
