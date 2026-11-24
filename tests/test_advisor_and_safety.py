"""Safety envelope and the constrained kiln advisor, validated on the simulator."""

from __future__ import annotations

import copy
import json

import numpy as np
import pytest
import yaml

from src.advisory.advisor import AdvisorConfig, KilnAdvisor
from src.domain import tags as T
from src.safety.envelope import (
    LeverLimit,
    ProcessLimit,
    SafetyEnvelope,
    UnsafeRecommendationError,
)
from src.sim.plant import simulate

ADVISOR_RAW = yaml.safe_load(open("config/advisor.yaml", encoding="utf-8"))
SAFETY_RAW = yaml.safe_load(open("config/safety_limits.yaml", encoding="utf-8"))


def _advisor(models, advisor_raw=None, safety_raw=None):
    return KilnAdvisor(
        AdvisorConfig.from_dict(advisor_raw or ADVISOR_RAW),
        SafetyEnvelope.from_dict(safety_raw or SAFETY_RAW),
        models,
    )


# --------------------------------------------------------------------------- #
# Envelope
# --------------------------------------------------------------------------- #
def test_lever_candidates_respect_range_and_step():
    env = SafetyEnvelope({"X": LeverLimit(0, 10, 2)}, {})
    assert env.lever_candidates("X", 9.0, [-3, -2, -1, 1, 2]) == [9.0, 7.0, 8.0, 10.0]


def test_check_levers_flags_range_step_and_unknown():
    env = SafetyEnvelope({"X": LeverLimit(0, 10, 2)}, {})
    kinds = {v.kind for v in env.check_levers({"X": 9.0, "Y": 0.0}, {"X": 12.0, "Y": 1.0})}
    assert kinds == {"lever_range", "lever_step", "unknown_lever"}
    assert env.check_levers({"X": 5.0}, {"X": 5.0}) == []


def test_process_check_uses_interval_when_asked():
    env = SafetyEnvelope({}, {"T": ProcessLimit(min=100, use_interval=True)})
    check = env.check_process("T", point=105, lower=99, upper=111)
    assert not check.ok and check.margin == pytest.approx(-1)
    env_point = SafetyEnvelope({}, {"T": ProcessLimit(min=100, use_interval=False)})
    assert env_point.check_process("T", 105, 99, 111).ok


def test_final_gate_raises():
    env = SafetyEnvelope({"X": LeverLimit(0, 10, 1)}, {})
    with pytest.raises(UnsafeRecommendationError):
        env.assert_safe({"X": 5.0}, {"X": 8.0}, [])


def test_envelope_rejects_inverted_limits():
    with pytest.raises(ValueError):
        LeverLimit(5, 1, 1)
    with pytest.raises(ValueError):
        ProcessLimit(min=5, max=1)


# --------------------------------------------------------------------------- #
# Advisor configuration
# --------------------------------------------------------------------------- #
def test_objective_must_be_a_lever():
    raw = copy.deepcopy(ADVISOR_RAW)
    raw["objective"]["tag"] = T.BZ_TEMP_C
    with pytest.raises(ValueError, match="levers"):
        AdvisorConfig.from_dict(raw)


def test_gain_bounds_must_agree_with_process_knowledge():
    raw = copy.deepcopy(ADVISOR_RAW)
    raw["gain_bounds"][T.SPECIFIC_HEAT_KCAL_KG][T.BZ_TEMP_C] = [-1.0, 1.0]
    with pytest.raises(ValueError, match="disagrees"):
        AdvisorConfig.from_dict(raw)


def test_advisor_requires_limits_for_every_model_target(constraint_models):
    raw = copy.deepcopy(SAFETY_RAW)
    del raw["process_limits"][T.KILN_INLET_CO_PPM]
    with pytest.raises(ValueError, match="No process limit"):
        _advisor(constraint_models, safety_raw=raw)


# --------------------------------------------------------------------------- #
# Advisor behaviour
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("k", [3150, 3450, 3750])
def test_recommendation_is_safe_when_actually_applied(sim, constraint_models, k):
    """Apply the advice in the simulator (same random draws) and check the
    real burning zone and CO over the next three hours stay inside limits."""
    advisor = _advisor(constraint_models)
    advice = advisor.advise(sim.wide.iloc[k - 60 : k])
    assert advice.status in {"recommend", "keep_current"}
    if advice.status != "recommend":
        return
    assert advice.objective_proposed < advice.objective_current
    overrides = {T.SPECIFIC_HEAT_KCAL_KG: (k, advice.proposed[T.SPECIFIC_HEAT_KCAL_KG])}
    if abs(advice.proposed[T.ID_FAN_PCT] - advice.current[T.ID_FAN_PCT]) > 1e-9:
        overrides[T.ID_FAN_PCT] = (k, advice.proposed[T.ID_FAN_PCT])
    after = simulate(setpoint_overrides=overrides).wide.iloc[k : k + 36]
    limits = SAFETY_RAW["process_limits"]
    assert after[T.BZ_TEMP_C].min() >= limits[T.BZ_TEMP_C]["min"]
    assert after[T.KILN_INLET_CO_PPM].max() <= limits[T.KILN_INLET_CO_PPM]["max"]
    assert after[T.SPECIFIC_HEAT_KCAL_KG].mean() < sim.wide[T.SPECIFIC_HEAT_KCAL_KG].iloc[k : k + 36].mean()


def test_recommendation_respects_lever_envelope(sim, constraint_models):
    advice = _advisor(constraint_models).advise(sim.wide.iloc[3390:3450])
    env = SafetyEnvelope.from_dict(SAFETY_RAW)
    if advice.proposed:
        assert env.check_levers(advice.current, advice.proposed) == []
        assert advice.operator_setpoints[T.GAS_FLOW_NM3H] >= 0
        assert advice.savings["heat_gcal_per_day"] > 0
    json.dumps(advice.to_dict(), allow_nan=False)  # strict JSON


def test_impossible_limits_give_no_safe_candidate(sim, constraint_models):
    raw = copy.deepcopy(SAFETY_RAW)
    raw["process_limits"][T.BZ_TEMP_C].update({"min": 1600, "max": 1700})
    advice = _advisor(constraint_models, safety_raw=raw).advise(sim.wide.iloc[3390:3450])
    assert advice.status == "no_safe_candidate"
    assert advice.proposed is None
    assert any("1600" in line for line in advice.explanation)


def test_small_gains_are_not_worth_an_operator_action(sim, constraint_models):
    raw = copy.deepcopy(ADVISOR_RAW)
    raw["objective"]["min_improvement"] = 100.0
    advice = _advisor(constraint_models, advisor_raw=raw).advise(sim.wide.iloc[3390:3450])
    assert advice.status == "keep_current"
    assert advice.proposed is None


class _BackwardsModel:
    """Wraps a model and flips the sign of the fuel effect (a confounded model)."""

    def __init__(self, inner):
        self._inner = inner
        self.target_name = inner.target_name
        self.feature_cols = inner.feature_cols
        self.metadata = inner.metadata
        self.is_fitted = True

    def predict_interval(self, X):
        cols = [c for c in X.columns if c.startswith(T.SPECIFIC_HEAT_KCAL_KG)]
        flipped = X.copy()
        flipped[cols] = 2 * 750.0 - X[cols]  # mirror around a fixed point
        return self._inner.predict_interval(flipped)


def test_lever_is_held_when_model_contradicts_physics(sim, constraint_models):
    models = dict(constraint_models)
    models["bz_temp"] = _BackwardsModel(constraint_models["bz_temp"])
    advice = _advisor(models).advise(sim.wide.iloc[3390:3450])
    assert T.SPECIFIC_HEAT_KCAL_KG in advice.held_levers
    fuel_check = [c for c in advice.direction_checks if c.lever == T.SPECIFIC_HEAT_KCAL_KG][0]
    assert fuel_check.status == "contradicted"
    if advice.proposed:
        assert advice.proposed[T.SPECIFIC_HEAT_KCAL_KG] == pytest.approx(
            advice.current[T.SPECIFIC_HEAT_KCAL_KG]
        )


def test_pessimistic_gain_shrinks_the_step_near_the_limit(sim, constraint_models):
    """Raising the minimum BZT close to the forecast must reduce (or stop) the cut."""
    loose = _advisor(constraint_models).advise(sim.wide.iloc[3390:3450])
    raw = copy.deepcopy(SAFETY_RAW)
    point = next(o.point for o in loose.baseline_outcomes if o.tag == T.BZ_TEMP_C)
    raw["process_limits"][T.BZ_TEMP_C]["min"] = float(np.floor(point - 12))
    tight = _advisor(constraint_models, safety_raw=raw).advise(sim.wide.iloc[3390:3450])
    cut = lambda a: (a.current[T.SPECIFIC_HEAT_KCAL_KG] - a.proposed[T.SPECIFIC_HEAT_KCAL_KG]) if a.proposed else 0.0  # noqa: E731
    assert cut(tight) <= cut(loose)


def test_advice_needs_enough_history(sim, constraint_models):
    with pytest.raises(ValueError, match="rows of history"):
        _advisor(constraint_models).advise(sim.wide.iloc[:3])
