"""Operator feedback statistics and shadow-mode readiness."""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import pytest

from src.advisory.feedback import FeedbackRecord, acceptance_stats
from src.control.shadow import AdviceRecord, DryRunWriter, ReadinessCriteria, evaluate_shadow, readiness
from src.domain import tags as T
from src.safety.envelope import SafetyEnvelope
from src.sim.plant import simulate

T0 = datetime(2026, 1, 1)


def test_feedback_validation():
    with pytest.raises(ValueError):
        FeedbackRecord(1, "rejected", "ali", T0)  # reason required
    with pytest.raises(ValueError):
        FeedbackRecord(1, "maybe", "ali", T0)
    with pytest.raises(ValueError):
        FeedbackRecord(1, "rejected", "ali", T0, reason="bored")


def test_acceptance_uses_latest_answer():
    records = [
        FeedbackRecord(1, "rejected", "ali", T0, reason="quality_risk", shift="A"),
        FeedbackRecord(1, "accepted", "ali", T0 + timedelta(minutes=1), shift="A"),
        FeedbackRecord(2, "rejected", "sara", T0, reason="process_unstable", shift="B"),
        FeedbackRecord(3, "modified", "sara", T0, reason="too_large_step", shift="B"),
    ]
    stats = acceptance_stats(5, records)
    assert stats.n_answered == 3 and stats.accepted == 1 and stats.modified == 1
    assert stats.acceptance_rate == pytest.approx(2 / 3)
    assert stats.response_rate == pytest.approx(3 / 5)
    assert stats.reasons == {"process_unstable": 1, "too_large_step": 1}


def test_dry_run_writer_only_records():
    writer = DryRunWriter()
    writer.write({T.ID_FAN_PCT: 70.0}, "test")
    assert writer.log[0]["setpoints"] == {T.ID_FAN_PCT: 70.0}


def test_shadow_replay_detects_followed_moves_and_readiness():
    env = SafetyEnvelope.from_yaml()
    k = 2000
    base = simulate()
    followed = simulate(setpoint_overrides={T.SPECIFIC_HEAT_KCAL_KG: (k, 744.0)})
    current = {
        T.SPECIFIC_HEAT_KCAL_KG: float(base.wide[T.SPECIFIC_HEAT_KCAL_KG].iloc[k - 3 : k].mean()),
        T.ID_FAN_PCT: float(base.wide[T.ID_FAN_PCT].iloc[k - 3 : k].mean()),
    }
    rec = AdviceRecord(
        timestamp=base.wide.index[k],
        status="recommend",
        current=current,
        proposed={**current, T.SPECIFIC_HEAT_KCAL_KG: 744.0},
        decision="accepted",
    )
    report = evaluate_shadow([rec], followed.wide, env, T.SPECIFIC_HEAT_KCAL_KG, {T.BZ_TEMP_C: 75})
    assert report.agreement_rate == 1.0
    assert report.violations_after_followed == 0
    assert report.realized_objective_change < 0
    ignored = evaluate_shadow([rec], base.wide, env, T.SPECIFIC_HEAT_KCAL_KG, {T.BZ_TEMP_C: 75})
    assert ignored.agreement_rate == 0.0

    verdict = readiness(report)
    assert verdict["ready_for_closed_loop_trial"] is False  # one record is not evidence
    assert verdict["criteria"]["violations_after_followed"]["passed"] is True
    assert "HAZOP" in verdict["note"]


def test_readiness_passes_only_when_every_criterion_does():
    from src.control.shadow import ShadowReport

    good = ShadowReport(300, 250, 90.0, 0.8, 0.85, 0, {T.BZ_TEMP_C: 0.9}, -5.0)
    assert readiness(good)["ready_for_closed_loop_trial"] is True
    bad = ShadowReport(300, 250, 90.0, 0.8, 0.85, 1, {T.BZ_TEMP_C: 0.9}, -5.0)
    assert readiness(bad)["ready_for_closed_loop_trial"] is False
    miscalibrated = ShadowReport(300, 250, 90.0, 0.8, 0.85, 0, {T.BZ_TEMP_C: 0.7}, -5.0)
    assert readiness(miscalibrated, ReadinessCriteria())["ready_for_closed_loop_trial"] is False


def test_shadow_requires_records():
    with pytest.raises(ValueError):
        evaluate_shadow([], pd.DataFrame(), SafetyEnvelope({}, {}), "x", {})
