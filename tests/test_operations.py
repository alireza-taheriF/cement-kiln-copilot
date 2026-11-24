"""Fuel transfers, free-lime soft sensor, condition monitoring, shift reports."""

from __future__ import annotations

import pandas as pd
import pytest

from src.domain import tags as T
from src.fuel.switch import (
    GAS_TO_MAZUT,
    MAZUT_TO_GAS,
    detect_transfers,
    estimate_o2_dip_per_share_rate,
    learn_ramp_policy,
    measure_impact,
    plan_transfer,
)
from src.maintenance.condition import Limits, assess_mills, assess_shell, assess_signal, worst_first
from src.reports.alarms import AlarmRule, derive_alarms, rules_from_limits, summarize
from src.reports.shift_report import numbers_consistent, render_en, render_fa, shift_facts, shift_window
from src.safety.envelope import SafetyEnvelope
from src.softsensor.free_lime import FreeLimeSoftSensor, select_delay


# --------------------------------------------------------------------------- #
# Fuel transfers
# --------------------------------------------------------------------------- #
def test_every_simulated_transfer_is_detected(sim):
    transfers = detect_transfers(sim.wide)
    assert len(transfers) == len(sim.fuel_events)
    for found, truth in zip(transfers, sim.fuel_events):
        assert found.direction == truth.direction
        assert abs(found.start - truth.start) <= pd.Timedelta(minutes=30)


def test_fast_transfers_upset_the_kiln_more(sim):
    impacts = [measure_impact(sim.wide, t) for t in detect_transfers(sim.wide)]
    fast = [i for i in impacts if i.transfer.ramp_minutes <= 45]
    co_fast = sum(i.co_minutes_above_limit for i in fast) / len(fast)
    assert co_fast > 0  # the 30/45-minute ramps drove CO over the limit
    assert min(i.o2_min for i in fast) < 1.0


def test_ramp_policy_prefers_slower_ramps_and_falls_back(sim):
    impacts = [measure_impact(sim.wide, t) for t in detect_transfers(sim.wide)]
    policy = learn_ramp_policy(impacts)
    assert policy.source[GAS_TO_MAZUT] == "history"
    assert policy.ramp_minutes[GAS_TO_MAZUT] >= 60
    empty = learn_ramp_policy([], default_ramp_minutes=75)
    assert empty.ramp_minutes[MAZUT_TO_GAS] == 75 and empty.source[MAZUT_TO_GAS] == "default"


def test_transfer_plan_keeps_heat_constant_and_biases_the_fan(sim):
    dip = estimate_o2_dip_per_share_rate(sim.wide)
    assert dip < 0
    plan = plan_transfer(sim.wide.iloc[:200], GAS_TO_MAZUT, 120, o2_dip_per_rate=dip, o2_per_fan_pct=0.12)
    heats = {round(s.heat_gcalh, 6) for s in plan.steps}
    assert len(heats) == 1
    assert plan.steps[0].mazut_kgh == 0 and plan.steps[-1].gas_nm3h == 0
    assert plan.steps[1].id_fan_pct > plan.steps[0].id_fan_pct
    assert plan.steps[-1].id_fan_pct == pytest.approx(plan.steps[0].id_fan_pct)
    assert plan.checklist_fa and len(plan.checklist) == len(plan.checklist_fa)
    with pytest.raises(ValueError):
        plan_transfer(sim.wide.iloc[:200], "sideways", 60)


# --------------------------------------------------------------------------- #
# Free-lime soft sensor
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def soft_sensor(sim):
    cut = sim.wide.index[3200]
    return FreeLimeSoftSensor.fit(sim.wide.loc[:cut], sim.lab[sim.lab["timestamp"] < cut])


def test_delay_search_finds_transport_plus_residence(sim):
    cut = sim.wide.index[3200]
    search = select_delay(
        sim.wide.loc[:cut],
        sim.lab[sim.lab["timestamp"] < cut],
        [T.BZ_TEMP_C, T.SPECIFIC_HEAT_KCAL_KG],
        60,
        range(15, 181, 15),
    )
    # 45 min transport + 30 min residence, read from a 60-min trailing window
    # (centred 30 min earlier): the best total delay is 45 min.
    assert search.best_delay_minutes == 45


def test_soft_sensor_beats_the_lab_average_out_of_sample(sim, soft_sensor):
    assert soft_sensor.cv_r2 > 0.6
    estimate = soft_sensor.estimate(sim.wide)
    test = sim.lab[sim.lab["timestamp"] >= sim.wide.index[3300]]
    errors = []
    for _, row in test.iterrows():
        at = row["timestamp"] - pd.Timedelta(minutes=soft_sensor.delay_minutes)
        errors.append(row["value"] - estimate.loc[:at].iloc[-1]["free_lime"])
    errors = pd.Series(errors)
    assert (errors**2).mean() ** 0.5 < 0.5 * test["value"].std()
    assert (errors.abs() <= soft_sensor.calibration.half_width).mean() >= 0.8


def test_bias_update_tracks_a_lab_offset(sim, soft_sensor):
    sensor = FreeLimeSoftSensor(**{**soft_sensor.__dict__, "bias": 0.0})
    row = sim.lab.iloc[-1]
    before = sensor.bias
    for _ in range(10):  # the lab keeps reading 0.3 % higher than the model
        sensor.update_bias(sim.wide, row["value"] + 0.3, row["timestamp"])
    assert sensor.bias > before + 0.2


# --------------------------------------------------------------------------- #
# Condition monitoring
# --------------------------------------------------------------------------- #
def test_faults_are_flagged_early_and_healthy_assets_are_quiet(sim):
    ref_end = sim.wide.index[0] + pd.Timedelta(days=7)
    early = sim.wide.loc[: sim.wide.index[0] + pd.Timedelta(days=11.5)]
    findings = {(f.asset, f.signal): f for f in assess_mills(early, [1, 2], reference_end=ref_end) + assess_shell(early, range(10), reference_end=ref_end)}
    assert findings[("cement mill 2", "vibration (mm/s)")].severity in {"warning", "alarm"}
    assert findings[("kiln shell", "position 4")].severity in {"warning", "alarm"}
    assert findings[("cement mill 1", "vibration (mm/s)")].severity == "ok"
    healthy = sim.wide.loc[: sim.wide.index[0] + pd.Timedelta(days=9.5)]
    quiet = assess_mills(healthy, [1, 2], reference_end=ref_end) + assess_shell(healthy, range(10), reference_end=ref_end)
    assert all(f.severity == "ok" for f in quiet)


def test_late_fault_is_an_alarm_and_sorted_first(sim):
    ref_end = sim.wide.index[0] + pd.Timedelta(days=7)
    findings = worst_first(assess_mills(sim.wide, [1, 2], reference_end=ref_end) + assess_shell(sim.wide, range(10), reference_end=ref_end))
    assert findings[0].severity == "alarm"
    assert findings[0].message_fa


def test_assess_signal_eta():
    idx = pd.date_range("2026-01-01", periods=24 * 12 * 4, freq="5min")
    series = pd.Series(range(len(idx)), index=idx, dtype=float) / 288.0 + 50  # +1 per day
    finding = assess_signal(series, "fan", "temp", Limits(warning=60, alarm=55), reference=series.iloc[:100])
    assert finding.slope_per_day == pytest.approx(1.0, rel=0.05)
    assert finding.hours_to_alarm == pytest.approx(24.0 * (55 - finding.value), rel=0.1)


# --------------------------------------------------------------------------- #
# Alarms and shift report
# --------------------------------------------------------------------------- #
def test_alarm_deadband_and_chattering():
    idx = pd.date_range("2026-01-01", periods=12, freq="1min")
    values = [0, 10, 9.5, 10, 0, 11, 0, 12, 0, 0, 0, 0]
    events = derive_alarms(pd.DataFrame({"X": values}, index=idx), [AlarmRule("X", "HI", 9, deadband=1)])
    assert len(events) == 3  # 9.5 is inside the deadband: no clear, no new alarm
    summary = summarize(events, hours=0.2)
    assert summary.chattering == ["X HI"]
    assert summary.bad_actors == ["X HI"]


def test_shift_window_boundaries():
    start, end, _ = shift_window(pd.Timestamp("2026-01-02 06:59"))
    assert (start, end) == (pd.Timestamp("2026-01-01 23:00"), pd.Timestamp("2026-01-02 07:00"))
    start, end, _ = shift_window(pd.Timestamp("2026-01-02 15:00"))
    assert (start, end) == (pd.Timestamp("2026-01-02 15:00"), pd.Timestamp("2026-01-02 23:00"))


def test_shift_report_numbers_come_from_data(sim):
    env = SafetyEnvelope.from_yaml()
    start, end, name = shift_window(pd.Timestamp("2026-01-06 10:00"))
    facts = shift_facts(sim.wide, start, end, lab=sim.lab, alarm_rules=rules_from_limits(env.process_limits), shift_name=name)
    window = sim.wide.loc[start:end].iloc[:-1]
    assert facts["clinker_t"] == pytest.approx(round(window[T.CLINKER_TPH].sum() * 5 / 60))
    assert facts["free_lime"]["n_samples"] == 4
    fa = render_fa(facts)
    assert "گزارش شیفت" in fa and "۰" in fa or "۱" in fa
    assert numbers_consistent(fa, fa)
    assert not numbers_consistent(fa + " ۹۹۹۹۹۹", fa)
    assert "Clinker" in render_en(facts)
