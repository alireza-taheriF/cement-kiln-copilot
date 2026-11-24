"""Mill scheduler (MILP) and M&V savings."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.domain import tags as T
from src.energy.mill_scheduler import Mill, ScheduleProblem, like_for_like, naive_schedule, optimize_schedule
from src.mv.baseline import daily_kiln_energy, daily_mill_energy, fit_baseline, savings_report
from src.sim.plant import SimConfig, simulate

MILLS = [Mill("M1", 4500, 105, start_cost=200_000), Mill("M2", 4600, 96, start_cost=200_000)]


def _problem(n=96, dt=0.5, dispatch_day=180.0, **overrides):
    hours = (np.arange(n) * dt) % 24
    quota = np.where((hours >= 19) & (hours < 23), 0, np.where((hours >= 11) & (hours < 15), 4800, 9500))
    grid = np.where((hours >= 17) & (hours < 23), 1500, np.where((hours >= 7) & (hours < 17), 1000, 600))
    kwargs = dict(
        mills=MILLS,
        slot_hours=dt,
        quota_kw=quota,
        grid_price=grid,
        free_market_price=np.full(n, 4000.0),
        free_market_cap_kw=np.full(n, 4600.0),
        dispatch_tph=np.where((hours >= 7) & (hours < 19), dispatch_day, 40.0),
        silo_initial_t=6000,
        silo_min_t=2000,
        silo_max_t=12000,
    )
    kwargs.update(overrides)
    return ScheduleProblem(**kwargs)


def test_optimal_schedule_respects_every_constraint():
    problem = _problem()
    s = optimize_schedule(problem)
    assert s.status == "optimal"
    kw = np.array([m.rated_kw for m in MILLS])[:, None]
    draw = (s.running * kw).sum(axis=0)
    assert np.all(draw <= np.asarray(problem.quota_kw) + np.asarray(problem.free_market_cap_kw) + 1e-6)
    assert np.allclose(draw, s.grid_kw + s.free_kw, atol=1e-6)
    assert np.all(s.grid_kw <= np.asarray(problem.quota_kw) + 1e-6)
    assert s.silo_t.min() >= problem.silo_min_t - 1e-6
    assert s.silo_t.max() <= problem.silo_max_t + 1e-6
    assert s.silo_t[-1] >= problem.silo_initial_t - 1e-6
    assert s.shortfall_t.sum() == pytest.approx(0.0, abs=1e-6)


def test_optimizer_never_loses_to_todays_rule_like_for_like():
    result = like_for_like(_problem(), [0.55, 0.45], [0.7, 0.65])
    cmp = result["comparison"]
    assert result["optimized"].silo_t[-1] >= result["baseline"].silo_t[-1] - 1e-6
    assert cmp["optimized"]["total_cost"] <= cmp["baseline"]["total_cost"] + 1e-6
    assert cmp["cost_saving_pct"] > 10  # tariff spread + free-market avoidance


def test_optimizer_moves_grinding_out_of_no_quota_hours():
    s = optimize_schedule(_problem(free_market_cap_kw=np.zeros(96)))
    hours = (np.arange(96) * 0.5) % 24
    blocked = (hours >= 19) & (hours < 23)
    assert not s.running[:, blocked].any()


def test_unavailable_mill_is_never_scheduled():
    availability = np.ones((2, 96), bool)
    availability[1, :] = False
    s = optimize_schedule(_problem(availability=availability))
    assert not s.running[1].any()


def test_naive_rule_reports_shortfall_when_capacity_is_short():
    problem = _problem(dispatch_day=400.0, silo_initial_t=2500)
    assert naive_schedule(problem, [0.75, 0.6], [0.95, 0.9]).shortfall_t.sum() > 0


def test_problem_validation():
    with pytest.raises(ValueError):
        _problem(grid_price=[1.0])
    with pytest.raises(ValueError):
        _problem(silo_initial_t=100)
    with pytest.raises(ValueError):
        Mill("bad", 0, 1)


# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def mv_runs():
    cfg = SimConfig(days=40, seed=11)
    base = simulate(cfg)
    k = 30 * 24 * 12
    copilot = simulate(cfg, setpoint_overrides={T.SPECIFIC_HEAT_KCAL_KG: (k, 740.0)})
    return daily_kiln_energy(base.wide), daily_kiln_energy(copilot.wide)


def test_baseline_passes_ashrae14_on_clean_data(mv_runs):
    base, _ = mv_runs
    model = fit_baseline(base.iloc[:30], "heat_gcal", ["clinker_t"], fit_intercept=False)
    assert model.passes_ashrae14()
    assert model.r2 > 0.8
    assert model.coefficients["clinker_t"] == pytest.approx(0.752, abs=0.01)  # ~752 kcal/kg


def test_real_saving_is_detected_and_sized(mv_runs):
    base, copilot = mv_runs
    model = fit_baseline(base.iloc[:30], "heat_gcal", ["clinker_t"], fit_intercept=False)
    report = savings_report(model, copilot.iloc[30:], unit="Gcal", unit_price=2.0, currency="IRT")
    true_saving = ((752.0 - 740.0) * copilot["clinker_t"].iloc[30:] / 1e3).sum()
    assert report.savings_significant
    assert report.savings == pytest.approx(true_saving, rel=0.25)
    assert report.savings_money == pytest.approx(2.0 * report.savings)


def test_no_change_is_not_reported_as_a_saving(mv_runs):
    base, _ = mv_runs
    model = fit_baseline(base.iloc[:30], "heat_gcal", ["clinker_t"], fit_intercept=False)
    report = savings_report(model, base.iloc[30:], unit="Gcal")
    assert abs(report.savings) < 2.5 * report.uncertainty


def test_failing_baseline_is_flagged():
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"e": rng.uniform(10, 190, 40), "x": rng.uniform(0, 1, 40)})
    model = fit_baseline(frame.iloc[:30], "e", ["x"])
    assert not model.passes_ashrae14()
    report = savings_report(model, frame.iloc[30:], unit="kWh")
    assert not report.baseline_valid and not report.savings_significant
    assert any("Guideline 14" in note for note in report.notes)


def test_daily_mill_energy(sim):
    daily = daily_mill_energy(sim.wide)
    assert (daily["mill_kwh"] / daily["cement_t"]).between(40, 50).all()
