"""Business KPIs and the plant simulator."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.domain import tags as T
from src.domain.kpis import (
    FuelProperties,
    add_derived_kpis,
    add_derived_kpis_long,
    gas_heat_share,
    heat_input_gcal_per_h,
    load_plant_config,
    mill_specific_power_kwh_per_t,
    specific_heat_kcal_per_kg,
)
from src.sim.plant import SimConfig, ground_truth_effects, simulate

FUELS = FuelProperties()


def test_specific_heat_matches_hand_calculation():
    # 3,000 t/d (125 t/h) at 750 kcal/kg needs 93.75 Gcal/h.
    gas = 93.75e6 / FUELS.gas_lhv_kcal_per_nm3
    heat = heat_input_gcal_per_h(gas, 0.0, FUELS)
    assert heat == pytest.approx(93.75)
    assert specific_heat_kcal_per_kg(np.array([heat]), np.array([125.0]))[0] == pytest.approx(750.0)


def test_specific_heat_is_undefined_when_kiln_is_down():
    value = specific_heat_kcal_per_kg(np.array([1.0]), np.array([0.5]))
    assert np.isnan(value[0])


def test_gas_share_and_mill_power():
    share = gas_heat_share(np.array([1000.0, 0.0]), np.array([0.0, 500.0]), FUELS)
    assert share.tolist() == [1.0, 0.0]
    kwh_t = mill_specific_power_kwh_per_t(np.array([4500.0, 4500.0]), np.array([100.0, 0.0]))
    assert kwh_t[0] == pytest.approx(45.0)
    assert np.isnan(kwh_t[1])


def test_fuel_properties_reject_nonsense():
    with pytest.raises(ValueError):
        FuelProperties(gas_lhv_kcal_per_nm3=0)


def test_add_derived_kpis_only_adds_what_inputs_allow():
    wide = pd.DataFrame({T.GAS_FLOW_NM3H: [10000.0], T.CLINKER_TPH: [120.0]})
    out = add_derived_kpis(wide, FUELS)
    assert {T.HEAT_INPUT_GCALH, T.GAS_HEAT_SHARE, T.SPECIFIC_HEAT_KCAL_KG} <= set(out.columns)
    assert not any("_KWH_T" in c for c in out.columns)


def test_add_derived_kpis_long_appends_rows():
    long = pd.DataFrame(
        {
            "timestamp": ["2026-01-01", "2026-01-01"],
            "tag_name": [T.GAS_FLOW_NM3H, T.CLINKER_TPH],
            "value": [10000.0, 120.0],
        }
    )
    out = add_derived_kpis_long(long, FUELS, only=[T.SPECIFIC_HEAT_KCAL_KG])
    assert set(out["tag_name"]) == {T.GAS_FLOW_NM3H, T.CLINKER_TPH, T.SPECIFIC_HEAT_KCAL_KG}


def test_plant_config_loads(tmp_path):
    cfg = load_plant_config("config/plant.yaml")
    assert cfg.fuels.mazut_lhv_kcal_per_kg > 0
    assert cfg.mill_ids == (1, 2)


def test_simulator_is_deterministic_and_plausible(sim):
    again = simulate()
    pd.testing.assert_frame_equal(sim.wide, again.wide)
    w = sim.wide
    assert 700 < w[T.SPECIFIC_HEAT_KCAL_KG].mean() < 800
    assert 1400 < w[T.BZ_TEMP_C].mean() < 1520
    assert w[T.GAS_HEAT_SHARE].min() == pytest.approx(0.0)
    assert len(sim.fuel_events) == 12
    assert sim.lab[T.FREE_LIME_PCT == sim.lab["parameter"]].shape[0] > 100


def test_simulator_respects_power_quota(sim):
    kw = sim.wide[T.mill_tag(T.MILL_KW_TEMPLATE, 1)] + sim.wide[T.mill_tag(T.MILL_KW_TEMPLATE, 2)]
    # Mill power never exceeds the quota by more than meter noise.
    assert (kw <= sim.power_quota_kw * 1.03 + 1).all()


def test_counterfactual_override_changes_only_the_future():
    cfg = SimConfig(days=2)
    base = simulate(cfg)
    k = 300
    moved = simulate(cfg, setpoint_overrides={T.SPECIFIC_HEAT_KCAL_KG: (k, 740.0)})
    pd.testing.assert_frame_equal(base.wide.iloc[:k], moved.wide.iloc[:k])
    # Less fuel -> colder burning zone an hour later (ground truth sign).
    later = slice(k + 12, k + 36)
    assert moved.wide[T.BZ_TEMP_C].iloc[later].mean() < base.wide[T.BZ_TEMP_C].iloc[later].mean()
    assert ground_truth_effects()[T.GAS_FLOW_NM3H][T.BZ_TEMP_C] == 1


def test_simulator_rejects_unknown_override():
    with pytest.raises(ValueError):
        simulate(SimConfig(days=1), setpoint_overrides={"FOO": (0, 1.0)})


def test_to_long_round_trips(sim):
    long = sim.to_long(include_derived=False)
    assert set(long.columns) == {"timestamp", "tag_name", "value"}
    assert T.SPECIFIC_HEAT_KCAL_KG not in set(long["tag_name"])
