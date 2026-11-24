"""Deterministic first-principles-flavoured simulator of a cement kiln line.

Why a simulator
---------------
Real plant data is not in the repository (it is the customer's), but every
module still needs data that behaves like a plant: transport delays, fuel
switches, lab samples that lag the process, mills competing for a power
quota, equipment that slowly degrades. The simulator provides that, with a
**known ground truth**, so tests can check that a model learnt the right
*direction* of an effect and not just a correlation.

It is not a validated process model. Coefficients are chosen to be in the
right range for a 3,000 t/d dry-process line, nothing more. Never use its
numbers for engineering decisions.

Process structure (one step = ``step_minutes``)
-----------------------------------------------
* Raw-meal feed wanders slowly; clinker = feed / clinker factor.
* A hidden *burnability* index sets the heat the clinker actually needs.
* Operators fire to a specific-heat setpoint with a safety margin.
* Gas supply is curtailed in scheduled windows; the burner is transferred
  to mazut with a linear ramp whose duration differs per event.
* O2 follows the ID fan and dips during fuel transfers; CO rises
  exponentially when O2 is low.
* Excess air and CO are heat losses: they reduce the *effective* heat.
* Burning-zone temperature responds to effective-heat surplus through a
  dead time and a first-order lag.
* Free lime follows burning-zone temperature with a residence delay; the
  lab samples it every two hours with a further transport delay.
* Two cement mills fill a silo under an electrical power quota.
* Mill 2 develops a bearing fault and the shell grows a hot spot near the
  end of the horizon (for the maintenance module).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.domain import tags as T
from src.domain.kpis import FuelProperties, add_derived_kpis


@dataclass(frozen=True)
class SimConfig:
    """Simulation parameters (defaults: generic 3,000 t/d line, 5-min steps)."""

    days: float = 14.0
    step_minutes: int = 5
    seed: int = 7
    start: str = "2026-01-01 00:00"
    clinker_factor: float = 1.58
    base_feed_tph: float = 197.5
    operator_specific_heat: float = 752.0  # what operators fire to (kcal/kg)
    required_specific_heat: float = 735.0  # what the clinker needs on average
    # Fuel-curtailment events: (start_day, duration_hours, transfer_ramp_minutes)
    gas_curtailments: tuple[tuple[float, float, int], ...] = (
        (1.5, 14.0, 30),
        (3.2, 10.0, 150),
        (5.6, 20.0, 45),
        (7.4, 8.0, 180),
        (9.1, 16.0, 60),
        (11.3, 12.0, 120),
    )
    # Mill data
    mill_rated_kw: tuple[float, float] = (4500.0, 4600.0)
    mill_rated_tph: tuple[float, float] = (105.0, 96.0)
    silo_capacity_t: float = 12000.0
    silo_initial_t: float = 6000.0
    # Hours (local) when the grid quota allows only one mill
    quota_one_mill_hours: tuple[int, ...] = (11, 12, 13, 14)
    # Hours when no mill may run
    quota_no_mill_hours: tuple[int, ...] = (19, 20, 21, 22)
    # Degradation
    bearing_fault_start_day: float = 10.0
    hot_spot_start_day: float = 11.0
    hot_spot_position: int = 4
    shell_positions: int = 10

    @property
    def n_steps(self) -> int:
        return int(self.days * 24 * 60 / self.step_minutes)

    def minutes_to_steps(self, minutes: float) -> int:
        return max(1, int(round(minutes / self.step_minutes)))


@dataclass
class FuelSwitchEvent:
    """Ground truth for one simulated gas curtailment."""

    start: pd.Timestamp
    end: pd.Timestamp
    ramp_minutes: int
    direction: str  # "gas_to_mazut" or "mazut_to_gas"


@dataclass
class SimulationResult:
    """Simulator output.

    Attributes
    ----------
    wide:
        Timestamp-indexed process tags plus derived KPIs.
    lab:
        Long-format lab results (``timestamp, parameter, value, sample_id``).
    fuel_events:
        Every fuel transfer, with its ramp duration.
    power_quota_kw:
        Allowed total mill power per step (the grid quota).
    true_free_lime:
        Free lime at the kiln outlet at every step (unobserved in practice).
    """

    wide: pd.DataFrame
    lab: pd.DataFrame
    fuel_events: List[FuelSwitchEvent]
    power_quota_kw: pd.Series
    true_free_lime: pd.Series
    config: SimConfig = field(repr=False, default_factory=SimConfig)

    def to_long(self, include_derived: bool = True) -> pd.DataFrame:
        """Long-format signals (``timestamp, tag_name, value``) for ingestion."""
        frame = self.wide
        if not include_derived:
            derived = {T.HEAT_INPUT_GCALH, T.GAS_HEAT_SHARE, T.SPECIFIC_HEAT_KCAL_KG}
            frame = frame[[c for c in frame.columns if c not in derived and "_KWH_T" not in c]]
        return (
            frame.rename_axis("timestamp")
            .reset_index()
            .melt(id_vars="timestamp", var_name="tag_name", value_name="value")
            .dropna(subset=["value"])
            .sort_values(["timestamp", "tag_name"], kind="mergesort")
            .reset_index(drop=True)
        )


def _ar1(rng: np.random.Generator, n: int, phi: float, sigma: float) -> np.ndarray:
    """Stationary AR(1) noise with innovation std ``sigma``."""
    out = np.empty(n)
    out[0] = rng.normal(0.0, sigma / np.sqrt(max(1e-9, 1 - phi**2)))
    eps = rng.normal(0.0, sigma, n)
    for k in range(1, n):
        out[k] = phi * out[k - 1] + eps[k]
    return out


def _gas_share_profile(cfg: SimConfig, index: pd.DatetimeIndex) -> tuple[np.ndarray, List[FuelSwitchEvent]]:
    """Gas heat share over time and the transfer events that shape it."""
    n = len(index)
    share = np.ones(n)
    events: List[FuelSwitchEvent] = []
    t0 = index[0]
    for start_day, hours, ramp_min in cfg.gas_curtailments:
        k_start = int(start_day * 24 * 60 / cfg.step_minutes)
        k_end = k_start + int(hours * 60 / cfg.step_minutes)
        ramp = cfg.minutes_to_steps(ramp_min)
        if k_start >= n:
            continue
        for k in range(k_start, min(n, k_end + ramp)):
            if k < k_start + ramp:
                level = 1.0 - (k - k_start + 1) / ramp
            elif k < k_end:
                level = 0.0
            else:
                level = (k - k_end + 1) / ramp
            share[k] = min(share[k], float(np.clip(level, 0.0, 1.0)))
        step = pd.Timedelta(minutes=cfg.step_minutes)
        events.append(
            FuelSwitchEvent(t0 + k_start * step, t0 + (k_start + ramp) * step, ramp_min, "gas_to_mazut")
        )
        if k_end < n:
            events.append(
                FuelSwitchEvent(t0 + k_end * step, t0 + (k_end + ramp) * step, ramp_min, "mazut_to_gas")
            )
    return share, events


def _power_quota(cfg: SimConfig, index: pd.DatetimeIndex) -> np.ndarray:
    """Grid quota for mill power (kW) per step."""
    both = sum(cfg.mill_rated_kw) * 1.05
    one = max(cfg.mill_rated_kw) * 1.05
    hours = index.hour
    quota = np.full(len(index), both)
    quota[np.isin(hours, cfg.quota_one_mill_hours)] = one
    quota[np.isin(hours, cfg.quota_no_mill_hours)] = 0.0
    return quota


def simulate(
    cfg: SimConfig | None = None,
    fuels: FuelProperties | None = None,
    setpoint_overrides: Optional[Dict[str, Tuple[int, float]]] = None,
) -> SimulationResult:
    """Run the simulator and return process tags, lab results and ground truth.

    Parameters
    ----------
    setpoint_overrides:
        Counterfactual operator moves: ``{lever: (from_step, value)}`` for
        ``SPECIFIC_HEAT_KCAL_KG`` (fired heat per kg clinker) and
        ``ID_FAN_PCT``. Random draws are unchanged, so the run differs from
        the baseline run *only* by the move — which is how tests check that a
        recommendation is actually safe when applied.
    """
    cfg = cfg or SimConfig()
    fuels = fuels or FuelProperties()
    rng = np.random.default_rng(cfg.seed)
    n = cfg.n_steps
    index = pd.date_range(cfg.start, periods=n, freq=f"{cfg.step_minutes}min")
    steps_per_hour = 60 / cfg.step_minutes

    # --- Feed, clinker, burnability -----------------------------------------
    feed = cfg.base_feed_tph + _ar1(rng, n, 0.995, 0.35)
    clinker = feed / cfg.clinker_factor
    burnability = _ar1(rng, n, 0.998, 0.04)  # hidden
    required = cfg.required_specific_heat + 8.0 * burnability / 0.6
    speed = 3.8 * feed / cfg.base_feed_tph + rng.normal(0, 0.01, n)

    # --- Operator actions ----------------------------------------------------
    sp_heat = cfg.operator_specific_heat + _ar1(rng, n, 0.97, 1.2)
    id_fan = np.empty(n)
    level = 75.0
    for k in range(n):
        if k % int(3 * steps_per_hour) == 0:  # operators touch the fan every ~3 h
            level = float(np.clip(level + rng.normal(0, 2.2), 68.0, 84.0))
        id_fan[k] = level + rng.normal(0, 0.25)

    overrides = setpoint_overrides or {}
    unknown = set(overrides) - {T.SPECIFIC_HEAT_KCAL_KG, T.ID_FAN_PCT}
    if unknown:
        raise ValueError(f"Unsupported setpoint override(s): {sorted(unknown)}.")
    if T.SPECIFIC_HEAT_KCAL_KG in overrides:
        k0, value = overrides[T.SPECIFIC_HEAT_KCAL_KG]
        sp_heat[int(k0) :] = float(value)
    if T.ID_FAN_PCT in overrides:
        k0, value = overrides[T.ID_FAN_PCT]
        id_fan[int(k0) :] = float(value)

    share, events = _gas_share_profile(cfg, index)
    transfer = np.abs(np.gradient(share)) * steps_per_hour  # share change per hour
    heat_gcalh = sp_heat * clinker * 1e3 / 1e6
    gas = heat_gcalh * share * 1e6 / fuels.gas_lhv_kcal_per_nm3
    mazut = heat_gcalh * (1 - share) * 1e6 / fuels.mazut_lhv_kcal_per_kg
    gas = np.clip(gas * (1 + rng.normal(0, 0.004, n)), 0, None)
    mazut = np.clip(mazut * (1 + rng.normal(0, 0.004, n)), 0, None)

    # --- Combustion ----------------------------------------------------------
    o2 = (
        2.2
        + 0.12 * (id_fan - 75.0)
        - 0.15 * (1 - share)
        - 0.9 * np.clip(transfer, 0, 2)
        + _ar1(rng, n, 0.8, 0.05)
    )
    o2 = np.clip(o2, 0.3, 6.0)
    co = 80 + 2500 * np.exp(-(o2 - 0.6) / 0.25) + 150 * np.clip(transfer, 0, 2) + rng.normal(0, 15, n)
    co = np.clip(co, 0, 20000)

    actual_heat = heat_gcalh * 1e6 / (clinker * 1e3)  # kcal/kg fired
    efficiency = 1.0 - 0.012 * (o2 - 2.0) - co / 1e5
    effective = actual_heat * efficiency

    # --- Burning zone (dead time + first-order lag) ---------------------------
    dead = cfg.minutes_to_steps(20)
    tau = 30 / cfg.step_minutes
    alpha = 1.0 / (1.0 + tau)
    bz_ss = 1450 + 1.1 * (effective - required) - 40 * (speed - 3.8) + 6.0 * (1 - share)
    bzt = np.empty(n)
    bzt[0] = bz_ss[0]
    for k in range(1, n):
        driver = bz_ss[max(0, k - dead)]
        bzt[k] = bzt[k - 1] + alpha * (driver - bzt[k - 1])
    bzt = bzt + _ar1(rng, n, 0.7, 1.5)

    # --- Free lime: residence delay, then lab sampling -------------------------
    residence = cfg.minutes_to_steps(30)
    bzt_lag = np.concatenate([np.full(residence, bzt[0]), bzt[:-residence]])
    free_lime = np.clip(1.3 - 0.025 * (bzt_lag - 1450) + rng.normal(0, 0.06, n), 0.3, 4.0)

    lab_every = int(2 * steps_per_hour)
    transport = cfg.minutes_to_steps(45)
    lab_rows = []
    for j, k in enumerate(range(transport + lab_every, n, lab_every)):
        sampled_at = k - transport  # clinker made then, analysed at k
        lab_rows.append(
            {
                "timestamp": index[k],
                "parameter": T.FREE_LIME_PCT,
                "value": round(float(free_lime[sampled_at] + rng.normal(0, 0.05)), 3),
                "sample_id": f"FL-{j:05d}",
            }
        )
    lab = pd.DataFrame(lab_rows, columns=["timestamp", "parameter", "value", "sample_id"])

    nox = np.clip(550 + 4.0 * (bzt - 1450) + 60 * (o2 - 2.2) + rng.normal(0, 12, n), 50, None)
    sec_air = 1000 + 0.6 * (bzt - 1450) + rng.normal(0, 4, n)

    wide = pd.DataFrame(
        {
            T.KILN_FEED_TPH: feed,
            T.CLINKER_TPH: clinker,
            T.GAS_FLOW_NM3H: gas,
            T.MAZUT_FLOW_KGH: mazut,
            T.KILN_SPEED_RPM: speed,
            T.ID_FAN_PCT: id_fan,
            T.BZ_TEMP_C: bzt,
            T.KILN_INLET_O2_PCT: o2,
            T.KILN_INLET_CO_PPM: co,
            T.NOX_MGNM3: nox,
            T.SEC_AIR_TEMP_C: sec_air,
        },
        index=index,
    )

    # --- Cement mills under a power quota --------------------------------------
    quota = _power_quota(cfg, index)
    hours = index.hour + index.minute / 60.0
    dispatch = np.where((hours >= 7) & (hours < 19), 230.0, 50.0)  # t/h trucks
    silo = cfg.silo_initial_t
    silo_level = np.empty(n)
    dt_h = cfg.step_minutes / 60.0
    mill_kw = np.zeros((2, n))
    mill_tph = np.zeros((2, n))
    # Naive operator policy (the baseline the scheduler must beat): each mill
    # has start/stop silo-fill thresholds (hysteresis); the quota is respected
    # by shedding mills in order, regardless of the tariff.
    start_below = (0.75, 0.60)
    stop_above = (0.95, 0.90)
    wants = [False, False]
    for k in range(n):
        fill = silo / cfg.silo_capacity_t
        for m in range(2):
            if fill < start_below[m]:
                wants[m] = True
            elif fill > stop_above[m]:
                wants[m] = False
        budget = quota[k]
        produced = 0.0
        for m in range(2):
            if wants[m] and budget >= cfg.mill_rated_kw[m]:
                budget -= cfg.mill_rated_kw[m]
                mill_kw[m, k] = cfg.mill_rated_kw[m] * (1 + rng.normal(0, 0.01))
                mill_tph[m, k] = cfg.mill_rated_tph[m] * (1 + rng.normal(0, 0.01))
                produced += mill_tph[m, k]
        silo = float(np.clip(silo + (produced - dispatch[k]) * dt_h, 0.0, cfg.silo_capacity_t))
        silo_level[k] = silo
    for m in range(2):
        wide[T.mill_tag(T.MILL_KW_TEMPLATE, m + 1)] = mill_kw[m]
        wide[T.mill_tag(T.MILL_TPH_TEMPLATE, m + 1)] = mill_tph[m]
    wide[T.CEMENT_SILO_T] = silo_level

    # --- Mill condition (bearing fault on mill 2) -------------------------------
    days = (index - index[0]) / pd.Timedelta(days=1)
    for m in range(2):
        base_vib = 2.4 + 0.3 * m
        growth = np.zeros(n)
        if m == 1:
            growth = np.clip(days - cfg.bearing_fault_start_day, 0, None) * 1.0
        vib = base_vib + growth + _ar1(rng, n, 0.6, 0.12)
        bearing = 64 + 3 * m + 4.0 * growth + _ar1(rng, n, 0.9, 0.3)
        is_on = wide[T.mill_tag(T.MILL_KW_TEMPLATE, m + 1)].to_numpy() > 0
        wide[T.mill_tag(T.MILL_VIBRATION_TEMPLATE, m + 1)] = np.where(is_on, vib, 0.3)
        wide[T.mill_tag(T.MILL_BEARING_TEMPLATE, m + 1)] = np.where(is_on, bearing, bearing - 15)

    # --- Kiln shell scanner ------------------------------------------------------
    positions = np.arange(cfg.shell_positions)
    profile = 230 + 90 * np.exp(-((positions - 3.0) ** 2) / 8.0)  # hottest near burning zone
    hot_growth = np.clip(days - cfg.hot_spot_start_day, 0, None) * 35.0
    for p in positions:
        extra = hot_growth if p == cfg.hot_spot_position else 0.0
        wide[T.shell_tag(int(p))] = profile[p] + 0.15 * (bzt - 1450) + extra + _ar1(rng, n, 0.9, 2.0)

    wide = add_derived_kpis(wide, fuels, mill_ids=(1, 2))
    wide.index.name = "timestamp"
    return SimulationResult(
        wide=wide,
        lab=lab,
        fuel_events=events,
        power_quota_kw=pd.Series(quota, index=index, name="MILL_POWER_QUOTA_KW"),
        true_free_lime=pd.Series(free_lime, index=index, name="TRUE_FREE_LIME_PCT"),
        config=cfg,
    )


def ground_truth_effects() -> Dict[str, Dict[str, int]]:
    """Signs of the true steady-state effects built into the simulator.

    Used by tests to check that a learnt model agrees with the physics.
    ``+1`` means increasing the input increases the output.
    """
    return {
        T.ID_FAN_PCT: {T.KILN_INLET_O2_PCT: +1, T.KILN_INLET_CO_PPM: -1, T.BZ_TEMP_C: -1},
        T.GAS_FLOW_NM3H: {T.BZ_TEMP_C: +1, T.SPECIFIC_HEAT_KCAL_KG: +1},
        T.MAZUT_FLOW_KGH: {T.BZ_TEMP_C: +1, T.SPECIFIC_HEAT_KCAL_KG: +1},
    }
