"""Gas ↔ mazut fuel-transfer advisor.

When gas is curtailed the burner is moved to mazut (and back when gas
returns). Done fast, the flame and the oxygen balance are upset: O2 dips,
CO spikes, the burning zone swings and free lime follows hours later. Done
too slowly, the plant over-fires or loses output. The right speed is plant
specific, and the plant's own history contains the answer.

This module

1. **detects** past transfers from the gas share of heat input,
2. **measures** the upset each one caused (burning-zone swing, CO minutes
   above limit, O2 minimum, excess fuel),
3. **learns** which ramp duration upset the kiln least, and
4. **plans** the next transfer step by step: burner flows that keep heat
   input constant, an ID-fan bias that offsets the expected O2 dip, and the
   checks the shift must do.

It never writes to the DCS.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from src.domain import tags as T
from src.domain.kpis import FuelProperties, add_derived_kpis

GAS_TO_MAZUT = "gas_to_mazut"
MAZUT_TO_GAS = "mazut_to_gas"


@dataclass(frozen=True)
class Transfer:
    """One detected fuel transfer."""

    start: pd.Timestamp
    end: pd.Timestamp
    direction: str

    @property
    def ramp_minutes(self) -> float:
        return (self.end - self.start) / pd.Timedelta(minutes=1)


@dataclass(frozen=True)
class TransferImpact:
    """How much one transfer upset the kiln."""

    transfer: Transfer
    bz_temp_std: float
    bz_temp_drop: float  # pre-transfer mean minus minimum during the window
    co_minutes_above_limit: float
    o2_min: float
    excess_heat_pct: float  # specific heat during the window vs before, %

    def score(self, co_limit_minutes_weight: float = 0.05) -> float:
        """Single instability score (lower is better) for ranking ramps."""
        return self.bz_temp_std + 0.5 * max(self.bz_temp_drop, 0.0) + co_limit_minutes_weight * self.co_minutes_above_limit

    def to_dict(self) -> dict:
        out = asdict(self)
        out["transfer"] = {
            "start": self.transfer.start.isoformat(),
            "end": self.transfer.end.isoformat(),
            "direction": self.transfer.direction,
            "ramp_minutes": self.transfer.ramp_minutes,
        }
        out["score"] = self.score()
        return out


@dataclass
class RampPolicy:
    """Recommended ramp per direction and the evidence behind it."""

    ramp_minutes: Dict[str, float]
    evidence: Dict[str, List[Dict[str, float]]]
    source: Dict[str, str]  # "history" | "default"


@dataclass
class TransferStep:
    """One step of a transfer plan."""

    minute: float
    gas_share: float
    gas_nm3h: float
    mazut_kgh: float
    heat_gcalh: float
    id_fan_pct: Optional[float]


@dataclass
class TransferPlan:
    direction: str
    ramp_minutes: float
    steps: List[TransferStep]
    checklist: List[str] = field(default_factory=list)
    checklist_fa: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Detection and impact
# --------------------------------------------------------------------------- #
def _ensure_share(wide: pd.DataFrame, fuels: Optional[FuelProperties]) -> pd.DataFrame:
    if T.GAS_HEAT_SHARE not in wide:
        wide = add_derived_kpis(wide, fuels or FuelProperties())
    if T.GAS_HEAT_SHARE not in wide:
        raise ValueError("gas/mazut flow tags are required to detect transfers.")
    return wide


def detect_transfers(
    wide: pd.DataFrame,
    high: float = 0.9,
    low: float = 0.1,
    fuels: Optional[FuelProperties] = None,
) -> List[Transfer]:
    """Transfers between mostly-gas (share >= ``high``) and mostly-mazut (<= ``low``).

    A transfer starts at the last sample in the old regime and ends at the
    first sample in the new one, so its duration is the ramp the crew used.
    """
    wide = _ensure_share(wide, fuels)
    share = wide[T.GAS_HEAT_SHARE].ffill()
    regime = pd.Series(np.where(share >= high, 1, np.where(share <= low, 0, -1)), index=share.index)
    transfers: List[Transfer] = []
    last_regime, last_time = None, None
    for ts, value in regime.items():
        if value == -1:
            continue
        if last_regime is not None and value != last_regime:
            direction = GAS_TO_MAZUT if last_regime == 1 else MAZUT_TO_GAS
            transfers.append(Transfer(pd.Timestamp(last_time), pd.Timestamp(ts), direction))
        last_regime, last_time = value, ts
    return transfers


def measure_impact(
    wide: pd.DataFrame,
    transfer: Transfer,
    settle_minutes: float = 120.0,
    pre_minutes: float = 120.0,
    co_limit_ppm: float = 600.0,
    fuels: Optional[FuelProperties] = None,
) -> TransferImpact:
    """Upset metrics over the transfer plus a settling window."""
    if T.SPECIFIC_HEAT_KCAL_KG not in wide:
        wide = add_derived_kpis(wide, fuels or FuelProperties())
    pre = wide.loc[transfer.start - pd.Timedelta(minutes=pre_minutes) : transfer.start]
    window = wide.loc[transfer.start : transfer.end + pd.Timedelta(minutes=settle_minutes)]
    if window.empty or pre.empty:
        raise ValueError("not enough data around the transfer.")
    step_min = float(pd.Series(wide.index).diff().median() / pd.Timedelta(minutes=1))
    bzt = window[T.BZ_TEMP_C]
    co = window.get(T.KILN_INLET_CO_PPM, pd.Series(dtype=float))
    o2 = window.get(T.KILN_INLET_O2_PCT, pd.Series(dtype=float))
    sh_pre = float(pre[T.SPECIFIC_HEAT_KCAL_KG].mean())
    sh_win = float(window[T.SPECIFIC_HEAT_KCAL_KG].mean())
    return TransferImpact(
        transfer=transfer,
        bz_temp_std=float(bzt.std()),
        bz_temp_drop=float(pre[T.BZ_TEMP_C].mean() - bzt.min()),
        co_minutes_above_limit=float((co > co_limit_ppm).sum() * step_min),
        o2_min=float(o2.min()) if not o2.empty else float("nan"),
        excess_heat_pct=100.0 * (sh_win - sh_pre) / sh_pre if sh_pre else 0.0,
    )


def learn_ramp_policy(
    impacts: Sequence[TransferImpact],
    default_ramp_minutes: float = 90.0,
    bucket_minutes: float = 30.0,
    min_events_per_bucket: int = 1,
) -> RampPolicy:
    """Ramp duration with the lowest mean instability score, per direction.

    Ramps are grouped in ``bucket_minutes`` buckets. With no history for a
    direction the engineering default is returned and flagged as such.
    """
    ramps: Dict[str, float] = {}
    evidence: Dict[str, List[Dict[str, float]]] = {}
    source: Dict[str, str] = {}
    for direction in (GAS_TO_MAZUT, MAZUT_TO_GAS):
        rows = [i for i in impacts if i.transfer.direction == direction]
        buckets: Dict[float, List[float]] = {}
        for impact in rows:
            key = bucket_minutes * round(impact.transfer.ramp_minutes / bucket_minutes)
            buckets.setdefault(key, []).append(impact.score())
        table = [
            {"ramp_minutes": k, "mean_score": float(np.mean(v)), "n_events": float(len(v))}
            for k, v in sorted(buckets.items())
            if len(v) >= min_events_per_bucket
        ]
        evidence[direction] = table
        if table:
            best = min(table, key=lambda r: (r["mean_score"], -r["ramp_minutes"]))
            ramps[direction] = best["ramp_minutes"]
            source[direction] = "history"
        else:
            ramps[direction] = default_ramp_minutes
            source[direction] = "default"
    return RampPolicy(ramps, evidence, source)


# --------------------------------------------------------------------------- #
# Planning
# --------------------------------------------------------------------------- #
def estimate_o2_dip_per_share_rate(wide: pd.DataFrame, fuels: Optional[FuelProperties] = None) -> float:
    """O2 change (%) per unit/h of gas-share change rate, from history.

    A negative value means O2 dips while the share is moving, in either
    direction. Estimated by least squares of O2 deviation from its 2-hour
    rolling median on |d share / dt|.
    """
    wide = _ensure_share(wide, fuels)
    if T.KILN_INLET_O2_PCT not in wide:
        return 0.0
    step_h = float(pd.Series(wide.index).diff().median() / pd.Timedelta(hours=1))
    rate = wide[T.GAS_HEAT_SHARE].diff().abs() / step_h
    window = max(3, int(round(2.0 / step_h)))
    dev = wide[T.KILN_INLET_O2_PCT] - wide[T.KILN_INLET_O2_PCT].rolling(window, center=True, min_periods=1).median()
    data = pd.concat([rate.rename("r"), dev.rename("d")], axis=1).dropna()
    data = data[data["r"] > 0]
    if len(data) < 5:
        return 0.0
    r = data["r"].to_numpy()
    return float(np.dot(r, data["d"].to_numpy()) / np.dot(r, r))


def plan_transfer(
    current: pd.DataFrame,
    direction: str,
    ramp_minutes: float,
    n_steps: int = 6,
    o2_dip_per_rate: float = 0.0,
    o2_per_fan_pct: Optional[float] = None,
    fan_limits: tuple[float, float] = (65.0, 88.0),
    fuels: Optional[FuelProperties] = None,
) -> TransferPlan:
    """Step-by-step burner flows that keep heat input constant through a transfer.

    Parameters
    ----------
    current:
        Recent wide frame; the last rows define heat input and fan speed.
    o2_dip_per_rate:
        From :func:`estimate_o2_dip_per_share_rate`.
    o2_per_fan_pct:
        Plant gain of O2 per % ID-fan speed (from a step test). With it, the
        plan adds the fan bias that offsets the expected O2 dip.
    """
    if direction not in (GAS_TO_MAZUT, MAZUT_TO_GAS):
        raise ValueError(f"direction must be {GAS_TO_MAZUT!r} or {MAZUT_TO_GAS!r}.")
    if ramp_minutes <= 0 or n_steps < 1:
        raise ValueError("ramp_minutes and n_steps must be positive.")
    fuels = fuels or FuelProperties()
    current = _ensure_share(current, fuels)
    tail = current.tail(3)
    heat = float(tail[T.HEAT_INPUT_GCALH].mean())
    fan0 = float(tail[T.ID_FAN_PCT].mean()) if T.ID_FAN_PCT in tail else None
    start_share = 1.0 if direction == GAS_TO_MAZUT else 0.0
    end_share = 1.0 - start_share
    rate = abs(end_share - start_share) / (ramp_minutes / 60.0)  # share per hour
    fan_bias = 0.0
    if o2_per_fan_pct and fan0 is not None and o2_dip_per_rate < 0:
        fan_bias = -o2_dip_per_rate * rate / o2_per_fan_pct

    steps: List[TransferStep] = []
    for i in range(n_steps + 1):
        frac = i / n_steps
        share = start_share + (end_share - start_share) * frac
        moving = 0 < i < n_steps
        fan = None
        if fan0 is not None:
            fan = float(np.clip(fan0 + (fan_bias if moving else 0.0), *fan_limits))
        steps.append(
            TransferStep(
                minute=ramp_minutes * frac,
                gas_share=share,
                gas_nm3h=heat * share * 1e6 / fuels.gas_lhv_kcal_per_nm3,
                mazut_kgh=heat * (1 - share) * 1e6 / fuels.mazut_lhv_kcal_per_kg,
                heat_gcalh=heat,
                id_fan_pct=fan,
            )
        )
    checklist = [
        "Confirm mazut is at the burner vendor's atomizing temperature/viscosity before the first step.",
        "Confirm atomizing air/steam and burner gun are in service.",
        "Hold each step until O2 and CO are stable; stop and hold if CO rises above the limit.",
        "Inform the lab: increase free-lime sampling for the next 4 hours.",
    ]
    checklist_fa = [
        "پیش از گام اول، دما/ویسکوزیتهٔ مازوت را مطابق مشخصات سازندهٔ مشعل برای اتمیزه‌شدن تأیید کنید.",
        "در مدار بودن هوای/بخار اتمیزه‌کننده و لنس مشعل را تأیید کنید.",
        "هر گام را تا پایدار شدن O2 و CO نگه دارید؛ اگر CO از حد مجاز بالا رفت، توقف کنید.",
        "به آزمایشگاه اطلاع دهید: نمونه‌گیری آهک آزاد را برای ۴ ساعت آینده بیشتر کنند.",
    ]
    if direction == MAZUT_TO_GAS:
        checklist[0] = "Confirm gas pressure at the burner train is stable before the first step."
        checklist_fa[0] = "پیش از گام اول، پایداری فشار گاز در خط مشعل را تأیید کنید."
    return TransferPlan(direction, float(ramp_minutes), steps, checklist, checklist_fa)
