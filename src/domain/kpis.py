"""Business KPIs computed from raw process tags.

The copilot is sold on numbers a plant manager already tracks:

* **specific heat consumption** (kcal per kg clinker) of the kiln,
* **specific power** (kWh per tonne) of each cement mill,
* the **gas share** of the kiln heat input, which drives cost when gas is
  curtailed and the burner runs on mazut.

These are computed here, from raw tags, so they can be stored as derived
signals and used as model targets instead of a sensor temperature.

All formulas are unit-explicit; fuel heating values come from
:class:`FuelProperties` (``config/plant.yaml``), never from constants buried
in code.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import yaml

from src.domain import tags as T

#: Below this clinker rate the kiln is considered stopped and specific
#: consumption is undefined (it would divide by ~0 during start-up).
MIN_CLINKER_TPH_FOR_KPI = 5.0

#: Below this throughput a mill is considered stopped.
MIN_MILL_TPH_FOR_KPI = 1.0


@dataclass(frozen=True)
class FuelProperties:
    """Lower heating values and densities used for heat-input accounting.

    Attributes
    ----------
    gas_lhv_kcal_per_nm3:
        Natural-gas lower heating value. Iranian pipeline gas is typically
        8,100-8,600 kcal/Nm3; use the plant's gas certificate.
    mazut_lhv_kcal_per_kg:
        Heavy fuel oil lower heating value (typically ~9,600 kcal/kg).
    mazut_density_kg_per_l:
        Mazut density, used to convert mass to purchased litres.
    """

    gas_lhv_kcal_per_nm3: float = 8200.0
    mazut_lhv_kcal_per_kg: float = 9600.0
    mazut_density_kg_per_l: float = 0.95

    def __post_init__(self) -> None:
        for name in ("gas_lhv_kcal_per_nm3", "mazut_lhv_kcal_per_kg", "mazut_density_kg_per_l"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive.")


@dataclass(frozen=True)
class PlantConfig:
    """Plant-level constants loaded from ``config/plant.yaml``."""

    fuels: FuelProperties
    clinker_factor: float  # t raw meal per t clinker
    mill_ids: tuple[int, ...]
    shell_positions: int
    currency: str
    gas_price_per_nm3: float
    mazut_price_per_l: float
    mazut_transport_per_l: float
    grid_power_price_per_kwh: float
    free_market_power_price_per_kwh: float


def load_plant_config(path: str | Path = "config/plant.yaml") -> PlantConfig:
    """Parse ``config/plant.yaml`` into a :class:`PlantConfig`."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a mapping.")
    fuels = FuelProperties(**raw.get("fuels", {}))
    prices = raw.get("prices", {})
    return PlantConfig(
        fuels=fuels,
        clinker_factor=float(raw.get("clinker_factor", 1.58)),
        mill_ids=tuple(int(m) for m in raw.get("mill_ids", [1, 2])),
        shell_positions=int(raw.get("shell_positions", 10)),
        currency=str(prices.get("currency", "IRT")),
        gas_price_per_nm3=float(prices.get("gas_per_nm3", 0.0)),
        mazut_price_per_l=float(prices.get("mazut_per_l", 0.0)),
        mazut_transport_per_l=float(prices.get("mazut_transport_per_l", 0.0)),
        grid_power_price_per_kwh=float(prices.get("grid_power_per_kwh", 0.0)),
        free_market_power_price_per_kwh=float(prices.get("free_market_power_per_kwh", 0.0)),
    )


# --------------------------------------------------------------------------- #
# Kiln heat KPIs
# --------------------------------------------------------------------------- #
def heat_input_gcal_per_h(
    gas_nm3_per_h: pd.Series | np.ndarray | float,
    mazut_kg_per_h: pd.Series | np.ndarray | float,
    fuels: FuelProperties,
) -> pd.Series | np.ndarray | float:
    """Total burner heat input in Gcal/h."""
    kcal_per_h = (
        np.asarray(gas_nm3_per_h, dtype=float) * fuels.gas_lhv_kcal_per_nm3
        + np.asarray(mazut_kg_per_h, dtype=float) * fuels.mazut_lhv_kcal_per_kg
    )
    result = kcal_per_h / 1e6
    if isinstance(gas_nm3_per_h, pd.Series):
        return pd.Series(result, index=gas_nm3_per_h.index, name=T.HEAT_INPUT_GCALH)
    return result


def specific_heat_kcal_per_kg(
    heat_gcal_per_h: pd.Series | np.ndarray,
    clinker_tph: pd.Series | np.ndarray,
) -> pd.Series | np.ndarray:
    """Specific heat consumption (kcal/kg clinker); NaN while the kiln is down.

    One Gcal is 1e6 kcal and one tonne is 1e3 kg, so
    ``kcal/kg = heat[Gcal/h] * 1e6 / (clinker[t/h] * 1e3)``. A 3,000 t/d
    line (125 t/h) burning 93.75 Gcal/h is at 750 kcal/kg.
    """
    heat = np.asarray(heat_gcal_per_h, dtype=float)
    clinker = np.asarray(clinker_tph, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        value = heat * 1e6 / (clinker * 1e3)
    value = np.where(clinker >= MIN_CLINKER_TPH_FOR_KPI, value, np.nan)
    if isinstance(heat_gcal_per_h, pd.Series):
        return pd.Series(value, index=heat_gcal_per_h.index, name=T.SPECIFIC_HEAT_KCAL_KG)
    return value


def gas_heat_share(
    gas_nm3_per_h: pd.Series | np.ndarray,
    mazut_kg_per_h: pd.Series | np.ndarray,
    fuels: FuelProperties,
) -> pd.Series | np.ndarray:
    """Fraction (0-1) of heat input that comes from natural gas."""
    gas_heat = np.asarray(gas_nm3_per_h, dtype=float) * fuels.gas_lhv_kcal_per_nm3
    mazut_heat = np.asarray(mazut_kg_per_h, dtype=float) * fuels.mazut_lhv_kcal_per_kg
    total = gas_heat + mazut_heat
    with np.errstate(divide="ignore", invalid="ignore"):
        share = np.where(total > 0, gas_heat / total, np.nan)
    if isinstance(gas_nm3_per_h, pd.Series):
        return pd.Series(share, index=gas_nm3_per_h.index, name=T.GAS_HEAT_SHARE)
    return share


def mill_specific_power_kwh_per_t(
    power_kw: pd.Series | np.ndarray,
    throughput_tph: pd.Series | np.ndarray,
) -> pd.Series | np.ndarray:
    """Mill specific power (kWh/t); NaN while the mill is stopped."""
    kw = np.asarray(power_kw, dtype=float)
    tph = np.asarray(throughput_tph, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        value = np.where(tph >= MIN_MILL_TPH_FOR_KPI, kw / tph, np.nan)
    if isinstance(power_kw, pd.Series):
        return pd.Series(value, index=power_kw.index)
    return value


# --------------------------------------------------------------------------- #
# Frame-level helpers
# --------------------------------------------------------------------------- #
def add_derived_kpis(
    wide: pd.DataFrame,
    fuels: FuelProperties,
    mill_ids: Iterable[int] = (1, 2),
) -> pd.DataFrame:
    """Return ``wide`` plus every KPI column whose raw inputs are present.

    Missing inputs are not an error: a plant that only exports kiln tags gets
    kiln KPIs and no mill KPIs.
    """
    out = wide.copy()
    if T.GAS_FLOW_NM3H in out or T.MAZUT_FLOW_KGH in out:
        gas = out.get(T.GAS_FLOW_NM3H, pd.Series(0.0, index=out.index))
        mazut = out.get(T.MAZUT_FLOW_KGH, pd.Series(0.0, index=out.index))
        gas = gas.fillna(0.0)
        mazut = mazut.fillna(0.0)
        out[T.HEAT_INPUT_GCALH] = heat_input_gcal_per_h(gas, mazut, fuels)
        out[T.GAS_HEAT_SHARE] = gas_heat_share(gas, mazut, fuels)
        if T.CLINKER_TPH in out:
            out[T.SPECIFIC_HEAT_KCAL_KG] = specific_heat_kcal_per_kg(
                out[T.HEAT_INPUT_GCALH], out[T.CLINKER_TPH]
            )
    for n in mill_ids:
        kw = T.mill_tag(T.MILL_KW_TEMPLATE, n)
        tph = T.mill_tag(T.MILL_TPH_TEMPLATE, n)
        if kw in out and tph in out:
            out[T.mill_tag(T.MILL_SPECIFIC_KWH_T_TEMPLATE, n)] = mill_specific_power_kwh_per_t(
                out[kw], out[tph]
            )
    return out


def add_derived_kpis_long(
    signals: pd.DataFrame,
    fuels: FuelProperties,
    mill_ids: Iterable[int] = (1, 2),
    only: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Long-format variant: append derived KPI rows to a ``timestamp/tag/value`` frame.

    Parameters
    ----------
    only:
        Optional whitelist of derived tags to append (default: all computable).
    """
    from src.ml.features import pivot_signals

    if signals.empty:
        return signals.copy()
    wide = pivot_signals(signals)
    enriched = add_derived_kpis(wide, fuels, mill_ids)
    derived_cols = [c for c in enriched.columns if c not in wide.columns]
    if only is not None:
        wanted = set(only)
        derived_cols = [c for c in derived_cols if c in wanted]
    if not derived_cols:
        return signals.copy()
    derived = (
        enriched[derived_cols]
        .rename_axis("timestamp")
        .reset_index()
        .melt(id_vars="timestamp", var_name="tag_name", value_name="value")
        .dropna(subset=["value"])
    )
    base = signals[["timestamp", "tag_name", "value"]].copy()
    base["timestamp"] = pd.to_datetime(base["timestamp"])
    return pd.concat([base, derived], ignore_index=True)
