"""Condition monitoring: kiln-shell hot spots and cement-mill bearings.

Both problems announce themselves days ahead in data the plant already
logs, and both are expensive when missed: a shell hot spot that becomes a
red spot stops the kiln for a refractory repair; a failing mill bearing
stops grinding at the worst moment of the power quota.

Method (per signal)
-------------------
* **Reference**: median and MAD over a reference window when the asset was
  healthy (default: the first part of the history), so the comparison is
  against *this* asset, not a textbook value.
* **Robust z-score** of the recent level against that reference.
* **Trend**: least-squares slope of hourly means over the recent window,
  in units per day, and the estimated hours until the alarm limit.
* **Severity**: ``alarm`` if the alarm limit is crossed or will be within
  ``eta_alarm_hours``; ``warning`` on the warning limit, a large robust
  z-score or a rising trend that reaches the alarm limit within a week.

Defaults: vibration zone limits follow ISO 10816-3 style velocity bands
(4.5 / 7.1 mm/s RMS); shell limits are typical refractory guidance. Both
must be replaced with the plant's own limits (``config/maintenance.yaml``).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from src.domain import tags as T


@dataclass(frozen=True)
class Limits:
    warning: float
    alarm: float
    z_warning: float = 4.0
    eta_alarm_hours: float = 48.0
    eta_warning_hours: float = 168.0


DEFAULT_LIMITS: Dict[str, Limits] = {
    "shell": Limits(warning=350.0, alarm=400.0),
    "vibration": Limits(warning=4.5, alarm=7.1),
    "bearing": Limits(warning=80.0, alarm=90.0),
}


@dataclass(frozen=True)
class Finding:
    asset: str
    signal: str
    severity: str  # "ok" | "warning" | "alarm"
    value: float
    reference: float
    robust_z: float
    slope_per_day: float
    hours_to_alarm: Optional[float]
    message: str
    message_fa: str

    def to_dict(self) -> dict:
        return asdict(self)


def _hourly(series: pd.Series) -> pd.Series:
    return series.resample("1h").mean().dropna()


def _slope_per_day(hourly: pd.Series) -> float:
    if len(hourly) < 3:
        return 0.0
    x = (hourly.index - hourly.index[0]) / pd.Timedelta(days=1)
    x = np.asarray(x, dtype=float)
    y = hourly.to_numpy(dtype=float)
    x = x - x.mean()
    denom = float((x * x).sum())
    return float((x * (y - y.mean())).sum() / denom) if denom > 0 else 0.0


def assess_signal(
    series: pd.Series,
    asset: str,
    signal: str,
    limits: Limits,
    reference: Optional[pd.Series] = None,
    recent_hours: float = 48.0,
    level_hours: float = 2.0,
    asset_fa: Optional[str] = None,
    signal_fa: Optional[str] = None,
) -> Finding:
    """Assess one signal; ``reference`` defaults to the first half of ``series``."""
    series = series.dropna()
    if series.empty:
        raise ValueError(f"{asset}/{signal}: no data.")
    end = series.index[-1]
    ref = reference.dropna() if reference is not None else series.iloc[: max(1, len(series) // 2)]
    median = float(ref.median())
    mad = float((ref - median).abs().median()) * 1.4826
    level = float(series.loc[end - pd.Timedelta(hours=level_hours) :].mean())
    z = (level - median) / mad if mad > 0 else 0.0
    slope = _slope_per_day(_hourly(series.loc[end - pd.Timedelta(hours=recent_hours) :]))
    eta = None
    if slope > 0 and level < limits.alarm:
        eta = (limits.alarm - level) / slope * 24.0

    severity = "ok"
    if level >= limits.alarm or (eta is not None and eta <= limits.eta_alarm_hours):
        severity = "alarm"
    elif (
        level >= limits.warning
        or z >= limits.z_warning
        or (eta is not None and eta <= limits.eta_warning_hours)
    ):
        severity = "warning"

    eta_txt = f", alarm limit in ~{eta:.0f} h at the current trend" if eta is not None else ""
    eta_fa = f"، با روند فعلی حدود {eta:.0f} ساعت تا حد آلارم" if eta is not None else ""
    return Finding(
        asset=asset,
        signal=signal,
        severity=severity,
        value=level,
        reference=median,
        robust_z=float(z),
        slope_per_day=slope,
        hours_to_alarm=eta,
        message=(
            f"{asset} {signal}: {level:.1f} vs reference {median:.1f} "
            f"(z={z:.1f}, trend {slope:+.2f}/day{eta_txt})."
        ),
        message_fa=(
            f"{asset_fa or asset} – {signal_fa or signal}: {level:.1f} در برابر مقدار مرجع {median:.1f} "
            f"(z={z:.1f}، روند {slope:+.2f} در روز{eta_fa})."
        ),
    )


def assess_shell(
    wide: pd.DataFrame,
    positions: Sequence[int],
    limits: Limits = DEFAULT_LIMITS["shell"],
    reference_end: Optional[pd.Timestamp] = None,
) -> List[Finding]:
    """One finding per shell-scanner position (0 = kiln outlet)."""
    findings = []
    for pos in positions:
        tag = T.shell_tag(pos)
        if tag not in wide:
            continue
        ref = wide[tag].loc[:reference_end] if reference_end is not None else None
        findings.append(
            assess_signal(
                wide[tag],
                "kiln shell",
                f"position {pos}",
                limits,
                ref,
                asset_fa="پوستهٔ کوره",
                signal_fa=f"موقعیت {pos}",
            )
        )
    return findings


def assess_mills(
    wide: pd.DataFrame,
    mill_ids: Sequence[int],
    vibration: Limits = DEFAULT_LIMITS["vibration"],
    bearing: Limits = DEFAULT_LIMITS["bearing"],
    reference_end: Optional[pd.Timestamp] = None,
    running_kw: float = 100.0,
) -> List[Finding]:
    """Vibration and bearing-temperature findings per mill, while running only."""
    findings = []
    for n in mill_ids:
        kw_tag = T.mill_tag(T.MILL_KW_TEMPLATE, n)
        running = wide[kw_tag] > running_kw if kw_tag in wide else pd.Series(True, index=wide.index)
        for template, signal, signal_fa, limits in (
            (T.MILL_VIBRATION_TEMPLATE, "vibration (mm/s)", "ارتعاش (mm/s)", vibration),
            (T.MILL_BEARING_TEMPLATE, "bearing temperature (°C)", "دمای یاتاقان (°C)", bearing),
        ):
            tag = T.mill_tag(template, n)
            if tag not in wide:
                continue
            series = wide[tag].where(running)
            ref = series.loc[:reference_end] if reference_end is not None else None
            if series.dropna().empty:
                continue
            findings.append(
                assess_signal(
                    series,
                    f"cement mill {n}",
                    signal,
                    limits,
                    ref,
                    asset_fa=f"آسیاب سیمان {n}",
                    signal_fa=signal_fa,
                )
            )
    return findings


def worst_first(findings: Sequence[Finding]) -> List[Finding]:
    rank = {"alarm": 0, "warning": 1, "ok": 2}
    return sorted(findings, key=lambda f: (rank[f.severity], -(f.robust_z)))
