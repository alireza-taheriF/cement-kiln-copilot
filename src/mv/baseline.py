"""Measurement & Verification (M&V) of energy savings.

A plant pays for savings only if both sides agree how they were measured. This
module implements the standard approach (IPMVP Option C style, statistics per
ASHRAE Guideline 14):

1. **Baseline period** (before the copilot): fit energy use against its
   drivers, e.g. daily fuel heat against clinker production and gas share.
2. **Check the fit** against ASHRAE Guideline 14 thresholds
   (CV(RMSE), NMBE). A baseline that fails them must not be used to bill.
3. **Reporting period** (with the copilot): predict what energy *would have
   been* under actual production and fuel mix (the *adjusted baseline*).
4. **Savings** = adjusted baseline - actual, with the Guideline 14
   fractional-savings uncertainty, then valued in money.

The method, the drivers, the thresholds and the prices must be written into
the contract before the PoC starts; :class:`MVReport` is designed to be
attached to the monthly invoice.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from src.domain import tags as T

#: ASHRAE Guideline 14 calibration thresholds (percent) by data granularity.
#: Monthly: CV(RMSE) <= 15, |NMBE| <= 5. Daily/hourly: CV(RMSE) <= 30, |NMBE| <= 10.
ASHRAE14_THRESHOLDS: Dict[str, tuple[float, float]] = {
    "monthly": (15.0, 5.0),
    "daily": (30.0, 10.0),
    "hourly": (30.0, 10.0),
}


@dataclass(frozen=True)
class BaselineModel:
    """Ordinary-least-squares baseline ``energy = b0 + sum b_i * driver_i``."""

    energy_column: str
    drivers: tuple[str, ...]
    coefficients: Dict[str, float]  # includes "intercept"
    n_observations: int
    r2: float
    cv_rmse_pct: float
    nmbe_pct: float
    rmse: float
    granularity: str
    p_values: Dict[str, float] = field(default_factory=dict)
    driver_ranges: Dict[str, tuple[float, float]] = field(default_factory=dict)
    residual_autocorrelation: float = 0.0  # lag-1, used for effective sample size
    fit_intercept: bool = True

    @property
    def n_parameters(self) -> int:
        return len(self.drivers) + (1 if self.fit_intercept else 0)

    @property
    def effective_n(self) -> float:
        """Guideline 14 effective sample size ``n (1 - rho) / (1 + rho)``."""
        rho = min(max(self.residual_autocorrelation, 0.0), 0.95)
        return self.n_observations * (1 - rho) / (1 + rho)

    def passes_ashrae14(self) -> bool:
        cv_max, nmbe_max = ASHRAE14_THRESHOLDS[self.granularity]
        return self.cv_rmse_pct <= cv_max and abs(self.nmbe_pct) <= nmbe_max

    def predict(self, frame: pd.DataFrame) -> pd.Series:
        missing = [d for d in self.drivers if d not in frame]
        if missing:
            raise ValueError(f"reporting data is missing driver column(s): {missing}.")
        value = np.full(len(frame), self.coefficients["intercept"], dtype=float)
        for driver in self.drivers:
            value = value + self.coefficients[driver] * frame[driver].to_numpy(dtype=float)
        return pd.Series(value, index=frame.index, name=f"{self.energy_column}_baseline")


def fit_baseline(
    frame: pd.DataFrame,
    energy_column: str,
    drivers: Sequence[str],
    granularity: str = "daily",
    fit_intercept: bool = True,
) -> BaselineModel:
    """Fit the baseline regression and its Guideline 14 statistics.

    Parameters
    ----------
    frame:
        One row per period (day or month) of the baseline period.
    energy_column:
        Energy use per period, e.g. ``heat_gcal`` or ``mill_kwh``.
    drivers:
        Independent variables, e.g. ``["clinker_t", "gas_share"]``.
    granularity:
        ``"daily"``, ``"hourly"`` or ``"monthly"`` (selects the thresholds).
    fit_intercept:
        Set ``False`` when physics says energy is proportional to the driver
        (kiln heat to clinker): an intercept then only adds extrapolation
        error outside the baseline production range.
    """
    if granularity not in ASHRAE14_THRESHOLDS:
        raise ValueError(f"granularity must be one of {list(ASHRAE14_THRESHOLDS)}.")
    cols = [energy_column, *drivers]
    missing = [c for c in cols if c not in frame]
    if missing:
        raise ValueError(f"baseline data is missing column(s): {missing}.")
    data = frame[cols].dropna()
    n = len(data)
    p = len(drivers) + (1 if fit_intercept else 0)
    if n <= p + 1:
        raise ValueError(f"need more than {p + 1} baseline periods, got {n}.")

    y = data[energy_column].to_numpy(dtype=float)
    D = data[list(drivers)].to_numpy(dtype=float)
    X = np.column_stack([np.ones(n), D]) if fit_intercept else D
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    fitted = X @ beta
    resid = y - fitted
    sse = float(resid @ resid)
    dof = n - p
    rmse = float(np.sqrt(sse / dof))
    y_mean = float(y.mean())
    ss_tot = float(((y - y_mean) ** 2).sum())
    r2 = 1.0 - sse / ss_tot if ss_tot > 0 else 0.0
    cv_rmse = 100.0 * rmse / y_mean
    nmbe = 100.0 * float(resid.sum()) / (dof * y_mean)

    # Coefficient p-values (classical OLS standard errors).
    p_values: Dict[str, float] = {}
    try:
        cov = (sse / dof) * np.linalg.inv(X.T @ X)
        se = np.sqrt(np.diag(cov))
        t_stats = beta / se
        names = (["intercept"] if fit_intercept else []) + list(drivers)
        p_values = {
            name: float(2 * stats.t.sf(abs(t), dof)) for name, t in zip(names, t_stats)
        }
    except np.linalg.LinAlgError:
        pass

    slopes = beta[1:] if fit_intercept else beta
    coefficients = {"intercept": float(beta[0]) if fit_intercept else 0.0}
    coefficients.update({d: float(b) for d, b in zip(drivers, slopes)})
    lag1 = float(np.corrcoef(resid[:-1], resid[1:])[0, 1]) if n > 3 and resid.std() > 0 else 0.0
    return BaselineModel(
        energy_column=energy_column,
        drivers=tuple(drivers),
        coefficients=coefficients,
        n_observations=n,
        r2=float(r2),
        cv_rmse_pct=float(cv_rmse),
        nmbe_pct=float(nmbe),
        rmse=rmse,
        granularity=granularity,
        p_values=p_values,
        driver_ranges={d: (float(data[d].min()), float(data[d].max())) for d in drivers},
        residual_autocorrelation=lag1 if np.isfinite(lag1) else 0.0,
        fit_intercept=fit_intercept,
    )


@dataclass
class MVReport:
    """Savings over a reporting period, ready for a monthly statement."""

    energy_column: str
    unit: str
    baseline_valid: bool
    baseline_stats: Dict[str, float]
    n_reporting_periods: int
    adjusted_baseline_total: float
    actual_total: float
    savings: float
    savings_pct: float
    uncertainty: float  # absolute, at the confidence level below
    confidence: float
    savings_significant: bool
    unit_price: Optional[float] = None
    savings_money: Optional[float] = None
    currency: Optional[str] = None
    periods: List[Dict[str, float]] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def savings_report(
    baseline: BaselineModel,
    reporting: pd.DataFrame,
    unit: str,
    confidence: float = 0.9,
    unit_price: Optional[float] = None,
    currency: Optional[str] = None,
) -> MVReport:
    """Avoided energy over the reporting period with Guideline 14 uncertainty.

    The absolute savings uncertainty uses ASHRAE Guideline 14 (Annex B) for
    a regression baseline without autocorrelation correction::

        U = t * 1.26 * CV * E_adj * sqrt((n / n') * (1 + 2/n') / m)

    with ``CV`` the baseline CV(RMSE) as a fraction, ``E_adj`` the total
    adjusted baseline, ``n`` baseline periods, ``n'`` the effective number
    after the residual lag-1 autocorrelation correction, and ``m``
    reporting periods. Savings smaller than ``U`` are not significant.
    """
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1).")
    data = reporting.dropna(subset=[baseline.energy_column, *baseline.drivers])
    m = len(data)
    if m == 0:
        raise ValueError("reporting period has no complete rows.")
    adjusted = baseline.predict(data)
    actual = data[baseline.energy_column].astype(float)
    e_adj = float(adjusted.sum())
    e_act = float(actual.sum())
    saved = e_adj - e_act
    n = baseline.n_observations
    n_eff = max(baseline.effective_n, baseline.n_parameters + 1.0)
    t_crit = float(stats.t.ppf(0.5 + confidence / 2, max(1.0, n_eff - baseline.n_parameters)))
    uncertainty = (
        t_crit
        * 1.26
        * (baseline.cv_rmse_pct / 100.0)
        * e_adj
        * np.sqrt((n / n_eff) * (1 + 2 / n_eff) / m)
    )

    notes: List[str] = []
    valid = baseline.passes_ashrae14()
    if not valid:
        cv_max, nmbe_max = ASHRAE14_THRESHOLDS[baseline.granularity]
        notes.append(
            f"Baseline fails ASHRAE Guideline 14 ({baseline.granularity}: CV(RMSE) <= {cv_max}%, "
            f"|NMBE| <= {nmbe_max}%): CV(RMSE) {baseline.cv_rmse_pct:.1f}%, "
            f"NMBE {baseline.nmbe_pct:.2f}%. Do not invoice on it."
        )
    for driver in baseline.drivers:
        # Extrapolation check: the baseline regression is only trusted inside
        # the driver range it was fitted on.
        lo, hi = float(data[driver].min()), float(data[driver].max())
        b_lo, b_hi = baseline.driver_ranges.get(driver, (lo, hi))
        span = max(b_hi - b_lo, 1e-9)
        if lo < b_lo - 0.1 * span or hi > b_hi + 0.1 * span:
            notes.append(
                f"{driver} in the reporting period ({lo:.3g}–{hi:.3g}) extends beyond the "
                f"baseline range ({b_lo:.3g}–{b_hi:.3g}); the adjusted baseline extrapolates."
            )

    periods = [
        {
            "period": str(idx),
            "adjusted_baseline": float(adjusted.loc[idx]),
            "actual": float(actual.loc[idx]),
            "savings": float(adjusted.loc[idx] - actual.loc[idx]),
        }
        for idx in data.index
    ]
    money = saved * unit_price if unit_price is not None else None
    return MVReport(
        energy_column=baseline.energy_column,
        unit=unit,
        baseline_valid=valid,
        baseline_stats={
            "n": float(n),
            "r2": baseline.r2,
            "cv_rmse_pct": baseline.cv_rmse_pct,
            "nmbe_pct": baseline.nmbe_pct,
            **{f"coef_{k}": v for k, v in baseline.coefficients.items()},
        },
        n_reporting_periods=m,
        adjusted_baseline_total=e_adj,
        actual_total=e_act,
        savings=saved,
        savings_pct=100.0 * saved / e_adj if e_adj else 0.0,
        uncertainty=float(uncertainty),
        confidence=confidence,
        savings_significant=bool(saved > uncertainty and valid),
        unit_price=unit_price,
        savings_money=money,
        currency=currency,
        periods=periods,
        notes=notes,
    )


# --------------------------------------------------------------------------- #
# Aggregation from process signals
# --------------------------------------------------------------------------- #
def daily_kiln_energy(wide: pd.DataFrame, step_minutes: Optional[float] = None) -> pd.DataFrame:
    """Daily kiln heat (Gcal), clinker (t) and mean gas share from a wide frame.

    Requires ``HEAT_INPUT_GCALH`` and ``CLINKER_TPH`` (see
    :func:`src.domain.kpis.add_derived_kpis`). Days with less than 90 % of
    their samples are dropped: a partial day biases the regression.
    """
    for tag in (T.HEAT_INPUT_GCALH, T.CLINKER_TPH):
        if tag not in wide:
            raise ValueError(f"wide frame is missing {tag}.")
    index = pd.DatetimeIndex(wide.index)
    if step_minutes is None:
        step_minutes = float(pd.Series(index).diff().median() / pd.Timedelta(minutes=1))
    dt_h = step_minutes / 60.0
    frame = pd.DataFrame(
        {
            "heat_gcal": wide[T.HEAT_INPUT_GCALH].to_numpy() * dt_h,
            "clinker_t": wide[T.CLINKER_TPH].to_numpy() * dt_h,
            "gas_share": wide.get(T.GAS_HEAT_SHARE, pd.Series(np.nan, index=wide.index)).to_numpy(),
        },
        index=index,
    )
    daily = frame.resample("1D").agg({"heat_gcal": "sum", "clinker_t": "sum", "gas_share": "mean"})
    counts = frame["heat_gcal"].resample("1D").count()
    expected = 24 * 60 / step_minutes
    daily = daily[counts >= 0.9 * expected]
    daily["specific_heat_kcal_kg"] = daily["heat_gcal"] * 1e6 / (daily["clinker_t"] * 1e3)
    return daily


def daily_mill_energy(
    wide: pd.DataFrame, mill_ids: Sequence[int] = (1, 2), step_minutes: Optional[float] = None
) -> pd.DataFrame:
    """Daily mill energy (kWh) and cement ground (t), summed over ``mill_ids``."""
    index = pd.DatetimeIndex(wide.index)
    if step_minutes is None:
        step_minutes = float(pd.Series(index).diff().median() / pd.Timedelta(minutes=1))
    dt_h = step_minutes / 60.0
    kwh = np.zeros(len(wide))
    tonnes = np.zeros(len(wide))
    for n in mill_ids:
        kwh += wide[T.mill_tag(T.MILL_KW_TEMPLATE, n)].fillna(0).to_numpy() * dt_h
        tonnes += wide[T.mill_tag(T.MILL_TPH_TEMPLATE, n)].fillna(0).to_numpy() * dt_h
    frame = pd.DataFrame({"mill_kwh": kwh, "cement_t": tonnes}, index=index)
    return frame.resample("1D").sum()
