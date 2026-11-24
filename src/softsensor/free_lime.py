"""Free-lime soft sensor: a lab value every minute instead of every two hours.

Free lime (CaO_f) is the standard clinker-quality check, but the lab reports
it every one to two hours, and the result describes clinker made well
before that. Operators react late, and over-fire "to be safe" in between.

Timing model
------------
::

    clinker made at  t_s = t_lab - transport          (sample travels to the lab)
    burnt under      process window ending t_s - residence
                     of length window

so a lab result at ``t_lab`` is explained by process conditions in
``[t_lab - transport - residence - window, t_lab - transport - residence]``.
:func:`select_delay` searches the total delay (transport + residence) that
best explains the lab history, because it is rarely known precisely.

Model
-----
Standardized ridge regression on window aggregates (mean, std, min, max)
of the input tags. With 10-30 lab samples a day a linear model is the
right complexity; the out-of-fold residuals from a chronological
cross-validation give a conformal prediction interval.

Online use
----------
Each new lab result updates an exponentially weighted **bias correction**
(lab minus soft sensor at the time that clinker was made), the usual way
soft sensors track slow raw-mix and calibration drift between retrainings.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from src.domain import tags as T
from src.ml.uncertainty import ConformalCalibration, calibrate

_AGGS = ("mean", "std", "min", "max")


def window_features(
    wide: pd.DataFrame,
    input_tags: Sequence[str],
    window_minutes: float,
) -> pd.DataFrame:
    """Trailing-window aggregates at every timestamp of ``wide``.

    Column ``{tag}_{agg}`` at time ``t`` summarises ``(t - window, t]``.
    """
    missing = [t for t in input_tags if t not in wide]
    if missing:
        raise ValueError(f"process data is missing tag(s): {missing}.")
    window = f"{int(window_minutes)}min"
    out = {}
    for tag in input_tags:
        roll = wide[tag].rolling(window, min_periods=2)
        for agg in _AGGS:
            out[f"{tag}_{agg}"] = getattr(roll, agg)()
    return pd.DataFrame(out, index=wide.index)


def align_lab(
    features: pd.DataFrame,
    lab: pd.DataFrame,
    delay_minutes: float,
    parameter: str = T.FREE_LIME_PCT,
    tolerance_minutes: float = 15.0,
) -> pd.DataFrame:
    """Join each lab result with the features ``delay_minutes`` before it.

    ``lab`` is long-format (``timestamp, parameter, value``). Rows whose
    feature timestamp has no process sample within ``tolerance_minutes``
    are dropped.
    """
    samples = lab[lab["parameter"] == parameter][["timestamp", "value"]].copy()
    samples["timestamp"] = pd.to_datetime(samples["timestamp"])
    samples["feature_time"] = samples["timestamp"] - pd.Timedelta(minutes=delay_minutes)
    samples = samples.sort_values("feature_time")
    feats = features.dropna().copy()
    feats.index = pd.DatetimeIndex(feats.index)
    feats = feats.rename_axis("feature_time").reset_index().sort_values("feature_time")
    merged = pd.merge_asof(
        samples,
        feats,
        on="feature_time",
        direction="backward",
        tolerance=pd.Timedelta(minutes=tolerance_minutes),
    )
    return merged.dropna().reset_index(drop=True)


def _pipeline() -> Pipeline:
    return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13)))


def _oof_predictions(X: np.ndarray, y: np.ndarray, n_splits: int) -> tuple[np.ndarray, np.ndarray]:
    """Out-of-fold predictions from an expanding-window chronological CV."""
    preds = np.full(len(y), np.nan)
    for train_idx, test_idx in TimeSeriesSplit(n_splits=n_splits).split(X):
        model = _pipeline().fit(X[train_idx], y[train_idx])
        preds[test_idx] = model.predict(X[test_idx])
    mask = ~np.isnan(preds)
    return preds[mask], y[mask]


@dataclass
class DelaySearch:
    """Result of the delay grid search."""

    best_delay_minutes: float
    scores: Dict[float, float]  # delay -> out-of-fold RMSE


def select_delay(
    wide: pd.DataFrame,
    lab: pd.DataFrame,
    input_tags: Sequence[str],
    window_minutes: float,
    candidates_minutes: Sequence[float],
    n_splits: int = 5,
) -> DelaySearch:
    """Total delay (transport + residence) with the lowest out-of-fold RMSE."""
    feats = window_features(wide, input_tags, window_minutes)
    scores: Dict[float, float] = {}
    for delay in candidates_minutes:
        data = align_lab(feats, lab, delay)
        if len(data) < 3 * n_splits:
            continue
        cols = [c for c in data.columns if c not in ("timestamp", "value", "feature_time")]
        pred, y = _oof_predictions(data[cols].to_numpy(), data["value"].to_numpy(), n_splits)
        scores[float(delay)] = float(np.sqrt(np.mean((pred - y) ** 2)))
    if not scores:
        raise ValueError("not enough aligned lab samples to search delays.")
    best = min(scores, key=scores.get)
    return DelaySearch(best, scores)


@dataclass
class FreeLimeSoftSensor:
    """Fitted soft sensor with conformal interval and online bias correction."""

    input_tags: List[str]
    window_minutes: float
    delay_minutes: float
    model: Pipeline
    calibration: ConformalCalibration
    cv_rmse: float
    cv_r2: float
    n_samples: int
    bias: float = 0.0
    bias_weight: float = 0.3  # EWMA weight of each new lab result
    feature_columns: List[str] = field(default_factory=list)

    # ------------------------------------------------------------------ #
    @classmethod
    def fit(
        cls,
        wide: pd.DataFrame,
        lab: pd.DataFrame,
        input_tags: Sequence[str] = (T.BZ_TEMP_C, T.SPECIFIC_HEAT_KCAL_KG, T.KILN_INLET_O2_PCT, T.KILN_FEED_TPH),
        window_minutes: float = 60.0,
        delay_minutes: Optional[float] = None,
        delay_candidates: Sequence[float] = tuple(range(30, 181, 15)),
        n_splits: int = 5,
        alpha: float = 0.1,
    ) -> "FreeLimeSoftSensor":
        """Fit on history. ``delay_minutes=None`` searches ``delay_candidates``."""
        if delay_minutes is None:
            delay_minutes = select_delay(
                wide, lab, input_tags, window_minutes, delay_candidates, n_splits
            ).best_delay_minutes
        feats = window_features(wide, input_tags, window_minutes)
        data = align_lab(feats, lab, delay_minutes)
        cols = [c for c in data.columns if c not in ("timestamp", "value", "feature_time")]
        if len(data) < 3 * n_splits:
            raise ValueError(f"need at least {3 * n_splits} aligned lab samples, got {len(data)}.")
        X = data[cols].to_numpy()
        y = data["value"].to_numpy()
        oof, y_oof = _oof_predictions(X, y, n_splits)
        rmse = float(np.sqrt(np.mean((oof - y_oof) ** 2)))
        ss_tot = float(((y_oof - y_oof.mean()) ** 2).sum())
        r2 = 1.0 - float(((oof - y_oof) ** 2).sum()) / ss_tot if ss_tot > 0 else 0.0
        return cls(
            input_tags=list(input_tags),
            window_minutes=float(window_minutes),
            delay_minutes=float(delay_minutes),
            model=_pipeline().fit(X, y),
            calibration=calibrate(y_oof, oof, alpha=alpha),
            cv_rmse=rmse,
            cv_r2=r2,
            n_samples=len(y),
            feature_columns=cols,
        )

    # ------------------------------------------------------------------ #
    def estimate(self, wide: pd.DataFrame) -> pd.DataFrame:
        """Free lime of the clinker leaving the kiln, at each timestamp.

        The value at ``t`` is computed from the process window ending at
        ``t``; it describes clinker that reaches the lab ``delay_minutes``
        later. Columns: ``free_lime``, ``lower``, ``upper``, ``lab_due_at``.
        """
        feats = window_features(wide, self.input_tags, self.window_minutes)[self.feature_columns]
        feats = feats.dropna()
        if feats.empty:
            raise ValueError(f"need at least {self.window_minutes:g} minutes of process data.")
        pred = self.model.predict(feats.to_numpy()) + self.bias
        half = self.calibration.half_width
        return pd.DataFrame(
            {
                "free_lime": pred,
                "lower": pred - half,
                "upper": pred + half,
                "lab_due_at": pd.DatetimeIndex(feats.index) + pd.Timedelta(minutes=self.delay_minutes),
            },
            index=feats.index,
        )

    def update_bias(self, wide: pd.DataFrame, lab_value: float, lab_time: pd.Timestamp) -> float:
        """Fold a new lab result into the bias correction; return the new bias."""
        feature_time = pd.Timestamp(lab_time) - pd.Timedelta(minutes=self.delay_minutes)
        history = wide.loc[: feature_time]
        if history.empty:
            raise ValueError("no process data before the lab sample's feature time.")
        estimate = self.estimate(history).iloc[-1]["free_lime"] - self.bias
        error = float(lab_value) - float(estimate)
        self.bias = (1 - self.bias_weight) * self.bias + self.bias_weight * error
        return self.bias

    def summary(self) -> dict:
        return {
            "input_tags": self.input_tags,
            "window_minutes": self.window_minutes,
            "delay_minutes": self.delay_minutes,
            "cv_rmse": self.cv_rmse,
            "cv_r2": self.cv_r2,
            "interval_half_width": self.calibration.half_width
            if math.isfinite(self.calibration.half_width)
            else None,
            "n_samples": self.n_samples,
            "bias": self.bias,
        }
