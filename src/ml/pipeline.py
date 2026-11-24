"""Database-free model fitting: wide frame in, calibrated model out.

:mod:`src.ml.train` adds I/O around this (database, artifacts, MLflow). Keeping
the numeric core here lets tests, the simulator and notebooks train exactly
the same model without a database.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union

import pandas as pd

from src.domain.kpis import FuelProperties, add_derived_kpis
from src.domain.tags import TAGS, TagKind
from src.ml.delays import estimate_delays, lags_from_delays
from src.ml.evaluate import evaluate_regressor, train_valid_split_time_series
from src.ml.features import make_supervised_dataset
from src.ml.models import EnergyKPIModel
from src.ml.uncertainty import ConformalCalibration, calibrate

logger = logging.getLogger(__name__)

DEFAULT_VALID_FRACTION = 0.2
DEFAULT_CONFORMAL_ALPHA = 0.1

LagSpec = Union[Sequence[int], str]

#: Estimators selectable from a training config (``estimator:`` key).
ESTIMATORS = ("lightgbm", "random_forest", "ridge")


def build_estimator(name: Optional[str]) -> Any:
    """Unfitted regressor for a config ``estimator`` name (``None``: default).

    ``ridge`` (standardized, regularization chosen by CV) is deterministic
    across machines and extrapolates gains linearly, which suits constraint
    models of a smooth process; tree ensembles suit strongly non-linear
    targets but cannot extrapolate.
    """
    if name is None or name == "lightgbm":
        return None  # EnergyKPIModel default: LightGBM, random-forest fallback
    if name == "random_forest":
        from sklearn.ensemble import RandomForestRegressor

        return RandomForestRegressor(n_estimators=200, min_samples_leaf=5, random_state=42, n_jobs=-1)
    if name == "ridge":
        import numpy as np
        from sklearn.linear_model import RidgeCV
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 13)))
    raise ValueError(f"estimator must be one of {ESTIMATORS}, got {name!r}.")


@dataclass
class FitResult:
    """Everything the trainer needs to persist and report a fitted model."""

    model: EnergyKPIModel
    X: pd.DataFrame
    X_train: pd.DataFrame
    X_valid: pd.DataFrame
    train_metrics: Dict[str, float]
    valid_metrics: Dict[str, float]
    conformal: ConformalCalibration
    delays: Dict[str, Dict[str, float]] = field(default_factory=dict)


def needs_derived_kpis(tags: Sequence[str]) -> bool:
    """Whether any requested tag is a derived KPI computed from raw tags."""
    return any(TAGS.get(t) is not None and TAGS[t].kind is TagKind.DERIVED for t in tags)


def prepare_wide(
    wide: pd.DataFrame,
    required_tags: Sequence[str],
    fuels: Optional[FuelProperties] = None,
) -> pd.DataFrame:
    """Add derived KPI columns when a required tag needs them."""
    if needs_derived_kpis(required_tags):
        wide = add_derived_kpis(wide, fuels or FuelProperties())
    missing = [t for t in required_tags if t not in wide.columns]
    if missing:
        raise ValueError(
            f"tag(s) not found in signals: {missing}. Available tags: {list(wide.columns)}."
        )
    return wide


def fit_kpi_model(
    wide: pd.DataFrame,
    target_tag: str,
    input_tags: List[str],
    lags: LagSpec,
    rolling_windows: List[int],
    horizon: int,
    config: Optional[Dict[str, Any]] = None,
    estimator: Any = None,
) -> FitResult:
    """Fit, evaluate and conformally calibrate a KPI forecasting model.

    Parameters
    ----------
    wide:
        Regular-grid, timestamp-indexed frame containing every tag needed.
    lags:
        List of lags for every input, or ``"auto"`` to add the delay-matched
        lag per input (:func:`src.ml.delays.lags_from_delays`) on top of
        ``config["base_lags"]``.
    config:
        Optional keys: ``target_mode`` (``"level"`` default, or ``"delta"``:
        learn the change of the target over the horizon on top of its current
        value, which tree models need when the level drifts outside the
        training range), ``base_lags`` (default ``[1]``), ``max_delay_rows``
        (default 24), ``delay_diff_rows`` (default 3), ``conformal_alpha`` (default 0.1), ``valid_fraction``.
    estimator:
        Optional unfitted scikit-learn style regressor. Default: the config's
        ``estimator`` name (:func:`build_estimator`), else LightGBM with a
        random-forest fallback.
    """
    cfg = config or {}
    if estimator is None:
        estimator = build_estimator(cfg.get("estimator"))
    auto = isinstance(lags, str)
    if auto and lags != "auto":
        raise ValueError(f"lags must be a list of ints or 'auto', got {lags!r}.")
    base_lags = [int(v) for v in cfg.get("base_lags", [1])] if auto else [int(v) for v in lags]

    lags_by_tag = None
    delays_summary: Dict[str, Dict[str, float]] = {}
    if auto:
        delays = estimate_delays(
            wide,
            target_tag,
            input_tags,
            max_delay=int(cfg.get("max_delay_rows", 24)),
            diff_rows=int(cfg.get("delay_diff_rows", 3)),
        )
        lags_by_tag = lags_from_delays(delays, horizon, base_lags)
        delays_summary = {
            tag: {"delay_rows": d.delay_rows, "correlation": round(d.correlation, 4)}
            for tag, d in delays.items()
        }

    target_mode = str(cfg.get("target_mode", "level"))
    if target_mode not in ("level", "delta"):
        raise ValueError("target_mode must be 'level' or 'delta'.")
    anchor = f"{target_tag}_lag_0"
    if target_mode == "delta":
        if target_tag not in input_tags:
            raise ValueError("target_mode 'delta' needs the target among input_tags.")
        if lags_by_tag is None:
            lags_by_tag = {tag: list(base_lags) for tag in input_tags}
        lags_by_tag[target_tag] = sorted(set(lags_by_tag.get(target_tag, base_lags)) | {0})

    X, y = make_supervised_dataset(
        pd.DataFrame(),
        target_tag=target_tag,
        input_tags=input_tags,
        lags=base_lags,
        rolling_windows=rolling_windows,
        horizon=horizon,
        lags_by_tag=lags_by_tag,
        wide=wide,
    )
    if X.empty:
        raise ValueError(
            "Supervised dataset is empty after feature/target alignment; "
            "not enough contiguous samples for the given lags/windows/horizon."
        )
    X_train, X_valid, y_train, y_valid = train_valid_split_time_series(
        X, y, valid_fraction=float(cfg.get("valid_fraction", DEFAULT_VALID_FRACTION))
    )
    model = EnergyKPIModel(model=estimator, target_name=target_tag)
    if target_mode == "delta":
        model.fit(X_train, y_train - X_train[anchor])
        model.metadata["target_mode"] = "delta"
        model.metadata["anchor_feature"] = anchor
    else:
        model.fit(X_train, y_train)
    train_metrics = evaluate_regressor(model, X_train, y_train)
    valid_metrics = evaluate_regressor(model, X_valid, y_valid)

    conformal = calibrate(
        y_valid,
        model.predict(X_valid),
        alpha=float(cfg.get("conformal_alpha", DEFAULT_CONFORMAL_ALPHA)),
    )
    model.metadata["conformal"] = conformal.to_dict()
    model.metadata["horizon_rows"] = int(horizon)
    model.metadata["input_tags"] = list(input_tags)
    if delays_summary:
        model.metadata["delays"] = delays_summary
    return FitResult(
        model=model,
        X=X,
        X_train=X_train,
        X_valid=X_valid,
        train_metrics=train_metrics,
        valid_metrics=valid_metrics,
        conformal=conformal,
        delays=delays_summary,
    )
