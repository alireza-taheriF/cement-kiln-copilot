"""Naive forecasts scored on the same chronological validation split.

Two references sit beside :class:`~src.ml.models.EnergyKPIModel`:

* **persistence** — at each forecast origin, predict the target measurement
  already observed at that timestamp. The future label (``y[t] = target[t +
  horizon]``) is never used as an input.
* **ordinary linear regression** — :class:`sklearn.linear_model.LinearRegression`
  fit on the training rows only, using the same feature columns as the
  energy model.

Both are scored with :func:`src.ml.evaluate.regression_metrics` on the
validation rows produced by
:func:`src.ml.evaluate.train_valid_split_time_series`. The split is
chronological. These helpers do not shuffle.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from src.ml.evaluate import regression_metrics
from src.ml.features import pivot_signals


def last_observed_target(
    signals_df: pd.DataFrame,
    target_tag: str,
    index: pd.Index,
) -> pd.Series:
    """Return the target value known at each forecast origin.

    Parameters
    ----------
    signals_df:
        Long-format signals (``timestamp``, ``tag_name``, ``value``), the same
        frame used to build the supervised dataset.
    target_tag:
        Tag being forecast.
    index:
        Forecast-origin timestamps, aligned to the supervised feature index.

    Returns
    -------
    pandas.Series
        ``target_tag`` at each origin. This is not the future label.

    Raises
    ------
    ValueError
        If the tag is missing, or any origin has no contemporaneous
        measurement. Missing origins are not filled from later rows.
    """
    wide = pivot_signals(signals_df)
    if target_tag not in wide.columns:
        raise ValueError(
            f"target tag {target_tag!r} is not in the signal table. "
            f"Available tags: {list(wide.columns)}."
        )

    observed = wide[target_tag].reindex(index)
    if observed.isna().any():
        missing = int(observed.isna().sum())
        raise ValueError(
            f"Last observed {target_tag} is missing for {missing} forecast "
            "origin(s). Refusing to fill those origins from future rows."
        )
    observed.name = target_tag
    return observed


def persistence_predict(last_observed: pd.Series) -> np.ndarray:
    """Predict each origin's already-observed target value.

    The mapping is the identity of ``last_observed``. It does not read any
    other row, so it cannot see a future label. Calling it twice on the same
    series returns the same array.

    Parameters
    ----------
    last_observed:
        Target measurements available at forecast time, one per row to score.

    Returns
    -------
    numpy.ndarray
        A copy of those measurements, in the same order.

    Raises
    ------
    TypeError
        If ``last_observed`` is not a Series.
    ValueError
        If the series is empty or contains a non-finite value.
    """
    if not isinstance(last_observed, pd.Series):
        raise TypeError(
            f"last_observed must be a pandas Series, got {type(last_observed)!r}."
        )
    values = np.asarray(last_observed.to_numpy(), dtype=float)
    if values.size == 0:
        raise ValueError("Cannot build a persistence forecast from an empty series.")
    if not np.isfinite(values).all():
        raise ValueError(
            "Persistence input contains a non-finite value. "
            "Refusing to substitute a future row."
        )
    return values.copy()


def fit_linear_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> LinearRegression:
    """Fit ordinary least squares on training rows only.

    Parameters
    ----------
    X_train:
        Training feature matrix. Column order is the model feature order.
    y_train:
        Training targets aligned to ``X_train``.

    Returns
    -------
    sklearn.linear_model.LinearRegression
        Estimator fitted on ``(X_train, y_train)`` and on no other rows.

    Raises
    ------
    TypeError
        If ``X_train`` is not a DataFrame or ``y_train`` is not a Series.
    ValueError
        If either input is empty or their lengths differ.
    """
    if not isinstance(X_train, pd.DataFrame):
        raise TypeError(f"X_train must be a pandas DataFrame, got {type(X_train)!r}.")
    if not isinstance(y_train, pd.Series):
        raise TypeError(f"y_train must be a pandas Series, got {type(y_train)!r}.")
    if X_train.empty or y_train.empty:
        raise ValueError("Linear regression requires a non-empty training split.")
    if len(X_train) != len(y_train):
        raise ValueError(
            f"X_train and y_train length mismatch: {len(X_train)} vs {len(y_train)}."
        )

    estimator = LinearRegression()
    estimator.fit(X_train, y_train)
    return estimator


def _require_chronological_split(
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
) -> None:
    """Reject a split whose validation index is not strictly after training."""
    if X_train.empty or X_valid.empty:
        raise ValueError("Train and validation splits must both be non-empty.")
    if not (
        X_train.index.is_monotonic_increasing and X_valid.index.is_monotonic_increasing
    ):
        raise ValueError(
            "Train and validation indexes must be monotonic increasing. "
            "Use train_valid_split_time_series; do not shuffle."
        )
    if X_train.index.max() >= X_valid.index.min():
        raise ValueError(
            "Validation rows are not strictly after the training rows. "
            "Use train_valid_split_time_series; do not shuffle."
        )


def _validation_observations(
    y_observed: pd.Series,
    X_valid: pd.DataFrame,
) -> pd.Series:
    """Align forecast-time target values to the validation rows."""
    if not isinstance(y_observed, pd.Series):
        raise TypeError(
            f"y_observed must be a pandas Series, got {type(y_observed)!r}."
        )
    aligned = y_observed.reindex(X_valid.index)
    if aligned.isna().any():
        raise ValueError(
            "y_observed does not cover every validation forecast origin. "
            "Refusing to fill missing origins from future rows."
        )
    return aligned


def _feature_frame(X_train: pd.DataFrame, X_valid: pd.DataFrame) -> pd.DataFrame:
    """Return ``X_valid`` in the training column order."""
    feature_cols = list(X_train.columns)
    missing = [column for column in feature_cols if column not in X_valid.columns]
    if missing:
        raise ValueError(
            f"Validation rows are missing feature column(s): {missing}."
        )
    return X_valid.loc[:, feature_cols]


def _score(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    """RMSE and MAE on ``y_true``, plus the validation row count."""
    metrics = regression_metrics(y_true, y_pred)
    return {
        "rmse": float(metrics["rmse"]),
        "mae": float(metrics["mae"]),
        "n_samples": int(len(y_true)),
    }


def evaluate_naive_baselines(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    y_observed: pd.Series,
    model_valid_rmse: float,
) -> dict[str, Any]:
    """Score persistence and linear regression on the model's validation rows.

    Parameters
    ----------
    X_train, y_train:
        Training partition from :func:`train_valid_split_time_series`.
        Linear regression is fit on these rows only.
    X_valid, y_valid:
        Validation partition. Both forecasts are scored on these rows only.
    y_observed:
        Target measurement at each forecast origin (not the future label).
        Must cover ``X_valid.index``. Values are the last observation available
        at that origin.
    model_valid_rmse:
        Held-out RMSE of :class:`~src.ml.models.EnergyKPIModel` on the same
        ``y_valid`` rows. It is reported beside the naive scores and is not
        recomputed here.

    Returns
    -------
    dict
        ``n_samples``, ``model_valid_rmse``, and for each of ``persistence``
        and ``linear_regression`` a dict with ``rmse``, ``mae``, and
        ``n_samples``.

    Raises
    ------
    ValueError
        If the split is empty, shuffled, or ``model_valid_rmse`` is not finite.
    """
    _require_chronological_split(X_train, X_valid)
    if len(X_valid) != len(y_valid):
        raise ValueError(
            f"X_valid and y_valid length mismatch: {len(X_valid)} vs {len(y_valid)}."
        )
    if not X_valid.index.equals(y_valid.index):
        raise ValueError("X_valid and y_valid must share the same index.")

    model_rmse = float(model_valid_rmse)
    if not np.isfinite(model_rmse):
        raise ValueError(f"model_valid_rmse must be finite, got {model_valid_rmse!r}.")

    observed_valid = _validation_observations(y_observed, X_valid)
    persistence_pred = persistence_predict(observed_valid)

    linear = fit_linear_regression(X_train, y_train)
    linear_pred = np.asarray(
        linear.predict(_feature_frame(X_train, X_valid)),
        dtype=float,
    )

    n_samples = int(len(y_valid))
    persistence_score = _score(y_valid, persistence_pred)
    linear_score = _score(y_valid, linear_pred)
    if persistence_score["n_samples"] != n_samples or linear_score["n_samples"] != n_samples:
        raise RuntimeError("Baseline scores were not computed on the validation rows.")

    return {
        "n_samples": n_samples,
        "model_valid_rmse": model_rmse,
        "persistence": persistence_score,
        "linear_regression": linear_score,
    }
