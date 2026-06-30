"""Evaluation metrics and time-series splitting for regression models.

These helpers operate on the :class:`~src.ml.models.EnergyKPIModel` wrapper
and on raw arrays, and are deliberately free of plotting/notebook concerns so
they can be used in training, CI gates, and serving alike.
"""

from __future__ import annotations

import logging
from typing import Sequence, Tuple, TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:  # pragma: no cover - import only for type checking
    from src.ml.models import EnergyKPIModel

logger = logging.getLogger(__name__)


def regression_metrics(
    y_true: Sequence[float],
    y_pred: Sequence[float],
) -> dict[str, float]:
    """Compute core regression error metrics.

    Parameters
    ----------
    y_true:
        Ground-truth target values.
    y_pred:
        Predicted target values.

    Returns
    -------
    dict[str, float]
        ``rmse``, ``mae`` and ``mape`` as native Python floats. ``mape`` is a
        percentage and is computed only over entries with a non-zero
        ``y_true`` (zero targets are ignored to avoid division-by-zero); it is
        ``nan`` when every target is zero.

    Raises
    ------
    ValueError
        If ``y_true`` and ``y_pred`` have different lengths or are empty.
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)

    if y_true_arr.shape[0] != y_pred_arr.shape[0]:
        raise ValueError(
            f"Length mismatch: y_true has {y_true_arr.shape[0]} entries, "
            f"y_pred has {y_pred_arr.shape[0]}."
        )
    if y_true_arr.shape[0] == 0:
        raise ValueError("Cannot compute metrics on empty inputs.")

    errors = y_pred_arr - y_true_arr
    rmse = float(np.sqrt(np.mean(np.square(errors))))
    mae = float(np.mean(np.abs(errors)))

    nonzero = y_true_arr != 0.0
    if not nonzero.any():
        mape = float("nan")
    else:
        mape = float(
            np.mean(
                np.abs(errors[nonzero] / y_true_arr[nonzero])
            )
            * 100.0
        )

    return {"rmse": rmse, "mae": mae, "mape": mape}


def evaluate_regressor(
    model: "EnergyKPIModel",
    X: pd.DataFrame,
    y: pd.Series,
) -> dict[str, float]:
    """Evaluate a fitted :class:`EnergyKPIModel` on ``(X, y)``.

    Parameters
    ----------
    model:
        A fitted model wrapper.
    X:
        Feature matrix (column order is enforced by the wrapper).
    y:
        Ground-truth targets aligned to ``X``.

    Returns
    -------
    dict[str, float]
        Regression metrics (see :func:`regression_metrics`) plus
        ``n_samples``, ``n_features`` and ``target_name``.
    """
    predictions = model.predict(X)
    metrics: dict[str, float] = regression_metrics(np.asarray(y, dtype=float), predictions)
    metrics["n_samples"] = int(len(X))
    metrics["n_features"] = int(len(model.feature_cols) if model.feature_cols else X.shape[1])
    metrics["target_name"] = model.target_name
    return metrics


def train_valid_split_time_series(
    X: pd.DataFrame,
    y: pd.Series,
    valid_fraction: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Chronologically split ``(X, y)`` into train/validation partitions.

    The split is order-preserving and never shuffles: the earliest rows form
    the training set and the most recent ``valid_fraction`` rows form the
    validation set. This avoids look-ahead leakage in time-series models.

    Parameters
    ----------
    X:
        Feature matrix, ordered chronologically.
    y:
        Target vector aligned to ``X``.
    valid_fraction:
        Fraction of the most recent rows used for validation. Must be strictly
        between 0 and 1.

    Returns
    -------
    tuple
        ``(X_train, X_valid, y_train, y_valid)`` with original index ordering
        preserved.

    Raises
    ------
    ValueError
        If ``valid_fraction`` is not in the open interval ``(0, 1)`` or the
        lengths of ``X`` and ``y`` differ.
    """
    if not 0.0 < valid_fraction < 1.0:
        raise ValueError(
            f"valid_fraction must be in (0, 1), got {valid_fraction}."
        )
    if len(X) != len(y):
        raise ValueError(
            f"X and y length mismatch: {len(X)} rows vs {len(y)} targets."
        )

    n_total = len(X)
    n_valid = max(1, int(round(n_total * valid_fraction)))
    n_train = n_total - n_valid
    if n_train <= 0:
        raise ValueError(
            "valid_fraction leaves no training rows; reduce valid_fraction."
        )

    X_train = X.iloc[:n_train]
    X_valid = X.iloc[n_train:]
    y_train = y.iloc[:n_train]
    y_valid = y.iloc[n_train:]
    return X_train, X_valid, y_train, y_valid
