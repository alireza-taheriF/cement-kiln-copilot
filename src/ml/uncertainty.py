"""Prediction intervals by split conformal prediction.

An operator should never see a single number without knowing how far off it
can be. Split conformal prediction gives a model-agnostic interval with a
finite-sample coverage guarantee, under exchangeability, from the residuals on
a held-out set:

    q = the ceil((n + 1)(1 - alpha))-th smallest |residual|
    interval = prediction ± q

For time series exchangeability is only approximate; the validation split is
chronological, so ``q`` reflects the most recent behaviour, which is what we
want. Coverage should still be monitored in production (see
:func:`empirical_coverage`).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class ConformalCalibration:
    """Calibration result stored in the model metadata."""

    alpha: float
    half_width: float
    n_calibration: int

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict) -> "ConformalCalibration":
        return cls(float(raw["alpha"]), float(raw["half_width"]), int(raw["n_calibration"]))


def calibrate(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    alpha: float = 0.1,
) -> ConformalCalibration:
    """Half-width of a ``1 - alpha`` split-conformal interval."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1).")
    residuals = np.abs(np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float))
    n = residuals.size
    if n == 0:
        raise ValueError("Cannot calibrate on an empty set.")
    rank = math.ceil((n + 1) * (1 - alpha))
    if rank > n:
        # Too few points for the requested level: the honest answer is "unbounded".
        half_width = float("inf")
    else:
        half_width = float(np.sort(residuals)[rank - 1])
    return ConformalCalibration(alpha=alpha, half_width=half_width, n_calibration=n)


def interval(
    predictions: Sequence[float],
    calibration: ConformalCalibration,
) -> tuple[np.ndarray, np.ndarray]:
    """Lower and upper interval bounds around ``predictions``."""
    preds = np.asarray(predictions, dtype=float)
    return preds - calibration.half_width, preds + calibration.half_width


def empirical_coverage(
    y_true: Sequence[float],
    lower: Sequence[float],
    upper: Sequence[float],
) -> float:
    """Share of observations inside their interval (for monitoring)."""
    y = np.asarray(y_true, dtype=float)
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    if y.size == 0:
        raise ValueError("No observations.")
    return float(np.mean((y >= lo) & (y <= hi)))
