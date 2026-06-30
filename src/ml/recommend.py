"""Model-based setpoint recommendation engine.

This engine proposes safe setpoint adjustments that move a predicted KPI
toward a desired target. It is **model-based**: every candidate is scored by
running the *real* trained :class:`~src.ml.models.EnergyKPIModel` over features
rebuilt with the *same* feature-engineering logic used by the prediction
endpoint. There is no heuristic shortcut.

Pipeline per candidate:

1. perturb adjustable tags around their current setpoints,
2. write the candidate values back into the long-format signal frame,
3. reconstruct the model's exact feature matrix,
4. predict and score against the desired target (lower is better).
"""

from __future__ import annotations

import itertools
import logging
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

if TYPE_CHECKING:  # pragma: no cover - import only for type checking
    from src.ml.models import EnergyKPIModel

logger = logging.getLogger(__name__)

_TAG_COL = "tag_name"
_VALUE_COL = "value"


def generate_candidates(
    current_setpoints: Dict[str, float],
    adjustable_tags: List[str],
    step_fractions: List[float],
    absolute_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    max_candidates: int = 200,
) -> List[Dict[str, float]]:
    """Generate candidate setpoint dictionaries by perturbing adjustable tags.

    For each adjustable tag and each fraction ``f`` the perturbed value is
    ``current_value * (1 + f)``. The Cartesian product of per-tag perturbations
    forms the candidate space; the unchanged baseline is always included first.

    Parameters
    ----------
    current_setpoints:
        Current setpoint value per tag.
    adjustable_tags:
        Tags allowed to vary. Each must exist in ``current_setpoints``.
    step_fractions:
        Relative perturbations (e.g. ``-0.05`` for -5%). ``0.0`` reproduces the
        current value.
    absolute_bounds:
        Optional ``tag -> (min, max)`` clipping bounds applied to every value.
    max_candidates:
        Hard cap on the number of returned candidates (deterministic
        truncation, baseline kept first).

    Returns
    -------
    list[dict[str, float]]
        Candidate setpoint dictionaries over ``adjustable_tags``.

    Raises
    ------
    ValueError
        If any adjustable tag is missing from ``current_setpoints`` or
        ``max_candidates`` is not positive.
    """
    if max_candidates <= 0:
        raise ValueError(f"max_candidates must be positive, got {max_candidates}.")

    missing = [tag for tag in adjustable_tags if tag not in current_setpoints]
    if missing:
        raise ValueError(
            f"adjustable_tags missing from current_setpoints: {missing}."
        )

    bounds = absolute_bounds or {}

    def _clip(tag: str, value: float) -> float:
        if tag in bounds:
            lo, hi = bounds[tag]
            return float(min(max(value, lo), hi))
        return float(value)

    # Per-tag candidate value lists (deduplicated, order-preserving).
    per_tag_values: List[List[float]] = []
    for tag in adjustable_tags:
        base = float(current_setpoints[tag])
        values: List[float] = []
        seen: set[float] = set()
        for fraction in step_fractions:
            value = _clip(tag, base * (1.0 + fraction))
            if value not in seen:
                seen.add(value)
                values.append(value)
        per_tag_values.append(values)

    # Baseline candidate (current values, unchanged except for clipping).
    baseline = {tag: _clip(tag, float(current_setpoints[tag])) for tag in adjustable_tags}
    candidates: List[Dict[str, float]] = [baseline]
    seen_keys: set[Tuple[Tuple[str, float], ...]] = {tuple(sorted(baseline.items()))}

    for combo in itertools.product(*per_tag_values):
        candidate = {tag: val for tag, val in zip(adjustable_tags, combo)}
        key = tuple(sorted(candidate.items()))
        if key in seen_keys:
            continue
        seen_keys.add(key)
        candidates.append(candidate)
        if len(candidates) >= max_candidates:
            break

    logger.info("Generated %d candidate setpoint(s)", len(candidates))
    return candidates


def apply_candidate_to_signals(
    request_df: pd.DataFrame,
    candidate: Dict[str, float],
) -> pd.DataFrame:
    """Overwrite candidate tags' values in a long-format signal frame.

    Parameters
    ----------
    request_df:
        Long-format frame with ``timestamp``, ``tag_name`` and ``value``.
    candidate:
        Mapping of ``tag_name -> constant value`` to write across all rows of
        that tag.

    Returns
    -------
    pandas.DataFrame
        A copy of ``request_df`` with each candidate tag's ``value`` replaced,
        preserving row/timestamp order and schema.
    """
    df = request_df.copy()
    for tag, value in candidate.items():
        df.loc[df[_TAG_COL] == tag, _VALUE_COL] = float(value)
    return df


def score_candidate(predictions: np.ndarray, desired_target: float) -> float:
    """Score predictions against a desired target (lower is better).

    Parameters
    ----------
    predictions:
        Model predictions across the horizon.
    desired_target:
        Target KPI value to move toward.

    Returns
    -------
    float
        Mean absolute error between predictions and ``desired_target``.

    Raises
    ------
    ValueError
        If ``predictions`` is empty.
    """
    preds = np.asarray(predictions, dtype=float)
    if preds.size == 0:
        raise ValueError("predictions is empty; cannot score candidate.")
    return float(np.mean(np.abs(preds - float(desired_target))))


def rank_candidates(
    request_df: pd.DataFrame,
    current_setpoints: Dict[str, float],
    desired_target: float,
    model: "EnergyKPIModel",
    adjustable_tags: List[str],
    step_fractions: List[float],
    absolute_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    top_k: int = 5,
) -> List[dict]:
    """Rank perturbed setpoint candidates by model-predicted closeness to target.

    Each candidate is applied to the signals, expanded into the model's exact
    feature matrix (via the prediction endpoint's feature builder), predicted,
    and scored. The best ``top_k`` candidates (lowest score) are returned.

    Parameters
    ----------
    request_df:
        Long-format signal frame for the recent window.
    current_setpoints:
        Current setpoint value per tag.
    desired_target:
        Desired KPI value to move toward.
    model:
        A fitted :class:`EnergyKPIModel`.
    adjustable_tags:
        Tags allowed to vary.
    step_fractions:
        Relative perturbations to explore.
    absolute_bounds:
        Optional clipping bounds per tag.
    top_k:
        Number of best candidates to return.

    Returns
    -------
    list[dict]
        Each item has ``candidate_setpoints``, ``predicted_mean``,
        ``predicted_min``, ``predicted_max`` and ``score``, sorted ascending by
        ``score``.

    Raises
    ------
    ValueError
        If ``top_k`` is not positive, or no candidate can be evaluated.
    """
    # Imported lazily to avoid an import cycle between the ML and API layers.
    from src.api.routers.prediction import make_prediction_features

    if top_k <= 0:
        raise ValueError(f"top_k must be positive, got {top_k}.")

    candidates = generate_candidates(
        current_setpoints,
        adjustable_tags,
        step_fractions,
        absolute_bounds=absolute_bounds,
    )

    results: List[dict] = []
    last_error: Optional[Exception] = None
    for candidate in candidates:
        try:
            candidate_df = apply_candidate_to_signals(request_df, candidate)
            X = make_prediction_features(candidate_df, model)
            predictions = model.predict(X)
            score = score_candidate(predictions, desired_target)
        except ValueError as exc:
            last_error = exc
            continue

        results.append(
            {
                "candidate_setpoints": candidate,
                "predicted_mean": float(np.mean(predictions)),
                "predicted_min": float(np.min(predictions)),
                "predicted_max": float(np.max(predictions)),
                "score": score,
            }
        )

    if not results:
        raise ValueError(
            "No candidate could be evaluated successfully; last error: "
            f"{last_error}"
        )

    # Stable ascending sort by score (ties keep generation order: baseline first).
    results.sort(key=lambda item: item["score"])
    return results[:top_k]
