"""Rebuild a trained model's exact feature matrix from raw signals.

Shared by the prediction endpoint, the recommendation engine and the advisor
so the three never disagree on how a feature is computed. The trained
feature names (``{tag}_lag_{n}``, ``{tag}_roll_{agg}_{n}``) are the contract:
they are parsed back into tags, lags and windows, rebuilt, and returned in the
model's own column order.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

from src.domain.kpis import FuelProperties, add_derived_kpis
from src.ml.features import create_lag_features_per_tag, create_rolling_features, pivot_signals
from src.ml.models import EnergyKPIModel
from src.ml.pipeline import needs_derived_kpis

_LAG_RE = re.compile(r"^(?P<tag>.+)_lag_(?P<n>\d+)$")
_ROLL_RE = re.compile(r"^(?P<tag>.+)_roll_(?P<agg>mean|std|min|max)_(?P<n>\d+)$")


def parse_feature_spec(
    feature_cols: List[str],
) -> Tuple[Dict[str, Set[int]], Set[str], Set[int]]:
    """Infer ``(lags_by_tag, roll_tags, roll_windows)`` from feature names.

    Raises
    ------
    ValueError
        If a feature column does not match a supported naming pattern.
    """
    lags_by_tag: Dict[str, Set[int]] = {}
    roll_tags: Set[str] = set()
    roll_windows: Set[int] = set()
    for col in feature_cols:
        lag_match = _LAG_RE.match(col)
        if lag_match:
            lags_by_tag.setdefault(lag_match.group("tag"), set()).add(int(lag_match.group("n")))
            continue
        roll_match = _ROLL_RE.match(col)
        if roll_match:
            roll_tags.add(roll_match.group("tag"))
            roll_windows.add(int(roll_match.group("n")))
            continue
        raise ValueError(
            f"Unsupported feature column {col!r}; only "
            "{tag}_lag_{n} and {tag}_roll_(mean|std|min|max)_{n} are supported."
        )
    return lags_by_tag, roll_tags, roll_windows


def required_base_tags(model: EnergyKPIModel) -> List[str]:
    """Raw or derived tags the model's features are built from."""
    lags_by_tag, roll_tags, _ = parse_feature_spec(list(model.feature_cols or []))
    return sorted(set(lags_by_tag) | roll_tags)


def features_from_wide(
    wide: pd.DataFrame,
    model: EnergyKPIModel,
    fuels: Optional[FuelProperties] = None,
    dropna: bool = True,
) -> pd.DataFrame:
    """Feature matrix with columns exactly ``model.feature_cols``.

    Derived KPI tags the model needs (e.g. ``SPECIFIC_HEAT_KCAL_KG``) are
    computed from raw tags when they are not already in ``wide``.

    Raises
    ------
    ValueError
        If the model is not fitted, a required tag is missing, or no rows
        survive feature construction.
    """
    if not model.is_fitted or not model.feature_cols:
        raise ValueError("Model is not fitted; cannot build features.")
    feature_cols = list(model.feature_cols)
    lags_by_tag, roll_tags, roll_windows = parse_feature_spec(feature_cols)
    needed = sorted(set(lags_by_tag) | roll_tags)

    missing_derived = [t for t in needed if t not in wide.columns]
    if missing_derived and needs_derived_kpis(missing_derived):
        wide = add_derived_kpis(wide, fuels or FuelProperties())
    missing = [t for t in needed if t not in wide.columns]
    if missing:
        raise ValueError(
            f"Request is missing required base tag(s): {missing}. "
            f"Provided tags: {list(wide.columns)}."
        )

    frames: List[pd.DataFrame] = []
    if lags_by_tag:
        frames.append(
            create_lag_features_per_tag(wide, {t: sorted(v) for t, v in sorted(lags_by_tag.items())})
        )
    if roll_tags:
        frames.append(create_rolling_features(wide, sorted(roll_tags), sorted(roll_windows)))
    features = pd.concat(frames, axis=1)

    not_built = [c for c in feature_cols if c not in features.columns]
    if not_built:
        raise ValueError(f"Could not reconstruct required feature column(s): {not_built}.")
    X = features[feature_cols]
    if dropna:
        X = X.dropna(axis=0, how="any")
    if X.empty:
        raise ValueError(
            "No rows remain after building lag/rolling features; provide a "
            "longer input window than the largest lag/rolling window."
        )
    return X


def features_from_long(
    signals: pd.DataFrame,
    model: EnergyKPIModel,
    fuels: Optional[FuelProperties] = None,
) -> pd.DataFrame:
    """Long-format (``timestamp, tag_name, value``) variant of :func:`features_from_wide`."""
    return features_from_wide(pivot_signals(signals), model, fuels=fuels)


def min_window_rows(model: EnergyKPIModel) -> int:
    """Smallest input window (rows) that yields at least one feature row."""
    lags_by_tag, _, roll_windows = parse_feature_spec(list(model.feature_cols or []))
    max_lag = max((max(v) for v in lags_by_tag.values()), default=0)
    max_window = max(roll_windows, default=1)
    return max(max_lag + 1, max_window)
