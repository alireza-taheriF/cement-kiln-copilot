"""Feature engineering for industrial time-series forecasting.

Transforms long-format process signals (as stored in the database) into
supervised machine-learning datasets for future-KPI prediction.

The canonical input is the long/tidy signal table::

    timestamp | tag_name | value

which is pivoted into a wide, timestamp-indexed frame and then expanded into
lag and rolling-window features. A future-shifted target is built from a
chosen tag so the resulting ``(X, y)`` pair can be fed directly to a
regressor.

Design notes
------------
* Pure pandas, no leakage: rolling/lag features only look backward; the
  target only looks forward.
* Deterministic column ordering: feature columns are generated in a stable,
  reproducible order driven by the caller-supplied tag/lag/window lists.
"""

from __future__ import annotations

import logging
from typing import List, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

#: Column names expected on the long-format signal input.
TIMESTAMP_COLUMN = "timestamp"
TAG_COLUMN = "tag_name"
VALUE_COLUMN = "value"

#: Rolling aggregations applied per (tag, window), in deterministic order.
_ROLLING_AGGS: tuple[str, ...] = ("mean", "std", "min", "max")


def _require_tags(wide_df: pd.DataFrame, tags: List[str], role: str) -> None:
    """Raise ``ValueError`` if any requested tag is absent from ``wide_df``.

    Parameters
    ----------
    wide_df:
        Wide, timestamp-indexed frame whose columns are tag names.
    tags:
        Tag names that must be present.
    role:
        Human-readable role of the tags (e.g. ``"input"``/``"target"``),
        used to build a clear error message.
    """
    missing = [tag for tag in tags if tag not in wide_df.columns]
    if missing:
        available = list(wide_df.columns)
        raise ValueError(
            f"{role} tag(s) not found in signals: {missing}. "
            f"Available tags: {available}."
        )


def pivot_signals(signals_df: pd.DataFrame) -> pd.DataFrame:
    """Pivot a long-format signal frame into a wide, timestamp-indexed frame.

    Parameters
    ----------
    signals_df:
        Long-format DataFrame with ``timestamp``, ``tag_name`` and ``value``
        columns.

    Returns
    -------
    pandas.DataFrame
        Wide frame indexed by ``timestamp`` (ascending) with one column per
        tag (columns sorted by name for determinism). On a duplicate
        ``(timestamp, tag_name)`` pair the last occurrence is kept.

    Raises
    ------
    ValueError
        If any of the required long-format columns is missing.
    """
    required = [TIMESTAMP_COLUMN, TAG_COLUMN, VALUE_COLUMN]
    missing = [c for c in required if c not in signals_df.columns]
    if missing:
        raise ValueError(
            f"signals_df is missing required column(s): {missing}. "
            f"Expected columns: {required}."
        )

    df = signals_df[required].copy()
    df[TIMESTAMP_COLUMN] = pd.to_datetime(df[TIMESTAMP_COLUMN])
    df[VALUE_COLUMN] = pd.to_numeric(df[VALUE_COLUMN], errors="coerce")

    # Keep last value on duplicate (timestamp, tag) before pivoting.
    df = df.sort_values(TIMESTAMP_COLUMN, kind="mergesort")
    df = df.drop_duplicates(subset=[TIMESTAMP_COLUMN, TAG_COLUMN], keep="last")

    wide = df.pivot(index=TIMESTAMP_COLUMN, columns=TAG_COLUMN, values=VALUE_COLUMN)
    wide = wide.sort_index(axis=0).sort_index(axis=1)
    wide.columns.name = None
    return wide


def create_lag_features(
    wide_df: pd.DataFrame,
    input_tags: List[str],
    lags: List[int],
) -> pd.DataFrame:
    """Create backward-shifted (lag) features for the given tags.

    For each ``tag`` in ``input_tags`` and each ``lag`` in ``lags`` a column
    ``{tag}_lag_{lag}`` is produced, holding the value observed ``lag`` rows
    earlier (lag units are rows; the input is assumed already resampled to a
    regular grid).

    Parameters
    ----------
    wide_df:
        Wide, timestamp-indexed frame whose columns are tag names.
    input_tags:
        Tags to lag. Defines the primary (outer) ordering of output columns.
    lags:
        Positive lag offsets in rows. Defines the secondary ordering.

    Returns
    -------
    pandas.DataFrame
        Frame indexed like ``wide_df`` containing only the generated lag
        columns, in deterministic ``(tag, lag)`` order.

    Raises
    ------
    ValueError
        If any requested input tag is missing from ``wide_df``.
    """
    _require_tags(wide_df, input_tags, role="input")

    features: dict[str, pd.Series] = {}
    for tag in input_tags:
        for lag in lags:
            features[f"{tag}_lag_{lag}"] = wide_df[tag].shift(lag)

    return pd.DataFrame(features, index=wide_df.index)


def create_rolling_features(
    wide_df: pd.DataFrame,
    input_tags: List[str],
    windows: List[int],
) -> pd.DataFrame:
    """Create rolling-window aggregation features for the given tags.

    For each ``tag`` in ``input_tags`` and each ``window`` in ``windows`` four
    columns are produced::

        {tag}_roll_mean_{window}
        {tag}_roll_std_{window}
        {tag}_roll_min_{window}
        {tag}_roll_max_{window}

    Each window requires a full set of observations (``min_periods == window``)
    so no partial, leakage-prone aggregates are emitted; the resulting leading
    NaNs are expected to be dropped downstream.

    Parameters
    ----------
    wide_df:
        Wide, timestamp-indexed frame whose columns are tag names.
    input_tags:
        Tags to aggregate. Defines the primary (outer) column ordering.
    windows:
        Rolling-window sizes in rows. Defines the secondary ordering.

    Returns
    -------
    pandas.DataFrame
        Frame indexed like ``wide_df`` containing only the generated rolling
        columns, in deterministic ``(tag, window, agg)`` order.

    Raises
    ------
    ValueError
        If any requested input tag is missing from ``wide_df``.
    """
    _require_tags(wide_df, input_tags, role="input")

    features: dict[str, pd.Series] = {}
    for tag in input_tags:
        for window in windows:
            roll = wide_df[tag].rolling(window=window, min_periods=window)
            features[f"{tag}_roll_mean_{window}"] = roll.mean()
            features[f"{tag}_roll_std_{window}"] = roll.std()
            features[f"{tag}_roll_min_{window}"] = roll.min()
            features[f"{tag}_roll_max_{window}"] = roll.max()

    return pd.DataFrame(features, index=wide_df.index)


def create_target(
    wide_df: pd.DataFrame,
    target_tag: str,
    horizon: int,
) -> pd.Series:
    """Build a future-shifted prediction target.

    The target at time ``t`` is the value of ``target_tag`` ``horizon`` rows in
    the future::

        y[t] = target[t + horizon]

    Parameters
    ----------
    wide_df:
        Wide, timestamp-indexed frame whose columns are tag names.
    target_tag:
        Tag to forecast.
    horizon:
        Strictly positive number of rows to look ahead.

    Returns
    -------
    pandas.Series
        Future-shifted target aligned to ``wide_df.index``, named
        ``{target_tag}_h{horizon}``.

    Raises
    ------
    ValueError
        If ``target_tag`` is missing or ``horizon`` is not positive.
    """
    _require_tags(wide_df, [target_tag], role="target")
    if horizon <= 0:
        raise ValueError(f"horizon must be a positive integer, got {horizon}.")

    target = wide_df[target_tag].shift(-horizon)
    target.name = f"{target_tag}_h{horizon}"
    return target


def make_supervised_dataset(
    signals_df: pd.DataFrame,
    target_tag: str,
    input_tags: List[str],
    lags: List[int],
    rolling_windows: List[int],
    horizon: int,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Build an aligned ``(X, y)`` supervised dataset from long-format signals.

    Pipeline
    --------
    1. Pivot signals into a wide, timestamp-indexed frame.
    2. Build lag features for ``input_tags``.
    3. Build rolling-window features for ``input_tags``.
    4. Concatenate features in deterministic order (lags, then rolling).
    5. Build the future-shifted target from ``target_tag``.
    6. Align ``X`` and ``y`` on the timestamp index and drop rows with NaNs
       (from leading lags/rolling windows and the trailing forecast horizon).

    Parameters
    ----------
    signals_df:
        Long-format DataFrame with ``timestamp``, ``tag_name``, ``value``.
    target_tag:
        Tag to forecast.
    input_tags:
        Tags used to build predictive features.
    lags:
        Lag offsets (rows) for lag features.
    rolling_windows:
        Window sizes (rows) for rolling features.
    horizon:
        Forecast horizon (rows) for the target.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.Series]
        ``X`` feature matrix and ``y`` target, sharing an identical index with
        no missing values.

    Raises
    ------
    ValueError
        If any of ``input_tags`` or ``target_tag`` is absent from the pivoted
        signals, or if ``horizon`` is not positive.
    """
    wide = pivot_signals(signals_df)

    # Validate up front for a single, clear error before doing any work.
    _require_tags(wide, input_tags, role="input")
    _require_tags(wide, [target_tag], role="target")

    lag_features = create_lag_features(wide, input_tags, lags)
    rolling_features = create_rolling_features(wide, input_tags, rolling_windows)

    X = pd.concat([lag_features, rolling_features], axis=1)
    y = create_target(wide, target_tag, horizon)

    logger.info(
        "Assembled %d candidate feature columns over %d timestamps",
        X.shape[1],
        X.shape[0],
    )

    # Align X and y, then drop any row containing a NaN in either.
    combined = X.copy()
    combined[y.name] = y
    combined = combined.dropna(axis=0, how="any")

    y_clean = combined[y.name]
    X_clean = combined.drop(columns=[y.name])

    logger.info("Final supervised dataset: %d rows x %d features", *X_clean.shape)
    return X_clean, y_clean
