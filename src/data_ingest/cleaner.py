"""Cleaning and alignment utilities for ingested plant data.

Cleaning is intentionally conservative: it handles obvious data-quality
issues (ordering, duplicates, sensor spikes, short gaps) on the long-format
signal stream and leaves modeling decisions to the ML layer.

The alignment helper joins irregular lab samples onto the signal stream
using a nearest-timestamp ("as-of") merge.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: z-score magnitude beyond which a value is treated as an outlier.
DEFAULT_ZSCORE_THRESHOLD = 4.0

#: maximum lab/signal timestamp gap tolerated during as-of alignment.
DEFAULT_LAB_TOLERANCE = pd.Timedelta(minutes=30)


def _clip_outliers_by_zscore(values: pd.Series, threshold: float) -> pd.Series:
    """Clip values whose z-score magnitude exceeds ``threshold``.

    Outliers are clipped (winsorized) to the threshold boundary rather than
    dropped, preserving the time index. A zero/NaN standard deviation (a
    flat-line series) is treated as having no outliers.
    """
    mean = values.mean()
    std = values.std(ddof=0)
    if not np.isfinite(std) or std == 0:
        return values
    lower = mean - threshold * std
    upper = mean + threshold * std
    return values.clip(lower=lower, upper=upper)


def clean_signals(
    df: pd.DataFrame,
    timestamp_column: str = "timestamp",
    tag_column: str = "tag_name",
    value_column: str = "value",
    zscore_threshold: float = DEFAULT_ZSCORE_THRESHOLD,
) -> pd.DataFrame:
    """Clean a long-format signal DataFrame.

    Steps (applied per tag where relevant):
      1. Sort by timestamp.
      2. Remove duplicate ``(timestamp, tag)`` rows (keep last).
      3. Clip outliers using a z-score threshold (default 4).
      4. Interpolate small gaps in ``value``.

    Parameters
    ----------
    df:
        Long-format DataFrame with timestamp, tag, and value columns.
    timestamp_column, tag_column, value_column:
        Column names, overridable for non-default schemas.
    zscore_threshold:
        Magnitude of the z-score beyond which values are clipped.

    Returns
    -------
    pandas.DataFrame
        Cleaned DataFrame, sorted by timestamp, index reset.
    """
    df = df.copy()
    df[timestamp_column] = pd.to_datetime(df[timestamp_column])
    df[value_column] = pd.to_numeric(df[value_column], errors="coerce")

    # 1. Sort by timestamp (stable so per-tag order is preserved).
    df = df.sort_values(timestamp_column, kind="mergesort")

    # 2. Remove duplicate (timestamp, tag) rows, keeping the latest reading.
    before = len(df)
    df = df.drop_duplicates(subset=[timestamp_column, tag_column], keep="last")
    dropped = before - len(df)
    if dropped:
        logger.warning("Dropped %d duplicate (timestamp, tag) row(s)", dropped)

    # 3 & 4. Per-tag outlier clipping + small-gap interpolation.
    def _process_group(group: pd.DataFrame) -> pd.DataFrame:
        group = group.sort_values(timestamp_column, kind="mergesort")
        group[value_column] = _clip_outliers_by_zscore(
            group[value_column], zscore_threshold
        )
        group[value_column] = (
            group[value_column]
            .interpolate(method="linear", limit=3, limit_direction="both")
        )
        return group

    df = (
        df.groupby(tag_column, group_keys=False, sort=False)
        .apply(_process_group)
        .reset_index(drop=True)
    )

    df = df.sort_values(timestamp_column, kind="mergesort").reset_index(drop=True)
    return df


def align_signals_with_lab(
    signals_df: pd.DataFrame,
    lab_df: pd.DataFrame,
    timestamp_column: str = "timestamp",
    tolerance: pd.Timedelta = DEFAULT_LAB_TOLERANCE,
) -> pd.DataFrame:
    """Align lab samples onto the signal stream via a nearest as-of merge.

    Each signal row is matched to the nearest lab sample within ``±tolerance``
    (default ±30 minutes) using :func:`pandas.merge_asof`. Lab columns are
    suffixed with ``_lab`` to avoid collisions with signal columns.

    Parameters
    ----------
    signals_df:
        Signal DataFrame (long or wide), sorted/sortable by timestamp.
    lab_df:
        Lab DataFrame containing a timestamp column.
    timestamp_column:
        Name of the timestamp column present in both frames.
    tolerance:
        Maximum absolute time difference for a match.

    Returns
    -------
    pandas.DataFrame
        ``signals_df`` rows enriched with the nearest lab sample's columns.
    """
    left = signals_df.copy()
    right = lab_df.copy()
    left[timestamp_column] = pd.to_datetime(left[timestamp_column])
    right[timestamp_column] = pd.to_datetime(right[timestamp_column])

    left = left.sort_values(timestamp_column, kind="mergesort").reset_index(drop=True)
    right = right.sort_values(timestamp_column, kind="mergesort").reset_index(drop=True)

    merged = pd.merge_asof(
        left,
        right,
        on=timestamp_column,
        direction="nearest",
        tolerance=tolerance,
        suffixes=("", "_lab"),
    )
    return merged
