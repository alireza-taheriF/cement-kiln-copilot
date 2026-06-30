"""Loaders and persisters for raw plant data.

This module is the I/O boundary of the ingestion layer. It is responsible
for:

* reading long-format signal and lab CSV exports into validated DataFrames,
* persisting those DataFrames into the normalized database schema
  (:class:`~src.db.models.Tag`, :class:`~src.db.models.Signal`,
  :class:`~src.db.models.LabResult`).

Validation against the Pydantic contracts in :mod:`src.data_ingest.schemas`
happens here so that downstream cleaning/modeling code can assume clean,
well-typed inputs.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import pandas as pd
from sqlalchemy.orm import Session

from src.db.models import LabResult, Signal, Tag

from .schemas import (
    LAB_CSV_COLUMNS,
    SIGNAL_CSV_COLUMNS,
    LabCSVRow,
    SignalCSVRow,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _require_columns(df: pd.DataFrame, expected: Iterable[str], source: str) -> None:
    """Raise a clear error if any expected column is missing."""
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(
            f"{source} is missing required column(s): {missing}. "
            f"Expected columns: {list(expected)}."
        )


# ---------------------------------------------------------------------------
# CSV loaders
# ---------------------------------------------------------------------------
def load_signals_from_csv(path: str) -> pd.DataFrame:
    """Load a long-format process-signal CSV into a validated DataFrame.

    Expected columns: ``timestamp``, ``tag_name``, ``value``.

    Each row is validated against :class:`SignalCSVRow`, guaranteeing parsed
    timestamps, non-empty tag names, and numeric values.

    Parameters
    ----------
    path:
        Filesystem path to the CSV export.

    Returns
    -------
    pandas.DataFrame
        Columns ``timestamp`` (datetime64), ``tag_name`` (str),
        ``value`` (float), sorted by ``timestamp``.
    """
    path = Path(path)
    logger.info("Loading signals from %s", path)
    df = pd.read_csv(path)
    _require_columns(df, SIGNAL_CSV_COLUMNS, source=str(path))
    df = df[SIGNAL_CSV_COLUMNS].copy()

    records = [SignalCSVRow(**row) for row in df.to_dict(orient="records")]
    out = pd.DataFrame([r.model_dump() for r in records], columns=SIGNAL_CSV_COLUMNS)
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=False)
    out = out.sort_values("timestamp").reset_index(drop=True)
    logger.info("Loaded %d signal rows", len(out))
    return out


def load_lab_from_csv(path: str) -> pd.DataFrame:
    """Load a laboratory QC CSV into a validated DataFrame.

    Expected columns: ``timestamp``, ``parameter``, ``value``, ``sample_id``.

    Each row is validated against :class:`LabCSVRow`.

    Parameters
    ----------
    path:
        Filesystem path to the CSV export.

    Returns
    -------
    pandas.DataFrame
        Columns ``timestamp`` (datetime64), ``parameter`` (str),
        ``value`` (float), ``sample_id`` (str or None), sorted by ``timestamp``.
    """
    path = Path(path)
    logger.info("Loading lab results from %s", path)
    df = pd.read_csv(path)
    _require_columns(df, LAB_CSV_COLUMNS, source=str(path))
    df = df[LAB_CSV_COLUMNS].copy()

    records = [LabCSVRow(**row) for row in df.to_dict(orient="records")]
    out = pd.DataFrame([r.model_dump() for r in records], columns=LAB_CSV_COLUMNS)
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=False)
    out = out.sort_values("timestamp").reset_index(drop=True)
    logger.info("Loaded %d lab rows", len(out))
    return out


# ---------------------------------------------------------------------------
# Database persisters
# ---------------------------------------------------------------------------
def _get_or_create_tag_ids(
    session: Session,
    tag_names: Iterable[str],
) -> dict[str, int]:
    """Map ``tag_name`` -> ``tag_id``, creating missing tags on the fly.

    Newly created tags are assigned the default ``type="process"``; their
    metadata (unit, description, target flag) can be enriched later.
    """
    unique_names = sorted({str(name) for name in tag_names})
    if not unique_names:
        return {}

    existing = (
        session.query(Tag.name, Tag.id)
        .filter(Tag.name.in_(unique_names))
        .all()
    )
    name_to_id: dict[str, int] = {name: tag_id for name, tag_id in existing}

    missing = [name for name in unique_names if name not in name_to_id]
    if missing:
        logger.info("Auto-creating %d new tag(s): %s", len(missing), missing)
        new_tags = [Tag(name=name, type="process") for name in missing]
        session.add_all(new_tags)
        session.flush()  # assign primary keys without committing
        for tag in new_tags:
            name_to_id[tag.name] = tag.id

    return name_to_id


def save_signals_to_db(df: pd.DataFrame, session: Session) -> int:
    """Persist a long-format signal DataFrame into the ``signals`` table.

    Behavior:
      * auto-creates :class:`Tag` rows for any unseen ``tag_name``;
      * maps ``tag_name`` -> ``tag_id``;
      * bulk-inserts the resulting ``(ts, tag_id, value)`` rows.

    Parameters
    ----------
    df:
        DataFrame with columns ``timestamp``, ``tag_name``, ``value``.
    session:
        An active SQLAlchemy session. The function flushes/commits within it.

    Returns
    -------
    int
        Number of signal rows inserted.
    """
    _require_columns(df, SIGNAL_CSV_COLUMNS, source="signals DataFrame")
    if df.empty:
        logger.info("No signal rows to persist.")
        return 0

    name_to_id = _get_or_create_tag_ids(session, df["tag_name"])

    timestamps = pd.to_datetime(df["timestamp"]).dt.to_pydatetime()
    mappings = [
        {
            "ts": ts,
            "tag_id": name_to_id[str(tag_name)],
            "value": float(value),
        }
        for ts, tag_name, value in zip(
            timestamps, df["tag_name"], df["value"]
        )
    ]

    session.bulk_insert_mappings(Signal, mappings)
    session.commit()
    logger.info("Inserted %d signal rows", len(mappings))
    return len(mappings)


def save_lab_to_db(df: pd.DataFrame, session: Session) -> int:
    """Persist a lab-result DataFrame into the ``lab_results`` table.

    Behavior:
      * bulk-inserts ``(ts, parameter, value, sample_id[, notes])`` rows.

    Parameters
    ----------
    df:
        DataFrame with columns ``timestamp``, ``parameter``, ``value`` and
        optionally ``sample_id`` and ``notes``.
    session:
        An active SQLAlchemy session.

    Returns
    -------
    int
        Number of lab rows inserted.
    """
    _require_columns(df, ["timestamp", "parameter", "value"], source="lab DataFrame")
    if df.empty:
        logger.info("No lab rows to persist.")
        return 0

    timestamps = pd.to_datetime(df["timestamp"]).dt.to_pydatetime()
    has_sample = "sample_id" in df.columns
    has_notes = "notes" in df.columns

    mappings = []
    for i, (ts, parameter, value) in enumerate(
        zip(timestamps, df["parameter"], df["value"])
    ):
        sample_id = df["sample_id"].iloc[i] if has_sample else None
        notes = df["notes"].iloc[i] if has_notes else None
        mappings.append(
            {
                "ts": ts,
                "parameter": str(parameter),
                "value": float(value),
                "sample_id": None if pd.isna(sample_id) else str(sample_id),
                "notes": None if pd.isna(notes) else str(notes),
            }
        )

    session.bulk_insert_mappings(LabResult, mappings)
    session.commit()
    logger.info("Inserted %d lab rows", len(mappings))
    return len(mappings)
