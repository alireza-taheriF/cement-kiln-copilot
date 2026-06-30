"""Tests for the data ingestion layer (loaders, cleaning, alignment, persistence)."""

from __future__ import annotations

import pandas as pd

from src.data_ingest.cleaner import align_signals_with_lab, clean_signals
from src.data_ingest.loader import (
    load_lab_from_csv,
    load_signals_from_csv,
    save_lab_to_db,
    save_signals_to_db,
)
from src.data_ingest.schemas import LAB_CSV_COLUMNS, SIGNAL_CSV_COLUMNS
from src.db.models import LabResult, Signal, Tag, get_engine, get_session_factory, init_db


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------
def _sample_signals() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-01-01 00:02",
                    "2026-01-01 00:00",
                    "2026-01-01 00:00",  # duplicate (timestamp, tag)
                    "2026-01-01 00:01",
                ]
            ),
            "tag_name": [
                "fuel_rate_tph",
                "fuel_rate_tph",
                "fuel_rate_tph",
                "fuel_rate_tph",
            ],
            "value": [11.4, 11.2, 11.2, None],
        }
    )


def _in_memory_session():
    engine = get_engine("sqlite:///:memory:")
    init_db(engine)
    return get_session_factory(engine)()


# ---------------------------------------------------------------------------
# Schema constants
# ---------------------------------------------------------------------------
def test_schema_column_constants_present():
    assert SIGNAL_CSV_COLUMNS == ["timestamp", "tag_name", "value"]
    assert LAB_CSV_COLUMNS == ["timestamp", "parameter", "value", "sample_id"]


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------
def test_clean_signals_sorts_and_drops_duplicates():
    df = clean_signals(_sample_signals())
    assert df["timestamp"].is_monotonic_increasing
    # 4 input rows minus 1 duplicate (timestamp, tag) = 3.
    assert len(df) == 3


def test_clean_signals_interpolates_small_gaps():
    df = clean_signals(_sample_signals())
    assert df["value"].isna().sum() == 0


def test_clean_signals_clips_outliers_by_zscore():
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=101, freq="min"),
            "tag_name": ["o2_pct"] * 101,
            "value": [2.8] * 100 + [50.0],  # last value is a gross spike
        }
    )
    out = clean_signals(df, zscore_threshold=4.0)
    assert out["value"].max() < 50.0


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------
def test_align_signals_with_lab_matches_within_tolerance():
    signals = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-01 00:00", "2026-01-01 02:00"]),
            "value": [11.2, 11.5],
        }
    )
    lab = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-01 00:10"]),
            "parameter": ["free_lime"],
            "value": [1.1],
        }
    )
    merged = align_signals_with_lab(signals, lab)
    # First signal is 10 min from the lab sample -> matched.
    assert merged.loc[0, "parameter"] == "free_lime"
    # Second signal is ~110 min away -> outside ±30 min, no match.
    assert pd.isna(merged.loc[1, "parameter"])


# ---------------------------------------------------------------------------
# CSV loaders
# ---------------------------------------------------------------------------
def test_load_signals_from_csv(tmp_path):
    csv = tmp_path / "signals.csv"
    csv.write_text(
        "timestamp,tag_name,value\n"
        "2026-01-01 00:01,fuel_rate_tph,11.3\n"
        "2026-01-01 00:00,fuel_rate_tph,11.2\n"
    )
    df = load_signals_from_csv(str(csv))
    assert list(df.columns) == SIGNAL_CSV_COLUMNS
    assert df["timestamp"].is_monotonic_increasing
    assert len(df) == 2


def test_load_lab_from_csv(tmp_path):
    csv = tmp_path / "lab.csv"
    csv.write_text(
        "timestamp,parameter,value,sample_id\n"
        "2026-01-01 00:00,free_lime,1.12,LAB-1\n"
    )
    df = load_lab_from_csv(str(csv))
    assert list(df.columns) == LAB_CSV_COLUMNS
    assert df.loc[0, "parameter"] == "free_lime"


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def test_save_signals_to_db_autocreates_tags():
    session = _in_memory_session()
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-01 00:00", "2026-01-01 00:01"]),
            "tag_name": ["fuel_rate_tph", "o2_pct"],
            "value": [11.2, 2.8],
        }
    )
    inserted = save_signals_to_db(df, session)
    assert inserted == 2
    assert session.query(Tag).count() == 2
    assert session.query(Signal).count() == 2


def test_save_lab_to_db_bulk_insert():
    session = _in_memory_session()
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-01 00:00", "2026-01-01 01:00"]),
            "parameter": ["free_lime", "free_lime"],
            "value": [1.12, 1.05],
            "sample_id": ["LAB-1", "LAB-2"],
        }
    )
    inserted = save_lab_to_db(df, session)
    assert inserted == 2
    assert session.query(LabResult).count() == 2
