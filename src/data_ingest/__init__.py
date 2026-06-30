"""Data ingestion package.

Responsible for validating, loading, cleaning, aligning, and persisting raw
plant data (long-format process-signal streams and laboratory QC
measurements) before it enters the persistence and ML layers.
"""

from .loader import (
    load_lab_from_csv,
    load_signals_from_csv,
    save_lab_to_db,
    save_signals_to_db,
)
from .cleaner import align_signals_with_lab, clean_signals
from .schemas import LabCSVRow, SignalCSVRow

__all__ = [
    "load_signals_from_csv",
    "load_lab_from_csv",
    "save_signals_to_db",
    "save_lab_to_db",
    "clean_signals",
    "align_signals_with_lab",
    "SignalCSVRow",
    "LabCSVRow",
]
