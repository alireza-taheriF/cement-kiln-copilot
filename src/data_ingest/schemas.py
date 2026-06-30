"""Pydantic schemas describing the expected shape of ingested CSV data.

These schemas are the contract between raw data sources (historian/DCS
exports stored in long/tidy format, and LIMS lab exports) and the rest of
the system. Validation happens at the ingestion boundary so downstream
layers can assume clean, well-typed inputs.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class SignalCSVRow(BaseModel):
    """A single row of a long-format process signal CSV export.

    Expected CSV columns: ``timestamp``, ``tag_name``, ``value``.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    timestamp: datetime = Field(..., description="Sample timestamp (tz-aware preferred).")
    tag_name: str = Field(..., min_length=1, description="Unique tag identifier.")
    value: float = Field(..., description="Numeric reading for the tag at timestamp.")


class LabCSVRow(BaseModel):
    """A single row of a laboratory QC CSV export.

    Expected CSV columns: ``timestamp``, ``parameter``, ``value``, ``sample_id``.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    timestamp: datetime = Field(..., description="Sampling/measurement timestamp.")
    parameter: str = Field(..., min_length=1, description="Measured lab parameter.")
    value: float = Field(..., description="Measured numeric value.")
    sample_id: Optional[str] = Field(None, description="Originating lab sample id.")


# Column-name constants — single source of truth for ingestion/cleaning.
SIGNAL_CSV_COLUMNS = list(SignalCSVRow.model_fields.keys())
LAB_CSV_COLUMNS = list(LabCSVRow.model_fields.keys())
