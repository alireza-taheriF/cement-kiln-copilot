"""Request schemas shared by the v1 routers."""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
from pydantic import BaseModel, Field, model_validator


class SignalWindow(BaseModel):
    """A window of process signals on one timestamp axis (canonical tags)."""

    timestamps: List[datetime] = Field(..., min_length=2)
    signals: Dict[str, List[Optional[float]]] = Field(...)

    @model_validator(mode="after")
    def _validate_shapes(self) -> "SignalWindow":
        if not self.signals:
            raise ValueError("signals must contain at least one tag.")
        n = len(self.timestamps)
        for tag, values in self.signals.items():
            if not tag or not tag.strip():
                raise ValueError("signal tag names must be non-empty.")
            if len(values) != n:
                raise ValueError(
                    f"signal {tag!r} has {len(values)} values but there are {n} timestamps."
                )
        return self

    def to_wide(self) -> pd.DataFrame:
        """Timestamp-indexed frame (timezones dropped to plant-local naive time)."""
        index = pd.DatetimeIndex(pd.to_datetime(self.timestamps))
        if index.tz is not None:
            index = index.tz_convert("UTC").tz_localize(None)
        frame = pd.DataFrame(
            {tag: pd.to_numeric(pd.Series(values, dtype="float64")).to_numpy() for tag, values in self.signals.items()},
            index=index,
        )
        frame.index.name = "timestamp"
        return frame.sort_index()


class LabRow(BaseModel):
    timestamp: datetime
    parameter: str
    value: float
    sample_id: Optional[str] = None


def lab_frame(rows: Optional[List[LabRow]]) -> Optional[pd.DataFrame]:
    if not rows:
        return None
    frame = pd.DataFrame([r.model_dump() for r in rows])
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True).dt.tz_localize(None)
    return frame
