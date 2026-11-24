"""Plant tag names → canonical tags, with unit conversion and quality checks.

``config/connectors.yaml`` holds one entry per plant tag::

    tags:
      K1_BZT_PV:     {canonical: BZ_TEMP_C}
      K1_MAZUT_TPH:  {canonical: MAZUT_FLOW_KGH, scale: 1000}   # t/h -> kg/h
      K1_O2_RAW:     {canonical: KILN_INLET_O2_PCT, valid: [0, 21]}

``canonical = scale * plant + offset``. Values outside ``valid`` (sensor
faults, bad quality) are dropped rather than clipped, so a stuck or
saturated transmitter never looks like a real process value. Unmapped plant
tags are dropped and reported.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from src.domain.tags import TAGS

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TagMapping:
    plant_tag: str
    canonical: str
    scale: float = 1.0
    offset: float = 0.0
    valid: Optional[Tuple[float, float]] = None


@dataclass
class MappingReport:
    """What the mapping did, for the data-audit report."""

    rows_in: int
    rows_out: int
    unmapped_tags: List[str]
    invalid_rows: Dict[str, int]
    unknown_canonical: List[str]


class TagMap:
    """Apply plant → canonical tag mappings to long-format signals."""

    def __init__(self, mappings: Mapping[str, TagMapping]) -> None:
        self.mappings = dict(mappings)

    @classmethod
    def from_config(cls, raw: Mapping) -> "TagMap":
        out: Dict[str, TagMapping] = {}
        for plant_tag, spec in (raw or {}).items():
            valid = spec.get("valid")
            out[str(plant_tag)] = TagMapping(
                plant_tag=str(plant_tag),
                canonical=str(spec["canonical"]),
                scale=float(spec.get("scale", 1.0)),
                offset=float(spec.get("offset", 0.0)),
                valid=(float(valid[0]), float(valid[1])) if valid else None,
            )
        return cls(out)

    def apply(self, signals: pd.DataFrame) -> tuple[pd.DataFrame, MappingReport]:
        """Return canonical long-format signals and a report."""
        rows_in = len(signals)
        present = sorted(signals["tag_name"].astype(str).unique()) if rows_in else []
        unmapped = [t for t in present if t not in self.mappings]
        frames = []
        invalid: Dict[str, int] = {}
        for plant_tag, mapping in self.mappings.items():
            part = signals[signals["tag_name"] == plant_tag]
            if part.empty:
                continue
            values = part["value"].astype(float) * mapping.scale + mapping.offset
            keep = np.isfinite(values)
            if mapping.valid is not None:
                lo, hi = mapping.valid
                keep &= (values >= lo) & (values <= hi)
            if (~keep).sum():
                invalid[plant_tag] = int((~keep).sum())
            frames.append(
                pd.DataFrame(
                    {
                        "timestamp": part["timestamp"][keep].to_numpy(),
                        "tag_name": mapping.canonical,
                        "value": values[keep].to_numpy(),
                    }
                )
            )
        out = (
            pd.concat(frames, ignore_index=True)
            if frames
            else pd.DataFrame(columns=["timestamp", "tag_name", "value"])
        )
        unknown = sorted({m.canonical for m in self.mappings.values()} - set(TAGS))
        if unmapped:
            logger.info("Dropping %d unmapped plant tag(s): %s", len(unmapped), unmapped[:20])
        return out, MappingReport(rows_in, len(out), unmapped, invalid, unknown)


def resample_regular(signals: pd.DataFrame, freq: str = "5min", max_gap: int = 3) -> pd.DataFrame:
    """Wide frame on a regular grid: mean per bin, short gaps interpolated.

    Gaps longer than ``max_gap`` bins stay NaN (a kiln stop or a dead
    transmitter must not be painted over).
    """
    if signals.empty:
        return pd.DataFrame()
    frame = signals.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    wide = frame.pivot_table(index="timestamp", columns="tag_name", values="value", aggfunc="mean")
    wide = wide.resample(freq).mean()
    wide = wide.interpolate(method="time", limit=max_gap, limit_area="inside")
    wide.columns.name = None
    return wide
