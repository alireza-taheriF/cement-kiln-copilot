"""Read-only plant data sources.

Every source implements :class:`ReadOnlySource`, whose interface has **only
read methods**. There is deliberately no write, no ``set_value`` and no
method call path that could change anything in the plant's DCS or
historian; ``tests/test_connectors.py`` checks that none appears in this
package. The copilot is advisory: it reads, the operator acts.

Sources
-------
* :class:`CsvExportSource` - historian/DCS CSV exports dropped in a folder
  (wide or long format), the most common first integration step because it
  needs no network change at the plant.
* :class:`ExcelLabSource` - lab results kept in Excel, with Jalali dates
  and Persian digits handled.
* :class:`OpcUaSource` - live/historical reads from an OPC UA server
  (optional dependency ``asyncua``). Connect it with a server account that
  is read-only on the server side as well.

All sources return long-format frames (``timestamp, tag_name, value`` for
signals; ``timestamp, parameter, value, sample_id`` for lab) in **plant**
tag names; :class:`~src.connectors.mapping.TagMap` converts them to
canonical tags and units.
"""

from __future__ import annotations

import abc
import glob
import logging
from datetime import datetime
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import pandas as pd

from src.connectors.jalali import looks_jalali, normalize_digits, parse_jalali_datetime

logger = logging.getLogger(__name__)

SIGNAL_COLUMNS = ["timestamp", "tag_name", "value"]
LAB_COLUMNS = ["timestamp", "parameter", "value", "sample_id"]


class ReadOnlySource(abc.ABC):
    """A plant data source that can only be read."""

    name: str

    @abc.abstractmethod
    def read_signals(
        self,
        tags: Optional[Sequence[str]] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Long-format signals in plant tag names."""

    def read_lab(
        self, start: Optional[datetime] = None, end: Optional[datetime] = None
    ) -> pd.DataFrame:
        """Long-format lab results; sources without lab data return an empty frame."""
        return pd.DataFrame(columns=LAB_COLUMNS)


def parse_timestamps(values: pd.Series, timezone: Optional[str] = None) -> pd.Series:
    """Parse Gregorian or Jalali timestamps (any digit script) to naive local time.

    With ``timezone`` set, timestamps are interpreted in that zone and
    converted to UTC, then made naive, so all sources share one clock.
    """
    sample = values.dropna().astype(str)
    if not sample.empty and looks_jalali(sample.iloc[0]):
        parsed = sample.map(parse_jalali_datetime).reindex(values.index)
        ts = pd.to_datetime(parsed)
    else:
        ts = pd.to_datetime(values.astype(str).map(normalize_digits), errors="coerce")
    if timezone:
        ts = ts.dt.tz_localize(timezone, ambiguous="NaT", nonexistent="NaT").dt.tz_convert("UTC").dt.tz_localize(None)
    return ts


def _to_number(values: pd.Series) -> pd.Series:
    """Numbers from strings with Persian digits, thousands commas and the Persian decimal mark."""
    if values.dtype.kind in "if":
        return values.astype(float)
    text = values.astype(str).map(normalize_digits).str.replace("٫", ".", regex=False)
    text = text.str.replace(",", "", regex=False).str.replace("٬", "", regex=False).str.strip()
    return pd.to_numeric(text, errors="coerce")


def _filter(frame: pd.DataFrame, tags, start, end, tag_col: str = "tag_name") -> pd.DataFrame:
    if tags is not None:
        frame = frame[frame[tag_col].isin(list(tags))]
    if start is not None:
        frame = frame[frame["timestamp"] >= pd.Timestamp(start)]
    if end is not None:
        frame = frame[frame["timestamp"] <= pd.Timestamp(end)]
    return frame.reset_index(drop=True)


class CsvExportSource(ReadOnlySource):
    """Historian/DCS CSV exports in a folder.

    Parameters
    ----------
    pattern:
        Glob of export files, e.g. ``/data/exports/*.csv``.
    layout:
        ``"wide"`` (one column per tag) or ``"long"`` (``tag``/``value`` columns).
    timestamp_column, tag_column, value_column:
        Column names in the export.
    timezone:
        Zone the export's clock is in (e.g. ``Asia/Tehran``); ``None`` keeps
        timestamps as they are.
    """

    def __init__(
        self,
        pattern: str,
        layout: str = "wide",
        timestamp_column: str = "timestamp",
        tag_column: str = "tag_name",
        value_column: str = "value",
        timezone: Optional[str] = None,
        name: str = "csv_export",
        **read_csv_kwargs,
    ) -> None:
        if layout not in ("wide", "long"):
            raise ValueError("layout must be 'wide' or 'long'.")
        self.pattern = pattern
        self.layout = layout
        self.timestamp_column = timestamp_column
        self.tag_column = tag_column
        self.value_column = value_column
        self.timezone = timezone
        self.name = name
        self.read_csv_kwargs = read_csv_kwargs

    def files(self) -> List[str]:
        return sorted(glob.glob(self.pattern))

    def read_signals(self, tags=None, start=None, end=None) -> pd.DataFrame:
        frames = []
        for path in self.files():
            raw = pd.read_csv(path, **self.read_csv_kwargs)
            if self.timestamp_column not in raw:
                raise ValueError(f"{path}: no timestamp column {self.timestamp_column!r}.")
            if self.layout == "wide":
                long = raw.melt(
                    id_vars=[self.timestamp_column], var_name="tag_name", value_name="value"
                )
            else:
                long = raw.rename(
                    columns={self.tag_column: "tag_name", self.value_column: "value"}
                )[[self.timestamp_column, "tag_name", "value"]]
            long = long.rename(columns={self.timestamp_column: "timestamp"})
            long["timestamp"] = parse_timestamps(long["timestamp"], self.timezone)
            long["value"] = _to_number(long["value"])
            long["tag_name"] = long["tag_name"].astype(str).str.strip()
            frames.append(long.dropna(subset=["timestamp", "value"]))
        if not frames:
            logger.warning("No files match %s", self.pattern)
            return pd.DataFrame(columns=SIGNAL_COLUMNS)
        out = pd.concat(frames, ignore_index=True).sort_values("timestamp", kind="mergesort")
        out = out.drop_duplicates(subset=["timestamp", "tag_name"], keep="last")
        return _filter(out[SIGNAL_COLUMNS], tags, start, end)


class ExcelLabSource(ReadOnlySource):
    """Lab results kept in an Excel workbook (one row per sample).

    Parameters
    ----------
    path:
        Workbook path.
    timestamp_column:
        Column holding the sampling time (Gregorian or Jalali). If the date
        and the time are in two columns, give ``time_column`` too.
    parameters:
        ``{excel column: canonical parameter}``, e.g.
        ``{"آهک آزاد": "FREE_LIME_PCT"}``.
    """

    def __init__(
        self,
        path: str,
        timestamp_column: str,
        parameters: Mapping[str, str],
        sheet: str | int = 0,
        time_column: Optional[str] = None,
        sample_id_column: Optional[str] = None,
        timezone: Optional[str] = None,
        name: str = "excel_lab",
    ) -> None:
        self.path = path
        self.sheet = sheet
        self.timestamp_column = timestamp_column
        self.time_column = time_column
        self.parameters = dict(parameters)
        self.sample_id_column = sample_id_column
        self.timezone = timezone
        self.name = name

    def read_signals(self, tags=None, start=None, end=None) -> pd.DataFrame:
        return pd.DataFrame(columns=SIGNAL_COLUMNS)

    def read_lab(self, start=None, end=None) -> pd.DataFrame:
        raw = pd.read_excel(self.path, sheet_name=self.sheet)
        if self.timestamp_column not in raw:
            raise ValueError(f"{self.path}: no column {self.timestamp_column!r}.")
        stamp = raw[self.timestamp_column].astype(str)
        if self.time_column:
            stamp = stamp.str.split().str[0] + " " + raw[self.time_column].astype(str)
        timestamps = parse_timestamps(stamp, self.timezone)
        frames = []
        for column, parameter in self.parameters.items():
            if column not in raw:
                raise ValueError(f"{self.path}: no column {column!r}.")
            frames.append(
                pd.DataFrame(
                    {
                        "timestamp": timestamps,
                        "parameter": parameter,
                        "value": _to_number(raw[column]),
                        "sample_id": raw[self.sample_id_column].astype(str)
                        if self.sample_id_column
                        else None,
                    }
                )
            )
        out = pd.concat(frames, ignore_index=True).dropna(subset=["timestamp", "value"])
        return _filter(out[LAB_COLUMNS], None, start, end, tag_col="parameter")


class OpcUaSource(ReadOnlySource):
    """Historical reads from an OPC UA server (requires ``pip install asyncua``).

    ``node_ids`` maps plant tag names to OPC UA node ids
    (``ns=2;s=K1.BZT.PV``). Only the history-read service is used.
    """

    def __init__(
        self,
        endpoint: str,
        node_ids: Mapping[str, str],
        username: Optional[str] = None,
        password: Optional[str] = None,
        timeout_s: float = 10.0,
        name: str = "opcua",
    ) -> None:
        self.endpoint = endpoint
        self.node_ids = dict(node_ids)
        self.username = username
        self.password = password
        self.timeout_s = timeout_s
        self.name = name

    def read_signals(self, tags=None, start=None, end=None) -> pd.DataFrame:
        import asyncio

        try:
            from asyncua import Client  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("OpcUaSource needs the optional 'asyncua' package.") from exc
        if start is None or end is None:
            raise ValueError("OPC UA history reads need both start and end.")
        wanted = list(tags) if tags is not None else list(self.node_ids)

        async def _read() -> List[dict]:
            rows: List[dict] = []
            client = Client(url=self.endpoint, timeout=self.timeout_s)
            if self.username:
                client.set_user(self.username)
                client.set_password(self.password or "")
            async with client:
                for tag in wanted:
                    node = client.get_node(self.node_ids[tag])
                    history = await node.read_raw_history(start, end)
                    for dv in history:
                        if dv.Value is None or dv.SourceTimestamp is None:
                            continue
                        rows.append(
                            {
                                "timestamp": dv.SourceTimestamp,
                                "tag_name": tag,
                                "value": float(dv.Value.Value),
                            }
                        )
            return rows

        rows = asyncio.run(_read())
        frame = pd.DataFrame(rows, columns=SIGNAL_COLUMNS)
        if not frame.empty:
            frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True).dt.tz_localize(None)
        return frame


def build_source(name: str, spec: Mapping) -> ReadOnlySource:
    """Instantiate a source from a ``config/connectors.yaml`` entry."""
    kind = spec.get("type")
    options = {k: v for k, v in spec.items() if k not in ("type", "enabled")}
    if kind == "csv_export":
        return CsvExportSource(name=name, **options)
    if kind == "excel_lab":
        return ExcelLabSource(name=name, **options)
    if kind == "opcua":
        return OpcUaSource(name=name, **options)
    raise ValueError(f"source {name!r}: unknown type {kind!r}.")


def read_all(sources: Iterable[ReadOnlySource], start=None, end=None) -> Dict[str, pd.DataFrame]:
    """Concatenate signals and lab results from every source."""
    signals, lab = [], []
    for source in sources:
        signals.append(source.read_signals(start=start, end=end))
        lab.append(source.read_lab(start=start, end=end))
    return {
        "signals": pd.concat(signals, ignore_index=True) if signals else pd.DataFrame(columns=SIGNAL_COLUMNS),
        "lab": pd.concat(lab, ignore_index=True) if lab else pd.DataFrame(columns=LAB_COLUMNS),
    }
