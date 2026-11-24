"""Read-only connectors: Jalali dates, CSV/Excel exports, tag mapping."""

from __future__ import annotations

import ast
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import pytest

from src.connectors.jalali import jalali_to_gregorian, normalize_digits, parse_jalali_datetime
from src.connectors.mapping import TagMap, resample_regular
from src.connectors.sources import CsvExportSource, ExcelLabSource, build_source, parse_timestamps
from src.connectors.sync import collect

PACKAGE = Path(__file__).resolve().parents[1] / "src" / "connectors"


@pytest.mark.parametrize(
    "jalali, gregorian",
    [
        ((1405, 9, 3), date(2026, 11, 24)),
        ((1404, 1, 1), date(2025, 3, 21)),  # Nowruz
        ((1403, 12, 30), date(2025, 3, 20)),  # leap year end
        ((1399, 12, 30), date(2021, 3, 20)),
        ((1400, 7, 1), date(2021, 9, 23)),
    ],
)
def test_jalali_conversion(jalali, gregorian):
    assert jalali_to_gregorian(*jalali) == gregorian


def test_persian_digits_and_datetime():
    assert normalize_digits("۱۲۳٤٥") == "12345"
    assert parse_jalali_datetime("۱۴۰۵/۰۹/۰۳ ۰۸:۳۰") == datetime(2026, 11, 24, 8, 30)
    with pytest.raises(ValueError):
        parse_jalali_datetime("2026-11-24")


def test_parse_timestamps_with_timezone():
    out = parse_timestamps(pd.Series(["1405/09/03 12:00"]), timezone="Asia/Tehran")
    assert out.iloc[0] == pd.Timestamp("2026-11-24 08:30")  # UTC+03:30


def test_wide_csv_export_with_persian_numbers(tmp_path):
    (tmp_path / "day1.csv").write_text(
        "Time,K1_BZT_PV,K1_MAZUT_TPH\n1405/09/03 08:00,۱۴۵۰٫۵,\"1,2\"\n1405/09/03 08:05,1451,1.3\n",
        encoding="utf-8",
    )
    src = CsvExportSource(str(tmp_path / "*.csv"), layout="wide", timestamp_column="Time")
    long = src.read_signals()
    assert len(long) == 4
    bzt = long[long["tag_name"] == "K1_BZT_PV"]["value"].tolist()
    assert bzt == [1450.5, 1451.0]
    assert long[long["tag_name"] == "K1_MAZUT_TPH"]["value"].tolist() == [12.0, 1.3]


def test_tag_map_scales_and_rejects_bad_values():
    signals = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=4, freq="5min"),
            "tag_name": ["K1_MAZUT_TPH", "K1_MAZUT_TPH", "K1_O2", "UNUSED"],
            "value": [1.5, 99.0, 2.1, 7.0],
        }
    )
    tag_map = TagMap.from_config(
        {
            "K1_MAZUT_TPH": {"canonical": "MAZUT_FLOW_KGH", "scale": 1000, "valid": [0, 15000]},
            "K1_O2": {"canonical": "KILN_INLET_O2_PCT"},
        }
    )
    out, report = tag_map.apply(signals)
    assert out[out["tag_name"] == "MAZUT_FLOW_KGH"]["value"].tolist() == [1500.0]
    assert report.invalid_rows == {"K1_MAZUT_TPH": 1}
    assert report.unmapped_tags == ["UNUSED"]


def test_resample_keeps_long_gaps_open():
    ts = pd.to_datetime(["2026-01-01 00:00", "2026-01-01 00:05", "2026-01-01 01:00"])
    signals = pd.DataFrame({"timestamp": ts, "tag_name": "X", "value": [1.0, 2.0, 3.0]})
    wide = resample_regular(signals, "5min", max_gap=3)
    assert wide["X"].isna().sum() > 0  # 55 min gap is not interpolated


def test_excel_lab_with_jalali_dates(tmp_path):
    path = tmp_path / "lab.xlsx"
    pd.DataFrame(
        {"تاریخ": ["1405/09/03", "1405/09/03"], "ساعت": ["08:00", "10:00"], "آهک آزاد": ["۱٫۲", 0.9], "شماره": ["a", "b"]}
    ).to_excel(path, index=False)
    lab = ExcelLabSource(
        str(path), "تاریخ", {"آهک آزاد": "FREE_LIME_PCT"}, time_column="ساعت", sample_id_column="شماره"
    ).read_lab()
    assert lab["value"].tolist() == [1.2, 0.9]
    assert lab["timestamp"].iloc[1] == pd.Timestamp("2026-11-24 10:00")
    assert set(lab["parameter"]) == {"FREE_LIME_PCT"}


def test_collect_end_to_end(tmp_path):
    (tmp_path / "x.csv").write_text("Time,K1_BZT_PV\n2026-01-01 00:00,1450\n", encoding="utf-8")
    config = {
        "sources": {"h": {"type": "csv_export", "pattern": str(tmp_path / "*.csv"), "timestamp_column": "Time"}},
        "tags": {"K1_BZT_PV": {"canonical": "BZ_TEMP_C"}},
    }
    result = collect(config)
    assert result["signals"]["tag_name"].tolist() == ["BZ_TEMP_C"]
    assert result["report"]["signal_rows_out"] == 1


def test_unknown_source_type():
    with pytest.raises(ValueError):
        build_source("x", {"type": "modbus"})


def test_connectors_never_write_to_the_plant():
    """No connector may call a write/actuation service of any protocol client."""
    forbidden = {"write_value", "write_values", "set_value", "write_attribute", "write_params", "call_method"}
    for path in PACKAGE.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                assert node.attr not in forbidden, f"{path.name} uses {node.attr}"
            if isinstance(node, ast.FunctionDef):
                assert not node.name.startswith(("write", "set_point", "actuate")), f"{path.name}: {node.name}"
