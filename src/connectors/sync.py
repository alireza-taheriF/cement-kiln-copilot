"""Pull plant data through the read-only sources into the copilot database.

Usage::

    python -m src.connectors.sync --config config/connectors.yaml --since 2026-11-01

Reads every enabled source, maps plant tags to canonical tags, writes
signals and lab results to ``DATABASE_URL``, and prints a data-quality
report (rows in/out, unmapped tags, rejected values). Run it from cron or a
scheduled task every few minutes; it only ever reads from the plant.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

from src.connectors.mapping import TagMap
from src.connectors.sources import build_source, read_all


def load_config(path: str | Path) -> dict:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if "sources" not in raw:
        raise ValueError(f"{path}: missing 'sources'.")
    return raw


def collect(config: dict, since: Optional[str] = None, until: Optional[str] = None) -> dict:
    """Read and map all enabled sources; return frames plus a report."""
    sources = [
        build_source(name, spec)
        for name, spec in config["sources"].items()
        if spec.get("enabled", True)
    ]
    start = pd.Timestamp(since) if since else None
    end = pd.Timestamp(until) if until else None
    raw = read_all(sources, start=start, end=end)
    tag_map = TagMap.from_config(config.get("tags", {}))
    signals, report = tag_map.apply(raw["signals"])
    return {
        "signals": signals,
        "lab": raw["lab"],
        "report": {
            "sources": [s.name for s in sources],
            "signal_rows_in": report.rows_in,
            "signal_rows_out": report.rows_out,
            "unmapped_tags": report.unmapped_tags,
            "rejected_values": report.invalid_rows,
            "unknown_canonical_tags": report.unknown_canonical,
            "lab_rows": int(len(raw["lab"])),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--config", default="config/connectors.yaml")
    parser.add_argument("--since")
    parser.add_argument("--until")
    parser.add_argument("--dry-run", action="store_true", help="Report only; write nothing.")
    args = parser.parse_args()

    result = collect(load_config(args.config), args.since, args.until)
    if not args.dry_run:
        from src.data_ingest.loader import save_lab_to_db, save_signals_to_db
        from src.db.models import get_session_factory, init_db

        init_db()
        session = get_session_factory()()
        try:
            result["report"]["signal_rows_written"] = save_signals_to_db(
                result["signals"], session, skip_existing=True
            )
            result["report"]["lab_rows_written"] = save_lab_to_db(result["lab"], session)
        finally:
            session.close()
    print(json.dumps(result["report"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
