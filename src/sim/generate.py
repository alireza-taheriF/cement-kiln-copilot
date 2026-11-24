"""Write the simulated demo plant to disk (and optionally seed the database).

Usage::

    python -m src.sim.generate --out data/demo            # CSV + JSON files
    python -m src.sim.generate --out data/demo --seed-db  # also load into DATABASE_URL

Files written to ``--out``:

* ``signals.csv``      long-format process tags (``timestamp, tag_name, value``)
* ``lab.csv``          lab results (``timestamp, parameter, value, sample_id``)
* ``fuel_events.json`` ground-truth fuel transfers
* ``power_quota.csv``  mill power quota per step

The output is deterministic for a given ``--seed``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.sim.plant import SimConfig, simulate


def write_demo(out_dir: str | Path, days: float = 14.0, seed: int = 7) -> dict[str, str]:
    """Simulate and write the demo files; return their paths."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    result = simulate(SimConfig(days=days, seed=seed))

    paths = {
        "signals": out / "signals.csv",
        "lab": out / "lab.csv",
        "fuel_events": out / "fuel_events.json",
        "power_quota": out / "power_quota.csv",
    }
    long = result.to_long(include_derived=False)
    long["value"] = long["value"].round(4)
    long.to_csv(paths["signals"], index=False)
    result.lab.to_csv(paths["lab"], index=False)
    paths["fuel_events"].write_text(
        json.dumps(
            [
                {
                    "start": e.start.isoformat(),
                    "end": e.end.isoformat(),
                    "ramp_minutes": e.ramp_minutes,
                    "direction": e.direction,
                }
                for e in result.fuel_events
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
    result.power_quota_kw.rename_axis("timestamp").reset_index().to_csv(
        paths["power_quota"], index=False
    )
    return {key: str(path) for key, path in paths.items()}


def seed_database(signals_csv: str, lab_csv: str) -> tuple[int, int]:
    """Drop/recreate tables and load the demo CSVs into ``DATABASE_URL``."""
    from src.data_ingest.loader import load_lab_from_csv, load_signals_from_csv
    from src.data_ingest.loader import save_lab_to_db, save_signals_to_db
    from src.db.models import Base, get_engine, get_session_factory

    engine = get_engine()
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    session = get_session_factory(engine)()
    try:
        n_signals = save_signals_to_db(load_signals_from_csv(signals_csv), session)
        n_lab = save_lab_to_db(load_lab_from_csv(lab_csv), session)
    finally:
        session.close()
    return n_signals, n_lab


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--out", default="data/demo")
    parser.add_argument("--days", type=float, default=14.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--seed-db", action="store_true", help="Load the CSVs into DATABASE_URL.")
    args = parser.parse_args()

    paths = write_demo(args.out, days=args.days, seed=args.seed)
    print(json.dumps(paths, indent=2))
    if args.seed_db:
        n_signals, n_lab = seed_database(paths["signals"], paths["lab"])
        print(f"Seeded {n_signals} signal rows and {n_lab} lab rows.")


if __name__ == "__main__":
    main()
