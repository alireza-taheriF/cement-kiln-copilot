"""Train the free-lime soft sensor from the database.

Usage::

    python -m src.softsensor.train --out artifacts/models/free_lime_softsensor.joblib

Reads process signals and lab results from ``DATABASE_URL``, resamples the
process data to a regular grid, searches the lab delay, fits
:class:`~src.softsensor.free_lime.FreeLimeSoftSensor` and saves it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from src.domain import tags as T
from src.domain.kpis import FuelProperties, add_derived_kpis, load_plant_config
from src.softsensor.free_lime import FreeLimeSoftSensor

DEFAULT_OUT = "artifacts/models/free_lime_softsensor.joblib"


def load_lab_from_db(session) -> pd.DataFrame:
    from src.db.models import LabResult

    rows = session.query(LabResult.ts, LabResult.parameter, LabResult.value, LabResult.sample_id).all()
    frame = pd.DataFrame(rows, columns=["timestamp", "parameter", "value", "sample_id"])
    if not frame.empty:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True).dt.tz_localize(None)
    return frame


def main() -> None:
    from src.db.models import get_session_factory
    from src.ml.features import pivot_signals
    from src.ml.train import load_signals_from_db

    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--freq", default="5min")
    parser.add_argument("--plant-config", default="config/plant.yaml")
    args = parser.parse_args()

    session = get_session_factory()()
    try:
        signals = load_signals_from_db(session)
        lab = load_lab_from_db(session)
    finally:
        session.close()
    if signals.empty or lab.empty:
        raise SystemExit("Need both process signals and lab results in the database.")
    signals["timestamp"] = pd.to_datetime(signals["timestamp"], utc=True).dt.tz_localize(None)
    wide = pivot_signals(signals).resample(args.freq).mean()
    fuels = (
        load_plant_config(args.plant_config).fuels
        if Path(args.plant_config).exists()
        else FuelProperties()
    )
    wide = add_derived_kpis(wide, fuels)
    sensor = FreeLimeSoftSensor.fit(wide, lab[lab["parameter"] == T.FREE_LIME_PCT])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(sensor, args.out)
    print(json.dumps(sensor.summary(), indent=2))


if __name__ == "__main__":
    main()
