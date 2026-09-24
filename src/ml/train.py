"""End-to-end training pipeline for cement-process KPI forecasting.

This module wires together the existing building blocks into a single,
production-grade trainer:

* read process signals from the database (long format),
* assemble a supervised dataset with :mod:`src.ml.features`,
* perform a chronological train/validation split,
* fit an :class:`~src.ml.models.EnergyKPIModel`,
* evaluate on both splits,
* persist the model artifact and return a structured training summary.

Run it as a module::

    python -m src.ml.train --config config/model_config.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml
from sqlalchemy.orm import Session

from src.db.models import Signal, Tag, get_engine, get_session_factory
from src.ml.evaluate import evaluate_regressor, train_valid_split_time_series
from src.ml.feature_stats import compute_feature_stats, feature_stats_path, write_feature_stats
from src.ml.features import make_supervised_dataset
from src.ml.models import EnergyKPIModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

#: Fraction of the most-recent rows held out for validation.
DEFAULT_VALID_FRACTION = 0.2

#: Output columns of :func:`load_signals_from_db`.
SIGNAL_COLUMNS = ["timestamp", "tag_name", "value"]

#: Keys that must be present in a training configuration file.
REQUIRED_CONFIG_KEYS = (
    "target_tag",
    "input_tags",
    "lags",
    "rolling_windows",
    "horizon",
    "model_output_path",
)


def load_signals_from_db(
    session: Session,
    tag_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Load long-format process signals from the database.

    Joins :class:`~src.db.models.Signal` with :class:`~src.db.models.Tag` and
    returns one row per ``(timestamp, tag)`` value.

    Parameters
    ----------
    session:
        An active SQLAlchemy session.
    tag_names:
        Optional whitelist of tag names. When provided, only signals for those
        tags are returned; otherwise all signals are loaded.

    Returns
    -------
    pandas.DataFrame
        Columns ``timestamp`` (timezone-aware where possible), ``tag_name``
        and ``value``, sorted ascending by ``timestamp``. An empty (but
        correctly typed) DataFrame is returned when no rows match.
    """
    query = (
        session.query(Signal.ts, Tag.name, Signal.value)
        .join(Tag, Signal.tag_id == Tag.id)
    )
    if tag_names is not None:
        query = query.filter(Tag.name.in_(list(tag_names)))
    query = query.order_by(Signal.ts.asc())

    rows = query.all()
    if not rows:
        logger.info("No signals found in database for tags=%s", tag_names)
        return pd.DataFrame(columns=SIGNAL_COLUMNS)

    df = pd.DataFrame(rows, columns=SIGNAL_COLUMNS)
    # Normalize to timezone-aware UTC timestamps; naive values are assumed UTC.
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.sort_values("timestamp").reset_index(drop=True)
    logger.info("Loaded %d signal rows for %d distinct tags", len(df), df["tag_name"].nunique())
    return df


def _training_config(
    target_tag: str,
    input_tags: List[str],
    lags: List[int],
    rolling_windows: List[int],
    horizon: int,
    model_output_path: str,
    config: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """YAML config when the caller has it, otherwise the trainer arguments."""
    if config is not None:
        return dict(config)
    return {
        "target_tag": target_tag,
        "input_tags": list(input_tags),
        "lags": list(lags),
        "rolling_windows": list(rolling_windows),
        "horizon": horizon,
        "model_output_path": model_output_path,
    }


def train_energy_model(
    session: Session,
    target_tag: str,
    input_tags: List[str],
    lags: List[int],
    rolling_windows: List[int],
    horizon: int,
    model_output_path: str,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Train and persist an :class:`EnergyKPIModel` for a single KPI tag.

    Parameters
    ----------
    session:
        Active SQLAlchemy session used to load training signals.
    target_tag:
        Tag whose future value is forecast.
    input_tags:
        Tags used to build predictive features. ``target_tag`` may legitimately
        appear here (autoregressive features) and is handled without error.
    lags:
        Lag offsets (rows) for lag features.
    rolling_windows:
        Window sizes (rows) for rolling features.
    horizon:
        Forecast horizon (rows).
    model_output_path:
        Destination path for the persisted model artifact.
    config:
        Optional full YAML configuration. When provided, it is logged to
        MLflow as run parameters. When omitted, the explicit trainer arguments
        are logged instead.

    Returns
    -------
    dict
        Structured training summary (sample counts, per-split metrics, backend
        and artifact path).

    Raises
    ------
    ValueError
        If the database has no signals for the requested tags, or no rows
        survive supervised-dataset assembly. Tag-existence errors raised by the
        feature layer are propagated unchanged.
    """
    # Deduplicate the DB query set while preserving order; the target tag is
    # always required even if it is also an input.
    required_tags: List[str] = list(dict.fromkeys([*input_tags, target_tag]))
    logger.info("Loading signals for tags: %s", required_tags)

    signals = load_signals_from_db(session, tag_names=required_tags)
    if signals.empty:
        raise ValueError(
            "No signals found in the database for the requested tags "
            f"{required_tags}; cannot train."
        )

    X, y = make_supervised_dataset(
        signals,
        target_tag=target_tag,
        input_tags=input_tags,
        lags=lags,
        rolling_windows=rolling_windows,
        horizon=horizon,
    )
    if X.empty:
        raise ValueError(
            "Supervised dataset is empty after feature/target alignment; "
            "not enough contiguous samples for the given lags/windows/horizon."
        )

    X_train, X_valid, y_train, y_valid = train_valid_split_time_series(
        X, y, valid_fraction=DEFAULT_VALID_FRACTION
    )

    model = EnergyKPIModel(target_name=target_tag)
    model.fit(X_train, y_train)

    train_metrics = evaluate_regressor(model, X_train, y_train)
    valid_metrics = evaluate_regressor(model, X_valid, y_valid)

    stats = compute_feature_stats(X_train)
    model.metadata["feature_stats"] = stats
    model.save(model_output_path)
    stats_path = feature_stats_path(model_output_path)
    write_feature_stats(stats_path, stats)

    logged_config = _training_config(
        target_tag,
        input_tags,
        lags,
        rolling_windows,
        horizon,
        model_output_path,
        config,
    )
    summary: Dict[str, Any] = {
        "target_tag": target_tag,
        "n_total_samples": int(len(X)),
        "n_train_samples": int(len(X_train)),
        "n_valid_samples": int(len(X_valid)),
        "n_features": int(len(model.feature_cols) if model.feature_cols else X.shape[1]),
        "train_metrics": train_metrics,
        "valid_metrics": valid_metrics,
        "model_output_path": model_output_path,
        "feature_stats_path": str(stats_path),
        "model_backend": model.metadata.get("model_backend"),
    }

    # Import lazily so loading this module does not require a tracking call.
    from src.ml.tracking import log_training_run

    summary["mlflow_run_id"] = log_training_run(
        logged_config,
        summary,
        model_output_path,
        str(stats_path),
    )
    summary_path = Path(model_output_path).parent / "training_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, default=str),
        encoding="utf-8",
    )
    logger.info("Training complete: %s", summary)
    return summary


def load_training_config(path: str) -> Dict[str, Any]:
    """Load and validate a training configuration YAML file.

    Parameters
    ----------
    path:
        Path to the YAML configuration.

    Returns
    -------
    dict
        The parsed configuration.

    Raises
    ------
    ValueError
        If the file does not parse to a mapping, or any required key
        (``target_tag``, ``input_tags``, ``lags``, ``rolling_windows``,
        ``horizon``, ``model_output_path``) is missing.
    """
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    if not isinstance(cfg, dict):
        raise ValueError(f"Config at {path!r} must be a mapping, got {type(cfg)!r}.")

    missing = [key for key in REQUIRED_CONFIG_KEYS if key not in cfg]
    if missing:
        raise ValueError(
            f"Config at {path!r} is missing required key(s): {missing}. "
            f"Required keys: {list(REQUIRED_CONFIG_KEYS)}."
        )
    return cfg


def main() -> None:
    """CLI entry point: train a model from a config file and print a summary.

    The database connection is resolved from the ``DATABASE_URL`` environment
    variable via :func:`src.db.models.get_engine` (no credentials are
    hardcoded). The JSON training summary is written to stdout.
    """
    parser = argparse.ArgumentParser(
        description="Train a cement-kiln-copilot KPI forecasting model.",
    )
    parser.add_argument(
        "--config",
        default="config/model_config.yaml",
        help="Path to the training configuration YAML.",
    )
    args = parser.parse_args()

    cfg = load_training_config(args.config)

    engine = get_engine()  # DATABASE_URL resolved from environment.
    session_factory = get_session_factory(engine)
    session = session_factory()
    try:
        summary = train_energy_model(
            session,
            target_tag=cfg["target_tag"],
            input_tags=list(cfg["input_tags"]),
            lags=list(cfg["lags"]),
            rolling_windows=list(cfg["rolling_windows"]),
            horizon=int(cfg["horizon"]),
            model_output_path=cfg["model_output_path"],
            config=cfg,
        )
    finally:
        session.close()

    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
