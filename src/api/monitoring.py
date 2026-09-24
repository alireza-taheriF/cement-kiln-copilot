"""Inference logging, Prometheus text metrics, and feature-drift scores.

Successful predict and recommend calls append one row to ``inference_events``
and update an in-process latency histogram. Failures in this module are
swallowed so a monitoring outage cannot change the advisory response.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import threading
import time
from datetime import datetime, timezone
from typing import Any, Mapping, Optional

import pandas as pd

from src.db.models import InferenceEvent, get_session_factory, init_db
from src.ml.feature_stats import feature_stats_path, read_feature_stats

logger = logging.getLogger(__name__)

PREDICT_ROUTE = "/v1/predict/energy-kpi"
RECOMMEND_ROUTE = "/v1/recommend/setpoints"
KNOWN_ROUTES = (PREDICT_ROUTE, RECOMMEND_ROUTE)

# Histogram edges in milliseconds. The last bucket is +Inf.
_BUCKET_EDGES_MS = (5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000)

_lock = threading.Lock()
_request_counts: dict[str, int] = {}
_latency_sums: dict[str, float] = {}
_latency_buckets: dict[str, list[int]] = {}
_ready_database_url: Optional[str] = None


def hash_numeric_input(
    signals: Mapping[str, list[float]],
    extra: Optional[Mapping[str, Any]] = None,
) -> str:
    """SHA-256 of the numeric payload in a stable tag order."""
    numeric: list[float] = []
    for tag in sorted(signals):
        numeric.extend(float(value) for value in signals[tag])
    if extra:
        for key in sorted(extra):
            value = extra[key]
            if isinstance(value, Mapping):
                for inner in sorted(value):
                    numeric.append(float(value[inner]))
            else:
                numeric.append(float(value))
    raw = json.dumps(numeric, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def feature_means(frame: pd.DataFrame) -> dict[str, float]:
    """Column means of the feature matrix actually sent to the model."""
    if frame.empty:
        return {}
    means = frame.mean(axis=0, numeric_only=True)
    return {str(name): float(value) for name, value in means.items()}


def _database_url() -> str:
    return os.getenv("DATABASE_URL", "sqlite:///./cement_copilot_dev.db")


def ensure_monitoring_tables() -> None:
    """Create tables for the current ``DATABASE_URL`` once per process."""
    global _ready_database_url
    url = _database_url()
    if _ready_database_url == url:
        return
    init_db()
    _ready_database_url = url


def _observe(route: str, latency_ms: float) -> None:
    with _lock:
        _request_counts[route] = _request_counts.get(route, 0) + 1
        _latency_sums[route] = _latency_sums.get(route, 0.0) + float(latency_ms)
        buckets = _latency_buckets.setdefault(
            route, [0] * (len(_BUCKET_EDGES_MS) + 1)
        )
        placed = False
        for index, edge in enumerate(_BUCKET_EDGES_MS):
            if latency_ms <= edge:
                buckets[index] += 1
                placed = True
                break
        if not placed:
            buckets[-1] += 1


def _persist_event(
    *,
    route: str,
    latency_ms: float,
    model_version: Optional[str],
    input_hash: str,
    means: dict[str, float],
) -> None:
    ensure_monitoring_tables()
    session = get_session_factory()()
    try:
        session.add(
            InferenceEvent(
                ts=datetime.now(timezone.utc),
                route=route,
                latency_ms=float(latency_ms),
                model_version=model_version,
                input_hash=input_hash,
                feature_means_json=json.dumps(means),
            )
        )
        session.commit()
    finally:
        session.close()


def record_inference(
    *,
    route: str,
    started: float,
    model_version: Optional[str],
    signals: Mapping[str, list[float]],
    feature_frame: pd.DataFrame,
    extra: Optional[Mapping[str, Any]] = None,
) -> None:
    """Record one successful advisory call. Never raises."""
    try:
        latency_ms = (time.perf_counter() - started) * 1000.0
        _observe(route, latency_ms)
        means = feature_means(feature_frame)
        digest = hash_numeric_input(signals, extra)
        version = None if model_version is None else str(model_version)
        _persist_event(
            route=route,
            latency_ms=latency_ms,
            model_version=version,
            input_hash=digest,
            means=means,
        )
    except Exception:
        logger.exception("Failed to record inference event for %s", route)


def render_prometheus() -> str:
    """Prometheus text exposition for request count and latency histogram."""
    with _lock:
        routes = sorted(set(KNOWN_ROUTES) | set(_request_counts))
        counts = dict(_request_counts)
        sums = dict(_latency_sums)
        buckets = {route: list(values) for route, values in _latency_buckets.items()}

    lines = [
        "# HELP cement_kiln_copilot_requests_total Successful advisory API requests.",
        "# TYPE cement_kiln_copilot_requests_total counter",
    ]
    for route in routes:
        lines.append(
            "cement_kiln_copilot_requests_total"
            f'{{route="{route}"}} {counts.get(route, 0)}'
        )

    lines.extend(
        [
            "# HELP cement_kiln_copilot_request_latency_ms Advisory request latency.",
            "# TYPE cement_kiln_copilot_request_latency_ms histogram",
        ]
    )
    for route in routes:
        raw = buckets.get(route) or [0] * (len(_BUCKET_EDGES_MS) + 1)
        cumulative = 0
        for index, edge in enumerate(_BUCKET_EDGES_MS):
            cumulative += raw[index] if index < len(raw) else 0
            lines.append(
                "cement_kiln_copilot_request_latency_ms_bucket"
                f'{{route="{route}",le="{edge}"}} {cumulative}'
            )
        cumulative += raw[-1] if raw else 0
        lines.append(
            "cement_kiln_copilot_request_latency_ms_bucket"
            f'{{route="{route}",le="+Inf"}} {cumulative}'
        )
        lines.append(
            "cement_kiln_copilot_request_latency_ms_sum"
            f'{{route="{route}"}} {sums.get(route, 0.0)}'
        )
        lines.append(
            "cement_kiln_copilot_request_latency_ms_count"
            f'{{route="{route}"}} {counts.get(route, 0)}'
        )
    return "\n".join(lines) + "\n"


def recent_prediction_means(n: int) -> list[dict[str, float]]:
    """Feature means from the last ``n`` successful prediction calls."""
    ensure_monitoring_tables()
    session = get_session_factory()()
    try:
        rows = (
            session.query(InferenceEvent)
            .filter(InferenceEvent.route == PREDICT_ROUTE)
            .order_by(InferenceEvent.id.desc())
            .limit(n)
            .all()
        )
        parsed: list[dict[str, float]] = []
        for row in rows:
            if not row.feature_means_json:
                continue
            payload = json.loads(row.feature_means_json)
            if isinstance(payload, dict):
                parsed.append({str(key): float(value) for key, value in payload.items()})
        return parsed
    finally:
        session.close()


def resolve_feature_stats(model: Any) -> Optional[dict[str, Any]]:
    """Stats stored on the model, otherwise the sidecar next to ``MODEL_PATH``."""
    metadata = getattr(model, "metadata", None) or {}
    stats = metadata.get("feature_stats")
    if isinstance(stats, dict) and "mean" in stats and "std" in stats:
        return stats
    model_path = os.getenv("MODEL_PATH")
    if not model_path:
        return None
    return read_feature_stats(feature_stats_path(model_path))


def per_feature_zscores(
    recent: list[dict[str, float]],
    stats: Mapping[str, Any],
) -> dict[str, float]:
    """Z-score of the recent feature mean against the training distribution."""
    means = stats.get("mean") or {}
    stds = stats.get("std") or {}
    scores: dict[str, float] = {}
    for name, train_mean in means.items():
        values = [row[name] for row in recent if name in row]
        if not values:
            continue
        std = float(stds.get(name, 1.0))
        if not math.isfinite(std) or abs(std) < 1e-12:
            std = 1.0
        recent_mean = sum(values) / len(values)
        score = (recent_mean - float(train_mean)) / std
        if math.isfinite(score):
            scores[str(name)] = float(score)
    return scores


def build_drift_report(model: Any, n: int) -> Optional[dict[str, Any]]:
    """Compare the last ``n`` prediction inputs to training feature stats.

    Returns ``None`` when no training stats are available.
    """
    stats = resolve_feature_stats(model)
    if stats is None:
        return None
    recent = recent_prediction_means(n)
    scores = per_feature_zscores(recent, stats)
    max_abs = max((abs(value) for value in scores.values()), default=0.0)
    version = (getattr(model, "metadata", None) or {}).get("trained_at")
    return {
        "n": len(recent),
        "features": scores,
        "max_abs_z": float(max_abs),
        "model_version": None if version is None else str(version),
    }


def _reset_prometheus_for_tests() -> None:
    """Clear in-process counters. Used by tests only."""
    with _lock:
        _request_counts.clear()
        _latency_sums.clear()
        _latency_buckets.clear()
