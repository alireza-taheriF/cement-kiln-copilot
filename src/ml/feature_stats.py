"""Training-feature distribution saved beside a model artifact.

Drift monitoring compares live prediction inputs to these mean/std values.
Zero-variance features use a standard deviation of 1.0 so the z-score stays
defined (it then equals the raw deviation from the training mean).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

#: Floor used when a training feature has no spread.
STD_FLOOR = 1.0


def feature_stats_path(model_output_path: str) -> Path:
    """Return the sidecar path next to a joblib model artifact."""
    model_path = Path(model_output_path)
    return model_path.with_name(f"{model_path.stem}.feature_stats.json")


def compute_feature_stats(X: pd.DataFrame) -> dict[str, Any]:
    """Mean and population standard deviation of each training feature."""
    if X.empty:
        raise ValueError("Cannot compute feature stats on an empty frame.")

    mean = X.mean(axis=0, numeric_only=True)
    std = X.std(axis=0, ddof=0, numeric_only=True)
    std = std.mask(std.abs() < 1e-12, STD_FLOOR)
    return {
        "mean": {str(name): float(value) for name, value in mean.items()},
        "std": {str(name): float(value) for name, value in std.items()},
    }


def write_feature_stats(path: Path, stats: dict[str, Any]) -> None:
    """Persist feature stats as JSON, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stats, indent=2), encoding="utf-8")


def read_feature_stats(path: Path) -> dict[str, Any] | None:
    """Load a feature-stats sidecar, or return ``None`` when it is absent."""
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return None
    if "mean" not in payload or "std" not in payload:
        return None
    return payload
