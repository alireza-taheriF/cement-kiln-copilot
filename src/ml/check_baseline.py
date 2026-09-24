"""Gate the demo training metric against the committed baseline.

CI trains on ``data/demo_signals.csv`` and then runs this module. The job
fails when the held-out RMSE is worse than ``tests/baselines/demo_metrics.json``
by more than the tolerance stored in that file.

Refresh the committed value after an intentional demo-metric change (from the
repository root, after training)::

    python -m src.ml.check_baseline --write
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SUMMARY = REPO_ROOT / "artifacts" / "models" / "training_summary.json"
DEFAULT_BASELINE = REPO_ROOT / "tests" / "baselines" / "demo_metrics.json"

METRIC_NAME = "valid_rmse"
DEFAULT_TOLERANCE = 0.05
DEFAULT_TOLERANCE_TYPE = "relative"


def extract_valid_rmse(summary: dict[str, Any]) -> float:
    """Pull held-out RMSE from a training summary."""
    try:
        value = summary["valid_metrics"]["rmse"]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "Training summary is missing valid_metrics.rmse."
        ) from exc
    return float(value)


def regression_limit(
    baseline: float,
    tolerance: float,
    tolerance_type: str,
) -> float:
    """Largest allowed metric. Lower RMSE is better."""
    if tolerance < 0:
        raise ValueError(f"tolerance must be >= 0, got {tolerance}.")
    if tolerance_type == "relative":
        return baseline * (1.0 + tolerance)
    if tolerance_type == "absolute":
        return baseline + tolerance
    raise ValueError(
        f"tolerance_type must be 'relative' or 'absolute', got {tolerance_type!r}."
    )


def is_regression(
    observed: float,
    baseline: float,
    tolerance: float,
    tolerance_type: str,
) -> bool:
    """Whether ``observed`` is worse than the baseline by more than tolerance."""
    return observed > regression_limit(baseline, tolerance, tolerance_type)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return payload


def write_baseline(summary_path: Path, baseline_path: Path) -> dict[str, Any]:
    """Create or refresh the baseline file from a training summary.

    An existing tolerance is preserved. The first write uses a small relative
    tolerance of 0.05.
    """
    summary = _load_json(summary_path)
    observed = extract_valid_rmse(summary)
    existing: dict[str, Any] = {}
    if baseline_path.exists():
        existing = _load_json(baseline_path)
    document = {
        "metric": METRIC_NAME,
        "value": observed,
        "tolerance": existing.get("tolerance", DEFAULT_TOLERANCE),
        "tolerance_type": existing.get("tolerance_type", DEFAULT_TOLERANCE_TYPE),
        "direction": "lower_is_better",
        "dataset": "data/demo_signals.csv",
        "config": "config/model_config.yaml",
    }
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    baseline_path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return document


def compare_to_baseline(summary_path: Path, baseline_path: Path) -> tuple[bool, str]:
    """Return ``(ok, message)`` for the demo metric gate."""
    if not summary_path.exists():
        return False, (
            f"Training summary not found at {summary_path}. "
            "Train on the bundled demo data first."
        )
    if not baseline_path.exists():
        return False, (
            f"Baseline not found at {baseline_path}. "
            "Create it with: python -m src.ml.check_baseline --write"
        )
    summary = _load_json(summary_path)
    baseline = _load_json(baseline_path)
    observed = extract_valid_rmse(summary)
    try:
        expected = float(baseline["value"])
        tolerance = float(baseline["tolerance"])
    except (KeyError, TypeError, ValueError) as exc:
        return False, f"Baseline file {baseline_path} is missing value/tolerance: {exc}"
    tolerance_type = str(baseline.get("tolerance_type", DEFAULT_TOLERANCE_TYPE))
    limit = regression_limit(expected, tolerance, tolerance_type)
    if is_regression(observed, expected, tolerance, tolerance_type):
        return False, (
            f"{METRIC_NAME} {observed:.6g} is worse than baseline {expected:.6g} "
            f"by more than tolerance {tolerance} ({tolerance_type}); "
            f"allowed limit is {limit:.6g}."
        )
    return True, (
        f"{METRIC_NAME} {observed:.6g} is within tolerance of baseline "
        f"{expected:.6g} (limit {limit:.6g}, tolerance {tolerance} {tolerance_type})."
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns 0 when the gate passes, 1 otherwise."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY))
    parser.add_argument("--baseline", default=str(DEFAULT_BASELINE))
    parser.add_argument(
        "--write",
        action="store_true",
        help="Refresh tests/baselines/demo_metrics.json from the latest summary.",
    )
    args = parser.parse_args(argv)
    summary_path = Path(args.summary)
    baseline_path = Path(args.baseline)

    if args.write:
        if not summary_path.exists():
            print(
                f"Training summary not found at {summary_path}. "
                "Train on the bundled demo data first.",
                file=sys.stderr,
            )
            return 1
        document = write_baseline(summary_path, baseline_path)
        print(
            f"Wrote {baseline_path} with {METRIC_NAME}={document['value']} "
            f"tolerance={document['tolerance']} ({document['tolerance_type']})."
        )
        return 0

    ok, message = compare_to_baseline(summary_path, baseline_path)
    print(message)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
