"""Process-wide, lazily loaded services for the API (configs, models).

Paths come from environment variables so the same image runs at any plant:

=====================  ==========================================
``PLANT_CONFIG``       ``config/plant.yaml``
``SAFETY_CONFIG``      ``config/safety_limits.yaml``
``ADVISOR_CONFIG``     ``config/advisor.yaml``
``FREE_LIME_MODEL``    ``artifacts/models/free_lime_softsensor.joblib``
=====================  ==========================================

Loading failures raise :class:`ServiceUnavailable`, which routers turn into
HTTP 503 with the reason, so a missing model is reported, not hidden.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any, Dict, Optional

import joblib

from src.advisory.advisor import KilnAdvisor
from src.domain.kpis import PlantConfig, load_plant_config
from src.safety.envelope import SafetyEnvelope


class ServiceUnavailable(RuntimeError):
    """A configuration file or model artifact is missing or invalid."""


# Re-entrant: a loader may itself load another service (the advisor loads the
# plant config), which with a plain Lock deadlocks.
_lock = threading.RLock()
_cache: Dict[str, Any] = {}


def _path(env: str, default: str) -> str:
    return os.getenv(env, default)


def _cached(key: str, loader):
    with _lock:
        if key not in _cache:
            try:
                _cache[key] = loader()
            except FileNotFoundError as exc:
                raise ServiceUnavailable(str(exc)) from exc
            except (ValueError, KeyError, TypeError) as exc:
                raise ServiceUnavailable(f"{key}: {exc}") from exc
        return _cache[key]


def reset_cache() -> None:
    """Forget loaded services (tests, or after retraining)."""
    with _lock:
        _cache.clear()


def plant_config() -> Optional[PlantConfig]:
    path = _path("PLANT_CONFIG", "config/plant.yaml")
    if not Path(path).exists():
        return None
    return _cached(f"plant:{path}", lambda: load_plant_config(path))


def envelope() -> SafetyEnvelope:
    path = _path("SAFETY_CONFIG", "config/safety_limits.yaml")
    return _cached(f"safety:{path}", lambda: SafetyEnvelope.from_yaml(path))


def advisor() -> KilnAdvisor:
    advisor_path = _path("ADVISOR_CONFIG", "config/advisor.yaml")
    safety_path = _path("SAFETY_CONFIG", "config/safety_limits.yaml")
    return _cached(
        f"advisor:{advisor_path}:{safety_path}",
        lambda: KilnAdvisor.from_paths(advisor_path, safety_path, plant=plant_config()),
    )


def free_lime_sensor():
    path = _path("FREE_LIME_MODEL", "artifacts/models/free_lime_softsensor.joblib")

    def _load():
        if not Path(path).exists():
            raise FileNotFoundError(
                f"free-lime soft sensor not found at {path}; run python -m src.softsensor.train"
            )
        return joblib.load(path)

    return _cached(f"free_lime:{path}", _load)


def demo_plant():
    """The simulated demo plant (14 days, seed 7), for demos without plant data."""
    from src.sim.plant import simulate

    return _cached("demo_plant", simulate)
