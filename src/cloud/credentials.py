"""Credential checks for cloud submit paths.

Dry-run never calls this module. Missing variables are reported by name only.
"""

from __future__ import annotations

import importlib
import os
import sys
from typing import Any, Sequence


def missing_env(names: Sequence[str]) -> list[str]:
    """Names that are unset or blank in the process environment."""
    missing: list[str] = []
    for name in names:
        if not os.environ.get(name, "").strip():
            missing.append(name)
    return missing


def ensure_submit_env(provider: str, names: Sequence[str]) -> None:
    """Exit 1 when the submit path does not have its expected environment."""
    missing = missing_env(names)
    if not missing:
        return
    print(
        f"Cannot submit the {provider} training job. "
        f"Missing environment variable(s): {', '.join(missing)}. "
        "The submit path needs a real cloud account and is not exercised in CI. "
        "Train locally instead:\n"
        "  python -m src.ml.train --config config/model_config.yaml",
        file=sys.stderr,
    )
    raise SystemExit(1)


def ensure_import(module_name: str, pip_name: str) -> Any:
    """Import an optional cloud SDK, or exit 1 with an install hint."""
    try:
        return importlib.import_module(module_name)
    except ImportError:
        print(
            f"Cannot submit: install the optional package '{pip_name}' "
            f"(module {module_name}). Dry-run does not need cloud SDKs, "
            "and CI does not submit jobs.",
            file=sys.stderr,
        )
        raise SystemExit(1)
