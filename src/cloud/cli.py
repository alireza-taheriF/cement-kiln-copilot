"""Shared ``--dry-run`` / ``--submit`` entrypoint for cloud job modules."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from typing import Any


def run_job_cli(
    build_spec: Callable[[], dict[str, Any]],
    submit: Callable[[], None],
    *,
    description: str,
) -> None:
    """Print a job spec, or submit it when ``--submit`` is passed.

    ``--dry-run`` does not read cloud credentials. With no flag, the parser
    exits non-zero: local training remains ``python -m src.ml.train``.
    """
    parser = argparse.ArgumentParser(description=description)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the job spec JSON and exit 0. No cloud credentials are required.",
    )
    mode.add_argument(
        "--submit",
        action="store_true",
        help="Submit the job only when the expected cloud credentials are set.",
    )
    args = parser.parse_args()
    if args.dry_run:
        print(json.dumps(build_spec(), indent=2))
        return
    submit()
