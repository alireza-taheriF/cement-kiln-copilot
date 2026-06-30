"""Health/liveness/readiness endpoints."""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok"}


@router.get("/ready")
async def ready() -> dict[str, str]:
    """Readiness probe.

    TODO: verify DB connectivity and that model artifacts are loaded.
    """
    return {"status": "ready"}
