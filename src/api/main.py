"""FastAPI application factory and entrypoint.

Run with::

    uvicorn src.api.main:app --reload
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src import __version__
from src.api.routers import (
    advisory,
    energy,
    health,
    monitoring,
    operations,
    prediction,
    recommendation,
)
from src.db.models import init_db

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

API_VERSION = "v1"


@asynccontextmanager
async def _lifespan(app: FastAPI):
    logger.info("Cement Kiln Copilot API starting up.")
    try:
        init_db()
    except Exception:
        logger.exception("Could not initialize the database on startup.")
    yield


def create_app() -> FastAPI:
    """Application factory: build and configure the FastAPI app."""
    app = FastAPI(
        title="Cement Kiln Copilot API",
        description=(
            "Open-loop decision support for cement kiln and mill energy optimization. "
            "Advisory only: nothing is ever written to the DCS."
        ),
        version=__version__,
        lifespan=_lifespan,
    )

    app.include_router(health.router, tags=["health"])
    app.include_router(monitoring.metrics_router, tags=["monitoring"])
    prefix = f"/{API_VERSION}"
    app.include_router(advisory.router, prefix=prefix, tags=["advisor"])
    app.include_router(energy.router, prefix=prefix, tags=["energy"])
    app.include_router(operations.router, prefix=prefix, tags=["operations"])
    app.include_router(prediction.router, prefix=prefix, tags=["prediction"])
    app.include_router(recommendation.router, prefix=prefix, tags=["recommendation"])
    app.include_router(monitoring.monitor_router, prefix=prefix, tags=["monitoring"])

    @app.get("/")
    async def root() -> dict[str, str]:
        return {"service": "cement-kiln-copilot", "version": __version__}

    return app


app = create_app()
