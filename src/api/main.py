"""FastAPI application factory and entrypoint.

Run with:

    uvicorn src.api.main:app --reload
"""

from __future__ import annotations

import logging

from fastapi import FastAPI

from src.api.routers import health, monitoring, prediction, recommendation
from src.db.models import init_db

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

API_VERSION = "v1"


def create_app() -> FastAPI:
    """Application factory: build and configure the FastAPI app."""
    app = FastAPI(
        title="Cement Kiln Copilot API",
        description="Advisory system for cement kiln & mill optimization.",
        version="0.1.0",
    )

    # Routers ---------------------------------------------------------------
    app.include_router(health.router, tags=["health"])
    app.include_router(monitoring.metrics_router, tags=["monitoring"])
    app.include_router(prediction.router, prefix=f"/{API_VERSION}", tags=["prediction"])
    app.include_router(
        recommendation.router, prefix=f"/{API_VERSION}", tags=["recommendation"]
    )
    app.include_router(
        monitoring.monitor_router, prefix=f"/{API_VERSION}", tags=["monitoring"]
    )

    @app.on_event("startup")
    async def _startup() -> None:
        logger.info("Cement Kiln Copilot API starting up.")
        try:
            init_db()
        except Exception:
            logger.exception("Could not initialize the database on startup.")

    @app.get("/")
    async def root() -> dict[str, str]:
        return {"service": "cement-kiln-copilot", "version": "0.1.0"}

    return app


app = create_app()
