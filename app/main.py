"""FastAPI application entry point.

Run with:
    .venv\\Scripts\\python.exe -m uvicorn app.main:app --reload --port 8000

Phase 0 is the skeleton: configuration, the app object, and a health endpoint.
Routers for the agent's tools, the dashboard, and the post-call webhook are
mounted here in later phases.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI

from app import __version__
from app.config import settings

logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
)
logger = logging.getLogger("app")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Log a readable startup banner, then hand over to the server."""
    logger.info("%s v%s starting (phase 0 - skeleton)", settings.app_name, __version__)
    # The dial-safety toggle is reported separately below; it is a switch, not
    # a credential, so listing it as "not configured" would read as a problem.
    ready = {
        name: ok
        for name, ok in settings.readiness().items()
        if name != "outbound_calls_enabled"
    }
    configured = sorted(name for name, ok in ready.items() if ok)
    pending = sorted(name for name, ok in ready.items() if not ok)
    logger.info("configured: %s", ", ".join(configured) or "nothing yet")
    logger.info("not yet configured: %s", ", ".join(pending) or "none")
    if not settings.enable_outbound_calls:
        logger.info("dial safety: outbound calls DISABLED (ENABLE_OUTBOUND_CALLS=false)")
    yield
    logger.info("%s shutting down", settings.app_name)


app = FastAPI(
    title="Voice Agent for Autopay Recovery",
    description=(
        "Backend for a voice agent that recovers failed autopay payments. "
        "All customer data is fictional and the payment processor is fully "
        "mocked — no real payments are ever made."
    ),
    version=__version__,
    lifespan=lifespan,
)

# Routers are mounted in later phases:
#   app.include_router(tools.router)     # Phase 2 — the agent's webhook tools
#   app.include_router(demo.router)      # Phase 3 — dashboard control plane
#   app.include_router(pages.router)     # Phase 3 — dashboard + voice page
#   app.include_router(webhooks.router)  # Phase 7 — post-call transcripts


@app.get("/healthz", tags=["system"])
async def healthz() -> dict[str, Any]:
    """Liveness probe and configuration readiness.

    Reports only whether each secret is *set*, never its value, so it is safe
    to hit through the public tunnel.
    """
    return {
        "status": "ok",
        "service": settings.app_name,
        "version": __version__,
        "phase": "0 - skeleton",
        "config": settings.readiness(),
    }


@app.get("/", tags=["system"])
async def root() -> dict[str, str]:
    """Placeholder root. The dashboard replaces this in Phase 3."""
    return {
        "service": settings.app_name,
        "version": __version__,
        "health": "/healthz",
        "docs": "/docs",
    }
