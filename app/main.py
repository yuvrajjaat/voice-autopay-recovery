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
from fastapi.exceptions import RequestValidationError
from fastapi.staticfiles import StaticFiles

from app import __version__
from app.config import settings
from app.errors import (
    ToolError,
    tool_error_handler,
    unhandled_error_handler,
    validation_error_handler,
)
from app.routers import demo, pages, tools, webhooks

logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
)
logger = logging.getLogger("app")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Log a readable startup banner, then hand over to the server."""
    logger.info(
        "%s v%s starting (phase 6 - voice agent)", settings.app_name, __version__
    )
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

# Every /tools/* failure, 422 included, is rendered as the same flat
# envelope, because the body is read by the agent's LLM and then spoken.
app.add_exception_handler(ToolError, tool_error_handler)
app.add_exception_handler(RequestValidationError, validation_error_handler)
app.add_exception_handler(Exception, unhandled_error_handler)

app.include_router(tools.router)  # Phase 2 — the agent's seven tools
app.include_router(demo.router)  # Phase 3 — local demo control plane
app.include_router(pages.router)  # Phase 3 — the dashboard page
app.include_router(webhooks.router)  # Phase 6 — post-call metadata

# Dashboard assets. Mounted from an absolute path so the server can be
# started from any working directory.
app.mount(
    "/static",
    StaticFiles(directory=str(settings.base_dir / "static")),
    name="static",
)



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
        "phase": "6 - elevenlabs voice agent",
        "config": settings.readiness(),
    }


@app.get("/", tags=["system"])
async def root() -> dict[str, str]:
    """Service index. The human-facing page is /dashboard."""
    return {
        "service": settings.app_name,
        "version": __version__,
        "health": "/healthz",
        "docs": "/docs",
        "dashboard": "/dashboard",
        "voice": "/voice",
    }
