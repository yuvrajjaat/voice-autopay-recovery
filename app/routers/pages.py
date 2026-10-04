"""The dashboard page.

One Jinja2-rendered HTML page plus the static assets it pulls. All data
arrives from ``/api/*`` after load, so the template itself carries no customer
data — and, importantly, no secret. The browser never sees
``TOOL_SHARED_SECRET``.
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app import __version__
from app.config import settings

router = APIRouter(tags=["dashboard"])

templates = Jinja2Templates(directory=str(settings.base_dir / "templates"))


@router.get("/dashboard", response_class=HTMLResponse, summary="Local demo dashboard")
async def dashboard(request: Request) -> HTMLResponse:
    """Render the operator dashboard.

    The tool-secret *name* is passed so the page can show a copy-pasteable
    curl command; the value is never sent.
    """
    return templates.TemplateResponse(
        request,
        "dashboard.html",
        {
            "version": __version__,
            "tool_auth_configured": bool(settings.tool_shared_secret),
            "outbound_calls_enabled": settings.enable_outbound_calls,
        },
    )
