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

from app.config import settings
from app.dial_safety import posture

router = APIRouter(tags=["dashboard"])

templates = Jinja2Templates(directory=str(settings.base_dir / "templates"))


@router.get("/dashboard", response_class=HTMLResponse, summary="Local demo dashboard")
async def dashboard(request: Request) -> HTMLResponse:
    """Render the operator dashboard.

    Only status indicators reach the template: whether the tool endpoints
    are usable at all, and the dial-safety posture. ``posture()`` returns
    booleans and a label, never the configured number, so no secret or
    phone number can reach the page.
    """
    return templates.TemplateResponse(
        request,
        "dashboard.html",
        {
            "tool_auth_configured": bool(settings.tool_shared_secret),
            "dial": posture(),
        },
    )
