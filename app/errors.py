"""Uniform error handling for the tool layer.

Why a custom envelope
---------------------
Tool responses are handed straight to the voice agent's LLM, which then says
something to a person. FastAPI's default error body (``{"detail": ...}``) and
its 422 validation body (a nested list of field errors) are both awkward for
that: the model has to guess what went wrong and improvise wording.

So every ``/tools/*`` failure returns the same flat shape:

    {"success": false, "error": "identity_not_verified",
     "message": "I need to verify the account before I can discuss payments."}

``error`` is a stable machine code for tests and the dashboard. ``message`` is
one short sentence the agent can say as-is. Neither ever contains a stack
trace, an internal path, or a secret.
"""

from __future__ import annotations

import logging

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.models import ToolErrorResponse

logger = logging.getLogger(__name__)


class ToolError(Exception):
    """A tool call that cannot proceed.

    Carries the HTTP status, a stable machine code, and a speakable message.
    """

    def __init__(self, status_code: int, error: str, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error = error
        self.message = message


def _envelope(status_code: int, error: str, message: str) -> JSONResponse:
    body = ToolErrorResponse(error=error, message=message)
    return JSONResponse(status_code=status_code, content=body.model_dump())


async def tool_error_handler(_request: Request, exc: Exception) -> JSONResponse:
    """Render a ToolError as the standard envelope."""
    assert isinstance(exc, ToolError)
    logger.info("tool error %s (%s): %s", exc.error, exc.status_code, exc.message)
    return _envelope(exc.status_code, exc.error, exc.message)


async def validation_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Turn a 422 into the same envelope, with a concise field summary.

    Only field names and their problems are echoed — never the submitted
    values, which could contain whatever a caller sent us.
    """
    assert isinstance(exc, RequestValidationError)
    problems = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error.get("loc", ()) if part != "body")
        problems.append(f"{location or 'body'}: {error.get('msg', 'invalid')}")
    summary = "; ".join(problems[:4]) or "the request was not valid"
    logger.info("validation error on %s: %s", request.url.path, summary)
    return _envelope(422, "invalid_request", f"That request wasn't valid ({summary}).")


async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Last resort: log the traceback server-side, return nothing revealing.

    Without this, an unexpected exception would surface to the agent as an
    HTML 500 page, which it would then try to read out.
    """
    logger.exception("unhandled error on %s", request.url.path, exc_info=exc)
    return _envelope(
        500,
        "internal_error",
        "Something went wrong on our side. Let me get a colleague to help.",
    )
