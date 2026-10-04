"""Shared-secret authentication for the tool endpoints.

The tool endpoints have to be reachable from ElevenLabs' cloud, which means
exposing them through a public tunnel. Anything on the open internet can then
POST to them, so each request must prove it came from our agent.

The mechanism is deliberately minimal: one shared secret, sent as
``X-Tool-Secret``, compared in constant time. The secret is stored as an
ElevenLabs workspace secret and attached to each tool's headers; it lives in
``.env`` on our side and nowhere in the repository.

Two failure modes, kept distinct on purpose:

* **Not configured** (``TOOL_SHARED_SECRET`` unset) → ``503``. The server
  refuses to serve tool calls at all rather than defaulting to open access.
  An unconfigured deployment failing loudly is much safer than one that
  quietly accepts every caller.
* **Wrong or missing header** → ``401``.

Neither path ever echoes the expected secret, and the secret is never logged.
"""

from __future__ import annotations

import logging
import secrets

from fastapi import Header

from app.config import settings
from app.errors import ToolError

logger = logging.getLogger(__name__)


async def require_tool_secret(
    x_tool_secret: str | None = Header(
        default=None,
        alias="X-Tool-Secret",
        description="Shared secret configured as TOOL_SHARED_SECRET.",
    ),
) -> None:
    """FastAPI dependency guarding every ``/tools/*`` route."""
    expected = settings.tool_shared_secret

    if not expected:
        logger.error(
            "a tool call was rejected because TOOL_SHARED_SECRET is not set; "
            "configure it in .env before exposing the tool endpoints"
        )
        raise ToolError(
            503,
            "tool_auth_not_configured",
            "The payment tools aren't available right now.",
        )

    # compare_digest keeps the comparison constant-time; passing "" when the
    # header is absent means a missing header takes the same path as a wrong
    # one, so neither case is distinguishable by timing.
    if not secrets.compare_digest(x_tool_secret or "", expected):
        logger.warning("rejected tool call with a missing or incorrect X-Tool-Secret")
        raise ToolError(
            401,
            "invalid_tool_secret",
            "This request isn't authorised.",
        )
