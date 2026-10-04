"""ElevenLabs post-call webhook.

ElevenLabs POSTs a transcript and some metadata after a conversation ends.
This endpoint *finalizes* the session: it stamps a completion time, the
provider's conversation id, the call duration and a transcript reference, then
records one ``voice_call_completed`` event carrying the final outcome.

Delivery is at-least-once, so finalisation is idempotent. A replay of the same
conversation is recognised and acknowledged without writing a second event or
disturbing what the first delivery recorded.

What it deliberately does not do
--------------------------------
It does **not** touch payment state, dispositions, verification flags, or any
ledger row. The seven tool endpoints are the only writers of those, and they
run during the call under our own validation. A transcript is a model's
summary of a conversation: useful evidence, not an authority. If the agent
said "your payment went through" but ``retry_payment`` never returned
``paid``, the ledger stays as it is and the transcript simply records that the
agent misspoke.

Authentication uses ``ELEVENLABS_WEBHOOK_SECRET`` with an HMAC over the raw
body. The endpoint is open to the internet through the tunnel, so an
unverified payload is rejected before it is parsed as anything meaningful.

Whose account is it? The session's, always. The session id arrives as a
dynamic variable and the customer comes from that session's binding. A
``customer_id`` anywhere in the payload is ignored, so a forged or confused
payload cannot attach a call to someone else's account.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import re
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Header, Request

from app import store
from app.config import settings
from app.errors import ToolError
from app.models import EventType, utc_now

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/webhooks", tags=["webhooks"])

#: Reject a signature whose timestamp is older than this, so a captured
#: request cannot be replayed indefinitely.
MAX_SIGNATURE_AGE_SECONDS = 30 * 60

#: Refuse to parse an oversized body.
MAX_BODY_BYTES = 2_000_000

#: Cap on stored transcript turns, so a long call cannot bloat the repo.
MAX_TRANSCRIPT_TURNS = 200

#: Runs this long are card or account numbers. A transcript is a model's
#: rendering of speech and should never carry one, but if a caller reads digits
#: aloud despite the agent refusing them, they get redacted on the way to disk.
_LONG_DIGIT_RUN = re.compile(r"\d{12,}")

TRANSCRIPT_DIR = Path(__file__).resolve().parent.parent.parent / "demo" / "transcripts"


def _transcript_dir() -> Path:
    """Where saved transcripts go.

    A function rather than a constant so the test suite can redirect it to
    a temporary directory. Without that, running the tests scatters files
    into the committed demo folder.
    """
    return TRANSCRIPT_DIR


def _verify_signature(raw_body: bytes, header: str | None) -> None:
    """Verify ElevenLabs' ``elevenlabs-signature`` header.

    Format is ``t=<unix>,v0=<hex hmac>`` over ``"<t>.<body>"``. Implemented
    here rather than via the SDK helper so the webhook needs no API client and
    works without an ElevenLabs connection.
    """
    secret = settings.elevenlabs_webhook_secret
    if not secret:
        logger.error(
            "a post-call webhook was rejected because ELEVENLABS_WEBHOOK_SECRET "
            "is not set; configure it before exposing this endpoint"
        )
        raise ToolError(
            503, "webhook_secret_not_configured", "Webhooks are not configured."
        )

    if not header:
        raise ToolError(401, "missing_signature", "Missing signature header.")

    timestamp: str | None = None
    provided: str | None = None
    for part in header.split(","):
        key, _, value = part.strip().partition("=")
        if key == "t":
            timestamp = value
        elif key.startswith("v"):
            provided = value

    if not timestamp or not provided:
        raise ToolError(401, "malformed_signature", "Malformed signature header.")

    try:
        age = abs(time.time() - int(timestamp))
    except ValueError:
        raise ToolError(401, "malformed_signature", "Malformed signature header.") from None

    if age > MAX_SIGNATURE_AGE_SECONDS:
        raise ToolError(401, "signature_expired", "Signature too old.")

    expected = hmac.new(
        secret.encode("utf-8"),
        f"{timestamp}.{raw_body.decode('utf-8', errors='replace')}".encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()

    if not hmac.compare_digest(expected, provided):
        logger.warning("rejected a post-call webhook with an invalid signature")
        raise ToolError(401, "invalid_signature", "Invalid signature.")


def _session_from_payload(data: dict[str, Any]) -> str | None:
    """Find our session id in the conversation's dynamic variables.

    The session id travels into the conversation as a dynamic variable, so it
    comes back on the post-call payload. Several shapes are tolerated because
    the exact nesting has varied across ElevenLabs API versions.
    """
    candidates = (
        data.get("conversation_initiation_client_data", {}),
        data.get("metadata", {}),
        data,
    )
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        variables = candidate.get("dynamic_variables")
        if isinstance(variables, dict) and variables.get("session_id"):
            return str(variables["session_id"])
    return None


def _sanitise(message: str) -> str:
    """Redact anything in a transcript line that looks like a credential."""
    return _LONG_DIGIT_RUN.sub("[redacted]", message)


def _write_transcript(
    session_id: str, conversation_id: str, data: dict[str, Any]
) -> tuple[Path | None, int]:
    """Save a sanitised transcript and return ``(path, turns_written)``.

    Only the file goes to disk; the session keeps a filename and a turn count,
    never the text. Missing or malformed transcript fields are not an error -
    plenty of payload types carry no transcript at all.
    """
    turns = data.get("transcript")
    if not isinstance(turns, list) or not turns:
        return None, 0

    lines = [
        f"ElevenLabs conversation {conversation_id}",
        f"backend session {session_id}",
        "=" * 66,
        "",
    ]
    written = 0
    for turn in turns[:MAX_TRANSCRIPT_TURNS]:
        if not isinstance(turn, dict):
            continue
        role = str(turn.get("role", "?"))
        message = str(turn.get("message") or "").strip()
        if message:
            lines.append(f"{role.capitalize()}: {_sanitise(message)}")
            written += 1

    if not written:
        return None, 0

    if len(turns) > MAX_TRANSCRIPT_TURNS:
        lines.append(f"\n[truncated after {MAX_TRANSCRIPT_TURNS} turns]")

    directory = _transcript_dir()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"voice-{session_id}.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path, written


def _already_finalized(session_id: str, conversation_id: str) -> bool:
    """Has this conversation already been finalized for this session?

    Webhook delivery is at-least-once. Keyed on the provider's conversation
    id so a genuine second conversation on the same session still records,
    while a replay of the same one does not.
    """
    for event in store.get_events(session_id):
        if event.get("event_type") != EventType.VOICE_CALL_COMPLETED.value:
            continue
        if (event.get("detail") or {}).get("conversation_id") == conversation_id:
            return True
    return False


@router.post(
    "/elevenlabs",
    summary="Post-call webhook (metadata only; never changes payment state)",
)
async def elevenlabs_post_call(
    request: Request,
    elevenlabs_signature: str | None = Header(default=None, alias="elevenlabs-signature"),
) -> dict[str, Any]:
    """Record completion metadata for a finished conversation."""
    raw_body = await request.body()
    if len(raw_body) > MAX_BODY_BYTES:
        raise ToolError(413, "payload_too_large", "That payload is too large.")

    _verify_signature(raw_body, elevenlabs_signature)

    try:
        payload = json.loads(raw_body)
    except json.JSONDecodeError:
        raise ToolError(400, "invalid_json", "That payload was not valid JSON.") from None
    if not isinstance(payload, dict):
        raise ToolError(400, "invalid_payload", "That payload was not an object.")

    event_type = str(payload.get("type", "unknown"))
    data = payload.get("data")
    if not isinstance(data, dict):
        data = {}

    conversation_id = str(data.get("conversation_id") or "unknown")
    session_id = _session_from_payload(data)
    metadata = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
    duration = metadata.get("call_duration_secs")

    logger.info(
        "post-call webhook: type=%s conversation=%s session=%s duration=%s",
        event_type,
        conversation_id,
        session_id or "-",
        duration,
    )

    if not session_id:
        # Nothing to attach it to. Accepted so ElevenLabs does not retry, but
        # explicitly recorded as unmatched.
        return {
            "received": True,
            "type": event_type,
            "conversation_id": conversation_id,
            "matched_session": False,
            "recorded": False,
        }

    try:
        session = store.get_session(session_id)
    except store.SessionNotFoundError:
        logger.warning("post-call webhook named unknown session %s", session_id)
        return {
            "received": True,
            "type": event_type,
            "conversation_id": conversation_id,
            "matched_session": False,
            "recorded": False,
        }

    # Idempotency: a replayed delivery is acknowledged, not re-recorded.
    if _already_finalized(session_id, conversation_id):
        logger.info(
            "post-call webhook for conversation %s already finalized; ignoring replay",
            conversation_id,
        )
        return {
            "received": True,
            "type": event_type,
            "conversation_id": conversation_id,
            "matched_session": True,
            "recorded": False,
            "duplicate": True,
            "outcome": session.outcome.value if session.outcome else None,
        }

    transcript_path, turns = _write_transcript(session_id, conversation_id, data)

    # Finalise with what actually arrived. Absent fields are left alone rather
    # than written as None, so a sparse delivery cannot erase what a previous
    # one recorded, and nothing here touches payment, disposition or
    # verification state - the tool calls own those.
    finalisation: dict[str, Any] = {
        "completed_at": utc_now(),
        "conversation_id": conversation_id,
    }
    if isinstance(duration, (int, float)):
        finalisation["call_duration_seconds"] = int(duration)
    if transcript_path is not None:
        finalisation["transcript_turns"] = turns
        finalisation["transcript_file"] = transcript_path.name

    session = store.update_session(session_id, **finalisation)

    outcome = session.outcome.value if session.outcome else "unresolved"
    store.record_event(
        session_id,
        session.customer_id,
        EventType.VOICE_CALL_COMPLETED,
        f"Call completed. Final outcome: {outcome}.",
        {
            "conversation_id": conversation_id,
            "event_type": event_type,
            "outcome": outcome,
            "completed_at": session.completed_at.isoformat()
            if session.completed_at
            else "",
            "duration_seconds": str(duration) if duration is not None else "unknown",
            "transcript_turns": str(turns) if turns else "0",
        },
    )

    return {
        "received": True,
        "type": event_type,
        "conversation_id": conversation_id,
        "matched_session": True,
        "recorded": True,
        "duplicate": False,
        "outcome": outcome,
        "transcript_saved": bool(transcript_path),
    }
