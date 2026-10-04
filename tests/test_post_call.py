"""Tests for post-call finalisation.

The post-call webhook is the one endpoint a third party POSTs to, and it runs
after the conversation is over — when nobody is watching. Two properties
matter most and are tested hardest here:

* **It cannot change what happened.** Payment state, dispositions and
  verification are written only by the tool calls, during the call, under our
  own validation. A transcript is a model's account of a conversation; it is
  evidence, not an authority.
* **It is idempotent.** Delivery is at-least-once, so a replay must not
  produce a second completion event or disturb the first one's record.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app import store
from app.config import settings
from app.main import app
from app.models import Disposition

WEBHOOK_SECRET = "wsec_phase7_test_secret"
TOOL_SECRET = "tool-secret-phase7"
AUTH = {"X-Tool-Secret": TOOL_SECRET}
CONVERSATION_ID = "conv_phase7_1"


@pytest.fixture(autouse=True)
def configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "elevenlabs_webhook_secret", WEBHOOK_SECRET)
    monkeypatch.setattr(settings, "tool_shared_secret", TOOL_SECRET)


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def new_session(client: TestClient, customer_id: str = "CUST-001") -> str:
    response = client.post(
        "/api/sessions", json={"customer_id": customer_id, "channel": "web"}
    )
    assert response.status_code == 201
    return response.json()["session_id"]


def tool(client: TestClient, name: str, payload: dict[str, Any]) -> Any:
    return client.post(f"/tools/{name}", json=payload, headers=AUTH)


def sign(body: bytes, secret: str = WEBHOOK_SECRET, timestamp: int | None = None) -> str:
    stamp = timestamp if timestamp is not None else int(time.time())
    digest = hmac.new(
        secret.encode(), f"{stamp}.{body.decode()}".encode(), hashlib.sha256
    ).hexdigest()
    return f"t={stamp},v0={digest}"


def payload(
    session_id: str,
    *,
    conversation_id: str = CONVERSATION_ID,
    duration: int | None = 95,
    transcript: list[dict[str, str]] | None = None,
    extra_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """A post-call payload shaped the way ElevenLabs sends one."""
    data: dict[str, Any] = {
        "conversation_id": conversation_id,
        "conversation_initiation_client_data": {
            "dynamic_variables": {"session_id": session_id}
        },
    }
    if duration is not None:
        data["metadata"] = {"call_duration_secs": duration}
    if transcript is None:
        transcript = [
            {"role": "agent", "message": "Hello, this is Ava."},
            {"role": "user", "message": "Yes, go ahead."},
        ]
    if transcript:
        data["transcript"] = transcript
    if extra_data:
        data.update(extra_data)
    return {
        "type": "post_call_transcription",
        "event_timestamp": int(time.time()),
        "data": data,
    }


def post(client: TestClient, body_dict: dict[str, Any], **kwargs: Any) -> Any:
    body = json.dumps(body_dict).encode()
    headers = {"elevenlabs-signature": kwargs.pop("signature", sign(body))}
    return client.post("/webhooks/elevenlabs", content=body, headers=headers)


def recovered_session(client: TestClient) -> str:
    """A session driven to a real recovery, the way the agent drives it."""
    session_id = new_session(client)
    tool(client, "verify_identity", {"session_id": session_id, "postal_code": "94107"})
    tool(client, "get_failed_payment_details", {"session_id": session_id})
    tool(client, "retry_payment", {"session_id": session_id})
    tool(
        client,
        "log_disposition",
        {"session_id": session_id, "disposition": "payment_recovered"},
    )
    return session_id


# ---------------------------------------------------------------------------
# The session-level outcome
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("customer_id", "postal_code", "disposition"),
    [
        ("CUST-002", "78702", "payment_link_prepared"),
        ("CUST-004", "60614", "retry_scheduled"),
        ("CUST-005", "30309", "customer_declined"),
        ("CUST-006", "55401", "escalated"),
        ("CUST-007", "19104", "verification_failed"),
        ("CUST-009", "85004", "unresolved"),
        ("CUST-010", "73102", "do_not_call"),
    ],
)
def test_every_existing_disposition_becomes_the_session_outcome(
    client: TestClient, customer_id: str, postal_code: str, disposition: str
) -> None:
    """The outcome is the disposition. No new vocabulary was invented."""
    session_id = new_session(client, customer_id)
    tool(
        client, "verify_identity", {"session_id": session_id, "postal_code": postal_code}
    )
    tool(
        client, "log_disposition", {"session_id": session_id, "disposition": disposition}
    )

    assert store.get_session(session_id).outcome is Disposition(disposition)
    assert client.get(f"/api/sessions/{session_id}").json()["outcome"] == disposition


def test_payment_recovered_is_the_outcome_of_a_real_recovery(
    client: TestClient,
) -> None:
    session_id = recovered_session(client)
    state = client.get(f"/api/sessions/{session_id}").json()
    assert state["outcome"] == "payment_recovered"
    assert state["payment_status"] == "recovered"


def test_the_outcome_belongs_to_its_own_session(client: TestClient) -> None:
    """A second conversation must not inherit the first one's outcome.

    The customer row keeps a last-known disposition, which is per-customer by
    design; the session view must report the conversation's own result.
    """
    first = recovered_session(client)
    assert client.get(f"/api/sessions/{first}").json()["outcome"] == "payment_recovered"

    second = new_session(client)
    assert store.get_session(second).outcome is None
    assert client.get(f"/api/sessions/{second}").json()["outcome"] is None


def test_an_unfinished_session_has_no_outcome(client: TestClient) -> None:
    session_id = new_session(client)
    tool(client, "verify_identity", {"session_id": session_id, "postal_code": "94107"})
    assert client.get(f"/api/sessions/{session_id}").json()["outcome"] is None


def test_reset_clears_the_outcome_and_completion(client: TestClient) -> None:
    session_id = recovered_session(client)
    post(client, payload(session_id))
    assert client.get(f"/api/sessions/{session_id}").json()["call_completed"] is True

    client.post(f"/api/sessions/{session_id}/reset")
    state = client.get(f"/api/sessions/{session_id}").json()
    assert state["outcome"] is None
    assert state["call_completed"] is False
    assert state["completed_at"] is None
    assert state["call_duration_seconds"] is None


# ---------------------------------------------------------------------------
# Finalisation
# ---------------------------------------------------------------------------


def test_a_valid_webhook_finalizes_the_session(client: TestClient) -> None:
    session_id = recovered_session(client)
    response = post(client, payload(session_id))

    assert response.status_code == 200
    body = response.json()
    assert body["received"] is True
    assert body["matched_session"] is True
    assert body["recorded"] is True
    assert body["duplicate"] is False
    assert body["outcome"] == "payment_recovered"

    session = store.get_session(session_id)
    assert session.completed_at is not None
    assert session.conversation_id == CONVERSATION_ID
    assert session.call_duration_seconds == 95
    assert session.transcript_turns == 2


def test_the_dashboard_shows_the_call_as_completed(client: TestClient) -> None:
    session_id = recovered_session(client)
    before = client.get(f"/api/sessions/{session_id}").json()
    assert before["call_completed"] is False

    post(client, payload(session_id))

    after = client.get(f"/api/sessions/{session_id}").json()
    assert after["call_completed"] is True
    assert after["completed_at"] is not None
    assert after["call_duration_seconds"] == 95
    assert after["transcript_turns"] == 2
    assert after["outcome"] == "payment_recovered"


def test_the_completion_event_carries_metadata_and_the_outcome(
    client: TestClient,
) -> None:
    session_id = recovered_session(client)
    post(client, payload(session_id))

    events = client.get(f"/api/sessions/{session_id}/events").json()
    completed = [e for e in events if e["event_type"] == "voice_call_completed"]
    assert len(completed) == 1

    detail = completed[0]["detail"]
    assert detail["conversation_id"] == CONVERSATION_ID
    assert detail["outcome"] == "payment_recovered"
    assert detail["duration_seconds"] == "95"
    assert detail["completed_at"]
    assert "Final outcome: payment_recovered" in completed[0]["summary"]


def test_the_completion_event_is_last_on_the_timeline(client: TestClient) -> None:
    session_id = recovered_session(client)
    post(client, payload(session_id))

    events = client.get(f"/api/sessions/{session_id}/events").json()
    assert events[-1]["event_type"] == "voice_call_completed"
    sequences = [event["sequence"] for event in events]
    assert sequences == sorted(sequences)


def test_finalisation_does_not_reopen_a_closed_session(client: TestClient) -> None:
    session_id = recovered_session(client)
    assert client.get(f"/api/sessions/{session_id}").json()["status"] == "closed"
    post(client, payload(session_id))
    assert client.get(f"/api/sessions/{session_id}").json()["status"] == "closed"


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


def test_a_replayed_webhook_records_nothing_twice(client: TestClient) -> None:
    """At-least-once delivery must not produce two completion events."""
    session_id = recovered_session(client)

    first = post(client, payload(session_id))
    assert first.json()["recorded"] is True
    assert first.json()["duplicate"] is False

    for _ in range(3):
        replay = post(client, payload(session_id))
        assert replay.status_code == 200
        assert replay.json()["recorded"] is False
        assert replay.json()["duplicate"] is True
        assert replay.json()["outcome"] == "payment_recovered"

    events = client.get(f"/api/sessions/{session_id}/events").json()
    completed = [e for e in events if e["event_type"] == "voice_call_completed"]
    assert len(completed) == 1, "a replay duplicated the completion event"


def test_a_replay_does_not_disturb_what_was_recorded(client: TestClient) -> None:
    session_id = recovered_session(client)
    post(client, payload(session_id))
    first = store.get_session(session_id)

    # A sparser replay: no duration, no transcript.
    post(client, payload(session_id, duration=None, transcript=[]))

    second = store.get_session(session_id)
    assert second.completed_at == first.completed_at
    assert second.call_duration_seconds == first.call_duration_seconds == 95
    assert second.transcript_turns == first.transcript_turns == 2


def test_a_genuinely_different_conversation_still_records(client: TestClient) -> None:
    """Idempotency keys on the conversation, not on the session."""
    session_id = recovered_session(client)
    post(client, payload(session_id, conversation_id="conv_one"))
    second = post(client, payload(session_id, conversation_id="conv_two"))

    assert second.json()["recorded"] is True
    assert second.json()["duplicate"] is False

    events = client.get(f"/api/sessions/{session_id}/events").json()
    completed = [e for e in events if e["event_type"] == "voice_call_completed"]
    assert {e["detail"]["conversation_id"] for e in completed} == {"conv_one", "conv_two"}


# ---------------------------------------------------------------------------
# It cannot change what happened
# ---------------------------------------------------------------------------


def test_a_webhook_cannot_invent_a_recovery(client: TestClient) -> None:
    """A transcript claiming success must not move the ledger."""
    session_id = new_session(client, "CUST-009")
    tool(client, "verify_identity", {"session_id": session_id, "postal_code": "85004"})
    tool(client, "retry_payment", {"session_id": session_id})  # always declines
    tool(
        client, "log_disposition", {"session_id": session_id, "disposition": "unresolved"}
    )

    post(
        client,
        payload(
            session_id,
            transcript=[
                {"role": "agent", "message": "Your payment of 12.99 went through."},
            ],
            extra_data={
                "analysis": {
                    "data_collection_results": {
                        "payment_status": {"value": "recovered"},
                        "disposition": {"value": "payment_recovered"},
                    }
                }
            },
        ),
    )

    state = client.get(f"/api/sessions/{session_id}").json()
    assert state["payment_status"] == "failed"
    assert state["outcome"] == "unresolved", "the webhook overwrote the real outcome"
    assert state["last_confirmation_number"] is None
    assert store.get_state("CUST-009").status.value == "failed"


def test_a_webhook_cannot_grant_verification(client: TestClient) -> None:
    session_id = new_session(client)
    post(
        client,
        payload(
            session_id,
            extra_data={"analysis": {"data_collection_results": {"verified": True}}},
        ),
    )
    assert store.get_session(session_id).identity_verified is False
    assert client.get(f"/api/sessions/{session_id}").json()["identity_verified"] is False


def test_a_webhook_customer_id_is_ignored(client: TestClient) -> None:
    """Whose account it is comes from the session binding, nothing else."""
    session_id = new_session(client, "CUST-001")
    post(
        client,
        payload(
            session_id,
            extra_data={
                "customer_id": "CUST-008",
                "metadata": {"call_duration_secs": 10, "customer_id": "CUST-008"},
            },
        ),
    )

    events = client.get(f"/api/sessions/{session_id}/events").json()
    completed = [e for e in events if e["event_type"] == "voice_call_completed"]
    assert len(completed) == 1
    assert client.get(f"/api/sessions/{session_id}").json()["customer_id"] == "CUST-001"

    # Nothing was attached to the customer the payload named.
    assert store.get_events("CUST-008") == []
    assert client.get("/api/sessions?customer_id=CUST-008").json() == []


def test_a_webhook_cannot_create_a_payment_attempt(client: TestClient) -> None:
    session_id = new_session(client)
    post(client, payload(session_id))
    assert store.get_payment_attempts("CUST-001") == []


# ---------------------------------------------------------------------------
# Authentication and malformed input
# ---------------------------------------------------------------------------


def test_an_unsigned_webhook_is_rejected(client: TestClient) -> None:
    session_id = new_session(client)
    body = json.dumps(payload(session_id)).encode()
    response = client.post("/webhooks/elevenlabs", content=body)
    assert response.status_code == 401
    assert response.json()["error"] == "missing_signature"
    assert store.get_session(session_id).completed_at is None


def test_a_wrongly_signed_webhook_is_rejected(client: TestClient) -> None:
    session_id = new_session(client)
    body = json.dumps(payload(session_id)).encode()
    response = client.post(
        "/webhooks/elevenlabs",
        content=body,
        headers={"elevenlabs-signature": sign(body, secret="not-the-secret")},
    )
    assert response.status_code == 401
    assert response.json()["error"] == "invalid_signature"
    assert store.get_session(session_id).completed_at is None


def test_an_unknown_session_is_acknowledged_but_not_recorded(
    client: TestClient,
) -> None:
    """Accepted so the provider stops retrying; recorded nowhere."""
    response = post(client, payload("sess_does_not_exist"))
    assert response.status_code == 200
    assert response.json()["matched_session"] is False
    assert response.json()["recorded"] is False
    assert store.get_events("sess_does_not_exist") == []


def test_a_payload_without_a_session_id_is_acknowledged(client: TestClient) -> None:
    body = {
        "type": "post_call_transcription",
        "data": {"conversation_id": "conv_orphan", "transcript": []},
    }
    response = post(client, body)
    assert response.status_code == 200
    assert response.json()["matched_session"] is False


def test_malformed_json_is_rejected_without_a_traceback(client: TestClient) -> None:
    body = b"{not json at all"
    response = client.post(
        "/webhooks/elevenlabs",
        content=body,
        headers={"elevenlabs-signature": sign(body)},
    )
    assert response.status_code == 400
    assert response.json()["error"] == "invalid_json"
    assert "Traceback" not in response.text


def test_a_non_object_payload_is_rejected(client: TestClient) -> None:
    body = b"[1, 2, 3]"
    response = client.post(
        "/webhooks/elevenlabs",
        content=body,
        headers={"elevenlabs-signature": sign(body)},
    )
    assert response.status_code == 400
    assert response.json()["error"] == "invalid_payload"


def test_a_payload_with_no_data_block_is_tolerated(client: TestClient) -> None:
    response = post(client, {"type": "post_call_transcription"})
    assert response.status_code == 200
    assert response.json()["matched_session"] is False


# ---------------------------------------------------------------------------
# Optional and partial fields
# ---------------------------------------------------------------------------


def test_a_missing_transcript_is_not_an_error(client: TestClient) -> None:
    session_id = recovered_session(client)
    response = post(client, payload(session_id, transcript=[]))

    assert response.status_code == 200
    assert response.json()["recorded"] is True
    assert response.json()["transcript_saved"] is False

    session = store.get_session(session_id)
    assert session.completed_at is not None
    assert session.transcript_turns is None
    assert session.transcript_file is None


def test_a_missing_duration_is_not_an_error(client: TestClient) -> None:
    session_id = recovered_session(client)
    response = post(client, payload(session_id, duration=None))

    assert response.status_code == 200
    session = store.get_session(session_id)
    assert session.completed_at is not None
    assert session.call_duration_seconds is None


def test_a_malformed_transcript_entry_is_skipped(client: TestClient) -> None:
    session_id = recovered_session(client)
    response = post(
        client,
        payload(
            session_id,
            transcript=[
                {"role": "agent", "message": "Hello."},
                "not a dict",  # type: ignore[list-item]
                {"role": "user", "message": ""},
                {"no_message_key": "x"},  # type: ignore[dict-item]
            ],
        ),
    )
    assert response.status_code == 200
    assert store.get_session(session_id).transcript_turns == 1


def test_a_post_call_audio_event_is_handled(client: TestClient) -> None:
    """Other payload types carry no transcript at all."""
    session_id = recovered_session(client)
    body = {
        "type": "post_call_audio",
        "data": {
            "conversation_id": "conv_audio",
            "conversation_initiation_client_data": {
                "dynamic_variables": {"session_id": session_id}
            },
        },
    }
    response = post(client, body)
    assert response.status_code == 200
    assert response.json()["recorded"] is True
    assert response.json()["type"] == "post_call_audio"


# ---------------------------------------------------------------------------
# Nothing sensitive is stored
# ---------------------------------------------------------------------------


def test_the_stored_session_holds_no_transcript_body(client: TestClient) -> None:
    """Only a filename and a count reach the runtime state."""
    session_id = recovered_session(client)
    post(
        client,
        payload(
            session_id,
            transcript=[{"role": "agent", "message": "A memorable sentence here."}],
        ),
    )

    stored = store.get_session(session_id).model_dump_json()
    assert "A memorable sentence here." not in stored
    assert "voice-" in stored  # the filename reference


def test_the_completion_event_holds_no_transcript_body(client: TestClient) -> None:
    session_id = recovered_session(client)
    post(
        client,
        payload(
            session_id,
            transcript=[{"role": "agent", "message": "A memorable sentence here."}],
        ),
    )
    events = json.dumps(client.get(f"/api/sessions/{session_id}/events").json())
    assert "A memorable sentence here." not in events


def test_no_secret_reaches_the_runtime_state_or_the_response(
    client: TestClient,
) -> None:
    session_id = recovered_session(client)
    response = post(client, payload(session_id))

    blob = response.text + store.get_session(session_id).model_dump_json()
    blob += json.dumps(client.get(f"/api/sessions/{session_id}/events").json())
    for secret in (WEBHOOK_SECRET, TOOL_SECRET):
        assert secret not in blob


def test_a_credential_read_aloud_is_redacted_from_the_transcript(
    client: TestClient, isolated_runtime: Path
) -> None:
    """The agent refuses card numbers, but if one is spoken anyway it does
    not reach disk intact."""
    session_id = recovered_session(client)
    post(
        client,
        payload(
            session_id,
            transcript=[
                {"role": "user", "message": "My card is 4111111111111111, use that."}
            ],
        ),
    )

    saved = store.get_session(session_id).transcript_file
    assert saved
    written = (isolated_runtime.parent / "transcripts" / saved).read_text(
        encoding="utf-8"
    )
    assert "4111111111111111" not in written
    assert "[redacted]" in written


def test_a_long_transcript_is_capped(client: TestClient) -> None:
    from app.routers.webhooks import MAX_TRANSCRIPT_TURNS

    session_id = recovered_session(client)
    post(
        client,
        payload(
            session_id,
            transcript=[
                {"role": "agent", "message": f"turn {index}"}
                for index in range(MAX_TRANSCRIPT_TURNS + 50)
            ],
        ),
    )
    assert store.get_session(session_id).transcript_turns == MAX_TRANSCRIPT_TURNS


def test_transcripts_are_written_outside_the_repository_in_tests(
    client: TestClient, isolated_runtime: Path
) -> None:
    """Guard against the suite scattering files into demo/transcripts."""
    session_id = recovered_session(client)
    post(client, payload(session_id))

    saved = store.get_session(session_id).transcript_file
    assert saved
    assert (isolated_runtime.parent / "transcripts" / saved).exists()
    repo_copy = Path("demo/transcripts") / saved
    assert not repo_copy.exists()


# ---------------------------------------------------------------------------
# Constraints that must still hold
# ---------------------------------------------------------------------------


def test_the_webhook_needs_no_elevenlabs_client() -> None:
    """Signature checking is local; the endpoint works with no API key."""
    text = Path("app/routers/webhooks.py").read_text(encoding="utf-8")
    assert "from elevenlabs" not in text
    assert "get_client" not in text
    assert "twilio" not in text.lower()


def test_the_dashboard_stays_observer_only(client: TestClient) -> None:
    script = client.get("/static/app.js").text
    for call_syntax in ('api("/tools', "api('/tools", 'fetch("/tools'):
        assert call_syntax not in script
    assert "TOOL_SHARED_SECRET" not in script
    assert "webhooks" not in script


def test_the_dashboard_gained_no_new_controls(client: TestClient) -> None:
    import re

    page = client.get("/dashboard").text
    assert sorted(set(re.findall(r'<button id="([^"]+)"', page))) == [
        "btn-copy",
        "btn-create",
        "btn-refresh",
        "btn-reset",
    ]
    assert "v0.1.0" not in page
    assert not re.search(r'type="(tel|number)"', page)
    assert not re.search(r"(?i)>\s*(call|dial)\b", page)


def test_dial_safety_is_still_untouched_by_the_voice_path() -> None:
    for module in ("app/routers/webhooks.py", "static/app.js", "static/voice.js"):
        text = Path(module).read_text(encoding="utf-8")
        assert "dial_safety" not in text
        assert "check_dial_allowed" not in text


def test_outbound_calling_is_still_disabled_by_default() -> None:
    from app.config import Settings

    assert Settings(_env_file=None).enable_outbound_calls is False
