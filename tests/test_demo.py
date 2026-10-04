"""Tests for the demo control plane and dashboard.

The data-exposure tests here are the important ones. The seed's postal code is
the verification answer, so any endpoint that leaked an address would hand
over the credential the agent checks — these assert, against the real seed,
that no such value appears in any response.
"""

from __future__ import annotations

import hashlib
import json
import socket
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app import store
from app.config import settings
from app.main import app
from app.models import AutopayStatus, Disposition

SECRET = "test-tool-secret-value"
AUTH = {"X-Tool-Secret": SECRET}


@pytest.fixture(autouse=True)
def configured_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "tool_shared_secret", SECRET)


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def new_session(client: TestClient, customer_id: str) -> str:
    response = client.post("/api/sessions", json={"customer_id": customer_id})
    assert response.status_code == 201, response.text
    return response.json()["session_id"]


def tool(client: TestClient, name: str, payload: dict[str, Any]) -> Any:
    return client.post(f"/tools/{name}", json=payload, headers=AUTH)


# ---------------------------------------------------------------------------
# Session creation
# ---------------------------------------------------------------------------


def test_valid_customer_creates_an_active_session(client: TestClient) -> None:
    response = client.post("/api/sessions", json={"customer_id": "CUST-001"})
    assert response.status_code == 201

    body = response.json()
    assert body["session_id"].startswith("sess_")
    assert body["customer_id"] == "CUST-001"
    assert body["customer_name"] == "Maya Thompson"
    assert body["status"] == "active"
    assert body["channel"] == "web"


def test_unknown_customer_is_rejected(client: TestClient) -> None:
    response = client.post("/api/sessions", json={"customer_id": "CUST-999"})
    assert response.status_code == 404
    assert response.json()["error"] == "unknown_customer"
    assert store.list_sessions() == []


def test_malformed_customer_id_is_rejected(client: TestClient) -> None:
    response = client.post("/api/sessions", json={"customer_id": "bobby-tables"})
    assert response.status_code == 422
    assert response.json()["error"] == "invalid_request"


def test_session_request_rejects_unexpected_fields(client: TestClient) -> None:
    response = client.post(
        "/api/sessions", json={"customer_id": "CUST-001", "identity_verified": True}
    )
    assert response.status_code == 422


def test_session_ids_are_unique(client: TestClient) -> None:
    ids = {new_session(client, "CUST-001") for _ in range(5)}
    assert len(ids) == 5


def test_session_creation_is_the_first_event(client: TestClient) -> None:
    session_id = new_session(client, "CUST-004")
    events = client.get(f"/api/sessions/{session_id}/events").json()
    assert events[0]["event_type"] == "session_created"
    assert events[0]["sequence"] == 1
    assert "Marcus Lee" in events[0]["summary"]


# ---------------------------------------------------------------------------
# Session inspection
# ---------------------------------------------------------------------------


def test_session_state_starts_at_the_seed_defaults(client: TestClient) -> None:
    session_id = new_session(client, "CUST-001")
    body = client.get(f"/api/sessions/{session_id}").json()

    assert body["session_id"] == session_id
    assert body["customer_id"] == "CUST-001"
    assert body["status"] == "active"
    assert body["identity_verified"] is False
    assert body["verification_attempts"] == 0
    assert body["verification_attempts_allowed"] == 2
    assert body["payment_status"] == "failed"
    assert body["amount_due"] == "49.00"
    assert body["retry_attempts"] == 0
    assert body["scheduled_for"] is None
    assert body["payment_link_prepared"] is False
    assert body["escalated"] is False
    assert body["disposition"] is None
    assert body["do_not_call"] is False


def test_unknown_session_state_is_a_404(client: TestClient) -> None:
    response = client.get("/api/sessions/sess_nope")
    assert response.status_code == 404
    assert response.json()["error"] == "unknown_session"


def test_session_state_tracks_tool_activity(client: TestClient) -> None:
    session_id = new_session(client, "CUST-001")
    tool(client, "verify_identity", {"session_id": session_id, "postal_code": "94107"})
    tool(client, "retry_payment", {"session_id": session_id})

    body = client.get(f"/api/sessions/{session_id}").json()
    assert body["identity_verified"] is True
    assert body["verification_attempts"] == 1
    assert body["payment_status"] == "recovered"
    assert body["retry_attempts"] == 1
    assert body["last_confirmation_number"] == "C-0254"
    assert body["tool_calls"] >= 2
    assert body["event_count"] >= 3


def test_session_state_exposes_no_secret_or_credential(client: TestClient) -> None:
    session_id = new_session(client, "CUST-001")
    response = client.get(f"/api/sessions/{session_id}")

    assert SECRET not in response.text
    for forbidden in ("expected_answer", "postal_code", "pm_mock_", "pay_mock_", "94107"):
        assert forbidden not in response.text, f"{forbidden} leaked into session state"


def test_session_state_becomes_closed_after_a_disposition(client: TestClient) -> None:
    session_id = new_session(client, "CUST-010")
    tool(client, "log_disposition", {"session_id": session_id, "disposition": "do_not_call"})

    body = client.get(f"/api/sessions/{session_id}").json()
    assert body["status"] == "closed"
    assert body["disposition"] == "do_not_call"
    assert body["do_not_call"] is True


# ---------------------------------------------------------------------------
# Customer endpoints
# ---------------------------------------------------------------------------


def test_customer_listing_returns_exactly_ten(client: TestClient) -> None:
    body = client.get("/api/customers").json()
    assert len(body) == 10
    assert [row["customer_id"] for row in body] == [f"CUST-{n:03d}" for n in range(1, 11)]


def test_customer_listing_has_the_fields_the_dashboard_needs(client: TestClient) -> None:
    first = client.get("/api/customers").json()[0]
    for field in (
        "customer_id",
        "name",
        "phone",
        "amount",
        "currency",
        "failure_reason",
        "failure_explanation",
        "scenario_label",
        "expected_path",
        "payment_status",
    ):
        assert field in first, f"{field} missing from the customer summary"


def test_customer_detail_is_returned(client: TestClient) -> None:
    body = client.get("/api/customers/CUST-006").json()
    assert body["customer_id"] == "CUST-006"
    assert body["name"] == "Tomas Varga"
    assert body["city"] == "Minneapolis"
    assert body["state"] == "MN"
    assert body["backup_method_available"] is False
    assert body["service_suspension_date"] == "2026-10-20"
    assert body["verification_method"] == "postal_code"
    assert body["attempts_before_this_call"] == 2


def test_unknown_customer_detail_is_a_404(client: TestClient) -> None:
    response = client.get("/api/customers/CUST-999")
    assert response.status_code == 404
    assert response.json()["error"] == "unknown_customer"


def test_customer_endpoints_never_expose_a_verification_answer(
    client: TestClient,
) -> None:
    """The decisive data-exposure test, run against the real seed.

    Every customer's postal code is their verification answer. Neither the
    listing nor any detail response may contain one.
    """
    answers = {c.verification.expected_answer for c in store.list_customers()}
    assert answers, "the seed should define verification answers"

    listing = client.get("/api/customers").text
    for answer in answers:
        assert answer not in listing, f"verification answer {answer} leaked into the listing"

    for customer in store.list_customers():
        detail = client.get(f"/api/customers/{customer.customer_id}").text
        for answer in answers:
            assert answer not in detail, (
                f"verification answer {answer} leaked into {customer.customer_id}"
            )


def test_customer_endpoints_expose_no_internal_identifiers(client: TestClient) -> None:
    listing = client.get("/api/customers").text
    detail = client.get("/api/customers/CUST-001").text
    for text in (listing, detail):
        assert "pm_mock_" not in text
        assert "pay_mock_" not in text
        assert "mock_retry_outcome" not in text
        assert "expected_answer" not in text
        assert "postal_code" not in text or "verification_method" in text


def test_customer_endpoints_never_expose_the_tool_secret(client: TestClient) -> None:
    assert SECRET not in client.get("/api/customers").text
    assert SECRET not in client.get("/api/customers/CUST-001").text


def test_customer_listing_reflects_the_live_ledger(client: TestClient) -> None:
    session_id = new_session(client, "CUST-001")
    tool(client, "verify_identity", {"session_id": session_id, "postal_code": "94107"})
    tool(client, "retry_payment", {"session_id": session_id})

    rows = {row["customer_id"]: row for row in client.get("/api/customers").json()}
    assert rows["CUST-001"]["payment_status"] == "recovered"
    assert rows["CUST-002"]["payment_status"] == "failed"


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------


def test_events_are_chronological(client: TestClient) -> None:
    session_id = new_session(client, "CUST-001")
    tool(client, "get_failed_payment_details", {"session_id": session_id})
    tool(client, "verify_identity", {"session_id": session_id, "postal_code": "94107"})
    tool(client, "get_failed_payment_details", {"session_id": session_id})
    tool(client, "retry_payment", {"session_id": session_id})
    tool(
        client,
        "log_disposition",
        {"session_id": session_id, "disposition": "payment_recovered"},
    )

    events = client.get(f"/api/sessions/{session_id}/events").json()
    sequences = [event["sequence"] for event in events]
    assert sequences == sorted(sequences)
    assert sequences == list(range(1, len(sequences) + 1))

    assert [event["event_type"] for event in events] == [
        "session_created",
        "payment_details_withheld",
        "identity_verified",
        "payment_details_viewed",
        "payment_retry_attempted",
        "payment_retry_succeeded",
        "disposition_logged",
    ]


def test_tool_actions_generate_the_expected_events(client: TestClient) -> None:
    session_id = new_session(client, "CUST-002")
    tool(client, "verify_identity", {"session_id": session_id, "postal_code": "78702"})
    tool(client, "retry_payment", {"session_id": session_id})
    tool(client, "send_payment_link", {"session_id": session_id, "channel": "email"})
    tool(client, "escalate_to_human", {"session_id": session_id, "reason": "needs help"})

    types = [e["event_type"] for e in client.get(f"/api/sessions/{session_id}/events").json()]
    assert "identity_verified" in types
    assert "payment_retry_skipped" in types
    assert "payment_link_prepared" in types
    assert "human_escalation_created" in types


def test_scheduling_generates_an_event(client: TestClient) -> None:
    session_id = new_session(client, "CUST-004")
    tool(client, "verify_identity", {"session_id": session_id, "postal_code": "60614"})
    when = (date.today() + timedelta(days=9)).isoformat()
    tool(client, "schedule_retry", {"session_id": session_id, "requested_date": when})

    events = client.get(f"/api/sessions/{session_id}/events").json()
    scheduled = [e for e in events if e["event_type"] == "payment_scheduled"]
    assert len(scheduled) == 1
    assert scheduled[0]["detail"]["scheduled_for"] == when


def test_verification_failure_events_do_not_record_the_answer(
    client: TestClient,
) -> None:
    """Neither the expected answer nor the caller's guess may be logged."""
    session_id = new_session(client, "CUST-007")
    # Both guesses are wrong, which is this customer's scripted scenario:
    # 19104 is the real answer and is deliberately never submitted here.
    tool(client, "verify_identity", {"session_id": session_id, "postal_code": "19147"})
    tool(client, "verify_identity", {"session_id": session_id, "postal_code": "19148"})

    text = client.get(f"/api/sessions/{session_id}/events").text
    assert "19104" not in text, "the verification answer must never be logged"
    assert "19147" not in text, "the caller's submitted code must never be logged"
    assert "19148" not in text, "the caller's submitted code must never be logged"

    types = [e["event_type"] for e in client.get(f"/api/sessions/{session_id}/events").json()]
    assert "identity_verification_failed" in types
    assert "identity_verification_locked" in types


def test_events_are_scoped_to_their_session(client: TestClient) -> None:
    first = new_session(client, "CUST-001")
    second = new_session(client, "CUST-005")
    tool(client, "verify_identity", {"session_id": first, "postal_code": "94107"})

    first_events = client.get(f"/api/sessions/{first}/events").json()
    second_events = client.get(f"/api/sessions/{second}/events").json()
    assert any(e["event_type"] == "identity_verified" for e in first_events)
    assert not any(e["event_type"] == "identity_verified" for e in second_events)
    assert len(second_events) == 1  # just its own session_created


def test_unknown_session_events_is_a_404(client: TestClient) -> None:
    response = client.get("/api/sessions/sess_nope/events")
    assert response.status_code == 404
    assert response.json()["error"] == "unknown_session"


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------


def test_reset_clears_session_state_and_reopens_it(client: TestClient) -> None:
    session_id = new_session(client, "CUST-001")
    tool(client, "verify_identity", {"session_id": session_id, "postal_code": "94107"})
    tool(client, "retry_payment", {"session_id": session_id})
    tool(
        client,
        "log_disposition",
        {"session_id": session_id, "disposition": "payment_recovered"},
    )
    assert client.get(f"/api/sessions/{session_id}").json()["status"] == "closed"

    response = client.post(f"/api/sessions/{session_id}/reset")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "active"
    assert body["records_cleared"] > 0
    assert body["seed_unchanged"] is True

    state = client.get(f"/api/sessions/{session_id}").json()
    assert state["status"] == "active"
    assert state["identity_verified"] is False
    assert state["payment_status"] == "failed"
    assert state["retry_attempts"] == 0
    assert state["disposition"] is None
    assert state["last_confirmation_number"] is None
    assert store.get_payment_attempts("CUST-001") == []


def test_the_same_scenario_can_be_run_again_after_reset(client: TestClient) -> None:
    """The whole point of reset: repeatable demos on one session id."""
    session_id = new_session(client, "CUST-001")

    for run in range(2):
        tool(client, "verify_identity", {"session_id": session_id, "postal_code": "94107"})
        retry = tool(client, "retry_payment", {"session_id": session_id}).json()
        assert retry["status"] == "paid", f"run {run} should recover"
        assert retry["confirmation_number"] == "C-0254", "deterministic across runs"
        tool(
            client,
            "log_disposition",
            {"session_id": session_id, "disposition": "payment_recovered"},
        )
        if run == 0:
            client.post(f"/api/sessions/{session_id}/reset")


def test_reset_leaves_the_seed_file_byte_identical(client: TestClient) -> None:
    seed = Path(__file__).resolve().parent.parent / "data" / "customers.json"
    before = hashlib.sha256(seed.read_bytes()).hexdigest()

    session_id = new_session(client, "CUST-003")
    tool(client, "verify_identity", {"session_id": session_id, "postal_code": "02139"})
    tool(client, "retry_payment", {"session_id": session_id})
    tool(client, "retry_payment", {"session_id": session_id, "payment_method": "backup"})
    client.post(f"/api/sessions/{session_id}/reset")

    assert hashlib.sha256(seed.read_bytes()).hexdigest() == before


def test_reset_does_not_affect_another_session(client: TestClient) -> None:
    first = new_session(client, "CUST-001")
    second = new_session(client, "CUST-005")

    tool(client, "verify_identity", {"session_id": first, "postal_code": "94107"})
    tool(client, "retry_payment", {"session_id": first})
    tool(client, "verify_identity", {"session_id": second, "postal_code": "30309"})
    tool(client, "retry_payment", {"session_id": second})

    client.post(f"/api/sessions/{first}/reset")

    reset_state = client.get(f"/api/sessions/{first}").json()
    kept_state = client.get(f"/api/sessions/{second}").json()

    assert reset_state["payment_status"] == "failed"
    assert reset_state["retry_attempts"] == 0
    assert kept_state["payment_status"] == "recovered"
    assert kept_state["retry_attempts"] == 1
    assert store.get_payment_attempts("CUST-005")
    assert store.get_payment_attempts("CUST-001") == []


def test_reset_keeps_other_sessions_events(client: TestClient) -> None:
    first = new_session(client, "CUST-001")
    second = new_session(client, "CUST-005")
    tool(client, "verify_identity", {"session_id": second, "postal_code": "30309"})

    client.post(f"/api/sessions/{first}/reset")

    assert len(client.get(f"/api/sessions/{second}/events").json()) == 2
    first_events = client.get(f"/api/sessions/{first}/events").json()
    assert [e["event_type"] for e in first_events] == ["session_reset"]


def test_reset_of_an_unknown_session_is_a_404(client: TestClient) -> None:
    response = client.post("/api/sessions/sess_nope/reset")
    assert response.status_code == 404


def test_reset_does_not_delete_the_runtime_file(
    client: TestClient, isolated_runtime: Path
) -> None:
    first = new_session(client, "CUST-001")
    new_session(client, "CUST-005")
    client.post(f"/api/sessions/{first}/reset")

    document = json.loads(isolated_runtime.read_text(encoding="utf-8"))
    assert len(document["sessions"]) == 2


# ---------------------------------------------------------------------------
# Dashboard page and assets
# ---------------------------------------------------------------------------


def test_dashboard_renders(client: TestClient) -> None:
    response = client.get("/dashboard")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Autopay Recovery" in response.text


def test_dashboard_never_contains_the_tool_secret(client: TestClient) -> None:
    """The page must not carry the secret, which is why it cannot call tools."""
    assert SECRET not in client.get("/dashboard").text


def test_dashboard_carries_no_customer_data(client: TestClient) -> None:
    """Data arrives from /api/* after load, so the template stays inert."""
    page = client.get("/dashboard").text
    assert "Maya Thompson" not in page
    assert "94107" not in page


def test_dashboard_assets_load(client: TestClient) -> None:
    css = client.get("/static/styles.css")
    assert css.status_code == 200
    assert "text/css" in css.headers["content-type"]

    js = client.get("/static/app.js")
    assert js.status_code == 200
    assert "/api/sessions" in js.text


def test_dashboard_assets_contain_no_secret_and_no_tool_calls(client: TestClient) -> None:
    js = client.get("/static/app.js").text
    assert SECRET not in js
    assert "TOOL_SHARED_SECRET" not in js
    for call_syntax in ('api("/tools', "api('/tools", 'fetch("/tools', "fetch('/tools"):
        assert call_syntax not in js, "the browser must never call a tool endpoint"


def test_root_points_at_the_dashboard(client: TestClient) -> None:
    body = client.get("/").json()
    assert body["dashboard"] == "/dashboard"
    assert body["health"] == "/healthz"


def test_healthz_still_works(client: TestClient) -> None:
    body = client.get("/healthz").json()
    assert body["status"] == "ok"
    assert all(isinstance(value, bool) for value in body["config"].values())


# ---------------------------------------------------------------------------
# Integration flows
# ---------------------------------------------------------------------------


def test_full_recovery_flow_through_the_control_plane(client: TestClient) -> None:
    """create -> inspect -> verify -> details -> retry -> close -> reset."""
    session_id = new_session(client, "CUST-001")

    assert client.get(f"/api/sessions/{session_id}").json()["identity_verified"] is False

    assert tool(
        client, "verify_identity", {"session_id": session_id, "postal_code": "94107"}
    ).json()["verified"]

    details = tool(client, "get_failed_payment_details", {"session_id": session_id}).json()
    assert details["amount"] == "49.00"

    retry = tool(client, "retry_payment", {"session_id": session_id}).json()
    assert retry["status"] == "paid"

    tool(
        client,
        "log_disposition",
        {"session_id": session_id, "disposition": "payment_recovered", "notes": "first retry"},
    )

    state = client.get(f"/api/sessions/{session_id}").json()
    assert state["payment_status"] == "recovered"
    assert state["disposition"] == "payment_recovered"
    assert state["disposition_notes"] == "first retry"
    assert state["status"] == "closed"

    events = client.get(f"/api/sessions/{session_id}/events").json()
    assert [e["event_type"] for e in events][-1] == "disposition_logged"

    client.post(f"/api/sessions/{session_id}/reset")
    after = client.get(f"/api/sessions/{session_id}").json()
    assert after["payment_status"] == "failed"
    assert after["status"] == "active"


def test_escalation_flow_through_the_control_plane(client: TestClient) -> None:
    """A second, structurally different scenario: unrecoverable -> human."""
    session_id = new_session(client, "CUST-006")

    tool(client, "verify_identity", {"session_id": session_id, "postal_code": "55401"})
    details = tool(client, "get_failed_payment_details", {"session_id": session_id}).json()
    assert details["next_action"] == "escalate"

    retry = tool(client, "retry_payment", {"session_id": session_id}).json()
    assert retry["status"] == "not_attempted"

    escalation = tool(
        client,
        "escalate_to_human",
        {"session_id": session_id, "reason": "bank closed the account"},
    ).json()
    assert escalation["ticket_id"].startswith("TCK-")

    tool(client, "log_disposition", {"session_id": session_id, "disposition": "escalated"})

    state = client.get(f"/api/sessions/{session_id}").json()
    assert state["escalated"] is True
    assert state["escalation_ticket"] == escalation["ticket_id"]
    assert state["payment_status"] == "escalated"
    assert state["disposition"] == "escalated"


def test_payment_link_flow_through_the_control_plane(client: TestClient) -> None:
    """A third scenario: expired card -> prepared link, never delivered."""
    session_id = new_session(client, "CUST-002")

    tool(client, "verify_identity", {"session_id": session_id, "postal_code": "78702"})
    tool(client, "retry_payment", {"session_id": session_id})
    link = tool(
        client, "send_payment_link", {"session_id": session_id, "channel": "email"}
    ).json()
    assert link["delivered"] is False

    tool(
        client,
        "log_disposition",
        {"session_id": session_id, "disposition": "payment_link_prepared"},
    )

    state = client.get(f"/api/sessions/{session_id}").json()
    assert state["payment_link_prepared"] is True
    assert state["payment_status"] == "link_sent"
    assert state["disposition"] == "payment_link_prepared"


# ---------------------------------------------------------------------------
# Offline guarantee
# ---------------------------------------------------------------------------


def test_the_control_plane_makes_no_external_connections(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Loopback stays open for the test harness; everything else is refused."""
    real_connect = socket.socket.connect
    real_getaddrinfo = socket.getaddrinfo
    loopback = {"127.0.0.1", "::1", "localhost", "testserver"}

    def guarded_connect(self: socket.socket, address: object) -> object:
        host = str(address[0]) if isinstance(address, tuple) and address else str(address)
        if host not in loopback:
            raise AssertionError(f"external connection attempted to {host}")
        return real_connect(self, address)

    def guarded_getaddrinfo(host: object, *args: object, **kwargs: object) -> object:
        if host is not None and str(host) not in loopback:
            raise AssertionError(f"DNS lookup attempted for {host}")
        return real_getaddrinfo(host, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket, "getaddrinfo", guarded_getaddrinfo)

    session_id = new_session(client, "CUST-005")
    assert client.get("/dashboard").status_code == 200
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/api/customers").status_code == 200
    assert client.get("/api/customers/CUST-005").status_code == 200
    assert client.get(f"/api/sessions/{session_id}").status_code == 200
    assert client.get(f"/api/sessions/{session_id}/events").status_code == 200
    assert client.post(f"/api/sessions/{session_id}/reset").status_code == 200


def test_phase_three_modules_import_no_provider_sdk() -> None:
    banned = ("elevenlabs", "twilio", "import requests", "import httpx", "stripe")
    for module in ("app/routers/demo.py", "app/routers/pages.py", "static/app.js"):
        text = Path(module).read_text(encoding="utf-8")
        for needle in banned:
            assert needle not in text, f"{module} must not reference {needle}"


def test_disposition_enum_is_still_closed(client: TestClient) -> None:
    """Guard against the control plane widening what the tools accept."""
    session_id = new_session(client, "CUST-001")
    response = tool(
        client, "log_disposition", {"session_id": session_id, "disposition": "whatever"}
    )
    assert response.status_code == 422
    assert set(Disposition) and AutopayStatus.RECOVERED.value == "recovered"


# ---------------------------------------------------------------------------
# Phase 4 UI reduction
# ---------------------------------------------------------------------------


def test_dashboard_has_no_version_label(client: TestClient) -> None:
    """A template version badge carries no demo value and was removed."""
    page = client.get("/dashboard").text
    assert "v0.1.0" not in page
    assert "{{ version }}" not in page


def test_dashboard_has_a_single_refresh_control(client: TestClient) -> None:
    """Two refresh buttons were redundant beside auto-refresh."""
    page = client.get("/dashboard").text
    assert 'id="btn-refresh"' in page
    assert "btn-refresh-state" not in page
    assert "btn-refresh-events" not in page


def test_dashboard_has_no_developer_curl_panel(client: TestClient) -> None:
    """Driving flows is the simulator's job, not the demo console's."""
    page = client.get("/dashboard").text
    assert "Drive a flow" not in page
    assert 'id="curl"' not in page
    assert "curl -s -X POST" not in page


def test_dashboard_keeps_only_the_useful_controls(client: TestClient) -> None:
    page = client.get("/dashboard").text
    for kept in ("btn-create", "btn-refresh", "btn-reset", "auto-refresh"):
        assert f'id="{kept}"' in page, f"{kept} should have been kept"
    for section in ("Choose a customer", "Session", "Event timeline"):
        assert section in page


def test_unused_control_plane_endpoints_are_gone(client: TestClient) -> None:
    """Nothing consumed /api/state or the session listing, so both went."""
    assert client.get("/api/state").status_code == 404
    # /api/sessions survives for POST, so a GET is method-not-allowed.
    assert client.get("/api/sessions").status_code == 405

    paths = client.get("/openapi.json").json()["paths"]
    assert "/api/state" not in paths
    assert "/api/sessions" in paths  # POST only
    assert set(paths["/api/sessions"]) == {"post"}
