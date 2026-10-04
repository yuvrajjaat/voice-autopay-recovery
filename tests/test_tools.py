"""Tests for the seven agent tool endpoints.

Everything here runs offline. The last two tests enforce that: one refuses
every non-loopback connection and DNS lookup while exercising all seven
endpoints, the other greps the Phase 2 modules for provider SDK imports. So a
future accidental call to ElevenLabs, Twilio, or a payment provider surfaces
as a test failure rather than a surprise in production.
"""

from __future__ import annotations

import re
import socket
from datetime import date, timedelta
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app import store
from app.config import settings
from app.main import app
from app.models import AutopayStatus, Disposition
from app.routers.tools import MAX_RETRIES_PER_SESSION, MAX_SCHEDULE_DAYS_AHEAD

SECRET = "test-tool-secret-value"
AUTH = {"X-Tool-Secret": SECRET}


@pytest.fixture(autouse=True)
def configured_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give the tool layer a secret for the duration of each test."""
    monkeypatch.setattr(settings, "tool_shared_secret", SECRET)


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def open_session(customer_id: str, session_id: str | None = None) -> str:
    """Create a session and return its id."""
    session = store.create_session(customer_id, session_id or f"sess_{customer_id}")
    return session.session_id


def verified_session(customer_id: str) -> str:
    """A session that has already passed verification.

    Verification is session-scoped, so this marks the session rather than the
    customer row.
    """
    session_id = open_session(customer_id)
    store.update_session(session_id, identity_verified=True, verification_attempts=1)
    return session_id


def call(
    client: TestClient,
    tool: str,
    payload: dict[str, Any],
    headers: dict[str, str] | None = None,
) -> Any:
    return client.post(
        f"/tools/{tool}",
        json=payload,
        headers=AUTH if headers is None else headers,
    )


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------


def test_valid_secret_is_accepted(client: TestClient) -> None:
    session_id = open_session("CUST-001")
    response = call(client, "get_failed_payment_details", {"session_id": session_id})
    assert response.status_code == 200


def test_missing_secret_is_rejected(client: TestClient) -> None:
    session_id = open_session("CUST-001")
    response = call(
        client, "get_failed_payment_details", {"session_id": session_id}, headers={}
    )
    assert response.status_code == 401
    body = response.json()
    assert body["success"] is False
    assert body["error"] == "invalid_tool_secret"


def test_wrong_secret_is_rejected(client: TestClient) -> None:
    session_id = open_session("CUST-001")
    response = call(
        client,
        "get_failed_payment_details",
        {"session_id": session_id},
        headers={"X-Tool-Secret": "not-the-secret"},
    )
    assert response.status_code == 401
    assert response.json()["error"] == "invalid_tool_secret"


def test_unconfigured_secret_fails_closed(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With TOOL_SHARED_SECRET unset the tools refuse everyone, not everyone."""
    monkeypatch.setattr(settings, "tool_shared_secret", None)
    session_id = open_session("CUST-001")
    response = call(client, "get_failed_payment_details", {"session_id": session_id})
    assert response.status_code == 503
    assert response.json()["error"] == "tool_auth_not_configured"


def test_error_responses_never_echo_the_secret(client: TestClient) -> None:
    response = call(
        client,
        "get_failed_payment_details",
        {"session_id": "whatever"},
        headers={"X-Tool-Secret": "wrong"},
    )
    assert SECRET not in response.text


@pytest.mark.parametrize(
    "tool",
    [
        "get_failed_payment_details",
        "verify_identity",
        "retry_payment",
        "schedule_retry",
        "send_payment_link",
        "escalate_to_human",
        "log_disposition",
    ],
)
def test_every_tool_requires_the_secret(client: TestClient, tool: str) -> None:
    response = client.post(f"/tools/{tool}", json={"session_id": "x"}, headers={})
    assert response.status_code == 401


# ---------------------------------------------------------------------------
# Session handling
# ---------------------------------------------------------------------------


def test_unknown_session_is_rejected(client: TestClient) -> None:
    response = call(client, "get_failed_payment_details", {"session_id": "sess_nope"})
    assert response.status_code == 404
    assert response.json()["error"] == "unknown_session"


def test_missing_session_id_is_a_validation_error(client: TestClient) -> None:
    response = call(client, "get_failed_payment_details", {})
    assert response.status_code == 422
    body = response.json()
    assert body["error"] == "invalid_request"
    assert "session_id" in body["message"]


def test_blank_session_id_is_rejected(client: TestClient) -> None:
    assert call(client, "get_failed_payment_details", {"session_id": ""}).status_code == 422


def test_a_tool_request_cannot_name_a_customer(client: TestClient) -> None:
    """The key anti-switching property: there is no customer field to set.

    A request carrying customer_id is a validation error, so a confused or
    manipulated model cannot redirect a session at another account.
    """
    session_id = open_session("CUST-001")
    response = call(
        client,
        "get_failed_payment_details",
        {"session_id": session_id, "customer_id": "CUST-008"},
    )
    assert response.status_code == 422
    assert response.json()["error"] == "invalid_request"


def test_sessions_stay_bound_to_their_own_customer(client: TestClient) -> None:
    first = open_session("CUST-001", "sess_a")
    second = open_session("CUST-005", "sess_b")

    body_one = call(client, "get_failed_payment_details", {"session_id": first}).json()
    body_two = call(client, "get_failed_payment_details", {"session_id": second}).json()

    assert body_one["customer_id"] == "CUST-001"
    assert body_two["customer_id"] == "CUST-005"


def test_no_api_exists_to_rebind_a_session() -> None:
    """Rebinding is not an operation this codebase offers."""
    session = store.create_session("CUST-001", "sess_bind")
    assert session.customer_id == "CUST-001"
    assert not any(
        name for name in dir(store) if "rebind" in name or "reassign" in name
    )


def test_duplicate_session_id_is_refused() -> None:
    open_session("CUST-001", "sess_dupe")
    with pytest.raises(ValueError, match="already exists"):
        store.create_session("CUST-002", "sess_dupe")


def test_session_for_unknown_customer_is_refused() -> None:
    with pytest.raises(store.CustomerNotFoundError):
        store.create_session("CUST-999")


def test_tool_calls_are_counted_on_the_session(client: TestClient) -> None:
    session_id = open_session("CUST-001")
    call(client, "get_failed_payment_details", {"session_id": session_id})
    call(client, "get_failed_payment_details", {"session_id": session_id})
    assert store.get_session(session_id).tool_calls == 2


def test_a_closed_session_rejects_further_tools(client: TestClient) -> None:
    session_id = verified_session("CUST-010")
    closing = call(
        client,
        "log_disposition",
        {"session_id": session_id, "disposition": "do_not_call"},
    )
    assert closing.status_code == 200

    after = call(client, "get_failed_payment_details", {"session_id": session_id})
    assert after.status_code == 409
    assert after.json()["error"] == "session_closed"


# ---------------------------------------------------------------------------
# get_failed_payment_details
# ---------------------------------------------------------------------------


def test_details_are_withheld_before_verification(client: TestClient) -> None:
    """Figures are gated in the backend, not only in the prompt."""
    session_id = open_session("CUST-001")
    body = call(client, "get_failed_payment_details", {"session_id": session_id}).json()

    assert body["success"] is True
    assert body["verified"] is False
    assert body["amount"] is None
    assert body["failure_reason"] is None
    assert body["card_description"] is None
    assert body["status"] == "withheld_pending_verification"
    assert body["next_action"] == "verify_identity"


def test_details_are_returned_once_verified(client: TestClient) -> None:
    session_id = verified_session("CUST-001")
    body = call(client, "get_failed_payment_details", {"session_id": session_id}).json()

    assert body["verified"] is True
    assert body["customer_id"] == "CUST-001"
    assert body["customer_name"] == "Maya Thompson"
    assert body["amount"] == "49.00"
    assert body["currency"] == "USD"
    assert body["status"] == "failed"
    assert body["failure_reason"] == "insufficient_funds"
    assert body["due_date"] == "2026-09-28"
    assert body["retry_worth_attempting"] is True
    assert body["backup_method_available"] is False
    assert body["next_action"] == "retry_payment"


def test_details_point_at_a_payment_link_for_a_terminal_decline(
    client: TestClient,
) -> None:
    session_id = verified_session("CUST-002")  # expired card
    body = call(client, "get_failed_payment_details", {"session_id": session_id}).json()
    assert body["retry_worth_attempting"] is False
    assert body["next_action"] == "payment_link"


def test_details_point_at_escalation_for_a_closed_account(client: TestClient) -> None:
    session_id = verified_session("CUST-006")
    body = call(client, "get_failed_payment_details", {"session_id": session_id}).json()
    assert body["next_action"] == "escalate"


def test_details_expose_no_internal_or_payment_identifiers(client: TestClient) -> None:
    """No method ids, payment ids, or structured card fields reach the agent.

    `card_description` deliberately carries the spoken phrase ("the visa
    ending 4417") because the agent has to say it; the underlying brand,
    expiry and method id stay server-side.
    """
    session_id = verified_session("CUST-003")
    response = call(client, "get_failed_payment_details", {"session_id": session_id})
    body = response.json()

    for forbidden in (
        "payment_method_id",
        "payment_id",
        "last4",
        "brand",
        "exp_month",
        "exp_year",
        "expected_answer",
        "scenario",
        "mock_retry_outcome",
    ):
        assert forbidden not in body, f"{forbidden} must not be exposed"

    assert "pm_mock_" not in response.text
    assert "pay_mock_" not in response.text
    assert not re.search(r"\d{12,}", response.text)


def test_suspension_date_is_surfaced_only_when_set(client: TestClient) -> None:
    quiet = call(
        client, "get_failed_payment_details", {"session_id": verified_session("CUST-001")}
    ).json()
    assert quiet["service_suspension_date"] is None

    delinquent = call(
        client, "get_failed_payment_details", {"session_id": verified_session("CUST-006")}
    ).json()
    assert delinquent["service_suspension_date"] == "2026-10-20"


# ---------------------------------------------------------------------------
# verify_identity
# ---------------------------------------------------------------------------


def test_successful_verification_updates_runtime_state(client: TestClient) -> None:
    session_id = open_session("CUST-001")
    body = call(
        client, "verify_identity", {"session_id": session_id, "postal_code": "94107"}
    ).json()

    assert body["success"] is True
    assert body["verified"] is True
    assert body["locked"] is False
    assert store.get_session(session_id).identity_verified is True
    assert store.get_session(session_id).verification_attempts == 1


def test_verification_tolerates_spacing_and_case(client: TestClient) -> None:
    session_id = open_session("CUST-001")
    body = call(
        client, "verify_identity", {"session_id": session_id, "postal_code": " 9 4 1 0 7 "}
    ).json()
    assert body["verified"] is True


def test_wrong_answer_leaves_one_attempt(client: TestClient) -> None:
    session_id = open_session("CUST-007")
    body = call(
        client, "verify_identity", {"session_id": session_id, "postal_code": "19147"}
    ).json()

    assert body["success"] is True
    assert body["verified"] is False
    assert body["attempts_remaining"] == 1
    assert body["locked"] is False
    assert store.get_session(session_id).identity_verified is False


def test_verification_locks_after_the_attempt_limit(client: TestClient) -> None:
    session_id = open_session("CUST-007")
    for _ in range(2):
        call(client, "verify_identity", {"session_id": session_id, "postal_code": "19147"})

    body = call(
        client, "verify_identity", {"session_id": session_id, "postal_code": "19104"}
    ).json()
    assert body["verified"] is False
    assert body["locked"] is True
    assert body["attempts_remaining"] == 0
    # The correct answer arriving after lockout must not unlock the account.
    assert store.get_session(session_id).identity_verified is False


def test_locked_verification_does_not_keep_counting_attempts(
    client: TestClient,
) -> None:
    session_id = open_session("CUST-007")
    for _ in range(4):
        call(client, "verify_identity", {"session_id": session_id, "postal_code": "00000"})
    assert store.get_session(session_id).verification_attempts == 2


def test_verification_is_idempotent_once_passed(client: TestClient) -> None:
    session_id = verified_session("CUST-001")
    body = call(
        client, "verify_identity", {"session_id": session_id, "postal_code": "00000"}
    ).json()
    assert body["verified"] is True
    assert store.get_session(session_id).verification_attempts == 1


# ---------------------------------------------------------------------------
# retry_payment
# ---------------------------------------------------------------------------


def test_retry_is_refused_before_verification(client: TestClient) -> None:
    session_id = open_session("CUST-001")
    response = call(client, "retry_payment", {"session_id": session_id})
    assert response.status_code == 403
    assert response.json()["error"] == "identity_not_verified"
    assert store.get_payment_attempts("CUST-001") == []


def test_successful_retry_returns_a_mock_transaction(client: TestClient) -> None:
    session_id = verified_session("CUST-001")
    body = call(client, "retry_payment", {"session_id": session_id}).json()

    assert body["success"] is True
    assert body["status"] == "paid"
    assert body["transaction_id"].startswith("mock_txn_")
    assert re.fullmatch(r"C-\d{4}", body["confirmation_number"])
    assert body["amount"] == "49.00"
    assert body["next_action"] == "none"

    assert store.get_state("CUST-001").status is AutopayStatus.RECOVERED
    assert len(store.get_payment_attempts("CUST-001")) == 1


def test_retry_attempt_is_linked_to_its_session(client: TestClient) -> None:
    session_id = verified_session("CUST-001")
    call(client, "retry_payment", {"session_id": session_id})
    assert store.get_payment_attempts("CUST-001")[0]["session_id"] == session_id


def test_failed_retry_suggests_a_link_or_escalation(client: TestClient) -> None:
    session_id = verified_session("CUST-009")  # insufficient funds, fails again
    body = call(client, "retry_payment", {"session_id": session_id}).json()

    assert body["success"] is False
    assert body["status"] == "declined"
    assert body["confirmation_number"] is None
    assert body["failure_reason"] == "insufficient_funds"
    assert body["next_action"] == "payment_link_or_escalation"
    assert store.get_state("CUST-009").status is AutopayStatus.FAILED


def test_terminal_decline_is_not_retried_at_all(client: TestClient) -> None:
    """An expired card cannot approve, so no attempt is spent proving it."""
    session_id = verified_session("CUST-002")
    body = call(client, "retry_payment", {"session_id": session_id}).json()

    assert body["success"] is False
    assert body["status"] == "not_attempted"
    assert body["next_action"] == "payment_link"
    assert store.get_payment_attempts("CUST-002") == []


def test_closed_account_routes_to_escalation(client: TestClient) -> None:
    session_id = verified_session("CUST-006")
    body = call(client, "retry_payment", {"session_id": session_id}).json()
    assert body["status"] == "not_attempted"
    assert body["next_action"] == "escalate"


def test_declined_primary_offers_the_backup_card(client: TestClient) -> None:
    session_id = verified_session("CUST-003")
    first = call(client, "retry_payment", {"session_id": session_id}).json()
    assert first["success"] is False
    assert first["next_action"] == "offer_backup_method"

    second = call(
        client, "retry_payment", {"session_id": session_id, "payment_method": "backup"}
    ).json()
    assert second["success"] is True
    assert second["status"] == "paid"
    assert store.get_state("CUST-003").status is AutopayStatus.RECOVERED


def test_requesting_an_absent_backup_card_is_refused(client: TestClient) -> None:
    session_id = verified_session("CUST-001")
    response = call(
        client, "retry_payment", {"session_id": session_id, "payment_method": "backup"}
    )
    assert response.status_code == 400
    assert response.json()["error"] == "payment_method_unavailable"


def test_retry_on_a_settled_balance_is_refused(client: TestClient) -> None:
    session_id = verified_session("CUST-001")
    assert call(client, "retry_payment", {"session_id": session_id}).json()["success"]

    response = call(client, "retry_payment", {"session_id": session_id})
    assert response.status_code == 409
    assert response.json()["error"] == "already_paid"
    assert len(store.get_payment_attempts("CUST-001")) == 1


def test_retries_are_capped_within_one_conversation(client: TestClient) -> None:
    session_id = verified_session("CUST-009")  # always declines
    for _ in range(MAX_RETRIES_PER_SESSION):
        assert call(client, "retry_payment", {"session_id": session_id}).status_code == 200

    response = call(client, "retry_payment", {"session_id": session_id})
    assert response.status_code == 409
    assert response.json()["error"] == "retry_limit_reached"
    assert len(store.get_payment_attempts("CUST-009")) == MAX_RETRIES_PER_SESSION


def test_invalid_payment_method_is_rejected(client: TestClient) -> None:
    session_id = verified_session("CUST-001")
    response = call(
        client, "retry_payment", {"session_id": session_id, "payment_method": "bitcoin"}
    )
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# schedule_retry
# ---------------------------------------------------------------------------


def test_scheduling_creates_a_record(client: TestClient) -> None:
    session_id = verified_session("CUST-004")
    when = date.today() + timedelta(days=11)
    body = call(
        client,
        "schedule_retry",
        {"session_id": session_id, "requested_date": when.isoformat()},
    ).json()

    assert body["success"] is True
    assert body["status"] == "scheduled"
    assert body["scheduled_for"] == when.isoformat()
    assert re.fullmatch(r"C-\d{4}", body["confirmation_number"])

    records = store.get_scheduled_retries("CUST-004")
    assert len(records) == 1
    assert records[0]["session_id"] == session_id
    assert records[0]["requested_action"] == "retry_primary"
    assert "No background job" in records[0]["note"]
    assert store.get_state("CUST-004").status is AutopayStatus.SCHEDULED


def test_scheduling_a_past_date_is_rejected(client: TestClient) -> None:
    session_id = verified_session("CUST-004")
    response = call(
        client,
        "schedule_retry",
        {
            "session_id": session_id,
            "requested_date": (date.today() - timedelta(days=1)).isoformat(),
        },
    )
    assert response.status_code == 400
    assert response.json()["error"] == "invalid_schedule"
    assert store.get_scheduled_retries("CUST-004") == []


def test_scheduling_today_is_rejected(client: TestClient) -> None:
    session_id = verified_session("CUST-004")
    response = call(
        client,
        "schedule_retry",
        {"session_id": session_id, "requested_date": date.today().isoformat()},
    )
    assert response.status_code == 400


def test_scheduling_too_far_ahead_is_rejected(client: TestClient) -> None:
    session_id = verified_session("CUST-004")
    too_far = date.today() + timedelta(days=MAX_SCHEDULE_DAYS_AHEAD + 1)
    response = call(
        client,
        "schedule_retry",
        {"session_id": session_id, "requested_date": too_far.isoformat()},
    )
    assert response.status_code == 400
    assert response.json()["error"] == "invalid_schedule"


def test_scheduling_requires_verification(client: TestClient) -> None:
    session_id = open_session("CUST-004")
    response = call(
        client,
        "schedule_retry",
        {
            "session_id": session_id,
            "requested_date": (date.today() + timedelta(days=5)).isoformat(),
        },
    )
    assert response.status_code == 403


def test_malformed_date_is_a_validation_error(client: TestClient) -> None:
    session_id = verified_session("CUST-004")
    response = call(
        client, "schedule_retry", {"session_id": session_id, "requested_date": "payday"}
    )
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# send_payment_link
# ---------------------------------------------------------------------------


def test_payment_link_is_a_mock_url_and_never_delivered(client: TestClient) -> None:
    session_id = verified_session("CUST-002")
    body = call(
        client, "send_payment_link", {"session_id": session_id, "channel": "email"}
    ).json()

    assert body["success"] is True
    assert body["delivered"] is False
    assert body["link"].startswith("https://example.test/pay/link_mock_")
    assert "Mock only" in body["delivery_note"]
    assert body["sent_to_masked"] == "d***@example.com"

    records = store.get_payment_links("CUST-002")
    assert len(records) == 1
    assert records[0]["delivered"] is False
    assert records[0]["session_id"] == session_id
    assert store.get_state("CUST-002").status is AutopayStatus.LINK_SENT


def test_payment_link_masks_a_phone_for_sms(client: TestClient) -> None:
    session_id = verified_session("CUST-002")
    body = call(
        client, "send_payment_link", {"session_id": session_id, "channel": "sms"}
    ).json()
    assert body["sent_to_masked"] == "+*******0101"
    assert "0101" in body["sent_to_masked"]


def test_payment_link_is_deterministic(client: TestClient) -> None:
    first = call(
        client,
        "send_payment_link",
        {"session_id": verified_session("CUST-002"), "channel": "email"},
    ).json()["link"]
    store.reset_runtime()
    second = call(
        client,
        "send_payment_link",
        {"session_id": verified_session("CUST-002"), "channel": "email"},
    ).json()["link"]
    assert first == second


def test_payment_link_requires_verification(client: TestClient) -> None:
    session_id = open_session("CUST-002")
    response = call(client, "send_payment_link", {"session_id": session_id})
    assert response.status_code == 403


def test_payment_link_rejects_an_unknown_channel(client: TestClient) -> None:
    session_id = verified_session("CUST-002")
    response = call(
        client, "send_payment_link", {"session_id": session_id, "channel": "whatsapp"}
    )
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# escalate_to_human
# ---------------------------------------------------------------------------


def test_escalation_creates_a_ticket(client: TestClient) -> None:
    session_id = verified_session("CUST-008")
    body = call(
        client,
        "escalate_to_human",
        {"session_id": session_id, "reason": "customer disputes the amount"},
    ).json()

    assert body["success"] is True
    assert body["status"] == "open"
    assert re.fullmatch(r"TCK-[0-9A-F]{6}", body["ticket_id"])
    assert body["callback_window"]

    records = store.get_escalations("CUST-008")
    assert len(records) == 1
    assert records[0]["reason"] == "customer disputes the amount"
    assert "No human is contacted" in records[0]["note"]
    assert store.get_state("CUST-008").status is AutopayStatus.ESCALATED


def test_escalation_works_without_verification(client: TestClient) -> None:
    """Someone who cannot verify is exactly who needs a person."""
    session_id = open_session("CUST-007")
    response = call(
        client,
        "escalate_to_human",
        {"session_id": session_id, "reason": "could not verify identity"},
    )
    assert response.status_code == 200
    assert store.get_escalations("CUST-007")


def test_escalation_requires_a_reason(client: TestClient) -> None:
    session_id = verified_session("CUST-008")
    assert call(client, "escalate_to_human", {"session_id": session_id}).status_code == 422
    assert (
        call(
            client, "escalate_to_human", {"session_id": session_id, "reason": ""}
        ).status_code
        == 422
    )


def test_escalation_stores_optional_notes(client: TestClient) -> None:
    session_id = verified_session("CUST-008")
    call(
        client,
        "escalate_to_human",
        {"session_id": session_id, "reason": "dispute", "notes": "two lines cancelled"},
    )
    assert store.get_escalations("CUST-008")[0]["notes"] == "two lines cancelled"


# ---------------------------------------------------------------------------
# log_disposition
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "disposition",
    [
        "payment_link_prepared",
        "retry_scheduled",
        "escalated",
        "customer_declined",
        "do_not_call",
        "verification_failed",
        "unresolved",
    ],
)
def test_every_agreed_disposition_is_accepted(
    client: TestClient, disposition: str
) -> None:
    session_id = open_session("CUST-001")
    response = call(
        client, "log_disposition", {"session_id": session_id, "disposition": disposition}
    )
    assert response.status_code == 200
    assert response.json()["disposition"] == disposition


def test_payment_recovered_requires_an_actual_payment(client: TestClient) -> None:
    """A recovery claim must be backed by an approved charge."""
    session_id = verified_session("CUST-009")  # retry always declines
    call(client, "retry_payment", {"session_id": session_id})

    response = call(
        client,
        "log_disposition",
        {"session_id": session_id, "disposition": "payment_recovered"},
    )
    assert response.status_code == 409
    assert response.json()["error"] == "disposition_conflicts_with_state"
    assert store.get_dispositions("CUST-009") == []

    # The honest label is accepted.
    assert call(
        client, "log_disposition", {"session_id": session_id, "disposition": "unresolved"}
    ).status_code == 200


def test_payment_recovered_is_accepted_after_a_successful_retry(
    client: TestClient,
) -> None:
    session_id = verified_session("CUST-001")
    assert call(client, "retry_payment", {"session_id": session_id}).json()["success"]
    response = call(
        client,
        "log_disposition",
        {"session_id": session_id, "disposition": "payment_recovered"},
    )
    assert response.status_code == 200


def test_invalid_disposition_is_rejected(client: TestClient) -> None:
    session_id = open_session("CUST-001")
    response = call(
        client,
        "log_disposition",
        {"session_id": session_id, "disposition": "customer_was_rude"},
    )
    assert response.status_code == 422
    assert response.json()["error"] == "invalid_request"
    assert store.get_dispositions("CUST-001") == []


def test_disposition_closes_the_session_and_records_notes(client: TestClient) -> None:
    session_id = verified_session("CUST-001")
    call(client, "retry_payment", {"session_id": session_id})  # earns the label
    body = call(
        client,
        "log_disposition",
        {
            "session_id": session_id,
            "disposition": "payment_recovered",
            "notes": "paid on the first retry",
        },
    ).json()

    assert body["session_closed"] is True
    assert body["do_not_call"] is False
    assert store.get_session(session_id).closed is True

    state = store.get_state("CUST-001")
    assert state.disposition is Disposition.PAYMENT_RECOVERED
    assert state.disposition_notes == "paid on the first retry"


def test_do_not_call_is_persisted_against_the_customer(client: TestClient) -> None:
    """The opt-out must outlive the call it was made on."""
    session_id = open_session("CUST-010")
    body = call(
        client,
        "log_disposition",
        {"session_id": session_id, "disposition": "do_not_call"},
    ).json()

    assert body["do_not_call"] is True
    state = store.get_state("CUST-010")
    assert state.do_not_call is True
    assert state.status is AutopayStatus.DO_NOT_CALL
    assert "won't call you again" in body["message"]


def test_do_not_call_survives_a_fresh_read(client: TestClient) -> None:
    session_id = open_session("CUST-010")
    call(
        client, "log_disposition", {"session_id": session_id, "disposition": "do_not_call"}
    )
    store._customers_cache = None
    assert store.get_state("CUST-010").do_not_call is True


# ---------------------------------------------------------------------------
# End-to-end scenarios across all ten fictional customers
# ---------------------------------------------------------------------------


def test_happy_path_end_to_end(client: TestClient) -> None:
    """Verify, explain, retry, recover, close — the headline demo."""
    session_id = open_session("CUST-001")

    withheld = call(
        client, "get_failed_payment_details", {"session_id": session_id}
    ).json()
    assert withheld["amount"] is None

    assert call(
        client, "verify_identity", {"session_id": session_id, "postal_code": "94107"}
    ).json()["verified"]

    details = call(client, "get_failed_payment_details", {"session_id": session_id}).json()
    assert details["amount"] == "49.00"

    retry = call(client, "retry_payment", {"session_id": session_id}).json()
    assert retry["status"] == "paid"

    closed = call(
        client,
        "log_disposition",
        {"session_id": session_id, "disposition": "payment_recovered"},
    ).json()
    assert closed["session_closed"] is True
    assert store.get_state("CUST-001").status is AutopayStatus.RECOVERED


def test_expired_card_path_end_to_end(client: TestClient) -> None:
    session_id = open_session("CUST-002")
    call(client, "verify_identity", {"session_id": session_id, "postal_code": "78702"})

    retry = call(client, "retry_payment", {"session_id": session_id}).json()
    assert retry["next_action"] == "payment_link"

    link = call(
        client, "send_payment_link", {"session_id": session_id, "channel": "email"}
    ).json()
    assert link["delivered"] is False

    call(
        client,
        "log_disposition",
        {"session_id": session_id, "disposition": "payment_link_prepared"},
    )
    assert store.get_state("CUST-002").disposition is Disposition.PAYMENT_LINK_PREPARED


def test_verification_failure_path_discloses_nothing(client: TestClient) -> None:
    session_id = open_session("CUST-007")
    for _ in range(2):
        call(
            client,
            "verify_identity",
            {"session_id": session_id, "postal_code": "19147"},
        )

    details = call(client, "get_failed_payment_details", {"session_id": session_id}).json()
    assert details["amount"] is None
    assert details["failure_reason"] is None

    assert call(client, "retry_payment", {"session_id": session_id}).status_code == 403

    call(
        client,
        "log_disposition",
        {"session_id": session_id, "disposition": "verification_failed"},
    )
    assert store.get_state("CUST-007").disposition is Disposition.VERIFICATION_FAILED


@pytest.mark.parametrize(
    ("customer_id", "postal_code", "disposition"),
    [
        ("CUST-001", "94107", "customer_declined"),
        ("CUST-002", "78702", "payment_link_prepared"),
        ("CUST-003", "02139", "escalated"),
        ("CUST-004", "60614", "retry_scheduled"),
        ("CUST-005", "30309", "customer_declined"),
        ("CUST-006", "55401", "escalated"),
        ("CUST-007", "19104", "verification_failed"),
        ("CUST-008", "98104", "escalated"),
        ("CUST-009", "85004", "unresolved"),
        ("CUST-010", "73102", "do_not_call"),
    ],
)
def test_each_customer_reaches_its_declared_outcome(
    client: TestClient, customer_id: str, postal_code: str, disposition: str
) -> None:
    """Every one of the ten records can be driven to a recorded outcome."""
    session_id = open_session(customer_id)
    call(
        client, "verify_identity", {"session_id": session_id, "postal_code": postal_code}
    )
    response = call(
        client, "log_disposition", {"session_id": session_id, "disposition": disposition}
    )
    assert response.status_code == 200
    assert store.get_dispositions(customer_id)[0]["disposition"] == disposition


# ---------------------------------------------------------------------------
# Offline guarantee
# ---------------------------------------------------------------------------


def test_the_whole_tool_layer_runs_with_no_network(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Block every off-box connection, then exercise all seven tools.

    Loopback has to stay open: asyncio's Windows event loop builds its own
    self-pipe with ``socket.socketpair()``, which connects over 127.0.0.1, and
    TestClient spins up a fresh loop per request. Banning sockets outright
    would therefore fail the harness rather than test the code. Refusing any
    non-loopback address is the assertion that actually matters — a real
    ElevenLabs, Twilio, or payment-provider call would be caught here.
    """
    real_connect = socket.socket.connect
    real_getaddrinfo = socket.getaddrinfo
    loopback = {"127.0.0.1", "::1", "localhost", "testserver"}

    def host_of(address: object) -> str:
        if isinstance(address, tuple) and address:
            return str(address[0])
        return str(address)

    def guarded_connect(self: socket.socket, address: object) -> object:
        host = host_of(address)
        if host not in loopback:
            raise AssertionError(f"a tool attempted an external connection to {host}")
        return real_connect(self, address)

    def guarded_getaddrinfo(host: object, *args: object, **kwargs: object) -> object:
        if host is not None and str(host) not in loopback:
            raise AssertionError(f"a tool attempted to resolve {host}")
        return real_getaddrinfo(host, *args, **kwargs)  # type: ignore[arg-type]

    def refuse_create_connection(*args: object, **kwargs: object) -> None:
        raise AssertionError("a tool attempted socket.create_connection")

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket, "getaddrinfo", guarded_getaddrinfo)
    monkeypatch.setattr(socket, "create_connection", refuse_create_connection)

    session_id = open_session("CUST-003")
    when = (date.today() + timedelta(days=7)).isoformat()

    assert call(client, "get_failed_payment_details", {"session_id": session_id}).status_code == 200
    assert call(
        client, "verify_identity", {"session_id": session_id, "postal_code": "02139"}
    ).status_code == 200
    assert call(client, "retry_payment", {"session_id": session_id}).status_code == 200
    assert call(
        client, "schedule_retry", {"session_id": session_id, "requested_date": when}
    ).status_code == 200
    assert call(
        client, "send_payment_link", {"session_id": session_id, "channel": "sms"}
    ).status_code == 200
    assert call(
        client, "escalate_to_human", {"session_id": session_id, "reason": "offline check"}
    ).status_code == 200
    assert call(
        client, "log_disposition", {"session_id": session_id, "disposition": "unresolved"}
    ).status_code == 200


def test_tool_modules_import_no_network_or_provider_sdk() -> None:
    """No ElevenLabs, Twilio, or HTTP client in the Phase 2 code paths."""
    from pathlib import Path

    banned = ("elevenlabs", "twilio", "import requests", "import httpx", "stripe")
    for module in ("app/routers/tools.py", "app/store.py", "app/security.py", "app/errors.py"):
        text = Path(module).read_text(encoding="utf-8")
        for needle in banned:
            assert needle not in text, f"{module} must not reference {needle}"


# ---------------------------------------------------------------------------
# Verification regressions
#
# A live voice call stated the correct postal code for CUST-001 and still
# ended as verification_failed, and a second session for the same customer
# failed before it compared anything. Root cause: identity_verified and
# verification_attempts lived on the customer row, so they persisted between
# conversations. These tests pin the corrected behaviour.
# ---------------------------------------------------------------------------


def test_cust_001_verifies_with_94107_on_the_first_attempt(client: TestClient) -> None:
    """The exact case that failed live."""
    session_id = open_session("CUST-001")
    body = call(
        client, "verify_identity", {"session_id": session_id, "postal_code": "94107"}
    ).json()

    assert body["verified"] is True
    assert body["locked"] is False
    assert body["attempts_remaining"] == 1
    assert store.get_session(session_id).identity_verified is True
    assert store.get_session(session_id).verification_attempts == 1


@pytest.mark.parametrize(
    "stated",
    [
        "94107",
        "9 4 1 0 7",
        " 94107 ",
        "94107.",
        "94107,",
        "9-4-1-0-7",
        "nine four one oh seven",
        "Nine Four One Zero Seven",
        "9 4 1 o 7",
    ],
)
def test_the_spoken_forms_a_voice_agent_sends_all_verify(
    client: TestClient, stated: str
) -> None:
    """Representation varies; the answer does not."""
    session_id = open_session("CUST-001")
    body = call(
        client, "verify_identity", {"session_id": session_id, "postal_code": stated}
    ).json()
    assert body["verified"] is True, f"{stated!r} should verify"


@pytest.mark.parametrize(
    "stated", ["19147", "00000", "9410", "941077", "ninety four one"]
)
def test_a_wrong_postal_code_still_fails(client: TestClient, stated: str) -> None:
    """Normalisation must not become a free pass."""
    session_id = open_session("CUST-001")
    body = call(
        client, "verify_identity", {"session_id": session_id, "postal_code": stated}
    ).json()
    assert body["verified"] is False, f"{stated!r} must not verify"
    assert store.get_session(session_id).identity_verified is False


def test_a_fresh_session_is_never_pre_locked_by_an_earlier_one(
    client: TestClient,
) -> None:
    """Bug 1: an earlier conversation must not lock the account forever."""
    first = open_session("CUST-001", "sess_first")
    for wrong in ("00000", "11111"):
        call(client, "verify_identity", {"session_id": first, "postal_code": wrong})
    assert store.get_session(first).verification_attempts == 2

    second = open_session("CUST-001", "sess_second")
    assert store.get_session(second).verification_attempts == 0

    body = call(
        client, "verify_identity", {"session_id": second, "postal_code": "94107"}
    ).json()
    assert body["verified"] is True, "a new conversation must get its own attempts"
    assert body["locked"] is False
    assert body["attempts_remaining"] == 1


def test_a_fresh_session_is_never_pre_verified_by_an_earlier_one(
    client: TestClient,
) -> None:
    """Bug 2, the security half: verification must not be inherited.

    Otherwise a second caller reaching the same account would be handed the
    balance without saying anything.
    """
    first = open_session("CUST-001", "sess_v1")
    assert call(
        client, "verify_identity", {"session_id": first, "postal_code": "94107"}
    ).json()["verified"]

    second = open_session("CUST-001", "sess_v2")
    assert store.get_session(second).identity_verified is False

    details = call(client, "get_failed_payment_details", {"session_id": second}).json()
    assert details["verified"] is False
    assert details["amount"] is None
    assert details["card_description"] is None

    assert call(client, "retry_payment", {"session_id": second}).status_code == 403


def test_a_verified_session_cannot_be_reverted_by_a_later_failure(
    client: TestClient,
) -> None:
    """A correct first attempt must survive whatever follows."""
    session_id = open_session("CUST-001")
    assert call(
        client, "verify_identity", {"session_id": session_id, "postal_code": "94107"}
    ).json()["verified"]

    # Further calls with a wrong code must not undo it, nor burn attempts.
    for wrong in ("00000", "11111", "99999"):
        body = call(
            client, "verify_identity", {"session_id": session_id, "postal_code": wrong}
        ).json()
        assert body["verified"] is True, "verification was revoked"
        assert body["locked"] is False

    session = store.get_session(session_id)
    assert session.identity_verified is True
    assert session.verification_attempts == 1, "idempotent calls must not count"

    # And the gated tools still work.
    assert call(client, "retry_payment", {"session_id": session_id}).json()["success"]


def test_another_sessions_failure_cannot_revoke_a_verified_session(
    client: TestClient,
) -> None:
    """Two conversations on one customer must not interfere."""
    verified = open_session("CUST-001", "sess_ok")
    assert call(
        client, "verify_identity", {"session_id": verified, "postal_code": "94107"}
    ).json()["verified"]

    other = open_session("CUST-001", "sess_bad")
    for wrong in ("00000", "11111"):
        call(client, "verify_identity", {"session_id": other, "postal_code": wrong})

    assert store.get_session(verified).identity_verified is True
    assert (
        call(client, "get_failed_payment_details", {"session_id": verified}).json()[
            "amount"
        ]
        == "49.00"
    )


def test_exactly_two_attempts_are_allowed_per_conversation(client: TestClient) -> None:
    """The cap is two comparisons; a third call compares nothing."""
    session_id = open_session("CUST-007")
    limit = store.get_customer("CUST-007").verification.max_attempts
    assert limit == 2

    first = call(
        client, "verify_identity", {"session_id": session_id, "postal_code": "00000"}
    ).json()
    assert first["attempts_remaining"] == 1
    assert first["locked"] is False

    second = call(
        client, "verify_identity", {"session_id": session_id, "postal_code": "11111"}
    ).json()
    assert second["attempts_remaining"] == 0
    assert second["locked"] is True

    # The third call is the locked branch: 200, but no comparison and no
    # further attempt recorded. This is why a live call logged three 200s.
    third = call(
        client, "verify_identity", {"session_id": session_id, "postal_code": "19104"}
    )
    assert third.status_code == 200
    assert third.json()["locked"] is True
    assert third.json()["verified"] is False
    assert (
        store.get_session(session_id).verification_attempts == 2
    ), "a locked call must not count as an attempt"
    assert (
        store.get_session(session_id).identity_verified is False
    ), "the correct code arriving after lockout must not unlock"


def test_the_received_postal_code_maps_into_the_request_model(
    client: TestClient,
) -> None:
    """The field name the provider sends must bind to the Pydantic field."""
    from app.models import VerifyIdentityRequest

    assert "postal_code" in VerifyIdentityRequest.model_fields
    session_id = open_session("CUST-001")

    # The exact body shape ElevenLabs posts.
    response = client.post(
        "/tools/verify_identity",
        json={"session_id": session_id, "postal_code": "94107"},
        headers=AUTH,
    )
    assert response.status_code == 200
    assert response.json()["verified"] is True

    # A misnamed field is a validation error, not a silent failure.
    bad = client.post(
        "/tools/verify_identity",
        json={"session_id": session_id, "postalCode": "94107"},
        headers=AUTH,
    )
    assert bad.status_code == 422


def test_a_long_spoken_answer_is_not_rejected_as_too_long(client: TestClient) -> None:
    """max_length used to be 16, which 422'd any spoken digit sequence."""
    session_id = open_session("CUST-001")
    response = call(
        client,
        "verify_identity",
        {"session_id": session_id, "postal_code": "nine four one oh seven"},
    )
    assert response.status_code == 200
    assert response.json()["verified"] is True


def test_verification_attempts_are_counted_after_comparison_not_before(
    client: TestClient,
) -> None:
    """A successful first attempt leaves one attempt unspent, not zero."""
    session_id = open_session("CUST-001")
    body = call(
        client, "verify_identity", {"session_id": session_id, "postal_code": "94107"}
    ).json()
    assert body["attempts_remaining"] == 1
    assert store.get_session(session_id).verification_attempts == 1


def test_the_diagnostic_log_never_records_the_expected_answer(
    client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    """The temporary logging must not leak the credential it checks."""
    session_id = open_session("CUST-001")
    with caplog.at_level("INFO", logger="app.routers.tools"):
        call(
            client, "verify_identity", {"session_id": session_id, "postal_code": "00000"}
        )

    assert "verify_identity diagnostic" in caplog.text
    assert "matched=False" in caplog.text
    assert "CUST-001" in caplog.text
    assert "94107" not in caplog.text, "the expected answer must never be logged"
