"""Tests for the deterministic mock payment processor."""

from __future__ import annotations

import re
import socket
from decimal import Decimal

import pytest

from app import store
from app.models import FailureCode, MockRetryOutcome, PaymentStatus
from app.payments import mock_processor
from app.payments.mock_processor import (
    PaymentMethodUnavailableError,
    check_payment_status,
    retry_payment,
)


# ---------------------------------------------------------------------------
# Successful and failed retries
# ---------------------------------------------------------------------------


def test_successful_retry_returns_structured_approval() -> None:
    customer = store.get_customer("CUST-001")
    result = retry_payment(customer)

    assert result.success is True
    assert result.status is PaymentStatus.PAID
    assert result.failure_code is None
    assert result.amount == Decimal("49.00")
    assert result.currency == "USD"
    assert result.payment_method == "primary"
    assert result.payment_method_id == "pm_mock_0001"
    assert result.processor == "mock"
    assert result.confirmation_number is not None
    assert "Approved" in result.message


def test_failed_retry_reports_a_decline_code_and_no_confirmation() -> None:
    customer = store.get_customer("CUST-002")  # expired card
    result = retry_payment(customer)

    assert result.success is False
    assert result.status is PaymentStatus.FAILED
    assert result.failure_code is FailureCode.EXPIRED_CARD
    assert result.confirmation_number is None
    assert result.message  # speakable explanation, not an error code
    assert "expired" in result.message.lower()


def test_backup_method_scenario_declines_primary_then_approves_backup() -> None:
    customer = store.get_customer("CUST-003")
    assert customer.scenario.mock_retry_outcome is MockRetryOutcome.SUCCEED_ON_BACKUP

    primary = retry_payment(customer, payment_method="primary")
    assert primary.success is False
    assert primary.failure_code is FailureCode.CARD_DECLINED

    backup = retry_payment(customer, payment_method="backup", attempt_number=2)
    assert backup.success is True
    assert backup.payment_method_id == "pm_mock_0003b"


def test_requesting_a_missing_backup_method_raises() -> None:
    customer = store.get_customer("CUST-001")  # no backup on file
    assert customer.autopay.backup_method is None
    with pytest.raises(PaymentMethodUnavailableError):
        retry_payment(customer, payment_method="backup")


@pytest.mark.parametrize(
    ("customer_id", "expected_success"),
    [
        ("CUST-001", True),
        ("CUST-002", False),
        ("CUST-003", False),  # succeed_on_backup declines the primary card
        ("CUST-004", True),
        ("CUST-005", True),
        ("CUST-006", False),
        ("CUST-007", True),
        ("CUST-008", False),
        ("CUST-009", False),
        ("CUST-010", True),
    ],
)
def test_every_customer_matches_its_declared_scenario(
    customer_id: str, expected_success: bool
) -> None:
    """The processor must follow the script written into the seed data."""
    result = retry_payment(store.get_customer(customer_id))
    assert result.success is expected_success


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_repeated_calls_return_identical_results() -> None:
    customer = store.get_customer("CUST-001")
    first = retry_payment(customer, attempt_number=1)
    second = retry_payment(customer, attempt_number=1)
    assert first.model_dump() == second.model_dump()


def test_identifiers_vary_by_attempt_and_method() -> None:
    customer = store.get_customer("CUST-003")
    attempt_one = retry_payment(customer, payment_method="primary", attempt_number=1)
    attempt_two = retry_payment(customer, payment_method="primary", attempt_number=2)
    backup = retry_payment(customer, payment_method="backup", attempt_number=1)

    ids = {attempt_one.transaction_id, attempt_two.transaction_id, backup.transaction_id}
    assert len(ids) == 3, "each (attempt, method) pair needs its own transaction id"


def test_transaction_ids_are_fake_and_well_formed() -> None:
    for customer in store.list_customers():
        result = retry_payment(customer)
        assert re.fullmatch(r"mock_txn_[0-9a-f]{10}", result.transaction_id), (
            f"{customer.customer_id}: {result.transaction_id}"
        )


def test_confirmation_numbers_are_short_and_speakable() -> None:
    for customer in store.list_customers():
        result = retry_payment(customer)
        if result.confirmation_number is not None:
            assert re.fullmatch(r"C-\d{4}", result.confirmation_number)


def test_identifier_builders_are_pure_functions() -> None:
    first = mock_processor.build_transaction_id("CUST-001", 1, "primary")
    second = mock_processor.build_transaction_id("CUST-001", 1, "primary")
    assert first == second
    assert first != mock_processor.build_transaction_id("CUST-002", 1, "primary")

    confirmation = mock_processor.build_confirmation_number("CUST-001", 1, "primary")
    assert confirmation == mock_processor.build_confirmation_number("CUST-001", 1, "primary")


def test_no_latency_is_simulated_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests must stay fast; the pause is opt-in for the tool layer."""
    calls: list[float] = []
    monkeypatch.setattr(mock_processor.time, "sleep", lambda seconds: calls.append(seconds))
    retry_payment(store.get_customer("CUST-001"))
    assert calls == []


def test_latency_is_applied_when_requested(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[float] = []
    monkeypatch.setattr(mock_processor.time, "sleep", lambda seconds: calls.append(seconds))
    retry_payment(store.get_customer("CUST-001"), latency_seconds=0.8)
    assert calls == [0.8]


# ---------------------------------------------------------------------------
# Status checks
# ---------------------------------------------------------------------------


def test_check_payment_status_describes_the_failure_without_charging() -> None:
    status = check_payment_status(store.get_customer("CUST-001"))

    assert status["customer_id"] == "CUST-001"
    assert status["status"] == "failed"
    assert status["amount"] == "49.00"
    assert status["failure_code"] == "insufficient_funds"
    assert status["backup_method_available"] is False
    assert status["retry_worth_attempting"] is True
    assert status["processor"] == "mock"
    # No transaction is created by a status check.
    assert "transaction_id" not in status


def test_terminal_declines_are_flagged_as_not_worth_retrying() -> None:
    """An expired or closed card cannot be fixed by trying again."""
    assert check_payment_status(store.get_customer("CUST-002"))["retry_worth_attempting"] is False
    assert check_payment_status(store.get_customer("CUST-006"))["retry_worth_attempting"] is False
    # Insufficient funds might clear, so a retry is reasonable.
    assert check_payment_status(store.get_customer("CUST-009"))["retry_worth_attempting"] is True


def test_backup_availability_is_reported() -> None:
    assert check_payment_status(store.get_customer("CUST-003"))["backup_method_available"] is True


# ---------------------------------------------------------------------------
# No external calls, no real credentials
# ---------------------------------------------------------------------------


def test_processor_makes_no_network_connections(monkeypatch: pytest.MonkeyPatch) -> None:
    """Block the socket layer, then exercise every code path through it."""

    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("the mock processor attempted a network connection")

    monkeypatch.setattr(socket, "socket", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)

    for customer in store.list_customers():
        check_payment_status(customer)
        retry_payment(customer)
        if customer.autopay.backup_method is not None:
            retry_payment(customer, payment_method="backup")


def test_processor_imports_no_network_or_payment_sdk() -> None:
    """Static check on the module source, so a future import is caught too."""
    source = mock_processor.__file__
    assert source is not None
    with open(source, encoding="utf-8") as handle:
        text = handle.read()

    banned = [
        "import requests",
        "import httpx",
        "import urllib",
        "import socket",
        "import stripe",
        "import razorpay",
        "import paypal",
        "import boto3",
    ]
    for statement in banned:
        assert statement not in text, f"mock processor must not {statement}"


def test_result_never_carries_a_raw_credential() -> None:
    """Only the stored method id travels through results."""
    for customer in store.list_customers():
        payload = retry_payment(customer).model_dump_json()
        assert "pm_mock_" in payload
        assert not re.search(r"\d{12,}", payload), "no credential-length digit runs"
