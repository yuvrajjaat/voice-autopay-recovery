"""Tests for the JSON-backed store layer."""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path

import pytest

from app import store
from app.models import AutopayStatus, Disposition
from app.payments.mock_processor import retry_payment


# ---------------------------------------------------------------------------
# Reading the seed
# ---------------------------------------------------------------------------


def test_load_all_ten_customers() -> None:
    customers = store.load_customers()
    assert len(customers) == 10
    assert set(customers) == {f"CUST-{n:03d}" for n in range(1, 11)}


def test_get_customer_by_id() -> None:
    customer = store.get_customer("CUST-005")
    assert customer.name == "Aisha Noor"
    assert customer.failed_payment.amount == pytest.approx(75.50)


def test_get_unknown_customer_raises() -> None:
    with pytest.raises(store.CustomerNotFoundError):
        store.get_customer("CUST-999")


def test_find_customer_by_phone_ignores_formatting() -> None:
    """Phone lookup compares digits only, so any format matches."""
    expected = "CUST-001"
    for variant in (
        "+14155550100",
        "14155550100",
        "+1 (415) 555-0100",
        "1-415-555-0100",
    ):
        found = store.find_customer_by_phone(variant)
        assert found is not None, variant
        assert found.customer_id == expected, variant


def test_find_customer_by_unknown_phone_returns_none() -> None:
    assert store.find_customer_by_phone("+15035550999") is None
    assert store.find_customer_by_phone("") is None


# ---------------------------------------------------------------------------
# Runtime state
# ---------------------------------------------------------------------------


def test_default_state_comes_from_the_seed_without_writing() -> None:
    state = store.get_state("CUST-001")
    assert state.status is AutopayStatus.FAILED
    assert state.identity_verified is False
    assert state.retry_attempts == 0
    assert state.disposition is None


def test_update_state_persists_changes() -> None:
    updated = store.update_state("CUST-001", identity_verified=True, verification_attempts=1)
    assert updated.identity_verified is True
    assert updated.verification_attempts == 1

    reread = store.get_state("CUST-001")
    assert reread.identity_verified is True
    assert reread.verification_attempts == 1


def test_state_survives_a_full_reload(isolated_runtime: Path) -> None:
    """Write, drop every in-memory cache, read the file back."""
    store.update_state("CUST-004", identity_verified=True, retry_attempts=2)

    store._customers_cache = None  # force the seed to be re-validated
    document = json.loads(isolated_runtime.read_text(encoding="utf-8"))
    assert document["customer_state"]["CUST-004"]["retry_attempts"] == 2

    assert store.get_state("CUST-004").retry_attempts == 2


def test_update_state_rejects_unknown_fields() -> None:
    with pytest.raises(ValueError, match="Unknown CustomerState field"):
        store.update_state("CUST-001", nonexistent_field=True)


def test_update_state_for_unknown_customer_raises() -> None:
    with pytest.raises(store.CustomerNotFoundError):
        store.update_state("CUST-999", identity_verified=True)


def test_all_states_covers_every_customer() -> None:
    states = store.all_states()
    assert len(states) == 10
    assert all(state.status is AutopayStatus.FAILED for state in states.values())


# ---------------------------------------------------------------------------
# Audit records
# ---------------------------------------------------------------------------


def test_record_payment_attempt_logs_and_marks_recovered() -> None:
    customer = store.get_customer("CUST-001")
    result = retry_payment(customer)

    attempt = store.record_payment_attempt("CUST-001", result, attempt_number=1)
    assert attempt.attempt_number == 1
    assert attempt.transaction_id == result.transaction_id

    state = store.get_state("CUST-001")
    assert state.status is AutopayStatus.RECOVERED
    assert state.retry_attempts == 1
    assert state.last_confirmation_number == result.confirmation_number

    history = store.get_payment_attempts("CUST-001")
    assert len(history) == 1
    assert history[0]["status"] == "paid"


def test_failed_attempt_leaves_the_customer_unrecovered() -> None:
    customer = store.get_customer("CUST-009")
    result = retry_payment(customer)
    assert result.success is False

    store.record_payment_attempt("CUST-009", result, attempt_number=1)
    state = store.get_state("CUST-009")
    assert state.status is AutopayStatus.FAILED
    assert state.retry_attempts == 1
    assert state.last_confirmation_number is None


def test_next_attempt_number_counts_only_our_own_retries() -> None:
    """The seed's attempts_so_far happened before we called; it doesn't count."""
    assert store.get_customer("CUST-009").failed_payment.attempts_so_far == 3
    assert store.next_attempt_number("CUST-009") == 1

    result = retry_payment(store.get_customer("CUST-009"))
    store.record_payment_attempt("CUST-009", result, attempt_number=1)
    assert store.next_attempt_number("CUST-009") == 2


def test_payment_attempts_are_scoped_per_customer() -> None:
    store.record_payment_attempt(
        "CUST-001", retry_payment(store.get_customer("CUST-001")), attempt_number=1
    )
    store.record_payment_attempt(
        "CUST-005", retry_payment(store.get_customer("CUST-005")), attempt_number=1
    )

    assert len(store.get_payment_attempts("CUST-001")) == 1
    assert len(store.get_payment_attempts("CUST-005")) == 1
    assert len(store.get_payment_attempts()) == 2


def test_record_scheduled_retry() -> None:
    retry = store.record_scheduled_retry("CUST-004", date(2026, 10, 15), "C-1234")
    assert retry.scheduled_for == date(2026, 10, 15)

    state = store.get_state("CUST-004")
    assert state.status is AutopayStatus.SCHEDULED
    assert state.scheduled_for == date(2026, 10, 15)
    assert state.last_confirmation_number == "C-1234"


def test_record_payment_link_is_never_marked_delivered() -> None:
    """Nothing is actually sent, so `delivered` must stay False."""
    link = store.record_payment_link("CUST-002", "sms", "+1******0101", "link_mock_01")
    assert link.delivered is False
    assert "Mock only" in link.note
    assert store.get_state("CUST-002").status is AutopayStatus.LINK_SENT


def test_record_escalation() -> None:
    escalation = store.record_escalation(
        "CUST-006", "TCK-1001", "bank closed the account", "within one business day"
    )
    assert escalation.ticket_id == "TCK-1001"
    assert store.get_state("CUST-006").status is AutopayStatus.ESCALATED


def test_record_disposition_stores_notes() -> None:
    state = store.record_disposition("CUST-008", Disposition.ESCALATED, "disputes the amount")
    assert state.disposition is Disposition.ESCALATED
    assert state.disposition_notes == "disputes the amount"


def test_do_not_call_sets_the_opt_out_flag() -> None:
    """Opt-out has to outlive the call it was made on."""
    state = store.record_disposition("CUST-010", Disposition.DO_NOT_CALL, "asked us to stop")
    assert state.do_not_call is True
    assert state.status is AutopayStatus.DO_NOT_CALL
    assert store.get_state("CUST-010").do_not_call is True


# ---------------------------------------------------------------------------
# File handling
# ---------------------------------------------------------------------------


def test_runtime_file_is_created_on_first_use(isolated_runtime: Path) -> None:
    assert not isolated_runtime.exists()
    store.load_runtime()
    assert isolated_runtime.exists()

    document = json.loads(isolated_runtime.read_text(encoding="utf-8"))
    assert document["version"] == store.RUNTIME_VERSION
    assert document["customer_state"] == {}


def test_reset_runtime_clears_all_state(isolated_runtime: Path) -> None:
    store.update_state("CUST-001", identity_verified=True)
    store.record_payment_attempt(
        "CUST-001", retry_payment(store.get_customer("CUST-001")), attempt_number=1
    )
    assert store.get_payment_attempts()

    store.reset_runtime()
    assert store.get_payment_attempts() == []
    assert store.get_state("CUST-001").identity_verified is False
    assert store.get_state("CUST-001").status is AutopayStatus.FAILED


def test_writes_leave_no_temporary_files_behind(isolated_runtime: Path) -> None:
    """Atomic replace must not litter the data directory."""
    for index in range(5):
        store.update_state("CUST-001", verification_attempts=index)

    leftovers = list(isolated_runtime.parent.glob(".runtime-*"))
    assert leftovers == [], f"temporary files left behind: {leftovers}"


def test_sequential_writes_never_corrupt_the_file(isolated_runtime: Path) -> None:
    """Many read-modify-write cycles, then the file must still parse."""
    for index in range(25):
        store.update_state("CUST-002", verification_attempts=index % 3)
        store.update_state("CUST-003", retry_attempts=index % 2)

    document = json.loads(isolated_runtime.read_text(encoding="utf-8"))
    assert document["customer_state"]["CUST-002"]["verification_attempts"] == 24 % 3
    assert document["customer_state"]["CUST-003"]["retry_attempts"] == 24 % 2


def test_corrupt_runtime_file_is_quarantined_and_replaced(isolated_runtime: Path) -> None:
    isolated_runtime.write_text("{ this is not json", encoding="utf-8")

    document = store.load_runtime()
    assert document["customer_state"] == {}
    assert isolated_runtime.with_suffix(".corrupt.json").exists()


def test_the_seed_file_is_never_modified(seed_path: Path) -> None:
    """Every write path must leave data/customers.json byte-identical."""
    before = hashlib.sha256(seed_path.read_bytes()).hexdigest()

    store.update_state("CUST-001", identity_verified=True)
    store.record_payment_attempt(
        "CUST-001", retry_payment(store.get_customer("CUST-001")), attempt_number=1
    )
    store.record_scheduled_retry("CUST-004", date(2026, 10, 15), "C-1234")
    store.record_payment_link("CUST-002", "email", "d***@example.com", "link_mock_02")
    store.record_escalation("CUST-006", "TCK-1", "closed account", "one business day")
    store.record_disposition("CUST-010", Disposition.DO_NOT_CALL, "opted out")
    store.reset_runtime()

    after = hashlib.sha256(seed_path.read_bytes()).hexdigest()
    assert before == after, "the seed file must be read-only at runtime"
