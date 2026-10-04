"""Deterministic mock payment processor.

This module is the whole "payment system" for the project. It is entirely
offline by construction:

* It imports no HTTP client, no socket, and no payment SDK. The test suite
  asserts that, and separately blocks ``socket.socket`` while calling every
  function here, so a future accidental network call fails the build.
* It charges a stored ``payment_method_id`` and never accepts a card number,
  CVC, token, or bank detail. There is no parameter to pass one through.
* Every outcome is decided by the customer's ``scenario.mock_retry_outcome``
  in the seed file. There is no randomness and no clock dependence, so a
  recorded demo behaves the same way on every run — which matters when you
  cannot retake a live phone call ten times.

Transaction and confirmation identifiers are derived by hashing
``customer_id`` + attempt number + method, so they are stable across runs but
still look like opaque processor references.
"""

from __future__ import annotations

import hashlib
import time
from decimal import Decimal
from typing import Literal

from app.models import (
    Customer,
    FailureCode,
    MockRetryOutcome,
    PaymentResult,
    PaymentStatus,
)

PROCESSOR_NAME = "mock"

PaymentMethodChoice = Literal["primary", "backup"]

#: Maps a scripted failure outcome to the decline code it reports.
_FAILURE_CODES: dict[MockRetryOutcome, FailureCode] = {
    MockRetryOutcome.FAIL_INSUFFICIENT_FUNDS: FailureCode.INSUFFICIENT_FUNDS,
    MockRetryOutcome.FAIL_EXPIRED_CARD: FailureCode.EXPIRED_CARD,
    MockRetryOutcome.FAIL_CLOSED_ACCOUNT: FailureCode.CLOSED_ACCOUNT,
    MockRetryOutcome.FAIL_DO_NOT_HONOR: FailureCode.DO_NOT_HONOR,
}

#: Speakable explanation per decline code. The agent reads these verbatim, so
#: a failure never produces improvised wording about someone's money.
_DECLINE_MESSAGES: dict[FailureCode, str] = {
    FailureCode.INSUFFICIENT_FUNDS: (
        "That didn't go through — the bank says there aren't enough funds available."
    ),
    FailureCode.EXPIRED_CARD: (
        "That card has expired, so the bank won't accept a charge on it."
    ),
    FailureCode.CARD_DECLINED: "The bank declined the charge.",
    FailureCode.CLOSED_ACCOUNT: (
        "The bank says the account behind that card is closed, "
        "so no charge can go through on it."
    ),
    FailureCode.INCORRECT_CVC: "The security code on file didn't match.",
    FailureCode.DO_NOT_HONOR: (
        "The bank declined the charge without giving a reason."
    ),
    FailureCode.PROCESSOR_NETWORK_ERROR: (
        "The payment network timed out before the charge completed."
    ),
    FailureCode.THREE_DS_REQUIRED: (
        "The bank wants extra verification that can't be completed automatically."
    ),
}

#: Decline codes where retrying the same card cannot possibly help, so the
#: agent should stop rather than burn a second attempt.
TERMINAL_FAILURE_CODES: frozenset[FailureCode] = frozenset(
    {
        FailureCode.EXPIRED_CARD,
        FailureCode.CLOSED_ACCOUNT,
        FailureCode.INCORRECT_CVC,
        FailureCode.THREE_DS_REQUIRED,
    }
)


class PaymentMethodUnavailableError(RuntimeError):
    """Raised when the requested method does not exist on the account."""


# ---------------------------------------------------------------------------
# Deterministic identifiers
# ---------------------------------------------------------------------------


def _digest(customer_id: str, attempt_number: int, method: str) -> str:
    """Stable hex digest for one (customer, attempt, method) triple."""
    seed = f"{customer_id}:{attempt_number}:{method}"
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()


def build_transaction_id(customer_id: str, attempt_number: int, method: str) -> str:
    """Fake processor reference, e.g. ``mock_txn_3f9a21c0d4``.

    Deterministic: the same inputs always yield the same id, which lets tests
    assert exact values and lets a demo be replayed.
    """
    return f"mock_txn_{_digest(customer_id, attempt_number, method)[:10]}"


def build_confirmation_number(customer_id: str, attempt_number: int, method: str) -> str:
    """Short confirmation number the agent can read aloud, e.g. ``C-8841``."""
    digits = int(_digest(customer_id, attempt_number, method)[:8], 16) % 10000
    return f"C-{digits:04d}"


def build_payment_link(customer_id: str, channel: str) -> tuple[str, str]:
    """Return ``(link_id, url)`` for a mock payment link.

    The host is ``example.test``: ``.test`` is an IANA-reserved TLD that can
    never resolve, so the URL is inert even if someone pastes it into a
    browser. Nothing is transmitted — the caller records the request and the
    agent describes the link as prepared, not sent.
    """
    link_id = f"link_mock_{_digest(customer_id, 0, channel)[:10]}"
    return link_id, f"https://example.test/pay/{link_id}"


def build_ticket_id(customer_id: str, reason: str) -> str:
    """Deterministic escalation ticket reference, e.g. ``TCK-4F91A2``."""
    seed = hashlib.sha256(f"{customer_id}:{reason}".encode("utf-8")).hexdigest()
    return f"TCK-{seed[:6].upper()}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def check_payment_status(customer: Customer) -> dict[str, object]:
    """Report the state of the failed autopay charge without charging anything.

    Read-only. Returns the plain facts the agent needs to explain the problem,
    plus ``retry_worth_attempting``, which is False when the decline code
    means a retry on the same card cannot succeed.
    """
    payment = customer.failed_payment
    terminal = payment.failure_code in TERMINAL_FAILURE_CODES
    return {
        "customer_id": customer.customer_id,
        "payment_id": payment.payment_id,
        "status": PaymentStatus.FAILED.value,
        "amount": f"{payment.amount:.2f}",
        "currency": payment.currency,
        "due_date": payment.due_date.isoformat(),
        "failed_at": payment.failed_at.isoformat(),
        "failure_code": payment.failure_code.value,
        "failure_reason": payment.failure_reason_spoken,
        "attempts_so_far": payment.attempts_so_far,
        "days_delinquent": payment.days_delinquent,
        "card": customer.autopay.primary_method.spoken,
        "backup_method_available": customer.autopay.backup_method is not None,
        "retry_worth_attempting": not terminal,
        "processor": PROCESSOR_NAME,
    }


def retry_payment(
    customer: Customer,
    payment_method: PaymentMethodChoice = "primary",
    attempt_number: int = 1,
    latency_seconds: float = 0.0,
) -> PaymentResult:
    """Run one mock authorization and return a structured result.

    The outcome comes from ``customer.scenario.mock_retry_outcome``:

    ``succeed``
        Approves on whichever method was asked for.
    ``succeed_on_backup``
        Declines on the primary card, approves on the backup. This is what
        drives the "let's try the other card on file" branch.
    ``fail_*``
        Declines with the matching code, every time.

    ``latency_seconds`` exists so the tool layer can add a realistic pause
    before the agent speaks the result. It defaults to zero so the test suite
    stays fast and deterministic.
    """
    method = customer.method(payment_method)
    if method is None:
        raise PaymentMethodUnavailableError(
            f"{customer.customer_id} has no {payment_method} payment method on file"
        )

    if latency_seconds > 0:
        time.sleep(latency_seconds)

    outcome = customer.scenario.mock_retry_outcome
    amount: Decimal = customer.failed_payment.amount
    transaction_id = build_transaction_id(
        customer.customer_id, attempt_number, payment_method
    )

    if outcome is MockRetryOutcome.SUCCEED:
        approved = True
        failure_code = None
    elif outcome is MockRetryOutcome.SUCCEED_ON_BACKUP:
        approved = payment_method == "backup"
        failure_code = None if approved else FailureCode.CARD_DECLINED
    else:
        approved = False
        failure_code = _FAILURE_CODES[outcome]

    if approved:
        confirmation_number = build_confirmation_number(
            customer.customer_id, attempt_number, payment_method
        )
        return PaymentResult(
            success=True,
            status=PaymentStatus.PAID,
            transaction_id=transaction_id,
            message=(
                f"Approved. {amount:.2f} {customer.failed_payment.currency} "
                f"was charged to {method.spoken}."
            ),
            amount=amount,
            currency=customer.failed_payment.currency,
            payment_method_id=method.payment_method_id,
            payment_method=payment_method,
            failure_code=None,
            confirmation_number=confirmation_number,
        )

    assert failure_code is not None  # every non-approved branch sets this
    return PaymentResult(
        success=False,
        status=PaymentStatus.FAILED,
        transaction_id=transaction_id,
        message=_DECLINE_MESSAGES[failure_code],
        amount=amount,
        currency=customer.failed_payment.currency,
        payment_method_id=method.payment_method_id,
        payment_method=payment_method,
        failure_code=failure_code,
        confirmation_number=None,
    )


def is_retry_worth_attempting(failure_code: FailureCode) -> bool:
    """Whether retrying the same card after this decline code could work."""
    return failure_code not in TERMINAL_FAILURE_CODES
