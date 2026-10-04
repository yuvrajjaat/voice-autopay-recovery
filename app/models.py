"""Pydantic models for the autopay recovery demo.

Two kinds of model live here:

* **Seed models** (``Customer`` and its children) mirror ``data/customers.json``
  exactly and are frozen. They validate the seed on load, so a malformed or
  incomplete record fails immediately with a precise message rather than
  surfacing as a ``KeyError`` mid-conversation.
* **Runtime models** (``CustomerState``, ``PaymentAttempt``,
  ``ScheduledRetry``, ``PaymentLink``, ``PaymentResult``) describe what the
  agent does during a call. These are mutable and persist to
  ``data/runtime.json``.

Money is ``Decimal`` throughout, parsed from decimal strings in the JSON.
Floats are never used for amounts.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_serializer

# A deliberately simple email shape check. Pydantic's EmailStr would be
# stricter but pulls in the `email-validator` package, and the only thing that
# actually matters here is asserted by the test suite instead: every address
# must sit on example.com, which cannot receive mail.
EMAIL_PATTERN = r"^[^@\s]+@[^@\s]+\.[A-Za-z]{2,}$"


def utc_now() -> datetime:
    """Timezone-aware UTC timestamp, used for every runtime record."""
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class AutopayStatus(str, Enum):
    """Where the customer's autopay stands in our ledger."""

    ACTIVE = "active"
    FAILED = "failed"
    RECOVERED = "recovered"
    SCHEDULED = "scheduled"
    ESCALATED = "escalated"
    LINK_SENT = "link_sent"
    DO_NOT_CALL = "do_not_call"


class FailureCode(str, Enum):
    """Decline reasons the mock processor can report.

    These mirror the vocabulary real card processors use, so the conversation
    sounds right, but nothing here touches a real network.
    """

    INSUFFICIENT_FUNDS = "insufficient_funds"
    EXPIRED_CARD = "expired_card"
    CARD_DECLINED = "card_declined"
    CLOSED_ACCOUNT = "closed_account"
    INCORRECT_CVC = "incorrect_cvc"
    DO_NOT_HONOR = "do_not_honor"
    PROCESSOR_NETWORK_ERROR = "processor_network_error"
    THREE_DS_REQUIRED = "3ds_required"


class MockRetryOutcome(str, Enum):
    """The deterministic script each fictional customer follows on retry.

    This is what makes a demo repeatable: CUST-001 always approves and
    CUST-002 always declines, so a recorded walkthrough cannot surprise you.
    """

    SUCCEED = "succeed"
    SUCCEED_ON_BACKUP = "succeed_on_backup"
    FAIL_INSUFFICIENT_FUNDS = "fail_insufficient_funds"
    FAIL_EXPIRED_CARD = "fail_expired_card"
    FAIL_CLOSED_ACCOUNT = "fail_closed_account"
    FAIL_DO_NOT_HONOR = "fail_do_not_honor"


class PaymentStatus(str, Enum):
    """Result of a single mock authorization."""

    PAID = "paid"
    FAILED = "failed"


class Disposition(str, Enum):
    """How a conversation ended. Written once, before hangup."""

    RECOVERED = "recovered"
    RETRY_FAILED = "retry_failed"
    RETRY_SCHEDULED = "retry_scheduled"
    LINK_SENT = "link_sent"
    ESCALATED = "escalated"
    VERIFICATION_FAILED = "verification_failed"
    DO_NOT_CALL = "do_not_call"
    DECLINED = "declined"
    WRONG_NUMBER = "wrong_number"
    NO_ANSWER = "no_answer"


# ---------------------------------------------------------------------------
# Seed models: a 1:1 mirror of data/customers.json
# ---------------------------------------------------------------------------


class FrozenModel(BaseModel):
    """Base for seed models: immutable, and rejects unexpected fields.

    ``extra="forbid"`` is deliberate. If someone adds a field to the JSON
    without adding it here, the load fails loudly instead of silently
    discarding data the rest of the code might expect.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")


class Address(FrozenModel):
    city: str
    state: str
    postal_code: str


class Verification(FrozenModel):
    """How the agent confirms it is speaking to the account holder."""

    method: Literal["postal_code"] = "postal_code"
    expected_answer: str
    max_attempts: int = Field(default=2, ge=1, le=5)


class Plan(FrozenModel):
    name: str
    billing_period: Literal["monthly", "annual"]


class PaymentMethod(FrozenModel):
    """A stored payment method: brand and last-4 only.

    There is deliberately no field for a card number, expiry-plus-CVC pair,
    token, or bank detail. The mock processor charges a
    ``payment_method_id``, never raw credentials, which is why "no real
    payments" holds structurally rather than by convention.
    """

    payment_method_id: str
    type: Literal["card"] = "card"
    brand: str
    last4: str = Field(min_length=4, max_length=4, pattern=r"^\d{4}$")
    exp_month: int = Field(ge=1, le=12)
    exp_year: int = Field(ge=2024, le=2040)

    @property
    def spoken(self) -> str:
        """How the agent should refer to this card out loud."""
        return f"the {self.brand} ending {self.last4}"


class Autopay(FrozenModel):
    enabled: bool
    status: AutopayStatus
    primary_method: PaymentMethod
    backup_method: PaymentMethod | None = None


class FailedPayment(FrozenModel):
    """The failed autopay charge this call is about."""

    payment_id: str
    amount: Decimal
    currency: str = "USD"
    due_date: date
    failed_at: datetime
    failure_code: FailureCode
    failure_reason_spoken: str
    attempts_so_far: int = Field(ge=0)
    days_delinquent: int = Field(ge=0)
    service_suspension_date: date | None = None

    @field_serializer("amount")
    def _amount_as_string(self, value: Decimal) -> str:
        return f"{value:.2f}"

    @property
    def amount_spoken(self) -> str:
        """Amount formatted for speech, e.g. "49.00"."""
        return f"{self.amount:.2f}"


class Scenario(FrozenModel):
    """Deterministic demo script attached to each fictional customer."""

    label: str
    expected_path: str
    mock_retry_outcome: MockRetryOutcome
    caller_intent: str
    caller_provides_answer: str


class Customer(FrozenModel):
    """One fictional customer, exactly as stored in the seed file."""

    customer_id: str = Field(pattern=r"^CUST-\d{3}$")
    name: str
    email: str = Field(pattern=EMAIL_PATTERN)
    phone: str = Field(pattern=r"^\+\d{10,15}$")
    timezone: str
    address: Address
    verification: Verification
    plan: Plan
    autopay: Autopay
    failed_payment: FailedPayment
    scenario: Scenario

    def method(self, which: Literal["primary", "backup"]) -> PaymentMethod | None:
        """Return the named payment method, or None if there is no backup."""
        return self.autopay.primary_method if which == "primary" else self.autopay.backup_method


# ---------------------------------------------------------------------------
# Runtime models: what the agent does, persisted to data/runtime.json
# ---------------------------------------------------------------------------


class RuntimeModel(BaseModel):
    """Base for mutable runtime records."""

    model_config = ConfigDict(extra="forbid")


class PaymentResult(RuntimeModel):
    """Structured outcome of one mock authorization.

    Returned by the processor and, later, read aloud by the agent — hence
    ``message``, which is written as a speakable sentence fragment rather than
    an error code.
    """

    success: bool
    status: PaymentStatus
    transaction_id: str
    message: str
    amount: Decimal
    currency: str = "USD"
    payment_method_id: str
    payment_method: Literal["primary", "backup"]
    failure_code: FailureCode | None = None
    confirmation_number: str | None = None
    processor: Literal["mock"] = "mock"

    @field_serializer("amount")
    def _amount_as_string(self, value: Decimal) -> str:
        return f"{value:.2f}"


class PaymentAttempt(RuntimeModel):
    """Audit record of one retry the agent triggered."""

    customer_id: str
    attempt_number: int = Field(ge=1)
    payment_method: Literal["primary", "backup"]
    status: PaymentStatus
    transaction_id: str
    amount: Decimal
    failure_code: FailureCode | None = None
    confirmation_number: str | None = None
    created_at: datetime = Field(default_factory=utc_now)

    @field_serializer("amount")
    def _amount_as_string(self, value: Decimal) -> str:
        return f"{value:.2f}"


class ScheduledRetry(RuntimeModel):
    """A retry the customer asked us to postpone."""

    customer_id: str
    scheduled_for: date
    confirmation_number: str
    created_at: datetime = Field(default_factory=utc_now)


class PaymentLink(RuntimeModel):
    """A 'secure payment link' the agent offered.

    Nothing is transmitted. The record exists so the demo can show that the
    agent refused to take a card number by voice and offered a link instead.
    """

    customer_id: str
    channel: Literal["sms", "email"]
    sent_to_masked: str
    link_id: str
    delivered: bool = False
    note: str = "Mock only. No SMS or email is ever sent by this project."
    created_at: datetime = Field(default_factory=utc_now)


class Escalation(RuntimeModel):
    """A handoff to a human agent."""

    customer_id: str
    ticket_id: str
    reason: str
    callback_window: str
    created_at: datetime = Field(default_factory=utc_now)


class CustomerState(RuntimeModel):
    """Mutable per-customer state, layered over the immutable seed.

    Kept separate from ``Customer`` so ``data/customers.json`` stays a clean,
    resettable seed that a demo run never rewrites.
    """

    customer_id: str
    status: AutopayStatus = AutopayStatus.FAILED
    identity_verified: bool = False
    verification_attempts: int = 0
    retry_attempts: int = 0
    do_not_call: bool = False
    disposition: Disposition | None = None
    disposition_notes: str | None = None
    last_transaction_id: str | None = None
    last_confirmation_number: str | None = None
    scheduled_for: date | None = None
    updated_at: datetime = Field(default_factory=utc_now)
