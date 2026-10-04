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
    """How a conversation ended. Written once, before hangup.

    A closed set: ``log_disposition`` rejects anything outside it, so the
    agent cannot invent an outcome label that later reporting cannot count.

    ``WRONG_NUMBER`` and ``NO_ANSWER`` cover the telephony branches of the
    conversation flow (reached the wrong person, or voicemail), which have to
    be recordable if a real call is ever placed.
    """

    PAYMENT_RECOVERED = "payment_recovered"
    PAYMENT_LINK_PREPARED = "payment_link_prepared"
    RETRY_SCHEDULED = "retry_scheduled"
    ESCALATED = "escalated"
    CUSTOMER_DECLINED = "customer_declined"
    DO_NOT_CALL = "do_not_call"
    VERIFICATION_FAILED = "verification_failed"
    UNRESOLVED = "unresolved"
    WRONG_NUMBER = "wrong_number"
    NO_ANSWER = "no_answer"


class NextAction(str, Enum):
    """What the agent should consider doing next.

    Returned by tools so the conversational decision is driven by backend
    state rather than left to the model's improvisation.
    """

    NONE = "none"
    VERIFY_IDENTITY = "verify_identity"
    RETRY_PAYMENT = "retry_payment"
    OFFER_BACKUP_METHOD = "offer_backup_method"
    PAYMENT_LINK = "payment_link"
    SCHEDULE_RETRY = "schedule_retry"
    ESCALATE = "escalate"
    PAYMENT_LINK_OR_ESCALATION = "payment_link_or_escalation"


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
    session_id: str | None = None
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
    """A retry the customer asked us to postpone.

    A record only. No job runs: nothing in this project has a scheduler, and
    the demo deliberately stops at recording the customer's instruction.
    """

    customer_id: str
    session_id: str | None = None
    scheduled_for: date
    requested_action: Literal["retry_primary", "retry_backup"] = "retry_primary"
    status: Literal["scheduled", "cancelled"] = "scheduled"
    confirmation_number: str
    note: str = "Demo record only. No background job is created."
    created_at: datetime = Field(default_factory=utc_now)


class PaymentLink(RuntimeModel):
    """A 'secure payment link' the agent offered.

    Nothing is transmitted. The record exists so the demo can show that the
    agent refused to take a card number by voice and offered a link instead.
    ``delivered`` is permanently False, which is what lets the agent describe
    the link truthfully as prepared rather than sent.
    """

    customer_id: str
    session_id: str | None = None
    channel: Literal["sms", "email"]
    sent_to_masked: str
    link_id: str
    url: str
    delivered: bool = False
    note: str = "Mock only. No SMS or email is ever sent by this project."
    created_at: datetime = Field(default_factory=utc_now)


class Escalation(RuntimeModel):
    """A handoff to a human agent. Nobody is actually contacted."""

    customer_id: str
    session_id: str | None = None
    ticket_id: str
    reason: str
    notes: str | None = None
    status: Literal["open"] = "open"
    callback_window: str
    note: str = "Demo record only. No human is contacted by this project."
    created_at: datetime = Field(default_factory=utc_now)


class Session(RuntimeModel):
    """One conversation, bound to exactly one customer.

    The binding is set at creation and there is deliberately no operation
    anywhere in the codebase that changes ``customer_id`` on an existing
    session. Tool requests carry only ``session_id``, so the agent has no
    parameter through which it could address a different customer — the
    cross-customer disclosure risk is removed structurally rather than being
    left to the prompt.
    """

    session_id: str
    customer_id: str
    channel: Literal["web", "phone", "simulator", "test"] = "test"
    closed: bool = False
    tool_calls: int = 0
    retry_attempts: int = 0
    # Verification is a property of THIS conversation, never of the customer.
    # Holding it on the customer row meant a caller who failed twice locked
    # the account for every future call, and - worse - a caller who verified
    # left the next session pre-verified without saying a word.
    identity_verified: bool = False
    verification_attempts: int = 0

    # How THIS conversation ended, set by log_disposition. The customer row
    # keeps its own `disposition` as the ledger's last-known outcome, but that
    # is per-customer: a second conversation would otherwise appear to have
    # inherited the first one's result, the same way verification used to.
    outcome: Disposition | None = None

    # Post-call metadata, written once by the provider's webhook. Identifiers,
    # counts and timestamps only - never a transcript body, never a secret.
    completed_at: datetime | None = None
    conversation_id: str | None = None
    call_duration_seconds: int | None = None
    transcript_turns: int | None = None
    transcript_file: str | None = None

    created_at: datetime = Field(default_factory=utc_now)
    last_tool_at: datetime | None = None


class CustomerState(RuntimeModel):
    """Mutable per-customer state, layered over the immutable seed.

    Kept separate from ``Customer`` so ``data/customers.json`` stays a clean,
    resettable seed that a demo run never rewrites.
    """

    customer_id: str
    status: AutopayStatus = AutopayStatus.FAILED
    retry_attempts: int = 0
    do_not_call: bool = False
    disposition: Disposition | None = None
    disposition_notes: str | None = None
    last_transaction_id: str | None = None
    last_confirmation_number: str | None = None
    scheduled_for: date | None = None
    updated_at: datetime = Field(default_factory=utc_now)


# ---------------------------------------------------------------------------
# Tool layer: request and response schemas
#
# These are the contract the voice agent sees. Three rules shape them:
#
# 1. Every request carries only `session_id` to identify the customer, and
#    `extra="forbid"` rejects anything else — so a stray `customer_id` from a
#    confused model is a validation error, never a silent customer switch.
# 2. Responses are flat, shallow objects with short string values, because the
#    agent reads them aloud. No nested structures to mangle in speech.
# 3. Amounts are pre-formatted strings ("49.00"), so the model never has to
#    decide how to pronounce a Decimal.
# ---------------------------------------------------------------------------


class ToolRequest(BaseModel):
    """Base for every tool request: a session id, and nothing extra."""

    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(min_length=1, max_length=64)


class SessionOnlyRequest(ToolRequest):
    """For tools that need no argument beyond the session."""


class VerifyIdentityRequest(ToolRequest):
    """The postal code the caller stated, to check against the account.

    Postal code is the only verification factor by design. Nothing that could
    be a real credential — card number, CVC, password, OTP — is accepted here
    or anywhere else in the tool layer.
    """

    postal_code: str = Field(
        min_length=1,
        max_length=64,
        description=(
            "The postal code the customer stated, digits only where possible "
            "(for example 94107). Spoken forms such as '9 4 1 0 7' or "
            "'nine four one oh seven' are also accepted."
        ),
    )


class RetryPaymentRequest(ToolRequest):
    payment_method: Literal["primary", "backup"] = "primary"


class ScheduleRetryRequest(ToolRequest):
    requested_date: date
    requested_action: Literal["retry_primary", "retry_backup"] = "retry_primary"


class SendPaymentLinkRequest(ToolRequest):
    channel: Literal["sms", "email"] = "email"


class EscalateRequest(ToolRequest):
    reason: str = Field(min_length=1, max_length=280)
    notes: str | None = Field(default=None, max_length=500)


class LogDispositionRequest(ToolRequest):
    disposition: Disposition
    notes: str | None = Field(default=None, max_length=500)


class ToolResponse(BaseModel):
    """Base for every tool response."""

    model_config = ConfigDict(extra="forbid")

    success: bool
    message: str


class FailedPaymentDetailsResponse(ToolResponse):
    """What the agent needs to explain the failure — and nothing more.

    Deliberately absent: ``payment_method_id``, ``payment_id``, card brand,
    expiry, and last-4 as separate fields. ``card_description`` carries the
    one phrase the agent actually has to say out loud.

    Amounts are withheld until ``verified`` is true. That rule lives here in
    the backend, not only in the system prompt, so it holds even if the model
    is talked into asking early.
    """

    verified: bool
    customer_id: str
    customer_name: str
    status: str
    amount: str | None = None
    currency: str | None = None
    due_date: str | None = None
    days_delinquent: int | None = None
    failure_reason: str | None = None
    failure_explanation: str | None = None
    card_description: str | None = None
    retry_worth_attempting: bool | None = None
    backup_method_available: bool | None = None
    service_suspension_date: str | None = None
    next_action: NextAction


class VerifyIdentityResponse(ToolResponse):
    verified: bool
    attempts_remaining: int
    locked: bool = False
    next_action: NextAction


class RetryPaymentResponse(ToolResponse):
    status: str
    transaction_id: str | None = None
    confirmation_number: str | None = None
    amount: str | None = None
    currency: str | None = None
    failure_reason: str | None = None
    next_action: NextAction


class ScheduleRetryResponse(ToolResponse):
    status: str
    scheduled_for: str
    confirmation_number: str
    next_action: NextAction


class SendPaymentLinkResponse(ToolResponse):
    link: str
    channel: str
    sent_to_masked: str
    delivered: bool = False
    delivery_note: str
    next_action: NextAction


class EscalationResponse(ToolResponse):
    status: str
    ticket_id: str
    callback_window: str
    next_action: NextAction


class DispositionResponse(ToolResponse):
    disposition: Disposition
    do_not_call: bool
    session_closed: bool
    next_action: NextAction


class ToolErrorResponse(BaseModel):
    """Uniform error envelope for every /tools/* failure.

    ``error`` is a stable machine code for the dashboard and tests;
    ``message`` is a short sentence the agent can say without rephrasing.
    Stack traces and internal detail never appear in either.
    """

    model_config = ConfigDict(extra="forbid")

    success: Literal[False] = False
    error: str
    message: str


# ---------------------------------------------------------------------------
# Session events
#
# The demo dashboard needs a chronological narrative of what happened on a
# call. The audit collections (attempts, links, escalations) each hold part of
# that story but not all of it — nothing records that identity was verified,
# or that details were looked up and withheld. So tools append to one
# append-only event log instead.
#
# Ordering uses a monotonic `sequence` rather than the timestamp: several
# events can land inside the same millisecond, and "chronological" has to be
# exact when the dashboard is being filmed.
# ---------------------------------------------------------------------------


class EventType(str, Enum):
    """Everything the dashboard can show on a session timeline."""

    SESSION_CREATED = "session_created"
    SESSION_RESET = "session_reset"
    PAYMENT_DETAILS_VIEWED = "payment_details_viewed"
    PAYMENT_DETAILS_WITHHELD = "payment_details_withheld"
    IDENTITY_VERIFIED = "identity_verified"
    IDENTITY_VERIFICATION_FAILED = "identity_verification_failed"
    IDENTITY_VERIFICATION_LOCKED = "identity_verification_locked"
    PAYMENT_RETRY_ATTEMPTED = "payment_retry_attempted"
    PAYMENT_RETRY_SUCCEEDED = "payment_retry_succeeded"
    PAYMENT_RETRY_DECLINED = "payment_retry_declined"
    PAYMENT_RETRY_SKIPPED = "payment_retry_skipped"
    PAYMENT_SCHEDULED = "payment_scheduled"
    PAYMENT_LINK_PREPARED = "payment_link_prepared"
    HUMAN_ESCALATION_CREATED = "human_escalation_created"
    DISPOSITION_LOGGED = "disposition_logged"
    DIAL_CHECK_ALLOWED = "dial_check_allowed"
    DIAL_CHECK_REFUSED = "dial_check_refused"
    VOICE_CALL_COMPLETED = "voice_call_completed"


class SessionEvent(RuntimeModel):
    """One entry on a session's timeline.

    ``detail`` holds a few short, safe strings for the dashboard. It must
    never carry a verification answer, a submitted postal code, a secret, or
    an internal payment identifier.
    """

    sequence: int
    session_id: str
    customer_id: str
    event_type: EventType
    summary: str
    detail: dict[str, str] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


# ---------------------------------------------------------------------------
# Demo control plane: request and response schemas
#
# These serve the local dashboard, not the voice agent. They are deliberately
# separate from the tool schemas above, and they are hand-built rather than
# dumps of the internal models — serialising `Customer` wholesale would leak
# the verification answer, the postal code it is derived from, and the
# internal payment identifiers.
# ---------------------------------------------------------------------------


class CreateSessionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    customer_id: str = Field(pattern=r"^CUST-\d{3}$")
    channel: Literal["web", "phone", "simulator", "test"] = "web"


class CreateSessionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str
    customer_id: str
    customer_name: str
    status: Literal["active", "closed"]
    channel: str
    created_at: datetime


class SessionStateResponse(BaseModel):
    """Everything the dashboard shows about a live session."""

    model_config = ConfigDict(extra="forbid")

    session_id: str
    customer_id: str
    customer_name: str
    status: Literal["active", "closed"]
    channel: str
    identity_verified: bool
    verification_attempts: int
    verification_attempts_allowed: int
    payment_status: AutopayStatus
    amount_due: str
    currency: str
    retry_attempts: int
    last_confirmation_number: str | None = None
    scheduled_for: date | None = None
    payment_link_prepared: bool
    escalated: bool
    escalation_ticket: str | None = None
    disposition: Disposition | None = None
    disposition_notes: str | None = None
    do_not_call: bool
    tool_calls: int
    event_count: int
    outcome: Disposition | None = None
    call_completed: bool = False
    completed_at: datetime | None = None
    call_duration_seconds: int | None = None
    transcript_turns: int | None = None
    created_at: datetime
    last_tool_at: datetime | None = None


class CustomerSummary(BaseModel):
    """A customer as the dashboard's selection list sees them.

    Note the absences: no postal code, no verification answer, no payment or
    payment-method identifiers. The postal code is the verification answer, so
    exposing the address would hand over the credential by another route.
    """

    model_config = ConfigDict(extra="forbid")

    customer_id: str
    name: str
    phone: str
    email: str
    plan_name: str
    amount: str
    currency: str
    failure_reason: FailureCode
    failure_explanation: str
    days_delinquent: int
    card_description: str
    scenario_label: str
    expected_path: str
    caller_intent: str
    payment_status: AutopayStatus
    do_not_call: bool


class CustomerDetail(CustomerSummary):
    """One customer, with the extra context the dashboard panel shows."""

    city: str
    state: str
    timezone: str
    billing_period: str
    due_date: date
    failed_at: datetime
    attempts_before_this_call: int
    service_suspension_date: date | None = None
    backup_method_available: bool
    verification_method: str
    verification_attempts_allowed: int


class SessionEventOut(BaseModel):
    """One timeline entry, as served to the dashboard."""

    model_config = ConfigDict(extra="forbid")

    sequence: int
    event_type: EventType
    summary: str
    detail: dict[str, str]
    created_at: datetime


class ResetSessionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str
    customer_id: str
    status: Literal["active", "closed"]
    message: str
    records_cleared: int
    seed_unchanged: bool = True
