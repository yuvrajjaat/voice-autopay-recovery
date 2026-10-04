"""Machine-readable specifications for the seven agent tools.

This is the single description of the tool surface, consumed by:

* the ElevenLabs provisioning script, which turns each spec into a webhook
  tool (URL, method, body schema, auth header);
* the offline simulator, which checks it only calls tools that exist;
* the test suite, which asserts every spec matches a real route.

Request schemas are **generated** from the Pydantic request models rather than
restated here. A hand-written copy would drift from the endpoint the first time
a field changed; ``model_json_schema()`` cannot.

Parameter ownership
-------------------
``injected_parameters`` are supplied by the voice platform as dynamic
variables; ``llm_parameters`` are the ones the model fills from what it hears.
``session_id`` is always injected, never LLM-filled — that is what prevents a
tool call being steered at a different customer's account.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.models import (
    EscalateRequest,
    LogDispositionRequest,
    RetryPaymentRequest,
    ScheduleRetryRequest,
    SendPaymentLinkRequest,
    SessionOnlyRequest,
    VerifyIdentityRequest,
)

#: Header carrying the shared secret. Its *value* lives in .env and in the
#: voice platform's secret store, never in this repository.
AUTH_HEADER = "X-Tool-Secret"

#: Path prefix every tool is mounted under.
TOOL_PREFIX = "/tools"


@dataclass(frozen=True)
class ToolSpec:
    """Everything needed to register one tool with a voice provider."""

    name: str
    description: str
    when_to_use: str
    response_description: str
    request_model: type
    llm_parameters: tuple[str, ...] = ()
    injected_parameters: tuple[str, ...] = ("session_id",)
    restrictions: tuple[str, ...] = ()
    requires_verification: bool = False
    method: str = field(default="POST")

    @property
    def path(self) -> str:
        """The endpoint path, derived from the tool name."""
        return f"{TOOL_PREFIX}/{self.name}"

    @property
    def request_schema(self) -> dict[str, Any]:
        """JSON Schema for the request body, generated from the model."""
        return self.request_model.model_json_schema()

    def url(self, base_url: str) -> str:
        """Absolute URL for a given public base, e.g. an ngrok domain."""
        return f"{base_url.rstrip('/')}{self.path}"

    def as_dict(self) -> dict[str, Any]:
        """Flat dictionary form, for provisioning payloads and inspection."""
        return {
            "name": self.name,
            "description": self.description,
            "method": self.method,
            "path": self.path,
            "request_schema": self.request_schema,
            "response_description": self.response_description,
            "when_to_use": self.when_to_use,
            "restrictions": list(self.restrictions),
            "llm_parameters": list(self.llm_parameters),
            "injected_parameters": list(self.injected_parameters),
            "requires_verification": self.requires_verification,
            "auth_header": AUTH_HEADER,
        }


TOOL_SPECS: tuple[ToolSpec, ...] = (
    ToolSpec(
        name="get_failed_payment_details",
        description=(
            "Look up the failed automatic payment for this call: the amount, "
            "why it failed, and whether a retry is worth attempting."
        ),
        when_to_use=(
            "Immediately after identity is verified, before explaining anything "
            "to the customer. Never state an amount you have not read from this."
        ),
        response_description=(
            "amount, currency, due_date, failure_reason, failure_explanation, "
            "card_description, retry_worth_attempting, backup_method_available, "
            "service_suspension_date, and next_action. Called before verification "
            "it returns verified=false with every figure omitted."
        ),
        request_model=SessionOnlyRequest,
        restrictions=(
            "Figures are withheld by the backend until verification passes.",
            "Carries no card number, payment id, or payment-method id.",
        ),
    ),
    ToolSpec(
        name="verify_identity",
        description=(
            "Check the postal code the customer stated against the one on the "
            "account."
        ),
        when_to_use=(
            "Once, near the start of the call, before discussing any account "
            "detail. Retry once if it does not match."
        ),
        response_description=(
            "verified (true/false), attempts_remaining, locked, and next_action."
        ),
        request_model=VerifyIdentityRequest,
        llm_parameters=("postal_code",),
        restrictions=(
            "Two attempts only; the backend locks further guesses.",
            "Never reveal, confirm, or partially read back the expected answer.",
            "Postal code is the only factor. Never ask for a card number, CVV, "
            "password, or one-time code.",
        ),
    ),
    ToolSpec(
        name="retry_payment",
        description="Retry the failed charge through the payment processor.",
        when_to_use=(
            "Only after the customer has clearly agreed to a retry in the "
            "current turn. Use payment_method='backup' when next_action says "
            "offer_backup_method and the customer agrees to the other card."
        ),
        response_description=(
            "status ('paid', 'declined', or 'not_attempted'), transaction_id, "
            "confirmation_number on success, failure_reason on decline, and "
            "next_action. Only 'paid' means the money moved."
        ),
        request_model=RetryPaymentRequest,
        llm_parameters=("payment_method",),
        requires_verification=True,
        restrictions=(
            "Never announce success unless status is exactly 'paid'.",
            "Two attempts per conversation; the backend refuses a third.",
            "Refused on a settled balance and on declines a retry cannot fix.",
        ),
    ),
    ToolSpec(
        name="schedule_retry",
        description="Record a note to retry the charge on a future date.",
        when_to_use=(
            "When the customer asks to be charged later and names a date, such "
            "as their next payday."
        ),
        response_description=(
            "status, scheduled_for, confirmation_number, and next_action."
        ),
        request_model=ScheduleRetryRequest,
        llm_parameters=("requested_date", "requested_action"),
        requires_verification=True,
        restrictions=(
            "Records a note only. No job runs, so never say it will charge "
            "automatically.",
            "The date must be in the future and within 60 days.",
        ),
    ),
    ToolSpec(
        name="send_payment_link",
        description=(
            "Prepare a secure payment link so the customer can enter card "
            "details themselves."
        ),
        when_to_use=(
            "Whenever the customer wants to use a different card, or when a "
            "retry cannot succeed on the card on file."
        ),
        response_description=(
            "link, channel, sent_to_masked, delivered (always false), "
            "delivery_note, and next_action."
        ),
        request_model=SendPaymentLinkRequest,
        llm_parameters=("channel",),
        requires_verification=True,
        restrictions=(
            "Nothing is transmitted: delivered is always false. Say the link "
            "has been put on the account, never that a text or email was sent.",
            "This is the answer to a customer offering a card number by voice.",
        ),
    ),
    ToolSpec(
        name="escalate_to_human",
        description="Raise a ticket for a human colleague to call the customer back.",
        when_to_use=(
            "When the customer asks for a person, disputes the charge, or the "
            "situation cannot be resolved on this call."
        ),
        response_description="status, ticket_id, callback_window, and next_action.",
        request_model=EscalateRequest,
        llm_parameters=("reason", "notes"),
        restrictions=(
            "Deliberately available without verification - a caller who cannot "
            "verify is often the one who most needs a person.",
            "Nobody is contacted in real time. Never say a colleague is joining "
            "the call or that you are transferring them.",
        ),
    ),
    ToolSpec(
        name="log_disposition",
        description="Record how the conversation ended and close the session.",
        when_to_use=(
            "Exactly once, as the last tool call of every conversation, before "
            "saying goodbye."
        ),
        response_description=(
            "disposition, do_not_call, session_closed, and next_action."
        ),
        request_model=LogDispositionRequest,
        llm_parameters=("disposition", "notes"),
        restrictions=(
            "disposition is a closed list; anything else is rejected.",
            "'payment_recovered' is refused unless a retry actually returned "
            "paid.",
            "'do_not_call' is persisted against the customer and outlives the "
            "call.",
            "Closes the session: no further tool call will be accepted.",
        ),
    ),
)


def tool_names() -> tuple[str, ...]:
    """The seven tool names, in call order."""
    return tuple(spec.name for spec in TOOL_SPECS)


def spec_by_name(name: str) -> ToolSpec:
    """Look up one specification, or raise ``KeyError``."""
    for spec in TOOL_SPECS:
        if spec.name == name:
            return spec
    raise KeyError(f"No tool spec named {name!r}")


def as_dicts() -> list[dict[str, Any]]:
    """All specifications in flat form, ready for a provisioning payload."""
    return [spec.as_dict() for spec in TOOL_SPECS]
