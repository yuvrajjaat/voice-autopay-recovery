"""The seven tools the voice agent calls.

Each route does four things and nothing more: validate the request (Pydantic),
authenticate it (the router-wide dependency), resolve and gate the session,
then delegate to ``store`` / ``mock_processor`` and format the result for
speech. Payment behaviour lives in ``mock_processor``; persistence lives in
``store``; neither is reimplemented here.

Guard rules enforced in code rather than left to the prompt
-----------------------------------------------------------
* Amounts are withheld until identity is verified.
* No charge, schedule, or payment link without verification.
* No retry on a balance already settled.
* No retry past two attempts in one conversation.
* No retry on a decline code where a retry mathematically cannot succeed.
* No tool calls at all once the conversation has been dispositioned.

A prompt can be argued with. These cannot.
"""

from __future__ import annotations

import logging
import secrets
from datetime import date, timedelta

from fastapi import APIRouter, Depends

from app import store
from app.errors import ToolError
from app.models import (
    AutopayStatus,
    Customer,
    CustomerState,
    Disposition,
    DispositionResponse,
    EscalateRequest,
    EscalationResponse,
    EventType,
    FailedPaymentDetailsResponse,
    FailureCode,
    LogDispositionRequest,
    NextAction,
    RetryPaymentRequest,
    RetryPaymentResponse,
    ScheduleRetryRequest,
    ScheduleRetryResponse,
    SendPaymentLinkRequest,
    SendPaymentLinkResponse,
    Session,
    SessionOnlyRequest,
    ToolErrorResponse,
    VerifyIdentityRequest,
    VerifyIdentityResponse,
)
from app.payments import mock_processor
from app.payments.mock_processor import PaymentMethodUnavailableError
from app.security import require_tool_secret

logger = logging.getLogger(__name__)

#: Retries allowed within a single conversation. The plan's guardrail: two.
MAX_RETRIES_PER_SESSION = 2

#: How far ahead a retry may be scheduled.
MAX_SCHEDULE_DAYS_AHEAD = 60

#: Simulated authorization delay, so the agent's pause sounds like a real
#: card network round-trip instead of an instant answer.
RETRY_LATENCY_SECONDS = 0.8

CALLBACK_WINDOW = "within one business day"

#: Which decline codes cannot be fixed by trying the same card again, and
#: what the agent should do instead.
_TERMINAL_NEXT_ACTION: dict[FailureCode, NextAction] = {
    FailureCode.EXPIRED_CARD: NextAction.PAYMENT_LINK,
    FailureCode.INCORRECT_CVC: NextAction.PAYMENT_LINK,
    FailureCode.THREE_DS_REQUIRED: NextAction.PAYMENT_LINK,
    FailureCode.CLOSED_ACCOUNT: NextAction.ESCALATE,
}

router = APIRouter(
    prefix="/tools",
    tags=["agent tools"],
    dependencies=[Depends(require_tool_secret)],
    responses={
        401: {"model": ToolErrorResponse, "description": "Missing or wrong tool secret"},
        403: {"model": ToolErrorResponse, "description": "Precondition not met"},
        404: {"model": ToolErrorResponse, "description": "Unknown session"},
        409: {"model": ToolErrorResponse, "description": "Conflicts with current state"},
        503: {"model": ToolErrorResponse, "description": "Tool auth not configured"},
    },
)


# ---------------------------------------------------------------------------
# Shared gates
# ---------------------------------------------------------------------------


def _resolve(session_id: str) -> tuple[Session, Customer, CustomerState]:
    """Resolve a session to its customer, or raise a speakable ToolError.

    This is the only path from a request to a customer. Because the request
    schema carries no customer field, the session's binding is the single
    source of truth for whose account is in play.
    """
    try:
        session = store.get_session(session_id)
    except store.SessionNotFoundError:
        raise ToolError(
            404,
            "unknown_session",
            "I've lost track of this call. Let me hand you to a colleague.",
        ) from None

    if session.closed:
        raise ToolError(
            409,
            "session_closed",
            "This call has already been wrapped up.",
        )

    try:
        customer = store.get_customer(session.customer_id)
    except store.CustomerNotFoundError:
        # Only reachable if the seed changed under a live session.
        raise ToolError(
            404,
            "unknown_customer",
            "I can't find that account. Let me hand you to a colleague.",
        ) from None

    return session, customer, store.get_state(session.customer_id)


def _require_verified(state: CustomerState) -> None:
    """Block anything that moves money or discloses figures."""
    if not state.identity_verified:
        raise ToolError(
            403,
            "identity_not_verified",
            "I need to confirm a couple of details before I can go into the account.",
        )


def _require_not_settled(state: CustomerState) -> None:
    if state.status is AutopayStatus.RECOVERED:
        raise ToolError(
            409,
            "already_paid",
            "That balance is already settled, so there's nothing left to pay.",
        )


def _mask_email(email: str) -> str:
    """Mask to a fixed width, so the mask doesn't leak the name's length."""
    local, _, domain = email.partition("@")
    visible = local[:1] if local else ""
    return f"{visible}***@{domain}"


def _mask_phone(phone: str) -> str:
    digits = "".join(character for character in phone if character.isdigit())
    if len(digits) <= 4:
        return "*" * len(digits)
    return f"+{'*' * (len(digits) - 4)}{digits[-4:]}"


# ---------------------------------------------------------------------------
# 1. get_failed_payment_details
# ---------------------------------------------------------------------------


@router.post(
    "/get_failed_payment_details",
    response_model=FailedPaymentDetailsResponse,
    summary="Look up the failed autopay charge for this call",
)
async def get_failed_payment_details(
    request: SessionOnlyRequest,
) -> FailedPaymentDetailsResponse:
    """Return the facts the agent needs to explain the failed payment.

    Called before verification, this returns ``verified: false`` with every
    figure omitted and ``next_action: verify_identity``. That is a soft answer
    rather than an error because asking in the wrong order is a normal
    conversational slip the agent should recover from by verifying — unlike
    attempting a charge unverified, which is a hard precondition failure.
    """
    session, customer, state = _resolve(request.session_id)
    store.touch_session(session.session_id)

    if not state.identity_verified:
        store.record_event(
            session.session_id,
            customer.customer_id,
            EventType.PAYMENT_DETAILS_WITHHELD,
            "Payment details requested before verification - figures withheld.",
        )
        return FailedPaymentDetailsResponse(
            success=True,
            verified=False,
            customer_id=customer.customer_id,
            customer_name=customer.name,
            status="withheld_pending_verification",
            message=(
                "I can't go into the account details until I've confirmed "
                "who I'm speaking with."
            ),
            next_action=NextAction.VERIFY_IDENTITY,
        )

    payment = customer.failed_payment
    status = mock_processor.check_payment_status(customer)
    worth_retrying = bool(status["retry_worth_attempting"])

    if worth_retrying:
        next_action = NextAction.RETRY_PAYMENT
    else:
        next_action = _TERMINAL_NEXT_ACTION.get(
            payment.failure_code, NextAction.PAYMENT_LINK_OR_ESCALATION
        )

    store.record_event(
        session.session_id,
        customer.customer_id,
        EventType.PAYMENT_DETAILS_VIEWED,
        f"Explained the failed {payment.amount_spoken} {payment.currency} autopay.",
        {
            "amount": payment.amount_spoken,
            "failure_reason": payment.failure_code.value,
            "retry_worth_attempting": str(worth_retrying).lower(),
        },
    )
    return FailedPaymentDetailsResponse(
        success=True,
        verified=True,
        customer_id=customer.customer_id,
        customer_name=customer.name,
        status=str(status["status"]),
        amount=payment.amount_spoken,
        currency=payment.currency,
        due_date=payment.due_date.isoformat(),
        days_delinquent=payment.days_delinquent,
        failure_reason=payment.failure_code.value,
        failure_explanation=payment.failure_reason_spoken,
        card_description=customer.autopay.primary_method.spoken,
        retry_worth_attempting=worth_retrying,
        backup_method_available=customer.autopay.backup_method is not None,
        service_suspension_date=(
            payment.service_suspension_date.isoformat()
            if payment.service_suspension_date
            else None
        ),
        message=(
            f"The autopay of {payment.amount_spoken} {payment.currency} due "
            f"{payment.due_date.isoformat()} didn't go through — "
            f"{payment.failure_reason_spoken}."
        ),
        next_action=next_action,
    )


# ---------------------------------------------------------------------------
# 2. verify_identity
# ---------------------------------------------------------------------------


@router.post(
    "/verify_identity",
    response_model=VerifyIdentityResponse,
    summary="Check the postal code the caller stated against the account",
)
async def verify_identity(request: VerifyIdentityRequest) -> VerifyIdentityResponse:
    """Confirm the caller is the account holder.

    Postal code is the only factor. Comparison ignores case and whitespace so
    a transcription like "9 4 1 0 7" still matches. Attempts are capped by the
    customer's ``max_attempts``; once spent, the tool stays locked for the
    rest of the conversation and no further guesses are counted.
    """
    session, customer, state = _resolve(request.session_id)
    store.touch_session(session.session_id)

    limit = customer.verification.max_attempts

    # Already verified: idempotent, and does not burn an attempt.
    if state.identity_verified:
        return VerifyIdentityResponse(
            success=True,
            verified=True,
            attempts_remaining=max(limit - state.verification_attempts, 0),
            message="Thanks, the account is already confirmed.",
            next_action=NextAction.RETRY_PAYMENT,
        )

    if state.verification_attempts >= limit:
        store.record_event(
            session.session_id,
            customer.customer_id,
            EventType.IDENTITY_VERIFICATION_LOCKED,
            "Further verification attempt refused - the limit was already reached.",
        )
        return VerifyIdentityResponse(
            success=True,
            verified=False,
            attempts_remaining=0,
            locked=True,
            message=(
                "I'm not able to confirm the account, so I can't discuss it on "
                "this call. Please use the number on your statement."
            ),
            next_action=NextAction.ESCALATE,
        )

    given = "".join(request.postal_code.split()).upper()
    expected = "".join(customer.verification.expected_answer.split()).upper()
    attempts_used = state.verification_attempts + 1
    matched = secrets.compare_digest(given, expected)

    store.update_state(
        customer.customer_id,
        identity_verified=matched,
        verification_attempts=attempts_used,
    )
    remaining = max(limit - attempts_used, 0)

    if matched:
        store.record_event(
            session.session_id,
            customer.customer_id,
            EventType.IDENTITY_VERIFIED,
            f"Identity verified on attempt {attempts_used}.",
            {"attempts_used": str(attempts_used)},
        )
        return VerifyIdentityResponse(
            success=True,
            verified=True,
            attempts_remaining=remaining,
            message="Thank you, that matches.",
            next_action=NextAction.RETRY_PAYMENT,
        )

    if remaining == 0:
        store.record_event(
            session.session_id,
            customer.customer_id,
            EventType.IDENTITY_VERIFICATION_LOCKED,
            "Verification failed on the final attempt - locked for this call.",
            {"attempts_used": str(attempts_used)},
        )
        return VerifyIdentityResponse(
            success=True,
            verified=False,
            attempts_remaining=0,
            locked=True,
            message=(
                "That still doesn't match what I have. I can't discuss the "
                "account on this call — please use the number on your statement."
            ),
            next_action=NextAction.ESCALATE,
        )

    store.record_event(
        session.session_id,
        customer.customer_id,
        EventType.IDENTITY_VERIFICATION_FAILED,
        f"Verification attempt {attempts_used} did not match.",
        {"attempts_remaining": str(remaining)},
    )
    return VerifyIdentityResponse(
        success=True,
        verified=False,
        attempts_remaining=remaining,
        message="That doesn't match what I have on file. Could you try once more?",
        next_action=NextAction.VERIFY_IDENTITY,
    )


# ---------------------------------------------------------------------------
# 3. retry_payment
# ---------------------------------------------------------------------------


@router.post(
    "/retry_payment",
    response_model=RetryPaymentResponse,
    summary="Retry the failed charge through the mock processor",
)
async def retry_payment(request: RetryPaymentRequest) -> RetryPaymentResponse:
    """Attempt the payment again. Mock processor only — no real money moves.

    Refuses, with a speakable reason, when: identity is unverified, the
    balance is already settled, two attempts have already been made in this
    conversation, the requested method isn't on file, or the decline code
    makes a retry pointless.
    """
    session, customer, state = _resolve(request.session_id)
    _require_verified(state)
    _require_not_settled(state)

    if session.retry_attempts >= MAX_RETRIES_PER_SESSION:
        raise ToolError(
            409,
            "retry_limit_reached",
            (
                "I've already tried that twice on this call, so I won't keep "
                "attempting it. Let's find another way."
            ),
        )

    # Refuse a retry that cannot possibly succeed, rather than spending an
    # attempt to prove it. Only applies to the card that already failed.
    failure_code = customer.failed_payment.failure_code
    if request.payment_method == "primary" and not mock_processor.is_retry_worth_attempting(
        failure_code
    ):
        next_action = _TERMINAL_NEXT_ACTION.get(
            failure_code, NextAction.PAYMENT_LINK_OR_ESCALATION
        )
        if customer.autopay.backup_method is not None:
            next_action = NextAction.OFFER_BACKUP_METHOD
        store.record_event(
            session.session_id,
            customer.customer_id,
            EventType.PAYMENT_RETRY_SKIPPED,
            f"Retry not attempted - {failure_code.value} is not recoverable on that card.",
            {"failure_reason": failure_code.value, "next_action": next_action.value},
        )
        return RetryPaymentResponse(
            success=False,
            status="not_attempted",
            failure_reason=failure_code.value,
            message=(
                f"There's no point retrying that card — "
                f"{customer.failed_payment.failure_reason_spoken}."
            ),
            next_action=next_action,
        )

    attempt_number = store.next_attempt_number(customer.customer_id)
    try:
        result = mock_processor.retry_payment(
            customer,
            payment_method=request.payment_method,
            attempt_number=attempt_number,
            latency_seconds=RETRY_LATENCY_SECONDS,
        )
    except PaymentMethodUnavailableError:
        raise ToolError(
            400,
            "payment_method_unavailable",
            "There isn't a second card on the account to try.",
        ) from None

    store.record_payment_attempt(
        customer.customer_id,
        result,
        attempt_number=attempt_number,
        session_id=session.session_id,
    )
    store.touch_session(session.session_id, retry=True)
    store.record_event(
        session.session_id,
        customer.customer_id,
        EventType.PAYMENT_RETRY_ATTEMPTED,
        f"Retry {attempt_number} sent to the mock processor "
        f"on the {request.payment_method} card.",
        {"attempt": str(attempt_number), "payment_method": request.payment_method},
    )

    if result.success:
        store.record_event(
            session.session_id,
            customer.customer_id,
            EventType.PAYMENT_RETRY_SUCCEEDED,
            f"Payment approved: {result.amount:.2f} {result.currency}.",
            {
                "amount": f"{result.amount:.2f}",
                "confirmation_number": result.confirmation_number or "",
                "transaction_id": result.transaction_id,
            },
        )
        return RetryPaymentResponse(
            success=True,
            status="paid",
            transaction_id=result.transaction_id,
            confirmation_number=result.confirmation_number,
            amount=f"{result.amount:.2f}",
            currency=result.currency,
            message=(
                f"That went through. {result.amount:.2f} {result.currency} is paid, "
                f"and your confirmation number is {result.confirmation_number}."
            ),
            next_action=NextAction.NONE,
        )

    # Declined: pick the most useful next step for this customer.
    assert result.failure_code is not None
    backup_untried = (
        customer.autopay.backup_method is not None and request.payment_method == "primary"
    )
    if backup_untried:
        next_action = NextAction.OFFER_BACKUP_METHOD
    else:
        next_action = _TERMINAL_NEXT_ACTION.get(
            result.failure_code, NextAction.PAYMENT_LINK_OR_ESCALATION
        )

    store.record_event(
        session.session_id,
        customer.customer_id,
        EventType.PAYMENT_RETRY_DECLINED,
        f"Payment declined: {result.failure_code.value}.",
        {"failure_reason": result.failure_code.value, "next_action": next_action.value},
    )
    return RetryPaymentResponse(
        success=False,
        status="declined",
        transaction_id=result.transaction_id,
        amount=f"{result.amount:.2f}",
        currency=result.currency,
        failure_reason=result.failure_code.value,
        message=result.message,
        next_action=next_action,
    )


# ---------------------------------------------------------------------------
# 4. schedule_retry
# ---------------------------------------------------------------------------


@router.post(
    "/schedule_retry",
    response_model=ScheduleRetryResponse,
    summary="Record a retry for a future date (no job is created)",
)
async def schedule_retry(request: ScheduleRetryRequest) -> ScheduleRetryResponse:
    """Write down that the customer wants the charge retried later.

    A record only. Nothing in this project runs on a schedule — there is no
    worker, queue, or cron — so the agent must describe this as a note on the
    account, which is exactly what it is.
    """
    session, customer, state = _resolve(request.session_id)
    _require_verified(state)
    _require_not_settled(state)

    today = date.today()
    if request.requested_date <= today:
        raise ToolError(
            400,
            "invalid_schedule",
            "That date has already passed. What date would suit you?",
        )
    if request.requested_date > today + timedelta(days=MAX_SCHEDULE_DAYS_AHEAD):
        raise ToolError(
            400,
            "invalid_schedule",
            (
                f"I can only schedule up to {MAX_SCHEDULE_DAYS_AHEAD} days out. "
                "Could you pick a sooner date?"
            ),
        )

    confirmation_number = mock_processor.build_confirmation_number(
        customer.customer_id,
        request.requested_date.toordinal(),
        request.requested_action,
    )
    retry = store.record_scheduled_retry(
        customer.customer_id,
        request.requested_date,
        confirmation_number,
        session_id=session.session_id,
        requested_action=request.requested_action,
    )
    store.touch_session(session.session_id)
    store.record_event(
        session.session_id,
        customer.customer_id,
        EventType.PAYMENT_SCHEDULED,
        f"Retry noted for {retry.scheduled_for.isoformat()} (record only, no job).",
        {
            "scheduled_for": retry.scheduled_for.isoformat(),
            "requested_action": retry.requested_action,
            "confirmation_number": confirmation_number,
        },
    )

    return ScheduleRetryResponse(
        success=True,
        status=retry.status,
        scheduled_for=retry.scheduled_for.isoformat(),
        confirmation_number=confirmation_number,
        message=(
            f"I've noted the retry for {retry.scheduled_for.isoformat()}. "
            f"Your reference is {confirmation_number}."
        ),
        next_action=NextAction.NONE,
    )


# ---------------------------------------------------------------------------
# 5. send_payment_link
# ---------------------------------------------------------------------------


@router.post(
    "/send_payment_link",
    response_model=SendPaymentLinkResponse,
    summary="Prepare a mock payment link (nothing is transmitted)",
)
async def send_payment_link(
    request: SendPaymentLinkRequest,
) -> SendPaymentLinkResponse:
    """Prepare a payment link instead of taking card details by voice.

    No SMS, email, or HTTP request is made — the link is recorded with
    ``delivered: false`` and points at ``example.test``, a reserved TLD that
    can never resolve. This tool exists so the agent has somewhere to go when
    a customer offers a new card number, which it must always refuse.
    """
    session, customer, state = _resolve(request.session_id)
    _require_verified(state)
    _require_not_settled(state)

    link_id, url = mock_processor.build_payment_link(customer.customer_id, request.channel)
    masked = (
        _mask_phone(customer.phone)
        if request.channel == "sms"
        else _mask_email(customer.email)
    )

    link = store.record_payment_link(
        customer.customer_id,
        request.channel,
        masked,
        link_id,
        url=url,
        session_id=session.session_id,
    )
    store.touch_session(session.session_id)

    store.record_event(
        session.session_id,
        customer.customer_id,
        EventType.PAYMENT_LINK_PREPARED,
        f"Payment link prepared for {link.sent_to_masked} (not sent).",
        {"channel": link.channel, "delivered": "false", "link_id": link.link_id},
    )

    destination = "number" if request.channel == "sms" else "email address"
    return SendPaymentLinkResponse(
        success=True,
        link=link.url,
        channel=link.channel,
        sent_to_masked=link.sent_to_masked,
        delivered=False,
        delivery_note=link.note,
        message=(
            f"I've prepared a secure payment link for the {destination} on your "
            "account, so you can enter the card details yourself rather than "
            "reading them to me."
        ),
        next_action=NextAction.NONE,
    )


# ---------------------------------------------------------------------------
# 6. escalate_to_human
# ---------------------------------------------------------------------------


@router.post(
    "/escalate_to_human",
    response_model=EscalationResponse,
    summary="Record a handoff to a human (nobody is contacted)",
)
async def escalate_to_human(request: EscalateRequest) -> EscalationResponse:
    """Raise a ticket for a person to pick up.

    Deliberately available without verification: someone who cannot confirm
    their postal code, or who disputes the charge entirely, is precisely the
    caller who most needs a human. Nothing is disclosed by escalating, so
    there is nothing to gate.
    """
    session, customer, _ = _resolve(request.session_id)

    ticket_id = mock_processor.build_ticket_id(customer.customer_id, request.reason)
    escalation = store.record_escalation(
        customer.customer_id,
        ticket_id,
        request.reason,
        CALLBACK_WINDOW,
        session_id=session.session_id,
        notes=request.notes,
    )
    store.touch_session(session.session_id)

    store.record_event(
        session.session_id,
        customer.customer_id,
        EventType.HUMAN_ESCALATION_CREATED,
        f"Escalated to a human: {request.reason}",
        {"ticket_id": ticket_id, "callback_window": CALLBACK_WINDOW},
    )
    return EscalationResponse(
        success=True,
        status=escalation.status,
        ticket_id=ticket_id,
        callback_window=CALLBACK_WINDOW,
        message=(
            f"I've passed this to a colleague — your reference is {ticket_id}, "
            f"and someone will be in touch {CALLBACK_WINDOW}."
        ),
        next_action=NextAction.NONE,
    )


# ---------------------------------------------------------------------------
# 7. log_disposition
# ---------------------------------------------------------------------------


@router.post(
    "/log_disposition",
    response_model=DispositionResponse,
    summary="Record the outcome and close the conversation",
)
async def log_disposition(request: LogDispositionRequest) -> DispositionResponse:
    """Record how the call ended, then close the session.

    The last tool of every conversation. ``disposition`` is a closed enum, so
    an invented outcome label is a 422 rather than an unreportable string.

    ``do_not_call`` is persisted against the customer, not just the session,
    because the request has to outlive the call it was made on. Nothing places
    outbound calls yet, but when Phase 9 does, this is the flag it checks.
    """
    session, customer, current = _resolve(request.session_id)

    # A recovery claim has to be backed by an actual approved payment.
    # Without this, a confused agent could close a declined call as
    # "payment_recovered" and the demo's recovery-rate table would overstate
    # itself — the one number this whole project exists to report.
    if (
        request.disposition is Disposition.PAYMENT_RECOVERED
        and current.status is not AutopayStatus.RECOVERED
    ):
        raise ToolError(
            409,
            "disposition_conflicts_with_state",
            "No payment has gone through on this account, so I can't close it as recovered.",
        )

    state = store.record_disposition(
        customer.customer_id,
        request.disposition,
        request.notes,
        session_id=session.session_id,
    )
    store.touch_session(session.session_id)
    store.record_event(
        session.session_id,
        customer.customer_id,
        EventType.DISPOSITION_LOGGED,
        f"Call closed as {request.disposition.value}.",
        {
            "disposition": request.disposition.value,
            "do_not_call": str(state.do_not_call).lower(),
        },
    )
    store.close_session(session.session_id)

    if request.disposition is Disposition.DO_NOT_CALL:
        message = "Understood — I've recorded that, and we won't call you again."
    else:
        message = "I've recorded the outcome of this call. Thanks for your time."

    logger.info(
        "session %s closed for %s: %s",
        session.session_id,
        customer.customer_id,
        request.disposition.value,
    )
    return DispositionResponse(
        success=True,
        disposition=request.disposition,
        do_not_call=state.do_not_call,
        session_closed=True,
        message=message,
        next_action=NextAction.NONE,
    )
