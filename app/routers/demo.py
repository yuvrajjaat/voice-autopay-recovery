"""Local demo control plane.

These endpoints serve the dashboard at ``/dashboard``. They are a different
audience from ``/tools/*``: a human operator on localhost, rather than the
voice agent's LLM. Three consequences follow.

**No tool secret.** The dashboard is a local inspection surface, not something
reachable through the public tunnel, so it carries no ``X-Tool-Secret``. That
is deliberate rather than lax: it means the shared secret never has to be
embedded in a web page the browser can read. The dashboard observes state and
manages session lifecycle; it does not invoke tools.

**Hand-built response models.** Nothing here serialises ``Customer``
wholesale. The seed's postal code *is* the verification answer, so dumping the
address would hand over the credential by a side door. Every field served is
named explicitly in ``app.models``, and a test asserts no customer's expected
answer appears in any response.

**No business logic.** Session creation, state, events, and reset all live in
``store``; these routes validate input, assemble a view, and return it.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter

from app import store
from app.errors import ToolError
from app.models import (
    AutopayStatus,
    CreateSessionRequest,
    CreateSessionResponse,
    Customer,
    CustomerDetail,
    CustomerSummary,
    EventType,
    ResetSessionResponse,
    Session,
    SessionEventOut,
    SessionStateResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["demo control plane"])


def _status_of(session: Session) -> str:
    return "closed" if session.closed else "active"


def _load_session(session_id: str) -> Session:
    try:
        return store.get_session(session_id)
    except store.SessionNotFoundError:
        raise ToolError(404, "unknown_session", f"No session {session_id!r}.") from None


def _load_customer(customer_id: str) -> Customer:
    try:
        return store.get_customer(customer_id)
    except store.CustomerNotFoundError:
        raise ToolError(
            404, "unknown_customer", f"No customer {customer_id!r}."
        ) from None


def _summary_fields(customer: Customer) -> dict[str, object]:
    """The customer fields that are safe to publish, in one place.

    Shared by the list and detail endpoints so the two cannot drift apart and
    accidentally expose different things.
    """
    state = store.get_state(customer.customer_id)
    payment = customer.failed_payment
    return {
        "customer_id": customer.customer_id,
        "name": customer.name,
        "phone": customer.phone,
        "email": customer.email,
        "plan_name": customer.plan.name,
        "amount": payment.amount_spoken,
        "currency": payment.currency,
        "failure_reason": payment.failure_code,
        "failure_explanation": payment.failure_reason_spoken,
        "days_delinquent": payment.days_delinquent,
        "card_description": customer.autopay.primary_method.spoken,
        "scenario_label": customer.scenario.label,
        "expected_path": customer.scenario.expected_path,
        "caller_intent": customer.scenario.caller_intent,
        "payment_status": state.status,
        "do_not_call": state.do_not_call,
    }


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------


@router.post(
    "/sessions",
    response_model=CreateSessionResponse,
    status_code=201,
    summary="Start a demo session for one of the ten fictional customers",
)
async def create_session(request: CreateSessionRequest) -> CreateSessionResponse:
    """Open a session, binding it to the chosen customer for its lifetime.

    The customer id is validated against the seed, so an id that is not one of
    the ten is a 404 rather than a session pointing at nothing.
    """
    customer = _load_customer(request.customer_id)
    session = store.create_session(customer.customer_id, channel=request.channel)

    store.record_event(
        session.session_id,
        customer.customer_id,
        EventType.SESSION_CREATED,
        f"Demo session opened for {customer.name} ({customer.scenario.label}).",
        {"channel": session.channel, "expected_path": customer.scenario.expected_path},
    )
    logger.info("demo session %s created for %s", session.session_id, customer.customer_id)

    return CreateSessionResponse(
        session_id=session.session_id,
        customer_id=customer.customer_id,
        customer_name=customer.name,
        status=_status_of(session),
        channel=session.channel,
        created_at=session.created_at,
    )


@router.get(
    "/sessions/{session_id}",
    response_model=SessionStateResponse,
    summary="Inspect everything the dashboard shows about a session",
)
async def get_session_state(session_id: str) -> SessionStateResponse:
    """Assemble the session's current state from the store.

    Derived rather than stored: ``payment_link_prepared``, ``escalated``, and
    ``escalation_ticket`` come from the audit collections, so the view cannot
    disagree with the records behind it.
    """
    session = _load_session(session_id)
    customer = _load_customer(session.customer_id)
    state = store.get_state(session.customer_id)

    escalations = store.get_escalations(session.customer_id)
    links = store.get_payment_links(session.customer_id)

    return SessionStateResponse(
        session_id=session.session_id,
        customer_id=customer.customer_id,
        customer_name=customer.name,
        status=_status_of(session),
        channel=session.channel,
        identity_verified=state.identity_verified,
        verification_attempts=state.verification_attempts,
        verification_attempts_allowed=customer.verification.max_attempts,
        payment_status=state.status,
        amount_due=customer.failed_payment.amount_spoken,
        currency=customer.failed_payment.currency,
        retry_attempts=state.retry_attempts,
        last_confirmation_number=state.last_confirmation_number,
        scheduled_for=state.scheduled_for,
        payment_link_prepared=bool(links),
        escalated=bool(escalations),
        escalation_ticket=escalations[-1]["ticket_id"] if escalations else None,
        disposition=state.disposition,
        disposition_notes=state.disposition_notes,
        do_not_call=state.do_not_call,
        tool_calls=session.tool_calls,
        event_count=len(store.get_events(session_id)),
        created_at=session.created_at,
        last_tool_at=session.last_tool_at,
    )


@router.get(
    "/sessions/{session_id}/events",
    response_model=list[SessionEventOut],
    summary="The session's timeline, oldest first",
)
async def get_session_events(session_id: str) -> list[SessionEventOut]:
    """Return the chronological narrative of the call.

    Ordering comes from the monotonic ``sequence`` assigned at write time, not
    from timestamps, so events written inside the same millisecond still read
    in the order they happened.
    """
    _load_session(session_id)
    # Project rather than validate the stored record wholesale: the stored
    # event also carries session_id and customer_id, which the caller already
    # knows, and SessionEventOut forbids extras so that the view stays an
    # explicit allow-list rather than a passthrough.
    return [
        SessionEventOut(
            sequence=event["sequence"],
            event_type=event["event_type"],
            summary=event["summary"],
            detail=event.get("detail") or {},
            created_at=event["created_at"],
        )
        for event in store.get_events(session_id)
    ]


@router.post(
    "/sessions/{session_id}/reset",
    response_model=ResetSessionResponse,
    summary="Clear this session's state so the scenario can be run again",
)
async def reset_session(session_id: str) -> ResetSessionResponse:
    """Reset one session without touching anything else.

    Clears the audit records tagged with this session, the customer's ledger
    row, and the session's own counters, then reopens the session under the
    same id so the dashboard can immediately run the scenario again.

    ``data/customers.json`` is never written. The seed is what a reset
    restores *to*, which is the whole reason the project keeps mutable state
    in a separate file.
    """
    session = _load_session(session_id)
    reopened, cleared = store.reset_session(session_id)

    return ResetSessionResponse(
        session_id=reopened.session_id,
        customer_id=reopened.customer_id,
        status=_status_of(reopened),
        message=(
            f"Session reset. {cleared} runtime record(s) cleared; "
            "the seed data is untouched."
        ),
        records_cleared=cleared,
    )


# ---------------------------------------------------------------------------
# Customers
# ---------------------------------------------------------------------------


@router.get(
    "/customers",
    response_model=list[CustomerSummary],
    summary="The ten fictional customers, with their current ledger status",
)
async def list_customers() -> list[CustomerSummary]:
    """The selection list for the dashboard."""
    return [
        CustomerSummary.model_validate(_summary_fields(customer))
        for customer in store.list_customers()
    ]


@router.get(
    "/customers/{customer_id}",
    response_model=CustomerDetail,
    summary="One fictional customer, with the extra demo context",
)
async def get_customer(customer_id: str) -> CustomerDetail:
    """Detail for the dashboard's customer panel.

    ``verification_method`` names the factor ("postal_code") without the
    value. The operator reads the actual postal codes from
    ``data/customers.json`` in their own checkout; they are deliberately not
    served over HTTP, because that value is the credential the agent checks.
    """
    customer = _load_customer(customer_id)
    payment = customer.failed_payment

    return CustomerDetail.model_validate(
        {
            **_summary_fields(customer),
            "city": customer.address.city,
            "state": customer.address.state,
            "timezone": customer.timezone,
            "billing_period": customer.plan.billing_period,
            "due_date": payment.due_date,
            "failed_at": payment.failed_at,
            "attempts_before_this_call": payment.attempts_so_far,
            "service_suspension_date": payment.service_suspension_date,
            "backup_method_available": customer.autopay.backup_method is not None,
            "verification_method": customer.verification.method,
            "verification_attempts_allowed": customer.verification.max_attempts,
        }
    )


@router.get(
    "/sessions",
    response_model=list[CreateSessionResponse],
    summary="Every session in the current runtime state",
)
async def list_sessions() -> list[CreateSessionResponse]:
    """Handy when a dashboard reload has lost track of the session id."""
    customers = store.load_customers()
    return [
        CreateSessionResponse(
            session_id=session.session_id,
            customer_id=session.customer_id,
            customer_name=customers[session.customer_id].name
            if session.customer_id in customers
            else session.customer_id,
            status=_status_of(session),
            channel=session.channel,
            created_at=session.created_at,
        )
        for session in sorted(store.list_sessions(), key=lambda s: s.created_at)
    ]


@router.get("/state", summary="The whole ledger, for the dashboard overview")
async def ledger() -> dict[str, object]:
    """One call the dashboard can poll for the ten-row status table."""
    states = store.all_states()
    counts: dict[str, int] = {}
    for state in states.values():
        counts[state.status.value] = counts.get(state.status.value, 0) + 1

    return {
        "customers": len(states),
        "status_counts": counts,
        "recovered": counts.get(AutopayStatus.RECOVERED.value, 0),
        "sessions": len(store.list_sessions()),
        "payment_attempts": len(store.get_payment_attempts()),
        "scheduled_retries": len(store.get_scheduled_retries()),
        "payment_links_prepared": len(store.get_payment_links()),
        "escalations": len(store.get_escalations()),
        "events": len(store.get_events()),
    }
