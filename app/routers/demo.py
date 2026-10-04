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
from app.agent.prompt import build_dynamic_variables
from app.errors import ToolError
from app.models import (
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
        identity_verified=session.identity_verified,
        verification_attempts=session.verification_attempts,
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


@router.get(
    "/sessions",
    response_model=list[CreateSessionResponse],
    summary="Sessions, newest first, optionally for one customer",
)
async def list_sessions(
    customer_id: str | None = None,
    limit: int = 20,
) -> list[CreateSessionResponse]:
    """List sessions so the dashboard can find one it did not create.

    This exists because the dashboard and the voice page are separate pages.
    A conversation started at ``/voice`` has a session the dashboard has never
    heard of, so without a way to look sessions up the dashboard sits on "No
    session yet" while a call runs. Newest first, because the interesting
    session is almost always the most recent one.

    Read-only, like the rest of the dashboard's view of the world: it creates
    nothing and changes nothing.
    """
    customers = store.load_customers()
    sessions = store.list_sessions()

    if customer_id is not None:
        # Validate rather than silently returning an empty list, so a typo in
        # a query string is distinguishable from "this customer has none".
        _load_customer(customer_id)
        sessions = [s for s in sessions if s.customer_id == customer_id]

    newest_first = sorted(sessions, key=lambda s: s.created_at, reverse=True)
    return [
        CreateSessionResponse(
            session_id=session.session_id,
            customer_id=session.customer_id,
            customer_name=(
                customers[session.customer_id].name
                if session.customer_id in customers
                else session.customer_id
            ),
            status=_status_of(session),
            channel=session.channel,
            created_at=session.created_at,
        )
        for session in newest_first[: max(1, min(limit, 100))]
    ]


# ---------------------------------------------------------------------------
# Voice connection
# ---------------------------------------------------------------------------


@router.get(
    "/voice/connection/{session_id}",
    summary="What the browser needs to open a voice conversation",
)
async def voice_connection(session_id: str) -> dict[str, object]:
    """Hand the page everything it needs to start talking - and no secrets.

    Two things come back. A **signed URL**, minted server-side, which lets the
    browser open one conversation without ever seeing the API key; the page
    falls back to the bare agent id only when a signed URL cannot be obtained
    (an agent with authentication disabled). And the **dynamic variables**,
    built by ``app.agent.prompt``, which carry the session id into the
    conversation so every tool call is bound to this customer.

    The variables deliberately exclude the amount, the decline reason, and the
    card: the agent must fetch those through ``get_failed_payment_details``
    after verification, so they cannot be spoken before identity is confirmed.
    """
    session = _load_session(session_id)
    customer = _load_customer(session.customer_id)

    from app.providers import elevenlabs_client as provider

    details: dict[str, object] = {
        "session_id": session.session_id,
        "customer_id": customer.customer_id,
        "customer_name": customer.name,
        "dynamic_variables": build_dynamic_variables(customer, session.session_id),
        "configured": provider.is_configured(),
        "agent_id": None,
        "signed_url": None,
        "mode": "unavailable",
        "message": "",
    }

    status = provider.status()
    if not status["configured"]:
        details["message"] = (
            "ElevenLabs is not configured. Set ELEVENLABS_API_KEY in .env."
        )
        return details
    if not status["agent_id"]:
        details["message"] = (
            "No agent provisioned yet. Run scripts/provision_agent.py, then set "
            "ELEVENLABS_AGENT_ID in .env."
        )
        return details

    details["agent_id"] = status["agent_id"]
    try:
        details["signed_url"] = provider.signed_url()
        details["mode"] = "signed_url"
    except Exception as error:  # noqa: BLE001 - reported, never raised at the page
        # A public agent needs no signed URL, and a transient API error should
        # not block the demo. Either way the key stays server-side.
        logger.warning("could not mint a signed URL: %s", type(error).__name__)
        details["mode"] = "agent_id"
        details["message"] = (
            "Using the agent id directly; a signed URL was unavailable."
        )
    return details


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
