"""The single gate every outbound call must pass.

The assignment allows calling only a number you control or have explicit
permission to call. This module is how that rule is enforced in code rather
than remembered by a human.

Intended architecture
---------------------
::

    call request  ->  dial_safety  ->  authorized?  ->  provider

There is deliberately **no** second copy of these checks inside a provider
adapter. Any future telephony code must call :func:`assert_dial_allowed`
before it is allowed to touch an API, so adding a provider cannot accidentally
add a bypass. If you are reading this while writing that adapter: call the
guard first, and do not re-implement any part of it.

What it refuses
---------------
Everything except one number. In particular a **customer's stored phone number
is not a callable destination**. The ten seed records hold fictional
``+1-555-01xx`` numbers, and nothing here will dial them: the only acceptable
destination is the one in ``DEMO_PHONE_NUMBER``, which the operator sets by
hand in ``.env``. A destination that arrives from a customer record, a model's
output, or a request body is checked against that value and nothing else.

Layers, in order
----------------
1. ``ENABLE_OUTBOUND_CALLS`` must be explicitly true. A fresh checkout is
   false, so a clone cannot dial at all.
2. The session, if given, must exist, be open, and belong to the customer.
3. The customer must not have opted out.
4. ``DEMO_PHONE_NUMBER`` must be configured and valid E.164.
5. The requested destination must be valid E.164.
6. It must equal the demo number exactly, after normalising both.

The opt-out check sits above the destination checks on purpose: a do-not-call
request is the one rejection that must hold no matter what number is asked
for, including the authorised one.
"""

from __future__ import annotations

import logging
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict

from app import store
from app.config import normalise_e164, settings
from app.models import EventType, utc_now

logger = logging.getLogger(__name__)


class DialOutcome(str, Enum):
    """The closed set of reasons a dial decision can carry.

    Constrained rather than free text so the telephony layer, the audit log,
    and the tests all agree on what happened.
    """

    # The single success value.
    AUTHORIZED_DEMO_NUMBER = "authorized_demo_number"

    # Rejections.
    OUTBOUND_CALLS_DISABLED = "outbound_calls_disabled"
    DEMO_NUMBER_NOT_CONFIGURED = "demo_number_not_configured"
    INVALID_DESTINATION = "invalid_destination"
    DESTINATION_NOT_AUTHORIZED = "destination_not_authorized"
    CUSTOMER_DO_NOT_CALL = "customer_do_not_call"
    INVALID_SESSION = "invalid_session"
    SESSION_CLOSED = "session_closed"
    CUSTOMER_MISMATCH = "customer_mismatch"
    UNKNOWN_CUSTOMER = "unknown_customer"


class DialDecision(BaseModel):
    """The structured answer a telephony layer consumes.

    ``destination`` is populated **only** when the call is authorised. A
    rejected decision carries ``None``, so a provider that ignores ``allowed``
    and reaches straight for the destination still has nothing to dial.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    allowed: bool
    destination: str | None
    reason: DialOutcome
    customer_id: str | None = None
    session_id: str | None = None
    checked_at: datetime

    def raise_if_rejected(self) -> DialDecision:
        """Return self when allowed, else raise :class:`DialBlocked`."""
        if not self.allowed:
            raise DialBlocked(self)
        return self


class DialBlocked(RuntimeError):
    """Raised when a dial request is refused.

    Carries the decision so a caller can report the reason without having to
    re-run the checks.
    """

    def __init__(self, decision: DialDecision) -> None:
        super().__init__(f"outbound call refused: {decision.reason.value}")
        self.decision = decision


def mask_number(value: str | None) -> str:
    """A loggable stand-in for a phone number.

    Keeps the last two digits and the length, which is enough to debug a
    mismatch, and drops everything that would identify the subscriber. Applied
    to authorised numbers too: the decision object carries the real value for
    the provider, the log never does.
    """
    if not value:
        return "<none>"
    digits = "".join(character for character in str(value) if character.isdigit())
    if len(digits) < 3:
        return f"<{len(digits)} digits>"
    return f"+{'*' * (len(digits) - 2)}{digits[-2:]} ({len(digits)} digits)"


def _decision(
    reason: DialOutcome,
    *,
    destination: str | None = None,
    customer_id: str | None = None,
    session_id: str | None = None,
) -> DialDecision:
    allowed = reason is DialOutcome.AUTHORIZED_DEMO_NUMBER
    return DialDecision(
        allowed=allowed,
        destination=destination if allowed else None,
        reason=reason,
        customer_id=customer_id,
        session_id=session_id,
        checked_at=utc_now(),
    )


def check_dial_allowed(
    destination: str | None = None,
    customer_id: str | None = None,
    session_id: str | None = None,
) -> DialDecision:
    """Decide whether one outbound call may be placed.

    Places no call and contacts nothing. It reads configuration and runtime
    state, writes an audit line, and returns a decision.

    Args:
        destination: the number the caller wants to dial, in any format. Must
            normalise to the configured demo number.
        customer_id: the account the call concerns, if known.
        session_id: the conversation the call belongs to, if one exists. When
            given, it is the authority on which customer is in play.

    Returns:
        A :class:`DialDecision`. ``allowed`` is true only when every layer
        passes, and ``destination`` is populated only in that case.
    """
    resolved_customer = customer_id

    # 1. The feature flag. Checked first so a fresh checkout gets the clearest
    #    possible answer rather than a complaint about a missing number.
    if not settings.enable_outbound_calls:
        return _record(
            _decision(
                DialOutcome.OUTBOUND_CALLS_DISABLED,
                customer_id=resolved_customer,
                session_id=session_id,
            ),
            destination,
        )

    # 2. Session binding. The session decides whose account this is; a caller
    #    cannot redirect an existing session at a different customer.
    if session_id is not None:
        try:
            session = store.get_session(session_id)
        except store.SessionNotFoundError:
            return _record(
                _decision(
                    DialOutcome.INVALID_SESSION,
                    customer_id=resolved_customer,
                    session_id=session_id,
                ),
                destination,
            )

        if session.closed:
            return _record(
                _decision(
                    DialOutcome.SESSION_CLOSED,
                    customer_id=session.customer_id,
                    session_id=session_id,
                ),
                destination,
            )

        if customer_id is not None and customer_id != session.customer_id:
            return _record(
                _decision(
                    DialOutcome.CUSTOMER_MISMATCH,
                    customer_id=session.customer_id,
                    session_id=session_id,
                ),
                destination,
            )

        resolved_customer = session.customer_id

    # 3. The opt-out. Above the destination checks so that asking for the
    #    authorised number cannot get around it.
    if resolved_customer is not None:
        try:
            state = store.get_state(resolved_customer)
        except store.CustomerNotFoundError:
            return _record(
                _decision(
                    DialOutcome.UNKNOWN_CUSTOMER,
                    customer_id=resolved_customer,
                    session_id=session_id,
                ),
                destination,
            )

        if state.do_not_call:
            return _record(
                _decision(
                    DialOutcome.CUSTOMER_DO_NOT_CALL,
                    customer_id=resolved_customer,
                    session_id=session_id,
                ),
                destination,
            )

    # 4. The one permitted destination has to exist.
    allowed_number = normalise_e164(settings.demo_phone_number)
    if allowed_number is None:
        return _record(
            _decision(
                DialOutcome.DEMO_NUMBER_NOT_CONFIGURED,
                customer_id=resolved_customer,
                session_id=session_id,
            ),
            destination,
        )

    # 5. The requested destination has to be a real number. Never repaired.
    requested = normalise_e164(destination)
    if requested is None:
        return _record(
            _decision(
                DialOutcome.INVALID_DESTINATION,
                customer_id=resolved_customer,
                session_id=session_id,
            ),
            destination,
        )

    # 6. And it has to be that number, exactly.
    if requested != allowed_number:
        return _record(
            _decision(
                DialOutcome.DESTINATION_NOT_AUTHORIZED,
                customer_id=resolved_customer,
                session_id=session_id,
            ),
            destination,
        )

    return _record(
        _decision(
            DialOutcome.AUTHORIZED_DEMO_NUMBER,
            destination=requested,
            customer_id=resolved_customer,
            session_id=session_id,
        ),
        destination,
    )


def assert_dial_allowed(
    destination: str | None = None,
    customer_id: str | None = None,
    session_id: str | None = None,
) -> DialDecision:
    """Like :func:`check_dial_allowed`, but raises on refusal.

    This is the call the eventual provider adapter should make, so that
    forgetting to inspect ``allowed`` cannot result in a call being placed.
    """
    return check_dial_allowed(destination, customer_id, session_id).raise_if_rejected()


def posture() -> dict[str, object]:
    """A safe summary of the current dial configuration.

    Booleans and a label only — never the configured number. Used by the
    dashboard's status indicator.
    """
    configured = normalise_e164(settings.demo_phone_number) is not None
    if not settings.enable_outbound_calls:
        label = "Disabled"
    elif not configured:
        label = "Enabled, but no number configured"
    else:
        label = "Demo number only"
    return {
        "enabled": settings.enable_outbound_calls,
        "demo_number_configured": configured,
        "label": label,
    }


def _record(decision: DialDecision, requested: str | None) -> DialDecision:
    """Audit one decision, then return it unchanged.

    Two destinations for the record:

    * the application log, always, with the number masked;
    * the session event timeline, when the decision belongs to a session that
      actually exists, so the dashboard shows refusals alongside tool calls.

    No secret, key, or full number reaches either.
    """
    masked = mask_number(requested)
    message = (
        "dial check %s: reason=%s destination=%s customer=%s session=%s"
    )
    arguments = (
        "ALLOWED" if decision.allowed else "REFUSED",
        decision.reason.value,
        masked,
        decision.customer_id or "-",
        decision.session_id or "-",
    )
    if decision.allowed:
        logger.info(message, *arguments)
    else:
        logger.warning(message, *arguments)

    # Only write an event when the session is real; an unknown session id has
    # nowhere to record to, and inventing one would be misleading.
    if decision.session_id and decision.reason is not DialOutcome.INVALID_SESSION:
        try:
            store.record_event(
                decision.session_id,
                decision.customer_id or "-",
                EventType.DIAL_CHECK_ALLOWED
                if decision.allowed
                else EventType.DIAL_CHECK_REFUSED,
                (
                    "Outbound call authorised for the configured demo number."
                    if decision.allowed
                    else f"Outbound call refused: {decision.reason.value}."
                ),
                {"reason": decision.reason.value, "destination": masked},
            )
        except store.SessionNotFoundError:  # pragma: no cover - defensive
            logger.debug("no session to record the dial decision against")

    return decision
