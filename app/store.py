"""JSON-backed persistence for the demo.

Design: a read-only seed plus a mutable overlay
-----------------------------------------------
``data/customers.json`` is an immutable seed. Nothing in this module ever
writes to it. Everything the agent does at runtime goes to
``data/runtime.json``, which is gitignored and can be deleted at any point to
return the demo to its starting state.

That split is what makes the demo repeatable. Mutating the seed in place would
mean a second recording of the same scenario behaves differently from the
first, and there would be no clean way to reset between takes.

Durability
----------
Writes go to a temporary file in the same directory and are then moved into
place with ``os.replace``, which is atomic on both Windows and POSIX. A reader
therefore sees either the old file or the new one, never a half-written one.
A module-level ``threading.RLock`` serialises read-modify-write cycles so two
concurrent requests cannot interleave and lose an update.

This is not a database and does not pretend to be: there is no transaction
across multiple calls and no multi-process safety. For a single local server
with ten records it is the right amount of machinery.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import tempfile
import threading
from datetime import date
from pathlib import Path
from typing import Any, Literal

from app.config import settings
from app.models import (
    AutopayStatus,
    Customer,
    CustomerState,
    Disposition,
    Escalation,
    EventType,
    PaymentAttempt,
    PaymentLink,
    PaymentResult,
    ScheduledRetry,
    Session,
    SessionEvent,
    utc_now,
)

logger = logging.getLogger(__name__)

RUNTIME_VERSION = 1

_lock = threading.RLock()

# Parsed seed, cached after first load. The seed never changes at runtime, so
# re-reading and re-validating it on every request would be wasted work.
_customers_cache: dict[str, Customer] | None = None


class CustomerNotFoundError(KeyError):
    """Raised when a customer id or phone number is not in the seed."""


class SessionNotFoundError(KeyError):
    """Raised when a session id has no record in runtime state."""


# ---------------------------------------------------------------------------
# Seed: customers.json (read-only)
# ---------------------------------------------------------------------------


def _customers_path() -> Path:
    return settings.customers_file


def load_customers(*, refresh: bool = False) -> dict[str, Customer]:
    """Load and validate all customers, keyed by ``customer_id``.

    Every record is parsed through the ``Customer`` model, so a malformed seed
    fails here with a precise field path rather than surfacing as a KeyError
    halfway through a conversation.
    """
    global _customers_cache
    with _lock:
        if _customers_cache is not None and not refresh:
            return _customers_cache

        path = _customers_path()
        if not path.exists():
            raise FileNotFoundError(
                f"Customer seed file not found at {path}. "
                "It ships with the repository and should not be deleted."
            )

        raw = json.loads(path.read_text(encoding="utf-8"))
        records = raw["customers"] if isinstance(raw, dict) else raw

        customers: dict[str, Customer] = {}
        for record in records:
            customer = Customer.model_validate(record)
            if customer.customer_id in customers:
                raise ValueError(f"Duplicate customer_id in seed: {customer.customer_id}")
            customers[customer.customer_id] = customer

        _customers_cache = customers
        logger.debug("loaded %d customers from %s", len(customers), path)
        return customers


def list_customers() -> list[Customer]:
    """All customers, in seed order."""
    return list(load_customers().values())


def get_customer(customer_id: str) -> Customer:
    """One customer by id, or raise ``CustomerNotFoundError``."""
    customers = load_customers()
    try:
        return customers[customer_id]
    except KeyError as exc:
        raise CustomerNotFoundError(f"No customer with id {customer_id!r}") from exc


def find_customer_by_phone(phone: str) -> Customer | None:
    """Look a customer up by phone number, tolerant of formatting.

    Compares on digits only, so ``+1 (415) 555-0100``, ``14155550100`` and
    ``+14155550100`` all match the same record.
    """
    wanted = _digits(phone)
    if not wanted:
        return None
    for customer in load_customers().values():
        if _digits(customer.phone) == wanted:
            return customer
    return None


def _digits(value: str) -> str:
    return "".join(character for character in value if character.isdigit())


# ---------------------------------------------------------------------------
# Runtime: runtime.json (mutable)
# ---------------------------------------------------------------------------


def _runtime_path() -> Path:
    return settings.runtime_file


def empty_runtime() -> dict[str, Any]:
    """The default runtime document, used on first run and on reset."""
    return {
        "version": RUNTIME_VERSION,
        "created_at": utc_now().isoformat(),
        "sessions": {},
        "customer_state": {},
        "payment_attempts": [],
        "scheduled_retries": [],
        "payment_links": [],
        "escalations": [],
        "dispositions": [],
        "events": [],
        "event_sequence": 0,
    }


def load_runtime() -> dict[str, Any]:
    """Read the runtime document, creating it if absent.

    A corrupt or truncated file is backed up to ``*.corrupt`` and replaced
    with a fresh document rather than crashing the server. The demo state is
    reproducible from the seed, so recovering is strictly better than failing.
    """
    with _lock:
        path = _runtime_path()
        if not path.exists():
            document = empty_runtime()
            _write_runtime(document)
            return document

        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            backup = path.with_suffix(".corrupt.json")
            path.replace(backup)
            logger.warning(
                "runtime.json was unreadable; moved to %s and started fresh", backup
            )
            document = empty_runtime()
            _write_runtime(document)
            return document

        # Fill in any key added by a later version of this module.
        for key, value in empty_runtime().items():
            document.setdefault(key, value)
        return document


def _write_runtime(document: dict[str, Any]) -> None:
    """Atomically replace runtime.json with ``document``."""
    path = _runtime_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    # Same directory as the target, so os.replace stays on one filesystem.
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=".runtime-",
        suffix=".tmp",
        delete=False,
    )
    try:
        with handle:
            json.dump(document, handle, indent=2, default=str)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(handle.name, path)
    except BaseException:
        Path(handle.name).unlink(missing_ok=True)
        raise


def reset_runtime() -> dict[str, Any]:
    """Discard all runtime state and return to the seed's starting point."""
    with _lock:
        document = empty_runtime()
        _write_runtime(document)
        logger.info("runtime state reset")
        return document


# ---------------------------------------------------------------------------
# Per-customer state
# ---------------------------------------------------------------------------


def get_state(customer_id: str) -> CustomerState:
    """Current mutable state for a customer, defaulted from the seed.

    Calling this for a customer that has no runtime record yet returns the
    starting state without writing anything.
    """
    customer = get_customer(customer_id)  # validates the id exists
    document = load_runtime()
    stored = document["customer_state"].get(customer_id)
    if stored is None:
        return CustomerState(
            customer_id=customer_id,
            status=customer.autopay.status,
        )
    return CustomerState.model_validate(stored)


def update_state(customer_id: str, **changes: Any) -> CustomerState:
    """Apply field changes to a customer's state and persist them.

    Unknown field names raise, so a typo cannot silently write a field the
    rest of the code will never read.
    """
    with _lock:
        state = get_state(customer_id)
        unknown = set(changes) - set(CustomerState.model_fields)
        if unknown:
            raise ValueError(
                f"Unknown CustomerState field(s): {', '.join(sorted(unknown))}"
            )

        updated = state.model_copy(update={**changes, "updated_at": utc_now()})
        # Re-validate so enums and dates are coerced the same way as on load.
        updated = CustomerState.model_validate(updated.model_dump())

        document = load_runtime()
        document["customer_state"][customer_id] = json.loads(updated.model_dump_json())
        _write_runtime(document)
        return updated


def all_states() -> dict[str, CustomerState]:
    """State for every customer, defaults included — the dashboard's view."""
    document = load_runtime()
    states: dict[str, CustomerState] = {}
    for customer_id, customer in load_customers().items():
        stored = document["customer_state"].get(customer_id)
        states[customer_id] = (
            CustomerState.model_validate(stored)
            if stored
            else CustomerState(customer_id=customer_id, status=customer.autopay.status)
        )
    return states


# ---------------------------------------------------------------------------
# Append-only audit records
# ---------------------------------------------------------------------------


def _append(collection: str, record: Any) -> None:
    """Append one Pydantic record to a runtime collection."""
    with _lock:
        document = load_runtime()
        document[collection].append(json.loads(record.model_dump_json()))
        _write_runtime(document)


def record_payment_attempt(
    customer_id: str,
    result: PaymentResult,
    attempt_number: int,
    session_id: str | None = None,
) -> PaymentAttempt:
    """Log a retry and advance the customer's status to match the outcome."""
    attempt = PaymentAttempt(
        customer_id=customer_id,
        session_id=session_id,
        attempt_number=attempt_number,
        payment_method=result.payment_method,
        status=result.status,
        transaction_id=result.transaction_id,
        amount=result.amount,
        failure_code=result.failure_code,
        confirmation_number=result.confirmation_number,
    )
    with _lock:
        _append("payment_attempts", attempt)
        update_state(
            customer_id,
            retry_attempts=attempt_number,
            status=AutopayStatus.RECOVERED if result.success else AutopayStatus.FAILED,
            last_transaction_id=result.transaction_id,
            last_confirmation_number=result.confirmation_number,
        )
    return attempt


def record_scheduled_retry(
    customer_id: str,
    scheduled_for: date,
    confirmation_number: str,
    session_id: str | None = None,
    requested_action: Literal["retry_primary", "retry_backup"] = "retry_primary",
) -> ScheduledRetry:
    """Log a postponed retry and mark the customer as scheduled."""
    retry = ScheduledRetry(
        customer_id=customer_id,
        session_id=session_id,
        scheduled_for=scheduled_for,
        requested_action=requested_action,
        confirmation_number=confirmation_number,
    )
    with _lock:
        _append("scheduled_retries", retry)
        update_state(
            customer_id,
            status=AutopayStatus.SCHEDULED,
            scheduled_for=scheduled_for,
            last_confirmation_number=confirmation_number,
        )
    return retry


def record_payment_link(
    customer_id: str,
    channel: Literal["sms", "email"],
    sent_to_masked: str,
    link_id: str,
    url: str = "",
    session_id: str | None = None,
) -> PaymentLink:
    """Log an offered payment link. Nothing is actually transmitted."""
    link = PaymentLink(
        customer_id=customer_id,
        session_id=session_id,
        channel=channel,
        sent_to_masked=sent_to_masked,
        link_id=link_id,
        url=url or f"https://example.test/pay/{link_id}",
        delivered=False,
    )
    with _lock:
        _append("payment_links", link)
        update_state(customer_id, status=AutopayStatus.LINK_SENT)
    return link


def record_escalation(
    customer_id: str,
    ticket_id: str,
    reason: str,
    callback_window: str,
    session_id: str | None = None,
    notes: str | None = None,
) -> Escalation:
    """Log a handoff to a human and mark the customer as escalated."""
    escalation = Escalation(
        customer_id=customer_id,
        session_id=session_id,
        ticket_id=ticket_id,
        reason=reason,
        notes=notes,
        callback_window=callback_window,
    )
    with _lock:
        _append("escalations", escalation)
        update_state(customer_id, status=AutopayStatus.ESCALATED)
    return escalation


def record_disposition(
    customer_id: str,
    disposition: Disposition,
    notes: str | None = None,
    session_id: str | None = None,
) -> CustomerState:
    """Record how a conversation ended.

    ``do_not_call`` is special: it sets the opt-out flag as well as the
    status, because that choice has to outlive the call it was made on.
    """
    with _lock:
        document = load_runtime()
        document["dispositions"].append(
            {
                "customer_id": customer_id,
                "session_id": session_id,
                "disposition": disposition.value,
                "notes": notes,
                "created_at": utc_now().isoformat(),
            }
        )
        _write_runtime(document)

        changes: dict[str, Any] = {
            "disposition": disposition,
            "disposition_notes": notes,
        }
        if disposition is Disposition.DO_NOT_CALL:
            changes["do_not_call"] = True
            changes["status"] = AutopayStatus.DO_NOT_CALL
        return update_state(customer_id, **changes)


def get_payment_attempts(customer_id: str | None = None) -> list[dict[str, Any]]:
    """Retry history, newest last. Filtered by customer when one is given."""
    attempts = load_runtime()["payment_attempts"]
    if customer_id is None:
        return list(attempts)
    return [a for a in attempts if a.get("customer_id") == customer_id]


def next_attempt_number(customer_id: str) -> int:
    """The attempt number the next retry should use, starting at 1.

    Counts only retries this project made, not the ``attempts_so_far`` that
    the seed records as having happened before we called.
    """
    return len(get_payment_attempts(customer_id)) + 1


# ---------------------------------------------------------------------------
# Sessions
#
# A session binds one conversation to one customer. There is deliberately no
# function here that changes an existing session's customer_id: rebinding is
# not an operation this codebase offers, so a tool call can never be steered
# to a different customer's data. Serving someone else means creating a new
# session, which is an explicit act by the caller that places the call.
# ---------------------------------------------------------------------------


def create_session(
    customer_id: str,
    session_id: str | None = None,
    channel: Literal["web", "phone", "simulator", "test"] = "test",
) -> Session:
    """Open a session for a customer.

    ``session_id`` may be supplied to make a run reproducible (tests and
    scripted demos do this); otherwise a random one is generated. The customer
    id is validated against the seed first, so a session can never point at a
    customer that does not exist.
    """
    get_customer(customer_id)  # raises CustomerNotFoundError on a bad id

    with _lock:
        document = load_runtime()
        if session_id is None:
            session_id = f"sess_{secrets.token_hex(8)}"
        elif session_id in document["sessions"]:
            raise ValueError(f"Session {session_id!r} already exists")

        session = Session(
            session_id=session_id,
            customer_id=customer_id,
            channel=channel,
        )
        document["sessions"][session_id] = json.loads(session.model_dump_json())
        _write_runtime(document)
        logger.info("session %s opened for %s (%s)", session_id, customer_id, channel)
        return session


def get_session(session_id: str) -> Session:
    """One session by id, or raise ``SessionNotFoundError``."""
    document = load_runtime()
    stored = document["sessions"].get(session_id)
    if stored is None:
        raise SessionNotFoundError(f"No session with id {session_id!r}")
    return Session.model_validate(stored)


def _save_session(session: Session) -> Session:
    with _lock:
        document = load_runtime()
        document["sessions"][session.session_id] = json.loads(session.model_dump_json())
        _write_runtime(document)
        return session


def touch_session(session_id: str, *, retry: bool = False) -> Session:
    """Record that a tool was called on this session.

    ``retry=True`` also increments the per-session retry counter, which is
    what caps retries within a single conversation.
    """
    with _lock:
        session = get_session(session_id)
        updated = session.model_copy(
            update={
                "tool_calls": session.tool_calls + 1,
                "retry_attempts": session.retry_attempts + (1 if retry else 0),
                "last_tool_at": utc_now(),
            }
        )
        return _save_session(updated)


def close_session(session_id: str) -> Session:
    """Mark a session finished. Later tool calls on it are rejected."""
    with _lock:
        session = get_session(session_id)
        return _save_session(session.model_copy(update={"closed": True}))


def list_sessions() -> list[Session]:
    """Every session, oldest first."""
    document = load_runtime()
    return [Session.model_validate(record) for record in document["sessions"].values()]


def get_scheduled_retries(customer_id: str | None = None) -> list[dict[str, Any]]:
    """Scheduled retry records, optionally filtered by customer."""
    records = load_runtime()["scheduled_retries"]
    if customer_id is None:
        return list(records)
    return [r for r in records if r.get("customer_id") == customer_id]


def get_payment_links(customer_id: str | None = None) -> list[dict[str, Any]]:
    """Payment link records, optionally filtered by customer."""
    records = load_runtime()["payment_links"]
    if customer_id is None:
        return list(records)
    return [r for r in records if r.get("customer_id") == customer_id]


def get_escalations(customer_id: str | None = None) -> list[dict[str, Any]]:
    """Escalation records, optionally filtered by customer."""
    records = load_runtime()["escalations"]
    if customer_id is None:
        return list(records)
    return [r for r in records if r.get("customer_id") == customer_id]


def get_dispositions(customer_id: str | None = None) -> list[dict[str, Any]]:
    """Disposition records, optionally filtered by customer."""
    records = load_runtime()["dispositions"]
    if customer_id is None:
        return list(records)
    return [r for r in records if r.get("customer_id") == customer_id]


# ---------------------------------------------------------------------------
# Session event log
# ---------------------------------------------------------------------------


def record_event(
    session_id: str,
    customer_id: str,
    event_type: EventType,
    summary: str,
    detail: dict[str, str] | None = None,
) -> SessionEvent:
    """Append one entry to a session's timeline.

    ``sequence`` comes from a monotonic counter in the runtime document rather
    than from the clock, so two events written in the same millisecond still
    have a defined order. The dashboard sorts on it.
    """
    with _lock:
        document = load_runtime()
        document["event_sequence"] = int(document.get("event_sequence", 0)) + 1
        event = SessionEvent(
            sequence=document["event_sequence"],
            session_id=session_id,
            customer_id=customer_id,
            event_type=event_type,
            summary=summary,
            detail=detail or {},
        )
        document["events"].append(json.loads(event.model_dump_json()))
        _write_runtime(document)
        return event


def get_events(session_id: str | None = None) -> list[dict[str, Any]]:
    """Timeline entries in chronological order, optionally for one session."""
    events = load_runtime()["events"]
    if session_id is not None:
        events = [e for e in events if e.get("session_id") == session_id]
    return sorted(events, key=lambda event: event.get("sequence", 0))


# ---------------------------------------------------------------------------
# Scoped reset
# ---------------------------------------------------------------------------

#: Collections whose records carry a session_id and so can be reset per session.
_SESSION_SCOPED_COLLECTIONS = (
    "payment_attempts",
    "scheduled_retries",
    "payment_links",
    "escalations",
    "dispositions",
    "events",
)


def reset_session(session_id: str) -> tuple[Session, int]:
    """Clear one session's runtime state and reopen it for another run.

    Returns the reopened session and how many records were discarded.

    Scope: every audit record tagged with this ``session_id``, plus the
    customer's ledger row, plus the session's own counters. Other sessions and
    their records are untouched, and ``data/customers.json`` is never opened
    for writing — the seed is what the reset restores *to*.

    One caveat worth knowing: the ledger row is keyed by customer, not by
    session. Two open sessions on the same customer therefore share one row,
    and resetting either clears it for both. The dashboard drives one session
    at a time, so this does not arise in practice.
    """
    with _lock:
        session = get_session(session_id)
        document = load_runtime()

        cleared = 0
        for collection in _SESSION_SCOPED_COLLECTIONS:
            before = len(document[collection])
            document[collection] = [
                record
                for record in document[collection]
                if record.get("session_id") != session_id
            ]
            cleared += before - len(document[collection])

        # Drop the customer's ledger row so get_state() falls back to the seed.
        if document["customer_state"].pop(session.customer_id, None) is not None:
            cleared += 1

        reopened = session.model_copy(
            update={
                "closed": False,
                "tool_calls": 0,
                "retry_attempts": 0,
                "last_tool_at": None,
            }
        )
        document["sessions"][session_id] = json.loads(reopened.model_dump_json())
        _write_runtime(document)

    record_event(
        session_id,
        session.customer_id,
        EventType.SESSION_RESET,
        "Session reset for another demo run.",
        {"records_cleared": str(cleared)},
    )
    logger.info("session %s reset (%d records cleared)", session_id, cleared)
    return reopened, cleared
