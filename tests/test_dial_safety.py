"""Tests for the outbound-call safety gate.

These are the regression tests that guard the eventual telephony integration.
Phase 9 will add code that can make a real phone ring; everything here exists
so that code cannot ring the wrong one.

The four cases in "The dangerous four" are the ones to read first.
"""

from __future__ import annotations

import socket
from pathlib import Path

import pytest

from app import store
from app.config import Settings, normalise_e164, settings
from app.dial_safety import (
    DialBlocked,
    DialDecision,
    DialOutcome,
    assert_dial_allowed,
    check_dial_allowed,
    mask_number,
    posture,
)
from app.models import Disposition

DEMO_NUMBER = "+14155550123"
OTHER_NUMBER = "+14155550199"


@pytest.fixture
def dialling_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """An otherwise-valid configuration: flag on, demo number set."""
    monkeypatch.setattr(settings, "enable_outbound_calls", True)
    monkeypatch.setattr(settings, "demo_phone_number", DEMO_NUMBER)


@pytest.fixture
def dialling_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """The default posture, stated explicitly."""
    monkeypatch.setattr(settings, "enable_outbound_calls", False)
    monkeypatch.setattr(settings, "demo_phone_number", DEMO_NUMBER)


# ---------------------------------------------------------------------------
# The dangerous four
# ---------------------------------------------------------------------------


def test_case_a_disabled_flag_refuses_even_the_authorised_number(
    dialling_disabled: None,
) -> None:
    """Case A: the right number is still refused while the flag is off."""
    decision = check_dial_allowed(DEMO_NUMBER)

    assert decision.allowed is False
    assert decision.reason is DialOutcome.OUTBOUND_CALLS_DISABLED
    assert decision.destination is None


def test_case_b_enabled_flag_refuses_any_other_number(
    dialling_enabled: None,
) -> None:
    """Case B: the flag being on authorises exactly one number, not dialling."""
    decision = check_dial_allowed(OTHER_NUMBER)

    assert decision.allowed is False
    assert decision.reason is DialOutcome.DESTINATION_NOT_AUTHORIZED
    assert decision.destination is None


def test_case_c_do_not_call_beats_the_authorised_number(
    dialling_enabled: None,
) -> None:
    """Case C: an opt-out refuses the call even with everything else correct."""
    store.record_disposition("CUST-010", Disposition.DO_NOT_CALL, "asked us to stop")
    assert store.get_state("CUST-010").do_not_call is True

    decision = check_dial_allowed(DEMO_NUMBER, customer_id="CUST-010")

    assert decision.allowed is False
    assert decision.reason is DialOutcome.CUSTOMER_DO_NOT_CALL
    assert decision.destination is None


def test_case_d_a_fully_valid_request_is_allowed(dialling_enabled: None) -> None:
    """Case D: flag on, authorised number, live session, no opt-out."""
    session_id = store.create_session("CUST-001", channel="phone").session_id

    decision = check_dial_allowed(
        DEMO_NUMBER, customer_id="CUST-001", session_id=session_id
    )

    assert decision.allowed is True
    assert decision.reason is DialOutcome.AUTHORIZED_DEMO_NUMBER
    assert decision.destination == DEMO_NUMBER
    assert decision.customer_id == "CUST-001"
    assert decision.session_id == session_id


# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------


def test_outbound_calling_is_disabled_by_default() -> None:
    """A fresh checkout must not be able to place a call."""
    fresh = Settings(_env_file=None)
    assert fresh.enable_outbound_calls is False
    assert fresh.demo_phone_number is None


def test_a_fresh_configuration_refuses_everything(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "enable_outbound_calls", False)
    monkeypatch.setattr(settings, "demo_phone_number", None)

    for destination in (None, "", DEMO_NUMBER, OTHER_NUMBER, "+1 (415) 555-0123"):
        decision = check_dial_allowed(destination)
        assert decision.allowed is False, destination


def test_missing_demo_number_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "enable_outbound_calls", True)
    monkeypatch.setattr(settings, "demo_phone_number", None)

    decision = check_dial_allowed(DEMO_NUMBER)
    assert decision.allowed is False
    assert decision.reason is DialOutcome.DEMO_NUMBER_NOT_CONFIGURED


def test_malformed_configured_demo_number_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bad value set programmatically must not become dialable."""
    monkeypatch.setattr(settings, "enable_outbound_calls", True)
    monkeypatch.setattr(settings, "demo_phone_number", "555-0123")

    decision = check_dial_allowed("555-0123")
    assert decision.allowed is False
    assert decision.reason is DialOutcome.DEMO_NUMBER_NOT_CONFIGURED


def test_env_example_configures_no_dialable_destination() -> None:
    """The committed template must not hand anyone a working destination."""
    text = (Path(__file__).resolve().parent.parent / ".env.example").read_text(
        encoding="utf-8"
    )

    values = {
        line.split("=", 1)[0].strip(): line.split("=", 1)[1].strip()
        for line in text.splitlines()
        if "=" in line and not line.lstrip().startswith("#")
    }

    assert values["ENABLE_OUTBOUND_CALLS"] == "false"
    assert values["DEMO_PHONE_NUMBER"] == "", (
        "the template must ship no number at all"
    )
    assert normalise_e164(values["DEMO_PHONE_NUMBER"]) is None
    assert values["AGENT_PHONE_NUMBER_ID"] == ""


def test_env_example_contains_no_real_credentials() -> None:
    """No assigned value may be a credential or a dialable number.

    Comments are allowed to show the required *format*, so the check
    distinguishes the two: assigned values must contain no long digit run at
    all, and any number appearing illustratively in a comment has to sit in
    the 555-0100..555-0199 block reserved for fiction.
    """
    import re

    path = Path(__file__).resolve().parent.parent / ".env.example"
    text = path.read_text(encoding="utf-8")
    assert "sk_replace_me" in text

    assigned, commented = [], []
    for line in text.splitlines():
        (commented if line.lstrip().startswith("#") else assigned).append(line)

    for line in assigned:
        value = line.split("=", 1)[1] if "=" in line else line
        assert not re.search(r"\d{10,}", value), f"long digit run assigned: {line!r}"

    for number in re.findall(r"\+?(\d{10,15})", "\n".join(commented)):
        assert number[1:4] + number[4:7] and number[4:7] == "555", (
            f"illustrative number {number} is not in the fictional 555 block"
        )
        assert 100 <= int(number[7:11]) <= 199, (
            f"illustrative number {number} is outside 555-0100..555-0199"
        )


# ---------------------------------------------------------------------------
# Destination validation
# ---------------------------------------------------------------------------


def test_the_configured_number_is_accepted(dialling_enabled: None) -> None:
    assert check_dial_allowed(DEMO_NUMBER).allowed is True


@pytest.mark.parametrize(
    "variant",
    [
        "+1 415 555 0123",
        "+1 (415) 555-0123",
        "+1-415-555-0123",
        "  +14155550123  ",
        "+1.415.555.0123",
    ],
)
def test_formatted_variants_of_the_configured_number_normalise(
    dialling_enabled: None, variant: str
) -> None:
    """Reformatting is allowed; the digits still have to match exactly."""
    decision = check_dial_allowed(variant)
    assert decision.allowed is True, variant
    assert decision.destination == DEMO_NUMBER


@pytest.mark.parametrize(
    "bad",
    [
        None,
        "",
        "   ",
        "555-0123",
        "14155550123",  # no leading +
        "+0415555012",  # zero country code
        "+1415",  # too short
        "+1415555012345678",  # too long
        "not-a-number",
        "+1415555012a",
        "tel:+14155550123",
    ],
)
def test_malformed_destinations_are_refused(dialling_enabled: None, bad: object) -> None:
    decision = check_dial_allowed(bad)  # type: ignore[arg-type]
    assert decision.allowed is False, bad
    assert decision.reason is DialOutcome.INVALID_DESTINATION
    assert decision.destination is None


def test_a_different_valid_number_is_refused(dialling_enabled: None) -> None:
    decision = check_dial_allowed(OTHER_NUMBER)
    assert decision.reason is DialOutcome.DESTINATION_NOT_AUTHORIZED


def test_a_number_one_digit_off_is_refused(dialling_enabled: None) -> None:
    """No fuzzy matching, no repair."""
    decision = check_dial_allowed("+14155550124")
    assert decision.allowed is False
    assert decision.reason is DialOutcome.DESTINATION_NOT_AUTHORIZED


def test_no_destination_at_all_is_refused(dialling_enabled: None) -> None:
    """There is no default destination, and the customer record is not one."""
    decision = check_dial_allowed(customer_id="CUST-001")
    assert decision.allowed is False
    assert decision.reason is DialOutcome.INVALID_DESTINATION


# ---------------------------------------------------------------------------
# Customer phone numbers are not destinations
# ---------------------------------------------------------------------------


def test_no_customer_phone_number_is_dialable(dialling_enabled: None) -> None:
    """The decisive test for the assignment's constraint.

    All ten seed records hold fictional numbers. None of them may be dialled,
    even with the feature flag on and a live session.
    """
    for customer in store.list_customers():
        session_id = store.create_session(customer.customer_id, channel="phone").session_id
        decision = check_dial_allowed(
            customer.phone,
            customer_id=customer.customer_id,
            session_id=session_id,
        )
        assert decision.allowed is False, (
            f"{customer.customer_id}'s stored number must not be callable"
        )
        assert decision.reason is DialOutcome.DESTINATION_NOT_AUTHORIZED
        assert decision.destination is None


def test_seed_numbers_are_valid_e164_so_the_refusal_is_meaningful() -> None:
    """The refusal above must be the allow-list working, not a parse failure."""
    for customer in store.list_customers():
        assert normalise_e164(customer.phone) == customer.phone, customer.customer_id


def test_a_model_supplied_number_is_refused(dialling_enabled: None) -> None:
    """Anything arriving from outside is checked against the one value."""
    for hostile in (
        "+12025550111",
        "+13035550144",
        "+16175550155",
        DEMO_NUMBER[:-1] + "9",
    ):
        assert check_dial_allowed(hostile).allowed is False, hostile


# ---------------------------------------------------------------------------
# Feature flag
# ---------------------------------------------------------------------------


def test_flag_false_refuses_an_otherwise_perfect_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "demo_phone_number", DEMO_NUMBER)
    session_id = store.create_session("CUST-001", channel="phone").session_id

    monkeypatch.setattr(settings, "enable_outbound_calls", False)
    refused = check_dial_allowed(DEMO_NUMBER, session_id=session_id)
    assert refused.allowed is False
    assert refused.reason is DialOutcome.OUTBOUND_CALLS_DISABLED

    monkeypatch.setattr(settings, "enable_outbound_calls", True)
    allowed = check_dial_allowed(DEMO_NUMBER, session_id=session_id)
    assert allowed.allowed is True


def test_the_flag_must_be_a_real_boolean_true(monkeypatch: pytest.MonkeyPatch) -> None:
    """Truthy strings do not count; pydantic parses the env var."""
    assert Settings(_env_file=None, enable_outbound_calls="false").enable_outbound_calls is False
    assert Settings(_env_file=None, enable_outbound_calls="true").enable_outbound_calls is True


# ---------------------------------------------------------------------------
# Do-not-call
# ---------------------------------------------------------------------------


def test_do_not_call_is_refused_via_the_session_too(dialling_enabled: None) -> None:
    """The opt-out is found through the session's customer binding."""
    session_id = store.create_session("CUST-010", channel="phone").session_id
    store.record_disposition("CUST-010", Disposition.DO_NOT_CALL, "opted out")

    decision = check_dial_allowed(DEMO_NUMBER, session_id=session_id)
    assert decision.allowed is False
    assert decision.reason is DialOutcome.CUSTOMER_DO_NOT_CALL
    assert decision.customer_id == "CUST-010"


def test_do_not_call_outranks_a_bad_destination(dialling_enabled: None) -> None:
    """The opt-out is reported first, because it is the stronger refusal."""
    store.record_disposition("CUST-010", Disposition.DO_NOT_CALL)
    decision = check_dial_allowed(OTHER_NUMBER, customer_id="CUST-010")
    assert decision.reason is DialOutcome.CUSTOMER_DO_NOT_CALL


def test_do_not_call_is_refused_without_a_demo_number(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Checked regardless of the configured number, as specified."""
    monkeypatch.setattr(settings, "enable_outbound_calls", True)
    monkeypatch.setattr(settings, "demo_phone_number", None)
    store.record_disposition("CUST-010", Disposition.DO_NOT_CALL)

    decision = check_dial_allowed(DEMO_NUMBER, customer_id="CUST-010")
    assert decision.reason is DialOutcome.CUSTOMER_DO_NOT_CALL


def test_customers_who_have_not_opted_out_are_unaffected(
    dialling_enabled: None,
) -> None:
    store.record_disposition("CUST-010", Disposition.DO_NOT_CALL)
    assert check_dial_allowed(DEMO_NUMBER, customer_id="CUST-001").allowed is True


def test_an_unknown_customer_is_refused(dialling_enabled: None) -> None:
    decision = check_dial_allowed(DEMO_NUMBER, customer_id="CUST-999")
    assert decision.allowed is False
    assert decision.reason is DialOutcome.UNKNOWN_CUSTOMER


# ---------------------------------------------------------------------------
# Session binding
# ---------------------------------------------------------------------------


def test_a_valid_session_is_accepted(dialling_enabled: None) -> None:
    session_id = store.create_session("CUST-004", channel="phone").session_id
    decision = check_dial_allowed(DEMO_NUMBER, session_id=session_id)
    assert decision.allowed is True
    assert decision.customer_id == "CUST-004"


def test_an_unknown_session_is_refused(dialling_enabled: None) -> None:
    decision = check_dial_allowed(DEMO_NUMBER, session_id="sess_nope")
    assert decision.allowed is False
    assert decision.reason is DialOutcome.INVALID_SESSION


def test_a_closed_session_is_refused(dialling_enabled: None) -> None:
    session_id = store.create_session("CUST-001", channel="phone").session_id
    store.close_session(session_id)

    decision = check_dial_allowed(DEMO_NUMBER, session_id=session_id)
    assert decision.allowed is False
    assert decision.reason is DialOutcome.SESSION_CLOSED


def test_a_mismatched_customer_and_session_is_refused(dialling_enabled: None) -> None:
    """The caller cannot point an existing session at another customer."""
    session_id = store.create_session("CUST-001", channel="phone").session_id

    decision = check_dial_allowed(
        DEMO_NUMBER, customer_id="CUST-008", session_id=session_id
    )
    assert decision.allowed is False
    assert decision.reason is DialOutcome.CUSTOMER_MISMATCH
    assert decision.customer_id == "CUST-001", "the session's binding wins"


def test_the_session_determines_the_customer(dialling_enabled: None) -> None:
    """A dial request need not name a customer; the session already does."""
    session_id = store.create_session("CUST-006", channel="phone").session_id
    decision = check_dial_allowed(DEMO_NUMBER, session_id=session_id)
    assert decision.customer_id == "CUST-006"


def test_dial_safety_reuses_the_phase_two_session_store() -> None:
    """No second session mechanism: the module reads app.store directly."""
    text = Path("app/dial_safety.py").read_text(encoding="utf-8")
    assert "store.get_session(" in text
    assert "store.get_state(" in text
    for reimplementation in ("class Session", "sessions = {}", "def create_session"):
        assert reimplementation not in text


# ---------------------------------------------------------------------------
# Decision shape
# ---------------------------------------------------------------------------


def test_a_rejected_decision_never_carries_a_destination(
    dialling_enabled: None,
) -> None:
    """A provider that ignores `allowed` still has nothing to dial."""
    for destination in (OTHER_NUMBER, "garbage", None, ""):
        decision = check_dial_allowed(destination)  # type: ignore[arg-type]
        assert decision.allowed is False
        assert decision.destination is None


def test_reasons_come_from_a_closed_set(dialling_enabled: None) -> None:
    decision = check_dial_allowed(OTHER_NUMBER)
    assert isinstance(decision.reason, DialOutcome)
    assert decision.reason.value in {member.value for member in DialOutcome}


def test_the_decision_is_immutable(dialling_enabled: None) -> None:
    decision = check_dial_allowed(DEMO_NUMBER)
    with pytest.raises(Exception):
        decision.allowed = False  # type: ignore[misc]


def test_the_decision_serialises_cleanly(dialling_enabled: None) -> None:
    payload = check_dial_allowed(DEMO_NUMBER).model_dump(mode="json")
    assert payload["allowed"] is True
    assert payload["reason"] == "authorized_demo_number"
    assert payload["destination"] == DEMO_NUMBER
    assert "checked_at" in payload


def test_the_decision_exposes_no_configuration(dialling_enabled: None) -> None:
    """No secret or setting leaks through the decision object."""
    monkeyed = check_dial_allowed(OTHER_NUMBER).model_dump_json()
    assert DEMO_NUMBER not in monkeyed, "a refusal must not echo the allowed number"
    assert set(DialDecision.model_fields) == {
        "allowed",
        "destination",
        "reason",
        "customer_id",
        "session_id",
        "checked_at",
    }


def test_assert_dial_allowed_raises_on_refusal(dialling_enabled: None) -> None:
    with pytest.raises(DialBlocked) as caught:
        assert_dial_allowed(OTHER_NUMBER)
    assert caught.value.decision.reason is DialOutcome.DESTINATION_NOT_AUTHORIZED


def test_assert_dial_allowed_returns_the_decision_when_allowed(
    dialling_enabled: None,
) -> None:
    decision = assert_dial_allowed(DEMO_NUMBER)
    assert decision.allowed is True
    assert decision.destination == DEMO_NUMBER


# ---------------------------------------------------------------------------
# Logging and audit
# ---------------------------------------------------------------------------


def test_numbers_are_masked_for_logging() -> None:
    masked = mask_number("+14155550123")
    assert masked.endswith("23 (11 digits)")
    assert "4155550" not in masked
    assert mask_number(None) == "<none>"
    assert mask_number("") == "<none>"
    assert "digits" in mask_number("12")


def test_a_refusal_is_logged_without_the_full_number(
    dialling_enabled: None, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level("WARNING", logger="app.dial_safety"):
        check_dial_allowed(OTHER_NUMBER, customer_id="CUST-001")

    text = caplog.text
    assert "destination_not_authorized" in text
    assert "CUST-001" in text
    assert OTHER_NUMBER not in text, "the full number must not be logged"
    assert DEMO_NUMBER not in text, "nor the configured one"


def test_an_authorised_call_is_logged_without_the_full_number(
    dialling_enabled: None, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level("INFO", logger="app.dial_safety"):
        check_dial_allowed(DEMO_NUMBER)
    assert "ALLOWED" in caplog.text
    assert DEMO_NUMBER not in caplog.text


def test_decisions_are_recorded_on_the_session_timeline(
    dialling_enabled: None,
) -> None:
    """Refusals show up beside tool calls, using the Phase 3 event log."""
    session_id = store.create_session("CUST-001", channel="phone").session_id

    check_dial_allowed(OTHER_NUMBER, session_id=session_id)
    check_dial_allowed(DEMO_NUMBER, session_id=session_id)

    events = store.get_events(session_id)
    types = [event["event_type"] for event in events]
    assert "dial_check_refused" in types
    assert "dial_check_allowed" in types

    for event in events:
        if event["event_type"].startswith("dial_check"):
            assert OTHER_NUMBER not in str(event)
            assert DEMO_NUMBER not in str(event)


def test_no_event_is_written_for_an_unknown_session(dialling_enabled: None) -> None:
    check_dial_allowed(DEMO_NUMBER, session_id="sess_nope")
    assert store.get_events("sess_nope") == []


def test_logs_never_contain_a_secret(
    dialling_enabled: None, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "tool_shared_secret", "super-secret-value")
    monkeypatch.setattr(settings, "elevenlabs_api_key", "sk_do_not_log_me")

    with caplog.at_level("DEBUG", logger="app.dial_safety"):
        check_dial_allowed(OTHER_NUMBER, customer_id="CUST-001")

    assert "super-secret-value" not in caplog.text
    assert "sk_do_not_log_me" not in caplog.text


# ---------------------------------------------------------------------------
# Posture (the dashboard indicator)
# ---------------------------------------------------------------------------


def test_posture_reports_disabled_by_default(dialling_disabled: None) -> None:
    assert posture() == {
        "enabled": False,
        "demo_number_configured": True,
        "label": "Disabled",
    }


def test_posture_reports_demo_number_only_when_armed(dialling_enabled: None) -> None:
    assert posture()["label"] == "Demo number only"


def test_posture_reports_a_missing_number(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "enable_outbound_calls", True)
    monkeypatch.setattr(settings, "demo_phone_number", None)
    assert posture()["label"] == "Enabled, but no number configured"


def test_posture_never_reveals_the_number(dialling_enabled: None) -> None:
    assert DEMO_NUMBER not in str(posture())


# ---------------------------------------------------------------------------
# No provider, no network
# ---------------------------------------------------------------------------


def test_dial_safety_imports_no_telephony_or_voice_provider() -> None:
    text = Path("app/dial_safety.py").read_text(encoding="utf-8").lower()
    for banned in (
        "import twilio",
        "from twilio",
        "import elevenlabs",
        "from elevenlabs",
        "import httpx",
        "import requests",
        "import urllib",
        "import socket",
    ):
        assert banned not in text, f"dial_safety must not {banned}"


def test_no_telephony_package_is_installed() -> None:
    """No provider dependency has crept into the environment."""
    import importlib.util

    for module in ("twilio", "elevenlabs.conversational_ai.twilio"):
        if module == "twilio":
            assert importlib.util.find_spec(module) is None, (
                "the twilio package must not be a dependency"
            )


def test_requirements_list_no_telephony_package() -> None:
    text = Path("requirements.txt").read_text(encoding="utf-8").lower()
    assert "twilio" not in text
    # elevenlabs is a declared dependency for Phase 6, but unused so far.
    assert "elevenlabs==" in text


def test_the_guard_attempts_no_connection(
    dialling_enabled: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every code path, with the socket layer refusing non-loopback traffic."""
    real_connect = socket.socket.connect
    real_getaddrinfo = socket.getaddrinfo
    loopback = {"127.0.0.1", "::1", "localhost", "testserver"}

    def guarded_connect(self: socket.socket, address: object) -> object:
        host = str(address[0]) if isinstance(address, tuple) and address else str(address)
        if host not in loopback:
            raise AssertionError(f"dial safety attempted to reach {host}")
        return real_connect(self, address)

    def guarded_getaddrinfo(host: object, *args: object, **kwargs: object) -> object:
        if host is not None and str(host) not in loopback:
            raise AssertionError(f"dial safety attempted to resolve {host}")
        return real_getaddrinfo(host, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket, "getaddrinfo", guarded_getaddrinfo)

    session_id = store.create_session("CUST-001", channel="phone").session_id
    store.record_disposition("CUST-010", Disposition.DO_NOT_CALL)

    check_dial_allowed(DEMO_NUMBER)
    check_dial_allowed(OTHER_NUMBER)
    check_dial_allowed("garbage")
    check_dial_allowed(None)
    check_dial_allowed(DEMO_NUMBER, customer_id="CUST-010")
    check_dial_allowed(DEMO_NUMBER, customer_id="CUST-999")
    check_dial_allowed(DEMO_NUMBER, session_id="sess_nope")
    check_dial_allowed(DEMO_NUMBER, session_id=session_id)
    posture()


def test_no_dial_endpoint_exists() -> None:
    """The gate is a function. Nothing exposes dialling over HTTP."""
    from fastapi.testclient import TestClient

    from app.main import app

    paths = TestClient(app).get("/openapi.json").json()["paths"]
    for forbidden in ("/call", "/dial", "/api/call", "/api/dial", "/tools/call"):
        assert forbidden not in paths

    assert not any("dial" in path.lower() for path in paths)
    assert not any("call" in path.lower() for path in paths)


def test_the_dashboard_offers_no_dialling_controls() -> None:
    """No number box, no call button: a person cannot dial from the page."""
    import re

    from fastapi.testclient import TestClient

    from app.main import app

    page = TestClient(app).get("/dashboard").text
    assert not re.search(r'type="(tel|number)"', page)
    assert not re.search(r"(?i)>\s*(call|dial)\b", page)
    assert "DEMO_PHONE_NUMBER" not in page


def test_the_dashboard_shows_the_dial_posture() -> None:
    from fastapi.testclient import TestClient

    from app.main import app

    page = TestClient(app).get("/dashboard").text
    assert "Outbound calling:" in page
