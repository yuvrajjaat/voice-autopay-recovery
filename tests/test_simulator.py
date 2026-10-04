"""Tests for the offline conversation simulator.

Each scenario is asserted on three things: the disposition it reaches, the
tools it did (or did not) call, and the backend state it leaves behind. The
last one matters most — a scenario that claimed success without the ledger
agreeing would be exactly the bug the project is built to avoid.
"""

from __future__ import annotations

import socket
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import store
from app.config import settings
from app.main import app
from app.models import AutopayStatus, Disposition
from scripts.simulate import (
    SCENARIOS,
    Intent,
    Result,
    main,
    run_all,
    run_scenario,
    scenario_for,
)


@pytest.fixture(autouse=True)
def configured_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the tool secret so the simulator does not mint a random one."""
    monkeypatch.setattr(settings, "tool_shared_secret", "test-tool-secret-value")


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def run(customer_id: str, client: TestClient) -> Result:
    return run_scenario(scenario_for(customer_id), client)


# ---------------------------------------------------------------------------
# Coverage of the scenario set
# ---------------------------------------------------------------------------


def test_there_is_a_scenario_for_every_fictional_customer() -> None:
    assert {scenario.customer_id for scenario in SCENARIOS} == {
        customer.customer_id for customer in store.list_customers()
    }


def test_every_required_disposition_is_exercised() -> None:
    reached = {scenario.expected_disposition for scenario in SCENARIOS}
    for required in (
        Disposition.PAYMENT_RECOVERED,
        Disposition.PAYMENT_LINK_PREPARED,
        Disposition.RETRY_SCHEDULED,
        Disposition.ESCALATED,
        Disposition.CUSTOMER_DECLINED,
        Disposition.DO_NOT_CALL,
        Disposition.VERIFICATION_FAILED,
        Disposition.UNRESOLVED,
    ):
        assert required in reached, f"no scenario reaches {required.value}"


def test_scenario_lookup_is_case_insensitive_and_strict() -> None:
    assert scenario_for("cust-001").customer_id == "CUST-001"
    with pytest.raises(KeyError):
        scenario_for("CUST-999")


# ---------------------------------------------------------------------------
# Scenario 1 - successful recovery
# ---------------------------------------------------------------------------


def test_cust_001_recovers(client: TestClient) -> None:
    result = run("CUST-001", client)

    assert result.disposition is Disposition.PAYMENT_RECOVERED
    assert result.passed
    assert result.tools_called == [
        "verify_identity",
        "get_failed_payment_details",
        "retry_payment",
        "log_disposition",
    ]
    assert result.final_state["payment_status"] == "recovered"
    assert result.final_state["last_confirmation_number"] == "C-0254"
    assert store.get_state("CUST-001").status is AutopayStatus.RECOVERED


def test_cust_001_only_claims_success_after_the_tool_confirms_it(
    client: TestClient,
) -> None:
    """The success line must come after a paid result, never before."""
    result = run("CUST-001", client)
    rendered = [line.render() for line in result.lines]

    paid_index = next(i for i, line in enumerate(rendered) if "status=paid" in line)
    claim_index = next(i for i, line in enumerate(rendered) if "gone through" in line)
    assert claim_index > paid_index


# ---------------------------------------------------------------------------
# Scenario 2 - payment link
# ---------------------------------------------------------------------------


def test_cust_002_prepares_a_payment_link(client: TestClient) -> None:
    result = run("CUST-002", client)

    assert result.disposition is Disposition.PAYMENT_LINK_PREPARED
    assert "send_payment_link" in result.tools_called
    assert "retry_payment" not in result.tools_called, (
        "an expired card should not be retried"
    )
    assert result.final_state["payment_link_prepared"] is True
    assert store.get_payment_links("CUST-002")[0]["delivered"] is False


def test_cust_002_refuses_a_card_number_by_voice(client: TestClient) -> None:
    """The agent must redirect to a link instead of taking the number."""
    transcript = run("CUST-002", client).transcript().lower()
    assert "read you the number" in transcript
    assert "can't take card numbers over the phone" in transcript


def test_cust_002_never_claims_a_message_was_sent(client: TestClient) -> None:
    transcript = run("CUST-002", client).transcript().lower()
    assert "nothing's been sent out" in transcript
    for false_claim in ("i've sent you a text", "check your email", "has been sent to"):
        assert false_claim not in transcript


# ---------------------------------------------------------------------------
# Scenario 3 - backup payment method
# ---------------------------------------------------------------------------


def test_cust_003_recovers_on_the_backup_card(client: TestClient) -> None:
    result = run("CUST-003", client)

    assert result.disposition is Disposition.PAYMENT_RECOVERED
    assert result.tools_called.count("retry_payment") == 2
    assert result.final_state["retry_attempts"] == 2
    assert result.final_state["payment_status"] == "recovered"

    attempts = store.get_payment_attempts("CUST-003")
    assert [attempt["payment_method"] for attempt in attempts] == ["primary", "backup"]
    assert attempts[0]["status"] == "failed"
    assert attempts[1]["status"] == "paid"


def test_cust_003_is_routed_by_the_backend_not_by_the_script(
    client: TestClient,
) -> None:
    """The backup card is offered because next_action said so."""
    transcript = run("CUST-003", client).transcript()
    assert "next_action=offer_backup_method" in transcript
    assert "another card on your account" in transcript


# ---------------------------------------------------------------------------
# Scenario 4 - scheduled retry
# ---------------------------------------------------------------------------


def test_cust_004_schedules_a_retry(client: TestClient) -> None:
    result = run("CUST-004", client)

    assert result.disposition is Disposition.RETRY_SCHEDULED
    assert "schedule_retry" in result.tools_called
    assert "retry_payment" not in result.tools_called
    assert result.final_state["payment_status"] == "scheduled"
    assert result.final_state["scheduled_for"] is not None
    assert len(store.get_scheduled_retries("CUST-004")) == 1


def test_cust_004_does_not_promise_an_automatic_charge(client: TestClient) -> None:
    transcript = run("CUST-004", client).transcript().lower()
    assert "it won't charge on its own" in transcript
    assert "will be charged automatically" not in transcript


# ---------------------------------------------------------------------------
# Scenario 5 - customer declines
# ---------------------------------------------------------------------------


def test_cust_005_accepts_a_refusal(client: TestClient) -> None:
    """This card would have approved; the agent still takes no for an answer."""
    assert store.get_customer("CUST-005").scenario.mock_retry_outcome.value == "succeed"

    result = run("CUST-005", client)
    assert result.disposition is Disposition.CUSTOMER_DECLINED
    assert "retry_payment" not in result.tools_called
    assert store.get_payment_attempts("CUST-005") == []
    assert store.get_state("CUST-005").status is AutopayStatus.FAILED


def test_cust_005_asks_only_once(client: TestClient) -> None:
    transcript = run("CUST-005", client).transcript()
    assert transcript.count("try that card again now?") == 1


# ---------------------------------------------------------------------------
# Scenario 6 - escalation
# ---------------------------------------------------------------------------


def test_cust_006_escalates(client: TestClient) -> None:
    result = run("CUST-006", client)

    assert result.disposition is Disposition.ESCALATED
    assert "escalate_to_human" in result.tools_called
    assert result.final_state["escalated"] is True
    assert result.final_state["escalation_ticket"].startswith("TCK-")
    assert len(store.get_escalations("CUST-006")) == 1


def test_cust_006_does_not_spend_an_attempt_on_a_closed_account(
    client: TestClient,
) -> None:
    result = run("CUST-006", client)
    assert "status=not_attempted" in result.transcript()
    assert store.get_payment_attempts("CUST-006") == []


def test_cust_006_does_not_claim_a_human_has_joined(client: TestClient) -> None:
    transcript = run("CUST-006", client).transcript().lower()
    assert "will call you back" in transcript
    for false_claim in ("transferring you", "putting you through", "is joining"):
        assert false_claim not in transcript


# ---------------------------------------------------------------------------
# Scenario 7 - verification failure
# ---------------------------------------------------------------------------


def test_cust_007_fails_verification(client: TestClient) -> None:
    result = run("CUST-007", client)

    assert result.disposition is Disposition.VERIFICATION_FAILED
    assert result.tools_called.count("verify_identity") == 2
    assert result.final_state["identity_verified"] is False
    assert store.get_payment_attempts("CUST-007") == []


def test_cust_007_discloses_nothing_about_the_account(client: TestClient) -> None:
    customer = store.get_customer("CUST-007")
    transcript = run("CUST-007", client).transcript()

    assert "get_failed_payment_details" not in transcript
    assert customer.failed_payment.amount_spoken not in transcript
    assert customer.verification.expected_answer not in transcript
    assert customer.autopay.primary_method.last4 not in transcript


def test_cust_007_still_gets_offered_a_human(client: TestClient) -> None:
    """Escalation without verification is deliberate, and exercised here."""
    result = run("CUST-007", client)
    assert "escalate_to_human" in result.tools_called
    assert store.get_escalations("CUST-007")


# ---------------------------------------------------------------------------
# Scenario 8 - disputed charge
# ---------------------------------------------------------------------------


def test_cust_008_escalates_a_dispute_without_retrying(client: TestClient) -> None:
    result = run("CUST-008", client)

    assert result.disposition is Disposition.ESCALATED
    assert "retry_payment" not in result.tools_called
    assert store.get_payment_attempts("CUST-008") == []
    assert store.get_escalations("CUST-008")[0]["reason"] == (
        "customer disputes the amount"
    )


def test_cust_008_does_not_argue_with_the_customer(client: TestClient) -> None:
    transcript = run("CUST-008", client).transcript().lower()
    assert "i understand" in transcript
    assert "won't try the payment while that's in question" in transcript


# ---------------------------------------------------------------------------
# Scenario 9 - retry declines again
# ---------------------------------------------------------------------------


def test_cust_009_ends_unresolved_without_a_false_claim(client: TestClient) -> None:
    result = run("CUST-009", client)

    assert result.disposition is Disposition.UNRESOLVED
    assert "retry_payment" in result.tools_called
    assert result.final_state["payment_status"] == "failed"

    transcript = result.transcript().lower()
    assert "didn't go through" in transcript
    assert "is paid" not in transcript


def test_cust_009_applies_no_pressure(client: TestClient) -> None:
    transcript = run("CUST-009", client).transcript().lower()
    assert "no problem at all" in transcript
    for pressure in ("must pay", "you need to pay", "suspend", "legal"):
        assert pressure not in transcript


# ---------------------------------------------------------------------------
# Scenario 10 - do not call
# ---------------------------------------------------------------------------


def test_cust_010_honours_do_not_call_immediately(client: TestClient) -> None:
    result = run("CUST-010", client)

    assert result.disposition is Disposition.DO_NOT_CALL
    assert result.tools_called == ["log_disposition"], (
        "nothing else should happen after an opt-out"
    )
    assert result.final_state["do_not_call"] is True
    assert store.get_state("CUST-010").status is AutopayStatus.DO_NOT_CALL


def test_cust_010_never_verifies_or_retries(client: TestClient) -> None:
    result = run("CUST-010", client)
    assert "verify_identity" not in result.tools_called
    assert "retry_payment" not in result.tools_called
    assert store.get_payment_attempts("CUST-010") == []


def test_cust_010_does_not_push_back(client: TestClient) -> None:
    transcript = run("CUST-010", client).transcript().lower()
    assert "won't call you again" in transcript
    for pushback in ("before you go", "just one thing", "are you sure"):
        assert pushback not in transcript


# ---------------------------------------------------------------------------
# The whole set
# ---------------------------------------------------------------------------


def test_every_scenario_reaches_its_expected_disposition(client: TestClient) -> None:
    results = run_all(client)
    failures = [
        f"{result.scenario.customer_id}: got "
        f"{result.disposition.value if result.disposition else None}, "
        f"expected {result.scenario.expected_disposition.value}"
        for result in results
        if not result.passed
    ]
    assert not failures, failures
    assert len(results) == 10


def test_every_scenario_closes_its_session(client: TestClient) -> None:
    for result in run_all(client):
        assert result.tools_called[-1] == "log_disposition", result.scenario.customer_id
        assert result.final_state["status"] == "closed"


def test_results_are_deterministic_across_runs(client: TestClient) -> None:
    """Two identical runs must produce identical transcripts."""
    first = [result.transcript() for result in run_all(client)]
    store.reset_runtime()
    second = [result.transcript() for result in run_all(client)]
    assert first == second


def test_transcripts_are_readable(client: TestClient) -> None:
    result = run("CUST-001", client)
    transcript = result.transcript()

    assert "Scenario: CUST-001" in transcript
    assert "Agent:" in transcript
    assert "Customer:" in transcript
    assert "Tool: verify_identity" in transcript
    assert "Result:" in transcript
    assert "Disposition: payment_recovered" in transcript
    assert all(ord(character) < 128 for character in transcript), (
        "transcripts should stay ASCII so they print on any console"
    )


def test_no_transcript_leaks_a_verification_answer_it_should_not_know(
    client: TestClient,
) -> None:
    """A transcript may contain the code the caller said, but no other."""
    answers = {
        customer.customer_id: customer.verification.expected_answer
        for customer in store.list_customers()
    }
    for result in run_all(client):
        transcript = result.transcript()
        own = answers[result.scenario.customer_id]
        for customer_id, answer in answers.items():
            if customer_id == result.scenario.customer_id or answer == own:
                continue
            assert answer not in transcript, (
                f"{result.scenario.customer_id} leaked {customer_id}'s answer"
            )


def test_simulator_only_calls_declared_tools(client: TestClient) -> None:
    from app.agent.tool_specs import tool_names

    known = set(tool_names())
    for result in run_all(client):
        assert set(result.tools_called) <= known, result.scenario.customer_id


def test_scenarios_use_the_real_backend_outcomes(client: TestClient) -> None:
    """A scenario must not assert a success the mock processor would not give."""
    for result in run_all(client):
        customer = store.get_customer(result.scenario.customer_id)
        if result.disposition is Disposition.PAYMENT_RECOVERED:
            assert customer.scenario.mock_retry_outcome.value in (
                "succeed",
                "succeed_on_backup",
            ), f"{customer.customer_id} cannot actually recover"


# ---------------------------------------------------------------------------
# Offline guarantee and CLI
# ---------------------------------------------------------------------------


def test_the_simulator_makes_no_external_connections(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Loopback stays open for the in-process client; everything else refused."""
    real_connect = socket.socket.connect
    real_getaddrinfo = socket.getaddrinfo
    loopback = {"127.0.0.1", "::1", "localhost", "testserver"}

    def guarded_connect(self: socket.socket, address: object) -> object:
        host = str(address[0]) if isinstance(address, tuple) and address else str(address)
        if host not in loopback:
            raise AssertionError(f"the simulator attempted to reach {host}")
        return real_connect(self, address)

    def guarded_getaddrinfo(host: object, *args: object, **kwargs: object) -> object:
        if host is not None and str(host) not in loopback:
            raise AssertionError(f"the simulator attempted to resolve {host}")
        return real_getaddrinfo(host, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket, "getaddrinfo", guarded_getaddrinfo)

    results = run_all(client)
    assert all(result.passed for result in results)


def test_the_simulator_imports_no_provider_or_llm_sdk() -> None:
    text = Path("scripts/simulate.py").read_text(encoding="utf-8").lower()
    for banned in (
        "import elevenlabs",
        "import twilio",
        "import openai",
        "import anthropic",
        "from elevenlabs",
        "from anthropic",
        "import requests",
    ):
        assert banned not in text, f"the simulator must not {banned}"


def test_the_simulator_does_not_reimplement_the_backend() -> None:
    """It must go through the tool endpoints, not the store or processor."""
    text = Path("scripts/simulate.py").read_text(encoding="utf-8")
    assert "client.post(\n            f\"/tools/{name}\"," in text
    for forbidden in (
        "mock_processor.retry_payment",
        "store.record_payment_attempt",
        "store.record_disposition",
        "store.record_escalation",
        "store.record_payment_link",
        "store.record_scheduled_retry",
    ):
        assert forbidden not in text, f"the simulator must not call {forbidden}"


def test_cli_list_exits_cleanly(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--list"]) == 0
    output = capsys.readouterr().out
    assert "CUST-001" in output
    assert "payment_recovered" in output


def test_cli_runs_one_scenario(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["-s", "CUST-001", "--quiet"]) == 0
    assert "1/1 scenarios reached the expected disposition" in capsys.readouterr().out


def test_cli_runs_every_scenario(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--quiet", "--reset"]) == 0
    assert "10/10 scenarios reached the expected disposition" in capsys.readouterr().out


def test_cli_rejects_an_unknown_scenario(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["-s", "CUST-999"]) == 2
    assert "No scenario for" in capsys.readouterr().err


def test_cli_prints_a_transcript_by_default(capsys: pytest.CaptureFixture[str]) -> None:
    main(["-s", "CUST-001"])
    output = capsys.readouterr().out
    assert "Agent:" in output
    assert "Tool: verify_identity" in output


def test_the_dashboard_sees_simulator_state(client: TestClient) -> None:
    """The control plane must reflect what the simulator did."""
    result = run("CUST-001", client)

    state = client.get(f"/api/sessions/{result.session_id}").json()
    assert state["channel"] == "simulator"
    assert state["payment_status"] == "recovered"
    assert state["disposition"] == "payment_recovered"

    events = client.get(f"/api/sessions/{result.session_id}/events").json()
    types = [event["event_type"] for event in events]
    assert types[0] == "session_created"
    assert "payment_retry_succeeded" in types
    assert types[-1] == "disposition_logged"

    rows = {row["customer_id"]: row for row in client.get("/api/customers").json()}
    assert rows["CUST-001"]["payment_status"] == "recovered"


def test_intents_are_all_used_by_some_scenario() -> None:
    """No dead intent in the enum."""
    used = {scenario.intent for scenario in SCENARIOS}
    unused = set(Intent) - used
    assert unused == {Intent.NEEDS_HUMAN}, (
        "only NEEDS_HUMAN is expected to be unused by the ten scenarios"
    )


def test_needs_human_intent_works_when_exercised(client: TestClient) -> None:
    """The one unused intent still has to function - it is in the prompt."""
    from scripts.simulate import Scenario

    scenario = Scenario(
        customer_id="CUST-001",
        title="Caller asks for a person immediately",
        intent=Intent.NEEDS_HUMAN,
        expected_disposition=Disposition.ESCALATED,
    )
    result = run_scenario(scenario, client)

    assert result.passed
    assert "verify_identity" not in result.tools_called
    assert "escalate_to_human" in result.tools_called
