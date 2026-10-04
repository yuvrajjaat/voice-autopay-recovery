"""Offline conversation simulator.

Runs the whole autopay-recovery conversation with no voice provider, no LLM,
and no network. Its job is to prove the *decision logic* — when to verify, when
a retry is pointless, when to fall back to a link, when to escalate, what
disposition a call earns — before a single paid voice minute is spent.

How it avoids being a second implementation
-------------------------------------------
It calls the real ``/tools/*`` endpoints through FastAPI's in-process test
client. Same routes, same guards, same mock processor, same JSON that
ElevenLabs receives in the live demo — just no socket in the middle. Nothing
about payment retries, verification, scheduling, links, escalation, or
dispositions is reimplemented here.

What stands in for the LLM
--------------------------
A deterministic decision engine, driven by two things: the scripted intent of
the fictional caller, and the ``next_action`` each tool returns. That is
exactly the contract the system prompt asks the model to follow, so a passing
scenario here means the prompt has a coherent path to walk. What it cannot
test is whether the model *will* follow it — that needs real voice minutes
and a live conversation.

Usage::

    python scripts/simulate.py                  # every scenario
    python scripts/simulate.py -s CUST-001      # one scenario
    python scripts/simulate.py --list           # what is available
    python scripts/simulate.py --save           # write demo/transcripts/
    python scripts/simulate.py --reset          # clear runtime state first
"""

from __future__ import annotations

import argparse
import secrets
import sys
from dataclasses import dataclass, field
from datetime import date, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

# Allow `python scripts/simulate.py` from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient  # noqa: E402

from app import store  # noqa: E402
from app.agent.prompt import build_first_message  # noqa: E402
from app.agent.tool_specs import tool_names  # noqa: E402
from app.config import settings  # noqa: E402
from app.main import app  # noqa: E402
from app.models import Customer, Disposition  # noqa: E402

TRANSCRIPT_DIR = Path(__file__).resolve().parent.parent / "demo" / "transcripts"


class Intent(str, Enum):
    """What the fictional caller wants. Stands in for a real human's goal."""

    RETRY_NOW = "retry_now"
    PAYMENT_LINK = "payment_link"
    SCHEDULE = "schedule"
    NEEDS_HUMAN = "needs_human"
    DISPUTES_CHARGE = "disputes_charge"
    DECLINES = "declines"
    DO_NOT_CALL = "do_not_call"
    CANNOT_VERIFY = "cannot_verify"


@dataclass(frozen=True)
class Scenario:
    """One scripted conversation against one fictional customer."""

    customer_id: str
    title: str
    intent: Intent
    expected_disposition: Disposition
    answers: tuple[str, ...] = ()
    note: str = ""


@dataclass
class Line:
    """One line of transcript."""

    speaker: str
    text: str

    def render(self) -> str:
        if self.speaker == "tool":
            return f"  Tool: {self.text}"
        if self.speaker == "result":
            return f"  Result: {self.text}"
        return f"{self.speaker.capitalize()}: {self.text}"


@dataclass
class Result:
    """Outcome of running one scenario."""

    scenario: Scenario
    lines: list[Line] = field(default_factory=list)
    tools_called: list[str] = field(default_factory=list)
    disposition: Disposition | None = None
    final_state: dict[str, Any] = field(default_factory=dict)
    session_id: str = ""

    @property
    def passed(self) -> bool:
        return self.disposition is self.scenario.expected_disposition

    def transcript(self) -> str:
        header = f"Scenario: {self.scenario.customer_id} - {self.scenario.title}"
        parts = ["=" * 66, header, "=" * 66, ""]
        if self.scenario.note:
            parts += [f"({self.scenario.note})", ""]
        parts += [line.render() for line in self.lines]
        parts += [
            "",
            "-" * 66,
            f"Disposition: {self.disposition.value if self.disposition else 'none'}"
            f"   expected: {self.scenario.expected_disposition.value}"
            f"   {'OK' if self.passed else 'MISMATCH'}",
            f"Tools called: {', '.join(self.tools_called) or 'none'}",
        ]
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# The conversation harness
# ---------------------------------------------------------------------------


class Conversation:
    """Drives one session, recording transcript lines as it goes."""

    def __init__(self, client: TestClient, customer: Customer, session_id: str) -> None:
        self.client = client
        self.customer = customer
        self.session_id = session_id
        self.lines: list[Line] = []
        self.tools_called: list[str] = []

    def agent(self, text: str) -> None:
        self.lines.append(Line("agent", text))

    def customer_says(self, text: str) -> None:
        self.lines.append(Line("customer", text))

    def call(self, name: str, **arguments: Any) -> dict[str, Any]:
        """Invoke a real tool endpoint and record it.

        Guards against calling a tool the specs do not declare, which is how a
        typo here would otherwise become a silent 404 against the live agent.
        """
        if name not in tool_names():
            raise ValueError(f"{name!r} is not a declared tool")

        payload = {"session_id": self.session_id, **arguments}
        response = self.client.post(
            f"/tools/{name}",
            json=payload,
            headers={"X-Tool-Secret": settings.tool_shared_secret or ""},
        )
        body = response.json()
        self.tools_called.append(name)

        shown = ", ".join(f"{key}={value}" for key, value in arguments.items())
        self.lines.append(Line("tool", f"{name}({shown})" if shown else f"{name}()"))
        self.lines.append(Line("result", _summarise(name, response.status_code, body)))
        return body


def _summarise(name: str, status_code: int, body: dict[str, Any]) -> str:
    """One readable line per tool result, for the transcript."""
    if status_code >= 400:
        return f"HTTP {status_code} {body.get('error')} - {body.get('message')}"

    if name == "verify_identity":
        return (
            f"verified={str(body['verified']).lower()}"
            f" attempts_remaining={body['attempts_remaining']}"
            f" locked={str(body['locked']).lower()}"
        )
    if name == "get_failed_payment_details":
        if not body["verified"]:
            return "verified=false - figures withheld"
        return (
            f"{body['amount']} {body['currency']} failed"
            f" ({body['failure_reason']})"
            f" retry_worth_attempting={str(body['retry_worth_attempting']).lower()}"
            f" next_action={body['next_action']}"
        )
    if name == "retry_payment":
        detail = f"status={body['status']}"
        if body.get("confirmation_number"):
            detail += f" confirmation={body['confirmation_number']}"
        if body.get("failure_reason"):
            detail += f" reason={body['failure_reason']}"
        return f"{detail} next_action={body['next_action']}"
    if name == "schedule_retry":
        return f"scheduled_for={body['scheduled_for']} ref={body['confirmation_number']}"
    if name == "send_payment_link":
        return f"link prepared delivered={str(body['delivered']).lower()}"
    if name == "escalate_to_human":
        return f"ticket={body['ticket_id']} callback={body['callback_window']}"
    if name == "log_disposition":
        return (
            f"disposition={body['disposition']}"
            f" do_not_call={str(body['do_not_call']).lower()}"
        )
    return str(body)


# ---------------------------------------------------------------------------
# The decision engine - what the LLM does in production
# ---------------------------------------------------------------------------


def _open_call(conv: Conversation, scenario: Scenario) -> None:
    conv.agent(build_first_message(conv.customer))

    if scenario.intent is Intent.DO_NOT_CALL:
        conv.customer_says("No. Take me off your list and don't call again.")
        return
    if scenario.intent is Intent.NEEDS_HUMAN:
        conv.customer_says("I'd rather just speak to a real person, please.")
        return
    conv.customer_says("Yes, now's fine. What's this about?")
    conv.agent(
        "An automatic payment on your account didn't go through. "
        "Before I say anything more, can you confirm the postal code on the account?"
    )


def _verify(conv: Conversation, scenario: Scenario) -> bool:
    """Run the verification exchange. Returns whether identity was confirmed."""
    answers = scenario.answers or (conv.customer.scenario.caller_provides_answer,)

    for index, answer in enumerate(answers):
        conv.customer_says(f"It's {answer}.")
        result = conv.call("verify_identity", postal_code=answer)

        if result.get("verified"):
            conv.agent("Thank you, that matches.")
            return True
        if result.get("locked"):
            conv.agent(
                "I'm sorry, I can't confirm the account, so I'm not able to discuss "
                "it on this call. Please use the number on your statement."
            )
            return False
        if index < len(answers) - 1:
            conv.agent("That doesn't match what I have. Could you try once more?")

    return False


def _explain(conv: Conversation) -> dict[str, Any]:
    details = conv.call("get_failed_payment_details")
    conv.agent(
        f"Your automatic payment of ${details['amount']} due "
        f"{details['due_date']} didn't go through - "
        f"{details['failure_explanation']}. Your service is still active."
    )
    return details


def _close(conv: Conversation, disposition: Disposition, notes: str | None = None) -> Disposition:
    conv.call("log_disposition", disposition=disposition.value, notes=notes)
    conv.agent("Thanks for your time today. Goodbye.")
    return disposition


def _do_retry(conv: Conversation, details: dict[str, Any]) -> Disposition:
    """The retry branch, routed entirely by what the backend returns."""
    conv.agent("Would you like me to try that card again now?")
    conv.customer_says("Yes, please try it again.")

    result = conv.call("retry_payment", payment_method="primary")

    if result["status"] == "paid":
        conv.agent(
            f"That's gone through. ${result['amount']} is paid, and your "
            f"confirmation number is {result['confirmation_number']}."
        )
        return _close(conv, Disposition.PAYMENT_RECOVERED, "recovered on the first retry")

    # The backend decides what is worth trying next; the agent follows it.
    next_action = result["next_action"]

    if next_action == "offer_backup_method":
        conv.agent(
            "That one was declined. There's another card on your account - "
            "would you like me to try that instead?"
        )
        conv.customer_says("Yes, go ahead and use that one.")
        backup = conv.call("retry_payment", payment_method="backup")
        if backup["status"] == "paid":
            conv.agent(
                f"That worked. ${backup['amount']} is paid, confirmation "
                f"{backup['confirmation_number']}."
            )
            return _close(conv, Disposition.PAYMENT_RECOVERED, "recovered on the backup card")
        conv.agent("That was declined as well, I'm afraid.")
        return _offer_link(conv, Disposition.UNRESOLVED)

    if next_action == "payment_link":
        conv.agent(
            "I won't keep trying that card - it won't go through. "
            "I can put a secure payment link on your account instead."
        )
        return _offer_link(conv, Disposition.PAYMENT_LINK_PREPARED)

    if next_action == "escalate":
        conv.agent(
            "That can't be fixed from my side, so I'd rather a colleague looked at it."
        )
        return _do_escalate(conv, "retry is not possible on the card on file")

    # Non-terminal decline: offer help, and take no for an answer.
    conv.agent(
        "That didn't go through either. I can put a payment link on the account, "
        "or make a note to try again another day - which would you prefer?"
    )
    conv.customer_says("Neither right now. I'll sort it out myself.")
    conv.agent("That's no problem at all.")
    return _close(conv, Disposition.UNRESOLVED, "retry declined; customer will self-serve")


def _offer_link(conv: Conversation, disposition: Disposition) -> Disposition:
    conv.customer_says("Yes, that's easier. Send me the link.")
    conv.call("send_payment_link", channel="email")
    conv.agent(
        "I've put a secure payment link on your account for the email we have "
        "on file - nothing's been sent out, it's there for you to use."
    )
    return _close(conv, disposition, "secure link prepared")


def _do_escalate(conv: Conversation, reason: str, notes: str | None = None) -> Disposition:
    result = conv.call("escalate_to_human", reason=reason, notes=notes)
    conv.agent(
        f"I've passed this to a colleague - your reference is "
        f"{result['ticket_id']}, and someone will call you back "
        f"{result['callback_window']}."
    )
    return _close(conv, Disposition.ESCALATED, reason)


def run_scenario(scenario: Scenario, client: TestClient | None = None) -> Result:
    """Run one scenario end to end against the real tool layer."""
    owns_client = client is None
    client = client or _make_client()

    customer = store.get_customer(scenario.customer_id)
    created = client.post(
        "/api/sessions", json={"customer_id": scenario.customer_id, "channel": "simulator"}
    )
    created.raise_for_status()
    session_id = created.json()["session_id"]

    conv = Conversation(client, customer, session_id)
    result = Result(scenario=scenario, session_id=session_id)

    try:
        disposition = _drive(conv, scenario)
    finally:
        result.lines = conv.lines
        result.tools_called = conv.tools_called

    result.disposition = disposition
    result.final_state = client.get(f"/api/sessions/{session_id}").json()

    if owns_client:
        client.close()
    return result


def _drive(conv: Conversation, scenario: Scenario) -> Disposition:
    """Choose the path. Intent and tool results decide, not a fixed script."""
    _open_call(conv, scenario)

    # Rule 7: an opt-out short-circuits everything, with no retry attempted.
    if scenario.intent is Intent.DO_NOT_CALL:
        conv.agent(
            "Understood - I've recorded that and we won't call you again. "
            "Sorry to have bothered you."
        )
        return _close(conv, Disposition.DO_NOT_CALL, "customer asked not to be called")

    # Escalation is available before verification, by design.
    if scenario.intent is Intent.NEEDS_HUMAN:
        conv.agent("Of course - let me get a colleague to call you back.")
        return _do_escalate(conv, "customer asked to speak to a person")

    if not _verify(conv, scenario):
        conv.agent("I can also have a colleague call you back about this, if that helps.")
        conv.customer_says("Yes, do that.")
        conv.call(
            "escalate_to_human",
            reason="identity could not be verified on the call",
            notes="caller could not confirm the postal code on the account",
        )
        conv.agent("I've made a note for a colleague to get in touch. Sorry about that.")
        return _close(conv, Disposition.VERIFICATION_FAILED, "postal code did not match")

    details = _explain(conv)

    if scenario.intent is Intent.DISPUTES_CHARGE:
        conv.customer_says(
            "That amount is wrong - I cancelled two of those lines back in August. "
            "I'm not paying it."
        )
        conv.agent(
            "I understand, and I won't try the payment while that's in question."
        )
        return _do_escalate(
            conv,
            "customer disputes the amount",
            "says two lines were cancelled in August",
        )

    if scenario.intent is Intent.DECLINES:
        conv.agent("Would you like me to try that card again now?")
        conv.customer_says("No thanks, I'll deal with it myself online.")
        conv.agent("That's absolutely fine - it'll stay on the account for you.")
        return _close(conv, Disposition.CUSTOMER_DECLINED, "customer will self-serve")

    if scenario.intent is Intent.SCHEDULE:
        conv.agent("Would you like me to try that card again now?")
        conv.customer_says("Not today - can you try again on payday?")
        conv.agent("Of course. What date would suit you?")
        when = date.today() + timedelta(days=11)
        conv.customer_says(f"The {when.day}th.")
        scheduled = conv.call("schedule_retry", requested_date=when.isoformat())
        conv.agent(
            f"I've made a note to try again on {scheduled['scheduled_for']} - "
            f"your reference is {scheduled['confirmation_number']}. "
            "It won't charge on its own; it's a note on the account."
        )
        return _close(conv, Disposition.RETRY_SCHEDULED, "customer asked to defer to payday")

    if scenario.intent is Intent.PAYMENT_LINK:
        conv.customer_says("I've got a different card here - let me read you the number.")
        conv.agent(
            "I'm sorry, I can't take card numbers over the phone. "
            "What I can do is put a secure payment link on your account "
            "so you can enter it yourself - would that work?"
        )
        return _offer_link(conv, Disposition.PAYMENT_LINK_PREPARED)

    return _do_retry(conv, details)


# ---------------------------------------------------------------------------
# Scenarios - one per fictional customer
# ---------------------------------------------------------------------------

SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        customer_id="CUST-001",
        title="Successful recovery",
        intent=Intent.RETRY_NOW,
        expected_disposition=Disposition.PAYMENT_RECOVERED,
        note="The headline path: verify, explain, agree, retry, approved.",
    ),
    Scenario(
        customer_id="CUST-002",
        title="Expired card, payment link prepared",
        intent=Intent.PAYMENT_LINK,
        expected_disposition=Disposition.PAYMENT_LINK_PREPARED,
        note="The customer offers a card number by voice; the agent must refuse.",
    ),
    Scenario(
        customer_id="CUST-003",
        title="Backup payment method",
        intent=Intent.RETRY_NOW,
        expected_disposition=Disposition.PAYMENT_RECOVERED,
        note="Primary declines; the backend offers the backup card and it approves.",
    ),
    Scenario(
        customer_id="CUST-004",
        title="Retry scheduled for payday",
        intent=Intent.SCHEDULE,
        expected_disposition=Disposition.RETRY_SCHEDULED,
        note="A note on the account, not a background job.",
    ),
    Scenario(
        customer_id="CUST-005",
        title="Customer declines help",
        intent=Intent.DECLINES,
        expected_disposition=Disposition.CUSTOMER_DECLINED,
        note="This card would have worked - the agent still takes no for an answer.",
    ),
    Scenario(
        customer_id="CUST-006",
        title="Closed account, escalated",
        intent=Intent.RETRY_NOW,
        expected_disposition=Disposition.ESCALATED,
        note="Unrecoverable by any retry, so the backend routes to a human.",
    ),
    Scenario(
        customer_id="CUST-007",
        title="Identity verification fails",
        intent=Intent.CANNOT_VERIFY,
        expected_disposition=Disposition.VERIFICATION_FAILED,
        answers=("19147", "19147"),
        note="Nothing about the account is disclosed; escalation is still offered.",
    ),
    Scenario(
        customer_id="CUST-008",
        title="Customer disputes the charge",
        intent=Intent.DISPUTES_CHARGE,
        expected_disposition=Disposition.ESCALATED,
        note="The agent does not defend the charge and does not retry it.",
    ),
    Scenario(
        customer_id="CUST-009",
        title="Retry declines again",
        intent=Intent.RETRY_NOW,
        expected_disposition=Disposition.UNRESOLVED,
        note="Third failure in a fortnight; no pressure applied, no false claim made.",
    ),
    Scenario(
        customer_id="CUST-010",
        title="Do not call",
        intent=Intent.DO_NOT_CALL,
        expected_disposition=Disposition.DO_NOT_CALL,
        note="Honoured immediately. No verification, no retry, no persuasion.",
    ),
)


def scenario_for(customer_id: str) -> Scenario:
    for scenario in SCENARIOS:
        if scenario.customer_id == customer_id.upper():
            return scenario
    raise KeyError(f"No scenario for {customer_id!r}")


def run_all(client: TestClient | None = None) -> list[Result]:
    """Run every scenario, sharing one client."""
    owns_client = client is None
    client = client or _make_client()
    try:
        return [run_scenario(scenario, client) for scenario in SCENARIOS]
    finally:
        if owns_client:
            client.close()


def _make_client() -> TestClient:
    """An in-process client, with a tool secret guaranteed to exist.

    The tool endpoints refuse every request when ``TOOL_SHARED_SECRET`` is
    unset — correctly, since an unconfigured deployment must not be open. The
    simulator is in-process and talks to no socket, so when no secret is
    configured it mints a random one for the run rather than shipping a
    hardcoded value.
    """
    if not settings.tool_shared_secret:
        settings.tool_shared_secret = secrets.token_urlsafe(32)
    return TestClient(app)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    # Windows consoles default to a legacy codepage, which mangles the
    # transcript and makes tools like grep treat the output as binary.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="Run the autopay recovery conversation offline.",
    )
    parser.add_argument(
        "-s", "--scenario", metavar="CUST-00N", help="run one scenario only"
    )
    parser.add_argument("--list", action="store_true", help="list the scenarios")
    parser.add_argument(
        "--save", action="store_true", help=f"write transcripts to {TRANSCRIPT_DIR}"
    )
    parser.add_argument(
        "--reset", action="store_true", help="clear runtime state before running"
    )
    parser.add_argument(
        "--quiet", action="store_true", help="summary table only, no transcripts"
    )
    arguments = parser.parse_args(argv)

    if arguments.list:
        print(f"{'CUSTOMER':<10} {'INTENT':<16} {'EXPECTED':<22} TITLE")
        print("-" * 86)
        for scenario in SCENARIOS:
            print(
                f"{scenario.customer_id:<10} {scenario.intent.value:<16} "
                f"{scenario.expected_disposition.value:<22} {scenario.title}"
            )
        return 0

    if arguments.reset:
        store.reset_runtime()
        print("runtime state cleared\n")

    try:
        scenarios = (
            [scenario_for(arguments.scenario)] if arguments.scenario else list(SCENARIOS)
        )
    except KeyError as error:
        print(error, file=sys.stderr)
        return 2

    client = _make_client()
    results = [run_scenario(scenario, client) for scenario in scenarios]
    client.close()

    if not arguments.quiet:
        for result in results:
            print(result.transcript())
            print()

    if arguments.save:
        TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
        for result in results:
            path = TRANSCRIPT_DIR / f"{result.scenario.customer_id.lower()}.txt"
            path.write_text(result.transcript() + "\n", encoding="utf-8")
        print(f"transcripts written to {TRANSCRIPT_DIR}\n")

    print(f"{'CUSTOMER':<10} {'DISPOSITION':<22} {'EXPECTED':<22} {'':<4} TOOLS")
    print("-" * 92)
    for result in results:
        print(
            f"{result.scenario.customer_id:<10} "
            f"{(result.disposition.value if result.disposition else '-'):<22} "
            f"{result.scenario.expected_disposition.value:<22} "
            f"{'OK' if result.passed else 'FAIL':<4} "
            f"{len(result.tools_called)}"
        )

    failures = [result for result in results if not result.passed]
    print()
    print(f"{len(results) - len(failures)}/{len(results)} scenarios reached the expected disposition")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
