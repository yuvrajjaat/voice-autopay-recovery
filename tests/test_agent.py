"""Tests for the agent prompt and the tool specifications.

The prompt tests are intentionally about *content*, not wording: they assert
that each hard rule is stated and that the tool names referenced actually
exist. They cannot prove a model will obey the prompt — only live voice
minutes can do that — but they do catch a rule being dropped or a tool being
renamed out from under it.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.agent import prompt as prompt_module
from app.agent.prompt import (
    AGENT_NAME,
    COMPANY_NAME,
    DYNAMIC_VARIABLES,
    SYSTEM_PROMPT,
    build_dynamic_variables,
    build_first_message,
)
from app.agent.tool_specs import (
    AUTH_HEADER,
    TOOL_PREFIX,
    TOOL_SPECS,
    ToolSpec,
    as_dicts,
    spec_by_name,
    tool_names,
)
from app import store
from app.main import app
from app.models import Disposition

LOWER_PROMPT = SYSTEM_PROMPT.lower()


# ---------------------------------------------------------------------------
# Prompt: identity and disclosure
# ---------------------------------------------------------------------------


def test_prompt_requires_identity_verification() -> None:
    assert "verify_identity" in SYSTEM_PROMPT
    assert "verify before you disclose" in LOWER_PROMPT


def test_prompt_forbids_disclosure_before_verification() -> None:
    """The rule must name what is withheld, not just gesture at it."""
    assert "until `verify_identity` has returned" in SYSTEM_PROMPT
    for withheld in ("amount", "due date", "card"):
        assert withheld in LOWER_PROMPT, f"{withheld} should be named as withheld"


def test_prompt_forbids_revealing_the_verification_answer() -> None:
    assert "never reveal or hint at the verification answer" in LOWER_PROMPT
    assert "read part of it back" in LOWER_PROMPT


def test_prompt_forbids_taking_payment_credentials_by_voice() -> None:
    assert "never take payment credentials by voice" in LOWER_PROMPT
    for banned in ("card number", "security code", "cvv", "one-time code", "password"):
        assert banned in LOWER_PROMPT, f"{banned} should be explicitly banned"


# ---------------------------------------------------------------------------
# Prompt: tool usage and truthfulness
# ---------------------------------------------------------------------------


def test_prompt_requires_tool_usage_rather_than_invention() -> None:
    collapsed = " ".join(SYSTEM_PROMPT.split())
    assert "never invent account facts" in LOWER_PROMPT
    assert "statuses come only from `get_failed_payment_details`" in collapsed
    assert "call these rather than guessing" in LOWER_PROMPT


def test_prompt_forbids_claiming_a_payment_succeeded_without_confirmation() -> None:
    assert 'only* if `retry_payment` returned' in SYSTEM_PROMPT
    assert '`status: "paid"`' in SYSTEM_PROMPT
    assert "only claim what a tool confirmed" in LOWER_PROMPT


def test_prompt_forbids_claiming_a_message_was_sent() -> None:
    """send_payment_link returns delivered: false, so the wording must match."""
    assert "`delivered: false`" in SYSTEM_PROMPT
    assert "never \"i've sent you a" in LOWER_PROMPT


def test_prompt_forbids_claiming_an_automatic_future_charge() -> None:
    assert "records a note" in LOWER_PROMPT
    assert 'never "it will charge automatically"' in LOWER_PROMPT


def test_prompt_forbids_claiming_a_human_has_joined() -> None:
    assert "never \"i'm transferring you" in LOWER_PROMPT
    assert "a colleague is joining" in LOWER_PROMPT


def test_prompt_requires_a_final_disposition() -> None:
    assert "always close with `log_disposition`" in LOWER_PROMPT
    assert "exactly one disposition" in LOWER_PROMPT


def test_prompt_lists_only_real_dispositions() -> None:
    """Every disposition the prompt offers must exist in the closed enum."""
    valid = {member.value for member in Disposition}
    section = SYSTEM_PROMPT.split("# Dispositions", 1)[1]
    mentioned = {value for value in valid if f"`{value}`" in section}
    assert mentioned == valid, f"prompt is missing: {sorted(valid - mentioned)}"


# ---------------------------------------------------------------------------
# Prompt: conduct
# ---------------------------------------------------------------------------


def test_prompt_honours_do_not_call_immediately() -> None:
    assert "honour a do-not-call request immediately" in LOWER_PROMPT
    assert "no pushback" in LOWER_PROMPT
    assert "no retry attempt" in LOWER_PROMPT


def test_prompt_forbids_pressure_and_threats() -> None:
    assert "never pressure, never threaten, never argue" in LOWER_PROMPT


def test_prompt_forbids_implying_suspension_without_evidence() -> None:
    assert "do not mention service suspension unless" in LOWER_PROMPT


def test_prompt_requires_admitting_it_is_automated() -> None:
    assert "you are not a human" in LOWER_PROMPT
    assert "say yes plainly" in LOWER_PROMPT


def test_prompt_asks_for_short_single_question_turns() -> None:
    assert "one or two short sentences" in LOWER_PROMPT
    assert "one question at a time" in LOWER_PROMPT


def test_prompt_covers_the_required_customer_intents() -> None:
    """Every intent from the phase brief needs an explicit instruction."""
    section = SYSTEM_PROMPT.split("# Handling what the customer says", 1)[1].lower()
    for cue in (
        "why are you calling",
        "now isn't a good time",
        "not giving you my postal code",
        "speak to a person",
        "different card",
        "try it again",
        "payday",
        "take me off your list",
    ):
        assert cue in section, f"the prompt does not handle: {cue}"


def test_prompt_tells_the_agent_to_follow_next_action() -> None:
    assert "every tool returns `next_action`" in LOWER_PROMPT
    assert "let what they say and what the tools return decide" in LOWER_PROMPT


def test_prompt_names_every_tool_that_exists() -> None:
    for name in tool_names():
        assert f"`{name}`" in SYSTEM_PROMPT, f"{name} is missing from the prompt"


def test_prompt_mentions_no_tool_that_does_not_exist() -> None:
    """Catch a renamed or invented tool before it reaches the provider.

    Scoped to the tool list, because elsewhere the prompt legitimately
    backticks next_action values and disposition names that are not tools.
    """
    import re

    section = SYSTEM_PROMPT.split("# Your tools", 1)[1].split("# The call", 1)[0]
    listed = set(re.findall(r"^- `([a-z_]+)`", section, flags=re.MULTILINE))
    assert listed == set(tool_names()), (
        f"tool list disagrees with the specs: {sorted(listed ^ set(tool_names()))}"
    )


# ---------------------------------------------------------------------------
# Opening line and dynamic variables
# ---------------------------------------------------------------------------


def test_first_message_is_short_and_discloses_automation() -> None:
    message = build_first_message()
    assert len(message.split()) < 60, "the opening line should stay brief"
    assert "automated" in message.lower()
    assert COMPANY_NAME in message
    assert AGENT_NAME in message
    assert message.rstrip().endswith("?"), "it should end by asking permission"


def test_first_message_reveals_no_account_facts() -> None:
    """Nothing is verified when the greeting is spoken, so it says nothing."""
    customer = store.get_customer("CUST-001")
    message = build_first_message(customer)
    assert customer.name in message
    assert "49" not in message
    assert "insufficient" not in message.lower()
    assert customer.verification.expected_answer not in message
    assert customer.autopay.primary_method.last4 not in message


def test_first_message_is_plain_ascii() -> None:
    """Typographic dashes mangle on a legacy Windows console."""
    message = build_first_message(store.get_customer("CUST-001"))
    assert all(ord(character) < 128 for character in message)


def test_dynamic_variables_carry_no_protected_facts() -> None:
    customer = store.get_customer("CUST-001")
    variables = build_dynamic_variables(customer, "sess_test")

    assert set(variables) == set(DYNAMIC_VARIABLES)
    assert variables["customer_name"] == customer.name
    assert variables["session_id"] == "sess_test"

    blob = " ".join(variables.values())
    assert customer.verification.expected_answer not in blob
    assert customer.failed_payment.amount_spoken not in blob
    assert customer.autopay.primary_method.last4 not in blob


def test_prompt_declares_every_dynamic_variable_it_uses() -> None:
    for name in DYNAMIC_VARIABLES:
        if name in ("company_name", "agent_name"):
            continue  # substituted at import time, not by the platform
        assert f"{{{{{name}}}}}" in SYSTEM_PROMPT, f"{name} is not used in the prompt"


def test_prompt_has_no_unresolved_placeholders() -> None:
    """A stray single-brace field would be a silent format bug."""
    assert "{agent_name}" not in SYSTEM_PROMPT
    assert "{company_name}" not in SYSTEM_PROMPT
    assert AGENT_NAME in SYSTEM_PROMPT
    assert COMPANY_NAME in SYSTEM_PROMPT


def test_prompt_module_imports_no_provider_sdk() -> None:
    source = prompt_module.__file__
    assert source is not None
    with open(source, encoding="utf-8") as handle:
        text = handle.read()
    lowered = text.lower()
    for banned in (
        "import elevenlabs",
        "from elevenlabs",
        "import twilio",
        "import openai",
        "import anthropic",
        "import httpx",
        "import requests",
    ):
        assert banned not in lowered, f"prompt.py must not {banned}"


# ---------------------------------------------------------------------------
# Tool specifications
# ---------------------------------------------------------------------------


def test_there_are_exactly_seven_tool_specs() -> None:
    assert len(TOOL_SPECS) == 7
    assert len(set(tool_names())) == 7


def test_every_spec_matches_a_real_route() -> None:
    """The decisive test: specs and endpoints cannot drift apart."""
    paths = TestClient(app).get("/openapi.json").json()["paths"]
    for spec in TOOL_SPECS:
        assert spec.path in paths, f"{spec.path} is not a route"
        assert spec.method.lower() in paths[spec.path], (
            f"{spec.path} does not accept {spec.method}"
        )


def test_every_tool_route_has_a_spec() -> None:
    """And the reverse: no endpoint is left undocumented."""
    paths = TestClient(app).get("/openapi.json").json()["paths"]
    routes = {path for path in paths if path.startswith(f"{TOOL_PREFIX}/")}
    assert routes == {spec.path for spec in TOOL_SPECS}


def test_all_specs_are_post() -> None:
    assert {spec.method for spec in TOOL_SPECS} == {"POST"}


def test_paths_are_derived_from_names() -> None:
    for spec in TOOL_SPECS:
        assert spec.path == f"{TOOL_PREFIX}/{spec.name}"


def test_request_schemas_come_from_the_real_models() -> None:
    """Generated, not restated, so a model change cannot leave a spec stale."""
    for spec in TOOL_SPECS:
        schema = spec.request_schema
        assert schema["type"] == "object"
        assert "session_id" in schema["properties"], spec.name
        assert schema.get("additionalProperties") is False, (
            f"{spec.name} should reject unexpected fields"
        )


def test_session_id_is_platform_injected_for_every_tool() -> None:
    """The anti-switching property, asserted at the spec level."""
    for spec in TOOL_SPECS:
        assert "session_id" in spec.injected_parameters, spec.name
        assert "session_id" not in spec.llm_parameters, (
            f"{spec.name} must not let the model choose the session"
        )


def test_llm_parameters_all_exist_in_the_request_schema() -> None:
    for spec in TOOL_SPECS:
        properties = set(spec.request_schema["properties"])
        for parameter in spec.llm_parameters:
            assert parameter in properties, f"{spec.name}.{parameter} is not a real field"


def test_specs_describe_usage_and_restrictions() -> None:
    for spec in TOOL_SPECS:
        assert spec.description.strip()
        assert spec.when_to_use.strip()
        assert spec.response_description.strip()


def test_tools_that_move_money_require_verification() -> None:
    for name in ("retry_payment", "schedule_retry", "send_payment_link"):
        assert spec_by_name(name).requires_verification, name


def test_escalation_and_disposition_work_without_verification() -> None:
    """Deliberate: a caller who cannot verify still needs a way out."""
    assert spec_by_name("escalate_to_human").requires_verification is False
    assert spec_by_name("log_disposition").requires_verification is False


def test_retry_spec_warns_against_announcing_unconfirmed_success() -> None:
    restrictions = " ".join(spec_by_name("retry_payment").restrictions).lower()
    assert "never announce success" in restrictions
    assert "'paid'" in restrictions


def test_payment_link_spec_states_nothing_is_transmitted() -> None:
    restrictions = " ".join(spec_by_name("send_payment_link").restrictions).lower()
    assert "delivered is always false" in restrictions


def test_schedule_spec_states_no_job_runs() -> None:
    restrictions = " ".join(spec_by_name("schedule_retry").restrictions).lower()
    assert "no job runs" in restrictions


def test_spec_urls_build_from_a_public_base() -> None:
    spec = spec_by_name("retry_payment")
    assert spec.url("https://example.ngrok-free.app") == (
        "https://example.ngrok-free.app/tools/retry_payment"
    )
    assert spec.url("https://example.ngrok-free.app/") == (
        "https://example.ngrok-free.app/tools/retry_payment"
    )


def test_as_dicts_is_serialisable_and_names_the_auth_header() -> None:
    import json

    payload = as_dicts()
    assert len(payload) == 7
    json.dumps(payload)  # must not raise
    assert all(entry["auth_header"] == AUTH_HEADER for entry in payload)


def test_specs_carry_no_secret_value() -> None:
    """The header name is in the repo; the value never is."""
    import json

    blob = json.dumps(as_dicts())
    assert AUTH_HEADER in blob
    assert "secret" not in blob.lower().replace("x-tool-secret", "").replace(
        "tool_shared_secret", ""
    )


def test_spec_lookup_rejects_an_unknown_name() -> None:
    import pytest

    with pytest.raises(KeyError):
        spec_by_name("refund_everything")


def test_tool_spec_is_immutable() -> None:
    import dataclasses

    import pytest

    spec = TOOL_SPECS[0]
    assert isinstance(spec, ToolSpec)
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.name = "something_else"  # type: ignore[misc]
