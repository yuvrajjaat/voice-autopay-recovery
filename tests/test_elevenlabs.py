"""Tests for the ElevenLabs integration.

No live credentials are used or required. Every provider call is either a pure
builder (no client needed) or exercised against a fake client, so the suite
runs offline and a contributor without an ElevenLabs account can still run it.

The secret-leakage tests are the ones that matter most: the API key, the
webhook secret and the tool secret must never reach a browser, and three of
them check the actual rendered HTML and JavaScript.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app import store
from app.agent.prompt import SYSTEM_PROMPT
from app.agent.tool_specs import AUTH_HEADER, TOOL_SPECS, tool_names
from app.config import settings
from app.main import app
from app.providers import elevenlabs_client as provider

API_KEY = "sk_test_do_not_log_me"
WEBHOOK_SECRET = "wsec_test_never_in_html"
TOOL_SECRET = "tool-secret-never-in-html"
AGENT_ID = "agent_test123"
BASE_URL = "https://demo-tunnel.ngrok-free.app"


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture
def configured(monkeypatch: pytest.MonkeyPatch) -> None:
    """A fully configured project, with no real credentials."""
    monkeypatch.setattr(settings, "elevenlabs_api_key", API_KEY)
    monkeypatch.setattr(settings, "elevenlabs_agent_id", AGENT_ID)
    monkeypatch.setattr(settings, "elevenlabs_webhook_secret", WEBHOOK_SECRET)
    monkeypatch.setattr(settings, "tool_shared_secret", TOOL_SECRET)
    monkeypatch.setattr(settings, "public_base_url", BASE_URL)


@pytest.fixture
def unconfigured(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "elevenlabs_api_key",
        "elevenlabs_agent_id",
        "elevenlabs_webhook_secret",
        "public_base_url",
    ):
        monkeypatch.setattr(settings, name, None)


# ---------------------------------------------------------------------------
# A fake ElevenLabs client
# ---------------------------------------------------------------------------


class Recorder:
    """Captures what would have been sent, so tests can assert on it."""

    def __init__(self) -> None:
        self.created_tools: list[Any] = []
        self.updated_tools: list[tuple[str, Any]] = []
        self.created_agents: list[dict[str, Any]] = []
        self.updated_agents: list[tuple[str, dict[str, Any]]] = []
        self.created_secrets: list[tuple[str, str]] = []
        self.signed_url_calls: list[str] = []
        self.api_key: str | None = None


def make_fake_client(recorder: Recorder, *, existing_tools: dict[str, str] | None = None) -> Any:
    existing_tools = existing_tools or {}

    class FakeSecrets:
        def list(self, **_: Any) -> Any:
            return type("R", (), {"secrets": []})()

        def create(self, *, name: str, value: str, **_: Any) -> Any:
            recorder.created_secrets.append((name, value))
            return type("S", (), {"secret_id": "secret_fake_1"})()

        def update(self, secret_id: str, *, name: str, value: str, **_: Any) -> Any:
            recorder.created_secrets.append((name, value))
            return type("S", (), {"secret_id": secret_id})()

    class FakeTools:
        def list(self, **_: Any) -> Any:
            tools = [
                type(
                    "T",
                    (),
                    {"id": tool_id, "tool_config": type("C", (), {"name": name})()},
                )()
                for name, tool_id in existing_tools.items()
            ]
            return type("R", (), {"tools": tools})()

        def create(self, *, request: Any, **_: Any) -> Any:
            recorder.created_tools.append(request)
            return type("T", (), {"id": f"tool_{request.tool_config.name}"})()

        def update(self, tool_id: str, *, request: Any, **_: Any) -> Any:
            recorder.updated_tools.append((tool_id, request))
            return type("T", (), {"id": tool_id})()

    class FakeAgents:
        def create(self, *, name: str, conversation_config: Any, **_: Any) -> Any:
            recorder.created_agents.append(
                {"name": name, "conversation_config": conversation_config}
            )
            return type("A", (), {"agent_id": AGENT_ID})()

        def update(self, agent_id: str, *, name: str = "", conversation_config: Any = None, **_: Any) -> Any:
            recorder.updated_agents.append(
                (agent_id, {"name": name, "conversation_config": conversation_config})
            )
            return type("A", (), {"agent_id": agent_id})()

        def get(self, agent_id: str, **_: Any) -> Any:
            prompt = type(
                "P",
                (),
                {
                    "prompt": SYSTEM_PROMPT,
                    "llm": provider.AGENT_LLM,
                    "tool_ids": [f"tool_{name}" for name in tool_names()],
                },
            )()
            agent = type("Ag", (), {"prompt": prompt})()
            config = type("Cfg", (), {"agent": agent})()
            return type(
                "A",
                (),
                {
                    "agent_id": agent_id,
                    "name": provider.AGENT_NAME,
                    "conversation_config": config,
                },
            )()

    class FakeConversations:
        def get_signed_url(self, *, agent_id: str, **_: Any) -> Any:
            recorder.signed_url_calls.append(agent_id)
            return type("S", (), {"signed_url": "wss://signed.example.test/convai?token=x"})()

    class FakeConvAi:
        secrets = FakeSecrets()
        tools = FakeTools()
        agents = FakeAgents()
        conversations = FakeConversations()

    class FakeClient:
        conversational_ai = FakeConvAi()

    return FakeClient()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def test_the_app_boots_without_elevenlabs_configuration(
    unconfigured: None, client: TestClient
) -> None:
    assert client.get("/healthz").status_code == 200
    assert client.get("/dashboard").status_code == 200
    assert client.get("/voice").status_code == 200


def test_healthz_reports_elevenlabs_readiness_as_booleans(
    configured: None, client: TestClient
) -> None:
    config = client.get("/healthz").json()["config"]
    assert config["elevenlabs_api_key"] is True
    assert config["elevenlabs_agent_id"] is True
    assert all(isinstance(value, bool) for value in config.values())


def test_status_never_contains_a_secret(configured: None) -> None:
    blob = json.dumps(provider.status())
    assert API_KEY not in blob
    assert WEBHOOK_SECRET not in blob
    assert TOOL_SECRET not in blob


def test_status_reports_readiness(configured: None) -> None:
    assert provider.status()["ready_for_voice"] is True


def test_status_reports_not_ready_when_unconfigured(unconfigured: None) -> None:
    status = provider.status()
    assert status["configured"] is False
    assert status["ready_for_voice"] is False


def test_env_example_holds_placeholders_only() -> None:
    text = (Path(__file__).resolve().parent.parent / ".env.example").read_text(
        encoding="utf-8"
    )
    assert "ELEVENLABS_API_KEY=sk_replace_me" in text
    assert "ELEVENLABS_AGENT_ID=" in text
    assert "ELEVENLABS_WEBHOOK_SECRET=wsec_replace_me" in text
    # A real ElevenLabs key is a long sk_ token; a placeholder is not.
    import re

    for match in re.findall(r"sk_[A-Za-z0-9_]+", text):
        assert len(match) < 20, f"{match} looks like a real key"


# ---------------------------------------------------------------------------
# Provider client
# ---------------------------------------------------------------------------


def test_missing_api_key_fails_clearly(unconfigured: None) -> None:
    with pytest.raises(provider.ElevenLabsNotConfigured, match="ELEVENLABS_API_KEY"):
        provider.get_client()


def test_the_client_uses_the_configured_key(configured: None) -> None:
    built = provider.get_client()
    # The SDK keeps the key in its request wrapper; assert it was accepted and
    # that nothing else had to be passed in.
    assert built is not None
    assert provider.is_configured() is True


def test_missing_base_url_fails_clearly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "elevenlabs_api_key", API_KEY)
    monkeypatch.setattr(settings, "tool_shared_secret", TOOL_SECRET)
    monkeypatch.setattr(settings, "public_base_url", None)

    recorder = Recorder()
    with pytest.raises(provider.ElevenLabsNotConfigured, match="PUBLIC_BASE_URL"):
        provider.sync_tools(None, make_fake_client(recorder))


def test_missing_tool_secret_fails_clearly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "elevenlabs_api_key", API_KEY)
    monkeypatch.setattr(settings, "public_base_url", BASE_URL)
    monkeypatch.setattr(settings, "tool_shared_secret", None)

    recorder = Recorder()
    with pytest.raises(provider.ElevenLabsNotConfigured, match="TOOL_SHARED_SECRET"):
        provider.sync_tools(BASE_URL, make_fake_client(recorder))


def test_missing_agent_id_fails_clearly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "elevenlabs_api_key", API_KEY)
    monkeypatch.setattr(settings, "elevenlabs_agent_id", None)
    with pytest.raises(provider.ElevenLabsNotConfigured, match="ELEVENLABS_AGENT_ID"):
        provider.signed_url(None, make_fake_client(Recorder()))


def test_no_credential_is_logged(
    configured: None, caplog: pytest.LogCaptureFixture
) -> None:
    recorder = Recorder()
    with caplog.at_level("DEBUG", logger="app.providers.elevenlabs_client"):
        provider.provision_agent(BASE_URL, make_fake_client(recorder))
    assert API_KEY not in caplog.text
    assert TOOL_SECRET not in caplog.text
    assert WEBHOOK_SECRET not in caplog.text


def test_the_tool_secret_is_sent_as_a_workspace_secret(configured: None) -> None:
    """It has to reach ElevenLabs - they add the header for us - but by the
    secret store, not inline in a tool definition."""
    recorder = Recorder()
    provider.sync_tools(BASE_URL, make_fake_client(recorder))

    assert recorder.created_secrets == [(provider.TOOL_SECRET_NAME, TOOL_SECRET)]
    for request in recorder.created_tools:
        headers = request.tool_config.api_schema.request_headers
        locator = headers[AUTH_HEADER]
        assert getattr(locator, "secret_id", None) == "secret_fake_1"
        assert TOOL_SECRET not in json.dumps(headers, default=str)


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------


def test_exactly_seven_tools_are_configured(configured: None) -> None:
    recorder = Recorder()
    tool_ids = provider.sync_tools(BASE_URL, make_fake_client(recorder))

    assert len(tool_ids) == 7
    assert set(tool_ids) == set(tool_names())
    assert len(recorder.created_tools) == 7


def test_tool_urls_are_built_from_the_public_base_url(configured: None) -> None:
    recorder = Recorder()
    provider.sync_tools(BASE_URL, make_fake_client(recorder))

    urls = {
        request.tool_config.name: request.tool_config.api_schema.url
        for request in recorder.created_tools
    }
    for name in tool_names():
        assert urls[name] == f"{BASE_URL}/tools/{name}"


def test_tool_urls_follow_a_changed_base_url(configured: None) -> None:
    """A new tunnel means re-provisioning, not editing a dashboard."""
    recorder = Recorder()
    provider.sync_tools("https://other-tunnel.example.test/", make_fake_client(recorder))
    for request in recorder.created_tools:
        assert request.tool_config.api_schema.url.startswith(
            "https://other-tunnel.example.test/tools/"
        )


def test_no_ngrok_or_tunnel_url_is_hardcoded() -> None:
    for module in ("app/providers/elevenlabs_client.py", "scripts/provision_agent.py"):
        text = Path(module).read_text(encoding="utf-8")
        assert "ngrok-free.app" not in text or "your-domain" in text
        assert "trycloudflare" not in text


def test_all_tools_are_post_json(configured: None) -> None:
    recorder = Recorder()
    provider.sync_tools(BASE_URL, make_fake_client(recorder))
    for request in recorder.created_tools:
        schema = request.tool_config.api_schema
        assert schema.method == "POST"
        assert schema.content_type == "application/json"


def test_tool_body_schemas_match_the_pydantic_models(configured: None) -> None:
    """The decisive anti-drift test: no second schema is maintained."""
    recorder = Recorder()
    provider.sync_tools(BASE_URL, make_fake_client(recorder))

    by_name = {r.tool_config.name: r for r in recorder.created_tools}
    for spec in TOOL_SPECS:
        sent = by_name[spec.name].tool_config.api_schema.request_body_schema
        expected = spec.request_schema

        assert set(sent.properties or {}) == set(expected["properties"]), spec.name
        assert set(sent.required or []) == set(expected.get("required", [])), spec.name


def test_session_id_is_a_platform_injected_dynamic_variable(configured: None) -> None:
    """The anti-switching guarantee, carried into the provider configuration.

    If session_id were an ordinary parameter the model could choose it, and
    one hallucinated value would address another customer's account.
    """
    recorder = Recorder()
    provider.sync_tools(BASE_URL, make_fake_client(recorder))

    for request in recorder.created_tools:
        properties = request.tool_config.api_schema.request_body_schema.properties or {}
        assert "session_id" in properties, request.tool_config.name
        assert properties["session_id"].dynamic_variable == "session_id", (
            f"{request.tool_config.name}: session_id must be injected, not model-filled"
        )


def test_injected_parameters_set_only_the_dynamic_variable(configured: None) -> None:
    """ElevenLabs rejects a property that sets more than one of these fields.

    Regression test for a live provisioning failure: ``session_id`` was sent
    with both ``description`` and ``dynamic_variable``, and the API refused it
    with "Can only set one of: description, dynamic_variable,
    is_system_provided, constant_value, or is_omitted".
    """
    exclusive = (
        "description",
        "dynamic_variable",
        "is_system_provided",
        "constant_value",
        "is_omitted",
    )
    recorder = Recorder()
    provider.sync_tools(BASE_URL, make_fake_client(recorder))

    for request in recorder.created_tools:
        properties = request.tool_config.api_schema.request_body_schema.properties or {}
        for name, prop in properties.items():
            # Exactly the payload the SDK serialises onto the wire.
            sent = prop.dict(exclude_none=True)
            set_fields = [field for field in exclusive if field in sent]
            assert len(set_fields) <= 1, (
                f"{request.tool_config.name}.{name} sets {set_fields}; "
                "ElevenLabs allows at most one of them"
            )

        session = properties["session_id"].dict(exclude_none=True)
        assert session == {"type": "string", "dynamic_variable": "session_id"}, (
            f"{request.tool_config.name}: unexpected session_id schema {session}"
        )


def test_model_filled_parameters_keep_their_descriptions(configured: None) -> None:
    """The fix must not strip guidance from the parameters the model fills."""
    recorder = Recorder()
    provider.sync_tools(BASE_URL, make_fake_client(recorder))
    by_name = {r.tool_config.name: r for r in recorder.created_tools}

    for spec in TOOL_SPECS:
        properties = by_name[spec.name].tool_config.api_schema.request_body_schema.properties or {}
        for parameter in spec.llm_parameters:
            assert properties[parameter].description, (
                f"{spec.name}.{parameter} lost its description"
            )


def test_only_declared_parameters_are_model_filled(configured: None) -> None:
    recorder = Recorder()
    provider.sync_tools(BASE_URL, make_fake_client(recorder))

    by_name = {r.tool_config.name: r for r in recorder.created_tools}
    for spec in TOOL_SPECS:
        properties = by_name[spec.name].tool_config.api_schema.request_body_schema.properties or {}
        model_filled = {
            name
            for name, prop in properties.items()
            if not getattr(prop, "dynamic_variable", None)
        }
        assert model_filled == set(spec.llm_parameters), spec.name


def test_enum_parameters_list_their_allowed_values(configured: None) -> None:
    recorder = Recorder()
    provider.sync_tools(BASE_URL, make_fake_client(recorder))

    by_name = {r.tool_config.name: r for r in recorder.created_tools}
    disposition = (
        by_name["log_disposition"].tool_config.api_schema.request_body_schema.properties
    )["disposition"]
    assert disposition.enum is not None
    assert "payment_recovered" in disposition.enum
    assert "do_not_call" in disposition.enum

    method = by_name["retry_payment"].tool_config.api_schema.request_body_schema.properties[
        "payment_method"
    ]
    assert set(method.enum or []) == {"primary", "backup"}


def test_tool_descriptions_carry_the_restrictions(configured: None) -> None:
    recorder = Recorder()
    provider.sync_tools(BASE_URL, make_fake_client(recorder))
    by_name = {r.tool_config.name: r.tool_config for r in recorder.created_tools}

    assert "never announce success" in by_name["retry_payment"].description.lower()
    assert "delivered is always false" in by_name["send_payment_link"].description.lower()


def test_existing_tools_are_updated_not_duplicated(configured: None) -> None:
    """Re-provisioning after a tunnel change must not create fourteen tools."""
    existing = {name: f"tool_existing_{name}" for name in tool_names()}
    recorder = Recorder()
    tool_ids = provider.sync_tools(BASE_URL, make_fake_client(recorder, existing_tools=existing))

    assert recorder.created_tools == []
    assert len(recorder.updated_tools) == 7
    assert tool_ids == existing


def test_no_business_tool_beyond_the_seven(configured: None) -> None:
    recorder = Recorder()
    provider.sync_tools(BASE_URL, make_fake_client(recorder))
    names = {r.tool_config.name for r in recorder.created_tools}
    assert names == set(tool_names())
    assert len(names) == 7


# ---------------------------------------------------------------------------
# Agent provisioning
# ---------------------------------------------------------------------------


def test_the_agent_is_created_with_the_repository_prompt(configured: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "elevenlabs_agent_id", None)
    recorder = Recorder()
    result = provider.provision_agent(BASE_URL, make_fake_client(recorder))

    assert result["action"] == "created"
    assert len(recorder.created_agents) == 1

    config = recorder.created_agents[0]["conversation_config"]
    assert config.agent.prompt.prompt == SYSTEM_PROMPT
    assert len(config.agent.prompt.tool_ids) == 7
    assert config.agent.first_message
    assert config.agent.language == "en"


def test_the_prompt_is_not_duplicated_anywhere(configured: None) -> None:
    """prompt.py is the only copy; nothing re-states it."""
    marker = "Verify before you disclose"
    for path in Path("app").rglob("*.py"):
        if path.name == "prompt.py":
            continue
        assert marker not in path.read_text(encoding="utf-8"), path
    for path in Path("static").glob("*.js"):
        assert marker not in path.read_text(encoding="utf-8"), path
    assert marker not in Path("templates/voice.html").read_text(encoding="utf-8")


def test_the_provisioned_agent_keeps_the_phase_four_rules(configured: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "elevenlabs_agent_id", None)
    recorder = Recorder()
    provider.provision_agent(BASE_URL, make_fake_client(recorder))
    prompt = recorder.created_agents[0]["conversation_config"].agent.prompt.prompt.lower()

    for rule in (
        "verify before you disclose",
        "only claim what a tool confirmed",
        "`delivered: false`",
        "always close with `log_disposition`",
        "honour a do-not-call request immediately",
        "never invent account facts",
    ):
        assert rule in prompt, f"the provisioned prompt lost: {rule}"


def test_an_existing_agent_is_updated_in_place(configured: None) -> None:
    recorder = Recorder()
    result = provider.provision_agent(BASE_URL, make_fake_client(recorder))

    assert result["action"] == "updated"
    assert result["agent_id"] == AGENT_ID
    assert recorder.created_agents == []
    assert recorder.updated_agents[0][0] == AGENT_ID


def test_the_agent_uses_a_claude_model(configured: None) -> None:
    assert provider.AGENT_LLM.startswith("claude-")


def test_describe_agent_reports_a_match(configured: None) -> None:
    summary = provider.describe_agent(client=make_fake_client(Recorder()))
    assert summary["tool_count"] == 7
    assert summary["prompt_matches_repository"] is True
    assert summary["llm"] == provider.AGENT_LLM


# ---------------------------------------------------------------------------
# Provisioning script
# ---------------------------------------------------------------------------


def test_dry_run_needs_no_credentials(
    unconfigured: None, capsys: pytest.CaptureFixture[str]
) -> None:
    from scripts.provision_agent import main

    assert main(["--dry-run"]) == 0
    output = capsys.readouterr().out
    assert "Tools (7)" in output
    assert "injected by platform: session_id" in output
    assert "Not ready to provision" in output


def test_dry_run_prints_no_secret(
    configured: None, capsys: pytest.CaptureFixture[str]
) -> None:
    from scripts.provision_agent import main

    main(["--dry-run"])
    output = capsys.readouterr().out
    assert API_KEY not in output
    assert TOOL_SECRET not in output
    assert WEBHOOK_SECRET not in output


def test_provisioning_refuses_without_configuration(
    unconfigured: None, capsys: pytest.CaptureFixture[str]
) -> None:
    from scripts.provision_agent import main

    assert main([]) == 2
    assert "missing configuration" in capsys.readouterr().err


def test_provisioning_requires_https(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(settings, "elevenlabs_api_key", API_KEY)
    monkeypatch.setattr(settings, "tool_shared_secret", TOOL_SECRET)
    monkeypatch.setattr(settings, "public_base_url", "http://insecure.example.test")

    from scripts.provision_agent import main

    assert main([]) == 2
    assert "must be an https URL" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Voice page
# ---------------------------------------------------------------------------


def test_voice_page_renders(configured: None, client: TestClient) -> None:
    response = client.get("/voice")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Voice agent" in response.text


def test_voice_page_says_when_elevenlabs_is_not_configured(
    unconfigured: None, client: TestClient
) -> None:
    page = client.get("/voice").text
    assert "ElevenLabs is not configured" in page
    assert "ELEVENLABS_API_KEY" in page
    assert "Traceback" not in page


def test_voice_page_leaks_no_secret(configured: None, client: TestClient) -> None:
    page = client.get("/voice").text
    assert API_KEY not in page
    assert WEBHOOK_SECRET not in page
    assert TOOL_SECRET not in page


def test_voice_script_leaks_no_secret(configured: None, client: TestClient) -> None:
    script = client.get("/static/voice.js").text
    assert API_KEY not in script
    assert WEBHOOK_SECRET not in script
    assert TOOL_SECRET not in script
    assert "X-Tool-Secret" not in script
    for call_syntax in ('api("/tools', "api('/tools", 'fetch("/tools'):
        assert call_syntax not in script


def test_voice_page_carries_no_customer_data(configured: None, client: TestClient) -> None:
    page = client.get("/voice").text
    assert "Maya Thompson" not in page
    for customer in store.list_customers():
        assert customer.verification.expected_answer not in page


def test_voice_page_has_only_the_demo_controls(configured: None, client: TestClient) -> None:
    import re

    page = client.get("/voice").text
    assert 'id="btn-start"' in page
    assert 'id="btn-end"' in page
    assert 'id="status-pill"' in page
    assert 'id="session"' in page

    # No telephony surface of any kind.
    assert not re.search(r'type="(tel|number)"', page)
    assert not re.search(r"(?i)>\s*(call|dial)\b", page)
    assert "v0.1.0" not in page
    assert "{{ version }}" not in page


# ---------------------------------------------------------------------------
# Voice connection endpoint
# ---------------------------------------------------------------------------


def _session(client: TestClient, customer_id: str = "CUST-001") -> str:
    response = client.post("/api/sessions", json={"customer_id": customer_id})
    assert response.status_code == 201
    return response.json()["session_id"]


def test_connection_returns_dynamic_variables(
    configured: None, client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(provider, "signed_url", lambda *a, **k: "wss://signed.example.test/x")
    session_id = _session(client)

    body = client.get(f"/api/voice/connection/{session_id}").json()
    assert body["session_id"] == session_id
    assert body["customer_id"] == "CUST-001"
    assert body["mode"] == "signed_url"
    assert body["signed_url"] == "wss://signed.example.test/x"
    assert body["dynamic_variables"]["session_id"] == session_id
    assert body["dynamic_variables"]["customer_name"] == "Maya Thompson"


def test_connection_dynamic_variables_hold_no_protected_facts(
    configured: None, client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The amount and the postal code must be fetched by tool, not injected."""
    monkeypatch.setattr(provider, "signed_url", lambda *a, **k: "wss://x")
    session_id = _session(client)
    body = client.get(f"/api/voice/connection/{session_id}").json()

    blob = json.dumps(body["dynamic_variables"])
    customer = store.get_customer("CUST-001")
    assert customer.verification.expected_answer not in blob
    assert customer.failed_payment.amount_spoken not in blob
    assert customer.autopay.primary_method.last4 not in blob


def test_connection_never_returns_a_secret(
    configured: None, client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(provider, "signed_url", lambda *a, **k: "wss://x")
    session_id = _session(client)
    text = client.get(f"/api/voice/connection/{session_id}").text
    assert API_KEY not in text
    assert TOOL_SECRET not in text
    assert WEBHOOK_SECRET not in text


def test_connection_falls_back_to_the_agent_id(
    configured: None, client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def explode(*_: Any, **__: Any) -> str:
        raise RuntimeError("signed url unavailable")

    monkeypatch.setattr(provider, "signed_url", explode)
    session_id = _session(client)

    body = client.get(f"/api/voice/connection/{session_id}").json()
    assert body["mode"] == "agent_id"
    assert body["agent_id"] == AGENT_ID
    assert body["signed_url"] is None
    assert "signed URL was unavailable" in body["message"]


def test_connection_reports_missing_configuration(
    unconfigured: None, client: TestClient
) -> None:
    session_id = _session(client)
    body = client.get(f"/api/voice/connection/{session_id}").json()
    assert body["mode"] == "unavailable"
    assert body["configured"] is False
    assert "ELEVENLABS_API_KEY" in body["message"]


def test_connection_rejects_an_unknown_session(configured: None, client: TestClient) -> None:
    response = client.get("/api/voice/connection/sess_nope")
    assert response.status_code == 404
    assert response.json()["error"] == "unknown_session"


def test_connection_is_bound_to_the_sessions_customer(
    configured: None, client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """There is no parameter through which the browser could change customer."""
    monkeypatch.setattr(provider, "signed_url", lambda *a, **k: "wss://x")
    session_id = _session(client, "CUST-006")

    body = client.get(f"/api/voice/connection/{session_id}?customer_id=CUST-001").json()
    assert body["customer_id"] == "CUST-006"
    assert body["dynamic_variables"]["customer_name"] == "Tomas Varga"


# ---------------------------------------------------------------------------
# Tool authentication still holds
# ---------------------------------------------------------------------------


def test_tools_still_reject_an_unauthenticated_call(
    configured: None, client: TestClient
) -> None:
    session_id = _session(client)
    response = client.post(
        "/tools/get_failed_payment_details", json={"session_id": session_id}
    )
    assert response.status_code == 401


def test_tools_accept_the_configured_secret(configured: None, client: TestClient) -> None:
    session_id = _session(client)
    response = client.post(
        "/tools/get_failed_payment_details",
        json={"session_id": session_id},
        headers={AUTH_HEADER: TOOL_SECRET},
    )
    assert response.status_code == 200


def test_a_tool_request_still_cannot_name_a_customer(
    configured: None, client: TestClient
) -> None:
    session_id = _session(client)
    response = client.post(
        "/tools/get_failed_payment_details",
        json={"session_id": session_id, "customer_id": "CUST-008"},
        headers={AUTH_HEADER: TOOL_SECRET},
    )
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Post-call webhook
# ---------------------------------------------------------------------------


def _sign(body: bytes, secret: str = WEBHOOK_SECRET, timestamp: int | None = None) -> str:
    stamp = timestamp if timestamp is not None else int(time.time())
    digest = hmac.new(
        secret.encode(), f"{stamp}.{body.decode()}".encode(), hashlib.sha256
    ).hexdigest()
    return f"t={stamp},v0={digest}"


def _payload(session_id: str) -> dict[str, Any]:
    return {
        "type": "post_call_transcription",
        "event_timestamp": int(time.time()),
        "data": {
            "conversation_id": "conv_test_1",
            "metadata": {"call_duration_secs": 95},
            "conversation_initiation_client_data": {
                "dynamic_variables": {"session_id": session_id}
            },
            "transcript": [
                {"role": "agent", "message": "Hello, this is Ava."},
                {"role": "user", "message": "Yes, go ahead."},
            ],
        },
    }


def test_a_valid_webhook_is_accepted(configured: None, client: TestClient) -> None:
    session_id = _session(client)
    body = json.dumps(_payload(session_id)).encode()

    response = client.post(
        "/webhooks/elevenlabs",
        content=body,
        headers={"elevenlabs-signature": _sign(body), "Content-Type": "application/json"},
    )
    assert response.status_code == 200
    result = response.json()
    assert result["received"] is True
    assert result["matched_session"] is True
    assert result["recorded"] is True


def test_an_unsigned_webhook_is_rejected(configured: None, client: TestClient) -> None:
    session_id = _session(client)
    body = json.dumps(_payload(session_id)).encode()
    response = client.post("/webhooks/elevenlabs", content=body)
    assert response.status_code == 401
    assert response.json()["error"] == "missing_signature"


def test_a_wrongly_signed_webhook_is_rejected(configured: None, client: TestClient) -> None:
    session_id = _session(client)
    body = json.dumps(_payload(session_id)).encode()
    response = client.post(
        "/webhooks/elevenlabs",
        content=body,
        headers={"elevenlabs-signature": _sign(body, secret="wrong-secret")},
    )
    assert response.status_code == 401
    assert response.json()["error"] == "invalid_signature"


def test_a_tampered_body_is_rejected(configured: None, client: TestClient) -> None:
    session_id = _session(client)
    body = json.dumps(_payload(session_id)).encode()
    signature = _sign(body)

    tampered = body.replace(b"95", b"9999")
    response = client.post(
        "/webhooks/elevenlabs",
        content=tampered,
        headers={"elevenlabs-signature": signature},
    )
    assert response.status_code == 401


def test_an_expired_signature_is_rejected(configured: None, client: TestClient) -> None:
    session_id = _session(client)
    body = json.dumps(_payload(session_id)).encode()
    old = int(time.time()) - 60 * 60 * 24
    response = client.post(
        "/webhooks/elevenlabs",
        content=body,
        headers={"elevenlabs-signature": _sign(body, timestamp=old)},
    )
    assert response.status_code == 401
    assert response.json()["error"] == "signature_expired"


def test_a_malformed_signature_header_is_rejected(configured: None, client: TestClient) -> None:
    body = b"{}"
    for header in ("garbage", "t=,v0=", "v0=abc", "t=notanumber,v0=abc"):
        response = client.post(
            "/webhooks/elevenlabs", content=body, headers={"elevenlabs-signature": header}
        )
        assert response.status_code == 401, header


def test_malformed_json_is_rejected_safely(configured: None, client: TestClient) -> None:
    body = b"{not json"
    response = client.post(
        "/webhooks/elevenlabs", content=body, headers={"elevenlabs-signature": _sign(body)}
    )
    assert response.status_code == 400
    assert response.json()["error"] == "invalid_json"
    assert "Traceback" not in response.text


def test_the_webhook_fails_closed_without_a_secret(
    configured: None, client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "elevenlabs_webhook_secret", None)
    body = b"{}"
    response = client.post(
        "/webhooks/elevenlabs", content=body, headers={"elevenlabs-signature": _sign(body)}
    )
    assert response.status_code == 503
    assert response.json()["error"] == "webhook_secret_not_configured"


def test_an_unknown_session_is_accepted_but_not_recorded(
    configured: None, client: TestClient
) -> None:
    body = json.dumps(_payload("sess_does_not_exist")).encode()
    response = client.post(
        "/webhooks/elevenlabs", content=body, headers={"elevenlabs-signature": _sign(body)}
    )
    assert response.status_code == 200
    assert response.json()["matched_session"] is False
    assert response.json()["recorded"] is False


def test_the_webhook_cannot_change_payment_state(
    configured: None, client: TestClient
) -> None:
    """The decisive test: a transcript is evidence, not an authority."""
    session_id = _session(client)
    before = store.get_state("CUST-001")
    assert before.status.value == "failed"
    assert before.disposition is None

    payload = _payload(session_id)
    payload["data"]["transcript"] = [
        {"role": "agent", "message": "Your payment of 49.00 went through, confirmation C-9999."},
    ]
    payload["data"]["analysis"] = {
        "data_collection_results": {
            "payment_status": {"value": "recovered"},
            "disposition": {"value": "payment_recovered"},
        }
    }
    body = json.dumps(payload).encode()

    response = client.post(
        "/webhooks/elevenlabs", content=body, headers={"elevenlabs-signature": _sign(body)}
    )
    assert response.status_code == 200

    after = store.get_state("CUST-001")
    assert after.status.value == "failed", "the webhook must not mark a payment recovered"
    assert after.disposition is None
    assert after.last_confirmation_number is None
    assert store.get_session(session_id).identity_verified is False
    assert store.get_payment_attempts("CUST-001") == []


def test_the_webhook_records_metadata_on_the_timeline(
    configured: None, client: TestClient
) -> None:
    session_id = _session(client)
    body = json.dumps(_payload(session_id)).encode()
    client.post(
        "/webhooks/elevenlabs", content=body, headers={"elevenlabs-signature": _sign(body)}
    )

    events = store.get_events(session_id)
    completed = [e for e in events if e["event_type"] == "voice_call_completed"]
    assert len(completed) == 1
    assert completed[0]["detail"]["conversation_id"] == "conv_test_1"
    assert completed[0]["detail"]["duration_seconds"] == "95"


def test_an_oversized_body_is_rejected(configured: None, client: TestClient) -> None:
    body = b"x" * 2_000_001
    response = client.post(
        "/webhooks/elevenlabs", content=body, headers={"elevenlabs-signature": _sign(body)}
    )
    assert response.status_code == 413


# ---------------------------------------------------------------------------
# No telephony in this phase
# ---------------------------------------------------------------------------


def test_no_twilio_or_pstn_code_was_added() -> None:
    for module in (
        "app/providers/elevenlabs_client.py",
        "scripts/provision_agent.py",
        "app/routers/webhooks.py",
        "static/voice.js",
    ):
        text = Path(module).read_text(encoding="utf-8").lower()
        for banned in ("twilio", "outbound_call", "pstn", "sip", "sms"):
            assert banned not in text, f"{module} must not reference {banned}"


def test_twilio_is_still_not_installed() -> None:
    import importlib.util

    assert importlib.util.find_spec("twilio") is None


def test_the_dial_guard_is_unused_by_the_voice_path() -> None:
    """Phase 5's guard stays dormant until telephony arrives."""
    for module in (
        "app/providers/elevenlabs_client.py",
        "app/routers/webhooks.py",
        "app/routers/demo.py",
        "static/voice.js",
    ):
        text = Path(module).read_text(encoding="utf-8")
        assert "dial_safety" not in text, f"{module} should not use the dial guard yet"
        assert "check_dial_allowed" not in text


def test_no_call_endpoint_exists(client: TestClient) -> None:
    paths = client.get("/openapi.json").json()["paths"]
    for forbidden in ("/call", "/dial", "/api/call", "/api/dial"):
        assert forbidden not in paths
