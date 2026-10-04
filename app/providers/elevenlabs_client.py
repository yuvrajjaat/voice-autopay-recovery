"""All ElevenLabs-specific code lives here.

Nothing else in the application constructs an ElevenLabs request. The rest of
the project deals in ``app.agent.prompt`` and ``app.agent.tool_specs``; this
module is the only place that knows those exist in a particular vendor's
shape.

What it does
------------
* Builds the seven webhook-tool definitions **from** ``tool_specs.py`` by
  translating our generated JSON Schema into ElevenLabs' schema objects. There
  is no second hand-maintained copy of the tool contract.
* Creates or updates the agent with the prompt from ``prompt.py``.
* Mints a short-lived signed URL so the browser can talk to a private agent
  without ever seeing the API key.

The one design point worth reading
----------------------------------
``session_id`` is translated into a **platform-injected dynamic variable**
(``LiteralJsonSchemaProperty(dynamic_variable="session_id")``), not an
LLM-filled parameter. ElevenLabs substitutes it from the conversation's
dynamic variables, so the model has no parameter through which it could aim a
tool call at a different customer's account. That is the Phase 2 session
binding carried intact into the provider.
"""

from __future__ import annotations

import logging
from typing import Any

from elevenlabs.client import ElevenLabs
from elevenlabs.types import (
    AgentConfig,
    ConvAiSecretLocator,
    ConversationalConfig,
    LiteralJsonSchemaProperty,
    ObjectJsonSchemaPropertyInput,
    PromptAgentApiModelOutput,
    ToolRequestModel,
    ToolRequestModelToolConfig_Webhook,
    WebhookToolApiSchemaConfigInput,
)

from app.agent.prompt import SYSTEM_PROMPT, build_first_message
from app.agent.tool_specs import AUTH_HEADER, TOOL_SPECS, ToolSpec
from app.config import settings

logger = logging.getLogger(__name__)

#: Shown in the ElevenLabs dashboard so the agent is identifiable.
AGENT_NAME = "Autopay Recovery (demo)"

#: Name of the workspace secret holding our X-Tool-Secret value.
TOOL_SECRET_NAME = "autopay_tool_shared_secret"

#: The reasoning model behind the agent.
AGENT_LLM = "claude-sonnet-5"

#: Low temperature: this agent reads back amounts and confirmation numbers,
#: where invention is the failure mode we care most about.
AGENT_TEMPERATURE = 0.3

#: Seconds ElevenLabs waits for one of our tool endpoints.
TOOL_TIMEOUT_SECONDS = 15

AGENT_LANGUAGE = "en"


class ElevenLabsNotConfigured(RuntimeError):
    """Raised when an operation needs configuration that is absent."""


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


def is_configured() -> bool:
    """Whether an API key is present. Never reveals the key itself."""
    return bool(settings.elevenlabs_api_key)


def get_client() -> ElevenLabs:
    """Build an API client, or fail with a clear message.

    The key is read at call time rather than import time, so the application
    boots fine without one and only the ElevenLabs-specific paths complain.
    """
    if not settings.elevenlabs_api_key:
        raise ElevenLabsNotConfigured(
            "ELEVENLABS_API_KEY is not set. Add it to .env (see .env.example)."
        )
    return ElevenLabs(api_key=settings.elevenlabs_api_key)


def status() -> dict[str, Any]:
    """A safe summary for the dashboard and the voice page.

    Booleans and identifiers only. The API key, the webhook secret, and the
    tool secret are never included — this is rendered into a web page.
    """
    return {
        "configured": is_configured(),
        "agent_id": settings.elevenlabs_agent_id,
        "public_base_url": settings.public_base_url,
        "tool_auth_configured": bool(settings.tool_shared_secret),
        "webhook_secret_configured": bool(settings.elevenlabs_webhook_secret),
        "ready_for_voice": bool(
            settings.elevenlabs_api_key
            and settings.elevenlabs_agent_id
            and settings.tool_shared_secret
            and settings.public_base_url
        ),
    }


# ---------------------------------------------------------------------------
# Schema translation
# ---------------------------------------------------------------------------

#: JSON Schema type -> the literal types ElevenLabs accepts.
_TYPE_MAP = {
    "string": "string",
    "integer": "integer",
    "number": "number",
    "boolean": "boolean",
}


def _resolve(schema: dict[str, Any], definitions: dict[str, Any]) -> dict[str, Any]:
    """Flatten one property schema into a plain type + enum.

    Handles the three shapes Pydantic actually emits for our models: a ``$ref``
    into ``$defs`` (an enum like ``Disposition``), an ``anyOf`` with a null
    branch (an optional field), and a plain typed property.
    """
    if "$ref" in schema:
        name = schema["$ref"].rsplit("/", 1)[-1]
        if name not in definitions:
            raise ValueError(f"unresolvable schema reference: {schema['$ref']}")
        return _resolve(definitions[name], definitions)

    if "anyOf" in schema:
        for branch in schema["anyOf"]:
            if branch.get("type") != "null":
                merged = _resolve(branch, definitions)
                # Keep the outer description, which Pydantic puts on the union.
                if schema.get("description") and "description" not in merged:
                    merged["description"] = schema["description"]
                return merged
        raise ValueError("an anyOf with only null branches cannot be sent")

    return dict(schema)


def _describe(spec: ToolSpec, name: str, resolved: dict[str, Any]) -> str:
    """Build the parameter description the model reads.

    Enum values are spelled out, because the model has to choose one by name
    and the backend rejects anything else.
    """
    parts: list[str] = []
    if resolved.get("description"):
        parts.append(str(resolved["description"]).strip().splitlines()[0])
    if resolved.get("format") == "date":
        parts.append("ISO date, YYYY-MM-DD.")
    if resolved.get("enum"):
        parts.append("One of: " + ", ".join(str(value) for value in resolved["enum"]) + ".")
    if not parts:
        parts.append(f"The {name.replace('_', ' ')} for {spec.name}.")
    return " ".join(parts)


def build_request_body_schema(spec: ToolSpec) -> ObjectJsonSchemaPropertyInput:
    """Translate a tool's generated JSON Schema into ElevenLabs' form.

    Derived from ``spec.request_schema``, which itself comes from the Pydantic
    request model — so the provider configuration cannot drift from the
    endpoint it calls.
    """
    schema = spec.request_schema
    definitions = schema.get("$defs", {})
    properties: dict[str, Any] = {}

    for name, raw in schema["properties"].items():
        resolved = _resolve(raw, definitions)
        json_type = resolved.get("type", "string")
        if json_type not in _TYPE_MAP:
            raise ValueError(f"{spec.name}.{name}: unsupported type {json_type!r}")

        if name in spec.injected_parameters:
            # An injected parameter is filled by the platform from the
            # conversation's dynamic variables, so the model never supplies it.
            #
            # ElevenLabs treats description, dynamic_variable,
            # is_system_provided, constant_value and is_omitted as mutually
            # exclusive and rejects a property that sets more than one
            # ("Can only set one of: ..."). A dynamic variable needs no
            # description anyway — nothing reads it, because the model is not
            # choosing the value.
            properties[name] = LiteralJsonSchemaProperty(
                type=_TYPE_MAP[json_type],
                dynamic_variable=name,
            )
            continue

        properties[name] = LiteralJsonSchemaProperty(
            type=_TYPE_MAP[json_type],
            description=_describe(spec, name, resolved),
            enum=[str(value) for value in resolved["enum"]] if resolved.get("enum") else None,
        )

    return ObjectJsonSchemaPropertyInput(
        type="object",
        properties=properties,
        required=list(schema.get("required", [])),
    )


def build_tool_request(spec: ToolSpec, base_url: str, secret_id: str) -> ToolRequestModel:
    """One webhook tool definition, pointed at our public tunnel."""
    description = f"{spec.description} {spec.when_to_use}".strip()
    if spec.restrictions:
        description += " Restrictions: " + " ".join(spec.restrictions)

    return ToolRequestModel(
        tool_config=ToolRequestModelToolConfig_Webhook(
            type="webhook",
            name=spec.name,
            description=description,
            response_timeout_secs=TOOL_TIMEOUT_SECONDS,
            api_schema=WebhookToolApiSchemaConfigInput(
                url=spec.url(base_url),
                method=spec.method,
                content_type="application/json",
                request_headers={AUTH_HEADER: ConvAiSecretLocator(secret_id=secret_id)},
                request_body_schema=build_request_body_schema(spec),
            ),
        )
    )


def build_conversation_config(tool_ids: list[str]) -> ConversationalConfig:
    """The agent configuration, built from ``app.agent.prompt``."""
    return ConversationalConfig(
        agent=AgentConfig(
            first_message=build_first_message(),
            language=AGENT_LANGUAGE,
            # The SDK's ConversationalConfig nests the *Output* prompt model
            # even on a create/update request; the two have identical fields.
            prompt=PromptAgentApiModelOutput(
                prompt=SYSTEM_PROMPT,
                llm=AGENT_LLM,
                temperature=AGENT_TEMPERATURE,
                tool_ids=tool_ids,
            ),
        ),
    )


# ---------------------------------------------------------------------------
# Provisioning
# ---------------------------------------------------------------------------


def ensure_tool_secret(client: ElevenLabs | None = None) -> str:
    """Store our X-Tool-Secret as a workspace secret and return its id.

    Reuses an existing secret of the same name so repeated provisioning does
    not litter the workspace. The value is sent to ElevenLabs (it has to be —
    they add the header on our behalf) but is never logged here.
    """
    if not settings.tool_shared_secret:
        raise ElevenLabsNotConfigured(
            "TOOL_SHARED_SECRET is not set; the tool endpoints would reject "
            "every call from the agent."
        )

    client = client or get_client()
    existing = client.conversational_ai.secrets.list()
    for secret in getattr(existing, "secrets", []) or []:
        if getattr(secret, "name", None) == TOOL_SECRET_NAME:
            secret_id = getattr(secret, "secret_id", None)
            if secret_id:
                client.conversational_ai.secrets.update(
                    secret_id, name=TOOL_SECRET_NAME, value=settings.tool_shared_secret
                )
                logger.info("reused workspace secret %s", TOOL_SECRET_NAME)
                return str(secret_id)

    created = client.conversational_ai.secrets.create(
        name=TOOL_SECRET_NAME, value=settings.tool_shared_secret
    )
    logger.info("created workspace secret %s", TOOL_SECRET_NAME)
    return str(created.secret_id)


def sync_tools(
    base_url: str | None = None, client: ElevenLabs | None = None
) -> dict[str, str]:
    """Create or update all seven webhook tools. Returns ``{name: tool_id}``.

    Idempotent: a tool whose name already exists is updated in place rather
    than duplicated, so provisioning can be re-run after changing a schema or
    moving the tunnel.
    """
    base_url = base_url or settings.public_base_url
    if not base_url:
        raise ElevenLabsNotConfigured(
            "PUBLIC_BASE_URL is not set; ElevenLabs would have no address to "
            "call our tools on."
        )

    client = client or get_client()
    secret_id = ensure_tool_secret(client)

    existing: dict[str, str] = {}
    listing = client.conversational_ai.tools.list()
    for tool in getattr(listing, "tools", []) or []:
        config = getattr(tool, "tool_config", None)
        name = getattr(config, "name", None)
        tool_id = getattr(tool, "id", None) or getattr(tool, "tool_id", None)
        if name and tool_id:
            existing[str(name)] = str(tool_id)

    tool_ids: dict[str, str] = {}
    for spec in TOOL_SPECS:
        request = build_tool_request(spec, base_url, secret_id)
        if spec.name in existing:
            updated = client.conversational_ai.tools.update(
                existing[spec.name], request=request
            )
            tool_ids[spec.name] = str(
                getattr(updated, "id", None) or getattr(updated, "tool_id", existing[spec.name])
            )
            logger.info("updated tool %s -> %s", spec.name, spec.url(base_url))
        else:
            created = client.conversational_ai.tools.create(request=request)
            tool_ids[spec.name] = str(
                getattr(created, "id", None) or getattr(created, "tool_id", "")
            )
            logger.info("created tool %s -> %s", spec.name, spec.url(base_url))

    return tool_ids


def provision_agent(
    base_url: str | None = None, client: ElevenLabs | None = None
) -> dict[str, Any]:
    """Create or update the agent and its tools. Returns a summary.

    If ``ELEVENLABS_AGENT_ID`` is already set the existing agent is updated,
    so the id in ``.env`` stays valid across re-provisioning.
    """
    client = client or get_client()
    tool_ids = sync_tools(base_url, client)
    config = build_conversation_config(list(tool_ids.values()))

    if settings.elevenlabs_agent_id:
        agent = client.conversational_ai.agents.update(
            settings.elevenlabs_agent_id,
            name=AGENT_NAME,
            conversation_config=config,
        )
        agent_id = str(getattr(agent, "agent_id", settings.elevenlabs_agent_id))
        action = "updated"
    else:
        agent = client.conversational_ai.agents.create(
            name=AGENT_NAME, conversation_config=config
        )
        agent_id = str(agent.agent_id)
        action = "created"

    logger.info("%s agent %s with %d tools", action, agent_id, len(tool_ids))
    return {
        "action": action,
        "agent_id": agent_id,
        "agent_name": AGENT_NAME,
        "llm": AGENT_LLM,
        "tools": tool_ids,
        "base_url": base_url or settings.public_base_url,
    }


def get_agent(agent_id: str | None = None, client: ElevenLabs | None = None) -> Any:
    """Fetch the configured agent, for local verification."""
    agent_id = agent_id or settings.elevenlabs_agent_id
    if not agent_id:
        raise ElevenLabsNotConfigured("ELEVENLABS_AGENT_ID is not set.")
    return (client or get_client()).conversational_ai.agents.get(agent_id)


def describe_agent(agent_id: str | None = None, client: ElevenLabs | None = None) -> dict[str, Any]:
    """A safe summary of the live agent: tool count, llm, prompt length."""
    agent = get_agent(agent_id, client)
    prompt_config = getattr(
        getattr(getattr(agent, "conversation_config", None), "agent", None), "prompt", None
    )
    tool_ids = list(getattr(prompt_config, "tool_ids", None) or [])
    prompt_text = getattr(prompt_config, "prompt", "") or ""
    return {
        "agent_id": str(getattr(agent, "agent_id", agent_id)),
        "name": getattr(agent, "name", None),
        "llm": getattr(prompt_config, "llm", None),
        "tool_count": len(tool_ids),
        "prompt_characters": len(prompt_text),
        "prompt_matches_repository": prompt_text.strip() == SYSTEM_PROMPT.strip(),
    }


# ---------------------------------------------------------------------------
# Browser conversation
# ---------------------------------------------------------------------------


def signed_url(agent_id: str | None = None, client: ElevenLabs | None = None) -> str:
    """A short-lived URL letting the browser open a conversation.

    This is why the API key can stay server-side: the page receives a
    time-limited URL for one conversation instead of a credential.
    """
    agent_id = agent_id or settings.elevenlabs_agent_id
    if not agent_id:
        raise ElevenLabsNotConfigured("ELEVENLABS_AGENT_ID is not set.")

    response = (client or get_client()).conversational_ai.conversations.get_signed_url(
        agent_id=agent_id
    )
    url = getattr(response, "signed_url", None)
    if not url:
        raise ElevenLabsNotConfigured("ElevenLabs returned no signed URL.")
    return str(url)
