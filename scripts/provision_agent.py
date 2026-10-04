"""Provision the ElevenLabs agent from the definitions in this repository.

The point of this script is that nothing has to be pasted into a dashboard.
The prompt comes from ``app/agent/prompt.py`` and the seven tools come from
``app/agent/tool_specs.py``, so a clone plus this script reproduces the exact
agent — and a change to either file is one re-run away from being live.

Usage::

    python scripts/provision_agent.py --dry-run   # print what would be sent
    python scripts/provision_agent.py             # create or update for real
    python scripts/provision_agent.py --verify    # inspect the live agent

``--dry-run`` needs no API key and no network: it builds the payloads locally
and prints them, which is the useful check when you have a tunnel but no
credentials yet.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agent.prompt import SYSTEM_PROMPT, build_first_message  # noqa: E402
from app.agent.tool_specs import AUTH_HEADER, TOOL_SPECS  # noqa: E402
from app.config import settings  # noqa: E402
from app.providers import elevenlabs_client as provider  # noqa: E402

REQUIRED_FOR_PROVISIONING = ("elevenlabs_api_key", "tool_shared_secret", "public_base_url")


def _fail(message: str) -> int:
    print(f"error: {message}", file=sys.stderr)
    return 2


def _check_configuration(*, needs_api_key: bool) -> str | None:
    """Return an error message when configuration is missing, else None."""
    missing: list[str] = []
    for name in REQUIRED_FOR_PROVISIONING:
        if name == "elevenlabs_api_key" and not needs_api_key:
            continue
        if not getattr(settings, name, None):
            missing.append(name.upper())

    if missing:
        return (
            "missing configuration: "
            + ", ".join(sorted(missing))
            + ".\nAdd it to .env (see .env.example). PUBLIC_BASE_URL must be the "
            "https address of your tunnel, for example "
            "https://your-domain.ngrok-free.app"
        )

    base_url = str(settings.public_base_url)
    if not base_url.startswith("https://"):
        return (
            f"PUBLIC_BASE_URL must be an https URL (got {base_url!r}). "
            "ElevenLabs will not call a plain-http webhook."
        )
    return None


def dry_run() -> int:
    """Print the payloads that would be sent. No key, no network."""
    base_url = settings.public_base_url or "https://PUBLIC_BASE_URL-not-set"

    print("Agent")
    print(f"  name          {provider.AGENT_NAME}")
    print(f"  llm           {provider.AGENT_LLM}")
    print(f"  temperature   {provider.AGENT_TEMPERATURE}")
    print(f"  language      {provider.AGENT_LANGUAGE}")
    print(f"  prompt        {len(SYSTEM_PROMPT)} characters from app/agent/prompt.py")
    print(f"  first message {build_first_message()[:72]}...")
    print()
    print(f"Tools ({len(TOOL_SPECS)}), authenticated with the {AUTH_HEADER} header")
    for spec in TOOL_SPECS:
        body = provider.build_request_body_schema(spec)
        injected = [
            name
            for name, prop in (body.properties or {}).items()
            if getattr(prop, "dynamic_variable", None)
        ]
        llm_filled = [
            name for name in (body.properties or {}) if name not in injected
        ]
        print(f"  {spec.name}")
        print(f"      {spec.method} {spec.url(base_url)}")
        print(f"      injected by platform: {', '.join(injected) or '-'}")
        print(f"      filled by the model:  {', '.join(llm_filled) or '-'}")
    print()

    problem = _check_configuration(needs_api_key=True)
    if problem:
        print("Not ready to provision:")
        print(f"  {problem.splitlines()[0]}")
    else:
        print("Configuration looks complete; re-run without --dry-run to apply.")
    return 0


def verify() -> int:
    """Fetch the live agent and report whether it matches this repository."""
    try:
        summary = provider.describe_agent()
    except provider.ElevenLabsNotConfigured as error:
        return _fail(str(error))

    print(json.dumps(summary, indent=2))
    print()
    if summary["tool_count"] != len(TOOL_SPECS):
        print(
            f"warning: the live agent has {summary['tool_count']} tools, "
            f"this repository defines {len(TOOL_SPECS)}. Re-run provisioning."
        )
    if not summary["prompt_matches_repository"]:
        print(
            "warning: the live prompt differs from app/agent/prompt.py. "
            "Re-run provisioning to push the repository version."
        )
    if summary["tool_count"] == len(TOOL_SPECS) and summary["prompt_matches_repository"]:
        print("the live agent matches this repository.")
    return 0


def provision() -> int:
    problem = _check_configuration(needs_api_key=True)
    if problem:
        return _fail(problem)

    try:
        result = provider.provision_agent()
    except provider.ElevenLabsNotConfigured as error:
        return _fail(str(error))

    print(f"{result['action']} agent {result['agent_id']}")
    print(f"  llm       {result['llm']}")
    print(f"  base url  {result['base_url']}")
    print(f"  tools     {len(result['tools'])}")
    for name, tool_id in result["tools"].items():
        print(f"      {name:<30} {tool_id}")
    print()

    if settings.elevenlabs_agent_id != result["agent_id"]:
        print("Add this to your .env, then restart the server:")
        print(f"  ELEVENLABS_AGENT_ID={result['agent_id']}")
    else:
        print("ELEVENLABS_AGENT_ID in .env already matches.")
    print()
    print("Next: open http://127.0.0.1:8000/voice and pick a customer.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Create or update the ElevenLabs agent from this repository.",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--dry-run",
        action="store_true",
        help="print the payloads without calling ElevenLabs (no API key needed)",
    )
    group.add_argument(
        "--verify",
        action="store_true",
        help="fetch the live agent and compare it with this repository",
    )
    arguments = parser.parse_args(argv)

    if arguments.dry_run:
        return dry_run()
    if arguments.verify:
        return verify()
    return provision()


if __name__ == "__main__":
    raise SystemExit(main())
