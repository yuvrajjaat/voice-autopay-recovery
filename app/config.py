"""Typed application configuration.

Values are read from environment variables, optionally loaded from a local
`.env` file. `.env` is gitignored; `.env.example` ships placeholders only.

Why everything is Optional
--------------------------
Phase 0 must boot with no configuration at all, so no integration variable is
required at import time. Instead, each phase calls ``settings.require(...)`` at
the point of use, which raises a single clear error naming exactly which
variables are missing — rather than failing deep inside a request handler with
a ``NoneType`` error.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Repository root: this file is <root>/app/config.py
BASE_DIR = Path(__file__).resolve().parent.parent

# E.164: a leading '+', a non-zero country code, then 7-14 more digits.
E164_PATTERN = re.compile(r"^\+[1-9]\d{7,14}$")

# Formatting characters a human might type or a transcript might contain.
_PHONE_NOISE = re.compile(r"[\s\-().]")


def normalise_e164(value: object) -> str | None:
    """Strip formatting and return the number in E.164, or None if invalid.

    Shared by the configuration validator (which raises on a bad value) and
    by ``app.dial_safety`` (which treats a bad value as a rejection), so both
    agree on exactly what a valid destination looks like.

    Deliberately total: anything it cannot normalise returns None rather than
    being guessed at or repaired. Silently "fixing" a malformed destination is
    how a call reaches the wrong person.
    """
    if value is None:
        return None
    text = _PHONE_NOISE.sub("", str(value))
    if not text:
        return None
    return text if E164_PATTERN.match(text) else None


class MissingConfigError(RuntimeError):
    """Raised when a feature is used before its configuration is supplied."""


class Settings(BaseSettings):
    """All runtime configuration for the project."""

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Application -------------------------------------------------------
    app_name: str = "voice-autopay-recovery"
    app_port: int = 8000
    log_level: str = "INFO"

    # --- ElevenLabs Agents (Phase 6+) --------------------------------------
    elevenlabs_api_key: str | None = None
    elevenlabs_agent_id: str | None = None
    elevenlabs_webhook_secret: str | None = None

    # --- Our own tool authentication (Phase 2+) ----------------------------
    tool_shared_secret: str | None = None

    # --- Public tunnel, so ElevenLabs can reach our tools (Phase 6+) -------
    public_base_url: str | None = None

    # --- Dial safety -------------------------------------------------------
    # Outbound telephony stays off until this is deliberately flipped to true.
    enable_outbound_calls: bool = False
    # The only destination this project is ever permitted to dial.
    demo_phone_number: str | None = None

    # --- Telephony identifier (Phase 9; not a credential) ------------------
    # Provider-neutral on purpose: the dial guard and the eventual provider
    # adapter both refer to "the number we call from", whoever supplies it.
    agent_phone_number_id: str | None = None

    # --- Optional: offline text simulator (Phase 4) ------------------------
    anthropic_api_key: str | None = None

    # ----------------------------------------------------------------- paths
    @property
    def base_dir(self) -> Path:
        return BASE_DIR

    @property
    def data_dir(self) -> Path:
        return BASE_DIR / "data"

    @property
    def customers_file(self) -> Path:
        return self.data_dir / "customers.json"

    @property
    def runtime_file(self) -> Path:
        return self.data_dir / "runtime.json"

    # ------------------------------------------------------------ validators
    @field_validator("demo_phone_number", mode="before")
    @classmethod
    def _normalise_phone(cls, value: object) -> str | None:
        """Treat a blank as unset; reject anything that is not E.164.

        A malformed number fails startup rather than being accepted and
        quietly rejected later, so a typo in .env is found immediately.
        """
        if value is None or not _PHONE_NOISE.sub("", str(value)):
            return None
        normalised = normalise_e164(value)
        if normalised is None:
            raise ValueError(
                f"DEMO_PHONE_NUMBER must be E.164 (e.g. +14155550123), got {value!r}"
            )
        return normalised

    @field_validator("public_base_url", "elevenlabs_api_key", "elevenlabs_agent_id",
                     "elevenlabs_webhook_secret", "tool_shared_secret",
                     "agent_phone_number_id", "anthropic_api_key",
                     mode="before")
    @classmethod
    def _blank_to_none(cls, value: object) -> object:
        """Treat an empty or whitespace-only env var as unset.

        `.env.example` ships several variables with empty values, which would
        otherwise arrive as "" and read as configured.
        """
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("public_base_url")
    @classmethod
    def _strip_trailing_slash(cls, value: str | None) -> str | None:
        return value.rstrip("/") if value else value

    @field_validator("log_level")
    @classmethod
    def _upper_log_level(cls, value: str) -> str:
        level = value.upper()
        allowed = {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"}
        if level not in allowed:
            raise ValueError(f"LOG_LEVEL must be one of {sorted(allowed)}, got {value!r}")
        return level

    # --------------------------------------------------------------- helpers
    def require(self, *names: str) -> None:
        """Assert that the named settings are present, or raise once, clearly.

        Example:
            settings.require("elevenlabs_api_key", "public_base_url")
        """
        missing = [name for name in names if not getattr(self, name, None)]
        if missing:
            env_names = ", ".join(sorted(name.upper() for name in missing))
            raise MissingConfigError(
                f"Missing required configuration: {env_names}. "
                "Add it to your .env file (see .env.example)."
            )

    def readiness(self) -> dict[str, bool]:
        """Which features are configured — booleans only, never any values.

        Safe to expose over HTTP: it reveals whether a secret is set, never
        what it is.
        """
        return {
            "elevenlabs_api_key": bool(self.elevenlabs_api_key),
            "elevenlabs_agent_id": bool(self.elevenlabs_agent_id),
            "elevenlabs_webhook_secret": bool(self.elevenlabs_webhook_secret),
            "tool_shared_secret": bool(self.tool_shared_secret),
            "public_base_url": bool(self.public_base_url),
            "demo_phone_number": bool(self.demo_phone_number),
            "agent_phone_number_id": bool(self.agent_phone_number_id),
            "outbound_calls_enabled": self.enable_outbound_calls,
        }


@lru_cache
def get_settings() -> Settings:
    """Cached settings accessor, suitable as a FastAPI dependency."""
    return Settings()


settings = get_settings()
