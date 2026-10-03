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
    elevenlabs_agent_phone_number_id: str | None = None

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
        """Strip formatting, treat blanks as unset, and enforce E.164."""
        if value is None:
            return None
        text = re.sub(r"[\s\-().]", "", str(value))
        if not text:
            return None
        if not E164_PATTERN.match(text):
            raise ValueError(
                f"DEMO_PHONE_NUMBER must be E.164 (e.g. +14155550123), got {value!r}"
            )
        return text

    @field_validator("public_base_url", "elevenlabs_api_key", "elevenlabs_agent_id",
                     "elevenlabs_webhook_secret", "tool_shared_secret",
                     "elevenlabs_agent_phone_number_id", "anthropic_api_key",
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
            "agent_phone_number_id": bool(self.elevenlabs_agent_phone_number_id),
            "outbound_calls_enabled": self.enable_outbound_calls,
        }


@lru_cache
def get_settings() -> Settings:
    """Cached settings accessor, suitable as a FastAPI dependency."""
    return Settings()


settings = get_settings()
