"""Phase 0 smoke tests: the app boots, health reports, secrets stay hidden."""

from fastapi.testclient import TestClient

from app.config import MissingConfigError, Settings
from app.main import app

client = TestClient(app)


def test_healthz_returns_ok() -> None:
    response = client.get("/healthz")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["service"] == "voice-autopay-recovery"
    assert "config" in body


def test_healthz_never_leaks_secret_values() -> None:
    """Readiness must be booleans only — the endpoint is publicly reachable."""
    config = client.get("/healthz").json()["config"]
    assert config, "readiness should not be empty"
    assert all(isinstance(value, bool) for value in config.values())


def test_root_points_at_health_and_docs() -> None:
    body = client.get("/").json()
    assert body["health"] == "/healthz"
    assert body["docs"] == "/docs"


def test_outbound_calls_are_disabled_by_default() -> None:
    """Dial safety: a fresh checkout must not be able to place a call."""
    assert Settings(_env_file=None).enable_outbound_calls is False


def test_require_names_the_missing_variables() -> None:
    settings = Settings(_env_file=None)
    try:
        settings.require("elevenlabs_api_key", "tool_shared_secret")
    except MissingConfigError as exc:
        message = str(exc)
        assert "ELEVENLABS_API_KEY" in message
        assert "TOOL_SHARED_SECRET" in message
    else:
        raise AssertionError("require() should have raised for unset variables")


def test_demo_phone_number_must_be_e164() -> None:
    assert Settings(_env_file=None, demo_phone_number="+1 (415) 555-0123").demo_phone_number == "+14155550123"
    assert Settings(_env_file=None, demo_phone_number="").demo_phone_number is None
    try:
        Settings(_env_file=None, demo_phone_number="555-0123")
    except ValueError:
        pass
    else:
        raise AssertionError("a non-E.164 number should be rejected")
