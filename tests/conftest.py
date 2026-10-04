"""Shared test fixtures.

The autouse fixture below redirects runtime state into a per-test temporary
file. Without it, running the suite would overwrite ``data/runtime.json`` and
quietly change the state of whatever demo you had set up.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app import store
from app.routers import webhooks


@pytest.fixture(autouse=True)
def isolated_runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Give each test its own runtime.json and a freshly validated seed.

    Patches the private path helper rather than adding a config knob that
    production code would never use. Clearing the seed cache means every test
    re-reads and re-validates the real ``data/customers.json``.
    """
    runtime_file = tmp_path / "runtime.json"
    monkeypatch.setattr(store, "_runtime_path", lambda: runtime_file)
    monkeypatch.setattr(store, "_customers_cache", None)
    # The post-call webhook saves transcripts; keep those out of the committed
    # demo/transcripts folder too.
    monkeypatch.setattr(webhooks, "_transcript_dir", lambda: tmp_path / "transcripts")
    return runtime_file


@pytest.fixture
def seed_path() -> Path:
    """The real committed seed file, for content assertions."""
    return Path(__file__).resolve().parent.parent / "data" / "customers.json"
