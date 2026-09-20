"""Test fixtures: an isolated database and a private copy of the demo dataset per test run."""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture()
def demo_data(tmp_path) -> Path:
    """A fresh writable copy of the demo dataset per test.

    Function scope on purpose: several tests mutate the source to prove live-data behaviour,
    and they must not leak into each other.
    """
    from scripts.seed_mock_jira import ISSUES, PROJECTS

    directory = tmp_path / "mock_jira"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "projects.json").write_text(json.dumps(PROJECTS, indent=2), encoding="utf-8")
    (directory / "issues.json").write_text(json.dumps(ISSUES, indent=2), encoding="utf-8")
    return directory


@pytest.fixture()
def settings(tmp_path, demo_data, monkeypatch):
    from app.config import get_settings, reset_settings_cache

    monkeypatch.setenv("ENV_FILE", str(tmp_path / "nonexistent.env"))
    monkeypatch.setenv("APP_MODE", "demo")
    monkeypatch.setenv("DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("MOCK_DATA_DIR", str(demo_data))
    monkeypatch.setenv("LLM_PROVIDER", "none")
    monkeypatch.setenv("LLM_API_KEY", "")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "hashing")
    monkeypatch.setenv("EMBEDDING_DIM", "256")
    monkeypatch.setenv("MCP_TRANSPORT", "inproc")
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("JIRA_ALLOWED_PROJECTS", "")
    reset_settings_cache()
    yield get_settings()
    reset_settings_cache()


@pytest.fixture()
def loop():
    """One event loop per test.

    The MCP session holds anyio streams bound to the loop that opened it, so startup, the test
    body and shutdown must all run on the same loop.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        yield loop
    finally:
        asyncio.set_event_loop(None)
        loop.close()


@pytest.fixture()
def run(loop):
    """Run a coroutine on the test's event loop."""
    def _run(coro):
        return loop.run_until_complete(coro)
    return _run


@pytest.fixture()
def services(settings, run):
    """A fully wired Services object, torn down after the test."""
    from app.services import Services

    instance = Services(settings)
    run(instance.startup())
    try:
        yield instance
    finally:
        run(instance.shutdown())


@pytest.fixture()
def synced_services(services, run):
    run(services.ingestion.sync(full=True))
    return services


@pytest.fixture()
def api_client(settings):
    """FastAPI TestClient with the real lifespan (services start and stop around the block)."""
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as client:
        yield client
