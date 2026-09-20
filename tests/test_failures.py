"""Failure modes: the system must degrade honestly, never crash and never invent."""
from __future__ import annotations

import httpx
import pytest

from app.connectors.base import AuthError, IssueQuery, RateLimitError, UnavailableError
from app.connectors.jira import JiraConnector, build_jql
from app.connectors.mock_jira import MockJiraConnector
from app.llm.base import LLMClient, LLMError, LLMResponse


class BrokenLLM(LLMClient):
    """An LLM that always fails, to prove the deterministic paths still work."""

    model = "broken"

    async def complete(self, system, user, *, temperature=None, max_tokens=None, json_object=False):
        raise LLMError("provider is down")


class EmptyLLM(LLMClient):
    model = "empty"

    async def complete(self, system, user, *, temperature=None, max_tokens=None, json_object=False):
        return LLMResponse(text="", model=self.model)


def test_source_unavailable_is_reported_not_raised(services, run):
    services.connector.fail_mode = "unavailable"
    result = run(services.mcp.call("search_issues", {"project_key": "ATLAS"}))
    assert result["ok"] is False
    assert result["error"]["kind"] == "unavailable"
    assert result["items"] == []


def test_agent_refuses_when_the_source_is_down(synced_services, run):
    synced_services.connector.fail_mode = "unavailable"
    result = run(synced_services.orchestrator.run(
        question="What is blocked in Project Atlas?", user_id="tester", session_id="down"))
    assert result["warnings"], "a source outage must be surfaced"
    assert not any(c["status"] == "ok" and c["result_count"] > 0
                   for c in result["tool_calls"] if c["tool_name"] == "get_blocked_issues")
    lowered = result["answer"].lower()
    # The local index still answers, but the answer must admit the live source was unreachable.
    assert ("retrieval was degraded" in lowered
            or "not going to guess" in lowered
            or result["insufficient_evidence"])


def test_missing_dataset_reports_a_useful_message(tmp_path, run):
    connector = MockJiraConnector(tmp_path / "does-not-exist")
    health = run(connector.health())
    assert health.ok is False
    assert "seed_mock_jira" in health.detail


def test_corrupted_dataset_is_treated_as_a_malformed_response(tmp_path, run):
    (tmp_path / "projects.json").write_text("{not json", encoding="utf-8")
    connector = MockJiraConnector(tmp_path)
    health = run(connector.health())
    assert health.ok is False and "Malformed" in health.detail


def test_empty_result_sets_are_handled(services, run):
    result = run(services.mcp.call("search_issues", {
        "project_key": "ATLAS", "labels": ["label-that-does-not-exist"]}))
    assert result["ok"] is True and result["count"] == 0


def test_live_mode_without_credentials_fails_fast(settings, monkeypatch):
    from app.config import get_settings, reset_settings_cache
    from app.connectors.factory import build_connector

    monkeypatch.setenv("APP_MODE", "live")
    monkeypatch.setenv("JIRA_BASE_URL", "")
    reset_settings_cache()
    with pytest.raises(RuntimeError, match="credentials are missing"):
        build_connector(get_settings())
    reset_settings_cache()


def test_incomplete_jira_credentials_raise_auth_error():
    with pytest.raises(AuthError):
        JiraConnector(base_url="https://x.atlassian.net", email="a@b.c", api_token="")


def test_jira_401_becomes_auth_error(run):
    transport = httpx.MockTransport(lambda request: httpx.Response(401, json={"errorMessages": ["nope"]}))
    client = httpx.AsyncClient(transport=transport, base_url="https://x.atlassian.net")
    connector = JiraConnector("https://x.atlassian.net", "a@b.c", "token", client=client)
    with pytest.raises(AuthError):
        run(connector.get_issue("ATLAS-1"))
    run(connector.close())


def test_jira_rate_limit_is_retried_then_succeeds(run):
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, headers={"Retry-After": "0"}, json={})
        return httpx.Response(200, json={"issues": [], "isLast": True})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="https://x.atlassian.net")
    connector = JiraConnector("https://x.atlassian.net", "a@b.c", "token", client=client, max_retries=3)
    page = run(connector.search_issues(IssueQuery(project_key="ATLAS")))
    assert calls["n"] == 2 and page.items == []
    run(connector.close())


def test_jira_server_error_exhausts_retries_and_reports_unavailable(run):
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(503, json={})),
        base_url="https://x.atlassian.net",
    )
    connector = JiraConnector("https://x.atlassian.net", "a@b.c", "token", client=client, max_retries=2)
    with pytest.raises(UnavailableError):
        run(connector.search_issues(IssueQuery(project_key="ATLAS")))
    run(connector.close())


def test_jql_builder_escapes_and_orders():
    jql = build_jql(IssueQuery(project_key='AT"LAS', unresolved_only=True, order_by="duedate",
                               order_dir="ASC"))
    assert '\\"' in jql
    assert "resolution = EMPTY" in jql
    assert jql.endswith("ORDER BY duedate ASC")


def test_planner_falls_back_when_the_llm_fails(settings, run):
    from app.agents.planner import Planner

    planner = Planner(BrokenLLM())
    plan = run(planner.plan(
        "What is blocked in Project Atlas?",
        tool_catalogue="- get_blocked_issues(project_key: string?)",
        tool_names=["get_blocked_issues"],
        known_projects=[{"key": "ATLAS", "name": "Project Atlas"}],
    ))
    assert plan.generated_by == "heuristic"
    assert plan.required_tools == ["get_blocked_issues"]


def test_synthesizer_falls_back_when_the_llm_returns_nothing(synced_services, run):
    from app.agents.synthesizer import Synthesizer

    synced_services.orchestrator.synthesizer = Synthesizer(EmptyLLM())
    result = run(synced_services.orchestrator.run(
        question="What is blocked in Project Atlas?", user_id="tester", session_id="fallback"))
    assert result["answer"]
    assert result["citation_stats"]["synthesizer"] == "heuristic"


def test_memory_extraction_survives_a_broken_llm(services, run):
    from app.memory.extractor import MemoryExtractor

    services.memory.extractor = MemoryExtractor(BrokenLLM())
    report = run(services.memory.observe(
        user_id="u9", session_id="s", run_id=None,
        user_message="Project Atlas is my highest priority this quarter."))
    assert report["extractor"] == "heuristic"
    assert report["actions"][0]["action"] == "created"


def test_agent_run_is_traced_even_when_a_tool_fails(synced_services, run):
    synced_services.connector.fail_mode = "unavailable"
    result = run(synced_services.orchestrator.run(
        question="What is overdue in Project Atlas?", user_id="tester", session_id="traced-failure"))
    trace = run(synced_services.trace_reader.get_trace(result["run_id"], "tester"))
    assert trace["run"]["status"] == "completed"
    assert any(call["status"] == "error" for call in trace["tool_calls"])
