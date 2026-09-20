"""HTTP surface tests through the real FastAPI lifespan."""
from __future__ import annotations


def test_health_reports_every_component(api_client):
    response = api_client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["app_mode"] == "demo"
    assert body["source"]["ok"] is True
    assert len(body["mcp"]["tools"]) >= 10
    assert body["llm"]["active"] is False       # heuristic mode in tests
    assert body["observability"]["local_tracing"] is True


def test_tool_catalogue_is_exposed(api_client):
    body = api_client.get("/api/tools").json()
    names = {tool["name"] for tool in body["tools"]}
    assert "get_project_activity" in names
    tool = next(t for t in body["tools"] if t["name"] == "search_issues")
    assert tool["description"] and tool["input_schema"]["type"] == "object"


def test_sync_then_projects(api_client):
    sync = api_client.post("/api/sync", json={"full": True})
    assert sync.status_code == 200
    stats = sync.json()["stats"]
    assert stats["issues_fetched"] >= 20 and stats["errors"] == []

    projects = api_client.get("/api/projects").json()
    assert projects["count"] == 3
    assert {p["key"] for p in projects["projects"]} == {"ATLAS", "APOLLO", "HELIOS"}


def test_chat_returns_answer_plan_evidence_and_trace(api_client):
    api_client.post("/api/sync", json={"full": True})
    response = api_client.post("/api/chat", json={
        "message": "What is blocked in Project Atlas and why?",
        "session_id": "api-session",
    })
    assert response.status_code == 200
    body = response.json()
    assert body["answer"]
    assert body["plan"]["sub_questions"]
    assert body["tool_calls"] and body["evidence"] and body["citations"]

    trace = api_client.get(f"/api/trace/{body['run_id']}").json()
    assert trace["run"]["run_id"] == body["run_id"]
    assert [step["name"] for step in trace["steps"]][0] == "memory_retrieval"

    runs = api_client.get("/api/runs").json()["runs"]
    assert any(r["run_id"] == body["run_id"] for r in runs)


def test_chat_with_baseline_comparison(api_client):
    api_client.post("/api/sync", json={"full": True})
    body = api_client.post("/api/chat", json={
        "message": "Summarize what changed in Project Atlas during Q2 2026 and identify the major risks.",
        "compare_baseline": True,
    }).json()
    assert body["baseline"]["tool_calls"] == 0
    assert body["baseline"]["citations"] == []
    assert len(body["tool_calls"]) > 0
    assert len(body["citations"]) > 0


def test_memory_lifecycle_over_http(api_client):
    api_client.post("/api/chat", json={
        "message": "Project Atlas is my highest priority this quarter.",
        "session_id": "mem-session",
    })
    memories = api_client.get("/api/memory").json()
    assert memories["count"] == 1
    memory_id = memories["memories"][0]["memory_id"]

    created = api_client.post("/api/memory", json={
        "content": "User reviews risk reports on Monday mornings.", "type": "context",
    })
    assert created.status_code == 200

    assert api_client.delete(f"/api/memory/{memory_id}").status_code == 200
    assert api_client.delete(f"/api/memory/{memory_id}").status_code == 404


def test_memory_is_not_visible_across_users(api_client):
    api_client.post("/api/chat", json={
        "message": "Project Apollo is my highest priority.", "user_id": "user-a"})
    assert api_client.get("/api/memory", params={"user_id": "user-b"}).json()["count"] == 0


def test_trace_of_another_user_is_not_readable(api_client):
    body = api_client.post("/api/chat", json={
        "message": "Who owns Project Apollo?", "user_id": "owner"}).json()
    assert api_client.get(f"/api/trace/{body['run_id']}", params={"user_id": "intruder"}).status_code == 404


def test_input_validation(api_client):
    assert api_client.post("/api/chat", json={"message": ""}).status_code == 422
    assert api_client.post("/api/chat", json={"message": "hi", "user_id": "bad id!"}).status_code == 400
    assert api_client.get("/api/trace/does-not-exist").status_code == 404


def test_ui_is_served(api_client):
    assert api_client.get("/").status_code == 200
    assert "Agent trace" in api_client.get("/").text
