"""Integration tests across the MCP boundary: agent -> MCP client -> server -> connector."""
from __future__ import annotations


def test_tools_are_discovered_over_the_protocol(services):
    names = services.mcp.tool_names()
    for expected in ("search_projects", "search_issues", "get_issue_changelog",
                     "get_blocked_issues", "get_overdue_issues", "semantic_search"):
        assert expected in names
    spec = services.mcp.tools["search_issues"]
    assert "project_key" in spec.input_schema.get("properties", {})
    assert spec.description


def test_search_projects_returns_normalised_items(services, run):
    result = run(services.mcp.call("search_projects", {}))
    assert result["ok"] is True
    keys = {item["source_id"] for item in result["items"]}
    assert {"ATLAS", "APOLLO"} <= keys
    for item in result["items"]:
        assert item["url"].startswith("http")


def test_search_issues_applies_structured_filters(services, run):
    result = run(services.mcp.call("search_issues", {
        "project_key": "ATLAS", "status_category": "Done",
        "resolved_after": "2026-04-01T00:00:00Z", "resolved_before": "2026-06-30T23:59:59Z",
    }))
    assert result["ok"] is True
    assert result["count"] >= 1
    assert all(item["data"]["resolved"] for item in result["items"])


def test_changelog_field_filter_isolates_deadline_moves(services, run):
    result = run(services.mcp.call("get_issue_changelog", {"issue_key": "ATLAS-1", "fields": ["duedate"]}))
    assert result["ok"] is True
    assert result["count"] == 3
    assert all(item["data"]["fields"] == ["duedate"] for item in result["items"])


def test_blocked_detection_uses_status_label_and_links(services, run):
    result = run(services.mcp.call("get_blocked_issues", {"project_key": "ATLAS"}))
    keys = {item["source_id"] for item in result["items"]}
    assert {"ATLAS-9", "ATLAS-11"} <= keys
    atlas9 = next(i for i in result["items"] if i["source_id"] == "ATLAS-9")
    assert any("blocked by ATLAS-11" in reason for reason in atlas9["data"]["blocked_reasons"])


def test_overdue_is_relative_to_as_of(services, run):
    end_of_q2 = run(services.mcp.call("get_overdue_issues", {
        "project_key": "ATLAS", "as_of": "2026-06-30T23:59:59Z"}))
    mid_q2 = run(services.mcp.call("get_overdue_issues", {
        "project_key": "ATLAS", "as_of": "2026-05-01T00:00:00Z"}))
    assert end_of_q2["count"] > mid_q2["count"]


def test_project_activity_counts_event_kinds(services, run):
    result = run(services.mcp.call("get_project_activity", {
        "project_key": "ATLAS", "start": "2026-04-01T00:00:00Z", "end": "2026-06-30T23:59:59Z"}))
    counters = result["counters"]
    assert counters["duedate_change"] >= 3
    assert counters["comment"] >= 5
    timestamps = [item["timestamp"] for item in result["items"] if item["timestamp"]]
    assert timestamps == sorted(timestamps)


def test_unknown_tool_and_bad_arguments_return_structured_errors(services, run):
    unknown = run(services.mcp.call("delete_everything", {}))
    assert unknown["ok"] is False and unknown["error"]["kind"] == "unknown_tool"

    bad_key = run(services.mcp.call("get_issue", {"issue_key": "not a key"}))
    assert bad_key["ok"] is False and bad_key["error"]["kind"] == "invalid_argument"

    missing = run(services.mcp.call("get_project", {"project_key": "ZZZ"}))
    assert missing["ok"] is False and missing["error"]["kind"] == "not_found"


def test_project_allow_list_is_enforced(settings, run):
    from app.connectors.factory import build_connector
    from app.mcp_client.client import MCPToolClient
    from app.mcp_server.server import build_server
    from app.mcp_server.tools import ToolContext

    context = ToolContext(connector=build_connector(settings), allowed_projects={"APOLLO"})
    client = MCPToolClient(settings, server_object=build_server(context))
    run(client.connect())
    try:
        denied = run(client.call("get_project", {"project_key": "ATLAS"}))
        assert denied["ok"] is False and denied["error"]["kind"] == "access_denied"
        allowed = run(client.call("get_project", {"project_key": "APOLLO"}))
        assert allowed["ok"] is True
        listed = run(client.call("search_projects", {}))
        assert {item["source_id"] for item in listed["items"]} == {"APOLLO"}
    finally:
        run(client.close())
