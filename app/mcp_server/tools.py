"""The tool layer exposed over MCP.

Every function here is a *capability*, not a query template: the agent picks which ones to
call and with what arguments. Each returns a uniform envelope so the executor can turn any
tool result into Evidence without knowing the tool:

    {"ok": true, "tool": ..., "source": "jira", "mode": "demo|live", "count": N,
     "items": [{"source_type", "source_id", "title", "url", "timestamp", "content", "data"}],
     "truncated": bool, "fetched_at": "..."}

Errors are returned as data (never raised across the protocol boundary) so the agent can
reason about them: {"ok": false, "error": {"kind": ..., "message": ...}}.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

from app.connectors.base import ConnectorError, IssueQuery, SourceConnector
from app.models.domain import ChangelogEntry, Comment, Issue, Project
from app.rag.retriever import SemanticRetriever
from app.security.guards import AccessDenied, InvalidArgument, clamp_limit, validate_issue_key, validate_project_key
from app.utils.dates import iso, parse_dt, utcnow
from app.utils.text import truncate

BLOCKED_STATUSES = {"blocked", "on hold", "waiting", "impeded"}
BLOCKED_LABELS = {"blocked", "impediment", "waiting-on-vendor"}
RISK_KEYWORDS = (
    "blocked", "blocker", "slip", "slipped", "delay", "delayed", "risk", "at risk", "escalate",
    "escalated", "waiting on", "dependency", "regression", "failed", "failing", "outage",
    "incident", "overdue", "no fix date", "cannot", "won't make", "will not make",
)


@dataclass
class ToolContext:
    """Everything the tools need. Built once per process (API or MCP subprocess)."""

    connector: SourceConnector
    retriever: Optional[SemanticRetriever] = None
    allowed_projects: Optional[set[str]] = None
    max_items: int = 100


def _envelope(tool: str, ctx: ToolContext, items: list[dict], truncated: bool = False, **extra: Any) -> dict:
    return {
        "ok": True,
        "tool": tool,
        "source": ctx.connector.source_name,
        "mode": ctx.connector.mode,
        "count": len(items),
        "items": items,
        "truncated": truncated,
        "fetched_at": iso(utcnow()),
        **extra,
    }


def _error(tool: str, kind: str, message: str, **extra: Any) -> dict:
    return {"ok": False, "tool": tool, "error": {"kind": kind, "message": message}, "items": [], "count": 0, **extra}


def _guard(fn):
    """Turn connector/validation failures into structured, agent-readable results."""

    async def wrapper(*args, **kwargs):
        tool = fn.__name__
        try:
            return await fn(*args, **kwargs)
        except AccessDenied as exc:
            return _error(tool, "access_denied", str(exc))
        except InvalidArgument as exc:
            return _error(tool, "invalid_argument", str(exc))
        except ConnectorError as exc:
            return _error(tool, exc.kind, str(exc), retryable=exc.retryable)
        except Exception as exc:  # last-resort guard: the protocol boundary must stay clean
            return _error(tool, "internal_error", f"{type(exc).__name__}: {exc}")

    wrapper.__name__ = fn.__name__
    wrapper.__doc__ = fn.__doc__
    return wrapper


# --- item serialisers -------------------------------------------------------
def issue_item(issue: Issue) -> dict:
    content = (
        f"{issue.key} - {issue.summary}\n"
        f"Status: {issue.status} ({issue.status_category}); Priority: {issue.priority or 'unset'}; "
        f"Type: {issue.issue_type}; Assignee: {issue.assignee or 'unassigned'}\n"
        f"Created: {iso(issue.created)}; Updated: {iso(issue.updated)}; "
        f"Due: {iso(issue.duedate) or 'none'}; Resolved: {iso(issue.resolutiondate) or 'no'}\n"
        f"Labels: {', '.join(issue.labels) or 'none'}\n"
        f"{truncate(issue.description, 700)}"
    )
    if issue.links:
        content += "\nLinks: " + "; ".join(f"{link.type} {link.issue_key} ({link.issue_status})" for link in issue.links)
    return {
        "source_type": "issue",
        "source_id": issue.key,
        "title": f"{issue.key} {issue.summary}",
        "url": issue.url,
        "timestamp": iso(issue.updated),
        "content": content,
        "data": {
            "key": issue.key, "project_key": issue.project_key, "summary": issue.summary,
            "status": issue.status, "status_category": issue.status_category,
            "priority": issue.priority, "issue_type": issue.issue_type, "assignee": issue.assignee,
            "labels": issue.labels, "created": iso(issue.created), "updated": iso(issue.updated),
            "duedate": iso(issue.duedate), "resolutiondate": iso(issue.resolutiondate),
            "resolved": issue.is_resolved, "parent_key": issue.parent_key,
            "links": [link.model_dump() for link in issue.links],
        },
    }


def comment_item(comment: Comment) -> dict:
    return {
        "source_type": "comment",
        "source_id": f"{comment.issue_key}/comment/{comment.id}",
        "title": f"Comment on {comment.issue_key} by {comment.author or 'unknown'}",
        "url": comment.url,
        "timestamp": iso(comment.created),
        "content": f"{comment.author or 'unknown'} on {iso(comment.created)}: {truncate(comment.body, 900)}",
        "data": {"issue_key": comment.issue_key, "author": comment.author, "comment_id": comment.id},
    }


def changelog_item(entry: ChangelogEntry) -> dict:
    changes = "; ".join(
        f"{item.field}: {item.from_value or 'none'} -> {item.to_value or 'none'}" for item in entry.items
    )
    return {
        "source_type": "changelog",
        "source_id": f"{entry.issue_key}/changelog/{entry.id}",
        "title": f"History on {entry.issue_key}",
        "url": entry.url,
        "timestamp": iso(entry.created),
        "content": f"{entry.author or 'unknown'} changed {entry.issue_key} on {iso(entry.created)}: {changes}",
        "data": {
            "issue_key": entry.issue_key,
            "author": entry.author,
            "created": iso(entry.created),
            "items": [item.model_dump() for item in entry.items],
            "fields": [item.field for item in entry.items],
        },
    }


def project_item(project: Project) -> dict:
    goals = project.raw_extra.get("goals") or []
    goal_text = "\n".join(f"- {g}" for g in goals)
    return {
        "source_type": "project",
        "source_id": project.key,
        "title": f"{project.key} - {project.name}",
        "url": project.url,
        "timestamp": iso(utcnow()),
        "content": f"{project.key} ({project.name}); lead: {project.lead or 'unknown'}\n"
                   f"{project.description}" + (f"\nStated goals:\n{goal_text}" if goal_text else ""),
        "data": {"key": project.key, "name": project.name, "lead": project.lead,
                 "category": project.category, "goals": goals},
    }


# --- tools ------------------------------------------------------------------
@_guard
async def search_projects(ctx: ToolContext, query: str = "", limit: int = 25) -> dict:
    projects = await ctx.connector.search_projects(query or None, limit=clamp_limit(limit, 25, 50))
    if ctx.allowed_projects:
        projects = [p for p in projects if p.key.upper() in ctx.allowed_projects]
    return _envelope("search_projects", ctx, [project_item(p) for p in projects])


@_guard
async def get_project(ctx: ToolContext, project_key: str) -> dict:
    key = validate_project_key(project_key, ctx.allowed_projects)
    project = await ctx.connector.get_project(key)
    if project is None:
        return _error("get_project", "not_found", f"Project {key} does not exist or is not visible")
    return _envelope("get_project", ctx, [project_item(project)])


@_guard
async def search_issues(
    ctx: ToolContext,
    project_key: str = "",
    text: str = "",
    statuses: Optional[Sequence[str]] = None,
    status_category: str = "",
    priorities: Optional[Sequence[str]] = None,
    issue_types: Optional[Sequence[str]] = None,
    labels: Optional[Sequence[str]] = None,
    assignee: str = "",
    created_after: str = "",
    created_before: str = "",
    updated_after: str = "",
    updated_before: str = "",
    resolved_after: str = "",
    resolved_before: str = "",
    due_after: str = "",
    due_before: str = "",
    unresolved_only: bool = False,
    order_by: str = "updated",
    limit: int = 25,
) -> dict:
    key = validate_project_key(project_key, ctx.allowed_projects) if project_key else None
    query = IssueQuery(
        project_key=key,
        text=text or None,
        statuses=list(statuses or []),
        status_category=status_category or None,
        priorities=list(priorities or []),
        issue_types=list(issue_types or []),
        labels=list(labels or []),
        assignee=assignee or None,
        created_after=parse_dt(created_after),
        created_before=parse_dt(created_before),
        updated_after=parse_dt(updated_after),
        updated_before=parse_dt(updated_before),
        resolved_after=parse_dt(resolved_after),
        resolved_before=parse_dt(resolved_before),
        due_after=parse_dt(due_after),
        due_before=parse_dt(due_before),
        unresolved_only=unresolved_only,
        order_by=order_by if order_by in {"updated", "created", "duedate", "priority"} else "updated",
    )
    limit = clamp_limit(limit, 25, ctx.max_items)
    issues = await ctx.connector.iter_issues(query, page_size=50, max_items=limit)
    if ctx.allowed_projects:
        issues = [i for i in issues if i.project_key.upper() in ctx.allowed_projects]
    return _envelope("search_issues", ctx, [issue_item(i) for i in issues],
                     truncated=len(issues) >= limit, jql_filters=query.model_dump(mode="json", exclude_none=True))


@_guard
async def get_issue(ctx: ToolContext, issue_key: str) -> dict:
    key = validate_issue_key(issue_key, ctx.allowed_projects)
    issue = await ctx.connector.get_issue(key)
    if issue is None:
        return _error("get_issue", "not_found", f"Issue {key} does not exist or is not visible")
    return _envelope("get_issue", ctx, [issue_item(issue)])


@_guard
async def get_issue_comments(ctx: ToolContext, issue_key: str, limit: int = 20) -> dict:
    key = validate_issue_key(issue_key, ctx.allowed_projects)
    comments = await ctx.connector.get_issue_comments(key, limit=clamp_limit(limit, 20, 50))
    return _envelope("get_issue_comments", ctx, [comment_item(c) for c in comments])


@_guard
async def get_issue_changelog(
    ctx: ToolContext, issue_key: str, fields: Optional[Sequence[str]] = None, limit: int = 50
) -> dict:
    key = validate_issue_key(issue_key, ctx.allowed_projects)
    entries = await ctx.connector.get_issue_changelog(key, limit=clamp_limit(limit, 50, 100))
    wanted = {f.lower() for f in (fields or [])}
    if wanted:
        entries = [
            e.model_copy(update={"items": [i for i in e.items if i.field.lower() in wanted]})
            for e in entries
        ]
        entries = [e for e in entries if e.items]
    return _envelope("get_issue_changelog", ctx, [changelog_item(e) for e in entries])


@_guard
async def get_project_activity(
    ctx: ToolContext, project_key: str, start: str = "", end: str = "", limit: int = 60
) -> dict:
    """Everything that happened in a project inside a time window: creations, resolutions,
    status transitions, due-date moves and comments, ordered chronologically."""
    key = validate_project_key(project_key, ctx.allowed_projects)
    start_dt, end_dt = parse_dt(start), parse_dt(end)
    limit = clamp_limit(limit, 60, ctx.max_items)

    touched = await ctx.connector.iter_issues(
        IssueQuery(project_key=key, updated_after=start_dt, updated_before=None, order_by="updated"),
        page_size=50, max_items=limit,
    )
    items: list[dict] = []
    counters = {"created": 0, "resolved": 0, "status_change": 0, "duedate_change": 0, "comment": 0}

    for issue in touched:
        if start_dt and issue.created and start_dt <= issue.created and (not end_dt or issue.created <= end_dt):
            counters["created"] += 1
            items.append({
                "source_type": "issue", "source_id": issue.key, "title": f"Created {issue.key}",
                "url": issue.url, "timestamp": iso(issue.created),
                "content": f"{issue.key} created on {iso(issue.created)}: {issue.summary} "
                           f"(type {issue.issue_type}, priority {issue.priority or 'unset'})",
                "data": {"kind": "created", "issue_key": issue.key, "summary": issue.summary,
                         "priority": issue.priority, "project_key": issue.project_key},
            })
        if issue.resolutiondate and (not start_dt or issue.resolutiondate >= start_dt) and \
                (not end_dt or issue.resolutiondate <= end_dt):
            counters["resolved"] += 1
            items.append({
                "source_type": "issue", "source_id": issue.key, "title": f"Resolved {issue.key}",
                "url": issue.url, "timestamp": iso(issue.resolutiondate),
                "content": f"{issue.key} resolved on {iso(issue.resolutiondate)}: {issue.summary}",
                "data": {"kind": "resolved", "issue_key": issue.key, "summary": issue.summary,
                         "project_key": issue.project_key},
            })
        try:
            entries = await ctx.connector.get_issue_changelog(issue.key, limit=50)
        except ConnectorError:
            entries = []
        for entry in entries:
            if not entry.created:
                continue
            if start_dt and entry.created < start_dt:
                continue
            if end_dt and entry.created > end_dt:
                continue
            for item in entry.items:
                kind = "duedate_change" if item.field.lower() in {"duedate", "due date"} else (
                    "status_change" if item.field.lower() == "status" else "field_change")
                if kind in counters:
                    counters[kind] += 1
                items.append({
                    "source_type": "changelog",
                    "source_id": f"{issue.key}/changelog/{entry.id}",
                    "title": f"{item.field} change on {issue.key}",
                    "url": entry.url,
                    "timestamp": iso(entry.created),
                    "content": f"{entry.author or 'unknown'} changed {item.field} on {issue.key} "
                               f"from '{item.from_value or 'none'}' to '{item.to_value or 'none'}' "
                               f"on {iso(entry.created)} ({issue.summary})",
                    "data": {"kind": kind, "issue_key": issue.key, "field": item.field,
                             "from": item.from_value, "to": item.to_value},
                })
        try:
            comments = await ctx.connector.get_issue_comments(issue.key, limit=30)
        except ConnectorError:
            comments = []
        for comment in comments:
            if not comment.created:
                continue
            if start_dt and comment.created < start_dt:
                continue
            if end_dt and comment.created > end_dt:
                continue
            counters["comment"] += 1
            item = comment_item(comment)
            item["data"]["kind"] = "comment"
            items.append(item)

    items.sort(key=lambda i: i["timestamp"] or "")
    truncated = len(items) > limit * 3
    return _envelope("get_project_activity", ctx, items[: limit * 3], truncated=truncated,
                     window={"start": iso(start_dt), "end": iso(end_dt)}, counters=counters,
                     issues_touched=len(touched))


@_guard
async def get_blocked_issues(ctx: ToolContext, project_key: str = "", limit: int = 25) -> dict:
    """Issues that are blocked by status, by label, or by an unresolved 'is blocked by' link."""
    key = validate_project_key(project_key, ctx.allowed_projects) if project_key else None
    issues = await ctx.connector.iter_issues(
        IssueQuery(project_key=key, unresolved_only=True, order_by="updated"),
        page_size=50, max_items=clamp_limit(limit, 25, ctx.max_items) * 4,
    )
    blocked: list[dict] = []
    for issue in issues:
        reasons: list[str] = []
        if issue.status.lower() in BLOCKED_STATUSES:
            reasons.append(f"status is '{issue.status}'")
        if {label.lower() for label in issue.labels} & BLOCKED_LABELS:
            reasons.append("carries a blocked label")
        for link in issue.links:
            if "blocked by" in link.type.lower() and link.issue_status.lower() != "done":
                reasons.append(f"blocked by {link.issue_key} ({link.issue_status})")
        if not reasons:
            continue
        item = issue_item(issue)
        item["content"] = f"BLOCKED: {'; '.join(reasons)}\n{item['content']}"
        item["data"]["blocked_reasons"] = reasons
        blocked.append(item)
        if len(blocked) >= clamp_limit(limit, 25, ctx.max_items):
            break
    return _envelope("get_blocked_issues", ctx, blocked)


@_guard
async def get_overdue_issues(ctx: ToolContext, project_key: str = "", as_of: str = "", limit: int = 25) -> dict:
    """Unresolved issues whose due date is in the past relative to `as_of` (default: now)."""
    key = validate_project_key(project_key, ctx.allowed_projects) if project_key else None
    reference = parse_dt(as_of) or utcnow()
    issues = await ctx.connector.iter_issues(
        IssueQuery(project_key=key, unresolved_only=True, due_before=reference, order_by="duedate",
                   order_dir="ASC"),
        page_size=50, max_items=clamp_limit(limit, 25, ctx.max_items),
    )
    items = []
    for issue in issues:
        if not issue.duedate:
            continue
        days = (reference - issue.duedate).days
        item = issue_item(issue)
        item["content"] = f"OVERDUE by {days} day(s) as of {iso(reference)}\n{item['content']}"
        item["data"]["days_overdue"] = days
        items.append(item)
    return _envelope("get_overdue_issues", ctx, items, as_of=iso(reference))


@_guard
async def search_comments(
    ctx: ToolContext, project_key: str, keywords: Optional[Sequence[str]] = None,
    start: str = "", end: str = "", limit: int = 25,
) -> dict:
    """Scan recent comments in a project for risk language or caller-supplied keywords.

    Useful when the question is about *why* something happened - the reasons live in
    discussion, not in structured fields."""
    key = validate_project_key(project_key, ctx.allowed_projects)
    terms = [k.lower() for k in (keywords or RISK_KEYWORDS)]
    start_dt, end_dt = parse_dt(start), parse_dt(end)
    issues = await ctx.connector.iter_issues(
        IssueQuery(project_key=key, updated_after=start_dt, order_by="updated"),
        page_size=50, max_items=clamp_limit(limit, 25, ctx.max_items) * 3,
    )
    matches: list[dict] = []
    for issue in issues:
        try:
            comments = await ctx.connector.get_issue_comments(issue.key, limit=30)
        except ConnectorError:
            continue
        for comment in comments:
            if start_dt and comment.created and comment.created < start_dt:
                continue
            if end_dt and comment.created and comment.created > end_dt:
                continue
            body = comment.body.lower()
            hits = [t for t in terms if t in body]
            if not hits:
                continue
            item = comment_item(comment)
            item["data"]["matched_terms"] = hits
            item["data"]["issue_summary"] = issue.summary
            matches.append(item)
            if len(matches) >= clamp_limit(limit, 25, ctx.max_items):
                return _envelope("search_comments", ctx, matches, truncated=True, matched_on=terms[:8])
    return _envelope("search_comments", ctx, matches, matched_on=terms[:8])


@_guard
async def semantic_search(
    ctx: ToolContext, query: str, project_key: str = "",
    source_types: Optional[Sequence[str]] = None, top_k: int = 6,
) -> dict:
    """Vector search over indexed issue and comment text. Use for open-ended 'what do people
    say about X' questions; use the structured tools for filters, dates and status."""
    if ctx.retriever is None:
        return _error("semantic_search", "unavailable",
                      "Semantic index is not configured in this process")
    key = validate_project_key(project_key, ctx.allowed_projects) if project_key else None
    evidence = await ctx.retriever.retrieve(
        query,
        top_k=clamp_limit(top_k, 6, 20),
        project_keys=[key] if key else None,
        source_types=list(source_types) if source_types else None,
        tool_used="semantic_search",
    )
    items = [
        {
            "source_type": e.source_type, "source_id": e.source_id, "title": e.title,
            "url": e.url, "timestamp": iso(e.timestamp), "content": e.content,
            "data": {"relevance": e.relevance, "injection_suspected": e.injection_suspected,
                     **e.metadata},
        }
        for e in evidence
    ]
    return _envelope("semantic_search", ctx, items, query=query)


TOOL_FUNCTIONS = {
    "search_projects": search_projects,
    "get_project": get_project,
    "search_issues": search_issues,
    "get_issue": get_issue,
    "get_issue_comments": get_issue_comments,
    "get_issue_changelog": get_issue_changelog,
    "get_project_activity": get_project_activity,
    "get_blocked_issues": get_blocked_issues,
    "get_overdue_issues": get_overdue_issues,
    "search_comments": search_comments,
    "semantic_search": semantic_search,
}
