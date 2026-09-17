"""DEMO-mode connector.

It reads Jira-shaped JSON documents from disk **on every call** (no in-memory freezing), so a
script that edits those files while the server is running produces genuinely changing data -
exactly the behaviour the live connector has. Same interface, same normalised models, same
error types. It is a local *source of truth stand-in*, not a pre-baked answer store.
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from app.connectors.base import (
    ConnectorError,
    IssueQuery,
    NotFoundError,
    SourceConnector,
    UnavailableError,
)
from app.models.domain import (
    ChangeItem,
    ChangelogEntry,
    Comment,
    Issue,
    IssueLink,
    Page,
    Project,
    SourceHealth,
)
from app.utils.dates import parse_dt, utcnow
from app.utils.text import collapse_whitespace

BASE_BROWSE_URL = "https://demo.atlassian.net/browse"
_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


class MockJiraConnector(SourceConnector):
    source_name = "jira"
    mode = "demo"

    def __init__(self, data_dir: str | Path, fail_mode: str = ""):
        self.data_dir = Path(data_dir)
        # fail_mode lets the failure tests exercise unavailable/auth paths deterministically.
        self.fail_mode = fail_mode

    # --- io ---------------------------------------------------------------
    def _read(self, name: str) -> Any:
        if self.fail_mode == "unavailable":
            raise UnavailableError("Demo source unavailable (simulated)")
        path = self.data_dir / name
        if not path.exists():
            raise UnavailableError(
                f"Demo dataset missing: {path}. Run `python scripts/seed_mock_jira.py` first."
            )
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:  # corrupted file behaves like a malformed API reply
            raise ConnectorError(f"Malformed demo dataset {name}: {exc}") from exc

    async def _projects_raw(self) -> list[dict]:
        return await asyncio.to_thread(self._read, "projects.json")

    async def _issues_raw(self) -> list[dict]:
        return await asyncio.to_thread(self._read, "issues.json")

    # --- normalisation ----------------------------------------------------
    def _to_project(self, raw: dict) -> Project:
        return Project(
            key=raw["key"],
            id=str(raw.get("id", "")),
            name=raw.get("name", ""),
            description=collapse_whitespace(raw.get("description", "")),
            lead=raw.get("lead", ""),
            category=raw.get("category", ""),
            url=f"{BASE_BROWSE_URL}/{raw['key']}",
            source="jira",
            raw_extra={"goals": raw.get("goals", []), "start_date": raw.get("start_date", "")},
        )

    def _to_issue(self, raw: dict) -> Issue:
        return Issue(
            key=raw["key"],
            id=str(raw.get("id", "")),
            project_key=raw["project_key"],
            summary=raw.get("summary", ""),
            description=collapse_whitespace(raw.get("description", "")),
            status=raw.get("status", ""),
            status_category=raw.get("status_category", ""),
            issue_type=raw.get("issue_type", ""),
            priority=raw.get("priority", ""),
            assignee=raw.get("assignee", ""),
            reporter=raw.get("reporter", ""),
            labels=list(raw.get("labels", [])),
            components=list(raw.get("components", [])),
            created=parse_dt(raw.get("created")),
            updated=parse_dt(raw.get("updated")),
            duedate=parse_dt(raw.get("duedate")),
            resolutiondate=parse_dt(raw.get("resolutiondate")),
            parent_key=raw.get("parent_key", "") or "",
            sprint=raw.get("sprint", ""),
            story_points=raw.get("story_points"),
            links=[IssueLink(**link) for link in raw.get("links", [])],
            url=f"{BASE_BROWSE_URL}/{raw['key']}",
            source="jira",
        )

    # --- interface --------------------------------------------------------
    async def health(self) -> SourceHealth:
        try:
            projects = await self._projects_raw()
            return SourceHealth(
                source="jira",
                mode=self.mode,
                ok=True,
                detail=f"demo dataset loaded ({len(projects)} projects) from {self.data_dir}",
            )
        except ConnectorError as exc:
            return SourceHealth(source="jira", mode=self.mode, ok=False, detail=str(exc))

    async def search_projects(self, query: Optional[str] = None, limit: int = 25) -> list[Project]:
        raw = await self._projects_raw()
        projects = [self._to_project(p) for p in raw]
        if query:
            q = query.lower().strip()
            projects = [
                p for p in projects
                if q in p.key.lower() or q in p.name.lower() or q in p.description.lower()
            ]
        return projects[:limit]

    async def get_project(self, key: str) -> Optional[Project]:
        for raw in await self._projects_raw():
            if raw["key"].upper() == key.upper():
                return self._to_project(raw)
        return None

    async def search_issues(self, query: IssueQuery, limit: int = 50, cursor: Optional[str] = None) -> Page:
        issues = [self._to_issue(r) for r in await self._issues_raw()]
        issues = [i for i in issues if _matches(i, query)]
        issues.sort(key=lambda i: _sort_key(i, query.order_by), reverse=query.order_dir.upper() == "DESC")

        offset = int(cursor) if cursor and cursor.isdigit() else 0
        window = issues[offset: offset + limit]
        next_offset = offset + len(window)
        is_last = next_offset >= len(issues)
        return Page(
            items=window,
            next_cursor=None if is_last else str(next_offset),
            is_last=is_last,
            fetched_at=utcnow(),
        )

    async def get_issue(self, key: str) -> Optional[Issue]:
        for raw in await self._issues_raw():
            if raw["key"].upper() == key.upper():
                return self._to_issue(raw)
        return None

    async def get_issue_comments(self, key: str, limit: int = 50) -> list[Comment]:
        for raw in await self._issues_raw():
            if raw["key"].upper() == key.upper():
                comments = [
                    Comment(
                        id=str(c["id"]),
                        issue_key=raw["key"],
                        author=c.get("author", ""),
                        body=collapse_whitespace(c.get("body", "")),
                        created=parse_dt(c.get("created")),
                        updated=parse_dt(c.get("updated") or c.get("created")),
                        url=f"{BASE_BROWSE_URL}/{raw['key']}?focusedCommentId={c['id']}",
                        source="jira",
                    )
                    for c in raw.get("comments", [])
                ]
                comments.sort(key=lambda c: c.created or _EPOCH)
                return comments[:limit]
        raise NotFoundError(f"Issue {key} not found")

    async def get_issue_changelog(self, key: str, limit: int = 100) -> list[ChangelogEntry]:
        for raw in await self._issues_raw():
            if raw["key"].upper() == key.upper():
                entries = [
                    ChangelogEntry(
                        id=str(h["id"]),
                        issue_key=raw["key"],
                        author=h.get("author", ""),
                        created=parse_dt(h.get("created")),
                        items=[
                            ChangeItem(
                                field=item.get("field", ""),
                                from_value=item.get("from"),
                                to_value=item.get("to"),
                            )
                            for item in h.get("items", [])
                        ],
                        url=f"{BASE_BROWSE_URL}/{raw['key']}?page=history",
                        source="jira",
                    )
                    for h in raw.get("changelog", [])
                ]
                entries.sort(key=lambda e: e.created or utcnow())
                return entries[:limit]
        raise NotFoundError(f"Issue {key} not found")


PRIORITY_RANK = {"highest": 5, "high": 4, "medium": 3, "low": 2, "lowest": 1}


def _sort_key(issue: Issue, order_by: str):
    """Return a homogeneous sort key for the requested ordering."""
    if order_by == "priority":
        return PRIORITY_RANK.get(issue.priority.lower(), 0)
    if order_by == "created":
        return issue.created or _EPOCH
    if order_by == "duedate":
        return issue.duedate or _EPOCH
    return issue.updated or _EPOCH


def _matches(issue: Issue, q: IssueQuery) -> bool:
    if q.project_key and issue.project_key.upper() != q.project_key.upper():
        return False
    if q.statuses and issue.status.lower() not in {s.lower() for s in q.statuses}:
        return False
    if q.status_category and issue.status_category.lower() != q.status_category.lower():
        return False
    if q.priorities and issue.priority.lower() not in {p.lower() for p in q.priorities}:
        return False
    if q.issue_types and issue.issue_type.lower() not in {t.lower() for t in q.issue_types}:
        return False
    if q.labels and not ({label.lower() for label in issue.labels} & {label.lower() for label in q.labels}):
        return False
    if q.assignee and q.assignee.lower() not in issue.assignee.lower():
        return False
    if q.unresolved_only and issue.is_resolved:
        return False
    if q.text:
        haystack = f"{issue.key} {issue.summary} {issue.description} {' '.join(issue.labels)}".lower()
        if not all(term in haystack for term in q.text.lower().split()):
            return False
    checks = [
        (issue.created, q.created_after, q.created_before),
        (issue.updated, q.updated_after, q.updated_before),
        (issue.resolutiondate, q.resolved_after, q.resolved_before),
        (issue.duedate, q.due_after, q.due_before),
    ]
    for value, after, before in checks:
        if after is not None:
            if value is None or value < after:
                return False
        if before is not None:
            if value is None or value > before:
                return False
    return True
