"""LIVE-mode Jira Cloud connector (REST API v3).

API notes (verified against Atlassian docs, 2026):
  * `GET/POST /rest/api/3/search` was removed from Jira Cloud. Issue search now uses the
    enhanced JQL endpoint `POST /rest/api/3/search/jql`, which paginates with
    `nextPageToken` instead of `startAt` and does not return a `total`.
  * Descriptions and comment bodies come back as ADF (Atlassian Document Format) JSON on
    the v3 API and must be flattened to text before indexing - see utils.text.adf_to_text.
  * Changelog is fetched via `GET /rest/api/3/issue/{key}/changelog` (paginated with
    startAt/maxResults) and comments via `GET /rest/api/3/issue/{key}/comment`.
  * Auth is HTTP Basic with `email:api_token` over HTTPS.
"""
from __future__ import annotations

import asyncio
import base64
from datetime import datetime
from typing import Any, Optional

import httpx

from app.connectors.base import (
    AuthError,
    ConnectorError,
    IssueQuery,
    NotFoundError,
    RateLimitError,
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
from app.utils.dates import jira_datetime, parse_dt, utcnow
from app.utils.text import adf_to_text, collapse_whitespace

ISSUE_FIELDS = [
    "summary", "description", "status", "issuetype", "priority", "assignee", "reporter",
    "labels", "components", "created", "updated", "duedate", "resolutiondate", "parent",
    "issuelinks",
]


def _jql_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def build_jql(query: IssueQuery) -> str:
    """Translate the source-agnostic IssueQuery into JQL."""
    clauses: list[str] = []
    if query.project_key:
        clauses.append(f'project = "{_jql_escape(query.project_key)}"')
    if query.statuses:
        joined = ", ".join(f'"{_jql_escape(s)}"' for s in query.statuses)
        clauses.append(f"status IN ({joined})")
    if query.status_category:
        clauses.append(f'statusCategory = "{_jql_escape(query.status_category)}"')
    if query.priorities:
        joined = ", ".join(f'"{_jql_escape(p)}"' for p in query.priorities)
        clauses.append(f"priority IN ({joined})")
    if query.issue_types:
        joined = ", ".join(f'"{_jql_escape(t)}"' for t in query.issue_types)
        clauses.append(f"issuetype IN ({joined})")
    if query.labels:
        joined = ", ".join(f'"{_jql_escape(label)}"' for label in query.labels)
        clauses.append(f"labels IN ({joined})")
    if query.assignee:
        clauses.append(f'assignee = "{_jql_escape(query.assignee)}"')
    if query.unresolved_only:
        clauses.append("resolution = EMPTY")
    if query.text:
        clauses.append(f'text ~ "{_jql_escape(query.text)}"')

    date_clauses = [
        ("created", ">=", query.created_after), ("created", "<=", query.created_before),
        ("updated", ">=", query.updated_after), ("updated", "<=", query.updated_before),
        ("resolutiondate", ">=", query.resolved_after), ("resolutiondate", "<=", query.resolved_before),
        ("duedate", ">=", query.due_after), ("duedate", "<=", query.due_before),
    ]
    for field, op, value in date_clauses:
        if value is not None:
            clauses.append(f'{field} {op} "{jira_datetime(value)}"')

    jql = " AND ".join(clauses) if clauses else "order by updated DESC"
    if clauses:
        order_field = {"updated": "updated", "created": "created", "duedate": "duedate",
                       "priority": "priority"}.get(query.order_by, "updated")
        direction = "DESC" if query.order_dir.upper() == "DESC" else "ASC"
        jql = f"{jql} ORDER BY {order_field} {direction}"
    return jql


class JiraConnector(SourceConnector):
    source_name = "jira"
    mode = "live"

    def __init__(
        self,
        base_url: str,
        email: str,
        api_token: str,
        *,
        timeout: float = 30.0,
        max_retries: int = 3,
        page_size: int = 50,
        client: Optional[httpx.AsyncClient] = None,
    ):
        if not (base_url and email and api_token):
            raise AuthError("Jira credentials are incomplete (JIRA_BASE_URL / JIRA_EMAIL / JIRA_API_TOKEN)")
        self.base_url = base_url.rstrip("/")
        self.page_size = page_size
        self.max_retries = max_retries
        token = base64.b64encode(f"{email}:{api_token}".encode()).decode()
        self._client = client or httpx.AsyncClient(
            base_url=self.base_url,
            timeout=timeout,
            headers={
                "Authorization": f"Basic {token}",
                "Accept": "application/json",
                "Content-Type": "application/json",
                "User-Agent": "agentic-rag-live-data/1.0",
            },
        )

    async def close(self) -> None:
        await self._client.aclose()

    # --- transport --------------------------------------------------------
    async def _request(self, method: str, path: str, **kwargs: Any) -> dict:
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                response = await self._client.request(method, path, **kwargs)
            except httpx.TimeoutException as exc:
                last_error = UnavailableError(f"Jira request timed out: {exc}")
            except httpx.HTTPError as exc:
                last_error = UnavailableError(f"Jira transport error: {exc}")
            else:
                if response.status_code in (200, 201, 204):
                    return response.json() if response.content else {}
                if response.status_code in (401, 403):
                    raise AuthError(f"Jira rejected the credentials (HTTP {response.status_code})")
                if response.status_code == 404:
                    raise NotFoundError(f"Jira resource not found: {path}")
                if response.status_code == 429:
                    retry_after = float(response.headers.get("Retry-After", 2 ** attempt))
                    last_error = RateLimitError("Jira rate limit hit", retry_after=retry_after)
                    await asyncio.sleep(min(retry_after, 30.0))
                    continue
                if response.status_code >= 500:
                    last_error = UnavailableError(f"Jira server error HTTP {response.status_code}")
                else:
                    raise ConnectorError(
                        f"Jira returned HTTP {response.status_code}: {response.text[:300]}"
                    )
            await asyncio.sleep(min(2 ** attempt * 0.5, 8.0))
        raise last_error or UnavailableError("Jira request failed")

    # --- normalisation ----------------------------------------------------
    def _browse(self, key: str) -> str:
        return f"{self.base_url}/browse/{key}"

    def _to_issue(self, raw: dict) -> Issue:
        fields = raw.get("fields", {}) or {}
        status = fields.get("status") or {}
        category = (status.get("statusCategory") or {}).get("name", "")
        links: list[IssueLink] = []
        for link in fields.get("issuelinks", []) or []:
            link_type = (link.get("type") or {})
            if link.get("outwardIssue"):
                target, direction = link["outwardIssue"], "outward"
                name = link_type.get("outward", link_type.get("name", "relates to"))
            elif link.get("inwardIssue"):
                target, direction = link["inwardIssue"], "inward"
                name = link_type.get("inward", link_type.get("name", "relates to"))
            else:
                continue
            target_fields = target.get("fields", {}) or {}
            links.append(IssueLink(
                type=name,
                direction=direction,
                issue_key=target.get("key", ""),
                issue_summary=target_fields.get("summary", ""),
                issue_status=((target_fields.get("status") or {}).get("name", "")),
            ))
        key = raw.get("key", "")
        return Issue(
            key=key,
            id=str(raw.get("id", "")),
            project_key=key.split("-")[0] if "-" in key else "",
            summary=fields.get("summary", "") or "",
            description=collapse_whitespace(adf_to_text(fields.get("description"))),
            status=status.get("name", ""),
            status_category=category,
            issue_type=(fields.get("issuetype") or {}).get("name", ""),
            priority=(fields.get("priority") or {}).get("name", ""),
            assignee=(fields.get("assignee") or {}).get("displayName", "") if fields.get("assignee") else "",
            reporter=(fields.get("reporter") or {}).get("displayName", "") if fields.get("reporter") else "",
            labels=list(fields.get("labels", []) or []),
            components=[c.get("name", "") for c in (fields.get("components") or [])],
            created=parse_dt(fields.get("created")),
            updated=parse_dt(fields.get("updated")),
            duedate=parse_dt(fields.get("duedate")),
            resolutiondate=parse_dt(fields.get("resolutiondate")),
            parent_key=(fields.get("parent") or {}).get("key", "") if fields.get("parent") else "",
            links=links,
            url=self._browse(key),
            source="jira",
        )

    # --- interface --------------------------------------------------------
    async def health(self) -> SourceHealth:
        try:
            data = await self._request("GET", "/rest/api/3/myself")
            return SourceHealth(
                source="jira", mode=self.mode, ok=True,
                detail=f"authenticated as {data.get('displayName', 'unknown')}",
            )
        except ConnectorError as exc:
            return SourceHealth(source="jira", mode=self.mode, ok=False, detail=str(exc))

    async def search_projects(self, query: Optional[str] = None, limit: int = 25) -> list[Project]:
        params: dict[str, Any] = {"maxResults": min(limit, 50), "expand": "description,lead"}
        if query:
            params["query"] = query
        data = await self._request("GET", "/rest/api/3/project/search", params=params)
        projects = []
        for raw in data.get("values", []):
            projects.append(Project(
                key=raw.get("key", ""),
                id=str(raw.get("id", "")),
                name=raw.get("name", ""),
                description=collapse_whitespace(raw.get("description", "") or ""),
                lead=(raw.get("lead") or {}).get("displayName", ""),
                category=((raw.get("projectCategory") or {}).get("name", "")),
                url=f"{self.base_url}/browse/{raw.get('key', '')}",
                source="jira",
            ))
        return projects

    async def get_project(self, key: str) -> Optional[Project]:
        try:
            raw = await self._request("GET", f"/rest/api/3/project/{key}")
        except NotFoundError:
            return None
        return Project(
            key=raw.get("key", key),
            id=str(raw.get("id", "")),
            name=raw.get("name", ""),
            description=collapse_whitespace(adf_to_text(raw.get("description")) or raw.get("description", "") or ""),
            lead=(raw.get("lead") or {}).get("displayName", ""),
            category=((raw.get("projectCategory") or {}).get("name", "")),
            url=f"{self.base_url}/browse/{raw.get('key', key)}",
            source="jira",
        )

    async def search_issues(self, query: IssueQuery, limit: int = 50, cursor: Optional[str] = None) -> Page:
        payload: dict[str, Any] = {
            "jql": build_jql(query),
            "maxResults": min(limit, self.page_size),
            "fields": ISSUE_FIELDS,
            "fieldsByKeys": False,
        }
        if cursor:
            payload["nextPageToken"] = cursor
        data = await self._request("POST", "/rest/api/3/search/jql", json=payload)
        issues = [self._to_issue(raw) for raw in data.get("issues", [])]
        next_token = data.get("nextPageToken")
        is_last = bool(data.get("isLast", next_token is None)) or not next_token
        return Page(items=issues, next_cursor=next_token, is_last=is_last, fetched_at=utcnow())

    async def get_issue(self, key: str) -> Optional[Issue]:
        try:
            raw = await self._request(
                "GET", f"/rest/api/3/issue/{key}", params={"fields": ",".join(ISSUE_FIELDS)}
            )
        except NotFoundError:
            return None
        return self._to_issue(raw)

    async def get_issue_comments(self, key: str, limit: int = 50) -> list[Comment]:
        comments: list[Comment] = []
        start_at = 0
        while len(comments) < limit:
            data = await self._request(
                "GET", f"/rest/api/3/issue/{key}/comment",
                params={"startAt": start_at, "maxResults": min(100, limit - len(comments)), "orderBy": "created"},
            )
            batch = data.get("comments", [])
            for raw in batch:
                comments.append(Comment(
                    id=str(raw.get("id", "")),
                    issue_key=key,
                    author=(raw.get("author") or {}).get("displayName", ""),
                    body=collapse_whitespace(adf_to_text(raw.get("body"))),
                    created=parse_dt(raw.get("created")),
                    updated=parse_dt(raw.get("updated") or raw.get("created")),
                    url=f"{self._browse(key)}?focusedCommentId={raw.get('id', '')}",
                    source="jira",
                ))
            start_at += len(batch)
            if not batch or start_at >= int(data.get("total", start_at)):
                break
        return comments[:limit]

    async def get_issue_changelog(self, key: str, limit: int = 100) -> list[ChangelogEntry]:
        entries: list[ChangelogEntry] = []
        start_at = 0
        while len(entries) < limit:
            data = await self._request(
                "GET", f"/rest/api/3/issue/{key}/changelog",
                params={"startAt": start_at, "maxResults": min(100, limit - len(entries))},
            )
            batch = data.get("values", [])
            for raw in batch:
                entries.append(ChangelogEntry(
                    id=str(raw.get("id", "")),
                    issue_key=key,
                    author=(raw.get("author") or {}).get("displayName", ""),
                    created=parse_dt(raw.get("created")),
                    items=[
                        ChangeItem(
                            field=item.get("field", ""),
                            from_value=item.get("fromString"),
                            to_value=item.get("toString"),
                        )
                        for item in raw.get("items", [])
                    ],
                    url=f"{self._browse(key)}?page=history",
                    source="jira",
                ))
            start_at += len(batch)
            if not batch or bool(data.get("isLast", True)):
                break
        entries.sort(key=lambda e: e.created or datetime.min.replace(tzinfo=utcnow().tzinfo))
        return entries[:limit]
