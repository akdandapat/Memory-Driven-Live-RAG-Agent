"""Connector abstraction.

The agent never talks to Jira. It talks to MCP tools, which talk to a *connector* that
implements this interface. Adding Gmail / Notion / Slack / GitHub later means implementing
`SourceConnector` again and registering it in the factory - the planner, executor, RAG,
memory and citation layers do not change.
"""
from __future__ import annotations

import abc
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

from app.models.domain import ChangelogEntry, Comment, Issue, Page, Project, SourceHealth


class ConnectorError(Exception):
    """Base class for connector failures (network, auth, rate limit, bad response)."""

    def __init__(self, message: str, *, kind: str = "error", retryable: bool = False):
        super().__init__(message)
        self.kind = kind
        self.retryable = retryable


class AuthError(ConnectorError):
    def __init__(self, message: str):
        super().__init__(message, kind="auth", retryable=False)


class RateLimitError(ConnectorError):
    def __init__(self, message: str, retry_after: float = 1.0):
        super().__init__(message, kind="rate_limit", retryable=True)
        self.retry_after = retry_after


class UnavailableError(ConnectorError):
    def __init__(self, message: str):
        super().__init__(message, kind="unavailable", retryable=True)


class NotFoundError(ConnectorError):
    def __init__(self, message: str):
        super().__init__(message, kind="not_found", retryable=False)


class IssueQuery(BaseModel):
    """Source-agnostic issue filter. The connector translates it (Jira -> JQL)."""

    project_key: Optional[str] = None
    text: Optional[str] = None
    statuses: list[str] = Field(default_factory=list)
    status_category: Optional[str] = None          # "To Do" | "In Progress" | "Done"
    priorities: list[str] = Field(default_factory=list)
    issue_types: list[str] = Field(default_factory=list)
    labels: list[str] = Field(default_factory=list)
    assignee: Optional[str] = None
    created_after: Optional[datetime] = None
    created_before: Optional[datetime] = None
    updated_after: Optional[datetime] = None
    updated_before: Optional[datetime] = None
    resolved_after: Optional[datetime] = None
    resolved_before: Optional[datetime] = None
    due_after: Optional[datetime] = None
    due_before: Optional[datetime] = None
    unresolved_only: bool = False
    order_by: str = "updated"                       # updated | created | duedate | priority
    order_dir: str = "DESC"


class SourceConnector(abc.ABC):
    """Read-only interface over a live data source."""

    source_name: str = "generic"
    mode: str = "live"

    @abc.abstractmethod
    async def health(self) -> SourceHealth: ...

    @abc.abstractmethod
    async def search_projects(self, query: Optional[str] = None, limit: int = 25) -> list[Project]: ...

    @abc.abstractmethod
    async def get_project(self, key: str) -> Optional[Project]: ...

    @abc.abstractmethod
    async def search_issues(
        self, query: IssueQuery, limit: int = 50, cursor: Optional[str] = None
    ) -> Page: ...

    @abc.abstractmethod
    async def get_issue(self, key: str) -> Optional[Issue]: ...

    @abc.abstractmethod
    async def get_issue_comments(self, key: str, limit: int = 50) -> list[Comment]: ...

    @abc.abstractmethod
    async def get_issue_changelog(self, key: str, limit: int = 100) -> list[ChangelogEntry]: ...

    async def close(self) -> None:
        """Release network resources. Default: nothing to do."""
        return None

    async def iter_issues(self, query: IssueQuery, page_size: int = 50, max_items: int = 500) -> list[Issue]:
        """Convenience pagination helper shared by every connector implementation."""
        collected: list[Issue] = []
        cursor: Optional[str] = None
        while len(collected) < max_items:
            page = await self.search_issues(query, limit=min(page_size, max_items - len(collected)), cursor=cursor)
            collected.extend(page.items)
            if page.is_last or not page.next_cursor:
                break
            cursor = page.next_cursor
        return collected[:max_items]
