"""Source-agnostic domain models.

Every connector (Jira today, Gmail/Notion/Slack tomorrow) normalises its payloads into
these shapes, so the agent, RAG layer and citation layer never learn Jira's wire format.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class Project(BaseModel):
    key: str
    id: str = ""
    name: str = ""
    description: str = ""
    lead: str = ""
    url: str = ""
    category: str = ""
    source: str = "jira"
    raw_extra: dict[str, Any] = Field(default_factory=dict)


class Comment(BaseModel):
    id: str
    issue_key: str
    author: str = ""
    body: str = ""
    created: Optional[datetime] = None
    updated: Optional[datetime] = None
    url: str = ""
    source: str = "jira"


class ChangeItem(BaseModel):
    field: str
    from_value: Optional[str] = None
    to_value: Optional[str] = None


class ChangelogEntry(BaseModel):
    id: str
    issue_key: str
    author: str = ""
    created: Optional[datetime] = None
    items: list[ChangeItem] = Field(default_factory=list)
    url: str = ""
    source: str = "jira"


class IssueLink(BaseModel):
    type: str
    direction: Literal["inward", "outward"] = "outward"
    issue_key: str
    issue_summary: str = ""
    issue_status: str = ""


class Issue(BaseModel):
    key: str
    id: str = ""
    project_key: str
    summary: str = ""
    description: str = ""
    status: str = ""
    status_category: str = ""
    issue_type: str = ""
    priority: str = ""
    assignee: str = ""
    reporter: str = ""
    labels: list[str] = Field(default_factory=list)
    components: list[str] = Field(default_factory=list)
    created: Optional[datetime] = None
    updated: Optional[datetime] = None
    duedate: Optional[datetime] = None
    resolutiondate: Optional[datetime] = None
    parent_key: str = ""
    sprint: str = ""
    story_points: Optional[float] = None
    links: list[IssueLink] = Field(default_factory=list)
    url: str = ""
    source: str = "jira"

    @property
    def is_resolved(self) -> bool:
        return bool(self.resolutiondate) or self.status_category.lower() == "done"


class Page(BaseModel):
    """Cursor-based page. Jira Cloud's enhanced JQL search uses nextPageToken."""

    items: list[Any] = Field(default_factory=list)
    next_cursor: Optional[str] = None
    is_last: bool = True
    fetched_at: datetime = Field(default_factory=datetime.utcnow)


class ActivityEvent(BaseModel):
    """A normalised 'something happened' record used for temporal reasoning."""

    kind: Literal["created", "resolved", "status_change", "duedate_change", "field_change", "comment"]
    issue_key: str
    project_key: str
    timestamp: Optional[datetime] = None
    actor: str = ""
    field: str = ""
    from_value: Optional[str] = None
    to_value: Optional[str] = None
    text: str = ""
    url: str = ""
    source_id: str = ""


class SourceHealth(BaseModel):
    source: str
    mode: str
    ok: bool
    detail: str = ""
    checked_at: datetime = Field(default_factory=datetime.utcnow)
