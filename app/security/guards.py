"""Authorisation, tool-argument validation and budget guards."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable

PROJECT_KEY_RE = re.compile(r"^[A-Z][A-Z0-9_]{1,29}$")
ISSUE_KEY_RE = re.compile(r"^[A-Z][A-Z0-9_]{1,29}-\d{1,9}$")
USER_ID_RE = re.compile(r"^[A-Za-z0-9._@\-]{1,64}$")


class AccessDenied(Exception):
    """Raised when a caller asks for a project outside the configured allow-list."""


class InvalidArgument(Exception):
    """Raised when a tool argument fails validation before it reaches the source system."""


def validate_project_key(key: str, allowed: Iterable[str] | None = None) -> str:
    key = (key or "").strip().upper()
    if not PROJECT_KEY_RE.match(key):
        raise InvalidArgument(f"Invalid project key: {key!r}")
    allowed = {a.upper() for a in (allowed or [])}
    if allowed and key not in allowed:
        raise AccessDenied(f"Project {key} is not in the configured allow-list")
    return key


def validate_issue_key(key: str, allowed: Iterable[str] | None = None) -> str:
    key = (key or "").strip().upper()
    if not ISSUE_KEY_RE.match(key):
        raise InvalidArgument(f"Invalid issue key: {key!r}")
    project = key.split("-", 1)[0]
    validate_project_key(project, allowed)
    return key


def validate_user_id(user_id: str) -> str:
    user_id = (user_id or "").strip()
    if not USER_ID_RE.match(user_id):
        raise InvalidArgument("Invalid user id")
    return user_id


def clamp_limit(value: Any, default: int = 25, maximum: int = 100) -> int:
    try:
        value = int(value)
    except (TypeError, ValueError):
        return default
    return max(1, min(value, maximum))


@dataclass
class ExecutionBudget:
    """Hard stop on agent work. Prevents infinite tool loops and context explosion."""

    max_tool_calls: int = 14
    max_rounds: int = 3
    max_evidence: int = 40
    tool_calls_used: int = 0
    rounds_used: int = 0
    seen_calls: set[str] = field(default_factory=set)

    def can_call(self) -> bool:
        return self.tool_calls_used < self.max_tool_calls

    def register_call(self, signature: str) -> str:
        """Return 'ok', 'duplicate' or 'budget_exceeded' for a proposed tool call."""
        if signature in self.seen_calls:
            return "duplicate"
        if not self.can_call():
            return "budget_exceeded"
        self.seen_calls.add(signature)
        self.tool_calls_used += 1
        return "ok"

    def next_round(self) -> bool:
        if self.rounds_used >= self.max_rounds:
            return False
        self.rounds_used += 1
        return True

    def snapshot(self) -> dict[str, int]:
        return {
            "tool_calls_used": self.tool_calls_used,
            "max_tool_calls": self.max_tool_calls,
            "rounds_used": self.rounds_used,
            "max_rounds": self.max_rounds,
        }
