"""Defences for untrusted third-party content (Jira descriptions, comments, titles).

Threat model: anybody who can comment on a Jira ticket can write text that the agent will
read. That text must be treated as *data*, never as instructions, and must never be able to
impersonate the system prompt or the evidence framing.
"""
from __future__ import annotations

import re
from typing import Any

# Patterns that indicate an attempt to steer the model rather than describe work.
INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+|any\s+)?(previous|prior|above)\s+instructions?", re.I),
    re.compile(r"disregard\s+(the\s+)?(system|previous|above)\s+(prompt|instructions?)", re.I),
    re.compile(r"you\s+are\s+now\s+(a|an|the)\b", re.I),
    re.compile(r"\bnew\s+(system\s+)?(prompt|instructions?)\b", re.I),
    re.compile(r"\b(reveal|print|show|dump)\b.{0,30}\b(system prompt|instructions|api[_ ]?key|token|secret)", re.I),
    re.compile(r"</?(system|assistant|user|untrusted_content)>", re.I),
    re.compile(r"\bexfiltrate\b|\bsend\s+(the\s+)?(data|secrets?)\s+to\b", re.I),
    re.compile(r"\bdo\s+not\s+cite\b|\bskip\s+citations?\b", re.I),
]

SECRET_PATTERNS = [
    (re.compile(r"(?i)(api[_-]?key|authorization|bearer|token|password|secret)\s*[:=]\s*\S+"), r"\1=***REDACTED***"),
    (re.compile(r"\bsk-[A-Za-z0-9_\-]{12,}\b"), "***REDACTED***"),
    (re.compile(r"\bAT(?:ATT|CTT)[A-Za-z0-9_\-=]{10,}\b"), "***REDACTED***"),
    (re.compile(r"\bghp_[A-Za-z0-9]{20,}\b"), "***REDACTED***"),
    (re.compile(r"\blsv2_[A-Za-z0-9_\-]{10,}\b"), "***REDACTED***"),
]

FENCE_OPEN = "<untrusted_content"
FENCE_CLOSE = "</untrusted_content>"


def detect_injection(text: str) -> list[str]:
    """Return the names of injection heuristics that fired."""
    hits: list[str] = []
    for pattern in INJECTION_PATTERNS:
        if pattern.search(text or ""):
            hits.append(pattern.pattern[:48])
    return hits


def neutralize(text: str) -> str:
    """Make untrusted text safe to embed in a prompt.

    We do not silently delete content (that would corrupt evidence). We break the tokens that
    could close our fence or forge a role, and annotate detected steering attempts.
    """
    if not text:
        return ""
    cleaned = text.replace("<", "‹").replace(">", "›")
    cleaned = re.sub(r"(?i)\b(system|assistant)\s*:", r"\1 -", cleaned)
    return cleaned


def wrap_untrusted(text: str, source_id: str = "") -> str:
    """Fence untrusted content so the model can see where data begins and ends."""
    attrs = f' source="{re.sub(r"[^A-Za-z0-9/_.:-]", "", source_id)}"' if source_id else ""
    return f"{FENCE_OPEN}{attrs}>\n{neutralize(text)}\n{FENCE_CLOSE}"


def redact_secrets(value: Any) -> Any:
    """Recursively redact anything that looks like a credential before logging/tracing."""
    if isinstance(value, str):
        out = value
        for pattern, replacement in SECRET_PATTERNS:
            out = pattern.sub(replacement, out)
        return out
    if isinstance(value, dict):
        redacted = {}
        for k, v in value.items():
            if re.search(r"(?i)(token|secret|password|api[_-]?key|authorization)", str(k)):
                redacted[k] = "***REDACTED***"
            else:
                redacted[k] = redact_secrets(v)
        return redacted
    if isinstance(value, list):
        return [redact_secrets(v) for v in value]
    return value
