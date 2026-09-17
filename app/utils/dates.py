"""Timestamp parsing/formatting and natural-language time-window resolution."""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Optional

ISO_FORMATS = (
    "%Y-%m-%dT%H:%M:%S.%f%z",
    "%Y-%m-%dT%H:%M:%S%z",
    "%Y-%m-%dT%H:%M:%S.%f",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d",
)

QUARTER_MONTHS = {1: (1, 3), 2: (4, 6), 3: (7, 9), 4: (10, 12)}


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def parse_dt(value: Optional[str | datetime]) -> Optional[datetime]:
    """Parse Jira/ISO timestamps into timezone-aware UTC datetimes."""
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    text = str(value).strip()
    # Jira returns +0530 style offsets; python needs them without a colon for %z on <3.11
    normalised = re.sub(r"([+-]\d{2}):(\d{2})$", r"\1\2", text)
    if normalised.endswith("Z"):
        normalised = normalised[:-1] + "+0000"
    for fmt in ISO_FORMATS:
        try:
            dt = datetime.strptime(normalised, fmt)
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    try:
        dt = datetime.fromisoformat(text)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def iso(value: Optional[datetime]) -> Optional[str]:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def jira_datetime(value: datetime) -> str:
    """Jira JQL datetime literal: 'yyyy-MM-dd HH:mm'."""
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M")


def quarter_bounds(year: int, quarter: int) -> tuple[datetime, datetime]:
    start_month, end_month = QUARTER_MONTHS[quarter]
    start = datetime(year, start_month, 1, tzinfo=timezone.utc)
    if end_month == 12:
        end = datetime(year + 1, 1, 1, tzinfo=timezone.utc) - timedelta(seconds=1)
    else:
        end = datetime(year, end_month + 1, 1, tzinfo=timezone.utc) - timedelta(seconds=1)
    return start, end


def current_quarter(now: Optional[datetime] = None) -> tuple[int, int]:
    now = now or utcnow()
    return now.year, (now.month - 1) // 3 + 1


def resolve_time_window(text: str, now: Optional[datetime] = None) -> dict[str, Optional[str]]:
    """Turn phrases such as 'in Q2', 'last 30 days', 'since May 2026' into an explicit window.

    Returns {'start': iso|None, 'end': iso|None, 'label': str}. A missing start/end means
    'unbounded' and the caller decides on a default.
    """
    now = now or utcnow()
    low = text.lower()

    m = re.search(r"\bq([1-4])\s*(?:of\s*)?(\d{4})?\b", low)
    if m:
        quarter = int(m.group(1))
        year = int(m.group(2)) if m.group(2) else now.year
        # "in Q4" asked in Q1 usually means the quarter that already happened.
        if not m.group(2) and quarter > (now.month - 1) // 3 + 1:
            year -= 1
        start, end = quarter_bounds(year, quarter)
        return {"start": iso(start), "end": iso(end), "label": f"Q{quarter} {year}"}

    if "this quarter" in low or "current quarter" in low:
        year, quarter = current_quarter(now)
        start, end = quarter_bounds(year, quarter)
        return {"start": iso(start), "end": iso(end), "label": f"Q{quarter} {year} (current)"}

    if "last quarter" in low or "previous quarter" in low:
        year, quarter = current_quarter(now)
        quarter -= 1
        if quarter == 0:
            quarter, year = 4, year - 1
        start, end = quarter_bounds(year, quarter)
        return {"start": iso(start), "end": iso(end), "label": f"Q{quarter} {year}"}

    m = re.search(r"\b(?:last|past|previous)\s+(\d{1,3})\s*(day|days|week|weeks|month|months)\b", low)
    if m:
        amount = int(m.group(1))
        unit = m.group(2)
        days = amount * {"day": 1, "days": 1, "week": 7, "weeks": 7, "month": 30, "months": 30}[unit]
        start = now - timedelta(days=days)
        return {"start": iso(start), "end": iso(now), "label": f"last {amount} {unit}"}

    if "last week" in low or "past week" in low:
        return {"start": iso(now - timedelta(days=7)), "end": iso(now), "label": "last 7 days"}
    if "last month" in low or "past month" in low:
        return {"start": iso(now - timedelta(days=30)), "end": iso(now), "label": "last 30 days"}
    if "yesterday" in low:
        return {"start": iso(now - timedelta(days=1)), "end": iso(now), "label": "last 24 hours"}
    if "today" in low:
        return {"start": iso(now - timedelta(days=1)), "end": iso(now), "label": "today"}
    if "this year" in low:
        start = datetime(now.year, 1, 1, tzinfo=timezone.utc)
        return {"start": iso(start), "end": iso(now), "label": str(now.year)}

    m = re.search(r"\bsince\s+(\d{4}-\d{2}-\d{2})\b", low)
    if m:
        start = parse_dt(m.group(1))
        return {"start": iso(start), "end": iso(now), "label": f"since {m.group(1)}"}

    return {"start": None, "end": None, "label": ""}


def within(value: Optional[datetime], start: Optional[datetime], end: Optional[datetime]) -> bool:
    if value is None:
        return False
    if start and value < start:
        return False
    if end and value > end:
        return False
    return True


def humanise(value: Optional[datetime]) -> str:
    if value is None:
        return "unknown date"
    return value.astimezone(timezone.utc).strftime("%Y-%m-%d")
