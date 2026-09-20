"""Unit tests for temporal resolution, ADF flattening and chunking."""
from __future__ import annotations

from datetime import datetime, timezone

from app.utils.dates import parse_dt, quarter_bounds, resolve_time_window, within
from app.utils.text import adf_to_text, chunk_text, keyword_overlap, tokenize


def test_quarter_bounds_are_inclusive():
    start, end = quarter_bounds(2026, 2)
    assert start == datetime(2026, 4, 1, tzinfo=timezone.utc)
    assert end.month == 6 and end.day == 30


def test_resolve_named_quarter():
    window = resolve_time_window("what changed in Q2 2026", now=datetime(2026, 9, 10, tzinfo=timezone.utc))
    assert window["start"].startswith("2026-04-01")
    assert window["end"].startswith("2026-06-30")
    assert window["label"] == "Q2 2026"


def test_bare_quarter_in_the_future_resolves_to_last_year():
    now = datetime(2026, 2, 1, tzinfo=timezone.utc)      # currently Q1
    window = resolve_time_window("summarise Q4", now=now)
    assert window["label"] == "Q4 2025"


def test_relative_windows():
    now = datetime(2026, 9, 10, tzinfo=timezone.utc)
    window = resolve_time_window("what happened in the last 14 days", now=now)
    assert window["start"].startswith("2026-08-27")
    assert resolve_time_window("who owns this?", now=now)["start"] is None


def test_parse_dt_handles_jira_offsets_and_z():
    assert parse_dt("2026-06-11T18:20:00.000+0530").tzinfo is not None
    assert parse_dt("2026-06-11T18:20:00Z").hour == 18
    assert parse_dt("") is None and parse_dt(None) is None


def test_within_bounds():
    start, end = quarter_bounds(2026, 2)
    assert within(parse_dt("2026-05-04T00:00:00Z"), start, end)
    assert not within(parse_dt("2026-07-04T00:00:00Z"), start, end)
    assert not within(None, start, end)


def test_adf_to_text_flattens_nested_nodes():
    adf = {
        "type": "doc",
        "content": [
            {"type": "paragraph", "content": [
                {"type": "text", "text": "Blocked by "},
                {"type": "mention", "attrs": {"text": "@vendor"}},
            ]},
            {"type": "paragraph", "content": [{"type": "text", "text": "No fix date."}]},
        ],
    }
    text = adf_to_text(adf)
    assert "Blocked by @vendor" in text
    assert "No fix date." in text


def test_chunking_respects_budget_and_overlaps():
    text = "\n".join(f"Paragraph {i} with a reasonable amount of prose in it." for i in range(40))
    chunks = chunk_text(text, chunk_chars=200, overlap=40)
    assert len(chunks) > 1
    assert all(len(c) <= 200 + 40 + 1 for c in chunks)
    assert chunk_text("", 200) == []


def test_keyword_overlap_and_tokenize():
    assert keyword_overlap("blocked by vendor", "the vendor blocked us") > 0.3
    assert keyword_overlap("apples", "oranges") == 0.0
    assert "the" not in tokenize("the vendor")
