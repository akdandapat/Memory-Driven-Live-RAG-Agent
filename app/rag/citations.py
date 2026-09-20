"""Citation construction and validation.

Rules enforced here:
  * every citation marker in the answer must resolve to a real evidence record;
  * markers that do not resolve are removed (a model cannot invent [E9]);
  * the citation payload carries source id, URL, timestamp and the tool that produced it,
    so the UI can link straight back to the Jira issue/comment/history entry.
"""
from __future__ import annotations

import re
from typing import Iterable, Sequence

from app.models.evidence import Citation, Evidence

MARKER_RE = re.compile(r"\[(E\d+(?:\s*,\s*E\d+)*)\]")


def build_citations(evidence: Sequence[Evidence], markers: Iterable[str]) -> list[Citation]:
    by_id = {e.evidence_id: e for e in evidence}
    citations: list[Citation] = []
    seen: set[str] = set()
    for marker in markers:
        item = by_id.get(marker)
        if item is None or marker in seen:
            continue
        seen.add(marker)
        citations.append(Citation(
            marker=marker,
            source=item.source,
            source_type=item.source_type,
            source_id=item.source_id,
            title=item.title,
            url=item.url,
            timestamp=item.timestamp,
            tool_used=item.tool_used,
            snippet=item.content[:280],
        ))
    return citations


def extract_markers(answer: str) -> list[str]:
    found: list[str] = []
    for group in MARKER_RE.findall(answer or ""):
        for marker in group.split(","):
            marker = marker.strip()
            if marker and marker not in found:
                found.append(marker)
    return found


def validate_and_clean(answer: str, evidence: Sequence[Evidence]) -> tuple[str, list[Citation], dict]:
    """Strip unresolvable markers, then build the citation list for what remains."""
    valid_ids = {e.evidence_id for e in evidence}
    invented: list[str] = []

    def _replace(match: re.Match) -> str:
        markers = [m.strip() for m in match.group(1).split(",")]
        keep = [m for m in markers if m in valid_ids]
        invented.extend(m for m in markers if m not in valid_ids)
        return f"[{', '.join(keep)}]" if keep else ""

    cleaned = MARKER_RE.sub(_replace, answer or "")
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r" +([.,;:])", r"\1", cleaned)

    markers = extract_markers(cleaned)
    citations = build_citations(evidence, markers)
    stats = {
        "citation_count": len(citations),
        "invented_markers_removed": len(invented),
        "invented_markers": sorted(set(invented)),
        "evidence_available": len(valid_ids),
        "evidence_cited": len(citations),
        "citation_coverage": round(len(citations) / len(valid_ids), 3) if valid_ids else 0.0,
    }
    return cleaned.strip(), citations, stats


def uncited_claim_sentences(answer: str) -> list[str]:
    """Lines that assert something factual but carry no marker (surfaced in the trace).

    Line-based rather than sentence-based, because the answers are markdown: a heading or a
    bullet is the unit a reader checks a citation against.
    """
    flagged: list[str] = []
    for line in (answer or "").splitlines():
        stripped = line.strip()
        if len(stripped) < 40 or "[E" in stripped:
            continue
        if stripped.startswith(("#", "|", ">")):
            continue
        if stripped.startswith("**") and stripped.endswith("**"):
            continue
        if stripped.startswith("_") and stripped.endswith("_"):
            continue
        if re.search(r"\b(insufficient|no evidence|not enough evidence|could not find|unable to|"
                     r"no claim is made|not going to guess|does not state)\b", stripped, re.I):
            continue
        # a bullet whose own line lacks a marker but whose parent block has one is still
        # reported: that is exactly the ambiguity a reviewer should see.
        flagged.append(stripped[:200])
    return flagged
