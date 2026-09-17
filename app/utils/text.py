"""Text normalisation helpers shared by the connector, RAG and heuristic layers."""
from __future__ import annotations

import re
from typing import Any, Iterable

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "than", "so", "of", "in", "on", "at",
    "to", "for", "with", "by", "from", "as", "is", "are", "was", "were", "be", "been", "being",
    "it", "its", "this", "that", "these", "those", "i", "me", "my", "we", "our", "you", "your",
    "he", "she", "they", "them", "his", "her", "their", "what", "which", "who", "whom", "how",
    "when", "where", "why", "do", "does", "did", "can", "could", "should", "would", "will",
    "shall", "may", "might", "must", "have", "has", "had", "not", "no", "yes", "please", "tell",
    "about", "into", "over", "under", "any", "all", "some", "most", "more", "also",
}

TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_\-\.]*")

# Words that carry no retrievable meaning in this domain. Their absence from retrieved text
# proves nothing, so both the sufficiency check and the unsupported-term check ignore them.
GENERIC_QUESTION_WORDS = {
    "project", "projects", "quarter", "quarterly", "risk", "risks", "risky", "major", "minor",
    "summarize", "summarise", "summary", "identify", "identifying", "changed", "change", "changes",
    "tell", "told", "personally", "personal", "focus", "focusing", "should", "during", "issue",
    "issues", "ticket", "tickets", "team", "status", "update", "updates", "give", "list", "show",
    "week", "weeks", "month", "months", "year", "years", "day", "days", "still", "right", "now",
    "currently", "current", "actually", "given", "need", "needs", "needed", "something",
    "anything", "everything", "thing", "things", "much", "many", "well", "really", "quite",
    "maybe", "probably", "please", "help", "want", "wants", "like", "make", "makes", "take",
    "takes", "come", "know", "think", "thinks", "ask", "asked", "look", "looking", "see", "seen",
    "use", "used", "using", "work", "working", "works", "worked", "going", "get", "gets", "also",
    "state", "states", "over", "under", "between", "across", "into", "about", "around", "there",
    "here", "than", "then", "when", "while", "with", "without", "good", "bad", "best", "worst",
    "big", "small", "high", "low", "next", "last", "first", "recent", "recently", "happen",
    "happened", "happening", "regarding", "kindly", "long", "short",
    # sub-question scaffolding produced by the planner itself
    "people", "reported", "report", "problems", "problem", "relate", "related", "relevant",
    "information", "goals", "goal", "scope", "lead", "attention", "surfaced", "above", "which",
    "what", "whose", "expect", "expected", "detail", "details", "activity",
    "dates", "date", "statuses", "moved", "move", "surfaced", "involved", "current",
}


def tokenize(text: str, drop_stopwords: bool = True) -> list[str]:
    tokens = TOKEN_RE.findall((text or "").lower())
    if drop_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    return tokens


def keyword_overlap(a: str, b: str) -> float:
    """Jaccard-style lexical overlap in [0, 1]. Cheap relevance signal, no network."""
    ta, tb = set(tokenize(a)), set(tokenize(b))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def collapse_whitespace(text: str) -> str:
    return re.sub(r"[ \t]+", " ", re.sub(r"\n{3,}", "\n\n", (text or "").strip()))


def truncate(text: str, limit: int, suffix: str = " …[truncated]") -> str:
    text = text or ""
    if len(text) <= limit:
        return text
    return text[: max(0, limit - len(suffix))] + suffix


def adf_to_text(node: Any) -> str:
    """Flatten Atlassian Document Format (Jira Cloud v3 rich text) into plain text.

    Jira returns descriptions/comments as ADF JSON trees; the RAG layer needs plain text.
    """
    if node is None:
        return ""
    if isinstance(node, str):
        return node
    if isinstance(node, list):
        return "".join(adf_to_text(n) for n in node)
    if not isinstance(node, dict):
        return str(node)

    node_type = node.get("type", "")
    if node_type == "text":
        return node.get("text", "")
    if node_type == "hardBreak":
        return "\n"
    if node_type == "mention":
        return "@" + str(node.get("attrs", {}).get("text", "")).lstrip("@")
    if node_type == "emoji":
        return str(node.get("attrs", {}).get("shortName", ""))
    if node_type in {"inlineCard", "blockCard"}:
        return str(node.get("attrs", {}).get("url", ""))

    inner = adf_to_text(node.get("content", []))
    if node_type in {"paragraph", "heading", "blockquote", "codeBlock", "listItem", "rule",
                     "panel", "tableRow"}:
        return inner + "\n"
    return inner


def chunk_text(text: str, chunk_chars: int = 900, overlap: int = 150) -> list[str]:
    """Paragraph-aware chunking with a character budget and small overlap."""
    text = collapse_whitespace(text)
    if not text:
        return []
    if len(text) <= chunk_chars:
        return [text]

    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks: list[str] = []
    buffer = ""
    for para in paragraphs:
        if len(buffer) + len(para) + 1 <= chunk_chars:
            buffer = f"{buffer}\n{para}".strip()
            continue
        if buffer:
            chunks.append(buffer)
        if len(para) <= chunk_chars:
            buffer = para
        else:
            start = 0
            while start < len(para):
                chunks.append(para[start: start + chunk_chars])
                start += max(1, chunk_chars - overlap)
            buffer = ""
    if buffer:
        chunks.append(buffer)

    # add overlap between neighbouring chunks so sentences are not cut without context
    if overlap > 0 and len(chunks) > 1:
        overlapped = [chunks[0]]
        for prev, cur in zip(chunks, chunks[1:]):
            overlapped.append((prev[-overlap:] + " " + cur).strip())
        return overlapped
    return chunks


def join_nonempty(parts: Iterable[str], sep: str = "\n") -> str:
    return sep.join(p for p in parts if p)
