# Agentic RAG over Live Data and Memory

An agent that answers questions about a **live issue tracker** by planning its own retrieval,
calling **MCP tools**, collecting **evidence**, reasoning across **multiple hops**, remembering
what matters **across sessions**, **citing** every claim, and exposing the whole trajectory as an
**operational trace**.

The reference question it is built around:

> *"Summarize what changed in Project Atlas during Q2 2026 and identify the major risks. Also tell
> me what I should personally focus on."*

A vector-store chatbot cannot answer that. It needs decomposition, structured retrieval over
changing data, temporal reasoning, causal linking, remembered user context, and citations.

---

## Table of contents

- [What this demonstrates](#what-this-demonstrates)
- [Why basic RAG is insufficient](#why-basic-rag-is-insufficient)
- [Architecture](#architecture)
- [Agent workflow](#agent-workflow)
- [MCP architecture](#mcp-architecture)
- [Live data architecture](#live-data-architecture)
- [RAG architecture](#rag-architecture)
- [Memory architecture](#memory-architecture)
- [Multi-hop reasoning](#multi-hop-reasoning)
- [Citations](#citations)
- [Observability](#observability)
- [Tech stack](#tech-stack)
- [Project structure](#project-structure)
- [Installation](#installation)
- [Environment variables](#environment-variables)
- [Demo mode](#demo-mode)
- [Live Jira mode](#live-jira-mode)
- [Running the MCP server standalone](#running-the-mcp-server-standalone)
- [API reference](#api-reference)
- [Database design](#database-design)
- [Testing](#testing)
- [Evaluation](#evaluation)
- [Security](#security)
- [End-to-end demo](#end-to-end-demo)
- [Video concept → current implementation](#video-concept--current-implementation)
- [Limitations](#limitations)
- [Future improvements](#future-improvements)
- [Troubleshooting](#troubleshooting)
- [Resume bullets](#resume-bullets)
- [Interview questions and answers](#interview-questions-and-answers)

---

## What this demonstrates

| Capability | Where it lives | How you can see it |
|---|---|---|
| Live data source | `app/connectors/` | `scripts/simulate_change.py` mutates the source; the answer changes after `/api/sync` |
| MCP tool layer | `app/mcp_server/`, `app/mcp_client/` | `GET /api/tools` lists what was discovered over `tools/list`; three transports |
| Query rewriting | `app/agents/rewriter.py` | Trace panel section 2; ask a follow-up like *"and what is overdue there?"* |
| Evidence sufficiency check | `app/agents/evaluator.py` | Trace panel section 5 shows the verdict, relevance and term coverage behind every sub-question status |
| Dynamic planning | `app/agents/planner.py` | Trace panel section 3; `tests/test_planner.py` asserts different questions produce different plans |
| Multiple tool calls | `app/agents/executor.py` | Trace panel section 5; the compound question issues 8–10 calls |
| Multi-hop reasoning | `planner.py` + `executor.py` | Sub-questions with `depends_on`; issue keys discovered in hop 1 feed hop 2 |
| Structured + semantic retrieval | `mcp_server/tools.py`, `app/rag/` | Filters and date ranges go to the source; open-ended language goes to the vector index |
| Persistent memory | `app/memory/` | Memory drawer in the UI; write in one session, recall in another |
| Memory conflict handling | `app/memory/manager.py` | State a new priority; the old record is marked `superseded`, not appended |
| Citations | `app/rag/citations.py` | Every `[E3]` links to the issue/comment/history entry it came from |
| Observability | `app/observability/` | `GET /api/trace/{run_id}`, the UI trace panel, optional LangSmith export |
| Honest failure | throughout | Source down → degraded notice; unsupported question → refusal, not invention |

---

## Why basic RAG is insufficient

```
BASIC RAG                             THIS SYSTEM
question                              question
  ↓                                     ↓ short-term context + long-term memory recall
embed                                   ↓ plan: decompose into sub-questions
  ↓                                     ↓ select tools per sub-question (MCP tools/list)
top-k chunks                            ↓ execute multiple calls, dedupe, budget
  ↓                                     ↓ evidence collection + scoring + injection screening
generate                                ↓ gap detection → second retrieval round
  ↓                                     ↓ multi-hop: hop-1 results parameterise hop-2 calls
answer                                  ↓ synthesis constrained to evidence
                                        ↓ citation validation (invented markers stripped)
                                        ↓ memory write-back (create/update/supersede/reject)
                                      answer + citations + trace
```

Concretely, for *"what changed in Q2 and what are the risks"*:

- **Chunks do not know about time.** "Changed in Q2" is a filter over `created`, `resolutiondate`
  and changelog timestamps. `search_issues(resolved_after=…, resolved_before=…)` answers it exactly;
  cosine similarity answers it approximately at best.
- **Current fields do not show their own history.** The text of ATLAS-1 says the due date is
  2026-07-24. Only `get_issue_changelog` shows it was 2026-06-12, then 2026-06-30, then 2026-07-24 —
  which is what turns "a deadline" into "repeated slippage".
- **Risk is a join, not a lookup.** Overdue ∧ unresolved ∧ blocked-by-an-unresolved-link ∧ what the
  comments say. No single chunk contains it.
- **The user's own context is not in the tracker.** "What should *I* focus on" needs memory.

The system ships the baseline so the difference is demonstrable, not asserted: toggle
**compare with basic RAG** in the UI, or `POST /api/chat {"compare_baseline": true}`. The baseline
does one embedding, one top-k lookup, one generation — 0 tool calls, 0 citations, no memory.

---

## Architecture

```
                     ┌──────────────────────────────────────────────┐
  Browser ──────────▶│ FastAPI  (app/main.py, app/api/routes.py)     │
  chat + trace panel │  /api/chat  /api/sync  /api/trace  /api/memory│
                     └───────────────┬──────────────────────────────┘
                                     ▼
                     ┌──────────────────────────────────────────────┐
                     │ Orchestrator (app/agents/orchestrator.py)    │
                     │  memory recall → rewrite → plan → execute →  │
                     │  assess → synthesize → cite → remember       │
                     └───┬───────────┬───────────┬──────────────┬───┘
                         │           │           │              │
              ┌──────────▼──┐ ┌──────▼──────┐ ┌──▼───────────┐ ┌▼──────────────┐
              │ Rewriter +  │ │ Executor +  │ │ Synthesizer  │ │ MemoryManager │
              │ Planner     │ │ Evaluator   │ │ grounded +   │ │ extract /     │
              │ resolve then│ │ tool select │ │ cited        │ │ conflict /    │
              │ decompose   │ │ budget/dedup│ │              │ │ supersede     │
              └─────────────┘ └──────┬──────┘ └──────────────┘ └───────┬───────┘
                                     ▼ MCP (tools/list, tools/call)     │
                     ┌──────────────────────────────────────────────┐   │
                     │ MCP server (app/mcp_server/server.py)        │   │
                     │  11 tools · inproc | stdio | streamable-http │   │
                     └───────────────┬──────────────────────────────┘   │
                                     ▼                                   │
                     ┌───────────────────────────┐  ┌───────────────────▼─────────┐
                     │ SourceConnector           │  │ SQLite                      │
                     │  JiraConnector (live)     │  │  sessions/messages          │
                     │  MockJiraConnector (demo) │  │  memories (+embeddings)     │
                     └───────────┬───────────────┘  │  documents/chunks/vectors   │
                                 ▼                   │  runs/steps/tool_calls/    │
                        Jira Cloud REST v3           │  evidence/memory_events    │
                        (source of truth)            └─────────────────────────────┘
```

**Component responsibilities**

| Component | Responsibility | Explicitly not responsible for |
|---|---|---|
| `connectors/` | Authenticate, paginate, retry, normalise into domain models | Deciding what to fetch |
| `mcp_server/tools.py` | Expose capabilities with schemas, validate arguments, enforce the allow-list | Planning or ranking |
| `mcp_client/` | Discover tools over the protocol, call them with timeouts, normalise results | Knowing which tools exist at import time |
| `agents/rewriter.py` | Resolve pronouns and ellipsis into a standalone question | Adding constraints the user did not express |
| `agents/planner.py` | Turn a question into sub-questions, entities and a time window | Executing anything |
| `agents/evaluator.py` | Judge whether a sub-question's evidence answers it, and name the gap | Fetching anything |
| `agents/executor.py` | Choose arguments, enforce budget, collect and score evidence | Writing prose |
| `agents/synthesizer.py` | Write the answer strictly from evidence | Fetching more data |
| `rag/` | Ingest, chunk, embed, index, retrieve, build and validate citations | Structured filtering |
| `memory/` | Decide what is worth keeping, retrieve selectively, resolve conflicts | Storing transcripts |
| `observability/` | Record the trajectory, redact secrets, export | Changing behaviour |

---

## Agent workflow

```
USER QUESTION
   ↓
MEMORY RETRIEVAL           short-term session context + top-k long-term records (scored, thresholded)
   ↓
QUERY REWRITE              resolve pronouns and ellipsis against the session, or leave the turn alone
   ↓
PLAN                       goal · strategy · entities · time window · sub-questions · tool hints
   ├── sq1  Which issues in ATLAS relate to the EU cutover date?      → semantic_search, search_issues
   ├── sq2  What are the project's stated goals?                      → get_project
   ├── sq3  What activity happened during Q2 2026?                    → get_project_activity
   ├── sq4  What is overdue or had its due date moved?                → get_overdue_issues
   ├── sq5  What is blocked, and by what?                             → get_blocked_issues
   └── sq6  For the issues found above, what changed and when?        → get_issue_changelog  (depends on sq1,sq3)
   ↓
TOOL CALLS                 executed over MCP · deduped · budget-capped · timed · recorded
   ↓
EVIDENCE                   normalised, scored, injection-screened, capped, ids E1…En
   ↓
SUFFICIENCY CHECK          per sub-question: is this evidence actually an answer? verdict + named gap
   ↓
GAP RETRY                  unresolved sub-questions get one relaxed round, steered by the named gap
   ↓
SYNTHESIS                  answer written only from evidence
   ↓
CITATION VALIDATION        markers resolved against real evidence; invented ones stripped
   ↓
MEMORY UPDATE              create · update · supersede · reject (each logged with a reason)
   ↓
FINAL ANSWER + TRACE
```

---

## MCP architecture

The agent never imports the tool functions. It calls `tools/list` and `tools/call` over the
Model Context Protocol, using the official Python SDK (`mcp>=2.0`). Three transports, one server
object:

| `MCP_TRANSPORT` | What happens | Use it for |
|---|---|---|
| `inproc` | The `MCPServer` object is hosted in the API process; `Client(server)` speaks the protocol in-memory | Default. No subprocess, no port, same protocol semantics |
| `stdio` | `python -m app.mcp_server.server` is launched as a subprocess; JSON-RPC over stdin/stdout | Proving process isolation; connecting Claude Desktop or an IDE |
| `http` | The server runs standalone over Streamable HTTP | Deploying the tool layer as its own service |

All three are exercised in this repo; `stdio` is verified end to end (subprocess launch → tool call
→ cited answer).

### The tools

| Tool | Purpose | When the agent should call it |
|---|---|---|
| `search_projects(query?, limit?)` | List/search projects | The question names a project in words and you need its key |
| `get_project(project_key)` | Description, lead, stated goals | Judging change or risk against objectives; ownership questions |
| `search_issues(project_key?, text?, statuses?, status_category?, priorities?, issue_types?, labels?, assignee?, created_*, updated_*, resolved_*, due_*, unresolved_only?, order_by?, limit?)` | Structured issue search | The primary tool: anything with a filter, date range, status or owner |
| `get_issue(issue_key)` | One issue in full, with links | A prior result points at a specific issue |
| `get_issue_comments(issue_key, limit?)` | Discussion thread | Causal evidence — *why* something is blocked |
| `get_issue_changelog(issue_key, fields?, limit?)` | Field-change history | "What changed and when"; `fields=['duedate']` for deadline moves |
| `get_project_activity(project_key, start?, end?, limit?)` | Chronological feed + per-kind counters | "What changed during \<period\>" in one call |
| `get_blocked_issues(project_key?, limit?)` | Blocked by status, label or unresolved link, with reasons | Risk and impediment questions |
| `get_overdue_issues(project_key?, as_of?, limit?)` | Unresolved past due, sorted by lateness | Deadline risk; set `as_of` to the end of the reported quarter |
| `search_comments(project_key, keywords?, start?, end?, limit?)` | Risk language across a project's comments | Causes and concerns not encoded in fields |
| `semantic_search(query, project_key?, source_types?, top_k?)` | Vector search over indexed prose | Open-ended language questions only |

Every tool returns the same envelope, so the executor converts any result into evidence without
knowing the tool:

```json
{"ok": true, "tool": "get_blocked_issues", "source": "jira", "mode": "demo", "count": 2,
 "items": [{"source_type": "issue", "source_id": "ATLAS-9", "title": "…", "url": "https://…",
            "timestamp": "2026-06-27T08:15:00Z", "content": "BLOCKED: …", "data": {…}}],
 "truncated": false, "fetched_at": "2026-09-10T…"}
```

Failures are **data, not exceptions**, so the agent can reason about them and the trace can show
them:

```json
{"ok": false, "tool": "get_project", "error": {"kind": "not_found", "message": "Project ZZZ …"},
 "items": [], "count": 0}
```

`kind` is one of `invalid_argument`, `access_denied`, `not_found`, `auth`, `rate_limit`,
`unavailable`, `timeout`, `unknown_tool`, `transport_error`, `internal_error`.

**Authentication.** Jira credentials live only in the connector process, read from the environment
as HTTP Basic `email:api_token` over HTTPS. They are never placed in a prompt, a trace or a tool
argument. Project access is constrained by `JIRA_ALLOWED_PROJECTS`, enforced inside every tool.

---

## Live data architecture

Three distinct stores, never confused:

| Store | What it is | Authority | Rebuildable |
|---|---|---|---|
| **Jira Cloud** (or the demo dataset) | Source of truth | Absolute | — |
| **`documents` / `chunks` / `chunk_embeddings`** | Local cache + vector index | Derived, may lag by one sync | Yes, from the source |
| **`memories`** | What the *user* told us | Authoritative about the user, says nothing about the tracker | No — delete is deliberate |

Structured tool calls (`search_issues`, `get_blocked_issues`, …) go **straight to the source** on
every call, so they are never stale. Only `semantic_search` reads the local index. That is the
design line: *facts and filters live, prose from the index.*

### Sync

```
POST /api/sync            → incremental (per-project watermark = max(source_updated) seen)
POST /api/sync {"full":true} → ignore watermarks and re-read everything
```

```
fetch (updated_after = watermark)
  → normalise into domain models
  → build document text
  → content hash: unchanged? stop here, nothing re-embedded
  → chunk (paragraph-aware, 900 chars, 150 overlap)
  → embed
  → upsert vectors, advance the watermark
```

A second sync immediately after a first reports `documents_upserted: 0, chunks_indexed: 0` — this is
asserted in `tests/test_rag_and_citations.py`.

### Watching data change

```bash
python scripts/simulate_change.py --scenario unblock          # vendor ships, ATLAS-11 → Done
python scripts/simulate_change.py --comment ATLAS-9 "New ETA is 2026-08-14."
python scripts/simulate_change.py --duedate ATLAS-1 2026-08-21
```

The demo connector re-reads the JSON files on **every call**, so these edits are live source
changes, not fixtures. Ask the question, mutate, sync, ask again.

### Jira API notes (verified, 2026)

- `GET/POST /rest/api/3/search` was **removed** from Jira Cloud. Issue search uses
  `POST /rest/api/3/search/jql`, which paginates with `nextPageToken` (not `startAt`) and returns no
  `total`. Implemented in `JiraConnector.search_issues`.
- v3 returns descriptions and comment bodies as **ADF** (Atlassian Document Format) JSON;
  `utils/text.adf_to_text` flattens the tree before indexing.
- Comments: `GET /rest/api/3/issue/{key}/comment`; history: `GET /rest/api/3/issue/{key}/changelog`
  (both still `startAt`/`maxResults`).
- Retries: exponential backoff on 5xx, `Retry-After` honoured on 429, 401/403 fail fast as
  `AuthError`.

---

## RAG architecture

RAG is a *component*, not the architecture.

```
acquisition (connector) → normalisation (domain models) → document representation
  → content hash → chunking → embedding → vector index → filtered retrieval
  → evidence construction → citation
```

**Document representation.** One document per issue (header + facts line + description + links), one
per comment, one per project (description + stated goals). The facts line matters: it puts status,
priority, assignee, due date and labels into the embedded text so lexical/semantic search can see
them at all.

**Retrieval scoring.** `0.75 × cosine + 0.25 × lexical overlap`. The lexical term stabilises ranking
under the offline hashing embedder and acts as a light keyword prior under a real encoder.

**Filtering.** Project, source type, issue key and `source_updated` range are applied *before*
scoring, so a project-scoped question never sees another project's chunks.

**Embeddings.** `EMBEDDING_PROVIDER=hashing` (default) is a deterministic hashed unigram+bigram
embedding: no API key, fully offline, but **lexical, not semantic** — it will not match paraphrases
the way a trained encoder does. `EMBEDDING_PROVIDER=openai` gives real semantic quality. This
trade-off is stated rather than hidden.

**Vector index.** Exact cosine over a numpy matrix backed by SQLite. Honest about scale: fine for
thousands of chunks, not a distributed ANN engine. The interface (`upsert` / `search` /
`invalidate`) is deliberately pgvector-shaped, so the swap is one class.

---

## Memory architecture

### Short-term (`app/memory/short_term.py`)
The current session: the last 8 turns verbatim plus a rolling summary of everything older. Scoped to
one session and one user; a session cannot be reused by a different user (`PermissionError`).

### Long-term (`app/memory/long_term.py`)
Structured records, not messages:

```json
{"memory_id": "mem_…", "user_id": "…", "type": "preference|context|fact|goal",
 "content": "User's stated top priority is Project Atlas.", "subject": "priority",
 "importance": 0.85, "confidence": 0.7, "status": "active|superseded|expired",
 "superseded_by": null, "source_session": "…", "source_run": "…",
 "created_at": "…", "updated_at": "…", "expires_at": null, "use_count": 3}
```

### What gets stored — and what does not

**Stored:** stated priorities, role and ownership, standing preferences about how answers should be
produced, long-lived goals, durable facts about the user's world.

**Rejected, with a logged reason:** questions ("what changed in Atlas?"), one-off instructions,
transient state, anything the *agent* inferred or retrieved, and anything resembling reasoning. The
agent's chain-of-thought is never stored — the extractor only sees the user's turn plus a short
answer summary.

Two extractors behind one interface: an LLM extractor with a strict JSON schema, and a pattern-based
extractor used when `LLM_PROVIDER=none`. Both outputs pass the same filters (minimum importance,
length, "is this reasoning?" check).

### Retrieval
`0.60 × semantic + 0.25 × importance + 0.15 × recency` (45-day half-life), thresholded at
`MEMORY_MIN_RELEVANCE`, capped at `MEMORY_TOP_K = 5`. Irrelevant memory is **not** injected — asking
about lattice QCD retrieves nothing even when the store is full.

### Conflict handling
For each candidate, the most similar active record of the same type is found. Subjects like
`priority`, `role` and `goal` are **singletons** — one active record each.

| Similarity | Action |
|---|---|
| ≥ 0.94 | `skipped_duplicate` — refresh timestamp/confidence, write nothing new |
| ≥ 0.80, same claim | `updated` |
| ≥ 0.80, different claim | `superseded` — new record written, old one marked `superseded_by` |
| below | `created` |

> "Project Atlas is my highest priority" → later → "Actually, Project Apollo has become my highest
> priority now" ⇒ one active record (Apollo), one superseded record (Atlas) retained for audit. The
> store never holds two contradictory priorities.

The store is bounded (`MEMORY_MAX_PER_USER`); pruning drops the least important, least used, oldest
records first.

---

## Query rewriting

The transcript lists *"the query rewrite"* as a stage in its own right, and follow-up turns prove
why. `And what about Apollo?` and `why did it slip?` carry no project, no period and no subject. A
planner that sees the raw turn plans the wrong retrieval, and everything downstream inherits the
mistake.

```
session:  "Summarize what changed in Project Atlas during Q2 2026 and identify the major risks."
turn:     "why did it slip?"
rewrite:  "why did Project Atlas slip in Q2 2026?"
resolved: ["it -> Project Atlas", "implicit period -> Q2 2026"]
```

The one hard rule: **resolve references, never add constraints.** A rewrite that invents a filter is
worse than no rewrite. That rule is enforced in four places, each with a test:

| Guard | Behaviour |
|---|---|
| Meta turns | `"What can you do?"`, greetings and capability questions are never rewritten — there is no reference to resolve, so any "resolution" is an invention |
| User statements | `"Project Apollo is my highest priority now."` is routed to memory, not resolved |
| Self-contained turns | A turn naming a project or issue key is left alone. Follow-up markers only count **at the start** — `"…in Project Atlas **and what** is it now?"` is one question, not a follow-up |
| Period carrying | The time window is only carried onto a genuinely elliptical turn, never bolted onto a question that states its own scope |
| No context | With nothing earlier in the session, the turn is returned unchanged |

Both paths return the same `RewriteResult` (original, rewritten, changed, resolved references,
reason, method), which is written to the trace as its own step. With no LLM configured the
deterministic path substitutes the most recent project or issue key from the session, handling
locative pronouns correctly (`"blocked there"` → `"blocked **in** Project Apollo"`).

## Evidence sufficiency

Something has to decide whether what came back actually answers the sub-question. Without it the
agent proceeds on thin evidence and the synthesizer writes something confident over nothing.

After each sub-question's tools run, the evaluator returns a verdict:

```json
{"sufficient": false, "missing": "weak match (top relevance 0.19, term coverage 25%); nothing found
 about: verityx, callbacks", "suggested_tool": "search_comments",
 "suggested_arguments": {"project_key": "ATLAS", "keywords": ["verityx"]},
 "method": "heuristic", "top_relevance": 0.19, "term_coverage": 0.25, "evidence_count": 3}
```

The deterministic check requires both a top relevance above threshold **and** coverage of the
sub-question's content-bearing terms, and treats evidence that only survived because an injection
attempt was downranked as no evidence at all. With an LLM configured, the model judges the same
evidence and may name a specific tool and arguments — and that suggestion becomes the **first call
of the retry round**, which is what makes the second round different from a repeat of the first.

Verdicts appear per sub-question in the UI trace and in the `sufficiency` field of the chat
response, so a reviewer can see exactly why a sub-question was marked answered or insufficient.

## Multi-hop reasoning

Hops are explicit in the plan (`depends_on`) and in the executor (`_discovered_issue_keys`), which
feeds the issue keys found in hop *n* into the arguments of hop *n+1*.

*"Why did the EU cutover date move in Project Atlas and what is it now?"*

```
hop 1  semantic_search("…EU cutover date…") + search_issues     → surfaces ATLAS-1, ATLAS-9, ATLAS-11
hop 2  get_issue_changelog(ATLAS-1, fields=[duedate,status])    → 2026-06-12 → 2026-06-30 → 2026-07-24
       get_issue_changelog(ATLAS-9)                             → In Progress → Blocked (2026-05-06)
hop 3  get_blocked_issues(ATLAS)                                → ATLAS-9 blocked by ATLAS-11
       search_comments(ATLAS)                                   → "Verityx have not shipped the 4.2 hotfix"
synthesis  deadline moves ∧ blocked dependency ∧ stated reason, each cited
```

**The agent does not assert a cause it cannot evidence.** The deterministic writer emits
*"The retrieved evidence shows what changed but does not state a cause, so no cause is asserted
here"* when the causal link is missing, and the LLM prompt forbids unevidenced causal claims.

---

## Citations

Every factual claim carries a marker resolving to a real evidence record:

```
ATLAS-11 is blocked (status is 'Blocked'; carries a blocked label) and overdue by 32 days,
priority Highest: Verityx SDK 4.2 hotfix delivery [E15, E24]
```

The evidence object behind `E15`:

```json
{"evidence_id": "E15", "source": "jira", "source_type": "issue", "source_id": "ATLAS-11",
 "url": "https://…/browse/ATLAS-11", "timestamp": "2026-06-28T12:00:00Z",
 "content": "OVERDUE by 32 day(s) as of …", "relevance": 0.71,
 "tool_used": "get_overdue_issues", "sub_question_id": "sq4", "injection_suspected": false}
```

Validation, in `app/rag/citations.py`:

1. Markers are extracted from the answer.
2. Any marker not matching a collected evidence id is **removed** and counted as
   `invented_markers_removed` — a model cannot smuggle in `[E9]`.
3. Surviving markers become citations carrying source id, URL, timestamp and the tool that produced
   them.
4. Lines that assert something with no marker are counted as `uncited_claims` and shown in the trace,
   so a reviewer can see exactly what is unbacked.

Evaluation measures `citation_validity` and `invented_citations_total` on every run.

---

## Observability

**Local tracing is always on** and powers the UI. Every run writes:

`agent_runs` (question, mode, status, latency, counts) · `run_steps` (memory_retrieval →
query_rewrite → planning → execution → sub_question:* → evidence_gap_retry → synthesis →
memory_update, each with inputs, outputs and duration) · `tool_calls` (name, arguments, status, result count, duration, error) ·
`evidence` (full records, `cited` flag) · `memory_events` (action + reason).

```bash
curl localhost:8000/api/trace/run_ab12… | jq '.steps[].name'
# "memory_retrieval" "query_rewrite" "planning" "execution" "sub_question:sq1" …
# "evidence_gap_retry" "synthesis" "memory_update"
```

**LangSmith** is an optional additional sink (`LANGSMITH_TRACING=true` + API key). The export
mirrors the real shape of a run rather than flattening it: **one root chain, one child span per
stage, and one child span per tool call**, so the whole path is inspectable in LangSmith exactly as
it is locally. Internal ids are mapped to UUIDs inside the exporter, so LangSmith's identifier
requirements never leak into the rest of the system. The SDK is imported lazily, so it is not a hard
dependency, and an export failure never breaks a request.

**Secrets never enter a trace.** Everything is passed through `security/sanitize.redact_secrets`,
which redacts token/key/password-shaped values recursively before anything is written or exported.

**Private reasoning is not exposed.** The trace shows operations, artefacts and short decision
summaries — plans, tool calls, arguments, evidence, citation statistics, memory decisions and their
reasons. It does not show hidden chain-of-thought.

---

## Tech stack

| Layer | Choice | Why |
|---|---|---|
| API | FastAPI + Uvicorn | Async, typed request/response models |
| Agent | Custom orchestrator | The planning/execution/budget logic *is* the project; a graph framework would hide it behind abstractions and add version risk |
| Tools | MCP Python SDK ≥ 2.0 (`MCPServer`, `Client`) | Real protocol, three transports, usable by other MCP hosts |
| Source | Jira Cloud REST v3 via `httpx` | Enhanced JQL search, ADF flattening, retries |
| LLM | OpenAI-compatible **or** Anthropic Messages, over plain HTTP | No vendor SDK pinning; works with OpenAI, Azure, OpenRouter, Ollama, vLLM |
| Embeddings | OpenAI, or a deterministic hashing embedder | The demo runs with no API key at all |
| Storage | SQLite (WAL) + numpy vector index | Zero infrastructure; pgvector-shaped interface for the swap |
| Observability | SQLite tracing + optional LangSmith | The UI needs a local trace regardless |
| UI | Static HTML/CSS/JS | No build step; the trace panel is the point |
| Tests | pytest (104 tests) | Unit, integration, e2e, failure injection |

**Deliberately not used:** a graph framework, a separate vector database, a message queue, Redis, or
a JS build pipeline. None of them would make the demonstrated behaviour more real.

---

## Project structure

```
agentic-rag-live-data/
├── README.md  INTERVIEW.md  .env.example  .gitignore  pyproject.toml  Makefile
├── Dockerfile  docker-compose.yml
├── app/
│   ├── main.py                    FastAPI app, static UI, lifespan
│   ├── config.py                  pydantic-settings; every secret from the environment
│   ├── services.py                composition root (the object graph)
│   ├── api/                       routes.py, schemas.py
│   ├── agents/                    rewriter · planner · executor · evaluator · synthesizer
│   │                              orchestrator · state · prompts
│   ├── baseline/naive_rag.py      the basic-RAG comparison
│   ├── mcp_server/                server.py (11 tools, 3 transports), tools.py (implementations)
│   ├── mcp_client/client.py       tools/list + tools/call, timeouts, result normalisation
│   ├── connectors/                base.py (interface) · jira.py (live) · mock_jira.py (demo) · factory.py
│   ├── rag/                       ingestion · embeddings · vector_index · retriever · citations
│   ├── memory/                    short_term · long_term · extractor · retriever · manager
│   ├── database/                  schema.sql · db.py
│   ├── observability/             tracing.py · langsmith_exporter.py
│   ├── models/                    domain.py · evidence.py
│   ├── security/                  sanitize.py (injection, redaction) · guards.py (authz, budget)
│   └── utils/                     dates.py (temporal resolution) · text.py (ADF, chunking)
├── frontend/                      index.html · app.js · styles.css
├── scripts/                       seed_mock_jira.py · sync.py · simulate_change.py · run_demo.py
├── tests/                         104 tests across 8 modules
├── evaluation/                    dataset.jsonl (19 cases) · metrics.py · run_eval.py
└── data/                          mock_jira/*.json (demo source) · app.db (cache, memory, traces)
```

---

## Installation

Python 3.10+ (3.12 recommended).

```bash
git clone <your-repo-url> && cd agentic-rag-live-data
make install          # or: python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
cp .env.example .env
make seed             # writes the demo dataset
make run              # http://localhost:8000
```

Then, in the UI: press **Sync source** once, and ask
*"Summarize what changed in Project Atlas during Q2 2026 and identify the major risks."*

With Docker:

```bash
docker compose up --build          # demo mode, dataset baked in
```

---

## Environment variables

Full list with comments in `.env.example`. The ones that matter:

| Variable | Default | Meaning |
|---|---|---|
| `APP_MODE` | `demo` | `demo` (seeded local source) or `live` (real Jira) |
| `JIRA_BASE_URL` / `JIRA_EMAIL` / `JIRA_API_TOKEN` | empty | Live credentials. Never commit these |
| `JIRA_ALLOWED_PROJECTS` | empty | Comma-separated allow-list; empty = everything the token can read |
| `LLM_PROVIDER` | `none` | `none` \| `openai` \| `anthropic`. `none` runs the deterministic planner/synthesizer |
| `LLM_MODEL` / `LLM_API_KEY` / `LLM_BASE_URL` | — | Point `LLM_BASE_URL` at Ollama/vLLM/OpenRouter to run locally |
| `EMBEDDING_PROVIDER` | `hashing` | `hashing` (offline, lexical) or `openai` (semantic) |
| `MCP_TRANSPORT` | `inproc` | `inproc` \| `stdio` \| `http` |
| `AGENT_MAX_TOOL_CALLS` / `AGENT_MAX_ROUNDS` / `AGENT_MAX_EVIDENCE` | 14 / 3 / 40 | Loop and context protection |
| `MEMORY_TOP_K` / `MEMORY_MIN_RELEVANCE` / `MEMORY_CONFLICT_SIMILARITY` | 5 / 0.30 / 0.80 | Memory selectivity |
| `LANGSMITH_TRACING` / `LANGSMITH_API_KEY` | `false` | Optional trace export |

---

## Demo mode

`APP_MODE=demo` uses `MockJiraConnector` against `data/mock_jira/*.json` — the **same
`SourceConnector` interface**, the same normalised models, the same error types, the same MCP tools.
The dataset is hand-authored so the multi-hop story is real: a vendor dependency (ATLAS-11) blocking
work (ATLAS-9) that pushes an epic's due date three times (ATLAS-1), PCI audit findings (ATLAS-18), a
failing load test (ATLAS-15), and a planted prompt-injection comment (ATLAS-22).

Everything works with **no API keys at all**: no LLM (deterministic planner/synthesizer), no
embedding provider (hashing embedder), no Jira account.

---

## Live Jira mode

```bash
# .env
APP_MODE=live
JIRA_BASE_URL=https://your-site.atlassian.net
JIRA_EMAIL=you@example.com
JIRA_API_TOKEN=...              # https://id.atlassian.com/manage-profile/security/api-tokens
JIRA_ALLOWED_PROJECTS=ATLAS,APOLLO
LLM_PROVIDER=openai
LLM_API_KEY=...
EMBEDDING_PROVIDER=openai
EMBEDDING_API_KEY=...
```

```bash
make run
curl -s localhost:8000/api/health | jq .source     # should report "authenticated as <you>"
python scripts/sync.py --full
```

Nothing above the connector changes. The agent, tools, memory, citations and traces are identical —
that is the point of the interface.

---

## Running the MCP server standalone

```bash
make mcp            # stdio  — python -m app.mcp_server.server
make mcp-http       # Streamable HTTP on :8765
```

To use the tool layer from **Claude Desktop** or another MCP host:

```json
{
  "mcpServers": {
    "jira-live-data": {
      "command": "/absolute/path/.venv/bin/python",
      "args": ["-m", "app.mcp_server.server"],
      "cwd": "/absolute/path/agentic-rag-live-data",
      "env": {"APP_MODE": "live", "JIRA_BASE_URL": "https://your-site.atlassian.net",
              "JIRA_EMAIL": "you@example.com", "JIRA_API_TOKEN": "..."}
    }
  }
}
```

To make *this* system talk to a remote MCP server: `MCP_TRANSPORT=http` and
`MCP_SERVER_URL=http://host:8765/mcp`.

---

## API reference

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/chat` | Run the agent. Body: `{message, session_id?, user_id?, compare_baseline?}` |
| `POST` | `/api/sync` | Incremental (or `{"full": true}`) refresh of the local index |
| `POST` | `/api/ingest` | Full re-index |
| `GET` | `/api/projects` | Projects visible through the connector |
| `GET` | `/api/tools` | The live MCP catalogue, as discovered over the protocol |
| `GET` | `/api/trace/{run_id}` | Full operational trace (owner only) |
| `GET` | `/api/runs` | Recent runs for a user |
| `GET` | `/api/memory` | Long-term records (`?include_inactive=true` shows superseded ones) |
| `POST` | `/api/memory` | Add a record explicitly |
| `DELETE` | `/api/memory/{id}` | Forget one record |
| `GET` | `/api/health`, `/health` | Component status |

```bash
curl -s localhost:8000/api/chat -H 'content-type: application/json' -d '{
  "message": "Summarize what changed in Project Atlas during Q2 2026 and identify the major risks.",
  "session_id": "demo", "compare_baseline": true}' | jq '{answer, tools: [.tool_calls[].tool_name], citations: [.citations[].source_id]}'
```

---

## Database design

One SQLite file, four logical areas. Nothing is duplicated across them.

| Table | Holds | Notes |
|---|---|---|
| `sessions`, `messages`, `session_summaries` | Conversation state | Short-term memory. Deleting a session loses nothing durable |
| `memories` | Curated long-term records + embeddings | `status` ∈ active/superseded/expired; `superseded_by` preserves history |
| `memory_events` | Every create/update/supersede/reject with a reason | Drives the memory section of the trace |
| `documents`, `chunks`, `chunk_embeddings` | Local cache + vector index | Fully rebuildable from the source; `content_hash` prevents redundant embedding |
| `sync_state` | Per-project watermark and stats | Enables incremental sync |
| `agent_runs`, `run_steps`, `tool_calls`, `evidence` | Observability | `evidence.cited` records what the answer actually used |

**Why SQLite:** it makes the repository runnable with zero infrastructure while keeping the SQL
portable. The migration path is documented rather than pretended: `documents`/`chunks` →
Postgres + pgvector, `memories` → the same, traces → LangSmith or ClickHouse.

---

## Testing

```bash
make test        # 104 tests, ~8s
```

| Module | Covers |
|---|---|
| `test_dates_and_text.py` | Quarter/relative-window resolution, Jira offset parsing, ADF flattening, chunking |
| `test_mcp_tools.py` | Tool discovery over the protocol, structured filters, changelog field filtering, blocked/overdue logic, activity counters, structured error envelopes, project allow-list |
| `test_planner.py` | Intent detection, dynamic decomposition (asserts different questions produce different plans), statement vs question routing, memory-supplied subjects, never planning a nonexistent tool |
| `test_memory.py` | Selective extraction, no duplication, supersession, cross-session recall, relevance thresholding, user isolation, session hijack prevention |
| `test_rewriter_and_evaluator.py` | Standalone turns left alone, pronoun and locative resolution, topic-shift period carrying, mid-sentence "and" not treated as a follow-up, meta turns and user statements never rewritten, no invented constraints; sufficiency on empty/off-topic/weak/injection-only evidence, verdicts in the payload, rewrite traced before planning |
| `test_rag_and_citations.py` | Full and incremental sync, re-indexing changed records, project filtering, injection flagging/downranking, invented-marker removal, citation resolution |
| `test_agent_e2e.py` | Multiple tool calls, multi-hop to the blocking dependency, budget enforcement, duplicate suppression, refusal on unanswerable questions, injection resistance, cross-session memory, live-data change moving the answer, trace completeness, trace access control |
| `test_failures.py` | Source outage, missing/corrupt dataset, empty results, missing credentials, Jira 401/429/503, JQL escaping, broken LLM (planner, synthesizer and extractor all fall back), tracing a failed run |
| `test_api.py` | Health, tool catalogue, sync, chat, baseline comparison, memory lifecycle, cross-user isolation, input validation, UI serving |

---

## Evaluation

```bash
make eval        # or: python evaluation/run_eval.py --out evaluation/results/latest.json
```

19 cases across single-hop, temporal, multi-hop, risk, memory-write, memory-recall, memory+RAG,
query-rewrite, insufficient-evidence, no-retrieval and security categories. Cases run in order and share a user,
because the memory cases deliberately depend on each other (write → recall in a new session →
conflict).

Measured per case: tool-selection recall, retrieval recall (expected sources present in evidence),
**plan quality** (fraction of the aspects a correct plan must cover, matched against sub-question
text and requested tools), **evidence relevance** (mean relevance of the evidence the answer
actually cited) and **evidence precision** (cited ÷ collected), **rewrite accuracy**, answer content
recall, citation validity, invented citations, abstention correctness, forbidden content, tool-call
count, evidence count, latency. Aggregated with per-category pass rates, plus
memory precision/recall computed from the final store.

**Every number the harness prints is measured from that run.** None are hard-coded, and none are
reproduced here as a claim — run it and see your own. On the default demo configuration
(`LLM_PROVIDER=none`, hashing embeddings) the suite currently passes all 19 cases; results will
differ with a real LLM and real embeddings, which is exactly why the harness exists.

---

## Security

Everything retrieved from the tracker is treated as **untrusted third-party input** — anyone who can
comment on a ticket can write it.

| Threat | Mitigation |
|---|---|
| Indirect prompt injection | Detection heuristics flag steering attempts; flagged evidence is downranked to 15% of its score, fenced in `<untrusted_content>` blocks, and never allowed near system instructions. The prompt states explicitly that fenced content is data. The dataset ships a real injection attempt (ATLAS-22) and `test_agent_e2e.py` asserts the agent ignores it |
| Role forgery inside content | Angle brackets and `system:`/`assistant:` prefixes are neutralised before the text reaches a prompt |
| Unauthorised project access | `JIRA_ALLOWED_PROJECTS` enforced inside every tool, on both the argument and the results |
| Cross-user memory leakage | Every memory query filters by `user_id`; delete is owner-scoped; sessions are bound to their user (hijack raises `PermissionError`) |
| Trace leakage | `GET /api/trace/{id}` returns 404 for a non-owner |
| Credential exposure | Secrets only in environment variables; `redact_secrets` scrubs traces and logs recursively; the Docker image contains no credentials |
| Malicious tool parameters | Project and issue keys are regex-validated, limits clamped, unknown arguments dropped against the tool's JSON schema |
| Excessive tool usage / infinite loops | `ExecutionBudget`: max calls, max rounds, duplicate-signature suppression, evidence cap, per-call timeout |
| Hallucinated citations | Markers validated against collected evidence; invented ones removed and counted |

**Not implemented** (and therefore not claimed): end-user authentication. The API takes `user_id` at
face value, which is fine for a single-user demo and wrong for production — put an identity provider
in front of it and derive `user_id` from a verified token.

---

## End-to-end demo

```bash
python scripts/run_demo.py
```

Runs, in order: ingest → basic-RAG baseline on the hard question → the agent on the same question →
**elliptical follow-ups being rewritten before planning** → a durable statement being remembered →
cross-session recall → a priority conflict being superseded → the live source changing and the
answer changing with it → an unanswerable question being refused.

Abridged real output (demo mode, no API keys):

```
> And what about Apollo?
  rewritten [heuristic]: And what about Apollo in Q2 2026?
  resolved:  ['implicit period -> Q2 2026']
  planned for project(s): ['APOLLO']

> why did it slip?
  rewritten [heuristic]: why did Project Apollo slip in Q2 2026?
  resolved:  ['it -> Project Apollo', 'implicit period -> Q2 2026']

plan[heuristic] strategy=risk_analysis window=Q2 2026
  - sq1 [answered] What are the goals, lead and scope of ATLAS?       tools: get_project
  - sq2 [answered] What activity happened in ATLAS during Q2 2026?    tools: get_project_activity
  - sq3 [answered] Which issues are overdue or had their due date moved?
  - sq4 [answered] Which issues in ATLAS are blocked, and by what?
  - sq5 [answered] What have people reported about problems in ATLAS?

tool calls:
  get_project           ok    1 items    {'project_key': 'ATLAS'}
  get_project_activity  ok   41 items    {'start': '2026-04-01T00:00:00Z', 'end': '2026-06-30T23:59:59Z'}
  get_overdue_issues    ok    5 items    {'as_of': '2026-06-30T23:59:59Z'}
  get_blocked_issues    ok    2 items
  search_comments       ok    7 items
  semantic_search       ok    3 items

**What changed in Q2 2026 in ATLAS.**
- Due date moved on ATLAS-1: none → 2026-06-12 (2026-04-08) [E5]
- Due date moved on ATLAS-1: 2026-06-12 → 2026-06-30 (2026-05-22) [E4]
- Due date moved on ATLAS-1: 2026-06-30 → 2026-07-24 (2026-06-11) [E2]

**Risks supported by the evidence.**
- ATLAS-11 is blocked (status is 'Blocked'; carries a blocked label) and overdue by 32 days,
  priority Highest: Verityx SDK 4.2 hotfix delivery [E15, E24]
- ATLAS-9 is blocked (… blocked by ATLAS-11 (Blocked)) and overdue by 25 days [E17, E25]
- ATLAS-1 had its due date moved 3 times, most recently to 2026-07-24 on 2026-06-11 —
  repeated slippage rather than a one-off [E5, E4, E2]
- Reported by Marcus Feld: "Still blocked. Verityx support ticket VX-88421 has been open for
  41 days with no fix date." [E9]

_Note: at least one retrieved comment contained text trying to issue instructions to this
assistant. It was treated as data only and excluded from the findings._
```

---

## Video concept → current implementation

| Video concept | Current implementation | Why |
|---|---|---|
| "Connect to Gmail, Notion, Jira, anything" | One source implemented properly (Jira) behind a `SourceConnector` interface | A shallow four-source integration demonstrates less than one real one. The interface is the extensibility claim |
| "Build a data connector using an MCP" | Real MCP server on the official SDK ≥ 2.0, 11 tools, three transports, usable by external hosts | MCP as a working protocol boundary, not a README mention |
| "Use LangSmith to trace the whole path" | Local SQLite tracing always on (powers the UI) + optional LangSmith export as a **root + per-stage + per-tool span tree** | The UI trace must work with no third-party account; LangSmith stays as an additional sink, but "the whole path" means a tree, not one span |
| "The query rewrite **or** the planner's generated questions" | Both, as separate traced stages: `query_rewrite` then `planning` | The transcript lists them as distinct demo elements, and follow-up turns need the first before the second |
| "Agent decomposes the question" | LLM planner constrained by the live tool catalogue, plus a deterministic intent/entity/time planner as fallback | The system must be runnable and reviewable without an API key |
| Static tutorial `/rest/api/3/search` | `POST /rest/api/3/search/jql` with `nextPageToken` | The legacy endpoint was removed from Jira Cloud |
| "Add persistent memory" | Extraction filter → conflict resolution → scored retrieval → bounded store, all traced | "Store everything" is the failure mode the video warns about |

---

## Limitations

Stated plainly rather than papered over.

1. **The default embedder is lexical, not semantic.** `hashing` matches words, not meaning. Set
   `EMBEDDING_PROVIDER=openai` for real paraphrase matching.
2. **The deterministic synthesizer is a report builder, not a writer.** With `LLM_PROVIDER=none` the
   answer is grouped, cited evidence — accurate but not fluent. Prose quality needs an LLM.
3. **Exact vector search only.** Fine for thousands of chunks; it will not scale to millions without
   swapping in pgvector/Qdrant.
4. **`semantic_search` can lag by one sync cycle.** Structured tools always hit the source; the index
   does not. The answer says when it fell back to the index.
5. **No end-user authentication.** `user_id` is taken at face value.
6. **Single-writer SQLite.** Concurrent heavy writes will contend; WAL mitigates but does not remove
   this.
7. **Injection detection is heuristic.** It catches the obvious patterns and downranks them; a novel
   phrasing could pass. The structural defences (fencing, data-only framing, no tool authority from
   content) matter more than the detector.
8. **Jira only.** Gmail/Notion/Slack are a documented extension path, not shipped code.
9. **English-centric heuristics.** Intent detection, risk keywords and memory patterns are English.
10. **`get_project_activity` fans out per issue.** On a very large project this is many API calls;
    it is bounded by `limit`, but a production version would use a bulk changelog endpoint.

---

## Future improvements

- Postgres + pgvector for documents, chunks and memories; keep the current interfaces.
- Webhook-driven ingestion (Jira webhooks → queue → incremental index) instead of pull-based sync.
- Hybrid retrieval: BM25 + vectors + reciprocal rank fusion.
- A learned reranker over collected evidence before synthesis.
- Additional connectors (GitHub first — it has the same issue/comment/history shape).
- LLM-judged evaluation for answer quality alongside the current structural metrics.
- Streaming responses (the trace is naturally incremental).
- Memory consolidation: periodically merge related records into higher-level summaries.
- Per-user OAuth to Jira so the agent inherits each user's real permissions.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `Demo dataset missing` | Seed never ran | `make seed` |
| `/api/health` shows `source.ok: false` in live mode | Bad credentials or URL | Check `JIRA_BASE_URL` has no trailing slash and the token is an **API token**, not a password |
| `The requested API has been removed. Please migrate to /rest/api/3/search/jql` | Old client code | This repo already uses the new endpoint — check you are not running a fork |
| Empty answers, `evidence: 0` | Index never built | `POST /api/sync {"full": true}` or press **Sync source** |
| `semantic_search` returns nothing relevant | Hashing embedder is lexical | Set `EMBEDDING_PROVIDER=openai` + `EMBEDDING_API_KEY` |
| Plan says `generated_by: heuristic` with an LLM configured | Provider error; the system fell back | Check `LLM_API_KEY`/`LLM_BASE_URL`; the run still completed |
| `MCP_TRANSPORT=stdio` hangs | Wrong working directory | Run from the project root so `python -m app.mcp_server.server` resolves |
| `MCP session is not open` | Server failed to start | Check logs; verify `mcp>=2.0` is installed |
| Docker healthcheck failing | Startup slower than the grace period | Increase `start-period`, or check `docker compose logs app` |
| Tests fail on a fresh clone | Dev extras missing | `pip install -e ".[dev]"` |

---

## Resume bullets

- Built an **agentic RAG system over a live Jira issue tracker**: an LLM planner decomposes a
  question into sub-questions, selects from **11 MCP tools discovered at runtime over the Model
  Context Protocol** (in-process, stdio and Streamable HTTP transports), executes multiple retrievals
  under a strict budget, and synthesises an answer in which **every claim is cited to a specific
  issue, comment or changelog entry**, with invented citation markers programmatically removed.

- Implemented **multi-hop retrieval combining structured and semantic access**: issue keys surfaced
  in one hop parameterise the next hop's changelog and comment queries, so questions like *"what
  caused the delay"* resolve a deadline-slip → blocked-dependency → vendor-comment chain, while
  date-, status- and owner-filtered questions bypass the vector store entirely and hit the live
  source through typed tools.

- Designed a **persistent two-tier memory system with conflict resolution**: an extraction filter
  stores only durable user statements (rejecting questions, transient state and model reasoning),
  scored retrieval injects at most five relevant records per turn, and contradicting statements
  **supersede** rather than append — backed by **104 automated tests**, a **19-case evaluation harness**
  measuring tool selection, citation validity, abstention and memory precision/recall, and **full
  operational tracing** (SQLite + optional LangSmith) with secret redaction and prompt-injection
  defences.

---

## Interview questions and answers

> A much fuller set — 67 questions across 19 sections, including the bugs found during the build and
> the trade-offs I would defend — is in **[INTERVIEW.md](INTERVIEW.md)**. The selection below covers
> the questions that come up most often.

**Why is this agentic RAG rather than RAG?**
Retrieval is a *decision*, not a fixed step. The system decomposes the question, chooses tools and
arguments, executes several retrievals, judges whether the evidence is sufficient, retries what came
back empty, and only then writes. Control flow is decided at run time from the question and from
intermediate results.

**Why isn't normal RAG enough?**
Three reasons the demo makes concrete: chunks have no notion of time (`resolved_after`/`before` is a
filter, not a similarity); current field values do not contain their own history (only the changelog
shows a due date moved three times); and risk is a join across overdue, blocked, linked and discussed
signals that no single chunk holds. Add the fourth: the user's own priorities are not in the tracker
at all.

**Why MCP? Why not just call the Jira API from the agent?**
Because it makes the tool layer a *protocol boundary* rather than an import. Three consequences that
matter: the agent discovers capabilities at run time via `tools/list`, so it cannot plan a tool that
does not exist; the same server is reusable by any MCP host (Claude Desktop, an IDE, another agent)
with no code change; and the tool layer can be deployed as its own process — this repo runs it
in-process, as a stdio subprocess and over HTTP with one environment variable. It also puts a clean
place for validation, authorisation and error shaping between the model and the credentials.

**Why is there a query rewriter as well as a planner?**
They solve different problems and fail differently. The rewriter resolves *reference* — turning
"why did it slip?" into a standalone question — using only the session transcript. The planner
decomposes a standalone question into retrieval work. Merging them would let a decomposition
mistake and a reference mistake hide in the same step; keeping them separate means the trace shows
exactly which one went wrong. The rewriter is also deliberately conservative: it may resolve
references but never add constraints, and it refuses to touch meta turns, user statements and
self-contained questions.

**How does the agent know its evidence is good enough?**
An explicit evaluation step after each sub-question. The deterministic check requires both a
relevance floor and coverage of the sub-question's content-bearing terms, and discounts evidence
that only survived because an injection attempt was downranked. With an LLM it judges the same
evidence and can name a specific tool and arguments for the gap — that suggestion becomes the first
call of the retry round, which is what stops round two being a rerun of round one. Every verdict is
in the trace with its relevance, coverage and named gap.

**How does the planner work?**
It receives the question, the live tool catalogue, the project list (fetched first, so it is
grounded), the retrieved memory and the session context. It returns a structured `Plan`: goal,
strategy, entities, an explicit ISO time window, and sub-questions with tool hints and `depends_on`
edges. Tool hints are filtered against the real catalogue. With no LLM configured, a deterministic
planner composes the same structure from detected intents, matched project entities and a resolved
time window — different questions genuinely produce different plans, which is asserted in the tests.

**How does the agent decide which tools to call?**
Per sub-question. With an LLM: the sub-question, the plan context, the issue keys discovered so far
and the calls already made go into a constrained JSON selection prompt, and arguments are filtered
against the tool's JSON schema. Without one: rules map intent and sub-question wording onto arguments
(completion wording → `resolved_after/before` + `status_category=Done`; deadline wording →
`unresolved_only` + `due_before`). Both paths then pass through the same budget, deduplication and
evidence pipeline.

**How is memory different from RAG?**
RAG retrieves from a corpus that exists independently of the user and is rebuilt from the source.
Memory is a small curated set of statements *about the user*, written by the system, that no source
system contains. Different write path (extraction and conflict resolution vs ingestion), different
retrieval budget (≤5 records vs top-k chunks), different lifecycle (superseded/expired vs
re-indexed).

**Why can't you store every conversation?**
Precision collapses and cost rises. Every stored message competes for the same top-k slots, so
retrieval starts surfacing chit-chat instead of the one durable preference that matters; contradictory
statements accumulate with no way to tell which is current; and the store grows without bound. Storing
the agent's own output is worse — it makes the system re-ingest its own inferences as facts.

**How do you prevent memory pollution?**
An extraction filter with an explicit taxonomy (preference/context/fact/goal) that rejects questions,
one-off instructions, transient state and anything resembling reasoning; an importance threshold; a
duplicate check at 0.94 similarity; a relevance threshold at retrieval so irrelevant records are never
injected; and a bounded store that prunes least-important, least-used, oldest first. Every rejection
is logged with its reason and visible in the trace.

**How do you handle conflicting memory?**
Similarity against active records of the same type, with singleton subjects (`priority`, `role`,
`goal`) treated as at most one active record. ≥0.94 → skip as duplicate. ≥0.80 with a different claim
→ write the new record and mark the old one `superseded_by` it. The old record is retained, not
deleted, so the history is auditable — but only the current one is ever retrieved.

**How do citations work?**
Every tool item becomes an `Evidence` record with source, source id, URL, timestamp, content,
relevance and the tool that produced it, and gets an id `E1…En`. The synthesizer sees only those
records and must cite by id. Afterwards, markers are extracted and validated: anything that does not
resolve to a collected record is stripped and counted. Surviving markers become citation objects the
UI renders as links to the issue, comment or history entry.

**How do you prevent hallucination?**
Structurally, not by asking nicely. The synthesizer sees only evidence; markers are validated against
real records; a coverage check flags question terms that appear nowhere in the evidence and prepends an
explicit "the evidence does not cover X" statement; sub-questions that returned nothing are listed as
not covered; lines with no citation are counted and shown in the trace; and the evaluation harness has
dedicated insufficient-evidence cases that fail if the agent answers instead of abstaining.

**How do you handle prompt injection?**
Layered. Untrusted content is fenced in `<untrusted_content>` blocks with an explicit data-only
instruction; role markers and angle brackets inside it are neutralised; injection patterns are detected
and matching evidence is downranked to 15% of its score and reported in the answer; retrieved content
has no authority to trigger tool calls (only the planner and executor choose tools, and arguments are
schema-validated); and credentials live outside the model's reach entirely. The demo dataset contains a
real injection attempt and a test asserts the agent neither complies nor leaks.

**How does live data differ from static RAG?**
The source of truth changes under you. That forces three things this system implements: structured
reads go to the source on every call rather than to a cache; the index is refreshed incrementally with
watermarks and content hashes so only changed records are re-embedded; and the answer distinguishes
live reads from index reads, saying so when it fell back to the cache.

**What happens if Jira is unavailable?**
Tool calls return `{"ok": false, "error": {"kind": "unavailable"}}` rather than raising. The executor
records the failure, the run continues, `semantic_search` can still answer from the local index, and
the answer opens with "Retrieval was degraded for this answer… treat it as possibly stale." If nothing
is retrievable, the agent says so instead of guessing. All of it is traced, and `test_failures.py`
covers outage, 401, 429 with retry, and 503 with exhausted retries.

**How do you evaluate agent quality?**
Structural metrics you can compute without a human: tool-selection recall against expected tools,
retrieval recall against expected source ids, answer content recall, citation validity and invented-
marker count, abstention correctness on unanswerable questions, forbidden-content rate, tool-call
count, evidence count and latency — plus memory precision/recall computed from the final store, with
per-category pass rates. The gap is answer *fluency*, which needs either human review or an LLM judge;
that is listed under future improvements rather than claimed.

**How do you prevent infinite tool loops?**
`ExecutionBudget`: a hard cap on total calls, a cap on rounds, a signature set that suppresses
identical calls (recorded as `skipped_duplicate`, not silently dropped), an evidence cap, per-item
truncation and a per-call timeout. Retries are bounded to one relaxed round over unresolved
sub-questions. The budget snapshot is returned with every response.

**How would you scale this?**
Move `documents`/`chunks`/`memories` to Postgres + pgvector behind the existing interfaces; replace
pull-based sync with Jira webhooks feeding a queue; run the MCP server as its own horizontally scaled
service (`MCP_TRANSPORT=http`); cache hot structured reads with short TTLs while keeping risk-critical
reads live; and shard traces into a store built for them. The agent code does not change — that is
what the connector and tool interfaces buy.

**How would you add Gmail / Notion / Slack?**
Implement `SourceConnector` for the source, normalise into the same domain models, register the new
tools on the MCP server with clear "when to call this" descriptions, and add ingestion mapping for the
document types. Planner, executor, memory, citations and tracing are untouched. Gmail maps threads →
issues and messages → comments; Notion maps pages → documents with block-level chunks; Slack maps
channels → projects and threads → discussions. The honest caveat: each source needs its own auth,
rate-limit and permission model, which is where the real work is.

**Why LangSmith?**
For run-level comparison across prompt and model changes, and for sharing traces with people who
should not have database access. It is optional here because the UI needs a trace regardless of any
third-party account, so local SQLite tracing is the primary sink and LangSmith is an additional
exporter that can fail without affecting a request.

**What exactly is stored in long-term memory?**
Short third-person statements the user made about themselves or their world, typed as
preference/context/fact/goal, with subject, importance, confidence, timestamps, provenance
(session/run) and an embedding. Not messages, not tool results, not tracker data (that is re-fetchable
and would go stale), and not model reasoning.

**How does multi-hop reasoning work here?**
The plan carries `depends_on` edges; the executor extracts issue keys from the evidence produced by
those dependencies and uses them as arguments for the next hop. "Why did the cutover date move" runs:
locate relevant issues semantically → fetch their changelogs (three due-date moves) → fetch blocked
issues and comments (vendor dependency, no fix date) → synthesise the chain with a citation on each
link.

**What are the major limitations?**
See the Limitations section — the honest headline ones are the lexical default embedder, the
report-style deterministic synthesizer without an LLM, exact-search-only vectors, no end-user
authentication, and heuristic (not guaranteed) injection detection.

---

## License

MIT. The demo dataset is fictional; any resemblance to a real payments migration is a coincidence
every payments engineer will recognise anyway.
