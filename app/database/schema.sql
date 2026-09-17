-- ---------------------------------------------------------------------------
-- Operational store (SQLite). One file, four logical areas:
--   1. conversation state   sessions / messages / session_summaries   (short-term memory)
--   2. long-term memory     memories                                   (cross-session facts)
--   3. local index/cache    documents / chunks / chunk_embeddings / sync_state
--   4. observability        agent_runs / run_steps / tool_calls / evidence / memory_events
-- Jira remains the SOURCE OF TRUTH; everything here is derived and disposable.
-- ---------------------------------------------------------------------------

PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS sessions (
    session_id     TEXT PRIMARY KEY,
    user_id        TEXT NOT NULL,
    title          TEXT DEFAULT '',
    created_at     TEXT NOT NULL,
    last_active_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id, last_active_at DESC);

CREATE TABLE IF NOT EXISTS messages (
    message_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    user_id    TEXT NOT NULL,
    role       TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content    TEXT NOT NULL,
    run_id     TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, created_at);

CREATE TABLE IF NOT EXISTS session_summaries (
    session_id     TEXT PRIMARY KEY REFERENCES sessions(session_id) ON DELETE CASCADE,
    summary        TEXT NOT NULL,
    covered_upto   TEXT NOT NULL,
    updated_at     TEXT NOT NULL
);

-- Long-term memory: curated, deduplicated, supersedable. NOT a message log.
CREATE TABLE IF NOT EXISTS memories (
    memory_id      TEXT PRIMARY KEY,
    user_id        TEXT NOT NULL,
    type           TEXT NOT NULL CHECK (type IN ('preference','context','fact','goal')),
    content        TEXT NOT NULL,
    subject        TEXT DEFAULT '',
    importance     REAL NOT NULL DEFAULT 0.5,
    confidence     REAL NOT NULL DEFAULT 0.5,
    status         TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active','superseded','expired')),
    superseded_by  TEXT,
    source_session TEXT,
    source_run     TEXT,
    created_at     TEXT NOT NULL,
    updated_at     TEXT NOT NULL,
    expires_at     TEXT,
    last_used_at   TEXT,
    use_count      INTEGER NOT NULL DEFAULT 0,
    embedding      BLOB,
    embedding_model TEXT DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_memories_user ON memories(user_id, status, updated_at DESC);

CREATE TABLE IF NOT EXISTS memory_events (
    event_id   TEXT PRIMARY KEY,
    run_id     TEXT,
    user_id    TEXT NOT NULL,
    memory_id  TEXT,
    action     TEXT NOT NULL,       -- created | updated | superseded | skipped_duplicate | rejected
    reason     TEXT DEFAULT '',
    content    TEXT DEFAULT '',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_memory_events_run ON memory_events(run_id);

-- Local index/cache of the live source (rebuildable at any time from Jira).
CREATE TABLE IF NOT EXISTS documents (
    doc_id         TEXT PRIMARY KEY,
    source         TEXT NOT NULL,
    source_type    TEXT NOT NULL,   -- issue | comment | project
    source_id      TEXT NOT NULL,
    project_key    TEXT NOT NULL DEFAULT '',
    issue_key      TEXT NOT NULL DEFAULT '',
    title          TEXT DEFAULT '',
    url            TEXT DEFAULT '',
    text           TEXT NOT NULL,
    content_hash   TEXT NOT NULL,
    source_created TEXT,
    source_updated TEXT,
    ingested_at    TEXT NOT NULL,
    metadata_json  TEXT DEFAULT '{}',
    UNIQUE (source, source_type, source_id)
);
CREATE INDEX IF NOT EXISTS idx_documents_project ON documents(project_key, source_updated DESC);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id      TEXT PRIMARY KEY,
    doc_id        TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    ordinal       INTEGER NOT NULL,
    text          TEXT NOT NULL,
    project_key   TEXT NOT NULL DEFAULT '',
    issue_key     TEXT NOT NULL DEFAULT '',
    source_type   TEXT NOT NULL DEFAULT '',
    source_updated TEXT,
    metadata_json TEXT DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
CREATE INDEX IF NOT EXISTS idx_chunks_project ON chunks(project_key, source_type);

CREATE TABLE IF NOT EXISTS chunk_embeddings (
    chunk_id  TEXT PRIMARY KEY REFERENCES chunks(chunk_id) ON DELETE CASCADE,
    model     TEXT NOT NULL,
    dim       INTEGER NOT NULL,
    vector    BLOB NOT NULL
);

-- Incremental sync watermarks so we do not reprocess the whole backlog every time.
CREATE TABLE IF NOT EXISTS sync_state (
    source        TEXT NOT NULL,
    resource      TEXT NOT NULL,   -- project key, or '_global'
    watermark     TEXT,            -- max(source_updated) seen
    last_sync_at  TEXT,
    stats_json    TEXT DEFAULT '{}',
    PRIMARY KEY (source, resource)
);

-- Observability -------------------------------------------------------------
CREATE TABLE IF NOT EXISTS agent_runs (
    run_id        TEXT PRIMARY KEY,
    session_id    TEXT,
    user_id       TEXT NOT NULL,
    question      TEXT NOT NULL,
    mode          TEXT NOT NULL DEFAULT 'agentic',   -- agentic | baseline
    status        TEXT NOT NULL DEFAULT 'running',   -- running | completed | failed
    answer        TEXT DEFAULT '',
    plan_json     TEXT DEFAULT '{}',
    started_at    TEXT NOT NULL,
    finished_at   TEXT,
    latency_ms    INTEGER DEFAULT 0,
    tool_calls    INTEGER DEFAULT 0,
    evidence_count INTEGER DEFAULT 0,
    memories_used INTEGER DEFAULT 0,
    error         TEXT DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_runs_session ON agent_runs(session_id, started_at DESC);

CREATE TABLE IF NOT EXISTS run_steps (
    step_id     TEXT PRIMARY KEY,
    run_id      TEXT NOT NULL REFERENCES agent_runs(run_id) ON DELETE CASCADE,
    ordinal     INTEGER NOT NULL,
    name        TEXT NOT NULL,
    step_type   TEXT NOT NULL,      -- chain | tool | retriever | memory | llm
    status      TEXT NOT NULL DEFAULT 'ok',
    input_json  TEXT DEFAULT '{}',
    output_json TEXT DEFAULT '{}',
    started_at  TEXT NOT NULL,
    ended_at    TEXT,
    duration_ms INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_steps_run ON run_steps(run_id, ordinal);

CREATE TABLE IF NOT EXISTS tool_calls (
    call_id        TEXT PRIMARY KEY,
    run_id         TEXT NOT NULL REFERENCES agent_runs(run_id) ON DELETE CASCADE,
    sub_question_id TEXT DEFAULT '',
    round_index    INTEGER DEFAULT 0,
    tool_name      TEXT NOT NULL,
    arguments_json TEXT DEFAULT '{}',
    status         TEXT NOT NULL DEFAULT 'ok',
    result_count   INTEGER DEFAULT 0,
    duration_ms    INTEGER DEFAULT 0,
    error          TEXT DEFAULT '',
    created_at     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_toolcalls_run ON tool_calls(run_id);

CREATE TABLE IF NOT EXISTS evidence (
    evidence_id     TEXT NOT NULL,
    run_id          TEXT NOT NULL REFERENCES agent_runs(run_id) ON DELETE CASCADE,
    source          TEXT NOT NULL,
    source_type     TEXT NOT NULL,
    source_id       TEXT NOT NULL,
    title           TEXT DEFAULT '',
    url             TEXT DEFAULT '',
    timestamp       TEXT,
    content         TEXT NOT NULL,
    relevance       REAL DEFAULT 0,
    tool_used       TEXT DEFAULT '',
    sub_question_id TEXT DEFAULT '',
    project_key     TEXT DEFAULT '',
    injection_suspected INTEGER DEFAULT 0,
    cited           INTEGER DEFAULT 0,
    PRIMARY KEY (run_id, evidence_id)
);
