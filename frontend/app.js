/* Demo UI: chat + operational agent trace. No build step, no framework. */
const state = {
  sessionId: null,
  userId: null,
  lastRun: null,
};

const el = (id) => document.getElementById(id);
const escapeHtml = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => (
  { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
));

const EXAMPLES = [
  "Summarize what changed in Project Atlas during Q2 2026 and identify the major risks.",
  "What caused the delay in Project Atlas?",
  "Project Atlas is my highest priority this quarter.",
  "Given my priorities, what should I focus on?",
  "Who owns Project Apollo and what is blocked there?",
  "What did I say my priority was?",
  "And what is overdue there?",
];

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!response.ok) {
    let detail = response.statusText;
    try { detail = (await response.json()).detail || detail; } catch (_) {}
    throw new Error(detail);
  }
  return response.json();
}

/* ---------- rendering helpers ---------- */
function renderAnswer(text, citations) {
  const byMarker = Object.fromEntries((citations || []).map((c) => [c.marker, c]));
  let html = escapeHtml(text);
  html = html.replace(/\[(E\d+(?:,\s*E\d+)*)\]/g, (match, group) => {
    const parts = group.split(",").map((m) => m.trim());
    const rendered = parts.map((marker) => {
      const citation = byMarker[marker];
      if (!citation) return `<span class="cite">${marker}</span>`;
      const label = `${citation.source_id} · ${citation.tool_used}`;
      const href = citation.url || "#";
      return `<a class="cite" href="${escapeHtml(href)}" target="_blank" rel="noopener" title="${escapeHtml(label)}">${marker}</a>`;
    });
    return `[${rendered.join(", ")}]`;
  });
  html = html.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
  html = html.replace(/_([^_\n]+)_/g, "<em>$1</em>");
  return html;
}

function addMessage(role, html, meta) {
  const node = document.createElement("div");
  node.className = `msg ${role}`;
  node.innerHTML = html + (meta ? `<div class="meta">${escapeHtml(meta)}</div>` : "");
  el("messages").appendChild(node);
  el("messages").scrollTop = el("messages").scrollHeight;
  return node;
}

function section(title, badge, bodyHtml, open = false) {
  const badgeClass = badge && badge.cls ? ` ${badge.cls}` : "";
  const badgeHtml = badge ? `<span class="badge${badgeClass}">${escapeHtml(badge.text)}</span>` : "";
  return `<details${open ? " open" : ""}>
    <summary><span>${escapeHtml(title)}</span>${badgeHtml}</summary>
    <div class="body">${bodyHtml}</div>
  </details>`;
}

function kv(pairs) {
  return `<div class="kv">${pairs
    .map(([k, v]) => `<span>${escapeHtml(k)}</span><span>${escapeHtml(String(v))}</span>`)
    .join("")}</div>`;
}

/* ---------- trace panel ---------- */
function renderTrace(result) {
  const plan = result.plan || {};
  const subQuestions = plan.sub_questions || [];
  const parts = [];

  parts.push(section("1 · User question", null, `<div>${escapeHtml(result.question || "")}</div>`, true));

  const rewrite = result.rewrite || {};
  parts.push(section(
    "2 · Query rewrite",
    { text: rewrite.changed ? `resolved (${rewrite.method || "-"})` : "not needed",
      cls: rewrite.changed ? "ok" : "" },
    rewrite.changed
      ? `<div><strong>${escapeHtml(rewrite.rewritten || "")}</strong></div>
         ${(rewrite.resolved || []).length ? `<ul class="plain">${(rewrite.resolved || []).map((r) => `<li><span class="tag">resolved</span>${escapeHtml(r)}</li>`).join("")}</ul>` : ""}
         <div class="meta">${escapeHtml(rewrite.reason || "")}</div>`
      : `<div class='spinner'>${escapeHtml(rewrite.reason || "The turn was already standalone.")}</div>`
  ));

  const memories = result.memories_used || [];
  parts.push(section(
    "3 · Memory retrieval",
    { text: `${memories.length} used`, cls: memories.length ? "ok" : "" },
    memories.length
      ? `<ul class="plain">${memories.map((m) => `<li><span class="tag">${escapeHtml(m.type)}</span>${escapeHtml(m.content)}
         <div class="meta">score ${m.score} · importance ${m.importance} · ${escapeHtml(m.memory_id)}</div></li>`).join("")}</ul>`
      : "<div class='spinner'>No long-term memory scored above the relevance threshold.</div>"
  ));

  parts.push(section(
    "4 · Plan",
    { text: plan.generated_by || "", cls: plan.generated_by === "llm" ? "ok" : "warn" },
    kv([
      ["goal", plan.goal || ""],
      ["strategy", plan.reasoning_strategy || ""],
      ["needs retrieval", plan.needs_retrieval],
      ["time window", plan.time_range ? `${plan.time_range.label || "-"} (${plan.time_range.start || "?"} → ${plan.time_range.end || "?"})` : "-"],
      ["projects", (plan.entities && plan.entities.project_keys || []).join(", ") || "-"],
      ["notes", plan.notes || "-"],
    ]),
    true
  ));

  parts.push(section(
    "5 · Sub-questions",
    { text: `${subQuestions.length}`, cls: "" },
    subQuestions.length
      ? `<ul class="plain">${subQuestions.map((sq) => {
          const verdict = (result.sufficiency || {})[sq.id] || {};
          return `<li>
          <span class="tag">${escapeHtml(sq.id)}</span>
          <span class="badge ${sq.status === "answered" ? "ok" : "warn"}">${escapeHtml(sq.status)}</span>
          <div>${escapeHtml(sq.question)}</div>
          <div class="meta">${escapeHtml(sq.rationale || "")}</div>
          <div class="meta">tools: ${escapeHtml((sq.tool_hints || []).join(", ") || "-")}${
            sq.depends_on && sq.depends_on.length ? ` · depends on ${escapeHtml(sq.depends_on.join(", "))}` : ""}</div>
          ${verdict.method ? `<div class="meta">sufficiency [${escapeHtml(verdict.method)}]: relevance ${verdict.top_relevance} · term coverage ${verdict.term_coverage}${
            verdict.missing ? ` · gap: ${escapeHtml(verdict.missing)}` : ""}</div>` : ""}
        </li>`; }).join("")}</ul>`
      : "<div class='spinner'>No decomposition: this question needed no retrieval.</div>"
  ));

  const calls = result.tool_calls || [];
  const failed = calls.filter((c) => c.status !== "ok").length;
  parts.push(section(
    "6 · Tool calls (MCP)",
    { text: `${calls.length} calls${failed ? `, ${failed} skipped/failed` : ""}`, cls: failed ? "warn" : "ok" },
    calls.length
      ? `<ul class="plain">${calls.map((c) => `<li>
          <span class="badge ${c.status === "ok" ? "ok" : "err"}">${escapeHtml(c.status)}</span>
          <strong>${escapeHtml(c.tool_name)}</strong>
          <span class="meta">${c.result_count} items · ${c.duration_ms}ms · ${escapeHtml(c.sub_question_id || "")}</span>
          <pre>${escapeHtml(JSON.stringify(c.arguments, null, 1))}</pre>
          ${c.error ? `<div class="meta">${escapeHtml(c.error)}</div>` : ""}
        </li>`).join("")}</ul>`
      : "<div class='spinner'>No tools were called.</div>"
  ));

  const evidence = result.evidence || [];
  parts.push(section(
    "7 · Evidence",
    { text: `${evidence.length} collected · ${evidence.filter((e) => e.cited).length} cited`, cls: "" },
    evidence.length
      ? `<ul class="plain">${evidence.map((e) => `<li>
          <span class="tag ${e.cited ? "cited" : ""}">${escapeHtml(e.evidence_id)}</span>
          ${e.injection_suspected ? '<span class="tag inj">injection suspected</span>' : ""}
          <a href="${escapeHtml(e.url || "#")}" target="_blank" rel="noopener">${escapeHtml(e.source_id)}</a>
          <span class="meta">${escapeHtml(e.source_type)} · ${escapeHtml(e.tool_used)} · relevance ${e.relevance}</span>
          <div class="meta">${escapeHtml((e.content || "").slice(0, 260))}</div>
        </li>`).join("")}</ul>`
      : "<div class='spinner'>No evidence was collected.</div>"
  ));

  const stats = result.citation_stats || {};
  parts.push(section(
    "8 · Synthesis & citations",
    { text: `${stats.citation_count || 0} citations`, cls: stats.invented_markers_removed ? "warn" : "ok" },
    kv([
      ["writer", stats.synthesizer || "-"],
      ["evidence cited", `${stats.evidence_cited || 0} / ${stats.evidence_available || 0}`],
      ["coverage", stats.citation_coverage ?? "-"],
      ["invented markers removed", stats.invented_markers_removed || 0],
      ["lines without a citation", stats.uncited_claims ?? "-"],
      ["insufficient evidence", result.insufficient_evidence],
    ])
  ));

  const update = result.memory_update || {};
  const actions = update.actions || [];
  parts.push(section(
    "9 · Memory update",
    { text: actions.length ? actions.map((a) => a.action).join(", ") : "no write", cls: actions.length ? "ok" : "" },
    `${kv([["extractor", update.extractor || "-"], ["candidates", update.candidates_considered ?? 0], ["pruned", update.pruned ?? 0]])}
     ${actions.length ? `<ul class="plain">${actions.map((a) => `<li><span class="tag">${escapeHtml(a.action)}</span>${escapeHtml(a.content || "")}
        ${a.superseded_memory_id ? `<div class="meta">replaced ${escapeHtml(a.superseded_memory_id)}: ${escapeHtml(a.previous_content || "")}</div>` : ""}</li>`).join("")}</ul>` : ""}
     ${(update.rejected || []).length ? `<div class="meta">rejected: ${escapeHtml((update.rejected || []).map((r) => `${r.content} (${r.reason})`).join(" | ").slice(0, 400))}</div>` : ""}`
  ));

  const budget = result.budget || {};
  parts.push(section(
    "10 · Run",
    { text: `${result.latency_ms || 0}ms`, cls: "" },
    kv([
      ["run id", result.run_id || "-"],
      ["tool budget", `${budget.tool_calls_used ?? 0} / ${budget.max_tool_calls ?? "-"}`],
      ["rounds", `${budget.rounds_used ?? 0} / ${budget.max_rounds ?? "-"}`],
      ["warnings", (result.warnings || []).join("; ") || "none"],
      ["full trace", `GET /api/trace/${result.run_id || ""}`],
    ])
  ));

  el("trace").innerHTML = parts.join("");
}

/* ---------- chat ---------- */
async function ask(question) {
  if (!question.trim()) return;
  el("send-btn").disabled = true;
  addMessage("user", escapeHtml(question));
  const pending = addMessage("assistant", "<span class='spinner'>planning → retrieving → synthesising…</span>");

  try {
    const result = await api("/api/chat", {
      method: "POST",
      body: JSON.stringify({
        message: question,
        session_id: state.sessionId,
        compare_baseline: el("compare-toggle").checked,
      }),
    });
    state.sessionId = result.session_id;
    state.lastRun = result;
    result.question = question;

    pending.innerHTML = renderAnswer(result.answer, result.citations) +
      `<div class="meta">${result.tool_calls.length} tool calls · ${result.evidence.length} evidence · ` +
      `${result.citations.length} citations · ${result.memories_used.length} memories · ${result.latency_ms}ms</div>`;

    if (result.baseline) {
      const baseline = result.baseline;
      addMessage("baseline",
        `<h4>Basic RAG baseline (no planning, no tools, no memory)</h4>${escapeHtml(baseline.answer || baseline.error || "")}` +
        `<div class="meta">${baseline.retrieved_chunks ? baseline.retrieved_chunks.length : 0} chunks · 0 tool calls · 0 citations · ${baseline.latency_ms || 0}ms</div>`);
    }
    renderTrace(result);
  } catch (error) {
    pending.innerHTML = `<span class="badge err">error</span> ${escapeHtml(error.message)}`;
  } finally {
    el("send-btn").disabled = false;
  }
}

/* ---------- drawers ---------- */
function openDrawer(title, html) {
  el("drawer-title").textContent = title;
  el("drawer-body").innerHTML = html;
  el("drawer").classList.remove("hidden");
}

async function showMemory() {
  openDrawer("Long-term memory", "<div class='spinner'>loading…</div>");
  try {
    const data = await api("/api/memory?include_inactive=true");
    const active = data.memories.filter((m) => m.status === "active");
    const inactive = data.memories.filter((m) => m.status !== "active");
    const render = (records) => records.map((m) => `<li>
        <span class="tag">${escapeHtml(m.type)}</span>${escapeHtml(m.content)}
        <div class="meta">importance ${m.importance} · used ${m.use_count}× · ${escapeHtml(m.created_at || "")} · ${escapeHtml(m.status)}</div>
        ${m.status === "active" ? `<button data-forget="${escapeHtml(m.memory_id)}">forget</button>` : ""}
      </li>`).join("");
    openDrawer("Long-term memory",
      `<p class="hint">Persisted across sessions for user <code>${escapeHtml(data.user_id)}</code>. Only durable statements are stored - never whole conversations.</p>
       <h4>Active (${active.length})</h4><ul class="plain">${render(active) || "<li>nothing yet</li>"}</ul>
       ${inactive.length ? `<h4>Superseded / expired (${inactive.length})</h4><ul class="plain">${render(inactive)}</ul>` : ""}`);
    el("drawer-body").querySelectorAll("[data-forget]").forEach((button) => {
      button.addEventListener("click", async () => {
        await api(`/api/memory/${button.dataset.forget}`, { method: "DELETE" });
        showMemory();
      });
    });
  } catch (error) {
    openDrawer("Long-term memory", `<span class="badge err">${escapeHtml(error.message)}</span>`);
  }
}

async function showTools() {
  openDrawer("MCP tool catalogue", "<div class='spinner'>loading…</div>");
  try {
    const data = await api("/api/tools");
    openDrawer("MCP tool catalogue",
      `<p class="hint">Discovered over MCP (<code>tools/list</code>), transport <code>${escapeHtml(data.transport)}</code>. The agent chooses among these at run time.</p>
       <ul class="plain">${data.tools.map((t) => `<li><strong>${escapeHtml(t.name)}</strong>
        <pre>${escapeHtml(t.signature)}</pre><div class="meta">${escapeHtml(t.description)}</div></li>`).join("")}</ul>`);
  } catch (error) {
    openDrawer("MCP tool catalogue", `<span class="badge err">${escapeHtml(error.message)}</span>`);
  }
}

async function refreshStatus() {
  try {
    const health = await api("/api/health");
    state.userId = state.userId || null;
    el("status-line").textContent =
      `${health.app_mode} mode · source ${health.source.ok ? "ok" : "unavailable"} · MCP ${health.mcp.transport} (${health.mcp.tools.length} tools) · ` +
      `LLM ${health.llm.active ? health.llm.model : "none (deterministic fallback)"} · ` +
      `index ${health.index.embedded_chunks} chunks · LangSmith ${health.observability.langsmith ? "on" : "off"}`;
    document.querySelector(".dot").className = health.source.ok ? "dot" : "dot bad";
  } catch (error) {
    el("status-line").textContent = `backend unreachable: ${error.message}`;
    document.querySelector(".dot").className = "dot bad";
  }
}

/* ---------- wiring ---------- */
el("chat-form").addEventListener("submit", (event) => {
  event.preventDefault();
  const question = el("input").value;
  el("input").value = "";
  ask(question);
});
el("input").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    el("chat-form").requestSubmit();
  }
});
el("sync-btn").addEventListener("click", async () => {
  el("sync-btn").disabled = true;
  el("sync-btn").textContent = "Syncing…";
  try {
    const result = await api("/api/sync", { method: "POST", body: JSON.stringify({ full: false }) });
    const stats = result.stats;
    addMessage("baseline", `<h4>Source sync</h4>${escapeHtml(
      `${stats.mode} sync: ${stats.issues_fetched} issues fetched, ${stats.documents_upserted} documents changed, ` +
      `${stats.documents_unchanged} unchanged, ${stats.chunks_indexed} chunks re-embedded.` +
      (stats.errors.length ? ` Errors: ${stats.errors.join("; ")}` : ""))}`);
    refreshStatus();
  } catch (error) {
    addMessage("baseline", `<span class="badge err">sync failed: ${escapeHtml(error.message)}</span>`);
  } finally {
    el("sync-btn").disabled = false;
    el("sync-btn").textContent = "Sync source";
  }
});
el("memory-btn").addEventListener("click", showMemory);
el("tools-btn").addEventListener("click", showTools);
el("drawer-close").addEventListener("click", () => el("drawer").classList.add("hidden"));

el("examples").innerHTML = EXAMPLES.map((q, i) => `<button data-example="${i}">${escapeHtml(q.length > 52 ? q.slice(0, 52) + "…" : q)}</button>`).join("");
el("examples").querySelectorAll("[data-example]").forEach((button) => {
  button.addEventListener("click", () => ask(EXAMPLES[Number(button.dataset.example)]));
});

refreshStatus();
addMessage("assistant",
  "Ask about a project and I will plan the retrieval, call the tracker through MCP tools, collect evidence and cite it. " +
  "Tell me something durable about yourself and I will remember it across sessions.");
