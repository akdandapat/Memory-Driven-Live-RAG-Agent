#!/usr/bin/env python3
"""Reproducible end-to-end demo, no server or UI required.

Shows, in order:
  1. an ingestion pass over the live source;
  2. the basic-RAG baseline answering a hard question badly;
  3. the agent answering the same question with plan, tools, evidence and citations;
  4. memory being written, then used in a later session;
  5. memory conflict handling (a priority that changes);
  6. live data changing underneath the agent and the answer moving with it.

    python scripts/run_demo.py
"""
from __future__ import annotations

import asyncio
import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.services import Services  # noqa: E402

USER = "demo-user"


def rule(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def show(result: dict, *, answer_chars: int = 1800) -> None:
    rewrite = result.get("rewrite", {})
    if rewrite.get("changed"):
        print(f"\nrewrite[{rewrite.get('method')}]: {rewrite.get('rewritten')}")
        print(f"  resolved: {rewrite.get('resolved')}")
    plan = result.get("plan", {})
    print(f"\nplan[{plan.get('generated_by')}] strategy={plan.get('reasoning_strategy')} "
          f"window={plan.get('time_range', {}).get('label') or '-'}")
    for sub_question in plan.get("sub_questions", []):
        verdict = (result.get("sufficiency") or {}).get(sub_question["id"], {})
        print(f"  - {sub_question['id']} [{sub_question['status']}] {sub_question['question']}")
        print(f"      tools: {', '.join(sub_question.get('tool_hints', [])) or '-'}")
        if verdict:
            gap = f" gap: {verdict.get('missing')}" if verdict.get("missing") else ""
            print(f"      sufficiency[{verdict.get('method')}]: relevance "
                  f"{verdict.get('top_relevance')} coverage {verdict.get('term_coverage')}{gap}")
    print("\ntool calls:")
    for call in result.get("tool_calls", []):
        print(f"  - {call['tool_name']:<22} {call['status']:<10} {call['result_count']:>3} items "
              f"{call['duration_ms']:>5}ms  {call['arguments']}")
    print(f"\nevidence: {len(result.get('evidence', []))} collected, "
          f"{sum(1 for e in result.get('evidence', []) if e['cited'])} cited")
    print(f"memories used: {[m['content'] for m in result.get('memories_used', [])]}")
    print("\nANSWER:")
    print(textwrap.indent(result["answer"][:answer_chars], "  "))
    print("\ncitations:")
    for citation in result.get("citations", [])[:8]:
        print(f"  [{citation['marker']}] {citation['source_id']} ({citation['tool_used']}) {citation['url']}")
    actions = result.get("memory_update", {}).get("actions", [])
    if actions:
        print("\nmemory written:")
        for action in actions:
            print(f"  - {action['action']}: {action.get('content', '')}")


async def main() -> None:
    services = Services()
    await services.startup()
    try:
        rule("0 · INGEST: live source -> normalised documents -> chunks -> vectors")
        stats = await services.ingestion.sync(full=True)
        print(stats.as_dict())

        question = ("Summarize what changed in Project Atlas during Q2 2026 and identify the major "
                    "risks. Also tell me what I should personally focus on.")

        rule("1 · BASIC RAG baseline (no planning, no tools, no memory)")
        baseline = await services.baseline.answer(question, USER, "demo-baseline")
        print(textwrap.indent(baseline["answer"][:1200], "  "))
        print(f"\n  tool calls: 0 · citations: 0 · chunks: {len(baseline['retrieved_chunks'])}")

        rule("2 · AGENTIC RAG, session A")
        result = await services.orchestrator.run(question=question, user_id=USER, session_id="demo-A")
        show(result)

        rule("3 · QUERY REWRITE: an elliptical follow-up is resolved before planning")
        for follow_up in ["And what about Apollo?", "why did it slip?"]:
            result = await services.orchestrator.run(
                question=follow_up, user_id=USER, session_id="demo-A")
            rewrite = result["rewrite"]
            print(f"\n> {follow_up}")
            print(f"  rewritten [{rewrite['method']}]: {rewrite['rewritten']}")
            print(f"  resolved:  {rewrite['resolved']}")
            print(f"  planned for project(s): {result['plan']['entities']['project_keys']}")

        rule("4 · MEMORY: the user states a durable priority (session A)")
        result = await services.orchestrator.run(
            question="Project Atlas is my highest priority this quarter.",
            user_id=USER, session_id="demo-A")
        print(result["answer"][:400])
        print("memory actions:", result["memory_update"]["actions"])

        rule("5 · CROSS-SESSION RECALL: new session, memory still applies")
        result = await services.orchestrator.run(
            question="Given my priorities, what should I focus on this week?",
            user_id=USER, session_id="demo-B")
        show(result, answer_chars=1200)

        rule("6 · MEMORY CONFLICT: the priority changes, the old record is superseded")
        result = await services.orchestrator.run(
            question="Actually, Project Apollo has become my highest priority now.",
            user_id=USER, session_id="demo-B")
        print("memory actions:", result["memory_update"]["actions"])
        print("\nactive memories:")
        for memory in await services.memory.list_memories(USER):
            print(f"  - [{memory['type']}] {memory['content']}")

        rule("7 · LIVE DATA: the source changes, the answer changes")
        subprocess.run([sys.executable, str(ROOT / "scripts" / "simulate_change.py"),
                        "--scenario", "unblock"], check=True)
        stats = await services.ingestion.sync()
        print(f"incremental sync: {stats.documents_upserted} documents changed, "
              f"{stats.chunks_indexed} chunks re-embedded")
        result = await services.orchestrator.run(
            question="What is still blocked in Project Atlas right now?",
            user_id=USER, session_id="demo-B")
        show(result, answer_chars=900)

        rule("8 · INSUFFICIENT EVIDENCE: the agent declines rather than inventing")
        result = await services.orchestrator.run(
            question="What is the customer churn rate for Project Atlas this quarter?",
            user_id=USER, session_id="demo-B")
        print(textwrap.indent(result["answer"][:800], "  "))
        print(f"\ninsufficient_evidence={result['insufficient_evidence']}")
    finally:
        await services.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
