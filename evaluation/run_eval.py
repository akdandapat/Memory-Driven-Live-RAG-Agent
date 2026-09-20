#!/usr/bin/env python3
"""Run the evaluation suite against a live instance of the system.

    python evaluation/run_eval.py                    # all cases
    python evaluation/run_eval.py --category memory_rag
    python evaluation/run_eval.py --out evaluation/results/run.json

Cases run in dataset order and share a user, because the memory cases deliberately depend on
earlier turns (write -> recall in a new session -> conflict). Every metric printed is measured
from the run; none of them are hard-coded.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluation.metrics import aggregate, evaluate_case  # noqa: E402
from app.services import Services  # noqa: E402

DATASET = Path(__file__).with_name("dataset.jsonl")
EVAL_USER = "eval-user"


def load_cases(path: Path, category: str | None) -> list[dict]:
    cases = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [c for c in cases if not category or c["category"] == category]


async def memory_quality(services, expected_terms: list[str]) -> tuple[float | None, float | None]:
    """Precision = stored records that were expected. Recall = expected terms that got stored."""
    records = await services.memory.list_memories(EVAL_USER)
    if not records:
        return (None, 0.0 if expected_terms else None)
    contents = [r["content"].lower() for r in records]
    relevant = sum(1 for c in contents if any(term.lower() in c for term in expected_terms))
    precision = round(relevant / len(contents), 3)
    covered = sum(1 for term in expected_terms if any(term.lower() in c for c in contents))
    recall = round(covered / len(expected_terms), 3) if expected_terms else None
    return precision, recall


async def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the agent")
    parser.add_argument("--dataset", default=str(DATASET))
    parser.add_argument("--category", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    cases = load_cases(Path(args.dataset), args.category)
    if not cases:
        print("No cases matched.")
        return 1

    services = Services()
    await services.startup()
    try:
        print(f"mode={services.settings.app_mode}  "
              f"llm={services.settings.llm_model if services.llm else 'none (heuristic)'}  "
              f"embeddings={services.embeddings.name}")
        stats = await services.ingestion.sync(full=True)
        print(f"indexed {stats.documents_upserted} documents / {stats.chunks_indexed} chunks\n")

        results = []
        session = "eval-session-1"
        expected_memory_terms: list[str] = []

        for index, case in enumerate(cases, start=1):
            if case.get("new_session"):
                session = f"eval-session-{index}"
            result = await services.orchestrator.run(
                question=case["question"], user_id=EVAL_USER, session_id=session
            )
            outcome = evaluate_case(case, result)
            results.append(outcome)
            expected_memory_terms.extend(case.get("expected_memory_terms", []))

            status = "PASS" if outcome.passed else "FAIL"
            plan_quality = "-" if outcome.plan_quality is None else f"{outcome.plan_quality:.2f}"
            print(f"[{status}] {case['id']:<26} {case['category']:<14} "
                  f"tools={outcome.tool_calls:<2} evidence={outcome.evidence:<3} "
                  f"cites={outcome.citation_count:<2} plan={plan_quality} "
                  f"{outcome.latency_ms:>5}ms")
            for failure in outcome.failures:
                print(f"         - {failure}")
            if args.verbose:
                print("         answer:", result["answer"][:300].replace("\n", " "))

        # Only the most recent priority statement should still be active.
        precision, recall = await memory_quality(services, ["Apollo", "priority"])
        summary = aggregate(results, precision, recall)

        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        for key, value in summary.items():
            if key == "by_category":
                continue
            print(f"  {key:<28} {value}")
        print("\n  by category:")
        for category, bucket in summary["by_category"].items():
            print(f"    {category:<16} {bucket['passed']}/{bucket['cases']} "
                  f"(pass rate {bucket['pass_rate']})")

        if args.out:
            out_path = Path(args.out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(
                {"summary": summary, "cases": [r.as_dict() for r in results]}, indent=2
            ), encoding="utf-8")
            print(f"\nwrote {out_path}")

        return 0 if summary["passed"] == summary["cases"] else 1
    finally:
        await services.shutdown()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
