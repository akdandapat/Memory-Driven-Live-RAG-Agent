#!/usr/bin/env python3
"""Mutate the DEMO dataset so you can watch live-data behaviour.

The demo connector reads these JSON files on every call, so a change here is a change in the
source of truth - exactly like someone editing a Jira ticket. Run a query, run this script,
run /api/sync, then ask again and the answer moves.

    python scripts/simulate_change.py --comment ATLAS-9 "Verityx confirmed 4.2 for 2026-08-14."
    python scripts/simulate_change.py --status ATLAS-9 "In Progress"
    python scripts/simulate_change.py --duedate ATLAS-1 2026-08-21
    python scripts/simulate_change.py --scenario unblock
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import get_settings  # noqa: E402


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def save(path: Path, data: list[dict]) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def find(issues: list[dict], key: str) -> dict:
    for issue in issues:
        if issue["key"].upper() == key.upper():
            return issue
    raise SystemExit(f"Issue {key} is not in the demo dataset")


def next_id(issue: dict, field: str, base: int) -> str:
    existing = [int(x["id"]) for x in issue.get(field, []) if str(x.get("id", "")).isdigit()]
    return str(max(existing + [base]) + 1)


def add_comment(issue: dict, body: str, author: str) -> None:
    issue.setdefault("comments", []).append({
        "id": next_id(issue, "comments", 90000), "author": author,
        "created": now_iso(), "body": body,
    })
    issue["updated"] = now_iso()


def change_field(issue: dict, field: str, to_value: str, author: str) -> None:
    from_value = issue.get(field)
    if field == "status":
        issue["status"] = to_value
        issue["status_category"] = {
            "Done": "Done", "To Do": "To Do", "Blocked": "In Progress",
        }.get(to_value, "In Progress")
        if to_value == "Done":
            issue["resolutiondate"] = now_iso()
    elif field == "duedate":
        issue["duedate"] = f"{to_value}T00:00:00Z"
        to_value = to_value
        from_value = (from_value or "")[:10] or None
    else:
        issue[field] = to_value
    issue.setdefault("changelog", []).append({
        "id": next_id(issue, "changelog", 90000), "author": author, "created": now_iso(),
        "items": [{"field": field, "from": from_value if isinstance(from_value, str) else None,
                   "to": to_value}],
    })
    issue["updated"] = now_iso()


def main() -> None:
    parser = argparse.ArgumentParser(description="Mutate the demo dataset")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--author", default="Priya Raman")
    parser.add_argument("--comment", nargs=2, metavar=("ISSUE", "BODY"))
    parser.add_argument("--status", nargs=2, metavar=("ISSUE", "STATUS"))
    parser.add_argument("--duedate", nargs=2, metavar=("ISSUE", "YYYY-MM-DD"))
    parser.add_argument("--priority", nargs=2, metavar=("ISSUE", "PRIORITY"))
    parser.add_argument("--scenario", choices=["unblock", "new_blocker"],
                        help="apply a prepared multi-field change")
    args = parser.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else get_settings().mock_dir
    path = data_dir / "issues.json"
    issues = load(path)
    applied: list[str] = []

    if args.comment:
        issue = find(issues, args.comment[0])
        add_comment(issue, args.comment[1], args.author)
        applied.append(f"comment added to {issue['key']}")
    if args.status:
        issue = find(issues, args.status[0])
        change_field(issue, "status", args.status[1], args.author)
        applied.append(f"{issue['key']} status -> {args.status[1]}")
    if args.duedate:
        issue = find(issues, args.duedate[0])
        change_field(issue, "duedate", args.duedate[1], args.author)
        applied.append(f"{issue['key']} due date -> {args.duedate[1]}")
    if args.priority:
        issue = find(issues, args.priority[0])
        change_field(issue, "priority", args.priority[1], args.author)
        applied.append(f"{issue['key']} priority -> {args.priority[1]}")

    if args.scenario == "unblock":
        vendor = find(issues, "ATLAS-11")
        change_field(vendor, "status", "Done", args.author)
        add_comment(vendor, "Verityx shipped SDK 4.2 today. Duplicate settlement callbacks are fixed "
                            "in their sandbox and we have validated it end to end.", "Priya Raman")
        blocked = find(issues, "ATLAS-9")
        change_field(blocked, "status", "In Progress", args.author)
        add_comment(blocked, "Unblocked now that 4.2 is out. Re-running the settlement contract tests.",
                    "Marcus Feld")
        applied.append("scenario 'unblock': ATLAS-11 done, ATLAS-9 back in progress")
    elif args.scenario == "new_blocker":
        issue = find(issues, "ATLAS-15")
        change_field(issue, "status", "Blocked", args.author)
        issue.setdefault("labels", [])
        if "blocked" not in issue["labels"]:
            issue["labels"].append("blocked")
        add_comment(issue, "Load test environment is down after the datacentre migration. No ETA from "
                           "infrastructure, so the 2x peak run cannot be scheduled.", "Aisha Bello")
        applied.append("scenario 'new_blocker': ATLAS-15 blocked on the load test environment")

    if not applied:
        parser.error("nothing to do - pass --comment/--status/--duedate/--priority/--scenario")

    save(path, issues)
    print("Applied:")
    for line in applied:
        print(f"  - {line}")
    print("\nNow run:  curl -XPOST localhost:8000/api/sync -H 'content-type: application/json' -d '{}'")
    print("or press 'Sync source' in the UI, then ask the question again.")


if __name__ == "__main__":
    main()
