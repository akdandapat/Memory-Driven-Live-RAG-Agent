#!/usr/bin/env python3
"""Pull changes from the live source into the local index.

    python scripts/sync.py                 # incremental, all visible projects
    python scripts/sync.py --full          # ignore watermarks and re-read everything
    python scripts/sync.py --project ATLAS # one project
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services import Services  # noqa: E402


async def main() -> int:
    parser = argparse.ArgumentParser(description="Sync the live source into the local index")
    parser.add_argument("--full", action="store_true", help="ignore watermarks")
    parser.add_argument("--project", action="append", dest="projects", help="project key (repeatable)")
    parser.add_argument("--no-comments", action="store_true", help="skip comment ingestion")
    args = parser.parse_args()

    services = Services()
    await services.startup()
    try:
        stats = await services.ingestion.sync(
            args.projects, full=args.full, include_comments=not args.no_comments
        )
        print(json.dumps(stats.as_dict(), indent=2))
        print(json.dumps(await services.ingestion.index_stats(), indent=2, default=str))
        return 1 if stats.errors else 0
    finally:
        await services.shutdown()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
