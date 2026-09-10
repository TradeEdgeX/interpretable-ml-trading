"""mlbot research review — paper audit of archive experiments."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from src.research.review import REVIEW_QUEUE_PATH, build_review_queue, review_experiment

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Paper-review experiments before replay")
    p.add_argument("experiment_id", nargs="?", help="One folder name")
    p.add_argument("--all", action="store_true")
    p.add_argument("--write", action="store_true", help=f"write {REVIEW_QUEUE_PATH}")
    args = p.parse_args(argv)

    if args.experiment_id:
        row = review_experiment(
            PROJECT_ROOT / "config" / "experiments" / args.experiment_id,
            repo_root=PROJECT_ROOT,
        )
        print(json.dumps(row.as_dict(), indent=2, ensure_ascii=False))
        return 0

    if not args.all and not args.write:
        p.error("pass an experiment id, or --all / --write")

    queue = build_review_queue(repo_root=PROJECT_ROOT)
    queue["generated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(
        f"experiments: {queue['count']}\n"
        f"actions: {queue['by_action']}\n"
        f"suggested_first: {len(queue['suggested_first'])}"
    )
    if args.write:
        out = PROJECT_ROOT / REVIEW_QUEUE_PATH
        out.write_text(
            json.dumps(queue, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(f"wrote {REVIEW_QUEUE_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
