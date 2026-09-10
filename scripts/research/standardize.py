"""mlbot research standardize — scaffold front-matter on legacy DECISION.md.

Does not re-run backtests. Does not write verdict.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.research.standardize_experiments import apply_scaffold, plan_all


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Scaffold English front-matter (empty verdict) onto legacy experiments"
    )
    p.add_argument(
        "--experiments-root",
        default=None,
        help="Override config/experiments (tests)",
    )
    p.add_argument(
        "--write",
        action="store_true",
        help="Write files (default is dry-run)",
    )
    p.add_argument("--stdout", action="store_true", help="Print JSON plan")
    args = p.parse_args(argv)

    root = (
        Path(args.experiments_root).resolve()
        if args.experiments_root
        else PROJECT_ROOT / "config" / "experiments"
    )
    plans = plan_all(root)
    counts = Counter(p.action for p in plans)

    if args.stdout:
        print(
            json.dumps(
                {
                    "write": args.write,
                    "counts": dict(counts),
                    "plans": [
                        {
                            "id": p.experiment_id,
                            "action": p.action,
                            "reason": p.reason,
                            "fields": p.fields,
                        }
                        for p in plans
                    ],
                },
                indent=2,
            )
        )
    else:
        mode = "WRITE" if args.write else "DRY-RUN"
        print(f"{mode}  experiments={len(plans)}")
        print("actions: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
        print(
            "verdict is never auto-filled. "
            "Close experiments by hand (or agent) then `mlbot research index`."
        )

    written = 0
    for plan in plans:
        if apply_scaffold(plan, write=args.write):
            written += 1
    if args.write:
        print(f"wrote front-matter on {written} DECISION.md files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
