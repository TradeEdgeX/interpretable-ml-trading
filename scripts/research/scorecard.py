"""mlbot research scorecard — harvest existing artifacts into the fixed card.

Never re-runs a backtest. Never writes verdict.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from src.research.scorecard import CENSUS_PATH, build_scorecard_census

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _render_summary(census: Dict[str, Any]) -> str:
    lines = [
        f"experiments: {census['count']}",
        "buckets: "
        + ", ".join(f"{k}={v}" for k, v in (census.get("buckets") or {}).items()),
        "slots:   "
        + ", ".join(f"{k}={v}" for k, v in (census.get("filled_slots") or {}).items()),
        f"empty legacy (review later, do not batch-delete): {len(census.get('empty_legacy') or [])}",
        f"has numbers but not trusted: {len(census.get('numbered_not_trusted') or [])}",
    ]
    for cls, buckets in (census.get("by_record_class") or {}).items():
        lines.append(f"  {cls}: " + ", ".join(f"{k}={v}" for k, v in buckets.items()))
    return "\n".join(lines)


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Harvest scorecards from on-disk artifacts (no re-run)"
    )
    p.add_argument(
        "--write",
        action="store_true",
        help=f"write {CENSUS_PATH}",
    )
    p.add_argument(
        "--output",
        default=CENSUS_PATH,
        help="census JSON path",
    )
    args = p.parse_args(argv)

    census = build_scorecard_census(repo_root=PROJECT_ROOT)
    census["generated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(_render_summary(census))
    if args.write:
        out = PROJECT_ROOT / args.output
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(census, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
