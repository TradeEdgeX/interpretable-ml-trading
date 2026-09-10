"""mlbot research stale — list promote evidence whose FeatureStore layer is newer.

Does not write verdict. Exit 1 with --check when any promote row is stale.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.research.evidence_staleness import scan_promote_staleness  # noqa: E402


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Flag promote rows whose FeatureStore layer is newer than the experiment date"
    )
    p.add_argument(
        "--experiments-root",
        default=None,
        help="Override config/experiments (for tests)",
    )
    p.add_argument(
        "--store-root",
        default=None,
        help="Override feature_store (for tests)",
    )
    p.add_argument(
        "--all-verdicts",
        action="store_true",
        help="Score reject/park/needs-more too (default: promote only)",
    )
    p.add_argument("--stdout", action="store_true", help="Print JSON")
    p.add_argument(
        "--check",
        action="store_true",
        help="Exit 1 if any scored promote row is stale",
    )
    args = p.parse_args(argv)

    exp_root = Path(args.experiments_root).resolve() if args.experiments_root else None
    store = Path(args.store_root).resolve() if args.store_root else None
    verdicts = (
        ("promote", "reject", "park", "needs-more")
        if args.all_verdicts
        else ("promote",)
    )
    rows = scan_promote_staleness(
        repo_root=PROJECT_ROOT,
        experiments_root=exp_root,
        store_root=store,
        verdicts=verdicts,
    )
    stale = [r for r in rows if r.stale]
    unknown = [r for r in rows if r.status == "unknown"]

    if args.stdout:
        print(json.dumps([r.as_dict() for r in rows], indent=2, ensure_ascii=False))
        return 1 if args.check and stale else 0

    print(f"scored: {len(rows)}  stale: {len(stale)}  unknown: {len(unknown)}")
    if not rows:
        return 0
    header = f"{'id':<46} {'verdict':<9} {'status':<8} layer"
    print(header)
    print("-" * len(header))
    for row in rows:
        mark = row.status
        layer = row.layer or "-"
        extra = f"  ({row.layer_date})" if row.layer_date else ""
        print(
            f"{row.experiment_id[:46]:<46} "
            f"{(row.verdict or '-'):<9} "
            f"{mark:<8} "
            f"{layer}{extra}"
        )
        if row.stale:
            for reason in row.reasons:
                print(f"  - {reason}")
    return 1 if args.check and stale else 0


if __name__ == "__main__":
    sys.exit(main())
