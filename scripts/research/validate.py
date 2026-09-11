"""mlbot research validate — structural template gate. Not a verdict."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.research.experiment_index import find_decision_file
from src.research.hypothesis_template import validate_decision_path


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Check a DECISION.md hypothesis template. Does not declare truth."
    )
    p.add_argument("experiment_id", help="Folder name under config/experiments/")
    p.add_argument(
        "--json",
        action="store_true",
        help="Print the report as JSON",
    )
    args = p.parse_args(argv)
    exp = PROJECT_ROOT / "config" / "experiments" / args.experiment_id
    decision = find_decision_file(exp) if exp.is_dir() else None
    if decision is None and exp.is_file():
        decision = exp
    if decision is None:
        print(f"ERROR: no DECISION.md for {args.experiment_id}", file=sys.stderr)
        return 3
    report = validate_decision_path(decision)
    if args.json:
        print(json.dumps(report.as_dict(), ensure_ascii=False, indent=2))
    else:
        print(f"template {'OK' if report.ok else 'INCOMPLETE'}  {decision}")
        if report.missing:
            print("missing: " + ", ".join(report.missing))
        for issue in report.issues:
            print(f"issue: {issue}")
        if report.ok:
            print("may orchestrate (still need a human 'measure this' to backtest)")
        else:
            print("fill the template before data / FeatureStore / event_backtest")
    return 0 if report.ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
