"""mlbot research harness — show / check family → eval harness mapping."""

from __future__ import annotations

import argparse
import sys
from typing import List

from src.research.harness_registry import (
    event_backtest_reject_reason,
    list_specs,
    spec_for,
)


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Show or check the family → eval harness contract"
    )
    p.add_argument("family", nargs="?", help="Strategy family (srb, rolling_trend, …)")
    p.add_argument(
        "--check-event-backtest",
        action="store_true",
        help="Exit 2 if FAMILY must not use event_backtest",
    )
    args = p.parse_args(argv)

    if not args.family:
        print(f"{'family':<24} {'harness':<28} forbid_EB")
        print("-" * 64)
        for spec in list_specs():
            flag = "yes" if spec.forbid_event_backtest else "no"
            print(f"{spec.family:<24} {spec.harness:<28} {flag}")
        return 0

    spec = spec_for(args.family)
    if not spec:
        print(f"unknown family: {args.family}", file=sys.stderr)
        known = ", ".join(s.family for s in list_specs()) or "(none)"
        print(f"(no required harness; known: {known})", file=sys.stderr)
        return 3

    print(f"family:   {spec.family}")
    print(f"harness:  {spec.harness}")
    print(f"entry:    {spec.entry}")
    if spec.notes:
        print(f"notes:    {spec.notes}")

    if args.check_event_backtest:
        reason = event_backtest_reject_reason([args.family])
        if reason:
            print(reason, file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
