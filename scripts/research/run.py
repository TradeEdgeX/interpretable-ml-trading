"""mlbot research run — execute one court's *_grid.yaml."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.research.court_run import CourtRunError, run_experiment  # noqa: E402


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Run one experiment's court grid (event_backtest, KS OFF)"
    )
    p.add_argument("experiment_id", help="Experiment directory name")
    p.add_argument(
        "--grid",
        default=None,
        help="*_grid.yaml name or path when the folder has more than one",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Remeasure even if verdict is already declared",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print argv; do not start the harness",
    )
    p.add_argument(
        "--no-refresh-index",
        action="store_true",
        help="Skip scorecard / GATE_INDEX / EXPERIMENT_INDEX rewrite",
    )
    args = p.parse_args(argv)

    grid = Path(args.grid) if args.grid else None
    try:
        return run_experiment(
            args.experiment_id,
            repo_root=PROJECT_ROOT,
            force=args.force,
            grid=grid,
            execute=not args.dry_run,
            refresh_index=not args.dry_run and not args.no_refresh_index,
        )
    except CourtRunError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
