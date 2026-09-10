"""mlbot research replay — new court dir for one archive experiment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

from src.research.court_run import CourtRunError, run_experiment
from src.research.replay import scaffold_replay

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Scaffold a new court that replays one archive experiment"
    )
    p.add_argument("experiment_id", help="Parent archive folder name")
    p.add_argument(
        "--run",
        action="store_true",
        help="Run the new court's *_grid.yaml (one id; event_backtest KS OFF)",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Allow --run when the new court already has a verdict",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the event_backtest argv; do not start the harness",
    )
    args = p.parse_args(argv)

    dest = scaffold_replay(args.experiment_id, repo_root=PROJECT_ROOT)
    print(f"replay court: {dest.relative_to(PROJECT_ROOT)}")
    if not args.run:
        print("not started. pass --run to execute this court's *_grid.yaml.")
        return 0
    print("running court grid…")
    try:
        return run_experiment(
            dest.name,
            repo_root=PROJECT_ROOT,
            force=args.force,
            execute=not args.dry_run,
            refresh_index=not args.dry_run,
        )
    except CourtRunError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
