"""mlbot research close — program court gate; optional human --declare.

Does not invent promote. Writes GATE_INDEX.json (program-owned).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.research.experiment_gate import (  # noqa: E402
    allowed_declare,
    build_gate_index,
    evaluate_experiment,
)
from src.research.experiment_index import (  # noqa: E402
    find_decision_file,
    load_known_segments,
    parse_front_matter,
)

DEFAULT_OUTPUT = "config/experiments/GATE_INDEX.json"


def _patch_verdict(decision: Path, verdict: str) -> None:
    text = decision.read_text(encoding="utf-8")
    raw, body = parse_front_matter(text)
    if not raw:
        raise SystemExit(f"no front-matter on {decision}; run standardize first")
    raw["verdict"] = verdict
    dumped = yaml.safe_dump(raw, sort_keys=False, allow_unicode=False)
    decision.write_text(f"---\n{dumped}---\n\n{body.lstrip()}", encoding="utf-8")


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Program court gate. Agents must not hand-write verdict."
    )
    p.add_argument(
        "experiment_id",
        nargs="?",
        help="Experiment directory name (omit with --all)",
    )
    p.add_argument("--all", action="store_true", help="Score every experiment")
    p.add_argument(
        "--experiments-root",
        default=None,
        help="Override config/experiments",
    )
    p.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"GATE_INDEX.json path (default {DEFAULT_OUTPUT})",
    )
    p.add_argument("--stdout", action="store_true")
    p.add_argument(
        "--declare",
        default=None,
        help="Human close: reject | park | needs-more | promote",
    )
    p.add_argument(
        "--yes",
        action="store_true",
        help="Required with --declare promote",
    )
    args = p.parse_args(argv)

    root = (
        Path(args.experiments_root).resolve()
        if args.experiments_root
        else PROJECT_ROOT / "config" / "experiments"
    )

    if args.all or not args.experiment_id:
        if args.declare:
            print("ERROR: --declare requires a single experiment_id", file=sys.stderr)
            return 2
        index = build_gate_index(repo_root=PROJECT_ROOT, experiments_root=root)
        if args.stdout:
            print(json.dumps(index, indent=2))
        else:
            out = PROJECT_ROOT / args.output
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
            s = index["summary"]
            print(f"wrote {out.relative_to(PROJECT_ROOT)}")
            print(f"experiments: {s['count']}")
            print(
                "gate: " + ", ".join(f"{k}={v}" for k, v in s["by_gate_status"].items())
            )
        return 0

    exp_dir = root / args.experiment_id
    if not exp_dir.is_dir():
        print(f"ERROR: missing experiment dir {exp_dir}", file=sys.stderr)
        return 2
    known = load_known_segments(PROJECT_ROOT)
    gate = evaluate_experiment(exp_dir, repo_root=PROJECT_ROOT, known_segments=known)
    print(
        f"{gate.experiment_id}  gate={gate.gate_status}  "
        f"suggested={gate.suggested_verdict or '-'}"
    )
    for reason in gate.reasons:
        print(f"  - {reason}")
    if gate.kpi:
        print("  kpi:", gate.kpi)

    if args.declare:
        err = allowed_declare(gate, args.declare, yes=args.yes)
        if err:
            print(f"ERROR: {err}", file=sys.stderr)
            return 2
        decision = find_decision_file(exp_dir)
        if not decision:
            print("ERROR: no DECISION.md", file=sys.stderr)
            return 2
        _patch_verdict(decision, args.declare.strip().lower())
        print(f"declared verdict={args.declare.strip().lower()} on {decision}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
