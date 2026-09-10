"""mlbot research index — build/query machine-readable experiment lineage.

Writes ``config/experiments/EXPERIMENT_INDEX.json`` so that experiment verdicts
survive context loss: an agent can ask "was this already rejected?" without
reading 300 Chinese decision documents.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from src.research.experiment_gate import merge_gate_into_index
from src.research.experiment_index import build_index, filter_rows

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_OUTPUT = "config/experiments/EXPERIMENT_INDEX.json"


def _render_table(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return "(no matching experiments)"
    header = (
        f"{'id':<46} {'class':<8} {'verdict':<11} {'src':<9} {'harness':<22} segments"
    )
    lines = [header, "-" * len(header)]
    for row in rows:
        segs = ",".join(row.get("segments") or []) or "-"
        lines.append(
            f"{str(row.get('id'))[:46]:<46} "
            f"{str(row.get('record_class') or '-'):<8} "
            f"{str(row.get('verdict') or '-'):<11} "
            f"{str(row.get('verdict_source')):<9} "
            f"{str(row.get('harness') or '-')[:22]:<22} "
            f"{segs}"
        )
    return "\n".join(lines)


def _render_summary(index: Dict[str, Any]) -> str:
    s = index["summary"]
    lines = [
        f"experiments: {s['count']}",
        "verdict: " + ", ".join(f"{k}={v}" for k, v in s["by_verdict"].items()),
        "source:  " + ", ".join(f"{k}={v}" for k, v in s["by_source"].items()),
        "class:   "
        + ", ".join(f"{k}={v}" for k, v in (s.get("by_record_class") or {}).items()),
    ]
    if s["invalid_count"]:
        lines.append(f"invalid front-matter: {s['invalid_count']}")
    gs = index.get("gate_summary") or {}
    by_gate = gs.get("by_gate_status") or {}
    if by_gate:
        lines.append("gate:    " + ", ".join(f"{k}={v}" for k, v in by_gate.items()))
    return "\n".join(lines)


def _print_issues(index: Dict[str, Any]) -> None:
    by_id = {r["id"]: r for r in index["rows"]}
    for exp_id in index["invalid"]:
        row = by_id.get(exp_id) or {}
        print(f"\n{exp_id}  ({row.get('decision_path') or 'no decision doc'})")
        for issue in row.get("issues") or []:
            print(f"  - {issue}")


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Build or query the experiment index (DECISION.md front-matter)"
    )
    p.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Index JSON path relative to repo root (default: {DEFAULT_OUTPUT})",
    )
    p.add_argument(
        "--experiments-root",
        default=None,
        help="Override config/experiments (for tests)",
    )
    p.add_argument("--stdout", action="store_true", help="Print JSON, do not write")
    p.add_argument(
        "--check",
        action="store_true",
        help="Exit 1 if any declared front-matter is invalid (CI gate)",
    )
    p.add_argument(
        "--verdict", default=None, help="Filter: promote/reject/park/needs-more"
    )
    p.add_argument("--strategy", default=None, help="Filter by strategy slug")
    p.add_argument("--tag", default=None, help="Filter by tag")
    p.add_argument(
        "--query", default=None, help="Substring match on id/topic/harness/tags"
    )
    p.add_argument(
        "--declared-only",
        action="store_true",
        help="Alias for --trusted",
    )
    p.add_argument(
        "--trusted",
        action="store_true",
        help="Only human-declared closes (agent default for 'already rejected?')",
    )
    p.add_argument(
        "--legacy",
        action="store_true",
        help="Only pre-court archive rows (not proof of reject/promote)",
    )
    p.add_argument(
        "--all-records",
        action="store_true",
        help="Do not filter by record_class (human archaeology)",
    )
    p.add_argument(
        "--record-class",
        default=None,
        help="trusted | open | legacy | example",
    )
    p.add_argument(
        "--stale",
        action="store_true",
        help="Only rows flagged evidence_stale by the last GATE_INDEX merge",
    )
    args = p.parse_args(argv)

    exp_root = Path(args.experiments_root).resolve() if args.experiments_root else None
    index = build_index(repo_root=PROJECT_ROOT, experiments_root=exp_root)
    gate_path = PROJECT_ROOT / "config" / "experiments" / "GATE_INDEX.json"
    if gate_path.is_file():
        try:
            merge_gate_into_index(
                index, json.loads(gate_path.read_text(encoding="utf-8"))
            )
        except (OSError, json.JSONDecodeError):
            pass

    querying = any(
        [
            args.verdict,
            args.strategy,
            args.tag,
            args.query,
            args.declared_only,
            args.trusted,
            args.legacy,
            args.record_class,
            args.stale,
        ]
    )

    if querying:
        record_classes = None
        if args.record_class:
            record_classes = [args.record_class]
        elif args.legacy:
            record_classes = ["legacy"]
        elif args.trusted or args.declared_only:
            record_classes = ["trusted"]
        elif not args.all_records:
            # Agent default: do not surface inferred archive as if it were a close.
            record_classes = ["trusted", "open"]
        rows = filter_rows(
            index["rows"],
            verdict=args.verdict,
            strategy=args.strategy,
            tag=args.tag,
            query=args.query,
            record_classes=record_classes,
        )
        if args.stale:
            rows = [r for r in rows if r.get("evidence_stale")]
        if args.stdout:
            print(json.dumps(rows, indent=2, ensure_ascii=False))
        else:
            print(_render_table(rows))
            print(f"\n{len(rows)} / {index['summary']['count']} experiments")
        return 0

    if args.stdout:
        print(json.dumps(index, indent=2, ensure_ascii=False))
    else:
        out = PROJECT_ROOT / args.output
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(f"wrote {args.output}")
        print(_render_summary(index))

    if index["summary"]["invalid_count"]:
        _print_issues(index)
        if args.check:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
