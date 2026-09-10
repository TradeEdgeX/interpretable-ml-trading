#!/usr/bin/env python3
"""Scaffold a new experiment directory under config/experiments/."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

TEMPLATE_DIR = PROJECT_ROOT / "config/experiments/_template"


def _render(text: str, *, topic: str, strategy: str, segment: str, date: str) -> str:
    return (
        text.replace("{{TOPIC}}", topic)
        .replace("{{STRATEGY}}", strategy)
        .replace("{{SEGMENT}}", segment)
        .replace("{{DATE}}", date)
    )


def init_experiment(
    topic: str,
    *,
    strategy: str = "ma_cross",
    layers: str = "regime",
    segment: str = "recent_6m_oos",
    force: bool = False,
) -> Path:
    """Create config/experiments/<topic>/ from _template/."""
    del layers  # reserved for README hints; phase1 yaml uses regime by default
    exp_dir = PROJECT_ROOT / "config/experiments" / topic
    if exp_dir.exists() and not force:
        raise FileExistsError(f"experiment dir exists: {exp_dir} (pass --force)")
    exp_dir.mkdir(parents=True, exist_ok=True)
    date = datetime.now(timezone.utc).strftime("%Y%m%d")

    copies = {
        "README.md": f"README.md",
        "DECISION.md": "DECISION.md",
        "rd_loop_phase1.yaml": f"rd_loop_{topic}_phase1.yaml",
        "promote_baseline.yaml": "promote_baseline.yaml",
    }
    for src_name, dst_name in copies.items():
        src = TEMPLATE_DIR / src_name
        if not src.is_file():
            continue
        text = _render(
            src.read_text(encoding="utf-8"),
            topic=topic,
            strategy=strategy,
            segment=segment,
            date=date,
        )
        (exp_dir / dst_name).write_text(text, encoding="utf-8")

    mb = exp_dir / "monitor_bundle"
    mb.mkdir(exist_ok=True)
    (mb / ".gitkeep").write_text("", encoding="utf-8")
    return exp_dir


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Initialize experiment directory scaffold")
    p.add_argument(
        "topic", help="Experiment directory name, e.g. 20260612_tpc_regime_baseline"
    )
    p.add_argument("--strategy", default="ma_cross")
    p.add_argument(
        "--layers", default="regime", help="Comma-separated layers (doc hint)"
    )
    p.add_argument("--segment", default="recent_6m_oos")
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)
    try:
        out = init_experiment(
            args.topic,
            strategy=args.strategy,
            layers=args.layers,
            segment=args.segment,
            force=args.force,
        )
    except FileExistsError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 3
    print(f"created {out}")
    print(f"  rd_loop: {out / f'rd_loop_{args.topic}_phase1.yaml'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
