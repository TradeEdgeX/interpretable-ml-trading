#!/usr/bin/env python3
"""Run the A-share small-cap vs large-cap hold panel.

Usage:
    PYTHONPATH=src python scripts/research/cohort_hold.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.research.cohort_hold import run_cohort, summarize  # noqa: E402


def _pct(x) -> str:
    if x is None or (isinstance(x, float) and (pd.isna(x))):
        return "无样本"
    return f"{100.0 * float(x):+.2f}%"


def _num(x, digits: int = 2) -> str:
    if x is None or (isinstance(x, float) and (pd.isna(x))):
        return "—"
    return f"{float(x):.{digits}f}"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--daily-dir", default="data/ashare/daily")
    p.add_argument("--basic", default="data/ashare/stock_basic/stock_basic.parquet")
    p.add_argument("--universe-dir", default="data/ashare/universe_hist")
    p.add_argument("--segments", default="config/market_segment_ashare.yaml")
    p.add_argument("--mcap-yi", type=float, default=100.0)
    p.add_argument("--hold-years", default="3,4")
    p.add_argument(
        "--out",
        default="results/tenbagger_smallcap/experiments/20260911_tenbagger_smallcap",
    )
    args = p.parse_args(argv)

    years = tuple(float(x) for x in str(args.hold_years).split(",") if x.strip())
    result = run_cohort(
        daily_dir=_REPO / args.daily_dir,
        basic_path=_REPO / args.basic,
        universe_dir=_REPO / args.universe_dir,
        segments_path=_REPO / args.segments,
        mcap_yi=float(args.mcap_yi),
        hold_years=years,
    )
    table = summarize(result)
    out = _REPO / args.out
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "cohort_kpis.csv", index=False)
    if result.holds:
        pd.DataFrame([h.__dict__ for h in result.holds]).to_parquet(
            out / "name_holds.parquet", index=False
        )
    (out / "coverage.json").write_text(
        json.dumps(result.coverage, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# cohort_hold · 入场日流通市值 ≤ 阈值 vs 对照 · 固定持有年数",
        "",
        f"阈值 {args.mcap_yi:g} 亿元。市值 = 当日成交额 / (换手率/100)。"
        "退市用最后收盘。未完成持有的入场季度不进表。",
        "",
        "| 组 | 段 | 持有 | 年化 | Calmar | 胜率 | MaxDD | Sharpe | n | 十倍率 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    if not table.empty:
        show = table.sort_values(["hold_years", "group", "segment"])
        for _, r in show.iterrows():
            wr = f"{100.0 * r['win_rate']:.1f}%" if pd.notna(r["win_rate"]) else "—"
            h10 = f"{100.0 * r['hit_10x']:.2f}%" if pd.notna(r["hit_10x"]) else "—"
            lines.append(
                f"| {r['group']} | {r['segment']} | {r['hold_years']:g}y | "
                f"{_pct(r['cagr'])} | {_num(r['calmar'])} | {wr} | "
                f"{_pct(r['maxdd'])} | {_num(r['sharpe'])} | {int(r['n'])} | {h10} |"
            )
    (out / "COHORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
    print(f"\nwrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
