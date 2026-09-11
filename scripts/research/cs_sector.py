#!/usr/bin/env python3
"""Run sector-rank CS + 10bp cost.

Usage:
    PYTHONPATH=src python scripts/research/cs_sector.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.research.cs_sector import run_cs_sector  # noqa: E402


def _pct(x) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{100.0 * float(x):+.2f}%"


def _num(x, digits: int = 2) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{float(x):.{digits}f}"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--daily-dir", default="data/ashare/daily")
    p.add_argument("--basic", default="data/ashare/stock_basic/stock_basic.parquet")
    p.add_argument("--industry", default="config/industry_map_ashare.yaml")
    p.add_argument("--segments", default="config/market_segment_ashare.yaml")
    p.add_argument("--mode", choices=("daily", "ls", "weekly"), default="daily")
    p.add_argument(
        "--out",
        default="results/ashare_cs_sector/experiments/20260911_ashare_cs_sector_cost",
    )
    args = p.parse_args(argv)
    result = run_cs_sector(
        daily_dir=_REPO / args.daily_dir,
        basic_path=_REPO / args.basic,
        industry_path=_REPO / args.industry,
        segments_path=_REPO / args.segments,
        mode=args.mode,
    )
    out = _REPO / args.out
    out.mkdir(parents=True, exist_ok=True)
    kpis = result["kpis"]
    books = result["books"]
    if not kpis.empty:
        kpis.to_csv(out / "book_kpis.csv", index=False)
    if not books.empty:
        books.to_parquet(out / "daily_books.parquet")
    (out / "coverage.json").write_text(
        json.dumps(result["coverage"], indent=2) + "\n", encoding="utf-8"
    )
    (out / "result.json").write_text(
        json.dumps(kpis.to_dict(orient="records") if not kpis.empty else [], indent=2)
        + "\n",
        encoding="utf-8",
    )
    lines = [
        "# cs_sector · 板块排序前 20% vs 同宇宙等权 · 单边 10bp",
        "",
        "行业是 `config/industry_map_ashare.yaml` 20 档粗分快照，不是申万一级、不是时点修订。闭棒：收盘打分，下一根开盘。",
        "",
        "| 组 | 段 | 年化 | Calmar | 胜率 | MaxDD | Sharpe | 天数 | 日均换手 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    if not kpis.empty:
        for _, r in kpis.sort_values(["segment", "group"]).iterrows():
            wr = f"{100.0 * r['win_rate']:.1f}%" if pd.notna(r["win_rate"]) else "—"
            lines.append(
                f"| {r['group']} | {r['segment']} | {_pct(r['cagr'])} | "
                f"{_num(r['calmar'])} | {wr} | {_pct(r['maxdd'])} | "
                f"{_num(r['sharpe'])} | {int(r['n_days'])} | {_num(r['to_mean'], 3)} |"
            )
    text = "\n".join(lines)
    (out / "CS_SECTOR.md").write_text(text + "\n", encoding="utf-8")
    print(text, flush=True)
    print(f"\nwrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
