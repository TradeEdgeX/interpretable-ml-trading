#!/usr/bin/env python3
"""Run the locked A-share momentum + amount cross-section.

Usage:
    PYTHONPATH=src python scripts/research/cs_panel.py
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

from src.research.cs_panel import run_cs_panel  # noqa: E402


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
    p.add_argument("--segments", default="config/market_segment_ashare.yaml")
    p.add_argument(
        "--out",
        default="results/ashare_cs_mom_amount/experiments/20260911_ashare_cs_mom_amount",
    )
    args = p.parse_args(argv)

    result = run_cs_panel(
        daily_dir=_REPO / args.daily_dir,
        basic_path=_REPO / args.basic,
        segments_path=_REPO / args.segments,
    )
    out = _REPO / args.out
    out.mkdir(parents=True, exist_ok=True)
    ic = result["ic"]
    kpis = result["kpis"]
    books = result["books"]
    ic.to_csv(out / "ic_by_segment.csv", index=False)
    if not kpis.empty:
        kpis.to_csv(out / "book_kpis.csv", index=False)
    if not books.empty:
        books.to_parquet(out / "daily_books.parquet")
    (out / "coverage.json").write_text(
        json.dumps(result["coverage"], indent=2) + "\n", encoding="utf-8"
    )
    reports = {}
    if not kpis.empty:
        for _, r in kpis.iterrows():
            key = f"{r['group']}_{r['segment']}"
            reports[key] = {
                "cagr": None if pd.isna(r["cagr"]) else float(r["cagr"]),
                "calmar": None if pd.isna(r["calmar"]) else float(r["calmar"]),
                "win_rate": None if pd.isna(r["win_rate"]) else float(r["win_rate"]),
                "maxdd": None if pd.isna(r["maxdd"]) else float(r["maxdd"]),
                "sharpe": None if pd.isna(r["sharpe"]) else float(r["sharpe"]),
            }
    (out / "result.json").write_text(
        json.dumps(reports, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# cs_panel · mom_20 + amount_z_20 · 前 20% vs 等权",
        "",
        "闭棒：收盘打分，下一根开盘进。IC 是探照灯。",
        "",
        "## IC",
        "",
        "| 列 | 段 | 日均 IC | ICIR | 天数 |",
        "|---|---|---:|---:|---:|",
    ]
    if not ic.empty:
        for _, r in ic.iterrows():
            lines.append(
                f"| {r['feature']} | {r['segment']} | {_num(r['ic_mean'], 4)} | "
                f"{_num(r['ic_ir'], 2)} | {int(r['n_days'])} |"
            )
    lines += [
        "",
        "## 书",
        "",
        "| 组 | 段 | 年化 | Calmar | 胜率 | MaxDD | Sharpe | 天数 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    if not kpis.empty:
        show = kpis.sort_values(["segment", "group"])
        for _, r in show.iterrows():
            wr = f"{100.0 * r['win_rate']:.1f}%" if pd.notna(r["win_rate"]) else "—"
            lines.append(
                f"| {r['group']} | {r['segment']} | {_pct(r['cagr'])} | "
                f"{_num(r['calmar'])} | {wr} | {_pct(r['maxdd'])} | "
                f"{_num(r['sharpe'])} | {int(r['n_days'])} |"
            )
    (out / "CS_PANEL.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
    print(f"\nwrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
