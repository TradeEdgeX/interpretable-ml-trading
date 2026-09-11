#!/usr/bin/env python3
"""000300 MA200-down short; +3% close flips long; MA20 exits long.

Usage:
    PYTHONPATH=src python scripts/research/regime_jump.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.research.regime_jump import run_regime_jump  # noqa: E402


def _pct(x) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{100.0 * float(x):+.2f}%"


def _num(x, digits: int = 2) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{float(x):.{digits}f}"


def main() -> int:
    result = run_regime_jump(
        index_path=_REPO / "data/ashare/daily/000300.SH.parquet",
        segments_path=_REPO / "config/market_segment_ashare.yaml",
    )
    out = _REPO / "results/ashare_regime_jump/experiments/20260911_ashare_regime_jump"
    out.mkdir(parents=True, exist_ok=True)
    kpis = result["kpis"]
    books = result["books"]
    if not kpis.empty:
        kpis.to_csv(out / "book_kpis.csv", index=False)
    if not books.empty:
        books.to_parquet(out / "daily_books.parquet")
    (out / "result.json").write_text(
        json.dumps(kpis.to_dict(orient="records") if not kpis.empty else [], indent=2)
        + "\n",
        encoding="utf-8",
    )
    lines = [
        "# regime_jump · 000300 · MA200 下空 / 收盘+3% 改多 / 跌破 MA20 退出",
        "",
        "闭棒。日历标签 bull_924 不是信号。",
        "",
        "| 段 | 年化 | Calmar | 胜率 | MaxDD | Sharpe | 天数 | 多头占比 | 空头占比 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    if not kpis.empty:
        for _, r in kpis.sort_values("segment").iterrows():
            wr = f"{100.0 * r['win_rate']:.1f}%" if pd.notna(r["win_rate"]) else "—"
            lines.append(
                f"| {r['segment']} | {_pct(r['cagr'])} | {_num(r['calmar'])} | "
                f"{wr} | {_pct(r['maxdd'])} | {_num(r['sharpe'])} | "
                f"{int(r['n_days'])} | {_num(r['long_share'], 3)} | "
                f"{_num(r['short_share'], 3)} |"
            )
    text = "\n".join(lines)
    (out / "REGIME_JUMP.md").write_text(text + "\n", encoding="utf-8")
    print(text, flush=True)
    print(f"\nwrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
