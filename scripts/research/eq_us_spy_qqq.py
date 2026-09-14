#!/usr/bin/env python3
"""SPY/QQQ buy-and-hold vs index timing (closed-bar daily).

Usage:
    PYTHONPATH=src python scripts/research/eq_us_spy_qqq.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.research.eq_us_spy_qqq import run_court, write_court  # noqa: E402


def _pct(x) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "—"
    return f"{100.0 * float(x):+.2f}%"


def _num(x, digits: int = 2) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "—"
    return f"{float(x):.{digits}f}"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--daily-dir", default="data/eq/us/daily")
    p.add_argument("--segments", default="config/market_segment_us.yaml")
    p.add_argument("--symbols", default="SPY,QQQ")
    p.add_argument("--lock-end", default="2026-08-17")
    p.add_argument(
        "--out",
        default="results/eq_us_spy_qqq/experiments/20260914_eq_us_spy_qqq_beta",
    )
    args = p.parse_args(argv)

    symbols = tuple(s.strip().upper() for s in str(args.symbols).split(",") if s.strip())
    result = run_court(
        daily_dir=_REPO / args.daily_dir,
        segments_path=_REPO / args.segments,
        symbols=symbols,
        lock_end=str(args.lock_end),
    )
    out = write_court(result, _REPO / args.out)
    table = pd.DataFrame(result.get("rows") or [])
    keep = table[table["mode"].isin(["buy_hold", "rsi_hold_40", "rsi_to_252_high", "dd20_hold_60", "ma200"])]
    for window, chunk in keep.groupby("window", sort=False):
        print(f"\n## {window}")
        print(
            f"{'sym':<5} {'mode':<16} {'cagr':>8} {'calmar':>7} {'wr':>7} "
            f"{'maxdd':>8} {'sharpe':>7} {'in_mkt':>7}"
        )
        for _, row in chunk.iterrows():
            print(
                f"{row['symbol']:<5} {row['mode']:<16} "
                f"{_pct(row['cagr']):>8} {_num(row['calmar']):>7} "
                f"{_pct(row['win_rate']):>7} {_pct(row['maxdd']):>8} "
                f"{_num(row['sharpe']):>7} {_pct(row['time_in_market']):>7}"
            )
    print(f"\nwrote {out}")
    print("pit_stock_picking:", result.get("pit_stock_picking"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
