#!/usr/bin/env python3
"""Oracle labeled-bear short vs closed-bar MA200 short on 000300.

Usage:
    PYTHONPATH=src python scripts/research/bear_short_index.py
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

from src.research.cohort_hold import book_kpis, load_segments  # noqa: E402
from src.research.cs_panel import COURT_SEGMENTS, _equity, _win_rate, attach_factors  # noqa: E402


def _kpis(rets: pd.Series) -> dict:
    r = pd.to_numeric(rets, errors="coerce").dropna()
    if r.empty:
        return {"cagr": math.nan, "calmar": math.nan, "win_rate": math.nan,
                "maxdd": math.nan, "sharpe": math.nan, "n": 0}
    out = book_kpis(_equity(r))
    out["win_rate"] = _win_rate(r)
    out["n"] = int(len(r))
    return out


def main() -> int:
    tab = pd.read_parquet(_REPO / "data/ashare/daily/000300.SH.parquet")
    idx = pd.DatetimeIndex(pd.to_datetime(tab["date"])).normalize()
    px = pd.DataFrame(
        {
            "open": pd.to_numeric(tab["open"], errors="coerce").to_numpy(),
            "close": pd.to_numeric(tab["close"], errors="coerce").to_numpy(),
            "amount": pd.to_numeric(tab["amount"], errors="coerce").to_numpy(),
        },
        index=idx,
    )
    px = px[~px.index.duplicated(keep="last")].sort_index()
    att = attach_factors(px[(px["open"] > 0) & (px["close"] > 0)])
    att["ma200"] = att["close"].rolling(200, min_periods=200).mean()
    att["short_ma"] = att["close"] < att["ma200"]
    books = pd.read_parquet(
        _REPO / "results/ashare_cs_mom_amount/experiments/"
        "20260911_ashare_cs_mom_amount/daily_books.parquet"
    )
    aligned = books.join(att[["book_ret", "short_ma"]], how="inner")
    aligned = aligned.rename(columns={"book_ret": "idx_ret"})
    segs = [s for s in load_segments(_REPO / "config/market_segment_ashare.yaml")
            if s["id"] in COURT_SEGMENTS]
    rows = []
    for s in segs:
        sl = aligned.loc[(aligned.index >= s["start"]) & (aligned.index <= s["end"])]
        packs = {
            "oracle_short": -sl["idx_ret"],
            "ma200_short_else_flat": (-sl["idx_ret"]).where(sl["short_ma"], 0.0),
            "cs_plus_oracle_short": (
                sl["long_ret"] - sl["idx_ret"] if s["id"] == "bear_2021" else sl["long_ret"]
            ),
            "ma200_short_else_cs": (-sl["idx_ret"]).where(sl["short_ma"], sl["long_ret"]),
        }
        for group, rets in packs.items():
            row = _kpis(rets)
            row["group"] = group
            row["segment"] = s["id"]
            rows.append(row)
    table = pd.DataFrame(rows)
    out = _REPO / "results/ashare_bear_short/experiments/20260911_ashare_bear_short_index"
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "book_kpis.csv", index=False)
    (out / "result.json").write_text(
        json.dumps(table.to_dict(orient="records"), indent=2) + "\n", encoding="utf-8"
    )
    print(table.to_string(index=False))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
