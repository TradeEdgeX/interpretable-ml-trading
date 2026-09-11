#!/usr/bin/env python3
"""Locked MA200 + ADX(14)>25 / mom_20<0 shorts on 000300.

Usage:
    PYTHONPATH=src python scripts/research/ma200_adx_mom.py
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
from src.research.index_filters import mom_20, wilder_adx  # noqa: E402

ADX_N = 14
ADX_MIN = 25.0


def _kpis(rets: pd.Series) -> dict:
    r = pd.to_numeric(rets, errors="coerce").dropna()
    if r.empty:
        return {
            "cagr": math.nan,
            "calmar": math.nan,
            "win_rate": math.nan,
            "maxdd": math.nan,
            "sharpe": math.nan,
            "n": 0,
        }
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
            "high": pd.to_numeric(tab["high"], errors="coerce").to_numpy(),
            "low": pd.to_numeric(tab["low"], errors="coerce").to_numpy(),
            "amount": pd.to_numeric(tab["amount"], errors="coerce").to_numpy(),
        },
        index=idx,
    )
    px = px[~px.index.duplicated(keep="last")].sort_index()
    px = px[(px["open"] > 0) & (px["close"] > 0)]
    att = attach_factors(px)
    att["ma200"] = att["close"].rolling(200, min_periods=200).mean()
    att["below_ma"] = att["close"] < att["ma200"]
    att["mom20"] = mom_20(att["close"])
    di = wilder_adx(att["high"], att["low"], att["close"], n=ADX_N)
    att = att.join(di)
    att["adx_on"] = att["adx"] > ADX_MIN
    att["adx_down"] = att["minus_di"] > att["plus_di"]
    short_ret = -att["book_ret"]
    packs = {
        "ma200_only": att["below_ma"],
        "ma200_adx25": att["below_ma"] & att["adx_on"],
        "ma200_mom20neg": att["below_ma"] & (att["mom20"] < 0),
        "adx25_di_short": att["adx_on"] & att["adx_down"],
    }
    segs = [
        s
        for s in load_segments(_REPO / "config/market_segment_ashare.yaml")
        if s["id"] in COURT_SEGMENTS
    ]
    rows = []
    for name, mask in packs.items():
        for s in segs:
            sl = att.loc[(att.index >= s["start"]) & (att.index <= s["end"])]
            m = mask.reindex(sl.index).fillna(False)
            rets = short_ret.reindex(sl.index).where(m, 0.0)
            row = _kpis(rets)
            row["group"] = name
            row["segment"] = s["id"]
            row["short_frac"] = float(m.mean()) if len(m) else math.nan
            rows.append(row)
    table = pd.DataFrame(rows)
    out = _REPO / "results/ashare_ma200_adx/experiments/20260911_ashare_ma200_adx_mom"
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
