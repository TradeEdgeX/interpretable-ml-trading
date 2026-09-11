"""Closed-bar 000300 regime jump: MA200-down short, +3% close flips long."""

from __future__ import annotations

from typing import List

import pandas as pd

from src.research.cohort_hold import book_kpis, load_segments
from src.research.cs_panel import COURT_SEGMENTS, _equity, _win_rate, attach_factors

JUMP = 0.03
MA_FAST = 20
MA_SLOW = 200


def regime_positions(
    close: pd.Series,
    *,
    ma_slow: pd.Series,
    ma_fast: pd.Series,
    jump: float = JUMP,
) -> pd.Series:
    """Position decided at T close for the next open (book_ret[T]). +1 / 0 / −1."""
    close = pd.to_numeric(close, errors="coerce")
    day_ret = close / close.shift(1) - 1.0
    pos: List[float] = []
    mode = "short_rule"
    for i in range(len(close)):
        c = close.iloc[i]
        slow = ma_slow.iloc[i]
        fast = ma_fast.iloc[i]
        r = day_ret.iloc[i]
        if pd.isna(c) or pd.isna(slow):
            pos.append(0.0)
            continue
        if mode == "long":
            if pd.notna(fast) and c < fast:
                mode = "short_rule"
        if mode == "short_rule" and pd.notna(r) and r >= jump:
            mode = "long"
        if mode == "long":
            pos.append(1.0)
        else:
            pos.append(-1.0 if c < slow else 0.0)
    return pd.Series(pos, index=close.index, dtype=float)


def load_index(path) -> pd.DataFrame:
    tab = pd.read_parquet(path)
    idx = pd.DatetimeIndex(pd.to_datetime(tab["date"])).normalize()
    px = pd.DataFrame(
        {
            "open": pd.to_numeric(tab["open"], errors="coerce").to_numpy(),
            "close": pd.to_numeric(tab["close"], errors="coerce").to_numpy(),
            "high": pd.to_numeric(tab.get("high", tab["close"]), errors="coerce").to_numpy(),
            "low": pd.to_numeric(tab.get("low", tab["close"]), errors="coerce").to_numpy(),
            "amount": pd.to_numeric(tab.get("amount", tab["close"]), errors="coerce").to_numpy(),
        },
        index=idx,
    )
    px = px[~px.index.duplicated(keep="last")].sort_index()
    return px[(px["open"] > 0) & (px["close"] > 0)]


def run_regime_jump(*, index_path, segments_path) -> dict:
    px = load_index(index_path)
    att = attach_factors(px)
    att["ma200"] = att["close"].rolling(MA_SLOW, min_periods=MA_SLOW).mean()
    att["ma20"] = att["close"].rolling(MA_FAST, min_periods=MA_FAST).mean()
    att["pos"] = regime_positions(att["close"], ma_slow=att["ma200"], ma_fast=att["ma20"])
    att["book"] = att["pos"] * att["book_ret"]
    books = att[["book", "pos", "book_ret"]].dropna(subset=["book"])
    segments = load_segments(segments_path)
    rows = []
    for seg in segments:
        if seg["id"] not in COURT_SEGMENTS:
            continue
        sl = books.loc[(books.index >= seg["start"]) & (books.index <= seg["end"])]
        if sl.empty:
            continue
        kpi = book_kpis(_equity(sl["book"]))
        kpi["win_rate"] = _win_rate(sl["book"])
        kpi["group"] = "regime_jump"
        kpi["segment"] = seg["id"]
        kpi["n_days"] = int(len(sl))
        kpi["long_share"] = float((sl["pos"] > 0).mean())
        kpi["short_share"] = float((sl["pos"] < 0).mean())
        rows.append(kpi)
    return {"books": books, "kpis": pd.DataFrame(rows)}
