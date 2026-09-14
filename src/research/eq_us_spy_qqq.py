"""US SPY/QQQ daily court: buy-and-hold vs index timing.

Closed-bar: a signal at close t is first tradable on day t+1.
Cash days earn 0. Kill switch off. Five KPIs; no total R.

Stock-picking / S&P 500 PIT is not in this extract (no point-in-time universe).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from src.research.cohort_hold import book_kpis, load_segments
from src.research.cs_panel import _equity, _win_rate

BH_START = "2014-08-01"
TIMING_START = "2015-08-01"
LOCK_END = "2026-08-17"
COURT_SEGMENTS = ("us_covid_2020", "us_bear_2022", "us_bull_2023_2024", "us_recent")
RSI_N = 14
RSI_TRIGGER = 30.0
MA_N = 200
HIGH_N = 252


def rsi_wilder(close: pd.Series, n: int = RSI_N) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / n, min_periods=n, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / n, min_periods=n, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def _hold_windows(trigger: pd.Series, hold_bars: int) -> pd.Series:
    """In-market on the *next* bar after each trigger close, for ``hold_bars`` days."""
    pos = pd.Series(0.0, index=trigger.index)
    idxs = [i for i, flag in enumerate(trigger.fillna(False).tolist()) if flag]
    n = len(pos)
    for i in idxs:
        start = i + 1
        end = min(n, start + int(hold_bars))
        if start < end:
            pos.iloc[start:end] = 1.0
    return pos


def _first_breach_dd(close: pd.Series, thresh: float = -0.20) -> pd.Series:
    """First close at or below ``thresh`` from the running peak (same-bar, then shift)."""
    peak = close.cummax()
    dd = close / peak - 1.0
    hit = dd <= float(thresh)
    first = hit & ~hit.shift(1, fill_value=False)
    return first


def _hold_until_level(trigger: pd.Series, level: pd.Series, close: pd.Series) -> pd.Series:
    """Enter the next bar after trigger; flatten the bar after close recovers the entry level."""
    pos = np.zeros(len(close), dtype=float)
    trig = trigger.fillna(False).to_numpy()
    px = close.to_numpy(dtype=float)
    lvl = level.to_numpy(dtype=float)
    in_pos = False
    target = math.nan
    for i in range(len(close)):
        if in_pos:
            if i > 0 and math.isfinite(target) and px[i - 1] >= target:
                in_pos = False
            else:
                pos[i] = 1.0
        if (not in_pos) and trig[i] and i + 1 < len(close):
            raw = float(lvl[i])
            if not math.isfinite(raw):
                continue
            in_pos = True
            target = raw
    return pd.Series(pos, index=close.index)


def positions_from_close(close: pd.Series) -> Dict[str, pd.Series]:
    rsi = rsi_wilder(close)
    ma = close.rolling(MA_N, min_periods=MA_N).mean()
    high_252 = close.rolling(HIGH_N, min_periods=HIGH_N).max()
    rsi_hit = rsi <= RSI_TRIGGER
    return {
        "buy_hold": pd.Series(1.0, index=close.index),
        "rsi_hold_40": _hold_windows(rsi_hit, 40),
        "rsi_to_252_high": _hold_until_level(rsi_hit, high_252, close),
        "dd20_hold_60": _hold_windows(_first_breach_dd(close), 60),
        "ma200": (close > ma).shift(1, fill_value=False).astype(float),
    }


def _kpis(rets: pd.Series, pos: pd.Series) -> Dict[str, float]:
    aligned = rets.reindex(pos.index).fillna(0.0)
    held = pos.reindex(rets.index).fillna(0.0)
    book = aligned * held
    eq = _equity(book)
    kpi = book_kpis(eq)
    in_mkt = held > 0
    kpi["win_rate"] = _win_rate(aligned[in_mkt]) if bool(in_mkt.any()) else float("nan")
    kpi["time_in_market"] = float(in_mkt.mean()) if len(held) else float("nan")
    kpi["n_days"] = int(len(book))
    return kpi


def slice_kpis(
    rets: pd.Series,
    positions: Dict[str, pd.Series],
    start,
    end,
) -> Dict[str, Dict[str, float]]:
    sl = rets.loc[(rets.index >= start) & (rets.index <= end)]
    out: Dict[str, Dict[str, float]] = {}
    for name, pos in positions.items():
        out[name] = _kpis(sl, pos.reindex(sl.index).fillna(0.0))
    return out


def load_px(daily_dir: Path, symbol: str) -> pd.Series:
    path = daily_dir / f"{symbol}.parquet"
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    px = pd.to_numeric(df["adjclose"], errors="coerce")
    if px.isna().all():
        px = pd.to_numeric(df["close"], errors="coerce")
    out = pd.Series(px.values, index=pd.DatetimeIndex(df["date"]), name=symbol)
    return out.sort_index()


def _fmt_row(symbol: str, window: str, name: str, kpi: Dict[str, float]) -> dict:
    return {
        "symbol": symbol,
        "window": window,
        "mode": name,
        "cagr": kpi.get("cagr"),
        "calmar": kpi.get("calmar"),
        "win_rate": kpi.get("win_rate"),
        "maxdd": kpi.get("maxdd"),
        "sharpe": kpi.get("sharpe"),
        "time_in_market": kpi.get("time_in_market"),
        "n_days": kpi.get("n_days"),
    }


def run_court(
    *,
    daily_dir: Path,
    segments_path: Path,
    symbols: Sequence[str] = ("SPY", "QQQ"),
    lock_end: str = LOCK_END,
) -> Dict:
    segs = load_segments(segments_path)
    lock = pd.Timestamp(lock_end).normalize()
    rows: List[dict] = []
    by_symbol: Dict[str, dict] = {}
    for sym in symbols:
        px = load_px(daily_dir, sym)
        rets = px.pct_change()
        positions = positions_from_close(px)
        bh = slice_kpis(rets, positions, pd.Timestamp(BH_START), lock)
        timing = slice_kpis(rets, positions, pd.Timestamp(TIMING_START), lock)
        by_symbol[sym] = {"buy_hold_lock": bh, "timing_lock": timing, "segments": {}}
        for name, kpi in bh.items():
            rows.append(_fmt_row(sym, f"{BH_START}_{lock_end}", name, kpi))
        for name, kpi in timing.items():
            rows.append(_fmt_row(sym, f"{TIMING_START}_{lock_end}", name, kpi))
        for seg in segs:
            if seg["id"] not in COURT_SEGMENTS:
                continue
            chunk = slice_kpis(rets, {"buy_hold": positions["buy_hold"]}, seg["start"], min(seg["end"], lock))
            by_symbol[sym]["segments"][seg["id"]] = chunk
            for name, kpi in chunk.items():
                rows.append(_fmt_row(sym, str(seg["id"]), name, kpi))

    headline = (by_symbol.get("QQQ") or {}).get("timing_lock", {}).get("buy_hold") or {}
    return {
        "cagr": headline.get("cagr"),
        "calmar": headline.get("calmar"),
        "win_rate": headline.get("win_rate"),
        "maxdd": headline.get("maxdd"),
        "sharpe": headline.get("sharpe"),
        "symbols": list(symbols),
        "lock_end": lock_end,
        "pit_stock_picking": "not_run_no_universe",
        "rows": rows,
        "by_symbol": by_symbol,
    }


def write_court(result: Dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "result.json"
    path.write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
    table = pd.DataFrame(result.get("rows") or [])
    if not table.empty:
        table.to_csv(out_dir / "kpis.csv", index=False)
    return path
