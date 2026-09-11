"""Closed-bar A-share cross-section: locked momentum + amount heat.

Not ``event_backtest``. Rank at day-T close; first fill is the next open.
Phase 1 is Spearman IC. The court is the daily long-top book vs equal-weight.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.research.cohort_hold import (
    book_kpis,
    is_st_name,
    load_segments,
    load_stock_basic,
)

LOOKBACK = 20
HOLD_BARS = 20
TOP_Q = 0.80
MIN_NAMES = 50
COURT_SEGMENTS = ("bear_2021", "bull_924", "chop_recent")


def mom_20(close: pd.Series) -> pd.Series:
    prev = close.shift(LOOKBACK)
    out = close / prev - 1.0
    return out.where(prev > 0)


def amount_z_20(amount: pd.Series) -> pd.Series:
    roll = amount.rolling(LOOKBACK, min_periods=LOOKBACK)
    sd = roll.std()
    return (amount - roll.mean()) / sd.replace(0.0, np.nan)


def attach_factors(tab: pd.DataFrame) -> pd.DataFrame:
    """Add mom / amount-z / next-open book return / 20-day open-to-open label."""
    out = tab.copy()
    out["mom_20"] = mom_20(out["close"])
    out["amount_z_20"] = amount_z_20(out["amount"])
    nxt = out["open"].shift(-1)
    nxt2 = out["open"].shift(-2)
    nxt_h = out["open"].shift(-(1 + HOLD_BARS))
    out["book_ret"] = nxt2 / nxt - 1.0
    out["fwd_20"] = nxt_h / nxt - 1.0
    return out


def _cs_z(series: pd.Series) -> pd.Series:
    mu = series.groupby(level=0).transform("mean")
    sd = series.groupby(level=0).transform("std")
    return (series - mu) / sd.replace(0.0, np.nan)


def add_score(long: pd.DataFrame) -> pd.DataFrame:
    out = long.copy()
    out["cs_mom"] = _cs_z(out["mom_20"])
    out["cs_amt"] = _cs_z(out["amount_z_20"])
    out["score"] = 0.5 * out["cs_mom"] + 0.5 * out["cs_amt"]
    return out


def rank_ic_day(score: pd.Series, fwd: pd.Series, *, min_n: int = MIN_NAMES) -> float:
    m = score.notna() & fwd.notna()
    if int(m.sum()) < min_n:
        return float("nan")
    rho, _p = spearmanr(score[m], fwd[m])
    return float(rho)


def ic_by_date(long: pd.DataFrame, col: str) -> pd.Series:
    rows: Dict[pd.Timestamp, float] = {}
    for day, grp in long.groupby(level=0):
        rows[pd.Timestamp(day)] = rank_ic_day(grp[col], grp["fwd_20"])
    return pd.Series(rows, dtype=float).sort_index()


def _segment_id(day: pd.Timestamp, segments: Sequence[dict]) -> Optional[str]:
    d = pd.Timestamp(day).normalize()
    for seg in segments:
        if seg["start"] <= d <= seg["end"]:
            return str(seg["id"])
    return None


def summarize_ic(
    long: pd.DataFrame, segments: Sequence[dict]
) -> pd.DataFrame:
    cols = ("mom_20", "amount_z_20", "score")
    rows: List[dict] = []
    for col in cols:
        series = ic_by_date(long, col)
        for seg in segments:
            if seg["id"] not in COURT_SEGMENTS:
                continue
            sl = series.loc[
                (series.index >= seg["start"]) & (series.index <= seg["end"])
            ].dropna()
            mean = float(sl.mean()) if len(sl) else float("nan")
            sd = float(sl.std(ddof=1)) if len(sl) > 1 else float("nan")
            ir = (
                mean / sd * math.sqrt(len(sl))
                if math.isfinite(mean) and math.isfinite(sd) and sd > 0
                else float("nan")
            )
            rows.append(
                {
                    "feature": col,
                    "segment": seg["id"],
                    "ic_mean": mean,
                    "ic_ir": ir,
                    "n_days": int(len(sl)),
                }
            )
    return pd.DataFrame(rows)


def score_weighted_ret(rets: pd.Series, score: pd.Series) -> float:
    """Long-only: weight ∝ max(score, 0). NaN if nothing is above 0."""
    w = pd.to_numeric(score, errors="coerce").clip(lower=0.0)
    r = pd.to_numeric(rets, errors="coerce")
    m = w.notna() & r.notna() & (w > 0)
    if int(m.sum()) == 0:
        return float("nan")
    ww = w[m]
    ww = ww / ww.sum()
    return float((ww * r[m]).sum())


def daily_books(long: pd.DataFrame, *, top_q: float = TOP_Q) -> pd.DataFrame:
    """One row per signal date: EW top, score-weighted top/all, vs EW universe."""
    rows: List[dict] = []
    for day, grp in long.groupby(level=0):
        g = grp.dropna(subset=["score", "book_ret"])
        if len(g) < MIN_NAMES:
            continue
        cut = float(g["score"].quantile(top_q))
        top = g.loc[g["score"] >= cut]
        low_cut = float(g["score"].quantile(1.0 - top_q))
        low = g.loc[g["score"] <= low_cut]
        if top.empty or low.empty:
            continue
        rows.append(
            {
                "date": pd.Timestamp(day).normalize(),
                "long_ret": float(top["book_ret"].mean()),
                "low_ret": float(low["book_ret"].mean()),
                "top_sw_ret": score_weighted_ret(top["book_ret"], top["score"]),
                "score_wt_ret": score_weighted_ret(g["book_ret"], g["score"]),
                "inv_score_wt_ret": score_weighted_ret(g["book_ret"], -g["score"]),
                "ew_ret": float(g["book_ret"].mean()),
                "n_long": int(len(top)),
                "n_low": int(len(low)),
                "n_all": int(len(g)),
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "date",
                "long_ret",
                "low_ret",
                "top_sw_ret",
                "score_wt_ret",
                "inv_score_wt_ret",
                "ew_ret",
                "n_long",
                "n_low",
                "n_all",
            ]
        )
    out = pd.DataFrame(rows).set_index("date").sort_index()
    out["excess_ret"] = out["long_ret"] - out["ew_ret"]
    return out


def _equity(rets: pd.Series) -> pd.Series:
    r = pd.to_numeric(rets, errors="coerce").fillna(0.0)
    return (1.0 + r).cumprod()


def _win_rate(rets: pd.Series) -> float:
    r = pd.to_numeric(rets, errors="coerce").dropna()
    if r.empty:
        return float("nan")
    return float((r > 0).mean())


def kpis_for_segment(books: pd.DataFrame, start, end) -> List[dict]:
    sl = books.loc[(books.index >= start) & (books.index <= end)]
    if sl.empty:
        return []
    out = []
    for group, col in (
        ("long_top", "long_ret"),
        ("long_low", "low_ret"),
        ("top_sw", "top_sw_ret"),
        ("score_wt", "score_wt_ret"),
        ("inv_score_wt", "inv_score_wt_ret"),
        ("ew", "ew_ret"),
    ):
        if col not in sl.columns:
            continue
        eq = _equity(sl[col])
        kpi = book_kpis(eq)
        kpi["win_rate"] = _win_rate(sl[col])
        kpi["group"] = group
        kpi["n_days"] = int(len(sl))
        out.append(kpi)
    return out


def segment_kpi_table(books: pd.DataFrame, segments: Sequence[dict]) -> pd.DataFrame:
    rows: List[dict] = []
    for seg in segments:
        if seg["id"] not in COURT_SEGMENTS:
            continue
        for row in kpis_for_segment(books, seg["start"], seg["end"]):
            row["segment"] = seg["id"]
            rows.append(row)
    return pd.DataFrame(rows)


def _load_one(path: Path) -> Optional[pd.DataFrame]:
    try:
        raw = pd.read_parquet(path)
    except Exception:  # noqa: BLE001
        return None
    need = {"date", "open", "close", "amount"}
    if not need.issubset(raw.columns):
        return None
    idx = pd.DatetimeIndex(pd.to_datetime(raw["date"], errors="coerce")).normalize()
    out = pd.DataFrame(
        {
            "open": pd.to_numeric(raw["open"], errors="coerce").to_numpy(),
            "close": pd.to_numeric(raw["close"], errors="coerce").to_numpy(),
            "amount": pd.to_numeric(raw["amount"], errors="coerce").to_numpy(),
        },
        index=idx,
    )
    out = out[~out.index.duplicated(keep="last")].sort_index()
    out = out[(out["open"] > 0) & (out["close"] > 0) & (out["amount"] > 0)]
    return out if len(out) >= LOOKBACK + HOLD_BARS + 3 else None


def a_share_symbols(basic: pd.DataFrame) -> List[str]:
    tab = basic.copy()
    if "type" in tab.columns:
        tab = tab[tab["type"].astype(str) == "1"]
    name_col = "code_name" if "code_name" in tab.columns else "name"
    if name_col in tab.columns:
        tab = tab[~tab[name_col].map(is_st_name)]
    tab["symbol"] = tab["symbol"].map(lambda s: str(s).zfill(6))
    return sorted(tab["symbol"].drop_duplicates().tolist())


def load_panel(daily_dir: Path, symbols: Iterable[str]) -> Dict[str, pd.DataFrame]:
    panel: Dict[str, pd.DataFrame] = {}
    want = set(symbols)
    files = sorted(daily_dir.glob("*.parquet"))
    for i, path in enumerate(files):
        sym = path.stem.split(".")[0].zfill(6)
        if want and sym not in want:
            continue
        tab = _load_one(path)
        if tab is not None:
            panel[sym] = attach_factors(tab)
        if (i + 1) % 800 == 0:
            print(f"  loaded {i + 1} files / {len(panel)} ok", flush=True)
    return panel


def stack_panel(panel: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames = []
    keep = ["mom_20", "amount_z_20", "book_ret", "fwd_20"]
    for sym, tab in panel.items():
        piece = tab[keep].copy()
        piece["symbol"] = sym
        frames.append(piece)
    if not frames:
        return pd.DataFrame(columns=keep + ["symbol"])
    long = pd.concat(frames, axis=0)
    long.index = pd.DatetimeIndex(long.index).normalize()
    long.index.name = "date"
    return add_score(long)


def run_cs_panel(
    *,
    daily_dir: Path,
    basic_path: Path,
    segments_path: Path,
) -> dict:
    basic = load_stock_basic(basic_path)
    symbols = a_share_symbols(basic)
    print(f"universe type=1 non-ST: {len(symbols)}", flush=True)
    panel = load_panel(daily_dir, symbols)
    print(f"panel names: {len(panel)}", flush=True)
    long = stack_panel(panel)
    segments = load_segments(segments_path)
    ic = summarize_ic(long, segments)
    books = daily_books(long)
    kpis = segment_kpi_table(books, segments)
    return {
        "coverage": {"universe": len(symbols), "panel": len(panel), "rows": int(len(long))},
        "ic": ic,
        "books": books,
        "kpis": kpis,
    }
