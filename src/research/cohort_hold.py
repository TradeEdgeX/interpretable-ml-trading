"""Closed-bar A-share buy-and-hold cohort (small-cap vs large-cap control).

Not ``event_backtest``. Entry uses that day's circulating cap
(``amount / (turnover/100)``), then holds a fixed number of trading years.
Delist during the hold → last printed close is the terminal.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

ST_NAME_RE = re.compile(r"(?i)(?:\*?ST|SST|PT)")
BARS_PER_YEAR = 252
MIN_LISTED_DAYS = 365


def circ_mcap_yi(amount: float, turnover: float) -> float:
    """Circulating cap in 亿元 from that bar's amount and turnover %."""
    try:
        amt = float(amount)
        turn = float(turnover)
    except (TypeError, ValueError):
        return float("nan")
    if not (amt > 0 and turn > 0):
        return float("nan")
    return amt / (turn / 100.0) / 1e8


def is_st_name(name: object) -> bool:
    return bool(name) and bool(ST_NAME_RE.search(str(name)))


def trading_year_bars(years: float) -> int:
    return int(round(float(years) * BARS_PER_YEAR))


def snap_trading_day(
    calendar: pd.DatetimeIndex, day: pd.Timestamp
) -> Optional[pd.Timestamp]:
    """Last closed bar on or before ``day``."""
    cal = pd.DatetimeIndex(calendar).normalize()
    day = pd.Timestamp(day).normalize()
    loc = int(cal.searchsorted(day, side="right") - 1)
    if loc < 0:
        return None
    return cal[loc]


def hold_end(
    calendar: pd.DatetimeIndex, start: pd.Timestamp, hold_bars: int
) -> Optional[pd.Timestamp]:
    """Last bar of a completed hold, or None if the tape is too short."""
    cal = pd.DatetimeIndex(calendar).normalize()
    entry = snap_trading_day(cal, start)
    if entry is None:
        return None
    loc = int(cal.get_loc(entry))
    end_i = loc + int(hold_bars)
    if end_i >= len(cal):
        return None
    return cal[end_i]


def max_drawdown(equity: pd.Series) -> float:
    if equity is None or equity.empty:
        return float("nan")
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min()) if len(dd) else float("nan")


def cagr(equity: pd.Series, *, periods_per_year: int = BARS_PER_YEAR) -> float:
    if equity is None or len(equity) < 2:
        return float("nan")
    start = float(equity.iloc[0])
    end = float(equity.iloc[-1])
    if start <= 0 or end <= 0:
        return float("nan")
    years = (len(equity) - 1) / float(periods_per_year)
    if years <= 0:
        return float("nan")
    return float((end / start) ** (1.0 / years) - 1.0)


def sharpe(daily_rets: pd.Series, *, periods_per_year: int = BARS_PER_YEAR) -> float:
    r = pd.to_numeric(daily_rets, errors="coerce").dropna()
    if len(r) < 5:
        return float("nan")
    sd = float(r.std(ddof=1))
    if sd <= 0 or not math.isfinite(sd):
        return float("nan")
    return float(r.mean() / sd * math.sqrt(periods_per_year))


def book_kpis(equity: pd.Series) -> Dict[str, float]:
    eq = pd.to_numeric(equity, errors="coerce").dropna()
    if eq.empty:
        return {
            "cagr": float("nan"),
            "calmar": float("nan"),
            "maxdd": float("nan"),
            "sharpe": float("nan"),
        }
    rets = eq.pct_change().dropna()
    c = cagr(eq)
    dd = max_drawdown(eq)
    calmar = float("nan")
    if math.isfinite(c) and math.isfinite(dd) and dd < 0:
        calmar = float(c / abs(dd))
    return {"cagr": c, "calmar": calmar, "maxdd": dd, "sharpe": sharpe(rets)}


def load_segments(path: str | Path) -> List[dict]:
    import yaml

    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    segs = data.get("segments") or []
    out = []
    for item in segs:
        if isinstance(item, dict) and item.get("id"):
            out.append(
                {
                    "id": str(item["id"]),
                    "start": pd.Timestamp(item["start_date"]).normalize(),
                    "end": pd.Timestamp(item["end_date"]).normalize(),
                }
            )
    return out


def segment_for(day: pd.Timestamp, segments: Sequence[dict]) -> Optional[str]:
    d = pd.Timestamp(day).normalize()
    for seg in segments:
        if seg["start"] <= d <= seg["end"]:
            return seg["id"]
    return None


def load_stock_basic(path: str | Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "symbol" not in df.columns and "code" in df.columns:
        df = df.copy()
        df["symbol"] = df["code"].map(lambda c: str(c).split(".")[-1].zfill(6))
    df["symbol"] = df["symbol"].map(lambda s: str(s).zfill(6))
    if "ipoDate" in df.columns:
        df["ipoDate"] = pd.to_datetime(df["ipoDate"], errors="coerce")
    return df


def load_universe_day(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "symbol" not in df.columns and "code" in df.columns:
        df = df.copy()
        df["symbol"] = df["code"].map(lambda c: str(c).split(".")[-1].zfill(6))
    df["symbol"] = df["symbol"].map(lambda s: str(s).zfill(6))
    return df


@dataclass
class NameHold:
    symbol: str
    entry: pd.Timestamp
    exit: pd.Timestamp
    mcap_yi: float
    multiple: float
    group: str
    segment: str
    hold_years: float
    delisted: bool = False


@dataclass
class CohortResult:
    holds: List[NameHold] = field(default_factory=list)
    books: Dict[Tuple[str, str, float], pd.Series] = field(default_factory=dict)
    coverage: Dict[str, int] = field(default_factory=dict)


def _day_index(dates) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(dates, errors="coerce")).normalize()


def _load_one_daily(path: Path) -> Optional[pd.DataFrame]:
    try:
        raw = pd.read_parquet(path)
    except Exception:  # noqa: BLE001
        return None
    need = {"date", "close"}
    if not need.issubset(raw.columns):
        return None
    out = pd.DataFrame(
        {
            "close": pd.to_numeric(raw["close"], errors="coerce").to_numpy(),
            "amount": pd.to_numeric(raw["amount"], errors="coerce").to_numpy()
            if "amount" in raw.columns
            else np.nan,
            "turnover": pd.to_numeric(raw["turnover"], errors="coerce").to_numpy()
            if "turnover" in raw.columns
            else np.nan,
        },
        index=_day_index(raw["date"]),
    )
    out = out[~out.index.duplicated(keep="last")].sort_index()
    out = out.dropna(subset=["close"])
    return out if len(out) >= 20 else None


def load_daily_panel(daily_dir: Path, symbols: Iterable[str]) -> Dict[str, pd.DataFrame]:
    panel: Dict[str, pd.DataFrame] = {}
    for i, sym in enumerate(sorted(set(symbols))):
        for cand in (daily_dir / f"{sym}.parquet", daily_dir / f"{sym}.SH.parquet"):
            if cand.is_file():
                tab = _load_one_daily(cand)
                if tab is not None:
                    panel[sym] = tab
                break
        if (i + 1) % 800 == 0:
            print(f"  loaded {i + 1} daily files / {len(panel)} ok", flush=True)
    return panel


def _mcap_on(tab: pd.DataFrame, day: pd.Timestamp) -> float:
    if day not in tab.index:
        hist = tab.loc[tab.index <= day]
        if hist.empty:
            return float("nan")
        row = hist.iloc[-1]
    else:
        row = tab.loc[day]
    return circ_mcap_yi(row.get("amount", np.nan), row.get("turnover", np.nan))


def _terminal_multiple(
    tab: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp
) -> Tuple[float, pd.Timestamp, bool]:
    if start not in tab.index:
        return float("nan"), start, False
    entry = float(tab.loc[start, "close"])
    if entry <= 0:
        return float("nan"), start, False
    path = tab.loc[(tab.index >= start) & (tab.index <= end), "close"]
    if path.empty:
        return float("nan"), start, False
    last_day = path.index[-1]
    last = float(path.iloc[-1])
    if last <= 0:
        return float("nan"), last_day, True
    delisted = last_day < pd.Timestamp(end).normalize()
    return last / entry, last_day, delisted


def _close_matrix(
    panel: Dict[str, pd.DataFrame], calendar: pd.DatetimeIndex
) -> pd.DataFrame:
    """Calendar × symbol close panel (NaN if the name is missing that day)."""
    cols = {}
    for sym, tab in panel.items():
        cols[sym] = tab["close"].reindex(calendar)
    return pd.DataFrame(cols, index=calendar)


def _vintage_equity(
    symbols: Sequence[str],
    closes: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    calendar: pd.DatetimeIndex,
) -> pd.Series:
    window = calendar[(calendar >= start) & (calendar <= end)]
    if len(window) < 2:
        return pd.Series(dtype=float)
    have = [s for s in symbols if s in closes.columns]
    if not have:
        return pd.Series(dtype=float)
    sub = closes.loc[window, have]
    daily = sub.pct_change(fill_method=None).mean(axis=1, skipna=True).fillna(0.0)
    eq = (1.0 + daily).cumprod()
    eq.iloc[0] = 1.0
    return eq


def _combine_books(curves: List[pd.Series], calendar: pd.DatetimeIndex) -> pd.Series:
    if not curves:
        return pd.Series(dtype=float)
    frame = pd.concat(curves, axis=1)
    rets = frame.pct_change(fill_method=None).mean(axis=1)
    start = frame.dropna(how="all").index[0]
    out = pd.Series(index=calendar[calendar >= start], dtype=float)
    out.iloc[0] = 1.0
    for i, t in enumerate(out.index[1:], start=1):
        r = rets.loc[t] if t in rets.index and pd.notna(rets.loc[t]) else 0.0
        out.iloc[i] = out.iloc[i - 1] * (1.0 + float(r))
    return out


def run_cohort(
    *,
    daily_dir: Path,
    basic_path: Path,
    universe_dir: Path,
    segments_path: Path,
    calendar_symbol: str = "000300.SH",
    mcap_yi: float = 100.0,
    hold_years: Sequence[float] = (3.0, 4.0),
    min_listed_days: int = MIN_LISTED_DAYS,
) -> CohortResult:
    basic = load_stock_basic(basic_path)
    ipo = (
        basic.drop_duplicates("symbol").set_index("symbol")["ipoDate"]
        if "ipoDate" in basic.columns
        else pd.Series(dtype="datetime64[ns]")
    )
    segments = load_segments(segments_path)
    snaps: Dict[pd.Timestamp, pd.DataFrame] = {}
    need: set[str] = set()
    for fp in sorted(Path(universe_dir).glob("*.parquet")):
        snap = load_universe_day(fp)
        asof = (
            pd.Timestamp(snap["asof"].iloc[0]).normalize()
            if "asof" in snap.columns and len(snap)
            else pd.Timestamp(fp.stem)
        )
        snaps[asof] = snap
        if "symbol" in snap.columns:
            need.update(snap["symbol"].astype(str).str.zfill(6).tolist())

    cal_path = Path(daily_dir) / f"{calendar_symbol}.parquet"
    if not cal_path.is_file():
        cal_path = Path(daily_dir) / "000300.SH.parquet"
    if cal_path.is_file():
        cal_tab = _load_one_daily(cal_path)
        calendar = cal_tab.index if cal_tab is not None else pd.DatetimeIndex([])
    else:
        calendar = pd.DatetimeIndex([])

    print(f"universe days={len(snaps)} symbols_in_snaps={len(need)}", flush=True)
    panel = load_daily_panel(Path(daily_dir), need)
    print(f"daily panel loaded={len(panel)}", flush=True)
    if calendar.empty and panel:
        calendar = pd.DatetimeIndex(sorted({d for tab in panel.values() for d in tab.index}))
    print("aligning close matrix…", flush=True)
    closes = _close_matrix(panel, calendar)
    print(f"close matrix {closes.shape}", flush=True)

    result = CohortResult(
        coverage={
            "universe_days": len(snaps),
            "snap_symbols": len(need),
            "daily_files": len(panel),
        }
    )
    for years in hold_years:
        hold_bars = trading_year_bars(years)
        for group in ("small", "large"):
            curves_by_seg: Dict[str, List[pd.Series]] = {s["id"]: [] for s in segments}
            curves_by_seg["all_complete"] = []
            n_vintages = 0
            for asof in sorted(snaps):
                entry = snap_trading_day(calendar, asof)
                if entry is None:
                    continue
                end = hold_end(calendar, entry, hold_bars)
                if end is None:
                    continue
                snap = snaps[asof]
                names: list[str] = []
                caps: Dict[str, float] = {}
                for rec in snap.itertuples(index=False):
                    sym = str(getattr(rec, "symbol", "")).zfill(6)
                    if not sym or sym[0] not in "036":
                        continue
                    status = str(getattr(rec, "tradeStatus", "1"))
                    if status != "1":
                        continue
                    name = getattr(rec, "code_name", "")
                    if is_st_name(name):
                        continue
                    if sym in ipo.index and pd.notna(ipo.loc[sym]):
                        if (asof - pd.Timestamp(ipo.loc[sym])).days < min_listed_days:
                            continue
                    tab = panel.get(sym)
                    if tab is None or entry not in tab.index:
                        continue
                    cap = _mcap_on(tab, entry)
                    if not math.isfinite(cap):
                        continue
                    if group == "small" and not (0 < cap <= mcap_yi):
                        continue
                    if group == "large" and not (cap > mcap_yi):
                        continue
                    names.append(sym)
                    caps[sym] = cap
                if len(names) < 10:
                    continue
                if n_vintages == 0:
                    print(
                        f"  first vintage {group} {years:g}y {entry.date()} n={len(names)}",
                        flush=True,
                    )
                seg = segment_for(entry, segments) or "unlabeled"
                eq = _vintage_equity(names, closes, entry, end, calendar)
                if eq.empty:
                    continue
                n_vintages += 1
                curves_by_seg.setdefault(seg, []).append(eq)
                curves_by_seg["all_complete"].append(eq)
                for sym in names:
                    mult, exit_day, dead = _terminal_multiple(panel[sym], entry, end)
                    if not math.isfinite(mult):
                        continue
                    result.holds.append(
                        NameHold(
                            symbol=sym,
                            entry=asof,
                            exit=exit_day,
                            mcap_yi=caps[sym],
                            multiple=mult,
                            group=group,
                            segment=seg,
                            hold_years=float(years),
                            delisted=dead,
                        )
                    )
            result.coverage[f"vintages_{group}_{years:g}y"] = n_vintages
            for seg_id, curves in curves_by_seg.items():
                if curves:
                    result.books[(group, seg_id, float(years))] = _combine_books(
                        curves, calendar
                    )
    return result


def summarize(result: CohortResult) -> pd.DataFrame:
    rows = []
    holds = pd.DataFrame([h.__dict__ for h in result.holds]) if result.holds else pd.DataFrame()
    keys = sorted(result.books)
    seen = set()
    for group, seg, years in keys:
        seen.add((group, seg, years))
        eq = result.books[(group, seg, years)]
        k = book_kpis(eq)
        if holds.empty:
            sub = holds
        else:
            if seg == "all_complete":
                sub = holds[(holds["group"] == group) & (holds["hold_years"] == years)]
            else:
                sub = holds[
                    (holds["group"] == group)
                    & (holds["hold_years"] == years)
                    & (holds["segment"] == seg)
                ]
        n = int(len(sub))
        wr = float((sub["multiple"] > 1.0).mean()) if n else float("nan")
        hit10 = float((sub["multiple"] >= 10.0).mean()) if n else float("nan")
        hit5 = float((sub["multiple"] >= 5.0).mean()) if n else float("nan")
        med = float(sub["multiple"].median()) if n else float("nan")
        n_dead = int(sub["delisted"].sum()) if n and "delisted" in sub.columns else 0
        # drop-Top-3 multiples: classification only
        ex3_hit10 = float("nan")
        if n >= 4:
            dropped = sub.sort_values("multiple", ascending=False).iloc[3:]
            ex3_hit10 = float((dropped["multiple"] >= 10.0).mean())
        rows.append(
            {
                "group": group,
                "segment": seg,
                "hold_years": years,
                "cagr": k["cagr"],
                "calmar": k["calmar"],
                "win_rate": wr,
                "maxdd": k["maxdd"],
                "sharpe": k["sharpe"],
                "n": n,
                "hit_5x": hit5,
                "hit_10x": hit10,
                "ex_top3_hit_10x": ex3_hit10,
                "median_multiple": med,
                "n_delisted": n_dead,
            }
        )
    return pd.DataFrame(rows)
