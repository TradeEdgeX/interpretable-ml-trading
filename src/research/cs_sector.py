"""Sector-rank cross-section + locked 10bp one-way rebalance cost.

Industry map is the latest snapshot (not PIT). Say so in the paper.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd

from src.research.cohort_hold import load_segments, load_stock_basic
from src.research.cohort_hold import book_kpis
from src.research.cs_panel import (
    COURT_SEGMENTS,
    MIN_NAMES,
    TOP_Q,
    _equity,
    _win_rate,
    a_share_symbols,
    load_panel,
    stack_panel,
)

COST_BP = 10.0
MIN_SECTOR = 8
BOT_Q = 0.20


def one_way_turnover(prev: pd.Series, new: pd.Series) -> float:
    prev = pd.to_numeric(prev, errors="coerce").fillna(0.0)
    new = pd.to_numeric(new, errors="coerce").fillna(0.0)
    both = pd.concat([prev, new], axis=1, keys=["p", "n"]).fillna(0.0)
    return 0.5 * float((both["n"] - both["p"]).abs().sum())


def cost_from_turnover(turnover: float, *, bp: float = COST_BP) -> float:
    return float(turnover) * (float(bp) / 10000.0)


def load_industry(path: Path) -> pd.Series:
    path = Path(path)
    if path.suffix.lower() in {".yaml", ".yml"}:
        import yaml

        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"industry yaml must be a symbol→name map: {path}")
        if "exact" in raw or "contains" in raw:
            raise ValueError(f"{path} is a coarse remap, not a symbol map")
        pairs = {}
        for key, val in raw.items():
            digits = str(key).strip("'\"").zfill(6)
            if digits.isdigit() and len(digits) == 6:
                pairs[digits] = str(val).strip()
        ser = pd.Series(pairs)
        return ser[ser.ne("") & ser.ne("nan")]
    raw = pd.read_parquet(path)
    raw["symbol"] = raw["symbol"].map(lambda s: str(s).zfill(6))
    col = "industry" if "industry" in raw.columns else "industryClassification"
    out = raw.dropna(subset=["symbol", col]).drop_duplicates("symbol", keep="last")
    return out.set_index("symbol")[col].astype(str)


def attach_industry(long: pd.DataFrame, industry: pd.Series) -> pd.DataFrame:
    out = long.copy()
    out["industry"] = out["symbol"].map(industry)
    return out.dropna(subset=["industry"])


def _sector_scores(g: pd.DataFrame) -> Optional[pd.Series]:
    cnt = g.groupby("industry")["symbol"].size()
    keep = cnt[cnt >= MIN_SECTOR].index
    if len(keep) < 5:
        return None
    sec = g.loc[g["industry"].isin(keep)].groupby("industry")[["mom_20", "amount_z_20"]].mean()
    for col in ("mom_20", "amount_z_20"):
        sd = float(sec[col].std())
        if not math.isfinite(sd) or sd <= 0:
            return None
        sec[col] = (sec[col] - float(sec[col].mean())) / sd
    return 0.5 * sec["mom_20"] + 0.5 * sec["amount_z_20"]


def week_end_dates(dates: Iterable) -> set:
    idx = pd.DatetimeIndex(pd.to_datetime(list(dates))).normalize().unique().sort_values()
    if idx.empty:
        return set()
    iso = idx.isocalendar()
    key = iso.year.astype(int) * 100 + iso.week.astype(int)
    last = pd.Series(idx, index=key).groupby(level=0).max()
    return {pd.Timestamp(x).normalize() for x in last}


def cagr_ex_top3(rets: pd.Series) -> float:
    r = pd.to_numeric(rets, errors="coerce").dropna()
    if len(r) < 8:
        return float("nan")
    dropped = r.drop(r.nlargest(3).index)
    if dropped.empty:
        return float("nan")
    return float(book_kpis(_equity(dropped))["cagr"])


def _ew_weights(symbols: Iterable[str]) -> pd.Series:
    names = pd.Index(sorted(set(symbols)))
    if len(names) == 0:
        return pd.Series(dtype=float)
    return pd.Series(1.0 / len(names), index=names)


def daily_sector_books(long: pd.DataFrame, *, top_q: float = TOP_Q, bp: float = COST_BP) -> pd.DataFrame:
    prev_fac: pd.Series = pd.Series(dtype=float)
    prev_ew: pd.Series = pd.Series(dtype=float)
    rows: List[dict] = []
    for day, grp in long.groupby(level=0):
        g = grp.dropna(subset=["mom_20", "amount_z_20", "book_ret", "industry", "symbol"])
        if len(g) < MIN_NAMES:
            continue
        scores = _sector_scores(g)
        if scores is None:
            continue
        cut = float(scores.quantile(top_q))
        long_secs = scores.index[scores >= cut]
        picked = g.loc[g["industry"].isin(long_secs)]
        if picked.empty:
            continue
        w_fac = _ew_weights(picked["symbol"])
        w_ew = _ew_weights(g["symbol"])
        ret_map = g.set_index("symbol")["book_ret"]
        fac_gross = float(ret_map.reindex(w_fac.index).dot(w_fac))
        ew_gross = float(ret_map.reindex(w_ew.index).dot(w_ew))
        to_fac = one_way_turnover(prev_fac, w_fac)
        to_ew = one_way_turnover(prev_ew, w_ew)
        rows.append(
            {
                "date": pd.Timestamp(day).normalize(),
                "fac_gross": fac_gross,
                "ew_gross": ew_gross,
                "fac_ret": fac_gross - cost_from_turnover(to_fac, bp=bp),
                "ew_ret": ew_gross - cost_from_turnover(to_ew, bp=bp),
                "to_fac": to_fac,
                "to_ew": to_ew,
                "n_fac": int(len(w_fac)),
                "n_all": int(len(w_ew)),
                "n_sec": int(len(scores)),
                "n_long_sec": int(len(long_secs)),
            }
        )
        prev_fac, prev_ew = w_fac, w_ew
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("date").sort_index()


def daily_sector_ls_books(long: pd.DataFrame, *, top_q: float = TOP_Q, bot_q: float = BOT_Q, bp: float = COST_BP) -> pd.DataFrame:
    prev_long: pd.Series = pd.Series(dtype=float)
    prev_short: pd.Series = pd.Series(dtype=float)
    rows: List[dict] = []
    for day, grp in long.groupby(level=0):
        g = grp.dropna(subset=["mom_20", "amount_z_20", "book_ret", "industry", "symbol"])
        if len(g) < MIN_NAMES:
            continue
        scores = _sector_scores(g)
        if scores is None:
            continue
        hi = float(scores.quantile(top_q))
        lo = float(scores.quantile(bot_q))
        long_secs = scores.index[scores >= hi]
        short_secs = scores.index[scores <= lo]
        long_names = g.loc[g["industry"].isin(long_secs)]
        short_names = g.loc[g["industry"].isin(short_secs)]
        if long_names.empty or short_names.empty:
            continue
        w_l = _ew_weights(long_names["symbol"])
        w_s = _ew_weights(short_names["symbol"])
        ret_map = g.set_index("symbol")["book_ret"]
        long_g = float(ret_map.reindex(w_l.index).dot(w_l))
        short_g = float(ret_map.reindex(w_s.index).dot(w_s))
        to_l = one_way_turnover(prev_long, w_l)
        to_s = one_way_turnover(prev_short, w_s)
        ls_gross = long_g - short_g
        rows.append(
            {
                "date": pd.Timestamp(day).normalize(),
                "ls_gross": ls_gross,
                "ls_ret": ls_gross - cost_from_turnover(to_l, bp=bp) - cost_from_turnover(to_s, bp=bp),
                "to_fac": to_l + to_s,
                "n_long": int(len(w_l)),
                "n_short": int(len(w_s)),
            }
        )
        prev_long, prev_short = w_l, w_s
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("date").sort_index()


def weekly_sector_books(long: pd.DataFrame, *, top_q: float = TOP_Q, bp: float = COST_BP) -> pd.DataFrame:
    ends = week_end_dates(long.index.get_level_values(0).unique())
    prev: pd.Series = pd.Series(dtype=float)
    rows: List[dict] = []
    for day, grp in long.groupby(level=0):
        g = grp.dropna(subset=["mom_20", "amount_z_20", "book_ret", "industry", "symbol"])
        if len(g) < MIN_NAMES:
            continue
        day_n = pd.Timestamp(day).normalize()
        rebalance = day_n in ends
        if rebalance:
            scores = _sector_scores(g)
            if scores is None:
                continue
            cut = float(scores.quantile(top_q))
            picked = g.loc[g["industry"].isin(scores.index[scores >= cut])]
            if picked.empty:
                continue
            w = _ew_weights(picked["symbol"])
            to = one_way_turnover(prev, w)
            prev = w
        else:
            if prev.empty:
                continue
            w = prev
            to = 0.0
        ret_map = g.set_index("symbol")["book_ret"]
        aligned = ret_map.reindex(w.index)
        if aligned.isna().any():
            keep = aligned.dropna()
            if keep.empty:
                continue
            w = w.reindex(keep.index)
            w = w / float(w.sum())
            aligned = keep
        fac_gross = float(aligned.dot(w))
        rows.append(
            {
                "date": day_n,
                "fac_gross": fac_gross,
                "fac_ret": fac_gross - cost_from_turnover(to, bp=bp),
                "to_fac": float(to),
                "n_fac": int(len(w)),
                "rebalance": bool(rebalance),
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("date").sort_index()


def segment_kpi_table(books: pd.DataFrame, segments, *, pairs=None) -> pd.DataFrame:
    if pairs is None:
        pairs = (
            ("sector_top", "fac_ret"),
            ("ew_cost", "ew_ret"),
            ("sector_top_gross", "fac_gross"),
            ("ew_gross", "ew_gross"),
            ("sector_ls", "ls_ret"),
            ("sector_ls_gross", "ls_gross"),
        )
    rows: List[dict] = []
    for seg in segments:
        if seg["id"] not in COURT_SEGMENTS:
            continue
        sl = books.loc[(books.index >= seg["start"]) & (books.index <= seg["end"])]
        if sl.empty:
            continue
        for group, col in pairs:
            if col not in sl.columns:
                continue
            kpi = book_kpis(_equity(sl[col]))
            kpi["win_rate"] = _win_rate(sl[col])
            kpi["group"] = group
            kpi["segment"] = seg["id"]
            kpi["n_days"] = int(len(sl))
            to_col = "to_fac" if ("sector" in group or group.startswith("ls")) else "to_ew"
            kpi["to_mean"] = float(sl[to_col].mean()) if to_col in sl.columns else float("nan")
            if col in sl.columns:
                kpi["cagr_ex_top3"] = cagr_ex_top3(sl[col])
            rows.append(kpi)
    return pd.DataFrame(rows)


def run_cs_sector(
    *,
    daily_dir: Path,
    basic_path: Path,
    industry_path: Path,
    segments_path: Path,
    mode: str = "daily",
) -> dict:
    from src.data_tools.ashare_baostock import fetch_stock_industry

    industry_path = Path(industry_path)
    if not industry_path.is_file():
        for candidate in (
            Path("config/industry_map_ashare.yaml"),
            Path("data/ashare/industry/sw_l1.parquet"),
        ):
            if candidate.is_file():
                print(f"industry missing at {industry_path}; using {candidate}", flush=True)
                industry_path = candidate
                break
        else:
            print(f"downloading industry → {industry_path}", flush=True)
            fetch_stock_industry(industry_path)
    basic = load_stock_basic(basic_path)
    symbols = a_share_symbols(basic)
    industry = load_industry(Path(industry_path))
    print(f"universe {len(symbols)}; industry {industry.nunique()} groups / {len(industry)} names", flush=True)
    panel = load_panel(daily_dir, symbols)
    long = stack_panel(panel)
    # stack_panel adds market-wide score; sector book uses raw mom/amount + industry
    long = attach_industry(long, industry)
    print(f"panel {len(panel)}; rows with industry {len(long)}; mode={mode}", flush=True)
    if mode == "ls":
        books = daily_sector_ls_books(long)
        pairs = (("sector_ls", "ls_ret"), ("sector_ls_gross", "ls_gross"))
    elif mode == "weekly":
        books = weekly_sector_books(long)
        pairs = (("sector_weekly", "fac_ret"), ("sector_weekly_gross", "fac_gross"))
    elif mode == "daily":
        books = daily_sector_books(long)
        pairs = None
    else:
        raise ValueError(f"unknown sector mode {mode!r}")
    segments = load_segments(segments_path)
    kpis = segment_kpi_table(books, segments, pairs=pairs)
    return {
        "coverage": {
            "universe": len(symbols),
            "panel": len(panel),
            "industry_n": int(industry.nunique()),
            "industry_pit": False,
            "mode": mode,
        },
        "books": books,
        "kpis": kpis,
    }
