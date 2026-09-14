"""AI mega-round calendar and AI-basket funding proxy (closed calendar)."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Literal

import numpy as np
import pandas as pd
import yaml

from src.features.registry import register_feature
from src.features.time_series.funding_rate_features import (
    _load_funding_rate_parquet,
    _rolling_robust_zscore,
)

DEFAULT_CALENDAR = "config/research/ai_financing_events.yaml"
DEFAULT_AI_BASKET = ("FETUSDT", "RENDERUSDT", "NEARUSDT", "TAOUSDT")


def load_ai_financing_events(
    calendar_path: str | Path = DEFAULT_CALENDAR,
    *,
    min_usd: float = 300_000_000.0,
) -> pd.DataFrame:
    """Load the locked public mega-round calendar.

    Returns a DataFrame indexed by announcement UTC midnight with columns
    ``usd``, ``company``, ``round``.
    """
    path = Path(calendar_path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    rows = raw.get("events") or []
    recs: list[dict] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        usd = float(row.get("usd") or 0.0)
        if usd < float(min_usd) and not bool(row.get("include")):
            continue
        dt = pd.Timestamp(str(row["date"]), tz="UTC").normalize()
        recs.append(
            {
                "date": dt,
                "usd": usd,
                "company": str(row.get("company") or ""),
                "round": str(row.get("round") or ""),
            }
        )
    if not recs:
        return pd.DataFrame(columns=["usd", "company", "round"])
    out = pd.DataFrame(recs).sort_values("date")
    out = out.drop_duplicates(subset=["date"], keep="last")
    return out.set_index("date")


def _bar_utc_dates(index: pd.DatetimeIndex) -> pd.Series:
    idx = index.tz_localize("UTC") if index.tz is None else index.tz_convert("UTC")
    return pd.Series(idx.normalize(), index=index)


def _first_bar_of_utc_day(dates: pd.Series) -> pd.Series:
    order = pd.Series(np.arange(len(dates), dtype=int), index=dates.index)
    first_i = order.groupby(dates.to_numpy(), sort=False).transform("min")
    return order.eq(first_i)


@register_feature(
    "compute_ai_financing_event_from_df",
    category="calendar",
    description=(
        "Public AI mega-round calendar. Event=1 on the first bar of the UTC "
        "day after the announcement date. Window starts that day."
    ),
    outputs=[
        "ai_financing_event",
        "ai_financing_in_window",
        "ai_financing_days_since",
        "ai_financing_log_usd",
    ],
)
def compute_ai_financing_event_from_df(
    df: pd.DataFrame,
    *,
    calendar_path: str = DEFAULT_CALENDAR,
    hold_days: int = 5,
    min_usd: float = 300_000_000.0,
    node_cache_version: str | None = None,
) -> pd.DataFrame:
    """Calendar feature: announcement day D is usable only from UTC date D+1.

    The column depends only on the bar's UTC date and the locked calendar.
    It does not use same-bar OHLC.
    """
    del node_cache_version
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("df index must be DatetimeIndex")
    hold = max(int(hold_days), 1)
    events = load_ai_financing_events(calendar_path, min_usd=float(min_usd))
    dates = _bar_utc_dates(df.index)
    out = pd.DataFrame(
        {
            "ai_financing_event": np.zeros(len(df), dtype=float),
            "ai_financing_in_window": np.zeros(len(df), dtype=float),
            "ai_financing_days_since": np.full(len(df), np.nan, dtype=float),
            "ai_financing_log_usd": np.full(len(df), np.nan, dtype=float),
        },
        index=df.index,
    )
    if events.empty:
        return out

    event_dates = events.index.sort_values()
    usd = events["usd"].astype(float)
    # Latest announcement strictly before this bar's UTC date → D is closed.
    pos = event_dates.searchsorted(dates.to_numpy(), side="left") - 1
    valid = pos >= 0
    if not bool(valid.any()):
        return out

    chosen = event_dates.take(np.clip(pos, 0, len(event_dates) - 1))
    last = pd.Series(chosen, index=df.index)
    days = (dates - last).dt.days.astype(float).to_numpy()
    days = np.where(valid, days, np.nan)
    log_usd = np.log10(np.clip(usd.reindex(chosen).to_numpy(), 1.0, None))
    log_usd = np.where(valid, log_usd, np.nan)

    in_window = valid & (days >= 1.0) & (days <= float(hold))
    first = _first_bar_of_utc_day(dates).to_numpy()
    event = in_window & first & np.isclose(days, 1.0)

    out["ai_financing_event"] = event.astype(float)
    out["ai_financing_in_window"] = in_window.astype(float)
    out["ai_financing_days_since"] = days
    out["ai_financing_log_usd"] = log_usd
    return out


def _asof_series_to_bars(bar_ts: pd.DatetimeIndex, series: pd.Series) -> np.ndarray:
    left = pd.DataFrame({"_ts": bar_ts, "_i": np.arange(len(bar_ts), dtype=int)})
    right = pd.DataFrame({"_ts": series.index, "v": series.to_numpy()}).dropna()
    if right.empty:
        return np.full(len(bar_ts), np.nan, dtype=float)
    merged = pd.merge_asof(
        left.sort_values("_ts"),
        right.sort_values("_ts"),
        on="_ts",
        direction="backward",
        allow_exact_matches=True,
    )
    vals = np.full(len(bar_ts), np.nan, dtype=float)
    vals[merged["_i"].to_numpy()] = merged["v"].to_numpy()
    return vals


@register_feature(
    "compute_ai_basket_funding_zscore_from_df",
    category="cross_symbol",
    description=(
        "Equal-weight mean of AI-narrative perpetual funding z-scores "
        "asof-joined onto host bars."
    ),
    outputs=["ai_basket_funding_zscore"],
)
def compute_ai_basket_funding_zscore_from_df(
    df: pd.DataFrame,
    *,
    funding_rate_dir: str = "data/funding_rate/parquet",
    symbols: Iterable[str] = DEFAULT_AI_BASKET,
    on_missing: Literal["nan", "raise"] = "nan",
    z_window: int = 50,
    z_min_periods: int = 20,
    node_cache_version: str | None = None,
) -> pd.DataFrame:
    del node_cache_version
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("df index must be DatetimeIndex")
    idx = (
        df.index.tz_localize("UTC")
        if df.index.tz is None
        else df.index.tz_convert("UTC")
    )
    stacked: list[np.ndarray] = []
    for raw in symbols:
        sym = str(raw).strip().upper()
        if not sym:
            continue
        try:
            native = _load_funding_rate_parquet(sym, funding_rate_dir)
        except Exception:
            if on_missing == "raise":
                raise
            continue
        z = _rolling_robust_zscore(
            native, window=int(z_window), min_periods=int(z_min_periods)
        )
        stacked.append(_asof_series_to_bars(idx, z))
    out = pd.DataFrame({"ai_basket_funding_zscore": np.nan}, index=df.index)
    if stacked:
        mat = np.vstack(stacked)
        with np.errstate(all="ignore"):
            out["ai_basket_funding_zscore"] = np.nanmean(mat, axis=0)
    elif on_missing == "raise":
        raise FileNotFoundError(
            f"No AI-basket funding parquet under {funding_rate_dir}"
        )
    return out
