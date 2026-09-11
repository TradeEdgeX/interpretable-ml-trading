"""Calendar / cross-symbol / tick P99 features for public court examples."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd

from src.features.registry import register_feature


@register_feature(
    "compute_weekday_monday_features_from_series",
    category="baseline",
    description="Weekday (Mon=0) and Monday close-to-close return on daily bars.",
    outputs=["weekday", "monday_return", "monday_down"],
)
def compute_weekday_monday_features_from_series(
    close: pd.Series,
    *,
    node_cache_version: str | None = None,
) -> pd.DataFrame:
    """Closed-bar calendar features.

    ``monday_return`` is NaN on non-Mondays. ``monday_down`` is 1.0 when
    Monday return < 0, else 0.0 on Mondays, NaN otherwise.
    """
    del node_cache_version  # cache-busting hook for FeatureStore meta
    c = pd.to_numeric(close, errors="coerce").astype(float)
    idx = c.index
    if not isinstance(idx, pd.DatetimeIndex):
        raise ValueError("close index must be DatetimeIndex")
    # Use Asia/Shanghai weekday for A-share court examples.
    local = idx.tz_convert("Asia/Shanghai") if idx.tz is not None else idx
    weekday = pd.Series(local.dayofweek, index=idx, dtype=float)
    ret = c.pct_change()
    monday_return = ret.where(weekday == 0.0, np.nan)
    monday_down = pd.Series(np.nan, index=idx, dtype=float)
    mon_mask = weekday == 0.0
    monday_down.loc[mon_mask] = (monday_return.loc[mon_mask] < 0.0).astype(float)
    return pd.DataFrame(
        {
            "weekday": weekday,
            "monday_return": monday_return,
            "monday_down": monday_down,
        },
        index=idx,
    )


def _infer_bar_timedelta(index: pd.DatetimeIndex) -> pd.Timedelta:
    if len(index) < 2:
        return pd.Timedelta(hours=2)
    diffs = pd.Series(index[1:]).reset_index(drop=True) - pd.Series(index[:-1]).reset_index(
        drop=True
    )
    # median positive diff
    pos = diffs[diffs > pd.Timedelta(0)]
    if pos.empty:
        return pd.Timedelta(hours=2)
    return pd.to_timedelta(pos.median())


def _load_btc_close_at_timeframe(
    data_path: str | Path,
    *,
    timeframe: str,
    start: Optional[pd.Timestamp],
    end: Optional[pd.Timestamp],
) -> pd.Series:
    from src.data_tools.data_utils import load_raw_data

    start_s = start.strftime("%Y-%m-%d") if start is not None else None
    end_s = end.strftime("%Y-%m-%d") if end is not None else None
    df = load_raw_data(
        data_path=str(data_path),
        symbol="BTCUSDT",
        start_date=start_s,
        end_date=end_s,
        timeframe=timeframe,
    )
    close = pd.to_numeric(df["close"], errors="coerce").astype(float)
    close.index = pd.to_datetime(df.index, utc=True)
    close = close.sort_index()
    close = close[~close.index.duplicated(keep="last")]
    return close


@register_feature(
    "compute_btc_prior_bar_return_from_df",
    category="cross_symbol",
    description=(
        "BTCUSDT closed-bar return aligned onto the host symbol bars via "
        "merge_asof(backward). Not btc_roc90 inject."
    ),
    outputs=["btc_prior_bar_return"],
)
def compute_btc_prior_bar_return_from_df(
    df: pd.DataFrame,
    *,
    data_path: str = "data/parquet_data",
    timeframe: str = "120T",
    on_missing: Literal["nan", "raise"] = "nan",
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
    out = pd.DataFrame(
        {"btc_prior_bar_return": np.nan},
        index=df.index,
    )
    try:
        btc_close = _load_btc_close_at_timeframe(
            data_path,
            timeframe=timeframe,
            start=idx.min() - pd.Timedelta(days=7),
            end=idx.max() + pd.Timedelta(days=2),
        )
    except Exception:
        if on_missing == "raise":
            raise
        return out

    btc_ret = btc_close.pct_change()
    # Prior completed BTC bar relative to host bar close: shift 1 on BTC series,
    # then asof-join so alts never see a BTC return that has not closed yet.
    btc_prior = btc_ret.shift(1)
    right = pd.DataFrame(
        {"_ts": btc_prior.index, "btc_prior_bar_return": btc_prior.to_numpy()}
    ).dropna()
    left = pd.DataFrame({"_ts": idx, "_i": np.arange(len(idx), dtype=int)})
    merged = pd.merge_asof(
        left.sort_values("_ts"),
        right.sort_values("_ts"),
        on="_ts",
        direction="backward",
    )
    vals = np.full(len(idx), np.nan, dtype=float)
    vals[merged["_i"].to_numpy()] = merged["btc_prior_bar_return"].to_numpy()
    out["btc_prior_bar_return"] = vals
    return out


@register_feature(
    "compute_bar_max_notional_p99_from_df",
    category="order_flow",
    description=(
        "Within-bar max trade notional vs rolling P99 of that max. "
        "Needs tick parquet; writes NaN when ticks are missing."
    ),
    outputs=[
        "bar_max_notional",
        "bar_max_notional_p99",
        "bar_max_notional_ge_p99",
    ],
)
def compute_bar_max_notional_p99_from_df(
    df: pd.DataFrame,
    *,
    data_path: str = "data/parquet_data",
    lookback: int = 100,
    q: float = 0.99,
    on_missing: Literal["nan", "raise"] = "nan",
    node_cache_version: str | None = None,
) -> pd.DataFrame:
    del node_cache_version
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("df index must be DatetimeIndex")

    cols = ["bar_max_notional", "bar_max_notional_p99", "bar_max_notional_ge_p99"]
    out = pd.DataFrame(np.nan, index=df.index, columns=cols)

    sym_col = (
        "_symbol"
        if "_symbol" in df.columns
        else ("symbol" if "symbol" in df.columns else None)
    )
    if sym_col is None:
        if on_missing == "raise":
            raise KeyError("df needs _symbol/symbol for tick join")
        return out

    root = Path(data_path)
    idx = (
        df.index.tz_localize("UTC")
        if df.index.tz is None
        else df.index.tz_convert("UTC")
    )
    # Bar open ≈ previous index; last bar uses median delta.
    delta = _infer_bar_timedelta(idx)
    opens = idx.to_series().shift(1)
    opens.iloc[0] = idx[0] - delta

    months_needed = {
        f"{ts.year:04d}-{ts.month:02d}" for ts in idx
    } | {f"{ts.year:04d}-{ts.month:02d}" for ts in opens.dropna()}

    for sym in pd.Series(df[sym_col]).astype(str).fillna("").unique():
        if not sym:
            continue
        mask = (pd.Series(df[sym_col]).astype(str) == sym).to_numpy()
        if not mask.any():
            continue
        files = []
        for m in sorted(months_needed):
            fp = root / f"{sym}_{m}.parquet"
            if fp.exists():
                files.append(fp)
        if not files:
            if on_missing == "raise":
                raise FileNotFoundError(f"no ticks for {sym} under {root}")
            continue

        tick_parts: list[pd.DataFrame] = []
        for fp in files:
            try:
                t = pd.read_parquet(fp, columns=["timestamp", "price", "volume"])
            except Exception:
                continue
            if t.empty:
                continue
            t["timestamp"] = pd.to_datetime(t["timestamp"], utc=True)
            tick_parts.append(t)
        if not tick_parts:
            continue
        ticks = pd.concat(tick_parts, ignore_index=True)
        ticks = ticks.dropna(subset=["timestamp", "price", "volume"])
        ticks["timestamp"] = pd.to_datetime(ticks["timestamp"], utc=True)
        ticks["notional"] = (
            pd.to_numeric(ticks["price"], errors="coerce")
            * pd.to_numeric(ticks["volume"], errors="coerce")
        )
        ticks = ticks.dropna(subset=["notional"]).sort_values("timestamp")
        t_times = ticks["timestamp"].to_numpy(dtype="datetime64[ns]")
        t_notional = ticks["notional"].to_numpy(dtype=float)

        max_notional = np.full(mask.sum(), np.nan, dtype=float)
        sub_idx = np.flatnonzero(mask)
        for j, i in enumerate(sub_idx):
            start_ts = opens.iloc[i]
            end_ts = idx[i]
            if pd.isna(start_ts) or pd.isna(end_ts):
                continue
            start_ts = pd.Timestamp(start_ts)
            end_ts = pd.Timestamp(end_ts)
            if start_ts.tzinfo is None:
                start_ts = start_ts.tz_localize("UTC")
            else:
                start_ts = start_ts.tz_convert("UTC")
            if end_ts.tzinfo is None:
                end_ts = end_ts.tz_localize("UTC")
            else:
                end_ts = end_ts.tz_convert("UTC")
            start64 = np.datetime64(start_ts.to_datetime64())
            end64 = np.datetime64(end_ts.to_datetime64())
            left = int(np.searchsorted(t_times, start64, side="left"))
            right = int(np.searchsorted(t_times, end64, side="left"))
            if right <= left:
                continue
            max_notional[j] = float(np.max(t_notional[left:right]))

        s = pd.Series(max_notional, index=df.index[sub_idx])
        p99 = s.rolling(window=int(lookback), min_periods=max(20, lookback // 5)).quantile(
            float(q)
        )
        ge = (s >= p99).astype(float)
        out.loc[s.index, "bar_max_notional"] = s
        out.loc[s.index, "bar_max_notional_p99"] = p99
        out.loc[s.index, "bar_max_notional_ge_p99"] = ge

    return out
