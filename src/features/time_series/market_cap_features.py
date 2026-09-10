from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.features.registry import register_feature


def _load_market_cap_daily(symbol: str, market_cap_dir: str) -> pd.Series:
    path = Path(market_cap_dir) / f"{symbol}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Market-cap parquet not found for symbol '{symbol}': {path}. "
            f"Run: python3 scripts/update_market_cap.py --config config/market_cap/market_cap.yaml --symbols {symbol}"
        )
    df = pd.read_parquet(path)
    # Expect either:
    # - index: date (UTC midnight), col: market_cap_usd
    # - columns: date + market_cap_usd
    if isinstance(df.index, pd.DatetimeIndex) and "market_cap_usd" in df.columns:
        s = pd.to_numeric(df["market_cap_usd"], errors="coerce")
        s.index = pd.to_datetime(df.index, utc=True).floor("D")
        return s.sort_index()
    if "date" in df.columns and "market_cap_usd" in df.columns:
        s = pd.to_numeric(df["market_cap_usd"], errors="coerce")
        idx = pd.to_datetime(df["date"], utc=True).floor("D")
        s.index = idx
        return s.sort_index()
    raise ValueError(
        f"Unexpected market-cap schema in {path}. Need market_cap_usd with a date index/column."
    )


def _infer_net_buy_qty(df: pd.DataFrame) -> pd.Series:
    """
    Best-effort "net buy" proxy in base units.

    Preference order:
    1) buy_qty - sell_qty (if present)
    2) cvd_change_1 (if present)
    3) diff(cvd) (if present)
    4) 0
    """
    if "buy_qty" in df.columns and "sell_qty" in df.columns:
        buy = pd.to_numeric(df["buy_qty"], errors="coerce").fillna(0.0)
        sell = pd.to_numeric(df["sell_qty"], errors="coerce").fillna(0.0)
        return buy - sell

    if "cvd_change_1" in df.columns:
        return pd.to_numeric(df["cvd_change_1"], errors="coerce").fillna(0.0)

    if "cvd" in df.columns:
        cvd = pd.to_numeric(df["cvd"], errors="coerce")
        return cvd.diff().fillna(0.0)

    return pd.Series(0.0, index=df.index)


@register_feature(
    "compute_market_cap_normalized_orderflow_from_df",
    category="market_cap",
    description="Market-cap normalized flow proxies: dollar volume / mcap, net-buy-$ / mcap.",
    outputs=[
        "market_cap_usd",
        "dollar_volume_over_mcap",
        "turnover_over_mcap",
        "net_buy_usd_over_mcap",
        "abs_net_buy_usd_over_mcap",
    ],
)
def compute_market_cap_normalized_orderflow_from_df(
    df: pd.DataFrame,
    *,
    market_cap_dir: str = "data/market_cap",
    min_market_cap_usd: float = 1e7,
    on_missing_market_cap: str = "nan",  # nan|zero|raise
    node_cache_version: str | None = None,  # reserved for cache invalidation (preferred)
    cache_version: str | None = None,  # backward compatible alias (do not use for new configs)
) -> pd.DataFrame:
    """
    Attach daily market cap and build normalized flow features.

    Notes:
    - Intended for multi-symbol training: uses df['_symbol'] to load per-symbol market cap.
    - For single-symbol datasets without '_symbol', we try to infer from df['symbol'].
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("df index must be a DatetimeIndex")

    sym_col = "_symbol" if "_symbol" in df.columns else ("symbol" if "symbol" in df.columns else None)
    if sym_col is None:
        raise KeyError("df must contain '_symbol' (preferred) or 'symbol' for market-cap join")

    close = pd.to_numeric(df["close"], errors="coerce").astype(float)
    vol = pd.to_numeric(df["volume"], errors="coerce").astype(float).fillna(0.0)

    dollar_volume = close * vol
    net_buy_qty = _infer_net_buy_qty(df)
    net_buy_usd = net_buy_qty.astype(float) * close

    out = pd.DataFrame(index=df.index)
    out["market_cap_usd"] = np.nan

    # Use UTC dates for joining
    idx_utc = df.index.tz_localize("UTC") if df.index.tz is None else df.index.tz_convert("UTC")

    for sym in pd.Series(df[sym_col]).astype(str).fillna("").unique():
        if not sym:
            continue
        try:
            mcap_daily = _load_market_cap_daily(sym, market_cap_dir=market_cap_dir)
        except Exception:
            if str(on_missing_market_cap).lower() == "raise":
                raise
            mcap_daily = pd.Series(dtype=float)

        mask = pd.Series(df[sym_col]).astype(str) == sym
        dates = idx_utc[mask].floor("D")
        if mcap_daily.empty:
            if str(on_missing_market_cap).lower() == "zero":
                out.loc[mask, "market_cap_usd"] = 0.0
            else:
                out.loc[mask, "market_cap_usd"] = np.nan
            continue

        # align by date:
        # - ffill handles "known cap up to today"
        # - if target dates are BEFORE the first known date (e.g. static snapshot written "as-of today"),
        #   ffill yields NaN; in that case we fill with the earliest available cap as a constant.
        mcap_aligned = mcap_daily.reindex(dates, method="ffill")
        if mcap_aligned.isna().all() and mcap_daily.notna().any():
            mcap_aligned = mcap_aligned.fillna(float(mcap_daily.dropna().iloc[0]))
        else:
            mcap_aligned = mcap_aligned.fillna(method="bfill")
        out.loc[mask, "market_cap_usd"] = mcap_aligned.to_numpy(dtype=float)

    denom = out["market_cap_usd"].astype(float).clip(lower=float(min_market_cap_usd))
    out["dollar_volume_over_mcap"] = (dollar_volume / denom).replace([np.inf, -np.inf], np.nan)
    out["turnover_over_mcap"] = (dollar_volume.abs() / denom).replace([np.inf, -np.inf], np.nan)
    out["net_buy_usd_over_mcap"] = (net_buy_usd / denom).replace([np.inf, -np.inf], np.nan)
    out["abs_net_buy_usd_over_mcap"] = (net_buy_usd.abs() / denom).replace(
        [np.inf, -np.inf], np.nan
    )

    return out


