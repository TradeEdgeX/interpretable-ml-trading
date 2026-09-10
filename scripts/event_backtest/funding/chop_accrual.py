"""Accrue historical funding rates for chop grid coin_m legs (BT-1.3)."""

from __future__ import annotations

from typing import Optional

import pandas as pd

from src.research.inverse_contract_pnl import is_long_side


def aggregated_funding_rate_decimal(
    entry_time: pd.Timestamp,
    exit_time: pd.Timestamp,
    funding_series: pd.Series,
    *,
    side: str,
) -> float:
    """Sum funding rate decimals paid over (entry, exit] for one leg.

    Returned value is passed to ``leg_close_economics(funding_rate=...)``:
    positive means net funding cost to the position.
    """
    if funding_series is None or funding_series.empty:
        return 0.0
    entry = _to_utc(entry_time)
    exit_ = _to_utc(exit_time)
    if exit_ <= entry:
        return 0.0
    series = funding_series.sort_index()
    idx = series.index
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")
    series = pd.Series(series.values, index=idx)
    window = series[(series.index > entry) & (series.index <= exit_)]
    if window.empty:
        return 0.0
    sign = 1.0 if is_long_side(side) else -1.0
    return float((window.astype(float) * sign).sum())


def _to_utc(ts: pd.Timestamp) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        return t.tz_localize("UTC")
    return t.tz_convert("UTC")
