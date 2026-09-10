"""Shared L1 swing + L3 wide_sr price series for trading maps (backtest + CMS)."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def simple_atr14(ohlc: pd.DataFrame, period: int = 14) -> pd.Series:
    """Chart-local ATR for L3 overlay (not FeatureStore parity-critical)."""
    h = ohlc["high"].astype(float)
    l = ohlc["low"].astype(float)
    c = ohlc["close"].astype(float)
    prev = c.shift(1)
    tr = pd.concat([(h - l).abs(), (h - prev).abs(), (l - prev).abs()], axis=1).max(
        axis=1
    )
    return tr.rolling(period, min_periods=max(3, period // 2)).mean()


def compute_sr_overlay_series(
    ohlc: pd.DataFrame, *, l1_lookback: int = 20
) -> pd.DataFrame:
    """L1 swing + L3 wide_sr price series aligned to ``ohlc`` index.

    Columns: ``l1_support``, ``l1_resistance``, ``l3_upper``, ``l3_lower``.
    """
    out = pd.DataFrame(index=ohlc.index)
    if ohlc.empty:
        return out
    hi = ohlc["high"].astype(float)
    lo = ohlc["low"].astype(float)
    cl = ohlc["close"].astype(float)
    lb = max(3, int(l1_lookback))
    out["l1_support"] = lo.rolling(lb, min_periods=3).min()
    out["l1_resistance"] = hi.rolling(lb, min_periods=3).max()
    try:
        from src.features.time_series.baseline_features import (
            compute_wide_sr_swing_from_series,
        )

        atr = simple_atr14(ohlc)
        wide = compute_wide_sr_swing_from_series(high=hi, low=lo, close=cl, atr=atr)
        out["l3_upper"] = wide["wide_sr_upper_px"]
        out["l3_lower"] = wide["wide_sr_lower_px"]
    except Exception as e:
        logger.warning("SR overlay: L3 wide_sr failed: %s", e)
        out["l3_upper"] = np.nan
        out["l3_lower"] = np.nan
    return out


def sr_overlay_points_from_candles(
    candles: list[dict],
    *,
    l1_lookback: int = 20,
) -> dict[str, list[dict]]:
    """Build CMS main-overlay point lists from OHLCV candle dicts (``time`` unix s)."""
    empty = {
        "sr_l1_support": [],
        "sr_l1_resistance": [],
        "sr_l3_upper": [],
        "sr_l3_lower": [],
    }
    if not candles:
        return empty
    idx = pd.to_datetime([int(c["time"]) for c in candles], unit="s", utc=True)
    ohlc = pd.DataFrame(
        {
            "high": [float(c.get("high") or c.get("close") or 0) for c in candles],
            "low": [float(c.get("low") or c.get("close") or 0) for c in candles],
            "close": [float(c.get("close") or 0) for c in candles],
        },
        index=idx,
    )
    series = compute_sr_overlay_series(ohlc, l1_lookback=l1_lookback)
    out: dict[str, list[dict]] = {}
    key_map = {
        "l1_support": "sr_l1_support",
        "l1_resistance": "sr_l1_resistance",
        "l3_upper": "sr_l3_upper",
        "l3_lower": "sr_l3_lower",
    }
    for col, key in key_map.items():
        pts: list[dict] = []
        if col not in series.columns:
            out[key] = pts
            continue
        for t, v in series[col].items():
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            if fv != fv:
                continue
            pts.append({"time": int(pd.Timestamp(t).timestamp()), "value": fv})
        out[key] = pts
    return out
