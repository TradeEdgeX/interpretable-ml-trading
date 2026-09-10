"""
压缩区突破策略标签：三元标签（方向+质量）

标签定义：
-1：向下有效突破
0：假突破/回补
+1：向上有效突破

有效突破 = 收盘站稳 + Vol > 20日均值
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional


def compute_compression_breakout_label(
    df: pd.DataFrame,
    price_col: str = "close",
    high_col: str = "high",
    low_col: str = "low",
    volume_col: str = "volume",
    atr_col: str = "atr",
    atr_window: int = 14,
    compression_col: Optional[str] = None,
    lookback_window: int = 10,
    confirmation_bars: int = 3,
    volume_lookback: int = 20,
    min_volume_ratio: float = 1.0,
    breakout_threshold: float = 1.5,
) -> pd.Series:
    """
    计算压缩区突破策略的三元标签

    Args:
        df: DataFrame with OHLCV data
        price_col: Price column
        high_col: High column
        low_col: Low column
        volume_col: Volume column
        atr_col: ATR column
        atr_window: ATR window if ATR column doesn't exist
        compression_col: Compression score column (if None, will compute)
        lookback_window: Window to detect compression
        confirmation_bars: Number of bars to confirm breakout
        volume_lookback: Window for average volume calculation
        min_volume_ratio: Minimum volume ratio for valid breakout
        breakout_threshold: ATR multiplier for breakout detection

    Returns:
        Series with ternary labels (-1, 0, +1, NaN)
    """
    # Ensure ATR exists
    if atr_col not in df.columns:
        if "high" in df.columns and "low" in df.columns and price_col in df.columns:
            try:
                import talib

                high = df["high"].values
                low = df["low"].values
                close = df[price_col].values
                atr_values = talib.ATR(high, low, close, timeperiod=atr_window)
                df[atr_col] = pd.Series(atr_values, index=df.index)
            except ImportError:
                tr = np.maximum(
                    df["high"] - df["low"],
                    np.maximum(
                        abs(df["high"] - df[price_col].shift(1)),
                        abs(df["low"] - df[price_col].shift(1)),
                    ),
                )
                df[atr_col] = tr.rolling(window=atr_window, min_periods=1).mean()
        else:
            if price_col in df.columns:
                df[atr_col] = df[price_col].rolling(window=atr_window).std()
            else:
                return pd.Series(np.nan, index=df.index)

    # Compute compression if not provided
    if compression_col is None or compression_col not in df.columns:
        # Use Bollinger Band width as compression proxy
        if (
            "bb_upper" in df.columns
            and "bb_lower" in df.columns
            and "bb_middle" in df.columns
        ):
            bb_width = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"].replace(
                0, np.nan
            )
            compression_score = 1.0 / (1.0 + bb_width)
        else:
            # Fallback: use ATR normalized by price
            atr_pct = df[atr_col] / df[price_col].replace(0, np.nan)
            compression_score = 1.0 / (1.0 + atr_pct * 100)
        df["compression_score"] = compression_score
        compression_col = "compression_score"

    labels = np.full(len(df), np.nan, dtype=float)

    # Pre-extract arrays
    compression_arr = df[compression_col].values
    atr_arr = df[atr_col].values
    high_arr = df[high_col].values
    low_arr = df[low_col].values
    price_arr = df[price_col].values
    volume_arr = df[volume_col].values if volume_col in df.columns else np.ones(len(df))

    max_i = len(df) - lookback_window - confirmation_bars - 1

    for i in range(lookback_window, max_i):
        # Check if we're in compression
        compression_window = compression_arr[i - lookback_window : i]
        avg_compression = np.nanmean(compression_window)

        if pd.isna(avg_compression) or avg_compression < 0.3:  # Not in compression
            continue

        # Get compression range
        comp_high = np.nanmax(high_arr[i - lookback_window : i])
        comp_low = np.nanmin(low_arr[i - lookback_window : i])
        comp_range = comp_high - comp_low

        if comp_range <= 0:
            continue

        atr = atr_arr[i]
        if pd.isna(atr) or atr <= 0:
            continue

        # Check for breakout in next few bars
        breakout_detected = False
        breakout_direction = 0  # 1 = up, -1 = down

        for j in range(1, min(confirmation_bars + 1, len(df) - i)):
            future_high = high_arr[i + j]
            future_low = low_arr[i + j]
            future_close = price_arr[i + j]
            future_volume = (
                volume_arr[i + j] if len(volume_arr) > i + j else volume_arr[i]
            )

            # Average volume before breakout
            if i >= volume_lookback:
                avg_vol = np.nanmean(volume_arr[i - volume_lookback : i])
            else:
                avg_vol = np.nanmean(volume_arr[:i]) if i > 0 else future_volume

            vol_ratio = future_volume / avg_vol if avg_vol > 0 else 1.0

            # Check for upward breakout
            if future_high > comp_high + breakout_threshold * atr:
                # Volume confirmation
                if vol_ratio >= min_volume_ratio:
                    # Check if price holds (closes above compression high)
                    if future_close > comp_high:
                        breakout_detected = True
                        breakout_direction = 1
                        break

            # Check for downward breakout
            if future_low < comp_low - breakout_threshold * atr:
                # Volume confirmation
                if vol_ratio >= min_volume_ratio:
                    # Check if price holds (closes below compression low)
                    if future_close < comp_low:
                        breakout_detected = True
                        breakout_direction = -1
                        break

        if breakout_detected:
            labels[i] = float(breakout_direction)
        else:
            # Still in compression or fake breakout
            labels[i] = 0.0

    return pd.Series(labels, index=df.index)
