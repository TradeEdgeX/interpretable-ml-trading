"""
SR 突破策略标签：连续标签（实现 R/R）

标签定义：
突破后最大有利偏移 / 最大不利偏移（MFE/MAE），截断在 [0, 3] 区间

动态检查：
一旦触达止损就停止扫描（因为 MAE 已经确定），max_holding_bars 只是寻找上限
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional


def _ensure_atr_inplace(
    work_df: pd.DataFrame,
    *,
    price_col: str,
    atr_col: str,
    atr_window: int,
) -> None:
    if atr_col in work_df.columns:
        return
    if (
        "high" in work_df.columns
        and "low" in work_df.columns
        and price_col in work_df.columns
    ):
        try:
            import talib

            high = work_df["high"].values
            low = work_df["low"].values
            close = work_df[price_col].values
            atr_values = talib.ATR(high, low, close, timeperiod=atr_window)
            work_df[atr_col] = pd.Series(atr_values, index=work_df.index)
            return
        except ImportError:
            tr = np.maximum(
                work_df["high"] - work_df["low"],
                np.maximum(
                    abs(work_df["high"] - work_df[price_col].shift(1)),
                    abs(work_df["low"] - work_df[price_col].shift(1)),
                ),
            )
            work_df[atr_col] = tr.rolling(window=atr_window, min_periods=1).mean()
            return

    # Fallback: rolling std proxy
    if price_col in work_df.columns:
        work_df[atr_col] = work_df[price_col].rolling(window=atr_window).std()
        return


def _auto_generate_signal_inplace(
    work_df: pd.DataFrame,
    *,
    signal_col: str,
    price_col: str,
    atr_col: str,
    signal_horizon: int,
    signal_threshold_atr: float,
) -> None:
    h = max(int(signal_horizon), 1)
    future = work_df[price_col].shift(-h)
    delta = future - work_df[price_col]
    atr_series = work_df[atr_col].replace(0, np.nan)
    thr = float(signal_threshold_atr) * atr_series
    sig = pd.Series(0.0, index=work_df.index)
    sig[delta > thr] = 1.0
    sig[delta < -thr] = -1.0
    work_df[signal_col] = sig


def _compute_sr_breakout_label_one_symbol(
    df: pd.DataFrame,
    *,
    signal_col: str,
    price_col: str,
    high_col: str,
    low_col: str,
    atr_col: str,
    atr_window: int,
    max_holding_bars: int,
    max_rr: float,
    stop_loss_r: float,
    auto_generate_signals: bool,
    signal_horizon: int,
    signal_threshold_atr: float,
) -> pd.Series:
    """Compute labels for a single symbol (no cross-symbol leakage)."""
    # Ensure ATR exists early (used both by auto-signal generation and RR normalization)
    work_df = df.copy()

    _ensure_atr_inplace(
        work_df, price_col=price_col, atr_col=atr_col, atr_window=atr_window
    )
    if atr_col not in work_df.columns:
        return pd.Series(np.nan, index=df.index)

    # Auto-generate signal if missing or effectively empty
    if (signal_col not in work_df.columns) or (
        auto_generate_signals
        and (
            work_df.get(signal_col, pd.Series(0, index=work_df.index))
            .fillna(0)
            .abs()
            .sum()
            == 0
        )
    ):
        if not auto_generate_signals:
            return pd.Series(np.nan, index=work_df.index)

        _auto_generate_signal_inplace(
            work_df,
            signal_col=signal_col,
            price_col=price_col,
            atr_col=atr_col,
            signal_horizon=signal_horizon,
            signal_threshold_atr=signal_threshold_atr,
        )

    signals = work_df[signal_col].values
    atr_arr = work_df[atr_col].values
    high_arr = work_df[high_col].values
    low_arr = work_df[low_col].values
    price_arr = work_df[price_col].values

    labels = np.full(len(df), np.nan, dtype=float)

    # 动态检查：一旦触达止损就停止，不需要等到 max_holding_bars
    max_i = len(df) - max_holding_bars - 1

    for i in range(max_i):
        signal = signals[i]

        if pd.isna(signal) or signal == 0:
            continue

        atr = atr_arr[i]
        if pd.isna(atr) or atr <= 0:
            continue

        # Entry price
        entry_price = price_arr[i]
        stop_loss = (
            entry_price - stop_loss_r * atr
            if signal > 0
            else entry_price + stop_loss_r * atr
        )

        # 动态扫描未来价格路径（最多 max_holding_bars，但一旦触达止损就停止）
        max_favorable = 0.0
        max_adverse = 0.0
        hit_stop_loss = False

        # 扫描范围：从 i+1 到 i+1+max_holding_bars（但不一定扫完）
        end_idx = min(i + 1 + max_holding_bars, len(df))

        for j in range(i + 1, end_idx):
            high = high_arr[j]
            low = low_arr[j]

            if pd.isna(high) or pd.isna(low):
                continue

            if signal > 0:  # Long
                # 检查是否触达止损（使用 low 模拟 intra-bar 执行）
                if not hit_stop_loss and low <= stop_loss:
                    hit_stop_loss = True
                    # 一旦触达止损，MAE 已经确定，立即停止扫描
                    max_adverse = entry_price - low
                    break

                # 更新 MFE 和 MAE
                max_favorable = max(max_favorable, high - entry_price)
                max_adverse = max(max_adverse, entry_price - low)
            else:  # Short
                # 检查是否触达止损（使用 high 模拟 intra-bar 执行）
                if not hit_stop_loss and high >= stop_loss:
                    hit_stop_loss = True
                    # 一旦触达止损，MAE 已经确定，立即停止扫描
                    max_adverse = high - entry_price
                    break

                # 更新 MFE 和 MAE
                max_favorable = max(max_favorable, entry_price - low)
                max_adverse = max(max_adverse, high - entry_price)

        # Calculate R/R: MFE / MAE (normalized by ATR)
        if max_adverse > 0:
            realized_rr = (max_favorable / atr) / (max_adverse / atr)
            # Cap at max_rr
            realized_rr = min(realized_rr, max_rr)
            # Floor at 0
            realized_rr = max(realized_rr, 0.0)
        else:
            # No adverse movement (ideal case)
            realized_rr = max_rr

        labels[i] = realized_rr

    return pd.Series(labels, index=df.index)


def compute_sr_breakout_label(
    df: pd.DataFrame,
    signal_col: str = "signal",
    price_col: str = "close",
    high_col: str = "high",
    low_col: str = "low",
    atr_col: str = "atr",
    atr_window: int = 14,
    max_holding_bars: int = 50,  # 只是寻找上限，实际会动态检查
    max_rr: float = 3.0,
    stop_loss_r: float = 1.0,
    # If the strategy config doesn't provide sparse breakout signals, we can auto-generate
    # a simple directional signal so the label pipeline can run.
    auto_generate_signals: bool = False,
    signal_horizon: int = 1,
    signal_threshold_atr: float = 0.0,
) -> pd.Series:
    """
    计算 SR 突破策略的连续标签（实现 R/R，动态检查）

    Args:
        df: DataFrame with OHLCV data and signals
        signal_col: Signal column (1=Long, -1=Short, 0=Hold)
        price_col: Price column
        high_col: High column
        low_col: Low column
        atr_col: ATR column
        atr_window: ATR window if ATR column doesn't exist
        max_holding_bars: Maximum holding period (只是寻找上限，实际会动态检查)
        max_rr: Maximum R/R to cap (default: 3.0)
        stop_loss_r: Stop loss in R units (default: 1.0)

    Returns:
        Series with continuous R/R labels (0 to max_rr, NaN for no signal)
    """
    # Multi-symbol safety: when df contains multiple symbols (pooled training),
    # compute the label per symbol to avoid cross-symbol look-ahead/leakage.
    sym_col = (
        "_symbol"
        if "_symbol" in df.columns
        else ("symbol" if "symbol" in df.columns else None)
    )

    # If we auto-generate signals, also write them back into df[signal_col] so that
    # backtest/execution can use signal direction consistently in pooled multi-symbol runs.
    if auto_generate_signals:
        if signal_col not in df.columns:
            df[signal_col] = 0.0
        sig_sum = pd.to_numeric(df[signal_col], errors="coerce").fillna(0.0).abs().sum()
        if sig_sum == 0:
            # generate per symbol to avoid leakage
            if sym_col is not None:
                sym_series = pd.Series(df[sym_col]).astype(str)
                for sym in sym_series.fillna("").unique():
                    if not sym:
                        continue
                    mask = (sym_series == sym).to_numpy()
                    if not bool(mask.any()):
                        continue
                    pos = np.flatnonzero(mask)
                    order = np.argsort(df.index.values[mask])
                    pos_sorted = pos[order]
                    g_sorted = df.iloc[pos_sorted].copy()
                    _ensure_atr_inplace(
                        g_sorted,
                        price_col=price_col,
                        atr_col=atr_col,
                        atr_window=atr_window,
                    )
                    if atr_col not in g_sorted.columns:
                        continue
                    _auto_generate_signal_inplace(
                        g_sorted,
                        signal_col=signal_col,
                        price_col=price_col,
                        atr_col=atr_col,
                        signal_horizon=signal_horizon,
                        signal_threshold_atr=signal_threshold_atr,
                    )
                    df.iloc[pos_sorted, df.columns.get_loc(signal_col)] = g_sorted[
                        signal_col
                    ].to_numpy()
            else:
                g = df.sort_index().copy()
                _ensure_atr_inplace(
                    g, price_col=price_col, atr_col=atr_col, atr_window=atr_window
                )
                if atr_col in g.columns:
                    _auto_generate_signal_inplace(
                        g,
                        signal_col=signal_col,
                        price_col=price_col,
                        atr_col=atr_col,
                        signal_horizon=signal_horizon,
                        signal_threshold_atr=signal_threshold_atr,
                    )
                    # Assign back by position since df may be large; order is preserved.
                    df.iloc[:, df.columns.get_loc(signal_col)] = g[
                        signal_col
                    ].to_numpy()
    if sym_col is not None:
        # IMPORTANT: df may have duplicated DatetimeIndex across symbols (pooled panel).
        # Never assign by index labels; assign by boolean mask / positional index to avoid
        # length mismatches with duplicate timestamps.
        out = pd.Series(np.nan, index=df.index, dtype=float)
        sym_series = pd.Series(df[sym_col]).astype(str)
        for sym in sym_series.fillna("").unique():
            if not sym:
                continue
            mask = (sym_series == sym).to_numpy()
            if not bool(mask.any()):
                continue

            # Sort rows within this symbol by timestamp (label uses "future bars" in order).
            pos = np.flatnonzero(mask)
            # argsort on datetime64[ns] values
            order = np.argsort(df.index.values[mask])
            pos_sorted = pos[order]

            g_sorted = df.iloc[pos_sorted]
            labels_sorted = _compute_sr_breakout_label_one_symbol(
                g_sorted,
                signal_col=signal_col,
                price_col=price_col,
                high_col=high_col,
                low_col=low_col,
                atr_col=atr_col,
                atr_window=atr_window,
                max_holding_bars=max_holding_bars,
                max_rr=max_rr,
                stop_loss_r=stop_loss_r,
                auto_generate_signals=auto_generate_signals,
                signal_horizon=signal_horizon,
                signal_threshold_atr=signal_threshold_atr,
            )
            out.iloc[pos_sorted] = labels_sorted.to_numpy()
        return out

    return _compute_sr_breakout_label_one_symbol(
        df.sort_index(),
        signal_col=signal_col,
        price_col=price_col,
        high_col=high_col,
        low_col=low_col,
        atr_col=atr_col,
        atr_window=atr_window,
        max_holding_bars=max_holding_bars,
        max_rr=max_rr,
        stop_loss_r=stop_loss_r,
        auto_generate_signals=auto_generate_signals,
        signal_horizon=signal_horizon,
        signal_threshold_atr=signal_threshold_atr,
    )
