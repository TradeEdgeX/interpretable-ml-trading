"""
使用预测波动率计算自适应R/R标签

这个函数类似于compute_adaptive_rr_label_with_future_vol，但使用预测波动率而不是未来波动率。
适用于实盘交易和回测。
"""

import pandas as pd
import numpy as np
from typing import Optional


def compute_adaptive_rr_label_with_predicted_vol(
    df: pd.DataFrame,
    predicted_vol: np.ndarray,
    signal_col: str = "signal",
    price_col: str = "close",
    atr_col: str = "atr",
    atr_window: int = 14,
    max_holding_bars: int = 50,
    stop_loss_multiplier: float = 1.0,
    take_profit_multiplier: float = 2.0,
    atr_lower_bound: float = 0.8,
    atr_upper_bound: float = 1.5,
    use_breakeven_stop: bool = True,
    entry_price_col: Optional[str] = None,
    entry_offset: int = 0,
) -> pd.Series:
    """
    基于预测波动率的自适应 R/R 标签

    核心思想：
    - 对于每个信号点，使用预测波动率（而不是未来波动率）
    - 使用该波动率动态调整止盈止损：
      - TP = entry ± (预测波动率 × take_profit_multiplier)
      - SL = entry ± (预测波动率 × stop_loss_multiplier)
    - 预测波动率会被限制在 [ATR × atr_lower_bound, ATR × atr_upper_bound] 范围内
    - 如果 use_breakeven_stop=True，当价格达到 stop_loss_multiplier × 预测波动率时，止损上移到保本

    适用于实盘交易和回测。

    Args:
        df: DataFrame with OHLCV data, signals, and ATR
        predicted_vol: Array of predicted volatility values (same length as df)
        signal_col: Column name for trading signals (1=Long, -1=Short, 0=Hold)
        price_col: Column name for price
        atr_col: Column name for ATR
        atr_window: ATR window if ATR column doesn't exist
        max_holding_bars: Maximum holding period
        stop_loss_multiplier: Stop loss multiplier relative to predicted volatility
        take_profit_multiplier: Take profit multiplier relative to predicted volatility
        atr_lower_bound: Lower bound for predicted volatility (as multiple of ATR)
        atr_upper_bound: Upper bound for predicted volatility (as multiple of ATR)
        use_breakeven_stop: If True, move stop loss to breakeven when price reaches stop_loss_multiplier × predicted_vol
        entry_price_col: Column to use for entry price
        entry_offset: Bars to wait after signal before entering

    Returns:
        Series with R/R labels (1=success, 0=failure, NaN=no trade)
    """
    if signal_col not in df.columns:
        return pd.Series(np.nan, index=df.index)

    # Ensure ATR exists
    if atr_col not in df.columns:
        if "high" in df.columns and "low" in df.columns and price_col in df.columns:
            import talib

            high = df["high"].values
            low = df["low"].values
            close = df[price_col].values
            atr_values = talib.ATR(high, low, close, timeperiod=atr_window)
            df[atr_col] = pd.Series(atr_values, index=df.index)
        else:
            if price_col in df.columns:
                df[atr_col] = df[price_col].rolling(window=atr_window).std()
            else:
                return pd.Series(np.nan, index=df.index)

    signals = df[signal_col]
    atr_series = df[atr_col]

    if entry_price_col and entry_price_col in df.columns:
        entry_series = df[entry_price_col]
    elif "open" in df.columns:
        entry_series = df["open"]
    else:
        entry_series = df[price_col]

    if entry_offset > 0:
        entry_prices = entry_series.shift(-entry_offset)
    else:
        entry_prices = entry_series.copy()

    # Clip predicted volatility to ATR bounds
    atr_values = atr_series.values
    final_vol = np.clip(
        predicted_vol,
        atr_values * atr_lower_bound,
        atr_values * atr_upper_bound,
    )

    # Initialize result
    labels = pd.Series(np.nan, index=df.index, dtype=float)

    high_col = "high" if "high" in df.columns else price_col
    low_col = "low" if "low" in df.columns else price_col

    signals_arr = signals.values
    entry_prices_arr = entry_prices.values
    high_arr = df[high_col].values
    low_arr = df[low_col].values

    min_future = max(entry_offset, 1)
    max_i = len(df) - max_holding_bars - min_future

    for i in range(max_i):
        signal = signals_arr[i]

        if pd.isna(signal) or signal == 0:
            continue

        entry_price = entry_prices_arr[i]

        # 确保pred_vol索引有效
        if i >= len(final_vol):
            continue
        pred_vol = final_vol[i]

        if pd.isna(entry_price) or pd.isna(pred_vol) or pred_vol <= 0:
            continue

        # Calculate adaptive stop loss and take profit based on predicted volatility
        if signal > 0:  # Long signal
            initial_stop_loss = entry_price - stop_loss_multiplier * pred_vol
            take_profit = entry_price + take_profit_multiplier * pred_vol
            breakeven_level = entry_price
            breakeven_trigger = entry_price + stop_loss_multiplier * pred_vol
        else:  # Short signal
            initial_stop_loss = entry_price + stop_loss_multiplier * pred_vol
            take_profit = entry_price - take_profit_multiplier * pred_vol
            breakeven_level = entry_price
            breakeven_trigger = entry_price - stop_loss_multiplier * pred_vol

        hit_tp_flag = False
        hit_sl_flag = False
        stop_loss = initial_stop_loss
        breakeven_activated_flag = False

        scan_start = i + max(entry_offset, 1)
        end_idx = min(scan_start + max_holding_bars, len(df))

        for j in range(scan_start, end_idx):
            high = high_arr[j]
            low = low_arr[j]

            if signal > 0:  # Long
                if (
                    use_breakeven_stop
                    and not breakeven_activated_flag
                    and high >= breakeven_trigger
                ):
                    stop_loss = breakeven_level
                    breakeven_activated_flag = True

                if not hit_tp_flag and high >= take_profit:
                    hit_tp_flag = True
                if not hit_sl_flag and low <= stop_loss:
                    hit_sl_flag = True
            else:  # Short
                if (
                    use_breakeven_stop
                    and not breakeven_activated_flag
                    and low <= breakeven_trigger
                ):
                    stop_loss = breakeven_level
                    breakeven_activated_flag = True

                if not hit_tp_flag and low <= take_profit:
                    hit_tp_flag = True
                if not hit_sl_flag and high >= stop_loss:
                    hit_sl_flag = True

            if hit_tp_flag and hit_sl_flag:
                break

        # Determine label
        # 优先TP：如果同时触发TP和SL，优先TP（TP先触发）
        if hit_tp_flag and not hit_sl_flag:
            labels.iloc[i] = 1.0
        elif hit_sl_flag and not hit_tp_flag:
            labels.iloc[i] = 0.0
        elif hit_tp_flag and hit_sl_flag:
            # 同时触发：检查哪个先触发（简化：优先TP）
            labels.iloc[i] = 1.0
        else:
            # 都没有触发（达到最大持仓期）
            labels.iloc[i] = 0.0

    return labels


def compute_adaptive_rr_label_with_predicted_vol_details(
    df: pd.DataFrame,
    predicted_vol: np.ndarray,
    signal_col: str = "signal",
    price_col: str = "close",
    atr_col: str = "atr",
    atr_window: int = 14,
    max_holding_bars: int = 50,
    stop_loss_multiplier: float = 1.0,
    take_profit_multiplier: float = 2.0,
    atr_lower_bound: float = 0.8,
    atr_upper_bound: float = 1.5,
    use_breakeven_stop: bool = True,
    entry_price_col: Optional[str] = None,
    entry_offset: int = 0,
) -> pd.DataFrame:
    """
    基于预测波动率的自适应 R/R 标签（返回详细信息）

    返回 DataFrame 包含以下列：
    - label: R/R 标签（1=成功，0=失败，NaN=无交易）
    - breakeven_activated: 是否触发了保本止损（True/False/NaN）
    - hit_tp: 是否触达止盈（True/False/NaN）
    - hit_sl: 是否触达止损（True/False/NaN）
    - final_result: 最终结果（"win"/"loss"/"breakeven_win"/"breakeven_loss"/NaN）
    - predicted_vol_used: 使用的预测波动率
    - stop_loss_price: 止损价格
    - take_profit_price: 止盈价格
    """
    if signal_col not in df.columns:
        return pd.DataFrame(
            {
                "label": pd.Series(np.nan, index=df.index),
                "breakeven_activated": pd.Series(np.nan, index=df.index),
                "hit_tp": pd.Series(np.nan, index=df.index),
                "hit_sl": pd.Series(np.nan, index=df.index),
                "final_result": pd.Series(np.nan, index=df.index, dtype=object),
                "predicted_vol_used": pd.Series(np.nan, index=df.index),
                "stop_loss_price": pd.Series(np.nan, index=df.index),
                "take_profit_price": pd.Series(np.nan, index=df.index),
            }
        )

    # Ensure ATR exists
    if atr_col not in df.columns:
        if "high" in df.columns and "low" in df.columns and price_col in df.columns:
            import talib

            high = df["high"].values
            low = df["low"].values
            close = df[price_col].values
            atr_values = talib.ATR(high, low, close, timeperiod=atr_window)
            df[atr_col] = pd.Series(atr_values, index=df.index)
        else:
            if price_col in df.columns:
                df[atr_col] = df[price_col].rolling(window=atr_window).std()
            else:
                return pd.DataFrame(
                    {
                        "label": pd.Series(np.nan, index=df.index),
                        "breakeven_activated": pd.Series(np.nan, index=df.index),
                        "hit_tp": pd.Series(np.nan, index=df.index),
                        "hit_sl": pd.Series(np.nan, index=df.index),
                        "final_result": pd.Series(np.nan, index=df.index, dtype=object),
                        "predicted_vol_used": pd.Series(np.nan, index=df.index),
                        "stop_loss_price": pd.Series(np.nan, index=df.index),
                        "take_profit_price": pd.Series(np.nan, index=df.index),
                    }
                )

    signals = df[signal_col]
    atr_series = df[atr_col]

    if entry_price_col and entry_price_col in df.columns:
        entry_series = df[entry_price_col]
    elif "open" in df.columns:
        entry_series = df["open"]
    else:
        entry_series = df[price_col]

    if entry_offset > 0:
        entry_prices = entry_series.shift(-entry_offset)
    else:
        entry_prices = entry_series.copy()

    # Clip predicted volatility to ATR bounds
    atr_values = atr_series.values
    final_vol = np.clip(
        predicted_vol,
        atr_values * atr_lower_bound,
        atr_values * atr_upper_bound,
    )

    # Initialize result
    result = pd.DataFrame(
        {
            "label": pd.Series(np.nan, index=df.index, dtype=float),
            "breakeven_activated": pd.Series(False, index=df.index, dtype=bool),
            "hit_tp": pd.Series(False, index=df.index, dtype=bool),
            "hit_sl": pd.Series(False, index=df.index, dtype=bool),
            "final_result": pd.Series(np.nan, index=df.index, dtype=object),
            "predicted_vol_used": pd.Series(np.nan, index=df.index),
            "stop_loss_price": pd.Series(np.nan, index=df.index),
            "take_profit_price": pd.Series(np.nan, index=df.index),
        }
    )

    high_col = "high" if "high" in df.columns else price_col
    low_col = "low" if "low" in df.columns else price_col

    signals_arr = signals.values
    entry_prices_arr = entry_prices.values
    high_arr = df[high_col].values
    low_arr = df[low_col].values

    min_future = max(entry_offset, 1)
    max_i = len(df) - max_holding_bars - min_future

    for i in range(max_i):
        signal = signals_arr[i]

        if pd.isna(signal) or signal == 0:
            continue

        entry_price = entry_prices_arr[i]

        # 确保pred_vol索引有效
        if i >= len(final_vol):
            continue
        pred_vol = final_vol[i]

        if pd.isna(entry_price) or pd.isna(pred_vol) or pred_vol <= 0:
            continue

        # Calculate adaptive stop loss and take profit based on predicted volatility
        if signal > 0:  # Long signal
            initial_stop_loss = entry_price - stop_loss_multiplier * pred_vol
            take_profit = entry_price + take_profit_multiplier * pred_vol
            breakeven_level = entry_price
            breakeven_trigger = entry_price + stop_loss_multiplier * pred_vol
        else:  # Short signal
            initial_stop_loss = entry_price + stop_loss_multiplier * pred_vol
            take_profit = entry_price - take_profit_multiplier * pred_vol
            breakeven_level = entry_price
            breakeven_trigger = entry_price - stop_loss_multiplier * pred_vol

        hit_tp_flag = False
        hit_sl_flag = False
        stop_loss = initial_stop_loss
        breakeven_activated_flag = False

        scan_start = i + max(entry_offset, 1)
        end_idx = min(scan_start + max_holding_bars, len(df))

        for j in range(scan_start, end_idx):
            high = high_arr[j]
            low = low_arr[j]

            if signal > 0:  # Long
                if (
                    use_breakeven_stop
                    and not breakeven_activated_flag
                    and high >= breakeven_trigger
                ):
                    stop_loss = breakeven_level
                    breakeven_activated_flag = True

                if not hit_tp_flag and high >= take_profit:
                    hit_tp_flag = True
                if not hit_sl_flag and low <= stop_loss:
                    hit_sl_flag = True
            else:  # Short
                if (
                    use_breakeven_stop
                    and not breakeven_activated_flag
                    and low <= breakeven_trigger
                ):
                    stop_loss = breakeven_level
                    breakeven_activated_flag = True

                if not hit_tp_flag and low <= take_profit:
                    hit_tp_flag = True
                if not hit_sl_flag and high >= stop_loss:
                    hit_sl_flag = True

            if hit_tp_flag and hit_sl_flag:
                break

        # Determine result
        # 优先TP：如果同时触发TP和SL，优先TP（TP先触发）
        if hit_tp_flag and not hit_sl_flag:
            result.loc[df.index[i], "label"] = 1.0
            if breakeven_activated_flag:
                result.loc[df.index[i], "final_result"] = "breakeven_win"
            else:
                result.loc[df.index[i], "final_result"] = "win"
        elif hit_sl_flag and not hit_tp_flag:
            result.loc[df.index[i], "label"] = 0.0
            if breakeven_activated_flag:
                result.loc[df.index[i], "final_result"] = "breakeven_loss"
            else:
                result.loc[df.index[i], "final_result"] = "loss"
        elif hit_tp_flag and hit_sl_flag:
            # 同时触发：优先TP（TP先触发）
            result.loc[df.index[i], "label"] = 1.0
            if breakeven_activated_flag:
                result.loc[df.index[i], "final_result"] = "breakeven_win"
            else:
                result.loc[df.index[i], "final_result"] = "win"
        else:
            # 都没有触发（达到最大持仓期）
            result.loc[df.index[i], "label"] = 0.0
            result.loc[df.index[i], "final_result"] = "loss"

        # 记录详细信息
        result.loc[df.index[i], "breakeven_activated"] = breakeven_activated_flag
        result.loc[df.index[i], "hit_tp"] = hit_tp_flag
        result.loc[df.index[i], "hit_sl"] = hit_sl_flag
        result.loc[df.index[i], "predicted_vol_used"] = pred_vol
        result.loc[df.index[i], "stop_loss_price"] = stop_loss
        result.loc[df.index[i], "take_profit_price"] = take_profit

    return result
