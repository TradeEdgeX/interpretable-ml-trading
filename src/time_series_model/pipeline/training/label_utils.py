"""
Utility functions for constructing training labels and applying target transforms.

Centralises the logic for:
- Log-magnitude targets used by the return regression model.
- Rolling quantile classification labels with look-ahead protection.
- Rolling RMS volatility proxy without leaking future information.
- Volatility-normalized targets (Sharpe-like targets).
- Historical quantile labels for evaluation and signal generation.
- Tradable mask for filtering low-quality samples.
- Trend strength as sample weights.
"""

from __future__ import annotations

from typing import Tuple, Optional

import numpy as np
import pandas as pd


def _ensure_atr(
    df: pd.DataFrame,
    atr_col: str,
    price_col: str,
    high_col: str,
    low_col: str,
    atr_window: int,
) -> pd.Series:
    """Return an ATR series, computing it if missing."""
    if atr_col in df.columns:
        atr_series = df[atr_col].copy()
    else:
        if not {price_col, high_col, low_col}.issubset(df.columns):
            return pd.Series(np.nan, index=df.index)

        try:
            import talib

            atr_values = talib.ATR(
                df[high_col].values,
                df[low_col].values,
                df[price_col].values,
                timeperiod=atr_window,
            )
            atr_series = pd.Series(atr_values, index=df.index)
        except ImportError:
            tr = np.maximum(
                df[high_col] - df[low_col],
                np.maximum(
                    (df[high_col] - df[price_col].shift(1)).abs(),
                    (df[low_col] - df[price_col].shift(1)).abs(),
                ),
            )
            atr_series = tr.rolling(window=atr_window, min_periods=1).mean()

    return atr_series.ffill()


def log_return_magnitude(y_return: pd.Series) -> pd.Series:
    """Project raw returns into log-amplitude space (>=0)."""
    log_mag = np.log1p(np.abs(y_return.to_numpy()))
    return pd.Series(log_mag, index=y_return.index, name="log_return_magnitude")


def invert_log_return_magnitude(values: np.ndarray | pd.Series) -> np.ndarray:
    """Invert log-amplitude predictions back to absolute return magnitude."""
    arr = np.asarray(values, dtype=float)
    arr = np.maximum(arr, 0.0)  # clip tiny negatives introduced by the model
    return np.expm1(arr)


def historical_rolling_volatility(
    y_return: pd.Series,
    window: int = 60,
    min_periods: int = 60,
) -> pd.Series:
    """
    Compute historical rolling RMS volatility (for use as a FEATURE).

    This function computes trailing volatility over the past window periods.
    At time t, it returns: RMS(r_{t-window+1}, r_{t-window+2}, ..., r_t)

    ✅ This is safe to use as a FEATURE because it only uses historical data.
    For future volatility LABELS, use `future_volatility_label()` instead.

    Args:
        y_return: Historical return series (e.g., from price.pct_change())
        window: Rolling window size
        min_periods: Minimum periods required for calculation

    Returns:
        Historical rolling RMS volatility series (suitable for features)
    """

    def _rms(x: np.ndarray) -> float:
        return float(np.sqrt(np.mean(np.square(x)))) if len(x) else np.nan

    rms = y_return.rolling(window=window, min_periods=min_periods).apply(_rms, raw=True)
    # Fallback to |r| when insufficient history is available
    rms = rms.fillna(np.abs(y_return))
    rms.name = "rolling_vol"
    return rms


# Backward compatibility alias
def rolling_rms_volatility(
    y_return: pd.Series,
    window: int = 60,
    min_periods: int = 60,
) -> pd.Series:
    """
    Deprecated alias for historical_rolling_volatility().

    Use historical_rolling_volatility() for clarity.
    """
    return historical_rolling_volatility(y_return, window, min_periods)


def future_volatility_label(
    price_series: pd.Series,
    horizon: int = 24,
    min_periods: Optional[int] = None,
) -> pd.Series:
    """
    Compute future volatility label: RMS of future single-period returns.

    This function computes the realized volatility over the future H periods
    as a supervision target. At time t, the label is:
        vol[t] = RMS(r_{t+1}, r_{t+2}, ..., r_{t+H})
    where r_{t+i} = (price[t+i] - price[t+i-1]) / price[t+i-1]

    ⚠️  This uses future information, which is correct for LABELS but would
    cause leakage if used as a FEATURE.

    Args:
        price_series: Price series (e.g., 'close')
        horizon: Number of future periods to compute volatility over
        min_periods: Minimum periods required (default: horizon)

    Returns:
        Series with future volatility labels
    """
    if min_periods is None:
        min_periods = horizon

    returns = price_series.pct_change()
    future_returns = []
    for i in range(len(price_series)):
        if i + horizon > len(price_series):
            future_returns.append(np.nan)
        else:
            future_window = returns.iloc[i + 1 : i + horizon + 1]
            if len(future_window) >= min_periods:
                rms = np.sqrt(np.mean(np.square(future_window.dropna())))
                future_returns.append(rms)
            else:
                future_returns.append(np.nan)

    return pd.Series(future_returns, index=price_series.index, name="future_volatility")


def historical_quantile_label(
    future_return: pd.Series,
    lookback_window: int = 60,
    hold_period: int = 5,
    min_samples: int = 30,
    asset_col: Optional[pd.Series] = None,
) -> pd.Series:
    """
    Compute historical quantile label: percentile rank of future return within
    historical distribution of returns.

    This is a robust, non-parametric label that:
    - Normalizes returns across different assets and time periods
    - Reduces sensitivity to outliers
    - Works well with tree-based models (LightGBM, XGBoost)

    At time t, the label is:
        quantile[t] = percentile_rank(future_return[t], historical_returns[t-window:t])

    Args:
        future_return: Future return series (e.g., from price.pct_change().shift(-hold_period))
        lookback_window: Number of historical periods to use for quantile calculation
        hold_period: Number of periods to hold (for computing future return)
        min_samples: Minimum number of samples required for quantile calculation
        asset_col: Optional asset identifier column (for cross-sectional ranking)

    Returns:
        Series with quantile labels (0-1 range)
    """
    result = pd.Series(np.nan, index=future_return.index, name="return_quantile")

    if asset_col is not None:
        # Cross-sectional ranking: rank within each asset group
        for asset in asset_col.unique():
            asset_mask = asset_col == asset
            asset_returns = future_return[asset_mask]
            if len(asset_returns) < min_samples:
                continue

            for i in range(lookback_window, len(asset_returns)):
                historical = asset_returns.iloc[i - lookback_window : i]
                if len(historical.dropna()) < min_samples:
                    continue

                current_return = asset_returns.iloc[i]
                if pd.isna(current_return):
                    continue

                quantile = (historical < current_return).sum() / len(historical)
                result.iloc[asset_returns.index[i]] = quantile
    else:
        # Time-series ranking: rank within historical window
        for i in range(lookback_window, len(future_return)):
            historical = future_return.iloc[i - lookback_window : i]
            if len(historical.dropna()) < min_samples:
                continue

            current_return = future_return.iloc[i]
            if pd.isna(current_return):
                continue

            quantile = (historical < current_return).sum() / len(historical)
            result.iloc[i] = quantile

    return result.rename("return_quantile")


def compute_rr_label(
    df: pd.DataFrame,
    signal_col: str = "signal",
    price_col: str = "close",
    atr_col: str = "atr",
    atr_window: int = 14,
    rr_ratio: float = 2.0,
    max_holding_bars: int = 24,
    stop_loss_r: float = 1.0,
    take_profit_r: float = 2.0,
    use_continuous_label: bool = False,
    entry_price_col: Optional[str] = None,
    entry_offset: int = 0,
    use_breakeven_stop: bool = False,  # 新增参数：是否使用保本止损
) -> pd.Series:
    """
    计算基于风险回报比（R/R）的标签，匹配 SR 策略的真实交易逻辑。

    策略逻辑：
    - 在高概率区域入场（由信号决定方向）
    - 设定 1R 止损
    - 捕捉至少 2R 的有利运行
    - 动态止盈止损
    - 可选：当价格达到 1R 时，止损上移到保本（use_breakeven_stop=True）

    标签计算：
    - 二元标签（use_continuous_label=False）：是否实现 ≥2R（1=成功，0=失败）
    - 连续标签（use_continuous_label=True）：实际实现的盈亏比（Realized R/R）

    Args:
        df: DataFrame with OHLCV data, signals, and ATR
        signal_col: Column name for trading signals (1=Long, -1=Short, 0=Hold)
        price_col: Column name for price (used for entry price)
        atr_col: Column name for ATR
        atr_window: ATR window if ATR column doesn't exist
        rr_ratio: Target risk-reward ratio (default: 2.0)
        max_holding_bars: Maximum holding period
        stop_loss_r: Stop loss in R units (default: 1.0)
        take_profit_r: Take profit in R units (default: 2.0)
        use_continuous_label: If True, return realized R/R; if False, return binary label
        entry_price_col: Column to use for entry price. Default: 'open' if available else price_col.
        entry_offset: Bars to wait after signal before entering (>=0). entry_offset=1 = next bar entry.
        use_breakeven_stop: If True, move stop loss to breakeven when price reaches 1R

    Returns:
        Series with R/R labels (binary or continuous)
    """
    if signal_col not in df.columns:
        # If no signal column, return NaN series
        return pd.Series(np.nan, index=df.index)

    # Ensure ATR exists
    if atr_col not in df.columns:
        # Compute ATR if not available
        if "high" in df.columns and "low" in df.columns and price_col in df.columns:
            import talib

            high = df["high"].values
            low = df["low"].values
            close = df[price_col].values
            atr_values = talib.ATR(high, low, close, timeperiod=atr_window)
            df[atr_col] = pd.Series(atr_values, index=df.index)
        else:
            # Fallback: use a simple volatility proxy
            if price_col in df.columns:
                df[atr_col] = df[price_col].rolling(window=atr_window).std()
            else:
                return pd.Series(np.nan, index=df.index)

    signals = df[signal_col]
    atr_series = df[atr_col]

    if entry_offset < 0:
        raise ValueError("entry_offset must be >= 0")

    if entry_price_col and entry_price_col not in df.columns:
        raise ValueError(
            f"entry_price_col='{entry_price_col}' not found in DataFrame columns"
        )

    if entry_price_col:
        entry_series = df[entry_price_col]
    elif "open" in df.columns:
        entry_series = df["open"]
    else:
        entry_series = df[price_col]

    if entry_offset > 0:
        entry_prices = entry_series.shift(-entry_offset)
    else:
        # For offset 0, still shift by 0 (current bar price)
        entry_prices = entry_series.copy()

    # Initialize labels with NaN (will be set to 0.0 for valid samples)
    labels = pd.Series(np.nan, index=df.index, dtype=float)

    # Get high/low columns for efficient access
    high_col = "high" if "high" in df.columns else price_col
    low_col = "low" if "low" in df.columns else price_col

    # Pre-extract arrays for faster access (avoid repeated iloc calls)
    signals_arr = signals.values
    entry_prices_arr = entry_prices.values
    atr_arr = atr_series.values
    high_arr = df[high_col].values
    low_arr = df[low_col].values

    # Process each sample
    min_future = max(entry_offset, 1)
    max_i = len(df) - max_holding_bars - min_future

    for i in range(max_i):
        signal = signals_arr[i]

        # Skip if no signal
        if pd.isna(signal) or signal == 0:
            continue

        entry_price = entry_prices_arr[i]
        atr = atr_arr[i]

        # Skip if invalid entry price or ATR
        if pd.isna(entry_price) or pd.isna(atr) or atr <= 0:
            continue

        # Calculate stop loss and take profit
        if signal > 0:  # Long signal
            initial_stop_loss = entry_price - stop_loss_r * atr
            take_profit = entry_price + take_profit_r * atr
            breakeven_level = entry_price  # 保本价
            breakeven_trigger = (
                entry_price + stop_loss_r * atr
            )  # 当价格达到1R时，止损上移到保本
        else:  # Short signal
            initial_stop_loss = entry_price + stop_loss_r * atr
            take_profit = entry_price - take_profit_r * atr
            breakeven_level = entry_price  # 保本价
            breakeven_trigger = (
                entry_price - stop_loss_r * atr
            )  # 当价格达到1R时，止损上移到保本

        # Scan future price path (up to max_holding_bars)
        # This is the core logic: check if TP or SL is hit first
        hit_tp = False
        hit_sl = False
        tp_bar = None
        sl_bar = None
        max_favorable = 0.0
        max_adverse = 0.0
        stop_loss = initial_stop_loss  # 当前止损（可能上移到保本）
        breakeven_activated = False  # 是否已激活保本止损

        # Scan from first tradable bar after entry
        scan_start = i + max(entry_offset, 1)
        end_idx = min(scan_start + max_holding_bars, len(df))

        for j in range(scan_start, end_idx):
            high = high_arr[j]
            low = low_arr[j]

            if signal > 0:  # Long
                # 检查是否达到1R，如果是，将止损上移到保本
                if (
                    use_breakeven_stop
                    and not breakeven_activated
                    and high >= breakeven_trigger
                ):
                    stop_loss = breakeven_level  # 止损上移到保本
                    breakeven_activated = True

                # Check if TP or SL hit (using high/low to simulate intra-bar execution)
                if not hit_tp and high >= take_profit:
                    hit_tp = True
                    tp_bar = j - i  # Bar number relative to entry
                if not hit_sl and low <= stop_loss:
                    hit_sl = True
                    sl_bar = j - i  # Bar number relative to entry

                # Track MFE/MAE for continuous label
                if use_continuous_label:
                    max_favorable = max(max_favorable, (high - entry_price) / atr)
                    max_adverse = max(max_adverse, (entry_price - low) / atr)
            else:  # Short
                # 检查是否达到1R，如果是，将止损上移到保本
                if (
                    use_breakeven_stop
                    and not breakeven_activated
                    and low <= breakeven_trigger
                ):
                    stop_loss = breakeven_level  # 止损上移到保本
                    breakeven_activated = True

                # Check if TP or SL hit
                if not hit_tp and low <= take_profit:
                    hit_tp = True
                    tp_bar = j - i
                if not hit_sl and high >= stop_loss:
                    hit_sl = True
                    sl_bar = j - i

                # Track MFE/MAE for continuous label
                if use_continuous_label:
                    max_favorable = max(max_favorable, (entry_price - low) / atr)
                    max_adverse = max(max_adverse, (high - entry_price) / atr)

            # Early exit: if both hit, check which came first
            if hit_tp and hit_sl:
                break

        # Calculate label
        if use_continuous_label:
            # Continuous label: Realized R/R
            # MFE / MAE, truncated at [0, take_profit_r]
            if max_adverse > 0:
                realized_rr = min(max_favorable, take_profit_r) / max(max_adverse, 0.1)
                realized_rr = min(realized_rr, take_profit_r)  # Cap at target R/R
            else:
                realized_rr = 0.0
            labels.iloc[i] = realized_rr
        else:
            # Binary label: Did we achieve ≥2R?
            # Success = hit TP before SL (or hit TP and never hit SL)
            if hit_tp and (
                not hit_sl
                or (tp_bar is not None and sl_bar is not None and tp_bar < sl_bar)
            ):
                labels.iloc[i] = 1.0  # Success: hit TP before SL
            else:
                labels.iloc[i] = (
                    0.0  # Failure: hit SL first, or timeout without hitting TP
                )

    return labels


def compute_rr_label_with_details(
    df: pd.DataFrame,
    signal_col: str = "signal",
    price_col: str = "close",
    atr_col: str = "atr",
    atr_window: int = 14,
    rr_ratio: float = 2.0,
    max_holding_bars: int = 24,
    stop_loss_r: float = 1.0,
    take_profit_r: float = 2.0,
    use_continuous_label: bool = False,
    entry_price_col: Optional[str] = None,
    entry_offset: int = 0,
    use_breakeven_stop: bool = False,
) -> pd.DataFrame:
    """
    计算基于风险回报比（R/R）的标签，并返回详细信息。

    返回 DataFrame 包含以下列：
    - label: R/R 标签（1=成功，0=失败，NaN=无交易）
    - breakeven_activated: 是否触发了保本止损（True/False/NaN）
    - hit_tp: 是否触达止盈（True/False/NaN）
    - hit_sl: 是否触达止损（True/False/NaN）
    - final_result: 最终结果（"win"/"loss"/"breakeven"/NaN）

    Args:
        与 compute_rr_label 相同

    Returns:
        DataFrame with detailed trade information
    """
    if signal_col not in df.columns:
        return pd.DataFrame(
            {
                "label": pd.Series(np.nan, index=df.index),
                "breakeven_activated": pd.Series(np.nan, index=df.index),
                "hit_tp": pd.Series(np.nan, index=df.index),
                "hit_sl": pd.Series(np.nan, index=df.index),
                "final_result": pd.Series(np.nan, index=df.index),
            }
        )

    # 使用 compute_rr_label 的逻辑，但记录详细信息
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
                        "final_result": pd.Series(np.nan, index=df.index),
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

    # Initialize result columns
    labels = pd.Series(np.nan, index=df.index, dtype=float)
    breakeven_activated = pd.Series(np.nan, index=df.index, dtype=bool)
    hit_tp = pd.Series(np.nan, index=df.index, dtype=bool)
    hit_sl = pd.Series(np.nan, index=df.index, dtype=bool)
    final_result = pd.Series(np.nan, index=df.index, dtype=object)

    high_col = "high" if "high" in df.columns else price_col
    low_col = "low" if "low" in df.columns else price_col

    signals_arr = signals.values
    entry_prices_arr = entry_prices.values
    atr_arr = atr_series.values
    high_arr = df[high_col].values
    low_arr = df[low_col].values

    min_future = max(entry_offset, 1)
    max_i = len(df) - max_holding_bars - min_future

    for i in range(max_i):
        signal = signals_arr[i]

        if pd.isna(signal) or signal == 0:
            continue

        entry_price = entry_prices_arr[i]
        atr = atr_arr[i]

        if pd.isna(entry_price) or pd.isna(atr) or atr <= 0:
            continue

        if signal > 0:  # Long signal
            initial_stop_loss = entry_price - stop_loss_r * atr
            take_profit = entry_price + take_profit_r * atr
            breakeven_level = entry_price
            breakeven_trigger = entry_price + stop_loss_r * atr
        else:  # Short signal
            initial_stop_loss = entry_price + stop_loss_r * atr
            take_profit = entry_price - take_profit_r * atr
            breakeven_level = entry_price
            breakeven_trigger = entry_price - stop_loss_r * atr

        hit_tp_flag = False
        hit_sl_flag = False
        tp_bar = None
        sl_bar = None
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
                    tp_bar = j - i
                if not hit_sl_flag and low <= stop_loss:
                    hit_sl_flag = True
                    sl_bar = j - i
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
                    tp_bar = j - i
                if not hit_sl_flag and high >= stop_loss:
                    hit_sl_flag = True
                    sl_bar = j - i

            if hit_tp_flag and hit_sl_flag:
                break

        # 计算标签和最终结果
        # ✅ 修复：优先判断 breakeven 情况
        if breakeven_activated_flag:
            # 如果触发了 breakeven stop，止损在入场价，不会亏
            if hit_tp_flag:
                # 保本后盈利
                labels.iloc[i] = 1.0
                final_result.iloc[i] = "breakeven_win"
            elif hit_sl_flag:
                # 保本后亏损（价格回到入场价以下/以上，触达保本止损）
                labels.iloc[i] = 0.0
                final_result.iloc[i] = "breakeven_loss"
            else:
                # 超时，但没有触达止损（止损在入场价，不会亏）
                labels.iloc[i] = 0.0  # 超时算失败，但不会亏
                final_result.iloc[i] = "breakeven_win"  # 保本，不会亏
        elif hit_tp_flag and (
            not hit_sl_flag
            or (tp_bar is not None and sl_bar is not None and tp_bar < sl_bar)
        ):
            # 没有触发 breakeven stop，直接盈利
            labels.iloc[i] = 1.0
            final_result.iloc[i] = "win"
        else:
            # 没有触发 breakeven stop，直接亏损
            labels.iloc[i] = 0.0
            final_result.iloc[i] = "loss"

        breakeven_activated.iloc[i] = breakeven_activated_flag
        hit_tp.iloc[i] = hit_tp_flag
        hit_sl.iloc[i] = hit_sl_flag

    return pd.DataFrame(
        {
            "label": labels,
            "breakeven_activated": breakeven_activated,
            "hit_tp": hit_tp,
            "hit_sl": hit_sl,
            "final_result": final_result,
        }
    )


def compute_adaptive_rr_label_with_future_vol(
    df: pd.DataFrame,
    signal_col: str = "signal",
    price_col: str = "close",
    atr_col: str = "atr",
    atr_window: int = 14,
    max_holding_bars: int = 50,
    stop_loss_multiplier: float = 1.0,
    take_profit_multiplier: float = 2.0,
    volatility_window: int = 10,
    use_breakeven_stop: bool = True,
    entry_price_col: Optional[str] = None,
    entry_offset: int = 0,
) -> pd.Series:
    """
    基于未来波动率的自适应 R/R 标签

    核心思想：
    - 对于每个信号点，计算未来 volatility_window 内的实际波动率
    - 使用该波动率动态调整止盈止损：
      - TP = entry ± (未来波动率 × take_profit_multiplier)
      - SL = entry ± (未来波动率 × stop_loss_multiplier)
    - 如果 use_breakeven_stop=True，当价格达到 stop_loss_multiplier × 未来波动率时，止损上移到保本

    注意：此函数使用未来波动率，仅适用于标签生成阶段，不适用于实盘交易。

    Args:
        df: DataFrame with OHLCV data, signals, and ATR
        signal_col: Column name for trading signals (1=Long, -1=Short, 0=Hold)
        price_col: Column name for price
        atr_col: Column name for ATR
        atr_window: ATR window if ATR column doesn't exist
        max_holding_bars: Maximum holding period
        stop_loss_multiplier: Stop loss multiplier relative to future volatility
        take_profit_multiplier: Take profit multiplier relative to future volatility
        volatility_window: Window size for calculating future volatility
        use_breakeven_stop: If True, move stop loss to breakeven when price reaches stop_loss_multiplier × future_vol
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
    prices = df[price_col]

    if entry_price_col and entry_price_col in df.columns:
        entry_series = df[entry_price_col]
    elif "open" in df.columns:
        entry_series = df["open"]
    else:
        entry_series = prices

    if entry_offset > 0:
        entry_prices = entry_series.shift(-entry_offset)
    else:
        entry_prices = entry_series.copy()

    labels = pd.Series(np.nan, index=df.index, dtype=float)

    high_col = "high" if "high" in df.columns else price_col
    low_col = "low" if "low" in df.columns else price_col

    signals_arr = signals.values
    entry_prices_arr = entry_prices.values
    prices_arr = prices.values
    high_arr = df[high_col].values
    low_arr = df[low_col].values

    # 计算未来波动率（使用未来窗口）
    returns = prices.pct_change()
    future_volatility = (
        returns.rolling(window=volatility_window, min_periods=1)
        .std()
        .shift(-volatility_window)
    )  # 使用未来窗口的波动率

    # 如果未来波动率为 NaN，使用当前 ATR 作为代理
    future_vol_arr = future_volatility.fillna(atr_series / prices).values

    min_future = max(entry_offset, 1)
    max_i = len(df) - max_holding_bars - min_future

    for i in range(max_i):
        signal = signals_arr[i]

        if pd.isna(signal) or signal == 0:
            continue

        entry_price = entry_prices_arr[i]
        future_vol = future_vol_arr[i]

        if pd.isna(entry_price) or pd.isna(future_vol) or future_vol <= 0:
            continue

        # 使用未来波动率计算止盈止损
        if signal > 0:  # Long
            initial_stop_loss = (
                entry_price - stop_loss_multiplier * future_vol * entry_price
            )
            take_profit = (
                entry_price + take_profit_multiplier * future_vol * entry_price
            )
            breakeven_level = entry_price
            breakeven_trigger = (
                entry_price + stop_loss_multiplier * future_vol * entry_price
            )
        else:  # Short
            initial_stop_loss = (
                entry_price + stop_loss_multiplier * future_vol * entry_price
            )
            take_profit = (
                entry_price - take_profit_multiplier * future_vol * entry_price
            )
            breakeven_level = entry_price
            breakeven_trigger = (
                entry_price - stop_loss_multiplier * future_vol * entry_price
            )

        stop_loss = initial_stop_loss
        breakeven_activated = False
        hit_tp = False
        hit_sl = False
        tp_bar = None
        sl_bar = None

        scan_start = i + max(entry_offset, 1)
        end_idx = min(scan_start + max_holding_bars, len(df))

        for j in range(scan_start, end_idx):
            high = high_arr[j]
            low = low_arr[j]

            if signal > 0:  # Long
                if (
                    use_breakeven_stop
                    and not breakeven_activated
                    and high >= breakeven_trigger
                ):
                    stop_loss = breakeven_level
                    breakeven_activated = True

                if not hit_tp and high >= take_profit:
                    hit_tp = True
                    tp_bar = j - i
                if not hit_sl and low <= stop_loss:
                    hit_sl = True
                    sl_bar = j - i
            else:  # Short
                if (
                    use_breakeven_stop
                    and not breakeven_activated
                    and low <= breakeven_trigger
                ):
                    stop_loss = breakeven_level
                    breakeven_activated = True

                if not hit_tp and low <= take_profit:
                    hit_tp = True
                    tp_bar = j - i
                if not hit_sl and high >= stop_loss:
                    hit_sl = True
                    sl_bar = j - i

            if hit_tp and hit_sl:
                break

        # 计算标签
        if hit_tp and (
            not hit_sl
            or (tp_bar is not None and sl_bar is not None and tp_bar < sl_bar)
        ):
            labels.iloc[i] = 1.0
        else:
            labels.iloc[i] = 0.0

    return labels


def simulate_rr_exits(
    df: pd.DataFrame,
    signal_col: str = "signal",
    price_col: str = "close",
    atr_col: str = "atr",
    atr_window: int = 14,
    max_holding_bars: int = 24,
    stop_loss_r: float = 1.0,
    take_profit_r: float = 2.0,
    entry_price_col: Optional[str] = None,
    entry_offset: int = 0,
    use_breakeven_stop: bool = False,  # 新增参数：是否使用保本止损
    use_time_exit: bool = True,  # 新增参数：是否启用“最长持仓”超时平仓
    use_trailing_stop: bool = False,  # 新增参数：是否启用 ATR 移动止损
    trailing_atr_mult: float = 1.0,  # 新增参数：移动止损距离（ATR 倍数）
) -> tuple[pd.Series, pd.Series]:
    """
    使用与 compute_rr_label 相同的 R/R 扫描逻辑，返回每笔交易的平仓位置。

    该函数用于回测场景：给定一列方向信号（1=多, -1=空, 0=不交易），基于 ATR 计算：
    - 1R 止损
    - take_profit_r * R 止盈（默认 2R）
    - 最长持仓 max_holding_bars
    - 可选：当价格达到 1R 时，止损上移到保本

    并为每个入场点找到第一个触达 TP/SL 的 bar（若都未触达，则在观察窗口末尾超时平仓），
    返回两个布尔 Series：
    - long_exits[i]  在多头仓位的平仓 bar 处为 True
    - short_exits[i] 在空头仓位的平仓 bar 处为 True

    注意：该函数不负责计算盈亏，只负责确定平仓时刻，便于与 vectorbt.from_signals 配合。
    """

    if signal_col not in df.columns:
        return (
            pd.Series(False, index=df.index),
            pd.Series(False, index=df.index),
        )

    work_df = df.copy()

    # Ensure ATR exists (重用 compute_rr_label 中的逻辑)
    if atr_col not in work_df.columns:
        if (
            "high" in work_df.columns
            and "low" in work_df.columns
            and price_col in work_df.columns
        ):
            import talib

            high = work_df["high"].values
            low = work_df["low"].values
            close = work_df[price_col].values
            atr_values = talib.ATR(high, low, close, timeperiod=atr_window)
            work_df[atr_col] = pd.Series(atr_values, index=work_df.index)
        else:
            if price_col in work_df.columns:
                work_df[atr_col] = work_df[price_col].rolling(window=atr_window).std()
            else:
                return (
                    pd.Series(False, index=work_df.index),
                    pd.Series(False, index=work_df.index),
                )

    signals = work_df[signal_col]

    if entry_price_col and entry_price_col in work_df.columns:
        entry_series = work_df[entry_price_col]
    elif "open" in work_df.columns:
        entry_series = work_df["open"]
    else:
        entry_series = work_df[price_col]

    if entry_offset > 0:
        entry_prices = entry_series.shift(-entry_offset)
    else:
        entry_prices = entry_series.copy()

    long_exits = pd.Series(False, index=work_df.index)
    short_exits = pd.Series(False, index=work_df.index)

    high_col = "high" if "high" in work_df.columns else price_col
    low_col = "low" if "low" in work_df.columns else price_col

    signals_arr = signals.values
    entry_prices_arr = entry_prices.values
    atr_arr = work_df[atr_col].values
    high_arr = work_df[high_col].values
    low_arr = work_df[low_col].values

    min_future = max(entry_offset, 1)
    # If time-exit is disabled, allow entries all the way to the end (we'll force-close at dataset end).
    max_i = (
        len(work_df) - max_holding_bars - min_future
        if use_time_exit
        else len(work_df) - min_future
    )

    for i in range(max_i):
        signal = signals_arr[i]

        if pd.isna(signal) or signal == 0:
            continue

        entry_price = entry_prices_arr[i]
        atr = atr_arr[i]

        if pd.isna(entry_price) or pd.isna(atr) or atr <= 0:
            continue

        if signal > 0:
            initial_stop_loss = entry_price - stop_loss_r * atr
            take_profit = entry_price + take_profit_r * atr
            breakeven_level = entry_price
            breakeven_trigger = entry_price + stop_loss_r * atr
        else:
            initial_stop_loss = entry_price + stop_loss_r * atr
            take_profit = entry_price - take_profit_r * atr
            breakeven_level = entry_price
            breakeven_trigger = entry_price - stop_loss_r * atr

        stop_loss = initial_stop_loss
        breakeven_activated = False

        scan_start = i + max(entry_offset, 1)
        end_idx = (
            min(scan_start + max_holding_bars, len(work_df))
            if use_time_exit
            else len(work_df)
        )

        exit_idx = None
        hit_tp = False
        hit_sl = False

        for j in range(scan_start, end_idx):
            high = high_arr[j]
            low = low_arr[j]
            atr_j = atr_arr[j]

            # ATR trailing stop: move stop in the direction of profit, never loosen it.
            # Long: stop = max(stop, high - trailing_atr_mult * atr)
            # Short: stop = min(stop, low + trailing_atr_mult * atr)
            if use_trailing_stop and (not pd.isna(atr_j)) and atr_j > 0:
                if signal > 0:
                    cand = high - float(trailing_atr_mult) * atr_j
                    if cand > stop_loss:
                        stop_loss = cand
                else:
                    cand = low + float(trailing_atr_mult) * atr_j
                    if cand < stop_loss:
                        stop_loss = cand

            if signal > 0:
                if (
                    use_breakeven_stop
                    and not breakeven_activated
                    and high >= breakeven_trigger
                ):
                    stop_loss = breakeven_level
                    breakeven_activated = True

                if not hit_tp and high >= take_profit:
                    exit_idx = j
                    hit_tp = True
                    break
                if not hit_sl and low <= stop_loss:
                    exit_idx = j
                    hit_sl = True
                    break
            else:
                if (
                    use_breakeven_stop
                    and not breakeven_activated
                    and low <= breakeven_trigger
                ):
                    stop_loss = breakeven_level
                    breakeven_activated = True

                if not hit_tp and low <= take_profit:
                    exit_idx = j
                    hit_tp = True
                    break
                if not hit_sl and high >= stop_loss:
                    exit_idx = j
                    hit_sl = True
                    break

        if exit_idx is None:
            exit_idx = end_idx - 1

        if exit_idx >= len(work_df):
            continue

        if signal > 0:
            long_exits.iloc[exit_idx] = True
        else:
            short_exits.iloc[exit_idx] = True

    return long_exits, short_exits


def classify_sr_reaction(
    df: pd.DataFrame,
    signal_col: str = "signal",
    sr_zone_col: Optional[str] = None,
    price_col: str = "close",
    high_col: str = "high",
    low_col: str = "low",
    atr_col: str = "atr",
    atr_window: int = 14,
    lookback_window: int = 5,
    threshold_factor: float = 0.5,
) -> pd.Series:
    """
    对每个 SR 信号，判断后续价格是"反向"（reversal）还是"突破"（breakout）。

    这是 SR 策略建模的关键拆分：
    - 反转型机会：价格回测 SR 区 → 试探（wick 进入）→ 快速反向运行（核心盈利来源）
    - 突破型机会：价格强势穿过 SR 区，继续沿原方向运行（趋势延续，非反转）

    逻辑：
    - Long 信号（期待从需求区反弹）：
        - 如果价格深跌破区域（< zone_price - threshold * ATR）→ 突破型（反向失败）
        - 如果价格快速上涨（> zone_price + 2 * ATR）且未深跌 → 突破型（强势突破）
        - 否则（有下影线试探后上涨）→ 反转型
    - Short 信号（期待从供给区回落）：
        - 如果价格强势突破区域（> zone_price + threshold * ATR）→ 突破型（反向失败）
        - 如果价格快速下跌（< zone_price - 2 * ATR）且未突破 → 突破型（强势突破）
        - 否则（有上影线试探后下跌）→ 反转型

    Args:
        df: DataFrame with OHLCV data and signals
        signal_col: Column name for trading signals (1=Long, -1=Short, 0=Hold)
        sr_zone_col: Column name for SR zone price (if None, will try to infer from features)
        price_col: Column name for price
        high_col: Column name for high
        low_col: Column name for low
        atr_col: Column name for ATR
        atr_window: ATR window if ATR column doesn't exist
        lookback_window: Number of future bars to observe
        threshold_factor: Factor for determining "deep penetration" (default: 0.5)

    Returns:
        Series with reaction types: 'reversal', 'breakout', or np.nan

    """
    if signal_col not in df.columns:
        return pd.Series(np.nan, index=df.index, name="sr_reaction")

    # Infer SR zone price if not provided
    if sr_zone_col is None or sr_zone_col not in df.columns:
        # Try common SR zone column names
        candidates = [
            "sr_zone_price",
            "nearest_sr",
            "vpvr_pvp",
            "wpt_price_reconstructed",
        ]
        for col in candidates:
            if col in df.columns:
                sr_zone_col = col
                break
        else:
            # Fallback to price
            sr_zone_col = price_col

    if sr_zone_col not in df.columns:
        return pd.Series(np.nan, index=df.index, name="sr_reaction")

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
                return pd.Series(np.nan, index=df.index, name="sr_reaction")

    signals = df[signal_col]
    zone_prices = df[sr_zone_col]
    atr_series = df[atr_col]
    prices = df[price_col]
    highs = df[high_col] if high_col in df.columns else prices
    lows = df[low_col] if low_col in df.columns else prices

    reactions = pd.Series(np.nan, index=df.index, name="sr_reaction")

    for i in range(len(df) - lookback_window):
        signal = signals.iloc[i]
        if pd.isna(signal) or signal == 0:
            continue

        zone_price = zone_prices.iloc[i]
        atr = atr_series.iloc[i]

        if pd.isna(zone_price) or pd.isna(atr) or atr <= 0:
            continue

        # Observe future price action
        future_window = slice(i + 1, min(i + 1 + lookback_window, len(df)))
        future_prices = prices.iloc[future_window]
        future_highs = highs.iloc[future_window]
        future_lows = lows.iloc[future_window]

        if len(future_prices) == 0:
            continue

        if signal > 0:  # Long signal (expecting reversal up)
            # Check for deep penetration (breakout down)
            min_low = future_lows.min()
            if min_low < zone_price - threshold_factor * atr:
                reactions.iloc[i] = "breakout"
                continue

            # Check for strong upward move (breakout up)
            max_high = future_highs.max()
            if (
                max_high > zone_price + 2 * atr
                and min_low >= zone_price - threshold_factor * atr
            ):
                reactions.iloc[i] = "breakout"
                continue

            # Otherwise: reversal (price tested zone and reversed)
            reactions.iloc[i] = "reversal"
        else:  # Short signal (expecting reversal down)
            # Check for strong upward move (breakout up)
            max_high = future_highs.max()
            if max_high > zone_price + threshold_factor * atr:
                reactions.iloc[i] = "breakout"
                continue

            # Check for strong downward move (breakout down)
            min_low = future_lows.min()
            if (
                min_low < zone_price - 2 * atr
                and max_high <= zone_price + threshold_factor * atr
            ):
                reactions.iloc[i] = "breakout"
                continue

            # Otherwise: reversal (price tested zone and reversed)
            reactions.iloc[i] = "reversal"

    return reactions


def compute_rr_label_by_reaction(
    df: pd.DataFrame,
    signal_col: str = "signal",
    reaction_col: str = "sr_reaction",
    price_col: str = "close",
    atr_col: str = "atr",
    atr_window: int = 14,
    rr_ratio: float = 2.0,
    max_holding_bars: int = 24,
    stop_loss_r: float = 1.0,
    take_profit_r: float = 2.0,
    reaction_type: Optional[str] = None,  # 'reversal' or 'breakout' or None (both)
    use_continuous_label: bool = False,
) -> pd.Series:
    """
    计算基于风险回报比（R/R）的标签，支持按反应类型（反转 vs 突破）分类。

    这是 SR 策略精细化建模的关键：将反转型和突破型机会分开处理。

    Args:
        df: DataFrame with OHLCV data, signals, and reaction types
        signal_col: Column name for trading signals (1=Long, -1=Short, 0=Hold)
        reaction_col: Column name for SR reaction type ('reversal' or 'breakout')
        price_col: Column name for price
        atr_col: Column name for ATR
        atr_window: ATR window if ATR column doesn't exist
        rr_ratio: Target risk-reward ratio (default: 2.0)
        max_holding_bars: Maximum holding period
        stop_loss_r: Stop loss in R units (default: 1.0)
        take_profit_r: Take profit in R units (default: 2.0)
        reaction_type: Filter by reaction type ('reversal', 'breakout', or None for both)
        use_continuous_label: If True, return realized R/R; if False, return binary label

    Returns:
        Series with R/R labels (binary or continuous), NaN for filtered samples
    """
    if signal_col not in df.columns:
        return pd.Series(np.nan, index=df.index)

    # Filter by reaction type if specified
    if reaction_type and reaction_col in df.columns:
        mask = df[reaction_col] == reaction_type
        if not mask.any():
            print(f"   ⚠️  Warning: No samples with reaction_type='{reaction_type}'")
            return pd.Series(np.nan, index=df.index)
    else:
        mask = pd.Series(True, index=df.index)

    # Compute R/R label for all samples first
    rr_labels = compute_rr_label(
        df,
        signal_col=signal_col,
        price_col=price_col,
        atr_col=atr_col,
        atr_window=atr_window,
        rr_ratio=rr_ratio,
        max_holding_bars=max_holding_bars,
        stop_loss_r=stop_loss_r,
        take_profit_r=take_profit_r,
        use_continuous_label=use_continuous_label,
    )

    # Apply reaction filter
    if reaction_type:
        rr_labels = rr_labels.where(mask)

    return rr_labels
