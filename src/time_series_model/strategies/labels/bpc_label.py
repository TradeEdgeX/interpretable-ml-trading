"""
BPC (BreakoutPullbackContinuation) 策略标签

语义：趋势中先回踩再延续原方向
核心逻辑：
1. 检测趋势方向（使用 trend_sign 或 SMA 斜率）
2. 检测回踩（价格回落但不破坏趋势结构）
3. 在回踩位置计算 forward RR（看是否延续原方向）
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from src.time_series_model.pipeline.training.label_utils import _ensure_atr
from src.time_series_model.pipeline.training.label_utils import compute_rr_label


def detect_pullback(
    df: pd.DataFrame,
    price_col: str = "close",
    high_col: str = "high",
    low_col: str = "low",
    lookback: int = 10,
    pullback_threshold: float = 0.3,
) -> tuple[pd.Series, pd.Series]:
    """
    检测回踩信号。

    回踩定义：
    - 上升趋势中：价格从近期高点回落一定比例（但没有创新低）
    - 下降趋势中：价格从近期低点反弹一定比例（但没有创新高）

    Args:
        lookback: 回看窗口
        pullback_threshold: 回踩阈值（相对于近期波动的比例）

    Returns:
        (long_pullback, short_pullback): 布尔 Series，分别表示多头回踩和空头回踩
    """
    close = df[price_col]
    high = df[high_col]
    low = df[low_col]

    # 近期高点和低点
    rolling_high = high.rolling(window=lookback, min_periods=1).max()
    rolling_low = low.rolling(window=lookback, min_periods=1).min()

    # 波动范围
    range_size = rolling_high - rolling_low
    range_size = range_size.clip(lower=1e-8)

    # 从高点回落的幅度
    drawdown_from_high = (rolling_high - close) / range_size
    # 从低点反弹的幅度
    bounce_from_low = (close - rolling_low) / range_size

    # 趋势方向（简单方法：比较当前价与 lookback 前的价格）
    price_change = close - close.shift(lookback)
    uptrend = price_change > 0
    downtrend = price_change < 0

    # 多头回踩：上升趋势中，从高点回落但回落幅度在阈值内
    long_pullback = (
        uptrend
        & (drawdown_from_high >= pullback_threshold)
        & (drawdown_from_high <= 0.7)
    )

    # 空头回踩：下降趋势中，从低点反弹但反弹幅度在阈值内
    short_pullback = (
        downtrend & (bounce_from_low >= pullback_threshold) & (bounce_from_low <= 0.7)
    )

    return long_pullback, short_pullback


def compute_bpc_label(
    df: pd.DataFrame,
    price_col: str = "close",
    high_col: str = "high",
    low_col: str = "low",
    atr_col: str = "atr",
    atr_window: int = 14,
    max_holding_bars: int = 50,
    stop_loss_r: float = 1.0,
    take_profit_r: float = 2.0,
    # 回踩检测参数
    pullback_lookback: int = 10,
    pullback_threshold: float = 0.3,
    # 趋势过滤（可选）
    trend_col: Optional[str] = "trend_sign",
    use_trend_filter: bool = True,
    # 方向
    combine_mode: str = "long_only",
) -> pd.Series:
    """
    计算 BPC (BreakoutPullbackContinuation) 标签。

    逻辑：
    1. 在趋势中检测回踩
    2. 在回踩位置计算顺趋势方向的 forward RR

    Args:
        pullback_lookback: 回踩检测的回看窗口
        pullback_threshold: 回踩阈值
        trend_col: 趋势方向列名（如果提供，用于加强趋势过滤）
        use_trend_filter: 是否使用趋势过滤
        combine_mode: "long_only", "short_only", "any_success"

    Returns:
        pd.Series: 连续 RR 标签，非回踩区域为 NaN
    """
    work_df = df.copy()
    atr_series = _ensure_atr(work_df, atr_col, price_col, high_col, low_col, atr_window)
    work_df[atr_col] = atr_series

    # 检测回踩
    long_pullback, short_pullback = detect_pullback(
        work_df,
        price_col=price_col,
        high_col=high_col,
        low_col=low_col,
        lookback=pullback_lookback,
        pullback_threshold=pullback_threshold,
    )

    # 趋势过滤（可选）
    if use_trend_filter and trend_col in work_df.columns:
        trend = work_df[trend_col].fillna(0)
        long_pullback &= trend > 0  # 只在上升趋势中做多
        short_pullback &= trend < 0  # 只在下降趋势中做空

    # 根据 combine_mode 确定信号
    if combine_mode == "long_only":
        signal_mask = long_pullback
        signal_direction = 1.0
    elif combine_mode == "short_only":
        signal_mask = short_pullback
        signal_direction = -1.0
    else:
        # any_success: 合并两个方向
        signal_mask = long_pullback | short_pullback
        signal_direction = 1.0  # 先计算多头，后面再处理

    # 计算 RR 标签
    work_df["__signal"] = signal_direction

    rr_series = compute_rr_label(
        work_df,
        signal_col="__signal",
        price_col=price_col,
        atr_col=atr_col,
        atr_window=atr_window,
        rr_ratio=take_profit_r / stop_loss_r if stop_loss_r != 0 else take_profit_r,
        max_holding_bars=max_holding_bars,
        stop_loss_r=stop_loss_r,
        take_profit_r=take_profit_r,
        use_continuous_label=True,
        entry_price_col="open",
        entry_offset=1,
    )

    # 应用回踩掩码
    rr_series = rr_series.where(signal_mask)
    rr_series.name = "rr_label"

    return rr_series
