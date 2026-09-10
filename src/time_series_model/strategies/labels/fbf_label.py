"""
FBF (FailedBreakoutFade) 策略标签

语义：假突破（突破后失败）→ 反手 fade
核心逻辑：
1. 检测突破：价格突破近期高点/低点
2. 检测失败：突破后未能延续，价格收回
3. 反向（fade）入场，计算 forward RR
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from src.time_series_model.pipeline.training.label_utils import _ensure_atr
from src.time_series_model.pipeline.training.label_utils import compute_rr_label


def detect_failed_breakout(
    df: pd.DataFrame,
    price_col: str = "close",
    high_col: str = "high",
    low_col: str = "low",
    lookback: int = 20,
    confirm_bars: int = 2,
) -> tuple[pd.Series, pd.Series]:
    """
    检测假突破信号。

    假突破定义：
    - 假突破向上：突破近期高点但随后收回（在 confirm_bars 内）
    - 假突破向下：突破近期低点但随后收回（在 confirm_bars 内）

    Returns:
        (failed_breakout_up, failed_breakout_down): 布尔 Series
    """
    close = df[price_col]
    high = df[high_col]
    low = df[low_col]

    # 近期高点和低点（不包括当前 bar）
    rolling_high = high.shift(1).rolling(window=lookback, min_periods=1).max()
    rolling_low = low.shift(1).rolling(window=lookback, min_periods=1).min()

    # 突破检测
    breakout_up = high > rolling_high  # 当前 bar 突破前高
    breakout_down = low < rolling_low  # 当前 bar 突破前低

    # 失败检测：突破后收回
    # 假突破向上：突破高点但收盘在高点之下
    failed_up = breakout_up & (close < rolling_high)
    # 假突破向下：突破低点但收盘在低点之上
    failed_down = breakout_down & (close > rolling_low)

    return failed_up, failed_down


def compute_fbf_label(
    df: pd.DataFrame,
    price_col: str = "close",
    high_col: str = "high",
    low_col: str = "low",
    atr_col: str = "atr",
    atr_window: int = 14,
    max_holding_bars: int = 50,
    stop_loss_r: float = 1.0,
    take_profit_r: float = 2.0,
    # 假突破检测参数
    breakout_lookback: int = 20,
    confirm_bars: int = 2,
    # 方向
    combine_mode: str = "long_only",
) -> pd.Series:
    """
    计算 FBF (FailedBreakoutFade) 标签。

    逻辑：
    1. 检测假突破
    2. 反向（fade）入场：假突破向上 → 做空，假突破向下 → 做多
    3. 计算 forward RR

    Args:
        breakout_lookback: 突破检测的回看窗口
        confirm_bars: 确认失败的 bar 数
        combine_mode: "long_only"（只做多，即 fade down），"short_only"（只做空，即 fade up），"any_success"

    Returns:
        pd.Series: 连续 RR 标签，无假突破处为 NaN
    """
    work_df = df.copy()
    atr_series = _ensure_atr(work_df, atr_col, price_col, high_col, low_col, atr_window)
    work_df[atr_col] = atr_series

    # 检测假突破
    failed_up, failed_down = detect_failed_breakout(
        work_df,
        price_col=price_col,
        high_col=high_col,
        low_col=low_col,
        lookback=breakout_lookback,
        confirm_bars=confirm_bars,
    )

    # Fade 方向：
    # - failed_up (假突破向上) → 做空 (fade)
    # - failed_down (假突破向下) → 做多 (fade)

    if combine_mode == "long_only":
        # 只做多 = 只 fade 假突破向下
        signal_mask = failed_down
        signal_direction = 1.0
    elif combine_mode == "short_only":
        # 只做空 = 只 fade 假突破向上
        signal_mask = failed_up
        signal_direction = -1.0
    else:
        # any_success: 两个方向都 fade
        signal_mask = failed_up | failed_down
        signal_direction = 1.0  # 默认，后面可以细化

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

    # 应用假突破掩码
    rr_series = rr_series.where(signal_mask)
    rr_series.name = "rr_label"

    return rr_series
