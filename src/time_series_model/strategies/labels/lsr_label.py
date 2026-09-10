"""
LSR (LiquiditySweepRejection) 策略标签

语义：流动性扫损（sweep）后价格拒绝并反向
核心逻辑：
1. 检测 sweep：wick 扫过前高/前低
2. 检测 rejection：收盘反向（未停留在 sweep 区域）
3. 反向入场，计算 forward RR
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from src.time_series_model.pipeline.training.label_utils import _ensure_atr
from src.time_series_model.pipeline.training.label_utils import compute_rr_label


def detect_liquidity_sweep(
    df: pd.DataFrame,
    price_col: str = "close",
    high_col: str = "high",
    low_col: str = "low",
    lookback: int = 20,
    atr_series: Optional[pd.Series] = None,
    sweep_atr_mult: float = 0.3,
) -> tuple[pd.Series, pd.Series]:
    """
    检测流动性扫损（liquidity sweep）信号。

    Sweep 定义：
    - Sweep High: wick 扫过前高，但收盘回到前高之下（形成上影线 rejection）
    - Sweep Low: wick 扫过前低，但收盘回到前低之上（形成下影线 rejection）

    Returns:
        (sweep_high_rejection, sweep_low_rejection): 布尔 Series
    """
    close = df[price_col]
    high = df[high_col]
    low = df[low_col]
    open_price = df.get("open", close)

    # 前高/前低（不包括当前 bar）
    rolling_high = high.shift(1).rolling(window=lookback, min_periods=1).max()
    rolling_low = low.shift(1).rolling(window=lookback, min_periods=1).min()

    # Sweep 检测
    # Sweep High: high 超过前高，但 close < 前高（形成 rejection）
    swept_high = high > rolling_high
    rejected_from_high = close < rolling_high

    # Sweep Low: low 低于前低，但 close > 前低（形成 rejection）
    swept_low = low < rolling_low
    rejected_from_low = close > rolling_low

    # 额外条件：wick 长度检查（确保是有意义的 sweep）
    if atr_series is not None:
        upper_wick = high - np.maximum(open_price, close)
        lower_wick = np.minimum(open_price, close) - low
        wick_threshold = atr_series * sweep_atr_mult

        # Sweep high 需要有明显的上影线
        significant_upper_wick = upper_wick > wick_threshold
        # Sweep low 需要有明显的下影线
        significant_lower_wick = lower_wick > wick_threshold

        sweep_high_rejection = swept_high & rejected_from_high & significant_upper_wick
        sweep_low_rejection = swept_low & rejected_from_low & significant_lower_wick
    else:
        sweep_high_rejection = swept_high & rejected_from_high
        sweep_low_rejection = swept_low & rejected_from_low

    return sweep_high_rejection, sweep_low_rejection


def compute_lsr_label(
    df: pd.DataFrame,
    price_col: str = "close",
    high_col: str = "high",
    low_col: str = "low",
    atr_col: str = "atr",
    atr_window: int = 14,
    max_holding_bars: int = 50,
    stop_loss_r: float = 1.0,
    take_profit_r: float = 2.0,
    # Sweep 检测参数
    sweep_lookback: int = 20,
    sweep_atr_mult: float = 0.3,
    # 方向
    combine_mode: str = "long_only",
) -> pd.Series:
    """
    计算 LSR (LiquiditySweepRejection) 标签。

    逻辑：
    1. 检测 liquidity sweep + rejection
    2. 反向入场：sweep high rejection → 做空，sweep low rejection → 做多
    3. 计算 forward RR

    Args:
        sweep_lookback: sweep 检测的回看窗口
        sweep_atr_mult: wick 长度阈值（ATR 倍数）
        combine_mode: "long_only"（只做多，sweep low rejection），"short_only"（只做空），"any_success"

    Returns:
        pd.Series: 连续 RR 标签，无 sweep rejection 处为 NaN
    """
    work_df = df.copy()
    atr_series = _ensure_atr(work_df, atr_col, price_col, high_col, low_col, atr_window)
    work_df[atr_col] = atr_series

    # 检测 liquidity sweep
    sweep_high, sweep_low = detect_liquidity_sweep(
        work_df,
        price_col=price_col,
        high_col=high_col,
        low_col=low_col,
        lookback=sweep_lookback,
        atr_series=atr_series,
        sweep_atr_mult=sweep_atr_mult,
    )

    # 反向入场：
    # - sweep_high (扫前高后拒绝) → 做空
    # - sweep_low (扫前低后拒绝) → 做多

    if combine_mode == "long_only":
        # 只做多 = 只在 sweep low rejection 后入场
        signal_mask = sweep_low
        signal_direction = 1.0
    elif combine_mode == "short_only":
        # 只做空 = 只在 sweep high rejection 后入场
        signal_mask = sweep_high
        signal_direction = -1.0
    else:
        # any_success: 两个方向
        signal_mask = sweep_high | sweep_low
        signal_direction = 1.0

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

    # 应用 sweep 掩码
    rr_series = rr_series.where(signal_mask)
    rr_series.name = "rr_label"

    return rr_series
