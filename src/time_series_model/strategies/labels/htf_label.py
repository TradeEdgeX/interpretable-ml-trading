"""
HTF (HTFBiasLTFEntry) 策略标签

语义：大周期定方向，小周期定入场
核心逻辑：
1. HTF（高时间框架）确定趋势偏向
2. LTF（低时间框架）寻找入场时机（如 wick rejection、订单流确认）
3. 顺 HTF 方向计算 forward RR
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from src.time_series_model.pipeline.training.label_utils import _ensure_atr
from src.time_series_model.pipeline.training.label_utils import compute_rr_label


def compute_htf_label(
    df: pd.DataFrame,
    price_col: str = "close",
    high_col: str = "high",
    low_col: str = "low",
    atr_col: str = "atr",
    atr_window: int = 14,
    max_holding_bars: int = 50,
    stop_loss_r: float = 1.0,
    take_profit_r: float = 2.0,
    # HTF 趋势参数
    htf_trend_col: Optional[str] = "htf_trend_sign",  # HTF 趋势方向列
    sma_col: Optional[str] = "sma_200",  # 备用：用 SMA 判断趋势
    # LTF 入场信号参数
    ltf_signal_col: Optional[str] = None,  # LTF 入场信号列（如果有）
    use_wick_rejection: bool = True,  # 使用 wick rejection 作为入场信号
    wick_atr_mult: float = 0.5,  # wick 长度阈值（ATR 倍数）
    # 方向
    combine_mode: str = "long_only",
) -> pd.Series:
    """
    计算 HTF (HTFBiasLTFEntry) 标签。

    逻辑：
    1. HTF 趋势方向确定交易偏向
    2. LTF 入场信号（wick rejection 或自定义）触发入场
    3. 顺 HTF 方向计算 forward RR

    Args:
        htf_trend_col: HTF 趋势方向列名（>0 多头偏向，<0 空头偏向）
        sma_col: SMA 列名（备用趋势判断）
        ltf_signal_col: LTF 入场信号列名（如果提供）
        use_wick_rejection: 是否使用 wick rejection 作为入场信号
        wick_atr_mult: wick rejection 阈值
        combine_mode: "long_only", "short_only", "any_success"

    Returns:
        pd.Series: 连续 RR 标签，无入场信号处为 NaN
    """
    work_df = df.copy()
    atr_series = _ensure_atr(work_df, atr_col, price_col, high_col, low_col, atr_window)
    work_df[atr_col] = atr_series

    close = work_df[price_col]
    high = work_df[high_col]
    low = work_df[low_col]
    open_price = work_df.get("open", close)

    # 确定 HTF 趋势偏向
    if htf_trend_col and htf_trend_col in work_df.columns:
        htf_trend = work_df[htf_trend_col].fillna(0)
    elif sma_col and sma_col in work_df.columns:
        sma = work_df[sma_col].fillna(close)
        htf_trend = np.sign(close - sma)
    else:
        # 默认：用 20 周期 SMA 斜率
        sma_20 = close.rolling(window=20, min_periods=1).mean()
        htf_trend = np.sign(sma_20 - sma_20.shift(5))

    htf_long = htf_trend > 0
    htf_short = htf_trend < 0

    # LTF 入场信号
    if ltf_signal_col and ltf_signal_col in work_df.columns:
        ltf_signal_long = work_df[ltf_signal_col] > 0
        ltf_signal_short = work_df[ltf_signal_col] < 0
    elif use_wick_rejection:
        # Wick rejection: 长下影线 = 多头信号，长上影线 = 空头信号
        body = abs(close - open_price)
        lower_wick = np.minimum(open_price, close) - low
        upper_wick = high - np.maximum(open_price, close)
        wick_threshold = atr_series * wick_atr_mult

        # 多头 wick rejection: 长下影线 + 收盘在上半部
        ltf_signal_long = (lower_wick > wick_threshold) & (close > (high + low) / 2)
        # 空头 wick rejection: 长上影线 + 收盘在下半部
        ltf_signal_short = (upper_wick > wick_threshold) & (close < (high + low) / 2)
    else:
        # 无 LTF 过滤：全量入场
        ltf_signal_long = pd.Series(True, index=work_df.index)
        ltf_signal_short = pd.Series(True, index=work_df.index)

    # 合并条件：HTF 偏向 + LTF 入场信号
    long_entry = htf_long & ltf_signal_long
    short_entry = htf_short & ltf_signal_short

    # 根据 combine_mode 确定信号
    if combine_mode == "long_only":
        signal_mask = long_entry
        signal_direction = 1.0
    elif combine_mode == "short_only":
        signal_mask = short_entry
        signal_direction = -1.0
    else:
        signal_mask = long_entry | short_entry
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

    # 应用入场掩码
    rr_series = rr_series.where(signal_mask)
    rr_series.name = "rr_label"

    return rr_series
