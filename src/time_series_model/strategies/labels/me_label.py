"""
ME (MomentumExpansion) 策略标签

语义：压缩后波动/区间扩张，放量突破
核心逻辑：
1. 检测压缩区（低波动）
2. 检测扩张/突破（高波动 + 方向性）
3. 计算 forward RR
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from src.time_series_model.pipeline.training.label_utils import _ensure_atr
from src.time_series_model.pipeline.training.label_utils import compute_rr_label


def compute_path_extreme_rr(
    df: pd.DataFrame,
    direction: str,
    horizon: int,
    atr_col: str,
    price_col: str = "close",
    high_col: str = "high",
    low_col: str = "low",
) -> pd.Series:
    """
    计算路径极值型的 forward_rr = (MFE - MAE) / ATR。

    这个值可以是负数（MAE 大于 MFE），用于识别"踩大坑"的情况。

    Args:
        direction: 'long' 或 'short'
        horizon: 前向观察的 bar 数量

    Returns:
        pd.Series: forward_rr 值，无信号处为 NaN
    """
    labels = np.full(len(df), np.nan)

    close = df[price_col].values
    high = df[high_col].values
    low = df[low_col].values
    atr = df[atr_col].values

    for i in range(len(df) - horizon):
        entry_price = close[i]
        risk_unit = atr[i]

        if pd.isna(risk_unit) or risk_unit <= 0:
            continue

        # 使用下一根 bar 到 horizon 范围内的 high/low
        future_high = high[i + 1 : i + 1 + horizon]
        future_low = low[i + 1 : i + 1 + horizon]

        if len(future_high) == 0 or len(future_low) == 0:
            continue

        if direction == "long":
            mfe = np.nanmax(future_high) - entry_price  # 最大有利偏移
            mae = entry_price - np.nanmin(future_low)  # 最大不利偏移
        else:  # short
            mfe = entry_price - np.nanmin(future_low)
            mae = np.nanmax(future_high) - entry_price

        # forward_rr = (MFE - MAE) / ATR
        labels[i] = (mfe - mae) / risk_unit

    return pd.Series(labels, index=df.index, name=f"forward_rr_{direction}")


def detect_compression(
    df: pd.DataFrame,
    atr_col: str = "atr",
    lookback: int = 20,
    compression_percentile: float = 30,
) -> pd.Series:
    """
    检测压缩区（低波动区域）。

    压缩定义：当前 ATR 处于近期历史的低百分位

    Returns:
        pd.Series: 布尔 Series，True 表示处于压缩区
    """
    atr = df[atr_col]

    # 计算 ATR 的滚动百分位
    atr_pct = atr.rolling(window=lookback, min_periods=1).apply(
        lambda x: (
            (x.iloc[-1] <= np.percentile(x, compression_percentile))
            if len(x) > 1
            else False
        ),
        raw=False,
    )

    return atr_pct.fillna(False).astype(bool)


def detect_expansion_breakout(
    df: pd.DataFrame,
    price_col: str = "close",
    high_col: str = "high",
    low_col: str = "low",
    volume_col: str = "volume",
    atr_col: str = "atr",
    compression_mask: pd.Series = None,
    breakout_lookback: int = 10,
    volume_mult: float = 1.2,
    atr_mult: float = 1.5,
) -> tuple[pd.Series, pd.Series]:
    """
    检测扩张/突破信号。

    突破定义：
    1. 之前处于压缩区
    2. 价格突破近期高/低
    3. 成交量放大
    4. ATR 扩张

    Returns:
        (breakout_up, breakout_down): 布尔 Series
    """
    close = df[price_col]
    high = df[high_col]
    low = df[low_col]
    atr = df[atr_col]

    # 近期高低点
    rolling_high = high.shift(1).rolling(window=breakout_lookback, min_periods=1).max()
    rolling_low = low.shift(1).rolling(window=breakout_lookback, min_periods=1).min()

    # 价格突破
    price_breakout_up = close > rolling_high
    price_breakout_down = close < rolling_low

    # 成交量放大
    if volume_col in df.columns:
        volume = df[volume_col]
        avg_volume = (
            volume.shift(1).rolling(window=breakout_lookback, min_periods=1).mean()
        )
        volume_expansion = volume > (avg_volume * volume_mult)
    else:
        volume_expansion = pd.Series(True, index=df.index)

    # ATR 扩张（波动率上升）
    avg_atr = atr.shift(1).rolling(window=breakout_lookback, min_periods=1).mean()
    atr_expansion = atr > (avg_atr * 1.1)  # ATR 上升 10%

    # 压缩后突破（如果提供了压缩掩码）
    if compression_mask is not None:
        # 检查前 N 个 bar 是否有压缩
        was_compressed = (
            compression_mask.shift(1).rolling(window=5, min_periods=1).sum() > 0
        )
    else:
        was_compressed = pd.Series(True, index=df.index)

    # 综合条件
    breakout_up = price_breakout_up & volume_expansion & was_compressed
    breakout_down = price_breakout_down & volume_expansion & was_compressed

    return breakout_up, breakout_down


def compute_me_label(
    df: pd.DataFrame,
    price_col: str = "close",
    high_col: str = "high",
    low_col: str = "low",
    volume_col: str = "volume",
    atr_col: str = "atr",
    atr_window: int = 14,
    max_holding_bars: int = 50,
    stop_loss_r: float = 1.0,
    take_profit_r: float = 2.0,
    # 压缩检测参数
    compression_lookback: int = 20,
    compression_percentile: float = 30,
    # 突破检测参数
    breakout_lookback: int = 10,
    volume_mult: float = 1.2,
    # 方向
    combine_mode: str = "long_only",
) -> pd.Series:
    """
    计算 ME (MomentumExpansion) 标签。

    逻辑：
    1. 检测压缩区（低波动）
    2. 检测扩张/突破（高波动 + 方向性）
    3. 顺势入场，计算 forward RR

    Args:
        compression_lookback: 压缩检测的回看窗口
        compression_percentile: 压缩判定的 ATR 百分位阈值
        breakout_lookback: 突破检测的回看窗口
        volume_mult: 成交量放大倍数
        combine_mode: "long_only", "short_only", "any_success"

    Returns:
        pd.Series: 连续 RR 标签，无突破处为 NaN
    """
    work_df = df.copy()
    atr_series = _ensure_atr(work_df, atr_col, price_col, high_col, low_col, atr_window)
    work_df[atr_col] = atr_series

    # 1. 检测压缩
    compression_mask = detect_compression(
        work_df,
        atr_col=atr_col,
        lookback=compression_lookback,
        compression_percentile=compression_percentile,
    )

    # 2. 检测扩张/突破
    breakout_up, breakout_down = detect_expansion_breakout(
        work_df,
        price_col=price_col,
        high_col=high_col,
        low_col=low_col,
        volume_col=volume_col,
        atr_col=atr_col,
        compression_mask=compression_mask,
        breakout_lookback=breakout_lookback,
        volume_mult=volume_mult,
    )

    # 3. 根据 combine_mode 确定信号
    if combine_mode == "long_only":
        signal_mask = breakout_up
        signal_direction = 1.0
    elif combine_mode == "short_only":
        signal_mask = breakout_down
        signal_direction = -1.0
    else:
        # any_success: 两个方向
        signal_mask = breakout_up | breakout_down
        # BUG FIX: 按突破方向分配 direction，而不是固定 = 1.0
        signal_direction = np.where(
            breakout_up, 1.0, np.where(breakout_down, -1.0, 0.0)
        )

    # 4. 计算 RR 标签
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

    # 应用信号掩码
    rr_series = rr_series.where(signal_mask)
    rr_series.name = "rr_label"

    return rr_series


def compute_me_failure_rr_extreme_label(
    df: pd.DataFrame,
    direction: str = "long",
    horizon: int = 50,
    invert: bool = True,
    price_col: str = "close",
    high_col: str = "high",
    low_col: str = "low",
    volume_col: str = "volume",
    atr_col: str = "atr",
    atr_window: int = 14,
    stop_loss_r: float = 1.0,
    take_profit_r: float = 2.0,
    compression_lookback: int = 20,
    compression_percentile: float = 30,
    breakout_lookback: int = 10,
    volume_mult: float = 1.2,
    failure_threshold: float = -0.8,
) -> pd.Series:
    """
    计算 ME failure_rr_extreme 标签（Gate 训练用）。

    识别动量扩张后会"踩大坑"的条件：
    - failure_rr_extreme = forward_rr < -0.8R

    使用路径极值型 RR：(MFE - MAE) / ATR，可以是负数。

    Args:
        invert: True = 返回 success_no_rr_extreme (1=不踩坑, 0=踩坑)
                False = 返回 failure_rr_extreme (1=踩坑, 0=不踩坑)
        failure_threshold: 失败阈值（默认 -0.8R）

    Returns:
        pd.Series: 二分类标签
    """
    work_df = df.copy()
    atr_series = _ensure_atr(work_df, atr_col, price_col, high_col, low_col, atr_window)
    work_df[atr_col] = atr_series

    # 1. 检测压缩和突破信号
    compression_mask = detect_compression(
        work_df,
        atr_col=atr_col,
        lookback=compression_lookback,
        compression_percentile=compression_percentile,
    )

    breakout_up, breakout_down = detect_expansion_breakout(
        work_df,
        price_col=price_col,
        high_col=high_col,
        low_col=low_col,
        volume_col=volume_col,
        atr_col=atr_col,
        compression_mask=compression_mask,
        breakout_lookback=breakout_lookback,
        volume_mult=volume_mult,
    )

    # 根据方向确定信号掩码
    if direction == "long":
        signal_mask = breakout_up
    else:
        signal_mask = breakout_down

    # 2. 计算路径极值型 RR（可以是负数）
    rr_series = compute_path_extreme_rr(
        work_df,
        direction=direction,
        horizon=horizon,
        atr_col=atr_col,
        price_col=price_col,
        high_col=high_col,
        low_col=low_col,
    )

    # 3. 只保留有突破信号的点
    rr_series = rr_series.where(signal_mask)

    # 4. 识别 failure_rr_extreme: forward_rr < failure_threshold
    failure_mask = rr_series < failure_threshold

    # 5. 生成标签
    if invert:
        # success_no_rr_extreme: 1=不踩坑, 0=踩坑
        label = (~failure_mask).astype(int)
        label = label.where(rr_series.notna())
        label.name = "success_no_rr_extreme"
    else:
        # failure_rr_extreme: 1=踩坑, 0=不踩坑
        label = failure_mask.astype(int)
        label = label.where(rr_series.notna())
        label.name = "failure_rr_extreme"

    return label


def compute_me_return_tree_label(
    df: pd.DataFrame,
    direction: str = "long",
    horizon: int = 50,
    filter_good_only: bool = True,
    price_col: str = "close",
    high_col: str = "high",
    low_col: str = "low",
    volume_col: str = "volume",
    atr_col: str = "atr",
    atr_window: int = 14,
    stop_loss_r: float = 1.0,
    take_profit_r: float = 2.0,
    compression_lookback: int = 20,
    compression_percentile: float = 30,
    breakout_lookback: int = 10,
    volume_mult: float = 1.2,
    good_threshold: float = -0.8,
) -> pd.Series:
    """
    计算 ME Return Tree 标签（Evidence 训练用）。

    目标：在 GOOD 样本（不踩坑）上学习如何放大 RR。

    使用路径极值型 RR：(MFE - MAE) / ATR。

    Args:
        filter_good_only: True = 只返回 GOOD 样本（forward_rr >= good_threshold），
                                 BAD 样本设为 NaN
        good_threshold: GOOD 样本的阈值（默认 -0.8R）

    Returns:
        pd.Series: forward_rr 连续值（GOOD 样本）或 NaN（BAD 样本）
    """
    work_df = df.copy()
    atr_series = _ensure_atr(work_df, atr_col, price_col, high_col, low_col, atr_window)
    work_df[atr_col] = atr_series

    # 1. 检测压缩和突破信号
    compression_mask = detect_compression(
        work_df,
        atr_col=atr_col,
        lookback=compression_lookback,
        compression_percentile=compression_percentile,
    )

    breakout_up, breakout_down = detect_expansion_breakout(
        work_df,
        price_col=price_col,
        high_col=high_col,
        low_col=low_col,
        volume_col=volume_col,
        atr_col=atr_col,
        compression_mask=compression_mask,
        breakout_lookback=breakout_lookback,
        volume_mult=volume_mult,
    )

    # 根据方向确定信号掩码
    if direction == "long":
        signal_mask = breakout_up
    else:
        signal_mask = breakout_down

    # 2. 计算路径极值型 RR
    rr_series = compute_path_extreme_rr(
        work_df,
        direction=direction,
        horizon=horizon,
        atr_col=atr_col,
        price_col=price_col,
        high_col=high_col,
        low_col=low_col,
    )

    # 3. 只保留有突破信号的点
    rr_series = rr_series.where(signal_mask)

    # 4. 过滤 GOOD 样本
    if filter_good_only:
        # GOOD = forward_rr >= good_threshold
        good_mask = rr_series >= good_threshold
        forward_rr = rr_series.where(good_mask)
    else:
        forward_rr = rr_series

    forward_rr.name = "forward_rr"
    return forward_rr
