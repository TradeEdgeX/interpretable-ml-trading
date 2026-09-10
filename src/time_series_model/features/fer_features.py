"""
FER (FailureExhaustionReversal) 专属特征函数

核心语义：单边博弈失败 → 反向清算
- 推进效率下降：ΔPrice / ΔDelta 下降
- 吸收特征：aggressor 但价格不涨
- Trapped Cluster：多头被困
- Impulse 失败：推进已死

所有特征遵循：
1. 无未来函数
2. 支持流式计算
3. 使用滚动窗口
"""

import numpy as np
import pandas as pd
from typing import Optional

EPS = 1e-9


def price_delta_efficiency_f(
    df: pd.DataFrame,
    close_col: str = "close",
    cvd_col: str = "cvd",
    window: int = 20,
    **kwargs
) -> pd.Series:
    """
    推进效率：ΔPrice / |ΔCVD|

    语义：
    - 高值：每单位订单流能推动价格大幅变动（效率高）
    - 低值：大量订单流但价格不动（效率低，吸收明显）

    FER信号：效率持续下降 = impulse失败

    Args:
        df: 数据
        close_col: 价格列
        cvd_col: CVD列
        window: 滚动窗口

    Returns:
        推进效率序列
    """
    close = df[close_col].values
    cvd = df[cvd_col].values if cvd_col in df.columns else np.zeros(len(df))
    n = len(df)

    efficiency = np.full(n, np.nan)

    for i in range(window, n):
        # 窗口内的价格变化和CVD变化
        price_change = abs(close[i] - close[i - window])
        cvd_change = abs(cvd[i] - cvd[i - window])

        if cvd_change > EPS:
            efficiency[i] = price_change / cvd_change
        else:
            efficiency[i] = 0.0  # CVD不变，效率为0

    return pd.Series(efficiency, index=df.index, name="price_delta_efficiency")


def aggressor_absorption_ratio_f(
    df: pd.DataFrame,
    close_col: str = "close",
    volume_col: str = "volume",
    cvd_col: str = "cvd",
    window: int = 10,
    **kwargs
) -> pd.Series:
    """
    吸收比率：aggressor买入但价格下跌的程度

    语义：
    - 高值：主动买入很多，但价格反而下跌（吸收明显）
    - 低值：主动买入推动价格上涨（正常）

    FER信号：吸收比率高 = 多头被困

    计算：
    1. 检测窗口内CVD上升（买入压力）
    2. 同时价格下跌
    3. 比率 = -ΔPrice / ΔCVD（归一化）

    Args:
        df: 数据
        close_col: 价格列
        volume_col: 成交量列
        cvd_col: CVD列
        window: 滚动窗口

    Returns:
        吸收比率序列
    """
    close = df[close_col].values
    cvd = df[cvd_col].values if cvd_col in df.columns else np.zeros(len(df))
    n = len(df)

    absorption = np.full(n, np.nan)

    for i in range(window, n):
        # 窗口内变化
        price_change = close[i] - close[i - window]
        cvd_change = cvd[i] - cvd[i - window]

        # 只关注买入压力（CVD上升）但价格下跌的情况
        if cvd_change > EPS and price_change < 0:
            # 吸收比率：价格下跌幅度 / CVD上升幅度
            absorption[i] = abs(price_change) / cvd_change
        elif cvd_change < -EPS and price_change > 0:
            # 反向情况：卖出压力但价格上涨（也是吸收）
            absorption[i] = abs(price_change) / abs(cvd_change)
        else:
            absorption[i] = 0.0  # 正常情况

    return pd.Series(absorption, index=df.index, name="aggressor_absorption_ratio")


def trapped_longs_ratio_f(
    df: pd.DataFrame,
    close_col: str = "close",
    high_col: str = "high",
    low_col: str = "low",
    volume_col: str = "volume",
    lookback: int = 20,
    **kwargs
) -> pd.Series:
    """
    多头被困比率

    语义：
    - 价格创新高后回落，高位成交量大的区域被困
    - 高值：大量多头被困在高位

    FER信号：被困比率高 = 反转机会

    计算逻辑：
    1. 找到lookback内的最高价
    2. 计算当前价格距离最高价的回撤
    3. 估计高位成交量占比

    Args:
        df: 数据
        close_col: 价格列
        high_col: 最高价列
        low_col: 最低价列
        volume_col: 成交量列
        lookback: 回溯窗口

    Returns:
        被困比率序列
    """
    close = df[close_col].values
    high = df[high_col].values
    volume = df[volume_col].values
    n = len(df)

    trapped_ratio = np.full(n, np.nan)

    for i in range(lookback, n):
        # 窗口内最高价（不包含当前点，避免未来函数）
        window_high = np.max(high[max(0, i - lookback) : i])
        current_price = close[i]

        # 回撤比例
        if window_high > EPS:
            drawdown = (window_high - current_price) / window_high
        else:
            drawdown = 0.0

        # 估计高位成交量（简化版：最近1/4窗口的成交量占比）
        recent_volume = np.sum(volume[max(0, i - lookback // 4) : i])
        total_volume = np.sum(volume[max(0, i - lookback) : i])

        if total_volume > EPS:
            high_volume_ratio = recent_volume / total_volume
        else:
            high_volume_ratio = 0.0

        # 被困比率 = 回撤 * 高位成交占比
        trapped_ratio[i] = drawdown * high_volume_ratio

    return pd.Series(trapped_ratio, index=df.index, name="trapped_longs_ratio")


def impulse_failure_score_f(
    df: pd.DataFrame,
    close_col: str = "close",
    atr_col: str = "atr",
    momentum_col: str = "momentum_score",
    efficiency_col: str = "price_delta_efficiency",
    window: int = 10,
    **kwargs
) -> pd.Series:
    """
    Impulse失败得分

    语义：
    - 综合判断impulse是否失败
    - 条件：动量还在，但推进效率死亡

    FER核心：钱还在冲，但价格不再配合

    计算：
    1. 检测是否有momentum（动能存在）
    2. 检测推进效率是否下降（推进死亡）
    3. 综合得分

    Args:
        df: 数据
        close_col: 价格列
        atr_col: ATR列
        momentum_col: 动量列
        efficiency_col: 推进效率列
        window: 窗口

    Returns:
        Impulse失败得分
    """
    # 先计算依赖特征（如果不存在）
    if efficiency_col not in df.columns:
        efficiency = price_delta_efficiency_f(df, close_col=close_col, window=window)
    else:
        efficiency = df[efficiency_col]

    momentum = (
        df[momentum_col].values if momentum_col in df.columns else np.zeros(len(df))
    )
    n = len(df)

    failure_score = np.full(n, np.nan)

    for i in range(window, n):
        # 动量水平（归一化）
        momentum_level = abs(momentum[i]) if not np.isnan(momentum[i]) else 0.0

        # 推进效率变化率（下降 = 失败）
        eff_current = efficiency.iloc[i] if not np.isnan(efficiency.iloc[i]) else 0.0
        eff_past = (
            efficiency.iloc[i - window]
            if not np.isnan(efficiency.iloc[i - window])
            else 0.0
        )

        if eff_past > EPS:
            eff_change_rate = (eff_current - eff_past) / eff_past
        else:
            eff_change_rate = 0.0

        # Impulse失败 = 动量高 + 效率下降
        # 得分越高 = 失败越明显
        if momentum_level > 0.5 and eff_change_rate < -0.3:
            # 有momentum但效率下降超过30%
            failure_score[i] = momentum_level * abs(eff_change_rate)
        else:
            failure_score[i] = 0.0

    return pd.Series(failure_score, index=df.index, name="impulse_failure_score")


def momentum_efficiency_decay_f(
    df: pd.DataFrame,
    close_col: str = "close",
    momentum_col: str = "momentum_score",
    window: int = 20,
    **kwargs
) -> pd.Series:
    """
    动量效率衰减

    语义：
    - 动量/价格变化比率的下降速度
    - 高值：动量虽在但价格推进减弱（效率衰减）

    Args:
        df: 数据
        close_col: 价格列
        momentum_col: 动量列
        window: 窗口

    Returns:
        效率衰减序列
    """
    close = df[close_col].values
    momentum = (
        df[momentum_col].values if momentum_col in df.columns else np.zeros(len(df))
    )
    n = len(df)

    decay = np.full(n, np.nan)

    for i in range(window, n):
        # 价格变化
        price_change = abs(close[i] - close[i - window])

        # 平均动量水平
        avg_momentum = np.nanmean(np.abs(momentum[i - window : i + 1]))

        if avg_momentum > EPS:
            # 效率 = 价格变化 / 动量
            current_efficiency = price_change / avg_momentum

            # 过去的效率
            past_price_change = (
                abs(close[i - window] - close[i - 2 * window]) if i >= 2 * window else 0
            )
            past_momentum = (
                np.nanmean(np.abs(momentum[i - 2 * window : i - window + 1]))
                if i >= 2 * window
                else avg_momentum
            )

            if past_momentum > EPS:
                past_efficiency = past_price_change / past_momentum

                # 衰减 = 效率下降比例
                if past_efficiency > EPS:
                    decay[i] = max(
                        0, (past_efficiency - current_efficiency) / past_efficiency
                    )
                else:
                    decay[i] = 0.0
            else:
                decay[i] = 0.0
        else:
            decay[i] = 0.0

    return pd.Series(decay, index=df.index, name="momentum_efficiency_decay")


# ============================================================
# 辅助特征
# ============================================================


def volume_price_divergence_f(
    df: pd.DataFrame,
    close_col: str = "close",
    volume_col: str = "volume",
    window: int = 10,
    **kwargs
) -> pd.Series:
    """
    成交量-价格背离

    语义：
    - 成交量上升但价格下跌（背离）
    - 高值：背离明显

    Args:
        df: 数据
        close_col: 价格列
        volume_col: 成交量列
        window: 窗口

    Returns:
        背离程度序列
    """
    close = df[close_col].values
    volume = df[volume_col].values
    n = len(df)

    divergence = np.full(n, np.nan)

    for i in range(window, n):
        # 价格变化方向
        price_change = close[i] - close[i - window]

        # 成交量变化
        volume_change = volume[i] - volume[i - window]

        # 背离：价格和成交量方向相反
        if price_change * volume_change < 0:
            # 归一化背离程度
            divergence[i] = abs(volume_change) / (abs(price_change) + EPS)
        else:
            divergence[i] = 0.0

    return pd.Series(divergence, index=df.index, name="volume_price_divergence")
