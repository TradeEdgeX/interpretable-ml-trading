"""统一的基础指标和特征工程模块

合并了 base_indicators.py 和 baseline_feature_engineering.py 的功能，
并添加了无量纲特征和优化的依赖关系管理。

主要改进：
1. 合并基础指标和 baseline 特征到一个模块
2. 添加 ZigZag、POC、HAL、Swing High/Low 的无量纲特征
3. 添加基础价格与量能相对变化特征
4. 优化依赖关系管理，支持按需计算
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

from src.features.registry import register_feature

# talib is heavy / often absent from lean CI (requirements-ci.txt). Import on use
# so OrderFlowListener → IncrementalFeatureComputer can load without sklearn/_talib().


def _talib():
    import talib

    return talib


# =============================================================================
# Baseline Feature Functions
# =============================================================================


@register_feature("compute_rsi", category="baseline")
def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """计算相对强弱指数 (RSI)."""
    series = pd.to_numeric(series, errors="coerce").astype(float)

    # 监控：检查输入数据质量
    try:
        from src.features.utils.data_monitor import check_data_quality

        check_data_quality(
            pd.DataFrame({"price": series}),
            data_source="RSI_CALC",
            stage="before_rsi_calc",
            raise_on_inf=False,
        )
    except Exception:
        pass

    # 检查输入数据：如果包含 inf/NaN 或全为 0，可能导致 RSI 计算异常
    if series.isna().all() or (series == 0).all():
        return pd.Series(np.nan, index=series.index)
    # 清理输入数据中的 inf 值，避免传递给 talib
    series_clean = series.replace([np.inf, -np.inf], np.nan)
    # 如果清理后数据不足，返回 NaN
    if series_clean.notna().sum() < period + 1:
        return pd.Series(np.nan, index=series.index)
    values = _talib().RSI(series_clean.values, timeperiod=period)
    rsi_series = pd.Series(values, index=series.index)
    # 清理输出中的 inf 值
    rsi_series = rsi_series.replace([np.inf, -np.inf], np.nan)
    # RSI has a warmup period; fill leading gaps with a neutral value to avoid NaNs.
    rsi_series = rsi_series.ffill().fillna(50.0)
    return rsi_series


@register_feature("compute_rsi_from_series", category="baseline")
def compute_rsi_from_series(close: pd.Series, period: int = 14) -> pd.DataFrame:
    """Narrow-IO version of RSI calculation that returns DataFrame."""
    rsi_series = compute_rsi(close, period=period)
    return rsi_series.rename("rsi").to_frame()


@register_feature("compute_macd", category="baseline")
def compute_macd(
    series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """计算MACD指标."""
    series = pd.to_numeric(series, errors="coerce").astype(float)
    macd_line, signal_line, histogram = _talib().MACD(
        series.values, fastperiod=fast, slowperiod=slow, signalperiod=signal
    )
    index = series.index
    return (
        pd.Series(macd_line, index=index),
        pd.Series(signal_line, index=index),
        pd.Series(histogram, index=index),
    )


@register_feature("compute_macd_from_series", category="baseline")
def compute_macd_from_series(
    *,
    series: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    atr_period: int = 14,
) -> pd.DataFrame:
    """Narrow-IO MACD, normalized by ATR.

    Returns:
        DataFrame with macd, macd_signal, macd_histogram, all normalized by ATR.
        This makes MACD cross-asset comparable.
        Typical range: [-3, 3] after normalization.
    """
    macd_line, signal_line, histogram = compute_macd(series, fast, slow, signal)
    atr = compute_atr(high, low, close, atr_period)

    # Normalize by ATR
    eps = 1e-8
    atr_safe = atr.replace(0, np.nan).fillna(eps)

    macd_norm = (macd_line / atr_safe).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    signal_norm = (
        (signal_line / atr_safe).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    )
    hist_norm = (histogram / atr_safe).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return pd.DataFrame(
        {
            "macd_atr": macd_norm,
            "macd_signal_atr": signal_norm,
            "macd_histogram_atr": hist_norm,
        }
    )


@register_feature("compute_bollinger_bands", category="baseline")
def compute_bollinger_bands(
    series: pd.Series, period: int = 20, std_dev: int = 2
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """计算布林带."""
    series = pd.to_numeric(series, errors="coerce").astype(float)
    upper, middle, lower = _talib().BBANDS(
        series.values, timeperiod=period, nbdevup=std_dev, nbdevdn=std_dev, matype=0
    )
    index = series.index
    return (
        pd.Series(upper, index=index),
        pd.Series(middle, index=index),
        pd.Series(lower, index=index),
    )


@register_feature("compute_atr", category="baseline")
def compute_atr(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    """计算平均真实波幅 (ATR)."""
    high = pd.to_numeric(high, errors="coerce").astype(float)
    low = pd.to_numeric(low, errors="coerce").astype(float)
    close = pd.to_numeric(close, errors="coerce").astype(float)
    atr_values = _talib().ATR(high.values, low.values, close.values, timeperiod=period)
    return pd.Series(atr_values, index=high.index)


@register_feature("compute_atr_from_series", category="baseline")
def compute_atr_from_series(
    *,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.DataFrame:
    """Narrow-IO ATR (Average True Range) in *price units*.

    Important:
        In this codebase, the canonical column name `atr` is used as a *price-unit scale*
        for label normalization and for converting normalized SR distances back to raw price.

        If you need a cross-asset comparable ATR, use one of:
        - `atr_ratio` (atr/close) via `compute_atr_ratio(_from_series)`
        - `natr_14` (Normalized ATR) from TA-Lib wrappers
        - `atr_percentile` (regime indicator) via `compute_atr_percentile(_from_series)`
    """
    close = pd.to_numeric(close, errors="coerce").astype(float)
    atr_raw = compute_atr(high, low, close, period)
    atr_raw = (
        pd.to_numeric(atr_raw, errors="coerce")
        .astype(float)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )
    return atr_raw.rename("atr").to_frame()


@register_feature("compute_zigzag", category="baseline")
def compute_zigzag(
    high: pd.Series,
    low: pd.Series,
    threshold: float = 0.05,
    return_high_low: bool = False,
    price_col: Optional[pd.Series] = None,
) -> pd.Series | Tuple[pd.Series, pd.Series, pd.Series]:
    """
    计算ZigZag指标（优化版：可同时计算高点和低点）

    ✅ 建议：使用 WPT 中高频重构价格（price_col）而非原始价格
    这样可以保留关键拐点，同时去除毛刺噪声。

    Args:
        high: 最高价序列（如果提供了 price_col，此参数将被忽略用于价格计算）
        low: 最低价序列（如果提供了 price_col，此参数将被忽略用于价格计算）
        threshold: 转折阈值（默认 0.05，即 5%）
        return_high_low: 是否同时返回高点和低点序列（默认 False）
        price_col: 可选的价格序列（如 WPT 中高频重构价格）。如果提供，将使用此价格
                   而非原始 high/low。默认 None，使用原始价格（向后兼容）

    Returns:
        如果 return_high_low=False: 返回 zigzag 序列
        如果 return_high_low=True: 返回 (zigzag, zz_high, zz_low) 元组
    """
    high = pd.to_numeric(high, errors="coerce")
    low = pd.to_numeric(low, errors="coerce")
    if len(high) < 2:
        if return_high_low:
            empty = pd.Series(index=high.index, dtype=float)
            return empty, empty, empty
        return pd.Series(index=high.index, dtype=float)

    zigzag = pd.Series(index=high.index, dtype=float)
    zz_high = pd.Series(index=high.index, dtype=float) if return_high_low else None
    zz_low = pd.Series(index=high.index, dtype=float) if return_high_low else None

    # 确定使用的价格序列
    if price_col is not None:
        # 使用 WPT 重构价格（同时作为 high 和 low）
        price_series = price_col
        last_pivot = price_series.iloc[0]
    else:
        # 使用原始价格
        price_series = None
        last_pivot = high.iloc[0]

    # Seed the first pivot to avoid leading NaNs before the first turn is detected.
    zigzag.iloc[0] = last_pivot
    if return_high_low:
        zz_high.iloc[0] = last_pivot
        zz_low.iloc[0] = last_pivot

    trend = None
    try:
        for i in range(1, len(high)):
            if price_col is not None:
                # 使用 WPT 重构价格
                current_price = price_series.iloc[i]
                current_high = current_price
                current_low = current_price
            else:
                # 使用原始价格
                current_high = high.iloc[i]
                current_low = low.iloc[i]

            if trend is None:
                if current_high >= last_pivot * (1 + threshold):
                    trend = "up"
                    last_pivot = current_high
                    zigzag.iloc[i] = current_high
                    if return_high_low:
                        zz_high.iloc[i] = current_high
                elif current_low <= last_pivot * (1 - threshold):
                    trend = "down"
                    last_pivot = current_low
                    zigzag.iloc[i] = current_low
                    if return_high_low:
                        zz_low.iloc[i] = current_low
            elif trend == "up":
                if current_low <= last_pivot * (1 - threshold):
                    # 趋势反转：从上涨转为下跌
                    trend = "down"
                    last_pivot = current_low
                    zigzag.iloc[i] = current_low
                    if return_high_low:
                        zz_low.iloc[i] = current_low
                elif current_high >= last_pivot:
                    # 继续上涨，更新高点
                    last_pivot = current_high
                    zigzag.iloc[i] = current_high
                    if return_high_low:
                        zz_high.iloc[i] = current_high
            else:  # trend == 'down'
                if current_high >= last_pivot * (1 + threshold):
                    # 趋势反转：从下跌转为上涨
                    trend = "up"
                    last_pivot = current_high
                    zigzag.iloc[i] = current_high
                    if return_high_low:
                        zz_high.iloc[i] = current_high
                elif current_low <= last_pivot:
                    # 继续下跌，更新低点
                    last_pivot = current_low
                    zigzag.iloc[i] = current_low
                    if return_high_low:
                        zz_low.iloc[i] = current_low

        zigzag = zigzag.ffill()
        if return_high_low:
            zz_high = zz_high.ffill()
            zz_low = zz_low.ffill()
    except Exception:
        zigzag = pd.Series(0, index=high.index, dtype=float)
        if return_high_low:
            zz_high = pd.Series(0, index=high.index, dtype=float)
            zz_low = pd.Series(0, index=high.index, dtype=float)

    if return_high_low:
        return zigzag, zz_high, zz_low
    return zigzag


@register_feature("calculate_sqs", category="baseline")
def calculate_sqs(
    sr_price: float,
    df: pd.DataFrame,
    window: int = 60,
    tolerance_factor: float = 0.5,
    sr_type: str = "support",  # 必须指定: 'support' 或 'resistance'
    max_lookahead_bars: int = 3,
    use_volume_confirmation: bool = True,  # 是否使用量价确认
    volume_lookback: int = 20,  # 成交量回看窗口（用于计算平均成交量）
    min_volume_ratio: float = 1.0,  # 最小成交量比率（低于此值不计入有效反应）
) -> float:
    """
    计算支撑/阻力位的质量评分（Structure Quality Score, SQS）

    ⚠️ 要求：df 必须是截止到【当前决策时刻之前】的历史数据（即不含未来K线）
    例如，在时刻 i 决策时，df = data.iloc[:i]

    Args:
        sr_price: 支撑或阻力价位
        df: 历史K线数据，必须包含 ['high', 'low', 'close', 'atr', 'volume']，索引为时间
        window: 回看窗口长度（单位：K线根数）
        tolerance_factor: ATR 容忍带系数，默认 0.5
        sr_type: 必须为 'support' 或 'resistance'
        max_lookahead_bars: 最大反应观察期（不超过窗口剩余长度）
        use_volume_confirmation: 是否使用量价确认（默认 True）
        volume_lookback: 成交量回看窗口（用于计算平均成交量）
        min_volume_ratio: 最小成交量比率（低于此值不计入有效反应，默认 1.0 表示不强制要求放量）

    Returns:
        SQS 分数（>=0，越高越好；无有效测试时返回 0.0）
    """
    if sr_type not in {"support", "resistance"}:
        raise ValueError("sr_type must be 'support' or 'resistance'")

    if len(df) < window or "atr" not in df.columns or df["atr"].empty:
        return 0.0

    # 使用窗口内最后一个 ATR（即最新可用ATR）
    current_atr = df["atr"].iloc[-1]
    # 检查 ATR 是否有效（必须是有限的正数）
    if not np.isfinite(current_atr) or current_atr <= 0:
        return 0.0

    tolerance = current_atr * tolerance_factor
    window_df = df.tail(window).copy()

    # 1. 找出触及 SR 区域的K线（价格区间与 [sr_price ± tolerance] 有交集）
    near_sr = (window_df["low"] <= sr_price + tolerance) & (
        window_df["high"] >= sr_price - tolerance
    )
    test_indices = window_df[near_sr].index.tolist()
    if not test_indices:
        return 0.0

    reactions = []
    n = len(window_df)

    for idx in test_indices:
        try:
            pos = window_df.index.get_loc(idx)
        except KeyError:
            continue

        # 确保后面还有至少1根K线用于观察反应
        if pos >= n - 1:
            continue

        # 动态确定实际可观察的反应期（不超过 max_lookahead_bars，也不越界）
        actual_lookahead = min(max_lookahead_bars, n - pos - 1)
        if actual_lookahead <= 0:
            continue

        future_slice = window_df.iloc[pos + 1 : pos + 1 + actual_lookahead]
        close_at_touch = window_df.loc[idx, "close"]

        # 【安全实现量价加权】：在反应循环内计算成交量统计
        # 关键原则：
        # 1. current_vol（测试点K线的成交量）可以使用，因为在该K线结束后是已知的
        # 2. avg_vol（基准平均成交量）必须来自更早的数据（不含当前K线）
        # 3. 使用 pos - volume_lookback : pos 确保不包含当前K线
        if use_volume_confirmation and "volume" in window_df.columns:
            # 用测试点之前的 volume_lookback 根K线计算平均成交量（不含当前K线）
            if pos >= volume_lookback:
                # 有足够历史数据：使用 pos - volume_lookback : pos（不包含 pos）
                ref_vols = window_df.iloc[pos - volume_lookback : pos]["volume"]
            else:
                # 数据不足：使用可用数据（至少1根，但不包含当前K线）
                ref_vols = window_df.iloc[: max(1, pos)]["volume"]

            # 清理 ref_vols 中的 inf/NaN 值，避免影响 avg_vol 计算
            ref_vols_clean = ref_vols.replace([np.inf, -np.inf], np.nan).dropna()
            if len(ref_vols_clean) > 0:
                avg_vol = ref_vols_clean.mean()
                if not np.isfinite(avg_vol) or avg_vol <= 0:
                    avg_vol = 1.0
            else:
                avg_vol = 1.0

            current_vol = window_df.loc[idx, "volume"]  # 当前K线成交量（可以使用）
            # 清理 current_vol 中的 inf/NaN 值
            if not np.isfinite(current_vol) or current_vol < 0:
                current_vol = 0.0

            EPS = 1e-10
            vol_ratio = current_vol / (avg_vol + EPS) if avg_vol > 0 else 1.0
            # 确保 vol_ratio 是有限值
            if not np.isfinite(vol_ratio):
                vol_ratio = 1.0

            # 体积确认因子（抑制极端值，限制在3倍以内）
            vol_factor = min(vol_ratio, 3.0)
            # 确保 vol_factor 是有限值
            if not np.isfinite(vol_factor):
                vol_factor = 1.0
        else:
            vol_factor = 1.0
            vol_ratio = 1.0

        if sr_type == "resistance":
            # 阻力：期望价格下跌 → 反应 = 触及时收盘价 - 未来最低价
            reaction = close_at_touch - future_slice["low"].min()
            if reaction > 0:  # 仅当确实下跌时才计入
                # 只有放量且方向正确的反应才计入
                if use_volume_confirmation and "volume" in window_df.columns:
                    if vol_ratio >= min_volume_ratio:
                        # 使用平方根加权，避免极端值影响过大
                        # 确保 current_atr 是有限正数（前面已检查，这里再次确认）
                        if np.isfinite(current_atr) and current_atr > 0:
                            weighted_reaction = (reaction / current_atr) * np.sqrt(
                                vol_factor
                            )
                            if np.isfinite(weighted_reaction):
                                reactions.append(weighted_reaction)
                else:
                    # 不使用量价确认，直接归一化
                    # 确保 current_atr 是有限正数（前面已检查，这里再次确认）
                    if np.isfinite(current_atr) and current_atr > 0:
                        normalized_reaction = reaction / current_atr
                        if np.isfinite(normalized_reaction):
                            reactions.append(normalized_reaction)
        else:  # support
            # 支撑：期望价格上涨 → 反应 = 未来最高价 - 触及时收盘价
            reaction = future_slice["high"].max() - close_at_touch
            if reaction > 0:  # 仅当确实反弹时才计入
                # 只有放量且方向正确的反应才计入
                if use_volume_confirmation and "volume" in window_df.columns:
                    if vol_ratio >= min_volume_ratio:
                        # 使用平方根加权，避免极端值影响过大
                        # 确保 current_atr 是有限正数（前面已检查，这里再次确认）
                        if np.isfinite(current_atr) and current_atr > 0:
                            weighted_reaction = (reaction / current_atr) * np.sqrt(
                                vol_factor
                            )
                            if np.isfinite(weighted_reaction):
                                reactions.append(weighted_reaction)
                else:
                    # 不使用量价确认，直接归一化
                    # 确保 current_atr 是有限正数（前面已检查，这里再次确认）
                    if np.isfinite(current_atr) and current_atr > 0:
                        normalized_reaction = reaction / current_atr
                        if np.isfinite(normalized_reaction):
                            reactions.append(normalized_reaction)

    # 汇总指标
    test_count = len(test_indices)
    valid_reaction_count = len(reactions)
    avg_reaction = np.mean(reactions) if reactions else 0.0
    recent_test_count = near_sr.tail(20).sum()  # 近20根K线内的测试次数

    # 标准化打分（抑制极端值，强调有效反应）
    test_score = min(test_count, 5) * 0.4
    reaction_score = min(avg_reaction * 2, 3.0) * 0.4  # avg_reaction=1.5 → 满分
    recency_score = min(recent_test_count, 3) * 0.2

    sqs = test_score + reaction_score + recency_score
    return float(sqs)


@register_feature("evaluate_level_quality_bidirectional", category="baseline")
def evaluate_level_quality_bidirectional(
    sr_price: float,
    df: pd.DataFrame,
    window: int = 60,
    tolerance_factor: float = 0.5,
    max_lookahead_bars: int = 3,
    use_volume_confirmation: bool = True,
    volume_lookback: int = 20,
    min_volume_ratio: float = 1.0,
) -> Dict[str, float | str]:
    """
    对未知类型的价格水平进行双向 SQS 评估（Bidirectional Testing）

    适用于：
    - 历史高点/低点（突破后角色可能转换）
    - 成交密集区中轴（POC）
    - OLS 通道中线、VWAP 等动态中轴
    - 其他无法明确判断是支撑还是阻力的水平

    Args:
        sr_price: 价格水平
        df: 历史K线数据，必须包含 ['high', 'low', 'close', 'atr', 'volume']
        window: 回看窗口长度
        tolerance_factor: ATR 容忍带系数
        max_lookahead_bars: 最大反应观察期
        use_volume_confirmation: 是否使用量价确认
        volume_lookback: 成交量回看窗口
        min_volume_ratio: 最小成交量比率

    Returns:
        包含以下键的字典：
        - sqs: 最佳 SQS 分数（支撑和阻力中的较高者）
        - type: 最佳类型（'support' 或 'resistance'）
        - support_sqs: 作为支撑的 SQS 分数
        - resistance_sqs: 作为阻力的 SQS 分数
    """
    # 分别计算支撑和阻力质量
    support_sqs = calculate_sqs(
        sr_price,
        df,
        window=window,
        tolerance_factor=tolerance_factor,
        sr_type="support",
        max_lookahead_bars=max_lookahead_bars,
        use_volume_confirmation=use_volume_confirmation,
        volume_lookback=volume_lookback,
        min_volume_ratio=min_volume_ratio,
    )

    resistance_sqs = calculate_sqs(
        sr_price,
        df,
        window=window,
        tolerance_factor=tolerance_factor,
        sr_type="resistance",
        max_lookahead_bars=max_lookahead_bars,
        use_volume_confirmation=use_volume_confirmation,
        volume_lookback=volume_lookback,
        min_volume_ratio=min_volume_ratio,
    )

    # 选择更优角色
    if support_sqs >= resistance_sqs:
        best_sqs = support_sqs
        best_type = "support"
    else:
        best_sqs = resistance_sqs
        best_type = "resistance"

    return {
        "sqs": float(best_sqs),
        "type": best_type,
        "support_sqs": float(support_sqs),
        "resistance_sqs": float(resistance_sqs),
    }


@register_feature("calculate_volume_price_confirmation", category="baseline")
def calculate_volume_price_confirmation(
    df: pd.DataFrame,
    breakout_idx: int,
    sr_price: float,
    lookback: int = 20,
    vol_threshold: float = 1.5,
    confirmation_bars: int = 3,
    sr_type: Optional[str] = None,
) -> float:
    """
    计算量价配合度（Volume-Price Confirmation）
    在突破发生时，评估量能是否支持

    Args:
        df: 包含 high, low, close, volume 的 K 线数据
        breakout_idx: 突破发生的索引位置
        sr_price: 突破的支撑/阻力价格
        lookback: 回看窗口大小（用于计算平均成交量）
        vol_threshold: 成交量倍数阈值（默认 1.5）
        confirmation_bars: 确认站稳的 K 线数量（默认 3）

    Returns:
        量价配合度分数（0-1，1 表示完全确认）
    """
    if breakout_idx < lookback or breakout_idx >= len(df) - confirmation_bars:
        return 0.0

    # 1. 成交量确认
    current_vol = df["volume"].iloc[breakout_idx]
    avg_vol = df["volume"].iloc[breakout_idx - lookback : breakout_idx].mean()
    if avg_vol <= 0:
        vol_ratio = 0.0
    else:
        vol_ratio = current_vol / avg_vol
    vol_confirmed = vol_ratio > vol_threshold

    # 2. 站稳确认：后几根 K 线是否持续在突破方向？
    if sr_type == "resistance":
        direction = 1
    elif sr_type == "support":
        direction = -1
    else:
        direction = 1 if df["close"].iloc[breakout_idx] >= sr_price else -1
    confirmed = True
    for i in range(1, min(confirmation_bars + 1, len(df) - breakout_idx)):
        if breakout_idx + i < len(df):
            price_diff = (df["close"].iloc[breakout_idx + i] - sr_price) * direction
            if price_diff <= 0:
                confirmed = False
                break

    # 综合评分：成交量确认权重 0.6，站稳确认权重 0.4
    score = 0.6 * (1.0 if vol_confirmed else 0.0) + 0.4 * (1.0 if confirmed else 0.0)

    return score


@register_feature("calculate_failed_breakout_reversal", category="baseline")
def calculate_failed_breakout_reversal(
    df: pd.DataFrame,
    breakout_idx: int,
    sr_price: float,
    sr_type: str,
    lookback: int = 20,
    confirmation_bars: int = 3,
    reversal_bars: int = 3,
) -> float:
    """
    计算失败突破反转强度（Failed Breakout Reversal）

    检测价格突破边界但未站稳，然后反向运动的情况。
    例如：价格突破 VWAP/均线/OLS 通道，但没站稳，然后反向运动回到边界内。

    Args:
        df: 包含 high, low, close, volume, atr 的 K 线数据
        breakout_idx: 突破发生的索引位置
        sr_price: 突破的支撑/阻力价格
        sr_type: 边界类型 ("support" 或 "resistance")
        lookback: 回看窗口大小（用于计算平均成交量）
        confirmation_bars: 确认站稳的 K 线数量（默认 3）
        reversal_bars: 确认反向运动的 K 线数量（默认 3）

    Returns:
        失败突破反转强度分数（0-1，1 表示强烈反转）
    """
    if breakout_idx < lookback or breakout_idx >= len(df) - max(
        confirmation_bars, reversal_bars
    ):
        return 0.0

    if "atr" not in df.columns or df["atr"].iloc[breakout_idx] <= 0:
        return 0.0

    # 1. 确认发生了突破
    if sr_type == "resistance":
        # 阻力：价格应该突破（close > sr_price）
        broke_out = df["close"].iloc[breakout_idx] > sr_price
        direction = 1  # 向上突破
    else:  # support
        # 支撑：价格应该突破（close < sr_price）
        broke_out = df["close"].iloc[breakout_idx] < sr_price
        direction = -1  # 向下突破

    if not broke_out:
        return 0.0

    # 2. 检测是否没站稳（在确认期内价格回到边界内）
    failed_confirmation = False
    for i in range(1, min(confirmation_bars + 1, len(df) - breakout_idx)):
        if breakout_idx + i < len(df):
            if sr_type == "resistance":
                # 阻力：如果价格回到阻力位下方，说明没站稳
                if df["close"].iloc[breakout_idx + i] <= sr_price:
                    failed_confirmation = True
                    break
            else:  # support
                # 支撑：如果价格回到支撑位上方，说明没站稳
                if df["close"].iloc[breakout_idx + i] >= sr_price:
                    failed_confirmation = True
                    break

    if not failed_confirmation:
        return 0.0  # 站稳了，不是失败突破

    # 3. 检测反向运动：后续 K 线是否持续反向（远离突破方向，回到边界内）
    reversal_confirmed = True
    reversal_start_idx = breakout_idx + confirmation_bars

    if reversal_start_idx >= len(df) - reversal_bars:
        return 0.0

    for i in range(reversal_bars):
        if reversal_start_idx + i < len(df):
            if sr_type == "resistance":
                # 阻力：价格应该持续下跌（回到阻力位下方）
                if (
                    df["close"].iloc[reversal_start_idx + i]
                    > df["close"].iloc[reversal_start_idx]
                ):
                    reversal_confirmed = False
                    break
                # 确认回到阻力位下方
                if df["close"].iloc[reversal_start_idx + i] > sr_price:
                    reversal_confirmed = False
                    break
            else:  # support
                # 支撑：价格应该持续上涨（回到支撑位上方）
                if (
                    df["close"].iloc[reversal_start_idx + i]
                    < df["close"].iloc[reversal_start_idx]
                ):
                    reversal_confirmed = False
                    break
                # 确认回到支撑位上方
                if df["close"].iloc[reversal_start_idx + i] < sr_price:
                    reversal_confirmed = False
                    break

    if not reversal_confirmed:
        return 0.0

    # 4. 计算反转幅度（以 ATR 为单位）
    if sr_type == "resistance":
        # 阻力：计算从突破点到反转后的跌幅
        breakout_price = df["close"].iloc[breakout_idx]
        reversal_price = df["close"].iloc[reversal_start_idx + reversal_bars - 1]
        reversal_magnitude = (breakout_price - reversal_price) / df["atr"].iloc[
            breakout_idx
        ]
    else:  # support
        # 支撑：计算从突破点到反转后的涨幅
        breakout_price = df["close"].iloc[breakout_idx]
        reversal_price = df["close"].iloc[reversal_start_idx + reversal_bars - 1]
        reversal_magnitude = (reversal_price - breakout_price) / df["atr"].iloc[
            breakout_idx
        ]

    # 5. 成交量确认：突破时成交量是否放大，反转时是否缩量或放量
    breakout_vol = df["volume"].iloc[breakout_idx]
    avg_vol = df["volume"].iloc[breakout_idx - lookback : breakout_idx].mean()
    if avg_vol <= 0:
        vol_ratio = 0.0
    else:
        vol_ratio = breakout_vol / avg_vol

    # 反转时的成交量
    reversal_vol = (
        df["volume"]
        .iloc[reversal_start_idx : reversal_start_idx + reversal_bars]
        .mean()
    )
    reversal_vol_ratio = reversal_vol / avg_vol if avg_vol > 0 else 0.0

    # 6. 综合评分
    # 突破幅度（0-1）：突破幅度越大，失败反转的信号越强
    if sr_type == "resistance":
        breakout_magnitude = (df["close"].iloc[breakout_idx] - sr_price) / df[
            "atr"
        ].iloc[breakout_idx]
    else:
        breakout_magnitude = (sr_price - df["close"].iloc[breakout_idx]) / df[
            "atr"
        ].iloc[breakout_idx]
    breakout_score = min(1.0, breakout_magnitude / 2.0)  # 2*ATR 的突破幅度为满分

    # 反转幅度（0-1）：反转幅度越大，信号越强
    reversal_score = min(1.0, reversal_magnitude / 2.0)  # 2*ATR 的反转幅度为满分

    # 成交量模式（0-1）：突破时放量，反转时缩量或放量都是信号
    vol_score = 0.0
    if vol_ratio > 1.5:  # 突破时放量
        if reversal_vol_ratio < 0.8:  # 反转时缩量（更典型）
            vol_score = 1.0
        elif reversal_vol_ratio > 1.2:  # 反转时也放量（恐慌性反转）
            vol_score = 0.8

    # 综合评分
    score = (
        0.3 * breakout_score  # 突破幅度权重 30%
        + 0.4 * reversal_score  # 反转幅度权重 40%
        + 0.3 * vol_score  # 成交量模式权重 30%
    )

    return score


def _get_sr_boundary_definitions(data: pd.DataFrame) -> List[Dict[str, str]]:
    """收集所有可用的 SR 边界定义"""
    boundaries: List[Dict[str, str]] = []

    def _register(name: str, column: str, sr_type: str, category: str) -> None:
        if column in data.columns:
            boundaries.append(
                {
                    "name": name,
                    "column": column,
                    "type": sr_type,
                    "category": category,
                }
            )

    _register("swing_high_s", "roll_high_s", "resistance", "swing_short")
    _register("swing_low_s", "roll_low_s", "support", "swing_short")
    _register("swing_high_l", "roll_high_l", "resistance", "swing_long")
    _register("swing_low_l", "roll_low_l", "support", "swing_long")
    _register("zigzag_high", "zz_high_value", "resistance", "zigzag")
    _register("zigzag_low", "zz_low_value", "support", "zigzag")
    _register("hal_high", "hal_high", "resistance", "hal")
    _register("hal_low", "hal_low", "support", "hal")
    _register("poc_level", "poc", "mid", "poc")
    _register("boll_upper", "bb_upper", "resistance", "bollinger")
    _register("boll_lower", "bb_lower", "support", "bollinger")
    _register("ols_upper", "ols_channel_upper", "resistance", "ols")
    _register("ols_lower", "ols_channel_lower", "support", "ols")
    _register("ols_mid", "ols_channel_mid", "mid", "ols")
    _register("vwap_level", "vwap", "mid", "vwap")

    return boundaries


def _compute_boundary_strengths(
    data: pd.DataFrame,
    boundaries: List[Dict[str, str]],
    window: int = 60,
    tolerance_factor: float = 0.5,
    cluster_weight: float = 0.15,
    compression_series: Optional[pd.Series] = None,
) -> Dict[str, pd.Series]:
    """
    计算每个边界的 SQS 强度，并考虑边界重合与压缩质量

    对于 mid 类型的边界（如 poc, ols_mid, vwap），不仅输出加权平均的 base，
    还输出原始分量和上下文特征，让模型学习交互关系：
    - {name}_support_sqs: 支撑方向的 SQS
    - {name}_resistance_sqs: 阻力方向的 SQS
    - {name}_price_above: 价格是否在边界上方 (1.0/0.0)
    - {name}_trend_down: 价格是否向下走 (1.0/0.0)
    - {name}_weight_support: 支撑权重
    - {name}_weight_resistance: 阻力权重
    - sqs_{name}: 加权平均的 base（保留用于向后兼容）
    """
    if "atr" not in data.columns or not boundaries:
        return {}

    atr_series = data["atr"].ffill()
    strengths: Dict[str, pd.Series] = {}
    sr_values = {b["name"]: data[b["column"]] for b in boundaries}
    comp_series = (
        compression_series.ffill()
        if compression_series is not None
        else pd.Series(0.0, index=data.index)
    )

    for boundary in boundaries:
        name = boundary["name"]
        column = boundary["column"]
        sr_type = boundary["type"]
        sr_series = sr_values[name]
        strength = pd.Series(0.0, index=data.index, dtype=float)

        # 对于 mid 类型，初始化额外的特征序列
        if sr_type == "mid":
            support_sqs_series = pd.Series(0.0, index=data.index, dtype=float)
            resistance_sqs_series = pd.Series(0.0, index=data.index, dtype=float)
            price_above_series = pd.Series(0.0, index=data.index, dtype=float)
            trend_down_series = pd.Series(0.0, index=data.index, dtype=float)
            weight_support_series = pd.Series(0.5, index=data.index, dtype=float)
            weight_resistance_series = pd.Series(0.5, index=data.index, dtype=float)

        for i in range(window, len(data)):
            sr_price = sr_series.iloc[i]
            if pd.isna(sr_price):
                continue

            # 【关键修复】：window_slice 不包含当前时刻 i，只使用历史数据 [i-window, i)
            # 这样 calculate_sqs 在计算反应强度时，可以使用 [i-window, i) 范围内的数据
            # 对于窗口内的每个测试点，可以使用该点之后、窗口结束之前的数据来计算反应
            window_slice = data.iloc[max(0, i - window) : i]
            try:
                # 对于 "mid" 类型的边界（如 poc, ols_mid, vwap），使用双向测试
                # 因为它们既可能是支撑也可能是阻力，取决于价格相对位置
                if sr_type == "mid":
                    # 使用双向测试，自动识别当前市场角色
                    level_quality = evaluate_level_quality_bidirectional(
                        sr_price,
                        window_slice,
                        window=window,
                        tolerance_factor=tolerance_factor,
                        use_volume_confirmation=True,
                    )
                    # 【关键修复】：对于 mid 类型，使用 support_sqs 和 resistance_sqs 的加权平均
                    # 而不是只选择较大的那个，避免偏向某一方向
                    # 权重根据价格趋势动态调整：
                    # - 价格在边界上方且向下走 → 边界作为支撑 → support 权重更高
                    # - 价格在边界下方且向上走 → 边界作为阻力 → resistance 权重更高
                    support_sqs_val = level_quality.get("support_sqs", 0.0)
                    resistance_sqs_val = level_quality.get("resistance_sqs", 0.0)

                    # 判断价格位置和趋势
                    current_price = data["close"].iloc[i]
                    if not pd.isna(current_price) and not pd.isna(sr_price):
                        price_above = current_price > sr_price

                        # 计算价格趋势（使用最近几根K线的平均变化）
                        lookback_trend = 3  # 使用最近3根K线判断趋势
                        if i >= lookback_trend:
                            recent_prices = data["close"].iloc[i - lookback_trend : i]
                            if len(recent_prices) > 1 and recent_prices.notna().all():
                                # 防止除零：如果起始价格为 0 或非常小，使用 pct_change 代替
                                start_price = recent_prices.iloc[0]
                                EPS = 1e-10
                                if abs(start_price) < EPS:
                                    # 如果起始价格接近 0，使用 pct_change 方法
                                    price_trend = recent_prices.pct_change().mean()
                                else:
                                    price_trend = (
                                        recent_prices.iloc[-1] - recent_prices.iloc[0]
                                    ) / (start_price + EPS)
                                # 检查结果是否有效
                                if not np.isfinite(price_trend):
                                    price_trend = 0.0
                                    print(
                                        f"   ⚠️  price_trend is inf/NaN at index {i}, start_price={start_price}"
                                    )
                                # price_trend > 0 表示上涨，< 0 表示下跌

                                # 动态权重分配：
                                # 1. 价格在边界上方且向下走 → 边界作为支撑 → support 权重更高
                                # 2. 价格在边界下方且向上走 → 边界作为阻力 → resistance 权重更高
                                if price_above and price_trend < 0:
                                    # 价格在边界上方且向下走，边界作为支撑
                                    weight_support = 0.7
                                    weight_resistance = 0.3
                                elif not price_above and price_trend > 0:
                                    # 价格在边界下方且向上走，边界作为阻力
                                    weight_support = 0.3
                                    weight_resistance = 0.7
                                else:
                                    # 其他情况（价格远离边界或趋势不明显），使用平衡权重
                                    weight_support = 0.5
                                    weight_resistance = 0.5
                            else:
                                # 数据不足，使用位置判断
                                if price_above:
                                    # 价格在边界上方，更可能作为支撑（如果回落）
                                    weight_support = 0.6
                                    weight_resistance = 0.4
                                else:
                                    # 价格在边界下方，更可能作为阻力（如果反弹）
                                    weight_support = 0.4
                                    weight_resistance = 0.6
                        else:
                            # 数据不足，使用位置判断
                            if price_above:
                                weight_support = 0.6
                                weight_resistance = 0.4
                            else:
                                weight_support = 0.4
                                weight_resistance = 0.6

                        base = (
                            support_sqs_val * weight_support
                            + resistance_sqs_val * weight_resistance
                        )

                        # 【增强方案】：不仅输出加权平均的 base，还输出原始分量和上下文特征
                        # 让模型自己学习交互关系，而不是依赖预定义的权重
                        support_sqs_series.iloc[i] = support_sqs_val
                        resistance_sqs_series.iloc[i] = resistance_sqs_val
                        price_above_series.iloc[i] = 1.0 if price_above else 0.0
                        trend_down_series.iloc[i] = 1.0 if price_trend < 0 else 0.0
                        weight_support_series.iloc[i] = weight_support
                        weight_resistance_series.iloc[i] = weight_resistance
                    else:
                        # 如果无法判断价格位置，使用简单平均
                        base = (support_sqs_val + resistance_sqs_val) / 2.0
                        # 仍然记录原始分量
                        support_sqs_series.iloc[i] = support_sqs_val
                        resistance_sqs_series.iloc[i] = resistance_sqs_val
                else:
                    # 对于明确类型的边界，使用量价确认增强的 SQS
                    base = calculate_sqs(
                        sr_price,
                        window_slice,
                        window=window,
                        tolerance_factor=tolerance_factor,
                        sr_type=sr_type,
                        use_volume_confirmation=True,  # 启用量价确认
                    )
            except Exception:
                base = 0.0
                # 如果发生异常，对于 mid 类型，保持特征为默认值（已在初始化时设置）

            tolerance = (
                atr_series.iloc[i] * tolerance_factor
                if not pd.isna(atr_series.iloc[i])
                else np.nan
            )
            cluster_bonus = 0.0
            if not np.isnan(tolerance):
                for other in boundaries:
                    if other["name"] == name:
                        continue
                    other_val = sr_values[other["name"]].iloc[i]
                    if pd.notna(other_val) and abs(other_val - sr_price) <= tolerance:
                        cluster_bonus += cluster_weight

            compression_bonus = (
                comp_series.iloc[i] if not np.isnan(comp_series.iloc[i]) else 0.0
            )
            score = base * (1.0 + cluster_bonus) + 0.2 * compression_bonus
            strength.iloc[i] = score

        # 对于 mid 类型，不仅输出加权平均的 base，还输出原始分量和上下文特征
        if sr_type == "mid":
            strengths[f"sqs_{name}"] = strength.shift(1).fillna(
                0.0
            )  # 保留加权平均的 base
            # 输出原始分量和上下文特征，让模型学习交互关系
            strengths[f"{name}_support_sqs"] = support_sqs_series.shift(1).fillna(0.0)
            strengths[f"{name}_resistance_sqs"] = resistance_sqs_series.shift(1).fillna(
                0.0
            )
            strengths[f"{name}_price_above"] = price_above_series.shift(1).fillna(0.0)
            strengths[f"{name}_trend_down"] = trend_down_series.shift(1).fillna(0.0)
            strengths[f"{name}_weight_support"] = weight_support_series.shift(1).fillna(
                0.5
            )
            strengths[f"{name}_weight_resistance"] = weight_resistance_series.shift(
                1
            ).fillna(0.5)
        else:
            strengths[f"sqs_{name}"] = strength.shift(1).fillna(0.0)

    return strengths


def _compute_breakout_confirmation_and_role_flip(
    data: pd.DataFrame,
    boundaries: List[Dict[str, str]],
    lookback: int = 20,
    confirmation_bars: int = 3,
    max_retest_bars: int = 10,
) -> Dict[str, pd.Series]:
    """
    计算突破确认和角色转换特征

    包括：
    1. 突破确认概率：基于量价关系判断真伪突破
    2. 角色转换概率：支撑/阻力角色转换的概率
    3. 转换状态显式标记：post_breakout_retest, post_breakdown_retest 等

    这些特征帮助模型理解"同一个位置，在不同市场环境下会扮演完全相反的角色"
    """
    if "atr" not in data.columns or not boundaries:
        return {}

    features: Dict[str, pd.Series] = {}
    atr_series = data["atr"].ffill()

    for boundary in boundaries:
        name = boundary["name"]
        column = boundary["column"]
        sr_type = boundary["type"]
        sr_series = data[column]

        # 初始化特征序列
        breakout_confirmation = pd.Series(0.0, index=data.index, dtype=float)
        role_flip_prob = pd.Series(0.0, index=data.index, dtype=float)
        post_breakout_retest = pd.Series(0.0, index=data.index, dtype=float)
        post_breakdown_retest = pd.Series(0.0, index=data.index, dtype=float)

        # 记录最近的突破事件（用于检测回踩）
        last_breakout_idx = -1
        last_breakout_direction = 0  # 1=向上突破, -1=向下突破
        last_breakout_price = np.nan

        for i in range(lookback + confirmation_bars + max_retest_bars, len(data)):
            sr_price = sr_series.iloc[i]
            if pd.isna(sr_price):
                continue

            current_price = data["close"].iloc[i]
            current_high = data["high"].iloc[i]
            current_low = data["low"].iloc[i]
            current_volume = data["volume"].iloc[i]

            # 计算 ATR 用于归一化
            current_atr = atr_series.iloc[i] if not pd.isna(atr_series.iloc[i]) else 1.0

            # 1. 检测突破（使用历史数据）
            breakout_idx = i - confirmation_bars - max_retest_bars
            if breakout_idx >= 0:
                prev_close = (
                    data["close"].iloc[breakout_idx - 1]
                    if breakout_idx > 0
                    else current_price
                )
                breakout_close = data["close"].iloc[breakout_idx]
                breakout_high = data["high"].iloc[breakout_idx]
                breakout_low = data["low"].iloc[breakout_idx]
                breakout_volume = data["volume"].iloc[breakout_idx]

                # 检测突破方向
                detected_breakout = False
                breakout_direction = 0

                if sr_type == "resistance":
                    if prev_close <= sr_price and breakout_high > sr_price:
                        detected_breakout = True
                        breakout_direction = 1
                elif sr_type == "support":
                    if prev_close >= sr_price and breakout_low < sr_price:
                        detected_breakout = True
                        breakout_direction = -1
                elif sr_type == "mid":
                    if (prev_close <= sr_price and breakout_close > sr_price) or (
                        prev_close >= sr_price and breakout_close < sr_price
                    ):
                        detected_breakout = True
                        breakout_direction = 1 if breakout_close > sr_price else -1

                if detected_breakout:
                    last_breakout_idx = breakout_idx
                    last_breakout_direction = breakout_direction
                    last_breakout_price = sr_price

                    # 计算突破确认概率（基于量价关系）
                    # 使用历史数据计算平均成交量
                    if breakout_idx >= lookback:
                        avg_vol = (
                            data["volume"]
                            .iloc[breakout_idx - lookback : breakout_idx]
                            .mean()
                        )
                    else:
                        avg_vol = (
                            data["volume"].iloc[:breakout_idx].mean()
                            if breakout_idx > 0
                            else 1.0
                        )

                    volume_ratio = breakout_volume / avg_vol if avg_vol > 0 else 1.0

                    # 突破幅度（归一化）
                    breakout_size = (
                        abs(breakout_close - sr_price) / current_atr
                        if current_atr > 0
                        else 0.0
                    )

                    # 突破后回踩速度（在 confirmation_bars 内是否回踩）
                    retrace_speed = 0.0
                    if breakout_idx + confirmation_bars < i:
                        post_breakout_slice = data.iloc[
                            breakout_idx + 1 : breakout_idx + 1 + confirmation_bars
                        ]
                        if len(post_breakout_slice) > 0:
                            if breakout_direction == 1:  # 向上突破
                                min_after = post_breakout_slice["low"].min()
                                retrace_pct = (
                                    (sr_price - min_after) / current_atr
                                    if current_atr > 0
                                    else 0.0
                                )
                                retrace_speed = max(
                                    0.0, retrace_pct
                                )  # 回踩越深，速度越快
                            else:  # 向下突破
                                max_after = post_breakout_slice["high"].max()
                                retrace_pct = (
                                    (max_after - sr_price) / current_atr
                                    if current_atr > 0
                                    else 0.0
                                )
                                retrace_speed = max(0.0, retrace_pct)

                    # 突破确认概率 = sigmoid(量能验证 * 0.5 + 突破幅度 * 0.3 - 回踩速度 * 0.2)
                    import math

                    confirmation_score = (
                        min(volume_ratio, 3.0) * 0.5
                        + min(breakout_size, 2.0) * 0.3
                        - min(retrace_speed, 1.5) * 0.2
                    )
                    breakout_confirmation.iloc[i] = 1.0 / (
                        1.0 + math.exp(-confirmation_score)
                    )  # Sigmoid

            # 2. 检测回踩（突破后回踩原边界）
            if last_breakout_idx >= 0 and i > last_breakout_idx:
                # 检查是否回踩到原边界附近（在 ATR 范围内）
                tolerance = current_atr * 0.5
                near_original_sr = abs(current_price - last_breakout_price) <= tolerance

                if last_breakout_direction == 1:  # 向上突破后回踩
                    if (
                        near_original_sr
                        and current_low <= last_breakout_price + tolerance
                    ):
                        post_breakout_retest.iloc[i] = 1.0
                elif last_breakout_direction == -1:  # 向下突破后回踩
                    if (
                        near_original_sr
                        and current_high >= last_breakout_price - tolerance
                    ):
                        post_breakdown_retest.iloc[i] = 1.0

            # 3. 计算角色转换概率（仅对 mid 类型）
            if sr_type == "mid":
                # 获取双向 SQS（如果已计算）
                support_sqs_col = f"{name}_support_sqs"
                resistance_sqs_col = f"{name}_resistance_sqs"

                if (
                    support_sqs_col in data.columns
                    and resistance_sqs_col in data.columns
                ):
                    support_sqs = (
                        data[support_sqs_col].iloc[i] if i < len(data) else 0.0
                    )
                    resistance_sqs = (
                        data[resistance_sqs_col].iloc[i] if i < len(data) else 0.0
                    )

                    # 支撑/阻力主导强度差
                    strength_diff = abs(support_sqs - resistance_sqs)

                    # 价格位置（+1=在边界上方，-1=在边界下方）
                    price_position = 1.0 if current_price > sr_price else -1.0

                    # 角色转换临界点（价格突破后回踩原阻力/支撑）
                    flip_zone = 0.0
                    if last_breakout_idx >= 0 and i > last_breakout_idx:
                        if (last_breakout_direction == 1 and price_position < 0) or (
                            last_breakout_direction == -1 and price_position > 0
                        ):
                            flip_zone = 1.0

                    # 转换概率 = sigmoid(强度差 * 0.7 + 位置验证 * 1.2)
                    import math

                    flip_score = strength_diff * 0.7 + flip_zone * 1.2
                    role_flip_prob.iloc[i] = 1.0 / (
                        1.0 + math.exp(-flip_score)
                    )  # Sigmoid

        # 保存特征（shift(1) 确保因果性）
        features[f"{name}_breakout_confirmation"] = breakout_confirmation.shift(
            1
        ).fillna(0.0)
        features[f"{name}_role_flip_prob"] = role_flip_prob.shift(1).fillna(0.0)
        features[f"{name}_post_breakout_retest"] = post_breakout_retest.shift(1).fillna(
            0.0
        )
        features[f"{name}_post_breakdown_retest"] = post_breakdown_retest.shift(
            1
        ).fillna(0.0)

    return features


def _add_breakout_quality_features(
    data: pd.DataFrame,
    boundaries: List[Dict[str, str]],
) -> pd.DataFrame:
    """
    添加4类12个核心特征，用于让模型自动学习突破质量判断

    整体特征体系设计（共4类12个核心特征）：

    A. 结构质量（3个）
    1. sqs - SR测试次数+反应强度+时间衰减（已有）
    2. dist_to_nearest_sr - 当前价距最近SR的距离（已有）
    3. sr_confluence - 是否多个周期SR重合（新增）

    B. 突破动能（3个）
    1. vol_ratio - 突破K线量比（已有 volume_ratio）
    2. order_flow_delta - 主动买卖差（新增，基于 delta 或 buy_qty - sell_qty）
    3. breakout_speed - 突破K线实体/影线比（新增）

    C. 动能持续性（3个）
    1. follow_through_1 - 第2根K线是否继续新高/新低（新增）
    2. follow_through_2 - 第3根K线是否站稳（新增）
    3. momentum_decay - 突破后3根K线的斜率变化（新增）

    D. 市场环境（3个）
    1. compression_score - 布林带宽度 / ATR 比值（已有 compression_confidence）
    2. trend_strength - ADX(14) 或 slope of MA50（新增）
    3. time_phase - 是否在活跃交易时段（已有 hour_sin, hour_cos）
    """
    if data.empty:
        return data

    # 确保有必要的列
    if "atr" not in data.columns:
        data["atr"] = _compute_atr(data)

    # A.3. SR重合度（sr_confluence）
    # 检查是否有多个边界在相近位置（ATR范围内）
    sr_confluence = pd.Series(0.0, index=data.index, dtype=float)
    if boundaries:
        for i in range(len(data)):
            current_price = data["close"].iloc[i]
            current_atr = (
                data["atr"].iloc[i] if not pd.isna(data["atr"].iloc[i]) else 1.0
            )
            tolerance = current_atr * 0.5

            # 收集所有非NaN的边界价格
            nearby_boundaries = []
            for boundary in boundaries:
                col = boundary["column"]
                if col in data.columns:
                    sr_price = data[col].iloc[i]
                    if (
                        not pd.isna(sr_price)
                        and abs(sr_price - current_price) <= tolerance * 2
                    ):
                        nearby_boundaries.append(sr_price)

            # 计算在 tolerance 范围内的边界数量
            if len(nearby_boundaries) >= 2:
                # 检查有多少个边界在 tolerance 范围内
                count = 0
                for sr1 in nearby_boundaries:
                    for sr2 in nearby_boundaries:
                        if sr1 != sr2 and abs(sr1 - sr2) <= tolerance:
                            count += 1
                sr_confluence.iloc[i] = min(count / 2.0, 3.0) / 3.0  # 归一化到 [0, 1]

    data["sr_confluence"] = sr_confluence.shift(1).fillna(0.0)

    # B.2. 订单流差值（order_flow_delta）
    # 注意：如果已经加载了 order_flow features，可以直接使用：
    # - cvd_normalized（单根K线，归一化，等同于 order_flow_delta）
    # - cvd_change_1（单根K线，原始值）
    # - cvd_change_5（5根K线周期）
    # - cvd_change_20（20根K线周期）
    #
    # 这里为了保持特征命名一致性，优先使用 cvd_normalized，如果没有则计算
    if "cvd_normalized" in data.columns:
        # 直接使用已有的 cvd_normalized（单根K线，归一化）
        data["order_flow_delta"] = data["cvd_normalized"].shift(1).fillna(0.0)
    elif "cvd_change_1" in data.columns:
        # 使用 cvd_change_1（单根K线，原始值），需要归一化
        if "volume" in data.columns:
            total_vol = data["volume"].replace(0, np.nan)
            order_flow_delta_normalized = (data["cvd_change_1"] / total_vol).fillna(0.0)
        else:
            order_flow_delta_normalized = data["cvd_change_1"]
        data["order_flow_delta"] = order_flow_delta_normalized.shift(1).fillna(0.0)
    elif "delta" in data.columns:
        # 使用 delta，需要归一化
        if "volume" in data.columns:
            total_vol = data["volume"].replace(0, np.nan)
            order_flow_delta_normalized = (data["delta"] / total_vol).fillna(0.0)
        else:
            order_flow_delta_normalized = data["delta"]
        data["order_flow_delta"] = order_flow_delta_normalized.shift(1).fillna(0.0)
    elif "buy_qty" in data.columns and "sell_qty" in data.columns:
        # 从 buy_qty 和 sell_qty 计算
        order_flow_delta = data["buy_qty"] - data["sell_qty"]
        if "volume" in data.columns:
            total_vol = data["volume"].replace(0, np.nan)
            order_flow_delta_normalized = (order_flow_delta / total_vol).fillna(0.0)
        else:
            order_flow_delta_normalized = order_flow_delta
        data["order_flow_delta"] = order_flow_delta_normalized.shift(1).fillna(0.0)
    else:
        # 如果没有订单流数据，使用0
        data["order_flow_delta"] = pd.Series(0.0, index=data.index)

    # B.3. 突破速度（breakout_speed）
    # 突破K线实体/影线比
    breakout_speed = pd.Series(0.0, index=data.index, dtype=float)
    for i in range(1, len(data)):
        high = data["high"].iloc[i]
        low = data["low"].iloc[i]
        open_price = data["open"].iloc[i]
        close = data["close"].iloc[i]

        # 实体大小
        body = abs(close - open_price)
        # 上影线
        upper_shadow = high - max(open_price, close)
        # 下影线
        lower_shadow = min(open_price, close) - low
        # 总影线
        total_shadow = upper_shadow + lower_shadow

        # 突破速度 = 实体 / (实体 + 影线)
        if body + total_shadow > 0:
            speed = body / (body + total_shadow)
        else:
            speed = 0.0

        breakout_speed.iloc[i] = speed

    data["breakout_speed"] = breakout_speed.shift(1).fillna(0.0)

    # C. 动能持续性特征
    # 需要检测突破事件，然后计算后续K线的表现
    follow_through_1 = pd.Series(0.0, index=data.index, dtype=float)
    follow_through_2 = pd.Series(0.0, index=data.index, dtype=float)
    momentum_decay = pd.Series(0.0, index=data.index, dtype=float)

    # 检测突破事件（相对于最近SR）
    if "dist_to_nearest_sr" in data.columns and len(boundaries) > 0:
        # 找到最近的SR边界
        nearest_sr = pd.Series(index=data.index, dtype=float)
        for boundary in boundaries:
            col = boundary["column"]
            if col in data.columns:
                if nearest_sr.isna().all():
                    nearest_sr = data[col].copy()
                else:
                    # 选择距离当前价格更近的边界
                    current_price = data["close"]
                    dist1 = abs(nearest_sr - current_price)
                    dist2 = abs(data[col] - current_price)
                    nearest_sr = pd.Series(
                        np.where(dist2 < dist1, data[col], nearest_sr), index=data.index
                    )

        for i in range(3, len(data)):
            if pd.isna(nearest_sr.iloc[i]):
                continue

            sr_price = nearest_sr.iloc[i]
            prev_close = data["close"].iloc[i - 1]
            curr_close = data["close"].iloc[i]
            curr_high = data["high"].iloc[i]
            curr_low = data["low"].iloc[i]

            # 检测突破方向
            breakout_direction = 0
            if prev_close <= sr_price and curr_high > sr_price:
                breakout_direction = 1  # 向上突破
            elif prev_close >= sr_price and curr_low < sr_price:
                breakout_direction = -1  # 向下突破

            if breakout_direction != 0:
                # C.1. follow_through_1: 第2根K线是否继续新高/新低
                if i + 1 < len(data):
                    next_high = data["high"].iloc[i + 1]
                    next_low = data["low"].iloc[i + 1]
                    if breakout_direction == 1:
                        # 向上突破：第2根K线是否创新高
                        follow_through_1.iloc[i + 1] = (
                            1.0 if next_high > curr_high else 0.0
                        )
                    else:
                        # 向下突破：第2根K线是否创新低
                        follow_through_1.iloc[i + 1] = (
                            1.0 if next_low < curr_low else 0.0
                        )

                # C.2. follow_through_2: 第3根K线是否站稳
                if i + 2 < len(data):
                    third_close = data["close"].iloc[i + 2]
                    if breakout_direction == 1:
                        # 向上突破：第3根K线收盘价是否仍在SR上方
                        follow_through_2.iloc[i + 2] = (
                            1.0 if third_close > sr_price else 0.0
                        )
                    else:
                        # 向下突破：第3根K线收盘价是否仍在SR下方
                        follow_through_2.iloc[i + 2] = (
                            1.0 if third_close < sr_price else 0.0
                        )

                # C.3. momentum_decay: 突破后3根K线的斜率变化
                if i + 3 < len(data):
                    # 计算突破后3根K线的价格变化
                    prices_after = data["close"].iloc[i + 1 : i + 4].values
                    if len(prices_after) == 3 and all(~np.isnan(prices_after)):
                        # 计算斜率（使用线性回归）
                        x = np.array([1, 2, 3])
                        y = prices_after
                        slope = np.polyfit(x, y, 1)[0]

                        # 归一化斜率（除以ATR）
                        current_atr = (
                            data["atr"].iloc[i]
                            if not pd.isna(data["atr"].iloc[i])
                            else 1.0
                        )
                        normalized_slope = (
                            slope / current_atr if current_atr > 0 else 0.0
                        )

                        # 动能衰减 = 1 - abs(斜率)（斜率越小，衰减越大）
                        momentum_decay.iloc[i + 3] = 1.0 - min(
                            abs(normalized_slope), 1.0
                        )

    data["follow_through_1"] = follow_through_1.shift(1).fillna(0.0)
    data["follow_through_2"] = follow_through_2.shift(1).fillna(0.0)
    data["momentum_decay"] = momentum_decay.shift(1).fillna(0.0)

    # D.2. 趋势强度（trend_strength）
    # 使用 ADX(14) 或 MA50 斜率
    if "close" in data.columns:
        # 计算 MA50
        ma50 = data["close"].rolling(window=50, min_periods=1).mean()

        # 计算 MA50 斜率（使用线性回归）
        trend_strength = pd.Series(0.0, index=data.index, dtype=float)
        for i in range(50, len(data)):
            if i >= 14:
                ma_window = ma50.iloc[i - 13 : i + 1].values
                if len(ma_window) == 14 and all(~np.isnan(ma_window)):
                    x = np.arange(14)
                    slope = np.polyfit(x, ma_window, 1)[0]
                    # 归一化斜率（除以当前价格）
                    current_price = data["close"].iloc[i]
                    normalized_slope = (
                        slope / current_price if current_price > 0 else 0.0
                    )
                    trend_strength.iloc[i] = normalized_slope * 100  # 放大100倍便于观察

        # 如果可以使用 TA-Lib，优先使用 ADX
        try:
            import talib

            high = data["high"].values
            low = data["low"].values
            close = data["close"].values
            adx = _talib().ADX(high, low, close, timeperiod=14)
            # ADX 范围是 0-100，归一化到 [0, 1]
            trend_strength = pd.Series(adx / 100.0, index=data.index)
        except Exception:
            pass  # 如果 TA-Lib 不可用，使用 MA50 斜率

    data["trend_strength"] = trend_strength.shift(1).fillna(0.0)

    return data


def _compute_boundary_volume_confirmations(
    data: pd.DataFrame,
    boundaries: List[Dict[str, str]],
    lookback: int = 20,
    vol_threshold: float = 1.5,
    confirmation_bars: int = 3,
) -> Dict[str, pd.Series]:
    """计算每个边界的量价配合度序列"""
    confirmations: Dict[str, pd.Series] = {}
    if not boundaries:
        return confirmations

    for boundary in boundaries:
        column = boundary["column"]
        sr_type = boundary["type"]
        sr_series = data[column]
        conf = pd.Series(0.0, index=data.index, dtype=float)

        # 【关键修复】：在时刻 i，只能使用历史数据 [0, i] 来检测突破和计算确认
        # 不能使用未来数据来确认是否站稳，因为这会导致数据泄漏
        # 解决方案：在突破发生后的 confirmation_bars 根K线之后，才计算确认分数
        # 这样在时刻 i，我们使用的是历史突破事件（发生在 i - confirmation_bars 之前）的确认结果
        for i in range(lookback + confirmation_bars, len(data)):
            # 检测突破：使用历史数据（i - confirmation_bars 时刻的突破）
            # 这样在时刻 i，我们计算的是历史突破的确认结果
            breakout_check_idx = i - confirmation_bars
            if breakout_check_idx < 0:
                continue

            sr_price = sr_series.iloc[breakout_check_idx - 1]
            if pd.isna(sr_price):
                continue

            prev_close = data["close"].iloc[breakout_check_idx - 1]
            curr_close = data["close"].iloc[breakout_check_idx]
            breakout = False
            direction = 0

            if (
                sr_type == "resistance"
                and prev_close <= sr_price
                and curr_close > sr_price
            ):
                breakout = True
                direction = 1
            elif (
                sr_type == "support"
                and prev_close >= sr_price
                and curr_close < sr_price
            ):
                breakout = True
                direction = -1
            elif sr_type == "mid":
                if (curr_close - sr_price) * (prev_close - sr_price) <= 0:
                    breakout = True
                    direction = 1 if curr_close > sr_price else -1

            if breakout:
                # 【修复】：只使用历史数据 [0, i] 来计算确认
                # 在时刻 i，我们已经有了 [breakout_check_idx, i] 的数据来确认是否站稳
                try:
                    score = calculate_volume_price_confirmation(
                        data.iloc[: i + 1],  # 只使用历史数据，不包含未来
                        breakout_check_idx,  # 突破发生在 breakout_check_idx
                        sr_price,
                        lookback=lookback,
                        vol_threshold=vol_threshold,
                        confirmation_bars=confirmation_bars,
                        sr_type=(
                            sr_type
                            if sr_type != "mid"
                            else ("resistance" if direction == 1 else "support")
                        ),
                    )
                except Exception:
                    score = 0.0
                conf.iloc[i] = score

        confirmations[f"volume_price_confirmation_{boundary['name']}"] = conf.shift(
            1
        ).fillna(0.0)

    return confirmations


def _add_price_action_features(
    data: pd.DataFrame,
    boundaries: List[Dict[str, str]],
    compression_series: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    添加价格行为特征（Price Action Features）

    包括：
    1. 突破状态（相对于最近SR）
    2. 反转信号（未到SR就回头）
    3. 假突破迹象
    4. SR结构特征（SR密度）
    5. 多时间框架上下文（趋势、压缩、波动率）
    """
    if data.empty or not boundaries:
        return data

    # 确保有必要的列
    if "atr" not in data.columns:
        data["atr"] = _compute_atr(data)

    # 1. 找到最近的SR边界
    # 收集所有边界价格
    all_boundary_prices = []
    for boundary in boundaries:
        col = boundary["column"]
        if col in data.columns:
            all_boundary_prices.append(data[col])

    if not all_boundary_prices:
        return data

    # 合并所有边界价格，找到最近的非NaN值
    boundary_df = pd.concat(all_boundary_prices, axis=1)

    # 计算到最近边界的距离和方向
    nearest_sr = pd.Series(index=data.index, dtype=float)
    dist_to_sr = pd.Series(index=data.index, dtype=float)
    direction_to_sr = pd.Series(index=data.index, dtype=float)
    nearest_sr_type = pd.Series(index=data.index, dtype=str)

    for i in range(len(data)):
        # 获取当前时刻所有非NaN的边界价格
        valid_boundaries = boundary_df.iloc[i].dropna()
        if len(valid_boundaries) == 0:
            continue

        current_price = data["close"].iloc[i]
        # 找到最近的边界
        distances = abs(valid_boundaries - current_price)
        nearest_idx = distances.idxmin()
        nearest_price = valid_boundaries[nearest_idx]

        nearest_sr.iloc[i] = nearest_price
        dist_to_sr.iloc[i] = (
            (current_price - nearest_price) / current_price
            if current_price > 0
            else 0.0
        )
        direction_to_sr.iloc[i] = 1.0 if current_price < nearest_price else -1.0

        # 找到对应的边界类型
        for boundary in boundaries:
            if boundary["column"] == nearest_idx:
                nearest_sr_type.iloc[i] = boundary["type"]
                break

    # 2. 突破状态（相对于最近SR）
    breakout_status = pd.Series(0, index=data.index, dtype=int)
    prev_close = data["close"].shift(1)

    for i in range(1, len(data)):
        if pd.isna(nearest_sr.iloc[i]) or pd.isna(prev_close.iloc[i]):
            continue

        current_high = data["high"].iloc[i]
        current_low = data["low"].iloc[i]
        current_close = data["close"].iloc[i]
        nearest_sr_price = nearest_sr.iloc[i]
        sr_type = nearest_sr_type.iloc[i]

        if sr_type == "resistance":
            # 刚上破阻力
            if (
                current_high > nearest_sr_price
                and prev_close.iloc[i] <= nearest_sr_price
            ):
                breakout_status.iloc[i] = 1
        elif sr_type == "support":
            # 刚下破支撑
            if (
                current_low < nearest_sr_price
                and prev_close.iloc[i] >= nearest_sr_price
            ):
                breakout_status.iloc[i] = -1
        elif sr_type == "mid":
            # 对于mid类型（如VWAP），检测穿越
            if (
                current_close > nearest_sr_price
                and prev_close.iloc[i] <= nearest_sr_price
            ):
                breakout_status.iloc[i] = 1
            elif (
                current_close < nearest_sr_price
                and prev_close.iloc[i] >= nearest_sr_price
            ):
                breakout_status.iloc[i] = -1

    data["breakout_status"] = breakout_status.shift(1).fillna(0)

    # 3. 反转信号（未到SR就回头）
    price_reversed_before_sr = pd.Series(False, index=data.index, dtype=bool)
    volume_spike_threshold = 1.5

    for i in range(1, len(data)):
        if pd.isna(nearest_sr.iloc[i]) or pd.isna(dist_to_sr.iloc[i]):
            continue

        # 应上涨但回落（距离SR为正，方向为正，但价格下跌）
        if dist_to_sr.iloc[i] > 0 and direction_to_sr.iloc[i] == 1:
            if data["close"].iloc[i] < data["close"].iloc[i - 1]:
                # 检查成交量是否放大
                if i >= 20:
                    avg_vol = data["volume"].iloc[i - 20 : i].mean()
                    if (
                        avg_vol > 0
                        and data["volume"].iloc[i] / avg_vol > volume_spike_threshold
                    ):
                        price_reversed_before_sr.iloc[i] = True
        # 应下跌但反弹（距离SR为负，方向为负，但价格上涨）
        elif dist_to_sr.iloc[i] < 0 and direction_to_sr.iloc[i] == -1:
            if data["close"].iloc[i] > data["close"].iloc[i - 1]:
                # 检查成交量是否放大
                if i >= 20:
                    avg_vol = data["volume"].iloc[i - 20 : i].mean()
                    if (
                        avg_vol > 0
                        and data["volume"].iloc[i] / avg_vol > volume_spike_threshold
                    ):
                        price_reversed_before_sr.iloc[i] = True

    data["price_reversed_before_sr"] = (
        price_reversed_before_sr.shift(1).fillna(False).astype(int)
    )

    # 4. 假突破迹象（突破后3根K线是否收回？）
    # 【关键修复】：在时刻 i，只能使用历史数据来判断假突破
    # 解决方案：在时刻 i，检查发生在 i - 3 的突破是否在后续被收回
    fake_breakout = pd.Series(False, index=data.index, dtype=bool)

    for i in range(3, len(data)):
        # 检查发生在 i - 3 的突破是否在后续被收回
        check_idx = i - 3
        if check_idx < 0 or breakout_status.iloc[check_idx] == 0:
            continue

        nearest_sr_price = nearest_sr.iloc[check_idx]
        if pd.isna(nearest_sr_price):
            continue

        # 检查突破后3根K线是否收回（使用历史数据）
        if breakout_status.iloc[check_idx] == 1:  # 向上突破
            # 如果后续收盘价回到阻力位下方，可能是假突破
            # 在时刻 i，我们已经有了 [check_idx, i] 的数据来判断
            if i >= check_idx + 1:
                # 检查从 check_idx + 1 到 i 的收盘价是否回到阻力位下方
                if (data["close"].iloc[check_idx + 1 : i + 1] < nearest_sr_price).any():
                    fake_breakout.iloc[i] = True
        elif breakout_status.iloc[check_idx] == -1:  # 向下突破
            # 如果后续收盘价回到支撑位上方，可能是假突破
            if i >= check_idx + 1:
                # 检查从 check_idx + 1 到 i 的收盘价是否回到支撑位上方
                if (data["close"].iloc[check_idx + 1 : i + 1] > nearest_sr_price).any():
                    fake_breakout.iloc[i] = True

    data["fake_breakout"] = fake_breakout.shift(1).fillna(False).astype(int)

    # 5. SR密度（是否处于SR密集区？）
    sr_density = pd.Series(0.0, index=data.index, dtype=float)
    tolerance_window = 0.5  # ATR倍数

    for i in range(len(data)):
        if pd.isna(data["atr"].iloc[i]) or data["atr"].iloc[i] <= 0:
            continue

        current_price = data["close"].iloc[i]
        tolerance = data["atr"].iloc[i] * tolerance_window

        # 计算在当前价格 ± tolerance 范围内的边界数量
        count = 0
        for boundary in boundaries:
            col = boundary["column"]
            if col in data.columns:
                sr_price = data[col].iloc[i]
                if pd.notna(sr_price):
                    if abs(sr_price - current_price) <= tolerance:
                        count += 1

        sr_density.iloc[i] = count

    data["sr_density"] = sr_density.shift(1).fillna(0.0)

    # 6. 多时间框架上下文
    # 6.1 趋势方向（基于均线，简化：使用50和200周期均线）
    if "close" in data.columns:
        ma50 = data["close"].rolling(window=50, min_periods=1).mean()
        ma200 = data["close"].rolling(window=200, min_periods=1).mean()
        trend_4h = pd.Series(0, index=data.index, dtype=int)
        trend_4h[ma50 > ma200] = 1
        trend_4h[ma50 < ma200] = -1
        data["trend_context"] = trend_4h.shift(1).fillna(0)

    # 6.2 压缩状态（基于布林带宽度）
    if (
        "bb_upper" in data.columns
        and "bb_lower" in data.columns
        and "bb_middle" in data.columns
    ):
        boll_width = (data["bb_upper"] - data["bb_lower"]) / data["bb_middle"].replace(
            0, np.nan
        )
        compression_score = 1.0 / (1.0 + boll_width)
        data["compression_score"] = compression_score.shift(1).fillna(0.0)
    else:
        # 如果没有布林带，使用ATR作为替代
        if "atr" in data.columns and "close" in data.columns:
            atr_normalized = data["atr"] / data["close"].replace(0, np.nan)
            compression_score = 1.0 / (1.0 + atr_normalized * 10)  # 缩放因子
            data["compression_score"] = compression_score.shift(1).fillna(0.0)

    # 6.3 波动率状态
    if "atr" in data.columns:
        atr_20_avg = data["atr"].rolling(window=20, min_periods=1).mean()
        volatility_regime = data["atr"] / atr_20_avg.replace(0, np.nan)
        data["volatility_regime"] = volatility_regime.shift(1).fillna(1.0)

    # 标准化距离特征
    data["dist_to_nearest_sr"] = dist_to_sr.shift(1).fillna(0.0)
    data["direction_to_nearest_sr"] = direction_to_sr.shift(1).fillna(0.0)

    # 【新增】：添加4类12个核心特征，用于让模型自动学习突破质量判断
    data = _add_breakout_quality_features(data, boundaries)

    return data


@register_feature("compute_bb_width_features", category="baseline")
def compute_bb_width_features(
    df: pd.DataFrame,
    *,
    period: int = 20,
    std_dev: int = 2,
    atr_window: int = 14,
) -> pd.DataFrame:
    """计算布林带宽度及其归一化特征。"""
    if "bb_upper" not in df.columns or "bb_lower" not in df.columns:
        upper, middle, lower = compute_bollinger_bands(
            df["close"], period=period, std_dev=std_dev
        )
        df["bb_upper"] = upper
        df["bb_middle"] = middle
        df["bb_lower"] = lower

    width = (df["bb_upper"] - df["bb_lower"]).abs()
    df["bb_width"] = width

    if "atr" not in df.columns:
        df["atr"] = compute_atr(df["high"], df["low"], df["close"], period=atr_window)

    df["bb_width_normalized"] = (
        (width / df["atr"].replace(0, np.nan))
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )
    return df


@register_feature("compute_range_ratio_5bar", category="baseline")
def compute_range_ratio_5bar(df: pd.DataFrame) -> pd.DataFrame:
    """计算 5/20 Bar 区间比率 z-score。"""
    if "hl" not in df.columns:
        df["hl"] = df["high"] - df["low"]

    short_range = df["hl"].rolling(5).mean()
    long_range = df["hl"].rolling(20).mean()
    ratio = (short_range / long_range.replace(0, np.nan)).replace(
        [np.inf, -np.inf], np.nan
    )
    ratio = ratio.fillna(1.0)
    ratio_log = np.log1p(ratio)
    mean = ratio_log.rolling(50, min_periods=5).mean()
    std = ratio_log.rolling(50, min_periods=5).std()
    df["range_ratio_5bar"] = (
        ((ratio_log - mean) / std.replace(0, np.nan))
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )
    return df


@register_feature("compute_volatility_reversal_score", category="baseline")
def compute_volatility_reversal_score(df: pd.DataFrame) -> pd.DataFrame:
    """ATR 回落 z-score，用于识别波动率反转。"""
    if "atr" not in df.columns:
        df["atr"] = compute_atr(df["high"], df["low"], df["close"])
    atr_mean = df["atr"].rolling(50).mean()
    atr_std = df["atr"].rolling(50).std()
    df["volatility_reversal_score"] = (
        (df["atr"] - atr_mean) / atr_std.replace(0, np.nan)
    ).fillna(0.0)
    return df


@register_feature("compute_price_range_symmetry", category="baseline")
def compute_price_range_symmetry(
    df: pd.DataFrame,
    *,
    feature_shift: int = 0,
) -> pd.DataFrame:
    """价格区间对称性（高/低/收盘关系），衡量上下影线不对称。"""
    high = df["high"].shift(feature_shift)
    low = df["low"].shift(feature_shift)
    close = df["close"].shift(feature_shift)

    numerator = high - close
    denominator = (close - low).replace(0, np.nan)
    raw = (numerator / denominator).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    log_val = np.log1p(np.abs(raw)) * np.sign(raw)
    mean = log_val.rolling(50, min_periods=5).mean()
    std = log_val.rolling(50, min_periods=5).std()
    df["price_range_symmetry"] = (
        ((log_val - mean) / std.replace(0, np.nan))
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )
    return df


@register_feature("compute_volume_anomaly", category="baseline")
def compute_volume_anomaly(df: pd.DataFrame) -> pd.DataFrame:
    """成交量异常 z-score。"""
    vol_ratio = df["volume"] / df["volume"].ewm(span=20, min_periods=10).mean()
    vol_ratio = vol_ratio.replace([np.inf, -np.inf], np.nan).fillna(1.0)
    vol_log = np.log1p(vol_ratio)
    mean = vol_log.rolling(50, min_periods=10).mean()
    std = vol_log.rolling(50, min_periods=10).std()
    df["volume_anomaly"] = (
        ((vol_log - mean) / std.replace(0, np.nan))
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )
    return df


@register_feature("compute_wick_ratios", category="baseline")
def compute_wick_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算上影线和下影线占比

    Args:
        df: DataFrame with high, low, open, close columns

    Returns:
        DataFrame with wick_upper_ratio and wick_lower_ratio columns added
    """
    if "wick_upper_ratio" in df.columns and "wick_lower_ratio" in df.columns:
        return df

    result = df.copy()
    range_val = result["high"] - result["low"]

    # 上影线 = max(high, close, open) - max(close, open)
    body_high = result[["close", "open"]].max(axis=1)
    upper_wick = result["high"] - body_high
    result["wick_upper_ratio"] = (upper_wick / range_val.replace(0, np.nan)).fillna(0.0)

    # 下影线 = min(close, open) - min(low, close, open)
    body_low = result[["close", "open"]].min(axis=1)
    lower_wick = body_low - result["low"]
    result["wick_lower_ratio"] = (lower_wick / range_val.replace(0, np.nan)).fillna(0.0)

    return result


@register_feature("compute_roc_5", category="baseline")
def compute_roc_5(df: pd.DataFrame) -> pd.DataFrame:
    """5 Bar ROC z-score。"""
    if "roc_5" in df.columns:
        return df
    roc_raw = df["close"].pct_change(5)
    roc_mean = roc_raw.rolling(window=50, min_periods=5).mean()
    roc_std = roc_raw.rolling(window=50, min_periods=5).std()
    roc_std = roc_std.clip(lower=roc_raw.abs().quantile(0.01))
    df["roc_5"] = (
        ((roc_raw - roc_mean) / roc_std.replace(0, np.nan))
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )
    return df


@register_feature("compute_acceleration_3", category="baseline")
def compute_acceleration_3(df: pd.DataFrame, feature_shift: int = 0) -> pd.DataFrame:
    """计算 acceleration_3 特征：ROC(3) 归一化后的差分（动量加速度）。"""
    if "acceleration_3" in df.columns:
        return df

    roc_3 = df["close"].pct_change(3)
    roc_3_mean = roc_3.rolling(window=50, min_periods=5).mean()
    roc_3_std = roc_3.rolling(window=50, min_periods=5).std()
    roc_3_std = roc_3_std.clip(lower=roc_3.abs().quantile(0.01))
    roc_3_norm = (
        ((roc_3 - roc_3_mean) / roc_3_std.replace(0, np.nan))
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )
    current = roc_3_norm.shift(feature_shift) if feature_shift > 0 else roc_3_norm
    prev = roc_3_norm.shift(feature_shift + 1)
    df["acceleration_3"] = current - prev
    return df


@register_feature("compute_trend_r2_20", category="baseline")
def compute_trend_r2_20(
    df: pd.DataFrame,
    *,
    feature_shift: int = 0,
) -> pd.DataFrame:
    """计算 20 Bar 趋势 R²。"""
    df["trend_r2_20"] = _trend_r2(df["close"], window=20, lag=feature_shift)
    return df


@register_feature("compute_trend_r2_50", category="baseline")
def compute_trend_r2_50(
    df: pd.DataFrame,
    *,
    feature_shift: int = 0,
) -> pd.DataFrame:
    """计算 50 Bar 趋势 R²。"""
    df["trend_r2_50"] = _trend_r2(df["close"], window=50, lag=feature_shift)
    return df


@register_feature("compute_slope_consistency_score", category="baseline")
def compute_slope_consistency_score(df: pd.DataFrame) -> pd.DataFrame:
    """多均线斜率一致性（EMA10/20/50）。"""
    ema10 = df["close"].ewm(span=10).mean()
    ema20 = df["close"].ewm(span=20).mean()
    ema50 = df["close"].ewm(span=50).mean()
    slope10 = np.sign(ema10.diff())
    slope20 = np.sign(ema20.diff())
    slope50 = np.sign(ema50.diff())
    df["slope_consistency_score"] = (
        (slope10 == slope20).astype(int)
        + (slope20 == slope50).astype(int)
        + (slope10 == slope50).astype(int)
    )
    return df


@register_feature("compute_atr_percentile", category="baseline")
def compute_atr_percentile(
    df: pd.DataFrame,
    *,
    window: int = 540,
    shift: int = 1,
) -> pd.DataFrame:
    """ATR 百分位（压缩检测）。

    warmup 不足（前 window-1 行）保持 NaN，禁止静默降级为 0.5。
    """
    if "atr" not in df.columns:
        df["atr"] = compute_atr(df["high"], df["low"], df["close"])
    if "atr_percentile" in df.columns:
        return df

    series = df["atr"].astype(float)

    def _percentile(arr: np.ndarray) -> float:
        if len(arr) == 0:
            return np.nan
        current = arr[-1]
        return float(np.mean(arr <= current))

    pct = series.rolling(window=window, min_periods=window).apply(_percentile, raw=True)
    if shift:
        pct = pct.shift(shift)
    df["atr_percentile"] = pct.clip(0.0, 1.0)
    return df


@register_feature("compute_trend_volatility_alignment", category="baseline")
def compute_trend_volatility_alignment(
    df: pd.DataFrame,
    *,
    feature_shift: int = 0,
    atr_percentile_window: int = 540,
) -> pd.DataFrame:
    """趋势方向与波动率状态的一致性。"""
    if "roc_5" not in df.columns:
        df = compute_roc_5(df)
    if "atr_percentile" not in df.columns:
        df = compute_atr_percentile(df, window=atr_percentile_window)
    df["trend_volatility_alignment"] = np.sign(df["roc_5"].shift(feature_shift)).fillna(
        0.0
    ) * df["atr_percentile"].fillna(0.0)
    return df


@register_feature("compute_compression_to_breakout_prob", category="baseline")
def compute_compression_to_breakout_prob(df: pd.DataFrame) -> pd.DataFrame:
    """压缩持续时间与未来动量的联动。"""
    if "compression_duration" not in df.columns or "roc_5" not in df.columns:
        return df
    df["compression_to_breakout_prob"] = df["compression_duration"].fillna(0.0) * df[
        "roc_5"
    ].fillna(0.0)
    return df


def _compute_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """计算 ATR（内部方法，使用类的静态方法）"""
    return compute_atr(df["high"], df["low"], df["close"], period=window)


# ========================================================================
# 特征添加静态方法
# ========================================================================


@register_feature("add_basic_indicators", category="baseline")
def add_basic_indicators(
    df: pd.DataFrame, required_features: Optional[set] = None
) -> pd.DataFrame:
    """
    添加基础技术指标到DataFrame（优化版：支持按需计算）

    Args:
        df: 包含OHLCV数据的DataFrame
        required_features: 需要计算的指标集合，None 表示计算所有
    """
    if df.empty:
        return df

    result = df.copy()

    # 确保所有列都是数值类型
    for col in ["open", "high", "low", "close", "volume"]:
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors="coerce")

    result = result.dropna(subset=["open", "high", "low", "close", "volume"])
    if result.empty:
        return result

    # 按需计算 RSI
    if required_features is None or "rsi" in required_features:
        if "rsi" not in result.columns:
            result["rsi"] = compute_rsi(result["close"])

    # 按需计算 MACD
    need_macd = required_features is None or any(
        f in required_features for f in ["macd", "macd_signal", "macd_histogram"]
    )
    if need_macd and "macd" not in result.columns:
        try:
            macd_line, signal_line, histogram = compute_macd(result["close"])
            result["macd"] = macd_line
            result["macd_signal"] = signal_line
            result["macd_histogram"] = histogram
        except Exception as e:
            print(f"Warning: Error computing MACD: {e}")
            result["macd"] = 0
            result["macd_signal"] = 0
            result["macd_histogram"] = 0

    # 按需计算 Bollinger Bands
    need_bb = required_features is None or any(
        f in required_features for f in ["bb_upper", "bb_middle", "bb_lower"]
    )
    if need_bb and "bb_upper" not in result.columns:
        try:
            upper_band, middle_band, lower_band = compute_bollinger_bands(
                result["close"]
            )
            result["bb_upper"] = upper_band
            result["bb_middle"] = middle_band
            result["bb_lower"] = lower_band
        except Exception as e:
            print(f"Warning: Error computing Bollinger Bands: {e}")
            result["bb_upper"] = result["close"]
            result["bb_middle"] = result["close"]
            result["bb_lower"] = result["close"]

    # 按需计算 ATR
    if required_features is None or "atr" in required_features:
        if "atr" not in result.columns:
            try:
                result["atr"] = compute_atr(
                    result["high"], result["low"], result["close"]
                )
            except Exception as e:
                print(f"Warning: Error computing ATR: {e}")
                result["atr"] = 0

    # 按需计算 ZigZag
    # 注意：这里只计算 zigzag，不计算高点和低点（高点和低点在 add_zigzag_dimensionless_features 中计算）
    if required_features is None or "zigzag" in required_features:
        if "zigzag" not in result.columns:
            try:
                result["zigzag"] = compute_zigzag(
                    result["high"], result["low"], return_high_low=False
                )
            except Exception as e:
                print(f"Warning: Error computing ZigZag: {e}")
                result["zigzag"] = 0

    # 按需计算价格变化和波动率
    if required_features is None or "price_change" in required_features:
        if "price_change" not in result.columns:
            result["price_change"] = result["close"].pct_change()

    if required_features is None or "volatility" in required_features:
        if "volatility" not in result.columns:
            price_change_numeric = pd.to_numeric(
                result["price_change"], errors="coerce"
            ).astype(float)
            values = _talib().STDDEV(price_change_numeric.values, timeperiod=14, nbdev=1)
            # 使用 shift(1) 确保时间对齐，避免使用未来信息
            result["volatility"] = pd.Series(
                values, index=price_change_numeric.index
            ).shift(1)

    # 按需计算成交量特征
    if (
        required_features is None
        or "volume_sma" in required_features
        or "volume_ratio" in required_features
    ):
        if "volume_sma" not in result.columns:
            volume_numeric = pd.to_numeric(result["volume"], errors="coerce").astype(
                float
            )
            values = _talib().SMA(volume_numeric.values, timeperiod=20)
            result["volume_sma"] = pd.Series(values, index=volume_numeric.index)
        if "volume_ratio" not in result.columns:
            result["volume_ratio"] = result["volume"] / result["volume_sma"].replace(
                0, np.nan
            )

    return result


@register_feature("ensure_basic_indicators", category="baseline")
def ensure_basic_indicators(
    df: pd.DataFrame, required_features: Optional[set] = None
) -> pd.DataFrame:
    """确保基础指标存在（优化版：支持按需计算）"""
    if df.empty:
        return df

    # 检查需要的指标是否都已存在
    if required_features:
        missing = required_features - set(df.columns)
        if not missing:
            return df

    return add_basic_indicators(df, required_features)


@register_feature("add_zigzag_dimensionless_features", category="baseline")
def add_zigzag_dimensionless_features(
    df: pd.DataFrame, required_features: Optional[set] = None
) -> pd.DataFrame:
    """
    添加 ZigZag 相关的无量纲特征

    新增特征：
    - price_to_zz_high_pct: 当前价格到最近 ZigZag 高点的相对距离
    - price_to_zz_low_pct: 当前价格到最近 ZigZag 低点的相对距离
    - zz_amplitude_pct: ZigZag 波幅（相对）
    - zz_duration: ZigZag 持续时间（bar 数，无量纲）
    - zz_slope: ZigZag 斜率（归一化）
    """
    if df.empty:
        return df

    result = df.copy()

    # 确保 zigzag 存在
    if "zigzag" not in result.columns:
        if required_features and any(
            "zz_" in f or "zigzag" in f for f in required_features
        ):
            result = ensure_basic_indicators(result, {"zigzag"})
        else:
            return result

    # 确保 atr 存在（zz_slope 需要 atr）
    if "atr" not in result.columns:
        if required_features and "zz_slope" in required_features:
            result = ensure_basic_indicators(result, {"atr"})
        elif required_features is None:
            # 如果没有指定 required_features，也确保 atr 存在
            result = ensure_basic_indicators(result, {"atr"})

    close = result["close"].replace(0, np.nan)

    # 确定使用的价格序列（优先使用 WPT 中高频重构价格）
    price_series = None
    if "wpt_price_reconstructed" in result.columns:
        # 自动检测 WPT 重构价格（中高频，保留关键拐点）
        price_series = result["wpt_price_reconstructed"]

    # 优化：直接计算 zigzag + 高点和低点（一次性完成）
    # 如果 zigzag 已存在，重新计算以确保高点和低点正确（性能影响可忽略）
    zigzag, zz_high, zz_low = compute_zigzag(
        result["high"], result["low"], return_high_low=True, price_col=price_series
    )
    result["zigzag"] = zigzag
    result["zz_high_value"] = zz_high
    result["zz_low_value"] = zz_low

    # 1. 当前价格距离最近 ZigZag 高/低点的相对距离
    if required_features is None or "price_to_zz_high_pct" in required_features:
        if "price_to_zz_high_pct" not in result.columns:
            result["price_to_zz_high_pct"] = ((zz_high - close) / close).replace(
                [np.inf, -np.inf], np.nan
            )

    if required_features is None or "price_to_zz_low_pct" in required_features:
        if "price_to_zz_low_pct" not in result.columns:
            result["price_to_zz_low_pct"] = ((close - zz_low) / close).replace(
                [np.inf, -np.inf], np.nan
            )

    # 2. ZigZag 波幅（相对）
    if required_features is None or "zz_amplitude_pct" in required_features:
        if "zz_amplitude_pct" not in result.columns:
            zz_low_safe = zz_low.replace(0, np.nan)
            result["zz_amplitude_pct"] = ((zz_high - zz_low) / zz_low_safe).replace(
                [np.inf, -np.inf], np.nan
            )

    # 3. ZigZag 持续时间（从上一个转折点至今的 bar 数）
    if required_features is None or "zz_duration" in required_features:
        if "zz_duration" not in result.columns:
            zigzag_diff = zigzag.diff()
            turn_points = (zigzag_diff * zigzag_diff.shift(1) < 0) | (
                (zigzag_diff != 0) & (zigzag_diff.shift(1) == 0)
            )

            duration = pd.Series(index=zigzag.index, dtype=float)
            last_turn_idx = 0
            for i in range(len(zigzag)):
                if turn_points.iloc[i]:
                    last_turn_idx = i
                duration.iloc[i] = i - last_turn_idx
            result["zz_duration"] = duration.fillna(0.0)

    # 4. ZigZag 斜率（归一化）
    if required_features is None or "zz_slope" in required_features:
        if "zz_slope" not in result.columns:
            if "atr" not in result.columns:
                raise ValueError(
                    "ATR is required for computing zz_slope. "
                    "Please ensure 'atr' is computed before calling this function."
                )

            window = 5
            zz_slope_raw = zigzag.diff(window) / window

            # 归一化：使用 ATR
            atr_safe = result["atr"].replace(0, np.nan)
            result["zz_slope"] = (zz_slope_raw / atr_safe).replace(
                [np.inf, -np.inf], np.nan
            )

    return result


@register_feature("add_poc_hal_dimensionless_features", category="baseline")
def add_poc_hal_dimensionless_features(
    df: pd.DataFrame,
    required_features: Optional[set] = None,
    poc_window: int = 160,
    price_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    添加 POC (Point of Control) 和 HAL (Value Area 70% 价格区间的上下界) 相关的无量纲特征

    ✅ 强烈建议：使用 WPT 低频重构价格（price_col='wpt_price_reconstructed'）
    这样可以过滤高频噪声，使 POC/HAL 更接近真实供需平衡点。

    注意：POC 和 HAL 的计算合并在一起，因为它们都基于相同的 volume profile 计算，
    避免重复计算浪费性能。

    POC 相关特征：
    - price_to_poc_pct: 当前价格到 POC 的相对距离
    - poc_position_ratio: POC 在价格区间中的位置（0-1）
    - poc_volume_ratio: POC 位置的成交量占比

    HAL 相关特征：
    - price_to_hal_high_pct: 当前价格到 HAL 高点的相对距离
    - price_to_hal_low_pct: 当前价格到 HAL 低点的相对距离
    - price_to_hal_mid_pct: 当前价格到 HAL 中点的相对距离
    - hal_bandwidth_pct: HAL 带宽（相对）

    Args:
        df: 输入 DataFrame
        required_features: 需要的特征集合（可选）
        poc_window: POC 计算窗口大小
        price_col: 可选的价格列名（如 'wpt_price_reconstructed'）。如果提供，将使用此列
                   而非原始 high/low。默认 None，使用原始价格（向后兼容）
    """
    if df.empty:
        return df

    result = df.copy()

    # 检查是否需要计算 POC 或 HAL
    need_poc = required_features is None or any("poc" in f for f in required_features)
    need_hal = required_features is None or any("hal" in f for f in required_features)

    if not (need_poc or need_hal):
        return result

    # 确定使用的价格序列
    # 优先使用 WPT 低频重构价格，如果不存在则使用原始价格
    price_series = None
    if price_col and price_col in result.columns:
        price_series = result[price_col]
    elif "wpt_price_reconstructed" in result.columns:
        # 自动检测 WPT 重构价格
        price_series = result["wpt_price_reconstructed"]

    # 计算 POC 和 HAL（一次性计算，避免重复）
    # 优化：如果 poc 列已存在且完整（非 NaN 值 > 50%），跳过计算
    need_compute = False
    if need_poc:
        poc_exists = "poc" in result.columns
        if poc_exists:
            poc_non_na_ratio = (
                result["poc"].notna().sum() / len(result) if len(result) > 0 else 0
            )
            if poc_non_na_ratio > 0.5:
                # poc 列已存在且完整，跳过计算
                need_poc = False
        if need_poc and (
            "poc" not in result.columns or "poc_volume_ratio" not in result.columns
        ):
            need_compute = True
    if need_hal and (
        "hal_high" not in result.columns or "hal_low" not in result.columns
    ):
        need_compute = True

    if need_compute:
        # 使用统一的 Volume Profile 实现
        from .utils_volume_profile import (
            compute_unified_volume_profile_features,
            compute_unified_volume_profile_derived_features,
        )

        result = compute_unified_volume_profile_features(
            result,
            window=poc_window,
            price_series=price_series,
        )

        # 映射到旧的特征名称（向后兼容）
        if "vp_poc" in result.columns:
            result["poc"] = result["vp_poc"]
        if "vp_poc_volume_ratio" in result.columns:
            result["poc_volume_ratio"] = result["vp_poc_volume_ratio"]
        if "vp_hal_high" in result.columns:
            result["hal_high"] = result["vp_hal_high"]
        if "vp_hal_low" in result.columns:
            result["hal_low"] = result["vp_hal_low"]
        if "vp_hal_mid" in result.columns:
            result["hal_mid"] = result["vp_hal_mid"]

    close = result["close"].replace(0, np.nan)
    poc = result["poc"]
    hal_high = result["hal_high"]
    hal_low = result["hal_low"]
    hal_mid = result["hal_mid"]
    high = result["high"]
    low = result["low"]

    # ========== POC 相关特征 ==========
    # 1. 当前价格到 POC 的相对距离
    if required_features is None or "price_to_poc_pct" in required_features:
        if "price_to_poc_pct" not in result.columns:
            result["price_to_poc_pct"] = ((poc - close) / close).replace(
                [np.inf, -np.inf], np.nan
            )

    # 2. POC 在价格区间中的位置（0-1）
    if required_features is None or "poc_position_ratio" in required_features:
        if "poc_position_ratio" not in result.columns:
            price_range = (high - low).replace(0, np.nan)
            result["poc_position_ratio"] = (
                ((poc - low) / price_range)
                .replace([np.inf, -np.inf], np.nan)
                .clip(0.0, 1.0)
            )

    # ========== HAL 相关特征 ==========
    # 1. 当前价格到 HAL 的相对距离
    if required_features is None or "price_to_hal_high_pct" in required_features:
        if "price_to_hal_high_pct" not in result.columns:
            result["price_to_hal_high_pct"] = ((hal_high - close) / close).replace(
                [np.inf, -np.inf], np.nan
            )

    if required_features is None or "price_to_hal_low_pct" in required_features:
        if "price_to_hal_low_pct" not in result.columns:
            result["price_to_hal_low_pct"] = ((close - hal_low) / close).replace(
                [np.inf, -np.inf], np.nan
            )

    if required_features is None or "price_to_hal_mid_pct" in required_features:
        if "price_to_hal_mid_pct" not in result.columns:
            # 使用 shift(1) 确保时间对齐，避免使用未来信息
            result["price_to_hal_mid_pct"] = (
                ((hal_mid - close) / close).replace([np.inf, -np.inf], np.nan).shift(1)
            )

    # 2. HAL 带宽（相对）
    if required_features is None or "hal_bandwidth_pct" in required_features:
        if "hal_bandwidth_pct" not in result.columns:
            hal_mid_safe = hal_mid.replace(0, np.nan)
            result["hal_bandwidth_pct"] = ((hal_high - hal_low) / hal_mid_safe).replace(
                [np.inf, -np.inf], np.nan
            )

    return result


@register_feature("add_swing_dimensionless_features", category="baseline")
def add_swing_dimensionless_features(
    df: pd.DataFrame,
    required_features: Optional[set] = None,
    swing_win_short: int = 20,
    swing_win_long: int = 60,
    price_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    添加 Swing High/Low 相关的无量纲特征

    ✅ 建议：使用 WPT 中频重构价格（price_col='wpt_price_reconstructed'）
    这样可以捕捉中期结构，同时过滤高频噪声。

    新增特征：
    - swing_high_pct_close: Swing High 相对收盘价的比率
    - swing_low_pct_close: Swing Low 相对收盘价的比率
    - swing_amplitude_pct: Swing 波幅（相对）

    Args:
        df: 输入 DataFrame
        required_features: 需要的特征集合（可选）
        swing_win_short: 短期 Swing 窗口大小
        swing_win_long: 长期 Swing 窗口大小
        price_col: 可选的价格列名（如 'wpt_price_reconstructed'）。如果提供，将使用此列
                   而非原始 high/low。默认 None，使用原始价格（向后兼容）
    """
    if df.empty:
        return df

    result = df.copy()

    close = result["close"].replace(0, np.nan)

    # 确定使用的价格序列（优先使用 WPT 中频重构价格）
    swing_price = None
    if price_col and price_col in result.columns:
        swing_price = result[price_col]
    elif "wpt_price_reconstructed" in result.columns:
        # 自动检测 WPT 重构价格（中频，捕捉中期结构）
        swing_price = result["wpt_price_reconstructed"]

    # 计算 Swing High/Low（如果不存在）
    if "roll_high_s" not in result.columns:
        if required_features and any("swing" in f for f in required_features):
            if swing_price is not None:
                # 使用 WPT 重构价格
                result["roll_high_s"] = swing_price.rolling(
                    swing_win_short, min_periods=1
                ).max()
                result["roll_low_s"] = swing_price.rolling(
                    swing_win_short, min_periods=1
                ).min()
                result["roll_high_l"] = swing_price.rolling(
                    swing_win_long, min_periods=1
                ).max()
                result["roll_low_l"] = swing_price.rolling(
                    swing_win_long, min_periods=1
                ).min()
            else:
                # 使用原始价格
                result["roll_high_s"] = (
                    result["high"].rolling(swing_win_short, min_periods=1).max()
                )
                result["roll_low_s"] = (
                    result["low"].rolling(swing_win_short, min_periods=1).min()
                )
                result["roll_high_l"] = (
                    result["high"].rolling(swing_win_long, min_periods=1).max()
                )
                result["roll_low_l"] = (
                    result["low"].rolling(swing_win_long, min_periods=1).min()
                )
        else:
            return result

    # 1. Swing High/Low 相对收盘价的比率
    if required_features is None or "swing_high_pct_close" in required_features:
        if "swing_high_pct_close" not in result.columns:
            result["swing_high_pct_close"] = (
                (result["roll_high_s"] - close) / close.replace(0, np.nan)
            ).replace([np.inf, -np.inf], np.nan)

    if required_features is None or "swing_low_pct_close" in required_features:
        if "swing_low_pct_close" not in result.columns:
            result["swing_low_pct_close"] = (
                (close - result["roll_low_s"]) / close.replace(0, np.nan)
            ).replace([np.inf, -np.inf], np.nan)

    # 2. Swing 波幅（相对）
    if required_features is None or "swing_amplitude_pct" in required_features:
        if "swing_amplitude_pct" not in result.columns:
            roll_low_s_safe = result["roll_low_s"].replace(0, np.nan)
            result["swing_amplitude_pct"] = (
                (result["roll_high_s"] - result["roll_low_s"]) / roll_low_s_safe
            ).replace([np.inf, -np.inf], np.nan)

    return result


@register_feature("add_ols_channel_features", category="baseline")
def add_ols_channel_features(
    df: pd.DataFrame,
    required_features: Optional[set] = None,
    window: int = 96,
) -> pd.DataFrame:
    """
    添加 OLS 通道特征（线性回归通道）

    生成：
    - ols_channel_mid：OLS 拟合的中心线
    - ols_channel_upper / lower：中心线 ± 残差标准差
    - ols_channel_width：通道宽度
    """
    if df.empty or "close" not in df.columns:
        return df

    need_channel = required_features is None or any(
        key in (required_features or set())
        for key in {
            "ols_channel_mid",
            "ols_channel_upper",
            "ols_channel_lower",
            "ols_channel_width",
        }
    )
    if not need_channel:
        return df

    result = df.copy()
    close = result["close"].astype(float)
    mid = pd.Series(np.nan, index=result.index, dtype=float)
    upper = pd.Series(np.nan, index=result.index, dtype=float)
    lower = pd.Series(np.nan, index=result.index, dtype=float)
    width = pd.Series(np.nan, index=result.index, dtype=float)

    x = np.arange(window)
    for i in range(window, len(result)):
        window_slice = close.iloc[i - window : i]
        if window_slice.isna().any():
            continue
        try:
            slope, intercept = np.polyfit(x, window_slice.values, 1)
            fitted = slope * x + intercept
            mid_val = slope * (window - 1) + intercept
            resid = window_slice.values - fitted
            resid_std = np.std(resid)
            mid.iloc[i] = mid_val
            upper.iloc[i] = mid_val + resid_std
            lower.iloc[i] = mid_val - resid_std
            width.iloc[i] = 2 * resid_std
        except Exception:
            continue

    result["ols_channel_mid"] = mid.ffill()
    result["ols_channel_upper"] = upper.ffill()
    result["ols_channel_lower"] = lower.ffill()
    result["ols_channel_width"] = width.ffill()

    return result


@register_feature("add_price_volume_relative_features", category="baseline")
def add_price_volume_relative_features(
    df: pd.DataFrame, required_features: Optional[set] = None
) -> pd.DataFrame:
    """
    添加基础价格与量能相对变化特征

    新增特征：
    - ret_1h, ret_4h, ret_24h: 对数收益率（1小时、4小时、24小时）
    - rv_4h, rv_24h: 已实现波动率
    - vol_ma_ratio: 成交量移动平均比率
    - vol_zscore: 成交量 Z-score
    """
    if df.empty:
        return df

    result = df.copy()
    close = result["close"].replace(0, np.nan)
    volume = result["volume"]

    # 1. 对数收益率（常用）
    # 注意：这里假设数据是 5 分钟 K 线，1h=12根，4h=48根，24h=288根
    # 实际应该根据时间框架动态计算
    periods_1h = 12  # 假设 5 分钟 K 线
    periods_4h = 48
    periods_24h = 288

    if required_features is None or "ret_1h" in required_features:
        if "ret_1h" not in result.columns:
            # 使用 shift(1) 确保时间对齐，避免使用未来信息
            result["ret_1h"] = np.log(close / close.shift(periods_1h)).shift(1)

    if required_features is None or "ret_4h" in required_features:
        if "ret_4h" not in result.columns:
            # 使用 shift(1) 确保时间对齐，避免使用未来信息
            result["ret_4h"] = np.log(close / close.shift(periods_4h)).shift(1)

    if required_features is None or "ret_24h" in required_features:
        if "ret_24h" not in result.columns:
            # 使用 shift(1) 确保时间对齐，避免使用未来信息
            result["ret_24h"] = np.log(close / close.shift(periods_24h)).shift(1)

    # 2. 已实现波动率（基于 ret_1h）
    if required_features is None or "rv_4h" in required_features:
        if "rv_4h" not in result.columns and "ret_1h" in result.columns:
            result["rv_4h"] = (
                result["ret_1h"]
                .rolling(window=periods_4h // periods_1h, min_periods=1)
                .std()
            )

    if required_features is None or "rv_24h" in required_features:
        if "rv_24h" not in result.columns and "ret_1h" in result.columns:
            result["rv_24h"] = (
                result["ret_1h"]
                .rolling(window=periods_24h // periods_1h, min_periods=1)
                .std()
            )

    # 3. 成交量异常度
    if required_features is None or "vol_ma_ratio" in required_features:
        if "vol_ma_ratio" not in result.columns:
            vol_ma = volume.rolling(window=periods_24h, min_periods=periods_24h).mean()
            vol_ma_ratio = (
                (volume / vol_ma.replace(0, np.nan))
                .replace([np.inf, -np.inf], np.nan)
                .fillna(1.0)
            )
            # 滚动统计衍生特征统一 shift(1)，确保在 t 时刻仅使用 t-1 及之前的数据
            result["vol_ma_ratio"] = vol_ma_ratio.shift(1)

    if required_features is None or "vol_zscore" in required_features:
        if "vol_zscore" not in result.columns:
            result["vol_zscore"] = _rolling_zscore(
                volume.astype(float),
                window=periods_24h,
                min_periods=periods_24h,
            )

    return result


@register_feature("add_common_derived_features", category="baseline")
def add_common_derived_features(
    df: pd.DataFrame,
    required_features: Optional[set] = None,
    rolling_zscore_windows: List[int] = None,
) -> pd.DataFrame:
    """
    添加常用衍生特征（优化版：支持按需计算，不强制计算所有基础指标）
    """
    if df.empty:
        return df

    result = df.copy()
    close = result["close"].replace(0, np.nan)

    # 解析依赖关系：确定需要哪些基础指标
    needed_basic = set()
    if required_features:
        # 分析需要哪些基础指标
        if any("rsi" in f for f in required_features):
            needed_basic.add("rsi")
        if any("macd" in f for f in required_features):
            needed_basic.update(["macd", "macd_signal", "macd_histogram"])
        if any("bb_" in f for f in required_features):
            needed_basic.update(["bb_upper", "bb_lower", "bb_middle"])
        if any("atr" in f for f in required_features):
            needed_basic.add("atr")
    else:
        # 如果没有指定，只确保必要的基础指标
        needed_basic = {"rsi", "atr"}  # 最小集合

    # 按需计算基础指标
    if needed_basic:
        result = ensure_basic_indicators(result, needed_basic)

    # 只在需要时计算特征
    if not required_features or "returns" in required_features:
        if "returns" not in result.columns:
            result["returns"] = close.pct_change()

    if not required_features or "log_returns" in required_features:
        if "log_returns" not in result.columns:
            shifted = close.shift(1).replace(0, np.nan)
            result["log_returns"] = np.log(close / shifted)

    if not required_features or "price_change" in required_features:
        if "price_change" not in result.columns:
            result["price_change"] = close.diff()

    if not required_features or "volatility" in required_features:
        if "volatility" not in result.columns:
            if "returns" in result.columns:
                returns_numeric = pd.to_numeric(
                    result["returns"], errors="coerce"
                ).astype(float)
                values = _talib().STDDEV(returns_numeric.values, timeperiod=20, nbdev=1)
                result["volatility"] = pd.Series(values, index=returns_numeric.index)
            else:
                price_change = close.pct_change()
                price_change_numeric = pd.to_numeric(
                    price_change, errors="coerce"
                ).astype(float)
                values = _talib().STDDEV(
                    price_change_numeric.values, timeperiod=20, nbdev=1
                )
                result["volatility"] = pd.Series(
                    values, index=price_change_numeric.index
                )

    # BB 相关特征
    if {"bb_upper", "bb_lower"}.issubset(result.columns):
        if not required_features or "bb_position" in required_features:
            if "bb_position" not in result.columns:
                denom = (result["bb_upper"] - result["bb_lower"]).replace(0, np.nan)
                result["bb_position"] = (
                    ((close - result["bb_lower"]) / denom)
                    .replace([np.inf, -np.inf], np.nan)
                    .fillna(0.5)
                )

        if not required_features or "bb_width" in required_features:
            if "bb_width" not in result.columns:
                # bb_width 归一化：除以 bb_middle（布林带中线）或 close
                # 这样消除量纲，使其成为相对值
                bb_width_raw = (result["bb_upper"] - result["bb_lower"]).abs()
                if "bb_middle" in result.columns:
                    bb_middle_safe = result["bb_middle"].replace(0, np.nan)
                    result["bb_width"] = (bb_width_raw / bb_middle_safe).replace(
                        [np.inf, -np.inf], np.nan
                    )
                else:
                    # 如果没有 bb_middle，使用 close
                    close_safe = close.replace(0, np.nan)
                    result["bb_width"] = (bb_width_raw / close_safe).replace(
                        [np.inf, -np.inf], np.nan
                    )

    # 归一化特征（价格归一化，保留用于向后兼容）
    # 注意：RSI 本身就是 0~100 的标准范围，不需要归一化
    if not required_features or "macd_normalized" in required_features:
        if "macd_normalized" not in result.columns and "macd" in result.columns:
            result["macd_normalized"] = (
                result["macd"] / close.replace(0, np.nan)
            ).replace([np.inf, -np.inf], np.nan)

    if not required_features or "atr_normalized" in required_features:
        if "atr_normalized" not in result.columns and "atr" in result.columns:
            result["atr_normalized"] = (
                result["atr"] / close.replace(0, np.nan)
            ).replace([np.inf, -np.inf], np.nan)

    # ========== 滚动 Z-score 特征（推荐：比价格归一化更优）==========
    # 使用多个滚动窗口标准化（Feature Stacking），让模型学习不同时间尺度的信息
    # 默认窗口：[50 (短期), 288 (24h), 500 (长期)]
    # 对于 5 分钟 K 线：50≈4h, 288=24h, 500≈2天
    if rolling_zscore_windows is None:
        rolling_zscore_windows = [50, 288, 500]

    # Helper function to generate z-score features for multiple windows
    def add_multi_window_zscore(
        base_col: str,
        feature_prefix: str,
        windows: List[int],
        required_features: Optional[set],
    ) -> None:
        """为某个基础指标生成多个窗口的 z-score 特征"""
        if base_col not in result.columns:
            return

        for window in windows:
            zscore_col = f"{feature_prefix}_zscore_w{window}"
            # Check if this specific feature is required
            if required_features is not None:
                # Check if any zscore variant is requested or this specific one
                if not any(f"{feature_prefix}_zscore" in f for f in required_features):
                    continue
                if zscore_col not in required_features and not any(
                    f.startswith(f"{feature_prefix}_zscore")
                    and f.endswith(f"_w{window}")
                    for f in required_features
                ):
                    # If specific windows are requested, only generate those
                    if any(
                        f"{feature_prefix}_zscore_w" in f for f in required_features
                    ):
                        continue

            if zscore_col not in result.columns:
                # 【关键修复】：强制 min_periods=window，确保满窗才输出，减少早期噪声
                result[zscore_col] = _rolling_zscore(
                    result[base_col], window=window, min_periods=window
                )

    # 1. RSI 滚动 Z-score（虽然 RSI 本身是 0-100，但 Z-score 能突出极端值）
    # 生成多个窗口：rsi_zscore_w50, rsi_zscore_w288, rsi_zscore_w500
    if required_features is None or any(
        "rsi_zscore" in f for f in required_features or [""]
    ):
        add_multi_window_zscore("rsi", "rsi", rolling_zscore_windows, required_features)

    # 2. MACD 滚动 Z-score（MACD 绝对值随价格变化，Z-score 标准化更优）
    if required_features is None or any(
        "macd_zscore" in f for f in required_features or [""]
    ):
        add_multi_window_zscore(
            "macd", "macd", rolling_zscore_windows, required_features
        )

    # 3. MACD Histogram 滚动 Z-score（波动更大，Z-score 能突出极端动量变化）
    if required_features is None or any(
        "macd_histogram_zscore" in f for f in required_features or [""]
    ):
        add_multi_window_zscore(
            "macd_histogram",
            "macd_histogram",
            rolling_zscore_windows,
            required_features,
        )

    # 4. Momentum 滚动 Z-score（不同资产的 ROC 量级差异巨大，Z-score 必须）
    for period in [5, 10, 20]:
        momentum_col = f"momentum_{period}"
        if required_features is None or any(
            f"momentum_{period}_zscore" in f for f in required_features or [""]
        ):
            add_multi_window_zscore(
                momentum_col,
                f"momentum_{period}",
                rolling_zscore_windows,
                required_features,
            )

    # 5. ATR 滚动 Z-score（ATR 绝对值与价格成正比，Z-score 可判断波动率异常）
    if required_features is None or any(
        "atr_zscore" in f for f in required_features or [""]
    ):
        add_multi_window_zscore("atr", "atr", rolling_zscore_windows, required_features)

    # 6. Volume 滚动 Z-score（交易量绝对值与流动性相关，Z-score 捕捉相对变化）
    if required_features is None or any(
        "volume_zscore" in f for f in required_features or [""]
    ):
        add_multi_window_zscore(
            "volume", "volume", rolling_zscore_windows, required_features
        )

    # 7. BB Width 滚动 Z-score（布林带宽度反映波动性，Z-score 标准化）
    if required_features is None or any(
        "bb_width_zscore" in f for f in required_features or [""]
    ):
        add_multi_window_zscore(
            "bb_width", "bb_width", rolling_zscore_windows, required_features
        )

    # 8. Volatility 滚动 Z-score（波动率指标的 Z-score）
    if required_features is None or any(
        "volatility_zscore" in f for f in required_features or [""]
    ):
        add_multi_window_zscore(
            "volatility", "volatility", rolling_zscore_windows, required_features
        )

    # Momentum features
    for period in [5, 10, 20]:
        momentum_col = f"momentum_{period}"
        if not required_features or momentum_col in required_features:
            if momentum_col not in result.columns:
                result[momentum_col] = close.pct_change(period)

    # SMA features
    sma_map = {5: "sma_5", 10: "sma_10", 20: "sma_20"}
    for window, col_name in sma_map.items():
        if not required_features or col_name in required_features:
            if col_name not in result.columns:
                close_numeric = pd.to_numeric(close, errors="coerce").astype(float)
                values = _talib().SMA(close_numeric.values, timeperiod=window)
                result[col_name] = pd.Series(values, index=close_numeric.index).fillna(
                    close
                )

    # SMA/EMA 相对 close 的百分比
    close_safe = close.replace(0, np.nan)
    for col_name in [
        "sma_5",
        "sma_10",
        "sma_20",
        "ema_5",
        "ema_10",
        "ema_20",
        "ema_50",
        "wma_20",
    ]:
        pct_col = f"{col_name}_pct_close"
        if not required_features or pct_col in required_features:
            if col_name in result.columns and pct_col not in result.columns:
                result[pct_col] = ((result[col_name] / close_safe - 1.0)).replace(
                    [np.inf, -np.inf], np.nan
                )

    # SMA ratios
    if not required_features or "sma_ratio_5_20" in required_features:
        if {"sma_5", "sma_20"}.issubset(
            result.columns
        ) and "sma_ratio_5_20" not in result.columns:
            # 使用 shift(1) 确保时间对齐，避免使用未来信息
            result["sma_ratio_5_20"] = (
                (result["sma_5"] / result["sma_20"].replace(0, np.nan))
                .replace([np.inf, -np.inf], np.nan)
                .fillna(1.0)
                .shift(1)
            )

    if not required_features or "sma_ratio_10_20" in required_features:
        if {"sma_10", "sma_20"}.issubset(
            result.columns
        ) and "sma_ratio_10_20" not in result.columns:
            # 使用 shift(1) 确保时间对齐，避免使用未来信息
            result["sma_ratio_10_20"] = (
                (result["sma_10"] / result["sma_20"].replace(0, np.nan))
                .replace([np.inf, -np.inf], np.nan)
                .fillna(1.0)
                .shift(1)
            )

    # Volume features
    if not required_features or "volume_sma_20" in required_features:
        if "volume_sma_20" not in result.columns:
            volume_numeric = pd.to_numeric(result["volume"], errors="coerce").astype(
                float
            )
            values = _talib().SMA(volume_numeric.values, timeperiod=20)
            result["volume_sma_20"] = pd.Series(
                values, index=volume_numeric.index
            ).fillna(result["volume"])

    if not required_features or "volume_ratio" in required_features:
        if "volume_ratio" not in result.columns:
            if "volume_sma_20" in result.columns:
                denom = result["volume_sma_20"].replace(0, np.nan)
                # 使用 shift(1) 确保时间对齐，避免使用未来信息
                result["volume_ratio"] = (
                    (result["volume"] / denom)
                    .replace([np.inf, -np.inf], np.nan)
                    .fillna(1.0)
                    .shift(1)
                )

    # Final cleanup: 处理所有数值列的 inf，保留 NaN 到预处理阶段
    numeric_cols = result.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col in result.columns:
            result[col] = result[col].replace([np.inf, -np.inf], np.nan)

    return result


def _rolling_percentile(
    series: pd.Series, window: int, min_periods: int = None, shift: bool = True
) -> pd.Series:
    """
    安全滚动百分位排名（严格因果，无未来信息）

    【核心原则：滚动窗口统计特征需要 shift(1)】
    - 虽然我们在 t 时刻可以使用 close[t]，但对于"滚动窗口统计特征"仍需 shift(1)
    - 原因：避免将当前值包含在历史分布中，即使我们已经排除了当前值
    - 更严格的做法：在 t 时刻使用的特征基于 t-1 及之前的数据计算

    【关键区分】
    - 基础滚动指标（sma, ema, atr, bb_width）：不需要 shift(1)
    - 滚动窗口统计特征（zscore, percentile, entropy）：需要 shift(1)

    Args:
        series: 输入序列
        window: 滚动窗口长度
        min_periods: 最小观测数才开始输出（默认=window，最稳健）
        shift: 是否 shift(1) 以确保完全因果（默认=True，推荐）

    Returns:
        滚动百分位排名序列（0~1，早期不足窗口处为 NaN）
    """
    if min_periods is None:
        min_periods = window  # 默认：必须满窗才输出，最稳健

    min_periods = min(min_periods, window)

    def _percentile(x: np.ndarray) -> float:
        """
        计算当前值在历史窗口中的百分位排名（严格因果，无自我参照偏差）

        【实现说明】
        - current = x[-1]：当前值（如 close[t]），作为"新来的考生"
        - history = x[:-1]：历史窗口（如 [t-N, t-1]），作为"老考生的成绩分数线"
        - percentile = (history <= current).sum() / len(history)
          表示：当前值在历史中的相对位置，完全基于历史评估当前状态
        """
        if len(x) < 2 or not np.isfinite(x[-1]):
            return np.nan
        current = x[-1]  # 当前值（如 close[t]），作为"新来的考生"
        history = x[:-1]  # ← 关键：只用历史（如 [t-N, t-1]），作为"老考生的成绩分数线"
        history = history[np.isfinite(history)]
        if len(history) == 0:
            return np.nan
        # 当前值在历史中的分位：(历史中 ≤ 当前值的数量) / 历史总数量
        # 这表示：当前值相对于历史的位置，完全基于历史评估当前状态
        return (history <= current).sum() / float(len(history))

    percentile_series = series.rolling(window=window, min_periods=min_periods).apply(
        _percentile, raw=True
    )

    # 【关键修复】：对滚动窗口统计特征强制 shift(1)，确保完全因果
    # 在 t 时刻使用的特征基于 t-1 及之前的数据计算
    if shift:
        percentile_series = percentile_series.shift(1)

    return percentile_series


def _rolling_zscore(
    series: pd.Series,
    window: int,
    min_periods: int = None,
    return_quality: bool = False,
    shift: bool = True,
):
    """
    安全滚动 Z-score（严格因果，无未来信息）

    【核心原则：滚动窗口统计特征需要 shift(1)】
    - 虽然我们在 t 时刻可以使用 close[t]，但对于"滚动窗口统计特征"仍需 shift(1)
    - 原因：避免将当前值包含在历史分布中，即使 rolling 本身是因果的
    - 更严格的做法：在 t 时刻使用的特征基于 t-1 及之前的数据计算

    【关键区分】
    - 基础滚动指标（sma, ema, atr, bb_width）：不需要 shift(1)
    - 滚动窗口统计特征（zscore, percentile, entropy）：需要 shift(1)

    【说明】
    - min_periods 小会导致早期统计量不稳定（小样本噪声）
    - 但 rolling() 只使用历史和当前数据，绝不包含未来
    - 提高 min_periods 能降低虚假相关，因为剔除了高噪声样本

    Args:
        series: 输入时间序列（如 ATR、volatility）
        window: 滚动窗口长度（如 288）
        min_periods: 最小观测数才开始输出（默认=window，最稳健）
        return_quality: 是否同时返回质量分数（0~1，1=完整窗口）
        shift: 是否 shift(1) 以确保完全因果（默认=True，推荐）

    Returns:
        zscore: 标准化后的序列（早期不足窗口处为 NaN）
        quality (可选): 每个点的统计质量 = 实际样本数 / window
    """
    if min_periods is None:
        min_periods = window  # 默认：必须满窗才输出，最稳健

    min_periods = min(min_periods, window)

    # 滚动统计量（只依赖历史和当前，无未来信息）
    rolling_mean = series.rolling(window=window, min_periods=min_periods).mean()
    rolling_std = series.rolling(window=window, min_periods=min_periods).std()

    # 计算 Z-score: (x - mean) / std
    zscore = (series - rolling_mean) / rolling_std.replace(0, np.nan)

    # 【关键修复】：对滚动窗口统计特征强制 shift(1)，确保完全因果
    # 在 t 时刻使用的特征基于 t-1 及之前的数据计算
    if shift:
        zscore = zscore.shift(1)

    if return_quality:
        # 计算质量分数：实际样本数 / window（0~1，1表示使用了完整窗口）
        count = series.rolling(window=window, min_periods=1).count()
        quality = count / window
        # Quality 也需要 shift(1) 以保持对齐
        if shift:
            quality = quality.shift(1)
        return zscore, quality

    # 处理 inf 和 NaN：将 inf 替换为 NaN，然后保留 NaN（不填充，让下游处理）
    zscore = zscore.replace([np.inf, -np.inf], np.nan)

    return zscore


def _trend_r2(prices: pd.Series, window: int = 20, *, lag: int = 0) -> pd.Series:
    """计算趋势R²特征（基于对数价格序列）"""
    log_price = np.log(prices.replace(0, np.nan)).ffill()

    def _compute_r2(series):
        if len(series) < 3:
            return 0.0
        try:
            x = np.arange(len(series))
            y = series.values
            slope, intercept = np.polyfit(x, y, 1)
            y_pred = slope * x + intercept
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
            return max(0.0, min(1.0, r2))
        except Exception:
            return 0.0

    r2_series = log_price.rolling(window=window, min_periods=3).apply(
        _compute_r2, raw=False
    )
    if lag == 0:
        return r2_series
    return r2_series.shift(lag)


def _price_entropy(close: pd.Series, window: int = 50) -> pd.Series:
    """价格方向熵"""
    ret = close.pct_change().fillna(0.0)
    sign = np.sign(ret).replace(0, 1)

    def _entropy(x: np.ndarray) -> float:
        if len(x) == 0:
            return np.nan
        p_up = (x > 0).mean()
        p_dn = 1.0 - p_up
        eps = 1e-9
        return -(p_up * np.log2(p_up + eps) + p_dn * np.log2(p_dn + eps)) / 1.0

    return sign.rolling(window=window, min_periods=1).apply(_entropy, raw=True)


def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Wilder-style RSI."""
    return compute_rsi(series, period)


def _rolling_skew(series: pd.Series, window: int) -> pd.Series:
    """Rolling skewness"""
    return series.rolling(window=window, min_periods=window).skew()


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def create_binary_labels_baseline(
    df: pd.DataFrame, *, forward_bars: int = 3, threshold: float = 0.005
) -> pd.DataFrame:
    """创建二分类标签"""
    df = df.copy()
    df["future_return"] = df["close"].shift(-forward_bars) / df["close"] - 1
    df["binary_signal"] = (df["future_return"] > threshold).astype(int)
    df["signal"] = df["binary_signal"]
    return df


def get_baseline_feature_columns(df: pd.DataFrame) -> List[str]:
    """获取 baseline 特征列"""
    exclude = {
        "open",
        "high",
        "low",
        "close",
        "volume",
        "signal",
        "binary_signal",
        "future_return",
        # Note: _symbol is now included as a categorical feature (not excluded)
        # This allows the model to learn both shared patterns and asset-specific behavior
        # 排除原始布林带值（有量纲），保留归一化的 bb_position 和 bb_width
        "bb_upper",
        "bb_lower",
        "bb_middle",
        # 排除原始 ATR（有量纲），保留归一化的 atr_normalized
        "atr",
        # 排除原始 zigzag（有量纲），保留归一化的 zigzag 特征
        "zigzag",
        # 排除 returns 和 log_returns（虽然无量纲，但不同资产分布差异大）
        # 它们主要用于计算其他特征（如 volatility），不作为最终特征
        "returns",
        "log_returns",
        # 排除 price_change（有量纲），保留归一化的特征
        "price_change",
        # 排除原始均线值（有量纲），保留归一化的 _pct_close 和 _ratio 特征
        "sma_5",
        "sma_10",
        "sma_20",
        "ema_5",
        "ema_10",
        "ema_20",
        "ema_50",
        "wma_20",
        # 注意：momentum_5/10/20 是百分比（pct_change），已经是无量纲的，保留
        # 排除原始成交量值（有量纲），保留归一化的 volume_ratio, vol_ma_ratio, vol_zscore 等
        "volume_sma_20",
        "volume_sma",  # 如果存在的话
        # 排除原始 MACD 值（有量纲），保留归一化的 macd_normalized
        "macd",
        "macd_signal",
        "macd_histogram",
        # 排除原始 POC/HAL 价格值（有量纲），保留归一化的 price_to_poc_pct 等
        "poc",
        "hal_high",
        "hal_low",
        "hal_mid",
        # 排除原始 VWAP 值（有量纲），保留归一化的 price_to_vwap_* 特征
        "vwap",
        # 排除滚动高低点（有量纲），保留归一化的 swing_*_pct_close 和 sr_dist_* 特征
        "roll_high_s",
        "roll_low_s",
        "roll_high_l",
        "roll_low_l",
        # 排除 ZigZag 原始值、OLS 通道等边界值（作为中间计算使用）
        "zz_high_value",
        "zz_low_value",
        "ols_channel_mid",
        "ols_channel_upper",
        "ols_channel_lower",
        # 排除 volatility（虽然基于收益率，但不同资产分布差异大）
        "volatility",
        # 注意：时间特征保留，它们可能包含真实的时间模式信息
    }
    exclude.update(
        [
            col
            for col in df.columns
            if (
                col.startswith("signal_")
                or col.startswith("binary_signal_")
                or col.startswith("future_return_")
            )
        ]
    )
    return [c for c in df.columns if c not in exclude]


@register_feature("compute_bb_width_features_from_series", category="baseline")
def compute_bb_width_features_from_series(
    *,
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    period: int = 20,
    std_dev: int = 2,
    atr_window: int = 14,
) -> pd.DataFrame:
    """
    Narrow-input / narrow-output BB width computation for the feature pipeline.

    Returns only normalized features (no raw price levels):
    - bb_width_normalized: BB width / ATR (cross-asset comparable, ~[0, 5])
    - bb_position: (close - bb_lower) / (bb_upper - bb_lower) (bounded [0, 1])

    Used with YAML `pass_full_df: false` + `column_mappings` to avoid passing/mutating
    a wide DataFrame.
    """
    close = pd.to_numeric(close, errors="coerce").astype(float)
    high = pd.to_numeric(high, errors="coerce").astype(float)
    low = pd.to_numeric(low, errors="coerce").astype(float)

    # Compute Bollinger Bands
    upper, middle, lower = compute_bollinger_bands(
        close, period=period, std_dev=std_dev
    )

    # Compute ATR for normalization
    atr = compute_atr(high, low, close, period=atr_window)

    # BB width normalized by ATR
    width = (upper - lower).abs()
    bb_width_norm = (
        (width / atr.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    )

    # BB position: where is close within the band? 0=lower, 1=upper
    bb_range = (upper - lower).replace(0, np.nan)
    bb_position = (
        ((close - lower) / bb_range)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.5)
        .clip(0.0, 1.0)
    )

    return pd.DataFrame(
        {
            "bb_width_normalized": bb_width_norm,
            "bb_position": bb_position,
        }
    )


@register_feature("compute_roc_5_from_series", category="baseline")
def compute_roc_5_from_series(
    *,
    close: pd.Series,
    period: int = 5,
    z_window: int = 50,
    min_periods: int = 5,
    clip_quantile: float = 0.01,
) -> pd.Series:
    """5-bar ROC z-score (narrow input/output for the feature pipeline)."""
    close = pd.to_numeric(close, errors="coerce").astype(float)
    roc_raw = close.pct_change(period)
    roc_mean = roc_raw.rolling(window=z_window, min_periods=min_periods).mean()
    roc_std = roc_raw.rolling(window=z_window, min_periods=min_periods).std()
    roc_std = roc_std.clip(lower=roc_raw.abs().quantile(clip_quantile))
    roc_5 = (
        ((roc_raw - roc_mean) / roc_std.replace(0, np.nan))
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )
    roc_5.name = "roc_5"
    return roc_5


@register_feature("compute_trend_confidence_from_series", category="baseline")
def compute_trend_confidence_from_series(
    *,
    close: pd.Series,
    horizons: Union[Sequence[int], Tuple[int, ...], List[int], None] = None,
) -> pd.DataFrame:
    """Multi-horizon sign agreement trend score (dual_add_trend / multi-leg).

    ``trend_confidence = mean(|sign(ret_h)|) * |mean(sign(ret_h))|`` over
    ``horizons`` (default 3, 5, 10 bars), matching legacy ``_add_trend_features``.
    """
    close = pd.to_numeric(close, errors="coerce").astype(float)
    if horizons is None or len(horizons) == 0:
        hs: Tuple[int, ...] = (3, 5, 10)
    else:
        hs = tuple(int(x) for x in horizons)
    rets = [close.pct_change(h) for h in hs]
    signs = pd.concat([np.sign(r) for r in rets], axis=1).fillna(0.0)
    signs.columns = list(range(len(hs)))
    mean_signs = signs.mean(axis=1)
    trend_direction_raw = np.sign(mean_signs)
    trend_confidence = signs.abs().mean(axis=1) * mean_signs.abs()
    trend_direction = np.where(trend_direction_raw >= 0, "UP", "DOWN")
    return pd.DataFrame(
        {
            "trend_confidence": trend_confidence,
            "trend_direction_raw": trend_direction_raw,
            "trend_direction": trend_direction,
        },
        index=close.index,
    )


@register_feature("compute_acceleration_3_from_series", category="baseline")
def compute_acceleration_3_from_series(
    *, close: pd.Series, feature_shift: int = 0
) -> pd.DataFrame:
    """Narrow-IO acceleration_3: normalized ROC(3) difference."""
    close = pd.to_numeric(close, errors="coerce").astype(float)
    roc_3 = close.pct_change(3)
    roc_3_mean = roc_3.rolling(window=50, min_periods=5).mean()
    roc_3_std = roc_3.rolling(window=50, min_periods=5).std()
    roc_3_std = roc_3_std.clip(lower=roc_3.abs().quantile(0.01))
    roc_3_norm = (
        ((roc_3 - roc_3_mean) / roc_3_std.replace(0, np.nan))
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )
    current = roc_3_norm.shift(feature_shift) if feature_shift > 0 else roc_3_norm
    prev = roc_3_norm.shift(feature_shift + 1)
    # Match legacy behavior: first rows can be NaN due to shifting.
    out = (current - prev).rename("acceleration_3")
    return out.to_frame()


@register_feature("compute_path_curvature_from_series", category="baseline")
def compute_path_curvature_from_series(
    *,
    close: pd.Series,
    smooth_window: int = 5,
    window: int = 5,
    z_window: int = 200,
    min_periods: int = 50,
) -> pd.DataFrame:
    """
    Path curvature (离散路径曲率): measure how violently the path changes direction.

    Design goals:
    - Unitless / cross-asset comparable: compute on returns (not raw price diffs).
    - Causal: rolling stats are shifted(1) via _rolling_zscore().

    Steps (simplified discrete approximation):
      1) Smooth close with rolling mean
      2) Velocity ≈ smoothed pct_change (unitless)
      3) Acceleration ≈ diff of velocity
      4) Curvature ≈ |acc| / (1 + vel^2)^(3/2)
      5) Smooth curvature and normalize (log1p + rolling z-score)
    """
    close = pd.to_numeric(close, errors="coerce").astype(float)
    close = close.replace(0, np.nan)

    # 1) Smooth price to reduce micro-noise (still causal)
    smoothed = close.rolling(window=int(smooth_window), min_periods=1).mean()

    # 2) Velocity in return space (unitless)
    velocity = smoothed.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # 3) Acceleration: change in velocity
    acceleration = velocity.diff().replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # 4) Curvature proxy (stable)
    denom = (1.0 + velocity**2) ** 1.5
    curvature = (acceleration.abs() / (denom + 1e-8)).replace([np.inf, -np.inf], np.nan)

    # 5) Smooth + normalize
    curvature_smoothed = curvature.rolling(window=int(window), min_periods=1).mean()
    curvature_log = np.log1p(curvature_smoothed.clip(lower=0.0).fillna(0.0))
    curvature_z = _rolling_zscore(
        curvature_log,
        window=int(z_window),
        min_periods=int(min_periods),
        shift=True,
    ).fillna(0.0)
    curvature_z.name = "path_curvature"
    return curvature_z.to_frame()


@register_feature("compute_volatility_cone_position_from_series", category="baseline")
def compute_volatility_cone_position_from_series(
    *,
    close: pd.Series,
    window: int = 20,
    lookback: int | None = None,
    lookback_days: int = 252,
    timeframe_minutes: int | None = None,
    min_periods: int | None = None,
) -> pd.DataFrame:
    """
    Volatility cone position (波动率锥位置):
    current rolling volatility's percentile rank within the past lookback window.

    Implementation notes:
    - Uses rolling std of returns as "current vol" proxy (unitless).
    - Percentile rank is computed causally via _rolling_percentile(..., shift=True).
      This uses only historical distribution (excluding current point) and then shift(1).

    Args:
        close: price series
        window: rolling window for volatility (e.g., 20)
        lookback: explicit lookback length in *bars* (overrides lookback_days conversion)
        lookback_days: lookback length in *days* (converted to bars using timeframe)
        timeframe_minutes: explicit timeframe in minutes (optional; if None and index is
            DatetimeIndex, try to infer from index frequency)
        min_periods: minimum periods for lookback percentile computation (default=lookback)

    Returns:
        DataFrame with 'volatility_cone_position' in [0,1]
    """
    close = pd.to_numeric(close, errors="coerce").astype(float)
    close = close.replace(0, np.nan)

    # Resolve lookback in bars
    if lookback is None:
        tf_min = None
        if timeframe_minutes is not None:
            tf_min = int(timeframe_minutes)
        else:
            # Best-effort: infer from DatetimeIndex (e.g., 240T, 4H, 5T)
            try:
                if isinstance(close.index, pd.DatetimeIndex) and len(close.index) >= 3:
                    inferred = pd.infer_freq(close.index)
                    if inferred:
                        from pandas.tseries.frequencies import to_offset

                        off = to_offset(inferred)
                        # Convert to minutes (nanoseconds -> minutes)
                        tf_min = int(round(off.delta.total_seconds() / 60.0))
            except Exception:
                tf_min = None

        if tf_min and tf_min > 0:
            bars_per_day = int(round((24 * 60) / tf_min))
            bars_per_day = max(1, bars_per_day)
            lookback = int(max(5, lookback_days * bars_per_day))
        else:
            # Fallback: keep backward-compatible behavior (252 bars)
            lookback = 252

    # Unitless returns
    rets = close.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Current rolling vol (unitless); do NOT annualize here (position is relative anyway)
    vol = rets.rolling(window=int(window), min_periods=max(2, int(window) // 2)).std()

    if min_periods is None:
        min_periods = int(lookback)

    pos = _rolling_percentile(
        vol.astype(float),
        window=int(lookback),
        min_periods=int(min_periods),
        shift=True,
    )
    out = pos.clip(0.0, 1.0).fillna(0.5).rename("volatility_cone_position")
    return out.to_frame()


@register_feature("compute_volume_anomaly_from_series", category="baseline")
def compute_volume_anomaly_from_series(*, volume: pd.Series) -> pd.DataFrame:
    """Narrow-IO volume_anomaly: EWM-based z-score of volume ratio."""
    volume = pd.to_numeric(volume, errors="coerce").astype(float)
    vol_ratio = volume / volume.ewm(span=20, min_periods=10).mean()
    vol_ratio = vol_ratio.replace([np.inf, -np.inf], np.nan).fillna(1.0)
    vol_log = np.log1p(vol_ratio)
    mean = vol_log.rolling(50, min_periods=10).mean()
    std = vol_log.rolling(50, min_periods=10).std()
    out = (
        ((vol_log - mean) / std.replace(0, np.nan))
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .rename("volume_anomaly")
    )
    return out.to_frame()


@register_feature("compute_trend_r2_20_from_series", category="baseline")
def compute_trend_r2_20_from_series(
    *, close: pd.Series, feature_shift: int = 0
) -> pd.DataFrame:
    """Narrow-IO 20-bar trend R²."""
    close = pd.to_numeric(close, errors="coerce").astype(float)
    # Match legacy behavior: initial rows can be NaN due to insufficient window.
    out = _trend_r2(close, window=20, lag=feature_shift)
    out.name = "trend_r2_20"
    return out.to_frame()


@register_feature("compute_trend_r2_50_from_series", category="baseline")
def compute_trend_r2_50_from_series(
    *, close: pd.Series, feature_shift: int = 0
) -> pd.DataFrame:
    """Narrow-IO 50-bar trend R²."""
    close = pd.to_numeric(close, errors="coerce").astype(float)
    # Match legacy behavior: initial rows can be NaN due to insufficient window.
    out = _trend_r2(close, window=50, lag=feature_shift)
    out.name = "trend_r2_50"
    return out.to_frame()


@register_feature("compute_slope_consistency_score_from_series", category="baseline")
def compute_slope_consistency_score_from_series(*, close: pd.Series) -> pd.DataFrame:
    """Narrow-IO EMA slope agreement (10/20/50)."""
    close = pd.to_numeric(close, errors="coerce").astype(float)
    ema10 = close.ewm(span=10).mean()
    ema20 = close.ewm(span=20).mean()
    ema50 = close.ewm(span=50).mean()
    slope10 = np.sign(ema10.diff())
    slope20 = np.sign(ema20.diff())
    slope50 = np.sign(ema50.diff())
    out = (
        (
            (slope10 == slope20).astype(int)
            + (slope20 == slope50).astype(int)
            + (slope10 == slope50).astype(int)
        )
        .fillna(0)
        .rename("slope_consistency_score")
    )
    return out.to_frame()


@register_feature("compute_volatility_reversal_score_from_series", category="baseline")
def compute_volatility_reversal_score_from_series(
    *, high: pd.Series, low: pd.Series, close: pd.Series
) -> pd.DataFrame:
    """Narrow-IO ATR mean-reversion z-score."""
    high = pd.to_numeric(high, errors="coerce").astype(float)
    low = pd.to_numeric(low, errors="coerce").astype(float)
    close = pd.to_numeric(close, errors="coerce").astype(float)
    atr = compute_atr(high, low, close)
    atr_mean = atr.rolling(50).mean()
    atr_std = atr.rolling(50).std()
    out = (
        ((atr - atr_mean) / atr_std.replace(0, np.nan))
        .fillna(0.0)
        .rename("volatility_reversal_score")
    )
    return out.to_frame()


@register_feature("compute_atr_percentile_from_series", category="baseline")
def compute_atr_percentile_from_series(
    *,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 540,
    shift: int = 1,
) -> pd.DataFrame:
    """Narrow-IO ATR percentile (compression detector).

    warmup 不足（前 window-1 行）保持 NaN，禁止静默降级为 0.5。
    """
    from src.features.live_tail_context import live_tail_slice_len, restore_live_tail_frame

    high = pd.to_numeric(high, errors="coerce").astype(float)
    low = pd.to_numeric(low, errors="coerce").astype(float)
    close = pd.to_numeric(close, errors="coerce").astype(float)
    # ATR (TA-Lib Wilder smoothing) is IIR/path-dependent, so compute it over the
    # FULL history first (C-fast), then slice only for the expensive rolling.apply.
    atr = compute_atr(high, low, close).astype(float)

    _full_index = close.index
    _slice_len = live_tail_slice_len(len(atr), warmup=int(window) + int(shift or 0))
    if _slice_len is not None:
        atr = atr.iloc[-_slice_len:]

    def _percentile(arr: np.ndarray) -> float:
        if len(arr) == 0:
            return np.nan
        current = arr[-1]
        return float(np.mean(arr <= current))

    pct = atr.rolling(window=window, min_periods=window).apply(_percentile, raw=True)
    if shift:
        pct = pct.shift(shift)
    out = pct.clip(0.0, 1.0).rename("atr_percentile")
    result = out.to_frame()
    if _slice_len is not None:
        result = restore_live_tail_frame(result, _full_index, fill_value=0.0)
    return result


@register_feature("compute_percentile_rank_from_series", category="baseline")
def compute_percentile_rank_from_series(
    *,
    series: pd.Series,
    window: int = 540,
    shift: int = 1,
    output_name: str = "percentile",
) -> pd.DataFrame:
    """
    Compute percentile rank of a series using rolling window.

    This is a generic function for computing percentile ranks of any feature.
    Used for features like cvd_change_5_pct, volume_ratio_pct, etc.

    warmup 不足（前 window-1 行）保持 NaN，禁止静默降级为 0.5。
    """
    from src.features.live_tail_context import live_tail_slice_len, restore_live_tail_frame

    _full_index = series.index
    _slice_len = live_tail_slice_len(
        len(series), warmup=int(window) + int(shift or 0)
    )
    if _slice_len is not None:
        series = series.iloc[-_slice_len:]

    series = pd.to_numeric(series, errors="coerce").astype(float)

    def _percentile(arr: np.ndarray) -> float:
        if len(arr) == 0:
            return np.nan
        current = arr[-1]
        history = arr[:-1]  # Use history only (causal)
        if len(history) == 0:
            return np.nan
        return float(np.mean(history <= current))

    pct = series.rolling(window=window, min_periods=window).apply(_percentile, raw=True)
    if shift:
        pct = pct.shift(shift)
    out = pct.clip(0.0, 1.0).rename(output_name)
    result = out.to_frame()
    if _slice_len is not None:
        # Positional pad: label reindex fails on duplicate tick minutes (OFCI).
        result = restore_live_tail_frame(result, _full_index, fill_value=0.0)
    return result


@register_feature("compute_cvd_change_5_pct_from_series", category="baseline")
def compute_cvd_change_5_pct_from_series(
    *,
    cvd_change_5: pd.Series,
    window: int = 540,
    shift: int = 1,
) -> pd.DataFrame:
    """Compute percentile rank of cvd_change_5 for cross-symbol stability."""
    return compute_percentile_rank_from_series(
        series=cvd_change_5,
        window=window,
        shift=shift,
        output_name="cvd_change_5_pct",
    )


@register_feature("compute_sqs_hal_high_pct_from_series", category="baseline")
def compute_sqs_hal_high_pct_from_series(
    *,
    sqs_hal_high: pd.Series,
    window: int = 540,
    shift: int = 1,
) -> pd.DataFrame:
    """Compute percentile rank of sqs_hal_high for cross-symbol comparability."""
    return compute_percentile_rank_from_series(
        series=sqs_hal_high,
        window=window,
        shift=shift,
        output_name="sqs_hal_high_pct",
    )


@register_feature("compute_sqs_hal_low_pct_from_series", category="baseline")
def compute_sqs_hal_low_pct_from_series(
    *,
    sqs_hal_low: pd.Series,
    window: int = 540,
    shift: int = 1,
) -> pd.DataFrame:
    """Compute percentile rank of sqs_hal_low for cross-symbol comparability."""
    return compute_percentile_rank_from_series(
        series=sqs_hal_low,
        window=window,
        shift=shift,
        output_name="sqs_hal_low_pct",
    )


@register_feature("compute_volume_ratio_pct_from_series", category="baseline")
def compute_volume_ratio_pct_from_series(
    *,
    volume: pd.Series,
    ratio_window: int = 20,
    percentile_window: int = 540,
    shift: int = 1,
) -> pd.DataFrame:
    """
    自包含版本：内部计算 volume_ratio 再转百分位

    1. volume_ratio = volume / rolling_mean(volume)
    2. volume_ratio_pct = percentile_rank(volume_ratio)
    """
    vol = pd.to_numeric(volume, errors="coerce").astype(float)

    # Step 1: 计算 volume_ratio
    rolling_mean = vol.rolling(window=ratio_window, min_periods=1).mean()
    rolling_mean_safe = rolling_mean.replace(0, np.nan)
    ratio = vol / rolling_mean_safe
    ratio = ratio.replace([np.inf, -np.inf], np.nan).clip(0.0, 10.0).fillna(1.0)

    # Step 2: 转百分位
    return compute_percentile_rank_from_series(
        series=ratio,
        window=percentile_window,
        shift=shift,
        output_name="volume_ratio_pct",
    )


@register_feature("compute_bb_width_normalized_pct_from_series", category="baseline")
def compute_bb_width_normalized_pct_from_series(
    *,
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    bb_period: int = 20,
    bb_std_dev: int = 2,
    atr_window: int = 14,
    percentile_window: int = 540,
    shift: int = 1,
) -> pd.DataFrame:
    """
    自包含版本：内部计算 bb_width_normalized 再转百分位

    1. bb_width_normalized = BB_width / ATR (跨资产可比)
    2. bb_width_normalized_pct = percentile_rank(bb_width_normalized)
    """
    close = pd.to_numeric(close, errors="coerce").astype(float)
    high = pd.to_numeric(high, errors="coerce").astype(float)
    low = pd.to_numeric(low, errors="coerce").astype(float)

    # Step 1: 计算 BB bands
    upper, middle, lower = compute_bollinger_bands(
        close, period=bb_period, std_dev=bb_std_dev
    )

    # Step 2: 计算 ATR
    atr = compute_atr(high, low, close, period=atr_window)

    # Step 3: BB width normalized by ATR
    width = (upper - lower).abs()
    bb_width_norm = (
        (width / atr.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    )

    # Step 4: 转百分位
    return compute_percentile_rank_from_series(
        series=bb_width_norm,
        window=percentile_window,
        shift=shift,
        output_name="bb_width_normalized_pct",
    )


@register_feature(
    "compute_cvd_change_5_normalized_pct_from_series", category="baseline"
)
def compute_cvd_change_5_normalized_pct_from_series(
    *,
    cvd_change_5_normalized: pd.Series,
    window: int = 540,
    shift: int = 1,
) -> pd.DataFrame:
    """Compute percentile rank of cvd_change_5_normalized for cross-symbol stability."""
    return compute_percentile_rank_from_series(
        series=cvd_change_5_normalized,
        window=window,
        shift=shift,
        output_name="cvd_change_5_normalized_pct",
    )


# Regime features (path_efficiency, jump_risk, price_dir_consistency, path_length)
# These should be computed in FeatureStore, not just in classify_regime
def _compute_jump_risk_from_series(
    *,
    close: pd.Series,
    atr: pd.Series,
    window: int = 10,
) -> pd.Series:
    """Compute jump risk (max abs return / std of returns)."""
    close = pd.to_numeric(close, errors="coerce").astype(float)
    atr = pd.to_numeric(atr, errors="coerce").astype(float)

    if len(close) < window:
        return pd.Series(np.nan, index=close.index)

    returns = close.pct_change().fillna(0.0)
    jump_risk = pd.Series(np.nan, index=close.index)

    for i in range(window, len(close)):
        window_returns = returns.iloc[i - window : i]
        if not window_returns.isna().all():
            max_abs_ret = window_returns.abs().max()
            std_ret = window_returns.std()
            if std_ret > 1e-9:
                jump_risk.iloc[i] = max_abs_ret / std_ret

    return jump_risk


def compute_jump_risk_from_series(
    *,
    close: pd.Series,
    atr: pd.Series,
    window: int = 10,
) -> pd.DataFrame:
    """Narrow-IO jump risk computation."""
    jump_risk = _compute_jump_risk_from_series(close=close, atr=atr, window=window)
    return jump_risk.rename("jump_risk").to_frame()


@register_feature("compute_jump_risk_pct_from_series", category="baseline")
def compute_jump_risk_pct_from_series(
    *,
    close: pd.Series,
    atr: pd.Series,
    window: int = 10,
    percentile_window: int = 540,
    shift: int = 1,
) -> pd.DataFrame:
    """Compute jump_risk and its percentile rank."""
    jump_risk = _compute_jump_risk_from_series(close=close, atr=atr, window=window)
    jump_risk_pct = compute_percentile_rank_from_series(
        series=jump_risk,
        window=percentile_window,
        shift=shift,
        output_name="jump_risk_pct",
    )
    return jump_risk_pct


def _compute_path_length_from_series(
    *,
    close: pd.Series,
    atr: pd.Series,
    window: int = 20,
) -> pd.Series:
    """Rolling path length in ATR units (sum of abs returns / ATR)."""
    close = pd.to_numeric(close, errors="coerce").astype(float)
    atr = pd.to_numeric(atr, errors="coerce").astype(float)

    if len(close) < window:
        return pd.Series(np.nan, index=close.index)

    diffs = close.diff().abs().fillna(0.0)
    atr_safe = atr + 1e-9
    path = diffs / atr_safe
    path_length = path.rolling(window=window, min_periods=window).sum()

    return path_length


@register_feature("compute_path_length_pct_from_series", category="baseline")
def compute_path_length_pct_from_series(
    *,
    close: pd.Series,
    atr: pd.Series,
    window: int = 20,
    percentile_window: int = 540,
    shift: int = 1,
) -> pd.DataFrame:
    """Compute path_length and its percentile rank."""
    path_length = _compute_path_length_from_series(close=close, atr=atr, window=window)
    path_length_pct = compute_percentile_rank_from_series(
        series=path_length,
        window=percentile_window,
        shift=shift,
        output_name="path_length_pct",
    )
    return path_length_pct


def _compute_path_efficiency_from_series(
    *,
    close: pd.Series,
    atr: pd.Series,
    window: int = 20,
) -> pd.Series:
    """Compute path efficiency: net_displacement / total_path_length."""
    close = pd.to_numeric(close, errors="coerce").astype(float)
    atr = pd.to_numeric(atr, errors="coerce").astype(float)

    if len(close) < window:
        return pd.Series(np.nan, index=close.index)

    # Net displacement: absolute change in price over window
    net_displacement = close.rolling(window=window).apply(
        lambda x: abs(x.iloc[-1] - x.iloc[0]) if len(x) == window else np.nan,
        raw=False,
    )

    # Total path length: sum of absolute returns over window
    diffs = close.diff().abs().fillna(0.0)
    atr_safe = atr + 1e-9
    path = diffs / atr_safe
    total_path_length = path.rolling(window=window, min_periods=window).sum()

    # Normalize net_displacement by ATR
    net_displacement_atr = net_displacement / atr_safe

    # Path efficiency: net_displacement / total_path_length
    path_efficiency = net_displacement_atr / total_path_length.replace(0, np.nan)
    path_efficiency = path_efficiency.clip(0.0, 1.0).fillna(np.nan)

    return path_efficiency


@register_feature("compute_path_efficiency_pct_from_series", category="baseline")
def compute_path_efficiency_pct_from_series(
    *,
    close: pd.Series,
    atr: pd.Series,
    window: int = 20,
    percentile_window: int = 540,
    shift: int = 1,
) -> pd.DataFrame:
    """Compute path_efficiency and its percentile rank."""
    path_efficiency = _compute_path_efficiency_from_series(
        close=close, atr=atr, window=window
    )
    path_efficiency_pct = compute_percentile_rank_from_series(
        series=path_efficiency,
        window=percentile_window,
        shift=shift,
        output_name="path_efficiency_pct",
    )
    return path_efficiency_pct


def _compute_price_direction_consistency_from_series(
    *,
    close: pd.Series,
    window: int = 20,
) -> pd.Series:
    """Compute rolling consistency of actual price direction (sign of returns)."""
    close = pd.to_numeric(close, errors="coerce").astype(float)

    if len(close) < window:
        return pd.Series(np.nan, index=close.index)

    returns = close.diff().fillna(0.0)
    signs = np.sign(returns)
    # abs(mean(sign)) = 1 if stable direction, ~0 if flipping
    consistency = signs.rolling(window=window, min_periods=window).mean().abs()

    return consistency


@register_feature("compute_price_dir_consistency_pct_from_series", category="baseline")
def compute_price_dir_consistency_pct_from_series(
    *,
    close: pd.Series,
    window: int = 20,
    percentile_window: int = 540,
    shift: int = 1,
) -> pd.DataFrame:
    """Compute price_dir_consistency and its percentile rank."""
    price_dir_consistency = _compute_price_direction_consistency_from_series(
        close=close, window=window
    )
    price_dir_consistency_pct = compute_percentile_rank_from_series(
        series=price_dir_consistency,
        window=percentile_window,
        shift=shift,
        output_name="price_dir_consistency_pct",
    )
    return price_dir_consistency_pct


@register_feature("compute_trend_volatility_alignment_from_series", category="baseline")
def compute_trend_volatility_alignment_from_series(
    *,
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    feature_shift: int = 0,
    atr_percentile_window: int = 540,
) -> pd.DataFrame:
    """Narrow-IO alignment of ROC sign and ATR percentile regime."""
    close = pd.to_numeric(close, errors="coerce").astype(float)
    roc_5 = compute_roc_5_from_series(close=close)
    atr_pct = compute_atr_percentile_from_series(
        high=high, low=low, close=close, window=atr_percentile_window, shift=1
    )["atr_percentile"]
    out = (
        np.sign(roc_5.shift(feature_shift)).fillna(0.0) * atr_pct.fillna(0.0)
    ).rename("trend_volatility_alignment")
    return out.to_frame()


@register_feature("compute_compression_duration_from_series", category="baseline")
def compute_compression_duration_from_series(
    *,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    percentile_window: int = 540,
    compression_threshold_pct: float = 0.2,
) -> pd.Series:
    """
    计算压缩持续时间（基于 ATR percentile）

    压缩持续时间：连续 bar 数，其中 ATR percentile 低于阈值
    基于原始 BaselineFeatureEngineer 的实现
    """
    high = pd.to_numeric(high, errors="coerce").astype(float)
    low = pd.to_numeric(low, errors="coerce").astype(float)
    close = pd.to_numeric(close, errors="coerce").astype(float)

    # 计算 ATR
    atr = compute_atr(high, low, close, period=14)

    # 计算 ATR percentile（滚动百分位排名）
    def _rolling_percentile(series: pd.Series, window: int) -> pd.Series:
        def _rank(x: np.ndarray) -> float:
            if len(x) <= 1 or not np.isfinite(x[-1]):
                return np.nan
            last = x[-1]
            arr = x[np.isfinite(x)]
            if len(arr) == 0:
                return np.nan
            return (arr <= last).sum() / float(len(arr))

        return series.rolling(window=window, min_periods=1).apply(_rank, raw=True)

    atr_percentile = _rolling_percentile(atr, window=percentile_window)

    # 压缩持续时间：连续 bar 数，其中 ATR percentile <= threshold
    threshold = compression_threshold_pct
    below = (atr_percentile.fillna(0.0) <= threshold).astype(int)

    # Run-length encoding: 计算连续 1 的长度
    run = np.zeros(len(below), dtype=int)
    cnt = 0
    for i, v in enumerate(below.values):
        if v == 1:
            cnt += 1
        else:
            cnt = 0
        run[i] = cnt

    # Normalize to a unitless ratio for cross-asset/timeframe comparability.
    denom = float(percentile_window) if float(percentile_window) > 0 else 1.0
    out = (pd.Series(run, index=close.index, name="compression_duration") / denom).clip(
        0.0, 1.0
    )
    return out


@register_feature("compute_recent_compression_decay_from_series", category="baseline")
def compute_recent_compression_decay_from_series(
    *,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    percentile_window: int = 540,
    compression_threshold_pct: float = 0.2,
    decay_rate: float = 0.97,
) -> pd.Series:
    """
    压缩衰减记忆特征 (ME Prefilter 核心)

    语义: "最近是否经历过压缩？"  ——  ME 需要先蓄势(压缩)再扩张。

    - 压缩期间(ATR percentile <= threshold): 记录压缩强度(越深越高)
    - 压缩结束后(ATR 扩张): 按 decay_rate 指数衰减
    - 输出 [0, 1]: 越高 = 最近越有过深度压缩

    与 atr_percentile 配合使用:
      atr_percentile > 0.7  AND  recent_compression_decay > 0.3
      = "当前正在扩张 + 最近刚从压缩出来"

    默认 decay_rate=0.97, 半衰期 ~23 bars (60T 约1天)。
    """
    high = pd.to_numeric(high, errors="coerce").astype(float)
    low = pd.to_numeric(low, errors="coerce").astype(float)
    close = pd.to_numeric(close, errors="coerce").astype(float)

    # 复用 compression_duration 的 ATR percentile 计算
    atr = compute_atr(high, low, close, period=14)

    def _rolling_percentile(series: pd.Series, window: int) -> pd.Series:
        def _rank(x: np.ndarray) -> float:
            if len(x) <= 1 or not np.isfinite(x[-1]):
                return np.nan
            last = x[-1]
            arr = x[np.isfinite(x)]
            if len(arr) == 0:
                return np.nan
            return (arr <= last).sum() / float(len(arr))

        return series.rolling(window=window, min_periods=1).apply(_rank, raw=True)

    atr_pct = _rolling_percentile(atr, window=percentile_window).fillna(0.5).values
    threshold = float(compression_threshold_pct)
    _decay = float(decay_rate)
    n = len(atr_pct)

    # 压缩强度: threshold 以下越深, 强度越高 [0, 1]
    compression_signal = np.maximum(0.0, (threshold - atr_pct) / threshold)

    # 指数衰减记忆: 压缩期间跟踪峰值, 扩张后衰减
    memory = np.zeros(n, dtype=float)
    for i in range(n):
        prev = memory[i - 1] * _decay if i > 0 else 0.0
        memory[i] = max(compression_signal[i], prev)

    out = pd.Series(
        np.clip(memory, 0.0, 1.0),
        index=close.index,
        name="recent_compression_decay",
    )
    return out


@register_feature("compute_compression_energy_from_series", category="baseline")
def compute_compression_energy_from_series(
    *, wpt_energy_cascade: pd.Series, bb_width_ratio: pd.Series
) -> pd.Series:
    """
    计算压缩能量（结合 WPT 能量和布林带压缩）

    基于原始 BaselineFeatureEngineer 的实现：
    compression_energy = (1.0 / bb_width) * volume_ratio
    这里使用 wpt_energy_cascade 替代 volume_ratio，bb_width_ratio 替代 bb_width
    """
    wpt_energy = (
        pd.to_numeric(wpt_energy_cascade, errors="coerce").astype(float).fillna(0.0)
    )
    bb_width = pd.to_numeric(bb_width_ratio, errors="coerce").astype(float)

    # 压缩能量 = (1.0 / bb_width_ratio) * wpt_energy_cascade
    # 注意：bb_width_ratio 越小，压缩越强，所以用 1/bb_width_ratio
    compression_energy_raw = (1.0 / bb_width.replace(0, np.nan)) * wpt_energy
    compression_energy_raw = compression_energy_raw.replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0)

    # 使用 log 转换避免极端值，然后标准化
    compression_energy_log = np.log1p(np.abs(compression_energy_raw)) * np.sign(
        compression_energy_raw
    )
    compression_energy_mean = compression_energy_log.rolling(50, min_periods=10).mean()
    compression_energy_std = compression_energy_log.rolling(50, min_periods=10).std()

    out = (
        (
            (compression_energy_log - compression_energy_mean)
            / compression_energy_std.replace(0, np.nan)
        )
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
        .clip(-5, 5)
    )

    out.name = "compression_energy"
    return out


@register_feature("compute_sma_slope_from_series", category="baseline")
def compute_sma_slope_from_series(*, sma_200: pd.Series, window: int = 5) -> pd.Series:
    """
    计算 SMA 200 斜率（均线变化率）

    使用线性回归计算斜率，归一化到价格水平
    """
    sma = pd.to_numeric(sma_200, errors="coerce").astype(float)

    # 计算斜率：使用差分或线性回归
    # 方法1：简单差分（归一化）
    sma_diff = sma.diff(window)
    sma_safe = sma.replace(0, np.nan)
    slope = (sma_diff / sma_safe).replace([np.inf, -np.inf], np.nan)

    # 方法2：使用线性回归（更稳健，但计算更慢）
    # 这里使用简单差分，如果需要更稳健可以用线性回归

    out = slope.fillna(0.0)
    out.name = "sma_200_slope"
    return out


@register_feature("compute_ma_slope_from_series", category="baseline")
def compute_ma_slope_from_series(
    *,
    ma: pd.Series,
    window: int = 10,
    output_column: str = "ma_slope",
) -> pd.DataFrame:
    """Generic N-bar MA slope, normalized by current MA level.

    slope_t = (ma_t - ma_{t-window}) / ma_t

    Used by ``ema_1200_slope_f`` (with column_mappings: ma=ema_1200).
    """
    ma_series = pd.to_numeric(ma, errors="coerce").astype(float)
    ma_diff = ma_series.diff(window)
    ma_safe = ma_series.replace(0, np.nan)
    slope = (ma_diff / ma_safe).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return pd.DataFrame({output_column: slope}, index=ma.index)


@register_feature("compute_abc_macro_regime_score_from_series", category="baseline")
def compute_abc_macro_regime_score_from_series(
    *,
    ema_1200_position: pd.Series,
    ema_1200_slope_10: pd.Series,
    atr_percentile: pd.Series,
    oi_zscore: pd.Series | None = None,
    funding_rate_zscore_50: pd.Series | None = None,
    position_min: float = 0.02,
    position_strong: float = 0.08,
    atr_active: float = 0.35,
    oi_expansion: float = 0.5,
    funding_support: float = 0.0,
    score_threshold_bull: float = 4.0,
    score_threshold_transition: float = 3.0,
) -> pd.DataFrame:
    """A-layer macro regime score for spot/fat-tail participation.

    This is a research proxy for "world state" rather than a trading signal:
    slow trend location, slope, volatility activity, and liquidity/funding
    support each contribute one point. Output state: 0=bear/avoid,
    1=transition, 2=bull/risk-on.
    """
    idx = ema_1200_position.index
    pos = pd.to_numeric(ema_1200_position, errors="coerce").reindex(idx).fillna(0.0)
    slope = pd.to_numeric(ema_1200_slope_10, errors="coerce").reindex(idx).fillna(0.0)
    atr = pd.to_numeric(atr_percentile, errors="coerce").reindex(idx).fillna(0.0)

    if oi_zscore is None:
        oi = pd.Series(0.0, index=idx)
    else:
        oi = pd.to_numeric(oi_zscore, errors="coerce").reindex(idx).fillna(0.0)
    if funding_rate_zscore_50 is None:
        funding = pd.Series(0.0, index=idx)
    else:
        funding = (
            pd.to_numeric(funding_rate_zscore_50, errors="coerce")
            .reindex(idx)
            .fillna(0.0)
        )

    score = pd.Series(0.0, index=idx)
    score += (pos >= float(position_min)).astype(float)
    score += (slope >= 0.0).astype(float)
    score += (pos >= float(position_strong)).astype(float)
    score += (atr >= float(atr_active)).astype(float)
    score += ((oi >= float(oi_expansion)) | (funding >= float(funding_support))).astype(
        float
    )

    state = pd.Series(0.0, index=idx)
    state.loc[score >= float(score_threshold_transition)] = 1.0
    state.loc[score >= float(score_threshold_bull)] = 2.0

    return pd.DataFrame(
        {
            "abc_macro_regime_score": score,
            "abc_macro_regime_state": state,
        },
        index=idx,
    )


@register_feature("compute_weekly_macro_cycle_exit_from_series", category="baseline")
def compute_weekly_macro_cycle_exit_from_series(
    *,
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    ema_span_weeks: int = 50,
    close_below_weeks: int = 2,
    llhl_weeks: int = 2,
) -> pd.DataFrame:
    """Weekly macro-cycle exit proxy, projected back to base timeframe.

    Outputs:
    - weekly_close_below_ema50_2w: weekly close below weekly EMA50 for N weeks
    - weekly_lower_high_lower_low_2w: consecutive weekly lower-high + lower-low
    - weekly_macro_cycle_exit_signal: 1 when both conditions hold, else 0
    """
    idx = close.index
    c = pd.to_numeric(close, errors="coerce").astype(float)
    h = pd.to_numeric(high, errors="coerce").astype(float)
    l = pd.to_numeric(low, errors="coerce").astype(float)

    weekly = (
        pd.DataFrame({"close": c, "high": h, "low": l}, index=idx)
        .sort_index()
        .resample("W-SUN", label="right", closed="right")
        .agg({"close": "last", "high": "max", "low": "min"})
    )
    weekly = weekly.dropna(subset=["close", "high", "low"])
    if weekly.empty:
        z = pd.Series(0.0, index=idx)
        return pd.DataFrame(
            {
                "weekly_close_below_ema50_2w": z,
                "weekly_lower_high_lower_low_2w": z,
                "weekly_macro_cycle_exit_signal": z,
            },
            index=idx,
        )

    wk_ema50 = (
        weekly["close"]
        .ewm(
            span=int(ema_span_weeks),
            adjust=False,
            min_periods=max(2, int(ema_span_weeks)),
        )
        .mean()
    )
    wk_below = weekly["close"] < wk_ema50
    wk_below_n = wk_below.astype(float).rolling(
        window=max(1, int(close_below_weeks)),
        min_periods=max(1, int(close_below_weeks)),
    ).sum() >= float(max(1, int(close_below_weeks)))

    wk_lh = weekly["high"] < weekly["high"].shift(1)
    wk_ll = weekly["low"] < weekly["low"].shift(1)
    wk_llhl = wk_lh & wk_ll
    wk_llhl_n = wk_llhl.astype(float).rolling(
        window=max(1, int(llhl_weeks)), min_periods=max(1, int(llhl_weeks))
    ).sum() >= float(max(1, int(llhl_weeks)))

    wk_exit = (wk_below_n & wk_llhl_n).astype(float)
    wk_below_n = wk_below_n.astype(float)
    wk_llhl_n = wk_llhl_n.astype(float)

    out = pd.DataFrame(
        {
            "weekly_close_below_ema50_2w": wk_below_n,
            "weekly_lower_high_lower_low_2w": wk_llhl_n,
            "weekly_macro_cycle_exit_signal": wk_exit,
        },
        index=weekly.index,
    ).reindex(idx, method="ffill")
    out = out.fillna(0.0)
    return out


@register_feature("compute_sma_position_from_series", category="baseline")
def compute_sma_position_from_series(
    *,
    close: pd.Series,
    sma_200: pd.Series,
) -> pd.DataFrame:
    """
    计算价格相对于 SMA 200 的归一化位置

    归一化方式: (close - sma_200) / close
    - 正值：价格在 SMA 上方（多头趋势）
    - 负值：价格在 SMA 下方（空头趋势）
    - 范围通常在 [-0.3, 0.3] 之间

    这个归一化方式使得不同价格水平的资产可以直接比较：
    - BTC: close=50000, sma_200=48000 → position = (50000-48000)/50000 = 0.04
    - ETH: close=3000, sma_200=2880 → position = (3000-2880)/3000 = 0.04

    两者都表示"价格比 SMA 高 4%"，可以直接比较。

    Returns:
        DataFrame with column: sma_200_position
    """
    close_clean = pd.to_numeric(close, errors="coerce").astype(float)
    sma_clean = pd.to_numeric(sma_200, errors="coerce").astype(float)

    # 避免除以零
    close_safe = close_clean.replace(0, np.nan)

    # (close - sma_200) / close
    position = (close_clean - sma_clean) / close_safe

    # 清理极端值
    position = position.replace([np.inf, -np.inf], np.nan)
    position = position.clip(-1.0, 1.0)  # 限制在 [-1, 1] 范围内
    position = position.fillna(0.0)

    return pd.DataFrame({"sma_200_position": position}, index=close.index)


@register_feature("compute_weekly_ema_position_from_ohlc", category="baseline")
def compute_weekly_ema_position_from_ohlc(
    *,
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    ema_span_weeks: int = 200,
    output_column: str = "weekly_ema_200_position",
    weekly_ema_context_dir: Optional[str] = None,
    weekly_ema_seed_path: Optional[str] = None,
    symbol: Optional[str] = None,
    _symbol: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """(current close - weekly EMA200) / current close on each base bar.

    Weekly EMA is computed on W-SUN aggregated closes (200-week span), then
    forward-filled to the base timeframe. Position uses the **live** base close
    at each bar, not the stale weekly close-only ratio.

    When ``weekly_ema_context_dir`` / ``weekly_ema_seed_path`` is set, load a
    long-history macro seed (Binance Vision spot daily) instead of inferring EMA
    from the short live bar buffer. Rows without a valid seeded EMA stay NaN
    (not 0.0) so missing macro history is not mistaken for "at EMA".
    """
    idx = close.index
    c = pd.to_numeric(close, errors="coerce").astype(float)

    # Resolve symbol from _symbol column when not explicitly provided
    if symbol is None and _symbol is not None:
        sym_vals = _symbol.dropna()
        if len(sym_vals) > 0:
            symbol = str(sym_vals.iloc[0])
    if not weekly_ema_seed_path and not weekly_ema_context_dir:
        weekly_ema_context_dir = (
            os.getenv("MLBOT_WEEKLY_EMA_SEED_ROOT", "").strip() or None
        )

    wk_ema_on_bar: Optional[pd.Series] = None
    if weekly_ema_seed_path or weekly_ema_context_dir:
        try:
            from pathlib import Path

            from src.live_data_stream.spot_weekly_ema_seed import (
                load_weekly_ema_seed,
                weekly_ema_position_series,
            )

            seed_df = None
            if weekly_ema_seed_path:
                p = Path(str(weekly_ema_seed_path))
                if p.is_file():
                    seed_df = pd.read_parquet(p)
                    if "week_ts" in seed_df.columns:
                        ts = pd.to_datetime(
                            seed_df["week_ts"], utc=True, errors="coerce"
                        )
                        seed_df = seed_df.set_index(ts).sort_index()
            elif weekly_ema_context_dir and symbol:
                seed_df = load_weekly_ema_seed(weekly_ema_context_dir, str(symbol))
            if (
                seed_df is not None
                and not seed_df.empty
                and "weekly_ema_200" in seed_df.columns
            ):
                ema = seed_df["weekly_ema_200"].dropna()
                out = weekly_ema_position_series(close=c, weekly_ema=ema)
                return pd.DataFrame({output_column: out.astype(float)}, index=idx)
        except Exception:
            pass

    # Auto-detect macro seed at default path (data/macro_seeds/{SYMBOL}.parquet)
    if symbol and not weekly_ema_seed_path and not weekly_ema_context_dir:
        try:
            from pathlib import Path

            from src.live_data_stream.spot_weekly_ema_seed import (
                seed_parquet_path,
                weekly_ema_position_series,
            )

            default_seed = seed_parquet_path(Path("data/macro_seeds"), str(symbol))
            if default_seed.is_file():
                seed_df = pd.read_parquet(default_seed)
                if "week_ts" in seed_df.columns:
                    ts = pd.to_datetime(seed_df["week_ts"], utc=True, errors="coerce")
                    seed_df = seed_df.set_index(ts).sort_index()
                ema_col = f"weekly_ema_{ema_span_weeks}"
                if not seed_df.empty and ema_col in seed_df.columns:
                    ema = seed_df[ema_col].dropna()
                    out = weekly_ema_position_series(close=c, weekly_ema=ema)
                    return pd.DataFrame({output_column: out.astype(float)}, index=idx)
                # Older seeds carry only weekly_ema_200; derive other spans from
                # the seed's own weekly_close with the builder's convention.
                if not seed_df.empty and "weekly_close" in seed_df.columns:
                    wc = pd.to_numeric(seed_df["weekly_close"], errors="coerce").dropna()
                    span = max(2, int(ema_span_weeks))
                    if len(wc) >= span:
                        ema = wc.ewm(span=span, adjust=False, min_periods=span).mean().dropna()
                        out = weekly_ema_position_series(close=c, weekly_ema=ema)
                        return pd.DataFrame({output_column: out.astype(float)}, index=idx)
        except Exception:
            pass

    weekly = (
        pd.DataFrame({"close": c}, index=idx)
        .sort_index()
        .resample("W-SUN", label="right", closed="right")
        .agg({"close": "last"})
    )
    weekly = weekly.dropna(subset=["close"])
    span = max(2, int(ema_span_weeks))
    if weekly.empty:
        out = pd.Series(np.nan, index=idx, dtype=float)
        return pd.DataFrame({output_column: out}, index=idx)
    wk_ema = weekly["close"].ewm(span=span, adjust=False, min_periods=span).mean()
    try:
        from src.live_data_stream.spot_weekly_ema_seed import weekly_ema_position_series

        out = weekly_ema_position_series(close=c, weekly_ema=wk_ema)
    except Exception:
        wk_ema_on_bar = wk_ema.reindex(idx, method="ffill")
        close_safe = c.replace(0, np.nan)
        out = ((c - wk_ema_on_bar) / close_safe).replace([np.inf, -np.inf], np.nan)
        out = out.where(wk_ema_on_bar.notna()).clip(-1.0, 1.0)
    return pd.DataFrame({output_column: out.astype(float)}, index=idx)


@register_feature("compute_price_vs_ma_position_from_series", category="baseline")
def compute_price_vs_ma_position_from_series(
    *,
    close: pd.Series,
    ma: pd.Series,
    output_column: str = "ma_position",
) -> pd.DataFrame:
    """(close - ma) / close, clipped to [-1, 1]. ``ma`` 由 column_mappings 绑定 (如 ema_1200)."""
    close_clean = pd.to_numeric(close, errors="coerce").astype(float)
    ma_clean = pd.to_numeric(ma, errors="coerce").astype(float)
    close_safe = close_clean.replace(0, np.nan)
    position = (close_clean - ma_clean) / close_safe
    position = position.replace([np.inf, -np.inf], np.nan).clip(-1.0, 1.0).fillna(0.0)
    return pd.DataFrame({output_column: position}, index=close.index)


@register_feature("compute_expanding_high_drawdown", category="baseline")
def compute_expanding_high_drawdown(
    *,
    close: pd.Series,
    high: pd.Series,
    output_column: str = "expanding_high_drawdown",
) -> pd.DataFrame:
    """Point-in-time close drawdown from the highest observed high, in [-1, 0]."""
    close_clean = pd.to_numeric(close, errors="coerce").astype(float)
    high_clean = pd.to_numeric(high, errors="coerce").astype(float)
    peak = high_clean.expanding(min_periods=1).max().replace(0.0, np.nan)
    drawdown = (close_clean / peak - 1.0).replace([np.inf, -np.inf], np.nan)
    drawdown = drawdown.clip(-1.0, 0.0)
    return pd.DataFrame({output_column: drawdown}, index=close.index)


@register_feature("compute_dmi_direction", category="baseline")
def compute_dmi_direction(
    *,
    plus_di: pd.Series,
    minus_di: pd.Series,
    output_column: str = "dmi_dir",
) -> pd.DataFrame:
    """Normalized DMI direction: (plus_di - minus_di) / (plus_di + minus_di).

    Bounded [-1, 1] by construction:
      +1 = purely bullish (+DI dominates)
      -1 = purely bearish (-DI dominates)
       0 = balanced / no clear direction

    Cross-symbol comparable — both numerator and denominator scale with price.
    """
    p = pd.to_numeric(plus_di, errors="coerce").astype(float)
    m = pd.to_numeric(minus_di, errors="coerce").astype(float)
    denom = p + m
    denom_safe = denom.replace(0, np.nan)
    direction = ((p - m) / denom_safe).replace([np.inf, -np.inf], np.nan)
    # When denom is 0, both DIs are 0 → no direction
    direction = direction.fillna(0.0).clip(-1.0, 1.0)
    return pd.DataFrame({output_column: direction.astype(float)}, index=plus_di.index)


@register_feature("compute_vwap_position_from_series", category="baseline")
def compute_vwap_position_from_series(
    *,
    close: pd.Series,
    volume: pd.Series,
    window: int = 20,
) -> pd.DataFrame:
    """
    计算价格相对于 VWAP 的归一化位置（跨品种可比）

    - price_to_vwap_pct = (close - vwap) / close
    - price_to_vwap_ratio = close / vwap

    Returns:
        DataFrame with columns: price_to_vwap_pct, price_to_vwap_ratio
    """
    close_clean = pd.to_numeric(close, errors="coerce").astype(float)
    vol_clean = pd.to_numeric(volume, errors="coerce").astype(float)

    vol_roll = vol_clean.rolling(window=window, min_periods=1).sum()
    vwap = (close_clean * vol_clean).rolling(
        window=window, min_periods=1
    ).sum() / vol_roll

    close_safe = close_clean.replace(0, np.nan)
    vwap_safe = vwap.replace(0, np.nan)

    price_to_vwap_pct = (close_clean - vwap) / close_safe
    price_to_vwap_ratio = close_clean / vwap_safe

    price_to_vwap_pct = (
        price_to_vwap_pct.replace([np.inf, -np.inf], np.nan).clip(-1.0, 1.0).fillna(0.0)
    )
    price_to_vwap_ratio = (
        price_to_vwap_ratio.replace([np.inf, -np.inf], np.nan)
        .clip(0.2, 5.0)
        .fillna(1.0)
    )

    return pd.DataFrame(
        {
            "price_to_vwap_pct": price_to_vwap_pct,
            "price_to_vwap_ratio": price_to_vwap_ratio,
        },
        index=close.index,
    )


@register_feature(
    "compute_typical_price_vwap_position_from_series", category="baseline"
)
def compute_typical_price_vwap_position_from_series(
    *,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    window: int = 1200,
    output_column: str = "macro_tp_vwap_1200_position",
) -> pd.DataFrame:
    """
    滚动典型价 VWAP: sum(tp*vol)/sum(vol), tp=(H+L+C)/3；
    输出 (close - vwap) / close，与 SMA 位置可比、便于 direction 用 sign。
    """
    h = pd.to_numeric(high, errors="coerce").astype(float)
    l = pd.to_numeric(low, errors="coerce").astype(float)
    c = pd.to_numeric(close, errors="coerce").astype(float)
    vol = pd.to_numeric(volume, errors="coerce").astype(float).clip(lower=0.0)
    tp = (h + l + c) / 3.0
    n = int(len(c))
    w = max(2, min(int(window), max(n, 2)))
    min_p = max(3, min(w // 10, max(50, w // 20)))
    num = (tp * vol).rolling(window=w, min_periods=min_p).sum()
    den = vol.rolling(window=w, min_periods=min_p).sum()
    vwap = num / den.replace(0, np.nan)
    close_safe = c.replace(0, np.nan)
    position = (c - vwap) / close_safe
    position = position.replace([np.inf, -np.inf], np.nan).clip(-1.0, 1.0).fillna(0.0)
    return pd.DataFrame({output_column: position}, index=close.index)


@register_feature("compute_volume_ratio_from_series", category="baseline")
def compute_volume_ratio_from_series(
    *,
    volume: pd.Series,
    window: int = 20,
) -> pd.DataFrame:
    """
    计算成交量相对于均值的归一化比率

    归一化方式: volume / rolling_mean_volume
    - 值 = 1.0：成交量等于均值
    - 值 > 1.0：成交量高于均值（放量）
    - 值 < 1.0：成交量低于均值（缩量）

    这个归一化方式使得不同资产的成交量可以直接比较。

    Returns:
        DataFrame with column: volume_ratio
    """
    vol = pd.to_numeric(volume, errors="coerce").astype(float)

    # 滚动均值（因果性：只使用历史数据）
    rolling_mean = vol.rolling(window=window, min_periods=1).mean()

    # 避免除以零
    rolling_mean_safe = rolling_mean.replace(0, np.nan)

    # volume / rolling_mean
    ratio = vol / rolling_mean_safe

    # 清理极端值
    ratio = ratio.replace([np.inf, -np.inf], np.nan)
    ratio = ratio.clip(0.0, 10.0)  # 限制在 [0, 10] 范围内
    ratio = ratio.fillna(1.0)  # 默认为 1.0（等于均值）

    return pd.DataFrame({"volume_ratio": ratio}, index=volume.index)


@register_feature("compute_zigzag_high_low_from_series", category="baseline")
def compute_zigzag_high_low_from_series(
    *,
    high: pd.Series,
    low: pd.Series,
    threshold: float = 0.05,
    price_col: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    Narrow-IO ZigZag computation that returns zigzag, zz_high_value, and zz_low_value.

    Args:
        high: 最高价序列
        low: 最低价序列
        threshold: 转折阈值（默认 0.05，即 5%）
        price_col: 可选的价格序列（如 WPT 中高频重构价格）

    Returns:
        DataFrame with columns: zigzag, zz_high_value, zz_low_value
    """
    zigzag, zz_high, zz_low = compute_zigzag(
        high=high,
        low=low,
        threshold=threshold,
        return_high_low=True,
        price_col=price_col,
    )
    result = pd.DataFrame(
        {
            "zigzag": zigzag,
            "zz_high_value": zz_high,
            "zz_low_value": zz_low,
        },
        index=high.index,
    )
    return result


@register_feature("compute_ofi_short_from_series", category="baseline")
def compute_ofi_short_from_series(
    *, vpin_signed_imbalance: pd.Series, vpin: pd.Series, window: int = 5
) -> pd.Series:
    """
    计算短期订单流不平衡（Order Flow Imbalance Short）

    基于 vpin_signed_imbalance 的短期移动平均或归一化版本
    """
    vpin_imb = pd.to_numeric(vpin_signed_imbalance, errors="coerce").astype(float)
    vpin_val = pd.to_numeric(vpin, errors="coerce").astype(float)

    # 方法1：短期移动平均（捕捉短期趋势）
    ofi_short_ma = vpin_imb.rolling(window=window, min_periods=1).mean()

    # 方法2：归一化版本（相对于 VPIN 水平）
    vpin_safe = vpin_val.replace(0, np.nan)
    ofi_short_norm = (vpin_imb / vpin_safe).replace([np.inf, -np.inf], np.nan)

    # 组合：使用移动平均，但归一化到合理范围
    # 或者直接使用移动平均
    out = ofi_short_ma.fillna(0.0)
    out.name = "ofi_short"
    return out


@register_feature(
    "compute_compression_to_breakout_prob_from_series", category="baseline"
)
def compute_compression_to_breakout_prob_from_series(
    *, compression_duration: pd.Series, roc_5: pd.Series
) -> pd.DataFrame:
    """
    Narrow-IO compression_duration × momentum proxy, mapped to a bounded [0,1] score.

    - compression_duration: already normalized to [0,1]
    - roc_5: rolling z-score signal -> squash with sigmoid to [0,1]
    """
    cd = (
        pd.to_numeric(compression_duration, errors="coerce")
        .astype(float)
        .fillna(0.0)
        .clip(0.0, 1.0)
    )
    r = pd.to_numeric(roc_5, errors="coerce").astype(float).fillna(0.0)
    mom = (1.0 / (1.0 + np.exp(-r))).clip(0.0, 1.0)
    out = (cd * mom).rename("compression_to_breakout_prob")
    return out.to_frame()


@register_feature("compute_range_ratio_5bar_from_series", category="baseline")
def compute_range_ratio_5bar_from_series(
    *, high: pd.Series, low: pd.Series
) -> pd.Series:
    """5/20 Bar range ratio z-score (narrow input/output)."""
    high = pd.to_numeric(high, errors="coerce").astype(float)
    low = pd.to_numeric(low, errors="coerce").astype(float)
    hl = high - low
    short_range = hl.rolling(5).mean()
    long_range = hl.rolling(20).mean()
    ratio = (short_range / long_range.replace(0, np.nan)).replace(
        [np.inf, -np.inf], np.nan
    )
    ratio = ratio.fillna(1.0)
    ratio_log = np.log1p(ratio)
    mean = ratio_log.rolling(50, min_periods=5).mean()
    std = ratio_log.rolling(50, min_periods=5).std()
    out = (
        ((ratio_log - mean) / std.replace(0, np.nan))
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )
    out.name = "range_ratio_5bar"
    return out


@register_feature("compute_price_range_symmetry_from_series", category="baseline")
def compute_price_range_symmetry_from_series(
    *,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    feature_shift: int = 0,
) -> pd.Series:
    """Upper/lower shadow asymmetry score (narrow input/output)."""
    high = pd.to_numeric(high, errors="coerce").astype(float).shift(feature_shift)
    low = pd.to_numeric(low, errors="coerce").astype(float).shift(feature_shift)
    close = pd.to_numeric(close, errors="coerce").astype(float).shift(feature_shift)
    numerator = high - close
    denominator = (close - low).replace(0, np.nan)
    raw = (numerator / denominator).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    log_val = np.log1p(np.abs(raw)) * np.sign(raw)
    mean = log_val.rolling(50, min_periods=5).mean()
    std = log_val.rolling(50, min_periods=5).std()
    out = (
        ((log_val - mean) / std.replace(0, np.nan))
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )
    out.name = "price_range_symmetry"
    return out


@register_feature("compute_wick_ratios_from_series", category="baseline")
def compute_wick_ratios_from_series(
    *,
    open: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
) -> pd.DataFrame:
    """Upper/lower wick ratios (narrow input/output)."""
    open = pd.to_numeric(open, errors="coerce").astype(float)
    high = pd.to_numeric(high, errors="coerce").astype(float)
    low = pd.to_numeric(low, errors="coerce").astype(float)
    close = pd.to_numeric(close, errors="coerce").astype(float)
    range_val = high - low
    body_high = pd.concat([close, open], axis=1).max(axis=1)
    upper_wick = high - body_high
    wick_upper_ratio = (
        (upper_wick / range_val.replace(0, np.nan))
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )
    body_low = pd.concat([close, open], axis=1).min(axis=1)
    lower_wick = body_low - low
    wick_lower_ratio = (
        (lower_wick / range_val.replace(0, np.nan))
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )
    return pd.DataFrame(
        {"wick_upper_ratio": wick_upper_ratio, "wick_lower_ratio": wick_lower_ratio}
    )


@register_feature("compute_poc_hal_features_from_series", category="baseline")
def compute_poc_hal_features_from_series(
    *,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    wpt_price_reconstructed: pd.Series | None = None,
    poc_window: int = 160,
    atr_period: int = 14,
    price_col: str = "wpt_price_reconstructed",
) -> pd.DataFrame:
    """
    Narrow-IO POC/HAL computation with normalized outputs.

    Returns normalized features (distance from close / ATR):
    - poc: (poc_raw - close) / ATR, typical range [-3, 3]
    - hal_high: (hal_high_raw - close) / ATR
    - hal_low: (hal_low_raw - close) / ATR
    - hal_mid: (hal_mid_raw - close) / ATR

    These normalized values represent "how many ATRs away" the SR level is,
    making them cross-asset comparable.
    """
    from src.features.live_tail_context import get_live_feature_tail

    _full_index = close.index
    _tail = get_live_feature_tail()
    if _tail is not None and len(close) > _tail:
        _warmup = int(poc_window) + int(atr_period) + 1
        _slice_len = min(len(close), _tail + _warmup)
        high = high.iloc[-_slice_len:]
        low = low.iloc[-_slice_len:]
        close = close.iloc[-_slice_len:]
        volume = volume.iloc[-_slice_len:]
        if wpt_price_reconstructed is not None:
            wpt_price_reconstructed = wpt_price_reconstructed.iloc[-_slice_len:]
    else:
        _tail = None

    high_s = pd.to_numeric(high, errors="coerce").astype(float)
    low_s = pd.to_numeric(low, errors="coerce").astype(float)
    close_s = pd.to_numeric(close, errors="coerce").astype(float)
    volume_s = pd.to_numeric(volume, errors="coerce").astype(float)

    df = pd.DataFrame(
        {
            "high": high_s,
            "low": low_s,
            "close": close_s,
            "volume": volume_s,
        }
    )

    if price_col and price_col not in df.columns:
        if (
            price_col == "wpt_price_reconstructed"
            and wpt_price_reconstructed is not None
        ):
            df[price_col] = pd.to_numeric(
                wpt_price_reconstructed, errors="coerce"
            ).astype(float)
        else:
            # Fallback: use close as price reference if a custom price_col wasn't provided
            df[price_col] = df["close"]

    out = add_poc_hal_dimensionless_features(
        df,
        required_features={"poc", "hal_high", "hal_low", "hal_mid"},
        poc_window=poc_window,
        price_col=price_col,
    )

    # Compute ATR for normalization
    atr = compute_atr(high_s, low_s, close_s, period=atr_period)
    atr_safe = atr.replace(0, np.nan).fillna(1e-8)

    # Normalize: (level - close) / ATR
    eps = 1e-8
    poc_norm = (
        ((out["poc"] - close_s) / (atr_safe + eps))
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )
    hal_high_norm = (
        ((out["hal_high"] - close_s) / (atr_safe + eps))
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )
    hal_low_norm = (
        ((out["hal_low"] - close_s) / (atr_safe + eps))
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )
    hal_mid_norm = (
        ((out["hal_mid"] - close_s) / (atr_safe + eps))
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )

    result = pd.DataFrame(
        {
            "poc": poc_norm,
            "hal_high": hal_high_norm,
            "hal_low": hal_low_norm,
            "hal_mid": hal_mid_norm,
        }
    )
    if _tail is not None:
        result = result.reindex(_full_index, fill_value=0.0)
    else:
        result = result.reindex(index=df.index)
    return result


@register_feature("compute_sqs_from_sr_price_series", category="baseline")
def compute_sqs_from_sr_price_series(
    *,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    atr: pd.Series,
    sr_price: pd.Series,
    window: int = 60,
    tolerance_factor: float = 0.5,
    sr_type: str = "resistance",
    output_name: str = "sqs",
) -> pd.Series:
    """
    Generic narrow-IO SQS computation for a dynamic SR price series (e.g., hal_high/hal_low).
    Uses only historical data up to i (no look-ahead).
    """
    high = pd.to_numeric(high, errors="coerce").astype(float)
    low = pd.to_numeric(low, errors="coerce").astype(float)
    close = pd.to_numeric(close, errors="coerce").astype(float)
    volume = pd.to_numeric(volume, errors="coerce").astype(float)
    atr = pd.to_numeric(atr, errors="coerce").astype(float)
    sr_price = pd.to_numeric(sr_price, errors="coerce").astype(float)

    out = np.zeros(len(close), dtype=float)
    for i in range(len(close)):
        if i < window:
            continue
        p = sr_price.iloc[i]
        if pd.isna(p) or p <= 0:
            continue
        start_idx = max(0, i - window + 1)
        hist_df = pd.DataFrame(
            {
                "high": high.iloc[start_idx : i + 1],
                "low": low.iloc[start_idx : i + 1],
                "close": close.iloc[start_idx : i + 1],
                "atr": atr.iloc[start_idx : i + 1],
                "volume": volume.iloc[start_idx : i + 1],
            }
        )
        try:
            sqs = calculate_sqs(
                sr_price=p,
                df=hist_df,
                window=min(window, len(hist_df)),
                tolerance_factor=tolerance_factor,
                sr_type=sr_type,
            )
            if not np.isnan(sqs):
                out[i] = float(sqs)
        except Exception:
            # Conservative fallback: keep 0
            pass

    s = pd.Series(out, index=close.index, name=output_name)
    return s


@register_feature("compute_sqs_hal_high_from_series", category="baseline")
def compute_sqs_hal_high_from_series(
    *,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    atr: pd.Series,
    hal_high: pd.Series,
    window: int = 60,
    tolerance_factor: float = 0.5,
    sr_type: str = "resistance",
) -> pd.DataFrame:
    # `poc_hal_features_close_f` provides HAL levels normalized as (level - close) / ATR.
    # `calculate_sqs` expects SR price in *raw price units* (to compare vs high/low).
    # Convert back: level_raw = hal_high_norm * atr + close.
    close_s = pd.to_numeric(close, errors="coerce").astype(float)
    atr_s = pd.to_numeric(atr, errors="coerce").astype(float)
    hal_high_norm = pd.to_numeric(hal_high, errors="coerce").astype(float)
    hal_high_raw = (hal_high_norm * atr_s + close_s).replace([np.inf, -np.inf], np.nan)
    sqs = compute_sqs_from_sr_price_series(
        high=high,
        low=low,
        close=close,
        volume=volume,
        atr=atr,
        sr_price=hal_high_raw,
        window=window,
        tolerance_factor=tolerance_factor,
        sr_type=sr_type,
        output_name="sqs_hal_high",
    )
    return pd.DataFrame({"sqs_hal_high": sqs})


@register_feature("compute_sqs_hal_low_from_series", category="baseline")
def compute_sqs_hal_low_from_series(
    *,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    atr: pd.Series,
    hal_low: pd.Series,
    window: int = 60,
    tolerance_factor: float = 0.5,
    sr_type: str = "support",
) -> pd.DataFrame:
    # `poc_hal_features_close_f` provides HAL levels normalized as (level - close) / ATR.
    # Convert back to raw SR price levels for SQS scoring.
    close_s = pd.to_numeric(close, errors="coerce").astype(float)
    atr_s = pd.to_numeric(atr, errors="coerce").astype(float)
    hal_low_norm = pd.to_numeric(hal_low, errors="coerce").astype(float)
    hal_low_raw = (hal_low_norm * atr_s + close_s).replace([np.inf, -np.inf], np.nan)
    sqs = compute_sqs_from_sr_price_series(
        high=high,
        low=low,
        close=close,
        volume=volume,
        atr=atr,
        sr_price=hal_low_raw,
        window=window,
        tolerance_factor=tolerance_factor,
        sr_type=sr_type,
        output_name="sqs_hal_low",
    )
    return pd.DataFrame({"sqs_hal_low": sqs})


@register_feature("compute_sr_strength_max_from_series", category="baseline")
def compute_sr_strength_max_from_series(
    *,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    atr: pd.Series,
    poc: pd.Series,
    hal_high: pd.Series,
    hal_low: pd.Series,
    window: int = 60,
    tolerance_factor: float = 0.5,
) -> pd.DataFrame:
    """
    Narrow-IO SR strength computation (returns only declared outputs).

    Returns normalized features:
    - sr_strength_max: already [0, 1] bounded (SQS strength)
    - dist_to_nearest_sr: normalized by ATR (how many ATRs away), typical [-3, 3]
    - direction_to_nearest_sr: -1 (below) or +1 (above), already normalized

    Note: poc, hal_high, hal_low inputs are expected to be normalized (from close / ATR).
    We need to "un-normalize" them to compute distances, then normalize the output.

    Live bounded-tail: when a live tail context is active (see
    ``src.features.live_tail_context``), only the last ``tail`` bars are computed
    and the outputs are reindexed back to the full index (older rows filled 0.0).
    This is bit-identical for the last output row (and the last
    ``tail - internal_window`` rows), which is all the live snapshot consumes.
    Backtest / FeatureStore builds never activate the context → full history.
    """
    from src.features.live_tail_context import get_live_feature_tail

    _full_index = close.index
    _tail = get_live_feature_tail()
    if _tail is not None and len(close) > _tail:
        high = high.iloc[-_tail:]
        low = low.iloc[-_tail:]
        close = close.iloc[-_tail:]
        atr = atr.iloc[-_tail:]
        poc = poc.iloc[-_tail:]
        hal_high = hal_high.iloc[-_tail:]
        hal_low = hal_low.iloc[-_tail:]
    else:
        _tail = None

    high_s = pd.to_numeric(high, errors="coerce").astype(float)
    low_s = pd.to_numeric(low, errors="coerce").astype(float)
    close_s = pd.to_numeric(close, errors="coerce").astype(float)
    atr_s = pd.to_numeric(atr, errors="coerce").astype(float)

    # Note: poc, hal_high, hal_low are now normalized inputs (distance / ATR)
    # We need to convert them back to price levels for internal computation
    # normalized = (price - close) / atr => price = normalized * atr + close
    poc_raw = pd.to_numeric(poc, errors="coerce").astype(float) * atr_s + close_s
    hal_high_raw = (
        pd.to_numeric(hal_high, errors="coerce").astype(float) * atr_s + close_s
    )
    hal_low_raw = (
        pd.to_numeric(hal_low, errors="coerce").astype(float) * atr_s + close_s
    )

    data = pd.DataFrame(
        {
            "high": high_s,
            "low": low_s,
            "close": close_s,
            "atr": atr_s,
            "poc": poc_raw,
            "hal_high": hal_high_raw,
            "hal_low": hal_low_raw,
            # seed columns so _compute_boundary_strengths also produces them
            "dist_to_nearest_sr": 0.0,
            "direction_to_nearest_sr": 0.0,
        },
        index=close.index,
    )

    boundaries = _get_sr_boundary_definitions(data)
    if not boundaries:
        empty = pd.DataFrame(
            {
                "sr_strength_max": pd.Series(0.0, index=data.index),
                "dist_to_nearest_sr": pd.Series(0.0, index=data.index),
                "direction_to_nearest_sr": pd.Series(0.0, index=data.index),
            }
        )
        if _tail is not None:
            empty = empty.reindex(_full_index, fill_value=0.0)
        return empty

    strengths = _compute_boundary_strengths(
        data=data,
        boundaries=boundaries,
        window=window,
        tolerance_factor=tolerance_factor,
        compression_series=None,
    )

    # compute max across returned strength components without building a big wide df
    sr_max = np.zeros(len(data), dtype=float)
    for s in strengths.values():
        if isinstance(s, pd.Series):
            v = pd.to_numeric(s, errors="coerce").fillna(0.0).values
            sr_max = np.maximum(sr_max, v)

    # 【关键修复】：调用 _add_price_action_features 来更新 dist_to_nearest_sr
    # 因为 _compute_boundary_strengths 只计算 SQS 强度，不计算距离
    # 确保必要的列存在（narrow-IO 接口可能没有 open 列）
    if "open" not in data.columns:
        data["open"] = data["close"]  # 使用 close 作为默认值
    if "volume" not in data.columns:
        data["volume"] = pd.Series(0.0, index=data.index)  # 使用默认值

    data = _add_price_action_features(data, boundaries, compression_series=None)

    # Normalize dist_to_nearest_sr by ATR
    # dist_to_nearest_sr is a percentage (dist / close), convert to ATR multiples
    dist_raw = (
        pd.to_numeric(data["dist_to_nearest_sr"], errors="coerce")
        .fillna(0.0)
        .astype(float)
    )
    # dist_raw is (sr_price - close) / close = pct distance
    # We want: (sr_price - close) / atr = dist_raw * close / atr
    atr_safe = atr_s.replace(0, np.nan).fillna(1e-8)
    dist_norm = (
        (dist_raw * close_s / atr_safe).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    )

    out = pd.DataFrame(index=data.index)
    out["sr_strength_max"] = pd.Series(sr_max, index=data.index).astype(float)
    out["dist_to_nearest_sr"] = dist_norm
    out["direction_to_nearest_sr"] = (
        pd.to_numeric(data["direction_to_nearest_sr"], errors="coerce")
        .fillna(0.0)
        .astype(float)
    )
    if _tail is not None:
        out = out.reindex(_full_index, fill_value=0.0)
    return out


@register_feature("compute_wide_sr_swing_from_series", category="baseline")
def compute_wide_sr_swing_from_series(
    *,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    atr: pd.Series,
    wide_window: int = 240,
    anchor_shift: int = 12,
) -> pd.DataFrame:
    """大级别 SR 代理（L3，全仓库唯一实现）：滚动 swing high / low。

    输入：
        wide_window: 回看 bar 数（默认 240，在 2H 主周期上 ≈ 1 个月）。
        anchor_shift: 向前平移 bar 数（默认 12，≈ 1 天），避免极值被"当下 bar"自含。

    与其它 SR 层的互补关系：
        L1 局部 SR  — `srb_regime.swing_sr_levels(20)`             ≈ 1-2 日
        L2 中期 SR  — `poc_hal_features_*_f` (poc_window=160)      ≈ 2 周
        L3 大级别 SR — 本特征 (wide_window=240, shift=12)           ≈ 1 个月

    输出列：
        wide_sr_upper_px        : 上沿价格（rolling max of high, shifted）
        wide_sr_lower_px        : 下沿价格（rolling min of low,  shifted）
        wide_sr_dist_atr        : min(|close - upper|, |close - lower|) / ATR
        wide_sr_side            : +1 更近上沿，-1 更近下沿
        wide_sr_range_width_atr : (upper - lower) / ATR
    """
    high_s = pd.to_numeric(high, errors="coerce").astype(float)
    low_s = pd.to_numeric(low, errors="coerce").astype(float)
    close_s = pd.to_numeric(close, errors="coerce").astype(float)
    atr_s = pd.to_numeric(atr, errors="coerce").astype(float).replace(0.0, np.nan)

    min_periods = max(wide_window // 2, 10)
    roll_high = (
        high_s.rolling(wide_window, min_periods=min_periods).max().shift(anchor_shift)
    )
    roll_low = (
        low_s.rolling(wide_window, min_periods=min_periods).min().shift(anchor_shift)
    )

    d_high = (close_s - roll_high).abs() / atr_s
    d_low = (close_s - roll_low).abs() / atr_s
    dist = np.minimum(d_high, d_low).replace([np.inf, -np.inf], np.nan)
    side = np.where(d_high <= d_low, 1.0, -1.0)
    width = (roll_high - roll_low) / atr_s

    return pd.DataFrame(
        {
            "wide_sr_upper_px": roll_high.astype(float),
            "wide_sr_lower_px": roll_low.astype(float),
            "wide_sr_dist_atr": dist.astype(float),
            "wide_sr_side": pd.Series(side, index=close.index).astype(float),
            "wide_sr_range_width_atr": width.replace([np.inf, -np.inf], np.nan).astype(
                float
            ),
        },
        index=close.index,
    )


# ============================================================================
# P5 非平稳性: Regime State / OOD Score
# ============================================================================


@register_feature("compute_regime_state_from_df", category="baseline")
def compute_regime_state_from_df(
    df: pd.DataFrame,
    **kwargs,
) -> pd.DataFrame:
    """Regime state as a feature — replicates RegimeDetector logic from live_pcm.py.

    Detection order (strict → loose):
      2 = HIGH_LEVERAGE : oi_zscore > 1.5  AND  funding_rate_abs_zscore > 2.0
      1 = HIGH_VOL      : atr_percentile > 0.7
      0 = NORMAL         : default

    If required columns are missing, defaults to 0 (NORMAL).

    Returns:
        DataFrame with column 'regime_state' (int: 0/1/2).
    """
    n = len(df)
    regime = np.zeros(n, dtype=np.int8)

    has_atr_pct = "atr_percentile" in df.columns
    has_oi_z = "oi_zscore" in df.columns
    has_fr_z = "funding_rate_abs_zscore_50" in df.columns

    if has_atr_pct:
        high_vol_mask = df["atr_percentile"].values > 0.7
        regime[high_vol_mask] = 1

    if has_oi_z and has_fr_z:
        high_lev_mask = (df["oi_zscore"].values > 1.5) & (
            df["funding_rate_abs_zscore_50"].values > 2.0
        )
        # HIGH_LEVERAGE overrides HIGH_VOL (stricter condition)
        regime[high_lev_mask] = 2

    return pd.DataFrame({"regime_state": regime}, index=df.index)


@register_feature("compute_ood_score_from_df", category="baseline")
def compute_ood_score_from_df(
    df: pd.DataFrame,
    *,
    baseline: Optional[Dict[str, Dict[str, float]]] = None,
    baseline_path: Optional[str] = None,
    **kwargs,
) -> pd.DataFrame:
    """Out-of-distribution score — fraction of features outside training [q05, q95].

    For each row, count how many features fall outside their training-period
    5th / 95th percentile range, then normalise by total checked features.
    Result is in [0, 1]; higher = more OOD.

    Args:
        df: full feature DataFrame
        baseline: {feat_name: {"q05": float, "q95": float, ...}}
                  If None, tries to load from *baseline_path*.
        baseline_path: path to feature_baseline.json (training_baseline.json)

    Returns:
        DataFrame with column 'ood_score' in [0.0, 1.0].
    """
    import json as _json

    # Resolve baseline dict
    if baseline is None and baseline_path:
        try:
            _p = (
                Path(baseline_path)
                if not isinstance(baseline_path, Path)
                else baseline_path
            )
            _raw = _json.loads(_p.read_text(encoding="utf-8"))
            baseline = _raw.get("feature_distributions", _raw)
        except Exception:
            baseline = None

    n = len(df)
    if not baseline:
        return pd.DataFrame({"ood_score": np.full(n, 0.0)}, index=df.index)

    # Vectorised: iterate features, accumulate OOD count per row
    # Use per-row denomination: NaN features are excluded from that row's denominator
    ood_counts = np.zeros(n, dtype=np.float64)
    n_valid_per_row = np.zeros(n, dtype=np.float64)
    for feat, stats in baseline.items():
        if feat not in df.columns:
            continue
        q05 = stats.get("q05") if stats.get("q05") is not None else stats.get("p5")
        q95 = stats.get("q95") if stats.get("q95") is not None else stats.get("p95")
        if q05 is None or q95 is None:
            continue
        vals = df[feat].values.astype(np.float64)
        nan_mask = np.isnan(vals)
        ood_mask = (vals < float(q05)) | (vals > float(q95))
        # NaN → not counted as OOD, and excluded from denominator
        ood_mask[nan_mask] = False
        ood_counts += ood_mask.astype(np.float64)
        n_valid_per_row += (~nan_mask).astype(np.float64)

    ood_score = ood_counts / np.maximum(n_valid_per_row, 1.0)
    return pd.DataFrame({"ood_score": ood_score.clip(0.0, 1.0)}, index=df.index)


@register_feature("compute_ema_vwap_1200_cross_from_positions", category="baseline")
def compute_ema_vwap_1200_cross_from_positions(
    *,
    close: pd.Series,
    ema_1200_position: pd.Series,
    macro_tp_vwap_1200_position: pd.Series,
    output_column_up: str = "ema_vwap_1200_cross_up",
    output_column_down: str = "ema_vwap_1200_cross_down",
) -> pd.DataFrame:
    """EMA1200 × VWAP1200 true cross flags from position columns.

    Reconstructs levels: ``ma = close * (1 - position)`` where
    ``position = (close - ma) / close``. Cross-up matches rolling champion
    entry (``rolling_decision.check_entry_signal``): prev ema<=vwap and
    cur ema>vwap. Cross-down is the symmetric death cross.
    """
    c = pd.to_numeric(close, errors="coerce").astype(float)
    ep = pd.to_numeric(ema_1200_position, errors="coerce").astype(float)
    vp = pd.to_numeric(macro_tp_vwap_1200_position, errors="coerce").astype(float)
    ema = c * (1.0 - ep)
    vwap = c * (1.0 - vp)
    prev_ema = ema.shift(1)
    prev_vwap = vwap.shift(1)
    cross_up = ((prev_ema <= prev_vwap) & (ema > vwap)).astype(float)
    cross_down = ((prev_ema >= prev_vwap) & (ema < vwap)).astype(float)
    # NaN inputs → 0 (no cross), not NaN (avoid blocking entry rules)
    cross_up = cross_up.where(ep.notna() & vp.notna() & c.notna(), 0.0)
    cross_down = cross_down.where(ep.notna() & vp.notna() & c.notna(), 0.0)
    return pd.DataFrame(
        {output_column_up: cross_up, output_column_down: cross_down},
        index=c.index,
    )


@register_feature("compute_ema_50_200_cross_from_series", category="baseline")
def compute_ema_50_200_cross_from_series(
    *,
    close: pd.Series,
    ema_50_position: pd.Series,
    ema_200: pd.Series,
    output_column_up: str = "ema_50_200_cross_up",
    output_column_down: str = "ema_50_200_cross_down",
    output_column_side: str = "ema_50_200_cross_side",
) -> pd.DataFrame:
    """Textbook EMA50 × EMA200 golden/death cross on the signal bar.

    Reconstructs EMA50 from ``ema_50_position = (close - ema50) / close``.
    ``ema_200`` is already in price units (``ema_200_value_f``).
    Cross-up: previous fast<=slow and current fast>slow.
    This is not ``ema_vwap_1200_cross`` and not ``sign(ema_1200_position)``.
    """
    c = pd.to_numeric(close, errors="coerce").astype(float)
    pos = pd.to_numeric(ema_50_position, errors="coerce").astype(float)
    slow = pd.to_numeric(ema_200, errors="coerce").astype(float)
    fast = c * (1.0 - pos)
    prev_fast = fast.shift(1)
    prev_slow = slow.shift(1)
    cross_up = ((prev_fast <= prev_slow) & (fast > slow)).astype(float)
    cross_down = ((prev_fast >= prev_slow) & (fast < slow)).astype(float)
    valid = pos.notna() & slow.notna() & c.notna()
    cross_up = cross_up.where(valid, 0.0)
    cross_down = cross_down.where(valid, 0.0)
    side = (cross_up - cross_down).astype(float)
    return pd.DataFrame(
        {
            output_column_up: cross_up,
            output_column_down: cross_down,
            output_column_side: side,
        },
        index=c.index,
    )


# ============================================================================
# 导出（保持向后兼容）
# ============================================================================

__all__ = [
    # 便捷函数
    "get_baseline_feature_columns",
    "create_binary_labels_baseline",
    # 基础指标计算函数
    "compute_rsi",
    "compute_macd",
    "compute_bollinger_bands",
    "compute_atr",
    "compute_zigzag",
    # 特征添加函数
    "add_basic_indicators",
    "ensure_basic_indicators",
    "add_zigzag_dimensionless_features",
    "add_poc_hal_dimensionless_features",
    "add_swing_dimensionless_features",
    "add_ols_channel_features",
    "add_price_volume_relative_features",
    "add_common_derived_features",
    "compute_regime_state_from_df",
    "compute_ood_score_from_df",
]
