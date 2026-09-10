"""
HTFBiasLTFEntry Archetype 专用特征模块

设计理念：
- HTF (高时间框架) 提供方向偏见（大趋势方向）
- LTF (低时间框架) 提供入场时机（精细入场点）
- 核心：HTF/LTF 一致性验证 + 结构对齐

核心输出：
1. htf_score_bias: HTF 方向偏见强度 [0-1]
2. ltf_score_entry: LTF 入场质量 [0-1]
3. htf_ltf_alignment: HTF/LTF 对齐度 [0-1]

规范遵循：
- 特征工程鲁棒性设计规范
- Archetype 语义化特征建模规范
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional

from src.features.registry import register_feature


# =============================================================================
# 📌 常量定义
# =============================================================================

# HTF 趋势参数
DEFAULT_HTF_LOOKBACK = 50         # HTF 趋势检测窗口
DEFAULT_LTF_LOOKBACK = 10         # LTF 入场检测窗口
HTF_CONSISTENCY_THRESHOLD = 0.6   # HTF 方向一致性阈值
LTF_ENTRY_VOL_MULT = 1.2          # LTF 入场放量倍数

# 特征版本
FEATURE_VERSION = "1.0"


# =============================================================================
# 🎯 主函数：HTF Bias + LTF Entry 软分数
# =============================================================================

@register_feature(
    "compute_htf_ltf_soft_phase_from_series",
    category="htf_ltf",
    description="HTFBiasLTFEntry soft phase scores",
    outputs=[
        # === ATOMIC: HTF Bias 原子信号 ===
        "htf_trend_strength",
        "htf_path_efficiency",
        "htf_dir_consistency",
        "htf_vol_trend_confirm",
        # === ATOMIC: LTF Entry 原子信号 ===
        "ltf_pullback_quality",
        "ltf_wick_rejection",
        "ltf_vol_entry_confirm",
        "ltf_cvd_entry_confirm",
        # === ATOMIC: Alignment 原子信号 ===
        "htf_ltf_dir_match",
        "htf_ltf_momentum_align",
        "htf_ltf_vol_sync",
        # === COMPOSITE: 组合分数 ===
        "htf_score_bias",
        "ltf_score_entry",
        "htf_ltf_score_alignment",
        # === CONTEXTUAL: 状态信号 ===
        "htf_direction",
        "ltf_direction",
        "htf_ltf_aligned",
    ],
)
def compute_htf_ltf_soft_phase_from_series(
    *,
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    volume: pd.Series,
    atr: pd.Series,
    # 订单流特征（可选）
    cvd_change_5: pd.Series = None,
    vpin: pd.Series = None,
    # 参数
    htf_lookback: int = 50,
    ltf_lookback: int = 10,
) -> pd.DataFrame:
    """
    HTFBiasLTFEntry 软阶段分数
    
    Args:
        close: 收盘价序列
        high: 最高价序列
        low: 最低价序列
        volume: 成交量序列
        atr: ATR 序列
        cvd_change_5: CVD 5周期变化（可选）
        vpin: VPIN 指标（可选）
        htf_lookback: HTF 检测窗口
        ltf_lookback: LTF 检测窗口
    
    Returns:
        DataFrame with HTF/LTF soft phase scores
    """
    # ========== 类型转换 ==========
    close = pd.to_numeric(close, errors="coerce").astype(float)
    high = pd.to_numeric(high, errors="coerce").astype(float)
    low = pd.to_numeric(low, errors="coerce").astype(float)
    volume = pd.to_numeric(volume, errors="coerce").astype(float).clip(lower=1)
    atr_s = pd.to_numeric(atr, errors="coerce").astype(float).clip(lower=1e-8)
    
    n = len(close)
    eps = 1e-8
    
    # ========== 1️⃣ HTF Bias: 大趋势方向偏见 ==========
    # 趋势强度：HTF 周期的收益 / (ATR * sqrt(周期))
    htf_return = close - close.shift(htf_lookback)
    htf_vol = atr_s * np.sqrt(htf_lookback)
    htf_trend_strength = (htf_return / htf_vol.clip(lower=eps)).abs().clip(0, 3) / 3
    
    # Path Efficiency：直线距离 / 实际路径
    # 注意：使用 min_periods=htf_lookback 确保流式一致性
    path_length = high.rolling(htf_lookback, min_periods=htf_lookback).max() - \
                  low.rolling(htf_lookback, min_periods=htf_lookback).min()
    straight_distance = htf_return.abs()
    htf_path_efficiency = (straight_distance / path_length.clip(lower=eps)).clip(0, 1).fillna(0.5)
    
    # 方向一致性：HTF 周期内方向一致的比例
    # 注意：使用 min_periods=htf_lookback 确保流式一致性
    direction = np.sign(close.diff())
    htf_dir_consistency = direction.rolling(htf_lookback, min_periods=htf_lookback).apply(
        lambda x: (x == np.sign(x.iloc[-1])).mean() if len(x) == htf_lookback else 0.5, raw=False
    ).fillna(0.5)
    
    # 成交量趋势确认
    vol_ma = volume.rolling(htf_lookback, min_periods=htf_lookback).mean().fillna(volume)
    vol_trend = (volume.rolling(ltf_lookback, min_periods=ltf_lookback).mean().fillna(volume) / vol_ma.clip(lower=eps) - 1).clip(-1, 1)
    htf_vol_trend_confirm = ((vol_trend > 0) * np.abs(vol_trend)).clip(0, 1)
    
    # HTF 方向
    htf_direction = np.sign(htf_return).fillna(0).astype(int)
    
    # 综合 HTF Bias 分数
    htf_score_bias = (
        htf_trend_strength * 0.3 +
        htf_path_efficiency * 0.3 +
        htf_dir_consistency * 0.25 +
        htf_vol_trend_confirm * 0.15
    ).clip(0, 1)
    
    # ========== 2️⃣ LTF Entry: 低时间框架入场质量 ==========
    # 回踩质量：LTF 周期的回踩深度
    # 注意：使用 min_periods=ltf_lookback 确保流式一致性
    ltf_high = high.rolling(ltf_lookback, min_periods=ltf_lookback).max().fillna(high)
    ltf_low = low.rolling(ltf_lookback, min_periods=ltf_lookback).min().fillna(low)
    ltf_range = (ltf_high - ltf_low).clip(lower=eps)
    
    # 多头回踩质量
    ltf_pullback_long = ((ltf_high - close) / ltf_range).clip(0, 1)
    # 空头回踩质量
    ltf_pullback_short = ((close - ltf_low) / ltf_range).clip(0, 1)
    # 根据 HTF 方向选择
    ltf_pullback_quality = np.where(
        htf_direction >= 0,
        1 - ltf_pullback_long.values,  # 多头：回踩浅 = 质量好
        1 - ltf_pullback_short.values
    )
    ltf_pullback_quality = pd.Series(ltf_pullback_quality, index=close.index).clip(0, 1)
    
    # Wick Rejection：影线拒绝
    body = (close - close.shift(1).fillna(close)).abs()
    full_range = (high - low).clip(lower=eps)
    upper_wick = high - np.maximum(close, close.shift(1).fillna(close))
    lower_wick = np.minimum(close, close.shift(1).fillna(close)) - low
    
    # 多头：下影线长 = 拒绝下跌
    # 空头：上影线长 = 拒绝上涨
    wick_rejection_long = (lower_wick / full_range).clip(0, 1)
    wick_rejection_short = (upper_wick / full_range).clip(0, 1)
    ltf_wick_rejection = np.where(
        htf_direction >= 0,
        wick_rejection_long.values,
        wick_rejection_short.values
    )
    ltf_wick_rejection = pd.Series(ltf_wick_rejection, index=close.index)
    
    # 入场成交量确认
    vol_ratio = volume / vol_ma.clip(lower=eps)
    ltf_vol_entry_confirm = (vol_ratio / LTF_ENTRY_VOL_MULT).clip(0, 1)
    
    # CVD 入场确认
    if cvd_change_5 is not None:
        cvd = pd.to_numeric(cvd_change_5, errors="coerce").fillna(0)
        # 使用固定窗口确保流式一致性
        cvd_mean = cvd.rolling(20, min_periods=20).mean().fillna(0)
        cvd_std = cvd.rolling(20, min_periods=20).std().fillna(1).clip(lower=eps)
        cvd_z = ((cvd - cvd_mean) / cvd_std).clip(-3, 3)
        ltf_cvd_entry_confirm = np.where(
            htf_direction >= 0,
            (cvd_z > 0).astype(float) * (cvd_z.abs() / 3).clip(0, 1),
            (cvd_z < 0).astype(float) * (cvd_z.abs() / 3).clip(0, 1)
        )
        ltf_cvd_entry_confirm = pd.Series(ltf_cvd_entry_confirm, index=close.index)
    else:
        ltf_cvd_entry_confirm = pd.Series(0.5, index=close.index)
    
    # LTF 方向
    ltf_direction = np.sign(close.diff(ltf_lookback)).fillna(0).astype(int)
    
    # 综合 LTF Entry 分数
    ltf_score_entry = (
        ltf_pullback_quality * 0.3 +
        ltf_wick_rejection * 0.25 +
        ltf_vol_entry_confirm * 0.25 +
        ltf_cvd_entry_confirm * 0.2
    ).clip(0, 1)
    
    # ========== 3️⃣ HTF/LTF Alignment: 对齐度 ==========
    # 方向匹配
    htf_ltf_dir_match = (htf_direction == ltf_direction).astype(float)
    
    # 动量对齐
    htf_momentum = htf_return / atr_s.clip(lower=eps)
    ltf_momentum = (close - close.shift(ltf_lookback)) / atr_s.clip(lower=eps)
    htf_ltf_momentum_align = (np.sign(htf_momentum) == np.sign(ltf_momentum)).astype(float)
    
    # 成交量同步
    vol_sync = (vol_trend > 0) & (htf_vol_trend_confirm > 0.3)
    htf_ltf_vol_sync = vol_sync.astype(float)
    
    # 综合对齐分数
    htf_ltf_score_alignment = (
        htf_ltf_dir_match * 0.4 +
        htf_ltf_momentum_align * 0.35 +
        htf_ltf_vol_sync * 0.25
    ).clip(0, 1)
    
    # 是否对齐（二元标记）
    htf_ltf_aligned = (htf_ltf_score_alignment > 0.6).astype(float)
    
    # ========== 输出 ==========
    result = pd.DataFrame({
        # === ATOMIC: HTF Bias ===
        "htf_trend_strength": htf_trend_strength,
        "htf_path_efficiency": htf_path_efficiency,
        "htf_dir_consistency": htf_dir_consistency,
        "htf_vol_trend_confirm": htf_vol_trend_confirm,
        # === ATOMIC: LTF Entry ===
        "ltf_pullback_quality": ltf_pullback_quality,
        "ltf_wick_rejection": ltf_wick_rejection,
        "ltf_vol_entry_confirm": ltf_vol_entry_confirm,
        "ltf_cvd_entry_confirm": ltf_cvd_entry_confirm,
        # === ATOMIC: Alignment ===
        "htf_ltf_dir_match": htf_ltf_dir_match,
        "htf_ltf_momentum_align": htf_ltf_momentum_align,
        "htf_ltf_vol_sync": htf_ltf_vol_sync,
        # === COMPOSITE ===
        "htf_score_bias": htf_score_bias,
        "ltf_score_entry": ltf_score_entry,
        "htf_ltf_score_alignment": htf_ltf_score_alignment,
        # === CONTEXTUAL ===
        "htf_direction": htf_direction,
        "ltf_direction": ltf_direction,
        "htf_ltf_aligned": htf_ltf_aligned,
    }, index=close.index)
    
    result.attrs['feature_version'] = FEATURE_VERSION
    return result


# =============================================================================
# 🧩 上下文特征
# =============================================================================

@register_feature(
    "compute_htf_ltf_context_from_series",
    category="htf_ltf",
    description="HTF/LTF context: VP position + SR proximity",
    outputs=[
        "htf_above_vp_poc",
        "htf_in_value_area",
        "ltf_near_sr",
        "htf_ltf_structure_valid",
    ],
)
def compute_htf_ltf_context_from_series(
    *,
    close: pd.Series,
    htf_direction: pd.Series,
    vp_poc: pd.Series = None,
    vp_hal_high: pd.Series = None,
    vp_hal_low: pd.Series = None,
    dist_to_nearest_sr: pd.Series = None,
    atr: pd.Series = None,
) -> pd.DataFrame:
    """
    HTF/LTF 上下文特征：VP 位置 + SR 距离
    
    用途：树模型发现"什么情况下 HTF/LTF 语义不成立"
    """
    close = pd.to_numeric(close, errors="coerce").astype(float)
    direction = pd.to_numeric(htf_direction, errors="coerce").fillna(0).astype(int)
    
    # HTF 是否在 POC 上方/下方（与方向一致）
    if vp_poc is not None:
        poc = pd.to_numeric(vp_poc, errors="coerce").fillna(close)
        above_poc = ((close > poc) & (direction > 0)).astype(float)
        below_poc = ((close < poc) & (direction < 0)).astype(float)
        htf_above_vp_poc = above_poc + below_poc
    else:
        htf_above_vp_poc = pd.Series(0.5, index=close.index)
    
    # HTF 是否在价值区间内
    if vp_hal_high is not None and vp_hal_low is not None:
        hal_h = pd.to_numeric(vp_hal_high, errors="coerce").fillna(close)
        hal_l = pd.to_numeric(vp_hal_low, errors="coerce").fillna(close)
        htf_in_value_area = ((close >= hal_l) & (close <= hal_h)).astype(float)
    else:
        htf_in_value_area = pd.Series(0.5, index=close.index)
    
    # LTF 是否接近 SR
    if dist_to_nearest_sr is not None:
        dist = pd.to_numeric(dist_to_nearest_sr, errors="coerce").abs().fillna(1)
        ltf_near_sr = (1 - dist.clip(0, 0.05) / 0.05).clip(0, 1)
    else:
        ltf_near_sr = pd.Series(0.5, index=close.index)
    
    # HTF/LTF 结构有效性
    htf_ltf_structure_valid = (
        htf_above_vp_poc * 0.4 +
        htf_in_value_area * 0.3 +
        ltf_near_sr * 0.3
    ).clip(0, 1)
    
    return pd.DataFrame({
        "htf_above_vp_poc": htf_above_vp_poc,
        "htf_in_value_area": htf_in_value_area,
        "ltf_near_sr": ltf_near_sr,
        "htf_ltf_structure_valid": htf_ltf_structure_valid,
    }, index=close.index)


@register_feature(
    "compute_htf_ltf_failure_signals_from_series",
    category="htf_ltf",
    description="HTF/LTF failure signals for tree model discovery",
    outputs=[
        "htf_trend_exhaustion",
        "ltf_false_entry",
        "htf_ltf_divergence",
        "htf_ltf_failure_score",
    ],
)
def compute_htf_ltf_failure_signals_from_series(
    *,
    close: pd.Series,
    htf_score_bias: pd.Series,
    ltf_score_entry: pd.Series,
    htf_ltf_score_alignment: pd.Series,
    cvd_change_5: pd.Series = None,
    shd_pct: pd.Series = None,
    lookback: int = 10,
) -> pd.DataFrame:
    """
    HTF/LTF 失败信号：供树模型发现语义不成立的条件
    
    - 趋势力竭：HTF 趋势即将结束
    - 假入场：LTF 入场信号失效
    - HTF/LTF 背离：方向不一致
    """
    close = pd.to_numeric(close, errors="coerce").astype(float)
    htf_bias = pd.to_numeric(htf_score_bias, errors="coerce").fillna(0.5).clip(0, 1)
    ltf_entry = pd.to_numeric(ltf_score_entry, errors="coerce").fillna(0.5).clip(0, 1)
    alignment = pd.to_numeric(htf_ltf_score_alignment, errors="coerce").fillna(0.5).clip(0, 1)
    
    # HTF 趋势力竭：趋势强度下降 + SHD 升高
    htf_bias_declining = (htf_bias < htf_bias.shift(lookback)).astype(float)
    if shd_pct is not None:
        shd = pd.to_numeric(shd_pct, errors="coerce").fillna(0.5).clip(0, 1)
        htf_trend_exhaustion = (htf_bias_declining * shd).clip(0, 1)
    else:
        htf_trend_exhaustion = htf_bias_declining * 0.5
    
    # LTF 假入场：入场分数高但价格反向
    price_dir = np.sign(close.diff(lookback // 2))
    entry_dir = np.where(ltf_entry > 0.5, 1, -1)
    ltf_false_entry = ((price_dir != entry_dir) & (ltf_entry > 0.6)).astype(float)
    
    # HTF/LTF 背离
    htf_ltf_divergence = (1 - alignment).clip(0, 1)
    
    # 综合失败分数
    htf_ltf_failure_score = (
        htf_trend_exhaustion * 0.35 +
        ltf_false_entry * 0.35 +
        htf_ltf_divergence * 0.3
    ).clip(0, 1)
    
    return pd.DataFrame({
        "htf_trend_exhaustion": htf_trend_exhaustion,
        "ltf_false_entry": ltf_false_entry,
        "htf_ltf_divergence": htf_ltf_divergence,
        "htf_ltf_failure_score": htf_ltf_failure_score,
    }, index=close.index)
