"""
AuctionExhaustionReversal Archetype 专用特征模块

设计理念：
- 波动极值：趋势末期的"信仰充值"（vol climax）
- 中等效率易衰竭：趋势没死，但信仰没油了
- 力竭反转：cluster 变大但推进变小，能量峰值后衰减

核心输出：
1. aer_score_climax: 极值强度 [0-1]
2. aer_score_exhaustion: 力竭强度 [0-1]
3. aer_score_reversal: 反转质量 [0-1]

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

DEFAULT_LOOKBACK = 20
VOL_CLIMAX_THRESHOLD = 0.85       # 波动极值百分位阈值
PATH_EFFICIENCY_MID_LOW = 0.4     # 中等效率下限
PATH_EFFICIENCY_MID_HIGH = 0.7    # 中等效率上限
EXHAUSTION_DELTA_THRESHOLD = 0.5  # 力竭 delta 阈值

FEATURE_VERSION = "1.0"


def _stream_safe_percentile(series: pd.Series, window: int) -> pd.Series:
    """
    流式安全的百分位计算
    
    确保流式和批量计算一致：
    - 使用固定窗口 window
    - min_periods = window（确保窗口内数据足够）
    - 窗口不足时返回 0.5（中性值）
    """
    result = series.rolling(window, min_periods=window).apply(
        lambda x: (x.iloc[-1] >= x).sum() / len(x) if len(x) == window else 0.5,
        raw=False
    )
    result = result.fillna(0.5)
    return result


# =============================================================================
# 🎯 主函数：Auction Exhaustion Reversal 软分数
# =============================================================================

@register_feature(
    "compute_auction_exhaustion_reversal_soft_phase_from_series",
    category="auction_exhaustion",
    description="AuctionExhaustionReversal soft phase scores",
    outputs=[
        # === ATOMIC: Climax 原子信号 ===
        "aer_vol_climax",
        "aer_atr_climax",
        "aer_range_climax",
        "aer_vol_surge",
        # === ATOMIC: Exhaustion 原子信号 ===
        "aer_path_efficiency_mid",
        "aer_momentum_decay",
        "aer_delta_without_progress",
        "aer_cluster_effort_waste",
        # === ATOMIC: Reversal 原子信号 ===
        "aer_cvd_divergence",
        "aer_price_stall",
        "aer_wick_absorption",
        # === COMPOSITE: 组合分数 ===
        "aer_score_climax",
        "aer_score_exhaustion",
        "aer_score_reversal",
        "aer_score_total",
        # === CONTEXTUAL: 状态信号 ===
        "aer_trend_direction",
        "aer_is_exhausting",
        "aer_path_length_sufficient",
    ],
)
def compute_auction_exhaustion_reversal_soft_phase_from_series(
    *,
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    volume: pd.Series,
    atr: pd.Series,
    # 可选特征
    cvd_change_5: pd.Series = None,
    vpin: pd.Series = None,
    trade_cluster_exhaustion_score: pd.Series = None,
    path_efficiency_pct: pd.Series = None,
    path_length_pct: pd.Series = None,
    # 参数
    lookback: int = 20,
    pct_window: int = 100,
) -> pd.DataFrame:
    """
    AuctionExhaustionReversal 软阶段分数
    
    Args:
        close: 收盘价序列
        high: 最高价序列
        low: 最低价序列
        volume: 成交量序列
        atr: ATR 序列
        cvd_change_5: CVD 5周期变化（可选）
        vpin: VPIN 指标（可选）
        trade_cluster_exhaustion_score: 交易聚集力竭分数（可选）
        path_efficiency_pct: 路径效率百分位（可选）
        path_length_pct: 路径长度百分位（可选）
        lookback: 检测窗口
        pct_window: 百分位计算窗口
    
    Returns:
        DataFrame with auction exhaustion reversal soft phase scores
    """
    # ========== 类型转换 ==========
    close = pd.to_numeric(close, errors="coerce").astype(float)
    high = pd.to_numeric(high, errors="coerce").astype(float)
    low = pd.to_numeric(low, errors="coerce").astype(float)
    volume = pd.to_numeric(volume, errors="coerce").astype(float).clip(lower=1)
    atr_s = pd.to_numeric(atr, errors="coerce").astype(float).clip(lower=1e-8)
    
    n = len(close)
    eps = 1e-8
    
    # ========== 1️⃣ Climax: 波动极值 ==========
    # 成交量极值（使用流式安全的百分位计算）
    vol_pct = _stream_safe_percentile(volume, pct_window)
    aer_vol_climax = (vol_pct > VOL_CLIMAX_THRESHOLD).astype(float) * vol_pct
    
    # ATR 极值
    atr_pct = _stream_safe_percentile(atr_s, pct_window)
    aer_atr_climax = (atr_pct > VOL_CLIMAX_THRESHOLD).astype(float) * atr_pct
    
    # 区间极值
    bar_range = high - low
    range_pct = _stream_safe_percentile(bar_range, pct_window)
    aer_range_climax = (range_pct > VOL_CLIMAX_THRESHOLD).astype(float) * range_pct
    
    # 成交量突增（使用固定窗口确保流式一致性）
    vol_ma = volume.rolling(lookback, min_periods=lookback).mean().fillna(volume)
    vol_ratio = volume / vol_ma.clip(lower=eps)
    aer_vol_surge = (vol_ratio / 2).clip(0, 1)
    
    # 综合 Climax 分数
    aer_score_climax = (
        aer_vol_climax * 0.3 +
        aer_atr_climax * 0.3 +
        aer_range_climax * 0.2 +
        aer_vol_surge * 0.2
    ).clip(0, 1)
    
    # ========== 2️⃣ Exhaustion: 力竭信号 ==========
    # 路径效率中等（趋势没死但效率下降）
    if path_efficiency_pct is not None:
        path_eff = pd.to_numeric(path_efficiency_pct, errors="coerce").fillna(0.5).clip(0, 1)
    else:
        # 计算简化版路径效率（使用固定窗口确保流式一致性）
        straight_dist = (close - close.shift(lookback)).abs()
        cumulative_range = bar_range.rolling(lookback, min_periods=lookback).sum().fillna(bar_range * lookback)
        path_eff = (straight_dist / cumulative_range.clip(lower=eps)).clip(0, 1)
    
    # 中等效率 = 在 [0.4, 0.7] 范围内
    in_mid_range = ((path_eff >= PATH_EFFICIENCY_MID_LOW) & (path_eff <= PATH_EFFICIENCY_MID_HIGH)).astype(float)
    center_distance = 1 - 2 * np.abs(path_eff - 0.55)
    aer_path_efficiency_mid = (in_mid_range * center_distance.clip(0, 1)).clip(0, 1)
    
    # 动量衰减
    momentum = close - close.shift(lookback // 2)
    momentum_prev = close.shift(lookback // 2) - close.shift(lookback)
    momentum_decay = (momentum.abs() < momentum_prev.abs() * 0.7).astype(float)
    decay_ratio = (1 - momentum.abs() / (momentum_prev.abs().clip(lower=eps))).clip(0, 1)
    aer_momentum_decay = (momentum_decay * decay_ratio).clip(0, 1)
    
    # Delta without progress：订单流弼但价格不动
    if cvd_change_5 is not None:
        cvd = pd.to_numeric(cvd_change_5, errors="coerce").fillna(0)
        cvd_std = cvd.abs().rolling(lookback, min_periods=lookback).mean().fillna(cvd.abs())
        cvd_std = cvd_std.clip(lower=eps)
        cvd_strong = cvd.abs() > cvd_std
        price_weak = (close - close.shift(5)).abs() < atr_s * 0.3
        aer_delta_without_progress = (cvd_strong & price_weak).astype(float)
    else:
        aer_delta_without_progress = pd.Series(0.0, index=close.index)
    
    # Cluster effort waste：大量努力但没有推进
    if trade_cluster_exhaustion_score is not None:
        cluster_exhaust = pd.to_numeric(trade_cluster_exhaustion_score, errors="coerce").fillna(0).clip(0, 1)
        aer_cluster_effort_waste = cluster_exhaust
    else:
        # 简化：放量但实体小
        body = (close - close.shift(1)).abs()
        full_range = bar_range.clip(lower=eps)
        small_body = (body / full_range) < 0.3
        vol_high = vol_pct > 0.6
        aer_cluster_effort_waste = (small_body & vol_high).astype(float)
    
    # 综合 Exhaustion 分数
    aer_score_exhaustion = (
        aer_path_efficiency_mid * 0.25 +
        aer_momentum_decay * 0.3 +
        aer_delta_without_progress * 0.25 +
        aer_cluster_effort_waste * 0.2
    ).clip(0, 1)
    
    # ========== 3️⃣ Reversal: 反转信号 ==========
    # CVD 背离
    if cvd_change_5 is not None:
        cvd = pd.to_numeric(cvd_change_5, errors="coerce").fillna(0)
        price_high = close >= close.rolling(lookback, min_periods=lookback).max().fillna(close)
        price_low = close <= close.rolling(lookback, min_periods=lookback).min().fillna(close)
        cvd_high = cvd >= cvd.rolling(lookback, min_periods=lookback).max().fillna(cvd)
        cvd_low = cvd <= cvd.rolling(lookback, min_periods=lookback).min().fillna(cvd)
        
        bearish_div = (price_high & ~cvd_high).astype(float)
        bullish_div = (price_low & ~cvd_low).astype(float)
        aer_cvd_divergence = (bearish_div + bullish_div).clip(0, 1)
    else:
        aer_cvd_divergence = pd.Series(0.0, index=close.index)
    
    # 价格停滞
    price_range_short = (high.rolling(5, min_periods=5).max().fillna(high) - low.rolling(5, min_periods=5).min().fillna(low))
    price_range_long = (high.rolling(lookback, min_periods=lookback).max().fillna(high) - low.rolling(lookback, min_periods=lookback).min().fillna(low))
    range_compression = price_range_short / price_range_long.clip(lower=eps)
    aer_price_stall = (1 - range_compression.clip(0, 1)).clip(0, 1)
    
    # Wick 吸收（长影线表示力竭）
    upper_wick = high - np.maximum(close, close.shift(1).fillna(close))
    lower_wick = np.minimum(close, close.shift(1).fillna(close)) - low
    total_wick = upper_wick + lower_wick
    aer_wick_absorption = (total_wick / bar_range.clip(lower=eps)).clip(0, 1)
    
    # 综合 Reversal 分数
    aer_score_reversal = (
        aer_cvd_divergence * 0.4 +
        aer_price_stall * 0.3 +
        aer_wick_absorption * 0.3
    ).clip(0, 1)
    
    # ========== 综合分数 ==========
    aer_score_total = (
        aer_score_climax * 0.3 +
        aer_score_exhaustion * 0.4 +
        aer_score_reversal * 0.3
    ).clip(0, 1)
    
    # ========== 状态信号 ==========
    # 趋势方向
    aer_trend_direction = np.sign(close - close.shift(lookback)).fillna(0).astype(int)
    
    # 是否在力竭中
    aer_is_exhausting = (aer_score_exhaustion > 0.5).astype(float)
    
    # 路径长度是否足够（需要有足够的趋势才能力竭）
    if path_length_pct is not None:
        path_len = pd.to_numeric(path_length_pct, errors="coerce").fillna(0.5).clip(0, 1)
        aer_path_length_sufficient = (path_len > 0.6).astype(float)
    else:
        cumulative_move = (close - close.shift(lookback)).abs()
        aer_path_length_sufficient = (cumulative_move > atr_s * 2).astype(float)
    
    # ========== 输出 ==========
    result = pd.DataFrame({
        # === ATOMIC: Climax ===
        "aer_vol_climax": aer_vol_climax,
        "aer_atr_climax": aer_atr_climax,
        "aer_range_climax": aer_range_climax,
        "aer_vol_surge": aer_vol_surge,
        # === ATOMIC: Exhaustion ===
        "aer_path_efficiency_mid": aer_path_efficiency_mid,
        "aer_momentum_decay": aer_momentum_decay,
        "aer_delta_without_progress": aer_delta_without_progress,
        "aer_cluster_effort_waste": aer_cluster_effort_waste,
        # === ATOMIC: Reversal ===
        "aer_cvd_divergence": aer_cvd_divergence,
        "aer_price_stall": aer_price_stall,
        "aer_wick_absorption": aer_wick_absorption,
        # === COMPOSITE ===
        "aer_score_climax": aer_score_climax,
        "aer_score_exhaustion": aer_score_exhaustion,
        "aer_score_reversal": aer_score_reversal,
        "aer_score_total": aer_score_total,
        # === CONTEXTUAL ===
        "aer_trend_direction": aer_trend_direction,
        "aer_is_exhausting": aer_is_exhausting,
        "aer_path_length_sufficient": aer_path_length_sufficient,
    }, index=close.index)
    
    result.attrs['feature_version'] = FEATURE_VERSION
    return result


# =============================================================================
# 🧩 失败信号特征
# =============================================================================

@register_feature(
    "compute_auction_exhaustion_reversal_failure_from_series",
    category="auction_exhaustion",
    description="AuctionExhaustionReversal failure signals for tree model discovery",
    outputs=[
        "aer_trend_continuation_risk",
        "aer_not_exhausted",
        "aer_no_divergence",
        "aer_failure_score",
    ],
)
def compute_auction_exhaustion_reversal_failure_from_series(
    *,
    close: pd.Series,
    atr: pd.Series,
    aer_score_climax: pd.Series,
    aer_score_exhaustion: pd.Series,
    aer_cvd_divergence: pd.Series,
    aer_trend_direction: pd.Series,
    jump_risk_pct: pd.Series = None,
    lookback: int = 10,
) -> pd.DataFrame:
    """
    AuctionExhaustionReversal 失败信号：供树模型发现语义不成立的条件
    
    - 趋势延续风险：趋势可能没有真正力竭
    - 未力竭：力竭信号不够强
    - 无背离：CVD 没有背离
    """
    close = pd.to_numeric(close, errors="coerce").astype(float)
    atr_s = pd.to_numeric(atr, errors="coerce").astype(float).clip(lower=1e-8)
    
    climax = pd.to_numeric(aer_score_climax, errors="coerce").fillna(0.5).clip(0, 1)
    exhaustion = pd.to_numeric(aer_score_exhaustion, errors="coerce").fillna(0.5).clip(0, 1)
    cvd_div = pd.to_numeric(aer_cvd_divergence, errors="coerce").fillna(0).clip(0, 1)
    trend_dir = pd.to_numeric(aer_trend_direction, errors="coerce").fillna(0).astype(int)
    
    # 趋势延续风险：趋势方向持续
    price_change = close - close.shift(lookback)
    trend_continues = (
        ((trend_dir > 0) & (price_change > atr_s * 0.5)) |
        ((trend_dir < 0) & (price_change < -atr_s * 0.5))
    ).astype(float)
    
    if jump_risk_pct is not None:
        jump = pd.to_numeric(jump_risk_pct, errors="coerce").fillna(0.5).clip(0, 1)
        # 跳空风险适中时趋势更可能延续
        aer_trend_continuation_risk = (trend_continues * 0.6 + (1 - np.abs(jump - 0.5) * 2) * 0.4).clip(0, 1)
    else:
        aer_trend_continuation_risk = trend_continues
    
    # 未力竭
    aer_not_exhausted = (1 - exhaustion).clip(0, 1)
    
    # 无背离
    aer_no_divergence = (1 - cvd_div).clip(0, 1)
    
    # 综合失败分数
    aer_failure_score = (
        aer_trend_continuation_risk * 0.35 +
        aer_not_exhausted * 0.35 +
        aer_no_divergence * 0.3
    ).clip(0, 1)
    
    return pd.DataFrame({
        "aer_trend_continuation_risk": aer_trend_continuation_risk,
        "aer_not_exhausted": aer_not_exhausted,
        "aer_no_divergence": aer_no_divergence,
        "aer_failure_score": aer_failure_score,
    }, index=close.index)


@register_feature(
    "compute_auction_exhaustion_reversal_context_from_series",
    category="auction_exhaustion",
    description="AuctionExhaustionReversal context: regime suitability",
    outputs=[
        "aer_jump_risk_suitable",
        "aer_reflex_risk",
        "aer_regime_suitable",
    ],
)
def compute_auction_exhaustion_reversal_context_from_series(
    *,
    close: pd.Series,
    jump_risk_pct: pd.Series = None,
    shd_pct: pd.Series = None,
    ofci_pct: pd.Series = None,
) -> pd.DataFrame:
    """
    AuctionExhaustionReversal 上下文特征：Regime 适合度
    
    AER 需要：
    - 跳空风险适中 (0.2-0.5)：太低没动能，太高风险大
    - 反身性风险不能过高
    """
    close = pd.to_numeric(close, errors="coerce").astype(float)
    
    # 跳空风险适合度
    if jump_risk_pct is not None:
        jump = pd.to_numeric(jump_risk_pct, errors="coerce").fillna(0.5).clip(0, 1)
        # AER 需要跳空风险在 0.2-0.5 之间
        in_range = ((jump >= 0.2) & (jump <= 0.5)).astype(float)
        distance_to_center = 1 - 2 * np.abs(jump - 0.35)
        aer_jump_risk_suitable = (in_range * distance_to_center.clip(0, 1)).clip(0, 1)
    else:
        aer_jump_risk_suitable = pd.Series(0.5, index=close.index)
    
    # 反身性风险
    if shd_pct is not None and ofci_pct is not None:
        shd = pd.to_numeric(shd_pct, errors="coerce").fillna(0.5).clip(0, 1)
        ofci = pd.to_numeric(ofci_pct, errors="coerce").fillna(0.5).clip(0, 1)
        aer_reflex_risk = np.maximum(shd, ofci)
    elif shd_pct is not None:
        aer_reflex_risk = pd.to_numeric(shd_pct, errors="coerce").fillna(0.5).clip(0, 1)
    else:
        aer_reflex_risk = pd.Series(0.5, index=close.index)
    
    # Regime 适合度
    aer_regime_suitable = (aer_jump_risk_suitable * (1 - aer_reflex_risk)).clip(0, 1)
    
    return pd.DataFrame({
        "aer_jump_risk_suitable": aer_jump_risk_suitable,
        "aer_reflex_risk": aer_reflex_risk,
        "aer_regime_suitable": aer_regime_suitable,
    }, index=close.index)
