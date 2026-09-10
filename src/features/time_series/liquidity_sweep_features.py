"""
LiquiditySweepRejection Archetype 专用特征模块

设计理念：
- 流动性扫止损：价格快速穿透止损区域后立即反转
- Sweep 检测：被止损引出来的流动性
- 反向 Rejection：sweep 后立刻回到高流动区

核心输出：
1. lsr_score_sweep: 扫止损强度 [0-1]
2. lsr_score_rejection: 拒绝强度 [0-1]
3. lsr_score_reversal: 反转质量 [0-1]

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
SWEEP_ATR_THRESHOLD = 0.3         # Sweep 最小 ATR 倍数
REVERSAL_SPEED_THRESHOLD = 0.5    # 反转速度阈值
CLUSTER_FAILED_VOL_MULT = 1.5     # 失败 cluster 的成交量倍数

FEATURE_VERSION = "1.0"


# =============================================================================
# 🎯 主函数：Liquidity Sweep Rejection 软分数
# =============================================================================

@register_feature(
    "compute_liquidity_sweep_rejection_soft_phase_from_series",
    category="liquidity_sweep",
    description="LiquiditySweepRejection soft phase scores",
    outputs=[
        # === ATOMIC: Sweep 原子信号 ===
        "lsr_sweep_up_detected",
        "lsr_sweep_down_detected",
        "lsr_sweep_depth",
        "lsr_sweep_speed",
        # === ATOMIC: Rejection 原子信号 ===
        "lsr_wick_rejection",
        "lsr_vol_spike",
        "lsr_price_snap_back",
        "lsr_liquidity_return",
        # === ATOMIC: Reversal 原子信号 ===
        "lsr_cvd_flip",
        "lsr_momentum_reversal",
        "lsr_cluster_failed",
        # === COMPOSITE: 组合分数 ===
        "lsr_score_sweep",
        "lsr_score_rejection",
        "lsr_score_reversal",
        "lsr_score_total",
        # === CONTEXTUAL: 状态信号 ===
        "lsr_sweep_side",
        "lsr_is_sweep",
        "lsr_near_sr",
    ],
)
def compute_liquidity_sweep_rejection_soft_phase_from_series(
    *,
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    open_: pd.Series,
    volume: pd.Series,
    atr: pd.Series,
    # 可选特征
    cvd_change_5: pd.Series = None,
    liquidity_void_detected: pd.Series = None,
    trade_cluster_exhaustion_score: pd.Series = None,
    dist_to_nearest_sr: pd.Series = None,
    # 参数
    lookback: int = 20,
) -> pd.DataFrame:
    """
    LiquiditySweepRejection 软阶段分数
    
    Args:
        close: 收盘价序列
        high: 最高价序列
        low: 最低价序列
        open_: 开盘价序列
        volume: 成交量序列
        atr: ATR 序列
        cvd_change_5: CVD 5周期变化（可选）
        liquidity_void_detected: 流动性真空检测（可选）
        trade_cluster_exhaustion_score: 交易聚集力竭分数（可选）
        dist_to_nearest_sr: 到最近 SR 的距离（可选）
        lookback: 检测窗口
    
    Returns:
        DataFrame with liquidity sweep rejection soft phase scores
    """
    # ========== 类型转换 ==========
    close = pd.to_numeric(close, errors="coerce").astype(float)
    high = pd.to_numeric(high, errors="coerce").astype(float)
    low = pd.to_numeric(low, errors="coerce").astype(float)
    open_s = pd.to_numeric(open_, errors="coerce").astype(float)
    volume = pd.to_numeric(volume, errors="coerce").astype(float).clip(lower=1)
    atr_s = pd.to_numeric(atr, errors="coerce").astype(float).clip(lower=1e-8)
    
    n = len(close)
    eps = 1e-8
    
    # ========== 1️⃣ Sweep: 扫止损检测 ==========
    # 近期高低点
    rolling_high = high.rolling(lookback, min_periods=1).max().shift(1)
    rolling_low = low.rolling(lookback, min_periods=1).min().shift(1)
    
    # Sweep up：高点突破后收盘回落
    sweep_up = (high > rolling_high) & (close < rolling_high)
    lsr_sweep_up_detected = sweep_up.astype(float)
    
    # Sweep down：低点突破后收盘回升
    sweep_down = (low < rolling_low) & (close > rolling_low)
    lsr_sweep_down_detected = sweep_down.astype(float)
    
    # Sweep 深度（ATR 倍数）
    sweep_up_depth = np.where(sweep_up, (high - rolling_high) / atr_s.clip(lower=eps), 0)
    sweep_down_depth = np.where(sweep_down, (rolling_low - low) / atr_s.clip(lower=eps), 0)
    lsr_sweep_depth = pd.Series(
        np.maximum(sweep_up_depth, sweep_down_depth), 
        index=close.index
    ).clip(0, 2) / 2
    
    # Sweep 速度（单根 K 线内完成 sweep + rejection）
    full_range = (high - low).clip(lower=eps)
    body = (close - open_s).abs()
    wick_ratio = 1 - body / full_range
    lsr_sweep_speed = (lsr_sweep_up_detected + lsr_sweep_down_detected) * wick_ratio
    
    # 综合 Sweep 分数
    lsr_score_sweep = (
        (lsr_sweep_up_detected + lsr_sweep_down_detected) * 0.3 +
        lsr_sweep_depth * 0.35 +
        lsr_sweep_speed * 0.35
    ).clip(0, 1)
    
    # ========== 2️⃣ Rejection: 拒绝信号 ==========
    # Wick 拒绝
    body_top = np.maximum(close, open_s)
    body_bottom = np.minimum(close, open_s)
    upper_wick = high - body_top
    lower_wick = body_bottom - low
    
    # Sweep up 后的上影线拒绝
    wick_up_rejection = (sweep_up).astype(float) * (upper_wick / full_range)
    # Sweep down 后的下影线拒绝
    wick_down_rejection = (sweep_down).astype(float) * (lower_wick / full_range)
    lsr_wick_rejection = (wick_up_rejection + wick_down_rejection).clip(0, 1)
    
    # 成交量尖峰
    vol_ma = volume.rolling(lookback, min_periods=1).mean()
    vol_ratio = volume / vol_ma.clip(lower=eps)
    lsr_vol_spike = (vol_ratio / CLUSTER_FAILED_VOL_MULT).clip(0, 1)
    
    # 价格弹回速度
    snap_back_up = np.where(sweep_up, rolling_high - close, 0)
    snap_back_down = np.where(sweep_down, close - rolling_low, 0)
    snap_back = np.maximum(snap_back_up, snap_back_down)
    lsr_price_snap_back = pd.Series(snap_back / atr_s.clip(lower=eps), index=close.index).clip(0, 2) / 2
    
    # 流动性回归
    if liquidity_void_detected is not None:
        lv = pd.to_numeric(liquidity_void_detected, errors="coerce").fillna(0).clip(0, 1)
        # Sweep 后回到非真空区 = 好
        lsr_liquidity_return = (1 - lv).clip(0, 1)
    else:
        lsr_liquidity_return = pd.Series(0.5, index=close.index)
    
    # 综合 Rejection 分数
    lsr_score_rejection = (
        lsr_wick_rejection * 0.3 +
        lsr_vol_spike * 0.25 +
        lsr_price_snap_back * 0.25 +
        lsr_liquidity_return * 0.2
    ).clip(0, 1)
    
    # ========== 3️⃣ Reversal: 反转信号 ==========
    # CVD 翻转
    if cvd_change_5 is not None:
        cvd = pd.to_numeric(cvd_change_5, errors="coerce").fillna(0)
        cvd_prev = cvd.shift(1).fillna(0)
        cvd_flip = (np.sign(cvd) != np.sign(cvd_prev)).astype(float)
        cvd_strength = cvd.abs() / cvd.abs().rolling(lookback).mean().clip(lower=eps)
        lsr_cvd_flip = cvd_flip * cvd_strength.clip(0, 2) / 2
    else:
        lsr_cvd_flip = pd.Series(0.5, index=close.index)
    
    # 动量反转
    price_dir = np.sign(close - close.shift(1))
    price_dir_prev = np.sign(close.shift(1) - close.shift(2))
    momentum_flip = (price_dir != price_dir_prev).astype(float)
    momentum_strength = (close - close.shift(1)).abs() / atr_s.clip(lower=eps)
    lsr_momentum_reversal = momentum_flip * momentum_strength.clip(0, 2) / 2
    
    # Cluster 失败（大成交量但价格不动）
    if trade_cluster_exhaustion_score is not None:
        cluster_exhaust = pd.to_numeric(trade_cluster_exhaustion_score, errors="coerce").fillna(0).clip(0, 1)
        lsr_cluster_failed = cluster_exhaust * lsr_vol_spike
    else:
        # 简化：放量但实体小
        small_body = body < full_range * 0.3
        lsr_cluster_failed = (lsr_vol_spike * small_body.astype(float)).clip(0, 1)
    
    # 综合 Reversal 分数
    lsr_score_reversal = (
        lsr_cvd_flip * 0.35 +
        lsr_momentum_reversal * 0.35 +
        lsr_cluster_failed * 0.3
    ).clip(0, 1)
    
    # ========== 综合分数 ==========
    lsr_score_total = (
        lsr_score_sweep * 0.35 +
        lsr_score_rejection * 0.35 +
        lsr_score_reversal * 0.3
    ).clip(0, 1)
    
    # ========== 状态信号 ==========
    # Sweep 方向（1=sweep up 后做空，-1=sweep down 后做多）
    lsr_sweep_side = np.where(
        sweep_up, -1,
        np.where(sweep_down, 1, 0)
    )
    lsr_sweep_side = pd.Series(lsr_sweep_side, index=close.index)
    
    lsr_is_sweep = ((lsr_sweep_up_detected + lsr_sweep_down_detected) > 0).astype(float)
    
    # SR 接近度
    if dist_to_nearest_sr is not None:
        dist = pd.to_numeric(dist_to_nearest_sr, errors="coerce").abs().fillna(1)
        lsr_near_sr = (1 - dist.clip(0, 0.03) / 0.03).clip(0, 1)
    else:
        lsr_near_sr = pd.Series(0.5, index=close.index)
    
    # ========== 输出 ==========
    result = pd.DataFrame({
        # === ATOMIC: Sweep ===
        "lsr_sweep_up_detected": lsr_sweep_up_detected,
        "lsr_sweep_down_detected": lsr_sweep_down_detected,
        "lsr_sweep_depth": lsr_sweep_depth,
        "lsr_sweep_speed": lsr_sweep_speed,
        # === ATOMIC: Rejection ===
        "lsr_wick_rejection": lsr_wick_rejection,
        "lsr_vol_spike": lsr_vol_spike,
        "lsr_price_snap_back": lsr_price_snap_back,
        "lsr_liquidity_return": lsr_liquidity_return,
        # === ATOMIC: Reversal ===
        "lsr_cvd_flip": lsr_cvd_flip,
        "lsr_momentum_reversal": lsr_momentum_reversal,
        "lsr_cluster_failed": lsr_cluster_failed,
        # === COMPOSITE ===
        "lsr_score_sweep": lsr_score_sweep,
        "lsr_score_rejection": lsr_score_rejection,
        "lsr_score_reversal": lsr_score_reversal,
        "lsr_score_total": lsr_score_total,
        # === CONTEXTUAL ===
        "lsr_sweep_side": lsr_sweep_side,
        "lsr_is_sweep": lsr_is_sweep,
        "lsr_near_sr": lsr_near_sr,
    }, index=close.index)
    
    result.attrs['feature_version'] = FEATURE_VERSION
    return result


# =============================================================================
# 🧩 失败信号特征
# =============================================================================

@register_feature(
    "compute_liquidity_sweep_rejection_failure_from_series",
    category="liquidity_sweep",
    description="LiquiditySweepRejection failure signals for tree model discovery",
    outputs=[
        "lsr_trend_continuation_risk",
        "lsr_weak_rejection",
        "lsr_no_liquidity_return",
        "lsr_failure_score",
    ],
)
def compute_liquidity_sweep_rejection_failure_from_series(
    *,
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    atr: pd.Series,
    lsr_score_sweep: pd.Series,
    lsr_score_rejection: pd.Series,
    lsr_sweep_side: pd.Series,
    jump_risk_pct: pd.Series = None,
    lookback: int = 10,
) -> pd.DataFrame:
    """
    LiquiditySweepRejection 失败信号：供树模型发现语义不成立的条件
    
    - 趋势延续风险：sweep 方向可能是真突破
    - 弱拒绝：拒绝信号不够强
    - 无流动性回归：价格没有回到高流动区
    """
    close = pd.to_numeric(close, errors="coerce").astype(float)
    high = pd.to_numeric(high, errors="coerce").astype(float)
    low = pd.to_numeric(low, errors="coerce").astype(float)
    atr_s = pd.to_numeric(atr, errors="coerce").astype(float).clip(lower=1e-8)
    
    sweep_score = pd.to_numeric(lsr_score_sweep, errors="coerce").fillna(0).clip(0, 1)
    rej_score = pd.to_numeric(lsr_score_rejection, errors="coerce").fillna(0).clip(0, 1)
    sweep_side = pd.to_numeric(lsr_sweep_side, errors="coerce").fillna(0).astype(int)
    
    # 趋势延续风险：价格继续向 sweep 方向移动
    price_change = close - close.shift(lookback)
    trend_continuation = (
        ((sweep_side == -1) & (price_change > atr_s * 0.5)) |  # sweep up 后继续涨
        ((sweep_side == 1) & (price_change < -atr_s * 0.5))    # sweep down 后继续跌
    ).astype(float)
    
    if jump_risk_pct is not None:
        jump = pd.to_numeric(jump_risk_pct, errors="coerce").fillna(0.5).clip(0, 1)
        # 高跳空风险 = 趋势延续风险高
        lsr_trend_continuation_risk = (trend_continuation * 0.5 + jump * 0.5).clip(0, 1)
    else:
        lsr_trend_continuation_risk = trend_continuation
    
    # 弱拒绝
    lsr_weak_rejection = (1 - rej_score).clip(0, 1)
    
    # 无流动性回归（价格持续在低流动区）
    rolling_mid = (high.rolling(lookback).max() + low.rolling(lookback).min()) / 2
    distance_from_mid = (close - rolling_mid).abs() / atr_s.clip(lower=1e-8)
    lsr_no_liquidity_return = distance_from_mid.clip(0, 2) / 2
    
    # 综合失败分数
    lsr_failure_score = (
        lsr_trend_continuation_risk * 0.4 +
        lsr_weak_rejection * 0.35 +
        lsr_no_liquidity_return * 0.25
    ).clip(0, 1)
    
    return pd.DataFrame({
        "lsr_trend_continuation_risk": lsr_trend_continuation_risk,
        "lsr_weak_rejection": lsr_weak_rejection,
        "lsr_no_liquidity_return": lsr_no_liquidity_return,
        "lsr_failure_score": lsr_failure_score,
    }, index=close.index)
