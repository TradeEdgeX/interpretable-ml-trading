"""
特征组合：交互特征和衍生特征

包含两类组合特征：
1. **交互特征**（Interaction）：两个特征的乘积（状态 × 动量）
   - 如：vpin × compression_energy = vpin_x_compression
   - 参考：docs/时序模型/高级特征：特征组合交互.md

2. **衍生特征**（Derived）：单个特征的变换或两个特征的其他运算
   - 如：abs(close - zz_high_value) = dist_to_zz_high（差值）
   - 如：cvd 的滚动斜率 = cvd_slope_5（变换）

所有特征都是独立的计算函数，可以在 feature_dependencies.yaml 中单独定义
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional

from src.features.registry import register_feature


@register_feature("compute_liquidity_void_x_wpt_risk", category="interaction")
def compute_liquidity_void_x_wpt_risk(
    df: pd.DataFrame,
    liquidity_void_col: str = "liquidity_void_detected",
    wpt_risk_col: str = "wpt_false_breakout_risk",
) -> pd.Series:
    """
    计算流动性真空 × WPT 假突破风险交互项
    
    Args:
        df: DataFrame with base features
        liquidity_void_col: Liquidity void detection column
        wpt_risk_col: WPT false breakout risk column
    
    Returns:
        Series with interaction feature
    """
    state = df.get(liquidity_void_col, pd.Series(0.0, index=df.index))
    momentum = df.get(wpt_risk_col, pd.Series(0.0, index=df.index))
    return (state.fillna(0) * momentum.fillna(0)).rename("liquidity_void_x_wpt_risk")


@register_feature("compute_liquidity_void_x_wpt_risk_from_series", category="interaction")
def compute_liquidity_void_x_wpt_risk_from_series(
    *,
    liquidity_void_detected: pd.Series,
    wpt_false_breakout_risk: pd.Series,
) -> pd.DataFrame:
    lv = pd.to_numeric(liquidity_void_detected, errors="coerce").fillna(0.0).astype(float)
    risk = pd.to_numeric(wpt_false_breakout_risk, errors="coerce").fillna(0.0).astype(float)
    return (lv * risk).rename("liquidity_void_x_wpt_risk").to_frame()


@register_feature("compute_liquidity_void_x_vpin_from_series", category="interaction")
def compute_liquidity_void_x_vpin_from_series(
    *,
    liquidity_void_detected: pd.Series,
    vpin: pd.Series,
    clip_vpin: float = 5.0,
) -> pd.DataFrame:
    """
    Heavy gate feature: liquidity void state × VPIN signal.

    This is intentionally a separate *heavy* node (depends on order-flow features) so the
    original `liquidity_void` can remain a cheap proxy.

    Args:
        liquidity_void_detected: 0/1 state series
        vpin: VPIN signal series (recommended: z-score like vpin_zscore_50)
        clip_vpin: clip VPIN signal to [-clip_vpin, clip_vpin] to avoid outliers dominating

    Returns:
        DataFrame with one column: liquidity_void_x_vpin
    """
    lv = pd.to_numeric(liquidity_void_detected, errors="coerce").fillna(0.0).astype(float)
    vp = pd.to_numeric(vpin, errors="coerce").fillna(0.0).astype(float)
    if clip_vpin is not None and float(clip_vpin) > 0:
        vp = vp.clip(lower=-float(clip_vpin), upper=float(clip_vpin))
    out = (lv * vp).rename("liquidity_void_x_vpin")
    return out.to_frame()


@register_feature("compute_exhaustion_at_liquidity_void_from_series", category="interaction")
def compute_exhaustion_at_liquidity_void_from_series(
    *,
    trade_cluster_exhaustion_score: pd.Series,
    liquidity_void_detected: pd.Series,
    liquidity_void_false_breakout_risk: Optional[pd.Series] = None,
    use_risk: bool = True,
    clip_score: float = 1.0,
) -> pd.DataFrame:
    """
    Composite semantic: Exhaustion-at-Liquidity-Void

    Intuition:
    - liquidity_void_detected indicates a "low resistance path" / sweep-like episode (proxy without L2)
    - trade_cluster_exhaustion_score indicates "effort without progress" (reversal-friendly)
    - optional: weight by liquidity_void_false_breakout_risk to focus on quick-reversal voids

    Returns:
      DataFrame with one column: exhaustion_at_liquidity_void (0..~1 after clipping)
    """
    ex = pd.to_numeric(trade_cluster_exhaustion_score, errors="coerce").fillna(0.0).astype(float)
    lv = pd.to_numeric(liquidity_void_detected, errors="coerce").fillna(0.0).astype(float)
    out = ex * lv
    if use_risk and liquidity_void_false_breakout_risk is not None:
        risk = (
            pd.to_numeric(liquidity_void_false_breakout_risk, errors="coerce")
            .fillna(0.0)
            .astype(float)
            .clip(lower=0.0, upper=1.0)
        )
        out = out * risk
    if clip_score is not None and float(clip_score) > 0:
        out = out.clip(lower=0.0, upper=float(clip_score)) / float(clip_score)
    return out.rename("exhaustion_at_liquidity_void").to_frame()


@register_feature("compute_vpin_semantic_scores_from_series", category="interaction")
def compute_vpin_semantic_scores_from_series(
    *,
    vpin_zscore_50: pd.Series,
    vpin_signed_imbalance_zscore_50: pd.Series,
    open: pd.Series,
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    atr: pd.Series,
    dist_to_nearest_sr: Optional[pd.Series] = None,
    clip_z: float = 5.0,
    disp_atr_threshold: float = 0.5,
    use_range_disp: bool = True,
    sr_prox_atr: float = 1.5,
) -> pd.DataFrame:
    """
    VPIN semantic mapping (reversal-aware):

    - vpin_stress_score: |z| clipped and normalized to [0,1]
    - vpin_directional_pressure: signed z clipped to [-1,1]
    - vpin_exhaustion_score: high stress + low displacement (effort without progress),
      optionally weighted by SR proximity.
    """
    eps = 1e-8
    clip_z = float(clip_z) if float(clip_z) > 0 else 5.0
    disp_thr = float(disp_atr_threshold) if float(disp_atr_threshold) > 0 else 0.5

    z = pd.to_numeric(vpin_zscore_50, errors="coerce").astype(float).fillna(0.0)
    z_signed = (
        pd.to_numeric(vpin_signed_imbalance_zscore_50, errors="coerce")
        .astype(float)
        .fillna(0.0)
    )

    vpin_stress = (z.abs().clip(0.0, clip_z) / clip_z).rename("vpin_stress_score")
    vpin_pressure = (z_signed.clip(-clip_z, clip_z) / clip_z).rename(
        "vpin_directional_pressure"
    )

    o = pd.to_numeric(open, errors="coerce").astype(float)
    c = pd.to_numeric(close, errors="coerce").astype(float)
    h = pd.to_numeric(high, errors="coerce").astype(float)
    l = pd.to_numeric(low, errors="coerce").astype(float)
    atr_s = pd.to_numeric(atr, errors="coerce").astype(float).fillna(atr.median()).clip(lower=eps)

    if bool(use_range_disp):
        disp = (h - l).abs()
    else:
        disp = (c - o).abs()
    disp_atr = (disp / atr_s).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)
    disp_norm = (disp_atr / disp_thr).clip(0.0, 1.0)

    sr_weight = 1.0
    if dist_to_nearest_sr is not None:
        # dist_to_nearest_sr is pct; convert to ATR multiples: |dist|*price/atr
        dist_pct = pd.to_numeric(dist_to_nearest_sr, errors="coerce").abs().astype(float)
        abs_dist = dist_pct * c
        dist_atr = (abs_dist / (atr_s + eps)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        sr_thr = float(sr_prox_atr) if float(sr_prox_atr) > 0 else 1.5
        sr_weight = (1.0 - (dist_atr / sr_thr).clip(0.0, 1.0)).fillna(0.0)

    vpin_exhaustion = (vpin_stress * (1.0 - disp_norm) * sr_weight).rename(
        "vpin_exhaustion_score"
    )

    return pd.DataFrame(
        {
            "vpin_stress_score": vpin_stress,
            "vpin_directional_pressure": vpin_pressure,
            "vpin_exhaustion_score": vpin_exhaustion,
        }
    )


@register_feature("compute_vpin_scene_semantic_scores_from_series", category="interaction")
def compute_vpin_scene_semantic_scores_from_series(
    *,
    vpin_zscore_50: pd.Series,
    vpin_signed_imbalance_zscore_50: pd.Series,
    open: pd.Series,
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    atr: pd.Series,
    # Context (optional, but strongly recommended)
    compression_score: Optional[pd.Series] = None,
    dist_to_nearest_sr: Optional[pd.Series] = None,
    volume_anomaly: Optional[pd.Series] = None,
    trend_r2_20: Optional[pd.Series] = None,
    # Params
    clip_z: float = 5.0,
    disp_atr_threshold: float = 0.5,
    sr_prox_atr: float = 1.5,
) -> pd.DataFrame:
    """
    VPIN multi-scene semantic mapping (Compression / Ignition / Absorption / Exhaustion).

    Goal: keep *raw* VPIN stats out of the model and feed scene-aligned scores instead.

    Outputs (all ~0..1):
    - vpin_compression_score: high stress + low displacement + high compression (pressure building)
    - vpin_ignition_score: high stress + high displacement (+ volume spike gate if provided)
    - vpin_absorption_score: high stress + low displacement, weighted by SR proximity (near SR)
    - vpin_exhaustion_scene_score: absorption-like but further weighted by (1 - trend_r2_20) if provided
    """
    eps = 1e-8
    clip_z = float(clip_z) if float(clip_z) > 0 else 5.0
    disp_thr = float(disp_atr_threshold) if float(disp_atr_threshold) > 0 else 0.5
    sr_thr = float(sr_prox_atr) if float(sr_prox_atr) > 0 else 1.5

    z = pd.to_numeric(vpin_zscore_50, errors="coerce").astype(float).fillna(0.0)
    z_signed = (
        pd.to_numeric(vpin_signed_imbalance_zscore_50, errors="coerce")
        .astype(float)
        .fillna(0.0)
    )
    stress = (z.abs().clip(0.0, clip_z) / clip_z).clip(0.0, 1.0)
    _pressure = (z_signed.clip(-clip_z, clip_z) / clip_z).clip(-1.0, 1.0)

    o = pd.to_numeric(open, errors="coerce").astype(float)
    c = pd.to_numeric(close, errors="coerce").astype(float)
    h = pd.to_numeric(high, errors="coerce").astype(float)
    l = pd.to_numeric(low, errors="coerce").astype(float)
    atr_s = pd.to_numeric(atr, errors="coerce").astype(float).fillna(atr.median()).clip(lower=eps)

    disp_atr = ((h - l).abs() / atr_s).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)
    disp_norm = (disp_atr / disp_thr).clip(0.0, 1.0)
    low_disp = (1.0 - disp_norm).clip(0.0, 1.0)
    high_disp = disp_norm

    comp_gate = 0.0
    if compression_score is not None:
        comp_gate = (
            pd.to_numeric(compression_score, errors="coerce")
            .astype(float)
            .fillna(0.0)
            .clip(0.0, 1.0)
        )

    sr_weight = 1.0
    if dist_to_nearest_sr is not None:
        # dist_to_nearest_sr is pct; convert to ATR multiples: |dist|*price/atr
        dist_pct = pd.to_numeric(dist_to_nearest_sr, errors="coerce").abs().astype(float)
        abs_dist = dist_pct * c
        dist_atr = (abs_dist / (atr_s + eps)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        sr_weight = (1.0 - (dist_atr / sr_thr).clip(0.0, 1.0)).fillna(0.0)

    vol_gate = 1.0
    if volume_anomaly is not None:
        # volume_anomaly is z-like; map [-3,3] -> [0,1]
        va = pd.to_numeric(volume_anomaly, errors="coerce").astype(float).fillna(0.0).clip(-3.0, 3.0)
        vol_gate = ((va + 3.0) / 6.0).clip(0.0, 1.0)

    trend_end_gate = 1.0
    if trend_r2_20 is not None:
        r2 = pd.to_numeric(trend_r2_20, errors="coerce").astype(float).fillna(0.0).clip(0.0, 1.0)
        trend_end_gate = (1.0 - r2).clip(0.0, 1.0)

    vpin_compression = (stress * low_disp * comp_gate).rename("vpin_compression_score")
    vpin_ignition = (stress * high_disp * vol_gate).rename("vpin_ignition_score")
    vpin_absorption = (stress * low_disp * sr_weight).rename("vpin_absorption_score")
    vpin_exhaustion_scene = (vpin_absorption * trend_end_gate).rename("vpin_exhaustion_scene_score")

    return pd.DataFrame(
        {
            "vpin_compression_score": vpin_compression,
            "vpin_ignition_score": vpin_ignition,
            "vpin_absorption_score": vpin_absorption,
            "vpin_exhaustion_scene_score": vpin_exhaustion_scene,
        }
    )


@register_feature("compute_tbr_imbalance_semantic_scores_from_series", category="interaction")
def compute_tbr_imbalance_semantic_scores_from_series(
    *,
    taker_buy_ratio: pd.Series,
    compression_score: Optional[pd.Series],
    open: pd.Series,
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    atr: pd.Series,
    clip_imb: float = 1.0,
    disp_atr_threshold: float = 0.5,
    use_range_disp: bool = True,
) -> pd.DataFrame:
    """
    Bar-level imbalance semantic mapping using taker_buy_ratio (0..1):

    - imbalance_ratio: (tbr-0.5)*2 in [-1,1]
    - imbalance_exhaustion_score: |imbalance| high but displacement low, gated by compression_score
    """
    eps = 1e-8
    disp_thr = float(disp_atr_threshold) if float(disp_atr_threshold) > 0 else 0.5
    clip_imb = float(clip_imb) if float(clip_imb) > 0 else 1.0

    tbr = pd.to_numeric(taker_buy_ratio, errors="coerce").astype(float).fillna(0.5)
    imb = ((tbr - 0.5) * 2.0).clip(-clip_imb, clip_imb) / clip_imb
    imb = imb.rename("imbalance_ratio")

    o = pd.to_numeric(open, errors="coerce").astype(float)
    c = pd.to_numeric(close, errors="coerce").astype(float)
    h = pd.to_numeric(high, errors="coerce").astype(float)
    l = pd.to_numeric(low, errors="coerce").astype(float)
    atr_s = pd.to_numeric(atr, errors="coerce").astype(float).fillna(atr.median()).clip(lower=eps)

    if bool(use_range_disp):
        disp = (h - l).abs()
    else:
        disp = (c - o).abs()
    disp_atr = (disp / atr_s).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)
    disp_norm = (disp_atr / disp_thr).clip(0.0, 1.0)

    if compression_score is None:
        comp_gate = 0.5
    else:
        comp_gate = (
            pd.to_numeric(compression_score, errors="coerce")
            .astype(float)
            .fillna(0.0)
            .clip(0.0, 1.0)
        )

    imb_exhaust = (imb.abs() * (1.0 - disp_norm) * comp_gate).rename(
        "imbalance_exhaustion_score"
    )

    return pd.DataFrame(
        {
            "imbalance_ratio": imb,
            "imbalance_exhaustion_score": imb_exhaust,
        }
    )


@register_feature("compute_fp_imbalance_exhaustion_from_series", category="interaction")
def compute_fp_imbalance_exhaustion_from_series(
    *,
    fp_max_imbalance_ratio: pd.Series,
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    atr: pd.Series,
    dist_to_nearest_sr: Optional[pd.Series] = None,
    imb_threshold: float = 3.0,
    imb_clip: float = 8.0,
    disp_atr_threshold: float = 0.5,
    sr_prox_atr: float = 1.5,
) -> pd.DataFrame:
    """
    Footprint imbalance → exhaustion semantic (reversal-friendly).

    Intuition:
    - fp_max_imbalance_ratio is often trend/continuation when taken raw.
    - Near SR, if imbalance is high but displacement is low → absorption/exhaustion → reversal-friendly.

    score = imb_strength * (1 - disp_norm) * sr_weight
      - imb_strength: normalize/clipped (>=imb_threshold gives positive strength)
      - disp_norm: clip((|high-low|/ATR)/disp_atr_threshold, 0..1)
      - sr_weight (optional): 1 - clip(dist_atr/sr_prox_atr, 0..1)
    """
    eps = 1e-8
    imb_thr = float(imb_threshold) if float(imb_threshold) > 0 else 3.0
    imb_clip_v = float(imb_clip) if float(imb_clip) > imb_thr else max(imb_thr + 1.0, 8.0)
    disp_thr = float(disp_atr_threshold) if float(disp_atr_threshold) > 0 else 0.5

    imb = pd.to_numeric(fp_max_imbalance_ratio, errors="coerce").astype(float).fillna(0.0)
    imb = imb.clip(lower=0.0, upper=imb_clip_v)
    imb_strength = ((imb - imb_thr) / (imb_clip_v - imb_thr + eps)).clip(0.0, 1.0)

    c = pd.to_numeric(close, errors="coerce").astype(float)
    h = pd.to_numeric(high, errors="coerce").astype(float)
    l = pd.to_numeric(low, errors="coerce").astype(float)
    atr_s = pd.to_numeric(atr, errors="coerce").astype(float).fillna(atr.median()).clip(lower=eps)

    disp_atr = ((h - l).abs() / atr_s).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)
    disp_norm = (disp_atr / disp_thr).clip(0.0, 1.0)

    sr_weight = 1.0
    if dist_to_nearest_sr is not None:
        dist_pct = pd.to_numeric(dist_to_nearest_sr, errors="coerce").abs().astype(float)
        abs_dist = dist_pct * c
        dist_atr = (abs_dist / (atr_s + eps)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        sr_thr = float(sr_prox_atr) if float(sr_prox_atr) > 0 else 1.5
        sr_weight = (1.0 - (dist_atr / sr_thr).clip(0.0, 1.0)).fillna(0.0)

    out = (imb_strength * (1.0 - disp_norm) * sr_weight).rename("fp_imbalance_exhaustion_score")
    return out.to_frame()


@register_feature("compute_fp_imbalance_scene_semantic_scores_from_series", category="interaction")
def compute_fp_imbalance_scene_semantic_scores_from_series(
    *,
    fp_max_imbalance_ratio: pd.Series,
    open: Optional[pd.Series] = None,
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    atr: pd.Series,
    # Context (optional, but recommended)
    compression_score: Optional[pd.Series] = None,
    dist_to_nearest_sr: Optional[pd.Series] = None,
    volume_anomaly: Optional[pd.Series] = None,
    trend_r2_20: Optional[pd.Series] = None,
    # Params
    imb_threshold: float = 3.0,
    imb_clip: float = 8.0,
    disp_atr_threshold: float = 0.5,
    sr_prox_atr: float = 1.5,
) -> pd.DataFrame:
    """
    Footprint max-imbalance multi-scene semantic mapping:

    - fp_imbalance_compression_score: high imbalance + low displacement + high compression
    - fp_imbalance_ignition_score: high imbalance + high displacement (+ volume spike gate if provided) (+ SR weight if provided)
    - fp_imbalance_absorption_score: high imbalance + low displacement near SR
    - fp_imbalance_exhaustion_scene_score: absorption-like but weighted by (1 - trend_r2_20) if provided
    """
    eps = 1e-8
    imb_thr = float(imb_threshold) if float(imb_threshold) > 0 else 3.0
    imb_clip_v = float(imb_clip) if float(imb_clip) > imb_thr else max(imb_thr + 1.0, 8.0)
    disp_thr = float(disp_atr_threshold) if float(disp_atr_threshold) > 0 else 0.5
    sr_thr = float(sr_prox_atr) if float(sr_prox_atr) > 0 else 1.5

    imb = pd.to_numeric(fp_max_imbalance_ratio, errors="coerce").astype(float).fillna(0.0)
    imb = imb.clip(lower=0.0, upper=imb_clip_v)
    imb_strength = ((imb - imb_thr) / (imb_clip_v - imb_thr + eps)).clip(0.0, 1.0)

    c = pd.to_numeric(close, errors="coerce").astype(float)
    h = pd.to_numeric(high, errors="coerce").astype(float)
    l = pd.to_numeric(low, errors="coerce").astype(float)
    atr_s = pd.to_numeric(atr, errors="coerce").astype(float).fillna(atr.median()).clip(lower=eps)

    disp_atr = ((h - l).abs() / atr_s).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)
    disp_norm = (disp_atr / disp_thr).clip(0.0, 1.0)
    low_disp = (1.0 - disp_norm).clip(0.0, 1.0)
    high_disp = disp_norm

    comp_gate = 0.0
    if compression_score is not None:
        comp_gate = (
            pd.to_numeric(compression_score, errors="coerce")
            .astype(float)
            .fillna(0.0)
            .clip(0.0, 1.0)
        )

    sr_weight = 1.0
    if dist_to_nearest_sr is not None:
        dist_pct = pd.to_numeric(dist_to_nearest_sr, errors="coerce").abs().astype(float)
        abs_dist = dist_pct * c
        dist_atr = (abs_dist / (atr_s + eps)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        sr_weight = (1.0 - (dist_atr / sr_thr).clip(0.0, 1.0)).fillna(0.0)

    vol_gate = 1.0
    if volume_anomaly is not None:
        va = pd.to_numeric(volume_anomaly, errors="coerce").astype(float).fillna(0.0).clip(-3.0, 3.0)
        vol_gate = ((va + 3.0) / 6.0).clip(0.0, 1.0)

    trend_end_gate = 1.0
    if trend_r2_20 is not None:
        r2 = pd.to_numeric(trend_r2_20, errors="coerce").astype(float).fillna(0.0).clip(0.0, 1.0)
        trend_end_gate = (1.0 - r2).clip(0.0, 1.0)

    fp_compression = (imb_strength * low_disp * comp_gate).rename("fp_imbalance_compression_score")
    fp_ignition = (imb_strength * high_disp * vol_gate * sr_weight).rename("fp_imbalance_ignition_score")
    fp_absorption = (imb_strength * low_disp * sr_weight).rename("fp_imbalance_absorption_score")
    fp_exhaustion_scene = (fp_absorption * trend_end_gate).rename("fp_imbalance_exhaustion_scene_score")

    return pd.DataFrame(
        {
            "fp_imbalance_compression_score": fp_compression,
            "fp_imbalance_ignition_score": fp_ignition,
            "fp_imbalance_absorption_score": fp_absorption,
            "fp_imbalance_exhaustion_scene_score": fp_exhaustion_scene,
        }
    )


@register_feature("compute_trade_cluster_scene_semantic_scores_from_series", category="interaction")
def compute_trade_cluster_scene_semantic_scores_from_series(
    *,
    trade_cluster_flow_intensity: pd.Series,
    trade_cluster_absorption_score: pd.Series,
    trade_cluster_exhaustion_score: pd.Series,
    compression_score: Optional[pd.Series] = None,
    volume_anomaly: Optional[pd.Series] = None,
    trend_r2_20: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    TradeCluster scene semantics built *on top of* existing TradeCluster semantic scores.

    This avoids feeding raw trade_cluster_* stats and reuses:
      - trade_cluster_absorption_score  (breakout/continuation-friendly)
      - trade_cluster_exhaustion_score  (reversal-friendly)
      - trade_cluster_flow_intensity    (activity/one-sidedness proxy)

    Outputs (0..1):
      - trade_cluster_compression_score
      - trade_cluster_ignition_score
      - trade_cluster_absorption_scene_score
      - trade_cluster_exhaustion_scene_score
    """
    flow = pd.to_numeric(trade_cluster_flow_intensity, errors="coerce").fillna(0.0).astype(float).clip(0.0, 1.0)
    absorp = pd.to_numeric(trade_cluster_absorption_score, errors="coerce").fillna(0.0).astype(float).clip(0.0, 1.0)
    exhaust = pd.to_numeric(trade_cluster_exhaustion_score, errors="coerce").fillna(0.0).astype(float).clip(0.0, 1.0)

    comp_gate = 0.0
    if compression_score is not None:
        comp_gate = pd.to_numeric(compression_score, errors="coerce").fillna(0.0).astype(float).clip(0.0, 1.0)

    vol_gate = 1.0
    if volume_anomaly is not None:
        va = pd.to_numeric(volume_anomaly, errors="coerce").fillna(0.0).astype(float).clip(-3.0, 3.0)
        vol_gate = ((va + 3.0) / 6.0).clip(0.0, 1.0)

    trend_gate = 1.0
    trend_end_gate = 1.0
    if trend_r2_20 is not None:
        r2 = pd.to_numeric(trend_r2_20, errors="coerce").fillna(0.0).astype(float).clip(0.0, 1.0)
        trend_gate = r2
        trend_end_gate = (1.0 - r2).clip(0.0, 1.0)

    # Compression: "effort without progress" during compression regime
    tc_compression = (exhaust * comp_gate).rename("trade_cluster_compression_score")
    # Ignition: absorption + activity + volume gate
    tc_ignition = (absorp * flow * vol_gate).clip(0.0, 1.0).rename("trade_cluster_ignition_score")
    # Absorption/Continuation: emphasize trend regime
    tc_absorption_scene = (absorp * trend_gate).rename("trade_cluster_absorption_scene_score")
    # Exhaustion/Reversal: emphasize trend ending
    tc_exhaustion_scene = (exhaust * trend_end_gate).rename("trade_cluster_exhaustion_scene_score")

    return pd.DataFrame(
        {
            "trade_cluster_compression_score": tc_compression,
            "trade_cluster_ignition_score": tc_ignition,
            "trade_cluster_absorption_scene_score": tc_absorption_scene,
            "trade_cluster_exhaustion_scene_score": tc_exhaustion_scene,
        }
    )


@register_feature("compute_liquidity_void_scene_semantic_scores_from_series", category="interaction")
def compute_liquidity_void_scene_semantic_scores_from_series(
    *,
    liquidity_void_speed: pd.Series,
    liquidity_void_price_impact: pd.Series,
    liquidity_void_retracement: pd.Series,
    liquidity_void_false_breakout_risk: pd.Series,
    wpt_breakout_confidence: Optional[pd.Series] = None,
    compression_score: Optional[pd.Series] = None,
    trend_r2_20: Optional[pd.Series] = None,
    speed_scale: float = 3.0,
    impact_scale: float = 3.0,
) -> pd.DataFrame:
    """
    LiquidityVoid scene semantics (0..1) - V2 连续化版本
    
    改进：用 speed_norm 作为 soft gate，替代 Bool 的 detected。
    这样每 bar 都有连续值，不再是 80% 时间为 0。
    
    语义：
      - compression: 压缩区 + 价格忘速度高
      - ignition: 快速穿越 + 低假突破风险 + WPT 确认
      - absorption: 低回撇 + 低假突破风险 + 趋势延续
      - exhaustion: 高回撇 + 高假突破风险 + 趋势衰竭
    
    Args:
        liquidity_void_speed: 价格穿越速度（ATR 归一化）
        liquidity_void_price_impact: 价格冲击
        liquidity_void_retracement: 回撇幅度 [0,1]
        liquidity_void_false_breakout_risk: 假突破风险 [0,1]
        wpt_breakout_confidence: WPT 突破置信度（可选）
        compression_score: 压缩得分（可选）
        trend_r2_20: 趋势 R²（可选）
        speed_scale: 速度归一化系数
        impact_scale: 冲击归一化系数
    
    Returns:
        DataFrame with 4 scene scores [0,1]:
        - liquidity_void_compression_score
        - liquidity_void_ignition_score
        - liquidity_void_absorption_score
        - liquidity_void_exhaustion_score
    
    Notes:
        - 使用 speed_norm 作为 soft gate，而非 Bool detected
        - 高速度 → 高 gate，低速度 → 低 gate
        - 这样每 bar 都有连续值，适合树模型
    """
    # === 1. 解析输入 ===
    speed = pd.to_numeric(liquidity_void_speed, errors="coerce").fillna(0.0).astype(float).clip(lower=0.0)
    impact = pd.to_numeric(liquidity_void_price_impact, errors="coerce").fillna(0.0).astype(float).clip(lower=0.0)
    retr = pd.to_numeric(liquidity_void_retracement, errors="coerce").fillna(0.0).astype(float).clip(0.0, 1.0)
    fake = pd.to_numeric(liquidity_void_false_breakout_risk, errors="coerce").fillna(0.0).astype(float).clip(0.0, 1.0)

    s_scale = float(speed_scale) if float(speed_scale) > 0 else 3.0
    i_scale = float(impact_scale) if float(impact_scale) > 0 else 3.0
    speed_norm = (speed / s_scale).clip(0.0, 1.0)
    impact_norm = (impact / i_scale).clip(0.0, 1.0)

    # === 2. Soft Gate: 用速度作为连续 gate，替代 Bool detected ===
    # 高速度 = 更可能是真正的流动性真空
    lv_gate = speed_norm

    # === 3. 可选的上下文门控 ===
    comp_gate = pd.Series(0.0, index=speed.index)
    if compression_score is not None:
        comp_gate = pd.to_numeric(compression_score, errors="coerce").fillna(0.0).astype(float).clip(0.0, 1.0)

    trend_gate = pd.Series(1.0, index=speed.index)
    trend_end_gate = pd.Series(1.0, index=speed.index)
    if trend_r2_20 is not None:
        r2 = pd.to_numeric(trend_r2_20, errors="coerce").fillna(0.0).astype(float).clip(0.0, 1.0)
        trend_gate = r2
        trend_end_gate = (1.0 - r2).clip(0.0, 1.0)

    wpt_gate = pd.Series(1.0, index=speed.index)
    if wpt_breakout_confidence is not None:
        wpt_gate = pd.to_numeric(wpt_breakout_confidence, errors="coerce").fillna(0.0).astype(float).clip(0.0, 1.0)

    # === 4. 计算 4 个场景分数 ===
    # compression: 压缩区 + 有速度
    lv_compression = (lv_gate * comp_gate).clip(0.0, 1.0).rename("liquidity_void_compression_score")
    
    # ignition: 快速穿越 + 高冲击 + 低假突破风险 + WPT 确认
    lv_ignition = (lv_gate * impact_norm * (1.0 - fake) * wpt_gate).clip(0.0, 1.0).rename("liquidity_void_ignition_score")
    
    # absorption: 低回撇 + 低假突破风险 + 趋势延续
    lv_absorption = (lv_gate * (1.0 - retr) * (1.0 - fake) * trend_gate).clip(0.0, 1.0).rename("liquidity_void_absorption_score")
    
    # exhaustion: 高回撇 + 高假突破风险 + 趋势衰竭
    lv_exhaustion = (lv_gate * retr * fake * trend_end_gate).clip(0.0, 1.0).rename("liquidity_void_exhaustion_score")

    return pd.DataFrame({
        "liquidity_void_compression_score": lv_compression,
        "liquidity_void_ignition_score": lv_ignition,
        "liquidity_void_absorption_score": lv_absorption,
        "liquidity_void_exhaustion_score": lv_exhaustion,
    })


@register_feature("compute_wpt_scene_semantic_scores_from_series", category="interaction")
def compute_wpt_scene_semantic_scores_from_series(
    *,
    wpt_breakout_confidence: pd.Series,
    wpt_false_breakout_risk: pd.Series,
    wpt_multi_scale_consistency: pd.Series,
    wpt_energy_cascade: pd.Series,
    compression_score: Optional[pd.Series] = None,
    trend_r2_20: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    WPT scene semantics (0..1):
      - compression: high compression_score + high multi-scale consistency + *not yet* igniting
      - ignition: breakout confidence gated by low false-breakout risk
      - absorption/continuation: ignition * energy cascade (trend-strength proxy)
      - exhaustion: false-breakout risk, emphasized when trend_r2 is low (trend ending)
    """
    bc = pd.to_numeric(wpt_breakout_confidence, errors="coerce").fillna(0.0).astype(float).clip(0.0, 1.0)
    fr = pd.to_numeric(wpt_false_breakout_risk, errors="coerce").fillna(0.0).astype(float).clip(0.0, 1.0)
    ms = pd.to_numeric(wpt_multi_scale_consistency, errors="coerce").fillna(0.0).astype(float).clip(0.0, 1.0)
    ec = pd.to_numeric(wpt_energy_cascade, errors="coerce").fillna(0.0).astype(float)
    # energy cascade is often in [-1,1] like; map positive to [0,1]
    ec_pos = ((ec + 1.0) / 2.0).clip(0.0, 1.0)

    comp_gate = 0.0
    if compression_score is not None:
        comp_gate = pd.to_numeric(compression_score, errors="coerce").fillna(0.0).astype(float).clip(0.0, 1.0)

    trend_end_gate = 1.0
    if trend_r2_20 is not None:
        r2 = pd.to_numeric(trend_r2_20, errors="coerce").fillna(0.0).astype(float).clip(0.0, 1.0)
        trend_end_gate = (1.0 - r2).clip(0.0, 1.0)

    wpt_compression = (comp_gate * ms * (1.0 - bc)).clip(0.0, 1.0).rename("wpt_compression_score")
    wpt_ignition = (bc * (1.0 - fr)).clip(0.0, 1.0).rename("wpt_ignition_score")
    wpt_absorption = (wpt_ignition * ec_pos).clip(0.0, 1.0).rename("wpt_absorption_score")
    wpt_exhaustion = (fr * trend_end_gate).clip(0.0, 1.0).rename("wpt_exhaustion_score")

    return pd.DataFrame(
        {
            "wpt_compression_score": wpt_compression,
            "wpt_ignition_score": wpt_ignition,
            "wpt_absorption_score": wpt_absorption,
            "wpt_exhaustion_score": wpt_exhaustion,
        }
    )


@register_feature("compute_volume_profile_scene_semantic_scores_from_series", category="interaction")
def compute_volume_profile_scene_semantic_scores_from_series(
    *,
    vp_width_ratio: pd.Series,
    vp_poc_deviation: pd.Series,
    vp_entropy: pd.Series,
    vp_lv_ratio: pd.Series,
    vp_hv_ratio: pd.Series,
    trend_r2_20: Optional[pd.Series] = None,
    entropy_scale: float = 2.0,
    poc_dev_scale: float = 2.0,
) -> pd.DataFrame:
    """
    Volume-profile scene semantics (0..1) derived from VPVR/VP features:
      - compression: narrow value area + low entropy (consensus + tight)
      - ignition: leaving value (POC deviation) + LV ratio (thin zone traversal proxy)
      - absorption/continuation: low entropy + wide/accepted value area + trend regime
      - exhaustion: high HV ratio (wall/acceptance) + trend ending
    """
    width = pd.to_numeric(vp_width_ratio, errors="coerce").fillna(0.0).astype(float).clip(0.0, 1.0)
    poc_dev = pd.to_numeric(vp_poc_deviation, errors="coerce").fillna(0.0).astype(float).clip(lower=0.0)
    ent = pd.to_numeric(vp_entropy, errors="coerce").fillna(0.0).astype(float).clip(lower=0.0)
    lv = pd.to_numeric(vp_lv_ratio, errors="coerce").fillna(0.0).astype(float).clip(0.0, 1.0)
    hv = pd.to_numeric(vp_hv_ratio, errors="coerce").fillna(0.0).astype(float).clip(0.0, 1.0)

    e_scale = float(entropy_scale) if float(entropy_scale) > 0 else 2.0
    d_scale = float(poc_dev_scale) if float(poc_dev_scale) > 0 else 2.0
    ent_norm = (ent / e_scale).clip(0.0, 1.0)
    poc_norm = (poc_dev / d_scale).clip(0.0, 1.0)

    trend_gate = 1.0
    trend_end_gate = 1.0
    if trend_r2_20 is not None:
        r2 = pd.to_numeric(trend_r2_20, errors="coerce").fillna(0.0).astype(float).clip(0.0, 1.0)
        trend_gate = r2
        trend_end_gate = (1.0 - r2).clip(0.0, 1.0)

    vp_compression = ((1.0 - width) * (1.0 - ent_norm)).clip(0.0, 1.0).rename("vp_compression_score")
    vp_ignition = (poc_norm * lv).clip(0.0, 1.0).rename("vp_ignition_score")
    vp_absorption = ((1.0 - ent_norm) * width * trend_gate).clip(0.0, 1.0).rename("vp_absorption_score")
    vp_exhaustion = (hv * trend_end_gate).clip(0.0, 1.0).rename("vp_exhaustion_score")

    return pd.DataFrame(
        {
            "vp_compression_score": vp_compression,
            "vp_ignition_score": vp_ignition,
            "vp_absorption_score": vp_absorption,
            "vp_exhaustion_score": vp_exhaustion,
        }
    )


@register_feature("compute_wick_scene_semantic_scores_from_series", category="interaction")
def compute_wick_scene_semantic_scores_from_series(
    *,
    wick_upper_ratio: pd.Series,
    wick_lower_ratio: pd.Series,
    compression_score: Optional[pd.Series] = None,
    trend_r2_20: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    Wick-based scene semantics (0..1).

    This is a cheap proxy (no ticks) for rejection/failed breakouts:
      - compression: small wicks + compression regime (quiet)
      - ignition: small wicks + trend regime (clean impulse)
      - absorption/continuation: small wicks + trend regime (follow-through)
      - exhaustion: large wicks + trend ending (rejection)
    """
    wu = pd.to_numeric(wick_upper_ratio, errors="coerce").fillna(0.0).astype(float).clip(0.0, 1.0)
    wl = pd.to_numeric(wick_lower_ratio, errors="coerce").fillna(0.0).astype(float).clip(0.0, 1.0)
    rej = pd.concat([wu, wl], axis=1).max(axis=1).clip(0.0, 1.0)
    calm = (1.0 - rej).clip(0.0, 1.0)

    comp_gate = 0.0
    if compression_score is not None:
        comp_gate = pd.to_numeric(compression_score, errors="coerce").fillna(0.0).astype(float).clip(0.0, 1.0)

    trend_gate = 1.0
    trend_end_gate = 1.0
    if trend_r2_20 is not None:
        r2 = pd.to_numeric(trend_r2_20, errors="coerce").fillna(0.0).astype(float).clip(0.0, 1.0)
        trend_gate = r2
        trend_end_gate = (1.0 - r2).clip(0.0, 1.0)

    wick_compression = (calm * comp_gate).rename("wick_compression_score")
    wick_ignition = (calm * trend_gate).rename("wick_ignition_score")
    wick_absorption = (calm * trend_gate).rename("wick_absorption_score")
    wick_exhaustion = (rej * trend_end_gate).rename("wick_exhaustion_score")

    return pd.DataFrame(
        {
            "wick_compression_score": wick_compression,
            "wick_ignition_score": wick_ignition,
            "wick_absorption_score": wick_absorption,
            "wick_exhaustion_score": wick_exhaustion,
        }
    )


# ========================================================================
# Dual-Source Scene Semantic Cross Features (乘法交叉)
# ========================================================================


@register_feature("compute_dual_compression_from_series", category="interaction")
def compute_dual_compression_from_series(
    *,
    funding_compression_score: pd.Series,
    vpin_compression_score: pd.Series,
) -> pd.DataFrame:
    """
    Dual-source compression confirmation: funding × VPIN.

    Semantic (BPC-aligned):
    - Both funding AND VPIN show compression simultaneously
    - Higher value = stronger structural compression evidence from two independent sources
    - Reduces false BPC signals: single-source compression may be noise

    Both inputs are bounded [0, 1], so the product is also [0, 1].
    """
    fc = pd.to_numeric(funding_compression_score, errors="coerce").fillna(0.0).astype(float).clip(0.0, 1.0)
    vc = pd.to_numeric(vpin_compression_score, errors="coerce").fillna(0.0).astype(float).clip(0.0, 1.0)
    return (fc * vc).rename("dual_compression_score").to_frame()


@register_feature("compute_dual_ignition_from_series", category="interaction")
def compute_dual_ignition_from_series(
    *,
    funding_ignition_score: pd.Series,
    fp_imbalance_ignition_score: pd.Series,
) -> pd.DataFrame:
    """
    Dual-source ignition confirmation: funding × footprint imbalance.

    Semantic (ME-aligned):
    - Both funding AND footprint show ignition simultaneously
    - Higher value = stronger momentum expansion evidence from two independent sources
    - Reduces false ME signals: single-source ignition may be noise

    Both inputs are bounded [0, 1], so the product is also [0, 1].
    """
    fi = pd.to_numeric(funding_ignition_score, errors="coerce").fillna(0.0).astype(float).clip(0.0, 1.0)
    fpi = pd.to_numeric(fp_imbalance_ignition_score, errors="coerce").fillna(0.0).astype(float).clip(0.0, 1.0)
    return (fi * fpi).rename("dual_ignition_score").to_frame()


@register_feature("compute_dual_exhaustion_from_series", category="interaction")
def compute_dual_exhaustion_from_series(
    *,
    funding_exhaustion_scene_score: pd.Series,
    vpin_exhaustion_scene_score: pd.Series,
) -> pd.DataFrame:
    """
    Dual-source exhaustion confirmation: funding × VPIN.

    Semantic (FER-aligned):
    - Both funding AND VPIN show exhaustion simultaneously
    - Higher value = stronger reversal evidence from two independent sources
    - Reduces false FER signals: single-source exhaustion may be noise

    Both inputs are bounded [0, 1], so the product is also [0, 1].
    """
    fe = pd.to_numeric(funding_exhaustion_scene_score, errors="coerce").fillna(0.0).astype(float).clip(0.0, 1.0)
    ve = pd.to_numeric(vpin_exhaustion_scene_score, errors="coerce").fillna(0.0).astype(float).clip(0.0, 1.0)
    return (fe * ve).rename("dual_exhaustion_score").to_frame()


@register_feature("compute_funding_oi_crowding_from_series", category="interaction")
def compute_funding_oi_crowding_from_series(
    *,
    funding_rate_abs_zscore_50: pd.Series,
    oi_zscore: pd.Series,
    funding_shift: float = 1.0,
    funding_scale: float = 1.0,
    oi_shift: float = 0.5,
    oi_scale: float = 1.0,
) -> pd.DataFrame:
    """Funding \u00d7 OI crowding confirmation score.

    True crowding = **high funding stress** + **high OI buildup**.
    Single funding alone may be short-term sentiment; OI confirms
    structural positioning (actual open contracts, not just fee premium).

    Formula::

        crowding = sigmoid(funding_abs_z) \u00d7 sigmoid(oi_z)

    Both sigmoids map z-scores to (0, 1), product is in [0, 1].

    Use-cases:
    * **LV**: crowded + high vol \u2192 liquidation cascade risk
    * **FER**: crowded trend \u2192 reversal fuel
    * **BPC**: crowded compression \u2192 stronger breakout potential
    """
    fr_z = pd.to_numeric(funding_rate_abs_zscore_50, errors="coerce").fillna(0.0).astype(float)
    oi_z = pd.to_numeric(oi_zscore, errors="coerce").fillna(0.0).astype(float)

    funding_stress = 1.0 / (1.0 + np.exp(-(fr_z - float(funding_shift)) / float(funding_scale)))
    oi_activity = 1.0 / (1.0 + np.exp(-(oi_z - float(oi_shift)) / float(oi_scale)))

    crowding = (funding_stress * oi_activity).clip(0.0, 1.0)
    return crowding.rename("funding_oi_crowding_score").to_frame()


@register_feature("compute_compression_energy_x_ofi_short", category="interaction")
def compute_compression_energy_x_ofi_short(
    df: pd.DataFrame,
    compression_col: str = "compression_energy",
    ofi_col: str = "ofi_short",
) -> pd.Series:
    """
    计算压缩能量 × 订单流强度交互项
    
    Args:
        df: DataFrame with base features
        compression_col: Compression energy column
        ofi_col: Order flow imbalance column
    
    Returns:
        Series with interaction feature
    """
    state = df.get(compression_col, pd.Series(0.0, index=df.index))
    momentum = df.get(ofi_col, pd.Series(0.0, index=df.index))
    return (state.fillna(0) * momentum.fillna(0)).rename("compression_energy_x_ofi_short")


@register_feature("compute_compression_energy_x_ofi_short_from_series", category="interaction")
def compute_compression_energy_x_ofi_short_from_series(
    *,
    compression_energy: pd.Series,
    ofi_short: pd.Series,
) -> pd.DataFrame:
    ce = pd.to_numeric(compression_energy, errors="coerce").fillna(0.0).astype(float)
    ofi = pd.to_numeric(ofi_short, errors="coerce").fillna(0.0).astype(float)
    return (ce * ofi).rename("compression_energy_x_ofi_short").to_frame()


@register_feature("compute_hurst_x_trend_r2", category="interaction")
def compute_hurst_x_trend_r2(
    df: pd.DataFrame,
    hurst_col: str = "hurst_close_rolling",
    trend_r2_col: str = "trend_r2_20",
) -> pd.Series:
    """
    计算 Hurst 指数 × 趋势 R² 交互项
    
    Args:
        df: DataFrame with base features
        hurst_col: Hurst exponent column
        trend_r2_col: Trend R² column
    
    Returns:
        Series with interaction feature
    """
    state = df.get(hurst_col, pd.Series(0.5, index=df.index))
    momentum = df.get(trend_r2_col, pd.Series(0.0, index=df.index))
    return (state.fillna(0.5) * momentum.fillna(0)).rename("hurst_x_trend_r2")


@register_feature("compute_hurst_x_trend_r2_from_series", category="interaction")
def compute_hurst_x_trend_r2_from_series(
    *,
    hurst_close_rolling: pd.Series,
    trend_r2_20: pd.Series,
) -> pd.DataFrame:
    h = pd.to_numeric(hurst_close_rolling, errors="coerce").fillna(0.5).astype(float)
    r2 = pd.to_numeric(trend_r2_20, errors="coerce").fillna(0.0).astype(float)
    return (h * r2).rename("hurst_x_trend_r2").to_frame()


@register_feature("compute_evt_x_trend_r2", category="interaction")
def compute_evt_x_trend_r2(
    df: pd.DataFrame,
    evt_col: str = "evt_tail_shape",
    trend_r2_col: str = "trend_r2_20",
) -> pd.Series:
    """
    计算 EVT 尾部风险 × 趋势强度交互项
    
    Args:
        df: DataFrame with base features
        evt_col: EVT tail shape column
        trend_r2_col: Trend R² column
    
    Returns:
        Series with interaction feature
    """
    state = df.get(evt_col, pd.Series(0.3, index=df.index))
    momentum = df.get(trend_r2_col, pd.Series(0.0, index=df.index))
    return (state.fillna(0.3) * momentum.fillna(0)).rename("evt_x_trend_r2")


@register_feature("compute_evt_x_trend_r2_from_series", category="interaction")
def compute_evt_x_trend_r2_from_series(
    *,
    evt_tail_shape: pd.Series,
    trend_r2_20: pd.Series,
) -> pd.DataFrame:
    evt = pd.to_numeric(evt_tail_shape, errors="coerce").fillna(0.3).astype(float)
    r2 = pd.to_numeric(trend_r2_20, errors="coerce").fillna(0.0).astype(float)
    return (evt * r2).rename("evt_x_trend_r2").to_frame()


@register_feature("compute_vpin_x_compression", category="interaction")
def compute_vpin_x_compression(
    df: pd.DataFrame,
    vpin_col: str = "vpin",
    compression_col: str = "compression_energy",
) -> pd.Series:
    """
    计算 VPIN × 压缩能量交互项
    
    Args:
        df: DataFrame with base features
        vpin_col: VPIN column
        compression_col: Compression energy column
    
    Returns:
        Series with interaction feature
    """
    state = df.get(vpin_col, pd.Series(0.0, index=df.index))
    momentum = df.get(compression_col, pd.Series(0.0, index=df.index))
    return (state.fillna(0) * momentum.fillna(0)).rename("vpin_x_compression")


@register_feature("compute_vpin_x_compression_from_series", category="interaction")
def compute_vpin_x_compression_from_series(
    *,
    vpin: pd.Series,
    compression_energy: pd.Series,
) -> pd.DataFrame:
    vp = pd.to_numeric(vpin, errors="coerce").fillna(0.0).astype(float)
    ce = pd.to_numeric(compression_energy, errors="coerce").fillna(0.0).astype(float)
    return (vp * ce).rename("vpin_x_compression").to_frame()


@register_feature("compute_vpin_x_trade_cluster_max_buy_run", category="interaction")
def compute_vpin_x_trade_cluster_max_buy_run(
    df: pd.DataFrame,
    vpin_col: str = "vpin",
    cluster_col: str = "trade_cluster_max_buy_run",
) -> pd.Series:
    """
    计算 VPIN × 最大连续买入长度交互项
    
    捕捉"高订单流不平衡 + 连续买入聚集"的组合信号
    可能表示知情交易者的策略性连续买入
    
    Args:
        df: DataFrame with base features
        vpin_col: VPIN column
        cluster_col: Trade cluster max buy run column
    
    Returns:
        Series with interaction feature
    """
    state = df.get(vpin_col, pd.Series(0.0, index=df.index))
    momentum = df.get(cluster_col, pd.Series(0.0, index=df.index))
    return (state.fillna(0) * momentum.fillna(0)).rename("vpin_x_trade_cluster_max_buy_run")


@register_feature("compute_vpin_x_trade_cluster_max_buy_run_from_series", category="interaction")
def compute_vpin_x_trade_cluster_max_buy_run_from_series(
    *,
    vpin: pd.Series,
    trade_cluster_max_buy_run: pd.Series,
) -> pd.DataFrame:
    vp = pd.to_numeric(vpin, errors="coerce").fillna(0.0).astype(float)
    run = pd.to_numeric(trade_cluster_max_buy_run, errors="coerce").fillna(0.0).astype(float)
    return (vp * run).rename("vpin_x_trade_cluster_max_buy_run").to_frame()


@register_feature("compute_vpin_zscore_x_trade_cluster_max_buy_run", category="interaction")
def compute_vpin_zscore_x_trade_cluster_max_buy_run(
    df: pd.DataFrame,
    vpin_zscore_col: str = "vpin_zscore_20",
    cluster_col: str = "trade_cluster_max_buy_run",
) -> pd.Series:
    """
    计算 VPIN Z-score × 最大连续买入长度交互项
    
    捕捉"异常高的订单流不平衡 + 连续买入聚集"的组合信号
    这是用户建议的交叉项，可能有超加成效应
    
    Args:
        df: DataFrame with base features
        vpin_zscore_col: VPIN Z-score column
        cluster_col: Trade cluster max buy run column
    
    Returns:
        Series with interaction feature
    """
    state = df.get(vpin_zscore_col, pd.Series(0.0, index=df.index))
    momentum = df.get(cluster_col, pd.Series(0.0, index=df.index))
    return (state.fillna(0) * momentum.fillna(0)).rename("vpin_zscore_x_trade_cluster_max_buy_run")


@register_feature("compute_vpin_zscore_x_trade_cluster_max_buy_run_from_series", category="interaction")
def compute_vpin_zscore_x_trade_cluster_max_buy_run_from_series(
    *,
    vpin_zscore_20: pd.Series,
    trade_cluster_max_buy_run: pd.Series,
) -> pd.DataFrame:
    vz = pd.to_numeric(vpin_zscore_20, errors="coerce").fillna(0.0).astype(float)
    run = pd.to_numeric(trade_cluster_max_buy_run, errors="coerce").fillna(0.0).astype(float)
    return (vz * run).rename("vpin_zscore_x_trade_cluster_max_buy_run").to_frame()


@register_feature("compute_vpin_signed_imbalance_x_trade_cluster_imbalance", category="interaction")
def compute_vpin_signed_imbalance_x_trade_cluster_imbalance(
    df: pd.DataFrame,
    vpin_signed_col: str = "vpin_signed_imbalance",
    cluster_imbalance_col: str = "trade_cluster_imbalance_ratio",
) -> pd.Series:
    """
    计算 VPIN Signed Imbalance × Trade Clustering Imbalance 交互项
    
    捕捉"订单流方向性 + 成交聚集方向性"的一致性
    两者方向一致时，信号更可靠
    
    Args:
        df: DataFrame with base features
        vpin_signed_col: VPIN signed imbalance column
        cluster_imbalance_col: Trade cluster imbalance ratio column
    
    Returns:
        Series with interaction feature
    """
    state = df.get(vpin_signed_col, pd.Series(0.0, index=df.index))
    momentum = df.get(cluster_imbalance_col, pd.Series(0.0, index=df.index))
    return (state.fillna(0) * momentum.fillna(0)).rename("vpin_signed_imbalance_x_trade_cluster_imbalance")


@register_feature("compute_vpin_signed_imbalance_x_trade_cluster_imbalance_from_series", category="interaction")
def compute_vpin_signed_imbalance_x_trade_cluster_imbalance_from_series(
    *,
    vpin_signed_imbalance: pd.Series,
    trade_cluster_imbalance_ratio: pd.Series,
) -> pd.DataFrame:
    vp = pd.to_numeric(vpin_signed_imbalance, errors="coerce").fillna(0.0).astype(float)
    imb = pd.to_numeric(trade_cluster_imbalance_ratio, errors="coerce").fillna(0.0).astype(float)
    return (vp * imb).rename("vpin_signed_imbalance_x_trade_cluster_imbalance").to_frame()


@register_feature("compute_vpin_x_trade_cluster_entropy", category="interaction")
def compute_vpin_x_trade_cluster_entropy(
    df: pd.DataFrame,
    vpin_col: str = "vpin",
    entropy_col: str = "trade_cluster_directional_entropy",
) -> pd.Series:
    """
    计算 VPIN × 方向熵交互项
    
    捕捉"订单流不平衡 + 成交混乱度"的组合
    高 VPIN + 低熵 = 大单主导且有序（知情交易）
    高 VPIN + 高熵 = 大单主导但混乱（可能假突破）
    
    Args:
        df: DataFrame with base features
        vpin_col: VPIN column
        entropy_col: Trade cluster directional entropy column
    
    Returns:
        Series with interaction feature
    """
    state = df.get(vpin_col, pd.Series(0.0, index=df.index))
    momentum = df.get(entropy_col, pd.Series(0.0, index=df.index))
    return (state.fillna(0) * momentum.fillna(0)).rename("vpin_x_trade_cluster_entropy")


@register_feature("compute_vpin_x_trade_cluster_entropy_from_series", category="interaction")
def compute_vpin_x_trade_cluster_entropy_from_series(
    *,
    vpin: pd.Series,
    trade_cluster_directional_entropy: pd.Series,
) -> pd.DataFrame:
    vp = pd.to_numeric(vpin, errors="coerce").fillna(0.0).astype(float)
    ent = pd.to_numeric(trade_cluster_directional_entropy, errors="coerce").fillna(0.0).astype(float)
    return (vp * ent).rename("vpin_x_trade_cluster_entropy").to_frame()


@register_feature("compute_sma_slope_x_price_pos", category="interaction")
def compute_sma_slope_x_price_pos(
    df: pd.DataFrame,
    sma_slope_col: str = "sma_200_slope",
    sma_col: str = "sma_200",
    close_col: str = "close",
) -> pd.Series:
    """
    计算均线斜率 × 价格位置交互项
    
    Args:
        df: DataFrame with base features
        sma_slope_col: SMA slope column
        sma_col: SMA column
        close_col: Close price column
    
    Returns:
        Series with interaction feature
    """
    state = df.get(sma_slope_col, pd.Series(0.0, index=df.index))
    # 计算价格位置
    if sma_col in df.columns and close_col in df.columns:
        price_pos = (df[close_col] / df[sma_col].replace(0, np.nan)).fillna(1.0)
    else:
        price_pos = pd.Series(1.0, index=df.index)
    return (state.fillna(0) * price_pos).rename("sma_slope_x_price_pos")


@register_feature("compute_sma_slope_x_price_pos_from_series", category="interaction")
def compute_sma_slope_x_price_pos_from_series(
    *,
    sma_200_slope: pd.Series,
    sma_200: pd.Series,
    close: pd.Series,
) -> pd.DataFrame:
    slope = pd.to_numeric(sma_200_slope, errors="coerce").fillna(0.0).astype(float)
    sma = pd.to_numeric(sma_200, errors="coerce").astype(float)
    cl = pd.to_numeric(close, errors="coerce").astype(float)
    price_pos = (cl / sma.replace(0, np.nan)).fillna(1.0)
    return (slope * price_pos).rename("sma_slope_x_price_pos").to_frame()


@register_feature("compute_vpin_x_wick_upper", category="interaction")
def compute_vpin_x_wick_upper(
    df: pd.DataFrame,
    vpin_col: str = "vpin",
    wick_col: str = "wick_upper_ratio",
) -> pd.Series:
    """
    计算 VPIN × 上影线占比交互项（反转策略专用）
    
    Args:
        df: DataFrame with base features
        vpin_col: VPIN column
        wick_col: Upper wick ratio column
    
    Returns:
        Series with interaction feature
    """
    state = df.get(vpin_col, pd.Series(0.0, index=df.index))
    momentum = df.get(wick_col, pd.Series(0.0, index=df.index))
    return (state.fillna(0) * momentum.fillna(0)).rename("vpin_x_wick_upper")


@register_feature("compute_vpin_x_wick_upper_from_series", category="interaction")
def compute_vpin_x_wick_upper_from_series(
    *,
    vpin: pd.Series,
    wick_upper_ratio: pd.Series,
) -> pd.DataFrame:
    vp = pd.to_numeric(vpin, errors="coerce").fillna(0.0).astype(float)
    wr = pd.to_numeric(wick_upper_ratio, errors="coerce").fillna(0.0).astype(float)
    return (vp * wr).rename("vpin_x_wick_upper").to_frame()


@register_feature("compute_vpin_x_wick_lower", category="interaction")
def compute_vpin_x_wick_lower(
    df: pd.DataFrame,
    vpin_col: str = "vpin",
    wick_col: str = "wick_lower_ratio",
) -> pd.Series:
    """
    计算 VPIN × 下影线占比交互项（反转策略专用）
    
    Args:
        df: DataFrame with base features
        vpin_col: VPIN column
        wick_col: Lower wick ratio column
    
    Returns:
        Series with interaction feature
    """
    state = df.get(vpin_col, pd.Series(0.0, index=df.index))
    momentum = df.get(wick_col, pd.Series(0.0, index=df.index))
    return (state.fillna(0) * momentum.fillna(0)).rename("vpin_x_wick_lower")


@register_feature("compute_vpin_x_wick_lower_from_series", category="interaction")
def compute_vpin_x_wick_lower_from_series(
    *,
    vpin: pd.Series,
    wick_lower_ratio: pd.Series,
) -> pd.DataFrame:
    vp = pd.to_numeric(vpin, errors="coerce").fillna(0.0).astype(float)
    wr = pd.to_numeric(wick_lower_ratio, errors="coerce").fillna(0.0).astype(float)
    return (vp * wr).rename("vpin_x_wick_lower").to_frame()


@register_feature("apply_rank_transform_to_interaction", category="interaction")
def apply_rank_transform_to_interaction(
    df: pd.DataFrame,
    interaction_col: str,
    groupby_col: Optional[str] = None,
) -> pd.Series:
    """
    对单个交互项做 rank transform
    
    Args:
        df: DataFrame with interaction feature
        interaction_col: Interaction column name
        groupby_col: Optional column for cross-sectional rank
    
    Returns:
        Series with rank-transformed interaction feature
    """
    if interaction_col not in df.columns:
        return pd.Series(dtype=float, index=df.index)
    
    if groupby_col and groupby_col in df.columns:
        # 横截面 rank（多标的场景）
        rank_series = df.groupby(groupby_col)[interaction_col].rank(pct=True, method="average")
    else:
        # 全局 rank（单标的时序场景）
        rank_series = df[interaction_col].rank(pct=True, method="average")
    
    return rank_series.fillna(0.5).rename(f"{interaction_col}_rank")


@register_feature("apply_rank_transform_to_interaction_from_series", category="interaction")
def apply_rank_transform_to_interaction_from_series(
    *,
    interaction: pd.Series,
) -> pd.DataFrame:
    """
    Narrow-IO rank transform for a single interaction series.
    Uses global rank(pct=True) and fills missing with 0.5, matching legacy behavior.
    """
    s = pd.to_numeric(interaction, errors="coerce").astype(float)
    ranked = s.rank(pct=True, method="average").fillna(0.5)
    return ranked.rename(f"{interaction.name or 'interaction'}_rank").to_frame()


@register_feature("apply_signed_rank_transform_to_interaction_from_series", category="interaction")
def apply_signed_rank_transform_to_interaction_from_series(
    *,
    interaction: pd.Series,
) -> pd.DataFrame:
    """
    Signed rank transform: 保留原始符号，只对绝对值做 rank。
    
    对于有正负方向的交互特征（如 vpin_signed_imbalance × trade_cluster_imbalance），
    直接做 rank 会丢失方向信息。此函数保留符号：
    - 输出范围: [-1, 1]
    - sign(x) × rank(|x|)
    """
    s = pd.to_numeric(interaction, errors="coerce").astype(float)
    sign = np.sign(s)
    abs_ranked = s.abs().rank(pct=True, method="average").fillna(0.5)
    signed_ranked = sign * abs_ranked
    return signed_ranked.rename(f"{interaction.name or 'interaction'}_signed_rank").to_frame()


# ========================================================================
# 衍生特征（Derived Features）：单个特征的变换或两个特征的其他运算
# ========================================================================

@register_feature("compute_sr_strength_combined", category="derived")
def compute_sr_strength_combined(
    df: pd.DataFrame,
    sqs_col: str = "sqs",
) -> pd.Series:
    """
    计算 SR 强度组合特征（单个特征的简单映射）
    
    Args:
        df: DataFrame with base features
        sqs_col: SQS column name
    
    Returns:
        Series with sr_strength_combined
    
    Raises:
        ValueError: 如果依赖列不存在
    """
    if sqs_col not in df.columns:
        raise ValueError(
            f"Required column '{sqs_col}' not found for sr_strength_combined. "
            f"Available columns: {list(df.columns)[:20]}..."
        )
    return df[sqs_col].fillna(0.0).rename("sr_strength_combined")


@register_feature("compute_sr_strength_combined_from_series", category="derived")
def compute_sr_strength_combined_from_series(*, sqs: pd.Series) -> pd.DataFrame:
    """Narrow-IO entrypoint for sr_strength_combined."""
    s = pd.to_numeric(sqs, errors="coerce").fillna(0.0).astype(float)
    return s.rename("sr_strength_combined").to_frame()


@register_feature(
    "compute_sr_strength_combined_from_hal_sqs_from_series", category="derived"
)
def compute_sr_strength_combined_from_hal_sqs_from_series(
    *, sqs_hal_high: pd.Series, sqs_hal_low: pd.Series
) -> pd.DataFrame:
    """
    Robust Narrow-IO entrypoint for sr_strength_combined.

    Some pipelines may not materialize an intermediate `sqs` column (combined SQS),
    but do produce `sqs_hal_high` and `sqs_hal_low`. This function derives the
    combined strength directly as max(high, low), avoiding hard dependency on `sqs`.
    """
    h = pd.to_numeric(sqs_hal_high, errors="coerce").fillna(0.0).astype(float)
    l = pd.to_numeric(sqs_hal_low, errors="coerce").fillna(0.0).astype(float)
    out = pd.Series(np.maximum(h.values, l.values), index=h.index, name="sr_strength_combined")
    return out.to_frame()


@register_feature("compute_dist_to_zz_high", category="derived")
def compute_dist_to_zz_high(
    df: pd.DataFrame,
    price_col: str = "close",
    zz_high_col: str = "zz_high_value",
) -> pd.Series:
    """
    计算到 ZigZag 高点的距离（两个特征的差值：abs(price - zz_high)）
    
    Args:
        df: DataFrame with base features
        price_col: Price column
        zz_high_col: ZigZag high value column
    
    Returns:
        Series with dist_to_zz_high
    
    Raises:
        ValueError: 如果依赖列不存在
    """
    missing_cols = [c for c in [price_col, zz_high_col] if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Required columns {missing_cols} not found for dist_to_zz_high. "
            f"Available columns: {list(df.columns)[:20]}..."
        )
    return (
        (df[price_col] - df[zz_high_col]).abs()
    ).fillna(0.0).rename("dist_to_zz_high")


@register_feature("compute_dist_to_zz_high_from_series", category="derived")
def compute_dist_to_zz_high_from_series(
    *, close: pd.Series, zz_high_value: pd.Series
) -> pd.DataFrame:
    """Narrow-IO entrypoint for dist_to_zz_high."""
    c = pd.to_numeric(close, errors="coerce").astype(float)
    z = pd.to_numeric(zz_high_value, errors="coerce").astype(float)
    return (c - z).abs().fillna(0.0).rename("dist_to_zz_high").to_frame()


@register_feature("compute_dist_to_zz_low", category="derived")
def compute_dist_to_zz_low(
    df: pd.DataFrame,
    price_col: str = "close",
    zz_low_col: str = "zz_low_value",
) -> pd.Series:
    """
    计算到 ZigZag 低点的距离（两个特征的差值：abs(price - zz_low)）
    
    Args:
        df: DataFrame with base features
        price_col: Price column
        zz_low_col: ZigZag low value column
    
    Returns:
        Series with dist_to_zz_low
    
    Raises:
        ValueError: 如果依赖列不存在
    """
    missing_cols = [c for c in [price_col, zz_low_col] if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Required columns {missing_cols} not found for dist_to_zz_low. "
            f"Available columns: {list(df.columns)[:20]}..."
        )
    return (
        (df[price_col] - df[zz_low_col]).abs()
    ).fillna(0.0).rename("dist_to_zz_low")


@register_feature("compute_dist_to_zz_low_from_series", category="derived")
def compute_dist_to_zz_low_from_series(
    *, close: pd.Series, zz_low_value: pd.Series
) -> pd.DataFrame:
    """Narrow-IO entrypoint for dist_to_zz_low."""
    c = pd.to_numeric(close, errors="coerce").astype(float)
    z = pd.to_numeric(zz_low_value, errors="coerce").astype(float)
    return (c - z).abs().fillna(0.0).rename("dist_to_zz_low").to_frame()


@register_feature("compute_dist_to_zz_high_atr", category="derived")
def compute_dist_to_zz_high_atr(
    df: pd.DataFrame,
    dist_col: str = "dist_to_zz_high",
    atr_col: str = "atr",
) -> pd.Series:
    """
    计算到 ZigZag 高点的距离（归一化到 ATR：dist / atr）
    
    Args:
        df: DataFrame with base features
        dist_col: Distance to ZZ high column
        atr_col: ATR column
    
    Returns:
        Series with dist_to_zz_high_atr
    
    Raises:
        ValueError: 如果依赖列不存在
    """
    missing_cols = [c for c in [dist_col, atr_col] if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Required columns {missing_cols} not found for dist_to_zz_high_atr. "
            f"Available columns: {list(df.columns)[:20]}..."
        )
    return (
        df[dist_col] / df[atr_col].replace(0, np.nan)
    ).fillna(0.0).rename("dist_to_zz_high_atr")


@register_feature("compute_dist_to_zz_high_atr_from_series", category="derived")
def compute_dist_to_zz_high_atr_from_series(
    *, dist_to_zz_high: pd.Series, atr: pd.Series
) -> pd.DataFrame:
    """Narrow-IO entrypoint for dist_to_zz_high_atr."""
    dist = pd.to_numeric(dist_to_zz_high, errors="coerce").astype(float)
    atr_s = pd.to_numeric(atr, errors="coerce").astype(float).replace(0, np.nan)
    return (dist / atr_s).fillna(0.0).rename("dist_to_zz_high_atr").to_frame()


@register_feature("compute_dist_to_zz_low_atr", category="derived")
def compute_dist_to_zz_low_atr(
    df: pd.DataFrame,
    dist_col: str = "dist_to_zz_low",
    atr_col: str = "atr",
) -> pd.Series:
    """
    计算到 ZigZag 低点的距离（归一化到 ATR：dist / atr）
    
    Args:
        df: DataFrame with base features
        dist_col: Distance to ZZ low column
        atr_col: ATR column
    
    Returns:
        Series with dist_to_zz_low_atr
    
    Raises:
        ValueError: 如果依赖列不存在
    """
    missing_cols = [c for c in [dist_col, atr_col] if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Required columns {missing_cols} not found for dist_to_zz_low_atr. "
            f"Available columns: {list(df.columns)[:20]}..."
        )
    return (
        df[dist_col] / df[atr_col].replace(0, np.nan)
    ).fillna(0.0).rename("dist_to_zz_low_atr")


@register_feature("compute_dist_to_zz_low_atr_from_series", category="derived")
def compute_dist_to_zz_low_atr_from_series(
    *, dist_to_zz_low: pd.Series, atr: pd.Series
) -> pd.DataFrame:
    """Narrow-IO entrypoint for dist_to_zz_low_atr."""
    dist = pd.to_numeric(dist_to_zz_low, errors="coerce").astype(float)
    atr_s = pd.to_numeric(atr, errors="coerce").astype(float).replace(0, np.nan)
    return (dist / atr_s).fillna(0.0).rename("dist_to_zz_low_atr").to_frame()


@register_feature("compute_cvd_slope", category="derived")
def compute_cvd_slope(
    df: pd.DataFrame,
    cvd_col: str = "cvd",
    window: int = 5,
) -> pd.Series:
    """
    计算 CVD 斜率（单个特征的滚动变换）
    
    Args:
        df: DataFrame with base features
        cvd_col: CVD column
        window: Rolling window size
    
    Returns:
        Series with cvd_slope
    
    Raises:
        ValueError: 如果依赖列不存在
    """
    if cvd_col not in df.columns:
        raise ValueError(
            f"Required column '{cvd_col}' not found for cvd_slope_{window}. "
            f"Available columns: {list(df.columns)[:20]}..."
        )
    if len(df) <= window:
        raise ValueError(
            f"DataFrame length ({len(df)}) must be greater than window ({window}) for cvd_slope_{window}"
        )
    
    def _compute_slope(x):
        if len(x) > 1:
            return np.polyfit(range(len(x)), x, 1)[0]
        return 0.0
    
    return (
        df[cvd_col]
        .rolling(window=window, min_periods=1)
        .apply(_compute_slope)
        .fillna(0.0)
        .rename(f"cvd_slope_{window}")
    )


@register_feature("compute_cvd_slope_from_series", category="derived")
def compute_cvd_slope_from_series(*, cvd: pd.Series, window: int = 5) -> pd.DataFrame:
    """Narrow-IO entrypoint for cvd_slope_{window}."""
    s = pd.to_numeric(cvd, errors="coerce").astype(float)
    if len(s) <= window:
        # keep legacy behavior of raising in df version, but return zeros for short series to keep pipeline robust
        out = pd.Series(0.0, index=s.index, name=f"cvd_slope_{window}")
        return out.to_frame()

    def _compute_slope(x):
        if len(x) > 1:
            return np.polyfit(range(len(x)), x, 1)[0]
        return 0.0

    out = (
        s.rolling(window=window, min_periods=1)
        .apply(_compute_slope)
        .fillna(0.0)
        .rename(f"cvd_slope_{window}")
    )
    return out.to_frame()


@register_feature("compute_path_efficiency_slope_from_series", category="derived")
def compute_path_efficiency_slope_from_series(
    *, path_efficiency_pct: pd.Series, window: int = 5
) -> pd.DataFrame:
    """
    Compute slope of path_efficiency_pct over rolling window.
    
    Used to detect "trendy but failing" conditions: efficiency is declining.
    Negative slope indicates efficiency is decreasing (trend is weakening).
    
    Args:
        path_efficiency_pct: Percentile rank of path efficiency (0-1)
        window: Rolling window size for slope calculation
    
    Returns:
        DataFrame with path_efficiency_slope_{window} column
    """
    s = pd.to_numeric(path_efficiency_pct, errors="coerce").astype(float)
    if len(s) <= window:
        out = pd.Series(0.0, index=s.index, name=f"path_efficiency_slope_{window}")
        return out.to_frame()

    def _compute_slope(x):
        if len(x) > 1:
            return np.polyfit(range(len(x)), x, 1)[0]
        return 0.0

    out = (
        s.rolling(window=window, min_periods=1)
        .apply(_compute_slope)
        .fillna(0.0)
        .rename(f"path_efficiency_slope_{window}")
    )
    return out.to_frame()


@register_feature("compute_price_dir_consistency_slope_from_series", category="derived")
def compute_price_dir_consistency_slope_from_series(
    *, price_dir_consistency_pct: pd.Series, window: int = 5
) -> pd.DataFrame:
    """
    Compute slope of price_dir_consistency_pct over rolling window.
    
    Used to detect "trendy but failing" conditions: consistency is declining.
    Negative slope indicates consistency is decreasing (trend is weakening).
    
    Args:
        price_dir_consistency_pct: Percentile rank of price direction consistency (0-1)
        window: Rolling window size for slope calculation
    
    Returns:
        DataFrame with price_dir_consistency_slope_{window} column
    """
    s = pd.to_numeric(price_dir_consistency_pct, errors="coerce").astype(float)
    if len(s) <= window:
        out = pd.Series(0.0, index=s.index, name=f"price_dir_consistency_slope_{window}")
        return out.to_frame()

    def _compute_slope(x):
        if len(x) > 1:
            return np.polyfit(range(len(x)), x, 1)[0]
        return 0.0

    out = (
        s.rolling(window=window, min_periods=1)
        .apply(_compute_slope)
        .fillna(0.0)
        .rename(f"price_dir_consistency_slope_{window}")
    )
    return out.to_frame()


@register_feature("compute_atr_ratio", category="derived")
def compute_atr_ratio(
    df: pd.DataFrame,
    atr_col: str = "atr",
    price_col: str = "close",
) -> pd.Series:
    """
    计算 ATR 比率（两个特征的比值：atr / price）
    
    Args:
        df: DataFrame with base features
        atr_col: ATR column
        price_col: Price column
    
    Returns:
        Series with atr_ratio
    
    Raises:
        ValueError: 如果依赖列不存在
    """
    missing_cols = [c for c in [atr_col, price_col] if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Required columns {missing_cols} not found for atr_ratio. "
            f"Available columns: {list(df.columns)[:20]}..."
        )
    return (
        df[atr_col] / df[price_col].replace(0, np.nan)
    ).fillna(0.0).rename("atr_ratio")


@register_feature("compute_atr_ratio_from_series", category="derived")
def compute_atr_ratio_from_series(
    *,
    atr: pd.Series,
    close: pd.Series,
) -> pd.Series:
    """Narrow-input ATR ratio: atr / close."""
    atr = pd.to_numeric(atr, errors="coerce").astype(float)
    close = pd.to_numeric(close, errors="coerce").astype(float)
    return (atr / close.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0).rename(
        "atr_ratio"
    )


@register_feature("compute_macdext_atr_normalized_from_series", category="derived")
def compute_macdext_atr_normalized_from_series(
    *,
    macdext: pd.Series,
    macdext_signal: pd.Series,
    macdext_histogram: pd.Series,
    close: pd.Series,
    atr: pd.Series,
) -> pd.DataFrame:
    """
    Normalize MACDEXT outputs by ATR (unitless), without changing upstream macdext_f semantics.

    Background:
    - In this repo, `macdext_f` is normalized by close (relative_close), so its outputs are ~unitless:
        macdext_rel = macdext_raw / close
    - To get a true ATR-normalized MACD (also unitless), we reconstruct:
        macdext_raw/atr = (macdext_rel * close) / atr

    This keeps compatibility while providing a scale-free MACD/ATR version for multi-asset modeling.
    """
    macdext = pd.to_numeric(macdext, errors="coerce").astype(float)
    macdext_signal = pd.to_numeric(macdext_signal, errors="coerce").astype(float)
    macdext_histogram = pd.to_numeric(macdext_histogram, errors="coerce").astype(float)
    close = pd.to_numeric(close, errors="coerce").astype(float)
    atr = pd.to_numeric(atr, errors="coerce").astype(float).replace(0, np.nan)

    scale = (close / atr).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out = pd.DataFrame(index=macdext.index)
    out["macdext_atr_norm"] = (macdext * scale).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out["macdext_signal_atr_norm"] = (macdext_signal * scale).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out["macdext_histogram_atr_norm"] = (macdext_histogram * scale).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


@register_feature("compute_macdfix_atr_normalized_from_series", category="derived")
def compute_macdfix_atr_normalized_from_series(
    *,
    macdfix: pd.Series,
    macdfix_signal: pd.Series,
    macdfix_histogram: pd.Series,
    close: pd.Series,
    atr: pd.Series,
) -> pd.DataFrame:
    """
    Normalize MACDFIX outputs by ATR (unitless), without changing upstream macdfix_f semantics.

    Same idea as compute_macdext_atr_normalized_from_series.
    """
    macdfix = pd.to_numeric(macdfix, errors="coerce").astype(float)
    macdfix_signal = pd.to_numeric(macdfix_signal, errors="coerce").astype(float)
    macdfix_histogram = pd.to_numeric(macdfix_histogram, errors="coerce").astype(float)
    close = pd.to_numeric(close, errors="coerce").astype(float)
    atr = pd.to_numeric(atr, errors="coerce").astype(float).replace(0, np.nan)

    scale = (close / atr).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out = pd.DataFrame(index=macdfix.index)
    out["macdfix_atr_norm"] = (macdfix * scale).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out["macdfix_signal_atr_norm"] = (macdfix_signal * scale).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out["macdfix_histogram_atr_norm"] = (macdfix_histogram * scale).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


@register_feature("compute_bb_width_ratio", category="derived")
def compute_bb_width_ratio(
    df: pd.DataFrame,
    bb_upper_col: str = "bb_upper",
    bb_lower_col: str = "bb_lower",
    bb_middle_col: str = "bb_middle",
) -> pd.Series:
    """
    计算 Bollinger Band 宽度比率（多个特征的组合：(upper - lower) / middle）
    
    Args:
        df: DataFrame with base features
        bb_upper_col: BB upper column
        bb_lower_col: BB lower column
        bb_middle_col: BB middle column
    
    Returns:
        Series with bb_width_ratio
    
    Raises:
        ValueError: 如果依赖列不存在
    """
    required_cols = [bb_upper_col, bb_lower_col, bb_middle_col]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Required columns {missing_cols} not found for bb_width_ratio. "
            f"Available columns: {list(df.columns)[:20]}..."
        )
    bb_width = (
        (df[bb_upper_col] - df[bb_lower_col]) / df[bb_middle_col].replace(0, np.nan)
    ).fillna(0.0)
    return bb_width.rename("bb_width_ratio")


@register_feature("compute_bb_width_ratio_from_series", category="derived")
def compute_bb_width_ratio_from_series(
    *,
    bb_upper: pd.Series,
    bb_lower: pd.Series,
    bb_middle: pd.Series,
) -> pd.Series:
    """Narrow-input BB width ratio: (upper - lower) / middle."""
    bb_upper = pd.to_numeric(bb_upper, errors="coerce").astype(float)
    bb_lower = pd.to_numeric(bb_lower, errors="coerce").astype(float)
    bb_middle = pd.to_numeric(bb_middle, errors="coerce").astype(float)
    out = (bb_upper - bb_lower) / bb_middle.replace(0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0).rename("bb_width_ratio")


@register_feature("compute_bb_width_ratio_from_price_from_series", category="derived")
def compute_bb_width_ratio_from_price_from_series(
    *,
    close: pd.Series,
    timeperiod: int = 20,
    nbdevup: float = 2.0,
    nbdevdn: float = 2.0,
    matype: int = 0,
) -> pd.Series:
    """
    BB width ratio directly from close (no dependency on intermediate raw-price BB columns).

    This exists because `bb_width_f` intentionally does NOT expose raw-price band levels
    (`bb_upper/bb_middle/bb_lower`) to keep the feature registry fully normalized.
    """
    import talib

    close = pd.to_numeric(close, errors="coerce").astype(float)
    upper, middle, lower = talib.BBANDS(
        close.values,
        timeperiod=int(timeperiod),
        nbdevup=float(nbdevup),
        nbdevdn=float(nbdevdn),
        matype=int(matype),
    )
    upper_s = pd.Series(upper, index=close.index)
    middle_s = pd.Series(middle, index=close.index).replace(0, np.nan)
    lower_s = pd.Series(lower, index=close.index)
    out = (upper_s - lower_s) / middle_s
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0).rename("bb_width_ratio")


@register_feature("compute_compression_score", category="derived")
def compute_compression_score(
    df: pd.DataFrame,
    bb_width_ratio_col: str = "bb_width_ratio",
) -> pd.Series:
    """
    计算压缩度分数（单个特征的变换：1 / (1 + bb_width_ratio)）
    
    Args:
        df: DataFrame with base features
        bb_width_ratio_col: BB width ratio column
    
    Returns:
        Series with compression_score
    
    Raises:
        ValueError: 如果依赖列不存在
    """
    if bb_width_ratio_col in df.columns:
        return (
            1.0 / (1.0 + df[bb_width_ratio_col])
        ).fillna(0.0).rename("compression_score")
    # 如果没有 bb_width_ratio，尝试从 BB 列计算
    bb_cols = ["bb_upper", "bb_lower", "bb_middle"]
    if all(col in df.columns for col in bb_cols):
        bb_width = (
            (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"].replace(0, np.nan)
        ).fillna(0.0)
        return (1.0 / (1.0 + bb_width)).fillna(0.0).rename("compression_score")
    
    # 如果都不存在，报错
    raise ValueError(
        f"Required column '{bb_width_ratio_col}' or BB columns {bb_cols} not found for compression_score. "
        f"Available columns: {list(df.columns)[:20]}..."
    )


@register_feature("compute_compression_score_from_series", category="derived")
def compute_compression_score_from_series(*, bb_width_ratio: pd.Series) -> pd.Series:
    """Narrow-input compression_score: 1 / (1 + bb_width_ratio)."""
    bb_width_ratio = pd.to_numeric(bb_width_ratio, errors="coerce").astype(float)
    out = 1.0 / (1.0 + bb_width_ratio)
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0).rename("compression_score")


@register_feature("compute_tbr_ma", category="derived")
def compute_tbr_ma(
    df: pd.DataFrame,
    tbr_col: str = "taker_buy_ratio",
    window: int = 5,
) -> pd.Series:
    """
    计算 TBR 移动平均（单个特征的滚动变换）
    
    Args:
        df: DataFrame with base features
        tbr_col: TBR column
        window: Moving average window
    
    Returns:
        Series with tbr_ma
    
    Raises:
        ValueError: 如果依赖列不存在
    """
    if tbr_col not in df.columns:
        raise ValueError(
            f"Required column '{tbr_col}' not found for tbr_ma_{window}. "
            f"Available columns: {list(df.columns)[:20]}..."
        )
    return (
        df[tbr_col].rolling(window=window, min_periods=1).mean()
    ).fillna(0.5).rename(f"tbr_ma_{window}")


@register_feature("compute_tbr_ma_from_series", category="derived")
def compute_tbr_ma_from_series(*, taker_buy_ratio: pd.Series, window: int = 5) -> pd.Series:
    """Narrow-input TBR moving average."""
    tbr = pd.to_numeric(taker_buy_ratio, errors="coerce").astype(float)
    out = tbr.rolling(window=window, min_periods=1).mean()
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.5).rename(f"tbr_ma_{window}")


@register_feature("compute_tbr_spike", category="derived")
def compute_tbr_spike(
    df: pd.DataFrame,
    tbr_col: str = "taker_buy_ratio",
    tbr_ma_col: str = "tbr_ma_5",
    spike_threshold: float = 1.5,
) -> pd.Series:
    """
    计算 TBR 突增信号（两个特征的比较：tbr > tbr_ma * threshold）
    
    Args:
        df: DataFrame with base features
        tbr_col: TBR column
        tbr_ma_col: TBR moving average column
        spike_threshold: Spike threshold multiplier
    
    Returns:
        Series with tbr_spike
    
    Raises:
        ValueError: 如果依赖列不存在
    """
    if tbr_col not in df.columns:
        raise ValueError(
            f"Required column '{tbr_col}' not found for tbr_spike. "
            f"Available columns: {list(df.columns)[:20]}..."
        )
    
    if tbr_ma_col in df.columns:
        tbr_ma = df[tbr_ma_col]
    else:
        # 如果没有 tbr_ma，尝试计算
        tbr_ma = df[tbr_col].rolling(window=5, min_periods=1).mean()
    
    spike = (df[tbr_col] > tbr_ma * spike_threshold).astype(float)
    return spike.rename("tbr_spike")


@register_feature("compute_tbr_spike_from_series", category="derived")
def compute_tbr_spike_from_series(
    *,
    taker_buy_ratio: pd.Series,
    tbr_ma_5: pd.Series,
    spike_threshold: float = 1.5,
) -> pd.Series:
    """Narrow-input TBR spike: taker_buy_ratio > tbr_ma_5 * threshold."""
    tbr = pd.to_numeric(taker_buy_ratio, errors="coerce").astype(float)
    ma = pd.to_numeric(tbr_ma_5, errors="coerce").astype(float)
    out = (tbr > ma * float(spike_threshold)).astype(float)
    out.name = "tbr_spike"
    return out


# ========================================================================
# 向后兼容：保留旧的批量函数（但推荐使用独立函数）
# ========================================================================


# 向后兼容：保留旧的批量函数（但推荐使用独立函数）
def build_interaction_features(
    df: pd.DataFrame,
    interaction_config: Optional[dict] = None,
) -> pd.DataFrame:
    """
    构建特征交互项（批量版本，向后兼容）
    
    推荐：使用独立的计算函数，在 feature_dependencies.yaml 中单独定义
    """
    df = df.copy()
    
    # 使用独立函数计算
    if "liquidity_void_detected" in df.columns or "wpt_false_breakout_risk" in df.columns:
        df["liquidity_void_x_wpt_risk"] = compute_liquidity_void_x_wpt_risk(df)
    
    if "compression_energy" in df.columns or "ofi_short" in df.columns:
        df["compression_energy_x_ofi_short"] = compute_compression_energy_x_ofi_short(df)
    
    if "hurst_close_rolling" in df.columns or "trend_r2_20" in df.columns:
        df["hurst_x_trend_r2"] = compute_hurst_x_trend_r2(df)
    
    if "evt_tail_shape" in df.columns or "trend_r2_20" in df.columns:
        df["evt_x_trend_r2"] = compute_evt_x_trend_r2(df)
    
    if "vpin" in df.columns:
        if "compression_energy" in df.columns:
            df["vpin_x_compression"] = compute_vpin_x_compression(df)
        if "wick_upper_ratio" in df.columns:
            df["vpin_x_wick_upper"] = compute_vpin_x_wick_upper(df)
        if "wick_lower_ratio" in df.columns:
            df["vpin_x_wick_lower"] = compute_vpin_x_wick_lower(df)
    
    if "sma_200_slope" in df.columns or "sma_200" in df.columns:
        df["sma_slope_x_price_pos"] = compute_sma_slope_x_price_pos(df)
    
    return df


def extract_interaction_features(
    df: pd.DataFrame,
    interaction_config: Optional[dict] = None,
    apply_rank: bool = True,
    groupby_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    提取特征交互项（完整流程，向后兼容）
    
    推荐：使用独立的计算函数，在 feature_dependencies.yaml 中单独定义
    """
    df = build_interaction_features(df, interaction_config)
    
    if apply_rank:
        interaction_cols = [col for col in df.columns if "_x_" in col]
        for col in interaction_cols:
            df[f"{col}_rank"] = apply_rank_transform_to_interaction(df, col, groupby_col)
    
    return df


@register_feature("compute_is_near_sr", category="derived")
def compute_is_near_sr(
    df: pd.DataFrame,
    dist_col: str = "dist_to_nearest_sr",
    atr_col: str = "atr",
    price_col: str = "close",
    dist_atr_mult: float = 1.5,
) -> pd.Series:
    """
    计算是否在SR附近的布尔列。
    
    基于 dist_to_nearest_sr 和 ATR，判断当前价格是否在SR附近（距离 <= dist_atr_mult * ATR）。
    
    注意：dist_to_nearest_sr 是相对百分比（如 0.05 表示 5%），需要转换为绝对价格距离后再与 ATR 比较。
    
    Args:
        df: DataFrame with base features
        dist_col: Distance to nearest SR column (default: "dist_to_nearest_sr")
        atr_col: ATR column (default: "atr")
        price_col: Price column for converting percentage to absolute distance (default: "close")
        dist_atr_mult: Distance threshold in ATR multiples (default: 1.5)
    
    Returns:
        Series with is_near_sr (boolean)
    
    Raises:
        ValueError: 如果依赖列不存在
    """
    missing_cols = [c for c in [dist_col, atr_col, price_col] if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Required columns {missing_cols} not found for is_near_sr. "
            f"Available columns: {list(df.columns)[:20]}..."
        )
    
    # dist_to_nearest_sr 是相对百分比（如 0.05 表示 5%）
    dist_to_sr_pct = df[dist_col].abs()
    atr = df[atr_col].fillna(df[atr_col].median())
    price = df[price_col]
    
    # 将百分比距离转换为绝对价格距离
    # 例如：dist_to_sr = 0.05 (5%), price = 100 -> abs_distance = 5
    abs_distance = dist_to_sr_pct * price
    
    # 计算归一化距离（单位：ATR）
    # 例如：abs_distance = 5, atr = 10 -> dist_normalized = 0.5 ATR
    dist_normalized = abs_distance / (atr + 1e-8)
    
    # 判断是否在SR附近
    is_near = dist_normalized <= dist_atr_mult
    
    return is_near.fillna(False).astype(bool).rename("is_near_sr")


@register_feature("compute_is_near_sr_from_series", category="derived")
def compute_is_near_sr_from_series(
    *,
    dist_to_nearest_sr: pd.Series,
    atr: pd.Series,
    close: pd.Series,
    dist_atr_mult: float = 1.5,
) -> pd.DataFrame:
    """
    Narrow-IO entrypoint for is_near_sr.
    
    注意：dist_to_nearest_sr 是相对百分比，需要转换为绝对价格距离后再与 ATR 比较。
    """
    dist_pct = pd.to_numeric(dist_to_nearest_sr, errors="coerce").abs()
    atr_s = pd.to_numeric(atr, errors="coerce").fillna(atr.median())
    price = pd.to_numeric(close, errors="coerce")
    
    # 将百分比距离转换为绝对价格距离
    abs_distance = dist_pct * price
    
    # 计算归一化距离（单位：ATR）
    dist_normalized = abs_distance / (atr_s + 1e-8)
    
    # 判断是否在SR附近
    is_near = (dist_normalized <= dist_atr_mult).fillna(False).astype(bool)
    return is_near.rename("is_near_sr").to_frame()


# =============================================================================
# DTW Scene Semantic Scores
# =============================================================================

@register_feature("compute_dtw_scene_semantic_scores_from_series", category="interaction")
def compute_dtw_scene_semantic_scores_from_series(
    *,
    # DTW 反转形态（看涨）
    dtw_hammer_dist_w15: Optional[pd.Series] = None,
    dtw_head_shoulder_bottom_dist_w15: Optional[pd.Series] = None,
    dtw_double_bottom_dist_w15: Optional[pd.Series] = None,
    dtw_bullish_engulfing_dist_w15: Optional[pd.Series] = None,
    # DTW 反转形态（看跌）
    dtw_shooting_star_dist_w15: Optional[pd.Series] = None,
    dtw_head_shoulder_top_dist_w15: Optional[pd.Series] = None,
    dtw_double_top_dist_w15: Optional[pd.Series] = None,
    dtw_bearish_engulfing_dist_w15: Optional[pd.Series] = None,
    # DTW 趋势形态
    dtw_bull_flag_dist_w25: Optional[pd.Series] = None,
    dtw_bear_flag_dist_w25: Optional[pd.Series] = None,
    dtw_triangle_dist_w25: Optional[pd.Series] = None,
    # 上下文
    compression_score: Optional[pd.Series] = None,
    trend_r2_20: Optional[pd.Series] = None,
    dist_to_nearest_sr: Optional[pd.Series] = None,
    # 参数
    dist_scale: float = 0.5,  # DTW距离归一化尺度
    sr_prox_threshold: float = 0.02,  # SR接近阈值（百分比）
) -> pd.DataFrame:
    """
    DTW Pattern Scene Semantic Scores (0..1).
    
    将 DTW 距离转换为语义场景分数：
    - dtw_reversal_bullish_score: 看涨反转形态匹配度（适用于 SR 反转做多）
    - dtw_reversal_bearish_score: 看跌反转形态匹配度（适用于 SR 反转做空）
    - dtw_continuation_bullish_score: 看涨延续形态匹配度（适用于趋势跟踪做多）
    - dtw_continuation_bearish_score: 看跌延续形态匹配度（适用于趋势跟踪做空）
    - dtw_compression_score: 压缩待突破形态（三角形 + 压缩上下文）
    - dtw_exhaustion_score: 衰竭形态（顶/底 + 趋势衰减）
    
    设计原理：
    - DTW距离越小 = 匹配度越高 = 分数越高
    - 使用 exp(-dist / scale) 将距离转换为 (0,1] 分数
    - 结合上下文（compression、trend、SR接近度）进行门控
    """
    eps = 1e-8
    scale = max(float(dist_scale), 0.1)
    sr_thr = max(float(sr_prox_threshold), 0.001)
    
    def dist_to_score(dist_series: Optional[pd.Series]) -> pd.Series:
        """将DTW距离转换为匹配度分数 (0,1]"""
        if dist_series is None:
            return pd.Series(0.0, index=pd.RangeIndex(1))
        d = pd.to_numeric(dist_series, errors="coerce").fillna(1.0).astype(float)
        # exp(-dist/scale): dist=0 -> 1.0, dist=scale -> ~0.37, dist=2*scale -> ~0.14
        return np.exp(-d.clip(0.0) / scale).clip(0.0, 1.0)
    
    # 获取索引（从任意非空序列）
    idx = None
    for s in [dtw_hammer_dist_w15, dtw_bull_flag_dist_w25, compression_score, trend_r2_20]:
        if s is not None:
            idx = s.index
            break
    if idx is None:
        idx = pd.RangeIndex(1)
    
    # ============ 反转形态（看涨）============
    bullish_reversal_patterns = [
        dist_to_score(dtw_hammer_dist_w15),
        dist_to_score(dtw_head_shoulder_bottom_dist_w15),
        dist_to_score(dtw_double_bottom_dist_w15),
        dist_to_score(dtw_bullish_engulfing_dist_w15),
    ]
    # 取最大值（最匹配的形态）
    bullish_rev = pd.concat([s.reindex(idx).fillna(0.0) for s in bullish_reversal_patterns], axis=1).max(axis=1)
    
    # ============ 反转形态（看跌）============
    bearish_reversal_patterns = [
        dist_to_score(dtw_shooting_star_dist_w15),
        dist_to_score(dtw_head_shoulder_top_dist_w15),
        dist_to_score(dtw_double_top_dist_w15),
        dist_to_score(dtw_bearish_engulfing_dist_w15),
    ]
    bearish_rev = pd.concat([s.reindex(idx).fillna(0.0) for s in bearish_reversal_patterns], axis=1).max(axis=1)
    
    # ============ 趋势延续形态 ============
    bull_cont = dist_to_score(dtw_bull_flag_dist_w25).reindex(idx).fillna(0.0)
    bear_cont = dist_to_score(dtw_bear_flag_dist_w25).reindex(idx).fillna(0.0)
    triangle = dist_to_score(dtw_triangle_dist_w25).reindex(idx).fillna(0.0)
    
    # ============ 上下文门控 ============
    # SR接近度门控（反转形态需要接近SR）
    sr_gate = 1.0
    if dist_to_nearest_sr is not None:
        sr_dist = pd.to_numeric(dist_to_nearest_sr, errors="coerce").abs().reindex(idx).fillna(1.0)
        # 距离越小，gate越高
        sr_gate = np.exp(-sr_dist / sr_thr).clip(0.0, 1.0)
    
    # 压缩度门控（三角形需要压缩上下文）
    comp_gate = 0.5  # 默认中性
    if compression_score is not None:
        comp_gate = pd.to_numeric(compression_score, errors="coerce").reindex(idx).fillna(0.5).clip(0.0, 1.0)
    
    # 趋势门控（延续形态需要趋势上下文）
    trend_gate = 0.5
    trend_end_gate = 0.5
    if trend_r2_20 is not None:
        r2 = pd.to_numeric(trend_r2_20, errors="coerce").reindex(idx).fillna(0.5).clip(0.0, 1.0)
        trend_gate = r2
        trend_end_gate = (1.0 - r2).clip(0.0, 1.0)
    
    # ============ 组装语义分数 ============
    # 反转形态：强调SR接近 + 趋势衰减
    dtw_rev_bull = (bullish_rev * sr_gate * (0.5 + 0.5 * trend_end_gate)).clip(0.0, 1.0)
    dtw_rev_bear = (bearish_rev * sr_gate * (0.5 + 0.5 * trend_end_gate)).clip(0.0, 1.0)
    
    # 延续形态：强调趋势上下文
    dtw_cont_bull = (bull_cont * (0.5 + 0.5 * trend_gate)).clip(0.0, 1.0)
    dtw_cont_bear = (bear_cont * (0.5 + 0.5 * trend_gate)).clip(0.0, 1.0)
    
    # 压缩形态：三角形 + 压缩上下文
    dtw_compression = (triangle * (0.5 + 0.5 * comp_gate)).clip(0.0, 1.0)
    
    # 衰竭形态：顶/底 + 趋势衰减
    tops_bottoms = pd.concat([
        dist_to_score(dtw_head_shoulder_top_dist_w15).reindex(idx).fillna(0.0),
        dist_to_score(dtw_head_shoulder_bottom_dist_w15).reindex(idx).fillna(0.0),
        dist_to_score(dtw_double_top_dist_w15).reindex(idx).fillna(0.0),
        dist_to_score(dtw_double_bottom_dist_w15).reindex(idx).fillna(0.0),
    ], axis=1).max(axis=1)
    dtw_exhaustion = (tops_bottoms * trend_end_gate).clip(0.0, 1.0)
    
    return pd.DataFrame({
        "dtw_reversal_bullish_score": dtw_rev_bull,
        "dtw_reversal_bearish_score": dtw_rev_bear,
        "dtw_continuation_bullish_score": dtw_cont_bull,
        "dtw_continuation_bearish_score": dtw_cont_bear,
        "dtw_compression_score": dtw_compression,
        "dtw_exhaustion_score": dtw_exhaustion,
    })


# ============================================================================
# CVD Divergence V2: 连续化背离特征
# ============================================================================

@register_feature("compute_cvd_divergence_v2_from_series", category="interaction")
def compute_cvd_divergence_v2_from_series(
    *,
    close: pd.Series,
    cvd: pd.Series,
    trend_strength: Optional[pd.Series] = None,
    position_window: int = 50,
    percentile_window: int = 540,
) -> pd.DataFrame:
    """
    CVD Divergence V2: 连续化背离特征
    
    相比 V1 (Bool) 的改进:
    1. 用 rank 替代 min-max，避免极值抖动
    2. 输出连续值，每 bar 都有值
    3. 新增 3 个工业级复合特征，区分"健康背离" vs "反转背离"
    
    Args:
        close: 收盘价序列
        cvd: CVD 序列
        trend_strength: 趋势强度 [-1, 1]，正=上行，负=下行（可选）
        position_window: 位置计算窗口
        percentile_window: 百分位历史窗口
    
    Returns:
        DataFrame with columns:
        - cvd_divergence_score: 背离得分 [-1, 1]，正=看涨背离，负=看跌背离
        - cvd_divergence_score_pct: 背离得分的历史百分位 [0, 1]
        - price_position: 价格在窗口内的相对位置 [0, 1]
        - trend_div_alignment: 趋势-背离对齐度（顺趋势 vs 逆趋势）
        - trend_div_tension: 趋势-背离张力（冲突强度，sqrt 非线性增强）
        - div_location_pressure: 背离位置压力（极端位置反转潜力）
    
    Notes:
        - NaN position is treated as neutral (mid-range=0.5) to avoid artificial reversal bias.
        - trend_div_tension uses sqrt() for better sensitivity to small conflicts.
        - Percentile rank uses <= comparison for stable handling of repeated values.
    """
    w = max(position_window, 10)
    price = pd.to_numeric(close, errors="coerce").astype(float)
    cvd_s = pd.to_numeric(cvd, errors="coerce").astype(float)
    
    # ===== 1. 计算相对位置（用 rank 替代 min-max）=====
    def _rolling_percentile_rank(series: pd.Series, window: int) -> pd.Series:
        """
        计算滚动窗口内的百分位排名 [0, 1]
        
        使用 <= 比较，确保：
        - 最小值 → 0
        - 最大值 → 1
        - 重复值 → 中性分布（不会系统性偏低）
        """
        def _pct_rank(x):
            if len(x) < 2:
                return 0.5
            rank = (x <= x.iloc[-1]).sum() - 1
            return rank / (len(x) - 1)
        return series.rolling(window=window, min_periods=max(10, window // 3)).apply(_pct_rank, raw=False)
    
    price_position = _rolling_percentile_rank(price, w)
    cvd_position = _rolling_percentile_rank(cvd_s, w)
    
    # ===== 2. 背离得分 =====
    # 正值：CVD 相对强，价格相对弱 → 看涨背离（吸筹）
    # 负值：CVD 相对弱，价格相对强 → 看跌背离（派发）
    divergence_score = (cvd_position - price_position).fillna(0.0).clip(-1.0, 1.0)
    
    # 百分位版本（用于 Router/Tree/NN）
    divergence_score_pct = _rolling_percentile_rank(divergence_score, percentile_window)
    
    # ===== 3. 三个工业级复合特征 =====
    # 趋势强度：如果没提供，用 0（只保留基础特征）
    if trend_strength is not None:
        T = pd.to_numeric(trend_strength, errors="coerce").fillna(0.0).clip(-1.0, 1.0)
    else:
        T = pd.Series(0.0, index=price.index)
    
    D = divergence_score
    # NaN position → 0.5 (neutral mid-range) to avoid artificial reversal bias
    P = price_position.fillna(0.5)
    
    # Feature 1: Trend–Divergence Alignment（顺趋势 vs 逆趋势）
    # 正值 = 顺趋势背离（健康），负值 = 逆趋势背离（反转风险）
    trend_div_alignment = (D * T).fillna(0.0).clip(-1.0, 1.0)
    
    # Feature 2: Trend–Divergence Tension（冲突强度）
    # 使用 sqrt() 非线性：小冲突更敏感，大冲突不过饱和
    trend_div_tension = np.sqrt((D.abs() * T.abs()).fillna(0.0)).clip(0.0, 1.0)
    
    # Feature 3: Divergence Location Pressure（位置压力）
    # 高值 = 背离 + 极端位置（反转潜力高）
    div_location_pressure = (D.abs() * (P - 0.5).abs() * 2).fillna(0.0).clip(0.0, 1.0)
    
    return pd.DataFrame({
        "cvd_divergence_score": divergence_score,
        "cvd_divergence_score_pct": divergence_score_pct.fillna(0.5),
        "price_position": P,
        "trend_div_alignment": trend_div_alignment,
        "trend_div_tension": trend_div_tension,
        "div_location_pressure": div_location_pressure,
    })


@register_feature("compute_price_momentum_divergence_from_series", category="interaction")
def compute_price_momentum_divergence_from_series(
    *,
    close: pd.Series,
    trend_strength: Optional[pd.Series] = None,
    velocity_span: int = 6,
    accel_span: int = 4,
    position_window: int = 50,
    velocity_position_window: int = 30,
    percentile_window: int = 540,
) -> pd.DataFrame:
    """
    Price-Momentum Divergence: 价格-动量背离特征
    
    与 CVD Divergence V2 完全同构，但度量的是"价格自己的推进力"而非"行为支撑"。
    
    语义差异：
    - CVD Divergence: "有没有人在背后支持这个价格？"
    - Momentum Divergence: "价格自己还有没有力气？"
    
    这两个是**正交维度**，应该同时使用。
    
    Args:
        close: 收盘价序列
        trend_strength: 趋势强度 [-1, 1]（可选，用于复合特征）
        velocity_span: 速度 EMA 平滑窗口 (推荐: 1h=5~6, 4h=8~10)
        accel_span: 加速度 EMA 平滑窗口 (通常 < velocity_span)
        position_window: 价格位置计算窗口（结构位置，较稳定）
        velocity_position_window: 速度位置计算窗口（更短，对"突然失速"更敏感）
        percentile_window: 百分位历史窗口
    
    Returns:
        DataFrame with columns:
        - price_velocity_pct: 速度的历史百分位 [0, 1]
        - price_accel_pct: 加速度的历史百分位 [0, 1]
        - price_momentum_div_score: 背离得分 [-1, 1]，正=动量超前，负=动量衰竭
        - price_momentum_div_score_pct: 背离得分的历史百分位 [0, 1]
        - momentum_div_tension: 动量-趋势张力 [0, 1]（sqrt 非线性）
        - momentum_location_pressure: 动量背离位置压力 [0, 1]
    
    Notes:
        - price_velocity 不是 return，是 "去噪的一阶时间导数"
        - price_accel 用于识别趋势衰竭/末端拐点
        - velocity_position_window < position_window 让背离对"突然失速"更敏感
        - NaN position is treated as neutral (mid-range=0.5) to avoid artificial reversal bias
    
    Example:
        双重背离（强反转信号）:
        - CVD divergence ↓ + Momentum divergence ↓
        
        结构分歧（容易假信号）:
        - CVD 支持 + 动量衰竭 → 多半是震荡/洗盘
    """
    # === 1. 解析输入 ===
    price = pd.to_numeric(close, errors="coerce").astype(float)
    w_price = max(position_window, 10)
    w_velocity = max(velocity_position_window, 10)
    v_span = max(velocity_span, 2)
    a_span = max(accel_span, 2)
    
    # === 2. 计算价格速度（平滑的一阶导数） ===
    # price_velocity = EMA(price.diff())
    # 这不是 return，是物理量："价格推进的速度"
    raw_velocity = price.diff()
    price_velocity = raw_velocity.ewm(span=v_span, adjust=False).mean()
    
    # === 3. 计算价格加速度（平滑的二阶导数） ===
    # price_accel = EMA(price_velocity.diff())
    # 用于识别趋势衰竭、末端拐点
    raw_accel = price_velocity.diff()
    price_accel = raw_accel.ewm(span=a_span, adjust=False).mean()
    
    # === 4. 计算滚动百分位排名 ===
    def _rolling_percentile_rank(series: pd.Series, window: int) -> pd.Series:
        """滚动窗口内的百分位排名 [0, 1]，使用 <= 比较"""
        def _pct_rank(x):
            if len(x) < 2:
                return 0.5
            rank = (x <= x.iloc[-1]).sum() - 1
            return rank / (len(x) - 1)
        return series.rolling(window=window, min_periods=max(10, window // 3)).apply(_pct_rank, raw=False)
    
    # 价格位置（结构位置，较稳定）、速度位置（更短窗口，对失速更敏感）
    price_position = _rolling_percentile_rank(price, w_price)
    velocity_position = _rolling_percentile_rank(price_velocity, w_velocity)
    
    # 速度、加速度的历史百分位
    price_velocity_pct = _rolling_percentile_rank(price_velocity, percentile_window)
    price_accel_pct = _rolling_percentile_rank(price_accel, percentile_window)
    
    # === 5. 动量背离得分 ===
    # 正值：动量相对强，价格相对弱 → 推进潜力
    # 负值：动量相对弱，价格相对强 → 推进衰竭
    momentum_div_score = (velocity_position - price_position).fillna(0.0).clip(-1.0, 1.0)
    momentum_div_score_pct = _rolling_percentile_rank(momentum_div_score, percentile_window)
    
    # === 6. 复合特征（与 CVD V2 同构） ===
    if trend_strength is not None:
        T = pd.to_numeric(trend_strength, errors="coerce").fillna(0.0).clip(-1.0, 1.0)
    else:
        T = pd.Series(0.0, index=price.index)
    
    D = momentum_div_score
    P = price_position.fillna(0.5)  # NaN → 中性
    
    # Momentum-Trend Tension: sqrt 非线性增强
    momentum_div_tension = np.sqrt((D.abs() * T.abs()).fillna(0.0)).clip(0.0, 1.0)
    
    # Momentum Location Pressure: 背离 + 极端位置 → 反转潜力
    momentum_location_pressure = (D.abs() * (P - 0.5).abs() * 2).fillna(0.0).clip(0.0, 1.0)
    
    return pd.DataFrame({
        "price_velocity_pct": price_velocity_pct.fillna(0.5),
        "price_accel_pct": price_accel_pct.fillna(0.5),
        "price_momentum_div_score": momentum_div_score,
        "price_momentum_div_score_pct": momentum_div_score_pct.fillna(0.5),
        "momentum_div_tension": momentum_div_tension,
        "momentum_location_pressure": momentum_location_pressure,
    })


@register_feature("compute_terminal_risk_score_from_series", category="interaction")
def compute_terminal_risk_score_from_series(
    *,
    price_position: pd.Series,
    price_velocity_pct: pd.Series,
    price_accel_pct: pd.Series,
    cvd_divergence_score: pd.Series,
    div_location_pressure: pd.Series,
) -> pd.DataFrame:
    """
    Terminal Risk Score: 末端风险评分 [0,1]
    
    Execution 层专用的连续化风险评估，回答：
    "我现在追进去，有多大概率是在吃趋势最后一口甚至接盘？"
    
    末端风险 = 三件事同时成立（连续化）：
    1. 价格已经在结构高/低位（位置）
    2. 价格自己在衰竭（Momentum divergence）
    3. 行为也开始不支持（CVD divergence）
    
    语义：末端 = "价格在极端 + 自己没力 + 背后没人"
    
    Args:
        price_position: 价格位置 [0,1]
        price_velocity_pct: 速度百分位 [0,1]
        price_accel_pct: 加速度百分位 [0,1]
        cvd_divergence_score: CVD 背离得分 [-1,1]
        div_location_pressure: 背离位置压力 [0,1]
    
    Returns:
        DataFrame with columns:
        - momentum_exhaustion_score: 动量衰竭分数 [0,1]（价格自己没力）
        - cvd_exhaustion_score: 行为不支持分数 [0,1]（CVD 背离）
        - location_amplifier: 位置放大器 [0,1]（只在极端放大）
        - terminal_risk_score: 末端风险综合得分 [0,1]
    
    Usage in Execution:
        terminal_risk_score | Execution 含义
        -------------------|------------------
        < 0.2              | 安全推进段
        0.2 ~ 0.4          | 需要谨慎（减 size）
        0.4 ~ 0.6          | 高风险追单
        > 0.6              | 基本是末端 / 易反转
    
    Notes:
        - 这不是反转信号，只是追单风险评估
        - 只给 Execution 层，不给 Router 决策方向
        - 用于: size 缩放 / entry 延迟 / stop 收紧
    """
    # === 1. 解析输入 ===
    P = pd.to_numeric(price_position, errors="coerce").fillna(0.5).clip(0.0, 1.0)
    V_pct = pd.to_numeric(price_velocity_pct, errors="coerce").fillna(0.5).clip(0.0, 1.0)
    A_pct = pd.to_numeric(price_accel_pct, errors="coerce").fillna(0.5).clip(0.0, 1.0)
    CVD_div = pd.to_numeric(cvd_divergence_score, errors="coerce").fillna(0.0).clip(-1.0, 1.0)
    div_loc_pressure = pd.to_numeric(div_location_pressure, errors="coerce").fillna(0.0).clip(0.0, 1.0)
    
    # === 2. 动量衰竭分数（价格自己没力） ===
    # velocity 低 + accel 低 → 衰竭
    # 用 (1 - pct) 表示衰竭程度
    momentum_exhaustion_score = ((1 - V_pct) * (1 - A_pct)).clip(0.0, 1.0)
    
    # === 3. 行为不支持分数（CVD 背离） ===
    # cvd_divergence_score 的绝对值表示背离强度
    # 乘以 div_location_pressure 放大极端位置的背离
    cvd_exhaustion_score = (CVD_div.abs() * div_loc_pressure).clip(0.0, 1.0)
    
    # === 4. 位置放大器（只在极端放大） ===
    # 0.5 是中位，|P - 0.5| * 2 在极端位置 (0 或 1) 等于 1
    location_amplifier = ((P - 0.5).abs() * 2).clip(0.0, 1.0)
    
    # === 5. 末端风险综合得分 ===
    # 三者相乘：只有三件事都成立时才会高
    terminal_risk_score = (
        momentum_exhaustion_score * cvd_exhaustion_score * location_amplifier
    ).clip(0.0, 1.0)
    
    return pd.DataFrame({
        "momentum_exhaustion_score": momentum_exhaustion_score,
        "cvd_exhaustion_score": cvd_exhaustion_score,
        "location_amplifier": location_amplifier,
        "terminal_risk_score": terminal_risk_score,
    })

@register_feature("compute_volume_participation_score_from_series", category="interaction")
def compute_volume_participation_score_from_series(
    *,
    volume: pd.Series,
    activity_window: int = 540,
    velocity_span: int = 5,
    stability_short_window: int = 5,
    stability_long_window: int = 20,
) -> pd.DataFrame:
    """
    Volume Participation Score: 市场参与度/活跃度评分 [0,1]
    
    Execution 层专用的连续化参与度评估，回答：
    "现在市场有没有人在认真交易？"
    
    语义：
    - 0 → 市场很"空"，追单/突破极易假
    - 1 → 市场很"热"，执行质量好、滑点可控
    
    Args:
        volume: 成交量序列
        activity_window: 活跃度历史窗口
        velocity_span: 速度EMA平滑窗口
        stability_short_window: 稳定性短期窗口
        stability_long_window: 稳定性长期窗口
    
    Returns:
        DataFrame with columns:
        - volume_activity_pct: 活跃度百分位 [0,1]（当前成交量在历史中的热度）
        - volume_velocity_pct: 速度百分位 [0,1]（突然开始交易的迹象）
        - volume_stability: 稳定性 [0,1]（不是一根假量）
        - volume_participation_score: 参与度综合得分 [0,1]
    
    Usage in Execution:
        volume_participation_score | Execution 含义
        -------------------------|------------------
        < 0.2                    | 市场冷清，避免追单
        0.2 ~ 0.4                | 一般参与度，谨慎执行
        0.4 ~ 0.7                | 良好参与度，可执行
        > 0.7                    | 高参与度，优质执行环境
    
    Notes:
        - 只给 Execution 层，不给 Router 决策方向
        - 用于: size 缩放 / entry 时机 / stop 设置
        - 与 terminal_risk_score 组合使用效果最佳
    """
    from src.features.live_tail_context import live_tail_slice_len

    _full_index = volume.index
    _slice_len = live_tail_slice_len(
        len(volume),
        warmup=int(activity_window)
        + int(velocity_span)
        + int(stability_long_window)
        + 1,
    )
    if _slice_len is not None:
        volume = volume.iloc[-_slice_len:]

    # === 1. 解析输入 ===
    vol = pd.to_numeric(volume, errors="coerce").astype(float)

    # === 2. 计算滚动百分位排名 ===
    def _rolling_percentile_rank(series: pd.Series, window: int) -> pd.Series:
        """滚动窗口内的百分位排名 [0, 1]，使用 <= 比较"""

        def _pct_rank(x):
            if len(x) < 2:
                return 0.5
            rank = (x <= x.iloc[-1]).sum() - 1
            return rank / (len(x) - 1)

        return series.rolling(window=window, min_periods=max(10, window // 3)).apply(
            _pct_rank, raw=False
        )

    # === 3. Volume 活跃度（相对历史）===
    volume_activity_pct = _rolling_percentile_rank(vol, activity_window)

    # === 4. Volume 加速度（突然有人）===
    raw_velocity = vol.diff()
    volume_velocity = raw_velocity.ewm(span=velocity_span, adjust=False).mean()
    volume_velocity_pct = _rolling_percentile_rank(volume_velocity, activity_window)

    # === 5. Volume 稳定性（不是一根假冲）===
    short_avg = vol.rolling(stability_short_window).mean()
    long_avg = vol.rolling(stability_long_window).mean()
    # 避免除零，确保比率有意义
    volume_stability = (short_avg / (long_avg + 1e-8)).clip(0.0, 2.0) / 2.0
    volume_stability = volume_stability.fillna(0.5)  # NaN 处理

    # === 6. 最终：参与度综合得分 ===
    # 使用 sqrt 对速度进行非线性处理，避免过度放大
    # 确保所有输入都经过NaN处理
    volume_activity_clean = volume_activity_pct.fillna(0.5)
    volume_velocity_clean = volume_velocity_pct.fillna(0.5)
    volume_stability_clean = volume_stability.fillna(0.5)

    volume_participation_score = (
        volume_activity_clean
        * np.sqrt(np.abs(volume_velocity_clean))
        * volume_stability_clean
    ).clip(0.0, 1.0)

    result = pd.DataFrame(
        {
            "volume_activity_pct": volume_activity_pct.fillna(0.5),
            "volume_velocity_pct": volume_velocity_pct.fillna(0.5),
            "volume_stability": volume_stability,
            "volume_participation_score": volume_participation_score,
        }
    )
    if _slice_len is not None:
        result = result.reindex(_full_index, fill_value=0.0)
    return result
