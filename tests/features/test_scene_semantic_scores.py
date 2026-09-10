import numpy as np
import pandas as pd

from src.features.time_series.utils_interaction_features import (
    compute_fp_imbalance_scene_semantic_scores_from_series,
    compute_liquidity_void_scene_semantic_scores_from_series,
    compute_trade_cluster_scene_semantic_scores_from_series,
    compute_vpin_scene_semantic_scores_from_series,
    compute_volume_profile_scene_semantic_scores_from_series,
    compute_wick_scene_semantic_scores_from_series,
    compute_wpt_scene_semantic_scores_from_series,
)


def _atr_like(n: int, v: float = 10.0) -> pd.Series:
    return pd.Series([v] * n)


def test_vpin_scene_semantic_scores_basic_monotonicity():
    """
    Smoke + monotonic checks:
    - compression score should be higher in low-disp + high-compression
    - ignition score should be higher in high-disp + high-volume
    - absorption score should be higher near SR
    - exhaustion_scene should be higher when trend_r2 is low (trend ending)
    """
    n = 4
    idx = pd.date_range("2025-01-01", periods=n, freq="h")

    vpin_z = pd.Series([5.0, 5.0, 5.0, 5.0], index=idx)
    vpin_signed_z = pd.Series([5.0, 5.0, 5.0, 5.0], index=idx)

    open_ = pd.Series([100.0] * n, index=idx)
    close = pd.Series([100.0] * n, index=idx)

    # displacement: bar0 low, bar1 high, bar2 low, bar3 low
    high = pd.Series([100.1, 115.0, 100.1, 100.1], index=idx)
    low = pd.Series([99.9, 85.0, 99.9, 99.9], index=idx)
    atr = _atr_like(n, 10.0).rename("atr").set_axis(idx)

    compression_score = pd.Series([1.0, 0.0, 0.0, 0.0], index=idx)
    dist_to_nearest_sr = pd.Series([0.0001, 0.0001, 0.10, 0.0001], index=idx)  # pct
    volume_anomaly = pd.Series([0.0, 3.0, 0.0, 0.0], index=idx)
    trend_r2_20 = pd.Series(
        [1.0, 1.0, 1.0, 0.0], index=idx
    )  # only last is "trend ending"

    out = compute_vpin_scene_semantic_scores_from_series(
        vpin_zscore_50=vpin_z,
        vpin_signed_imbalance_zscore_50=vpin_signed_z,
        open=open_,
        close=close,
        high=high,
        low=low,
        atr=atr,
        compression_score=compression_score,
        dist_to_nearest_sr=dist_to_nearest_sr,
        volume_anomaly=volume_anomaly,
        trend_r2_20=trend_r2_20,
        clip_z=5.0,
        disp_atr_threshold=0.5,
        sr_prox_atr=1.5,
    )

    for col in out.columns:
        assert out[col].notna().all()
        assert out[col].between(0.0, 1.0).all()

    # compression should be strongest at bar0 (high compression + low disp)
    assert out["vpin_compression_score"].iloc[0] > out["vpin_compression_score"].iloc[1]
    assert out["vpin_compression_score"].iloc[0] > out["vpin_compression_score"].iloc[2]

    # ignition should be strongest at bar1 (high disp + high vol gate)
    assert out["vpin_ignition_score"].iloc[1] > out["vpin_ignition_score"].iloc[0]
    assert out["vpin_ignition_score"].iloc[1] > out["vpin_ignition_score"].iloc[2]

    # absorption: near SR (bar0/1/3) should beat far SR (bar2)
    assert out["vpin_absorption_score"].iloc[0] > out["vpin_absorption_score"].iloc[2]
    assert out["vpin_absorption_score"].iloc[3] > out["vpin_absorption_score"].iloc[2]

    # exhaustion_scene: same SR/disp context, but lower trend_r2 (bar3) -> higher exhaustion_scene
    assert (
        out["vpin_exhaustion_scene_score"].iloc[3]
        > out["vpin_exhaustion_scene_score"].iloc[0]
    )


def test_fp_imbalance_scene_semantic_scores_basic_monotonicity():
    n = 4
    idx = pd.date_range("2025-01-01", periods=n, freq="h")

    fp_imb = pd.Series([8.0, 8.0, 8.0, 8.0], index=idx)
    close = pd.Series([100.0] * n, index=idx)

    # displacement: bar0 low, bar1 high, bar2 low, bar3 low
    high = pd.Series([100.1, 115.0, 100.1, 100.1], index=idx)
    low = pd.Series([99.9, 85.0, 99.9, 99.9], index=idx)
    atr = _atr_like(n, 10.0).rename("atr").set_axis(idx)

    compression_score = pd.Series([1.0, 0.0, 0.0, 0.0], index=idx)
    dist_to_nearest_sr = pd.Series([0.0001, 0.0001, 0.10, 0.0001], index=idx)  # pct
    volume_anomaly = pd.Series([0.0, 3.0, 0.0, 0.0], index=idx)
    trend_r2_20 = pd.Series([1.0, 1.0, 1.0, 0.0], index=idx)

    out = compute_fp_imbalance_scene_semantic_scores_from_series(
        fp_max_imbalance_ratio=fp_imb,
        close=close,
        high=high,
        low=low,
        atr=atr,
        compression_score=compression_score,
        dist_to_nearest_sr=dist_to_nearest_sr,
        volume_anomaly=volume_anomaly,
        trend_r2_20=trend_r2_20,
        imb_threshold=3.0,
        imb_clip=8.0,
        disp_atr_threshold=0.5,
        sr_prox_atr=1.5,
    )

    for col in out.columns:
        assert out[col].notna().all()
        assert out[col].between(0.0, 1.0).all()

    assert (
        out["fp_imbalance_compression_score"].iloc[0]
        > out["fp_imbalance_compression_score"].iloc[1]
    )
    assert (
        out["fp_imbalance_ignition_score"].iloc[1]
        > out["fp_imbalance_ignition_score"].iloc[0]
    )
    assert (
        out["fp_imbalance_absorption_score"].iloc[0]
        > out["fp_imbalance_absorption_score"].iloc[2]
    )
    assert (
        out["fp_imbalance_exhaustion_scene_score"].iloc[3]
        > out["fp_imbalance_exhaustion_scene_score"].iloc[0]
    )


def test_liquidity_void_scene_semantic_scores_basic():
    """V2 连续化版本: 用 speed 作为 soft gate"""
    n = 4
    idx = pd.date_range("2025-01-01", periods=n, freq="h")

    # speed 作为 soft gate，不再需要 detected
    speed = pd.Series([0.1, 3.0, 0.1, 0.1], index=idx)  # ignition at bar1 (high speed)
    impact = pd.Series([0.1, 3.0, 0.1, 0.1], index=idx)
    retr = pd.Series([0.1, 0.1, 0.1, 0.9], index=idx)  # exhaustion at bar3 (high retr)
    fake = pd.Series(
        [0.1, 0.1, 0.9, 0.9], index=idx
    )  # penalize bar2/3 for ignition/absorption
    wpt_bc = pd.Series([0.0, 1.0, 0.0, 0.0], index=idx)
    compression_score = pd.Series([1.0, 0.0, 0.0, 0.0], index=idx)
    trend_r2_20 = pd.Series([1.0, 1.0, 1.0, 0.0], index=idx)

    out = compute_liquidity_void_scene_semantic_scores_from_series(
        liquidity_void_speed=speed,
        liquidity_void_price_impact=impact,
        liquidity_void_retracement=retr,
        liquidity_void_false_breakout_risk=fake,
        wpt_breakout_confidence=wpt_bc,
        compression_score=compression_score,
        trend_r2_20=trend_r2_20,
        speed_scale=3.0,
        impact_scale=3.0,
    )

    for col in out.columns:
        assert out[col].notna().all()
        assert out[col].between(0.0, 1.0).all()

    assert (
        out["liquidity_void_compression_score"].iloc[0]
        > out["liquidity_void_compression_score"].iloc[1]
    )
    assert (
        out["liquidity_void_ignition_score"].iloc[1]
        > out["liquidity_void_ignition_score"].iloc[0]
    )
    assert (
        out["liquidity_void_exhaustion_score"].iloc[3]
        > out["liquidity_void_exhaustion_score"].iloc[0]
    )


def test_liquidity_void_scene_v2_continuous_gate():
    """测试 V2 的核心改进：连续化 gate，而非 Bool"""
    n = 100
    idx = pd.date_range("2025-01-01", periods=n, freq="h")

    # 创建渐变的 speed（0 → 3）
    speed = pd.Series(np.linspace(0, 3, n), index=idx)
    impact = pd.Series([1.0] * n, index=idx)
    retr = pd.Series([0.2] * n, index=idx)
    fake = pd.Series([0.2] * n, index=idx)

    out = compute_liquidity_void_scene_semantic_scores_from_series(
        liquidity_void_speed=speed,
        liquidity_void_price_impact=impact,
        liquidity_void_retracement=retr,
        liquidity_void_false_breakout_risk=fake,
        speed_scale=3.0,
        impact_scale=3.0,
    )

    # 核心断言：输出应该是连续的，不是大量 0
    ignition = out["liquidity_void_ignition_score"]

    # 应该有多个不同的值（不是只有 0 和 1）
    unique_values = ignition.round(2).nunique()
    assert unique_values > 10, f"V2 应该输出连续值，但只有 {unique_values} 个不同值"

    # 应该单调递增（speed 增加 → ignition 增加）
    assert ignition.iloc[-1] > ignition.iloc[0], "high speed 应该产生更高的 ignition"

    # 不应该有 80% 是 0（这是 V1 的问题）
    zero_ratio = (ignition < 0.01).mean()
    assert zero_ratio < 0.5, f"V2 不应该有大量 0 值，但 {zero_ratio:.1%} 是接近 0"


def test_liquidity_void_scene_semantic_correctness():
    """测试 4 个场景的语义正确性"""
    n = 4
    idx = pd.date_range("2025-01-01", periods=n, freq="h")

    # 设计 4 个不同场景
    # bar0: compression (高压缩 + 高速度)
    # bar1: ignition (高速度 + 高冲击 + 低假突破风险 + WPT确认)
    # bar2: absorption (高速度 + 低回撇 + 低假突破 + 高趋势)
    # bar3: exhaustion (高速度 + 高回撇 + 高假突破 + 低趋势)

    speed = pd.Series([3.0, 3.0, 3.0, 3.0], index=idx)  # 都有高速度
    impact = pd.Series([0.5, 3.0, 0.5, 0.5], index=idx)  # bar1 高冲击
    retr = pd.Series([0.1, 0.1, 0.1, 0.9], index=idx)  # bar3 高回撇
    fake = pd.Series([0.1, 0.1, 0.1, 0.9], index=idx)  # bar3 高假突破风险

    wpt_bc = pd.Series([0.0, 1.0, 0.0, 0.0], index=idx)  # bar1 WPT 确认
    compression_score = pd.Series([1.0, 0.0, 0.0, 0.0], index=idx)  # bar0 压缩
    trend_r2_20 = pd.Series([0.5, 0.5, 0.9, 0.1], index=idx)  # bar2 高趋势, bar3 低趋势

    out = compute_liquidity_void_scene_semantic_scores_from_series(
        liquidity_void_speed=speed,
        liquidity_void_price_impact=impact,
        liquidity_void_retracement=retr,
        liquidity_void_false_breakout_risk=fake,
        wpt_breakout_confidence=wpt_bc,
        compression_score=compression_score,
        trend_r2_20=trend_r2_20,
        speed_scale=3.0,
        impact_scale=3.0,
    )

    # 语义正确性断言
    # compression 在 bar0 最高
    assert (
        out["liquidity_void_compression_score"].iloc[0]
        == out["liquidity_void_compression_score"].max()
    ), "bar0 应该有最高的 compression_score"

    # ignition 在 bar1 最高
    assert (
        out["liquidity_void_ignition_score"].iloc[1]
        == out["liquidity_void_ignition_score"].max()
    ), "bar1 应该有最高的 ignition_score"

    # absorption 在 bar2 最高（高趋势 + 低回撇 + 低假突破）
    assert (
        out["liquidity_void_absorption_score"].iloc[2]
        == out["liquidity_void_absorption_score"].max()
    ), "bar2 应该有最高的 absorption_score"

    # exhaustion 在 bar3 最高（低趋势 + 高回撇 + 高假突破）
    assert (
        out["liquidity_void_exhaustion_score"].iloc[3]
        == out["liquidity_void_exhaustion_score"].max()
    ), "bar3 应该有最高的 exhaustion_score"


def test_wpt_scene_semantic_scores_basic():
    n = 4
    idx = pd.date_range("2025-01-01", periods=n, freq="h")

    bc = pd.Series([0.0, 1.0, 0.0, 0.0], index=idx)
    fr = pd.Series([0.0, 0.0, 0.0, 1.0], index=idx)
    ms = pd.Series([1.0, 1.0, 1.0, 1.0], index=idx)
    ec = pd.Series([0.0, 1.0, 0.0, 0.0], index=idx)
    compression_score = pd.Series([1.0, 0.0, 0.0, 0.0], index=idx)
    trend_r2_20 = pd.Series([1.0, 1.0, 1.0, 0.0], index=idx)

    out = compute_wpt_scene_semantic_scores_from_series(
        wpt_breakout_confidence=bc,
        wpt_false_breakout_risk=fr,
        wpt_multi_scale_consistency=ms,
        wpt_energy_cascade=ec,
        compression_score=compression_score,
        trend_r2_20=trend_r2_20,
    )
    for col in out.columns:
        assert out[col].notna().all()
        assert out[col].between(0.0, 1.0).all()

    assert out["wpt_compression_score"].iloc[0] > out["wpt_compression_score"].iloc[1]
    assert out["wpt_ignition_score"].iloc[1] > out["wpt_ignition_score"].iloc[0]
    assert out["wpt_exhaustion_score"].iloc[3] > out["wpt_exhaustion_score"].iloc[0]


def test_volume_profile_scene_semantic_scores_basic():
    n = 4
    idx = pd.date_range("2025-01-01", periods=n, freq="h")

    width = pd.Series([0.1, 0.1, 0.9, 0.9], index=idx)
    ent = pd.Series([0.1, 0.1, 0.1, 0.9], index=idx)
    poc_dev = pd.Series([0.0, 2.0, 0.0, 0.0], index=idx)
    lv = pd.Series([0.0, 1.0, 0.0, 0.0], index=idx)
    hv = pd.Series([0.0, 0.0, 0.0, 1.0], index=idx)
    trend_r2_20 = pd.Series([1.0, 1.0, 1.0, 0.0], index=idx)

    out = compute_volume_profile_scene_semantic_scores_from_series(
        vp_width_ratio=width,
        vp_poc_deviation=poc_dev,
        vp_entropy=ent,
        vp_lv_ratio=lv,
        vp_hv_ratio=hv,
        trend_r2_20=trend_r2_20,
        entropy_scale=2.0,
        poc_dev_scale=2.0,
    )
    for col in out.columns:
        assert out[col].notna().all()
        assert out[col].between(0.0, 1.0).all()

    assert out["vp_compression_score"].iloc[0] > out["vp_compression_score"].iloc[2]
    assert out["vp_ignition_score"].iloc[1] > out["vp_ignition_score"].iloc[0]
    assert out["vp_exhaustion_score"].iloc[3] > out["vp_exhaustion_score"].iloc[0]


def test_wick_scene_semantic_scores_basic():
    n = 4
    idx = pd.date_range("2025-01-01", periods=n, freq="h")

    # bar0 calm + compression, bar3 rejection + trend end
    wu = pd.Series([0.0, 0.0, 0.0, 1.0], index=idx)
    wl = pd.Series([0.0, 0.0, 0.0, 0.0], index=idx)
    compression_score = pd.Series([1.0, 0.0, 0.0, 0.0], index=idx)
    trend_r2_20 = pd.Series([1.0, 1.0, 1.0, 0.0], index=idx)

    out = compute_wick_scene_semantic_scores_from_series(
        wick_upper_ratio=wu,
        wick_lower_ratio=wl,
        compression_score=compression_score,
        trend_r2_20=trend_r2_20,
    )
    for col in out.columns:
        assert out[col].notna().all()
        assert out[col].between(0.0, 1.0).all()

    assert out["wick_compression_score"].iloc[0] > out["wick_compression_score"].iloc[1]
    assert out["wick_exhaustion_score"].iloc[3] > out["wick_exhaustion_score"].iloc[0]


def test_trade_cluster_scene_semantic_scores_basic():
    n = 4
    idx = pd.date_range("2025-01-01", periods=n, freq="h")

    flow = pd.Series([1.0, 1.0, 1.0, 1.0], index=idx)
    absorp = pd.Series([0.1, 1.0, 0.1, 0.1], index=idx)
    exhaust = pd.Series([1.0, 0.1, 0.1, 1.0], index=idx)
    compression_score = pd.Series([1.0, 0.0, 0.0, 0.0], index=idx)
    volume_anomaly = pd.Series([0.0, 3.0, 0.0, 0.0], index=idx)
    trend_r2_20 = pd.Series([1.0, 1.0, 1.0, 0.0], index=idx)

    out = compute_trade_cluster_scene_semantic_scores_from_series(
        trade_cluster_flow_intensity=flow,
        trade_cluster_absorption_score=absorp,
        trade_cluster_exhaustion_score=exhaust,
        compression_score=compression_score,
        volume_anomaly=volume_anomaly,
        trend_r2_20=trend_r2_20,
    )
    for col in out.columns:
        assert out[col].notna().all()
        assert out[col].between(0.0, 1.0).all()

    assert (
        out["trade_cluster_compression_score"].iloc[0]
        > out["trade_cluster_compression_score"].iloc[1]
    )
    assert (
        out["trade_cluster_ignition_score"].iloc[1]
        > out["trade_cluster_ignition_score"].iloc[0]
    )
    assert (
        out["trade_cluster_exhaustion_scene_score"].iloc[3]
        > out["trade_cluster_exhaustion_scene_score"].iloc[0]
    )
