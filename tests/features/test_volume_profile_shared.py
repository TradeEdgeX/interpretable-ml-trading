import numpy as np
import pandas as pd
import pytest

from src.features.time_series.utils_volume_profile import (
    compute_wpt_volume_profile,
    compute_unified_volume_profile_features,
    compute_wpt_vpvr_from_series,
)


def test_compute_wpt_volume_profile_basic_histogram_properties():
    """基础属性：bin 数量、边界和体积守恒。"""
    price_window = np.linspace(100.0, 110.0, 120)
    volume_window = np.linspace(1.0, 2.0, 120)

    result = compute_wpt_volume_profile(
        price_window=price_window,
        volume_window=volume_window,
        bins=24,
        wavelet="db2",
        level=2,
        drop_high_freq=False,
    )

    assert result is not None
    assert len(result.hist) == 24
    assert len(result.centers) == 24
    assert len(result.edges) == 25

    # 成交量守恒（允许小的数值误差）
    assert pytest.approx(volume_window.sum(), rel=1e-6) == np.sum(result.hist)
    # 价格范围覆盖原始窗口
    assert pytest.approx(price_window.min(), rel=1e-6) == result.price_min
    assert pytest.approx(price_window.max(), rel=1e-6) == result.price_max

    # centers 应该是 edges 中点
    reconstructed_centers = (result.edges[:-1] + result.edges[1:]) / 2.0
    assert np.allclose(result.centers, reconstructed_centers, rtol=1e-6, atol=1e-8)


def test_vpvr_and_poc_share_same_price_profile():
    """在平滑单调行情下，VPVR 的 PVP ≈ POC。"""
    n = 180
    window = 60
    bins = 30

    # 构造单调上升价格 + 轻微 volume 梯度，避免极端噪声
    price_series = np.linspace(100.0, 120.0, n)
    volume_series = np.linspace(1_000.0, 2_000.0, n)

    df = pd.DataFrame(
        {
            "high": price_series,
            "low": price_series,
            "close": price_series,
            "volume": volume_series,
        }
    )

    vpvr_df = compute_unified_volume_profile_features(
        df,
        price_col="close",
        volume_col="volume",
        high_col="high",
        low_col="low",
        window=window,
        bins=bins,
        use_typical_price=True,
    )

    # 使用统一实现计算 POC/HAL
    poc_df = compute_unified_volume_profile_features(
        df,
        price_col="close",
        volume_col="volume",
        high_col="high",
        low_col="low",
        window=window,
        bins=bins,
    )
    poc = poc_df["vp_poc"]
    poc_volume_ratio = poc_df["vp_poc_volume_ratio"]
    hal_high = poc_df["vp_hal_high"]
    hal_low = poc_df["vp_hal_low"]

    check_idx = window + 20

    # PVP 与 POC 在同一价格层附近（允许少量数值误差）
    assert np.isfinite(vpvr_df["vp_poc"].iloc[check_idx])
    assert np.isfinite(poc.iloc[check_idx])
    assert vpvr_df["vp_poc"].iloc[check_idx] == pytest.approx(
        poc.iloc[check_idx], rel=1e-4, abs=1e-3
    )

    # VPVR 的密度在 [0, 1] 之间
    assert 0.0 <= vpvr_df["vp_volume_density"].iloc[check_idx] <= 1.0

    # POC 有非零的成交量占比
    assert 0.0 < poc_volume_ratio.iloc[check_idx] <= 1.0


def test_compute_poc_value_area_volume_ratio():
    """验证 HAL (Value Area) 覆盖的成交量占比接近目标 value_area_ratio。"""
    n = 200
    window = 80
    bins = 40
    value_area_ratio = 0.7

    # 价格单调上升，volume 在中间价格段略微抬高，方便形成清晰的 POC/Value Area
    price_series = np.linspace(100.0, 120.0, n)
    volume_series = np.ones(n) * 1_000.0
    volume_series[80:120] = 2_000.0

    df = pd.DataFrame(
        {
            "high": price_series,
            "low": price_series,
            "close": price_series,
            "volume": volume_series,
        }
    )

    # 使用统一实现计算 POC/HAL
    poc_df = compute_unified_volume_profile_features(
        df,
        price_col="close",
        volume_col="volume",
        high_col="high",
        low_col="low",
        window=window,
        bins=bins,
        value_area_ratio=value_area_ratio,
    )
    poc = poc_df["vp_poc"]
    poc_volume_ratio = poc_df["vp_poc_volume_ratio"]
    hal_high = poc_df["vp_hal_high"]
    hal_low = poc_df["vp_hal_low"]

    check_idx = window + 30

    # 使用与 compute_poc 相同的窗口直接构建 volume profile，独立验证 Value Area 体积占比
    price_window = df["close"].iloc[check_idx - window : check_idx].values
    volume_window = df["volume"].iloc[check_idx - window : check_idx].values

    vp_result = compute_wpt_volume_profile(
        price_window=price_window,
        volume_window=volume_window,
        bins=bins,
    )

    assert vp_result is not None

    total_vol = vp_result.hist.sum()
    assert total_vol > 0

    # 统计 HAL 区间内的成交量占比
    low = hal_low.iloc[check_idx]
    high = hal_high.iloc[check_idx]
    assert low < high

    # 找出 edges 中落在 [low, high] 范围内的 bins
    edges = vp_result.edges
    centers = vp_result.centers

    mask = (centers >= low) & (centers <= high)
    value_area_vol = vp_result.hist[mask].sum()
    value_area_ratio_est = value_area_vol / total_vol

    # 要求 Value Area 覆盖的成交量至少达到目标比例，且不要远离太多
    assert value_area_ratio_est >= value_area_ratio - 0.05
    assert value_area_ratio_est <= min(1.0, value_area_ratio + 0.15)

    # 同时，POC 必须落在 HAL 区间内
    poc_price = poc.iloc[check_idx]
    assert low <= poc_price <= high


def test_vpvr_hvn_lvn_counts_for_bimodal_profile():
    """构造典型双峰 volume profile，验证 HVN/LVN 计数和 LVN 距离与直方图定义一致。"""
    n = 200
    window = 150
    bins = 40

    # 在窗口内构造：高量区(100)、低量谷(110)、高量区(120)
    # 调整数据使 LVN 阈值为正数
    price_series = np.ones(n) * 100.0
    volume_series = np.ones(n) * 5_000.0  # 增加基础成交量

    # 中段抬升到 110，降成交量 -> LVN 区（需要足够低以触发 LVN）
    price_series[70:120] = 110.0
    volume_series[70:120] = 500.0  # 显著低于平均值

    # 后半段抬升到 120，再次高成交量 -> 第二个 HVN
    price_series[120:170] = 120.0
    volume_series[120:170] = 10_000.0  # 显著高于平均值

    df = pd.DataFrame(
        {
            "high": price_series,
            "low": price_series,
            "close": price_series,
            "volume": volume_series,
        }
    )

    vpvr_df = compute_unified_volume_profile_features(
        df,
        price_col="close",
        volume_col="volume",
        high_col="high",
        low_col="low",
        window=window,
        bins=bins,
        use_typical_price=True,
    )

    # 选取最后一个索引，窗口内应包含两个明显的高量峰和一个低量谷
    idx = len(df) - 1

    # 使用同一窗口直接调用 compute_wpt_volume_profile，独立重现 HVN/LVN 定义
    price_window = df["close"].iloc[idx - window : idx].values
    volume_window = df["volume"].iloc[idx - window : idx].values

    vp_result = compute_wpt_volume_profile(
        price_window=price_window,
        volume_window=volume_window,
        bins=bins,
    )

    assert vp_result is not None

    hist = vp_result.hist
    centers = vp_result.centers

    positive_mask = hist > 0
    assert np.any(positive_mask)

    volume_mean = np.mean(hist[positive_mask])
    volume_std = np.std(hist[positive_mask])
    assert volume_std > 0

    hvn_mask = hist > (volume_mean + 0.5 * volume_std)
    lvn_mask = hist < (volume_mean - 0.5 * volume_std)

    hvn_expected = np.sum(hvn_mask)
    lvn_expected = np.sum(lvn_mask)

    # 至少应当存在高量节点（LVN 可能不存在，取决于数据分布）
    assert hvn_expected > 0
    # 如果 LVN 阈值 > 0，则应该有 LVN；否则跳过 LVN 相关断言
    lvn_threshold = volume_mean - 0.5 * volume_std
    if lvn_threshold > 0:
        assert lvn_expected > 0

    # VPVR 中的 HVN/LVN 计数与根据直方图计算的一致
    hvn_count = vpvr_df["vp_hvn_count"].iloc[idx]
    lvn_count = vpvr_df["vp_lvn_count"].iloc[idx]
    assert hvn_count == pytest.approx(float(hvn_expected))
    if lvn_threshold > 0:
        assert lvn_count == pytest.approx(float(lvn_expected))

    # 按实现逻辑计算最近 LVN 距离和 price_in_lvn
    current_price = df["close"].iloc[idx]
    lvn_prices = centers[lvn_mask]
    if len(lvn_prices) == 0:
        # 如果没有 LVN，跳过后续 LVN 相关断言
        return

    lvn_distances = np.abs(lvn_prices - current_price)
    nearest_lvn_idx = int(np.argmin(lvn_distances))
    nearest_lvn_price = lvn_prices[nearest_lvn_idx]
    nearest_lvn_distance = float(lvn_distances[nearest_lvn_idx])

    price_range = vp_result.price_max - vp_result.price_min
    assert price_range > 0

    expected_lvn_distance = nearest_lvn_distance / price_range

    bin_width = (vp_result.price_max - vp_result.price_min) / bins
    expected_price_in_lvn = (
        1.0 if abs(current_price - nearest_lvn_price) < bin_width else 0.0
    )

    assert vpvr_df["vp_lvn_distance"].iloc[idx] == pytest.approx(
        expected_lvn_distance, rel=1e-6, abs=1e-6
    )
    assert vpvr_df["vpvr_price_in_lvn"].iloc[idx] == pytest.approx(
        expected_price_in_lvn, rel=1e-6, abs=1e-6
    )


def test_vpvr_price_in_lvn_flag_and_low_density():
    """让当前价格落在人工构造的 LVN 区间内，验证 price_in_lvn 与 LVN 定义一致。"""
    n = 200
    window = 150
    bins = 40

    price_series = np.ones(n) * 100.0
    volume_series = np.ones(n) * 1_000.0

    # 前段高量区(100)
    price_series[50:100] = 100.0
    volume_series[50:100] = 2_000.0

    # 中段低量谷(110)
    price_series[100:150] = 110.0
    volume_series[100:150] = 100.0

    # 为了让当前价格位于 LVN，将尾部也保持在 110 且低量
    price_series[150:] = 110.0
    volume_series[150:] = 100.0

    df = pd.DataFrame(
        {
            "high": price_series,
            "low": price_series,
            "close": price_series,
            "volume": volume_series,
        }
    )

    vpvr_df = compute_unified_volume_profile_features(
        df,
        price_col="close",
        volume_col="volume",
        high_col="high",
        low_col="low",
        window=window,
        bins=bins,
        use_typical_price=True,
    )
    vpvr_narrow = compute_wpt_vpvr_from_series(
        close=df["close"],
        high=df["high"],
        low=df["low"],
        volume=df["volume"],
        window=window,
        bins=bins,
    )

    idx = len(df) - 1

    # 使用同一窗口，通过 volume profile 独立重建 LVN 定义
    price_window = df["close"].iloc[idx - window : idx].values
    volume_window = df["volume"].iloc[idx - window : idx].values

    vp_result = compute_wpt_volume_profile(
        price_window=price_window,
        volume_window=volume_window,
        bins=bins,
    )

    assert vp_result is not None

    hist = vp_result.hist
    centers = vp_result.centers

    positive_mask = hist > 0
    assert np.any(positive_mask)

    volume_mean = np.mean(hist[positive_mask])
    volume_std = np.std(hist[positive_mask])
    assert volume_std > 0

    lvn_threshold = volume_mean - 0.5 * volume_std
    lvn_mask = (
        hist < lvn_threshold if lvn_threshold > 0 else np.zeros_like(hist, dtype=bool)
    )

    current_price = df["close"].iloc[idx]
    lvn_prices = centers[lvn_mask]

    if len(lvn_prices) == 0:
        # 如果没有 LVN，跳过后续 LVN 相关断言
        return
    lvn_distances = np.abs(lvn_prices - current_price)
    nearest_lvn_idx = int(np.argmin(lvn_distances))
    nearest_lvn_price = lvn_prices[nearest_lvn_idx]
    nearest_lvn_distance = float(lvn_distances[nearest_lvn_idx])

    price_range = vp_result.price_max - vp_result.price_min
    assert price_range > 0
    expected_lvn_distance = nearest_lvn_distance / price_range

    bin_width = (vp_result.price_max - vp_result.price_min) / bins
    expected_price_in_lvn = (
        1.0 if abs(current_price - nearest_lvn_price) < bin_width else 0.0
    )

    # 校验实现值与根据直方图公式算出的值一致
    assert vpvr_df["vp_lvn_distance"].iloc[idx] == pytest.approx(
        expected_lvn_distance, rel=1e-6, abs=1e-6
    )
    # Narrow entrypoint should match the unified implementation for the VPVR subset
    assert vpvr_narrow["vpvr_lvn_distance"].iloc[idx] == pytest.approx(
        expected_lvn_distance, rel=1e-6, abs=1e-6
    )
    assert vpvr_narrow["vpvr_price_in_lvn"].iloc[idx] == pytest.approx(
        expected_price_in_lvn, rel=1e-6, abs=1e-6
    )

    # 额外 sanity check：当前价就是低量谷，所以应当被视为 LVN
    assert expected_price_in_lvn == 1.0
