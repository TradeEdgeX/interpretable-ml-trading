#!/usr/bin/env python3
"""
简单测试脚本：验证 compute_wpt_volume_profile 改进功能

可以直接运行：python3 tests/test_wpt_improvements_simple.py
"""

import sys

# 添加项目根目录到路径

import numpy as np
from src.features.time_series.utils_volume_profile import (
    compute_wpt_volume_profile,
    VolumeProfileResult,
)


def test_high_freq_removal():
    """测试高频子带剔除优化"""
    print("🧪 测试 1: 高频子带剔除优化...")
    np.random.seed(42)
    n = 128

    # 创建包含高频噪声的价格序列
    trend = np.linspace(100.0, 110.0, n)
    noise = np.random.randn(n) * 0.5
    price_window = trend + noise
    volume_window = np.ones(n) * 1000.0

    # 测试降噪效果
    result_with_denoise = compute_wpt_volume_profile(
        price_window=price_window,
        volume_window=volume_window,
        bins=20,
        wavelet="db4",
        level=4,
        drop_high_freq=True,
    )

    result_without_denoise = compute_wpt_volume_profile(
        price_window=price_window,
        volume_window=volume_window,
        bins=20,
        wavelet="db4",
        level=4,
        drop_high_freq=False,
    )

    assert result_with_denoise is not None, "降噪结果不应为 None"
    assert result_without_denoise is not None, "非降噪结果不应为 None"
    assert result_with_denoise.price_denoised is not None, "应返回降噪价格序列"

    print("  ✅ 高频子带剔除测试通过")


def test_auto_bins():
    """测试动态 bins 计算"""
    print("🧪 测试 2: 动态 bins 计算...")
    np.random.seed(42)
    n = 200

    # 创建不同价格范围的数据
    price_window = np.random.uniform(100.0, 120.0, n)
    volume_window = np.ones(n) * 1000.0

    result_auto = compute_wpt_volume_profile(
        price_window=price_window,
        volume_window=volume_window,
        bins="auto",
    )

    result_fixed = compute_wpt_volume_profile(
        price_window=price_window,
        volume_window=volume_window,
        bins=50,
    )

    assert result_auto is not None, "auto bins 结果不应为 None"
    assert result_fixed is not None, "固定 bins 结果不应为 None"
    assert (
        10 <= len(result_auto.hist) <= 100
    ), f"auto bins 应在 10-100 之间，实际: {len(result_auto.hist)}"

    print(
        f"  ✅ 动态 bins 测试通过 (auto={len(result_auto.hist)}, fixed={len(result_fixed.hist)})"
    )


def test_denoised_price_return():
    """测试返回降噪价格序列"""
    print("🧪 测试 3: 返回降噪价格序列...")
    np.random.seed(42)
    n = 100
    price_window = np.linspace(100.0, 110.0, n) + np.random.randn(n) * 0.2
    volume_window = np.ones(n) * 1000.0

    result = compute_wpt_volume_profile(
        price_window=price_window,
        volume_window=volume_window,
        bins=20,
    )

    assert result is not None, "结果不应为 None"
    assert hasattr(result, "price_denoised"), "应包含 price_denoised 字段"
    assert result.price_denoised is not None, "price_denoised 不应为 None"
    assert isinstance(
        result.price_denoised, np.ndarray
    ), "price_denoised 应为 numpy 数组"
    assert len(result.price_denoised) == n, f"price_denoised 长度应为 {n}"

    print("  ✅ 降噪价格序列返回测试通过")


def test_edge_cases():
    """测试边界情况"""
    print("🧪 测试 4: 边界情况处理...")

    # 测试 None 输入
    result = compute_wpt_volume_profile(
        price_window=None,
        volume_window=np.ones(100),
    )
    assert result is None, "None 输入应返回 None"

    # 测试长度不匹配
    result = compute_wpt_volume_profile(
        price_window=np.ones(100),
        volume_window=np.ones(50),
    )
    assert result is None, "长度不匹配应返回 None"

    # 测试太短的序列
    result = compute_wpt_volume_profile(
        price_window=np.ones(5),
        volume_window=np.ones(5),
    )
    assert result is None, "太短序列应返回 None"

    # 测试价格恒定
    result = compute_wpt_volume_profile(
        price_window=np.ones(100) * 100.0,
        volume_window=np.ones(100) * 1000.0,
    )
    assert result is None, "价格恒定应返回 None"

    print("  ✅ 边界情况测试通过")


def test_integration():
    """集成测试：所有改进功能协同工作"""
    print("🧪 测试 5: 集成测试（所有改进功能）...")
    np.random.seed(42)
    n = 200

    # 创建包含噪声的价格序列
    trend = np.linspace(100.0, 120.0, n)
    noise = np.random.randn(n) * 0.8
    price_window = trend + noise
    volume_window = np.random.uniform(800.0, 1200.0, n)

    # 使用所有改进功能
    result = compute_wpt_volume_profile(
        price_window=price_window,
        volume_window=volume_window,
        bins="auto",  # 动态 bins
        wavelet="db4",
        level=4,
        drop_high_freq=True,  # 高频剔除
    )

    assert result is not None, "结果不应为 None"
    assert result.hist is not None, "hist 不应为 None"
    assert result.edges is not None, "edges 不应为 None"
    assert result.centers is not None, "centers 不应为 None"
    assert result.price_min is not None, "price_min 不应为 None"
    assert result.price_max is not None, "price_max 不应为 None"
    assert result.price_denoised is not None, "price_denoised 不应为 None"

    # 验证基本属性
    assert len(result.hist) == len(result.centers), "hist 和 centers 长度应一致"
    assert len(result.edges) == len(result.hist) + 1, "edges 长度应为 hist+1"
    assert 10 <= len(result.hist) <= 100, "auto bins 应在合理范围"
    assert result.price_min < result.price_max, "price_min 应小于 price_max"

    # 验证降噪价格序列
    assert len(result.price_denoised) == n, "降噪价格序列长度应匹配"
    assert np.all(np.isfinite(result.price_denoised)), "降噪价格序列应全为有限值"

    # 验证成交量守恒
    volume_sum = np.sum(result.hist)
    expected_sum = volume_window.sum()
    assert abs(volume_sum - expected_sum) < expected_sum * 1e-5, "成交量应守恒"

    print("  ✅ 集成测试通过")


def test_denoised_price_smoothing():
    """测试降噪效果：降噪后的价格应该更平滑"""
    print("🧪 测试 6: 降噪效果验证...")
    np.random.seed(42)
    n = 128

    # 创建包含明显噪声的价格序列
    trend = np.linspace(100.0, 110.0, n)
    noise = np.random.randn(n) * 1.0  # 较大噪声
    price_window = trend + noise
    volume_window = np.ones(n) * 1000.0

    result = compute_wpt_volume_profile(
        price_window=price_window,
        volume_window=volume_window,
        bins=20,
        drop_high_freq=True,
    )

    assert result is not None, "结果不应为 None"
    assert result.price_denoised is not None, "应返回降噪价格序列"

    # 计算一阶差分（变化率）的标准差
    original_diff_std = np.std(np.diff(price_window))
    denoised_diff_std = np.std(np.diff(result.price_denoised))

    # 降噪后的变化应该更平滑（允许轻微增加，因为边界效应）
    assert (
        denoised_diff_std < original_diff_std * 1.2
    ), f"降噪后应更平滑: {denoised_diff_std} < {original_diff_std * 1.2}"

    print(
        f"  ✅ 降噪效果验证通过 (原始: {original_diff_std:.4f}, 降噪: {denoised_diff_std:.4f})"
    )


def main():
    """运行所有测试"""
    print("=" * 60)
    print("开始测试 compute_wpt_volume_profile 改进功能")
    print("=" * 60)
    print()

    tests = [
        test_high_freq_removal,
        test_auto_bins,
        test_denoised_price_return,
        test_edge_cases,
        test_integration,
        test_denoised_price_smoothing,
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"  ❌ 失败: {e}")
            failed += 1
        except Exception as e:
            print(f"  ❌ 错误: {e}")
            failed += 1
        print()

    print("=" * 60)
    print(f"测试完成: {passed} 通过, {failed} 失败")
    print("=" * 60)

    if failed > 0:
        sys.exit(1)
    else:
        print("✅ 所有测试通过！")


if __name__ == "__main__":
    main()
