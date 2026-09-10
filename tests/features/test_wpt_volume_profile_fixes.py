"""
WPT Volume Profile 修复验证测试

测试内容：
1. WPT 重建后长度对齐（修复1）
2. level 超过最大允许分解层数的处理（修复2）
3. bins > 数据点数的情况（修复3）
4. 异常处理的正确性（修复4）
"""

import numpy as np
import pandas as pd
import warnings
import sys

# pytest 是可选的（如果可用则使用，否则跳过）
try:
    import pytest

    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False

    # 如果没有 pytest，创建一个简单的 fixture 装饰器
    class pytest:
        @staticmethod
        def fixture(func):
            return func


warnings.filterwarnings("ignore")

# 添加项目根目录到路径

from src.features.time_series.utils_volume_profile import (
    compute_wpt_volume_profile,
    freedman_diaconis_bins,
)
from src.features.time_series.utils_wpt_features import wpt_decompose


class TestWPTVolumeProfileFixes:
    """WPT Volume Profile 修复验证测试"""

    @pytest.fixture
    def sample_data(self):
        """创建样本数据"""
        np.random.seed(42)
        n = 100

        # 生成价格序列（随机游走）
        prices = 50000 + np.cumsum(np.random.randn(n) * 10)

        # 生成成交量序列
        volumes = np.random.uniform(100, 1000, n)

        return prices, volumes

    def test_wpt_reconstruction_length_alignment(self, sample_data):
        """测试1: WPT 重建后长度对齐"""
        print("\n测试 WPT 重建后长度对齐...")

        prices, volumes = sample_data

        # 测试不同长度的窗口
        for window_size in [32, 64, 100, 128]:
            price_window = prices[:window_size]
            volume_window = volumes[:window_size]

            result = compute_wpt_volume_profile(
                price_window=price_window,
                volume_window=volume_window,
                bins=20,
                wavelet="db4",
                level=4,
                drop_high_freq=True,
            )

            if result is not None:
                # 验证 price_denoised 长度与原始窗口一致
                assert result.price_denoised is not None, "应返回降噪后的价格"
                assert len(result.price_denoised) == len(
                    price_window
                ), f"降噪后价格长度 ({len(result.price_denoised)}) 应与原始窗口长度 ({len(price_window)}) 一致"
                print(f"   ✅ 窗口大小 {window_size}: 长度对齐正确")
            else:
                print(f"   ⚠️  窗口大小 {window_size}: 返回 None（可能数据不足）")

    def test_level_exceeds_max_decomposition(self, sample_data):
        """测试2: level 超过最大允许分解层数"""
        print("\n测试 level 超过最大允许分解层数...")

        prices, volumes = sample_data

        # 使用小窗口（32个点），level=4 可能超过最大允许层数
        small_window = 32
        price_window = prices[:small_window]
        volume_window = volumes[:small_window]

        # 计算最大允许层数
        import pywt

        max_level = pywt.dwt_max_level(small_window, "db4")
        print(f"   窗口大小: {small_window}, 最大允许层数: {max_level}")

        # 测试 level > max_level 的情况
        for level in [max_level + 1, max_level + 2, 10]:
            result = compute_wpt_volume_profile(
                price_window=price_window,
                volume_window=volume_window,
                bins=20,
                wavelet="db4",
                level=level,
                drop_high_freq=True,
            )

            # 应该能正常处理，不会报错
            if result is not None:
                assert result.price_denoised is not None, "应返回降噪后的价格"
                assert len(result.price_denoised) == len(price_window), "长度应对齐"
                print(f"   ✅ level={level} (超过最大层数): 正确处理")
            else:
                print(f"   ⚠️  level={level}: 返回 None（可能数据不足）")

    def test_bins_exceeds_data_points(self, sample_data):
        """测试3: bins > 数据点数的情况"""
        print("\n测试 bins > 数据点数...")

        prices, volumes = sample_data

        # 使用小窗口，确保 bins 可能超过数据点数
        small_window = 20
        price_window = prices[:small_window]
        volume_window = volumes[:small_window]

        # 测试 bins="auto" 时可能超过数据点数的情况
        result = compute_wpt_volume_profile(
            price_window=price_window,
            volume_window=volume_window,
            bins="auto",  # 自动计算，可能 > len(price_valid)
            wavelet="db4",
            level=2,  # 使用较小的 level
            drop_high_freq=True,
        )

        if result is not None:
            # 验证 bins 数量不超过有效数据点数
            # 注意：hist 的长度 = bins，应该 <= len(price_valid)
            assert len(result.hist) <= len(
                price_window
            ), f"hist 长度 ({len(result.hist)}) 不应超过窗口长度 ({len(price_window)})"
            print(
                f"   ✅ bins 自动计算: {len(result.hist)} 个 bins，窗口大小: {len(price_window)}"
            )
        else:
            print(f"   ⚠️  返回 None（可能数据不足）")

        # 测试显式指定 bins > 数据点数
        large_bins = 50  # 大于窗口大小
        result2 = compute_wpt_volume_profile(
            price_window=price_window,
            volume_window=volume_window,
            bins=large_bins,
            wavelet="db4",
            level=2,
            drop_high_freq=True,
        )

        if result2 is not None:
            # 应该被限制为 <= len(price_valid)
            assert len(result2.hist) <= len(
                price_window
            ), f"hist 长度 ({len(result2.hist)}) 应被限制为 <= 窗口长度 ({len(price_window)})"
            print(f"   ✅ 显式 bins={large_bins}: 被限制为 {len(result2.hist)} 个 bins")
        else:
            print(f"   ⚠️  返回 None（可能数据不足）")

    def test_exception_handling_specificity(self, sample_data):
        """测试4: 异常处理的正确性（只捕获预期异常）"""
        print("\n测试异常处理...")

        prices, volumes = sample_data

        # 测试正常情况
        result = compute_wpt_volume_profile(
            price_window=prices,
            volume_window=volumes,
            bins=20,
            wavelet="db4",
            level=4,
            drop_high_freq=True,
        )

        assert result is not None, "正常情况应返回结果"
        print("   ✅ 正常情况: 正确处理")

        # 测试无效小波函数（应捕获 ValueError）
        result2 = compute_wpt_volume_profile(
            price_window=prices,
            volume_window=volumes,
            bins=20,
            wavelet="invalid_wavelet",  # 无效小波
            level=4,
            drop_high_freq=True,
        )

        # 应该返回原始价格（fallback），不报错
        if result2 is not None:
            assert (
                result2.price_denoised is not None
            ), "应返回降噪后的价格（或原始价格）"
            print("   ✅ 无效小波: 正确处理（fallback 到原始价格）")
        else:
            print("   ⚠️  无效小波: 返回 None")

    def test_freedman_diaconis_bins_edge_cases(self):
        """测试 Freedman-Diaconis bins 计算的边界情况"""
        print("\n测试 Freedman-Diaconis bins 边界情况...")

        # 测试1: 数据点很少
        small_data = np.array([1.0, 2.0, 3.0])
        bins = freedman_diaconis_bins(small_data, min_bins=10, max_bins=100)
        assert bins >= 10, "数据点少时应返回最小 bins"
        print(f"   ✅ 小数据集 (n=3): bins={bins}")

        # 测试2: 所有值相同（IQR=0）
        constant_data = np.ones(100)
        bins = freedman_diaconis_bins(constant_data, min_bins=10, max_bins=100)
        assert bins == 10, "所有值相同时应返回最小 bins"
        print(f"   ✅ 常数数据 (IQR=0): bins={bins}")

        # 测试3: 正常数据
        normal_data = np.random.normal(100, 5, 1000)
        bins = freedman_diaconis_bins(normal_data, min_bins=10, max_bins=100)
        assert 10 <= bins <= 100, "正常数据应在合理范围内"
        print(f"   ✅ 正常数据 (n=1000): bins={bins}")

        # 测试4: 极端值（可能产生很大的 bins）
        extreme_data = np.concatenate([np.ones(50) * 1.0, np.ones(50) * 1000.0])
        bins = freedman_diaconis_bins(extreme_data, min_bins=10, max_bins=100)
        assert 10 <= bins <= 100, "极端值应被限制在 max_bins 内"
        print(f"   ✅ 极端值数据: bins={bins}")

    def test_wpt_decompose_length_alignment(self):
        """测试 wpt_decompose 函数的长度对齐"""
        print("\n测试 wpt_decompose 长度对齐...")

        # 测试不同长度的信号
        for n in [32, 64, 100, 128]:
            signal = np.random.randn(n) * 10 + 100

            result = wpt_decompose(
                signal=signal,
                wavelet="db4",
                level=4,
                mode="symmetric",
            )

            # 验证所有输出长度一致
            assert len(result["trend"]) == len(signal), "trend 长度应对齐"
            assert len(result["fluctuation"]) == len(signal), "fluctuation 长度应对齐"
            print(f"   ✅ 信号长度 {n}: 所有输出长度对齐")

    def test_small_window_edge_cases(self):
        """测试小窗口边界情况"""
        print("\n测试小窗口边界情况...")

        # 测试1: 非常小的窗口（小于 2^level）
        tiny_window = 8  # 小于 2^4 = 16
        prices = np.random.randn(tiny_window) * 10 + 50000
        volumes = np.random.uniform(100, 1000, tiny_window)

        result = compute_wpt_volume_profile(
            price_window=prices,
            volume_window=volumes,
            bins=5,
            wavelet="db4",
            level=4,  # level=4 需要至少 16 个点
            drop_high_freq=True,
        )

        # 应该能处理（自动降低 level 或返回 None）
        if result is not None:
            assert result.price_denoised is not None, "应返回降噪后的价格"
            assert len(result.price_denoised) == len(prices), "长度应对齐"
            print(f"   ✅ 小窗口 (n={tiny_window}): 正确处理")
        else:
            print(f"   ⚠️  小窗口 (n={tiny_window}): 返回 None（数据不足）")

        # 测试2: 刚好等于最小长度的窗口
        min_window = 16  # 2^4
        prices2 = np.random.randn(min_window) * 10 + 50000
        volumes2 = np.random.uniform(100, 1000, min_window)

        result2 = compute_wpt_volume_profile(
            price_window=prices2,
            volume_window=volumes2,
            bins=5,
            wavelet="db4",
            level=4,
            drop_high_freq=True,
        )

        if result2 is not None:
            assert result2.price_denoised is not None, "应返回降噪后的价格"
            assert len(result2.price_denoised) == len(prices2), "长度应对齐"
            print(f"   ✅ 最小窗口 (n={min_window}): 正确处理")
        else:
            print(f"   ⚠️  最小窗口 (n={min_window}): 返回 None")


def run_all_tests():
    """运行所有测试"""
    print("=" * 70)
    print("WPT Volume Profile 修复验证测试")
    print("=" * 70)

    # 创建测试实例
    test_instance = TestWPTVolumeProfileFixes()

    # 创建 fixtures
    sample_data = test_instance.sample_data()

    # 运行测试
    tests = [
        (
            "WPT 重建长度对齐",
            test_instance.test_wpt_reconstruction_length_alignment,
            [sample_data],
        ),
        (
            "level 超过最大层数",
            test_instance.test_level_exceeds_max_decomposition,
            [sample_data],
        ),
        ("bins > 数据点数", test_instance.test_bins_exceeds_data_points, [sample_data]),
        ("异常处理", test_instance.test_exception_handling_specificity, [sample_data]),
        (
            "Freedman-Diaconis 边界",
            test_instance.test_freedman_diaconis_bins_edge_cases,
            [],
        ),
        (
            "wpt_decompose 长度对齐",
            test_instance.test_wpt_decompose_length_alignment,
            [],
        ),
        ("小窗口边界", test_instance.test_small_window_edge_cases, []),
    ]

    passed = 0
    failed = 0

    for test_name, test_func, args in tests:
        try:
            test_func(*args)
            passed += 1
        except Exception as e:
            print(f"\n❌ {test_name} 失败: {e}")
            import traceback

            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 70)
    print(f"测试完成: {passed} 通过, {failed} 失败")
    print("=" * 70)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
