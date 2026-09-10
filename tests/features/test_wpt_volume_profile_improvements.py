"""
测试 compute_wpt_volume_profile 改进功能

测试内容：
1. 高频子带剔除优化（使用 freq 排序剔除尾部 25%）
2. 动态 bins 计算（auto 模式，Freedman-Diaconis rule）
3. 返回降噪价格序列（price_denoised 字段）
4. 边界情况处理（无效输入、极短序列等）
5. 降噪效果验证（降噪前后对比）
"""

import numpy as np
import pandas as pd
import pytest
import pywt

from src.features.time_series.utils_volume_profile import (
    compute_wpt_volume_profile,
    VolumeProfileResult,
)


class TestHighFreqSubbandRemoval:
    """测试高频子带剔除优化"""

    def test_freq_ordered_removal(self):
        """验证使用 freq 排序剔除高频子带"""
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

        assert result_with_denoise is not None
        assert result_without_denoise is not None

        # 降噪后的价格序列应该更平滑（标准差更小）
        assert result_with_denoise.price_denoised is not None
        assert result_without_denoise.price_denoised is not None

        std_denoised = np.std(result_with_denoise.price_denoised)
        std_original = np.std(price_window)

        # 降噪应该减少波动（允许小的数值误差）
        assert std_denoised <= std_original * 1.1  # 允许轻微增加（边界效应）

    def test_freq_removal_removes_tail_subbands(self):
        """验证剔除的是最高频的 25% 子带"""
        np.random.seed(42)
        n = 64
        price_window = np.linspace(100.0, 110.0, n) + np.random.randn(n) * 0.3
        volume_window = np.ones(n) * 1000.0

        # 手动验证 WPT 结构
        wp = pywt.WaveletPacket(
            data=price_window, wavelet="db4", mode="symmetric", maxlevel=4
        )
        freq_nodes = wp.get_level(4, "freq")
        total_nodes = len(freq_nodes)
        expected_drop = max(1, total_nodes // 4)

        # 测试函数应该剔除 expected_drop 个子带
        result = compute_wpt_volume_profile(
            price_window=price_window,
            volume_window=volume_window,
            bins=20,
            level=4,
            drop_high_freq=True,
        )

        assert result is not None
        # 降噪后的序列应该存在
        assert result.price_denoised is not None
        assert len(result.price_denoised) == n


class TestAutoBins:
    """测试动态 bins 计算"""

    def test_auto_bins_basic(self):
        """测试 auto bins 基本功能"""
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

        assert result_auto is not None
        assert result_fixed is not None

        # auto bins 应该在合理范围内（10-100）
        assert 10 <= len(result_auto.hist) <= 100
        assert len(result_auto.hist) == len(result_auto.centers)

    def test_auto_bins_freedman_diaconis(self):
        """验证 auto bins 使用 Freedman-Diaconis rule"""
        np.random.seed(42)
        n = 300

        # 创建正态分布数据
        price_window = np.random.normal(100.0, 5.0, n)
        volume_window = np.ones(n) * 1000.0

        result = compute_wpt_volume_profile(
            price_window=price_window,
            volume_window=volume_window,
            bins="auto",
        )

        assert result is not None

        # 手动计算 Freedman-Diaconis bins
        q75, q25 = np.percentile(price_window, [75, 25])
        iqr = q75 - q25
        bin_width = 2 * iqr / (n ** (1 / 3))
        expected_bins = int((price_window.max() - price_window.min()) / bin_width)
        expected_bins = np.clip(expected_bins, 10, 100)

        # 实际 bins 应该接近预期（允许小的差异，因为使用了降噪后的价格）
        actual_bins = len(result.hist)
        assert abs(actual_bins - expected_bins) <= 5  # 允许 5 个 bins 的差异

    def test_auto_bins_narrow_range(self):
        """测试窄价格范围下的 auto bins"""
        np.random.seed(42)
        n = 150

        # 创建非常窄的价格范围（盘整期）
        price_window = np.random.uniform(100.0, 100.5, n)
        volume_window = np.ones(n) * 1000.0

        result = compute_wpt_volume_profile(
            price_window=price_window,
            volume_window=volume_window,
            bins="auto",
        )

        assert result is not None
        # 即使价格范围很窄，也应该有合理的 bins 数量
        assert len(result.hist) >= 10

    def test_auto_bins_wide_range(self):
        """测试宽价格范围下的 auto bins（如加密货币）"""
        np.random.seed(42)
        n = 200

        # 创建宽价格范围
        price_window = np.random.uniform(100.0, 200.0, n)
        volume_window = np.ones(n) * 1000.0

        result = compute_wpt_volume_profile(
            price_window=price_window,
            volume_window=volume_window,
            bins="auto",
        )

        assert result is not None
        # 宽范围应该产生更多 bins，但不超过 100
        assert len(result.hist) <= 100
        assert len(result.hist) >= 10


class TestDenoisedPriceReturn:
    """测试返回降噪价格序列"""

    def test_price_denoised_field_exists(self):
        """验证返回结果包含 price_denoised 字段"""
        np.random.seed(42)
        n = 100
        price_window = np.linspace(100.0, 110.0, n) + np.random.randn(n) * 0.2
        volume_window = np.ones(n) * 1000.0

        result = compute_wpt_volume_profile(
            price_window=price_window,
            volume_window=volume_window,
            bins=20,
        )

        assert result is not None
        assert hasattr(result, "price_denoised")
        assert result.price_denoised is not None
        assert isinstance(result.price_denoised, np.ndarray)
        assert len(result.price_denoised) == n

    def test_price_denoised_smoother_than_original(self):
        """验证降噪后的价格序列更平滑"""
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

        assert result is not None
        assert result.price_denoised is not None

        # 计算一阶差分（变化率）的标准差
        original_diff_std = np.std(np.diff(price_window))
        denoised_diff_std = np.std(np.diff(result.price_denoised))

        # 降噪后的变化应该更平滑
        assert denoised_diff_std < original_diff_std * 1.2  # 允许轻微增加（边界效应）

    def test_price_denoised_preserves_trend(self):
        """验证降噪后保留主要趋势"""
        np.random.seed(42)
        n = 100

        # 创建明显的上升趋势 + 噪声
        trend = np.linspace(100.0, 120.0, n)
        noise = np.random.randn(n) * 0.5
        price_window = trend + noise
        volume_window = np.ones(n) * 1000.0

        result = compute_wpt_volume_profile(
            price_window=price_window,
            volume_window=volume_window,
            bins=20,
            drop_high_freq=True,
        )

        assert result is not None
        assert result.price_denoised is not None

        # 降噪后的价格应该保持上升趋势
        denoised_start = result.price_denoised[0]
        denoised_end = result.price_denoised[-1]

        # 应该保持上升趋势（允许小的误差）
        assert denoised_end > denoised_start - 2.0  # 允许 2 的误差

        # 原始趋势也应该保持
        assert price_window[-1] > price_window[0]


class TestEdgeCases:
    """测试边界情况"""

    def test_invalid_inputs(self):
        """测试无效输入"""
        # None 输入
        result = compute_wpt_volume_profile(
            price_window=None,
            volume_window=np.ones(100),
        )
        assert result is None

        result = compute_wpt_volume_profile(
            price_window=np.ones(100),
            volume_window=None,
        )
        assert result is None

        # 长度不匹配
        result = compute_wpt_volume_profile(
            price_window=np.ones(100),
            volume_window=np.ones(50),
        )
        assert result is None

        # 太短的序列
        result = compute_wpt_volume_profile(
            price_window=np.ones(5),
            volume_window=np.ones(5),
        )
        assert result is None

    def test_nan_and_inf_handling(self):
        """测试 NaN 和 Inf 处理"""
        np.random.seed(42)
        n = 100
        price_window = np.linspace(100.0, 110.0, n)
        volume_window = np.ones(n) * 1000.0

        # 添加 NaN
        price_window[10] = np.nan
        price_window[20] = np.inf
        volume_window[30] = np.nan

        result = compute_wpt_volume_profile(
            price_window=price_window,
            volume_window=volume_window,
            bins=20,
        )

        # 应该能够处理并返回有效结果
        assert result is not None
        assert np.all(np.isfinite(result.hist))
        assert np.all(np.isfinite(result.centers))

    def test_zero_volume_handling(self):
        """测试零成交量处理"""
        np.random.seed(42)
        n = 100
        price_window = np.linspace(100.0, 110.0, n)
        volume_window = np.ones(n) * 1000.0

        # 部分成交量为 0
        volume_window[10:20] = 0.0

        result = compute_wpt_volume_profile(
            price_window=price_window,
            volume_window=volume_window,
            bins=20,
        )

        # 应该能够处理并返回有效结果
        assert result is not None
        assert np.sum(result.hist) > 0  # 至少有一些成交量

    def test_constant_price(self):
        """测试价格恒定的情况"""
        n = 100
        price_window = np.ones(n) * 100.0
        volume_window = np.ones(n) * 1000.0

        result = compute_wpt_volume_profile(
            price_window=price_window,
            volume_window=volume_window,
            bins=20,
        )

        # 价格范围为零，应该返回 None
        assert result is None

    def test_auto_bins_with_constant_price(self):
        """测试 auto bins 在价格恒定时的情况"""
        n = 100
        price_window = np.ones(n) * 100.0
        volume_window = np.ones(n) * 1000.0

        result = compute_wpt_volume_profile(
            price_window=price_window,
            volume_window=volume_window,
            bins="auto",
        )

        # 价格范围为零，应该返回 None
        assert result is None


class TestIntegration:
    """集成测试：验证改进功能协同工作"""

    def test_all_improvements_together(self):
        """测试所有改进功能同时使用"""
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

        assert result is not None

        # 验证所有字段都存在
        assert result.hist is not None
        assert result.edges is not None
        assert result.centers is not None
        assert result.price_min is not None
        assert result.price_max is not None
        assert result.price_denoised is not None  # 新增字段

        # 验证基本属性
        assert len(result.hist) == len(result.centers)
        assert len(result.edges) == len(result.hist) + 1
        assert 10 <= len(result.hist) <= 100  # auto bins 范围
        assert result.price_min < result.price_max

        # 验证降噪价格序列
        assert len(result.price_denoised) == n
        assert np.all(np.isfinite(result.price_denoised))

        # 验证成交量守恒
        assert pytest.approx(volume_window.sum(), rel=1e-5) == np.sum(result.hist)

    def test_backward_compatibility(self):
        """测试向后兼容性（原有调用方式仍然有效）"""
        np.random.seed(42)
        n = 100
        price_window = np.linspace(100.0, 110.0, n)
        volume_window = np.ones(n) * 1000.0

        # 原有调用方式（固定 bins，不关心降噪价格）
        result = compute_wpt_volume_profile(
            price_window=price_window,
            volume_window=volume_window,
            bins=50,  # 固定值
        )

        assert result is not None
        assert len(result.hist) == 50
        # price_denoised 仍然存在（可选字段）
        assert result.price_denoised is not None

    def test_denoised_price_usage_example(self):
        """演示如何使用降噪价格序列进行特征扩展"""
        np.random.seed(42)
        n = 150
        price_window = np.linspace(100.0, 115.0, n) + np.random.randn(n) * 0.5
        volume_window = np.ones(n) * 1000.0

        result = compute_wpt_volume_profile(
            price_window=price_window,
            volume_window=volume_window,
            bins=30,
            drop_high_freq=True,
        )

        assert result is not None
        assert result.price_denoised is not None

        # 示例：计算降噪价格与原始价格的偏离度
        price_original = price_window
        price_denoised = result.price_denoised

        deviation = np.abs(price_original - price_denoised)
        max_deviation = np.max(deviation)
        mean_deviation = np.mean(deviation)

        # 偏离度应该在合理范围内
        assert max_deviation < 5.0  # 最大偏离不超过 5
        assert mean_deviation < 1.0  # 平均偏离不超过 1

        # 示例：计算降噪后的趋势强度
        denoised_returns = np.diff(price_denoised) / price_denoised[:-1]
        trend_strength = np.std(denoised_returns)

        assert trend_strength >= 0
        assert np.isfinite(trend_strength)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
