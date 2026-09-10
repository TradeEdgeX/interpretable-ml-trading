"""
测试 inf 值根本原因修复

验证所有修复都能正确处理输入数据中的 inf 值，而不是简单地替换为 NaN
"""

import pytest
import numpy as np
import pandas as pd

# 添加项目根目录到路径

from src.features.time_series.baseline_features import (
    compute_rsi,
    calculate_sqs,
    _get_sr_boundary_definitions,
    _compute_boundary_strengths,
)
from src.features.time_series.utils_hurst_features import (
    compute_hurst_dfa,
    extract_hurst_features,
)
from src.features.time_series.utils_order_flow_features import (
    compute_trade_clustering_from_ticks,
)


class TestInfRootCauseFixes:
    """测试 inf 值根本原因修复"""

    def test_sr_strength_max_with_inf_volume(self):
        """测试 sr_strength_max 在 volume 包含 inf 时的处理"""
        # 创建包含 inf volume 的测试数据
        n_samples = 100
        dates = pd.date_range("2024-01-01", periods=n_samples, freq="1H")
        data = pd.DataFrame(
            {
                "open": 100.0 + np.random.randn(n_samples) * 0.1,
                "high": 100.5 + np.random.randn(n_samples) * 0.1,
                "low": 99.5 + np.random.randn(n_samples) * 0.1,
                "close": 100.0 + np.random.randn(n_samples) * 0.1,
                "volume": 1000.0 + np.random.randn(n_samples) * 100,
                "atr": 0.5 + np.random.randn(n_samples) * 0.1,
            },
            index=dates,
        )

        # 在 volume 中插入 inf 值
        data.loc[dates[10:15], "volume"] = np.inf
        data.loc[dates[20:25], "volume"] = -np.inf

        # 计算 SR 强度特征
        boundaries = _get_sr_boundary_definitions(data)
        boundary_strengths = _compute_boundary_strengths(
            data,
            boundaries,
            window=60,
            tolerance_factor=0.5,
        )

        # 检查结果
        for name, series in boundary_strengths.items():
            # 不应该包含 inf 值
            assert not np.isinf(series).any(), f"{name} contains inf values"
            # 不应该包含 -inf 值
            assert not np.isinf(-series).any(), f"{name} contains -inf values"

        # 计算 sr_strength_max
        strength_columns = list(boundary_strengths.keys())
        if strength_columns:
            strength_df = data[strength_columns]
            sr_strength_max = strength_df.max(axis=1)
            # 不应该包含 inf 值
            assert not np.isinf(
                sr_strength_max
            ).any(), "sr_strength_max contains inf values"
            assert not np.isinf(
                -sr_strength_max
            ).any(), "sr_strength_max contains -inf values"

    def test_hurst_features_with_inf_input(self):
        """测试 Hurst 特征在输入数据包含 inf 时的处理"""
        # 创建包含 inf 的测试数据
        n_samples = 200
        dates = pd.date_range("2024-01-01", periods=n_samples, freq="1H")
        data = pd.DataFrame(
            {
                "close": 100.0 + np.cumsum(np.random.randn(n_samples) * 0.1),
                "cvd": np.cumsum(np.random.randn(n_samples) * 10),
                "volume": 1000.0 + np.random.randn(n_samples) * 100,
            },
            index=dates,
        )

        # 在价格中插入 inf 值
        data.loc[dates[50:55], "close"] = np.inf
        data.loc[dates[100:105], "close"] = -np.inf

        # 计算 Hurst 特征
        extract_hurst_features(
            data,
            price_col="close",
            cvd_col="cvd",
            volume_col="volume",
            rolling_window=50,
            update_freq=1,
            clip_pct=0.5,
        )

        # 检查结果
        for col in ["hurst_price_rolling", "hurst_cvd_rolling", "hurst_volume_rolling"]:
            if col in data.columns:
                # 不应该包含 inf 值
                assert not np.isinf(data[col]).any(), f"{col} contains inf values"
                assert not np.isinf(-data[col]).any(), f"{col} contains -inf values"

    def test_rsi_with_inf_input(self):
        """测试 RSI 在输入数据包含 inf 时的处理"""
        # 创建包含 inf 的测试数据
        n_samples = 100
        dates = pd.date_range("2024-01-01", periods=n_samples, freq="1H")
        price_series = pd.Series(
            100.0 + np.cumsum(np.random.randn(n_samples) * 0.1), index=dates
        )

        # 在价格中插入 inf 值
        price_series.iloc[20:25] = np.inf
        price_series.iloc[50:55] = -np.inf

        # 计算 RSI
        rsi_series = compute_rsi(price_series, period=14)

        # 检查结果
        # 不应该包含 inf 值
        assert not np.isinf(rsi_series).any(), "RSI contains inf values"
        assert not np.isinf(-rsi_series).any(), "RSI contains -inf values"

    def test_trade_clustering_with_inf_input(self):
        """测试 Trade Clustering 在输入数据包含 inf 时的处理"""
        # 创建包含 inf 的测试 tick 数据
        n_ticks = 1000
        timestamps = pd.date_range("2024-01-01", periods=n_ticks, freq="1S")
        ticks = pd.DataFrame(
            {
                "price": 100.0 + np.random.randn(n_ticks) * 0.1,
                "volume": 1.0 + np.random.rand(n_ticks) * 0.5,
                "side": np.random.choice([1, -1], n_ticks),
            },
            index=timestamps,
        )

        # 在 volume 中插入 inf 值（虽然不应该有，但测试健壮性）
        ticks.loc[timestamps[100:110], "volume"] = np.inf
        ticks.loc[timestamps[200:210], "volume"] = -np.inf

        # 计算 Trade Clustering
        cluster_df, final_state = compute_trade_clustering_from_ticks(
            ticks,
            window_size=100,
            initial_state=None,
        )

        # 检查结果
        for col in cluster_df.columns:
            # 不应该包含 inf 值
            assert not np.isinf(cluster_df[col]).any(), f"{col} contains inf values"
            assert not np.isinf(-cluster_df[col]).any(), f"{col} contains -inf values"

    def test_hurst_dfa_with_inf_input(self):
        """测试 compute_hurst_dfa 在输入数据包含 inf 时的处理"""
        # 创建包含 inf 的测试序列
        n_samples = 100
        returns = np.random.randn(n_samples) * 0.01
        returns[20:25] = np.inf
        returns[50:55] = -np.inf

        # 计算 Hurst
        hurst = compute_hurst_dfa(returns)

        # 检查结果
        # 应该返回 NaN（因为输入包含 inf），而不是 inf
        assert not np.isinf(hurst), "Hurst should return NaN for inf input, not inf"
        assert np.isnan(hurst) or (
            0.0 <= hurst <= 1.0
        ), "Hurst should be NaN or in [0, 1]"

    def test_price_trend_calculation_with_zero_price(self):
        """测试价格趋势计算在起始价格为 0 时的处理"""
        # 创建起始价格为 0 的测试数据
        n_samples = 100
        dates = pd.date_range("2024-01-01", periods=n_samples, freq="1H")
        data = pd.DataFrame(
            {
                "open": 100.0 + np.random.randn(n_samples) * 0.1,
                "high": 100.5 + np.random.randn(n_samples) * 0.1,
                "low": 99.5 + np.random.randn(n_samples) * 0.1,
                "close": 100.0 + np.random.randn(n_samples) * 0.1,
                "volume": 1000.0 + np.random.randn(n_samples) * 100,
                "atr": 0.5 + np.random.randn(n_samples) * 0.1,
            },
            index=dates,
        )

        # 设置起始价格为 0
        data.loc[dates[10], "close"] = 0.0

        # 计算 SR 强度特征
        boundaries = _get_sr_boundary_definitions(data)
        boundary_strengths = _compute_boundary_strengths(
            data,
            boundaries,
            window=60,
            tolerance_factor=0.5,
        )

        # 检查结果
        for name, series in boundary_strengths.items():
            # 不应该包含 inf 值
            assert not np.isinf(
                series
            ).any(), f"{name} contains inf values after zero price"

    def test_volume_ratio_calculation_with_inf(self):
        """测试成交量比率计算在 volume 包含 inf 时的处理"""
        # 创建包含 inf volume 的测试数据
        n_samples = 100
        dates = pd.date_range("2024-01-01", periods=n_samples, freq="1H")
        data = pd.DataFrame(
            {
                "open": 100.0 + np.random.randn(n_samples) * 0.1,
                "high": 100.5 + np.random.randn(n_samples) * 0.1,
                "low": 99.5 + np.random.randn(n_samples) * 0.1,
                "close": 100.0 + np.random.randn(n_samples) * 0.1,
                "volume": 1000.0 + np.random.randn(n_samples) * 100,
                "atr": 0.5 + np.random.randn(n_samples) * 0.1,
            },
            index=dates,
        )

        # 在 volume 中插入 inf 值
        data.loc[dates[10:15], "volume"] = np.inf

        # 计算 SQS（会使用 volume 计算 vol_ratio）
        sr_price = 100.0
        window_df = data.tail(60)

        # 直接测试 calculate_sqs
        sqs = calculate_sqs(
            sr_price,
            window_df,
            window=60,
            tolerance_factor=0.5,
            sr_type="support",
            use_volume_confirmation=True,
        )

        # 检查结果
        assert np.isfinite(sqs), "SQS should be finite even with inf volume"
        assert sqs >= 0, "SQS should be non-negative"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
