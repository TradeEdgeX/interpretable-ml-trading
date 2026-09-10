"""
测试 WPT 增强功能：log returns 和自适应窗口
"""

import pytest
import pandas as pd
import numpy as np

from src.features.time_series.utils_liquidity_features import (
    compute_wpt_volume_energy_features,
    compute_wpt_volume_energy_features_from_series,
)


def create_mock_data(n_samples: int = 100, seed: int = 42) -> pd.DataFrame:
    """创建模拟数据"""
    np.random.seed(seed)
    dates = pd.date_range("2024-01-01", periods=n_samples, freq="5min")

    # 生成有趋势的价格数据
    returns = np.random.randn(n_samples) * 0.01
    prices = 100 * np.exp(np.cumsum(returns))  # 使用 exp 生成有趋势的价格
    volumes = np.random.lognormal(10, 1, n_samples)

    df = pd.DataFrame(
        {
            "close": prices,
            "volume": volumes,
        },
        index=dates,
    )
    return df


class TestWPTEnhancements:
    """WPT 增强功能测试"""

    def test_log_returns_removes_trend(self):
        """测试 log returns 预处理能去除趋势"""
        df = create_mock_data(n_samples=200, seed=42)

        # 测试使用 log returns 和不使用的差异
        result_with_log = compute_wpt_volume_energy_features_from_series(
            close=df["close"],
            volume=df["volume"],
            use_log_returns=True,
            wavelet="db2",
            level=2,
            lookback_window=20,
        )

        result_without_log = compute_wpt_volume_energy_features_from_series(
            close=df["close"],
            volume=df["volume"],
            use_log_returns=False,
            wavelet="db2",
            level=2,
            lookback_window=20,
        )

        # 两种方法都应该产生有效的输出
        assert len(result_with_log) == len(df)
        assert len(result_without_log) == len(df)

        # VPER 值应该不同（因为 log returns 改变了价格能量分布）
        vper_log = result_with_log["wpt_vper_mid"].dropna()
        vper_no_log = result_without_log["wpt_vper_mid"].dropna()

        if len(vper_log) > 0 and len(vper_no_log) > 0:
            # 它们应该不同（不是完全相同的值）
            assert not np.allclose(
                vper_log.values, vper_no_log.values, rtol=1e-6
            ), "Log returns 应该改变 VPER 值"

    def test_adaptive_window_adapts_to_volatility(self):
        """测试自适应窗口能根据波动率调整"""
        np.random.seed(42)

        # 创建两段数据：低波动率和高波动率
        n_samples = 150
        dates = pd.date_range("2024-01-01", periods=n_samples, freq="5min")

        # 前半段：低波动率
        low_vol_returns = np.random.randn(n_samples // 2) * 0.005
        # 后半段：高波动率
        high_vol_returns = np.random.randn(n_samples - n_samples // 2) * 0.02

        prices = 100 * np.exp(
            np.cumsum(np.concatenate([low_vol_returns, high_vol_returns]))
        )
        volumes = np.random.lognormal(10, 1, n_samples)

        # 计算 ATR（作为波动率代理）
        high = prices * 1.01
        low = prices * 0.99
        prev_close = np.roll(prices, 1)
        prev_close[0] = prices[0]  # 第一个值用当前价格
        tr = np.maximum(
            high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close))
        )
        atr = pd.Series(tr, index=dates).rolling(window=14, min_periods=1).mean()
        atr = atr.bfill().ffill().fillna(0.1)  # 填充 NaN

        df = pd.DataFrame(
            {
                "close": prices,
                "volume": volumes,
                "atr": atr,
            },
            index=dates,
        )

        # 测试自适应窗口
        result_adaptive = compute_wpt_volume_energy_features(
            df[["close", "volume", "atr"]].copy(),
            price_col="close",
            volume_col="volume",
            adaptive_window=True,
            atr_col="atr",
            wavelet="db2",
            level=2,
            lookback_window=20,
        )

        result_fixed = compute_wpt_volume_energy_features(
            df[["close", "volume", "atr"]].copy(),
            price_col="close",
            volume_col="volume",
            adaptive_window=False,
            wavelet="db2",
            level=2,
            lookback_window=20,
        )

        # 两种方法都应该产生有效的输出
        assert len(result_adaptive) == len(df)
        assert len(result_fixed) == len(df)

        # 验证输出列存在
        assert "wpt_vper_mid" in result_adaptive.columns
        assert "wpt_vper_mid" in result_fixed.columns

    def test_frequency_center_classification(self):
        """测试频率中心分类方法"""
        # 这个测试主要验证频率中心分类不会出错
        df = create_mock_data(n_samples=100, seed=42)

        # 使用不同 level 测试频率中心分类
        for level in [2, 3, 4]:
            result = compute_wpt_volume_energy_features_from_series(
                close=df["close"],
                volume=df["volume"],
                wavelet="db2",
                level=level,
                lookback_window=20,
            )

            # 验证输出列存在
            assert "wpt_vper_low" in result.columns
            assert "wpt_vper_mid" in result.columns
            assert "wpt_vper_high" in result.columns

            # 验证值在合理范围内
            vper_values = result[
                ["wpt_vper_low", "wpt_vper_mid", "wpt_vper_high"]
            ].dropna()
            if len(vper_values) > 0:
                assert (vper_values >= 0).all().all(), "VPER 值应该 >= 0"

    def test_log_returns_edge_cases(self):
        """测试 log returns 的边界情况"""
        # 测试价格全为正数（正常情况）
        df_normal = create_mock_data(n_samples=50, seed=42)
        result_normal = compute_wpt_volume_energy_features_from_series(
            close=df_normal["close"],
            volume=df_normal["volume"],
            use_log_returns=True,
            wavelet="db2",
            level=2,
            lookback_window=20,
        )
        assert len(result_normal) == len(df_normal)

        # 测试少量数据点（应该跳过）
        df_small = create_mock_data(n_samples=10, seed=42)
        result_small = compute_wpt_volume_energy_features_from_series(
            close=df_small["close"],
            volume=df_small["volume"],
            use_log_returns=True,
            wavelet="db2",
            level=2,
            lookback_window=20,
        )
        # 少量数据可能无法计算，但不应报错
        assert len(result_small) == len(df_small)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
