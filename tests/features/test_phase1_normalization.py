"""
Phase 1 归一化测试

验证 Phase 1 归一化的正确性和数据分布：
1. ATR: atr / close -> [0.001, 0.1]
2. MACD: macd / atr -> [-3, 3]
3. BB Width: bb_width_normalized, bb_position
4. POC/HAL: (level - close) / atr -> [-5, 5]
5. SR Strength: dist_to_nearest_sr 归一化为 ATR 倍数
"""

import numpy as np
import pandas as pd
import pytest

from src.features.time_series.baseline_features import (
    compute_atr_from_series,
    compute_macd_from_series,
    compute_bb_width_features_from_series,
    compute_poc_hal_features_from_series,
)


def create_realistic_data(
    n_samples: int = 500,
    base_price: float = 100,
    volatility: float = 0.02,
    seed: int = 42,
) -> pd.DataFrame:
    """创建模拟真实市场的数据"""
    np.random.seed(seed)
    dates = pd.date_range("2024-01-01", periods=n_samples, freq="4H")

    # 生成带有趋势和波动的价格
    returns = np.random.randn(n_samples) * volatility
    prices = base_price * np.exp(np.cumsum(returns))

    # 生成 OHLCV
    high = prices * (1 + np.abs(np.random.randn(n_samples) * 0.005))
    low = prices * (1 - np.abs(np.random.randn(n_samples) * 0.005))
    open_ = prices * (1 + np.random.randn(n_samples) * 0.003)
    volume = np.random.uniform(1000, 10000, n_samples)

    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": prices,
            "volume": volume,
        },
        index=dates,
    )


class TestPhase1Normalization:
    """Phase 1 归一化测试"""

    def test_atr_normalization(self):
        """测试 ATR 归一化: atr / close"""
        df = create_realistic_data(200)

        result = compute_atr_from_series(
            high=df["high"],
            low=df["low"],
            close=df["close"],
            period=14,
        )

        atr = result["atr"].dropna()

        # 验证基本属性
        assert len(atr) > 100, "应该有足够的有效值"
        assert (atr >= 0).all(), "归一化 ATR 应该 >= 0"

        # 验证归一化范围
        # 对于正常波动的资产，atr/close 通常在 [0.001, 0.1]
        assert atr.mean() < 0.1, f"归一化 ATR 均值应该 < 0.1，实际: {atr.mean():.4f}"
        assert atr.max() < 0.5, f"归一化 ATR 最大值应该 < 0.5，实际: {atr.max():.4f}"

        # 验证分布合理性
        assert atr.std() > 0.0001, "ATR 应该有一定的变化"

    def test_atr_cross_asset_comparability(self):
        """测试 ATR 跨资产可比性"""
        # 不同价格水平的资产
        assets = {
            "high_price": create_realistic_data(200, base_price=50000, volatility=0.02),
            "mid_price": create_realistic_data(200, base_price=3000, volatility=0.02),
            "low_price": create_realistic_data(200, base_price=100, volatility=0.02),
        }

        atr_stats = {}
        for name, df in assets.items():
            result = compute_atr_from_series(
                high=df["high"],
                low=df["low"],
                close=df["close"],
            )
            atr = result["atr"].dropna()
            atr_stats[name] = {"mean": atr.mean(), "std": atr.std()}

        # 归一化后，不同价格水平的资产应该有相似的 ATR 分布
        means = [s["mean"] for s in atr_stats.values()]
        # 允许 5 倍以内的差异（因为随机波动）
        assert (
            max(means) / (min(means) + 1e-10) < 5
        ), f"归一化 ATR 应该跨资产可比，实际: {atr_stats}"

    def test_macd_normalization(self):
        """测试 MACD 归一化: macd / atr"""
        df = create_realistic_data(200)

        result = compute_macd_from_series(
            series=df["close"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
        )

        for col in ["macd", "macd_signal", "macd_histogram"]:
            vals = result[col].dropna()

            # 验证基本属性
            assert len(vals) > 100, f"{col} 应该有足够的有效值"

            # 验证归一化范围
            # 归一化后应该在 [-5, 5] 范围内（99% 分位数）
            q99 = vals.abs().quantile(0.99)
            assert q99 < 10, f"{col} 99%分位应该 < 10，实际: {q99:.2f}"

            # 验证分布合理性（不是常量）
            assert vals.std() > 0.01, f"{col} 应该有一定的变化"

    def test_bb_width_normalization(self):
        """测试 BB Width 归一化"""
        df = create_realistic_data(200)

        result = compute_bb_width_features_from_series(
            close=df["close"],
            high=df["high"],
            low=df["low"],
        )

        # 验证输出列（只有归一化的列）
        assert "bb_width_normalized" in result.columns
        assert "bb_position" in result.columns
        # 不应该有原始价格列
        assert "bb_upper" not in result.columns
        assert "bb_lower" not in result.columns
        assert "bb_middle" not in result.columns

        # bb_position 应该在 [0, 1]
        bb_pos = result["bb_position"].dropna()
        assert (bb_pos >= 0).all(), "bb_position 应该 >= 0"
        assert (bb_pos <= 1).all(), "bb_position 应该 <= 1"
        assert (
            bb_pos.mean() > 0.3 and bb_pos.mean() < 0.7
        ), f"bb_position 均值应该接近 0.5，实际: {bb_pos.mean():.2f}"

        # bb_width_normalized 应该是正数
        bb_width = result["bb_width_normalized"].dropna()
        assert (bb_width >= 0).all(), "bb_width_normalized 应该 >= 0"
        assert (
            bb_width.mean() < 10
        ), f"bb_width_normalized 均值应该合理，实际: {bb_width.mean():.2f}"

    def test_poc_hal_normalization(self):
        """测试 POC/HAL 归一化: (level - close) / atr"""
        df = create_realistic_data(400)  # 需要更多数据

        result = compute_poc_hal_features_from_series(
            high=df["high"],
            low=df["low"],
            close=df["close"],
            volume=df["volume"],
            poc_window=160,
        )

        for col in ["poc", "hal_high", "hal_low", "hal_mid"]:
            vals = result[col].dropna()

            if len(vals) > 50:
                # 归一化后的值表示 ATR 倍数，可正可负
                # 99% 分位数应该在 [-10, 10]
                q99 = vals.abs().quantile(0.99)
                assert q99 < 15, f"{col} 99%分位应该 < 15，实际: {q99:.2f}"

                # 验证分布合理性
                assert vals.std() > 0.01, f"{col} 应该有一定的变化"

    def test_poc_hal_cross_asset_comparability(self):
        """测试 POC/HAL 跨资产可比性"""
        assets = {
            "high_price": create_realistic_data(400, base_price=50000),
            "low_price": create_realistic_data(400, base_price=100),
        }

        poc_stats = {}
        for name, df in assets.items():
            result = compute_poc_hal_features_from_series(
                high=df["high"],
                low=df["low"],
                close=df["close"],
                volume=df["volume"],
                poc_window=160,
            )
            poc = result["poc"].dropna()
            if len(poc) > 50:
                poc_stats[name] = {
                    "mean": poc.mean(),
                    "std": poc.std(),
                    "q99": poc.abs().quantile(0.99),
                }

        if len(poc_stats) >= 2:
            # 归一化后，不同价格水平的资产应该有相似的分布
            q99_values = [s["q99"] for s in poc_stats.values()]
            # 允许 5 倍以内的差异
            assert (
                max(q99_values) / (min(q99_values) + 1e-10) < 5
            ), f"归一化 POC 应该跨资产可比，实际: {poc_stats}"

    def test_no_inf_nan_overflow(self):
        """测试没有 inf/nan/overflow"""
        df = create_realistic_data(300)

        # ATR
        atr_result = compute_atr_from_series(
            high=df["high"], low=df["low"], close=df["close"]
        )
        assert not np.isinf(atr_result["atr"]).any(), "ATR 不应该有 inf"

        # MACD
        macd_result = compute_macd_from_series(
            series=df["close"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
        )
        for col in macd_result.columns:
            assert not np.isinf(macd_result[col]).any(), f"{col} 不应该有 inf"

        # BB Width
        bb_result = compute_bb_width_features_from_series(
            close=df["close"], high=df["high"], low=df["low"]
        )
        for col in bb_result.columns:
            assert not np.isinf(bb_result[col]).any(), f"{col} 不应该有 inf"

        # POC/HAL
        poc_result = compute_poc_hal_features_from_series(
            high=df["high"],
            low=df["low"],
            close=df["close"],
            volume=df["volume"],
            poc_window=160,
        )
        for col in poc_result.columns:
            assert not np.isinf(poc_result[col]).any(), f"{col} 不应该有 inf"

    def test_edge_cases(self):
        """测试边界情况"""
        # 很少的数据
        df_small = create_realistic_data(50)

        # ATR 应该还能工作
        atr_result = compute_atr_from_series(
            high=df_small["high"],
            low=df_small["low"],
            close=df_small["close"],
        )
        assert len(atr_result) == 50

        # 零波动
        df_flat = pd.DataFrame(
            {
                "open": [100.0] * 100,
                "high": [100.0] * 100,
                "low": [100.0] * 100,
                "close": [100.0] * 100,
                "volume": [1000.0] * 100,
            },
            index=pd.date_range("2024-01-01", periods=100, freq="4H"),
        )

        atr_flat = compute_atr_from_series(
            high=df_flat["high"],
            low=df_flat["low"],
            close=df_flat["close"],
        )
        # 零波动应该返回 0 或接近 0
        assert (atr_flat["atr"].dropna() < 0.01).all(), "零波动的 ATR 应该接近 0"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
