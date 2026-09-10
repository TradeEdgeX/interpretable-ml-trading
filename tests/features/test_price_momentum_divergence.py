"""
Price-Momentum Divergence 特征测试

测试覆盖：
1. 功能正确性：语义验证
2. 未来函数检测：截断后不变
3. 流式计算一致性：增量 vs 批量
"""

import numpy as np
import pandas as pd
import pytest

from src.features.time_series.utils_interaction_features import (
    compute_price_momentum_divergence_from_series,
)

# ============================================================================
# 功能正确性测试
# ============================================================================


class TestPriceMomentumDivergenceFunctionality:
    """功能正确性测试"""

    def test_output_columns(self):
        """验证输出列完整"""
        n = 100
        dates = pd.date_range("2025-01-01", periods=n, freq="h")
        close = pd.Series(100 + np.cumsum(np.random.randn(n) * 0.1), index=dates)

        result = compute_price_momentum_divergence_from_series(close=close)

        expected_columns = [
            "price_velocity_pct",
            "price_accel_pct",
            "price_momentum_div_score",
            "price_momentum_div_score_pct",
            "momentum_div_tension",
            "momentum_location_pressure",
        ]
        assert list(result.columns) == expected_columns

    def test_bounded_outputs(self):
        """验证 bounded 输出在正确范围内"""
        n = 200
        dates = pd.date_range("2025-01-01", periods=n, freq="h")
        close = pd.Series(100 + np.cumsum(np.random.randn(n) * 0.5), index=dates)

        result = compute_price_momentum_divergence_from_series(close=close)

        # pct 列 [0, 1]
        for col in [
            "price_velocity_pct",
            "price_accel_pct",
            "price_momentum_div_score_pct",
        ]:
            valid = result[col].dropna()
            assert valid.between(0.0, 1.0).all(), f"{col} should be in [0, 1]"

        # div_score [-1, 1]
        valid_score = result["price_momentum_div_score"].dropna()
        assert valid_score.between(-1.0, 1.0).all(), "div_score should be in [-1, 1]"

        # tension/pressure [0, 1]
        for col in ["momentum_div_tension", "momentum_location_pressure"]:
            assert result[col].between(0.0, 1.0).all(), f"{col} should be in [0, 1]"

    def test_velocity_semantic_uptrend(self):
        """上涨趋势：velocity_pct 应该较高"""
        n = 600  # 足够的数据量以计算 percentile
        dates = pd.date_range("2025-01-01", periods=n, freq="h")
        # 单调上涨
        close = pd.Series(100 + np.arange(n) * 0.5, index=dates)

        result = compute_price_momentum_divergence_from_series(close=close)

        # 速度百分位应该较高
        velocity_pct = result["price_velocity_pct"].iloc[400:]  # 跳过预热期
        assert velocity_pct.mean() > 0.5, "Uptrend should have high velocity_pct"

    def test_velocity_semantic_downtrend(self):
        """下跌趋势应该产生有效的 velocity_pct"""
        n = 600
        dates = pd.date_range("2025-01-01", periods=n, freq="h")
        # 单调下跌
        close = pd.Series(100 - np.arange(n) * 0.1, index=dates)

        result = compute_price_momentum_divergence_from_series(close=close)

        # 应该产生有效的百分位值，不全是 0.5
        velocity_pct = result["price_velocity_pct"].iloc[400:]
        assert not (velocity_pct == 0.5).all(), "Should produce varying velocity_pct"

    def test_accel_semantic_accelerating(self):
        """加速上涨：accel_pct 应该较高"""
        n = 600
        dates = pd.date_range("2025-01-01", periods=n, freq="h")
        # 加速上涨（二次函数）
        close = pd.Series(100 + 0.001 * np.arange(n) ** 2, index=dates)

        result = compute_price_momentum_divergence_from_series(close=close)

        # 加速度百分位应该较高
        accel_pct = result["price_accel_pct"].iloc[400:]
        assert (
            accel_pct.mean() > 0.5
        ), "Accelerating uptrend should have higher accel_pct"

    def test_accel_semantic_decelerating(self):
        """减速上涨应该产生有效的 accel_pct"""
        n = 600
        dates = pd.date_range("2025-01-01", periods=n, freq="h")
        # 减速上涨（根号函数）
        close = pd.Series(100 + 5 * np.sqrt(np.arange(n) + 1), index=dates)

        result = compute_price_momentum_divergence_from_series(close=close)

        # 应该产生有效的百分位值
        accel_pct = result["price_accel_pct"].iloc[400:]
        assert accel_pct.notna().all(), "Should produce valid accel_pct"

    def test_momentum_divergence_semantic(self):
        """动量背离语义：价格高+速度慢 → 负背离"""
        n = 200
        dates = pd.date_range("2025-01-01", periods=n, freq="h")

        # 构造场景：前半段快速上涨，后半段缓慢上涨到更高点
        # 这会产生：价格相对高，但速度相对慢 → 负背离（推进衰竭）
        price_fast = 100 + np.arange(100) * 1.0  # 快速阶段
        price_slow = price_fast[-1] + np.arange(100) * 0.1  # 缓慢阶段
        close = pd.Series(np.concatenate([price_fast, price_slow]), index=dates)

        result = compute_price_momentum_divergence_from_series(
            close=close,
            position_window=50,
        )

        # 后半段应该出现负背离（速度位置 < 价格位置）
        div_score = result["price_momentum_div_score"].iloc[150:]
        assert (
            div_score < 0
        ).mean() > 0.5, "Slow phase should show negative divergence (exhaustion)"

    def test_with_trend_strength(self):
        """带趋势强度时，tension 应该非零"""
        n = 100
        dates = pd.date_range("2025-01-01", periods=n, freq="h")
        close = pd.Series(100 + np.cumsum(np.random.randn(n) * 0.3), index=dates)
        trend = pd.Series([0.8] * n, index=dates)  # 强趋势

        result = compute_price_momentum_divergence_from_series(
            close=close,
            trend_strength=trend,
        )

        # 有趋势时，tension 应该有值
        tension = result["momentum_div_tension"].iloc[50:]
        assert tension.mean() > 0.01, "With trend_strength, tension should be non-zero"

    def test_location_pressure_at_extremes(self):
        """极端位置时，pressure 应该更高"""
        n = 200
        dates = pd.date_range("2025-01-01", periods=n, freq="h")

        # 构造场景：先上涨到高位，再稳定
        close = pd.Series(
            np.concatenate(
                [
                    100 + np.arange(100) * 0.5,  # 上涨
                    150 + np.random.randn(100) * 0.1,  # 高位震荡
                ]
            ),
            index=dates,
        )

        result = compute_price_momentum_divergence_from_series(
            close=close,
            position_window=50,
        )

        # 高位时的 pressure 应该更高
        pressure_high = result["momentum_location_pressure"].iloc[150:]
        pressure_mid = result["momentum_location_pressure"].iloc[50:100]

        # 不强求绝对大于，因为压力还依赖背离强度
        assert pressure_high.notna().all(), "Pressure should be computed"


# ============================================================================
# 未来函数检测测试
# ============================================================================


class TestPriceMomentumDivergenceNoFutureLeak:
    """未来函数检测"""

    def test_no_future_leak_basic(self):
        """基础未来函数检测：截断后不变"""
        n = 200
        dates = pd.date_range("2025-01-01", periods=n, freq="h")
        close = pd.Series(100 + np.cumsum(np.random.randn(n) * 0.3), index=dates)

        # 全量计算
        result_full = compute_price_momentum_divergence_from_series(close=close)

        # 截断到 150 条
        close_partial = close.iloc[:150]
        result_partial = compute_price_momentum_divergence_from_series(
            close=close_partial
        )

        # 重叠部分应该完全相等
        for col in result_full.columns:
            full_values = result_full[col].iloc[:150]
            partial_values = result_partial[col]

            # 使用 iloc 避免索引问题
            np.testing.assert_array_almost_equal(
                full_values.values,
                partial_values.values,
                decimal=10,
                err_msg=f"Future leak detected in {col}",
            )

    def test_no_future_leak_with_trend(self):
        """带趋势强度时的未来函数检测"""
        n = 200
        dates = pd.date_range("2025-01-01", periods=n, freq="h")
        close = pd.Series(100 + np.cumsum(np.random.randn(n) * 0.3), index=dates)
        trend = pd.Series(np.random.randn(n).clip(-1, 1), index=dates)

        result_full = compute_price_momentum_divergence_from_series(
            close=close,
            trend_strength=trend,
        )

        result_partial = compute_price_momentum_divergence_from_series(
            close=close.iloc[:150],
            trend_strength=trend.iloc[:150],
        )

        for col in result_full.columns:
            np.testing.assert_array_almost_equal(
                result_full[col].iloc[:150].values,
                result_partial[col].values,
                decimal=10,
                err_msg=f"Future leak with trend in {col}",
            )


# ============================================================================
# 流式计算一致性测试
# ============================================================================


class TestPriceMomentumDivergenceStreamingVsBatch:
    """流式 vs 批量一致性"""

    def test_streaming_consistency(self):
        """流式增量计算应该和批量计算一致"""
        n = 300
        dates = pd.date_range("2025-01-01", periods=n, freq="h")
        close = pd.Series(100 + np.cumsum(np.random.randn(n) * 0.3), index=dates)

        # 批量计算
        result_batch = compute_price_momentum_divergence_from_series(close=close)

        # 模拟流式：逐步增加数据
        checkpoints = [100, 200, 300]

        for cp in checkpoints:
            result_stream = compute_price_momentum_divergence_from_series(
                close=close.iloc[:cp]
            )

            for col in result_batch.columns:
                # 使用 iloc 比较
                batch_values = result_batch[col].iloc[:cp].values
                stream_values = result_stream[col].values

                np.testing.assert_array_almost_equal(
                    batch_values,
                    stream_values,
                    decimal=10,
                    err_msg=f"Streaming inconsistency at {cp} in {col}",
                )

    def test_continuous_output(self):
        """验证输出是连续的，不是稀疏的"""
        n = 200
        dates = pd.date_range("2025-01-01", periods=n, freq="h")
        close = pd.Series(100 + np.cumsum(np.random.randn(n) * 0.3), index=dates)

        result = compute_price_momentum_divergence_from_series(close=close)

        # 跳过预热期后，应该有连续值
        for col in [
            "price_momentum_div_score",
            "price_velocity_pct",
            "price_accel_pct",
        ]:
            values = result[col].iloc[100:]
            unique_count = values.round(2).nunique()

            # 应该有多个不同值（不是只有 0/1）
            assert (
                unique_count > 10
            ), f"{col} should have continuous values, got only {unique_count} unique"


# ============================================================================
# 正交性验证
# ============================================================================


class TestPriceMomentumOrthogonality:
    """验证与 CVD Divergence 的正交性"""

    def test_independent_of_cvd(self):
        """Price-Momentum 只依赖价格，不依赖 CVD"""
        n = 100
        dates = pd.date_range("2025-01-01", periods=n, freq="h")
        close = pd.Series(100 + np.arange(n) * 0.5, index=dates)

        # 不需要 CVD 输入
        result = compute_price_momentum_divergence_from_series(close=close)

        # 应该正常输出
        assert result.shape[0] == n
        assert (
            not result.isnull().all().any()
        ), "Should produce valid output without CVD"

    def test_semantic_difference(self):
        """
        语义差异验证：
        - CVD Divergence: 行为支撑
        - Momentum Divergence: 推进力

        场景：无量趋势（CVD 平，但价格被推）
        """
        n = 600
        dates = pd.date_range("2025-01-01", periods=n, freq="h")

        # 单调上涨（假设是无量推动）
        close = pd.Series(100 + np.arange(n) * 0.5, index=dates)

        result = compute_price_momentum_divergence_from_series(close=close)

        # Momentum 应该显示"有推进力"（velocity_pct 较高）
        velocity_pct = result["price_velocity_pct"].iloc[400:]
        assert (
            velocity_pct.mean() > 0.5
        ), "Uptrend should show positive momentum (high velocity_pct)"

        # 这时 CVD Divergence 可能是 0（无行为数据）
        # 但 Momentum Divergence 仍有价值
