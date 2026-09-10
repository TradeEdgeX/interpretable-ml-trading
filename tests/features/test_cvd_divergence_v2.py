#!/usr/bin/env python3
"""
CVD Divergence V2 特征测试

覆盖三个维度：
1. 功能正确性测试
2. 流式计算一致性测试
3. 未来函数测试（无 look-ahead bias）
"""

import pytest
import numpy as np
import pandas as pd

from src.features.time_series.utils_interaction_features import (
    compute_cvd_divergence_v2_from_series,
)


class TestCVDDivergenceV2Functionality:
    """功能正确性测试"""

    def test_basic_output_columns(self):
        """测试输出列是否完整"""
        dates = pd.date_range("2024-01-01", periods=200, freq="1min")
        close = pd.Series(np.random.randn(200).cumsum() + 100, index=dates)
        cvd = pd.Series(np.random.randn(200).cumsum() * 1000, index=dates)

        result = compute_cvd_divergence_v2_from_series(
            close=close, cvd=cvd, position_window=50
        )

        expected_cols = [
            "cvd_divergence_score",
            "cvd_divergence_score_pct",
            "price_position",
            "trend_div_alignment",
            "trend_div_tension",
            "div_location_pressure",
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_divergence_score_range(self):
        """测试背离得分范围 [-1, 1]"""
        dates = pd.date_range("2024-01-01", periods=200, freq="1min")
        close = pd.Series(np.random.randn(200).cumsum() + 100, index=dates)
        cvd = pd.Series(np.random.randn(200).cumsum() * 1000, index=dates)

        result = compute_cvd_divergence_v2_from_series(
            close=close, cvd=cvd, position_window=50
        )

        score = result["cvd_divergence_score"].dropna()
        assert score.min() >= -1.0, "divergence_score 低于 -1"
        assert score.max() <= 1.0, "divergence_score 高于 1"

    def test_bullish_divergence_semantic(self):
        """测试看涨背离语义：价格低但 CVD 高 → 正值"""
        dates = pd.date_range("2024-01-01", periods=100, freq="1min")
        # 价格持续下跌
        close = pd.Series(100 - np.arange(100) * 0.1, index=dates)
        # CVD 持续上升
        cvd = pd.Series(np.arange(100) * 100, index=dates)

        result = compute_cvd_divergence_v2_from_series(
            close=close, cvd=cvd, position_window=50
        )

        # 后半段应该有正的背离得分（看涨背离）
        score = result["cvd_divergence_score"].iloc[-20:].dropna()
        assert score.mean() > 0, "价格下跌+CVD上升应该产生正的背离得分（看涨背离）"

    def test_bearish_divergence_semantic(self):
        """测试看跌背离语义：价格高但 CVD 低 → 负值"""
        dates = pd.date_range("2024-01-01", periods=100, freq="1min")
        # 价格持续上涨
        close = pd.Series(100 + np.arange(100) * 0.1, index=dates)
        # CVD 持续下降
        cvd = pd.Series(-np.arange(100) * 100, index=dates)

        result = compute_cvd_divergence_v2_from_series(
            close=close, cvd=cvd, position_window=50
        )

        # 后半段应该有负的背离得分（看跌背离）
        score = result["cvd_divergence_score"].iloc[-20:].dropna()
        assert score.mean() < 0, "价格上涨+CVD下降应该产生负的背离得分（看跌背离）"

    def test_trend_div_alignment_with_trend(self):
        """测试趋势-背离对齐度"""
        dates = pd.date_range("2024-01-01", periods=100, freq="1min")
        # 价格下跌但 CVD 上升 → 产生背离
        close = pd.Series(100 - np.arange(100) * 0.1, index=dates)
        cvd = pd.Series(np.arange(100) * 100, index=dates)
        # 正趋势强度
        trend_strength = pd.Series(0.8, index=dates)

        result = compute_cvd_divergence_v2_from_series(
            close=close, cvd=cvd, trend_strength=trend_strength, position_window=50
        )

        alignment = result["trend_div_alignment"].iloc[-20:].dropna()
        # 有背离 + 有趋势强度时，alignment 应该非零
        assert not np.allclose(alignment, 0), "有背离和趋势强度时，alignment 应该非零"

    def test_trend_div_tension_sqrt_effect(self):
        """测试 tension 的 sqrt 非线性效果"""
        dates = pd.date_range("2024-01-01", periods=100, freq="1min")
        close = pd.Series(100 + np.arange(100) * 0.1, index=dates)
        cvd = pd.Series(np.arange(100) * 100, index=dates)
        trend_strength = pd.Series(0.5, index=dates)

        result = compute_cvd_divergence_v2_from_series(
            close=close, cvd=cvd, trend_strength=trend_strength, position_window=50
        )

        tension = result["trend_div_tension"].dropna()
        assert tension.max() <= 1.0, "tension 应该 <= 1"
        assert tension.min() >= 0.0, "tension 应该 >= 0"

    def test_price_position_nan_fallback(self):
        """测试 price_position NaN 回填为 0.5"""
        dates = pd.date_range("2024-01-01", periods=20, freq="1min")
        close = pd.Series([100.0] * 20, index=dates)
        cvd = pd.Series([0.0] * 20, index=dates)

        result = compute_cvd_divergence_v2_from_series(
            close=close, cvd=cvd, position_window=50  # 窗口大于数据量
        )

        # 窗口不足时，price_position 应该回填为 0.5
        position = result["price_position"]
        assert not position.isna().any(), "price_position 不应该有 NaN"

    def test_percentile_rank_handles_repeated_values(self):
        """测试百分位排名能正确处理重复值（plateau 场景）"""
        dates = pd.date_range("2024-01-01", periods=100, freq="1min")
        # CVD plateau（大量重复值）
        cvd = pd.Series([1000.0] * 50 + [1000.0 + i for i in range(50)], index=dates)
        close = pd.Series(100 + np.arange(100) * 0.01, index=dates)

        result = compute_cvd_divergence_v2_from_series(
            close=close, cvd=cvd, position_window=50
        )

        # 即使有重复值，得分也应该在合理范围内
        score = result["cvd_divergence_score"].dropna()
        assert score.min() >= -1.0 and score.max() <= 1.0


class TestCVDDivergenceV2NoFutureLeak:
    """未来函数测试（无 look-ahead bias）"""

    def test_no_future_leak_divergence_score(self):
        """测试：修改未来数据不应影响历史背离得分"""
        dates = pd.date_range("2024-01-01", periods=200, freq="1min")
        np.random.seed(42)
        close = pd.Series(np.random.randn(200).cumsum() + 100, index=dates)
        cvd = pd.Series(np.random.randn(200).cumsum() * 1000, index=dates)

        # 计算第一次
        result1 = compute_cvd_divergence_v2_from_series(
            close=close, cvd=cvd, position_window=50, percentile_window=100
        )
        score1 = result1["cvd_divergence_score"].copy()

        # 修改未来数据（从 t=150 开始）
        close_future = close.copy()
        cvd_future = cvd.copy()
        close_future.iloc[150:] = 999.0
        cvd_future.iloc[150:] = -999999.0

        # 重新计算
        result2 = compute_cvd_divergence_v2_from_series(
            close=close_future,
            cvd=cvd_future,
            position_window=50,
            percentile_window=100,
        )
        score2 = result2["cvd_divergence_score"].copy()

        # 验证历史数据不受影响（t < 150 - window）
        check_idx = dates[:100]
        score1_check = score1.loc[check_idx].dropna()
        score2_check = score2.loc[check_idx].dropna()

        if len(score1_check) > 0 and len(score2_check) > 0:
            common_idx = score1_check.index.intersection(score2_check.index)
            if len(common_idx) > 0:
                diff = (
                    score1_check.loc[common_idx] - score2_check.loc[common_idx]
                ).abs()
                max_diff = diff.max()
                assert (
                    max_diff < 1e-6
                ), f"未来数据变化影响了历史背离得分，最大差异: {max_diff}"

    def test_no_future_leak_alignment(self):
        """测试：修改未来数据不应影响历史 alignment"""
        dates = pd.date_range("2024-01-01", periods=200, freq="1min")
        np.random.seed(42)
        close = pd.Series(np.random.randn(200).cumsum() + 100, index=dates)
        cvd = pd.Series(np.random.randn(200).cumsum() * 1000, index=dates)
        trend = pd.Series(np.random.uniform(-1, 1, 200), index=dates)

        result1 = compute_cvd_divergence_v2_from_series(
            close=close, cvd=cvd, trend_strength=trend, position_window=50
        )
        align1 = result1["trend_div_alignment"].copy()

        # 修改未来数据
        close_future = close.copy()
        close_future.iloc[150:] = 999.0
        trend_future = trend.copy()
        trend_future.iloc[150:] = -1.0

        result2 = compute_cvd_divergence_v2_from_series(
            close=close_future, cvd=cvd, trend_strength=trend_future, position_window=50
        )
        align2 = result2["trend_div_alignment"].copy()

        check_idx = dates[:100]
        align1_check = align1.loc[check_idx].dropna()
        align2_check = align2.loc[check_idx].dropna()

        if len(align1_check) > 0 and len(align2_check) > 0:
            common_idx = align1_check.index.intersection(align2_check.index)
            if len(common_idx) > 0:
                diff = (
                    align1_check.loc[common_idx] - align2_check.loc[common_idx]
                ).abs()
                max_diff = diff.max()
                assert (
                    max_diff < 1e-6
                ), f"未来数据变化影响了历史 alignment，最大差异: {max_diff}"

    def test_no_future_leak_all_features(self):
        """测试：所有输出特征都不应有未来泄露"""
        dates = pd.date_range("2024-01-01", periods=200, freq="1min")
        np.random.seed(123)
        close = pd.Series(np.random.randn(200).cumsum() + 100, index=dates)
        cvd = pd.Series(np.random.randn(200).cumsum() * 1000, index=dates)
        trend = pd.Series(np.random.uniform(-1, 1, 200), index=dates)

        result1 = compute_cvd_divergence_v2_from_series(
            close=close, cvd=cvd, trend_strength=trend, position_window=50
        )

        # 修改未来数据
        close_mod = close.copy()
        cvd_mod = cvd.copy()
        close_mod.iloc[150:] = 500.0
        cvd_mod.iloc[150:] = 500000.0

        result2 = compute_cvd_divergence_v2_from_series(
            close=close_mod, cvd=cvd_mod, trend_strength=trend, position_window=50
        )

        # 验证所有特征
        for col in result1.columns:
            v1 = result1[col].iloc[:100].dropna()
            v2 = result2[col].iloc[:100].dropna()
            common = v1.index.intersection(v2.index)
            if len(common) > 0:
                diff = (v1.loc[common] - v2.loc[common]).abs().max()
                assert diff < 1e-6, f"特征 {col} 存在未来泄露，最大差异: {diff}"


class TestCVDDivergenceV2StreamingVsBatch:
    """流式计算一致性测试"""

    def test_streaming_vs_batch_divergence_score(self):
        """测试：流式计算与批量计算结果一致"""
        dates = pd.date_range("2024-01-01", periods=300, freq="1min")
        np.random.seed(42)
        close = pd.Series(np.random.randn(300).cumsum() + 100, index=dates)
        cvd = pd.Series(np.random.randn(300).cumsum() * 1000, index=dates)

        # 批量计算
        batch_result = compute_cvd_divergence_v2_from_series(
            close=close, cvd=cvd, position_window=50, percentile_window=100
        )

        # 流式计算（分块）
        chunk_size = 50
        streaming_results = []

        for i in range(0, len(close), chunk_size):
            # 每次取从开始到当前位置的所有数据
            end_idx = min(i + chunk_size, len(close))
            chunk_close = close.iloc[:end_idx]
            chunk_cvd = cvd.iloc[:end_idx]

            chunk_result = compute_cvd_divergence_v2_from_series(
                close=chunk_close,
                cvd=chunk_cvd,
                position_window=50,
                percentile_window=100,
            )
            # 只保留当前块的结果
            chunk_result_filtered = chunk_result.iloc[i:end_idx]
            streaming_results.append(chunk_result_filtered)

        if streaming_results:
            streaming_combined = pd.concat(streaming_results)
        else:
            streaming_combined = pd.DataFrame()

        # 比较结果
        common_idx = batch_result.index.intersection(streaming_combined.index)
        if len(common_idx) > 0:
            batch_score = batch_result.loc[common_idx, "cvd_divergence_score"].dropna()
            stream_score = streaming_combined.loc[
                common_idx, "cvd_divergence_score"
            ].dropna()

            common_valid = batch_score.index.intersection(stream_score.index)
            if len(common_valid) > 0:
                diff = (
                    batch_score.loc[common_valid] - stream_score.loc[common_valid]
                ).abs()
                max_diff = diff.max()
                assert max_diff < 1e-5, f"流式与批量计算不一致，最大差异: {max_diff}"

    def test_streaming_vs_batch_alignment(self):
        """测试：流式计算与批量计算 alignment 一致"""
        dates = pd.date_range("2024-01-01", periods=300, freq="1min")
        np.random.seed(42)
        close = pd.Series(np.random.randn(300).cumsum() + 100, index=dates)
        cvd = pd.Series(np.random.randn(300).cumsum() * 1000, index=dates)
        trend = pd.Series(np.random.uniform(-1, 1, 300), index=dates)

        # 批量计算
        batch_result = compute_cvd_divergence_v2_from_series(
            close=close, cvd=cvd, trend_strength=trend, position_window=50
        )

        # 流式计算
        chunk_size = 50
        streaming_results = []

        for i in range(0, len(close), chunk_size):
            end_idx = min(i + chunk_size, len(close))
            chunk_result = compute_cvd_divergence_v2_from_series(
                close=close.iloc[:end_idx],
                cvd=cvd.iloc[:end_idx],
                trend_strength=trend.iloc[:end_idx],
                position_window=50,
            )
            streaming_results.append(chunk_result.iloc[i:end_idx])

        streaming_combined = pd.concat(streaming_results)

        common_idx = batch_result.index.intersection(streaming_combined.index)
        if len(common_idx) > 0:
            batch_align = batch_result.loc[common_idx, "trend_div_alignment"].dropna()
            stream_align = streaming_combined.loc[
                common_idx, "trend_div_alignment"
            ].dropna()

            common_valid = batch_align.index.intersection(stream_align.index)
            if len(common_valid) > 0:
                diff = (
                    batch_align.loc[common_valid] - stream_align.loc[common_valid]
                ).abs()
                max_diff = diff.max()
                assert (
                    max_diff < 1e-5
                ), f"流式与批量 alignment 不一致，最大差异: {max_diff}"

    def test_streaming_vs_batch_all_features(self):
        """测试：所有特征的流式与批量计算一致"""
        dates = pd.date_range("2024-01-01", periods=300, freq="1min")
        np.random.seed(123)
        close = pd.Series(np.random.randn(300).cumsum() + 100, index=dates)
        cvd = pd.Series(np.random.randn(300).cumsum() * 1000, index=dates)
        trend = pd.Series(np.random.uniform(-1, 1, 300), index=dates)

        # 批量
        batch_result = compute_cvd_divergence_v2_from_series(
            close=close, cvd=cvd, trend_strength=trend, position_window=50
        )

        # 流式
        chunk_size = 60
        streaming_results = []
        for i in range(0, len(close), chunk_size):
            end_idx = min(i + chunk_size, len(close))
            chunk_result = compute_cvd_divergence_v2_from_series(
                close=close.iloc[:end_idx],
                cvd=cvd.iloc[:end_idx],
                trend_strength=trend.iloc[:end_idx],
                position_window=50,
            )
            streaming_results.append(chunk_result.iloc[i:end_idx])

        streaming_combined = pd.concat(streaming_results)

        for col in batch_result.columns:
            batch_col = batch_result[col].dropna()
            stream_col = streaming_combined[col].dropna()
            common = batch_col.index.intersection(stream_col.index)
            if len(common) > 0:
                diff = (batch_col.loc[common] - stream_col.loc[common]).abs().max()
                assert diff < 1e-5, f"特征 {col} 流式与批量不一致，最大差异: {diff}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
