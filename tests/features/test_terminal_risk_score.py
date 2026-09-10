"""
Terminal Risk Score 特征测试

测试覆盖：
1. 功能正确性：语义验证
2. 未来函数检测
3. 流式计算一致性
"""

import numpy as np
import pandas as pd
import pytest

from src.features.time_series.utils_interaction_features import (
    compute_terminal_risk_score_from_series,
)


class TestTerminalRiskScoreFunctionality:
    """功能正确性测试"""

    def test_output_columns(self):
        """验证输出列完整"""
        n = 100
        idx = pd.date_range("2025-01-01", periods=n, freq="h")

        result = compute_terminal_risk_score_from_series(
            price_position=pd.Series([0.5] * n, index=idx),
            price_velocity_pct=pd.Series([0.5] * n, index=idx),
            price_accel_pct=pd.Series([0.5] * n, index=idx),
            cvd_divergence_score=pd.Series([0.0] * n, index=idx),
            div_location_pressure=pd.Series([0.0] * n, index=idx),
        )

        expected = [
            "momentum_exhaustion_score",
            "cvd_exhaustion_score",
            "location_amplifier",
            "terminal_risk_score",
        ]
        assert list(result.columns) == expected

    def test_bounded_outputs(self):
        """验证所有输出在 [0,1] 范围内"""
        n = 200
        idx = pd.date_range("2025-01-01", periods=n, freq="h")

        result = compute_terminal_risk_score_from_series(
            price_position=pd.Series(np.random.rand(n), index=idx),
            price_velocity_pct=pd.Series(np.random.rand(n), index=idx),
            price_accel_pct=pd.Series(np.random.rand(n), index=idx),
            cvd_divergence_score=pd.Series(np.random.randn(n).clip(-1, 1), index=idx),
            div_location_pressure=pd.Series(np.random.rand(n), index=idx),
        )

        for col in result.columns:
            assert result[col].between(0.0, 1.0).all(), f"{col} should be in [0,1]"

    def test_safe_zone_low_risk(self):
        """安全推进段：位置中性、动量充足 → 低风险"""
        n = 100
        idx = pd.date_range("2025-01-01", periods=n, freq="h")

        # 安全场景：中位位置 + 高动量 + 无背离
        result = compute_terminal_risk_score_from_series(
            price_position=pd.Series([0.5] * n, index=idx),  # 中位位置
            price_velocity_pct=pd.Series([0.8] * n, index=idx),  # 高速度
            price_accel_pct=pd.Series([0.7] * n, index=idx),  # 正加速
            cvd_divergence_score=pd.Series([0.0] * n, index=idx),  # 无背离
            div_location_pressure=pd.Series([0.0] * n, index=idx),  # 无位置压力
        )

        # 末端风险应该很低
        assert (
            result["terminal_risk_score"].mean() < 0.1
        ), "Safe zone should have low risk"

    def test_terminal_zone_high_risk(self):
        """末端场景：极端位置 + 动量衰竭 + CVD 背离 → 高风险"""
        n = 100
        idx = pd.date_range("2025-01-01", periods=n, freq="h")

        # 末端场景：价格在高位 + 速度低 + 有背离
        result = compute_terminal_risk_score_from_series(
            price_position=pd.Series([0.95] * n, index=idx),  # 极端高位
            price_velocity_pct=pd.Series([0.1] * n, index=idx),  # 低速度
            price_accel_pct=pd.Series([0.1] * n, index=idx),  # 负加速
            cvd_divergence_score=pd.Series([-0.8] * n, index=idx),  # 强背离
            div_location_pressure=pd.Series([0.8] * n, index=idx),  # 高位置压力
        )

        # 末端风险应该较高
        assert (
            result["terminal_risk_score"].mean() > 0.3
        ), "Terminal zone should have high risk"

    def test_semantic_components(self):
        """验证各组件语义正确性"""
        n = 100
        idx = pd.date_range("2025-01-01", periods=n, freq="h")

        result = compute_terminal_risk_score_from_series(
            price_position=pd.Series([0.9] * n, index=idx),  # 高位
            price_velocity_pct=pd.Series([0.2] * n, index=idx),  # 低速度
            price_accel_pct=pd.Series([0.2] * n, index=idx),  # 低加速度
            cvd_divergence_score=pd.Series([-0.5] * n, index=idx),  # 背离
            div_location_pressure=pd.Series([0.6] * n, index=idx),  # 位置压力
        )

        # momentum_exhaustion_score = (1-0.2) * (1-0.2) = 0.64
        assert abs(result["momentum_exhaustion_score"].iloc[0] - 0.64) < 0.01

        # cvd_exhaustion_score = 0.5 * 0.6 = 0.3
        assert abs(result["cvd_exhaustion_score"].iloc[0] - 0.3) < 0.01

        # location_amplifier = |0.9 - 0.5| * 2 = 0.8
        assert abs(result["location_amplifier"].iloc[0] - 0.8) < 0.01

        # terminal_risk_score = 0.64 * 0.3 * 0.8 ≈ 0.1536
        assert abs(result["terminal_risk_score"].iloc[0] - 0.1536) < 0.01

    def test_location_amplifier_extremes(self):
        """位置放大器在极端和中位的行为"""
        n = 3
        idx = pd.date_range("2025-01-01", periods=n, freq="h")

        result = compute_terminal_risk_score_from_series(
            price_position=pd.Series([0.0, 0.5, 1.0], index=idx),  # 低/中/高
            price_velocity_pct=pd.Series([0.5] * n, index=idx),
            price_accel_pct=pd.Series([0.5] * n, index=idx),
            cvd_divergence_score=pd.Series([0.5] * n, index=idx),
            div_location_pressure=pd.Series([0.5] * n, index=idx),
        )

        loc_amp = result["location_amplifier"]
        assert loc_amp.iloc[0] == 1.0, "位置 0 应该是极端"
        assert loc_amp.iloc[1] == 0.0, "位置 0.5 应该是中性"
        assert loc_amp.iloc[2] == 1.0, "位置 1 应该是极端"


class TestTerminalRiskScoreNoFutureLeak:
    """未来函数检测"""

    def test_no_future_leak(self):
        """截断后结果不变"""
        n = 200
        idx = pd.date_range("2025-01-01", periods=n, freq="h")

        # 生成随机数据
        pp = pd.Series(np.random.rand(n), index=idx)
        vp = pd.Series(np.random.rand(n), index=idx)
        ap = pd.Series(np.random.rand(n), index=idx)
        cvd = pd.Series(np.random.randn(n).clip(-1, 1), index=idx)
        dlp = pd.Series(np.random.rand(n), index=idx)

        # 全量计算
        result_full = compute_terminal_risk_score_from_series(
            price_position=pp,
            price_velocity_pct=vp,
            price_accel_pct=ap,
            cvd_divergence_score=cvd,
            div_location_pressure=dlp,
        )

        # 截断计算
        result_partial = compute_terminal_risk_score_from_series(
            price_position=pp.iloc[:150],
            price_velocity_pct=vp.iloc[:150],
            price_accel_pct=ap.iloc[:150],
            cvd_divergence_score=cvd.iloc[:150],
            div_location_pressure=dlp.iloc[:150],
        )

        for col in result_full.columns:
            np.testing.assert_array_almost_equal(
                result_full[col].iloc[:150].values,
                result_partial[col].values,
                decimal=10,
                err_msg=f"Future leak in {col}",
            )


class TestTerminalRiskScoreStreaming:
    """流式一致性"""

    def test_streaming_consistency(self):
        """流式 vs 批量一致"""
        n = 300
        idx = pd.date_range("2025-01-01", periods=n, freq="h")

        pp = pd.Series(np.random.rand(n), index=idx)
        vp = pd.Series(np.random.rand(n), index=idx)
        ap = pd.Series(np.random.rand(n), index=idx)
        cvd = pd.Series(np.random.randn(n).clip(-1, 1), index=idx)
        dlp = pd.Series(np.random.rand(n), index=idx)

        # 批量
        result_batch = compute_terminal_risk_score_from_series(
            price_position=pp,
            price_velocity_pct=vp,
            price_accel_pct=ap,
            cvd_divergence_score=cvd,
            div_location_pressure=dlp,
        )

        # 流式
        for cp in [100, 200, 300]:
            result_stream = compute_terminal_risk_score_from_series(
                price_position=pp.iloc[:cp],
                price_velocity_pct=vp.iloc[:cp],
                price_accel_pct=ap.iloc[:cp],
                cvd_divergence_score=cvd.iloc[:cp],
                div_location_pressure=dlp.iloc[:cp],
            )

            for col in result_batch.columns:
                np.testing.assert_array_almost_equal(
                    result_batch[col].iloc[:cp].values,
                    result_stream[col].values,
                    decimal=10,
                )
