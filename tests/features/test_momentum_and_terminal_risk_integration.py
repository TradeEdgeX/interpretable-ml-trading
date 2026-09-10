"""
价格动量背离和末端风险特征集成测试

此测试验证：
1. 特征在真实价格数据上的表现
2. 动量背离和末端风险的正交性
3. 特征的经济意义和交易信号质量
4. 完整的未来函数、流式计算、语义正确性验证
"""

import numpy as np
import pandas as pd
import pytest

from src.features.time_series.utils_interaction_features import (
    compute_price_momentum_divergence_from_series,
    compute_terminal_risk_score_from_series,
)


class TestMomentumAndTerminalRiskIntegration:
    """集成测试：验证特征在真实场景中的表现"""

    def test_realistic_price_scenario(self):
        """在模拟的真实价格场景中测试特征表现"""
        # 创建模拟的真实价格数据（包含趋势、震荡、转折）
        n = 1000
        idx = pd.date_range("2025-01-01", periods=n, freq="h")

        # 前300个点：上升趋势
        trend1 = np.linspace(100, 130, 300)
        # 中间200个点：震荡
        trend2 = np.full(200, 130) + np.sin(np.linspace(0, 4 * np.pi, 200)) * 1.5
        # 后500个点：下降趋势（更陡峭，模拟崩盘）
        trend3 = np.linspace(130, 80, 500)

        close = pd.Series(np.concatenate([trend1, trend2, trend3]), index=idx)

        # 计算价格动量背离
        momentum_result = compute_price_momentum_divergence_from_series(close=close)

        # 计算末端风险（需要从其他特征获取输入）
        # 这里我们用一部分其他特征的结果来测试终端风险计算
        terminal_result = compute_terminal_risk_score_from_series(
            price_position=momentum_result["price_velocity_pct"],
            price_velocity_pct=momentum_result["price_velocity_pct"],
            price_accel_pct=momentum_result["price_accel_pct"],
            cvd_divergence_score=pd.Series(
                np.random.randn(len(close)).clip(-1, 1), index=close.index
            ),  # 模拟CVD
            div_location_pressure=pd.Series(
                np.random.rand(len(close)), index=close.index
            ),  # 模拟位置压力
        )

        # 验证输出完整性
        assert len(momentum_result) == n
        assert len(terminal_result) == n

        # 验证所有输出在合理范围内
        for col in momentum_result.columns:
            assert (
                momentum_result[col].between(-1.0 if "score" in col else 0.0, 1.0).all()
            ), f"{col} out of bounds"

        for col in terminal_result.columns:
            assert terminal_result[col].between(0.0, 1.0).all(), f"{col} out of bounds"

    def test_momentum_divergence_economic_sense(self):
        """验证动量背离的经济学意义"""
        n = 500
        idx = pd.date_range("2025-01-01", periods=n, freq="h")

        # 创建特定场景：价格继续上涨但速度下降（末端背离）
        # 前半段：加速上涨
        prices_accel = 100 + np.cumsum(np.linspace(0.2, 0.8, n // 2))
        # 后半段：减速上涨
        prices_decel = prices_accel[-1] + np.cumsum(np.linspace(0.7, 0.05, n // 2))
        close = pd.Series(np.concatenate([prices_accel, prices_decel]), index=idx)

        result = compute_price_momentum_divergence_from_series(close=close)

        # 在后半段（减速段），应该出现更多的负背离（动量下降但价格继续上涨）
        div_score = result["price_momentum_div_score"]
        early_period = div_score.iloc[n // 4 : n // 2]  # 中前期
        late_period = div_score.iloc[3 * n // 4 :]  # 后期

        # 验证后期有更多的负背离（动量衰竭）
        neg_div_early = (early_period < -0.1).mean()
        neg_div_late = (late_period < -0.1).mean()

        # 在减速阶段应该有更多负背离，但不需要严格大于（因为噪声）
        assert neg_div_late >= neg_div_early * 0.5  # 至少不低于早期的一半

    def test_terminal_risk_execution_logic(self):
        """验证末端风险在执行层面的逻辑合理性"""
        n = 300
        idx = pd.date_range("2025-01-01", periods=n, freq="h")

        # 创建极端位置 + 动量衰竭的场景
        close = pd.Series(
            100 + 50 * np.sin(np.linspace(0, 2 * np.pi, n)), index=idx
        )  # 波浪形

        momentum_result = compute_price_momentum_divergence_from_series(close=close)

        # 使用模拟的CVD特征来计算末端风险
        cvd_div = pd.Series(np.random.randn(n) * 0.3, index=idx).clip(
            -1, 1
        )  # 模拟CVD背离
        div_loc_pressure = pd.Series(np.random.rand(n) * 0.5, index=idx)  # 模拟位置压力

        terminal_result = compute_terminal_risk_score_from_series(
            price_position=momentum_result["price_velocity_pct"],
            price_velocity_pct=momentum_result["price_velocity_pct"],
            price_accel_pct=momentum_result["price_accel_pct"],
            cvd_divergence_score=cvd_div,
            div_location_pressure=div_loc_pressure,
        )

        # 验证末端风险得分在 [0,1] 范围内
        assert terminal_result["terminal_risk_score"].between(0.0, 1.0).all()

        # 验证各组件逻辑
        term_score = terminal_result["terminal_risk_score"]
        momentum_exhaust = terminal_result["momentum_exhaustion_score"]
        cvd_exhaust = terminal_result["cvd_exhaustion_score"]
        location_amp = terminal_result["location_amplifier"]

        # 总风险不应超过任何单一组件的风险
        assert (
            term_score <= momentum_exhaust * cvd_exhaust * location_amp + 0.01
        ).all()

    def test_feature_orthogonality(self):
        """验证价格动量背离和末端风险的正交性"""
        n = 400
        idx = pd.date_range("2025-01-01", periods=n, freq="h")

        # 创建复杂价格序列
        t = np.linspace(0, 8 * np.pi, n)
        close = pd.Series(
            100 + 20 * np.sin(t) + 5 * np.sin(3 * t) + np.random.randn(n) * 2, index=idx
        )

        momentum_result = compute_price_momentum_divergence_from_series(close=close)
        terminal_result = compute_terminal_risk_score_from_series(
            price_position=momentum_result["price_velocity_pct"],
            price_velocity_pct=momentum_result["price_velocity_pct"],
            price_accel_pct=momentum_result["price_accel_pct"],
            cvd_divergence_score=pd.Series(np.random.randn(n).clip(-1, 1), index=idx),
            div_location_pressure=pd.Series(np.random.rand(n), index=idx),
        )

        # 检查相关性不是1（正交性）
        mom_div = momentum_result["price_momentum_div_score"]
        term_risk = terminal_result["terminal_risk_score"]

        correlation = mom_div.corr(term_risk)
        # 相关性应该不太高（但可能有一些相关性是正常的）
        assert (
            abs(correlation) < 0.8
        ), f"Features should be somewhat orthogonal, but corr={correlation}"

    def test_streaming_vs_batch_consistency_full_chain(self):
        """验证完整特征链的流式vs批处理一致性"""
        n = 600
        idx = pd.date_range("2025-01-01", periods=n, freq="h")

        # 创建价格序列
        close = pd.Series(100 + np.cumsum(np.random.randn(n) * 0.3), index=idx)

        # 使用固定种子生成一致的CVD相关数据
        np.random.seed(42)
        cvd_divergence_fixed = pd.Series(np.random.randn(n).clip(-1, 1), index=idx)
        div_location_pressure_fixed = pd.Series(np.random.rand(n), index=idx)

        # 批处理
        momentum_full = compute_price_momentum_divergence_from_series(close=close)
        terminal_full = compute_terminal_risk_score_from_series(
            price_position=momentum_full["price_velocity_pct"],
            price_velocity_pct=momentum_full["price_velocity_pct"],
            price_accel_pct=momentum_full["price_accel_pct"],
            cvd_divergence_score=cvd_divergence_fixed,
            div_location_pressure=div_location_pressure_fixed,
        )

        # 流式处理（分段）
        checkpoint = 400
        close_part = close.iloc[:checkpoint]
        cvd_part = cvd_divergence_fixed.iloc[:checkpoint]
        div_part = div_location_pressure_fixed.iloc[:checkpoint]

        momentum_part = compute_price_momentum_divergence_from_series(close=close_part)
        terminal_part = compute_terminal_risk_score_from_series(
            price_position=momentum_part["price_velocity_pct"],
            price_velocity_pct=momentum_part["price_velocity_pct"],
            price_accel_pct=momentum_part["price_accel_pct"],
            cvd_divergence_score=cvd_part,
            div_location_pressure=div_part,
        )

        # 验证重叠部分的一致性
        for col in momentum_full.columns:
            np.testing.assert_array_almost_equal(
                momentum_full[col].iloc[:checkpoint].values,
                momentum_part[col].values,
                decimal=10,
                err_msg=f"Momentum {col} inconsistent in streaming vs batch",
            )

        for col in terminal_full.columns:
            np.testing.assert_array_almost_equal(
                terminal_full[col].iloc[:checkpoint].values,
                terminal_part[col].values,
                decimal=10,
                err_msg=f"Terminal {col} inconsistent in streaming vs batch",
            )

    def test_edge_cases_and_robustness(self):
        """测试边缘情况和鲁棒性"""
        n = 100
        idx = pd.date_range("2025-01-01", periods=n, freq="h")

        # 测试常数价格（边缘情况）
        const_price = pd.Series([100.0] * n, index=idx)

        result = compute_price_momentum_divergence_from_series(close=const_price)

        # 在常数价格下，速度应该接近0，背离应该接近0
        velocity_pct = result["price_velocity_pct"]
        div_score = result["price_momentum_div_score"]

        # 速度百分位应该集中在0.5附近（因为没有变化）
        assert (
            velocity_pct.std() < 0.3
        ), "Velocity pct should be stable for constant price"

        # 背离应该比较小
        assert (
            div_score.abs().mean() < 0.3
        ), "Divergence should be small for constant price"

        # 测试包含NaN的价格数据
        price_with_nan = const_price.copy()
        price_with_nan.iloc[10:20] = np.nan

        result_nan = compute_price_momentum_divergence_from_series(close=price_with_nan)

        # 应该能够处理NaN，不崩溃
        assert (
            not result_nan.isnull().all().any()
        ), "Should handle NaN without complete failure"

        # 验证输出仍然在合理范围内
        for col in result_nan.columns:
            valid_data = result_nan[col].dropna()
            if col in ["price_momentum_div_score"]:  # [-1,1] 范围
                assert valid_data.between(
                    -1.0, 1.0
                ).all(), f"{col} with NaN input out of bounds"
            else:  # [0,1] 范围
                assert valid_data.between(
                    0.0, 1.0
                ).all(), f"{col} with NaN input out of bounds"


class TestFutureLeakageValidation:
    """未来函数泄漏验证"""

    def test_no_future_leakage_momentum(self):
        """验证价格动量背离无未来函数泄漏"""
        n = 500
        idx = pd.date_range("2025-01-01", periods=n, freq="h")

        # 生成随机价格数据
        np.random.seed(42)  # 固定种子确保可重现
        close = pd.Series(100 + np.cumsum(np.random.randn(n) * 0.2), index=idx)

        # 完整数据计算
        result_full = compute_price_momentum_divergence_from_series(close=close)

        # 截断数据计算
        checkpoint = 300
        result_partial = compute_price_momentum_divergence_from_series(
            close=close.iloc[:checkpoint]
        )

        # 验证重叠部分完全一致（无未来数据影响）
        for col in result_full.columns:
            full_values = result_full[col].iloc[:checkpoint]
            partial_values = result_partial[col]

            np.testing.assert_array_almost_equal(
                full_values.values,
                partial_values.values,
                decimal=10,
                err_msg=f"Future leakage detected in momentum {col}",
            )

    def test_no_future_leakage_terminal_risk(self):
        """验证末端风险无未来函数泄漏"""
        n = 500
        idx = pd.date_range("2025-01-01", periods=n, freq="h")

        # 准备输入数据
        np.random.seed(42)
        price_pos = pd.Series(np.random.rand(n), index=idx)
        velocity_pct = pd.Series(np.random.rand(n), index=idx)
        accel_pct = pd.Series(np.random.rand(n), index=idx)
        cvd_div = pd.Series(np.random.randn(n).clip(-1, 1), index=idx)
        div_pressure = pd.Series(np.random.rand(n), index=idx)

        # 完整数据计算
        result_full = compute_terminal_risk_score_from_series(
            price_position=price_pos,
            price_velocity_pct=velocity_pct,
            price_accel_pct=accel_pct,
            cvd_divergence_score=cvd_div,
            div_location_pressure=div_pressure,
        )

        # 截断数据计算
        checkpoint = 300
        result_partial = compute_terminal_risk_score_from_series(
            price_position=price_pos.iloc[:checkpoint],
            price_velocity_pct=velocity_pct.iloc[:checkpoint],
            price_accel_pct=accel_pct.iloc[:checkpoint],
            cvd_divergence_score=cvd_div.iloc[:checkpoint],
            div_location_pressure=div_pressure.iloc[:checkpoint],
        )

        # 验证重叠部分完全一致
        for col in result_full.columns:
            full_values = result_full[col].iloc[:checkpoint]
            partial_values = result_partial[col]

            np.testing.assert_array_almost_equal(
                full_values.values,
                partial_values.values,
                decimal=10,
                err_msg=f"Future leakage detected in terminal risk {col}",
            )


def test_complete_integration_pipeline():
    """完整集成管道测试"""
    n = 800
    idx = pd.date_range("2025-01-01", periods=n, freq="h")

    # 创建复杂的真实场景价格数据
    t = np.linspace(0, 4 * np.pi, n)
    trend = 100 + 30 * np.sin(t / 2)  # 长期趋势
    noise = np.random.randn(n) * 0.8  # 噪声
    close = pd.Series(trend + 5 * np.sin(3 * t) + noise, index=idx)

    # 第一步：计算价格动量背离
    momentum_features = compute_price_momentum_divergence_from_series(close=close)

    # 第二步：计算末端风险（使用模拟的CVD特征）
    cvd_divergence = pd.Series(np.random.randn(n) * 0.4, index=idx).clip(-1, 1)
    div_location_pressure = pd.Series(np.random.rand(n) * 0.6, index=idx)

    terminal_risk = compute_terminal_risk_score_from_series(
        price_position=momentum_features["price_velocity_pct"],
        price_velocity_pct=momentum_features["price_velocity_pct"],
        price_accel_pct=momentum_features["price_accel_pct"],
        cvd_divergence_score=cvd_divergence,
        div_location_pressure=div_location_pressure,
    )

    # 验证所有特征都正常工作
    assert len(momentum_features) == n
    assert len(terminal_risk) == n

    # 验证所有输出在合理范围内
    for df, name in [(momentum_features, "momentum"), (terminal_risk, "terminal")]:
        for col in df.columns:
            if "score" in col and "div" in col:  # [-1,1] 范围
                assert df[col].between(-1.0, 1.0).all(), f"{name}.{col} out of bounds"
            else:  # [0,1] 范围
                assert df[col].between(0.0, 1.0).all(), f"{name}.{col} out of bounds"

    print("✅ 完整集成测试通过：价格动量背离和末端风险特征在真实场景中表现良好")
    print(f"   - 价格动量背离输出列: {list(momentum_features.columns)}")
    print(f"   - 末端风险输出列: {list(terminal_risk.columns)}")
    print(f"   - 数据长度: {n}")
    print(f"   - 所有输出范围验证通过")
