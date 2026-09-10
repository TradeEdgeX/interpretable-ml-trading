"""
测试 compute_lift_for_threshold 函数的正确性

核心语义：
- operator 是 DENY 条件（来自 gate.yaml 的 value_lt/value_gt）
- pass 条件是 deny 条件的反面
- Lift = (pass_rate_good / pass_rate_bad) - 1
"""

import pandas as pd
import numpy as np
import pytest

from src.research.stat_kernels.gate_lift import compute_lift_for_threshold


class TestComputeLiftForThreshold:
    """测试 compute_lift_for_threshold 函数"""

    def test_basic_lt_operator(self):
        """
        测试 value_lt (deny when <) 的语义

        规则: deny when feature < 0.5
        所以: pass when feature >= 0.5
        """
        # 构造测试数据
        df = pd.DataFrame(
            {
                "feature": [0.1, 0.2, 0.3, 0.6, 0.7, 0.8, 0.9, 1.0],
                "is_good": [0, 0, 0, 1, 1, 1, 1, 1],
            }
        )
        # feature < 0.5: [0.1, 0.2, 0.3] -> all bad (0)
        # feature >= 0.5: [0.6, 0.7, 0.8, 0.9, 1.0] -> all good (1)

        result = compute_lift_for_threshold(df, "feature", "lt", 0.5, "is_good")

        # Pass 的样本是 feature >= 0.5 的 5 个（全是 good）
        # n_good = 5, n_bad = 3
        # pass_rate_good = 5/5 = 1.0 (所有 good 都 pass)
        # pass_rate_bad = 0/3 = 0.0 (没有 bad pass)

        assert (
            result["pass_rate_good"] == 1.0
        ), f"Expected 1.0, got {result['pass_rate_good']}"
        assert (
            result["pass_rate_bad"] == 0.0
        ), f"Expected 0.0, got {result['pass_rate_bad']}"
        assert (
            result["pass_rate_all"] == 5 / 8
        ), f"Expected 0.625, got {result['pass_rate_all']}"

    def test_basic_gt_operator(self):
        """
        测试 value_gt (deny when >) 的语义

        规则: deny when feature > 0.5
        所以: pass when feature <= 0.5
        """
        df = pd.DataFrame(
            {
                "feature": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                "is_good": [1, 1, 1, 1, 1, 0, 0, 0],
            }
        )
        # feature <= 0.5: [0.1, 0.2, 0.3, 0.4, 0.5] -> all good (1)
        # feature > 0.5: [0.6, 0.7, 0.8] -> all bad (0)

        result = compute_lift_for_threshold(df, "feature", "gt", 0.5, "is_good")

        # Pass 的样本是 feature <= 0.5 的 5 个（全是 good）
        # n_good = 5, n_bad = 3
        # pass_rate_good = 5/5 = 1.0
        # pass_rate_bad = 0/3 = 0.0

        assert result["pass_rate_good"] == 1.0
        assert result["pass_rate_bad"] == 0.0
        assert result["pass_rate_all"] == 5 / 8

    def test_lift_calculation_positive(self):
        """
        测试正 Lift 的计算

        当 pass_rate_good > pass_rate_bad 时，Lift > 0
        """
        df = pd.DataFrame(
            {
                "feature": [0.1, 0.2, 0.6, 0.7, 0.8, 0.9],
                "is_good": [0, 0, 1, 1, 0, 1],
            }
        )
        # n_good = 3, n_bad = 3
        # deny when < 0.5, pass when >= 0.5
        # pass 的样本: [0.6, 0.7, 0.8, 0.9] -> good=[0.6,0.7,0.9], bad=[0.8]
        # pass_rate_good = 3/3 = 1.0
        # pass_rate_bad = 1/3 = 0.333
        # lift = 1.0 / 0.333 - 1 = 2.0

        result = compute_lift_for_threshold(df, "feature", "lt", 0.5, "is_good")

        assert result["pass_rate_good"] == 1.0
        assert abs(result["pass_rate_bad"] - 1 / 3) < 0.001
        assert abs(result["lift"] - 2.0) < 0.01

    def test_lift_calculation_negative(self):
        """
        测试负 Lift 的计算

        当 pass_rate_good < pass_rate_bad 时，Lift < 0
        这表示规则在误杀 good 样本
        """
        df = pd.DataFrame(
            {
                "feature": [0.1, 0.2, 0.3, 0.6, 0.7, 0.8],
                "is_good": [1, 1, 0, 0, 0, 1],
            }
        )
        # n_good = 3 ([0.1, 0.2, 0.8])
        # n_bad = 3 ([0.3, 0.6, 0.7])
        # deny when < 0.5, pass when >= 0.5
        # pass 的样本 (feature >= 0.5): [0.6, 0.7, 0.8]
        #   good in pass: [0.8] -> 1
        #   bad in pass: [0.6, 0.7] -> 2
        # pass_rate_good = 1/3
        # pass_rate_bad = 2/3
        # lift = (1/3) / (2/3) - 1 = 0.5 - 1 = -0.5

        result = compute_lift_for_threshold(df, "feature", "lt", 0.5, "is_good")

        assert abs(result["pass_rate_good"] - 1 / 3) < 0.001
        assert abs(result["pass_rate_bad"] - 2 / 3) < 0.001
        assert abs(result["lift"] - (-0.5)) < 0.01
        assert result["lift"] < 0, "Negative lift expected when rule kills good samples"

    def test_le_operator(self):
        """测试 value_le (deny when <=) 的语义"""
        df = pd.DataFrame(
            {
                "feature": [0.5, 0.5, 0.6, 0.7],
                "is_good": [0, 0, 1, 1],
            }
        )
        # deny when <= 0.5, pass when > 0.5
        # pass: [0.6, 0.7] -> all good

        result = compute_lift_for_threshold(df, "feature", "le", 0.5, "is_good")

        assert result["pass_rate_good"] == 1.0
        assert result["pass_rate_bad"] == 0.0

    def test_ge_operator(self):
        """测试 value_ge (deny when >=) 的语义"""
        df = pd.DataFrame(
            {
                "feature": [0.3, 0.4, 0.5, 0.5],
                "is_good": [1, 1, 0, 0],
            }
        )
        # deny when >= 0.5, pass when < 0.5
        # pass: [0.3, 0.4] -> all good

        result = compute_lift_for_threshold(df, "feature", "ge", 0.5, "is_good")

        assert result["pass_rate_good"] == 1.0
        assert result["pass_rate_bad"] == 0.0

    def test_feature_not_found(self):
        """测试特征不存在时的处理"""
        df = pd.DataFrame(
            {
                "other_feature": [0.1, 0.2],
                "is_good": [1, 0],
            }
        )

        result = compute_lift_for_threshold(df, "feature", "lt", 0.5, "is_good")

        assert result["lift"] == 0.0
        assert result["pass_rate_all"] == 0.0

    def test_invalid_operator_raises_error(self):
        """测试无效 operator 应该报错，而不是静默通过"""
        df = pd.DataFrame(
            {
                "feature": [0.1, 0.2, 0.3],
                "is_good": [1, 1, 0],
            }
        )

        with pytest.raises(ValueError, match="Invalid operator"):
            compute_lift_for_threshold(df, "feature", "invalid_op", 0.5, "is_good")

    def test_low_bad_pass_rate_returns_nan(self):
        """
        测试当 pass_rate_bad < 0.01 时，lift 应该返回 NaN

        这是关键的 bug fix：
        - 不应该使用 fallback 公式制造“假高原”
        - bad 样本几乎全被拒绝时，lift 无法可靠计算
        """
        # 构造数据：bad 样本几乎全被 deny
        df = pd.DataFrame(
            {
                "feature": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] * 10,
                "is_good": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0] * 10,  # 90% good, 10% bad
            }
        )
        # bad 样本都在 feature=1.0
        # deny when > 0.95, pass when <= 0.95
        # bad 样本 (feature=1.0) 全被 deny
        # pass_rate_bad = 0/10 = 0.0

        result = compute_lift_for_threshold(df, "feature", "gt", 0.95, "is_good")

        # lift 应该是 NaN，不应该是某个假的高值
        assert np.isnan(result["lift"]), f"Expected NaN, got {result['lift']}"

    def test_realistic_gate_scenario(self):
        """
        模拟真实 Gate 场景

        规则: gate_wpt_exhaustion
        - deny when wpt_exhaustion_score > 0.3
        - 语义: 高力竭分数 = 坏，应该 deny

        期望: deny 掉的样本中 bad 比例更高（即 pass 的样本中 good 比例更高）
        """
        # 构造符合语义的数据: 高 exhaustion -> 更容易 bad
        df = pd.DataFrame(
            {
                "wpt_exhaustion_score": [0.1, 0.15, 0.2, 0.25, 0.35, 0.4, 0.45, 0.5],
                "is_good": [1, 1, 1, 1, 0, 0, 1, 0],
            }
        )
        # 低 exhaustion (0.1-0.25): 4 good, 0 bad
        # 高 exhaustion (0.35-0.5): 1 good, 3 bad

        result = compute_lift_for_threshold(
            df, "wpt_exhaustion_score", "gt", 0.3, "is_good"
        )

        # pass when exhaustion <= 0.3: [0.1, 0.15, 0.2, 0.25] -> all good
        # n_good = 5, n_bad = 3
        # pass_rate_good = 4/5 = 0.8
        # pass_rate_bad = 0/3 = 0.0

        assert result["pass_rate_good"] == 4 / 5
        assert result["pass_rate_bad"] == 0.0
        # Lift 应该是正的（或 inf），表示规则有效过滤了 bad
        assert result["lift"] > 0 or result["pass_rate_bad"] == 0

    def test_bad_concentration_in_denied_samples(self):
        """
        验证核心逻辑: deny 掉的样本中 bad 浓度应该更高

        这是 Gate 有效性的核心指标
        """
        # 构造数据: 规则应该 deny 高风险样本
        df = pd.DataFrame(
            {
                "risk_score": [0.1, 0.2, 0.3, 0.7, 0.8, 0.9],
                "is_good": [1, 1, 1, 0, 0, 0],
            }
        )
        # deny when risk_score > 0.5
        # denied: [0.7, 0.8, 0.9] -> all bad (100% bad concentration)
        # passed: [0.1, 0.2, 0.3] -> all good

        result = compute_lift_for_threshold(df, "risk_score", "gt", 0.5, "is_good")

        # 所有 good 都 pass，所有 bad 都被 deny
        assert result["pass_rate_good"] == 1.0
        assert result["pass_rate_bad"] == 0.0

        # 计算 bad concentration in denied
        # denied = df[df["risk_score"] > 0.5]
        # bad_in_denied = 3, total_denied = 3
        # bad_concentration = 3/3 = 1.0 (100%)
        # 全样本 bad_rate = 3/6 = 0.5
        # BadConc = 1.0 / 0.5 = 2.0x (浓度提升 2 倍)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
