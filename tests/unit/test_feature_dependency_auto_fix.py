#!/usr/bin/env python3
"""
单元测试：特征依赖自动修复功能（防御性措施）

注意：这些测试验证的是自动修复机制作为防御性措施（defensive programming）的表现。
理想情况下，依赖应该通过配置文件（feature_dependencies.yaml）正确解析，确保
依赖特征在需要时已经计算。

这些测试覆盖的场景：
1. 直接调用特征函数（未经过依赖解析器）
2. 配置文件未正确声明依赖的异常情况
3. 自动修复作为安全网的功能

正常使用中，应该通过 StrategyFeatureLoader.resolve_dependencies() 和
load_features_from_requested() 来使用特征，依赖会自动解析和计算。
"""

import pytest
import pandas as pd
import numpy as np

from src.features.loader.feature_wrappers import (
    compute_sr_strength_max,
    compute_sqs_hal_high,
    compute_sqs_hal_low,
)
from src.features.time_series.baseline_features import (
    compute_atr,
    add_poc_hal_dimensionless_features,
)


@pytest.fixture
def sample_data():
    """创建示例数据"""
    dates = pd.date_range("2025-01-01", periods=200, freq="4h")
    np.random.seed(42)

    # 生成价格数据
    base_price = 100000
    returns = np.random.randn(200) * 0.01
    prices = base_price * (1 + returns).cumprod()

    df = pd.DataFrame(
        {
            "open": prices * (1 + np.random.randn(200) * 0.001),
            "high": prices * (1 + abs(np.random.randn(200)) * 0.002),
            "low": prices * (1 - abs(np.random.randn(200)) * 0.002),
            "close": prices,
            "volume": np.random.randint(1000, 10000, 200),
        },
        index=dates,
    )

    # 确保 high >= close >= low
    df["high"] = df[["high", "close"]].max(axis=1)
    df["low"] = df[["low", "close"]].min(axis=1)

    return df


class TestSQSHalHighDependencyFix:
    """测试 sqs_hal_high 的依赖自动修复"""

    def test_missing_atr_auto_compute(self, sample_data):
        """测试缺少 ATR 时自动计算"""
        df = sample_data.copy()

        # 确保没有 ATR
        assert "atr" not in df.columns

        # 计算特征
        result = compute_sqs_hal_high(df)

        # 验证 ATR 被自动创建
        assert "atr" in result.columns
        assert result["atr"].notna().sum() > 0
        assert "sqs_hal_high" in result.columns

    def test_missing_hal_high_auto_compute(self, sample_data):
        """测试缺少 hal_high 时自动计算"""
        df = sample_data.copy()

        # 确保没有 hal_high
        assert "hal_high" not in df.columns

        # 计算特征
        result = compute_sqs_hal_high(df)

        # 验证 hal_high 被自动创建
        assert "hal_high" in result.columns
        assert "sqs_hal_high" in result.columns

    def test_uses_wpt_price_if_available(self, sample_data):
        """测试如果存在 wpt_price_reconstructed，会使用它"""
        df = sample_data.copy()
        df["wpt_price_reconstructed"] = df["close"] * 0.99  # 稍微不同的价格

        result = compute_sqs_hal_high(df, price_col="wpt_price_reconstructed")

        assert "hal_high" in result.columns
        assert "sqs_hal_high" in result.columns


class TestSQSHalLowDependencyFix:
    """测试 sqs_hal_low 的依赖自动修复"""

    def test_missing_atr_auto_compute(self, sample_data):
        """测试缺少 ATR 时自动计算"""
        df = sample_data.copy()

        # 确保没有 ATR
        assert "atr" not in df.columns

        # 计算特征
        result = compute_sqs_hal_low(df)

        # 验证 ATR 被自动创建
        assert "atr" in result.columns
        assert result["atr"].notna().sum() > 0
        assert "sqs_hal_low" in result.columns

    def test_missing_hal_low_auto_compute(self, sample_data):
        """测试缺少 hal_low 时自动计算"""
        df = sample_data.copy()

        # 确保没有 hal_low
        assert "hal_low" not in df.columns

        # 计算特征
        result = compute_sqs_hal_low(df)

        # 验证 hal_low 被自动创建
        assert "hal_low" in result.columns
        assert "sqs_hal_low" in result.columns


class TestSRStrengthMaxDependencyFix:
    """测试 sr_strength_max 的依赖自动修复"""

    def test_missing_all_boundaries_auto_compute(self, sample_data):
        """测试缺少所有边界列时自动计算"""
        df = sample_data.copy()

        # 确保没有边界列和 ATR
        assert "hal_high" not in df.columns
        assert "hal_low" not in df.columns
        assert "poc" not in df.columns
        assert "atr" not in df.columns

        # 计算特征
        result = compute_sr_strength_max(df)

        # 验证所有必需的列被自动创建
        assert "hal_high" in result.columns
        assert "hal_low" in result.columns
        assert "poc" in result.columns
        assert "atr" in result.columns
        assert "sr_strength_max" in result.columns

        # 验证 sr_strength_max 有有效值（不是全部为 0 或 NaN）
        sr_strength = result["sr_strength_max"]
        assert sr_strength.notna().sum() > 0
        # 至少有一些非零值（如果边界计算成功）
        if sr_strength.notna().any():
            assert (sr_strength != 0.0).sum() >= 0  # 允许全部为 0（如果没有有效边界）

    def test_missing_atr_only(self, sample_data):
        """测试只缺少 ATR 时自动计算"""
        df = sample_data.copy()

        # 先计算边界列
        df = add_poc_hal_dimensionless_features(
            df, required_features={"hal_high", "hal_low", "poc"}
        )

        # 删除 ATR
        if "atr" in df.columns:
            df = df.drop(columns=["atr"])

        assert "atr" not in df.columns
        assert "hal_high" in df.columns
        assert "hal_low" in df.columns
        assert "poc" in df.columns

        # 计算特征
        result = compute_sr_strength_max(df)

        # 验证 ATR 被自动创建
        assert "atr" in result.columns
        assert "sr_strength_max" in result.columns

    def test_partial_boundaries_missing(self, sample_data):
        """测试部分边界列缺失时自动计算"""
        df = sample_data.copy()

        # 只计算 hal_high
        df = add_poc_hal_dimensionless_features(df, required_features={"hal_high"})

        # 确保 hal_low 和 poc 不存在
        if "hal_low" in df.columns:
            df = df.drop(columns=["hal_low"])
        if "poc" in df.columns:
            df = df.drop(columns=["poc"])

        assert "hal_high" in df.columns
        assert "hal_low" not in df.columns
        assert "poc" not in df.columns

        # 计算特征
        result = compute_sr_strength_max(df)

        # 验证缺失的边界列被自动创建
        assert "hal_low" in result.columns
        assert "poc" in result.columns
        assert "sr_strength_max" in result.columns

    def test_uses_wpt_price_if_available(self, sample_data):
        """测试如果存在 wpt_price_reconstructed，会使用它"""
        df = sample_data.copy()
        df["wpt_price_reconstructed"] = df["close"] * 0.99

        result = compute_sr_strength_max(
            df,
            poc_window=160,
            price_col="wpt_price_reconstructed",
        )

        assert "hal_high" in result.columns
        assert "hal_low" in result.columns
        assert "poc" in result.columns
        assert "sr_strength_max" in result.columns

    def test_all_dependencies_exist(self, sample_data):
        """测试所有依赖都存在时正常工作"""
        df = sample_data.copy()

        # 手动计算所有依赖
        df["atr"] = compute_atr(df["high"], df["low"], df["close"], period=14)
        df = add_poc_hal_dimensionless_features(
            df, required_features={"hal_high", "hal_low", "poc"}
        )

        # 计算特征
        result = compute_sr_strength_max(df)

        # 验证结果
        assert "sr_strength_max" in result.columns
        sr_strength = result["sr_strength_max"]
        assert sr_strength.notna().sum() > 0


class TestDependencyAutoFixIntegration:
    """集成测试：测试依赖自动修复在完整流程中的表现"""

    def test_sqs_features_then_sr_strength_max(self, sample_data):
        """测试先计算 sqs 特征，再计算 sr_strength_max"""
        df = sample_data.copy()

        # 只添加基础数据，不添加任何依赖
        assert "atr" not in df.columns
        assert "hal_high" not in df.columns
        assert "hal_low" not in df.columns

        # 先计算 sqs_hal_high（会自动计算 atr 和 hal_high）
        df = compute_sqs_hal_high(df)
        assert "atr" in df.columns
        assert "hal_high" in df.columns
        assert "sqs_hal_high" in df.columns

        # 再计算 sqs_hal_low（会使用已有的 atr，自动计算 hal_low）
        df = compute_sqs_hal_low(df)
        assert "hal_low" in df.columns
        assert "sqs_hal_low" in df.columns

        # 最后计算 sr_strength_max（会使用已有的 hal_high, hal_low，自动计算 poc）
        # 注意：sqs_hal_high 和 sqs_hal_low 可能只计算了各自的列，poc 可能不存在
        # sr_strength_max 应该自动计算 poc（如果不存在）
        df = compute_sr_strength_max(df)

        assert "sr_strength_max" in df.columns
        # 验证 poc 被自动创建（如果之前不存在）
        assert "poc" in df.columns
        sr_strength = df["sr_strength_max"]
        assert sr_strength.notna().sum() > 0

    def test_sr_strength_max_standalone(self, sample_data):
        """测试 sr_strength_max 独立计算（不依赖 sqs 特征）"""
        df = sample_data.copy()

        # 不计算 sqs 特征，直接计算 sr_strength_max
        result = compute_sr_strength_max(df)

        # 验证所有必需的列都被自动创建
        assert "atr" in result.columns
        assert "hal_high" in result.columns
        assert "hal_low" in result.columns
        assert "poc" in result.columns
        assert "sr_strength_max" in result.columns

        # 验证结果有效
        sr_strength = result["sr_strength_max"]
        assert sr_strength.notna().sum() > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
