"""
验证特征相关性的 pytest 测试

注意：这个测试使用模拟数据，主要用于验证相关性测试方法本身的正确性。
对于实际的特征验证，应该使用各个特征专门的测试文件：
- test_vpin_future_leak_and_multi_asset.py - VPIN 特征验证
- test_wpt_future_leak_and_multi_asset.py - WPT 特征验证
- test_spectrum_features.py - Spectrum 特征验证
- test_complex_features_comprehensive.py - 其他复杂特征验证

这个测试的目的是：
1. 作为相关性验证方法的参考实现
2. 可以用于验证简单的通用特征
3. 各个特征测试可以借鉴这里的验证逻辑
"""

import pytest
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def feature_data_with_labels():
    """
    加载特征数据和标签（使用模拟数据）

    注意：这个 fixture 使用模拟数据，主要用于测试验证方法本身。
    对于实际特征验证，应该使用各个特征专门的测试。
    """
    np.random.seed(42)
    n_samples = 1000
    dates = pd.date_range("2024-01-01", periods=n_samples, freq="4h")

    # 生成价格数据
    price_base = 50000
    returns = np.random.randn(n_samples) * 0.01
    prices = price_base * (1 + returns).cumprod()

    df = pd.DataFrame(
        {
            "open": prices * (1 + np.random.randn(n_samples) * 0.001),
            "high": prices * (1 + np.abs(np.random.randn(n_samples)) * 0.002),
            "low": prices * (1 - np.abs(np.random.randn(n_samples)) * 0.002),
            "close": prices,
            "volume": np.random.uniform(1000, 10000, n_samples),
        },
        index=dates,
    )

    # 生成一些简单的特征（模拟）
    df["rsi"] = 50 + np.random.randn(n_samples) * 10
    df["atr"] = (df["high"] - df["low"]).rolling(14).mean()
    df["macd"] = np.random.randn(n_samples) * 2
    df["volatility"] = df["close"].pct_change().rolling(20).std()

    # 生成未来收益（模拟）
    df["future_return"] = df["close"].pct_change(24).shift(-24)
    df["volatility_normalized_target"] = df["future_return"] / (df["volatility"] + 1e-8)

    # 选择要测试的特征
    feature_cols = ["rsi", "atr", "macd", "volatility"]

    # 分割数据
    train_size = int(len(df) * 0.85)
    train_labels = df.iloc[:train_size].copy()

    return train_labels, feature_cols


class TestFeatureCorrelationMethods:
    """
    特征相关性验证方法的测试类

    这个类主要用于验证相关性测试方法本身的正确性，
    而不是测试所有实际特征。实际特征验证应该在各特征专门的测试文件中进行。
    """

    def test_lag_correlation_method(self, feature_data_with_labels):
        """
        测试滞后相关性验证方法

        这个方法可以作为各个特征测试的参考实现。
        """
        labels, features = feature_data_with_labels

        # 选择前几个特征进行测试
        test_features = features[:2]

        for feat in test_features:
            if feat not in labels.columns:
                continue

            base = labels[[feat, "future_return"]].dropna()
            if len(base) < 100:
                continue

            # 计算 lag 0 和 lag 1 的相关性
            corr_0, _ = spearmanr(base[feat], base["future_return"])

            # lag 1
            base_lag1 = base.copy()
            base_lag1[feat] = base_lag1[feat].shift(1)
            base_lag1 = base_lag1.dropna()
            if len(base_lag1) >= 100:
                corr_1, _ = spearmanr(base_lag1[feat], base_lag1["future_return"])

                # 相关性应该缓慢衰减
                decay = abs(corr_0) - abs(corr_1)
                print(
                    f"   {feat}: Lag 0={corr_0:.4f}, Lag 1={corr_1:.4f}, 衰减={decay:.4f}"
                )

                # 如果相关性急剧下降（>0.05），可能需要检查
                if abs(decay) > 0.05:
                    print(f"   ⚠️  {feat}: 相关性衰减较大，可能需要检查时间对齐")

    def test_shuffled_correlation_method(self, feature_data_with_labels):
        """
        测试随机打乱相关性验证方法

        这个方法可以作为各个特征测试的参考实现。
        """
        labels, features = feature_data_with_labels

        # 选择前几个特征进行测试
        test_features = features[:2]
        n_shuffle = 5  # 减少打乱次数以加快测试

        for feat in test_features:
            if feat not in labels.columns:
                continue

            base = labels[[feat, "future_return"]].dropna()
            if len(base) < 100:
                continue

            # 真实相关性
            corr_real, p_real = spearmanr(base[feat], base["future_return"])

            # 打乱后的相关性
            shuffled_corrs = []
            for i in range(n_shuffle):
                shuffled_returns = (
                    base["future_return"].sample(frac=1, random_state=42 + i).values
                )
                corr_shuffled, _ = spearmanr(base[feat], shuffled_returns)
                shuffled_corrs.append(corr_shuffled)

            mean_corr_shuffled = np.mean(shuffled_corrs)
            ratio = (
                abs(mean_corr_shuffled) / abs(corr_real) if corr_real != 0 else np.nan
            )

            print(
                f"   {feat}: 真实={corr_real:.4f}, 打乱后均值={mean_corr_shuffled:.4f}, 比率={ratio:.2%}"
            )

            # 如果打乱后相关性仍然很高（>0.5），可能存在问题
            if (
                not np.isnan(ratio)
                and ratio > 0.5
                and abs(corr_real) > 0.01
                and p_real < 0.05
            ):
                print(f"   ⚠️  {feat}: 打乱后相关性仍然较高，可能存在虚假相关")

    def test_feature_values_validity(self, feature_data_with_labels):
        """测试特征值有效性（通用验证方法）"""
        labels, features = feature_data_with_labels

        for feat in features:
            if feat not in labels.columns:
                continue

            values = labels[feat].dropna()
            if len(values) == 0:
                continue

            # 检查是否有异常值
            has_inf = np.isinf(values).any()
            assert not has_inf, f"特征 {feat} 包含 Inf 值"

            # 检查是否有过多 NaN
            nan_ratio = labels[feat].isna().sum() / len(labels)
            if nan_ratio > 0.8:
                print(f"   ⚠️  {feat}: NaN 比例较高 ({nan_ratio:.2%})")

            print(f"   ✅ {feat}: 有效值 {len(values)}, NaN 比例 {nan_ratio:.2%}")
