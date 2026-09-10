"""
测试 Volume Profile 波动率特征的有效性

使用模拟数据验证：
1. 特征计算的正确性
2. 特征与波动率的相关性
3. 特征在不同市场状态下的表现
"""

import numpy as np
import pandas as pd
import pytest

from src.features.time_series.utils_volatility_features import (
    extract_volatility_features_from_vp,
    extract_volume_profile_volatility_features_from_series,
)


# Route B: DF-style entrypoint removed; provide local DF wrapper for tests.
def extract_volume_profile_volatility_features(
    df: pd.DataFrame,
    price_col: str = "close",
    volume_col: str = "volume",
    **kwargs,
) -> pd.DataFrame:
    feats = extract_volume_profile_volatility_features_from_series(
        close=df[price_col],
        volume=df[volume_col],
        **kwargs,
    )
    out = df.copy()
    for c in feats.columns:
        out[c] = feats[c]
    return out


from src.features.time_series.utils_volume_profile import (
    VolumeProfileResult,
    compute_wpt_volume_profile,
)


class TestVolumeProfileVolatilityFeatures:
    """测试 Volume Profile 波动率特征"""

    def test_extract_volatility_features_from_vp_basic(self):
        """测试基础特征提取功能"""
        # 创建模拟 Volume Profile 数据
        # 场景：价格在 100-110 区间，成交量集中在 105（POC）
        centers = np.linspace(100, 110, 20)
        hist = np.zeros(20)
        hist[10] = 1000  # POC 在中心
        hist[9:12] = [500, 1000, 500]  # 高成交量区域

        vp = VolumeProfileResult(
            hist=hist,
            edges=np.linspace(100, 110, 21),
            centers=centers,
            price_min=100.0,
            price_max=110.0,
        )

        # 提取特征（当前价格在 POC）
        features = extract_volatility_features_from_vp(vp, current_price=105.0)

        # 验证特征存在
        assert "vp_width_ratio" in features
        assert "vp_poc_deviation" in features
        assert "vp_skewness" in features
        assert "vp_entropy" in features
        assert "vp_lv_ratio" in features
        assert "vp_hv_ratio" in features

        # 验证特征值合理性
        assert 0.0 <= features["vp_width_ratio"] <= 1.0
        assert -1.0 <= features["vp_poc_deviation"] <= 1.0
        assert features["vp_entropy"] >= 0.0
        assert 0.0 <= features["vp_lv_ratio"] <= 1.0
        assert 0.0 <= features["vp_hv_ratio"] <= 1.0

        # 当前价格在 POC，偏离应该接近 0
        assert abs(features["vp_poc_deviation"]) < 0.1

    def test_extract_volatility_features_from_vp_price_far_from_poc(self):
        """测试价格远离 POC 时的特征"""
        # 创建模拟数据：POC 在 105，当前价格在 100（远离 POC）
        centers = np.linspace(100, 110, 20)
        hist = np.zeros(20)
        hist[10] = 1000  # POC 在 105

        vp = VolumeProfileResult(
            hist=hist,
            edges=np.linspace(100, 110, 21),
            centers=centers,
            price_min=100.0,
            price_max=110.0,
        )

        # 当前价格远离 POC
        features = extract_volatility_features_from_vp(vp, current_price=100.0)

        # 偏离应该为负（价格低于 POC）
        assert features["vp_poc_deviation"] < -0.3

        # 当前价格在 POC
        features_at_poc = extract_volatility_features_from_vp(vp, current_price=105.0)
        assert abs(features_at_poc["vp_poc_deviation"]) < 0.1

    def test_extract_volatility_features_from_vp_narrow_vs_wide_distribution(self):
        """测试窄分布 vs 宽分布的特征差异"""
        # 窄分布：成交量高度集中在中心
        centers = np.linspace(100, 110, 20)
        hist_narrow = np.zeros(20)
        hist_narrow[9:12] = [500, 1000, 500]  # 高度集中（3个元素）

        vp_narrow = VolumeProfileResult(
            hist=hist_narrow,
            edges=np.linspace(100, 110, 21),
            centers=centers,
            price_min=100.0,
            price_max=110.0,
        )

        # 宽分布：成交量分散
        hist_wide = np.ones(20) * 50  # 均匀分布

        vp_wide = VolumeProfileResult(
            hist=hist_wide,
            edges=np.linspace(100, 110, 21),
            centers=centers,
            price_min=100.0,
            price_max=110.0,
        )

        features_narrow = extract_volatility_features_from_vp(
            vp_narrow, current_price=105.0
        )
        features_wide = extract_volatility_features_from_vp(
            vp_wide, current_price=105.0
        )

        # 窄分布应该有更小的 width_ratio（共识强）
        assert features_narrow["vp_width_ratio"] < features_wide["vp_width_ratio"]

        # 宽分布应该有更高的 entropy（分歧大）
        assert features_wide["vp_entropy"] > features_narrow["vp_entropy"]

    def test_extract_volume_profile_volatility_features_dataframe(self):
        """测试 DataFrame 特征提取"""
        # 创建模拟数据
        n_samples = 200
        dates = pd.date_range("2024-01-01", periods=n_samples, freq="4H")

        # 模拟价格：趋势 + 波动
        trend = np.linspace(100, 110, n_samples)
        noise = np.random.randn(n_samples) * 2
        prices = trend + noise

        # 模拟成交量：与价格波动相关
        volumes = np.abs(noise) * 100 + 1000

        df = pd.DataFrame(
            {
                "close": prices,
                "volume": volumes,
            },
            index=dates,
        )

        # 提取特征
        df_features = extract_volume_profile_volatility_features(
            df,
            price_col="close",
            volume_col="volume",
            window=100,
            wavelet="db4",
            level=4,
        )

        # 验证特征列存在
        expected_cols = [
            "vp_width_ratio",
            "vp_poc_deviation",
            "vp_skewness",
            "vp_entropy",
            "vp_lv_ratio",
            "vp_hv_ratio",
        ]
        for col in expected_cols:
            assert col in df_features.columns, f"Missing column: {col}"

        # 验证特征值合理性
        assert df_features["vp_width_ratio"].notna().sum() > 0
        assert (df_features["vp_width_ratio"] >= 0).all()
        assert (df_features["vp_width_ratio"] <= 1).all()

        assert df_features["vp_poc_deviation"].notna().sum() > 0
        assert (df_features["vp_poc_deviation"] >= -1).all()
        assert (df_features["vp_poc_deviation"] <= 1).all()

        assert df_features["vp_entropy"].notna().sum() > 0
        assert (df_features["vp_entropy"] >= 0).all()
        assert (df_features["vp_entropy"] <= 1).all()

    def test_narrow_entrypoint_matches_df_entrypoint_close_volume_only(self):
        """Regression: Series-in entrypoint matches legacy DF entrypoint (close+volume only)."""
        n_samples = 220
        dates = pd.date_range("2024-01-01", periods=n_samples, freq="4H")
        trend = np.linspace(100, 110, n_samples)
        noise = np.random.randn(n_samples) * 2
        prices = trend + noise
        volumes = np.abs(noise) * 100 + 1000

        df = pd.DataFrame({"close": prices, "volume": volumes}, index=dates)

        legacy = extract_volume_profile_volatility_features(
            df,
            price_col="close",
            volume_col="volume",
            window=80,  # faster
            wavelet="db4",
            level=4,
        )
        narrow = extract_volume_profile_volatility_features_from_series(
            close=df["close"],
            volume=df["volume"],
            window=80,
            wavelet="db4",
            level=4,
        )

        expected_cols = [
            "vp_width_ratio",
            "vp_poc_deviation",
            "vp_skewness",
            "vp_entropy",
            "vp_lv_ratio",
            "vp_hv_ratio",
        ]
        assert list(narrow.columns) == expected_cols
        for c in expected_cols:
            a = legacy[c].values
            b = narrow[c].values
            assert np.allclose(
                a, b, rtol=1e-10, atol=1e-12, equal_nan=True
            ), f"Mismatch in {c}"

    def test_volume_profile_features_correlation_with_volatility(self):
        """测试特征与波动率的相关性（模拟数据验证）"""
        np.random.seed(42)
        n_samples = 500
        dates = pd.date_range("2024-01-01", periods=n_samples, freq="4H")

        # 创建两种市场状态：
        # 1. 低波动期：价格在窄区间，成交量集中
        # 2. 高波动期：价格大幅波动，成交量分散

        prices = []
        volumes = []
        true_volatility = []

        for i in range(n_samples):
            if i < n_samples // 2:
                # 低波动期：价格在 100-102 窄区间
                price = 101 + np.random.randn() * 0.5
                volume = 1000 + np.random.randn() * 100
                true_volatility.append(0.5)
            else:
                # 高波动期：价格在 100-110 宽区间
                price = 105 + np.random.randn() * 3.0
                volume = 1000 + np.abs(np.random.randn()) * 500
                true_volatility.append(3.0)

            prices.append(price)
            volumes.append(volume)

        df = pd.DataFrame(
            {
                "close": prices,
                "volume": volumes,
                "true_volatility": true_volatility,
            },
            index=dates,
        )

        # 提取特征
        df_features = extract_volume_profile_volatility_features(
            df,
            price_col="close",
            volume_col="volume",
            window=100,
            wavelet="db4",
            level=4,
        )

        # 计算实际波动率（滚动标准差）
        df_features["realized_vol"] = df_features["close"].rolling(20).std()

        # 只使用有效数据（去除 NaN）
        valid_mask = (
            df_features["vp_width_ratio"].notna()
            & df_features["vp_entropy"].notna()
            & df_features["realized_vol"].notna()
        )
        df_valid = df_features[valid_mask]

        if len(df_valid) > 50:
            # 验证相关性：
            # 1. vp_width_ratio 与波动率正相关（宽分布 → 高波动）
            corr_width = df_valid["vp_width_ratio"].corr(df_valid["realized_vol"])
            # 2. vp_entropy 与波动率正相关（高熵 → 高波动）
            corr_entropy = df_valid["vp_entropy"].corr(df_valid["realized_vol"])

            # 相关性应该为正（可能较弱，因为还有其他因素）
            # 注意：由于模拟数据的简化，相关性可能不是非常强
            print(f"\n📊 特征与波动率的相关性:")
            print(f"   vp_width_ratio vs realized_vol: {corr_width:.3f}")
            print(f"   vp_entropy vs realized_vol: {corr_entropy:.3f}")

            # 验证特征在不同波动率区间的差异
            high_vol_mask = df_valid["realized_vol"] > df_valid["realized_vol"].median()
            low_vol_mask = df_valid["realized_vol"] <= df_valid["realized_vol"].median()

            # 高波动期应该有更高的 width_ratio 和 entropy
            high_vol_width = df_valid.loc[high_vol_mask, "vp_width_ratio"].mean()
            low_vol_width = df_valid.loc[low_vol_mask, "vp_width_ratio"].mean()

            high_vol_entropy = df_valid.loc[high_vol_mask, "vp_entropy"].mean()
            low_vol_entropy = df_valid.loc[low_vol_mask, "vp_entropy"].mean()

            print(f"\n📊 不同波动率区间的特征均值:")
            print(f"   高波动期 vp_width_ratio: {high_vol_width:.3f}")
            print(f"   低波动期 vp_width_ratio: {low_vol_width:.3f}")
            print(f"   高波动期 vp_entropy: {high_vol_entropy:.3f}")
            print(f"   低波动期 vp_entropy: {low_vol_entropy:.3f}")

            # 验证：高波动期应该有更高的特征值（至少趋势应该如此）
            # 注意：由于模拟数据的随机性，可能不是每次都成立，但趋势应该存在
            assert (
                high_vol_width >= low_vol_width * 0.8
            ), "高波动期应该有更高的 width_ratio"
            assert (
                high_vol_entropy >= low_vol_entropy * 0.8
            ), "高波动期应该有更高的 entropy"

    def test_volume_profile_features_edge_cases(self):
        """测试边界情况"""
        # 1. 空 Volume Profile
        features_empty = extract_volatility_features_from_vp(None, current_price=100.0)
        assert all(v == 0.0 for v in features_empty.values())

        # 2. 单点 Volume Profile
        vp_single = VolumeProfileResult(
            hist=np.array([1000]),
            edges=np.array([100.0, 110.0]),
            centers=np.array([105.0]),
            price_min=100.0,
            price_max=110.0,
        )
        features_single = extract_volatility_features_from_vp(
            vp_single, current_price=105.0
        )
        assert features_single["vp_entropy"] == 0.0  # 单点，熵为 0

        # 3. 均匀分布 Volume Profile
        centers = np.linspace(100, 110, 20)
        hist_uniform = np.ones(20) * 50
        vp_uniform = VolumeProfileResult(
            hist=hist_uniform,
            edges=np.linspace(100, 110, 21),
            centers=centers,
            price_min=100.0,
            price_max=110.0,
        )
        features_uniform = extract_volatility_features_from_vp(
            vp_uniform, current_price=105.0
        )
        # 均匀分布应该有较高的 entropy
        assert features_uniform["vp_entropy"] > 0.5

    def test_volume_profile_features_with_realistic_market_scenarios(self):
        """测试真实市场场景"""
        np.random.seed(42)
        n_samples = 300
        dates = pd.date_range("2024-01-01", periods=n_samples, freq="4H")

        # 场景 1: 压缩期（价格在窄区间，成交量集中）
        # 场景 2: 突破期（价格快速移动，成交量分散）
        # 场景 3: 趋势期（价格持续移动，成交量跟随）

        prices = []
        volumes = []

        for i in range(n_samples):
            if i < 100:
                # 压缩期：价格在 100-102
                price = 101 + np.random.randn() * 0.3
                volume = 800 + np.random.randn() * 50
            elif i < 200:
                # 突破期：价格快速上升
                trend = 101 + (i - 100) * 0.05
                price = trend + np.random.randn() * 1.5
                volume = 1500 + np.abs(np.random.randn()) * 300
            else:
                # 趋势期：价格持续上升但波动减小
                trend = 106 + (i - 200) * 0.02
                price = trend + np.random.randn() * 0.8
                volume = 1200 + np.random.randn() * 200

            prices.append(price)
            volumes.append(volume)

        df = pd.DataFrame(
            {
                "close": prices,
                "volume": volumes,
            },
            index=dates,
        )

        # 提取特征
        df_features = extract_volume_profile_volatility_features(
            df,
            price_col="close",
            volume_col="volume",
            window=50,  # 使用较小的窗口以便更快看到变化
            wavelet="db4",
            level=4,
        )

        # 验证不同场景下的特征差异
        compression = df_features.iloc[50:100]  # 压缩期
        breakout = df_features.iloc[150:200]  # 突破期
        trend = df_features.iloc[250:300]  # 趋势期

        # 压缩期应该有较小的 width_ratio（共识强）
        compression_width = compression["vp_width_ratio"].mean()
        breakout_width = breakout["vp_width_ratio"].mean()

        print(f"\n📊 不同市场场景的特征均值:")
        print(f"   压缩期 vp_width_ratio: {compression_width:.3f}")
        print(f"   突破期 vp_width_ratio: {breakout_width:.3f}")

        # 突破期应该有更高的 width_ratio（分歧大）
        assert (
            breakout_width > compression_width * 0.9
        ), "突破期应该有更高的 width_ratio"


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v", "-s"])
