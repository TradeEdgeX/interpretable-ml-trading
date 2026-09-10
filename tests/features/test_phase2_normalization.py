"""
Phase 2/3 归一化测试

验证 Phase 2/3 归一化的正确性和数据分布：
1. SMA Position: (close - sma_200) / close -> [-1, 1]
2. Volume Ratio: volume / rolling_mean_volume -> [0, 10]
3. DTW Similarity: exp(-dist/scale) -> [0, 1]
"""

import numpy as np
import pandas as pd
import pytest

from src.features.time_series.baseline_features import (
    compute_sma_position_from_series,
    compute_volume_ratio_from_series,
)
from src.features.time_series.utils_dtw_features import (
    extract_dtw_features_from_series,
)


def create_realistic_data(
    n_samples: int = 500, base_price: float = 100, seed: int = 42
) -> pd.DataFrame:
    """创建模拟真实市场的数据"""
    np.random.seed(seed)
    dates = pd.date_range("2024-01-01", periods=n_samples, freq="4H")

    # 生成带有趋势和波动的价格
    returns = np.random.randn(n_samples) * 0.02
    prices = base_price * np.exp(np.cumsum(returns))

    # 生成 SMA 200（模拟）
    sma_200 = (
        pd.Series(prices).rolling(window=min(200, n_samples // 2), min_periods=1).mean()
    )

    # 生成成交量
    volume = np.random.lognormal(mean=10, sigma=0.5, size=n_samples)

    return pd.DataFrame(
        {
            "close": prices,
            "sma_200": sma_200.values,
            "volume": volume,
        },
        index=dates,
    )


class TestPhase2Normalization:
    """Phase 2/3 归一化测试"""

    def test_sma_position_normalization(self):
        """测试 SMA Position 归一化"""
        df = create_realistic_data(300)

        result = compute_sma_position_from_series(
            close=pd.Series(df["close"].values, index=df.index),
            sma_200=pd.Series(df["sma_200"].values, index=df.index),
        )

        position = result["sma_200_position"]

        # 验证基本属性
        assert len(position) == len(df), "长度应该匹配"

        # 验证范围 [-1, 1]
        assert position.min() >= -1.0, f"最小值应该 >= -1，实际: {position.min():.4f}"
        assert position.max() <= 1.0, f"最大值应该 <= 1，实际: {position.max():.4f}"

        # 验证分布合理性
        assert position.std() > 0.001, "SMA position 应该有变化"

    def test_sma_position_cross_asset_comparability(self):
        """测试 SMA Position 跨资产可比性"""
        # 不同价格水平的资产
        assets = {
            "high_price": create_realistic_data(300, base_price=50000),
            "low_price": create_realistic_data(300, base_price=100),
        }

        position_stats = {}
        for name, df in assets.items():
            result = compute_sma_position_from_series(
                close=pd.Series(df["close"].values, index=df.index),
                sma_200=pd.Series(df["sma_200"].values, index=df.index),
            )
            position = result["sma_200_position"]
            position_stats[name] = {
                "mean": position.mean(),
                "std": position.std(),
                "range": position.max() - position.min(),
            }

        # 不同价格水平的资产应该有相似的分布
        ranges = [s["range"] for s in position_stats.values()]
        # 允许 3 倍以内的差异
        assert (
            max(ranges) / (min(ranges) + 1e-10) < 3
        ), f"不同资产的 SMA position 范围应该相似，实际: {position_stats}"

    def test_volume_ratio_normalization(self):
        """测试 Volume Ratio 归一化"""
        df = create_realistic_data(300)

        result = compute_volume_ratio_from_series(
            volume=pd.Series(df["volume"].values, index=df.index),
            window=20,
        )

        ratio = result["volume_ratio"]

        # 验证基本属性
        assert len(ratio) == len(df), "长度应该匹配"

        # 验证范围 [0, 10]
        assert ratio.min() >= 0.0, f"最小值应该 >= 0，实际: {ratio.min():.4f}"
        assert ratio.max() <= 10.0, f"最大值应该 <= 10，实际: {ratio.max():.4f}"

        # 验证均值接近 1.0
        assert 0.5 < ratio.mean() < 2.0, f"均值应该接近 1，实际: {ratio.mean():.4f}"

    def test_volume_ratio_cross_asset_comparability(self):
        """测试 Volume Ratio 跨资产可比性"""
        np.random.seed(42)

        # 不同成交量水平的资产
        n = 300
        idx = pd.date_range("2024-01-01", periods=n, freq="4H")

        assets = {
            "high_volume": pd.Series(
                np.random.lognormal(mean=15, sigma=0.5, size=n), index=idx
            ),
            "low_volume": pd.Series(
                np.random.lognormal(mean=8, sigma=0.5, size=n), index=idx
            ),
        }

        ratio_stats = {}
        for name, volume in assets.items():
            result = compute_volume_ratio_from_series(volume=volume, window=20)
            ratio = result["volume_ratio"]
            ratio_stats[name] = {
                "mean": ratio.mean(),
                "std": ratio.std(),
            }

        # 不同成交量水平的资产，归一化后的比率应该有相似的分布
        means = [s["mean"] for s in ratio_stats.values()]
        # 均值都应该接近 1.0
        for name, stats in ratio_stats.items():
            assert (
                0.5 < stats["mean"] < 2.0
            ), f"{name} 均值应该接近 1，实际: {stats['mean']:.4f}"

    def test_dtw_similarity_normalization(self):
        """测试 DTW 相似度归一化"""
        np.random.seed(42)
        n = 100
        idx = pd.date_range("2024-01-01", periods=n, freq="4H")
        close = pd.Series(100 + np.cumsum(np.random.randn(n) * 0.5), index=idx)

        result = extract_dtw_features_from_series(
            close=close,
            window=15,
            normalize_distance=True,
        )

        # 检查所有 dist 列（现在是相似度分数）
        dist_cols = [col for col in result.columns if "dist" in col]

        for col in dist_cols[:5]:  # 只检查前 5 个
            vals = result[col].dropna()
            if len(vals) > 0:
                # 验证范围 [0, 1]
                assert (
                    vals.min() >= 0.0
                ), f"{col} 最小值应该 >= 0，实际: {vals.min():.4f}"
                assert (
                    vals.max() <= 1.0
                ), f"{col} 最大值应该 <= 1，实际: {vals.max():.4f}"

    def test_dtw_similarity_cross_asset_comparability(self):
        """测试 DTW 相似度跨资产可比性"""
        np.random.seed(42)

        # 不同价格水平的资产
        n = 100
        idx = pd.date_range("2024-01-01", periods=n, freq="4H")

        assets = {
            "high_price": pd.Series(
                50000 + np.cumsum(np.random.randn(n) * 100), index=idx
            ),
            "low_price": pd.Series(
                100 + np.cumsum(np.random.randn(n) * 0.5), index=idx
            ),
        }

        similarity_stats = {}
        for name, close in assets.items():
            result = extract_dtw_features_from_series(
                close=close,
                window=15,
                normalize_distance=True,
            )
            # 取第一个 dist 列
            dist_cols = [col for col in result.columns if "dist" in col]
            if dist_cols:
                vals = result[dist_cols[0]].dropna()
                if len(vals) > 0:
                    similarity_stats[name] = {
                        "mean": vals.mean(),
                        "std": vals.std(),
                    }

        if len(similarity_stats) >= 2:
            # 不同价格水平的资产应该有相似的分布
            means = [s["mean"] for s in similarity_stats.values()]
            # 允许 3 倍以内的差异
            assert (
                max(means) / (min(means) + 1e-10) < 3
            ), f"不同资产的 DTW 相似度应该可比，实际: {similarity_stats}"

    def test_no_inf_nan_in_normalized_features(self):
        """测试归一化特征没有 inf/nan"""
        df = create_realistic_data(200)

        # SMA Position
        sma_result = compute_sma_position_from_series(
            close=pd.Series(df["close"].values, index=df.index),
            sma_200=pd.Series(df["sma_200"].values, index=df.index),
        )
        assert not np.isinf(
            sma_result["sma_200_position"]
        ).any(), "SMA position 不应该有 inf"
        assert not np.isnan(
            sma_result["sma_200_position"]
        ).any(), "SMA position 不应该有 nan"

        # Volume Ratio
        vol_result = compute_volume_ratio_from_series(
            volume=pd.Series(df["volume"].values, index=df.index),
        )
        assert not np.isinf(
            vol_result["volume_ratio"]
        ).any(), "Volume ratio 不应该有 inf"
        assert not np.isnan(
            vol_result["volume_ratio"]
        ).any(), "Volume ratio 不应该有 nan"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
