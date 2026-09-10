"""
Momentum 特征测试

测试内容：
1. 无未来函数测试（修改未来数据不影响历史特征值）⭐⭐⭐⭐⭐
2. 多资产归一化测试（特征分布对齐）⭐⭐⭐⭐
3. 流式vs批量一致性测试 ⭐⭐⭐⭐
4. 特征数学正确性验证
"""

import numpy as np
import pandas as pd
import pytest

from src.features.time_series.baseline_features import (
    add_common_derived_features,
)
from src.features.time_series.utils_order_flow_features import (
    compute_vpin_momentum_features_from_series,
)


def create_mock_data(n_samples: int = 500, seed: int = 42) -> pd.DataFrame:
    """创建模拟数据用于测试"""
    np.random.seed(seed)
    dates = pd.date_range("2024-01-01", periods=n_samples, freq="4H")

    # 生成价格数据
    returns = np.random.randn(n_samples) * 0.01
    prices = 100 * np.exp(np.cumsum(returns))

    df = pd.DataFrame(
        {
            "open": prices * (1 + np.random.randn(n_samples) * 0.001),
            "high": prices * (1 + np.abs(np.random.randn(n_samples) * 0.002)),
            "low": prices * (1 - np.abs(np.random.randn(n_samples) * 0.002)),
            "close": prices,
            "volume": np.random.uniform(1000, 10000, n_samples),
        },
        index=dates,
    )

    return df


class TestMomentumFeatures:
    """Momentum 特征测试"""

    def test_momentum_basic(self):
        """基础功能测试"""
        df = create_mock_data(200)

        # 计算 momentum 特征（通过 add_common_derived_features）
        result = add_common_derived_features(
            df, required_features={"momentum_5", "momentum_10", "momentum_20"}
        )

        # 检查输出列
        for period in [5, 10, 20]:
            col = f"momentum_{period}"
            assert col in result.columns
            assert len(result) == len(df)

            # 检查数值合理性（momentum 是百分比变化，可以是任意值）
            valid_data = result[col].dropna()
            if len(valid_data) > 0:
                assert np.isfinite(valid_data).all()

    def test_vpin_momentum_basic(self):
        """VPIN momentum 基础功能测试"""
        df = create_mock_data(200)

        # 创建 VPIN 序列
        vpin = pd.Series(np.random.uniform(0, 1, len(df)), index=df.index)

        result = compute_vpin_momentum_features_from_series(vpin=vpin)

        # 检查输出列
        assert "vpin_momentum" in result.columns
        assert len(result) == len(df)

    def test_no_future_leak(self):
        """
        测试1：无未来函数（修改未来数据不影响历史特征值）⭐⭐⭐⭐⭐
        这是底线，必须确保特征计算不包含未来信息
        """
        df = create_mock_data(300)

        # 计算第一次特征
        result1 = add_common_derived_features(
            df, required_features={"momentum_5", "momentum_10", "momentum_20"}
        )
        momentum_5_1 = result1["momentum_5"].copy()

        # 修改未来数据
        df_future_modified = df.copy()
        if len(df) > 100:
            df_future_modified.loc[df_future_modified.index[100] :, "close"] *= 2.0

            # 重新计算特征
            result2 = add_common_derived_features(
                df_future_modified,
                required_features={"momentum_5", "momentum_10", "momentum_20"},
            )
            momentum_5_2 = result2["momentum_5"].copy()

            # 检查前50个时间点的特征值（应该不受未来数据影响）
            check_idx = df.index[:50]
            momentum_1_check = momentum_5_1.loc[check_idx].dropna()
            momentum_2_check = momentum_5_2.loc[check_idx].dropna()

            if len(momentum_1_check) > 0 and len(momentum_2_check) > 0:
                diff = (momentum_1_check - momentum_2_check).abs()
                max_diff = diff.max()

                assert (
                    max_diff < 1e-6
                ), f"未来数据变化不应影响历史 Momentum 特征值，最大差异: {max_diff}"

    def test_normalization_multi_asset(self):
        """
        测试2：多资产归一化（特征分布对齐）⭐⭐⭐⭐

        验证：
        - 不同价格水平的资产，Momentum 特征（百分比变化）应该天然归一化
        - Momentum 是百分比变化，不依赖于价格水平
        """
        np.random.seed(42)
        n = 200

        # 不同价格水平的资产
        assets = {
            "BTC": 50000 + np.cumsum(np.random.randn(n) * 100),
            "ETH": 3000 + np.cumsum(np.random.randn(n) * 10),
            "SOL": 100 + np.cumsum(np.random.randn(n) * 0.5),
        }

        results = []
        for symbol, prices in assets.items():
            dates = pd.date_range("2024-01-01", periods=n, freq="4H")
            df = pd.DataFrame({"close": prices}, index=dates)
            result = add_common_derived_features(
                df, required_features={"momentum_5", "momentum_10", "momentum_20"}
            )
            result["_symbol"] = symbol
            results.append(result)

        combined = pd.concat(results, ignore_index=False)

        # 检查：不同资产的特征值应该在相似范围内（百分比变化）
        for period in [5, 10, 20]:
            col = f"momentum_{period}"
            valid_data = combined[col].dropna()
            if len(valid_data) > 0:
                # 检查不同资产的特征分布是否相似
                by_symbol = combined.groupby("_symbol")[col].agg(["mean", "std"])
                # 均值应该在合理范围内（百分比变化通常在 -10% 到 +10% 之间）
                # 允许更大的范围，因为不同资产可能有不同的波动特性
                assert (
                    by_symbol["mean"].abs() < 1.0
                ).all()  # 平均动量应该在 ±100% 以内

    def test_streaming_vs_batch_consistency(self):
        """
        测试3：流式 vs 批量一致性 ⭐⭐⭐⭐
        对生产部署至关重要：生产环境往往是流式推理，而训练是批量计算
        """
        df = create_mock_data(300)

        # 批量计算（一次性计算所有数据）
        batch_result = add_common_derived_features(
            df, required_features={"momentum_5", "momentum_10", "momentum_20"}
        )

        # 流式计算（逐行模拟，每次只处理到当前时间点）
        window = 20  # momentum_20 需要至少 20 个数据点
        streaming_results = []
        for i in range(window, len(df)):
            df_stream = df.iloc[: i + 1].copy()
            stream_result = add_common_derived_features(
                df_stream,
                required_features={"momentum_5", "momentum_10", "momentum_20"},
            )
            if len(stream_result) > 0:
                # 取最后一行（当前时间点的特征）
                streaming_results.append(stream_result.iloc[-1])

        if len(streaming_results) > 0:
            streaming_df = pd.DataFrame(streaming_results)
            streaming_df.index = df.index[window:][: len(streaming_df)]

            # 比较关键特征
            key_col = "momentum_5"
            if key_col in batch_result.columns and key_col in streaming_df.columns:
                batch_vals = batch_result[key_col].iloc[window:].dropna()
                stream_vals = streaming_df[key_col].dropna()

                # 找到共同索引
                common_idx = batch_vals.index.intersection(stream_vals.index)
                if len(common_idx) > 10:  # 至少需要10个数据点
                    diff = (
                        batch_vals.loc[common_idx] - stream_vals.loc[common_idx]
                    ).abs()
                    max_diff = diff.max()
                    mean_diff = diff.mean()

                    # 允许一定的数值误差（由于滚动窗口计算的微小差异）
                    assert max_diff < 1e-5, (
                        f"流式与批量计算不一致，最大差异: {max_diff:.8f}, "
                        f"平均差异: {mean_diff:.8f}"
                    )

    def test_momentum_math_correctness(self):
        """测试：Momentum 特征数学正确性"""
        df = create_mock_data(100)

        # 测试 momentum_5：应该是 close.pct_change(5)
        result = add_common_derived_features(df, required_features={"momentum_5"})

        # 手动计算验证
        momentum_manual = df["close"].pct_change(5)

        # 与特征值比较（允许微小误差）
        momentum_feature = result["momentum_5"]
        common_idx = momentum_manual.index.intersection(momentum_feature.index)
        if len(common_idx) > 0:
            diff = (
                momentum_manual.loc[common_idx] - momentum_feature.loc[common_idx]
            ).abs()
            max_diff = diff.max()

            assert max_diff < 1e-10, f"Momentum 计算不正确: 最大差异={max_diff:.10f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
