"""
独立DTW特征测试

测试内容：
1. 无未来函数测试（修改未来数据不影响历史特征值）⭐⭐⭐⭐⭐
2. 多资产归一化测试（特征分布对齐）⭐⭐⭐⭐
3. 流式vs批量一致性测试 ⭐⭐⭐⭐
4. 单个模板特征提取正确性
"""

import numpy as np
import pandas as pd
import pytest

try:
    from src.features.time_series.utils_dtw_individual import (
        extract_dtw_template_features,
        extract_all_dtw_template_features,
    )

    DTW_AVAILABLE = True
except ImportError:
    DTW_AVAILABLE = False
    pytestmark = pytest.mark.skip(reason="DTW features not available")


def create_mock_data(n_samples: int = 200, seed: int = 42) -> pd.DataFrame:
    """创建模拟数据用于测试"""
    np.random.seed(seed)
    dates = pd.date_range("2024-01-01", periods=n_samples, freq="5min")

    # 生成价格数据
    returns = np.random.randn(n_samples) * 0.01
    prices = 100 * np.exp(np.cumsum(returns))

    df = pd.DataFrame(
        {
            "close": prices,
        },
        index=dates,
    )

    return df


@pytest.mark.skipif(not DTW_AVAILABLE, reason="DTW features not available")
class TestDTWIndividual:
    """独立DTW特征测试类"""

    def test_extract_single_template(self):
        """测试：提取单个DTW模板特征"""
        df = create_mock_data(n_samples=200)

        # 测试提取单个模板
        result = extract_dtw_template_features(
            df,
            template_name="hammer",
            price_col="close",
            window=20,
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(df)

        # 检查是否有DTW特征列
        dtw_cols = [col for col in result.columns if "dtw" in col.lower()]
        assert len(dtw_cols) > 0, "应该有DTW特征列"

    def test_extract_all_templates(self):
        """测试：提取所有DTW模板特征"""
        df = create_mock_data(n_samples=200)

        # 测试提取所有模板
        result = extract_all_dtw_template_features(
            df,
            price_col="close",
            window=20,
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(df)

        # 检查是否有多个DTW特征列
        dtw_cols = [col for col in result.columns if "dtw" in col.lower()]
        assert len(dtw_cols) > 1, "应该有多个DTW特征列"

    def test_no_future_leak(self):
        """
        测试1：无未来函数（修改未来数据不影响历史特征值）⭐⭐⭐⭐⭐
        这是底线，必须确保特征计算不包含未来信息
        """
        df = create_mock_data(n_samples=300, seed=42)
        window = 20

        # 计算第一次特征
        result1 = extract_dtw_template_features(
            df,
            template_name="hammer",
            price_col="close",
            window=window,
        )
        # 选择一个DTW特征列
        dtw_cols = [col for col in result1.columns if "dtw" in col.lower()]
        if len(dtw_cols) > 0:
            feature_col = dtw_cols[0]
            feature_1 = result1[feature_col].copy()

            # 修改未来数据（从 t=150 开始）
            df_future_modified = df.copy()
            df_future_modified.loc[df_future_modified.index[150] :, "close"] *= 2.0

            # 重新计算特征
            result2 = extract_dtw_template_features(
                df_future_modified,
                template_name="hammer",
                price_col="close",
                window=window,
            )
            feature_2 = result2[feature_col].copy()

            # 检查前 100 个时间点的特征值（应该不受未来数据影响）
            check_idx = df.index[:100]
            feat_1_check = feature_1.loc[check_idx].dropna()
            feat_2_check = feature_2.loc[check_idx].dropna()

            if len(feat_1_check) > 0 and len(feat_2_check) > 0:
                diff = (feat_1_check - feat_2_check).abs()
                max_diff = diff.max()

                assert (
                    max_diff < 1e-6
                ), f"未来数据变化不应影响历史特征值，最大差异: {max_diff}"

    def test_normalization_multi_asset(self):
        """
        测试2：多资产归一化（特征分布对齐）⭐⭐⭐⭐

        验证：
        - 不同价格水平的资产，特征分布应该对齐
        - 特征值应该在相似范围内，便于多资产训练
        """
        np.random.seed(42)
        n = 200
        dates = pd.date_range("2024-01-01", periods=n, freq="5min")

        # 创建不同价格水平的资产
        assets = {
            "BTC": 50000 + np.cumsum(np.random.randn(n) * 100),
            "ETH": 3000 + np.cumsum(np.random.randn(n) * 10),
            "SOL": 100 + np.cumsum(np.random.randn(n) * 0.5),
        }

        results = []
        for symbol, prices in assets.items():
            df = pd.DataFrame(
                {
                    "close": prices,
                },
                index=dates,
            )

            # 计算特征
            result = extract_dtw_template_features(
                df,
                template_name="hammer",
                price_col="close",
                window=20,
            )
            result["_symbol"] = symbol
            results.append(result)

        combined = pd.concat(results, ignore_index=False)

        # 检查不同资产的特征分布
        dtw_cols = [col for col in combined.columns if "dtw" in col.lower()]
        if len(dtw_cols) > 0:
            col = dtw_cols[0]
            valid_data = combined[col].dropna()
            if len(valid_data) > 0:
                by_symbol = combined.groupby("_symbol")[col].agg(["mean", "std"])

                # 检查均值范围
                mean_range = by_symbol["mean"].max() - by_symbol["mean"].min()

                # DTW特征应该对不同资产的价格水平不敏感（因为内部做了归一化）
                # 允许一定的差异，因为不同资产的价格走势不同
                assert mean_range < 10.0, (
                    f"{col} 在不同资产间的均值差异过大: {mean_range:.4f}，"
                    f"可能归一化不正确。各资产均值: {by_symbol['mean'].to_dict()}"
                )

    def test_streaming_vs_batch_consistency(self):
        """
        测试3：流式 vs 批量一致性 ⭐⭐⭐⭐
        对生产部署至关重要：生产环境往往是流式推理，而训练是批量计算
        """
        df = create_mock_data(n_samples=200, seed=42)
        window = 20

        # 批量计算（一次性计算所有数据）
        batch_result = extract_dtw_template_features(
            df,
            template_name="hammer",
            price_col="close",
            window=window,
        )

        # 流式计算（分块处理，模拟生产环境）
        streaming_results = []
        for i in range(window, len(df)):
            df_stream = df.iloc[: i + 1].copy()
            stream_result = extract_dtw_template_features(
                df_stream,
                template_name="hammer",
                price_col="close",
                window=window,
            )
            if len(stream_result) > 0:
                # 取最后一行（当前时间点的特征）
                streaming_results.append(stream_result.iloc[-1])

        if len(streaming_results) > 0:
            streaming_df = pd.DataFrame(streaming_results)
            streaming_df.index = df.index[window:][: len(streaming_df)]

            # 比较关键特征
            dtw_cols = [col for col in batch_result.columns if "dtw" in col.lower()]
            if len(dtw_cols) > 0:
                key_col = dtw_cols[0]
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
                            f"流式与批量计算不一致 ({key_col})，最大差异: {max_diff:.8f}, "
                            f"平均差异: {mean_diff:.8f}"
                        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
