"""
DTW 特征测试（补充完整测试）

测试内容：
1. 无未来函数测试（修改未来数据不影响历史特征值）⭐⭐⭐⭐⭐
2. 多资产归一化测试（特征分布对齐）⭐⭐⭐⭐
3. 流式vs批量一致性测试 ⭐⭐⭐⭐
4. 特征数学正确性验证
"""

import numpy as np
import pandas as pd
import pytest

from src.features.time_series.utils_dtw_features import (
    extract_dtw_features,
    extract_dtw_features_from_series,
)


def test_dtw_from_series_matches_df_entrypoint_small():
    idx = pd.date_range("2024-01-01", periods=80, freq="5min")
    close = pd.Series(
        np.linspace(100, 110, len(idx)) + np.sin(np.arange(len(idx)) / 5), index=idx
    )
    dist = pd.Series(np.linspace(0.5, 2.0, len(idx)), index=idx)
    atr = pd.Series(np.linspace(1.0, 1.2, len(idx)), index=idx)

    df = pd.DataFrame(
        {"close": close, "dist_to_nearest_sr": dist, "atr": atr}, index=idx
    )

    params = dict(
        window=[15, 20],
        template_filter=["hammer", "double_bottom"],
        compute_only_near_sr=False,
        sr_dist_col="dist_to_nearest_sr",
        sr_threshold=1.5,
        normalize_distance=True,
        warping_window=0.1,
        use_c=True,
    )

    df_out = extract_dtw_features(df, price_col="close", **params)
    s_out = extract_dtw_features_from_series(
        close=close, dist_to_nearest_sr=dist, atr=atr, **params
    )

    assert list(df_out.columns) == list(s_out.columns)
    # allow object columns (dtw_best_match_*) to compare exactly as strings
    for c in df_out.columns:
        assert df_out[c].equals(s_out[c])


def create_mock_data(n_samples: int = 200, seed: int = 42) -> pd.DataFrame:
    """创建模拟数据用于测试"""
    np.random.seed(seed)
    dates = pd.date_range("2024-01-01", periods=n_samples, freq="4H")

    # 生成价格数据
    prices = 100 + np.cumsum(np.random.randn(n_samples) * 0.5)

    df = pd.DataFrame(
        {
            "close": prices,
            "dist_to_nearest_sr": np.random.uniform(0.5, 2.0, n_samples),
            "atr": np.random.uniform(1.0, 1.2, n_samples),
        },
        index=dates,
    )

    return df


class TestDTWFeaturesComplete:
    """DTW 特征完整测试（补充 test_advanced_features.py 中的测试）"""

    def test_dtw_no_future_leak(self):
        """
        测试1：无未来函数（修改未来数据不影响历史特征值）⭐⭐⭐⭐⭐
        """
        df = create_mock_data(200)
        window = 20

        # 计算第一次 DTW 特征
        result1 = extract_dtw_features(df, price_col="close", window=window)
        dtw_min_1 = result1["dtw_min_dist"].copy()

        # 修改未来数据（t=100 之后）
        df_future_modified = df.copy()
        df_future_modified.loc[df_future_modified.index[100] :, "close"] *= 2.0

        # 重新计算 DTW 特征
        result2 = extract_dtw_features(
            df_future_modified, price_col="close", window=window
        )
        dtw_min_2 = result2["dtw_min_dist"].copy()

        # 检查前 50 个时间点的特征值（应该不受未来数据影响）
        check_idx = df.index[:50]
        dtw_1_check = dtw_min_1.loc[check_idx].dropna()
        dtw_2_check = dtw_min_2.loc[check_idx].dropna()

        if len(dtw_1_check) > 0 and len(dtw_2_check) > 0:
            diff = (dtw_1_check - dtw_2_check).abs()
            max_diff = diff.max()

            # 允许微小的数值误差（浮点数精度）
            assert (
                max_diff < 1e-6
            ), f"未来数据变化不应影响历史 DTW 特征值，最大差异: {max_diff}"

    def test_dtw_normalization_multi_asset(self):
        """
        测试2：多资产归一化（特征分布对齐）⭐⭐⭐⭐
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
            df = pd.DataFrame(
                {
                    "close": prices,
                    "dist_to_nearest_sr": np.random.uniform(0.5, 2.0, n),
                    "atr": np.random.uniform(1.0, 1.2, n),
                },
                index=dates,
            )
            result = extract_dtw_features(df, price_col="close", window=20)
            result["_symbol"] = symbol
            results.append(result)

        combined = pd.concat(results, ignore_index=False)

        # 检查：不同资产的DTW距离分布应该相似（因为价格已归一化）
        for col in ["dtw_min_dist", "dtw_hammer_dist"]:
            if col in combined.columns:
                valid_data = combined[col].dropna()
                if len(valid_data) > 0:
                    by_symbol = combined.groupby("_symbol")[col].agg(["mean", "std"])
                    # 不同资产的距离均值应该在相似范围内
                    mean_range = by_symbol["mean"].max() - by_symbol["mean"].min()
                    # 如果归一化正确，不同资产的均值差异不应该太大
                    assert (
                        mean_range < 10.0
                    ), f"{col} 在不同资产间的均值差异过大: {mean_range:.4f}"

    def test_dtw_streaming_vs_batch_consistency(self):
        """
        测试3：流式 vs 批量一致性 ⭐⭐⭐⭐
        """
        df = create_mock_data(200)
        window = 20

        # 批量计算（一次性处理）
        batch_result = extract_dtw_features(df, price_col="close", window=window)

        # 流式计算（分块处理，模拟在线推理）
        chunk_size = 30
        streaming_results = []

        for i in range(0, len(df), chunk_size):
            chunk = df.iloc[i : i + chunk_size].copy()
            chunk_result = extract_dtw_features(chunk, price_col="close", window=window)
            streaming_results.append(chunk_result)

        # 合并流式结果
        streaming_result = pd.concat(streaming_results, axis=0)

        # 比较关键特征（跳过前 window 行）
        skip_rows = window
        if len(batch_result) > skip_rows and len(streaming_result) > skip_rows:
            batch_valid = batch_result.iloc[skip_rows:]["dtw_min_dist"].dropna()
            streaming_valid = streaming_result.iloc[skip_rows:]["dtw_min_dist"].dropna()

            # 找到共同索引
            common_idx = batch_valid.index.intersection(streaming_valid.index)
            if len(common_idx) > 0:
                batch_vals = batch_valid.loc[common_idx]
                streaming_vals = streaming_valid.loc[common_idx]
                diff = (batch_vals - streaming_vals).abs()

                # 允许一定误差（因为分块计算可能导致边界处理略有不同）
                max_diff = diff.max()
                assert max_diff < 1e-5, f"流式与批量计算不一致，最大差异: {max_diff}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
