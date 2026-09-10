"""
Trend 特征测试

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
    compute_trend_r2_20_from_series,
    compute_trend_r2_50_from_series,
    compute_slope_consistency_score_from_series,
)


def create_mock_data(n_samples: int = 500, seed: int = 42) -> pd.DataFrame:
    """创建模拟数据用于测试"""
    np.random.seed(seed)
    dates = pd.date_range("2024-01-01", periods=n_samples, freq="4H")

    # 生成价格数据（带趋势）
    trend = np.linspace(0, 0.1, n_samples)  # 上升趋势
    noise = np.random.randn(n_samples) * 0.01
    prices = 100 * np.exp(np.cumsum(trend + noise))

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


class TestTrendFeatures:
    """Trend 特征测试"""

    def test_trend_r2_20_basic(self):
        """基础功能测试"""
        df = create_mock_data(200)
        result = compute_trend_r2_20_from_series(close=df["close"])

        # 检查输出列
        assert "trend_r2_20" in result.columns
        assert len(result) == len(df)

        # 检查数值合理性（R² 应该在 [0, 1] 范围内）
        valid_data = result["trend_r2_20"].dropna()
        if len(valid_data) > 0:
            assert (valid_data >= 0).all() and (valid_data <= 1).all()

    def test_trend_r2_50_basic(self):
        """基础功能测试"""
        df = create_mock_data(200)
        result = compute_trend_r2_50_from_series(close=df["close"])

        # 检查输出列
        assert "trend_r2_50" in result.columns
        assert len(result) == len(df)

        # 检查数值合理性
        valid_data = result["trend_r2_50"].dropna()
        if len(valid_data) > 0:
            assert (valid_data >= 0).all() and (valid_data <= 1).all()

    def test_slope_consistency_score_basic(self):
        """基础功能测试"""
        df = create_mock_data(200)
        result = compute_slope_consistency_score_from_series(close=df["close"])

        # 检查输出列
        assert "slope_consistency_score" in result.columns
        assert len(result) == len(df)

        # 检查数值合理性（应该在 [0, 3] 范围内，因为是对3个EMA斜率一致性的计数）
        valid_data = result["slope_consistency_score"].dropna()
        if len(valid_data) > 0:
            assert (valid_data >= 0).all() and (valid_data <= 3).all()

    def test_no_future_leak(self):
        """
        测试1：无未来函数（修改未来数据不影响历史特征值）⭐⭐⭐⭐⭐
        这是底线，必须确保特征计算不包含未来信息
        """
        df = create_mock_data(300)

        # 计算第一次特征
        result1 = compute_trend_r2_20_from_series(close=df["close"])
        trend_r2_1 = result1["trend_r2_20"].copy()

        # 修改未来数据
        df_future_modified = df.copy()
        if len(df) > 100:
            df_future_modified.loc[df_future_modified.index[100] :, "close"] *= 2.0

            # 重新计算特征
            result2 = compute_trend_r2_20_from_series(close=df_future_modified["close"])
            trend_r2_2 = result2["trend_r2_20"].copy()

            # 检查前50个时间点的特征值（应该不受未来数据影响）
            check_idx = df.index[:50]
            trend_1_check = trend_r2_1.loc[check_idx].dropna()
            trend_2_check = trend_r2_2.loc[check_idx].dropna()

            if len(trend_1_check) > 0 and len(trend_2_check) > 0:
                diff = (trend_1_check - trend_2_check).abs()
                max_diff = diff.max()

                assert (
                    max_diff < 1e-6
                ), f"未来数据变化不应影响历史 Trend 特征值，最大差异: {max_diff}"

    def test_normalization_multi_asset(self):
        """
        测试2：多资产归一化（特征分布对齐）⭐⭐⭐⭐

        验证：
        - 不同价格水平的资产，Trend R² 特征应该在相似范围内
        - Trend R² 是基于对数价格的，应该天然归一化
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
            result = compute_trend_r2_20_from_series(close=df["close"])
            result["_symbol"] = symbol
            results.append(result)

        combined = pd.concat(results, ignore_index=False)

        # 检查：不同资产的特征值应该在相似范围内（R² 都在 [0, 1]）
        valid_data = combined["trend_r2_20"].dropna()
        if len(valid_data) > 0:
            assert (valid_data >= 0).all() and (valid_data <= 1).all()

            # 检查不同资产的特征分布是否相似
            by_symbol = combined.groupby("_symbol")["trend_r2_20"].agg(["mean", "std"])
            # 均值应该在合理范围内（R² 通常在 0-1 之间）
            assert (by_symbol["mean"] >= 0).all()
            assert (by_symbol["mean"] <= 1).all()

    def test_streaming_vs_batch_consistency(self):
        """
        测试3：流式 vs 批量一致性 ⭐⭐⭐⭐
        对生产部署至关重要：生产环境往往是流式推理，而训练是批量计算
        """
        df = create_mock_data(300)

        # 批量计算（一次性计算所有数据）
        batch_result = compute_trend_r2_20_from_series(close=df["close"])

        # 流式计算（逐行模拟，每次只处理到当前时间点）
        window = 20  # trend_r2_20 需要至少 20 个数据点
        streaming_results = []
        for i in range(window, len(df)):
            df_stream = df.iloc[: i + 1].copy()
            stream_result = compute_trend_r2_20_from_series(close=df_stream["close"])
            if len(stream_result) > 0:
                # 取最后一行（当前时间点的特征）
                streaming_results.append(stream_result.iloc[-1])

        if len(streaming_results) > 0:
            streaming_df = pd.DataFrame(streaming_results)
            streaming_df.index = df.index[window:][: len(streaming_df)]

            # 比较关键特征
            key_col = "trend_r2_20"
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

    def test_trend_math_correctness(self):
        """测试：Trend 特征数学正确性"""
        df = create_mock_data(100)

        # 测试 trend_r2_20：应该基于对数价格的线性回归 R²
        result = compute_trend_r2_20_from_series(close=df["close"])

        # 手动计算 R² 验证
        log_price = np.log(df["close"].replace(0, np.nan).ffill())
        window = 20

        # 计算最后一个窗口的 R²
        if len(log_price) >= window:
            last_window = log_price.iloc[-window:]
            x = np.arange(len(last_window))
            y = last_window.values

            # 线性回归
            slope, intercept = np.polyfit(x, y, 1)
            y_pred = slope * x + intercept
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2_manual = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
            r2_manual = max(0.0, min(1.0, r2_manual))

            # 与特征值比较（允许微小误差）
            r2_feature = result["trend_r2_20"].iloc[-1]
            if not np.isnan(r2_feature):
                assert abs(r2_feature - r2_manual) < 0.01, (
                    f"Trend R² 计算不正确: 特征值={r2_feature:.6f}, "
                    f"手动计算={r2_manual:.6f}"
                )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
