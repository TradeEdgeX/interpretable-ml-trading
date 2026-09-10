"""
流动性特征测试

测试内容：
1. 无未来函数测试（修改未来数据不影响历史特征值）⭐⭐⭐⭐⭐
2. 多资产归一化测试（特征分布对齐）⭐⭐⭐⭐
3. 流式vs批量一致性测试 ⭐⭐⭐⭐
4. 流动性真空区识别正确性验证
"""

import numpy as np
import pandas as pd
import pytest

from src.features.time_series.utils_liquidity_features import (
    extract_liquidity_features,
    compute_liquidity_void_features,
    compute_liquidity_void_features_from_series,
    compute_wpt_volume_energy_features,
    compute_wpt_volume_energy_features_from_series,
    # build_wpt_denoised_vpvr 已删除，使用 compute_unified_volume_profile_features
)


def create_mock_data(n_samples: int = 500, seed: int = 42) -> pd.DataFrame:
    """创建模拟数据用于测试"""
    np.random.seed(seed)
    dates = pd.date_range("2024-01-01", periods=n_samples, freq="5min")

    # 生成价格数据
    returns = np.random.randn(n_samples) * 0.01
    prices = 100 * np.exp(np.cumsum(returns))

    # 生成其他数据
    high = prices * (1 + np.abs(np.random.randn(n_samples) * 0.005))
    low = prices * (1 - np.abs(np.random.randn(n_samples) * 0.005))
    volume = np.random.lognormal(10, 1, n_samples)

    df = pd.DataFrame(
        {
            "close": prices,
            "high": high,
            "low": low,
            "volume": volume,
        },
        index=dates,
    )

    # 计算 ATR
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - df["close"].shift(1)).abs(),
            (df["low"] - df["close"].shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["atr"] = tr.rolling(window=14, min_periods=1).mean()
    df["atr"] = df["atr"].clip(lower=1e-6)

    return df


class TestLiquidityFeatures:
    """流动性特征测试类"""

    def test_basic_liquidity_features(self):
        """测试：基础流动性特征计算"""
        df = create_mock_data(n_samples=200)

        # 测试流动性真空特征
        result = compute_liquidity_void_features(
            df,
            price_col="close",
            volume_col="volume",
            high_col="high",
            low_col="low",
            atr_col="atr",
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(df)

        # 检查是否有流动性特征列
        liquidity_cols = [
            col
            for col in result.columns
            if "liquidity" in col.lower() or "void" in col.lower()
        ]
        assert len(liquidity_cols) > 0, "应该有流动性特征列"

    def test_extract_liquidity_features(self):
        """测试：提取流动性特征"""
        df = create_mock_data(n_samples=200)

        result = extract_liquidity_features(
            df,
            price_col="close",
            volume_col="volume",
            high_col="high",
            low_col="low",
            atr_col="atr",
            feature_type="void",  # 只测试 void 特征，避免 WPT 计算耗时
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(df)

    def test_liquidity_void_narrow_entrypoint_matches_legacy(self):
        """回归：narrow Series-in entrypoint 与 legacy DF 版本输出一致（slim 输入）。"""
        df = create_mock_data(n_samples=300, seed=123)
        legacy = compute_liquidity_void_features(
            df[["close", "volume", "atr"]].copy(),
            price_col="close",
            volume_col="volume",
            atr_col="atr",
        )
        narrow = compute_liquidity_void_features_from_series(
            close=df["close"],
            volume=df["volume"],
            atr=df["atr"],
        )
        cols = [
            "liquidity_void_detected",
            "liquidity_void_speed",
            "liquidity_void_volume_ratio",
            "liquidity_void_retracement",
            "liquidity_void_false_breakout_risk",
        ]
        for c in cols:
            assert np.allclose(
                legacy[c].values,
                narrow[c].values,
                equal_nan=True,
                rtol=1e-12,
                atol=1e-12,
            ), f"mismatch col={c}"

    def test_wpt_volume_energy_narrow_entrypoint_matches_legacy_small(self):
        """回归：wpt_volume_energy narrow entrypoint 与 legacy 版本一致（小样本以控制耗时）。"""
        df = create_mock_data(n_samples=90, seed=7)
        legacy = compute_wpt_volume_energy_features(
            df[["close", "volume"]].copy(),
            price_col="close",
            volume_col="volume",
            wavelet="db2",
            level=2,
            lookback_window=20,
        )
        narrow = compute_wpt_volume_energy_features_from_series(
            close=df["close"],
            volume=df["volume"],
            wavelet="db2",
            level=2,
            lookback_window=20,
        )
        cols = [
            "wpt_vper_low",
            "wpt_vper_mid",
            "wpt_vper_high",
            "wpt_energy_cascade",
            "wpt_multi_scale_consistency",
            "wpt_breakout_confidence",
            "wpt_false_breakout_risk",
        ]
        for c in cols:
            assert np.allclose(
                legacy[c].values, narrow[c].values, equal_nan=True, rtol=1e-9, atol=1e-9
            ), f"mismatch col={c}"

    def test_no_future_leak(self):
        """
        测试1：无未来函数（修改未来数据不影响历史特征值）⭐⭐⭐⭐⭐
        这是底线，必须确保特征计算不包含未来信息
        """
        df = create_mock_data(n_samples=500, seed=42)

        # 计算第一次特征
        result1 = compute_liquidity_void_features(
            df,
            price_col="close",
            volume_col="volume",
            high_col="high",
            low_col="low",
            atr_col="atr",
        )
        # 选择一个流动性特征列
        liquidity_cols = [
            col
            for col in result1.columns
            if "liquidity" in col.lower() or "void" in col.lower()
        ]
        if len(liquidity_cols) > 0:
            feature_col = liquidity_cols[0]
            feature_1 = result1[feature_col].copy()

            # 修改未来数据（从 t=250 开始）
            df_future_modified = df.copy()
            df_future_modified.loc[df_future_modified.index[250] :, "close"] *= 2.0
            df_future_modified.loc[df_future_modified.index[250] :, "high"] *= 2.0
            df_future_modified.loc[df_future_modified.index[250] :, "low"] *= 2.0
            df_future_modified.loc[df_future_modified.index[250] :, "volume"] *= 2.0

            # 重新计算 ATR
            tr = pd.concat(
                [
                    df_future_modified["high"] - df_future_modified["low"],
                    (
                        df_future_modified["high"]
                        - df_future_modified["close"].shift(1)
                    ).abs(),
                    (
                        df_future_modified["low"] - df_future_modified["close"].shift(1)
                    ).abs(),
                ],
                axis=1,
            ).max(axis=1)
            df_future_modified["atr"] = tr.rolling(window=14, min_periods=1).mean()
            df_future_modified["atr"] = df_future_modified["atr"].clip(lower=1e-6)

            # 重新计算特征
            result2 = compute_liquidity_void_features(
                df_future_modified,
                price_col="close",
                volume_col="volume",
                high_col="high",
                low_col="low",
                atr_col="atr",
            )
            feature_2 = result2[feature_col].copy()

            # 检查前 200 个时间点的特征值（应该不受未来数据影响）
            check_idx = df.index[:200]
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
            high = prices * (1 + np.abs(np.random.randn(n) * 0.005))
            low = prices * (1 - np.abs(np.random.randn(n) * 0.005))
            volume = np.random.lognormal(10, 1, n)

            df = pd.DataFrame(
                {
                    "close": prices,
                    "high": high,
                    "low": low,
                    "volume": volume,
                },
                index=dates,
            )

            # 计算 ATR
            tr = pd.concat(
                [
                    df["high"] - df["low"],
                    (df["high"] - df["close"].shift(1)).abs(),
                    (df["low"] - df["close"].shift(1)).abs(),
                ],
                axis=1,
            ).max(axis=1)
            df["atr"] = tr.rolling(window=14, min_periods=1).mean()
            df["atr"] = df["atr"].clip(lower=1e-6)

            # 计算特征
            result = compute_liquidity_void_features(
                df,
                price_col="close",
                volume_col="volume",
                high_col="high",
                low_col="low",
                atr_col="atr",
            )
            result["_symbol"] = symbol
            results.append(result)

        combined = pd.concat(results, ignore_index=False)

        # 检查不同资产的特征分布
        liquidity_cols = [
            col
            for col in combined.columns
            if "liquidity" in col.lower() or "void" in col.lower()
        ]
        if len(liquidity_cols) > 0:
            col = liquidity_cols[0]
            valid_data = combined[col].dropna()
            if len(valid_data) > 0:
                by_symbol = combined.groupby("_symbol")[col].agg(["mean", "std"])

                # 检查均值范围
                mean_range = by_symbol["mean"].max() - by_symbol["mean"].min()

                # 流动性特征应该对不同资产的价格水平不敏感（如果归一化正确）
                # 允许一定的差异，因为不同资产的基础特征可能不同
                assert mean_range < 1.0, (
                    f"{col} 在不同资产间的均值差异过大: {mean_range:.4f}，"
                    f"可能归一化不正确。各资产均值: {by_symbol['mean'].to_dict()}"
                )

    def test_streaming_vs_batch_consistency(self):
        """
        测试3：流式 vs 批量一致性 ⭐⭐⭐⭐
        对生产部署至关重要：生产环境往往是流式推理，而训练是批量计算
        """
        df = create_mock_data(n_samples=300, seed=42)
        window = 50

        # 批量计算（一次性计算所有数据）
        batch_result = compute_liquidity_void_features(
            df,
            price_col="close",
            volume_col="volume",
            high_col="high",
            low_col="low",
            atr_col="atr",
        )

        # 流式计算（分块处理，模拟生产环境）
        streaming_results = []
        for i in range(window, len(df)):
            df_stream = df.iloc[: i + 1].copy()
            stream_result = compute_liquidity_void_features(
                df_stream,
                price_col="close",
                volume_col="volume",
                high_col="high",
                low_col="low",
                atr_col="atr",
            )
            if len(stream_result) > 0:
                # 取最后一行（当前时间点的特征）
                streaming_results.append(stream_result.iloc[-1])

        if len(streaming_results) > 0:
            streaming_df = pd.DataFrame(streaming_results)
            streaming_df.index = df.index[window:][: len(streaming_df)]

            # 比较关键特征
            liquidity_cols = [
                col
                for col in batch_result.columns
                if "liquidity" in col.lower() or "void" in col.lower()
            ]
            if len(liquidity_cols) > 0:
                key_col = liquidity_cols[0]
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

    def test_wpt_breakout_confidence_range(self):
        """
        测试修复：wpt_breakout_confidence 应该在 [0, 1] 范围内

        Bug修复：添加 np.clip 确保值不会超出 [0, 1] 范围
        """
        df = create_mock_data(n_samples=150, seed=42)

        result = compute_wpt_volume_energy_features(
            df[["close", "volume"]].copy(),
            price_col="close",
            volume_col="volume",
            wavelet="db2",
            level=2,
            lookback_window=20,
        )

        # 检查 breakout_confidence 是否在 [0, 1] 范围内
        confidence = result["wpt_breakout_confidence"].dropna()
        if len(confidence) > 0:
            assert (
                confidence.min() >= 0.0
            ), f"wpt_breakout_confidence 应该 >= 0，最小值: {confidence.min()}"
            assert (
                confidence.max() <= 1.0
            ), f"wpt_breakout_confidence 应该 <= 1，最大值: {confidence.max()}"

        # 测试 narrow-IO 版本
        narrow_result = compute_wpt_volume_energy_features_from_series(
            close=df["close"],
            volume=df["volume"],
            wavelet="db2",
            level=2,
            lookback_window=20,
        )

        narrow_confidence = narrow_result["wpt_breakout_confidence"].dropna()
        if len(narrow_confidence) > 0:
            assert (
                narrow_confidence.min() >= 0.0
            ), f"narrow IO wpt_breakout_confidence 应该 >= 0，最小值: {narrow_confidence.min()}"
            assert (
                narrow_confidence.max() <= 1.0
            ), f"narrow IO wpt_breakout_confidence 应该 <= 1，最大值: {narrow_confidence.max()}"

    def test_price_impact_calculation(self):
        """
        测试：liquidity_void_price_impact 计算正确性

        验证 price_impact = (high - low) / volume 的计算逻辑
        """
        df = create_mock_data(n_samples=200, seed=42)

        # 创建 high 和 low
        df["high"] = df["close"] + np.abs(np.random.randn(len(df)) * 0.2)
        df["low"] = df["close"] - np.abs(np.random.randn(len(df)) * 0.2)

        result = compute_liquidity_void_features_from_series(
            close=df["close"],
            volume=df["volume"],
            high=df["high"],
            low=df["low"],
            atr=df["atr"],
        )

        # 验证 price_impact 列存在
        assert (
            "liquidity_void_price_impact" in result.columns
        ), "应该包含 price_impact 列"

        # 计算期望值
        expected_price_impact = (df["high"] - df["low"]) / (df["volume"] + 1e-8)

        # 对于检测到流动性真空的点，price_impact 应该等于期望值
        detected_mask = result["liquidity_void_detected"] == 1.0
        if detected_mask.sum() > 0:
            detected_price_impact = result.loc[
                detected_mask, "liquidity_void_price_impact"
            ]
            detected_expected = expected_price_impact[detected_mask]

            # 计算差异（考虑浮点误差）
            diff = (detected_price_impact - detected_expected).abs()
            max_diff = diff.max()

            assert (
                max_diff < 1e-6
            ), f"检测点的 price_impact 计算不正确，最大差异: {max_diff}"

        # 对于未检测到的点，price_impact 应该为 0.0
        not_detected_mask = result["liquidity_void_detected"] == 0.0
        if not_detected_mask.sum() > 0:
            not_detected_price_impact = result.loc[
                not_detected_mask, "liquidity_void_price_impact"
            ]
            assert (
                not_detected_price_impact == 0.0
            ).all(), "未检测到的点的 price_impact 应该为 0.0"

        # 验证 price_impact 值的合理性
        all_price_impact = result["liquidity_void_price_impact"].dropna()
        if len(all_price_impact) > 0:
            assert (
                all_price_impact >= 0.0
            ).all(), "price_impact 应该 >= 0（价格范围 / 成交量）"

    def test_price_impact_without_high_low(self):
        """
        测试：在没有提供 high/low 时，price_impact 使用估算值

        验证当 high/low 未提供时，代码能正常工作
        """
        df = create_mock_data(n_samples=100, seed=42)

        # 不提供 high 和 low
        result = compute_liquidity_void_features_from_series(
            close=df["close"],
            volume=df["volume"],
            atr=df["atr"],
        )

        # 应该仍然包含 price_impact 列
        assert "liquidity_void_price_impact" in result.columns

        # price_impact 应该计算出来（即使使用估算值）
        price_impact = result["liquidity_void_price_impact"]
        assert len(price_impact) == len(df), "price_impact 长度应该匹配"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
