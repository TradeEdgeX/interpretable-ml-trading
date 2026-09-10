"""
测试扩展波动率特征提取器
验证所有特征是否正确生成，使用模拟数据
"""

import numpy as np
import pandas as pd
import pytest

from src.features.time_series.utils_volatility_features import (
    extract_extended_volatility_features,
)

# 期望的所有特征列表（从feature_dependencies.yaml和volatility_model.yaml）
EXPECTED_FEATURES = [
    # 1. Multi-scale historical volatility (4个)
    "vol_raw_5",
    "vol_raw_10",
    "vol_raw_20",
    "vol_raw_60",
    # 2. ATR-derived features (15个)
    "vol_atr_norm",
    "vol_atr_ma_5",
    "vol_atr_ma_10",
    "vol_atr_ma_20",
    "vol_atr_std_5",
    "vol_atr_std_10",
    "vol_atr_std_20",
    "vol_atr_max_5",
    "vol_atr_max_10",
    "vol_atr_max_20",
    "vol_atr_min_5",
    "vol_atr_min_10",
    "vol_atr_min_20",
    "vol_atr_ratio_20",
    "vol_atr_change",
    "vol_atr_change_abs",
    # 3. Lag features (3个)
    "vol_lag_1",
    "vol_lag_2",
    "vol_lag_3",
    # 4. Trend features (4个)
    "vol_slope_5",
    "vol_slope_10",
    "vol_slope_20",
    "vol_accel",
    # 5. Moving averages (6个)
    "vol_ma_5",
    "vol_ma_10",
    "vol_ma_20",
    "vol_ema_5",
    "vol_ema_10",
    "vol_ema_20",
    # 6. Regime features (2个)
    "vol_zscore",
    "vol_percentile_approx",
    # 7. Range features (4个)
    "vol_range_10",
    "vol_range_20",
    "vol_range_pos_10",
    "vol_range_pos_20",
    # 8. Momentum features (3个)
    "vol_mom_3",
    "vol_mom_5",
    "vol_mom_10",
]

# 总共应该有41个特征
EXPECTED_FEATURE_COUNT = len(EXPECTED_FEATURES)


def create_mock_data(n_samples: int = 500, seed: int = 42) -> pd.DataFrame:
    """
    创建模拟数据用于测试

    Args:
        n_samples: 样本数量
        seed: 随机种子

    Returns:
        包含价格和ATR数据的DataFrame
    """
    np.random.seed(seed)

    # 创建时间索引
    dates = pd.date_range("2024-01-01", periods=n_samples, freq="5min")

    # 生成价格数据（随机游走 + 趋势）
    returns = np.random.randn(n_samples) * 0.01
    # 添加一些趋势和波动率聚集
    trend = np.sin(np.arange(n_samples) / 50) * 0.001
    volatility_cluster = np.abs(np.random.randn(n_samples)) * 0.005
    returns = returns + trend + volatility_cluster

    prices = 100 * np.exp(np.cumsum(returns))

    # 生成high/low（价格上下波动）
    high = prices * (1 + np.abs(np.random.randn(n_samples) * 0.005))
    low = prices * (1 - np.abs(np.random.randn(n_samples) * 0.005))

    df = pd.DataFrame(
        {
            "close": prices,
            "high": high,
            "low": low,
            "volume": np.random.lognormal(10, 1, n_samples),
        },
        index=dates,
    )

    # 计算ATR（使用真实的高低差）
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - df["close"].shift(1)).abs(),
            (df["low"] - df["close"].shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["atr"] = tr.rolling(window=14, min_periods=1).mean()

    # 确保ATR不为0（避免除零错误）
    df["atr"] = df["atr"].clip(lower=1e-6)

    return df


def test_extract_extended_volatility_features_basic():
    """测试基本功能：特征是否正确生成"""
    df = create_mock_data(n_samples=500)

    # 提取特征
    result = extract_extended_volatility_features(
        df,
        price_col="close",
        atr_col="atr",
        window=20,
        lag_periods=[1, 2, 3],
    )

    # 验证返回的是DataFrame
    assert isinstance(result, pd.DataFrame), "结果应该是DataFrame"

    # 验证索引匹配
    assert len(result) == len(df), "结果长度应该与输入数据相同"
    assert result.index.equals(df.index), "索引应该匹配"

    # 验证特征数量
    assert (
        len(result.columns) == EXPECTED_FEATURE_COUNT
    ), f"特征数量应该是{EXPECTED_FEATURE_COUNT}，实际是{len(result.columns)}"

    # 验证所有期望的特征都存在
    missing_features = set(EXPECTED_FEATURES) - set(result.columns)
    assert len(missing_features) == 0, f"缺少以下特征: {missing_features}"

    # 验证没有多余的特征
    extra_features = set(result.columns) - set(EXPECTED_FEATURES)
    assert len(extra_features) == 0, f"存在多余的特征: {extra_features}"


def test_extract_extended_volatility_features_values():
    """测试特征值是否合理（非NaN、非Inf、有限值）"""
    df = create_mock_data(n_samples=500)

    result = extract_extended_volatility_features(
        df,
        price_col="close",
        atr_col="atr",
        window=20,
        lag_periods=[1, 2, 3],
    )

    # 检查每个特征
    for col in result.columns:
        # 前20行可能有NaN（因为滚动窗口），但之后应该都是有效值
        valid_data = result[col].iloc[100:]  # 跳过前100行

        # 检查是否有NaN
        nan_count = valid_data.isna().sum()
        assert nan_count == 0, f"特征 {col} 在有效数据中有 {nan_count} 个NaN值"

        # 检查是否有Inf
        inf_count = np.isinf(valid_data).sum()
        assert inf_count == 0, f"特征 {col} 在有效数据中有 {inf_count} 个Inf值"

        # 检查是否都是有限值
        finite_count = np.isfinite(valid_data).sum()
        assert finite_count == len(
            valid_data
        ), f"特征 {col} 在有效数据中有 {len(valid_data) - finite_count} 个非有限值"


def test_extract_extended_volatility_features_ranges():
    """测试特征值是否在合理范围内"""
    df = create_mock_data(n_samples=500)

    result = extract_extended_volatility_features(
        df,
        price_col="close",
        atr_col="atr",
        window=20,
        lag_periods=[1, 2, 3],
    )

    # 跳过前100行（滚动窗口初始化）
    valid_data = result.iloc[100:]

    # 检查vol_raw_*应该在合理范围（波动率通常是0-0.1）
    for col in ["vol_raw_5", "vol_raw_10", "vol_raw_20", "vol_raw_60"]:
        if col in valid_data.columns:
            max_val = valid_data[col].max()
            min_val = valid_data[col].min()
            assert min_val >= 0, f"{col} 的最小值应该是非负的，实际是 {min_val}"
            assert max_val < 1.0, f"{col} 的最大值应该小于1.0，实际是 {max_val}"

    # 检查vol_atr_norm应该在合理范围
    if "vol_atr_norm" in valid_data.columns:
        max_val = valid_data["vol_atr_norm"].max()
        min_val = valid_data["vol_atr_norm"].min()
        assert min_val >= 0, f"vol_atr_norm 的最小值应该是非负的，实际是 {min_val}"
        assert max_val < 0.1, f"vol_atr_norm 的最大值应该小于0.1，实际是 {max_val}"

    # 检查vol_percentile_approx应该在[0, 1]范围内
    if "vol_percentile_approx" in valid_data.columns:
        max_val = valid_data["vol_percentile_approx"].max()
        min_val = valid_data["vol_percentile_approx"].min()
        assert min_val >= 0, f"vol_percentile_approx 的最小值应该>=0，实际是 {min_val}"
        assert max_val <= 1, f"vol_percentile_approx 的最大值应该<=1，实际是 {max_val}"

    # 检查vol_range_pos_*应该在[0, 1]范围内
    for col in ["vol_range_pos_10", "vol_range_pos_20"]:
        if col in valid_data.columns:
            max_val = valid_data[col].max()
            min_val = valid_data[col].min()
            assert min_val >= 0, f"{col} 的最小值应该>=0，实际是 {min_val}"
            assert max_val <= 1, f"{col} 的最大值应该<=1，实际是 {max_val}"


def test_extract_extended_volatility_features_without_atr():
    """测试没有ATR列时的行为"""
    df = create_mock_data(n_samples=500)
    # 移除ATR列
    df_no_atr = df.drop(columns=["atr"])

    result = extract_extended_volatility_features(
        df_no_atr,
        price_col="close",
        atr_col="atr",  # 指定不存在的列
        window=20,
        lag_periods=[1, 2, 3],
    )

    # 应该仍然生成非ATR相关的特征
    assert len(result.columns) > 0, "即使没有ATR，也应该生成一些特征"

    # 验证vol_raw_*特征仍然存在
    for col in ["vol_raw_5", "vol_raw_10", "vol_raw_20", "vol_raw_60"]:
        assert col in result.columns, f"特征 {col} 应该存在（不依赖ATR）"

    # 验证ATR相关特征不存在
    atr_features = [col for col in result.columns if "atr" in col.lower()]
    assert len(atr_features) == 0, f"不应该有ATR相关特征，但找到了: {atr_features}"


def test_extract_extended_volatility_features_custom_lag_periods():
    """测试自定义lag_periods"""
    df = create_mock_data(n_samples=500)

    # 使用自定义lag_periods
    result = extract_extended_volatility_features(
        df,
        price_col="close",
        atr_col="atr",
        window=20,
        lag_periods=[1, 5, 10],  # 自定义滞后
    )

    # 验证lag特征
    assert "vol_lag_1" in result.columns
    assert "vol_lag_5" in result.columns
    assert "vol_lag_10" in result.columns
    assert "vol_lag_2" not in result.columns  # 不应该存在
    assert "vol_lag_3" not in result.columns  # 不应该存在


def test_extract_extended_volatility_features_custom_window():
    """测试自定义window参数"""
    df = create_mock_data(n_samples=500)

    # 使用不同的window
    result = extract_extended_volatility_features(
        df,
        price_col="close",
        atr_col="atr",
        window=30,  # 自定义窗口
        lag_periods=[1, 2, 3],
    )

    # 验证特征仍然正确生成
    assert len(result.columns) == EXPECTED_FEATURE_COUNT
    assert "vol_raw_5" in result.columns
    assert "vol_zscore" in result.columns


def test_extract_extended_volatility_features_edge_cases():
    """测试边界情况"""
    # 测试非常小的数据集
    df_small = create_mock_data(n_samples=50)
    result_small = extract_extended_volatility_features(
        df_small,
        price_col="close",
        atr_col="atr",
        window=20,
        lag_periods=[1, 2, 3],
    )
    assert len(result_small) == 50
    assert len(result_small.columns) == EXPECTED_FEATURE_COUNT

    # 测试价格全为0的情况（应该被clip处理）
    df_zero = create_mock_data(n_samples=100)
    df_zero["close"] = 0.0
    result_zero = extract_extended_volatility_features(
        df_zero,
        price_col="close",
        atr_col="atr",
        window=20,
        lag_periods=[1, 2, 3],
    )
    # 应该不会崩溃，但特征值可能都是0或NaN
    assert len(result_zero) == 100


def test_extract_extended_volatility_features_feature_relationships():
    """测试特征之间的逻辑关系"""
    df = create_mock_data(n_samples=500)

    result = extract_extended_volatility_features(
        df,
        price_col="close",
        atr_col="atr",
        window=20,
        lag_periods=[1, 2, 3],
    )

    valid_data = result.iloc[100:]

    # vol_range_* 应该是 vol_max - vol_min，所以应该 >= 0
    for col in ["vol_range_10", "vol_range_20"]:
        if col in valid_data.columns:
            assert (valid_data[col] >= 0).all(), f"{col} 应该都是非负的"

    # vol_range_pos_* 应该在[0, 1]范围内
    for col in ["vol_range_pos_10", "vol_range_pos_20"]:
        if col in valid_data.columns:
            assert (valid_data[col] >= 0).all(), f"{col} 应该都是非负的"
            assert (valid_data[col] <= 1).all(), f"{col} 应该都<=1"

    # vol_atr_ratio_20 应该是当前ATR / 20期均值，应该接近1（如果波动率稳定）
    if "vol_atr_ratio_20" in valid_data.columns:
        ratio = valid_data["vol_atr_ratio_20"]
        assert (ratio > 0).all(), "vol_atr_ratio_20 应该都是正数"
        # 大部分值应该在合理范围内（比如0.5到2.0）
        reasonable_ratio = ((ratio >= 0.1) & (ratio <= 10.0)).sum() / len(ratio)
        assert reasonable_ratio > 0.8, (
            f"vol_atr_ratio_20 的大部分值应该在[0.1, 10.0]范围内，"
            f"实际只有 {reasonable_ratio*100:.1f}%"
        )


class TestExtendedVolatilityFeaturesCritical:
    """
    扩展波动率特征的四种关键测试

    1. 无未来函数测试（修改未来数据不影响历史特征值）⭐⭐⭐⭐⭐
    2. 多资产归一化测试（特征分布对齐）⭐⭐⭐⭐
    3. 流式vs批量一致性测试 ⭐⭐⭐⭐
    4. lag衰减平滑测试 ⭐⭐⭐
    """

    def test_no_future_leak(self):
        """
        测试1：无未来函数（修改未来数据不影响历史特征值）⭐⭐⭐⭐⭐
        这是底线，必须确保特征计算不包含未来信息
        """
        df = create_mock_data(n_samples=500, seed=42)
        window = 20

        # 计算第一次特征
        result1 = extract_extended_volatility_features(
            df,
            price_col="close",
            atr_col="atr",
            window=window,
            lag_periods=[1, 2, 3],
        )
        vol_raw_1 = result1["vol_raw_20"].copy()

        # 修改未来数据（从 t=250 开始）
        df_future_modified = df.copy()
        df_future_modified.loc[df_future_modified.index[250] :, "close"] *= 2.0
        df_future_modified.loc[df_future_modified.index[250] :, "high"] *= 2.0
        df_future_modified.loc[df_future_modified.index[250] :, "low"] *= 2.0

        # 重新计算 ATR
        tr = pd.concat(
            [
                df_future_modified["high"] - df_future_modified["low"],
                (
                    df_future_modified["high"] - df_future_modified["close"].shift(1)
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
        result2 = extract_extended_volatility_features(
            df_future_modified,
            price_col="close",
            atr_col="atr",
            window=window,
            lag_periods=[1, 2, 3],
        )
        vol_raw_2 = result2["vol_raw_20"].copy()

        # 检查前 200 个时间点的特征值（应该不受未来数据影响）
        check_idx = df.index[:200]
        vol_1_check = vol_raw_1.loc[check_idx].dropna()
        vol_2_check = vol_raw_2.loc[check_idx].dropna()

        if len(vol_1_check) > 0 and len(vol_2_check) > 0:
            diff = (vol_1_check - vol_2_check).abs()
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
            # 生成 high/low
            high = prices * (1 + np.abs(np.random.randn(n) * 0.005))
            low = prices * (1 - np.abs(np.random.randn(n) * 0.005))

            df = pd.DataFrame(
                {
                    "close": prices,
                    "high": high,
                    "low": low,
                    "volume": np.random.lognormal(10, 1, n),
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
            result = extract_extended_volatility_features(
                df,
                price_col="close",
                atr_col="atr",
                window=20,
                lag_periods=[1, 2, 3],
            )
            result["_symbol"] = symbol
            results.append(result)

        combined = pd.concat(results, ignore_index=False)

        # 检查不同资产的特征分布
        for col in ["vol_raw_20", "vol_zscore", "vol_percentile_approx"]:
            if col in combined.columns:
                valid_data = combined[col].dropna()
                if len(valid_data) > 0:
                    by_symbol = combined.groupby("_symbol")[col].agg(["mean", "std"])

                    # 检查均值范围
                    mean_range = by_symbol["mean"].max() - by_symbol["mean"].min()
                    mean_cv = mean_range / (by_symbol["mean"].abs().mean() + 1e-10)

                    # 对于 vol_raw_20，不同资产的均值差异不应该太大
                    if col == "vol_raw_20":
                        assert mean_cv < 2.0, (
                            f"{col} 在不同资产间的均值变异系数过大: {mean_cv:.4f}，"
                            f"可能归一化不正确。各资产均值: {by_symbol['mean'].to_dict()}"
                        )

                    # 对于 vol_zscore 和 vol_percentile_approx，应该更接近
                    if col in ["vol_zscore", "vol_percentile_approx"]:
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
        window = 20

        # 批量计算（一次性计算所有数据）
        batch_result = extract_extended_volatility_features(
            df,
            price_col="close",
            atr_col="atr",
            window=window,
            lag_periods=[1, 2, 3],
        )

        # 流式计算（分块处理，模拟生产环境）
        streaming_results = []
        for i in range(window, len(df)):
            df_stream = df.iloc[: i + 1].copy()
            stream_result = extract_extended_volatility_features(
                df_stream,
                price_col="close",
                atr_col="atr",
                window=window,
                lag_periods=[1, 2, 3],
            )
            if len(stream_result) > 0:
                # 取最后一行（当前时间点的特征）
                streaming_results.append(stream_result.iloc[-1])

        if len(streaming_results) > 0:
            streaming_df = pd.DataFrame(streaming_results)
            streaming_df.index = df.index[window:][: len(streaming_df)]

            # 比较关键特征
            key_cols = ["vol_raw_20", "vol_zscore", "vol_lag_1"]
            for key_col in key_cols:
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

    def test_lag_effect_decay_smoothly(self):
        """
        测试4：lag 特征有效性应缓慢衰减 ⭐⭐⭐

        验证：
        - lag_1, lag_2, lag_3 与目标变量的相关性应该递减
        - 不应出现断崖式下降（如 lag_1=0.3, lag_2=0.01）
        """
        df = create_mock_data(n_samples=500, seed=42)

        # 提取包含 lag 特征的特征
        features = extract_extended_volatility_features(
            df,
            price_col="close",
            atr_col="atr",
            window=20,
            lag_periods=[1, 2, 3],
        )

        # 构造简单目标（未来波动率，仅用于测试）
        log_ret = np.log(df["close"] / df["close"].shift(1))
        # 未来5期的波动率（注意：仅用于测试相关性衰减模式）
        target = log_ret.rolling(5).std().shift(-5)

        # 检查 lag 特征
        lag_cols = ["vol_lag_1", "vol_lag_2", "vol_lag_3"]
        correlations = []
        lag_periods = []

        for col in lag_cols:
            if col in features.columns:
                # 找到共同的有效索引
                feature_valid = features[col].dropna()
                target_valid = target.dropna()
                common_idx = feature_valid.index.intersection(target_valid.index)

                if len(common_idx) > 50:  # 需要足够的数据点
                    corr = feature_valid.loc[common_idx].corr(
                        target_valid.loc[common_idx]
                    )
                    if not np.isnan(corr):
                        correlations.append(abs(corr))
                        # 从列名提取 lag 周期
                        lag_period = int(col.split("_")[-1])
                        lag_periods.append(lag_period)

        # 验证相关性应该大致递减
        if len(correlations) >= 2:
            # 按 lag 周期排序
            sorted_data = sorted(zip(lag_periods, correlations))
            lag_periods_sorted = [x[0] for x in sorted_data]
            correlations_sorted = [x[1] for x in sorted_data]

            # 检查是否递减（允许小幅波动）
            for i in range(len(correlations_sorted) - 1):
                # lag 增加时，相关性应该递减或至少不会大幅增加
                if correlations_sorted[i] > 0.1:  # 只有当相关性足够大时才检查
                    ratio = correlations_sorted[i + 1] / correlations_sorted[i]
                    # lag_{i+1} 的相关性不应该比 lag_i 低太多（不应低于 0.1 倍）
                    assert ratio > 0.1, (
                        f"lag_{lag_periods_sorted[i+1]} 相关性断崖式下降: "
                        f"lag_{lag_periods_sorted[i]}={correlations_sorted[i]:.4f}, "
                        f"lag_{lag_periods_sorted[i+1]}={correlations_sorted[i+1]:.4f}, "
                        f"ratio={ratio:.4f}"
                    )

    def test_lag_features_correlation_with_base(self):
        """
        测试：lag 特征与基础特征的相关性

        验证：
        - lag_1 应该与当前特征高度相关
        - lag_2, lag_3 的相关性应该递减
        """
        df = create_mock_data(n_samples=500, seed=42)

        features = extract_extended_volatility_features(
            df,
            price_col="close",
            atr_col="atr",
            window=20,
            lag_periods=[1, 2, 3],
        )

        # 使用 vol_raw_20 作为基础特征
        base_col = "vol_raw_20"
        if base_col in features.columns:
            base_feature = features[base_col].dropna()

            lag_cols = ["vol_lag_1", "vol_lag_2", "vol_lag_3"]
            correlations = []

            for col in lag_cols:
                if col in features.columns:
                    lag_feature = features[col].dropna()
                    common_idx = base_feature.index.intersection(lag_feature.index)

                    if len(common_idx) > 50:
                        corr = base_feature.loc[common_idx].corr(
                            lag_feature.loc[common_idx]
                        )
                        if not np.isnan(corr):
                            correlations.append(abs(corr))

            # lag_1 应该与基础特征高度相关（因为就是前1期的值）
            if len(correlations) > 0:
                assert (
                    correlations[0] > 0.7
                ), f"lag_1 应该与基础特征高度相关，实际: {correlations[0]:.4f}"

                # 后续 lag 的相关性应该递减
                if len(correlations) >= 2:
                    assert (
                        correlations[1] > 0.3
                    ), f"lag_2 与基础特征的相关性过低: {correlations[1]:.4f}"

                    # 检查是否递减
                    if correlations[0] > 0.5:
                        ratio = correlations[1] / correlations[0]
                        assert ratio > 0.3, (
                            f"lag_2 相关性断崖式下降: "
                            f"lag_1={correlations[0]:.4f}, lag_2={correlations[1]:.4f}, "
                            f"ratio={ratio:.4f}"
                        )


if __name__ == "__main__":
    # 运行所有测试
    pytest.main([__file__, "-v", "--tb=short"])
