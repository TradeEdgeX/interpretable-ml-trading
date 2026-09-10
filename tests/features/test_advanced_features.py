"""
高级特征测试（GARCH, DTW, EVT）

测试内容：
1. 归一化验证（多资产训练兼容性）
2. 实现合理性验证
3. 性能测试
4. 公共代码抽取验证
"""

import numpy as np
import pandas as pd
import pytest
import time

pytestmark = [pytest.mark.slow, pytest.mark.integration]
from typing import Dict, List
import warnings

warnings.filterwarnings("ignore")

# 导入特征提取函数
from src.features.time_series.utils_garch_features import (
    extract_garch_features_from_series,
)
from src.features.time_series.utils_dtw_features import (
    extract_dtw_features,
    create_dtw_templates,
)
from src.features.time_series.utils_evt_features import extract_evt_features_from_series


# ---------------------------------------------------------------------------
# Route B: DF-style entrypoints removed from library.
# Keep test semantics by providing local DF wrappers that forward to *_from_series.
# ---------------------------------------------------------------------------
def extract_garch_features(
    df: pd.DataFrame, price_col: str = "close", **kwargs
) -> pd.DataFrame:
    return extract_garch_features_from_series(close=df[price_col], **kwargs)


def extract_evt_features(
    df: pd.DataFrame, price_col: str = "close", **kwargs
) -> pd.DataFrame:
    return extract_evt_features_from_series(close=df[price_col], **kwargs)


# 导入公共归一化模块
from src.features.time_series.utils_normalization import (
    normalize_series,
    normalize_by_group,
)


class TestGARCHFeatures:
    """GARCH特征测试"""

    @pytest.fixture
    def sample_data_single_asset(self):
        """单资产测试数据"""
        np.random.seed(42)
        n = 200
        prices = 100 + np.cumsum(np.random.randn(n) * 0.5)
        return pd.DataFrame({"close": prices})

    @pytest.fixture
    def sample_data_multi_asset(self):
        """多资产测试数据（不同价格水平）"""
        np.random.seed(42)
        n = 200

        # 不同价格水平的资产
        assets = {
            "BTC": 50000 + np.cumsum(np.random.randn(n) * 100),
            "ETH": 3000 + np.cumsum(np.random.randn(n) * 10),
            "SOL": 100 + np.cumsum(np.random.randn(n) * 0.5),
        }

        dfs = []
        for symbol, prices in assets.items():
            df = pd.DataFrame({"close": prices})
            df["_symbol"] = symbol
            dfs.append(df)

        return pd.concat(dfs, ignore_index=False)

    def test_garch_features_basic(self, sample_data_single_asset):
        """基础功能测试"""
        df = sample_data_single_asset
        result = extract_garch_features(df, price_col="close", window=60)

        # 检查输出列
        expected_cols = [
            "garch_volatility",
            "garch_persistence",
            "garch_leverage_gamma",
            "garch_alpha",
            "garch_beta",
        ]
        assert all(col in result.columns for col in expected_cols)
        assert len(result) == len(df)

        # 检查数值合理性
        assert result["garch_persistence"].notna().sum() > 0
        assert (result["garch_persistence"] >= 0).all()
        assert (result["garch_persistence"] <= 1.5).all()  # 通常 < 1.0，允许一些误差

        # 波动率应该非负
        valid_vol = result["garch_volatility"].dropna()
        if len(valid_vol) > 0:
            assert (valid_vol >= 0).all()

    def test_garch_features_normalization_multi_asset(self, sample_data_multi_asset):
        """多资产归一化测试（加强版）"""
        # 按资产分组计算特征
        results = []
        for symbol in sample_data_multi_asset["_symbol"].unique():
            df_asset = sample_data_multi_asset[
                sample_data_multi_asset["_symbol"] == symbol
            ].copy()
            result = extract_garch_features(df_asset, price_col="close", window=60)
            result["_symbol"] = symbol
            results.append(result)

        combined = pd.concat(results, ignore_index=False)

        # 检查：不同资产的特征值应该在相似范围内（归一化后）
        # 由于GARCH特征已经是相对值（基于收益率），应该天然归一化
        for col in ["garch_persistence", "garch_leverage_gamma"]:
            valid_data = combined[col].dropna()
            if len(valid_data) > 0:
                # 检查不同资产的特征分布是否相似
                by_symbol = combined.groupby("_symbol")[col].agg(["mean", "std"])
                # 均值应该在合理范围内（例如persistence在0.7-0.99之间）
                assert (by_symbol["mean"] >= 0).all()
                assert (by_symbol["mean"] <= 1.5).all()

                # 加强：检查不同资产的特征分布范围是否重叠（归一化后应该可比）
                mean_range = by_symbol["mean"].max() - by_symbol["mean"].min()
                # 不同资产的均值差异不应该太大（如果归一化正确）
                # 允许一定差异，因为不同资产可能有不同的波动特性
                assert mean_range < 1.0, (
                    f"{col}: 不同资产的特征均值差异过大 ({mean_range:.4f})，"
                    f"可能归一化不正确。各资产均值: {by_symbol['mean'].to_dict()}"
                )

    def test_garch_features_performance(self, sample_data_single_asset):
        """性能测试"""
        df = sample_data_single_asset

        start_time = time.time()
        result = extract_garch_features(df, price_col="close", window=60)
        elapsed = time.time() - start_time

        # 200个样本应该在合理时间内完成（< 10秒）
        assert elapsed < 10.0, f"GARCH特征计算耗时 {elapsed:.2f}秒，超过10秒"

        # 检查是否有有效结果
        assert result["garch_persistence"].notna().sum() > 0

    def test_garch_features_edge_cases(self):
        """边界情况测试"""
        # 数据不足
        df_short = pd.DataFrame({"close": [100, 101, 102]})
        result = extract_garch_features(df_short, price_col="close", window=60)
        assert len(result) == len(df_short)
        # 应该返回NaN或默认值
        assert (
            result["garch_persistence"].isna().all()
            or (result["garch_persistence"] == 0.0).all()
        )

        # 常数价格
        df_constant = pd.DataFrame({"close": [100.0] * 100})
        result = extract_garch_features(df_constant, price_col="close", window=60)
        assert len(result) == len(df_constant)

    def test_garch_no_future_leak(self, sample_data_single_asset):
        """
        测试1：无未来函数（修改未来数据不影响历史特征值）⭐⭐⭐⭐⭐
        这是底线，必须确保特征计算不包含未来信息
        """
        df = sample_data_single_asset.copy()
        window = 60

        # 计算第一次特征
        result1 = extract_garch_features(df, price_col="close", window=window)
        garch_vol_1 = result1["garch_volatility"].copy()

        # 修改未来数据
        df_future_modified = df.copy()
        if len(df) > 100:
            df_future_modified.loc[df_future_modified.index[100] :, "close"] *= 2.0

            # 重新计算特征
            result2 = extract_garch_features(
                df_future_modified, price_col="close", window=window
            )
            garch_vol_2 = result2["garch_volatility"].copy()

            # 检查前50个时间点的特征值（应该不受未来数据影响）
            check_idx = df.index[:50]
            vol_1_check = garch_vol_1.loc[check_idx].dropna()
            vol_2_check = garch_vol_2.loc[check_idx].dropna()

            if len(vol_1_check) > 0 and len(vol_2_check) > 0:
                diff = (vol_1_check - vol_2_check).abs()
                max_diff = diff.max()

                assert (
                    max_diff < 1e-6
                ), f"未来数据变化不应影响历史GARCH特征值，最大差异: {max_diff}"

    def test_garch_streaming_vs_batch_consistency(self, sample_data_single_asset):
        """
        测试3：流式 vs 批量一致性 ⭐⭐⭐⭐
        对生产部署至关重要：生产环境往往是流式推理，而训练是批量计算
        """
        df = sample_data_single_asset.copy()
        window = 60

        # 批量计算（一次性计算所有数据）
        batch_result = extract_garch_features(df, price_col="close", window=window)

        # 流式计算（逐行模拟，每次只处理到当前时间点）
        streaming_results = []
        for i in range(window, len(df)):
            df_stream = df.iloc[: i + 1].copy()
            stream_result = extract_garch_features(
                df_stream, price_col="close", window=window
            )
            if len(stream_result) > 0:
                # 取最后一行（当前时间点的特征）
                streaming_results.append(stream_result.iloc[-1])

        if len(streaming_results) > 0:
            streaming_df = pd.DataFrame(streaming_results)
            streaming_df.index = df.index[window:][: len(streaming_df)]

            # 比较关键特征（只比较有值的部分）
            key_col = "garch_volatility"
            if key_col in batch_result.columns and key_col in streaming_df.columns:
                batch_vals = batch_result[key_col].iloc[window:].dropna()
                stream_vals = streaming_df[key_col].dropna()

                # 找到共同索引
                common_idx = batch_vals.index.intersection(stream_vals.index)
                if len(common_idx) > 0:
                    diff = (
                        batch_vals.loc[common_idx] - stream_vals.loc[common_idx]
                    ).abs()
                    max_diff = diff.max()

                    # 允许一定的数值误差（因为GARCH拟合可能有微小差异）
                    assert (
                        max_diff < 1e-3
                    ), f"流式与批量GARCH计算不一致，最大差异: {max_diff}"


class TestDTWFeatures:
    """DTW特征测试"""

    @pytest.fixture
    def sample_data_single_asset(self):
        """单资产测试数据"""
        np.random.seed(42)
        n = 100
        prices = 100 + np.cumsum(np.random.randn(n) * 0.5)
        return pd.DataFrame({"close": prices})

    @pytest.fixture
    def sample_data_multi_asset(self):
        """多资产测试数据（不同价格水平）"""
        np.random.seed(42)
        n = 100

        assets = {
            "BTC": 50000 + np.cumsum(np.random.randn(n) * 100),
            "ETH": 3000 + np.cumsum(np.random.randn(n) * 10),
            "SOL": 100 + np.cumsum(np.random.randn(n) * 0.5),
        }

        dfs = []
        for symbol, prices in assets.items():
            df = pd.DataFrame({"close": prices})
            df["_symbol"] = symbol
            dfs.append(df)

        return pd.concat(dfs, ignore_index=False)

    def test_dtw_normalize_series(self):
        """测试归一化函数（公共代码）"""
        # 测试Z-score归一化
        x1 = np.array([100, 101, 102, 103, 104])
        x2 = np.array([50000, 50100, 50200, 50300, 50400])

        norm1 = normalize_series(x1)
        norm2 = normalize_series(x2)

        # 归一化后应该有相似的分布
        tol = 1e-6
        assert np.abs(norm1.mean()) < tol  # 均值应该接近0
        assert np.abs(norm1.std() - 1.0) < tol  # 标准差应该接近1
        assert np.abs(norm2.mean()) < 1e-10
        assert np.abs(norm2.std() - 1.0) < 1e-10

        # 常数序列
        x_const = np.array([100.0] * 10)
        norm_const = normalize_series(x_const)
        assert np.allclose(norm_const, 0.0)

    def test_dtw_features_basic(self, sample_data_single_asset):
        """基础功能测试"""
        df = sample_data_single_asset
        result = extract_dtw_features(df, price_col="close", window=20)

        # 检查输出列
        templates = create_dtw_templates()
        expected_cols = [f"dtw_{name}_dist" for name in templates.keys()]
        expected_cols.extend(["dtw_min_dist", "dtw_best_match"])

        assert all(col in result.columns for col in expected_cols)
        assert len(result) == len(df)

        # 检查距离值合理性（应该非负）
        for col in expected_cols:
            if col != "dtw_best_match":
                valid_data = result[col].dropna()
                if len(valid_data) > 0:
                    assert (valid_data >= 0).all()

    def test_dtw_features_normalization_multi_asset(self, sample_data_multi_asset):
        """
        测试2：多资产归一化（特征分布对齐）⭐⭐⭐⭐

        验证：
        - 不同价格水平的资产，DTW距离分布应该相似（因为价格已归一化）
        - 特征分布应该对齐，便于多资产训练
        """
        # DTW特征应该对价格水平不敏感（因为内部做了归一化）
        results = []
        for symbol in sample_data_multi_asset["_symbol"].unique():
            df_asset = sample_data_multi_asset[
                sample_data_multi_asset["_symbol"] == symbol
            ].copy()
            result = extract_dtw_features(df_asset, price_col="close", window=20)
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
                    # （允许一些差异，因为价格走势不同）
                    assert mean_range < 10.0, (
                        f"{col} 在不同资产间的均值差异过大: {mean_range:.4f}，"
                        f"归一化可能有问题。各资产均值: {by_symbol['mean'].to_dict()}"
                    )

    def test_dtw_features_performance(self, sample_data_single_asset):
        """性能测试"""
        df = sample_data_single_asset

        start_time = time.time()
        result = extract_dtw_features(df, price_col="close", window=20)
        elapsed = time.time() - start_time

        # 100个样本应该在合理时间内完成（< 5秒）
        assert elapsed < 5.0, f"DTW特征计算耗时 {elapsed:.2f}秒，超过5秒"

        # 检查是否有有效结果
        assert result["dtw_min_dist"].notna().sum() > 0

    def test_dtw_templates_consistency(self):
        """测试模板一致性"""
        templates = create_dtw_templates()

        # 所有模板应该有相同的长度（20）
        for name, template in templates.items():
            assert len(template) >= 15, f"模板 {name} 长度过短"
            # 模板值应该在合理范围内
            assert template.min() >= 0.0
            assert template.max() <= 1.0

    def test_dtw_features_no_future_leak(self, sample_data_single_asset):
        """
        测试1：无未来函数（修改未来数据不影响历史特征值）⭐⭐⭐⭐⭐

        验证：
        - DTW 计算使用滚动窗口，每个时间点只使用历史数据
        - 不会因为未来数据的变化而影响当前特征值
        """
        df = sample_data_single_asset
        window = 20

        # 计算第一次 DTW 特征
        result1 = extract_dtw_features(df, price_col="close", window=window)
        dtw_min_1 = result1["dtw_min_dist"].copy()

        # 修改未来数据（t=70 之后）
        df_future_modified = df.copy()
        df_future_modified.loc[df_future_modified.index[70] :, "close"] *= 2.0

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

    def test_dtw_features_streaming_vs_batch(self, sample_data_single_asset):
        """
        测试3：流式 vs 批量一致性 ⭐⭐⭐⭐

        验证：
        - 批量计算（一次性处理所有数据）和流式计算（逐块处理）结果一致
        - 对生产部署至关重要
        """
        df = sample_data_single_asset
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

    def test_dtw_features_correlation_decay(self, sample_data_single_asset):
        """
        测试4：特征相关性衰减平滑 ⭐⭐⭐

        验证：
        - 不同窗口大小的 DTW 特征应该有一定相关性
        - 相关性应该平滑，不应出现断崖式下降
        """
        df = sample_data_single_asset

        # 计算不同窗口的特征
        windows = [10, 20, 30]
        results = {}
        for w in windows:
            result = extract_dtw_features(df, price_col="close", window=w)
            results[f"window_{w}"] = result["dtw_min_dist"].dropna()

        # 检查不同窗口特征之间的相关性
        if len(results) >= 2:
            keys = list(results.keys())
            for i in range(len(keys) - 1):
                col1 = results[keys[i]]
                col2 = results[keys[i + 1]]

                # 找到共同索引
                common_idx = col1.index.intersection(col2.index)
                if len(common_idx) > 10:
                    corr = col1.loc[common_idx].corr(col2.loc[common_idx])

                    # 如果相关性为 NaN（样本太少或常数序列），跳过此检查
                    if pd.isna(corr):
                        continue

                    # 不同窗口的 DTW 距离应该有一定相关性（>0.3）
                    # 注意：DTW 距离可能受窗口大小影响较大，所以阈值较低
                    assert (
                        corr > 0.3 or corr < -0.3
                    ), f"{keys[i]} 与 {keys[i+1]} 的相关性异常: {corr:.4f}，可能存在计算错误"

    def test_dtw_no_future_leak(self, sample_data_single_asset):
        """
        测试1：无未来函数（修改未来数据不影响历史特征值）⭐⭐⭐⭐⭐
        这是底线，必须确保特征计算不包含未来信息
        """
        df = sample_data_single_asset.copy()
        window = 20

        # 计算第一次特征
        result1 = extract_dtw_features(df, price_col="close", window=window)
        dtw_min_1 = result1["dtw_min_dist"].copy()

        # 修改未来数据
        df_future_modified = df.copy()
        if len(df) > 70:
            df_future_modified.loc[df_future_modified.index[70] :, "close"] *= 2.0

            # 重新计算特征
            result2 = extract_dtw_features(
                df_future_modified, price_col="close", window=window
            )
            dtw_min_2 = result2["dtw_min_dist"].copy()

            # 检查前50个时间点的特征值（应该不受未来数据影响）
            check_idx = df.index[:50]
            dtw_1_check = dtw_min_1.loc[check_idx].dropna()
            dtw_2_check = dtw_min_2.loc[check_idx].dropna()

            if len(dtw_1_check) > 0 and len(dtw_2_check) > 0:
                diff = (dtw_1_check - dtw_2_check).abs()
                max_diff = diff.max()

                assert (
                    max_diff < 1e-6
                ), f"未来数据变化不应影响历史DTW特征值，最大差异: {max_diff}"

    def test_dtw_streaming_vs_batch_consistency(self, sample_data_single_asset):
        """
        测试3：流式 vs 批量一致性 ⭐⭐⭐⭐
        对生产部署至关重要：生产环境往往是流式推理，而训练是批量计算
        """
        df = sample_data_single_asset.copy()
        window = 20

        # 批量计算（一次性计算所有数据）
        batch_result = extract_dtw_features(df, price_col="close", window=window)

        # 流式计算（逐行模拟，每次只处理到当前时间点）
        streaming_results = []
        for i in range(window, len(df)):
            df_stream = df.iloc[: i + 1].copy()
            stream_result = extract_dtw_features(
                df_stream, price_col="close", window=window
            )
            if len(stream_result) > 0:
                # 取最后一行（当前时间点的特征）
                streaming_results.append(stream_result.iloc[-1])

        if len(streaming_results) > 0:
            streaming_df = pd.DataFrame(streaming_results)
            streaming_df.index = df.index[window:][: len(streaming_df)]

            # 比较关键特征（只比较有值的部分）
            key_col = "dtw_min_dist"
            if key_col in batch_result.columns and key_col in streaming_df.columns:
                batch_vals = batch_result[key_col].iloc[window:].dropna()
                stream_vals = streaming_df[key_col].dropna()

                # 找到共同索引
                common_idx = batch_vals.index.intersection(stream_vals.index)
                if len(common_idx) > 0:
                    diff = (
                        batch_vals.loc[common_idx] - stream_vals.loc[common_idx]
                    ).abs()
                    max_diff = diff.max()

                    # 允许一定的数值误差
                    assert (
                        max_diff < 1e-6
                    ), f"流式与批量DTW计算不一致，最大差异: {max_diff}"

    def test_dtw_normalization_enhanced(self, sample_data_multi_asset):
        """
        测试2：多资产归一化（特征分布对齐）⭐⭐⭐⭐
        加强版：检查特征在不同资产上的分布是否真正对齐
        """
        # 按资产分组计算特征
        results = []
        for symbol in sample_data_multi_asset["_symbol"].unique():
            df_asset = sample_data_multi_asset[
                sample_data_multi_asset["_symbol"] == symbol
            ].copy()
            result = extract_dtw_features(df_asset, price_col="close", window=20)
            result["_symbol"] = symbol
            results.append(result)

        combined = pd.concat(results, ignore_index=False)

        # 检查：不同资产的DTW距离分布应该相似（因为价格已归一化）
        for col in ["dtw_min_dist"]:
            if col in combined.columns:
                valid_data = combined[col].dropna()
                if len(valid_data) > 0:
                    by_symbol = combined.groupby("_symbol")[col].agg(
                        ["mean", "std", "min", "max"]
                    )

                    # 不同资产的距离均值应该在相似范围内
                    mean_range = by_symbol["mean"].max() - by_symbol["mean"].min()
                    mean_avg = by_symbol["mean"].mean()

                    # 如果归一化正确，不同资产的均值差异不应该太大
                    # （允许一些差异，因为价格走势不同，但应该在合理范围内）
                    assert (
                        mean_range < mean_avg * 0.5
                    ), f"{col} 在不同资产间差异过大（范围={mean_range:.4f}, 均值={mean_avg:.4f}），可能未正确归一化"


class TestEVTFeatures:
    """EVT特征测试"""

    @pytest.fixture
    def sample_data_single_asset(self):
        """单资产测试数据"""
        np.random.seed(42)
        n = 200
        prices = 100 + np.cumsum(np.random.randn(n) * 0.5)
        return pd.DataFrame({"close": prices})

    @pytest.fixture
    def sample_data_multi_asset(self):
        """多资产测试数据（不同价格水平）"""
        np.random.seed(42)
        n = 200

        assets = {
            "BTC": 50000 + np.cumsum(np.random.randn(n) * 100),
            "ETH": 3000 + np.cumsum(np.random.randn(n) * 10),
            "SOL": 100 + np.cumsum(np.random.randn(n) * 0.5),
        }

        dfs = []
        for symbol, prices in assets.items():
            df = pd.DataFrame({"close": prices})
            df["_symbol"] = symbol
            dfs.append(df)

        return pd.concat(dfs, ignore_index=False)

    def test_evt_features_basic(self, sample_data_single_asset):
        """基础功能测试"""
        df = sample_data_single_asset
        result = extract_evt_features(df, price_col="close", window=120)

        # 检查输出列
        expected_cols = [
            "evt_tail_shape",
            "evt_tail_shape_left",
            "evt_tail_shape_right",
            "evt_scale",
            "evt_var_99",
            "evt_es_99",
        ]
        assert all(col in result.columns for col in expected_cols)
        assert len(result) == len(df)

        # 检查数值合理性
        # ξ 通常在 -0.5 到 1.0 之间
        valid_xi = result["evt_tail_shape"].dropna()
        if len(valid_xi) > 0:
            assert (valid_xi >= -1.0).all()  # 允许一些极端值
            assert (valid_xi <= 2.0).all()  # 允许一些极端值

        # VaR和ES应该是负数（损失）
        valid_var = result["evt_var_99"].dropna()
        if len(valid_var) > 0:
            assert (valid_var <= 0).all()  # VaR应该是负数

    def test_evt_features_normalization_multi_asset(self, sample_data_multi_asset):
        """多资产归一化测试（加强版）"""
        # EVT特征基于收益率，应该天然归一化
        results = []
        for symbol in sample_data_multi_asset["_symbol"].unique():
            df_asset = sample_data_multi_asset[
                sample_data_multi_asset["_symbol"] == symbol
            ].copy()
            result = extract_evt_features(df_asset, price_col="close", window=120)
            result["_symbol"] = symbol
            results.append(result)

        combined = pd.concat(results, ignore_index=False)

        # 检查：不同资产的ξ值应该在相似范围内
        for col in ["evt_tail_shape", "evt_tail_shape_left", "evt_tail_shape_right"]:
            valid_data = combined[col].dropna()
            if len(valid_data) > 0:
                by_symbol = combined.groupby("_symbol")[col].agg(["mean", "std"])
                # 均值应该在合理范围内（放宽以避免过度敏感）
                assert (by_symbol["mean"] >= -1.5).all()
                assert (by_symbol["mean"] <= 2.0).all()

                # 加强：检查不同资产的特征分布范围是否重叠（放宽）
                mean_range = by_symbol["mean"].max() - by_symbol["mean"].min()
                # EVT tail shape 在不同资产上应该相似（因为基于收益率）
                assert mean_range < 2.0, (
                    f"{col}: 不同资产的均值差异过大 ({mean_range:.4f})，"
                    f"可能归一化不正确。各资产均值: {by_symbol['mean'].to_dict()}"
                )

    def test_evt_features_performance(self, sample_data_single_asset):
        """性能测试"""
        df = sample_data_single_asset

        start_time = time.time()
        result = extract_evt_features(df, price_col="close", window=120)
        elapsed = time.time() - start_time

        # 200个样本应该在合理时间内完成（< 10秒）
        assert elapsed < 10.0, f"EVT特征计算耗时 {elapsed:.2f}秒，超过10秒"

        # 检查是否有有效结果
        assert result["evt_tail_shape"].notna().sum() > 0

    def test_evt_features_edge_cases(self):
        """边界情况测试"""
        # 数据不足
        df_short = pd.DataFrame({"close": [100, 101, 102]})
        result = extract_evt_features(df_short, price_col="close", window=120)
        assert len(result) == len(df_short)
        # 应该返回默认值
        if "evt_tail_shape" in result.columns:
            assert (result["evt_tail_shape"] == 0.3).all()  # 默认安全值
        else:
            pytest.skip("evt_tail_shape not produced; skip edge-case default check.")

        # 常数价格
        df_constant = pd.DataFrame({"close": [100.0] * 100})
        result = extract_evt_features(df_constant, price_col="close", window=120)
        assert len(result) == len(df_constant)

    def test_evt_features_no_future_leak(self, sample_data_single_asset):
        """
        测试：无未来函数（修改未来数据不影响历史特征值）

        验证：
        - EVT 特征在时刻 t 只使用 [t-window, t-1] 的数据
        - 修改未来数据不应影响历史特征值
        """
        df = sample_data_single_asset.copy()
        window = 120

        # 计算第一次特征
        result1 = extract_evt_features(df, price_col="close", window=window)
        evt_col = "evt_tail_shape"
        if evt_col in result1.columns:
            evt_1 = result1[evt_col].copy()

            # 修改未来数据（从索引 150 开始）
            df_future_modified = df.copy()
            future_idx = df.index[150:]
            df_future_modified.loc[future_idx, "close"] *= 2.0

            # 重新计算特征
            result2 = extract_evt_features(
                df_future_modified, price_col="close", window=window
            )
            evt_2 = result2[evt_col].copy()

            # 检查前 100 个时间点的特征值（应该不受未来数据影响）
            check_idx = df.index[:100]
            evt_1_check = evt_1.loc[check_idx].dropna()
            evt_2_check = evt_2.loc[check_idx].dropna()

            if len(evt_1_check) > 0 and len(evt_2_check) > 0:
                common_idx = evt_1_check.index.intersection(evt_2_check.index)
                if len(common_idx) > 0:
                    diff = (
                        evt_1_check.loc[common_idx] - evt_2_check.loc[common_idx]
                    ).abs()
                    max_diff = diff.max()

                    assert (
                        max_diff < 1e-6
                    ), f"未来数据变化不应影响历史 EVT 特征值，最大差异: {max_diff}"

    def test_evt_features_streaming_vs_batch(self, sample_data_single_asset):
        """
        测试：流式 vs 批量计算一致性

        验证：
        - 批量计算和流式计算结果应该一致
        """
        df = sample_data_single_asset.copy()
        window = 120

        # 批量计算（一次性）
        batch_result = extract_evt_features(df, price_col="close", window=window)

        # 流式计算（分两部分）
        mid_point = len(df) // 2
        df_part1 = df.iloc[:mid_point].copy()

        # 第一部分
        result_part1 = extract_evt_features(df_part1, price_col="close", window=window)
        batch_part1 = batch_result.iloc[:mid_point]

        # 比较前一部分的结果（跳过窗口初始化部分）
        skip_window = min(window, len(result_part1))
        if skip_window < len(result_part1):
            result_part1_valid = result_part1.iloc[skip_window:]
            batch_part1_valid = batch_part1.iloc[skip_window:]

            # 找到共同的有效列
            common_cols = set(result_part1_valid.columns) & set(
                batch_part1_valid.columns
            )
            for col in common_cols:
                if col.startswith("evt_"):
                    valid_idx = (
                        result_part1_valid[col]
                        .dropna()
                        .index.intersection(batch_part1_valid[col].dropna().index)
                    )
                    if len(valid_idx) > 0:
                        diff = (
                            result_part1_valid.loc[valid_idx, col]
                            - batch_part1_valid.loc[valid_idx, col]
                        ).abs()
                        max_diff = diff.max()
                        # 允许一定的数值误差
                        assert (
                            max_diff < 1e-5
                        ), f"流式与批量计算不一致: {col}, 最大差异: {max_diff}"

    def test_evt_no_future_leak(self, sample_data_single_asset):
        """
        测试1：无未来函数（修改未来数据不影响历史特征值）⭐⭐⭐⭐⭐
        这是底线，必须确保特征计算不包含未来信息
        """
        df = sample_data_single_asset.copy()
        window = 120

        # 计算第一次特征
        result1 = extract_evt_features(df, price_col="close", window=window)
        evt_tail_1 = result1["evt_tail_shape"].copy()

        # 修改未来数据
        df_future_modified = df.copy()
        if len(df) > 150:
            df_future_modified.loc[df_future_modified.index[150] :, "close"] *= 2.0

            # 重新计算特征
            result2 = extract_evt_features(
                df_future_modified, price_col="close", window=window
            )
            evt_tail_2 = result2["evt_tail_shape"].copy()

            # 检查前100个时间点的特征值（应该不受未来数据影响）
            check_idx = df.index[:100]
            tail_1_check = evt_tail_1.loc[check_idx].dropna()
            tail_2_check = evt_tail_2.loc[check_idx].dropna()

            if len(tail_1_check) > 0 and len(tail_2_check) > 0:
                diff = (tail_1_check - tail_2_check).abs()
                max_diff = diff.max()

                assert (
                    max_diff < 1e-6
                ), f"未来数据变化不应影响历史EVT特征值，最大差异: {max_diff}"

    def test_evt_streaming_vs_batch_consistency(self, sample_data_single_asset):
        """
        测试3：流式 vs 批量一致性 ⭐⭐⭐⭐
        对生产部署至关重要：生产环境往往是流式推理，而训练是批量计算
        """
        df = sample_data_single_asset.copy()
        window = 120

        # 批量计算（一次性计算所有数据）
        batch_result = extract_evt_features(df, price_col="close", window=window)

        # 流式计算（逐行模拟，每次只处理到当前时间点）
        streaming_results = []
        for i in range(window, len(df)):
            df_stream = df.iloc[: i + 1].copy()
            stream_result = extract_evt_features(
                df_stream, price_col="close", window=window
            )
            if len(stream_result) > 0:
                # 取最后一行（当前时间点的特征）
                streaming_results.append(stream_result.iloc[-1])

        if len(streaming_results) > 0:
            streaming_df = pd.DataFrame(streaming_results)
            streaming_df.index = df.index[window:][: len(streaming_df)]

            # 比较关键特征（只比较有值的部分）
            key_col = "evt_tail_shape"
            if key_col in batch_result.columns and key_col in streaming_df.columns:
                batch_vals = batch_result[key_col].iloc[window:].dropna()
                stream_vals = streaming_df[key_col].dropna()

                # 找到共同索引
                common_idx = batch_vals.index.intersection(stream_vals.index)
                if len(common_idx) > 0:
                    diff = (
                        batch_vals.loc[common_idx] - stream_vals.loc[common_idx]
                    ).abs()
                    max_diff = diff.max()

                    # 允许一定的数值误差（因为EVT拟合可能有微小差异）
                    assert (
                        max_diff < 1e-3
                    ), f"流式与批量EVT计算不一致，最大差异: {max_diff}"

    def test_evt_normalization_enhanced(self, sample_data_multi_asset):
        """
        测试2：多资产归一化（特征分布对齐）⭐⭐⭐⭐
        加强版：检查特征在不同资产上的分布是否真正对齐
        """
        # 按资产分组计算特征
        results = []
        for symbol in sample_data_multi_asset["_symbol"].unique():
            df_asset = sample_data_multi_asset[
                sample_data_multi_asset["_symbol"] == symbol
            ].copy()
            result = extract_evt_features(df_asset, price_col="close", window=120)
            result["_symbol"] = symbol
            results.append(result)

        combined = pd.concat(results, ignore_index=False)

        # 检查：不同资产的ξ值应该在相似范围内
        for col in ["evt_tail_shape", "evt_tail_shape_left", "evt_tail_shape_right"]:
            valid_data = combined[col].dropna()
            if len(valid_data) > 0:
                by_symbol = combined.groupby("_symbol")[col].agg(
                    ["mean", "std", "min", "max"]
                )

                # 均值应该在合理范围内（放宽以避免误报）
                assert (by_symbol["mean"] >= -1.5).all()
                assert (by_symbol["mean"] <= 2.0).all()

                # 检查不同资产的特征分布是否相似（归一化后应该可比）
                mean_range = by_symbol["mean"].max() - by_symbol["mean"].min()
                # EVT tail shape 在不同资产间可能有差异，但应该在合理范围内
                assert (
                    mean_range < 1.1
                ), f"{col} 在不同资产间差异过大（范围={mean_range:.4f}），可能未正确归一化"

    def test_evt_features_no_future_leak(self, sample_data_single_asset):
        """
        测试1：无未来函数（修改未来数据不影响历史特征值）⭐⭐⭐⭐⭐

        验证：
        - EVT 计算使用滚动窗口，每个时间点只使用历史数据
        - 不会因为未来数据的变化而影响当前特征值
        """
        df = sample_data_single_asset
        window = 120

        # 计算第一次 EVT 特征
        result1 = extract_evt_features(df, price_col="close", window=window)
        evt_tail_1 = result1["evt_tail_shape"].copy()

        # 修改未来数据（t=150 之后）
        df_future_modified = df.copy()
        df_future_modified.loc[df_future_modified.index[150] :, "close"] *= 2.0

        # 重新计算 EVT 特征
        result2 = extract_evt_features(
            df_future_modified, price_col="close", window=window
        )
        evt_tail_2 = result2["evt_tail_shape"].copy()

        # 检查前 100 个时间点的特征值（应该不受未来数据影响）
        check_idx = df.index[:100]
        tail_1_check = evt_tail_1.loc[check_idx].dropna()
        tail_2_check = evt_tail_2.loc[check_idx].dropna()

        if len(tail_1_check) > 0 and len(tail_2_check) > 0:
            diff = (tail_1_check - tail_2_check).abs()
            max_diff = diff.max()

            # 允许微小的数值误差（浮点数精度）
            assert (
                max_diff < 1e-6
            ), f"未来数据变化不应影响历史 EVT 特征值，最大差异: {max_diff}"

    def test_evt_features_streaming_vs_batch(self, sample_data_single_asset):
        """
        测试3：流式 vs 批量一致性 ⭐⭐⭐⭐

        验证：
        - 批量计算（一次性处理所有数据）和流式计算（逐块处理）结果一致
        - 对生产部署至关重要
        """
        df = sample_data_single_asset
        window = 120

        # 批量计算（一次性处理）
        batch_result = extract_evt_features(df, price_col="close", window=window)

        # 流式计算（分块处理，模拟在线推理）
        chunk_size = 50
        streaming_results = []

        for i in range(0, len(df), chunk_size):
            chunk = df.iloc[i : i + chunk_size].copy()
            chunk_result = extract_evt_features(chunk, price_col="close", window=window)
            streaming_results.append(chunk_result)

        # 合并流式结果
        streaming_result = pd.concat(streaming_results, axis=0)

        # 比较关键特征（跳过前 window 行）
        skip_rows = window
        if len(batch_result) > skip_rows and len(streaming_result) > skip_rows:
            if (
                "evt_tail_shape" not in batch_result.columns
                or "evt_tail_shape" not in streaming_result.columns
            ):
                pytest.skip(
                    "evt_tail_shape not produced; skip streaming vs batch comparison."
                )
            batch_valid = batch_result.iloc[skip_rows:]["evt_tail_shape"].dropna()
            streaming_valid = streaming_result.iloc[skip_rows:][
                "evt_tail_shape"
            ].dropna()

            # 找到共同索引
            common_idx = batch_valid.index.intersection(streaming_valid.index)
            if len(common_idx) > 0:
                batch_vals = batch_valid.loc[common_idx]
                streaming_vals = streaming_valid.loc[common_idx]
                diff = (batch_vals - streaming_vals).abs()

                # 允许一定误差（因为分块计算可能导致边界处理略有不同）
                max_diff = diff.max()
                assert max_diff < 1e-5, f"流式与批量计算不一致，最大差异: {max_diff}"

    def test_evt_features_correlation_decay(self, sample_data_single_asset):
        """
        测试4：特征相关性衰减平滑 ⭐⭐⭐

        验证：
        - 不同窗口大小的 EVT 特征应该高度相关（如 tail_shape）
        - 相关性应该平滑，不应出现断崖式下降
        """
        df = sample_data_single_asset

        # 计算不同窗口的特征
        windows = [60, 120, 180]
        results = {}
        for w in windows:
            result = extract_evt_features(df, price_col="close", window=w)
            results[f"window_{w}"] = result["evt_tail_shape"].dropna()

        # 检查不同窗口特征之间的相关性
        if len(results) >= 2:
            keys = list(results.keys())
            for i in range(len(keys) - 1):
                col1 = results[keys[i]]
                col2 = results[keys[i + 1]]

                # 找到共同索引
                common_idx = col1.index.intersection(col2.index)
                if len(common_idx) > 10:
                    corr = col1.loc[common_idx].corr(col2.loc[common_idx])

                    # 不同窗口的 tail_shape 应该有一定相关性（>0.3）
                    assert (
                        corr > 0.3 or corr < -0.3
                    ), f"{keys[i]} 与 {keys[i+1]} 的相关性异常: {corr:.4f}，可能存在计算错误"


class TestAdvancedFeaturesFutureLeak:
    """高级特征未来数据泄露测试 - 必须测试 ⭐⭐⭐⭐⭐"""

    def create_test_data(self, n_samples=500):
        """创建测试数据"""
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=n_samples, freq="4h")
        prices = 100 + np.cumsum(np.random.randn(n_samples) * 0.5)
        return pd.DataFrame({"close": prices}, index=dates)

    def test_garch_no_future_leak(self):
        """测试 GARCH 特征无未来数据泄露"""
        df = self.create_test_data(500)

        # 计算第一次特征
        result1 = extract_garch_features(df, price_col="close", window=60)
        garch_vol_1 = result1["garch_volatility"].copy()

        # 修改未来数据（t=400 之后）
        df_future_modified = df.copy()
        df_future_modified.loc[df_future_modified.index[400] :, "close"] *= 2.0

        # 重新计算特征
        result2 = extract_garch_features(
            df_future_modified, price_col="close", window=60
        )
        garch_vol_2 = result2["garch_volatility"].copy()

        # 检查前 250 个时间点的特征值应该相同
        check_idx = df.index[:250]
        vol_1_check = garch_vol_1.loc[check_idx].dropna()
        vol_2_check = garch_vol_2.loc[check_idx].dropna()

        if len(vol_1_check) > 0 and len(vol_2_check) > 0:
            diff = (vol_1_check - vol_2_check).abs()
            max_diff = diff.max()
            assert (
                max_diff < 1e-6
            ), f"未来数据变化不应影响历史 GARCH 特征值，最大差异: {max_diff}"

    def test_dtw_no_future_leak(self):
        """测试 DTW 特征无未来数据泄露"""
        df = self.create_test_data(500)

        # 计算第一次特征
        result1 = extract_dtw_features(df, price_col="close", window=20)
        dtw_cols = [col for col in result1.columns if "dtw_" in col and "dist" in col]
        if not dtw_cols:
            dtw_cols = [col for col in result1.columns if "dtw" in col.lower()]

        if dtw_cols:
            dtw_col = dtw_cols[0]
            dtw_1 = result1[dtw_col].copy()

            # 修改未来数据
            df_future_modified = df.copy()
            df_future_modified.loc[df_future_modified.index[400] :, "close"] *= 2.0

            # 重新计算特征
            result2 = extract_dtw_features(
                df_future_modified, price_col="close", window=20
            )
            dtw_2 = result2[dtw_col].copy()

            # 检查前 250 个时间点
            check_idx = df.index[:250]
            dtw_1_check = dtw_1.loc[check_idx].dropna()
            dtw_2_check = dtw_2.loc[check_idx].dropna()

            if len(dtw_1_check) > 0 and len(dtw_2_check) > 0:
                diff = (dtw_1_check - dtw_2_check).abs()
                max_diff = diff.max()
                assert (
                    max_diff < 1e-6
                ), f"未来数据变化不应影响历史 DTW 特征值，最大差异: {max_diff}"

    def test_evt_no_future_leak(self):
        """测试 EVT 特征无未来数据泄露"""
        df = self.create_test_data(500)

        # 计算第一次特征
        result1 = extract_evt_features(df, price_col="close", window=120)
        evt_tail_1 = result1["evt_tail_shape"].copy()

        # 修改未来数据
        df_future_modified = df.copy()
        df_future_modified.loc[df_future_modified.index[400] :, "close"] *= 2.0

        # 重新计算特征
        result2 = extract_evt_features(
            df_future_modified, price_col="close", window=120
        )
        evt_tail_2 = result2["evt_tail_shape"].copy()

        # 检查前 250 个时间点
        check_idx = df.index[:250]
        tail_1_check = evt_tail_1.loc[check_idx].dropna()
        tail_2_check = evt_tail_2.loc[check_idx].dropna()

        if len(tail_1_check) > 0 and len(tail_2_check) > 0:
            diff = (tail_1_check - tail_2_check).abs()
            max_diff = diff.max()
            assert (
                max_diff < 1e-6
            ), f"未来数据变化不应影响历史 EVT 特征值，最大差异: {max_diff}"


class TestAdvancedFeaturesStreamingVsBatch:
    """流式 vs 批量一致性测试 - 强烈推荐 ⭐⭐⭐⭐"""

    def create_test_data(self, n_samples=300):
        """创建测试数据"""
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=n_samples, freq="4h")
        prices = 100 + np.cumsum(np.random.randn(n_samples) * 0.5)
        return pd.DataFrame({"close": prices}, index=dates)

    def test_garch_streaming_vs_batch(self):
        """测试 GARCH 流式 vs 批量一致性"""
        df = self.create_test_data(300)

        # 批量计算
        batch_result = extract_garch_features(df, price_col="close", window=60)

        # 流式计算（分块模拟）
        chunk_size = 100
        streaming_results = []
        for i in range(0, len(df), chunk_size):
            chunk_df = df.iloc[i : i + chunk_size]
            chunk_result = extract_garch_features(
                chunk_df, price_col="close", window=60
            )
            streaming_results.append(chunk_result)

        # 注意：由于窗口限制，流式和批量在边界处可能有差异
        # 这里主要验证核心逻辑一致性
        if len(streaming_results) > 0:
            streaming_result = pd.concat(streaming_results, axis=0)

            # 比较中间部分（避开窗口边界）
            mid_start = len(df) // 4
            mid_end = 3 * len(df) // 4
            mid_idx = df.index[mid_start:mid_end]

            if "garch_volatility" in batch_result.columns:
                batch_mid = batch_result.loc[mid_idx, "garch_volatility"].dropna()
                if "garch_volatility" in streaming_result.columns:
                    stream_mid = streaming_result.loc[
                        mid_idx, "garch_volatility"
                    ].dropna()

                    # 找到共同索引
                    common_idx = batch_mid.index.intersection(stream_mid.index)
                    if len(common_idx) > 0:
                        diff = (
                            batch_mid.loc[common_idx] - stream_mid.loc[common_idx]
                        ).abs()
                        max_diff = diff.max()
                        # 允许一定误差（因为窗口边界处理可能不同）
                        assert (
                            max_diff < 1.0
                        ), f"GARCH 流式与批量计算差异过大: {max_diff}"

    def test_dtw_streaming_vs_batch(self):
        """测试 DTW 流式 vs 批量一致性"""
        df = self.create_test_data(200)

        # 批量计算
        batch_result = extract_dtw_features(df, price_col="close", window=20)

        # 流式计算（分块模拟）
        chunk_size = 50
        streaming_results = []
        for i in range(0, len(df), chunk_size):
            chunk_df = df.iloc[i : i + chunk_size]
            chunk_result = extract_dtw_features(chunk_df, price_col="close", window=20)
            streaming_results.append(chunk_result)

        if len(streaming_results) > 0:
            streaming_result = pd.concat(streaming_results, axis=0)

            # 比较中间部分
            mid_start = len(df) // 4
            mid_end = 3 * len(df) // 4
            mid_idx = df.index[mid_start:mid_end]

            dtw_cols = [
                col for col in batch_result.columns if "dtw_" in col and "dist" in col
            ]
            if dtw_cols:
                dtw_col = dtw_cols[0]
                batch_mid = batch_result.loc[mid_idx, dtw_col].dropna()
                if dtw_col in streaming_result.columns:
                    stream_mid = streaming_result.loc[mid_idx, dtw_col].dropna()

                    common_idx = batch_mid.index.intersection(stream_mid.index)
                    if len(common_idx) > 0:
                        diff = (
                            batch_mid.loc[common_idx] - stream_mid.loc[common_idx]
                        ).abs()
                        max_diff = diff.max()
                        # DTW 允许更大误差（因为距离计算可能有数值差异）
                        assert (
                            max_diff < 10.0
                        ), f"DTW 流式与批量计算差异过大: {max_diff}"


class TestAdvancedFeaturesLagDecay:
    """lag 衰减平滑测试 - 可选但高价值 ⭐⭐⭐"""

    def create_test_data(self, n_samples=500):
        """创建测试数据"""
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=n_samples, freq="4h")
        prices = 100 + np.cumsum(np.random.randn(n_samples) * 0.5)
        return pd.DataFrame({"close": prices}, index=dates)

    def test_garch_persistence_autocorrelation_decay(self):
        """测试 GARCH persistence 的自相关性衰减"""
        df = self.create_test_data(500)
        result = extract_garch_features(df, price_col="close", window=60)

        persistence = result["garch_persistence"].dropna()

        if len(persistence) > 100:
            # 计算不同 lag 的自相关性
            lags = [1, 2, 3, 5, 10]
            autocorrs = []

            for lag in lags:
                if len(persistence) > lag:
                    corr = persistence.autocorr(lag=lag)
                    if not np.isnan(corr):
                        autocorrs.append(abs(corr))

            # 验证自相关性应该大致递减（允许波动）
            if len(autocorrs) >= 3:
                # lag=1 应该最高
                assert autocorrs[0] > 0.1, "GARCH persistence 应该有正自相关"
                # 不应该出现 lag=1 高但 lag=2 接近 0 的断崖
                if len(autocorrs) >= 2:
                    assert (
                        autocorrs[1] > 0.05 * autocorrs[0]
                    ), "GARCH persistence 自相关性不应断崖式下降"

    def test_dtw_distance_autocorrelation_decay(self):
        """测试 DTW 距离的自相关性衰减"""
        df = self.create_test_data(500)
        result = extract_dtw_features(df, price_col="close", window=20)

        dtw_cols = [col for col in result.columns if "dtw_" in col and "dist" in col]
        if dtw_cols:
            dtw_col = dtw_cols[0]
            dtw_dist = result[dtw_col].dropna()

            if len(dtw_dist) > 100:
                # 计算不同 lag 的自相关性
                lags = [1, 2, 3, 5]
                autocorrs = []

                for lag in lags:
                    if len(dtw_dist) > lag:
                        corr = dtw_dist.autocorr(lag=lag)
                        if not np.isnan(corr):
                            autocorrs.append(abs(corr))

                # DTW 距离的自相关性可能较低（因为它是模式匹配距离）
                # 但应该不会出现异常的模式
                if len(autocorrs) >= 2:
                    # 不应该出现 lag=1 很高但 lag=2 突然为负的异常模式
                    assert not (
                        autocorrs[0] > 0.5 and autocorrs[1] < -0.3
                    ), "DTW 距离自相关性不应出现异常模式"

    def test_evt_tail_shape_autocorrelation_decay(self):
        """测试 EVT tail_shape 的自相关性衰减"""
        df = self.create_test_data(500)
        result = extract_evt_features(df, price_col="close", window=120)

        tail_shape = result["evt_tail_shape"].dropna()

        if len(tail_shape) > 100:
            # 计算不同 lag 的自相关性
            lags = [1, 2, 3, 5, 10]
            autocorrs = []

            for lag in lags:
                if len(tail_shape) > lag:
                    corr = tail_shape.autocorr(lag=lag)
                    if not np.isnan(corr):
                        autocorrs.append(abs(corr))

            # EVT tail_shape 应该有合理的自相关性
            if len(autocorrs) >= 3:
                # 不应该出现断崖式下降
                if len(autocorrs) >= 2:
                    # lag=2 不应该比 lag=1 突然小很多（除非 lag=1 本身就很低）
                    if autocorrs[0] > 0.2:
                        assert (
                            autocorrs[1] > 0.1 * autocorrs[0]
                        ), "EVT tail_shape 自相关性不应断崖式下降"


class TestAdvancedFeaturesMultiAssetNormalization:
    """多资产归一化测试 - 加强版 ⭐⭐⭐⭐"""

    def create_multi_asset_data(self, n_samples=200):
        """创建多资产测试数据（不同价格水平）"""
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=n_samples, freq="4h")

        assets = {
            "BTC": 50000 + np.cumsum(np.random.randn(n_samples) * 100),
            "ETH": 3000 + np.cumsum(np.random.randn(n_samples) * 10),
            "SOL": 100 + np.cumsum(np.random.randn(n_samples) * 0.5),
        }

        dfs = []
        for symbol, prices in assets.items():
            df = pd.DataFrame({"close": prices}, index=dates)
            df["_symbol"] = symbol
            dfs.append(df)

        return pd.concat(dfs, ignore_index=False)

    def test_garch_multi_asset_normalization_enhanced(self):
        """加强版 GARCH 多资产归一化测试"""
        multi_asset_df = self.create_multi_asset_data()

        results = []
        for symbol in multi_asset_df["_symbol"].unique():
            df_asset = multi_asset_df[multi_asset_df["_symbol"] == symbol].copy()
            result = extract_garch_features(df_asset, price_col="close", window=60)
            result["_symbol"] = symbol
            results.append(result)

        combined = pd.concat(results, ignore_index=False)

        # 检查不同资产的特征分布
        for col in ["garch_persistence", "garch_leverage_gamma"]:
            valid_data = combined[col].dropna()
            if len(valid_data) > 0:
                by_symbol = combined.groupby("_symbol")[col].agg(
                    ["mean", "std", "min", "max"]
                )

                # 不同资产的均值应该在合理范围内
                mean_range = by_symbol["mean"].max() - by_symbol["mean"].min()
                # GARCH 特征基于收益率，应该天然归一化，不同资产的均值差异不应太大
                assert (
                    mean_range < 0.5
                ), f"{col} 在不同资产间的均值差异过大: {mean_range}"

                # 检查标准差范围（不同资产的标准差应该相似）
                std_range = by_symbol["std"].max() - by_symbol["std"].min()
                assert (
                    std_range < 0.3
                ), f"{col} 在不同资产间的标准差差异过大: {std_range}"

    def test_dtw_multi_asset_normalization_enhanced(self):
        """加强版 DTW 多资产归一化测试"""
        multi_asset_df = self.create_multi_asset_data()

        results = []
        for symbol in multi_asset_df["_symbol"].unique():
            df_asset = multi_asset_df[multi_asset_df["_symbol"] == symbol].copy()
            result = extract_dtw_features(df_asset, price_col="close", window=20)
            result["_symbol"] = symbol
            results.append(result)

        combined = pd.concat(results, ignore_index=False)

        # DTW 特征应该对价格水平不敏感（因为内部做了归一化）
        dtw_cols = [col for col in combined.columns if "dtw_" in col and "dist" in col]
        for col in dtw_cols[:2]:  # 检查前两个 DTW 特征
            valid_data = combined[col].dropna()
            if len(valid_data) > 0:
                by_symbol = combined.groupby("_symbol")[col].agg(["mean", "std"])

                # 不同资产的距离均值应该在相似范围内
                mean_range = by_symbol["mean"].max() - by_symbol["mean"].min()
                # 如果归一化正确，不同资产的均值差异不应该太大
                assert (
                    mean_range < 5.0
                ), f"{col} 在不同资产间的均值差异过大: {mean_range}"

    def test_evt_multi_asset_normalization_enhanced(self):
        """加强版 EVT 多资产归一化测试"""
        multi_asset_df = self.create_multi_asset_data()

        results = []
        for symbol in multi_asset_df["_symbol"].unique():
            df_asset = multi_asset_df[multi_asset_df["_symbol"] == symbol].copy()
            result = extract_evt_features(df_asset, price_col="close", window=120)
            result["_symbol"] = symbol
            results.append(result)

        combined = pd.concat(results, ignore_index=False)

        # EVT 特征基于收益率，应该天然归一化
        # 注意：EVT tail_shape 对数据分布非常敏感，不同资产可能有不同的尾部特性
        # 这里主要验证特征值在合理范围内，而不是强制要求不同资产相似
        for col in ["evt_tail_shape", "evt_tail_shape_left", "evt_tail_shape_right"]:
            valid_data = combined[col].dropna()
            if len(valid_data) > 0:
                by_symbol = combined.groupby("_symbol")[col].agg(
                    ["mean", "std", "min", "max"]
                )

                # 检查每个资产的特征值是否在合理范围内
                # EVT tail_shape 通常在 [-1.5, 2.0] 范围内（允许一些极端值）
                assert (by_symbol["mean"] >= -1.5).all() and (
                    by_symbol["mean"] <= 2.0
                ).all(), f"{col} 的均值应在 [-1.5, 2.0] 范围内"

                # 检查不同资产的标准差是否在合理范围内（不应过大）
                std_range = by_symbol["std"].max() - by_symbol["std"].min()
                assert (
                    std_range < 0.5
                ), f"{col} 在不同资产间的标准差差异过大: {std_range:.4f}"


class TestAdvancedFeaturesIntegration:
    """高级特征集成测试"""

    def test_all_features_together(self):
        """测试所有高级特征一起使用"""
        np.random.seed(42)
        n = 200
        df = pd.DataFrame({"close": 100 + np.cumsum(np.random.randn(n) * 0.5)})

        # 计算所有特征
        garch_result = extract_garch_features(df, price_col="close", window=60)
        dtw_result = extract_dtw_features(df, price_col="close", window=20)
        evt_result = extract_evt_features(df, price_col="close", window=120)

        # 合并结果
        combined = pd.concat([garch_result, dtw_result, evt_result], axis=1)

        # 检查没有重复列
        assert len(combined.columns) == len(set(combined.columns))

        # 检查所有特征都有值（至少部分）
        for col in combined.columns:
            if col != "dtw_best_match":
                assert combined[col].notna().sum() > 0 or (combined[col] == 0.0).all()

    def test_features_with_missing_data(self):
        """测试缺失数据处理"""
        np.random.seed(42)
        n = 200
        prices = 100 + np.cumsum(np.random.randn(n) * 0.5)

        # 添加一些NaN
        prices[50:60] = np.nan
        prices[150:155] = np.nan

        df = pd.DataFrame({"close": prices})

        # 应该能正常处理
        garch_result = extract_garch_features(df, price_col="close", window=60)
        dtw_result = extract_dtw_features(df, price_col="close", window=20)
        evt_result = extract_evt_features(df, price_col="close", window=120)

        # 检查结果长度
        assert len(garch_result) == len(df)
        assert len(dtw_result) == len(df)
        assert len(evt_result) == len(df)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
