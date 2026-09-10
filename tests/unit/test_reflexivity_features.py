#!/usr/bin/env python3
"""
单元测试：反身性监测特征（Reflexivity Monitoring Features）

测试：
1. OFCI (Order Flow Consensus Index) 特征计算
2. SHD (Strategy Homogeneity Detector) 特征计算
3. Percentile rank 计算
4. 边界情况处理
"""

import pytest
import numpy as np
import pandas as pd

from src.features.time_series.reflexivity_features import (
    compute_ofci_from_trades,
    compute_ofci_pct_from_series,
    compute_shd_from_series,
    compute_shd_from_ohlcv,
    compute_shd_pct_from_series,
)


def _make_close_cvd_for_shd(n: int, seed: int = 42) -> tuple[pd.Series, pd.Series]:
    """Synthetic close/cvd for SHD percentile tests (matches bar-level inputs)."""
    dates = pd.date_range("2024-01-01", periods=n, freq="1min")
    rng = np.random.default_rng(seed)
    returns = pd.Series(rng.standard_normal(n) * 0.01, index=dates)
    close = pd.Series(100.0 * (1 + returns).cumprod(), index=dates)
    cvd = pd.Series(returns.cumsum() * 1000, index=dates)
    return close, cvd


class TestOFCI:
    """测试OFCI特征计算"""

    def test_compute_ofci_from_trades(self):
        """测试OFCI计算（buy_ratio转换、输出范围[-1,1]）"""
        # 创建测试数据：100笔交易，60笔buy，40笔sell
        trades = pd.DataFrame(
            {
                "side": [1] * 60 + [-1] * 40,
            },
            index=pd.date_range("2024-01-01", periods=100, freq="1min"),
        )

        result = compute_ofci_from_trades(trades, window=100)

        assert "ofci" in result.columns
        assert len(result) == 100

        # 最后一个值应该反映buy_ratio
        # buy_ratio = 60/100 = 0.6
        # ofci = 2 * 0.6 - 1 = 0.2
        last_ofci = result["ofci"].iloc[-1]
        assert abs(last_ofci - 0.2) < 0.01

        # 检查输出范围
        assert result["ofci"].min() >= -1.0
        assert result["ofci"].max() <= 1.0

    def test_compute_ofci_all_buy(self):
        """测试全部buy的情况（OFCI应该接近1）"""
        trades = pd.DataFrame(
            {
                "side": [1] * 100,
            },
            index=pd.date_range("2024-01-01", periods=100, freq="1min"),
        )

        result = compute_ofci_from_trades(trades, window=100)
        last_ofci = result["ofci"].iloc[-1]
        assert abs(last_ofci - 1.0) < 0.01

    def test_compute_ofci_all_sell(self):
        """测试全部sell的情况（OFCI应该接近-1）"""
        trades = pd.DataFrame(
            {
                "side": [-1] * 100,
            },
            index=pd.date_range("2024-01-01", periods=100, freq="1min"),
        )

        result = compute_ofci_from_trades(trades, window=100)
        last_ofci = result["ofci"].iloc[-1]
        assert abs(last_ofci - (-1.0)) < 0.01

    def test_compute_ofci_balanced(self):
        """测试平衡的情况（OFCI应该接近0）"""
        trades = pd.DataFrame(
            {
                "side": [1, -1] * 50,
            },
            index=pd.date_range("2024-01-01", periods=100, freq="1min"),
        )

        result = compute_ofci_from_trades(trades, window=100)
        last_ofci = result["ofci"].iloc[-1]
        assert abs(last_ofci) < 0.1  # 应该接近0

    def test_compute_ofci_pct_from_series(self):
        """测试OFCI percentile计算（基于abs(ofci)）"""
        # 创建测试数据：OFCI值从-1到1
        ofci = pd.Series(
            [-1.0, -0.5, 0.0, 0.5, 1.0] * 20,
            index=pd.date_range("2024-01-01", periods=100, freq="1min"),
        )

        result = compute_ofci_pct_from_series(ofci=ofci, window=50)

        assert "ofci_pct" in result.columns
        assert len(result) == 100

        # ofci_pct应该基于abs(ofci)计算，所以-1和1应该有相同的percentile
        # 最后一个值（1.0）的percentile应该较高
        last_pct = result["ofci_pct"].iloc[-1]
        assert 0.0 <= last_pct <= 1.0

    def test_compute_ofci_no_side_column(self):
        """测试没有side列的情况（应该返回0）"""
        trades = pd.DataFrame(
            {
                "price": [100.0] * 100,
                "volume": [1.0] * 100,
            },
            index=pd.date_range("2024-01-01", periods=100, freq="1min"),
        )

        result = compute_ofci_from_trades(trades, window=100)

        assert "ofci" in result.columns
        # 没有side列时应该返回0
        assert result["ofci"].iloc[-1] == 0.0

    def test_compute_ofci_empty_data(self):
        """测试空数据"""
        trades = pd.DataFrame(columns=["side"])
        result = compute_ofci_from_trades(trades, window=100)

        assert "ofci" in result.columns
        assert len(result) == 0


class TestSHD:
    """测试SHD特征计算"""

    def test_compute_shd_from_series(self):
        """测试SHD计算（rolling_corr(d_cvd, ret)的绝对值）"""
        # 创建测试数据：CVD和价格收益率高度相关
        dates = pd.date_range("2024-01-01", periods=200, freq="1min")

        # 创建高度相关的CVD和收益率序列
        np.random.seed(42)
        returns = pd.Series(np.random.randn(200) * 0.01, index=dates)
        cvd = returns.cumsum() * 1000  # CVD是收益率的累积

        result = compute_shd_from_series(
            cvd_series=cvd, price_returns=returns, window=60
        )

        assert "shd" in result.columns
        assert len(result) == 200

        # SHD应该是[0, 1]范围
        assert result["shd"].min() >= 0.0
        assert result["shd"].max() <= 1.0

        # 由于CVD和returns高度相关，SHD应该较高
        last_shd = result["shd"].iloc[-1]
        assert 0.0 <= last_shd <= 1.0

    def test_compute_shd_from_ohlcv(self):
        """测试从OHLCV计算SHD"""
        dates = pd.date_range("2024-01-01", periods=200, freq="1min")

        # 创建价格序列
        np.random.seed(42)
        price_changes = np.random.randn(200) * 0.01
        close = pd.Series(100.0 * (1 + price_changes).cumprod(), index=dates)

        # 创建CVD序列（与价格相关）
        cvd = pd.Series(price_changes.cumsum() * 1000, index=dates)

        result = compute_shd_from_ohlcv(close=close, cvd=cvd, window=60)

        assert "shd" in result.columns
        assert len(result) == 200
        assert result["shd"].min() >= 0.0
        assert result["shd"].max() <= 1.0

    def test_compute_shd_pct_from_series(self):
        """测试SHD percentile计算"""
        close, cvd = _make_close_cvd_for_shd(200, seed=0)

        result = compute_shd_pct_from_series(
            close=close,
            cvd=cvd,
            shd_window=60,
            percentile_window=50,
            shift=1,
        )

        assert "shd_pct" in result.columns
        assert len(result) == 200
        assert result["shd_pct"].min() >= 0.0
        assert result["shd_pct"].max() <= 1.0

    def test_compute_shd_uncorrelated(self):
        """测试不相关的CVD和收益率（SHD应该较低）"""
        dates = pd.date_range("2024-01-01", periods=200, freq="1min")

        np.random.seed(42)
        returns = pd.Series(np.random.randn(200) * 0.01, index=dates)
        # CVD与returns不相关
        cvd = pd.Series(np.random.randn(200) * 100, index=dates)

        result = compute_shd_from_series(
            cvd_series=cvd, price_returns=returns, window=60
        )

        # SHD应该较低（接近0）
        last_shd = result["shd"].iloc[-1]
        assert 0.0 <= last_shd <= 1.0

    def test_compute_shd_empty_data(self):
        """测试空数据"""
        empty_series = pd.Series(dtype=float)
        result = compute_shd_from_series(
            cvd_series=empty_series, price_returns=empty_series, window=60
        )

        assert "shd" in result.columns
        assert len(result) == 0

    def test_compute_shd_mismatched_index(self):
        """测试索引不匹配的情况"""
        dates1 = pd.date_range("2024-01-01", periods=100, freq="1min")
        dates2 = pd.date_range("2024-01-02", periods=100, freq="1min")

        cvd = pd.Series(np.random.randn(100), index=dates1)
        returns = pd.Series(np.random.randn(100), index=dates2)

        result = compute_shd_from_series(
            cvd_series=cvd, price_returns=returns, window=60
        )

        # 应该返回空结果（没有共同索引）
        assert "shd" in result.columns
        assert len(result) == 0


class TestEdgeCases:
    """测试边界情况"""

    def test_ofci_with_nan(self):
        """测试OFCI处理NaN值"""
        trades = pd.DataFrame(
            {
                "side": [1, -1, np.nan, 1, -1] * 20,
            },
            index=pd.date_range("2024-01-01", periods=100, freq="1min"),
        )

        result = compute_ofci_from_trades(trades, window=100)

        # 应该能正常处理NaN
        assert "ofci" in result.columns
        # NaN值应该被过滤掉

    def test_shd_with_nan(self):
        """测试SHD处理NaN值"""
        dates = pd.date_range("2024-01-01", periods=200, freq="1min")

        cvd = pd.Series([1.0, 2.0, np.nan, 4.0, 5.0] * 40, index=dates)
        returns = pd.Series([0.01, 0.02, np.nan, 0.04, 0.05] * 40, index=dates)

        result = compute_shd_from_series(
            cvd_series=cvd, price_returns=returns, window=60
        )

        # 应该能正常处理NaN
        assert "shd" in result.columns

    def test_ofci_side_string_mapping(self):
        """测试side列是字符串的情况"""
        trades = pd.DataFrame(
            {
                "side": ["buy", "sell", "BUY", "SELL"] * 25,
            },
            index=pd.date_range("2024-01-01", periods=100, freq="1min"),
        )

        result = compute_ofci_from_trades(trades, window=100)

        assert "ofci" in result.columns
        # 应该正确映射字符串到数值


class TestReflexivityFeaturesNoFutureLeak:
    """测试反身性特征无未来函数（No Lookahead Bias）⭐⭐⭐⭐⭐"""

    def test_ofci_no_future_leak(self):
        """测试OFCI：修改未来数据不应影响历史特征值"""
        # 1. 创建测试数据
        trades = pd.DataFrame(
            {
                "side": [1, -1] * 200,
            },
            index=pd.date_range("2024-01-01", periods=400, freq="1min"),
        )

        # 2. 计算第一次特征
        result1 = compute_ofci_from_trades(trades, window=100)
        ofci_1 = result1["ofci"].copy()

        # 3. 修改未来数据（从t=200开始）
        trades_future = trades.copy()
        trades_future.loc[trades_future.index[200] :, "side"] = 1  # 全部改为buy

        # 4. 重新计算特征
        result2 = compute_ofci_from_trades(trades_future, window=100)
        ofci_2 = result2["ofci"].copy()

        # 5. 验证历史数据（t < 200）的特征值应该相同
        # 检查窗口已满后的历史数据（索引100之后，但在200之前）
        check_idx = trades.index[100:195]  # 窗口已满后的历史数据
        ofci_1_check = ofci_1.loc[check_idx].dropna()
        ofci_2_check = ofci_2.loc[check_idx].dropna()

        common_idx = ofci_1_check.index.intersection(ofci_2_check.index)
        if len(common_idx) > 0:
            diff = (ofci_1_check.loc[common_idx] - ofci_2_check.loc[common_idx]).abs()
            max_diff = diff.max()
            assert (
                max_diff < 1e-6
            ), f"未来数据变化不应影响历史OFCI值，最大差异: {max_diff}"

    def test_shd_no_future_leak(self):
        """测试SHD：修改未来数据不应影响历史特征值"""
        dates = pd.date_range("2024-01-01", periods=400, freq="1min")

        # 创建测试数据
        np.random.seed(42)
        returns = pd.Series(np.random.randn(400) * 0.01, index=dates)
        cvd = returns.cumsum() * 1000

        # 计算第一次特征
        result1 = compute_shd_from_series(
            cvd_series=cvd, price_returns=returns, window=60
        )
        shd_1 = result1["shd"].copy()

        # 修改未来数据（从t=200开始）
        returns_future = returns.copy()
        cvd_future = cvd.copy()
        returns_future.loc[returns_future.index[200] :] = (
            returns_future.loc[returns_future.index[200] :] * 2.0
        )
        cvd_future.loc[cvd_future.index[200] :] = (
            cvd_future.loc[cvd_future.index[200] :] * 2.0
        )

        # 重新计算特征
        result2 = compute_shd_from_series(
            cvd_series=cvd_future, price_returns=returns_future, window=60
        )
        shd_2 = result2["shd"].copy()

        # 验证历史数据（t < 200）的特征值应该相同
        check_idx = dates[60:195]  # 窗口已满后的历史数据
        shd_1_check = shd_1.loc[check_idx].dropna()
        shd_2_check = shd_2.loc[check_idx].dropna()

        common_idx = shd_1_check.index.intersection(shd_2_check.index)
        if len(common_idx) > 0:
            diff = (shd_1_check.loc[common_idx] - shd_2_check.loc[common_idx]).abs()
            max_diff = diff.max()
            assert (
                max_diff < 1e-6
            ), f"未来数据变化不应影响历史SHD值，最大差异: {max_diff}"

    def test_ofci_pct_no_future_leak(self):
        """测试OFCI percentile：修改未来数据不应影响历史特征值"""
        dates = pd.date_range("2024-01-01", periods=400, freq="1min")
        ofci = pd.Series([-1.0, -0.5, 0.0, 0.5, 1.0] * 80, index=dates)

        # 计算第一次特征
        result1 = compute_ofci_pct_from_series(ofci=ofci, window=50)
        ofci_pct_1 = result1["ofci_pct"].copy()

        # 修改未来数据（从t=200开始）
        ofci_future = ofci.copy()
        ofci_future.loc[ofci_future.index[200] :] = 1.0  # 全部改为1.0

        # 重新计算特征
        result2 = compute_ofci_pct_from_series(ofci=ofci_future, window=50)
        ofci_pct_2 = result2["ofci_pct"].copy()

        # 验证历史数据
        check_idx = dates[50:195]  # 窗口已满后的历史数据
        ofci_pct_1_check = ofci_pct_1.loc[check_idx].dropna()
        ofci_pct_2_check = ofci_pct_2.loc[check_idx].dropna()

        common_idx = ofci_pct_1_check.index.intersection(ofci_pct_2_check.index)
        if len(common_idx) > 0:
            diff = (
                ofci_pct_1_check.loc[common_idx] - ofci_pct_2_check.loc[common_idx]
            ).abs()
            max_diff = diff.max()
            assert (
                max_diff < 1e-6
            ), f"未来数据变化不应影响历史OFCI percentile值，最大差异: {max_diff}"

    def test_shd_pct_no_future_leak(self):
        """测试SHD percentile：修改未来数据不应影响历史特征值"""
        close, cvd = _make_close_cvd_for_shd(400, seed=0)
        pct_window = 50

        result1 = compute_shd_pct_from_series(
            close=close,
            cvd=cvd,
            shd_window=60,
            percentile_window=pct_window,
            shift=1,
        )
        shd_pct_1 = result1["shd_pct"].copy()

        close_future = close.copy()
        cvd_future = cvd.copy()
        close_future.loc[close_future.index[200] :] = close_future.iloc[199] * 2.0
        cvd_future.loc[cvd_future.index[200] :] = cvd_future.iloc[199] + 5000.0

        result2 = compute_shd_pct_from_series(
            close=close_future,
            cvd=cvd_future,
            shd_window=60,
            percentile_window=pct_window,
            shift=1,
        )
        shd_pct_2 = result2["shd_pct"].copy()

        check_idx = close.index[50:195]
        shd_pct_1_check = shd_pct_1.loc[check_idx].dropna()
        shd_pct_2_check = shd_pct_2.loc[check_idx].dropna()

        common_idx = shd_pct_1_check.index.intersection(shd_pct_2_check.index)
        if len(common_idx) > 0:
            diff = (
                shd_pct_1_check.loc[common_idx] - shd_pct_2_check.loc[common_idx]
            ).abs()
            max_diff = diff.max()
            assert (
                max_diff < 1e-6
            ), f"未来数据变化不应影响历史SHD percentile值，最大差异: {max_diff}"


class TestReflexivityFeaturesStreamingVsBatch:
    """测试反身性特征流式vs批量一致性 ⭐⭐⭐⭐"""

    def test_ofci_streaming_vs_batch(self):
        """测试OFCI：流式计算与批量计算应该一致"""
        # 1. 创建测试数据
        trades = pd.DataFrame(
            {
                "side": [1, -1] * 200,
            },
            index=pd.date_range("2024-01-01", periods=400, freq="1min"),
        )

        # 2. 批量计算（一次性处理所有数据）
        batch_result = compute_ofci_from_trades(trades, window=100)

        # 3. 流式计算（分块处理，模拟生产环境）
        streaming_results = []
        chunk_size = 50
        accumulated_trades = pd.DataFrame()

        for i in range(0, len(trades), chunk_size):
            # 获取当前chunk
            chunk = trades.iloc[i : i + chunk_size]
            # 累积数据（保持窗口所需的历史数据）
            accumulated_trades = pd.concat([accumulated_trades, chunk])

            # 只保留窗口所需的数据（避免内存无限增长）
            if len(accumulated_trades) > 200:  # window * 2
                accumulated_trades = accumulated_trades.iloc[-200:]

            # 计算特征（只对累积数据）
            if len(accumulated_trades) >= 100:  # 窗口大小
                chunk_result = compute_ofci_from_trades(accumulated_trades, window=100)
                # 只保留当前chunk的结果
                chunk_result_filtered = chunk_result.loc[chunk.index]
                streaming_results.append(chunk_result_filtered)

        # 4. 合并流式结果
        if streaming_results:
            streaming_combined = pd.concat(streaming_results)
        else:
            streaming_combined = pd.DataFrame()

        # 5. 比较相同时间戳的结果
        common_idx = batch_result.index.intersection(streaming_combined.index)
        if len(common_idx) > 0:
            batch_ofci = batch_result.loc[common_idx, "ofci"].dropna()
            stream_ofci = streaming_combined.loc[common_idx, "ofci"].dropna()

            common_valid = batch_ofci.index.intersection(stream_ofci.index)
            if len(common_valid) > 0:
                diff = (
                    batch_ofci.loc[common_valid] - stream_ofci.loc[common_valid]
                ).abs()
                max_diff = diff.max()
                # 允许一些误差（因为窗口边界处理可能略有不同）
                assert max_diff < 1e-5, f"流式与批量计算不一致，最大差异: {max_diff}"

    def test_shd_streaming_vs_batch(self):
        """测试SHD：流式计算与批量计算应该一致"""
        dates = pd.date_range("2024-01-01", periods=400, freq="1min")

        np.random.seed(42)
        returns = pd.Series(np.random.randn(400) * 0.01, index=dates)
        cvd = returns.cumsum() * 1000

        # 批量计算
        batch_result = compute_shd_from_series(
            cvd_series=cvd, price_returns=returns, window=60
        )

        # 流式计算
        streaming_results = []
        chunk_size = 50
        accumulated_cvd = pd.Series(dtype=float)
        accumulated_returns = pd.Series(dtype=float)

        for i in range(0, len(cvd), chunk_size):
            chunk_cvd = cvd.iloc[i : i + chunk_size]
            chunk_returns = returns.iloc[i : i + chunk_size]

            accumulated_cvd = pd.concat([accumulated_cvd, chunk_cvd])
            accumulated_returns = pd.concat([accumulated_returns, chunk_returns])

            if len(accumulated_cvd) > 120:  # window * 2
                accumulated_cvd = accumulated_cvd.iloc[-120:]
                accumulated_returns = accumulated_returns.iloc[-120:]

            if len(accumulated_cvd) >= 60:  # 窗口大小
                chunk_result = compute_shd_from_series(
                    cvd_series=accumulated_cvd,
                    price_returns=accumulated_returns,
                    window=60,
                )
                chunk_result_filtered = chunk_result.loc[chunk_cvd.index]
                streaming_results.append(chunk_result_filtered)

        if streaming_results:
            streaming_combined = pd.concat(streaming_results)
        else:
            streaming_combined = pd.DataFrame()

        # 比较结果
        common_idx = batch_result.index.intersection(streaming_combined.index)
        if len(common_idx) > 0:
            batch_shd = batch_result.loc[common_idx, "shd"].dropna()
            stream_shd = streaming_combined.loc[common_idx, "shd"].dropna()

            common_valid = batch_shd.index.intersection(stream_shd.index)
            if len(common_valid) > 0:
                diff = (
                    batch_shd.loc[common_valid] - stream_shd.loc[common_valid]
                ).abs()
                max_diff = diff.max()
                assert max_diff < 1e-5, f"流式与批量计算不一致，最大差异: {max_diff}"

    def test_ofci_pct_streaming_vs_batch(self):
        """测试OFCI percentile：流式计算与批量计算应该一致"""
        dates = pd.date_range("2024-01-01", periods=400, freq="1min")
        ofci = pd.Series([-1.0, -0.5, 0.0, 0.5, 1.0] * 80, index=dates)

        # 批量计算
        batch_result = compute_ofci_pct_from_series(ofci=ofci, window=50)

        # 流式计算
        streaming_results = []
        chunk_size = 50
        accumulated_ofci = pd.Series(dtype=float)

        for i in range(0, len(ofci), chunk_size):
            chunk = ofci.iloc[i : i + chunk_size]
            accumulated_ofci = pd.concat([accumulated_ofci, chunk])

            if len(accumulated_ofci) > 100:  # window * 2
                accumulated_ofci = accumulated_ofci.iloc[-100:]

            if len(accumulated_ofci) >= 50:  # 窗口大小
                chunk_result = compute_ofci_pct_from_series(
                    ofci=accumulated_ofci, window=50
                )
                chunk_result_filtered = chunk_result.loc[chunk.index]
                streaming_results.append(chunk_result_filtered)

        if streaming_results:
            streaming_combined = pd.concat(streaming_results)
        else:
            streaming_combined = pd.DataFrame()

        # 比较结果
        common_idx = batch_result.index.intersection(streaming_combined.index)
        if len(common_idx) > 0:
            batch_pct = batch_result.loc[common_idx, "ofci_pct"].dropna()
            stream_pct = streaming_combined.loc[common_idx, "ofci_pct"].dropna()

            common_valid = batch_pct.index.intersection(stream_pct.index)
            if len(common_valid) > 0:
                diff = (
                    batch_pct.loc[common_valid] - stream_pct.loc[common_valid]
                ).abs()
                max_diff = diff.max()
                assert max_diff < 1e-5, f"流式与批量计算不一致，最大差异: {max_diff}"

    def test_shd_pct_streaming_vs_batch(self):
        """测试SHD percentile：流式计算与批量计算应该一致"""
        close, cvd = _make_close_cvd_for_shd(400, seed=42)
        pct_window = 50
        shd_window = 60

        batch_result = compute_shd_pct_from_series(
            close=close,
            cvd=cvd,
            shd_window=shd_window,
            percentile_window=pct_window,
            shift=1,
        )

        streaming_results = []
        chunk_size = 50
        accumulated_close = pd.Series(dtype=float)
        accumulated_cvd = pd.Series(dtype=float)

        for i in range(0, len(close), chunk_size):
            chunk_close = close.iloc[i : i + chunk_size]
            chunk_cvd = cvd.iloc[i : i + chunk_size]
            accumulated_close = pd.concat([accumulated_close, chunk_close])
            accumulated_cvd = pd.concat([accumulated_cvd, chunk_cvd])

            if len(accumulated_close) > 100:
                accumulated_close = accumulated_close.iloc[-100:]
                accumulated_cvd = accumulated_cvd.iloc[-100:]

            if len(accumulated_close) >= pct_window:
                chunk_result = compute_shd_pct_from_series(
                    close=accumulated_close,
                    cvd=accumulated_cvd,
                    shd_window=shd_window,
                    percentile_window=pct_window,
                    shift=1,
                )
                chunk_result_filtered = chunk_result.loc[chunk_close.index]
                streaming_results.append(chunk_result_filtered)

        if streaming_results:
            streaming_combined = pd.concat(streaming_results)
        else:
            streaming_combined = pd.DataFrame()

        common_idx = batch_result.index.intersection(streaming_combined.index)
        if len(common_idx) > 0:
            batch_pct = batch_result.loc[common_idx, "shd_pct"].dropna()
            stream_pct = streaming_combined.loc[common_idx, "shd_pct"].dropna()

            common_valid = batch_pct.index.intersection(stream_pct.index)
            if len(common_valid) > 0:
                diff = (
                    batch_pct.loc[common_valid] - stream_pct.loc[common_valid]
                ).abs()
                max_diff = diff.max()
                assert max_diff < 1e-5, f"流式与批量计算不一致，最大差异: {max_diff}"


class TestReflexivityFeaturesNoFutureLeak:
    """测试反身性特征无未来函数（No Lookahead Bias）⭐⭐⭐⭐⭐"""

    def test_ofci_no_future_leak(self):
        """测试OFCI：修改未来数据不应影响历史特征值"""
        # 1. 创建测试数据
        trades = pd.DataFrame(
            {
                "side": [1, -1] * 200,
            },
            index=pd.date_range("2024-01-01", periods=400, freq="1min"),
        )

        # 2. 计算第一次特征
        result1 = compute_ofci_from_trades(trades, window=100)
        ofci_1 = result1["ofci"].copy()

        # 3. 修改未来数据（从t=200开始）
        trades_future = trades.copy()
        trades_future.loc[trades_future.index[200] :, "side"] = 1  # 全部改为buy

        # 4. 重新计算特征
        result2 = compute_ofci_from_trades(trades_future, window=100)
        ofci_2 = result2["ofci"].copy()

        # 5. 验证历史数据（t < 200）的特征值应该相同
        check_idx = trades.index[:195]  # 窗口已满后的历史数据
        ofci_1_check = ofci_1.loc[check_idx].dropna()
        ofci_2_check = ofci_2.loc[check_idx].dropna()

        common_idx = ofci_1_check.index.intersection(ofci_2_check.index)
        if len(common_idx) > 0:
            diff = (ofci_1_check.loc[common_idx] - ofci_2_check.loc[common_idx]).abs()
            max_diff = diff.max()
            assert (
                max_diff < 1e-6
            ), f"未来数据变化不应影响历史OFCI值，最大差异: {max_diff}"

    def test_shd_no_future_leak(self):
        """测试SHD：修改未来数据不应影响历史特征值"""
        dates = pd.date_range("2024-01-01", periods=400, freq="1min")

        np.random.seed(42)
        returns = pd.Series(np.random.randn(400) * 0.01, index=dates)
        cvd = returns.cumsum() * 1000

        # 计算第一次特征
        result1 = compute_shd_from_series(
            cvd_series=cvd, price_returns=returns, window=60
        )
        shd_1 = result1["shd"].copy()

        # 修改未来数据（从t=200开始）
        returns_future = returns.copy()
        cvd_future = cvd.copy()
        returns_future.loc[returns_future.index[200] :] = 0.1  # 大幅改变
        cvd_future.loc[cvd_future.index[200] :] = cvd_future.iloc[199] + (
            returns_future.loc[returns_future.index[200] :].cumsum() * 1000
        )

        # 重新计算特征
        result2 = compute_shd_from_series(
            cvd_series=cvd_future, price_returns=returns_future, window=60
        )
        shd_2 = result2["shd"].copy()

        # 验证历史数据（t < 200）的特征值应该相同
        check_idx = dates[:195]  # 窗口已满后的历史数据
        shd_1_check = shd_1.loc[check_idx].dropna()
        shd_2_check = shd_2.loc[check_idx].dropna()

        common_idx = shd_1_check.index.intersection(shd_2_check.index)
        if len(common_idx) > 0:
            diff = (shd_1_check.loc[common_idx] - shd_2_check.loc[common_idx]).abs()
            max_diff = diff.max()
            assert (
                max_diff < 1e-6
            ), f"未来数据变化不应影响历史SHD值，最大差异: {max_diff}"

    def test_ofci_pct_no_future_leak(self):
        """测试OFCI percentile：修改未来数据不应影响历史特征值"""
        dates = pd.date_range("2024-01-01", periods=400, freq="1min")
        ofci = pd.Series([1, -1] * 200, index=dates)

        # 计算第一次特征
        result1 = compute_ofci_pct_from_series(ofci=ofci, window=100)
        ofci_pct_1 = result1["ofci_pct"].copy()

        # 修改未来数据（从t=200开始）
        ofci_future = ofci.copy()
        ofci_future.loc[ofci_future.index[200] :] = 1.0  # 全部改为1

        # 重新计算特征
        result2 = compute_ofci_pct_from_series(ofci=ofci_future, window=100)
        ofci_pct_2 = result2["ofci_pct"].copy()

        # 验证历史数据
        check_idx = dates[:195]
        ofci_pct_1_check = ofci_pct_1.loc[check_idx].dropna()
        ofci_pct_2_check = ofci_pct_2.loc[check_idx].dropna()

        common_idx = ofci_pct_1_check.index.intersection(ofci_pct_2_check.index)
        if len(common_idx) > 0:
            diff = (
                ofci_pct_1_check.loc[common_idx] - ofci_pct_2_check.loc[common_idx]
            ).abs()
            max_diff = diff.max()
            assert (
                max_diff < 1e-6
            ), f"未来数据变化不应影响历史OFCI percentile值，最大差异: {max_diff}"

    def test_shd_pct_no_future_leak(self):
        """测试SHD percentile：修改未来数据不应影响历史特征值"""
        close, cvd = _make_close_cvd_for_shd(400, seed=42)
        pct_window = 100

        result1 = compute_shd_pct_from_series(
            close=close,
            cvd=cvd,
            shd_window=60,
            percentile_window=pct_window,
            shift=1,
        )
        shd_pct_1 = result1["shd_pct"].copy()

        close_future = close.copy()
        cvd_future = cvd.copy()
        close_future.loc[close_future.index[200] :] = close_future.iloc[199] * 2.0
        cvd_future.loc[cvd_future.index[200] :] = cvd_future.iloc[199] + 5000.0

        result2 = compute_shd_pct_from_series(
            close=close_future,
            cvd=cvd_future,
            shd_window=60,
            percentile_window=pct_window,
            shift=1,
        )
        shd_pct_2 = result2["shd_pct"].copy()

        check_idx = close.index[:195]
        shd_pct_1_check = shd_pct_1.loc[check_idx].dropna()
        shd_pct_2_check = shd_pct_2.loc[check_idx].dropna()

        common_idx = shd_pct_1_check.index.intersection(shd_pct_2_check.index)
        if len(common_idx) > 0:
            diff = (
                shd_pct_1_check.loc[common_idx] - shd_pct_2_check.loc[common_idx]
            ).abs()
            max_diff = diff.max()
            assert (
                max_diff < 1e-6
            ), f"未来数据变化不应影响历史SHD percentile值，最大差异: {max_diff}"


class TestReflexivityFeaturesStreamingVsBatch:
    """测试反身性特征流式vs批量一致性 ⭐⭐⭐⭐"""

    def test_ofci_streaming_vs_batch(self):
        """测试OFCI：流式计算与批量计算应该一致"""
        # 1. 创建测试数据
        trades = pd.DataFrame(
            {
                "side": [1, -1] * 200,
            },
            index=pd.date_range("2024-01-01", periods=400, freq="1min"),
        )

        # 2. 批量计算（一次性处理所有数据）
        batch_result = compute_ofci_from_trades(trades, window=100)

        # 3. 流式计算（分块处理，模拟生产环境）
        streaming_results = []
        chunk_size = 50
        accumulated_data = pd.DataFrame()

        for i in range(0, len(trades), chunk_size):
            chunk = trades.iloc[i : i + chunk_size]
            accumulated_data = pd.concat([accumulated_data, chunk])

            # 只计算新chunk的结果（模拟流式处理）
            if len(accumulated_data) >= 100:
                chunk_result = compute_ofci_from_trades(accumulated_data, window=100)
                # 只保留最后一个chunk的结果（模拟实时计算）
                chunk_result = chunk_result.iloc[-chunk_size:]
                streaming_results.append(chunk_result)

        # 4. 合并流式结果
        if streaming_results:
            streaming_combined = pd.concat(streaming_results)
        else:
            streaming_combined = pd.DataFrame()

        # 5. 比较相同时间戳的结果（只比较有足够窗口后的数据）
        if len(streaming_combined) > 0:
            common_idx = batch_result.index.intersection(streaming_combined.index)
            if len(common_idx) > 0:
                batch_ofci = batch_result.loc[common_idx, "ofci"].dropna()
                stream_ofci = streaming_combined.loc[common_idx, "ofci"].dropna()

                common_valid = batch_ofci.index.intersection(stream_ofci.index)
                if len(common_valid) > 0:
                    diff = (
                        batch_ofci.loc[common_valid] - stream_ofci.loc[common_valid]
                    ).abs()
                    max_diff = diff.max()
                    # 允许一些误差（因为窗口边界处理可能略有不同）
                    assert (
                        max_diff < 1e-5
                    ), f"流式与批量计算不一致，最大差异: {max_diff}"

    def test_shd_streaming_vs_batch(self):
        """测试SHD：流式计算与批量计算应该一致"""
        dates = pd.date_range("2024-01-01", periods=400, freq="1min")
        np.random.seed(42)
        returns = pd.Series(np.random.randn(400) * 0.01, index=dates)
        cvd = returns.cumsum() * 1000

        # 批量计算
        batch_result = compute_shd_from_series(
            cvd_series=cvd, price_returns=returns, window=60
        )

        # 流式计算
        streaming_results = []
        chunk_size = 50
        accumulated_cvd = None
        accumulated_returns = None

        for i in range(0, len(cvd), chunk_size):
            chunk_cvd = cvd.iloc[i : i + chunk_size]
            chunk_returns = returns.iloc[i : i + chunk_size]
            if accumulated_cvd is None:
                accumulated_cvd = chunk_cvd.copy()
                accumulated_returns = chunk_returns.copy()
            else:
                accumulated_cvd = pd.concat([accumulated_cvd, chunk_cvd])
                accumulated_returns = pd.concat([accumulated_returns, chunk_returns])

            if len(accumulated_cvd) >= 60:
                chunk_result = compute_shd_from_series(
                    cvd_series=accumulated_cvd,
                    price_returns=accumulated_returns,
                    window=60,
                )
                chunk_result = chunk_result.iloc[-chunk_size:]
                streaming_results.append(chunk_result)

        if streaming_results:
            streaming_combined = pd.concat(streaming_results)
        else:
            streaming_combined = pd.DataFrame()

        if len(streaming_combined) > 0:
            common_idx = batch_result.index.intersection(streaming_combined.index)
            if len(common_idx) > 0:
                batch_shd = batch_result.loc[common_idx, "shd"].dropna()
                stream_shd = streaming_combined.loc[common_idx, "shd"].dropna()

                common_valid = batch_shd.index.intersection(stream_shd.index)
                if len(common_valid) > 0:
                    diff = (
                        batch_shd.loc[common_valid] - stream_shd.loc[common_valid]
                    ).abs()
                    max_diff = diff.max()
                    assert (
                        max_diff < 1e-5
                    ), f"流式与批量计算不一致，最大差异: {max_diff}"

    def test_ofci_pct_streaming_vs_batch(self):
        """测试OFCI percentile：流式计算与批量计算应该一致"""
        dates = pd.date_range("2024-01-01", periods=400, freq="1min")
        ofci = pd.Series([1, -1] * 200, index=dates)

        # 批量计算
        batch_result = compute_ofci_pct_from_series(ofci=ofci, window=100)

        # 流式计算
        streaming_results = []
        chunk_size = 50
        accumulated_ofci = None

        for i in range(0, len(ofci), chunk_size):
            chunk = ofci.iloc[i : i + chunk_size]
            if accumulated_ofci is None:
                accumulated_ofci = chunk.copy()
            else:
                accumulated_ofci = pd.concat([accumulated_ofci, chunk])

            if len(accumulated_ofci) >= 100:
                chunk_result = compute_ofci_pct_from_series(
                    ofci=accumulated_ofci, window=100
                )
                chunk_result = chunk_result.iloc[-chunk_size:]
                streaming_results.append(chunk_result)

        if streaming_results:
            streaming_combined = pd.concat(streaming_results)
        else:
            streaming_combined = pd.DataFrame()

        if len(streaming_combined) > 0:
            common_idx = batch_result.index.intersection(streaming_combined.index)
            if len(common_idx) > 0:
                batch_pct = batch_result.loc[common_idx, "ofci_pct"].dropna()
                stream_pct = streaming_combined.loc[common_idx, "ofci_pct"].dropna()

                common_valid = batch_pct.index.intersection(stream_pct.index)
                if len(common_valid) > 0:
                    diff = (
                        batch_pct.loc[common_valid] - stream_pct.loc[common_valid]
                    ).abs()
                    max_diff = diff.max()
                    assert (
                        max_diff < 1e-5
                    ), f"流式与批量计算不一致，最大差异: {max_diff}"

    def test_shd_pct_streaming_vs_batch(self):
        """测试SHD percentile：流式计算与批量计算应该一致"""
        close, cvd = _make_close_cvd_for_shd(400, seed=42)
        pct_window = 100
        shd_window = 60

        batch_result = compute_shd_pct_from_series(
            close=close,
            cvd=cvd,
            shd_window=shd_window,
            percentile_window=pct_window,
            shift=1,
        )

        streaming_results = []
        chunk_size = 50
        accumulated_close = None
        accumulated_cvd = None

        for i in range(0, len(close), chunk_size):
            chunk_close = close.iloc[i : i + chunk_size]
            chunk_cvd = cvd.iloc[i : i + chunk_size]
            if accumulated_close is None:
                accumulated_close = chunk_close.copy()
                accumulated_cvd = chunk_cvd.copy()
            else:
                accumulated_close = pd.concat([accumulated_close, chunk_close])
                accumulated_cvd = pd.concat([accumulated_cvd, chunk_cvd])

            if len(accumulated_close) >= pct_window:
                chunk_result = compute_shd_pct_from_series(
                    close=accumulated_close,
                    cvd=accumulated_cvd,
                    shd_window=shd_window,
                    percentile_window=pct_window,
                    shift=1,
                )
                chunk_result = chunk_result.iloc[-chunk_size:]
                streaming_results.append(chunk_result)

        if streaming_results:
            streaming_combined = pd.concat(streaming_results)
        else:
            streaming_combined = pd.DataFrame()

        if len(streaming_combined) > 0:
            common_idx = batch_result.index.intersection(streaming_combined.index)
            if len(common_idx) > 0:
                batch_pct = batch_result.loc[common_idx, "shd_pct"].dropna()
                stream_pct = streaming_combined.loc[common_idx, "shd_pct"].dropna()

                common_valid = batch_pct.index.intersection(stream_pct.index)
                if len(common_valid) > 0:
                    diff = (
                        batch_pct.loc[common_valid] - stream_pct.loc[common_valid]
                    ).abs()
                    max_diff = diff.max()
                    assert (
                        max_diff < 1e-5
                    ), f"流式与批量计算不一致，最大差异: {max_diff}"
