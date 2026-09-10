"""
Session & Microstructure 特征测试

测试内容：
1. 未来函数检测（确保特征不使用未来信息）⭐⭐⭐⭐⭐
2. 流式 vs 批量一致性测试 ⭐⭐⭐⭐
3. 功能正确性测试 ⭐⭐⭐
"""

import numpy as np
import pandas as pd
import pytest

from src.features.time_series.session_features import (
    compute_session_features_from_series,
    compute_bars_since_extreme_from_series,
    SESSION_BOUNDARIES,
    OVERLAP_START,
    OVERLAP_END,
)


# =============================================================================
# 📊 测试数据生成器
# =============================================================================


def create_ohlcv_data(
    n_samples: int = 500,
    base_price: float = 50000.0,
    volatility: float = 0.02,
    seed: int = 42,
    freq: str = "4h",
) -> pd.DataFrame:
    np.random.seed(seed)
    timestamps = pd.date_range("2024-01-01 00:00:00", periods=n_samples, freq=freq)
    returns = np.random.randn(n_samples) * volatility
    log_prices = np.log(base_price) + np.cumsum(returns)
    close = np.exp(log_prices)

    high_spread = np.abs(np.random.randn(n_samples)) * volatility * base_price
    low_spread = np.abs(np.random.randn(n_samples)) * volatility * base_price
    open_offset = np.random.randn(n_samples) * volatility * base_price * 0.5

    high = close + high_spread
    low = close - low_spread
    open_price = close + open_offset

    high = np.maximum(high, np.maximum(close, open_price))
    low = np.minimum(low, np.minimum(close, open_price))

    tr = np.maximum(
        high - low,
        np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))),
    )
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).rolling(14, min_periods=1).mean().values

    return pd.DataFrame(
        {
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.random.uniform(100, 1000, n_samples),
            "atr": atr,
        },
        index=timestamps,
    )


# =============================================================================
# 📋 Session Features 测试
# =============================================================================


class TestSessionFeaturesNoFutureLeak:
    """session_features 无未来函数测试"""

    @pytest.fixture
    def sample_data(self):
        return create_ohlcv_data(n_samples=500, freq="4h", seed=42)

    def test_session_features_no_future_leak(self, sample_data):
        """session_features 纯时间函数，修改未来价格不影响任何历史值"""
        df = sample_data
        result1 = compute_session_features_from_series(close=df["close"])

        df_future = df.copy()
        df_future.loc[df_future.index[300:], "close"] *= 2.0
        result2 = compute_session_features_from_series(close=df_future["close"])

        check_idx = df.index[:250]
        for feat in ["session_id", "hour_sin", "hour_cos", "is_session_overlap"]:
            diff = (
                (result1.loc[check_idx, feat] - result2.loc[check_idx, feat])
                .abs()
                .max()
            )
            assert diff < 1e-10, f"未来数据影响了历史 {feat}，差异: {diff}"


class TestSessionFeaturesStreamingVsBatch:
    """session_features 流式 vs 批量一致性"""

    @pytest.fixture
    def sample_data(self):
        return create_ohlcv_data(n_samples=500, freq="1h", seed=42)

    def test_streaming_vs_batch_consistency(self, sample_data):
        """分块计算与整体计算结果完全一致"""
        df = sample_data
        batch_result = compute_session_features_from_series(close=df["close"])

        chunk_size = 100
        streaming_results = []
        for i in range(0, len(df), chunk_size):
            chunk = df.iloc[i : i + chunk_size]
            chunk_result = compute_session_features_from_series(close=chunk["close"])
            streaming_results.append(chunk_result)

        streaming_combined = pd.concat(streaming_results, axis=0)

        for feat in ["session_id", "hour_sin", "hour_cos", "is_session_overlap"]:
            diff = (batch_result[feat] - streaming_combined[feat]).abs().max()
            assert diff < 1e-10, f"流式与批量 {feat} 不一致，差异: {diff}"

    def test_incremental_append_consistency(self, sample_data):
        """追加数据不改变已有结果"""
        df = sample_data
        result_partial = compute_session_features_from_series(
            close=df["close"].iloc[:300]
        )
        result_full = compute_session_features_from_series(close=df["close"])

        check_idx = df.index[:300]
        for feat in ["session_id", "hour_sin", "hour_cos", "is_session_overlap"]:
            diff = (result_partial[feat] - result_full.loc[check_idx, feat]).abs().max()
            assert diff < 1e-10, f"追加数据改变了 {feat}，差异: {diff}"


class TestSessionFeaturesFunctional:
    """session_features 功能正确性"""

    def test_session_id_classification(self):
        """session_id 正确分类各时段"""
        hours = [0, 3, 7, 8, 10, 13, 14, 17, 20, 21, 23]
        expected_sessions = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3]

        idx = pd.to_datetime([f"2024-01-01 {h:02d}:00:00" for h in hours])
        close = pd.Series(100.0, index=idx)
        result = compute_session_features_from_series(close=close)

        for i, (h, exp) in enumerate(zip(hours, expected_sessions)):
            actual = result["session_id"].iloc[i]
            assert actual == exp, f"hour={h}: expected session={exp}, got {actual}"

    def test_hour_sin_cos_cyclic(self):
        """hour_sin/cos 周期编码正确: sin²+cos²=1"""
        idx = pd.date_range("2024-01-01", periods=24, freq="1h")
        close = pd.Series(100.0, index=idx)
        result = compute_session_features_from_series(close=close)

        sum_sq = result["hour_sin"] ** 2 + result["hour_cos"] ** 2
        assert (sum_sq - 1.0).abs().max() < 1e-10, "sin²+cos² ≠ 1"

    def test_hour_sin_cos_values(self):
        """hour_sin/cos 在已知点的精确值"""
        # hour=0 → sin=0, cos=1
        # hour=6 → sin=1, cos≈0
        # hour=12 → sin≈0, cos=-1
        # hour=18 → sin=-1, cos≈0
        idx = pd.to_datetime(
            [
                "2024-01-01 00:00",
                "2024-01-01 06:00",
                "2024-01-01 12:00",
                "2024-01-01 18:00",
            ]
        )
        close = pd.Series(100.0, index=idx)
        result = compute_session_features_from_series(close=close)

        assert abs(result["hour_sin"].iloc[0] - 0.0) < 1e-10
        assert abs(result["hour_cos"].iloc[0] - 1.0) < 1e-10
        assert abs(result["hour_sin"].iloc[1] - 1.0) < 1e-10
        assert abs(result["hour_sin"].iloc[3] - (-1.0)) < 1e-10

    def test_overlap_indicator(self):
        """is_session_overlap 正确标记 EU-US 重叠时段 14:00-16:00"""
        idx = pd.date_range("2024-01-01", periods=24, freq="1h")
        close = pd.Series(100.0, index=idx)
        result = compute_session_features_from_series(close=close)

        for i, ts in enumerate(idx):
            expected = 1.0 if OVERLAP_START <= ts.hour < OVERLAP_END else 0.0
            actual = result["is_session_overlap"].iloc[i]
            assert (
                actual == expected
            ), f"hour={ts.hour}: overlap expected={expected}, got={actual}"

    def test_all_features_present(self):
        """所有预期输出特征都存在"""
        idx = pd.date_range("2024-01-01", periods=10, freq="4h")
        close = pd.Series(100.0, index=idx)
        result = compute_session_features_from_series(close=close)

        expected = {"session_id", "hour_sin", "hour_cos", "is_session_overlap"}
        assert set(result.columns) == expected

    def test_no_inf_no_nan(self):
        """输出无 Inf/NaN"""
        df = create_ohlcv_data(n_samples=200, freq="1h")
        result = compute_session_features_from_series(close=df["close"])

        for col in result.columns:
            assert not np.isinf(result[col]).any(), f"{col} 包含 Inf"
            assert not result[col].isna().any(), f"{col} 包含 NaN"

    def test_value_ranges(self):
        """特征值域正确"""
        df = create_ohlcv_data(n_samples=200, freq="1h")
        result = compute_session_features_from_series(close=df["close"])

        assert result["session_id"].isin([0, 1, 2, 3]).all()
        assert result["hour_sin"].between(-1, 1).all()
        assert result["hour_cos"].between(-1, 1).all()
        assert result["is_session_overlap"].isin([0, 1]).all()

    def test_non_datetime_index_fallback(self):
        """RangeIndex 退化为全 0（不崩溃）"""
        close = pd.Series(100.0, index=range(10))
        result = compute_session_features_from_series(close=close)

        assert (result["session_id"] == 0).all()
        assert (result["hour_sin"] == 0).all()
        assert (result["hour_cos"] == 0).all()
        assert (result["is_session_overlap"] == 0).all()


# =============================================================================
# 📋 Bars Since Extreme 测试
# =============================================================================


class TestBarsSinceExtremeNoFutureLeak:
    """bars_since_extreme 无未来函数测试"""

    @pytest.fixture
    def sample_data(self):
        return create_ohlcv_data(n_samples=500, freq="4h", seed=42)

    def test_no_future_leak(self, sample_data):
        """修改未来数据不影响历史特征值"""
        df = sample_data
        result1 = compute_bars_since_extreme_from_series(high=df["high"], low=df["low"])

        df_future = df.copy()
        df_future.loc[df_future.index[300:], "high"] *= 2.0
        df_future.loc[df_future.index[300:], "low"] *= 0.5
        result2 = compute_bars_since_extreme_from_series(
            high=df_future["high"], low=df_future["low"]
        )

        check_idx = df.index[:250]
        for feat in ["bars_since_local_high", "bars_since_local_low"]:
            diff = (
                (result1.loc[check_idx, feat] - result2.loc[check_idx, feat])
                .abs()
                .max()
            )
            assert diff < 1e-10, f"未来数据影响了历史 {feat}，差异: {diff}"

    def test_rolling_window_no_lookahead(self, sample_data):
        """t 时刻只用 [t-lookback+1, t] 的数据"""
        df = sample_data

        # 在 t=200 制造极端高点
        df_shock = df.copy()
        df_shock.loc[df_shock.index[200], "high"] *= 10.0

        result_shock = compute_bars_since_extreme_from_series(
            high=df_shock["high"], low=df_shock["low"]
        )
        result_orig = compute_bars_since_extreme_from_series(
            high=df["high"], low=df["low"]
        )

        # t=199 不应受 t=200 影响
        diff = abs(
            result_shock.loc[df.index[199], "bars_since_local_high"]
            - result_orig.loc[df.index[199], "bars_since_local_high"]
        )
        assert diff < 1e-10, f"t=199 受到 t=200 数据影响，差异: {diff}"


class TestBarsSinceExtremeStreamingVsBatch:
    """bars_since_extreme 流式 vs 批量一致性"""

    @pytest.fixture
    def sample_data(self):
        return create_ohlcv_data(n_samples=500, freq="4h", seed=42)

    def test_streaming_vs_batch(self, sample_data):
        """分块计算（带 warmup）与整体计算结果一致"""
        df = sample_data
        lookback = 20

        batch_result = compute_bars_since_extreme_from_series(
            high=df["high"], low=df["low"], lookback=lookback
        )

        chunk_size = 100
        warmup = lookback * 2
        streaming_results = []

        for i in range(warmup, len(df), chunk_size):
            start = max(0, i - warmup)
            end = min(i + chunk_size, len(df))
            chunk_result = compute_bars_since_extreme_from_series(
                high=df["high"].iloc[start:end],
                low=df["low"].iloc[start:end],
                lookback=lookback,
            )
            actual_start = i - start
            streaming_results.append(chunk_result.iloc[actual_start:])

        streaming_combined = pd.concat(streaming_results, axis=0)
        streaming_combined = streaming_combined[
            ~streaming_combined.index.duplicated(keep="first")
        ]

        common_idx = batch_result.index.intersection(streaming_combined.index)
        for feat in ["bars_since_local_high", "bars_since_local_low"]:
            diff = (
                (
                    batch_result.loc[common_idx, feat]
                    - streaming_combined.loc[common_idx, feat]
                )
                .abs()
                .max()
            )
            assert diff < 1e-6, f"流式与批量 {feat} 不一致，差异: {diff}"

    def test_incremental_append(self, sample_data):
        """追加数据不改变已有结果（warmup 后区域）"""
        df = sample_data
        result_partial = compute_bars_since_extreme_from_series(
            high=df["high"].iloc[:300], low=df["low"].iloc[:300]
        )
        result_full = compute_bars_since_extreme_from_series(
            high=df["high"], low=df["low"]
        )

        check_idx = df.index[50:250]
        for feat in ["bars_since_local_high", "bars_since_local_low"]:
            diff = (
                (result_partial.loc[check_idx, feat] - result_full.loc[check_idx, feat])
                .abs()
                .max()
            )
            assert diff < 1e-10, f"追加数据改变了 {feat}，差异: {diff}"


class TestBarsSinceExtremeFunctional:
    """bars_since_extreme 功能正确性"""

    def test_at_new_high_value_is_zero(self):
        """刚创新高时 bars_since_local_high = 0"""
        n = 30
        idx = pd.date_range("2024-01-01", periods=n, freq="4h")
        # 单调递增 → 每个 bar 都是新高
        high = pd.Series(np.arange(n, dtype=float) + 100.0, index=idx)
        low = high - 1.0

        result = compute_bars_since_extreme_from_series(high=high, low=low)
        # 每个 bar 都是窗口最高点 → bars_since_local_high = 0
        assert (result["bars_since_local_high"] == 0.0).all()

    def test_decreasing_price_high_increases(self):
        """价格持续下跌时 bars_since_local_high 递增"""
        n = 50
        idx = pd.date_range("2024-01-01", periods=n, freq="4h")
        # 单调递减
        high = pd.Series(200.0 - np.arange(n, dtype=float), index=idx)
        low = high - 1.0

        result = compute_bars_since_extreme_from_series(high=high, low=low, lookback=20)
        # 在 lookback 窗口内，最高点总是窗口开头
        # 所以 bars_since_local_high 应该递增到 lookback-1 然后平台
        for i in range(1, min(20, n)):
            assert result["bars_since_local_high"].iloc[i] > 0

    def test_feature_value_ranges(self):
        """输出在 [0, 1] 范围"""
        df = create_ohlcv_data(n_samples=300)
        result = compute_bars_since_extreme_from_series(high=df["high"], low=df["low"])

        for feat in ["bars_since_local_high", "bars_since_local_low"]:
            assert result[feat].min() >= 0.0, f"{feat} < 0"
            assert result[feat].max() <= 1.0, f"{feat} > 1"

    def test_no_inf_no_nan(self):
        """输出无 Inf/NaN"""
        df = create_ohlcv_data(n_samples=200)
        result = compute_bars_since_extreme_from_series(high=df["high"], low=df["low"])
        for col in result.columns:
            assert not np.isinf(result[col]).any(), f"{col} 包含 Inf"
            assert not result[col].isna().any(), f"{col} 包含 NaN"

    def test_all_features_present(self):
        """所有预期输出特征都存在"""
        df = create_ohlcv_data(n_samples=50)
        result = compute_bars_since_extreme_from_series(high=df["high"], low=df["low"])
        expected = {"bars_since_local_high", "bars_since_local_low"}
        assert set(result.columns) == expected

    def test_symmetry_high_low(self):
        """新低时 bars_since_local_low = 0"""
        n = 30
        idx = pd.date_range("2024-01-01", periods=n, freq="4h")
        # 单调递减 → 每个 bar 都是新低
        low = pd.Series(200.0 - np.arange(n, dtype=float), index=idx)
        high = low + 1.0

        result = compute_bars_since_extreme_from_series(high=high, low=low)
        assert (result["bars_since_local_low"] == 0.0).all()

    def test_lookback_parameter(self):
        """lookback 参数影响窗口大小"""
        df = create_ohlcv_data(n_samples=200, seed=42)

        result_short = compute_bars_since_extreme_from_series(
            high=df["high"], low=df["low"], lookback=5
        )
        result_long = compute_bars_since_extreme_from_series(
            high=df["high"], low=df["low"], lookback=50
        )

        # 短 lookback 的极值更近（因为窗口更小），
        # 所以 bars_since_local_high 的平均值应该更小
        mean_short = result_short["bars_since_local_high"].mean()
        mean_long = result_long["bars_since_local_high"].mean()
        assert (
            mean_short <= mean_long + 0.01
        ), f"短 lookback 均值 {mean_short:.4f} > 长 lookback {mean_long:.4f}"

    def test_multi_asset_comparability(self):
        """不同价格水平的资产，bars_since_extreme 分布相似"""
        assets = {
            "BTC": (50000.0, 0.02),
            "SOL": (100.0, 0.03),
        }
        means = {}
        for name, (price, vol) in assets.items():
            df = create_ohlcv_data(base_price=price, volatility=vol, seed=42)
            result = compute_bars_since_extreme_from_series(
                high=df["high"], low=df["low"]
            )
            means[name] = result["bars_since_local_high"].mean()

        # 均值差异不应超过 0.15（因为波动率不同会有小差异）
        diff = abs(means["BTC"] - means["SOL"])
        assert diff < 0.15, f"资产间均值差异过大: {diff:.4f}"
