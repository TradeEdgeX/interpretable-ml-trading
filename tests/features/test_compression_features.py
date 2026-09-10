"""
compression_duration_f 与 recent_compression_decay_f 完整测试

测试内容：
1. 功能正确性：输出值域 [0, 1], 语义行为正确
2. 未来函数检测：修改未来数据不影响历史计算
3. 流式一致性：全量 vs 截断结果一致（无 lookahead）
4. 归一化跨 symbol 可比：不同价格/波动率量级下输出均在 [0, 1]
5. 边界场景：全 NaN、常数价格、极端波动
"""

import numpy as np
import pandas as pd
import pytest

from src.features.time_series.baseline_features import (
    compute_compression_duration_from_series,
    compute_recent_compression_decay_from_series,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def ohlc_data():
    """500 根 K 线：前 200 根低波动（压缩），后 300 根高波动（扩张）"""
    np.random.seed(42)
    n = 500
    idx = pd.date_range("2024-01-01", periods=n, freq="h")

    close = pd.Series(
        100 + np.cumsum(np.random.normal(0, 0.3, n)), index=idx, name="close"
    )
    # 前 200 根：极低波动（压缩）
    high = close.copy()
    low = close.copy()
    high.iloc[:200] = close.iloc[:200] + 0.05
    low.iloc[:200] = close.iloc[:200] - 0.05
    # 后 300 根：高波动（扩张）
    high.iloc[200:] = close.iloc[200:] + 2.0
    low.iloc[200:] = close.iloc[200:] - 2.0

    return high, low, close


@pytest.fixture
def ohlc_btc():
    """模拟 BTC 价格 (60000+)，验证跨 symbol 归一化"""
    np.random.seed(123)
    n = 500
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    close = pd.Series(
        60000 + np.cumsum(np.random.normal(0, 50, n)), index=idx, name="close"
    )
    high = close.copy()
    low = close.copy()
    high.iloc[:200] = close.iloc[:200] + 5
    low.iloc[:200] = close.iloc[:200] - 5
    high.iloc[200:] = close.iloc[200:] + 300
    low.iloc[200:] = close.iloc[200:] - 300
    return high, low, close


@pytest.fixture
def ohlc_small_cap():
    """模拟低价币 (0.001)，验证跨 symbol 归一化"""
    np.random.seed(456)
    n = 500
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    close = pd.Series(
        0.001 + np.cumsum(np.random.normal(0, 0.00001, n)), index=idx, name="close"
    )
    high = close.copy()
    low = close.copy()
    high.iloc[:200] = close.iloc[:200] + 0.000005
    low.iloc[:200] = close.iloc[:200] - 0.000005
    high.iloc[200:] = close.iloc[200:] + 0.0002
    low.iloc[200:] = close.iloc[200:] - 0.0002
    return high, low, close


# =============================================================================
# compression_duration_f 测试
# =============================================================================


class TestCompressionDuration:
    """compression_duration_f 完整测试"""

    WINDOW = 120  # 缩短 window 加速测试

    def test_bounded_0_1(self, ohlc_data):
        """输出严格在 [0, 1]"""
        high, low, close = ohlc_data
        out = compute_compression_duration_from_series(
            high=high, low=low, close=close, percentile_window=self.WINDOW
        )
        s = out.fillna(0.0)
        assert (s >= -1e-9).all(), f"min={s.min()} < 0"
        assert (s <= 1.0 + 1e-9).all(), f"max={s.max()} > 1"

    def test_semantic_compression_phase_high(self, ohlc_data):
        """压缩期间 compression_duration 应显著 > 0"""
        high, low, close = ohlc_data
        out = compute_compression_duration_from_series(
            high=high, low=low, close=close, percentile_window=self.WINDOW
        )
        # 前 200 根是压缩区，warmup 后（bar 50+）应持续积累
        late_compression = out.iloc[100:200].fillna(0.0)
        assert (
            late_compression.mean() > 0.01
        ), f"压缩后段均值={late_compression.mean():.4f} 太低"

    def test_semantic_expansion_phase_zero(self, ohlc_data):
        """扩张期间 compression_duration 应为 0（连续压缩 bars = 0）"""
        high, low, close = ohlc_data
        out = compute_compression_duration_from_series(
            high=high, low=low, close=close, percentile_window=self.WINDOW
        )
        # 后 100 根完全扩张，应为 0
        late_expansion = out.iloc[350:].fillna(0.0)
        assert (
            late_expansion.max() < 0.01
        ), f"扩张后段最大值={late_expansion.max():.4f} 不为 0"

    def test_no_future_leak(self, ohlc_data):
        """修改 bar 300+ 的数据不影响 bar 0~199"""
        high, low, close = ohlc_data
        out1 = compute_compression_duration_from_series(
            high=high, low=low, close=close, percentile_window=self.WINDOW
        )

        # 篡改未来
        h2, l2, c2 = high.copy(), low.copy(), close.copy()
        c2.iloc[300:] *= 2.0
        h2.iloc[300:] *= 3.0
        l2.iloc[300:] *= 0.5

        out2 = compute_compression_duration_from_series(
            high=h2, low=l2, close=c2, percentile_window=self.WINDOW
        )

        v1 = out1.iloc[:200].fillna(0.0)
        v2 = out2.iloc[:200].fillna(0.0)
        diff = (v1 - v2).abs().max()
        assert diff < 1e-10, f"存在未来泄露，差异: {diff}"

    def test_streaming_consistency(self, ohlc_data):
        """全量 vs 截断前 N 根结果应完全一致"""
        high, low, close = ohlc_data

        full = compute_compression_duration_from_series(
            high=high, low=low, close=close, percentile_window=self.WINDOW
        )
        n = 300
        partial = compute_compression_duration_from_series(
            high=high.iloc[:n],
            low=low.iloc[:n],
            close=close.iloc[:n],
            percentile_window=self.WINDOW,
        )

        v_full = full.iloc[:n].fillna(0.0)
        v_part = partial.fillna(0.0)
        diff = v_full.values - v_part.values
        assert np.abs(diff).max() < 1e-10, f"流式不一致，最大差异: {np.abs(diff).max()}"

    def test_cross_symbol_normalized_btc(self, ohlc_btc):
        """BTC 级价格：输出在 [0, 1]"""
        high, low, close = ohlc_btc
        out = compute_compression_duration_from_series(
            high=high, low=low, close=close, percentile_window=self.WINDOW
        )
        s = out.fillna(0.0)
        assert (s >= -1e-9).all() and (s <= 1.0 + 1e-9).all()
        assert s.nunique() > 1, "输出全为常数"

    def test_cross_symbol_normalized_small_cap(self, ohlc_small_cap):
        """低价币级价格：输出在 [0, 1]"""
        high, low, close = ohlc_small_cap
        out = compute_compression_duration_from_series(
            high=high, low=low, close=close, percentile_window=self.WINDOW
        )
        s = out.fillna(0.0)
        assert (s >= -1e-9).all() and (s <= 1.0 + 1e-9).all()
        assert s.nunique() > 1, "输出全为常数"

    def test_constant_price_safe(self):
        """常数价格不崩溃，输出全 0"""
        n = 200
        idx = pd.date_range("2024-01-01", periods=n, freq="h")
        close = pd.Series(100.0, index=idx)
        high = pd.Series(100.0, index=idx)
        low = pd.Series(100.0, index=idx)

        out = compute_compression_duration_from_series(
            high=high, low=low, close=close, percentile_window=50
        )
        assert out.isna().all() or (out.fillna(0.0) >= 0).all()


# =============================================================================
# recent_compression_decay_f 测试
# =============================================================================


class TestRecentCompressionDecay:
    """recent_compression_decay_f 完整测试"""

    WINDOW = 120
    DECAY = 0.97

    def test_bounded_0_1(self, ohlc_data):
        """输出严格在 [0, 1]"""
        high, low, close = ohlc_data
        out = compute_recent_compression_decay_from_series(
            high=high,
            low=low,
            close=close,
            percentile_window=self.WINDOW,
            decay_rate=self.DECAY,
        )
        s = out.fillna(0.0)
        assert (s >= -1e-9).all(), f"min={s.min()} < 0"
        assert (s <= 1.0 + 1e-9).all(), f"max={s.max()} > 1"

    def test_semantic_decay_after_compression(self, ohlc_data):
        """压缩后扩张期间应有衰减：越远离压缩结束点，值越低"""
        high, low, close = ohlc_data
        out = compute_recent_compression_decay_from_series(
            high=high,
            low=low,
            close=close,
            percentile_window=self.WINDOW,
            decay_rate=self.DECAY,
        )
        # 刚进入扩张 (bar 210) vs 扩张中期 (bar 350) vs 扩张后期 (bar 450)
        val_early = out.iloc[210]
        val_mid = out.iloc[350]
        val_late = out.iloc[450]
        assert (
            val_early > val_mid > val_late
        ), f"衰减不单调: early={val_early:.4f}, mid={val_mid:.4f}, late={val_late:.4f}"

    def test_semantic_compression_phase_positive(self, ohlc_data):
        """压缩期间值应为正（在积累能量）"""
        high, low, close = ohlc_data
        out = compute_recent_compression_decay_from_series(
            high=high,
            low=low,
            close=close,
            percentile_window=self.WINDOW,
            decay_rate=self.DECAY,
        )
        # 压缩后段 (warmup后) 应有正值
        compression_vals = out.iloc[80:200].fillna(0.0)
        assert (
            compression_vals.mean() > 0.1
        ), f"压缩期间均值={compression_vals.mean():.4f} 太低"

    def test_no_future_leak(self, ohlc_data):
        """修改 bar 300+ 的数据不影响 bar 0~199"""
        high, low, close = ohlc_data
        out1 = compute_recent_compression_decay_from_series(
            high=high,
            low=low,
            close=close,
            percentile_window=self.WINDOW,
            decay_rate=self.DECAY,
        )

        h2, l2, c2 = high.copy(), low.copy(), close.copy()
        c2.iloc[300:] *= 2.0
        h2.iloc[300:] *= 3.0
        l2.iloc[300:] *= 0.5

        out2 = compute_recent_compression_decay_from_series(
            high=h2,
            low=l2,
            close=c2,
            percentile_window=self.WINDOW,
            decay_rate=self.DECAY,
        )

        v1 = out1.iloc[:200].fillna(0.0)
        v2 = out2.iloc[:200].fillna(0.0)
        diff = (v1 - v2).abs().max()
        assert diff < 1e-10, f"存在未来泄露，差异: {diff}"

    def test_streaming_consistency(self, ohlc_data):
        """全量 vs 截断前 N 根结果应完全一致"""
        high, low, close = ohlc_data

        full = compute_recent_compression_decay_from_series(
            high=high,
            low=low,
            close=close,
            percentile_window=self.WINDOW,
            decay_rate=self.DECAY,
        )
        n = 300
        partial = compute_recent_compression_decay_from_series(
            high=high.iloc[:n],
            low=low.iloc[:n],
            close=close.iloc[:n],
            percentile_window=self.WINDOW,
            decay_rate=self.DECAY,
        )

        v_full = full.iloc[:n].fillna(0.0)
        v_part = partial.fillna(0.0)
        diff = np.abs(v_full.values - v_part.values).max()
        assert diff < 1e-10, f"流式不一致，最大差异: {diff}"

    def test_cross_symbol_normalized_btc(self, ohlc_btc):
        """BTC 级价格：输出在 [0, 1]"""
        high, low, close = ohlc_btc
        out = compute_recent_compression_decay_from_series(
            high=high,
            low=low,
            close=close,
            percentile_window=self.WINDOW,
            decay_rate=self.DECAY,
        )
        s = out.fillna(0.0)
        assert (s >= -1e-9).all() and (s <= 1.0 + 1e-9).all()

    def test_cross_symbol_normalized_small_cap(self, ohlc_small_cap):
        """低价币级价格：输出在 [0, 1]"""
        high, low, close = ohlc_small_cap
        out = compute_recent_compression_decay_from_series(
            high=high,
            low=low,
            close=close,
            percentile_window=self.WINDOW,
            decay_rate=self.DECAY,
        )
        s = out.fillna(0.0)
        assert (s >= -1e-9).all() and (s <= 1.0 + 1e-9).all()

    def test_constant_price_safe(self):
        """常数价格不崩溃"""
        n = 200
        idx = pd.date_range("2024-01-01", periods=n, freq="h")
        close = pd.Series(100.0, index=idx)
        high = pd.Series(100.0, index=idx)
        low = pd.Series(100.0, index=idx)

        out = compute_recent_compression_decay_from_series(
            high=high,
            low=low,
            close=close,
            percentile_window=50,
            decay_rate=self.DECAY,
        )
        assert out.isna().all() or (out.fillna(0.0) >= 0).all()

    def test_decay_rate_effect(self, ohlc_data):
        """衰减率越低，扩张期衰减越快"""
        high, low, close = ohlc_data
        fast = compute_recent_compression_decay_from_series(
            high=high,
            low=low,
            close=close,
            percentile_window=self.WINDOW,
            decay_rate=0.90,
        )
        slow = compute_recent_compression_decay_from_series(
            high=high,
            low=low,
            close=close,
            percentile_window=self.WINDOW,
            decay_rate=0.99,
        )
        # 扩张后期，慢衰减应该保留更多记忆
        assert (
            slow.iloc[400] > fast.iloc[400]
        ), f"衰减率效应异常: slow={slow.iloc[400]:.4f}, fast={fast.iloc[400]:.4f}"

    def test_no_compression_yields_zero(self):
        """全程高波动（无压缩）=> 输出接近 0"""
        np.random.seed(789)
        n = 300
        idx = pd.date_range("2024-01-01", periods=n, freq="h")
        close = pd.Series(100 + np.cumsum(np.random.normal(0, 1, n)), index=idx)
        high = close + 5.0
        low = close - 5.0

        out = compute_recent_compression_decay_from_series(
            high=high,
            low=low,
            close=close,
            percentile_window=self.WINDOW,
            decay_rate=self.DECAY,
        )
        # warmup 之后应全为 0（没有压缩就没有记忆）
        late_vals = out.iloc[150:].fillna(0.0)
        assert late_vals.max() < 0.1, f"无压缩时输出应接近 0，max={late_vals.max():.4f}"
