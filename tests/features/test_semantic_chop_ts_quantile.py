"""
semantic_chop_ts_quantile：同品种 rolling 分位 [0,1]（因果窗口）。

覆盖：
- 值域 / 形状 / 常数退化
- 未来函数：改动未来 bar 不改变历史 ts_q
- 流式：前缀序列末端 == 全量序列同 index 的值
- 与 TPC soft phase 集成列 tpc_semantic_chop_ts_q 一致（对同一 chop 序列）
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scipy.stats import percentileofscore

from src.features.time_series.bpc_features import compute_tpc_soft_phase_from_series
from src.features.time_series.semantic_chop_ts_quantile import (
    DEFAULT_CHOP_TS_MIN_PERIODS,
    DEFAULT_CHOP_TS_WINDOW,
    semantic_chop_ts_quantile,
)


@pytest.fixture
def hourly_index() -> pd.DatetimeIndex:
    return pd.date_range("2020-01-01", periods=900, freq="2h", tz="UTC")


class TestSemanticChopTsQuantileCore:
    def test_shape_matches_input(self, hourly_index: pd.DatetimeIndex):
        chop = np.random.default_rng(1).random(len(hourly_index))
        out = semantic_chop_ts_quantile(chop, hourly_index, window=200, min_periods=40)
        assert out.shape == (len(hourly_index),)

    def test_early_nan_then_finite(self, hourly_index: pd.DatetimeIndex):
        chop = np.linspace(0.01, 0.99, len(hourly_index))
        min_p = 50
        out = semantic_chop_ts_quantile(
            chop, hourly_index, window=200, min_periods=min_p
        )
        assert np.all(np.isnan(out[: min_p - 1]))
        assert np.all(np.isfinite(out[min_p - 1 + 10 :]))

    def test_values_in_unit_interval_when_finite(self, hourly_index: pd.DatetimeIndex):
        rng = np.random.default_rng(2)
        chop = rng.random(len(hourly_index)) * 0.8 + 0.05
        out = semantic_chop_ts_quantile(chop, hourly_index, window=200, min_periods=40)
        fin = out[np.isfinite(out)]
        assert fin.min() >= -1e-9
        assert fin.max() <= 1.0 + 1e-9

    def test_constant_window_returns_zero(self, hourly_index: pd.DatetimeIndex):
        chop = np.full(len(hourly_index), 0.37)
        out = semantic_chop_ts_quantile(chop, hourly_index, window=100, min_periods=50)
        assert np.nanmax(np.abs(out[np.isfinite(out)])) < 1e-9


class TestSemanticChopTsQuantileNoFutureLeak:
    def test_future_perturbation_preserves_past(self, hourly_index: pd.DatetimeIndex):
        n = len(hourly_index)
        rng = np.random.default_rng(3)
        chop = rng.random(n) * 0.5 + 0.1
        window, min_p = 160, 40
        q_full = semantic_chop_ts_quantile(
            chop, hourly_index, window=window, min_periods=min_p
        )
        cut = n // 2
        chop2 = chop.copy()
        chop2[cut:] = chop2[cut:] * 3.0 + 0.2
        q2 = semantic_chop_ts_quantile(
            chop2, hourly_index, window=window, min_periods=min_p
        )
        # 在 cut 之前，窗口内从未见过 cut 之后的数据 → 应逐点相等
        np.testing.assert_allclose(
            q_full[:cut],
            q2[:cut],
            rtol=0,
            atol=1e-12,
            equal_nan=True,
        )


class TestSemanticChopTsQuantileMatchesLegacyScipyRolling:
    def test_matches_pandas_rolling_scipy_reference(self):
        """Regression: NumPy loop must match legacy rolling + percentileofscore."""
        n = 160
        idx = pd.date_range("2021-03-01", periods=n, freq="h", tz="UTC")
        rng = np.random.default_rng(42)
        chop = rng.random(n) * 0.72 + 0.08
        window, min_p = 55, 22

        def _pct_last(x: np.ndarray) -> float:
            if x.size < min_p:
                return float("nan")
            xv = np.asarray(x, dtype=float)
            xv = xv[np.isfinite(xv)]
            if xv.size < 2:
                return float("nan")
            if float(np.nanstd(xv)) < 1e-12:
                return 0.0
            last = float(xv[-1])
            return float(percentileofscore(xv, last, kind="mean")) / 100.0

        s = pd.Series(chop)
        ref = (
            s.rolling(window, min_periods=min_p)
            .apply(_pct_last, raw=True)
            .to_numpy(dtype=float)
        )
        out = semantic_chop_ts_quantile(chop, idx, window=window, min_periods=min_p)
        np.testing.assert_allclose(ref, out, rtol=0, atol=1e-12, equal_nan=True)


class TestSemanticChopTsQuantileStreaming:
    def test_prefix_last_equals_full_at_index(self, hourly_index: pd.DatetimeIndex):
        """流式：仅用 [0..i] 前缀算出的末端 ts_q 应等于全量序列在 i 处的值。"""
        rng = np.random.default_rng(4)
        chop = rng.random(len(hourly_index)) * 0.6 + 0.05
        window, min_p = 200, 50
        q_full = semantic_chop_ts_quantile(
            chop, hourly_index, window=window, min_periods=min_p
        )
        for i in range(min_p, len(hourly_index), 17):
            q_pre = semantic_chop_ts_quantile(
                chop[: i + 1],
                hourly_index[: i + 1],
                window=window,
                min_periods=min_p,
            )
            assert np.isfinite(q_pre[-1])
            assert abs(float(q_pre[-1]) - float(q_full[i])) < 1e-10


class TestSemanticChopTsQuantileTPCIntegration:
    def test_tpc_soft_phase_exposes_ts_q_in_range(self):
        n = 700
        idx = pd.date_range("2021-06-01", periods=n, freq="2h", tz="UTC")
        rng = np.random.default_rng(5)
        close = pd.Series(100 + np.cumsum(rng.normal(0, 0.4, n)), index=idx)
        high = close * (1 + np.abs(rng.normal(0, 0.002, n)))
        low = close * (1 - np.abs(rng.normal(0, 0.002, n)))
        atr = (high - low).rolling(14, min_periods=1).mean().clip(lower=1e-6)
        vol = pd.Series(rng.lognormal(12, 0.3, n), index=idx)
        cvd = pd.Series(rng.normal(0, 1e5, n), index=idx)
        bb = pd.Series(rng.random(n), index=idx)
        ep = pd.Series(rng.normal(0, 0.08, n), index=idx)
        df = compute_tpc_soft_phase_from_series(
            close=close,
            high=high,
            low=low,
            atr=atr,
            volume=vol,
            cvd_change_5=cvd,
            bb_width_normalized=bb,
            ema_1200_position=ep,
        )
        assert "tpc_semantic_chop_ts_q" in df.columns
        tsq = df["tpc_semantic_chop_ts_q"].to_numpy(dtype=float)
        chop = df["tpc_semantic_chop"].to_numpy(dtype=float)
        ref = semantic_chop_ts_quantile(
            chop,
            df.index,
            window=DEFAULT_CHOP_TS_WINDOW,
            min_periods=DEFAULT_CHOP_TS_MIN_PERIODS,
        )
        np.testing.assert_allclose(tsq, ref, rtol=0, atol=1e-12, equal_nan=True)
        fin = tsq[np.isfinite(tsq)]
        assert fin.min() >= -1e-9
        assert fin.max() <= 1.0 + 1e-9
