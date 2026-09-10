"""Tests for funding-rate feature functions.

Covers (four-dimension verification):
1. No look-ahead: future bars do not change past values
2. Streaming consistency: first K bars identical alone vs with future data
3. Robust zscore correctness: median/MAD math vs mean/std
4. Functional correctness: output columns, bounded, causal merge, change semantics
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.time_series.funding_rate_features import (
    compute_funding_rate_features_from_df,
    compute_funding_scene_semantic_scores_from_df,
    _rolling_robust_zscore,
)

# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

SYM = "BTCUSDT"
ALL_COLS = [
    "funding_rate",
    "funding_rate_abs",
    "funding_rate_change_1",
    "funding_rate_zscore_50",
    "funding_rate_abs_zscore_50",
]


def _write_funding_parquet(tmp_dir, funding_rates: np.ndarray, freq: str = "8h"):
    """Write a mock funding-rate parquet file."""
    n = len(funding_rates)
    idx = pd.date_range("2024-01-01", periods=n, freq=freq, tz="UTC")
    df = pd.DataFrame(
        {"_symbol": SYM, "funding_rate": funding_rates},
        index=idx,
    )
    df.index.name = "datetime"
    df.to_parquet(tmp_dir / f"{SYM}_2024-01_funding_rate.parquet")


def _make_bars(n: int = 200, freq: str = "4h", seed: int = 42) -> pd.DataFrame:
    """Create bar DataFrame aligned with mock funding data."""
    rng = np.random.RandomState(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq=freq, tz="UTC")
    close = 50000.0 + rng.randn(n).cumsum() * 100
    return pd.DataFrame(
        {
            "open": close + rng.randn(n) * 10,
            "high": close + abs(rng.randn(n)) * 50,
            "low": close - abs(rng.randn(n)) * 50,
            "close": close,
            "volume": abs(rng.randn(n)) * 1e6 + 1e5,
            "_symbol": SYM,
        },
        index=idx,
    )


# ─────────────────────────────────────────────────────────────
# 1. Causal merge_asof (no look-ahead)
# ─────────────────────────────────────────────────────────────


class TestFundingRateJoinCausal:
    """Verify merge_asof backward join correctness."""

    def test_funding_rate_join_is_causal(self, tmp_path):
        """Exact value mapping: 04:00 sees 00:00, 08:00 sees 08:00, etc."""
        idx = pd.to_datetime(
            [
                "2024-01-01 00:00:00+00:00",
                "2024-01-01 08:00:00+00:00",
                "2024-01-01 16:00:00+00:00",
            ]
        )
        fr = pd.DataFrame(
            {"_symbol": SYM, "funding_rate": [0.001, 0.002, -0.001]}, index=idx
        )
        fr.index.name = "datetime"
        fr.to_parquet(tmp_path / f"{SYM}_2024-01_funding_rate.parquet")

        bar_idx = pd.date_range("2024-01-01 04:00:00+00:00", periods=5, freq="4h")
        df = pd.DataFrame({"close": 100.0, "_symbol": SYM}, index=bar_idx)

        out = compute_funding_rate_features_from_df(df, funding_rate_dir=str(tmp_path))

        got = out["funding_rate"].to_list()
        # 04:00→00:00(0.001), 08:00→08:00(0.002), 12:00→08:00(0.002),
        # 16:00→16:00(-0.001), 20:00→16:00(-0.001)
        assert np.isclose(got[0], 0.001)
        assert np.isclose(got[1], 0.002)
        assert np.isclose(got[2], 0.002)
        assert np.isclose(got[3], -0.001)
        assert np.isclose(got[4], -0.001)


# ─────────────────────────────────────────────────────────────
# 2. No look-ahead + streaming consistency (end-to-end)
# ─────────────────────────────────────────────────────────────


class TestFundingNoLookAheadStreaming:
    """First K bars must be identical whether computed alone or with future bars."""

    def test_no_look_ahead_and_streaming(self, tmp_path):
        rng = np.random.RandomState(42)
        n_fr = 300  # 300 × 8h = 100 days of funding data
        fr_vals = 0.0001 + rng.randn(n_fr) * 0.0005
        _write_funding_parquet(tmp_path, fr_vals)

        bars = _make_bars(200)
        k = 80

        result_full = compute_funding_rate_features_from_df(
            bars, funding_rate_dir=str(tmp_path), z_window=50, z_min_periods=20
        )
        result_prefix = compute_funding_rate_features_from_df(
            bars.iloc[:k], funding_rate_dir=str(tmp_path), z_window=50, z_min_periods=20
        )

        for col in ALL_COLS:
            pd.testing.assert_series_equal(
                result_prefix[col],
                result_full[col].iloc[:k],
                check_names=False,
                rtol=1e-10,
            )


# ─────────────────────────────────────────────────────────────
# 3. Robust zscore correctness
# ─────────────────────────────────────────────────────────────


class TestRobustZscore:
    """Verify _rolling_robust_zscore uses median/MAD, not mean/std."""

    def test_robust_zscore_manual_calculation(self):
        """For known input, verify robust z matches (x - median) / (MAD * 1.4826)."""
        # Create simple series where we can manually compute
        x = pd.Series([1.0, 2.0, 3.0, 4.0, 100.0])  # 100 is an outlier
        result = _rolling_robust_zscore(x, window=5, min_periods=5)

        # At index 4 (full window):
        # median([1,2,3,4,100]) = 3.0
        # deviations: [2, 1, 0, 1, 97]
        # MAD = median([2,1,0,1,97]) = 1.0
        # scale = 1.0 * 1.4826 = 1.4826
        # robust_z(100) = (100 - 3) / 1.4826 = 65.42...
        assert result.iloc[4] > 60  # very high: outlier detected

        # Compare with mean/std approach: would be much lower due to std inflation
        mean = x.mean()  # 22.0
        std = x.std(ddof=0)  # ~38.0
        naive_z = (100 - mean) / std  # ~2.05
        assert result.iloc[4] > naive_z * 10  # robust z is 30x higher for outliers

    def test_robust_zscore_resistant_to_spike(self):
        """A single spike should not pollute subsequent z-scores."""
        rng = np.random.RandomState(42)
        n = 100
        x = pd.Series(rng.randn(n) * 0.001)  # normal funding-like data
        x.iloc[30] = 0.1  # extreme spike at position 30

        result = _rolling_robust_zscore(x, window=20, min_periods=10)

        # After the spike leaves the window (position 51+), z-scores should be normal
        post_spike = result.iloc[55:80].dropna()
        assert (
            post_spike.abs().mean() < 3.0
        ), f"Post-spike z-scores still elevated: mean={post_spike.abs().mean():.2f}"

    def test_robust_zscore_nan_warmup(self):
        """First min_periods-1 values should be NaN."""
        x = pd.Series(range(20), dtype=float)
        result = _rolling_robust_zscore(x, window=10, min_periods=5)
        assert result.iloc[:4].isna().all()
        assert result.iloc[5:].notna().any()


# ─────────────────────────────────────────────────────────────
# 4. Functional correctness
# ─────────────────────────────────────────────────────────────


class TestFundingFunctionalCorrectness:
    """Output columns, shapes, and feature semantics."""

    def test_output_columns_and_shape(self, tmp_path):
        rng = np.random.RandomState(42)
        _write_funding_parquet(tmp_path, 0.0001 + rng.randn(300) * 0.0005)
        bars = _make_bars(100)
        out = compute_funding_rate_features_from_df(
            bars, funding_rate_dir=str(tmp_path)
        )
        assert isinstance(out, pd.DataFrame)
        assert len(out) == 100
        assert set(ALL_COLS).issubset(set(out.columns))

    def test_change_is_native_interval_diff(self, tmp_path):
        """funding_rate_change_1 should be diff on native 8h frequency, not bar-level.

        On 4h bars, consecutive bars that see the SAME funding observation
        should have the SAME change value (the previous native diff).
        """
        # 3 funding observations: 0.001, 0.003, -0.001
        fr_vals = np.array([0.001, 0.003, -0.001])
        _write_funding_parquet(tmp_path, fr_vals)

        # 6 bars at 4h: 00:00, 04:00, 08:00, 12:00, 16:00, 20:00
        bar_idx = pd.date_range("2024-01-01 00:00+00:00", periods=6, freq="4h")
        df = pd.DataFrame({"close": 100.0, "_symbol": SYM}, index=bar_idx)
        out = compute_funding_rate_features_from_df(df, funding_rate_dir=str(tmp_path))

        change = out["funding_rate_change_1"]
        # bar 00:00 → funding 00:00 (0.001), change = NaN (first observation)
        assert np.isnan(change.iloc[0])
        # bar 04:00 → funding 00:00 (0.001), change = NaN (still first, same funding obs)
        assert np.isnan(change.iloc[1])
        # bar 08:00 → funding 08:00 (0.003), change = 0.003 - 0.001 = 0.002
        assert np.isclose(change.iloc[2], 0.002)
        # bar 12:00 → funding 08:00 (0.003), change = 0.002 (same funding obs)
        assert np.isclose(change.iloc[3], 0.002)
        # bar 16:00 → funding 16:00 (-0.001), change = -0.001 - 0.003 = -0.004
        assert np.isclose(change.iloc[4], -0.004)

    def test_zscore_valid_after_warmup(self, tmp_path):
        """After z_min_periods warmup, zscore should have non-NaN values."""
        rng = np.random.RandomState(42)
        _write_funding_parquet(tmp_path, 0.0001 + rng.randn(300) * 0.0005)
        bars = _make_bars(200)
        out = compute_funding_rate_features_from_df(
            bars, funding_rate_dir=str(tmp_path), z_window=50, z_min_periods=20
        )
        valid = out["funding_rate_zscore_50"].dropna()
        assert len(valid) > 50, f"Expected >50 valid zscore values, got {len(valid)}"

    def test_graceful_missing_data(self):
        """on_missing='nan' should produce all-NaN output without raising."""
        bars = _make_bars(10)
        out = compute_funding_rate_features_from_df(
            bars, funding_rate_dir="/nonexistent/path", on_missing="nan"
        )
        assert isinstance(out, pd.DataFrame)
        assert len(out) == 10
        assert set(ALL_COLS).issubset(set(out.columns))

    def test_raises_on_missing_when_configured(self):
        """on_missing='raise' should raise FileNotFoundError."""
        bars = _make_bars(10)
        with pytest.raises(FileNotFoundError):
            compute_funding_rate_features_from_df(
                bars, funding_rate_dir="/nonexistent/path", on_missing="raise"
            )

    def test_requires_symbol_column(self):
        """Should raise KeyError if no _symbol column."""
        idx = pd.date_range("2024-01-01", periods=5, freq="4h", tz="UTC")
        df = pd.DataFrame({"close": range(5)}, index=idx)
        with pytest.raises(KeyError, match="_symbol"):
            compute_funding_rate_features_from_df(df, on_missing="raise")

    def test_requires_datetimeindex(self):
        """Should raise ValueError if index is not DatetimeIndex."""
        df = pd.DataFrame({"close": [1.0], "_symbol": [SYM]})
        with pytest.raises(ValueError, match="DatetimeIndex"):
            compute_funding_rate_features_from_df(df)

    def test_zscore_computed_on_native_not_bars(self, tmp_path):
        """Verify zscore is NOT just bar-level rolling (which would be biased).

        With 4h bars and 8h funding, consecutive bars see the same funding value.
        If zscore were computed on bar-level, these duplicates would produce
        artificially low std → inflated z-scores. Native computation avoids this.
        """
        # Create funding with clear pattern: positive, then spike
        n_fr = 60
        fr_vals = np.full(n_fr, 0.0001)
        fr_vals[50] = 0.01  # spike at observation 50
        _write_funding_parquet(tmp_path, fr_vals)

        # 4h bars: 2 bars per funding observation
        bars = _make_bars(120, freq="4h")
        out = compute_funding_rate_features_from_df(
            bars, funding_rate_dir=str(tmp_path), z_window=30, z_min_periods=10
        )

        # The two bars that see the same funding observation should have
        # IDENTICAL zscore values (since zscore was computed on native freq)
        z = out["funding_rate_zscore_50"]
        # Find pairs of bars mapping to same funding
        for i in range(2, len(z) - 1, 2):
            if pd.notna(z.iloc[i]) and pd.notna(z.iloc[i + 1]):
                # Both bars see the same funding obs → same zscore
                assert np.isclose(z.iloc[i], z.iloc[i + 1], rtol=1e-10), (
                    f"Bar {i} and {i+1} should have same zscore but got "
                    f"{z.iloc[i]:.6f} vs {z.iloc[i+1]:.6f}"
                )


# ─────────────────────────────────────────────────────────────
# 5. Scene semantic scores
# ─────────────────────────────────────────────────────────────


class TestFundingSceneScores:
    """Test funding scene semantic score computation."""

    def test_bounded_0_1(self):
        idx = pd.date_range("2024-01-01", periods=10, freq="4h", tz="UTC")
        df = pd.DataFrame(
            {
                "funding_rate_abs_zscore_50": np.linspace(0.0, 4.0, 10),
                "compression_score": np.linspace(0.0, 1.0, 10),
                "trend_r2_20": np.linspace(1.0, 0.0, 10),
            },
            index=idx,
        )
        out = compute_funding_scene_semantic_scores_from_df(df)
        for c in [
            "funding_compression_score",
            "funding_ignition_score",
            "funding_absorption_score",
            "funding_exhaustion_scene_score",
        ]:
            assert c in out.columns
            assert ((out[c] >= 0.0) & (out[c] <= 1.0)).all()

    def test_no_look_ahead_pointwise(self):
        """Scene scores are pointwise → first K rows should match."""
        n = 50
        idx = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC")
        rng = np.random.RandomState(42)
        df = pd.DataFrame(
            {
                "funding_rate_abs_zscore_50": rng.randn(n) * 2,
                "compression_score": rng.rand(n),
                "trend_r2_20": rng.rand(n),
            },
            index=idx,
        )
        full = compute_funding_scene_semantic_scores_from_df(df)
        half = compute_funding_scene_semantic_scores_from_df(df.iloc[:25])
        for col in full.columns:
            pd.testing.assert_series_equal(
                half[col], full[col].iloc[:25], check_names=False
            )

    def test_requires_datetimeindex(self):
        df = pd.DataFrame(
            {
                "funding_rate_abs_zscore_50": [1.0],
                "compression_score": [0.5],
                "trend_r2_20": [0.5],
            }
        )
        with pytest.raises(ValueError, match="DatetimeIndex"):
            compute_funding_scene_semantic_scores_from_df(df)
