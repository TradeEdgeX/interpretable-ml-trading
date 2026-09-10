"""Tests for OI (Open Interest) feature functions.

Covers:
- compute_oi_features_from_df: shape, bounded, no look-ahead, NaN handling
- compute_oi_scene_semantic_scores_from_df: shape, bounded [0,1], math correctness
- oi_flow_zscore end-to-end: no look-ahead, streaming consistency, functional correctness
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.time_series.open_interest_features import (
    compute_oi_features_from_df,
    compute_oi_scene_semantic_scores_from_df,
)

# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────


def _make_bar_df(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """Create a minimal kline-like DataFrame with DatetimeIndex."""
    rng = np.random.RandomState(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC")
    close = 50000.0 + rng.randn(n).cumsum() * 100
    return pd.DataFrame(
        {
            "open": close + rng.randn(n) * 10,
            "high": close + abs(rng.randn(n)) * 50,
            "low": close - abs(rng.randn(n)) * 50,
            "close": close,
            "volume": abs(rng.randn(n)) * 1e6 + 1e5,
            "_symbol": "BTCUSDT",
        },
        index=idx,
    )


def _make_scene_df(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """Create a DataFrame with the required columns for scene semantic scores."""
    rng = np.random.RandomState(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC")
    return pd.DataFrame(
        {
            "oi_zscore": rng.randn(n) * 2,
            "oi_flow_zscore": rng.randn(n) * 2,
            "compression_score": rng.rand(n),
            "trend_r2_20": rng.rand(n),
            "atr_percentile": rng.rand(n),
        },
        index=idx,
    )


# ─────────────────────────────────────────────────────────────
# Tests for compute_oi_scene_semantic_scores_from_df
# (These don't need actual parquet files)
# ─────────────────────────────────────────────────────────────


class TestOISceneSemanticScores:
    """Test OI scene semantic score computation."""

    def test_output_shape_and_columns(self):
        df = _make_scene_df(200)
        result = compute_oi_scene_semantic_scores_from_df(df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 200
        expected = {
            "oi_compression_score",
            "oi_ignition_score",
            "oi_absorption_score",
            "oi_exhaustion_score",
            "oi_trend_divergence_score",
        }
        assert expected.issubset(set(result.columns))

    def test_bounded_0_1(self):
        df = _make_scene_df(500, seed=99)
        result = compute_oi_scene_semantic_scores_from_df(df)
        for col in result.columns:
            assert result[col].min() >= -1e-9, f"{col} below 0"
            assert result[col].max() <= 1.0 + 1e-9, f"{col} above 1"

    def test_compression_high_when_oi_building_compressed_no_trend(self):
        """When OI z-score high, compression high, trend low → compression score should be high."""
        n = 100
        idx = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC")
        df = pd.DataFrame(
            {
                "oi_zscore": [3.0] * n,  # high OI activity
                "compression_score": [0.9] * n,  # high compression
                "trend_r2_20": [0.1] * n,  # low trend
            },
            index=idx,
        )
        result = compute_oi_scene_semantic_scores_from_df(df)
        # compression_score = sigmoid((3-0.5)/1) * 0.9 * 0.9 ≈ 0.924 * 0.81 ≈ 0.749
        assert result["oi_compression_score"].mean() > 0.5

    def test_ignition_high_when_oi_building_trend_present(self):
        """When OI z-score high, no compression, high trend → ignition should be high."""
        n = 100
        idx = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC")
        df = pd.DataFrame(
            {
                "oi_zscore": [3.0] * n,
                "compression_score": [0.1] * n,
                "trend_r2_20": [0.9] * n,
            },
            index=idx,
        )
        result = compute_oi_scene_semantic_scores_from_df(df)
        assert result["oi_ignition_score"].mean() > 0.5

    def test_exhaustion_high_when_oi_unwinding(self):
        """When OI z-score very negative (unwinding), no comp, no trend → exhaustion high."""
        n = 100
        idx = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC")
        df = pd.DataFrame(
            {
                "oi_zscore": [-3.0] * n,  # OI declining = unwinding
                "compression_score": [0.1] * n,
                "trend_r2_20": [0.1] * n,
            },
            index=idx,
        )
        result = compute_oi_scene_semantic_scores_from_df(df)
        # unwinding = 1 - sigmoid((-3-0.5)/1) = 1 - sigmoid(-3.5) ≈ 0.971
        # exhaustion = 0.971 * 0.9 * 0.9 ≈ 0.786
        assert result["oi_exhaustion_score"].mean() > 0.5

    def test_nan_handling(self):
        """NaN in inputs should produce valid (not NaN) outputs due to fillna."""
        n = 50
        idx = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC")
        df = pd.DataFrame(
            {
                "oi_zscore": [np.nan] * n,
                "oi_flow_zscore": [np.nan] * n,
                "compression_score": [np.nan] * n,
                "trend_r2_20": [np.nan] * n,
            },
            index=idx,
        )
        result = compute_oi_scene_semantic_scores_from_df(df)
        # Should not raise, all NaN → fillna(0) → produce valid numbers
        assert not result.isna().all().all()

    def test_no_look_ahead(self):
        """Row i output should only depend on row i inputs (pointwise)."""
        n = 100
        idx = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC")
        rng = np.random.RandomState(123)
        oi_z = rng.randn(n) * 2
        oi_flow_z = rng.randn(n) * 2
        comp = rng.rand(n)
        trend = rng.rand(n)

        df_full = pd.DataFrame(
            {
                "oi_zscore": oi_z,
                "oi_flow_zscore": oi_flow_z,
                "compression_score": comp,
                "trend_r2_20": trend,
            },
            index=idx,
        )
        result_full = compute_oi_scene_semantic_scores_from_df(df_full)

        # Compute first 50 rows alone
        df_half = df_full.iloc[:50].copy()
        result_half = compute_oi_scene_semantic_scores_from_df(df_half)

        # First 50 rows should be identical
        for col in result_full.columns:
            pd.testing.assert_series_equal(
                result_half[col],
                result_full[col].iloc[:50],
                check_names=False,
            )

    def test_requires_datetimeindex(self):
        """Should raise if index is not DatetimeIndex."""
        df = pd.DataFrame(
            {"oi_zscore": [1.0], "compression_score": [0.5], "trend_r2_20": [0.5]}
        )
        with pytest.raises(ValueError, match="DatetimeIndex"):
            compute_oi_scene_semantic_scores_from_df(df)

    def test_missing_column_graceful(self):
        """When columns are missing, should use default (0.0) via .get()."""
        n = 50
        idx = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC")
        # Only provide oi_zscore, missing compression_score and trend_r2_20
        df = pd.DataFrame({"oi_zscore": np.random.randn(n)}, index=idx)
        # The function uses df.get() which returns Series of NaN, then fillna(0)
        result = compute_oi_scene_semantic_scores_from_df(df)
        assert len(result) == n
        assert set(result.columns) == {
            "oi_compression_score",
            "oi_ignition_score",
            "oi_absorption_score",
            "oi_exhaustion_score",
            "oi_trend_divergence_score",
        }

    def test_trend_divergence_high_when_trend_strong_flow_stalled(self):
        """When trend high, OI flow negative, volatility high → divergence high.

        This is the ME → FER “last leg” signal: price trending on existing
        positions, no new capital, volatility elevated.
        Formula: trend × sigmoid(-flow_z / 1.5) × atr_percentile
        """
        n = 100
        idx = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC")
        df = pd.DataFrame(
            {
                "oi_zscore": [1.0] * n,
                "oi_flow_zscore": [-2.0] * n,  # flow NEGATIVE = no new capital
                "compression_score": [0.3] * n,
                "trend_r2_20": [0.9] * n,  # strong trend
                "atr_percentile": [0.85] * n,  # high volatility
            },
            index=idx,
        )
        result = compute_oi_scene_semantic_scores_from_df(df)
        # sigmoid(-(-2)/1.5) = sigmoid(1.33) ≈ 0.791
        # divergence = 0.9 × 0.791 × 0.85 ≈ 0.605
        assert result["oi_trend_divergence_score"].mean() > 0.5

    def test_trend_divergence_low_when_flow_strong(self):
        """When trend high AND OI flow also high → divergence should be low.

        Healthy trend with participation = no divergence.
        """
        n = 100
        idx = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC")
        df = pd.DataFrame(
            {
                "oi_zscore": [1.0] * n,
                "oi_flow_zscore": [3.0] * n,  # flow STRONG = new capital entering
                "compression_score": [0.3] * n,
                "trend_r2_20": [0.9] * n,
                "atr_percentile": [0.85] * n,  # even with high vol, flow kills it
            },
            index=idx,
        )
        result = compute_oi_scene_semantic_scores_from_df(df)
        # sigmoid(-3/1.5) = sigmoid(-2) ≈ 0.119
        # divergence = 0.9 × 0.119 × 0.85 ≈ 0.091
        assert result["oi_trend_divergence_score"].mean() < 0.15

    def test_trend_divergence_suppressed_by_low_volatility(self):
        """Even with negative flow + trend, low volatility suppresses score.

        Low volatility = quiet decay, not the “last leg” blowoff.
        """
        n = 100
        idx = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC")
        df = pd.DataFrame(
            {
                "oi_zscore": [1.0] * n,
                "oi_flow_zscore": [-2.0] * n,  # flow negative
                "compression_score": [0.3] * n,
                "trend_r2_20": [0.9] * n,  # strong trend
                "atr_percentile": [0.15] * n,  # LOW volatility
            },
            index=idx,
        )
        result = compute_oi_scene_semantic_scores_from_df(df)
        # sigmoid(2/1.5) ≈ 0.791
        # divergence = 0.9 × 0.791 × 0.15 ≈ 0.107
        assert result["oi_trend_divergence_score"].mean() < 0.15


class TestOIFlowZscoreEndToEnd:
    """End-to-end tests for oi_flow_zscore with mock OI parquet data.

    Covers:
    - No look-ahead: future bars do not change past values
    - Streaming consistency: first K bars identical when computed alone vs full
    - Functional correctness: flow_zscore semantics
    """

    @staticmethod
    def _write_oi_parquet(oi_dir, oi_usd: np.ndarray) -> None:
        """Write a mock OI parquet file into *oi_dir*."""
        n = len(oi_usd)
        idx = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
        oi_df = pd.DataFrame(
            {"oi_usd": oi_usd, "oi_contracts": oi_usd / 50000.0},
            index=idx,
        )
        oi_df.index.name = "datetime"
        oi_df.to_parquet(oi_dir / "BTCUSDT_2024-01_oi_5m.parquet")

    def test_no_look_ahead_and_streaming(self, tmp_path):
        """First K bars must be identical whether computed alone or with future bars.

        Simultaneously verifies:
        1. No look-ahead: future data does not contaminate past values
        2. Streaming consistency: appending new bars preserves old results
        """
        rng = np.random.RandomState(42)
        n_oi = 17280  # ~60 days at 5min
        oi_usd = 1e9 + rng.randn(n_oi).cumsum() * 1e6
        self._write_oi_parquet(tmp_path, oi_usd)

        bars = _make_bar_df(200)
        k = 80  # split point

        result_full = compute_oi_features_from_df(
            bars, oi_dir=str(tmp_path), z_window=50, z_min_periods=20
        )
        result_prefix = compute_oi_features_from_df(
            bars.iloc[:k], oi_dir=str(tmp_path), z_window=50, z_min_periods=20
        )

        for col in [
            "oi_usd",
            "oi_change_pct",
            "oi_zscore",
            "oi_delta_price_sign",
            "oi_flow_zscore",
        ]:
            pd.testing.assert_series_equal(
                result_prefix[col],
                result_full[col].iloc[:k],
                check_names=False,
                rtol=1e-10,
            )

    def test_flow_zscore_valid_after_warmup(self, tmp_path):
        """After z_min_periods warmup, oi_flow_zscore must have non-NaN values."""
        rng = np.random.RandomState(42)
        n_oi = 17280
        oi_usd = 1e9 + rng.randn(n_oi).cumsum() * 1e6
        self._write_oi_parquet(tmp_path, oi_usd)

        bars = _make_bar_df(200)
        result = compute_oi_features_from_df(
            bars, oi_dir=str(tmp_path), z_window=50, z_min_periods=20
        )

        valid = result["oi_flow_zscore"].dropna()
        # After warmup: 1 bar for diff() + z_min_periods → expect >100 valid
        assert (
            len(valid) > 100
        ), f"Expected >100 valid flow_zscore values, got {len(valid)}"

    def test_flow_zscore_positive_when_oi_accelerating(self, tmp_path):
        """When OI increase accelerates (quadratic), flow_zscore should be positive.

        Semantic: \u0394OI keeps growing \u2192 recent \u0394OI > rolling mean \u0394OI \u2192 z > 0.
        """
        n_oi = 17280
        t = np.arange(n_oi, dtype=float)
        oi_usd = 1e9 + (t**2) * 10.0  # quadratic = accelerating OI
        self._write_oi_parquet(tmp_path, oi_usd)

        bars = _make_bar_df(200)
        result = compute_oi_features_from_df(
            bars, oi_dir=str(tmp_path), z_window=50, z_min_periods=20
        )

        last_50 = result["oi_flow_zscore"].iloc[-50:]
        valid = last_50.dropna()
        assert len(valid) > 0, "No valid flow_zscore in last 50 bars"
        assert (
            valid.mean() > 0
        ), f"Expected positive flow_zscore when OI accelerating, got mean={valid.mean():.4f}"

    def test_flow_zscore_negative_when_oi_decelerating(self, tmp_path):
        """When OI increase decelerates (sqrt growth), flow_zscore should be negative.

        Semantic: \u0394OI shrinking over time \u2192 recent \u0394OI < rolling mean \u2192 z < 0.
        """
        n_oi = 17280
        t = np.arange(1, n_oi + 1, dtype=float)
        oi_usd = 1e9 + np.sqrt(t) * 1e6  # sqrt = decelerating OI
        self._write_oi_parquet(tmp_path, oi_usd)

        bars = _make_bar_df(200)
        result = compute_oi_features_from_df(
            bars, oi_dir=str(tmp_path), z_window=50, z_min_periods=20
        )

        last_50 = result["oi_flow_zscore"].iloc[-50:]
        valid = last_50.dropna()
        assert len(valid) > 0, "No valid flow_zscore in last 50 bars"
        assert (
            valid.mean() < 0
        ), f"Expected negative flow_zscore when OI decelerating, got mean={valid.mean():.4f}"


class TestOIFeaturesFromDf:
    """Test raw OI feature extraction (requires mock or on_missing='nan')."""

    def test_graceful_missing_data(self):
        """When OI data doesn't exist, on_missing='nan' should produce NaN columns."""
        df = _make_bar_df(50)
        result = compute_oi_features_from_df(
            df, oi_dir="/nonexistent/path", on_missing="nan"
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 50
        expected_cols = {
            "oi_usd",
            "oi_change_pct",
            "oi_zscore",
            "oi_delta_price_sign",
            "oi_flow_zscore",
        }
        assert expected_cols.issubset(set(result.columns))

    def test_raises_on_missing_when_configured(self):
        """When on_missing='raise', should raise if no OI data."""
        df = _make_bar_df(10)
        with pytest.raises(FileNotFoundError):
            compute_oi_features_from_df(
                df, oi_dir="/nonexistent/path", on_missing="raise"
            )

    def test_requires_symbol_column(self):
        """Should raise if neither _symbol nor symbol column exists."""
        idx = pd.date_range("2024-01-01", periods=10, freq="4h", tz="UTC")
        df = pd.DataFrame({"close": range(10)}, index=idx)
        with pytest.raises(KeyError, match="_symbol"):
            compute_oi_features_from_df(df, on_missing="raise")

    def test_requires_datetimeindex(self):
        """Should raise if index is not DatetimeIndex."""
        df = pd.DataFrame({"close": [1.0], "_symbol": ["BTCUSDT"]})
        with pytest.raises(ValueError, match="DatetimeIndex"):
            compute_oi_features_from_df(df)
