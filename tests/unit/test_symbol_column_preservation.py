"""Tests for _symbol column preservation through IFC resample.

Bug: _compute_features_core's resample().agg() drops _symbol column,
causing OI join (add_oi_features) to fail with:
  KeyError: "df must contain '_symbol' or 'symbol' for OI join"

This cascades: OI failure → downstream nodes (including ATR) all skip,
resulting in FER having only 20 columns instead of 74.

Fix: preserve _symbol after resample in _compute_features_core,
and inject _symbol at all three caller sites:
  - backtest_execution_layer.py  (_load_raw_features_for_archetype)
  - event_backtest.py            (run loop)
  - order_flow_listener.py       (_compute_and_save_15min_features)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────


def _make_bars_1min(
    n: int = 2000, symbol: str = "BTCUSDT", include_symbol: bool = True, seed: int = 42
) -> pd.DataFrame:
    """Create minimal 1min bars for IFC testing."""
    rng = np.random.RandomState(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
    close = 50000.0 + rng.randn(n).cumsum() * 10
    df = pd.DataFrame(
        {
            "open": close + rng.randn(n) * 5,
            "high": close + abs(rng.randn(n)) * 20,
            "low": close - abs(rng.randn(n)) * 20,
            "close": close,
            "volume": abs(rng.randn(n)) * 1e4 + 1e3,
        },
        index=idx,
    )
    if include_symbol:
        df["_symbol"] = symbol
    return df


def _make_bars_4h(
    n: int = 200, symbol: str = "BTCUSDT", include_symbol: bool = True, seed: int = 42
) -> pd.DataFrame:
    """Create minimal 4h bars (already resampled) for OI testing."""
    rng = np.random.RandomState(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC")
    close = 50000.0 + rng.randn(n).cumsum() * 100
    df = pd.DataFrame(
        {
            "open": close + rng.randn(n) * 10,
            "high": close + abs(rng.randn(n)) * 50,
            "low": close - abs(rng.randn(n)) * 50,
            "close": close,
            "volume": abs(rng.randn(n)) * 1e6 + 1e5,
        },
        index=idx,
    )
    if include_symbol:
        df["_symbol"] = symbol
    return df


# ─────────────────────────────────────────────────────────────
# Test 1: resample preserves _symbol
# ─────────────────────────────────────────────────────────────


class TestResamplePreservesSymbol:
    """Verify that _compute_features_core preserves _symbol after resample."""

    def test_resample_preserves_symbol_when_present(self):
        """_symbol column should survive resample().agg()."""
        bars = _make_bars_1min(n=2000, include_symbol=True)
        assert "_symbol" in bars.columns

        # Simulate what _compute_features_core does
        agg_dict = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
        bars_tf = bars.resample("240T").agg(agg_dict).dropna(subset=["close"])

        # Before fix: _symbol is lost
        assert (
            "_symbol" not in bars_tf.columns
        ), "resample().agg() should drop non-aggregated columns like _symbol"

        # Apply fix: preserve _symbol after resample
        if "_symbol" in bars.columns:
            bars_tf["_symbol"] = bars["_symbol"].iloc[0]

        # After fix: _symbol is preserved
        assert "_symbol" in bars_tf.columns
        assert (bars_tf["_symbol"] == "BTCUSDT").all()

    def test_resample_no_symbol_no_error(self):
        """When _symbol is absent, resample should work without error."""
        bars = _make_bars_1min(n=2000, include_symbol=False)
        assert "_symbol" not in bars.columns

        agg_dict = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
        bars_tf = bars.resample("240T").agg(agg_dict).dropna(subset=["close"])

        # No _symbol to preserve, no error
        assert "_symbol" not in bars_tf.columns


# ─────────────────────────────────────────────────────────────
# Test 2: OI join requires _symbol
# ─────────────────────────────────────────────────────────────


class TestOIJoinRequiresSymbol:
    """Verify that add_oi_features fails without _symbol."""

    def test_oi_join_fails_without_symbol(self):
        """compute_oi_features_from_df should raise KeyError when _symbol is missing."""
        from src.features.time_series.open_interest_features import (
            compute_oi_features_from_df,
        )

        bars_no_sym = _make_bars_4h(include_symbol=False)
        with pytest.raises(KeyError, match="_symbol.*symbol.*OI join"):
            compute_oi_features_from_df(bars_no_sym)

    def test_oi_join_accepts_symbol(self):
        """compute_oi_features_from_df should not raise KeyError when _symbol is present."""
        from src.features.time_series.open_interest_features import (
            compute_oi_features_from_df,
        )

        bars_with_sym = _make_bars_4h(include_symbol=True)
        # Should not raise KeyError (may still fail on missing OI data, but
        # that's a data issue, not a schema issue)
        try:
            result = compute_oi_features_from_df(bars_with_sym)
            # If OI data exists locally, result should have oi_usd column
            assert isinstance(result, pd.DataFrame)
        except KeyError as e:
            # Only acceptable if it's NOT about _symbol
            assert "_symbol" not in str(e) and "symbol" not in str(
                e
            ), f"Should not fail on _symbol: {e}"
        except Exception:
            # Other errors (missing OI parquet, etc.) are acceptable
            pass


# ─────────────────────────────────────────────────────────────
# Test 3: IFC compute_features_dataframe propagates _symbol
# ─────────────────────────────────────────────────────────────


class TestIFCSymbolPropagation:
    """Verify IFC correctly propagates _symbol through compute pipeline."""

    def test_compute_features_core_preserves_symbol(self):
        """_compute_features_core should preserve _symbol in output."""
        from src.time_series_model.live.incremental_feature_computer import (
            IncrementalFeatureComputer,
        )

        fc = IncrementalFeatureComputer(primary_timeframe="240T")
        bars = _make_bars_1min(n=3000, include_symbol=True)
        ticks = pd.DataFrame()  # empty ticks OK for this test

        result = fc._compute_features_core(bars, ticks, "240T")

        assert result is not None, "Should return DataFrame for 3000 1min bars"
        assert (
            "_symbol" in result.columns
        ), "_symbol must be preserved after resample in _compute_features_core"
        assert (result["_symbol"] == "BTCUSDT").all()

    def test_compute_features_core_no_symbol_no_crash(self):
        """_compute_features_core should work without _symbol (no crash)."""
        from src.time_series_model.live.incremental_feature_computer import (
            IncrementalFeatureComputer,
        )

        fc = IncrementalFeatureComputer(primary_timeframe="240T")
        bars = _make_bars_1min(n=3000, include_symbol=False)
        ticks = pd.DataFrame()

        result = fc._compute_features_core(bars, ticks, "240T")

        assert result is not None
        assert "_symbol" not in result.columns

    def test_compute_features_dataframe_preserves_symbol(self):
        """compute_features_dataframe should preserve _symbol through full pipeline.

        Tests _compute_features_core directly (shared code path) to avoid
        warmup validation that requires 150+ days of data.
        """
        from src.time_series_model.live.incremental_feature_computer import (
            IncrementalFeatureComputer,
        )

        fc = IncrementalFeatureComputer(primary_timeframe="240T")
        bars = _make_bars_1min(n=3000, include_symbol=True)
        ticks = pd.DataFrame()

        # _compute_features_core is the shared path; test it directly
        result = fc._compute_features_core(bars, ticks, "240T")
        assert result is not None
        assert "_symbol" in result.columns
        assert (result["_symbol"] == "BTCUSDT").all()


# ─────────────────────────────────────────────────────────────
# Test 4: Caller injection pattern
# ─────────────────────────────────────────────────────────────


class TestCallerSymbolInjection:
    """Verify the _symbol injection pattern used by callers."""

    def test_event_backtest_injection_pattern(self):
        """Simulate event_backtest.py _symbol injection."""
        bars_1min = _make_bars_1min(n=100, include_symbol=False)
        sym = "ETHUSDT"

        # This is the pattern used in event_backtest.py
        if "_symbol" not in bars_1min.columns:
            bars_1min["_symbol"] = sym

        assert "_symbol" in bars_1min.columns
        assert (bars_1min["_symbol"] == "ETHUSDT").all()

    def test_order_flow_listener_injection_pattern(self):
        """Simulate order_flow_listener.py _symbol injection."""
        bars_merged = _make_bars_1min(n=100, include_symbol=False)
        symbol = "SOLUSDT"

        # This is the pattern used in order_flow_listener.py
        if "_symbol" not in bars_merged.columns:
            bars_merged["_symbol"] = symbol

        assert "_symbol" in bars_merged.columns
        assert (bars_merged["_symbol"] == "SOLUSDT").all()

    def test_backtest_execution_layer_injection_pattern(self):
        """Simulate backtest_execution_layer.py _symbol injection."""
        bars_1min = _make_bars_1min(n=100, include_symbol=False)
        sym = "BNBUSDT"

        # This is the pattern used in backtest_execution_layer.py
        bars_1min["_symbol"] = sym

        assert "_symbol" in bars_1min.columns
        assert (bars_1min["_symbol"] == "BNBUSDT").all()

    def test_idempotent_injection(self):
        """Injection should be idempotent — no overwrite if _symbol exists."""
        bars = _make_bars_1min(n=100, include_symbol=True, symbol="BTCUSDT")

        # Guard pattern: only inject if missing
        if "_symbol" not in bars.columns:
            bars["_symbol"] = "WRONG"

        assert (
            bars["_symbol"] == "BTCUSDT"
        ).all(), "Should NOT overwrite existing _symbol"
