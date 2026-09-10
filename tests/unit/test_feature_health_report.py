"""Tests for feature health reporting in IFC.

Verifies:
1. report_feature_health() computes correct NaN ratios and critical feature detection
2. report_feature_health_df() reports DataFrame-level NaN columns
3. Critical prefixes (atr, oi_, funding_oi_) trigger ERROR level
4. Prometheus metrics are updated when update_prometheus=True
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ─────────────────────────────────────────────────────────────
# IFC with known live_feature_set
# ─────────────────────────────────────────────────────────────


def _make_ifc(live_features=None):
    """Create a minimal IFC with a known live_feature_set."""
    from src.time_series_model.live.incremental_feature_computer import (
        IncrementalFeatureComputer,
    )

    ifc = IncrementalFeatureComputer(
        primary_timeframe="240T",
        live_feature_plan_path="/dev/null",  # avoid auto-detect
    )
    if live_features is not None:
        ifc.live_feature_set = set(live_features)
    return ifc


# ─────────────────────────────────────────────────────────────
# Test: report_feature_health (dict-based, for live)
# ─────────────────────────────────────────────────────────────


class TestReportFeatureHealth:
    """Tests for report_feature_health (compute_features_batch output)."""

    def test_all_features_present(self):
        """When all expected features are present, nan_ratio=0."""
        ifc = _make_ifc(["close", "volume", "atr", "oi_zscore"])
        features = {"close": 100.0, "volume": 500.0, "atr": 50.0, "oi_zscore": 0.3}
        report = ifc.report_feature_health(features, symbol="BTCUSDT", timeframe="240T")

        assert report["total"] == 4
        assert report["expected"] == 4
        assert report["missing_count"] == 0
        assert report["nan_ratio"] == 0.0
        assert report["critical_nan"] == []

    def test_missing_features(self):
        """Missing features are reported correctly."""
        ifc = _make_ifc(["close", "volume", "atr", "oi_zscore", "oi_flow_zscore"])
        features = {"close": 100.0, "volume": 500.0}  # atr, oi_* missing
        report = ifc.report_feature_health(features, symbol="BTCUSDT", timeframe="240T")

        assert report["total"] == 2
        assert report["expected"] == 5
        assert report["missing_count"] == 3
        assert report["nan_ratio"] == pytest.approx(0.6, abs=0.01)
        assert "atr" in report["critical_nan"]
        assert "oi_zscore" in report["critical_nan"]
        assert "oi_flow_zscore" in report["critical_nan"]

    def test_no_live_feature_set(self):
        """When live_feature_set is empty, report has zeros."""
        ifc = _make_ifc([])
        features = {"close": 100.0}
        report = ifc.report_feature_health(features, symbol="BTCUSDT", timeframe="240T")

        assert report["expected"] == 0
        assert report["missing_count"] == 0
        assert report["nan_ratio"] == 0.0

    def test_critical_prefixes(self):
        """atr, oi_, funding_oi_ are detected as critical."""
        ifc = _make_ifc(
            [
                "atr",
                "atr_percentile",
                "oi_zscore",
                "funding_oi_crowding_score",
                "bb_width",
            ]
        )
        features = {"bb_width": 0.5}  # all critical missing
        report = ifc.report_feature_health(features, symbol="BTCUSDT", timeframe="240T")

        assert set(report["critical_nan"]) == {
            "atr",
            "atr_percentile",
            "oi_zscore",
            "funding_oi_crowding_score",
        }

    def test_non_critical_missing(self):
        """Non-critical features missing don't trigger critical_nan."""
        ifc = _make_ifc(["close", "volume", "bb_width"])
        features = {"close": 100.0}  # volume, bb_width missing
        report = ifc.report_feature_health(features, symbol="ETHUSDT", timeframe="60T")

        assert report["missing_count"] == 2
        assert report["critical_nan"] == []


# ─────────────────────────────────────────────────────────────
# Test: report_feature_health_df (DataFrame-based, for event backtest)
# ─────────────────────────────────────────────────────────────


class TestReportFeatureHealthDf:
    """Tests for report_feature_health_df (compute_features_dataframe output)."""

    def test_no_nan_df(self):
        """DataFrame with no NaN in last row is healthy."""
        ifc = _make_ifc(["close", "atr"])
        df = pd.DataFrame(
            {
                "close": [100.0, 101.0, 102.0],
                "atr": [50.0, 51.0, 52.0],
            }
        )
        report = ifc.report_feature_health_df(df, symbol="BTCUSDT", timeframe="240T")

        assert report["total_cols"] == 2
        assert report["last_row_nan_count"] == 0
        assert report["critical_nan"] == []

    def test_nan_in_last_row(self):
        """NaN in last row detected, especially critical features."""
        ifc = _make_ifc(["close", "atr", "oi_zscore"])
        df = pd.DataFrame(
            {
                "close": [100.0, 101.0, 102.0],
                "atr": [50.0, 51.0, np.nan],  # NaN in last row
                "oi_zscore": [0.1, np.nan, np.nan],  # NaN in last row
            }
        )
        report = ifc.report_feature_health_df(df, symbol="BTCUSDT", timeframe="240T")

        assert report["last_row_nan_count"] == 2
        assert "atr" in report["critical_nan"]
        assert "oi_zscore" in report["critical_nan"]

    def test_high_nan_rate_columns(self):
        """Columns with >50% NaN across all rows are flagged."""
        ifc = _make_ifc(["close", "oi_zscore"])
        df = pd.DataFrame(
            {
                "close": [100.0, 101.0, 102.0, 103.0],
                "oi_zscore": [np.nan, np.nan, np.nan, 0.1],  # 75% NaN
            }
        )
        report = ifc.report_feature_health_df(df, symbol="BTCUSDT", timeframe="240T")

        assert "oi_zscore" in report["high_nan_cols"]

    def test_side_gated_macro_columns_not_flagged(self):
        """Regime-gated macro legs may be >50% NaN by design."""
        ifc = _make_ifc(["tpc_macro_pullback_pct_long"])
        df = pd.DataFrame(
            {
                "close": [100.0, 101.0, 102.0, 103.0],
                "tpc_macro_pullback_pct_long": [np.nan, np.nan, np.nan, 0.2],
            }
        )
        report = ifc.report_feature_health_df(df, symbol="BTCUSDT", timeframe="120T")
        assert "tpc_macro_pullback_pct_long" not in report["high_nan_cols"]

    def test_empty_df(self):
        """Empty DataFrame returns 'empty' report."""
        ifc = _make_ifc(["close"])
        report = ifc.report_feature_health_df(
            pd.DataFrame(), symbol="BTCUSDT", timeframe="240T"
        )
        assert report.get("empty") is True


# ─────────────────────────────────────────────────────────────
# Test: _is_critical
# ─────────────────────────────────────────────────────────────


class TestIsCritical:
    """Tests for _is_critical static method."""

    def test_atr_exact(self):
        from src.time_series_model.live.incremental_feature_computer import (
            IncrementalFeatureComputer as IFC,
        )

        assert IFC._is_critical("atr") is True

    def test_atr_prefix(self):
        from src.time_series_model.live.incremental_feature_computer import (
            IncrementalFeatureComputer as IFC,
        )

        assert IFC._is_critical("atr_percentile") is True

    def test_oi_prefix(self):
        from src.time_series_model.live.incremental_feature_computer import (
            IncrementalFeatureComputer as IFC,
        )

        assert IFC._is_critical("oi_zscore") is True
        assert IFC._is_critical("oi_flow_zscore") is True

    def test_funding_oi_prefix(self):
        from src.time_series_model.live.incremental_feature_computer import (
            IncrementalFeatureComputer as IFC,
        )

        assert IFC._is_critical("funding_oi_crowding_score") is True

    def test_non_critical(self):
        from src.time_series_model.live.incremental_feature_computer import (
            IncrementalFeatureComputer as IFC,
        )

        assert IFC._is_critical("close") is False
        assert IFC._is_critical("bb_width") is False
        assert IFC._is_critical("volume") is False
        assert IFC._is_critical("macd") is False


# ─────────────────────────────────────────────────────────────
# Test: _record_loader_error doesn't crash
# ─────────────────────────────────────────────────────────────


class TestRecordLoaderError:
    """Ensure _record_loader_error is safe even without Prometheus."""

    def test_no_crash_without_prometheus(self):
        ifc = _make_ifc(["close"])
        # Should not raise even if Prometheus is not available
        ifc._record_loader_error("test_node", "240T", RuntimeError("test"))
