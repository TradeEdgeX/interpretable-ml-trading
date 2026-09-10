"""
Tests for incremental feature store build logic.

Covers:
1. _find_missing_features: detect which features need computing
2. _find_donor_months: cross-layer reuse discovery
3. FeatureStore.write_month with merge_existing: column merging
4. _get_expected_output_columns: output column derivation
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ── Imports from the build script ──
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_feature_store_from_config import (
    _find_donor_months,
    _find_missing_features,
    _get_expected_output_columns,
)
from src.feature_store.feature_store import FeatureStore, FeatureStoreSpec


# ── Fixtures ──


@pytest.fixture
def features_cfg():
    """Minimal feature dependency config for testing."""
    return {
        "roc_5_f": {
            "compute_func": "compute_roc_5_from_series",
            "output_columns": ["roc_5"],
            "dependencies": [],
        },
        "roc_10_f": {
            "compute_func": "compute_roc_10_from_series",
            "output_columns": ["roc_10"],
            "dependencies": [],
        },
        "roc_20_f": {
            "compute_func": "compute_roc_20_from_series",
            "output_columns": ["roc_20"],
            "dependencies": [],
        },
        "bb_width_f": {
            "compute_func": "compute_bb_width",
            "output_columns": ["bb_upper", "bb_lower", "bb_width"],
            "dependencies": [],
        },
        "atr_ratio_f": {
            "compute_func": "compute_atr_ratio",
            "output_columns": ["atr_ratio"],
            "dependencies": [],
        },
    }


@pytest.fixture
def sample_df():
    """Create a small sample DataFrame spanning 2 months."""
    idx = pd.date_range("2024-01-01", "2024-02-28", freq="1h")
    np.random.seed(42)
    return pd.DataFrame(
        {
            "open": np.random.randn(len(idx)).cumsum() + 100,
            "high": np.random.randn(len(idx)).cumsum() + 101,
            "low": np.random.randn(len(idx)).cumsum() + 99,
            "close": np.random.randn(len(idx)).cumsum() + 100,
            "volume": np.abs(np.random.randn(len(idx))) * 1000,
            "symbol": "BTCUSDT",
        },
        index=idx,
    )


@pytest.fixture
def tmp_store(tmp_path):
    """Create a temporary FeatureStore."""
    return FeatureStore(tmp_path)


# ── Tests for _get_expected_output_columns ──


class TestGetExpectedOutputColumns:
    def test_basic(self, features_cfg):
        requested = ["roc_5_f", "bb_width_f"]
        result = _get_expected_output_columns(features_cfg, requested)
        assert result == {"roc_5", "bb_upper", "bb_lower", "bb_width"}

    def test_unknown_feature_passthrough(self, features_cfg):
        """Unknown feature names are treated as their own output column."""
        requested = ["roc_5_f", "unknown_col"]
        result = _get_expected_output_columns(features_cfg, requested)
        assert "roc_5" in result
        assert "unknown_col" in result

    def test_empty(self, features_cfg):
        assert _get_expected_output_columns(features_cfg, []) == set()


# ── Tests for _find_missing_features ──


class TestFindMissingFeatures:
    def test_all_present(self, features_cfg):
        existing = {"roc_5", "roc_10", "roc_20", "bb_upper", "bb_lower", "bb_width"}
        requested = ["roc_5_f", "roc_10_f", "roc_20_f", "bb_width_f"]
        result = _find_missing_features(features_cfg, requested, existing)
        assert result == []

    def test_some_missing(self, features_cfg):
        existing = {"roc_5", "bb_upper", "bb_lower", "bb_width"}
        requested = ["roc_5_f", "roc_10_f", "roc_20_f", "bb_width_f"]
        result = _find_missing_features(features_cfg, requested, existing)
        assert result == ["roc_10_f", "roc_20_f"]

    def test_partial_multi_output(self, features_cfg):
        """bb_width_f has 3 outputs; if any is missing, the feature is 'missing'."""
        existing = {"roc_5", "bb_upper", "bb_lower"}  # bb_width missing
        requested = ["roc_5_f", "bb_width_f"]
        result = _find_missing_features(features_cfg, requested, existing)
        assert result == ["bb_width_f"]

    def test_all_missing(self, features_cfg):
        existing = {"open", "close"}
        requested = ["roc_5_f", "roc_10_f"]
        result = _find_missing_features(features_cfg, requested, existing)
        assert result == ["roc_5_f", "roc_10_f"]


# ── Tests for _find_donor_months ──


class TestFindDonorMonths:
    def test_finds_donor(self, tmp_store, sample_df, tmp_path):
        """When another layer has the month, it should be found as donor."""
        # Write to donor layer
        donor_spec = FeatureStoreSpec(
            layer="old_layer", symbol="BTCUSDT", timeframe="60T"
        )
        jan_df = sample_df.loc["2024-01"]
        tmp_store.write_month(donor_spec, "2024-01", jan_df, overwrite=True)

        # Query for the current layer (doesn't have the month)
        current_spec = FeatureStoreSpec(
            layer="new_layer", symbol="BTCUSDT", timeframe="60T"
        )
        donors = _find_donor_months(tmp_store, current_spec, ["2024-01"], tmp_path)
        assert "2024-01" in donors
        assert donors["2024-01"].layer == "old_layer"

    def test_no_donor_for_different_timeframe(self, tmp_store, sample_df, tmp_path):
        """Donor must match symbol + timeframe."""
        donor_spec = FeatureStoreSpec(
            layer="old_layer", symbol="BTCUSDT", timeframe="240T"
        )
        jan_df = sample_df.loc["2024-01"]
        tmp_store.write_month(donor_spec, "2024-01", jan_df, overwrite=True)

        current_spec = FeatureStoreSpec(
            layer="new_layer", symbol="BTCUSDT", timeframe="60T"
        )
        donors = _find_donor_months(tmp_store, current_spec, ["2024-01"], tmp_path)
        assert "2024-01" not in donors

    def test_skips_own_layer(self, tmp_store, sample_df, tmp_path):
        """Should not return the current layer as a donor."""
        spec = FeatureStoreSpec(layer="my_layer", symbol="BTCUSDT", timeframe="60T")
        jan_df = sample_df.loc["2024-01"]
        tmp_store.write_month(spec, "2024-01", jan_df, overwrite=True)

        donors = _find_donor_months(tmp_store, spec, ["2024-01"], tmp_path)
        assert donors == {}

    def test_empty_months_needed(self, tmp_store, tmp_path):
        spec = FeatureStoreSpec(layer="layer", symbol="BTCUSDT", timeframe="60T")
        assert _find_donor_months(tmp_store, spec, [], tmp_path) == {}


# ── Tests for FeatureStore merge_existing ──


class TestFeatureStoreMergeExisting:
    def test_merge_adds_columns(self, tmp_store, sample_df):
        """write_month with merge_existing=True should add new columns to existing data."""
        spec = FeatureStoreSpec(layer="test", symbol="BTCUSDT", timeframe="60T")
        jan_df = sample_df.loc["2024-01"].copy()

        # Phase 1: write with initial columns
        phase1_df = jan_df[["open", "close", "volume"]].copy()
        phase1_df["feat_a"] = 1.0
        phase1_df["feat_b"] = 2.0
        tmp_store.write_month(
            spec,
            "2024-01",
            phase1_df,
            base_columns=["open", "close", "volume"],
            feature_columns=["feat_a", "feat_b"],
            overwrite=True,
        )

        # Phase 2: write new features with merge
        phase2_df = jan_df[["open", "close"]].copy()
        phase2_df["feat_c"] = 3.0
        phase2_df["feat_d"] = 4.0
        tmp_store.write_month(
            spec,
            "2024-01",
            phase2_df,
            base_columns=["open", "close"],
            feature_columns=["feat_c", "feat_d"],
            merge_existing=True,
        )

        # Verify: all columns present
        result = tmp_store.read_month(spec, "2024-01")
        assert "feat_a" in result.columns, "Original feature should be preserved"
        assert "feat_b" in result.columns, "Original feature should be preserved"
        assert "feat_c" in result.columns, "New feature should be added"
        assert "feat_d" in result.columns, "New feature should be added"

    def test_merge_preserves_values(self, tmp_store, sample_df):
        """Merged data should preserve original values for existing columns."""
        spec = FeatureStoreSpec(layer="test", symbol="BTCUSDT", timeframe="60T")
        jan_df = sample_df.loc["2024-01"].copy()

        # Phase 1
        phase1_df = jan_df[["close"]].copy()
        phase1_df["feat_a"] = np.arange(len(phase1_df), dtype=float)
        tmp_store.write_month(spec, "2024-01", phase1_df, overwrite=True)

        # Phase 2: add feat_b
        phase2_df = jan_df[["close"]].copy()
        phase2_df["feat_b"] = np.arange(len(phase2_df), dtype=float) * 10
        tmp_store.write_month(spec, "2024-01", phase2_df, merge_existing=True)

        result = tmp_store.read_month(spec, "2024-01")
        # feat_a values preserved
        np.testing.assert_array_equal(
            result["feat_a"].values, np.arange(len(result), dtype=float)
        )
        # feat_b values added
        np.testing.assert_array_equal(
            result["feat_b"].values, np.arange(len(result), dtype=float) * 10
        )

    def test_meta_columns_updated_after_merge(self, tmp_store, sample_df):
        """After merge, the meta.json columns list should reflect all columns."""
        spec = FeatureStoreSpec(layer="test", symbol="BTCUSDT", timeframe="60T")
        jan_df = sample_df.loc["2024-01"].copy()

        # Phase 1
        phase1_df = jan_df[["close"]].copy()
        phase1_df["feat_a"] = 1.0
        tmp_store.write_month(spec, "2024-01", phase1_df, overwrite=True)

        # Phase 2
        phase2_df = jan_df[["close"]].copy()
        phase2_df["feat_b"] = 2.0
        tmp_store.write_month(spec, "2024-01", phase2_df, merge_existing=True)

        # Check meta
        meta = tmp_store.read_month_meta(spec, "2024-01")
        cols = meta.get("columns", [])
        assert "feat_a" in cols
        assert "feat_b" in cols
