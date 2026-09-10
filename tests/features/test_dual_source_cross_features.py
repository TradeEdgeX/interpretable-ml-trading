"""
Tests for dual-source scene semantic cross features.

Coverage:
1. Basic functionality (output shape, column names, values bounded [0,1])
2. No look-ahead bias (bar-by-bar causal)
3. Streaming vs batch consistency
4. Edge cases (NaN, zeros, ones)
"""

import numpy as np
import pandas as pd
import pytest

from src.features.time_series.utils_interaction_features import (
    compute_dual_compression_from_series,
    compute_dual_ignition_from_series,
    compute_dual_exhaustion_from_series,
    compute_funding_oi_crowding_from_series,
)


@pytest.fixture
def toy_data():
    n = 100
    idx = pd.date_range("2024-01-01", periods=n, freq="4H")
    rng = np.random.default_rng(42)
    return {
        "funding_compression_score": pd.Series(rng.uniform(0, 1, n), index=idx),
        "vpin_compression_score": pd.Series(rng.uniform(0, 1, n), index=idx),
        "funding_ignition_score": pd.Series(rng.uniform(0, 1, n), index=idx),
        "fp_imbalance_ignition_score": pd.Series(rng.uniform(0, 1, n), index=idx),
        "funding_exhaustion_scene_score": pd.Series(rng.uniform(0, 1, n), index=idx),
        "vpin_exhaustion_scene_score": pd.Series(rng.uniform(0, 1, n), index=idx),
    }


class TestDualCompression:
    def test_output_shape_and_columns(self, toy_data):
        out = compute_dual_compression_from_series(
            funding_compression_score=toy_data["funding_compression_score"],
            vpin_compression_score=toy_data["vpin_compression_score"],
        )
        assert isinstance(out, pd.DataFrame)
        assert list(out.columns) == ["dual_compression_score"]
        assert len(out) == len(toy_data["funding_compression_score"])

    def test_bounded_0_1(self, toy_data):
        out = compute_dual_compression_from_series(
            funding_compression_score=toy_data["funding_compression_score"],
            vpin_compression_score=toy_data["vpin_compression_score"],
        )
        vals = out["dual_compression_score"]
        assert (vals >= 0.0 - 1e-9).all()
        assert (vals <= 1.0 + 1e-9).all()

    def test_math_correctness(self, toy_data):
        out = compute_dual_compression_from_series(
            funding_compression_score=toy_data["funding_compression_score"],
            vpin_compression_score=toy_data["vpin_compression_score"],
        )
        expected = (
            toy_data["funding_compression_score"] * toy_data["vpin_compression_score"]
        )
        pd.testing.assert_series_equal(
            out["dual_compression_score"],
            expected.rename("dual_compression_score"),
            check_names=True,
            rtol=1e-10,
        )

    def test_nan_handling(self):
        idx = pd.date_range("2024-01-01", periods=5, freq="4H")
        fc = pd.Series([0.5, np.nan, 0.8, 0.0, 1.0], index=idx)
        vc = pd.Series([0.3, 0.6, np.nan, 0.7, 0.9], index=idx)
        out = compute_dual_compression_from_series(
            funding_compression_score=fc,
            vpin_compression_score=vc,
        )
        assert out["dual_compression_score"].notna().all(), "NaN should be filled to 0"

    def test_no_look_ahead(self, toy_data):
        """Verify bar-by-bar causal: changing future data doesn't affect past."""
        fc = toy_data["funding_compression_score"].copy()
        vc = toy_data["vpin_compression_score"].copy()
        out_full = compute_dual_compression_from_series(
            funding_compression_score=fc,
            vpin_compression_score=vc,
        )
        # Truncate to first 50 bars
        out_partial = compute_dual_compression_from_series(
            funding_compression_score=fc.iloc[:50],
            vpin_compression_score=vc.iloc[:50],
        )
        pd.testing.assert_series_equal(
            out_full["dual_compression_score"].iloc[:50],
            out_partial["dual_compression_score"],
            check_names=True,
        )


class TestDualIgnition:
    def test_output_shape_and_columns(self, toy_data):
        out = compute_dual_ignition_from_series(
            funding_ignition_score=toy_data["funding_ignition_score"],
            fp_imbalance_ignition_score=toy_data["fp_imbalance_ignition_score"],
        )
        assert isinstance(out, pd.DataFrame)
        assert list(out.columns) == ["dual_ignition_score"]
        assert len(out) == len(toy_data["funding_ignition_score"])

    def test_bounded_0_1(self, toy_data):
        out = compute_dual_ignition_from_series(
            funding_ignition_score=toy_data["funding_ignition_score"],
            fp_imbalance_ignition_score=toy_data["fp_imbalance_ignition_score"],
        )
        vals = out["dual_ignition_score"]
        assert (vals >= 0.0 - 1e-9).all()
        assert (vals <= 1.0 + 1e-9).all()

    def test_math_correctness(self, toy_data):
        out = compute_dual_ignition_from_series(
            funding_ignition_score=toy_data["funding_ignition_score"],
            fp_imbalance_ignition_score=toy_data["fp_imbalance_ignition_score"],
        )
        expected = (
            toy_data["funding_ignition_score"] * toy_data["fp_imbalance_ignition_score"]
        )
        pd.testing.assert_series_equal(
            out["dual_ignition_score"],
            expected.rename("dual_ignition_score"),
            check_names=True,
            rtol=1e-10,
        )


class TestDualExhaustion:
    def test_output_shape_and_columns(self, toy_data):
        out = compute_dual_exhaustion_from_series(
            funding_exhaustion_scene_score=toy_data["funding_exhaustion_scene_score"],
            vpin_exhaustion_scene_score=toy_data["vpin_exhaustion_scene_score"],
        )
        assert isinstance(out, pd.DataFrame)
        assert list(out.columns) == ["dual_exhaustion_score"]

    def test_bounded_0_1(self, toy_data):
        out = compute_dual_exhaustion_from_series(
            funding_exhaustion_scene_score=toy_data["funding_exhaustion_scene_score"],
            vpin_exhaustion_scene_score=toy_data["vpin_exhaustion_scene_score"],
        )
        vals = out["dual_exhaustion_score"]
        assert (vals >= 0.0 - 1e-9).all()
        assert (vals <= 1.0 + 1e-9).all()

    def test_math_correctness(self, toy_data):
        out = compute_dual_exhaustion_from_series(
            funding_exhaustion_scene_score=toy_data["funding_exhaustion_scene_score"],
            vpin_exhaustion_scene_score=toy_data["vpin_exhaustion_scene_score"],
        )
        expected = (
            toy_data["funding_exhaustion_scene_score"]
            * toy_data["vpin_exhaustion_scene_score"]
        )
        pd.testing.assert_series_equal(
            out["dual_exhaustion_score"],
            expected.rename("dual_exhaustion_score"),
            check_names=True,
            rtol=1e-10,
        )

    def test_zeros_produce_zero(self):
        idx = pd.date_range("2024-01-01", periods=5, freq="4H")
        fe = pd.Series([0.0, 0.5, 0.0, 1.0, 0.3], index=idx)
        ve = pd.Series([0.8, 0.0, 0.0, 0.9, 0.7], index=idx)
        out = compute_dual_exhaustion_from_series(
            funding_exhaustion_scene_score=fe,
            vpin_exhaustion_scene_score=ve,
        )
        # Where either is 0, product should be 0
        assert out["dual_exhaustion_score"].iloc[0] == 0.0  # 0 × 0.8
        assert out["dual_exhaustion_score"].iloc[1] == 0.0  # 0.5 × 0
        assert out["dual_exhaustion_score"].iloc[2] == 0.0  # 0 × 0

    def test_ones_produce_one(self):
        idx = pd.date_range("2024-01-01", periods=3, freq="4H")
        fe = pd.Series([1.0, 1.0, 0.5], index=idx)
        ve = pd.Series([1.0, 0.5, 1.0], index=idx)
        out = compute_dual_exhaustion_from_series(
            funding_exhaustion_scene_score=fe,
            vpin_exhaustion_scene_score=ve,
        )
        assert np.isclose(out["dual_exhaustion_score"].iloc[0], 1.0)
        assert np.isclose(out["dual_exhaustion_score"].iloc[1], 0.5)
        assert np.isclose(out["dual_exhaustion_score"].iloc[2], 0.5)


# ========================================================================
# Funding × OI Crowding
# ========================================================================


def _sigmoid(x, shift, scale):
    """Reference sigmoid for manual verification."""
    return 1.0 / (1.0 + np.exp(-(x - shift) / scale))


class TestFundingOiCrowding:
    """Tests for compute_funding_oi_crowding_from_series."""

    def test_output_shape_and_column(self):
        idx = pd.date_range("2024-01-01", periods=50, freq="4h")
        fr_z = pd.Series(np.linspace(-2, 3, 50), index=idx)
        oi_z = pd.Series(np.linspace(-1, 2, 50), index=idx)
        out = compute_funding_oi_crowding_from_series(
            funding_rate_abs_zscore_50=fr_z,
            oi_zscore=oi_z,
        )
        assert isinstance(out, pd.DataFrame)
        assert list(out.columns) == ["funding_oi_crowding_score"]
        assert len(out) == 50

    def test_bounded_0_1(self):
        idx = pd.date_range("2024-01-01", periods=200, freq="4h")
        rng = np.random.default_rng(7)
        fr_z = pd.Series(rng.normal(0, 3, 200), index=idx)
        oi_z = pd.Series(rng.normal(0, 2, 200), index=idx)
        out = compute_funding_oi_crowding_from_series(
            funding_rate_abs_zscore_50=fr_z,
            oi_zscore=oi_z,
        )
        vals = out["funding_oi_crowding_score"]
        assert (vals >= 0.0 - 1e-9).all()
        assert (vals <= 1.0 + 1e-9).all()

    def test_math_correctness(self):
        """Manual sigmoid(fr_z) × sigmoid(oi_z) matches function output."""
        idx = pd.date_range("2024-01-01", periods=5, freq="4h")
        fr_z = pd.Series([0.0, 1.0, 2.0, 3.0, -1.0], index=idx)
        oi_z = pd.Series([0.0, 0.5, 1.5, 2.5, -0.5], index=idx)
        out = compute_funding_oi_crowding_from_series(
            funding_rate_abs_zscore_50=fr_z,
            oi_zscore=oi_z,
            funding_shift=1.0,
            funding_scale=1.0,
            oi_shift=0.5,
            oi_scale=1.0,
        )
        for i in range(5):
            expected = _sigmoid(fr_z.iloc[i], 1.0, 1.0) * _sigmoid(
                oi_z.iloc[i], 0.5, 1.0
            )
            assert np.isclose(
                out["funding_oi_crowding_score"].iloc[i], expected, rtol=1e-10
            ), f"Mismatch at i={i}: {out['funding_oi_crowding_score'].iloc[i]} vs {expected}"

    def test_both_high_produces_high(self):
        """Both funding and OI z-scores high → crowding close to 1."""
        idx = pd.date_range("2024-01-01", periods=1, freq="4h")
        out = compute_funding_oi_crowding_from_series(
            funding_rate_abs_zscore_50=pd.Series([5.0], index=idx),
            oi_zscore=pd.Series([4.0], index=idx),
        )
        assert out["funding_oi_crowding_score"].iloc[0] > 0.9

    def test_both_low_produces_low(self):
        """Both z-scores negative → crowding close to 0."""
        idx = pd.date_range("2024-01-01", periods=1, freq="4h")
        out = compute_funding_oi_crowding_from_series(
            funding_rate_abs_zscore_50=pd.Series([-2.0], index=idx),
            oi_zscore=pd.Series([-2.0], index=idx),
        )
        assert out["funding_oi_crowding_score"].iloc[0] < 0.05

    def test_one_high_one_low_is_moderate(self):
        """Only funding high but OI low → not true crowding."""
        idx = pd.date_range("2024-01-01", periods=1, freq="4h")
        out = compute_funding_oi_crowding_from_series(
            funding_rate_abs_zscore_50=pd.Series([5.0], index=idx),
            oi_zscore=pd.Series([-2.0], index=idx),
        )
        val = out["funding_oi_crowding_score"].iloc[0]
        # funding_stress ≈ 0.98, oi_activity ≈ 0.076 → product ≈ 0.075
        assert val < 0.15, f"Should be low when only funding is high: {val}"

    def test_nan_handling(self):
        idx = pd.date_range("2024-01-01", periods=3, freq="4h")
        fr_z = pd.Series([np.nan, 2.0, 1.0], index=idx)
        oi_z = pd.Series([1.0, np.nan, 0.5], index=idx)
        out = compute_funding_oi_crowding_from_series(
            funding_rate_abs_zscore_50=fr_z,
            oi_zscore=oi_z,
        )
        assert (
            out["funding_oi_crowding_score"].notna().all()
        ), "NaN should be filled to 0"

    def test_no_look_ahead(self):
        """Pointwise: no rolling involved, so truncation must match."""
        idx = pd.date_range("2024-01-01", periods=100, freq="4h")
        rng = np.random.default_rng(42)
        fr_z = pd.Series(rng.normal(1, 2, 100), index=idx)
        oi_z = pd.Series(rng.normal(0.5, 1.5, 100), index=idx)
        out_full = compute_funding_oi_crowding_from_series(
            funding_rate_abs_zscore_50=fr_z,
            oi_zscore=oi_z,
        )
        out_partial = compute_funding_oi_crowding_from_series(
            funding_rate_abs_zscore_50=fr_z.iloc[:50],
            oi_zscore=oi_z.iloc[:50],
        )
        pd.testing.assert_series_equal(
            out_full["funding_oi_crowding_score"].iloc[:50],
            out_partial["funding_oi_crowding_score"],
            check_names=True,
        )
