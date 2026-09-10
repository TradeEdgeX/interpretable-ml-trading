"""Tests for T5β liquidation cluster proxy features."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.time_series.t5_liquidation_proxy_features import (
    compute_liquidation_cluster_proxy_from_df,
)


def _make_df(n: int = 100, seed: int = 0) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    idx = pd.date_range("2024-06-01", periods=n, freq="2h", tz="UTC")
    return pd.DataFrame(
        {
            "oi_flow_zscore": rng.randn(n) * 2,
            "funding_rate_abs_zscore_50": rng.rand(n) * 4,
            "atr_percentile": rng.rand(n),
            "oi_ignition_score": rng.rand(n),
            "oi_exhaustion_score": rng.rand(n),
        },
        index=idx,
    )


class TestLiquidationClusterProxy:
    def test_output_columns(self):
        out = compute_liquidation_cluster_proxy_from_df(_make_df())
        assert set(out.columns) == {
            "liquidation_cluster_score",
            "liquidation_cascade_proxy_score",
            "liquidation_reversal_proxy_score",
        }

    def test_bounded_0_1(self):
        out = compute_liquidation_cluster_proxy_from_df(_make_df(200, seed=7))
        for col in out.columns:
            assert out[col].min() >= -1e-9
            assert out[col].max() <= 1.0 + 1e-9

    def test_high_stress_when_flow_and_funding_extreme(self):
        n = 50
        idx = pd.date_range("2024-06-01", periods=n, freq="2h", tz="UTC")
        df = pd.DataFrame(
            {
                "oi_flow_zscore": [4.0] * n,
                "funding_rate_abs_zscore_50": [4.0] * n,
                "atr_percentile": [0.9] * n,
                "oi_ignition_score": [0.8] * n,
                "oi_exhaustion_score": [0.1] * n,
            },
            index=idx,
        )
        out = compute_liquidation_cluster_proxy_from_df(df)
        assert out["liquidation_cluster_score"].mean() > 0.5
        assert out["liquidation_cascade_proxy_score"].mean() > 0.4
        assert out["liquidation_reversal_proxy_score"].mean() < 0.2

    def test_low_when_inputs_mild(self):
        n = 50
        idx = pd.date_range("2024-06-01", periods=n, freq="2h", tz="UTC")
        df = pd.DataFrame(
            {
                "oi_flow_zscore": [0.0] * n,
                "funding_rate_abs_zscore_50": [0.0] * n,
                "atr_percentile": [0.1] * n,
                "oi_ignition_score": [0.2] * n,
                "oi_exhaustion_score": [0.2] * n,
            },
            index=idx,
        )
        out = compute_liquidation_cluster_proxy_from_df(df)
        assert out["liquidation_cluster_score"].mean() < 0.05

    def test_cascade_vs_reversal_split(self):
        n = 50
        idx = pd.date_range("2024-06-01", periods=n, freq="2h", tz="UTC")
        base = {
            "oi_flow_zscore": [3.0] * n,
            "funding_rate_abs_zscore_50": [3.0] * n,
            "atr_percentile": [0.8] * n,
        }
        cascade = compute_liquidation_cluster_proxy_from_df(
            pd.DataFrame(
                {
                    **base,
                    "oi_ignition_score": [0.9] * n,
                    "oi_exhaustion_score": [0.1] * n,
                },
                index=idx,
            )
        )
        reversal = compute_liquidation_cluster_proxy_from_df(
            pd.DataFrame(
                {
                    **base,
                    "oi_ignition_score": [0.1] * n,
                    "oi_exhaustion_score": [0.9] * n,
                },
                index=idx,
            )
        )
        assert (
            cascade["liquidation_cascade_proxy_score"].mean()
            > reversal["liquidation_cascade_proxy_score"].mean()
        )
        assert (
            reversal["liquidation_reversal_proxy_score"].mean()
            > cascade["liquidation_reversal_proxy_score"].mean()
        )

    def test_no_look_ahead_pointwise(self):
        df = _make_df(80, seed=3)
        full = compute_liquidation_cluster_proxy_from_df(df)
        half = compute_liquidation_cluster_proxy_from_df(df.iloc[:40])
        for col in full.columns:
            pd.testing.assert_series_equal(
                half[col], full[col].iloc[:40], check_names=False
            )

    def test_requires_datetimeindex(self):
        df = pd.DataFrame({"oi_flow_zscore": [1.0]})
        with pytest.raises(ValueError, match="DatetimeIndex"):
            compute_liquidation_cluster_proxy_from_df(df)

    def test_missing_columns_use_defaults(self):
        n = 20
        idx = pd.date_range("2024-06-01", periods=n, freq="2h", tz="UTC")
        df = pd.DataFrame(index=idx)
        out = compute_liquidation_cluster_proxy_from_df(df)
        assert len(out) == n
        assert out["liquidation_cluster_score"].notna().all()
