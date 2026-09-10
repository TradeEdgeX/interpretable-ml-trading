"""
T5β feature stack — causal / no-look-ahead verification.

Covers the full proxy chain used before scan (Phase 1B+):
  OI join → Funding join → scene semantics → funding_oi_crowding → liquidation proxy

Critical invariant: merge_asof(direction='backward') on OI/funding; downstream scores are
pointwise on already-aligned columns.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features.time_series.funding_rate_features import (
    compute_funding_rate_features_from_df,
    compute_funding_scene_semantic_scores_from_df,
)
from src.features.time_series.open_interest_features import (
    compute_oi_features_from_df,
    compute_oi_scene_semantic_scores_from_df,
)
from src.features.time_series.t5_liquidation_proxy_features import (
    compute_liquidation_cluster_proxy_from_df,
)
from src.features.time_series.utils_interaction_features import (
    compute_funding_oi_crowding_from_series,
)

SYM = "ETHUSDT"


def _write_oi_parquet(
    path: Path, oi_usd: np.ndarray, start: str = "2024-01-01"
) -> None:
    n = len(oi_usd)
    idx = pd.date_range(start, periods=n, freq="5min", tz="UTC")
    df = pd.DataFrame(
        {"oi_usd": oi_usd, "oi_contracts": oi_usd / 3000.0, "_symbol": SYM},
        index=idx,
    )
    df.index.name = "datetime"
    df.to_parquet(path)


def _write_funding_parquet(
    path: Path, rates: np.ndarray, start: str = "2024-01-01"
) -> None:
    n = len(rates)
    idx = pd.date_range(start, periods=n, freq="8h", tz="UTC")
    df = pd.DataFrame({"_symbol": SYM, "funding_rate": rates}, index=idx)
    df.index.name = "datetime"
    df.to_parquet(path)


def _make_bars(n: int, *, seed: int = 0, start: str = "2024-01-15") -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    idx = pd.date_range(start, periods=n, freq="2h", tz="UTC")
    close = 3000.0 + rng.randn(n).cumsum() * 5.0
    return pd.DataFrame(
        {
            "open": close + rng.randn(n),
            "high": close + abs(rng.randn(n)) * 3,
            "low": close - abs(rng.randn(n)) * 3,
            "close": close,
            "volume": abs(rng.randn(n)) * 1e5 + 1e4,
            "compression_score": rng.rand(n),
            "trend_r2_20": rng.rand(n),
            "atr_percentile": rng.rand(n),
            "_symbol": SYM,
        },
        index=idx,
    )


def _compute_t5_stack(
    bars: pd.DataFrame,
    *,
    oi_dir: str,
    funding_dir: str,
) -> pd.DataFrame:
    oi_raw = compute_oi_features_from_df(
        bars, oi_dir=oi_dir, z_window=50, z_min_periods=20
    )
    fr_raw = compute_funding_rate_features_from_df(
        bars, funding_rate_dir=funding_dir, z_window=50, z_min_periods=20
    )
    merged = bars.join(oi_raw).join(fr_raw)
    oi_scene = compute_oi_scene_semantic_scores_from_df(merged)
    fr_scene = compute_funding_scene_semantic_scores_from_df(merged)
    merged = merged.join(oi_scene).join(fr_scene)
    crowding = compute_funding_oi_crowding_from_series(
        funding_rate_abs_zscore_50=merged["funding_rate_abs_zscore_50"],
        oi_zscore=merged["oi_zscore"],
    )
    liq = compute_liquidation_cluster_proxy_from_df(merged)
    return merged.join(crowding).join(liq)


class TestMergeAsofCausal:
    """OI / funding joins must only see observations at or before the bar timestamp."""

    def test_oi_merge_asof_does_not_use_future_observation(self, tmp_path):
        # OI ticks every 5m; bar at 10:00 must not see 10:05 tick.
        oi_idx = pd.date_range("2024-01-15 09:00", periods=30, freq="5min", tz="UTC")
        oi_vals = np.linspace(1e8, 2e8, len(oi_idx))
        oi_df = pd.DataFrame(
            {"oi_usd": oi_vals, "oi_contracts": oi_vals / 3000, "_symbol": SYM},
            index=oi_idx,
        )
        oi_df.index.name = "datetime"
        oi_df.to_parquet(tmp_path / f"{SYM}_2024-01_oi_5m.parquet")

        bar_ts = pd.DatetimeIndex(["2024-01-15 10:00:00"], tz="UTC")
        bars = pd.DataFrame({"close": [3000.0], "_symbol": [SYM]}, index=bar_ts)
        out = compute_oi_features_from_df(bars, oi_dir=str(tmp_path))

        # Last OI at or before 10:00 is 10:00 itself (if aligned) or 09:55
        expected_ts = oi_idx[oi_idx <= bar_ts[0]][-1]
        expected_oi = oi_vals[oi_idx == expected_ts][0]
        assert np.isclose(out["oi_usd"].iloc[0], expected_oi)

        # Mutate all OI strictly AFTER 10:00 — bar value must not change.
        oi_df2 = oi_df.copy()
        future_mask = oi_df2.index > bar_ts[0]
        oi_df2.loc[future_mask, "oi_usd"] *= 10.0
        oi_df2.to_parquet(tmp_path / f"{SYM}_2024-01_oi_5m.parquet")
        out2 = compute_oi_features_from_df(bars, oi_dir=str(tmp_path))
        assert np.isclose(out2["oi_usd"].iloc[0], out["oi_usd"].iloc[0])

    def test_funding_merge_asof_does_not_use_future_observation(self, tmp_path):
        fr_idx = pd.to_datetime(
            [
                "2024-01-15 00:00:00+00:00",
                "2024-01-15 08:00:00+00:00",
                "2024-01-15 16:00:00+00:00",
            ]
        )
        fr = pd.DataFrame(
            {"_symbol": SYM, "funding_rate": [0.0001, 0.0002, 0.0009]},
            index=fr_idx,
        )
        fr.index.name = "datetime"
        fr.to_parquet(tmp_path / f"{SYM}_2024-01_funding_rate.parquet")

        bar_ts = pd.DatetimeIndex(["2024-01-15 10:00:00"], tz="UTC")
        bars = pd.DataFrame({"close": [3000.0], "_symbol": [SYM]}, index=bar_ts)
        out = compute_funding_rate_features_from_df(
            bars, funding_rate_dir=str(tmp_path)
        )
        # 10:00 bar sees latest funding at or before 10:00 → 08:00 obs (0.0002)
        assert np.isclose(out["funding_rate"].iloc[0], 0.0002)

        fr2 = fr.copy()
        fr2.loc[fr_idx > bar_ts[0], "funding_rate"] = 0.05
        fr2.to_parquet(tmp_path / f"{SYM}_2024-01_funding_rate.parquet")
        out2 = compute_funding_rate_features_from_df(
            bars, funding_rate_dir=str(tmp_path)
        )
        assert np.isclose(out2["funding_rate"].iloc[0], out["funding_rate"].iloc[0])


class TestT5StackNoFutureLeak:
    """End-to-end: appending future bars or future source rows must not alter history."""

    @pytest.fixture()
    def data_dirs(self, tmp_path):
        oi_dir = tmp_path / "oi"
        fr_dir = tmp_path / "fr"
        oi_dir.mkdir()
        fr_dir.mkdir()
        rng = np.random.RandomState(7)
        n_oi = 20_000
        oi_usd = 5e8 + rng.randn(n_oi).cumsum() * 1e6
        _write_oi_parquet(oi_dir / f"{SYM}_2024-01_oi_5m.parquet", oi_usd)
        n_fr = 400
        fr = 0.0001 + rng.randn(n_fr) * 0.0002
        _write_funding_parquet(fr_dir / f"{SYM}_2024-01_funding_rate.parquet", fr)
        return str(oi_dir), str(fr_dir)

    T5_COLS = [
        "oi_usd",
        "oi_flow_zscore",
        "oi_ignition_score",
        "oi_exhaustion_score",
        "funding_rate_abs_zscore_50",
        "funding_oi_crowding_score",
        "liquidation_cluster_score",
        "liquidation_cascade_proxy_score",
        "liquidation_reversal_proxy_score",
    ]

    def test_appending_future_bars_does_not_change_past(self, data_dirs):
        oi_dir, fr_dir = data_dirs
        n = 120
        k = 70
        bars = _make_bars(n, seed=1)
        full = _compute_t5_stack(bars, oi_dir=oi_dir, funding_dir=fr_dir)
        prefix = _compute_t5_stack(bars.iloc[:k], oi_dir=oi_dir, funding_dir=fr_dir)

        for col in self.T5_COLS:
            if col not in full.columns:
                continue
            pd.testing.assert_series_equal(
                prefix[col],
                full[col].iloc[:k],
                check_names=False,
                rtol=1e-10,
                obj=f"future-leak appending bars: {col}",
            )

    def test_appending_future_oi_rows_does_not_change_past_bars(self, tmp_path):
        oi_dir = tmp_path / "oi"
        fr_dir = tmp_path / "fr"
        oi_dir.mkdir()
        fr_dir.mkdir()
        rng = np.random.RandomState(11)
        n_oi = 15_000
        base_oi = 4e8 + rng.randn(n_oi).cumsum() * 1e6
        _write_oi_parquet(oi_dir / f"{SYM}_2024-01_oi_5m.parquet", base_oi)
        fr = 0.0001 + rng.randn(300) * 0.0001
        _write_funding_parquet(fr_dir / f"{SYM}_2024-01_funding_rate.parquet", fr)

        bars = _make_bars(100, seed=3)
        before = _compute_t5_stack(bars, oi_dir=str(oi_dir), funding_dir=str(fr_dir))

        # Extend OI series into the future with a huge spike (new download batch).
        extra = base_oi[-1] + np.arange(5000) * 1e7
        combined = np.concatenate([base_oi, extra])
        _write_oi_parquet(oi_dir / f"{SYM}_2024-01_oi_5m.parquet", combined)

        after = _compute_t5_stack(bars, oi_dir=str(oi_dir), funding_dir=str(fr_dir))

        for col in self.T5_COLS:
            if col not in before.columns:
                continue
            pd.testing.assert_series_equal(
                before[col],
                after[col],
                check_names=False,
                rtol=1e-10,
                obj=f"future-leak appending OI rows: {col}",
            )

    def test_appending_future_funding_rows_does_not_change_past_bars(self, tmp_path):
        oi_dir = tmp_path / "oi"
        fr_dir = tmp_path / "fr"
        oi_dir.mkdir()
        fr_dir.mkdir()
        rng = np.random.RandomState(13)
        n_oi = 15_000
        _write_oi_parquet(
            oi_dir / f"{SYM}_2024-01_oi_5m.parquet",
            4e8 + rng.randn(n_oi).cumsum() * 1e6,
        )
        base_fr = 0.0001 + rng.randn(200) * 0.0001
        _write_funding_parquet(fr_dir / f"{SYM}_2024-01_funding_rate.parquet", base_fr)

        bars = _make_bars(100, seed=5)
        before = _compute_t5_stack(bars, oi_dir=str(oi_dir), funding_dir=str(fr_dir))

        spike_fr = np.concatenate([base_fr, np.full(50, 0.02)])
        _write_funding_parquet(fr_dir / f"{SYM}_2024-01_funding_rate.parquet", spike_fr)

        after = _compute_t5_stack(bars, oi_dir=str(oi_dir), funding_dir=str(fr_dir))

        for col in self.T5_COLS:
            if col not in before.columns:
                continue
            pd.testing.assert_series_equal(
                before[col],
                after[col],
                check_names=False,
                rtol=1e-10,
                obj=f"future-leak appending funding rows: {col}",
            )


class TestFundingSceneCausal:
    def test_funding_scene_no_look_ahead_pointwise(self):
        n = 80
        idx = pd.date_range("2024-06-01", periods=n, freq="2h", tz="UTC")
        rng = np.random.RandomState(4)
        df = pd.DataFrame(
            {
                "funding_rate_abs_zscore_50": rng.randn(n) * 2,
                "compression_score": rng.rand(n),
                "trend_r2_20": rng.rand(n),
            },
            index=idx,
        )
        full = compute_funding_scene_semantic_scores_from_df(df)
        half = compute_funding_scene_semantic_scores_from_df(df.iloc[:40])
        for col in full.columns:
            pd.testing.assert_series_equal(
                half[col], full[col].iloc[:40], check_names=False
            )


class TestLiquidationProxyCausal:
    def test_extreme_future_inputs_do_not_change_past_rows(self):
        n = 100
        k = 60
        idx = pd.date_range("2024-06-01", periods=n, freq="2h", tz="UTC")
        rng = np.random.RandomState(8)
        base = pd.DataFrame(
            {
                "oi_flow_zscore": rng.randn(n),
                "funding_rate_abs_zscore_50": rng.rand(n) * 3,
                "atr_percentile": rng.rand(n),
                "oi_ignition_score": rng.rand(n),
                "oi_exhaustion_score": rng.rand(n),
            },
            index=idx,
        )
        full_base = compute_liquidation_cluster_proxy_from_df(base)

        mutated = base.copy()
        mutated.loc[mutated.index[k:], "oi_flow_zscore"] = 99.0
        mutated.loc[mutated.index[k:], "funding_rate_abs_zscore_50"] = 99.0
        full_mut = compute_liquidation_cluster_proxy_from_df(mutated)

        for col in full_base.columns:
            pd.testing.assert_series_equal(
                full_base[col].iloc[:k],
                full_mut[col].iloc[:k],
                check_names=False,
                obj=f"future-leak mutation: {col}",
            )
