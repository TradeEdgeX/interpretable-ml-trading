"""Holdout embargo helpers for train_strategy_pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.train_strategy_pipeline import (
    apply_post_label_filters,
    holdout_train_end_exclusive,
    resolve_holdout_embargo_bars,
)


def test_resolve_holdout_embargo_bars_explicit_wins() -> None:
    assert resolve_holdout_embargo_bars(holdout_embargo_bars=5, label_horizon=3) == 5


def test_resolve_holdout_embargo_bars_from_label_horizon() -> None:
    assert (
        resolve_holdout_embargo_bars(holdout_embargo_bars=None, label_horizon=20) == 20
    )


def test_resolve_holdout_embargo_bars_zero_when_unset() -> None:
    assert (
        resolve_holdout_embargo_bars(holdout_embargo_bars=None, label_horizon=None) == 0
    )


def test_holdout_train_end_exclusive_240t_three_bars() -> None:
    hs = pd.Timestamp("2025-10-01")
    cut = holdout_train_end_exclusive(hs, embargo_bars=3, bar_minutes=240)
    # 3 × 240min = 12h before holdout start
    assert cut == pd.Timestamp("2025-09-30 12:00:00")


def test_purge_train_indices_timestamp_horizon() -> None:
    from src.time_series_model.strategies.models.strategy_trainer import (
        purge_train_indices,
    )

    idx = pd.date_range("2025-01-01", periods=10, freq="4h", tz="UTC")
    ts = pd.Series(idx)
    train_idx = np.arange(7)
    val_idx = np.arange(7, 10)
    kept = purge_train_indices(
        train_idx, val_idx, timestamps=ts, horizon_bars=3, bar_minutes=240
    )
    # val_start = 2025-01-02 04:00; cutoff = val_start - 12h = 2025-01-01 16:00
    # keep t < 16:00 → bars 0,1,2,3 (00:00, 04:00, 08:00, 12:00)
    assert list(kept) == [0, 1, 2, 3]


def test_purge_train_indices_row_fallback() -> None:
    from src.time_series_model.strategies.models.strategy_trainer import (
        purge_train_indices,
    )

    kept = purge_train_indices(
        np.arange(10), np.arange(10, 15), timestamps=None, horizon_bars=3
    )
    assert list(kept) == [0, 1, 2, 3, 4, 5, 6]


def test_apply_post_label_filters_min_compression_mask() -> None:
    df = pd.DataFrame(
        {
            "label": [1.0, 2.0, 3.0],
            "compression_duration": [0.0, 0.02, 0.5],
        }
    )
    out = apply_post_label_filters(
        df,
        [
            {"column": "compression_duration", "notna": True},
            {"column": "compression_duration", "min": 0.01},
        ],
        feature_cols=[],
    )
    assert len(out) == 2
    assert out["compression_duration"].tolist() == [0.02, 0.5]
