import numpy as np
import pandas as pd


def test_multiclass_prob_threshold_produces_entries():
    from src.time_series_model.strategies.backtesting.vectorbt_backtest import (
        VectorBTBacktest,
    )

    index = pd.date_range("2024-01-01", periods=5, freq="1H")
    # 3-class proba: [short, neutral, long]
    proba = np.array(
        [
            [0.10, 0.80, 0.10],  # neutral
            [0.46, 0.40, 0.14],  # short >= 0.45
            [0.10, 0.60, 0.48],  # long >= 0.45
            [0.46, 0.30, 0.46],  # both >= 0.45 -> pick higher (tie -> long)
            [0.05, 0.90, 0.05],  # neutral
        ],
        dtype=float,
    )

    le, se, lx, sx = VectorBTBacktest._multiclass_entries_from_proba(
        proba=proba,
        index=index,
        long_class=2,
        short_class=0,
        neutral_class=1,
        entry_mode="prob_threshold",
        entry_threshold=0.45,
    )
    assert int(le.sum()) == 2
    assert int(se.sum()) == 1
    assert bool(lx.any()) is False
    assert bool(sx.any()) is False


def test_multiclass_prob_quantile_produces_entries():
    from src.time_series_model.strategies.backtesting.vectorbt_backtest import (
        VectorBTBacktest,
    )

    index = pd.date_range("2024-01-01", periods=5, freq="1H")
    # 3-class proba: [short, neutral, long]
    proba = np.array(
        [
            [0.05, 0.70, 0.25],
            [0.10, 0.70, 0.20],
            [0.20, 0.60, 0.20],
            [0.60, 0.30, 0.10],
            [0.55, 0.40, 0.05],
        ],
        dtype=float,
    )
    # q80: should pick the top ~20% for each side (small N, quantile interpolation).
    le, se, lx, sx = VectorBTBacktest._multiclass_entries_from_proba(
        proba=proba,
        index=index,
        long_class=2,
        short_class=0,
        neutral_class=1,
        entry_mode="prob_quantile",
        entry_quantile=0.8,
    )
    assert int(le.sum()) >= 1
    assert int(se.sum()) >= 1
    assert bool(lx.any()) is False
    assert bool(sx.any()) is False
