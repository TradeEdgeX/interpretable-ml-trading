"""EMA50×EMA200 cross is an event, not a level."""

from __future__ import annotations

import pandas as pd

from src.features.time_series.baseline_features import (
    compute_ema_50_200_cross_from_series,
)


def test_golden_and_death_cross_on_handmade_series() -> None:
    # close=100; reconstruct ema50 = close * (1 - pos)
    # bar0: fast 99, slow 100  (pos = 0.01)
    # bar1: fast 100, slow 100 (pos = 0)     still not strictly >
    # bar2: fast 101, slow 100 (pos = -0.01) golden
    # bar3: fast 99, slow 100  (pos = 0.01)  death
    close = pd.Series([100.0, 100.0, 100.0, 100.0])
    ema_50_position = pd.Series([0.01, 0.0, -0.01, 0.01])
    ema_200 = pd.Series([100.0, 100.0, 100.0, 100.0])
    out = compute_ema_50_200_cross_from_series(
        close=close, ema_50_position=ema_50_position, ema_200=ema_200
    )
    assert list(out["ema_50_200_cross_up"]) == [0.0, 0.0, 1.0, 0.0]
    assert list(out["ema_50_200_cross_down"]) == [0.0, 0.0, 0.0, 1.0]
    assert list(out["ema_50_200_cross_side"]) == [0.0, 0.0, 1.0, -1.0]


def test_node_is_registered() -> None:
    from src.features.registry import ensure_features_registered, get_compute_func

    ensure_features_registered()
    fn = get_compute_func("compute_ema_50_200_cross_from_series")
    assert fn is compute_ema_50_200_cross_from_series
