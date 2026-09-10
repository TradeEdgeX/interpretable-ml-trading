import numpy as np
import pandas as pd

from src.time_series_model.strategies.labels.triple_barrier_meta import (
    compute_meta_label,
    compute_triple_barrier_result,
    primary_side_from_position,
)


def _bars(path: list[tuple[float, float, float]]) -> pd.DataFrame:
    rows = [{"close": c, "high": h, "low": lo, "atr": 1.0} for c, h, lo in path]
    return pd.DataFrame(rows)


def test_long_hits_tp_first() -> None:
    path = [(100.0, 100.0, 100.0)]
    path.append((100.5, 101.2, 100.2))
    path.extend([(100.5, 100.6, 100.4)] * 19)
    side = pd.Series([1.0] + [0.0] * 20)
    b = compute_triple_barrier_result(_bars(path), side, horizon=20)
    assert b.iloc[0] == 1.0
    assert compute_meta_label(b).iloc[0] == 1.0


def test_long_hits_sl_first() -> None:
    path = [(100.0, 100.0, 100.0)]
    path.append((99.5, 99.8, 98.8))
    path.extend([(105.0, 105.0, 104.5)] * 19)
    side = pd.Series([1.0] + [0.0] * 20)
    b = compute_triple_barrier_result(_bars(path), side, horizon=20)
    assert b.iloc[0] == -1.0
    assert compute_meta_label(b).iloc[0] == 0.0


def test_timeout_is_zero_not_correct() -> None:
    path = [(100.0, 100.0, 100.0)]
    path.extend([(100.2, 100.4, 99.9)] * 20)
    side = pd.Series([1.0] + [0.0] * 20)
    b = compute_triple_barrier_result(_bars(path), side, horizon=20)
    assert b.iloc[0] == 0.0
    assert compute_meta_label(b).iloc[0] == 0.0


def test_same_bar_both_barriers_sl_wins() -> None:
    path = [(100.0, 100.0, 100.0), (100.0, 101.5, 98.5)]
    path.extend([(100.0, 100.1, 99.9)] * 19)
    side = pd.Series([1.0] + [0.0] * 20)
    b = compute_triple_barrier_result(_bars(path), side, horizon=20)
    assert b.iloc[0] == -1.0


def test_short_hits_tp_below() -> None:
    path = [(100.0, 100.0, 100.0), (99.2, 99.4, 98.7)]
    path.extend([(99.5, 99.6, 99.4)] * 19)
    side = pd.Series([-1.0] + [0.0] * 20)
    b = compute_triple_barrier_result(_bars(path), side, horizon=20)
    assert b.iloc[0] == 1.0


def test_no_side_stays_nan() -> None:
    path = [(100.0, 100.0, 100.0)] + [(101.0, 102.0, 100.0)] * 20
    side = pd.Series([0.0] * 21)
    b = compute_triple_barrier_result(_bars(path), side, horizon=20)
    assert np.isnan(b.iloc[0])
    assert np.isnan(compute_meta_label(b).iloc[0])


def test_primary_side_deadzone() -> None:
    pos = pd.Series([0.20, -0.15, 0.05, np.nan])
    side = primary_side_from_position(pos, deadzone=0.10)
    assert list(side.iloc[:3]) == [1.0, -1.0, 0.0]
    assert np.isnan(side.iloc[3]) or side.iloc[3] == 0.0
