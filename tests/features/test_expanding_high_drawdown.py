import pandas as pd
import pytest

from src.features.time_series.baseline_features import (
    compute_expanding_high_drawdown,
)


def test_expanding_high_drawdown_is_point_in_time_and_bounded():
    idx = pd.date_range("2026-01-01", periods=4, freq="1D")
    close = pd.Series([100.0, 80.0, 120.0, 60.0], index=idx)
    high = pd.Series([100.0, 90.0, 125.0, 70.0], index=idx)

    got = compute_expanding_high_drawdown(close=close, high=high)[
        "expanding_high_drawdown"
    ]

    assert got.tolist() == pytest.approx([0.0, -0.2, -0.04, -0.52])
