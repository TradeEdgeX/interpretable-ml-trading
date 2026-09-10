"""Unit tests for dual_add max_adds ablation aggregation helpers."""

from __future__ import annotations

import pandas as pd
import pytest

from scripts.experiment_dual_add_max_adds_ablation import (
    _forward_blacklist,
    _portfolio_dd_from_segments,
    _segment_quantiles,
)


def test_portfolio_dd_from_segments_ordered():
    segments = pd.DataFrame(
        {
            "end": pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC"),
            "pnl_per_capital": [0.01, 0.02, -0.05, 0.01],
        }
    )
    dd = _portfolio_dd_from_segments(segments)
    assert dd == pytest.approx(-0.05)


def test_segment_quantiles_empty():
    q = _segment_quantiles(pd.DataFrame())
    assert pd.isna(q["mean_segment_pnl"])


def test_forward_blacklist_rejects_max_adds():
    with pytest.raises(ValueError):
        _forward_blacklist(["--max-adds-per-side", "2"])
