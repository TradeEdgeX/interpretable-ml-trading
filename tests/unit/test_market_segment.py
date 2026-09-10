"""Tests for config/market_segment.yaml loader."""

from __future__ import annotations

import pytest

from scripts.event_backtest.market_segment import (
    expand_segment_matrix,
    load_market_segments,
    resolve_segment_run,
)
from scripts.event_backtest.variant_grid import _normalize_runs


def test_load_market_segments_has_canonical_ids() -> None:
    segs = load_market_segments()
    assert "bear_2022" in segs
    assert "bull_2023_2024" in segs
    assert segs["bull_2023_2024"]["start_date"] == "2023-06-01"
    assert segs["bull_2023_2024"]["end_date"] == "2025-01-01"


def test_resolve_segment_run_fills_dates() -> None:
    run = resolve_segment_run({"variant": "x", "segment": "bear_2022"})
    assert run["start_date"] == "2022-01-01"
    assert run["end_date"] == "2023-11-01"
    assert run.get("segment_label") == "Bear"


def test_expand_segment_matrix() -> None:
    grid = {
        "segment_matrix": {
            "segments": ["bear_2022", "recent"],
            "variants": [
                {"suffix": "G0", "strategies_root": "config/strategies"},
            ],
        }
    }
    runs = expand_segment_matrix(grid)
    assert len(runs) == 2
    ids = {r["segment"] for r in runs}
    assert ids == {"bear_2022", "recent"}


def test_normalize_runs_from_segment_matrix() -> None:
    grid = {
        "strategy": "tpc",
        "segment_matrix": {
            "segments": ["bear_2022"],
            "variants": [{"suffix": "G0", "strategies_root": "config/strategies"}],
        },
    }
    runs = _normalize_runs(grid)
    assert len(runs) == 1
    assert runs[0]["start_date"] == "2022-01-01"


def test_resolve_unknown_segment_raises() -> None:
    with pytest.raises(KeyError):
        resolve_segment_run({"segment": "no_such_segment"})
