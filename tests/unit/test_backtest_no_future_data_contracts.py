"""CI contracts: the public court must not consume future (same-bar-at-open) data.

Canonical: docs/lessons.md#closed-bar · docs/math.md §1
Cursor: .cursor/rules/backtest-no-future-data.mdc
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.event_backtest import (
    _align_feature_index_to_bar_close,
    _feature_asof_from_sym_tf_features,
)

ROOT = Path(__file__).resolve().parents[2]


def test_event_backtest_source_keeps_closed_bar_align_path() -> None:
    """Static gate: do not silently drop close-align from the feature-load path."""
    src = (ROOT / "scripts" / "event_backtest" / "backtester.py").read_text(
        encoding="utf-8"
    )
    assert "features_df = _align_feature_index_to_bar_close(features_df, tf)" in src
    assert src.count("_align_feature_index_to_bar_close(features_df, tf)") >= 2


def test_event_backtest_align_hides_unclosed_open_labelled_bar() -> None:
    """Open-labelled 02:00 bar is not visible before its close (04:00)."""
    raw_idx = pd.to_datetime(["2024-01-01 00:00:00", "2024-01-01 02:00:00"], utc=True)
    raw = pd.DataFrame({"signal": [1, 2]}, index=raw_idx)
    aligned = _align_feature_index_to_bar_close(raw, "120T")
    mid = pd.Timestamp("2024-01-01 03:00:00", tz="UTC")
    assert len(raw[raw.index <= mid]) == 2  # naive leak
    assert len(aligned[aligned.index <= mid]) == 1
    assert float(aligned[aligned.index <= mid]["signal"].iloc[-1]) == 1.0


def test_feature_asof_does_not_see_later_close_aligned_row() -> None:
    """asof at 03:00 must not read the 04:00-close row (open-labelled 02:00)."""
    raw_idx = pd.to_datetime(
        ["2024-01-01 00:00:00", "2024-01-01 02:00:00"], utc=True
    )
    raw = pd.DataFrame({"chop": [0.10, 0.99]}, index=raw_idx)
    aligned = _align_feature_index_to_bar_close(raw, "120T")
    bundle = {"tf_features": {"120T": aligned}}
    mid = pd.Timestamp("2024-01-01 03:00:00", tz="UTC")
    assert _feature_asof_from_sym_tf_features(bundle, mid, "chop") == 0.10
    close = pd.Timestamp("2024-01-01 04:00:00", tz="UTC")
    assert _feature_asof_from_sym_tf_features(bundle, close, "chop") == 0.99
