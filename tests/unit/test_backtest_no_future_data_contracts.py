"""CI contracts: backtest harnesses must not consume future (same-bar-at-open) data.

Canonical: docs/architecture/20260724_timeline_open_bar_feature_lookahead_SEVERE_CN.md
Cursor: .cursor/rules/backtest-no-future-data.mdc
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.backtest_multileg_timeline import _lookup, _prev_signal_ts
from scripts.event_backtest import _align_feature_index_to_bar_close
from src.time_series_model.grid.subbar_replay import (
    merge_signal_features_onto_execution_bars,
    timeframe_to_timedelta,
)

ROOT = Path(__file__).resolve().parents[2]


def test_timeline_source_keeps_closed_bar_decision_path() -> None:
    """Static gate: do not silently drop the closed-bar lag from the hot path."""
    src = (ROOT / "scripts" / "backtest_multileg_timeline.py").read_text(
        encoding="utf-8"
    )
    assert "def _prev_signal_ts" in src
    assert "closed_ts = _prev_signal_ts(fire_ts, lane_freq)" in src
    assert "_lookup(feat_map, sym, closed_ts)" in src
    assert '"closed_bar_decision": True' in src
    # Regression: never look up features at fire_ts in the on_bar block again.
    assert "_lookup(feat_map, sym, fire_ts)" not in src
    assert "_lookup(feats, sym, fire_ts)" not in src


def test_timeline_decision_pair_never_sees_forming_or_same_open_row() -> None:
    """Wall T + prev closed = only knowable row under 2h_close."""
    idx = pd.to_datetime(
        [
            "2026-07-23 18:00:00",
            "2026-07-23 20:00:00",
            "2026-07-23 22:00:00",
        ],
        utc=True,
    )
    feats = {
        "BTCUSDT": pd.DataFrame(
            {"grid_semantic_chop": [0.0, 0.75, 0.99], "close": [1.0, 2.0, 3.0]},
            index=idx,
        )
    }
    fire = pd.Timestamp("2026-07-23 22:00:00", tz="UTC")
    closed = _prev_signal_ts(fire, "2h")
    assert closed == idx[1]
    row = _lookup(feats, "BTCUSDT", closed)
    assert row["grid_semantic_chop"] == 0.75
    assert _lookup(feats, "BTCUSDT", fire)["grid_semantic_chop"] == 0.99  # poison


def test_event_backtest_align_hides_unclosed_open_labelled_bar() -> None:
    """Open-labelled 02:00 bar is not visible before its close (04:00)."""
    raw_idx = pd.to_datetime(["2024-01-01 00:00:00", "2024-01-01 02:00:00"], utc=True)
    raw = pd.DataFrame({"signal": [1, 2]}, index=raw_idx)
    aligned = _align_feature_index_to_bar_close(raw, "120T")
    mid = pd.Timestamp("2024-01-01 03:00:00", tz="UTC")
    assert len(raw[raw.index <= mid]) == 2  # naive leak
    assert len(aligned[aligned.index <= mid]) == 1
    assert float(aligned[aligned.index <= mid]["signal"].iloc[-1]) == 1.0


def test_chop_grid_merge_asof_requires_signal_bar_close() -> None:
    """With signal_bar_delta, mid-bar exec must not see the still-forming signal row."""
    sig_idx = pd.to_datetime(["2026-07-23 18:00:00", "2026-07-23 20:00:00"], utc=True)
    df_sig = pd.DataFrame(
        {"grid_semantic_chop": [0.10, 0.99], "box_pos_60": [0.2, 0.9]},
        index=sig_idx,
    )
    # Execution bar inside (20:00, 22:00) — 20:00 signal not closed yet.
    exec_idx = pd.to_datetime(["2026-07-23 21:00:00"], utc=True)
    ohlc = pd.DataFrame(
        {"open": [1.0], "high": [1.1], "low": [0.9], "close": [1.0], "volume": [1.0]},
        index=exec_idx,
    )
    delta = timeframe_to_timedelta("2h")
    joined = merge_signal_features_onto_execution_bars(
        ohlc, df_sig, signal_bar_delta=delta
    )
    assert float(joined["grid_semantic_chop"].iloc[0]) == 0.10

    # Without delta → LEAK (documents why delta is mandatory on research path).
    leaked = merge_signal_features_onto_execution_bars(
        ohlc, df_sig, signal_bar_delta=None
    )
    assert float(leaked["grid_semantic_chop"].iloc[0]) == 0.99
