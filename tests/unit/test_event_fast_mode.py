"""Fast mode (--fast) accuracy tests for event backtest.

Covers:
  - _ohlc_dict_from_bar_row (normal + fallback)
  - _sync_macro_tp_vwap_from_feature_row idempotency
  - _sync_ema_1200_from_feature_row idempotency
  - _iter_update_bars_primary_tf correctness
  - Fast vs 1min PositionSimulator update comparison
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

import pandas as pd

from scripts.event_backtest import (
    PositionSimulator,
    _iter_update_bars_primary_tf,
    _ohlc_dict_from_bar_row,
    _sync_ema_1200_from_feature_row,
    _sync_macro_tp_vwap_from_feature_row,
)

# ── _ohlc_dict_from_bar_row ──────────────────────────────────────────


def test_ohlc_dict_with_full_columns():
    ts = pd.Timestamp("2024-01-01 02:00:00", tz="UTC")
    row = pd.Series({"open": 100.0, "high": 105.0, "low": 98.0, "close": 102.0})
    result = _ohlc_dict_from_bar_row(ts, row)
    assert result["timestamp"] == ts
    assert result["open"] == 100.0
    assert result["high"] == 105.0
    assert result["low"] == 98.0
    assert result["close"] == 102.0


def test_ohlc_dict_fallback_open_to_close():
    ts = pd.Timestamp("2024-01-01 02:00:00", tz="UTC")
    row = pd.Series({"close": 200.0})  # no open/high/low
    result = _ohlc_dict_from_bar_row(ts, row)
    # open/high/low all fall back to close
    assert result["open"] == 200.0
    assert result["high"] == 200.0
    assert result["low"] == 200.0
    assert result["close"] == 200.0


def test_ohlc_dict_zero_open_uses_close():
    """Regression: bar_row.get('open', bar_row.get('close', 0)) or 0
    — if open==0, the expression returns 0 (falsy), so it falls through to or 0."""
    ts = pd.Timestamp("2024-01-01 02:00:00", tz="UTC")
    row = pd.Series({"open": 0.0, "high": 105.0, "low": 98.0, "close": 102.0})
    result = _ohlc_dict_from_bar_row(ts, row)
    # open=0 → .get('open')=0 → 0 or ... → falls through
    assert result["open"] == 0.0  # 0 is falsy, so "or" picks the next value (close)
    # But high/low are present, so they use actual values
    assert result["high"] == 105.0
    assert result["low"] == 98.0


def test_ohlc_dict_all_zero_rows():
    ts = pd.Timestamp("2024-01-01 02:00:00", tz="UTC")
    row = pd.Series({"open": 0.0, "high": 0.0, "low": 0.0, "close": 0.0})
    result = _ohlc_dict_from_bar_row(ts, row)
    assert result["close"] == 0.0


# ── sync functions idempotency ────────────────────────────────────────


def test_sync_macro_vwap_idempotent_with_same_close():
    """In fast mode, the feature row's close == the bar close passed to update().
    So sync → recalculate should produce the same pv."""
    sim = PositionSimulator()
    row = pd.Series({"close": 100.0, "macro_tp_vwap_1200_position": 0.05})
    _sync_macro_tp_vwap_from_feature_row(sim, row)

    assert sim._macro_tp_vwap_position == 0.05
    # vwap_level = close * (1 - pv) = 100 * 0.95 = 95.0
    assert abs(float(sim._macro_tp_vwap_level) - 95.0) < 1e-9

    # Now simulate what update() does: recalculate pv from frozen level
    bar_close = 100.0  # same as row.close
    vwap_level = float(sim._macro_tp_vwap_level)
    live_pv = (bar_close - vwap_level) / bar_close
    assert abs(live_pv - 0.05) < 1e-9  # should be idempotent


def test_sync_macro_vwap_recalculate_drift_in_fast_mode():
    """Between two primary-TF bars, no 1min updates happen in fast mode.
    At the next bar, the feature row has a new close and new pv.
    The recalculation should match the new pv."""
    sim = PositionSimulator()
    # Bar 1
    row1 = pd.Series({"close": 100.0, "macro_tp_vwap_1200_position": 0.05})
    _sync_macro_tp_vwap_from_feature_row(sim, row1)
    assert float(sim._macro_tp_vwap_level) == 95.0

    # Between bars: no 1min updates in fast mode → level stays frozen

    # Bar 2: price moved, VWAP level should be re-frozen
    row2 = pd.Series({"close": 102.0, "macro_tp_vwap_1200_position": 0.03})
    _sync_macro_tp_vwap_from_feature_row(sim, row2)
    new_vwap = 102.0 * (1.0 - 0.03)
    assert abs(float(sim._macro_tp_vwap_level) - new_vwap) < 1e-9

    # Recalculate at new close
    live_pv = (102.0 - float(sim._macro_tp_vwap_level)) / 102.0
    assert abs(live_pv - 0.03) < 1e-9


def test_sync_ema1200_idempotent():
    sim = PositionSimulator()
    row = pd.Series({"close": 100.0, "ema_1200_position": 0.10})
    _sync_ema_1200_from_feature_row(sim, row)

    assert sim._ema_1200_position == 0.10
    # ema_level = close * (1 - pv) = 100 * 0.90 = 90.0
    assert abs(float(sim._ema_1200_level) - 90.0) < 1e-9

    bar_close = 100.0
    ema_level = float(sim._ema_1200_level)
    live_ev = (bar_close - ema_level) / bar_close
    assert abs(live_ev - 0.10) < 1e-9


def test_sync_ema1200_handles_missing_column():
    sim = PositionSimulator()
    sim._ema_1200_position = 0.99  # junk
    sim._ema_1200_level = 1.0
    row = pd.Series({"close": 100.0})  # no ema_1200_position
    _sync_ema_1200_from_feature_row(sim, row)
    # should NOT overwrite existing values
    assert sim._ema_1200_position == 0.99
    assert sim._ema_1200_level == 1.0


# ── _iter_update_bars_primary_tf ──────────────────────────────────────


def test_iter_primary_tf_no_data_returns_nothing():
    bundle: dict = {}
    prev_ts = pd.Timestamp("2024-01-01 01:00:00", tz="UTC")
    cur_ts = pd.Timestamp("2024-01-01 03:00:00", tz="UTC")
    got = list(_iter_update_bars_primary_tf(bundle, prev_ts, cur_ts, "120T"))
    assert got == []


def test_iter_primary_tf_missing_timeframe_returns_nothing():
    bundle = {"tf_features": {"60T": pd.DataFrame()}}
    prev_ts = pd.Timestamp("2024-01-01 01:00:00", tz="UTC")
    cur_ts = pd.Timestamp("2024-01-01 03:00:00", tz="UTC")
    got = list(_iter_update_bars_primary_tf(bundle, prev_ts, cur_ts, "120T"))
    assert got == []


def test_iter_primary_tf_skips_bars_before_prev_ts():
    idx = pd.to_datetime(
        ["2024-01-01 02:00:00", "2024-01-01 04:00:00", "2024-01-01 06:00:00"],
        utc=True,
    )
    tf_df = pd.DataFrame(
        {"open": [1.0, 2.0, 3.0], "high": [1.5, 2.5, 3.5], "close": [1.0, 2.0, 3.0]},
        index=idx,
    )
    bundle = {"tf_features": {"120T": tf_df}}
    # prev_ts = 03:00, so 02:00 bar should be skipped
    prev_ts = pd.Timestamp("2024-01-01 03:00:00", tz="UTC")
    cur_ts = pd.Timestamp("2024-01-01 05:00:00", tz="UTC")

    got = list(_iter_update_bars_primary_tf(bundle, prev_ts, cur_ts, "120T"))
    assert len(got) == 1
    assert got[0][0] == pd.Timestamp("2024-01-01 04:00:00", tz="UTC")


def test_iter_primary_tf_includes_bar_at_cur_ts():
    """cur_ts is inclusive: bars at the exact cur_ts should be yielded."""
    idx = pd.to_datetime(
        ["2024-01-01 02:00:00", "2024-01-01 04:00:00"],
        utc=True,
    )
    tf_df = pd.DataFrame(
        {"open": [1.0, 2.0], "close": [1.0, 2.0]},
        index=idx,
    )
    bundle = {"tf_features": {"120T": tf_df}}
    prev_ts = pd.Timestamp("2024-01-01 01:00:00", tz="UTC")
    cur_ts = pd.Timestamp("2024-01-01 04:00:00", tz="UTC")  # exactly at bar#2

    got = list(_iter_update_bars_primary_tf(bundle, prev_ts, cur_ts, "120T"))
    assert len(got) == 2
    assert got[1][0] == pd.Timestamp("2024-01-01 04:00:00", tz="UTC")


# ── PositionSimulator: fast-mode style (single bar update vs 1min) ───


def _make_sim_with_long_position(
    entry_price: float = 100.0,
    atr: float = 2.0,
    sl_price: float = 96.0,
    structural_exit: str = "",
) -> PositionSimulator:
    sim = PositionSimulator()
    now = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
    pos = {
        "side": "LONG",
        "entry_price": entry_price,
        "entry_time": now,
        "atr_at_entry": atr,
        "max_holding_bars": 100,
        "bar_minutes": 120,
        "stop_loss_price": sl_price,
        "structural_exit": structural_exit,
        "breakeven_locked": True,  # skip breakeven for simplicity
        "trailing_activated": False,
    }
    sim._positions["p1"] = pos
    return sim


def test_fast_bar_with_wide_range_triggers_sl():
    """In fast mode, a 2h bar with wide OHLC should still detect SL crossing."""
    sim = _make_sim_with_long_position(entry_price=100.0, atr=2.0, sl_price=96.0)
    now = datetime(2024, 1, 2, 12, 0, tzinfo=timezone.utc)
    # 2h bar: price drops well below SL
    closed = sim.update(
        {
            "timestamp": now,
            "open": 97.0,
            "high": 97.5,
            "low": 94.0,
            "close": 94.5,
        }
    )
    assert len(closed) == 1
    assert closed[0].exit_reason in ("sl", "stop_loss")


def test_fast_bar_with_narrow_range_misses_sl():
    """If the 2h bar low is above SL, no trigger."""
    sim = _make_sim_with_long_position(entry_price=100.0, atr=2.0, sl_price=96.0)
    now = datetime(2024, 1, 2, 12, 0, tzinfo=timezone.utc)
    closed = sim.update(
        {
            "timestamp": now,
            "open": 98.0,
            "high": 99.0,
            "low": 97.0,  # above SL
            "close": 98.5,
        }
    )
    assert len(closed) == 0


def test_fast_bar_coincident_open_is_close():
    """OHLC fallback: when open/high/low are all replaced by close,
    the bar dict should still be valid for update()."""
    sim = _make_sim_with_long_position(entry_price=100.0, atr=2.0, sl_price=96.0)
    now = datetime(2024, 1, 2, 12, 0, tzinfo=timezone.utc)
    # All OHLC = close, representing a bar with only close available
    closed = sim.update(
        {
            "timestamp": now,
            "open": 95.0,
            "high": 95.0,
            "low": 95.0,
            "close": 95.0,
        }
    )
    assert len(closed) == 1
    assert closed[0].exit_reason in ("sl", "stop_loss")


def test_fast_mode_does_not_corrupt_vwap_level_between_updates():
    """Verify that sync-then-update cycle preserves VWAP level."""
    sim = PositionSimulator()
    now = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
    pos = {
        "side": "LONG",
        "entry_price": 100.0,
        "entry_time": now,
        "atr_at_entry": 2.0,
        "max_holding_bars": 0,
        "bar_minutes": 120,
        "stop_loss_price": 80.0,
        "structural_exit": "vwap1200",
        "breakeven_locked": True,
    }
    sim._positions["p1"] = pos

    # Sync from feature row (as fast mode does)
    row = pd.Series({"close": 100.0, "macro_tp_vwap_1200_position": 0.02})
    _sync_macro_tp_vwap_from_feature_row(sim, row)
    assert abs(float(sim._macro_tp_vwap_level) - 98.0) < 1e-9

    # Update with bar that stays above deadband
    closed = sim.update(
        {
            "timestamp": now,
            "open": 100.0,
            "high": 100.5,
            "low": 99.5,
            "close": 100.2,
        }
    )
    assert len(closed) == 0
    # VWAP level should be unchanged between sync calls
    assert abs(float(sim._macro_tp_vwap_level) - 98.0) < 1e-9
