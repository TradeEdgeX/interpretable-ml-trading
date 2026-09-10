"""Tests for scripts/simulate_exec_trail.py structural scale-out replay."""

from __future__ import annotations

import pandas as pd
import pytest

from scripts.simulate_exec_trail import attach_opposite_rails, simulate_trade


def _bars_with_flat_ols_rail() -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=120, freq="2h")
    n = len(idx)
    base = 100.0
    df = pd.DataFrame(
        {
            "open": [base] * n,
            "high": [base + 0.5] * n,
            "low": [base - 0.5] * n,
            "close": [base] * n,
            "atr": [1.0] * n,
            "ols_upper": [102.0] * n,
            "ols_lower": [98.0] * n,
            "range_upper": [101.0] * n,
            "range_lower": [99.0] * n,
        },
        index=idx,
    )
    # Post-entry bar: wick hits opposite OLS rail for a long
    df.loc[idx[60], ["high", "low", "close"]] = [103.0, 99.5, 100.0]
    df.loc[idx[61], ["high", "low", "close"]] = [100.5, 99.0, 99.2]
    return df


def test_scale_out_opposite_ols_partial_then_breakeven_runner():
    bars = _bars_with_flat_ols_rail()
    entry_time = bars.index[59]
    exec_cfg = {
        "stop_loss": {
            "initial_r": 1.0,
            "trailing": {"enabled": False},
        },
        "take_profit": {"enabled": False, "target_r": 2.0},
        "breakeven": {"enabled": False},
        "holding": {"time_stop_bars": 10},
    }
    so = {
        "enabled": True,
        "target": "opposite_ols",
        "fraction": 0.5,
        "move_sl_to_be_after_scale": True,
    }
    r = simulate_trade(
        bars,
        entry_time=entry_time,
        side="LONG",
        entry_price=100.0,
        atr_at_entry=1.0,
        exec_cfg=exec_cfg,
        scale_out=so,
    )
    assert r["scale_out_done"] is True
    assert r["scale_out_fill"] == pytest.approx(102.0)
    # 0.5 * 2R on first leg + 0.5 * ~-0.8R on runner stopped at BE 100
    assert r["pnl_r"] == pytest.approx(0.5 * 2.0 + 0.5 * 0.0)
    assert "scale_out" in r["exit_reason"] or r["exit_reason"] in (
        "scale_out+sl",
        "scale_out+breakeven_sl",
    )


def test_sl_same_bar_wins_before_scale_out():
    """If low hits SL and high would hit rail, stop-out first (conservative)."""
    idx = pd.date_range("2024-01-01", periods=5, freq="2h")
    df = pd.DataFrame(
        {
            "open": [100.0, 100.0, 100.0, 100.0, 100.0],
            "high": [100.5, 105.0, 100.0, 100.0, 100.0],
            "low": [99.5, 98.0, 100.0, 100.0, 100.0],
            "close": [100.0, 99.0, 100.0, 100.0, 100.0],
            "atr": [1.0] * 5,
            "ols_upper": [102.0] * 5,
            "ols_lower": [98.0] * 5,
            "range_upper": [101.0] * 5,
            "range_lower": [99.0] * 5,
        },
        index=idx,
    )
    r = simulate_trade(
        df,
        entry_time=idx[0],
        side="LONG",
        entry_price=100.0,
        atr_at_entry=1.0,
        exec_cfg={
            "stop_loss": {"initial_r": 1.0, "trailing": {"enabled": False}},
            "take_profit": {"enabled": False, "target_r": 2.0},
            "breakeven": {"enabled": False},
            "holding": {"time_stop_bars": 10},
        },
        scale_out={"enabled": True, "target": "opposite_ols", "fraction": 0.5},
    )
    assert r["scale_out_done"] is False
    assert r["pnl_r"] == pytest.approx(-1.0)


def test_attach_opposite_rails_matches_close_only_series():
    idx = pd.date_range("2024-06-01", periods=120, freq="2h")
    # Slight noise so OLS residual width is never exactly zero (avoids NaN rails).
    x = (
        pd.Series(range(120), index=idx, dtype=float)
        + 100.0
        + 0.01 * (pd.Series(range(120), index=idx) % 7)
    )
    raw = pd.DataFrame(
        {
            "open": x,
            "high": x + 0.2,
            "low": x - 0.2,
            "close": x,
            "atr": [1.0] * 120,
        },
        index=idx,
    )
    out = attach_opposite_rails(raw, ols_window=32, range_window=10)
    assert "ols_upper" in out.columns and "range_upper" in out.columns
    assert pd.notna(out["ols_upper"].iloc[-1])
    assert out["range_upper"].iloc[-1] >= out["close"].iloc[-1]
