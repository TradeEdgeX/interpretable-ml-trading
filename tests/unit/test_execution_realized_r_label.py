"""Unit tests for execution-aligned realized-R labels."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.time_series_model.strategies.labels.execution_config_utils import (
    normalize_execution_config,
    normalize_take_profit_block,
)
from src.time_series_model.strategies.labels.execution_realized_r_label import (
    compute_realized_r_under_execution,
)


def test_normalize_take_profit_r_to_target_r():
    raw = {"enabled": True, "r": 1.0}
    out = normalize_take_profit_block(raw)
    assert out["target_r"] == 1.0
    assert "r" not in out


def test_signed_label_long_favorable_synthetic():
    """Upward drift: long sim should beat short sim under tight TP."""
    n = 40
    close = np.linspace(100.0, 110.0, n)
    high = close + 0.5
    low = close - 0.5
    atr = np.full(n, 1.0)
    df = pd.DataFrame(
        {
            "close": close,
            "high": high,
            "low": low,
            "atr": atr,
            "symbol": "TEST",
        }
    )
    block = compute_realized_r_under_execution(
        df,
        exec_config={
            "stop_loss": {"initial_r": 50.0, "trailing": {"enabled": False}},
            "take_profit": {"enabled": True, "target_r": 1.0},
            "holding": {"max_holding_bars": 6, "time_stop_bars": 6},
        },
        rr_floor=0.0,
    )
    # Mid-sample bars with enough forward path should lean long
    mid = block.iloc[10:20]
    assert mid["realized_r_long"].notna().any()
    assert (mid["label"] > 0).mean() > 0.5


def test_tp_target_r_used_not_default():
    cfg = normalize_execution_config(
        {
            "take_profit": {"enabled": True, "target_r": 0.5},
            "stop_loss": {"initial_r": 50.0},
        }
    )
    assert cfg["take_profit"]["target_r"] == 0.5


def test_tp_trigger_matches_target_r():
    """TP uses target_r × ATR price level; realized R = target_r / initial_r at TP."""
    close = np.array([100.0, 100.0, 102.0, 102.0, 102.0])
    high = close + 0.1
    low = close - 0.1
    atr = np.full(len(close), 1.0)
    df = pd.DataFrame(
        {"close": close, "high": high, "low": low, "atr": atr, "symbol": "TEST"}
    )
    initial_r = 1.5
    target_r = 1.0
    block = compute_realized_r_under_execution(
        df,
        exec_config={
            "stop_loss": {"initial_r": initial_r, "trailing": {"enabled": False}},
            "take_profit": {"enabled": True, "target_r": target_r},
            "holding": {"max_holding_bars": 6, "time_stop_bars": 6},
        },
        rr_floor=0.0,
    )
    r_long = block.iloc[0]["realized_r_long"]
    assert pd.notna(r_long)
    expected = target_r / initial_r
    assert abs(float(r_long) - expected) < 0.05
