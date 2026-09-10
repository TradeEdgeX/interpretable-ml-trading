"""Tests for entry RR snotio scan kernel."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.research.execution_kernel.entry_rr_scan import (
    prepare_entry_rr_frame,
    scan_snotio_entry_rr_thresholds,
)


def _synthetic_entry_rr_df(n: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    close = 100 + np.cumsum(rng.normal(0, 0.5, size=n))
    high = close + rng.uniform(0.1, 0.5, size=n)
    low = close - rng.uniform(0.1, 0.5, size=n)
    feat = rng.normal(size=n)
    direction = np.where(feat > 0, 1.0, -1.0)
    return pd.DataFrame(
        {
            "symbol": ["BTC"] * n,
            "high": high,
            "low": low,
            "close": close,
            "atr": np.full(n, 1.0),
            "entry_direction": direction,
            "gate_decision": ["allow"] * n,
            "pulse_z": feat,
        }
    )


def test_prepare_entry_rr_frame_applies_gate():
    df = _synthetic_entry_rr_df(50)
    df.loc[0:5, "gate_decision"] = "deny"
    prepared = prepare_entry_rr_frame(df, "srb", apply_gate=True)
    assert int((prepared["entry_direction"] != 0).sum()) == 44


def test_scan_snotio_entry_rr_thresholds_runs():
    df = _synthetic_entry_rr_df(200)
    prepared = prepare_entry_rr_frame(df, "srb", apply_gate=False)
    exec_config = {
        "stop_loss": {"type": "fixed", "initial_r": 1.5},
        "take_profit": {"enabled": True, "target_r": 3.0},
        "holding": {"max_holding_bars": 50, "time_stop_bars": 50},
    }
    mask = pd.Series(True, index=prepared.index)
    rows = scan_snotio_entry_rr_thresholds(
        prepared,
        "pulse_z",
        "<=",
        [0.0, 0.5, 1.0],
        mask,
        exec_config,
        min_trades=5,
    )
    assert len(rows) == 3
    assert all(r.get("sim") == "entry_rr" for r in rows)
    assert any(not r["too_few"] for r in rows)


def test_prepare_entry_rr_frame_missing_ohlc_raises():
    df = pd.DataFrame({"entry_direction": [1.0, -1.0], "pulse_z": [0.1, 0.2]})
    with pytest.raises(ValueError, match="missing columns"):
        prepare_entry_rr_frame(df, "srb")


def test_prepare_entry_rr_frame_prefers_strategy_breakout_direction():
    n = 10
    df = pd.DataFrame(
        {
            "high": [101.0] * n,
            "low": [99.0] * n,
            "close": [100.0] * n,
            "atr": [1.0] * n,
            "bpc_breakout_direction": [1.0] * n,
            "tpc_breakout_direction": [-1.0] * n,
            "pulse_z": np.linspace(-1, 1, n),
        }
    )
    prepared_tpc = prepare_entry_rr_frame(df, "tpc", apply_gate=False)
    assert (prepared_tpc["entry_direction"] == -1.0).all()
    prepared_bpc = prepare_entry_rr_frame(df, "bpc", apply_gate=False)
    assert (prepared_bpc["entry_direction"] == 1.0).all()
