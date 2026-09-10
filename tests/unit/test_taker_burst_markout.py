"""Synthetic 1s clock: absorbed fade vs displaced continuation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.research.execution_kernel.taker_burst_markout import (
    BurstSpec,
    attach_markout,
    detect_bursts,
    run_month,
    summarize_markout,
)


def _sec(n: int, *, start: str = "2026-04-01") -> pd.DataFrame:
    ts = pd.date_range(start, periods=n, freq="s", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": ts,
            "last": np.full(n, 100.0),
            "high": np.full(n, 100.0),
            "low": np.full(n, 100.0),
            "vwap": np.full(n, 100.0),
            "buy_qty": np.full(n, 1.0),
            "sell_qty": np.full(n, 1.0),
            "n_trades": np.full(n, 2.0),
        }
    )


def test_absorbed_burst_then_fade() -> None:
    spec = BurstSpec(
        window_s=3,
        lookback_s=80,
        vol_quantile=0.9,
        min_imbalance=0.5,
        refractory_s=10,
        pre_range_s=10,
        touch_bp=5.0,
        fee_rt_bp=10.0,
        horizons_s=(5, 30),
    )
    sec = _sec(200)
    # Quiet hour so P90 is low, then a 3s buy burst that does not break range.
    sec.loc[100:102, "buy_qty"] = 50.0
    sec.loc[100:102, "sell_qty"] = 0.0
    # Fade 8bp over the next 5s (still inside prior 10s range of 100).
    for i, px in enumerate([99.98, 99.96, 99.94, 99.93, 99.92], start=103):
        sec.loc[i, ["last", "high", "low", "vwap"]] = px
    events = detect_bursts(sec, spec)
    assert not events.empty
    burst = events.iloc[0]
    assert burst["sign"] == 1
    assert burst["sleeve"] == "absorbed"
    marked = attach_markout(sec, events, spec)
    assert marked["y_5s_bp"].iloc[0] < -5.0
    summary = summarize_markout(marked, spec, month="synth", symbol="BTCUSDT")
    abs_row = summary.loc[summary["sleeve"] == "absorbed"].iloc[0]
    assert abs_row["fade_net_5s_bp"] > abs_row["mom_net_5s_bp"]


def test_displaced_burst_continues() -> None:
    spec = BurstSpec(
        window_s=3,
        lookback_s=80,
        vol_quantile=0.9,
        min_imbalance=0.5,
        refractory_s=10,
        pre_range_s=10,
        touch_bp=5.0,
        fee_rt_bp=10.0,
        horizons_s=(5, 30),
    )
    sec = _sec(200)
    sec.loc[100:102, "buy_qty"] = 50.0
    sec.loc[100:102, "sell_qty"] = 0.0
    # Burst itself prints through the prior 100 high.
    for i, px in enumerate([100.2, 100.4, 100.6], start=100):
        sec.loc[i, ["last", "high", "low", "vwap"]] = px
        sec.loc[i, "high"] = px
    # Continues another 8bp.
    for i, px in enumerate([100.7, 100.75, 100.8, 100.82, 100.85], start=103):
        sec.loc[i, ["last", "high", "low", "vwap"]] = px
    events, summary = run_month(sec, spec, month="synth", symbol="BTCUSDT")
    assert not events.empty
    assert bool(events.iloc[0]["displaced"])
    assert events["y_5s_bp"].iloc[0] > 0
    disp = summary.loc[summary["sleeve"] == "displaced"].iloc[0]
    assert disp["p_same_first"] >= 0.0
    assert disp["mean_cont_5s_bp"] > 0
