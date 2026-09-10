"""Public court surface: no SRB hooks, no coin-m wallet construction."""

from __future__ import annotations

import inspect

import pytest

from scripts.event_backtest.backtester import EventBacktester
from scripts.event_backtest.simulator.position import PositionSimulator


def test_backtester_import_does_not_load_srb_hooks() -> None:
    src = inspect.getsource(EventBacktester)
    assert "SrbEventBacktestHooks" not in src
    assert "resolve_srb_add_path" not in src
    assert "CoinMarginState" not in src


def test_coin_m_court_refuses() -> None:
    with pytest.raises(ValueError, match="coin_m is not in this public court"):
        EventBacktester(strategies=["ma_cross"], margin_mode="coin_m")


def test_usd_m_simulator_needs_no_wallet() -> None:
    sim = PositionSimulator(margin_mode="usd_m")
    assert sim._coin_margin_state is None
    assert not sim._is_coin_m()
    assert sim._positions == {}
