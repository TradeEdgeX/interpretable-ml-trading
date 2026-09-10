"""Timeline account unified metrics."""

from __future__ import annotations

import pytest
import pandas as pd

from scripts.multileg_timeline_account import MultilegTimelineAccount
from scripts.pipeline.multileg_portfolio_metrics import sharpe_from_equity_snapshots
from src.order_management.mock_binance_api import MockBinanceAPI


def test_to_summary_includes_unified_metrics() -> None:
    mock = MockBinanceAPI()
    acct = MultilegTimelineAccount(initial_equity=10_000.0, mock=mock)
    mock.set_wallet(10_500.0)
    mock.closed_leg_pnls.extend([50.0, -20.0, 30.0])
    acct.snapshot_equity("2026-01-01")
    acct.snapshot_equity("2026-01-02")
    s = acct.to_summary(
        one_r_usdt=120.0,
        start="2026-01-01",
        end="2027-01-01",
    )
    assert s["return_pct"] == pytest.approx(5.0)
    assert s["return_pct_annualized"] == pytest.approx(5.0, rel=0.01)
    assert s["realized_pnl_usdt"] == pytest.approx(500.0)
    assert s["total_r"] == pytest.approx(500.0 / 120.0)
    assert s["closed_legs"] == 3
    assert s["win_rate"] == pytest.approx(2 / 3)
    assert s["sharpe_annualized"] == s["sharpe_daily"]


def test_sharpe_from_equity_snapshots_positive_trend() -> None:
    snaps = [
        {"ts": "2026-01-01", "equity": 100.0},
        {"ts": "2026-01-02", "equity": 101.0},
        {"ts": "2026-01-03", "equity": 102.0},
    ]
    assert sharpe_from_equity_snapshots(snaps) > 0


def test_cagr_pct_accepts_tz_aware_timestamps() -> None:
    from scripts.pipeline.multileg_portfolio_metrics import cagr_pct

    out = cagr_pct(
        equity_start=10_000.0,
        equity_end=11_000.0,
        start=pd.Timestamp("2025-01-01", tz="UTC"),
        end=pd.Timestamp("2026-01-01", tz="UTC"),
    )
    assert out == pytest.approx(10.0, rel=0.01)
