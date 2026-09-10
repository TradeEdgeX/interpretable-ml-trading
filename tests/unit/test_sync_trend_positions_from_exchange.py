"""Tests for exchange → SQLite trend position sync."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from scripts.sync_trend_positions_from_exchange import sync_trend_positions
from src.order_management.models import Position, PositionSide, PositionStatus
from src.order_management.storage import Storage


def _open_row(
    *,
    pid: str,
    symbol: str,
    side: str,
    qty: float,
    entry: float,
) -> Position:
    return Position(
        position_id=pid,
        symbol=symbol,
        side=PositionSide.LONG if side == "long" else PositionSide.SHORT,
        entry_time=datetime.now(timezone.utc),
        entry_price=entry,
        initial_size=qty,
        current_size=qty,
        total_cost=entry * qty,
        status=PositionStatus.OPEN,
        strategy_id="tpc",
    )


def test_sync_inserts_short_from_exchange(tmp_path: Path) -> None:
    db = tmp_path / "order_management.db"
    storage = Storage(str(db))
    api = MagicMock()
    api.get_positions.return_value = [
        {
            "symbol": "XRP/USDT:USDT",
            "side": "short",
            "size": 161.2,
            "entry_price": 1.104,
            "unrealized_pnl": -5.0,
        }
    ]
    report = sync_trend_positions(
        api=api,
        db_path=db,
        dry_run=False,
        close_stale=True,
    )
    assert report["summary"]["insert"] == 1
    rows = storage.get_open_positions("XRPUSDT")
    assert len(rows) == 1
    assert rows[0].side == PositionSide.SHORT
    assert rows[0].current_size == pytest.approx(161.2)


def test_sync_closes_stale_local_when_exchange_flat(tmp_path: Path) -> None:
    db = tmp_path / "order_management.db"
    storage = Storage(str(db))
    storage.create_position(
        _open_row(pid="old1", symbol="ETHUSDT", side="short", qty=0.2, entry=2100.0)
    )
    api = MagicMock()
    api.get_positions.return_value = []
    api.get_ticker_price.return_value = 2050.0
    report = sync_trend_positions(api=api, db_path=db, dry_run=False)
    assert report["summary"]["close"] == 1
    assert storage.get_open_positions() == []
    closed = storage.get_position("old1")
    assert closed is not None
    assert closed.status == PositionStatus.CLOSED
    assert closed.exit_reason == "exchange_sync_flat"
    assert closed.exit_price == pytest.approx(2050.0)
    assert closed.realized_pnl == pytest.approx((2100.0 - 2050.0) * 0.2)
    api.get_ticker_price.assert_called_once_with("ETHUSDT")
