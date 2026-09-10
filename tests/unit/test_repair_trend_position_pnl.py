"""Unit tests for exchange close matching in repair_trend_position_pnl_from_exchange."""

from __future__ import annotations

from scripts.repair_trend_position_pnl_from_exchange import (
    _needs_repair,
    _pick_exchange_close,
    _realized_pnl,
)


def test_needs_repair_exchange_sync_flat() -> None:
    assert _needs_repair(
        {
            "status": "closed",
            "exit_time": "2026-06-12T02:21:53+00:00",
            "exit_reason": "exchange_sync_flat",
            "realized_pnl": 0.0,
            "entry_price": 100.0,
            "exit_price": 100.0,
        }
    )


def test_needs_repair_skip_when_pnl_and_prices_ok() -> None:
    assert not _needs_repair(
        {
            "status": "closed",
            "exit_time": "2026-05-18T07:46:41+00:00",
            "exit_reason": None,
            "realized_pnl": -6.64,
            "entry_price": 2254.11,
            "exit_price": 2123.9,
        }
    )


def test_pick_exchange_close_prefers_stop_for_short() -> None:
    entry_ms = 1_700_000_000_000
    exit_ms = entry_ms + 3600_000
    orders = [
        {
            "orderId": "1",
            "status": "FILLED",
            "side": "BUY",
            "positionSide": "SHORT",
            "type": "MARKET",
            "avgPrice": "1900",
            "executedQty": "0.1",
            "updateTime": exit_ms + 1000,
        },
        {
            "orderId": "2",
            "status": "FILLED",
            "side": "BUY",
            "positionSide": "SHORT",
            "type": "STOP_MARKET",
            "avgPrice": "1850",
            "executedQty": "0.1",
            "updateTime": exit_ms,
        },
    ]
    picked = _pick_exchange_close(
        orders, pos_side="short", entry_ms=entry_ms, exit_ms=exit_ms
    )
    assert picked is not None
    assert str(picked["orderId"]) == "2"


def test_realized_pnl_short() -> None:
    assert _realized_pnl("short", 2000.0, 1900.0, 0.5) == 50.0
