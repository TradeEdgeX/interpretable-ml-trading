"""Tests for dapi position normalization and coin OM execution reports."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from order_management.coin_order_manager import CoinOrderManager
from order_management.exchange.trend_position_views import (
    map_report_to_signal_symbol,
    normalize_dapi_positions,
    signal_symbol_for_exec,
)
from order_management.models import OrderSide, OrderStatus, OrderType


def test_signal_symbol_for_exec_sol() -> None:
    assert signal_symbol_for_exec("SOLUSD_PERP") == "SOLUSDT"


def test_normalize_dapi_position_risk_row() -> None:
    rows = normalize_dapi_positions(
        [
            {
                "symbol": "SOLUSD_PERP",
                "positionAmt": "12",
                "entryPrice": "150.5",
                "markPrice": "151.0",
                "unRealizedProfit": "0.001",
                "leverage": "5",
            }
        ]
    )
    assert len(rows) == 1
    assert rows[0]["symbol"] == "SOLUSDT"
    assert rows[0]["exec_symbol"] == "SOLUSD_PERP"
    assert rows[0]["size"] == 12.0
    assert rows[0]["side"] == "long"
    assert rows[0]["notional"] == 120.0


def test_normalize_dapi_row_omits_notional_when_mapping_unknown() -> None:
    """No mapping → leave the guard's size*mark fallback rather than gross=0."""
    rows = normalize_dapi_positions(
        [
            {
                "symbol": "DOGEUSD_PERP",
                "positionAmt": "5",
                "entryPrice": "0.2",
                "markPrice": "0.21",
            }
        ]
    )
    assert len(rows) == 1
    assert "notional" not in rows[0]


def test_map_report_to_signal_symbol() -> None:
    mapped = map_report_to_signal_symbol(
        {"symbol": "SOLUSD_PERP", "order_id": "1", "side": "BUY"}
    )
    assert mapped["symbol"] == "SOLUSDT"
    assert mapped["exec_symbol"] == "SOLUSD_PERP"


def test_coin_order_manager_handle_execution_report_delegates() -> None:
    storage = MagicMock()
    storage.get_order_by_binance_id.return_value = None
    storage.get_order_by_client_id.return_value = None
    om = CoinOrderManager(storage, exchange=MagicMock(), shadow=False)
    delegate = MagicMock()
    expected = MagicMock()
    delegate.handle_execution_report.return_value = expected
    om._order_sync = delegate

    report = {"symbol": "SOLUSD_PERP", "order_id": "99", "status": "FILLED"}
    out = om.handle_execution_report(report)
    assert out is expected
    args = delegate.handle_execution_report.call_args[0][0]
    assert args["symbol"] == "SOLUSDT"


def test_dapi_user_stream_import() -> None:
    from order_management.exchange.dapi_user_stream import DapiUserStream

    assert DapiUserStream is not None
