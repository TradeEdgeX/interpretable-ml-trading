"""Fapi ccxt position normalization for TruthSync."""

from __future__ import annotations

from order_management.exchange.trend_position_views import (
    normalize_exchange_positions,
    normalize_fapi_ccxt_position_row,
)


def test_fapi_ccxt_row_maps_symbol_to_signal() -> None:
    row = normalize_fapi_ccxt_position_row(
        {
            "symbol": "ETH/USDT:USDT",
            "side": "long",
            "size": 2.0,
            "entry_price": 2100.0,
        }
    )
    assert row is not None
    assert row["symbol"] == "ETHUSDT"
    assert row["exec_symbol"] == "ETH/USDT:USDT"


def test_normalize_exchange_positions_fapi_batch() -> None:
    rows = normalize_exchange_positions(
        [
            {
                "symbol": "ETH/USDT:USDT",
                "side": "long",
                "size": 1.0,
                "entry_price": 2000.0,
            }
        ]
    )
    assert len(rows) == 1
    assert rows[0]["symbol"] == "ETHUSDT"
