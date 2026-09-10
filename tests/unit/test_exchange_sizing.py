"""Tests for exchange sizing and built-in dapi symbol mappings."""

from __future__ import annotations

from order_management.exchange.sizing import (
    contracts_from_notional_usd,
    contracts_from_risk_coin,
    usd_notional_from_qty,
)
from order_management.exchange.symbol_map import (
    contract_multiplier_for,
    load_symbol_map,
    resolve_symbol_mapping,
)
from order_management.exchange.types import MarginKind


def test_builtin_symbol_map_btc() -> None:
    m = resolve_symbol_mapping("BTCUSDT")
    assert m.exec_symbol == "BTCUSD_PERP"
    assert m.contract_multiplier == 100
    assert m.collateral_asset == "BTC"


def test_builtin_symbol_map_sol() -> None:
    assert contract_multiplier_for("SOLUSDT") == 10
    m = resolve_symbol_mapping("SOLUSDT")
    assert m.exec_symbol == "SOLUSD_PERP"


def test_contracts_from_notional_usd() -> None:
    n = contracts_from_notional_usd(
        notional_usd=10_000, price=100_000, contract_multiplier=100
    )
    assert n == 100


def test_contracts_from_risk_coin() -> None:
    n = contracts_from_risk_coin(
        risk_coin=0.01,
        entry_price=100_000,
        stop_pct=0.02,
        contract_multiplier=100,
    )
    assert n >= 1


def test_usd_notional_usd_m_is_qty_times_price() -> None:
    n = usd_notional_from_qty(
        qty=0.02,
        price=50_000.0,
        margin_kind=MarginKind.USD_M,
    )
    assert n == 1_000.0


def test_usd_notional_coin_m_is_contracts_times_multiplier() -> None:
    """Prod 2026-08-21: qty*price inflated BNB ~16x and blocked every SRB open."""
    n = usd_notional_from_qty(
        qty=33.0,
        price=667.0,
        margin_kind=MarginKind.COIN_M,
        contract_multiplier=10.0,
    )
    assert n == 330.0
    wrong_linear = 33.0 * 667.0
    assert n < wrong_linear / 10.0


def test_load_symbol_map_non_empty() -> None:
    table = load_symbol_map()
    assert "BTCUSDT" in table
    assert len(table) >= 3
