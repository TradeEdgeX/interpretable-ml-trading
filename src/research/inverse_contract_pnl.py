"""Inverse-perp (coin-margined) PnL arithmetic.

Not wired into the public court. The court sizes and marks USD-M linear
contracts. Call this helper only when you explicitly need reverse-contract math.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Side = Literal["LONG", "SHORT", "BUY", "SELL"]


def is_long_side(side: str) -> bool:
    return str(side or "").upper() in {"LONG", "BUY"}


def coin_m_gross_pnl_collateral(
    *,
    contracts: float,
    entry_price: float,
    exit_price: float,
    side: str,
    contract_multiplier: float = 100.0,
) -> float:
    """Gross PnL in margin collateral on an inverse perpetual."""
    if contracts <= 0 or entry_price <= 0 or exit_price <= 0:
        return 0.0
    m = float(contract_multiplier)
    if is_long_side(side):
        return contracts * m * (1.0 / entry_price - 1.0 / exit_price)
    return contracts * m * (1.0 / exit_price - 1.0 / entry_price)


def coin_m_gross_pnl_usd(
    *,
    contracts: float,
    entry_price: float,
    exit_price: float,
    side: str,
    contract_multiplier: float = 100.0,
) -> float:
    coin = coin_m_gross_pnl_collateral(
        contracts=contracts,
        entry_price=entry_price,
        exit_price=exit_price,
        side=side,
        contract_multiplier=contract_multiplier,
    )
    return coin * float(exit_price)


def coin_m_fee_collateral(
    *,
    contracts: float,
    fill_price: float,
    fee_rate: float,
    contract_multiplier: float = 100.0,
) -> float:
    if contracts <= 0 or fill_price <= 0 or fee_rate <= 0:
        return 0.0
    return (
        float(contracts)
        * float(contract_multiplier)
        * float(fee_rate)
        / float(fill_price)
    )


def coin_m_contracts_from_risk_coin(
    *,
    risk_coin: float,
    entry_price: float,
    stop_pct: float,
    contract_multiplier: float = 100.0,
) -> float:
    if risk_coin <= 0 or entry_price <= 0 or stop_pct <= 1e-9:
        return 0.0
    return (
        float(risk_coin)
        * float(entry_price)
        / (float(contract_multiplier) * float(stop_pct))
    )


@dataclass
class CoinMarginState:
    """Collateral-denominated wallet. Optional helper — not used by the court."""

    wallet_coin: float
    contract_multiplier: float = 100.0

    def equity_usdt(self, mark_price: float) -> float:
        if mark_price <= 0:
            return 0.0
        return float(self.wallet_coin) * float(mark_price)

    def apply_realized(self, pnl_collateral: float) -> None:
        self.wallet_coin = float(self.wallet_coin) + float(pnl_collateral or 0.0)


def linear_gross_pnl_usd(
    *,
    qty_base: float,
    entry_price: float,
    exit_price: float,
    side: str,
) -> float:
    if qty_base <= 0:
        return 0.0
    if is_long_side(side):
        return qty_base * (exit_price - entry_price)
    return qty_base * (entry_price - exit_price)


def grid_gross_pnl_pct(
    *,
    entry: float,
    exit_px: float,
    side: str,
    margin_mode: str = "usd_m",
) -> float:
    """Gross return as a fraction (fee excluded).

    usd_m (linear): (exit-entry)/entry for a long.
    coin_m (inverse): (exit-entry)/exit for a long.
    """
    if entry <= 0 or exit_px <= 0:
        return 0.0
    if str(margin_mode).lower() == "coin_m":
        if is_long_side(side):
            return (exit_px - entry) / exit_px
        return (entry - exit_px) / exit_px
    if is_long_side(side):
        return (exit_px - entry) / entry
    return (entry - exit_px) / entry


def contracts_from_linear_qty(
    qty_base: float,
    entry_price: float,
    contract_multiplier: float = 100.0,
) -> float:
    """Match USD notional: contracts × multiplier ≈ qty × entry."""
    if entry_price <= 0:
        return 0.0
    return qty_base * entry_price / float(contract_multiplier)


def coin_m_pnl_r_from_trade_row(
    row: dict,
    *,
    risk_usd: float,
    contract_multiplier: float = 100.0,
) -> float:
    """Recompute pnl_r as if inverse with the same linear qty / risk budget."""
    qty = float(row.get("qty_base") or 0.0)
    entry = float(row.get("entry_price") or 0.0)
    exit_p = float(row.get("exit_price") or 0.0)
    side = str(row.get("side") or "LONG")
    if qty <= 0 and entry > 0:
        notional = float(row.get("notional_usdt") or 0.0)
        if notional > 0:
            qty = notional / entry
    if qty <= 0 or risk_usd <= 0:
        return float(row.get("pnl_r") or 0.0)
    contracts = contracts_from_linear_qty(qty, entry, contract_multiplier)
    gross = coin_m_gross_pnl_usd(
        contracts=contracts,
        entry_price=entry,
        exit_price=exit_p,
        side=side,
        contract_multiplier=contract_multiplier,
    )
    return gross / risk_usd * float(row.get("size_multiplier", 1.0) or 1.0)
