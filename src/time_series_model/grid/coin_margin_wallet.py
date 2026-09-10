"""Shared COIN-M wallet state — backtest + future chop native ledger."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CoinMarginState:
    """Collateral-denominated account (one pool per margin asset / symbol run)."""

    wallet_coin: float
    contract_multiplier: float = 100.0

    def equity_usdt(self, mark_price: float) -> float:
        if mark_price <= 0:
            return 0.0
        return float(self.wallet_coin) * float(mark_price)

    def apply_realized(self, pnl_collateral: float) -> None:
        self.wallet_coin = float(self.wallet_coin) + float(pnl_collateral or 0.0)
