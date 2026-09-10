"""Historical funding overlay for COIN-M event backtest."""

from scripts.event_backtest.funding.chop_accrual import aggregated_funding_rate_decimal
from scripts.event_backtest.funding.loader import (
    load_funding_rate_series,
    load_funding_rates_for_symbols,
)
from scripts.event_backtest.funding.overlay import (
    CoinMFundingOverlay,
    funding_payment_collateral,
)

__all__ = [
    "CoinMFundingOverlay",
    "aggregated_funding_rate_decimal",
    "funding_payment_collateral",
    "load_funding_rate_series",
    "load_funding_rates_for_symbols",
]
