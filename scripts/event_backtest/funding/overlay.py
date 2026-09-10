"""Apply historical funding payments to COIN-M wallet state during event backtest."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import pandas as pd

from src.research.inverse_contract_pnl import is_long_side


def funding_payment_collateral(
    *,
    contracts: float,
    mark_price: float,
    funding_rate: float,
    side: str,
    contract_multiplier: float,
) -> float:
    """Signed collateral payment (+ receive, - pay) for one inverse position."""
    if contracts <= 0.0 or mark_price <= 0.0:
        return 0.0
    notional_coin = float(contracts) * float(contract_multiplier) / float(mark_price)
    sign = 1.0 if is_long_side(side) else -1.0
    return -sign * notional_coin * float(funding_rate)


class CoinMFundingOverlay:
    """Apply parquet funding events up to the current timeline timestamp."""

    def __init__(self, funding_by_symbol: Mapping[str, pd.Series]):
        self._funding = {
            str(sym).upper(): s.sort_index() for sym, s in funding_by_symbol.items()
        }
        self._cursor: Dict[str, int] = {sym: 0 for sym in self._funding}
        self.total_paid_collateral: float = 0.0

    @property
    def enabled(self) -> bool:
        return bool(self._funding)

    def apply_through(
        self,
        ts: pd.Timestamp,
        simulators: Mapping[str, Any],
        mark_by_sym: Mapping[str, float],
    ) -> None:
        ts = (
            pd.Timestamp(ts).tz_convert("UTC")
            if pd.Timestamp(ts).tzinfo
            else pd.Timestamp(ts, tz="UTC")
        )
        for sym, sim in simulators.items():
            series = self._funding.get(str(sym).upper())
            if series is None or not getattr(sim, "_coin_margin_state", None):
                continue
            state = sim._coin_margin_state
            idx = int(self._cursor.get(str(sym).upper(), 0))
            mark = float(mark_by_sym.get(str(sym), 0.0) or 0.0)
            if mark <= 0.0:
                continue
            mult = float(getattr(sim, "contract_multiplier", state.contract_multiplier))
            while idx < len(series):
                f_ts = pd.Timestamp(series.index[idx])
                if f_ts.tzinfo is None:
                    f_ts = f_ts.tz_localize("UTC")
                else:
                    f_ts = f_ts.tz_convert("UTC")
                if f_ts > ts:
                    break
                rate = float(series.iloc[idx])
                for pos in getattr(sim, "_positions", {}).values():
                    contracts = float(pos.get("_contracts", 0.0) or 0.0)
                    if contracts <= 0.0:
                        entry = float(pos.get("entry_price", 0.0) or 0.0)
                        qty = float(pos.get("_qty_base", 0.0) or 0.0)
                        if entry > 0.0 and qty > 0.0:
                            contracts = qty * entry / mult
                    if contracts <= 0.0:
                        continue
                    pay = funding_payment_collateral(
                        contracts=contracts,
                        mark_price=mark,
                        funding_rate=rate,
                        side=str(pos.get("side") or "LONG"),
                        contract_multiplier=mult,
                    )
                    state.apply_realized(pay)
                    self.total_paid_collateral += pay
                idx += 1
            self._cursor[str(sym).upper()] = idx
