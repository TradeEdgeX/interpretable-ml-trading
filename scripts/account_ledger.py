"""Lightweight reusable account ledger utilities for research backtests.

This module provides deterministic bookkeeping primitives for:
- quote notional -> base quantity conversion
- weighted average entry updates for inventory-style accumulation
- realized USDT PnL calculation (with entry/exit fees)
- optional cash tracking per isolated account

The caller controls account isolation by constructing one ledger instance per
account/backtest stream (e.g. spot_accum / trend / multi-leg).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional


@dataclass
class LedgerLot:
    lot_id: str
    account: str
    strategy: str
    symbol: str
    side: str
    qty_base: float
    vwap_entry: float
    entry_notional_usdt: float
    entry_fee_usdt: float
    opened_at: datetime
    cash_mode: str = "cash_notional"  # cash_notional | fee_only


@dataclass
class CloseResult:
    lot_id: str
    qty_base: float
    entry_price: float
    exit_price: float
    entry_notional_usdt: float
    exit_notional_usdt: float
    entry_fee_usdt: float
    exit_fee_usdt: float
    gross_pnl_usdt: float
    realized_pnl_usdt: float


class AccountLedger:
    def __init__(self, *, account: str, initial_cash_usdt: float = 0.0) -> None:
        self.account = str(account)
        self.initial_cash_usdt = float(initial_cash_usdt or 0.0)
        self.cash_usdt = float(initial_cash_usdt or 0.0)
        self.realized_pnl_usdt = 0.0
        self.total_fees_usdt = 0.0
        self._lots: Dict[str, LedgerLot] = {}

    def deposit(self, *, amount_usdt: float) -> float:
        amt = float(amount_usdt or 0.0)
        if amt <= 0.0:
            return self.cash_usdt
        self.cash_usdt += amt
        return self.cash_usdt

    def get_lot(self, lot_id: str) -> Optional[LedgerLot]:
        return self._lots.get(str(lot_id))

    def open_lot(
        self,
        *,
        lot_id: str,
        strategy: str,
        symbol: str,
        side: str,
        notional_usdt: float,
        entry_price: float,
        fee_rate: float = 0.0,
        opened_at: Optional[datetime] = None,
        cash_mode: str = "cash_notional",
        allow_scale_down: bool = True,
    ) -> tuple[bool, float, str]:
        """Open a new lot and update cash according to cash_mode.

        Returns:
            ok, filled_notional_usdt, reason
        """
        lid = str(lot_id)
        px = float(entry_price or 0.0)
        if px <= 0.0:
            return False, 0.0, "bad_entry_price"
        desired = max(0.0, float(notional_usdt or 0.0))
        if desired <= 0.0:
            return False, 0.0, "bad_notional"
        fr = max(0.0, float(fee_rate or 0.0))
        mode = str(cash_mode or "cash_notional")

        filled = desired
        if mode == "cash_notional":
            cash_need = desired * (1.0 + fr)
            if self.cash_usdt + 1e-12 < cash_need:
                if not allow_scale_down:
                    return False, 0.0, "insufficient_cash"
                denom = max(1e-12, (1.0 + fr))
                filled = max(0.0, self.cash_usdt / denom)
                if filled <= 0.0:
                    return False, 0.0, "insufficient_cash"
        else:
            fee_need = desired * fr
            if self.cash_usdt + 1e-12 < fee_need:
                if not allow_scale_down:
                    return False, 0.0, "insufficient_cash_fee"
                filled = (
                    max(0.0, self.cash_usdt / max(1e-12, fr)) if fr > 0 else desired
                )
                if filled <= 0.0:
                    return False, 0.0, "insufficient_cash_fee"

        entry_fee = filled * fr
        qty = filled / px
        if qty <= 0.0:
            return False, 0.0, "bad_qty"

        if mode == "cash_notional":
            self.cash_usdt -= filled + entry_fee
        else:
            self.cash_usdt -= entry_fee
        self.total_fees_usdt += entry_fee

        self._lots[lid] = LedgerLot(
            lot_id=lid,
            account=self.account,
            strategy=str(strategy or ""),
            symbol=str(symbol or ""),
            side=str(side or "").upper(),
            qty_base=qty,
            vwap_entry=px,
            entry_notional_usdt=filled,
            entry_fee_usdt=entry_fee,
            opened_at=opened_at or datetime.utcnow(),
            cash_mode=mode,
        )
        return True, filled, ""

    def merge_lot(
        self,
        *,
        lot_id: str,
        add_notional_usdt: float,
        add_price: float,
        fee_rate: float = 0.0,
        allow_scale_down: bool = True,
    ) -> tuple[bool, float, str]:
        lot = self._lots.get(str(lot_id))
        if lot is None:
            return False, 0.0, "lot_not_found"
        px = float(add_price or 0.0)
        if px <= 0.0:
            return False, 0.0, "bad_add_price"
        desired = max(0.0, float(add_notional_usdt or 0.0))
        if desired <= 0.0:
            return False, 0.0, "bad_add_notional"
        fr = max(0.0, float(fee_rate or 0.0))
        filled = desired

        if lot.cash_mode == "cash_notional":
            cash_need = desired * (1.0 + fr)
            if self.cash_usdt + 1e-12 < cash_need:
                if not allow_scale_down:
                    return False, 0.0, "insufficient_cash"
                filled = max(0.0, self.cash_usdt / max(1e-12, 1.0 + fr))
                if filled <= 0.0:
                    return False, 0.0, "insufficient_cash"
        else:
            fee_need = desired * fr
            if self.cash_usdt + 1e-12 < fee_need:
                if not allow_scale_down:
                    return False, 0.0, "insufficient_cash_fee"
                filled = (
                    max(0.0, self.cash_usdt / max(1e-12, fr)) if fr > 0 else desired
                )
                if filled <= 0.0:
                    return False, 0.0, "insufficient_cash_fee"

        add_fee = filled * fr
        add_qty = filled / px
        if lot.cash_mode == "cash_notional":
            self.cash_usdt -= filled + add_fee
        else:
            self.cash_usdt -= add_fee
        self.total_fees_usdt += add_fee

        q_old = max(0.0, float(lot.qty_base or 0.0))
        q_new = q_old + add_qty
        if q_new <= 0.0:
            return False, 0.0, "bad_total_qty"
        lot.vwap_entry = (lot.vwap_entry * q_old + px * add_qty) / q_new
        lot.qty_base = q_new
        lot.entry_notional_usdt += filled
        lot.entry_fee_usdt += add_fee
        return True, filled, ""

    def close_lot(
        self,
        *,
        lot_id: str,
        exit_price: float,
        fee_rate: float = 0.0,
    ) -> Optional[CloseResult]:
        lid = str(lot_id)
        lot = self._lots.pop(lid, None)
        if lot is None:
            return None
        px = float(exit_price or 0.0)
        if px <= 0.0:
            return None
        fr = max(0.0, float(fee_rate or 0.0))

        qty = max(0.0, float(lot.qty_base or 0.0))
        if qty <= 0.0:
            return None
        exit_notional = qty * px
        exit_fee = exit_notional * fr
        sign = 1.0 if lot.side in {"LONG", "BUY"} else -1.0
        gross = (px - lot.vwap_entry) * qty * sign
        realized = gross - lot.entry_fee_usdt - exit_fee

        if lot.cash_mode == "cash_notional":
            self.cash_usdt += exit_notional - exit_fee
        else:
            self.cash_usdt += gross - exit_fee
        self.total_fees_usdt += exit_fee
        self.realized_pnl_usdt += realized

        return CloseResult(
            lot_id=lid,
            qty_base=qty,
            entry_price=float(lot.vwap_entry),
            exit_price=px,
            entry_notional_usdt=float(lot.entry_notional_usdt),
            exit_notional_usdt=float(exit_notional),
            entry_fee_usdt=float(lot.entry_fee_usdt),
            exit_fee_usdt=float(exit_fee),
            gross_pnl_usdt=float(gross),
            realized_pnl_usdt=float(realized),
        )
