"""Aggregate resting depth into price-bin walls (T5α)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np

PriceQty = Tuple[float, float]


@dataclass(frozen=True)
class WallSnapshot:
    mid: float
    spread_bps: float
    bucket_pct: float
    wall_bid_notional_usd_max: float
    wall_ask_notional_usd_max: float
    wall_bid_price: float
    wall_ask_price: float
    best_bid: float
    best_ask: float


def _parse_levels(levels: Sequence[Sequence[str | float]]) -> List[PriceQty]:
    out: List[PriceQty] = []
    for row in levels:
        if not row or len(row) < 2:
            continue
        price = float(row[0])
        qty = float(row[1])
        if price > 0 and qty > 0:
            out.append((price, qty))
    return out


def _bin_side_notional(
    levels: Iterable[PriceQty],
    *,
    mid: float,
    bin_width: float,
    side: str,
) -> Tuple[float, float]:
    """Return (max_bin_notional_usd, price_at_max_bin)."""
    bins: dict[int, tuple[float, float]] = {}
    for price, qty in levels:
        if bin_width <= 0:
            continue
        idx = int(np.floor(price / bin_width))
        notional = price * qty
        prev = bins.get(idx, (0.0, 0.0))
        total = prev[0] + notional
        # representative price = volume-weighted average in bin
        if total > 0:
            rep_price = (prev[0] * prev[1] + notional * price) / total
        else:
            rep_price = price
        bins[idx] = (total, rep_price)

    if not bins:
        return 0.0, mid

    best_idx = max(bins, key=lambda k: bins[k][0])
    max_notional, rep_price = bins[best_idx]
    if side == "bid":
        rep_price = min(rep_price, mid)
    else:
        rep_price = max(rep_price, mid)
    return float(max_notional), float(rep_price)


def aggregate_walls_from_depth(
    bids: Sequence[Sequence[str | float]],
    asks: Sequence[Sequence[str | float]],
    *,
    bucket_pct: float = 0.005,
) -> WallSnapshot:
    bid_levels = _parse_levels(bids)
    ask_levels = _parse_levels(asks)
    if not bid_levels or not ask_levels:
        raise ValueError("empty bid or ask book")

    best_bid = max(p for p, _ in bid_levels)
    best_ask = min(p for p, _ in ask_levels)
    if best_ask <= best_bid:
        raise ValueError(f"crossed book: bid={best_bid} ask={best_ask}")

    mid = (best_bid + best_ask) / 2.0
    spread_bps = (best_ask - best_bid) / mid * 10_000.0
    bin_width = max(mid * float(bucket_pct), 1e-12)

    bid_notional, bid_price = _bin_side_notional(
        bid_levels, mid=mid, bin_width=bin_width, side="bid"
    )
    ask_notional, ask_price = _bin_side_notional(
        ask_levels, mid=mid, bin_width=bin_width, side="ask"
    )

    return WallSnapshot(
        mid=float(mid),
        spread_bps=float(spread_bps),
        bucket_pct=float(bucket_pct),
        wall_bid_notional_usd_max=bid_notional,
        wall_ask_notional_usd_max=ask_notional,
        wall_bid_price=bid_price,
        wall_ask_price=ask_price,
        best_bid=float(best_bid),
        best_ask=float(best_ask),
    )


# Tier mins (USD) per §15.2 — informational defaults for docs/tests.
WALL_USD_MIN_BY_SYMBOL_PREFIX: dict[str, float] = {
    "BTC": 50_000_000.0,
    "ETH": 15_000_000.0,
    "BNB": 5_000_000.0,
    "SOL": 5_000_000.0,
    "XRP": 2_000_000.0,
    "HYPE": 2_000_000.0,
}


def wall_usd_min_for_symbol(symbol: str) -> float:
    sym = str(symbol).strip().upper().replace("USDT", "")
    return WALL_USD_MIN_BY_SYMBOL_PREFIX.get(sym, 2_000_000.0)
