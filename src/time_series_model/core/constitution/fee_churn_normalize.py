"""Normalize userTrades PnL / commission to USD for Fee-Churn Guard."""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Mapping, Optional

logger = logging.getLogger(__name__)

_STABLE_USD = frozenset({"USDT", "USD", "BUSD", "USDC"})


def _mark_price_usd(asset: str, mark_prices: Mapping[str, float]) -> Optional[float]:
    a = str(asset or "").upper()
    if not a:
        return None
    if a in _STABLE_USD:
        return 1.0
    px = mark_prices.get(f"{a}USDT")
    if px is None:
        px = mark_prices.get(a)
    try:
        out = float(px) if px is not None else 0.0
    except (TypeError, ValueError):
        return None
    return out if out > 0 else None


def _collateral_asset_for_symbol(symbol: str) -> str:
    sym = str(symbol or "").upper()
    if not sym:
        return "USDT"
    try:
        from order_management.exchange.symbol_map import resolve_symbol_mapping

        return str(resolve_symbol_mapping(sym).collateral_asset or "USDT").upper()
    except KeyError:
        return "USDT"


def amount_to_usd(
    amount: float,
    asset: str,
    mark_prices: Mapping[str, float],
) -> Optional[float]:
    """Convert *amount* in *asset* to approximate USD using *mark_prices*."""
    try:
        native = float(amount)
    except (TypeError, ValueError):
        return None
    if native == 0.0:
        return 0.0
    px = _mark_price_usd(asset, mark_prices)
    if px is None:
        return None
    return native * px


def _resolve_ticker_source(api: Any) -> Optional[Any]:
    """Find an object exposing ``get_ticker_price`` for *api*.

    Several dapi facades in this codebase wrap the REST client that actually
    implements ``get_ticker_price`` (e.g. ``DapiMultiLegApi._api`` is a
    ``BinanceCoinMFuturesAPI``, ``DapiExchangeAdapter.underlying_api`` too),
    without proxying the method themselves. Without this fallback, C-layer
    coin_m (chop_grid dapi / ``MULTILEG_DAPI``) would silently fail to
    resolve any marks and every commission/pnl row would be dropped —
    degrading straight back to the unreliable WS-commission fallback this
    module exists to avoid.
    """
    if callable(getattr(api, "get_ticker_price", None)):
        return api
    for attr in ("_api", "underlying_api", "exchange"):
        inner = getattr(api, attr, None)
        if inner is not None and callable(getattr(inner, "get_ticker_price", None)):
            return inner
    return None


def resolve_mark_prices_for_trades(
    trades: Iterable[Mapping[str, Any]],
    api: Any,
    *,
    margin_kind: str = "usd_m",
) -> Dict[str, float]:
    """Fetch ticker marks needed to normalize dapi (and mixed-asset) trades."""
    marks: Dict[str, float] = {"USDT": 1.0, "USD": 1.0, "BUSD": 1.0, "USDC": 1.0}
    if str(margin_kind or "usd_m").lower() == "usd_m":
        return marks

    needed: set[str] = set()
    for t in trades:
        if not isinstance(t, dict):
            continue
        comm_asset = str(t.get("commissionAsset") or "USDT").upper()
        if comm_asset not in _STABLE_USD:
            needed.add(f"{comm_asset}USDT")
        sym = str(t.get("symbol") or "").upper()
        coll = _collateral_asset_for_symbol(sym)
        if coll not in _STABLE_USD:
            needed.add(f"{coll}USDT")

    if not needed:
        return marks

    ticker_src = _resolve_ticker_source(api)
    if ticker_src is None:
        logger.warning(
            "fee-churn: no get_ticker_price on api=%s (margin_kind=%s) — "
            "coin_m commission/pnl rows needing %s cannot be normalized and "
            "will be dropped from this sync (guard may fall back to WS commission)",
            type(api).__name__,
            margin_kind,
            sorted(needed),
        )
        return marks
    get_px = ticker_src.get_ticker_price

    for pair in sorted(needed):
        try:
            px = float(get_px(pair))
        except (TypeError, ValueError):
            continue
        except Exception:
            logger.debug("fee-churn: ticker fetch failed pair=%s", pair, exc_info=True)
            continue
        if px <= 0:
            continue
        asset = pair.replace("USDT", "")
        marks[asset] = px
        marks[pair] = px
    return marks


def normalize_user_trade_row(
    row: Mapping[str, Any],
    *,
    mark_prices: Mapping[str, float],
    margin_kind: str = "usd_m",
) -> Optional[Dict[str, Any]]:
    """Return fee-churn batch row with ``realized_pnl_usd`` / ``commission`` in USD."""
    oid = str(row.get("orderId") or row.get("order_id") or "").strip()
    if not oid:
        return None
    try:
        comm_native = float(row.get("commission") or 0.0)
    except (TypeError, ValueError):
        comm_native = 0.0
    try:
        rp_native = float(row.get("realizedPnl") or row.get("realized_pnl") or 0.0)
    except (TypeError, ValueError):
        rp_native = 0.0

    comm_asset = str(
        row.get("commissionAsset") or row.get("commission_asset") or "USDT"
    )
    sym = str(row.get("symbol") or "")

    if str(margin_kind or "usd_m").lower() == "usd_m":
        comm_usd = comm_native
        rp_usd = rp_native
    else:
        comm_usd = amount_to_usd(comm_native, comm_asset, mark_prices)
        if comm_usd is None:
            logger.warning(
                "fee-churn: skip trade order=%s — no mark for commission asset %s",
                oid,
                comm_asset,
            )
            return None
        pnl_asset = _collateral_asset_for_symbol(sym)
        rp_usd = amount_to_usd(rp_native, pnl_asset, mark_prices)
        if rp_usd is None:
            logger.warning(
                "fee-churn: skip trade order=%s — no mark for pnl asset %s",
                oid,
                pnl_asset,
            )
            return None

    return {
        "order_id": oid,
        "exchange_order_id": oid,
        "commission": comm_usd,
        "realized_pnl_usd": rp_usd,
        "symbol": sym,
        "ts": row.get("time") or row.get("ts"),
    }


def aggregate_user_trades_for_fee_churn(
    trades: Iterable[Mapping[str, Any]],
    *,
    mark_prices: Mapping[str, float],
    margin_kind: str = "usd_m",
) -> List[Dict[str, Any]]:
    """Sum partial fills per ``orderId`` with USD-normalized pnl/commission."""
    by_order: Dict[str, Dict[str, Any]] = {}
    for raw in trades:
        if not isinstance(raw, dict):
            continue
        norm = normalize_user_trade_row(
            raw, mark_prices=mark_prices, margin_kind=margin_kind
        )
        if norm is None:
            continue
        oid = norm["order_id"]
        row = by_order.get(oid)
        if row is None:
            by_order[oid] = dict(norm)
            continue
        row["commission"] = float(row.get("commission") or 0.0) + float(
            norm["commission"]
        )
        row["realized_pnl_usd"] = float(row.get("realized_pnl_usd") or 0.0) + float(
            norm["realized_pnl_usd"]
        )
        row["ts"] = max(row.get("ts") or 0, norm.get("ts") or 0)
    return list(by_order.values())


def sync_fee_churn_from_user_trades(
    guard: Any,
    *,
    api: Any,
    symbols: Iterable[str],
    margin_kind: str = "usd_m",
    lookback_ms: int = 1_800_000,
    strategy: Optional[str] = None,
) -> int:
    """REST ``userTrades`` → confirm pending / record round-trips. Returns newly blocked count."""
    import time

    if guard is None or not getattr(getattr(guard, "config", None), "enabled", False):
        return 0
    if api is None or not hasattr(api, "get_user_trades"):
        return 0

    now_ms = int(time.time() * 1000)
    start_ms = now_ms - int(lookback_ms)
    all_rows: List[Dict[str, Any]] = []

    for sym in symbols:
        sym_u = str(sym or "").upper().strip()
        if not sym_u:
            continue
        try:
            trades = api.get_user_trades(
                symbol=sym_u,
                start_time_ms=start_ms,
                end_time_ms=now_ms,
                limit=1000,
            )
        except Exception:
            logger.debug(
                "fee-churn sync: userTrades failed symbol=%s", sym_u, exc_info=True
            )
            continue
        if not trades:
            continue
        marks = resolve_mark_prices_for_trades(trades, api, margin_kind=margin_kind)
        all_rows.extend(
            aggregate_user_trades_for_fee_churn(
                trades, mark_prices=marks, margin_kind=margin_kind
            )
        )

    if not all_rows:
        guard.expire_stale_pending(max_wait_minutes=15)
        return 0

    newly = guard.confirm_commission_batch(all_rows, strategy=strategy)
    guard.expire_stale_pending(max_wait_minutes=15)
    return newly
