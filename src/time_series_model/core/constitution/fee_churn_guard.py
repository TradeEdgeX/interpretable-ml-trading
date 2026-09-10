"""Fee-churn guard: detect friction-dominated round-trips and block new entries.

Rule (v1): within ``window_minutes``, if ``bad_trade_count`` or more closing
round-trips are "bad" (net loss dominated by commission), block ``place`` for
that (lane, strategy) scope.

See docs/architecture/account_safety_gate_CN.md §10.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from src.live_data_stream.constitution_config import load_constitution_dict

logger = logging.getLogger(__name__)

_CLOSING_PROTECTION_TYPES = frozenset({"stop_loss", "take_profit", "market_exit"})


def is_bad_trade(
    *,
    realized_pnl_usd: float,
    commission_usd: float,
    loss_dominance_k: float = 2.0,
) -> bool:
    """Return True when friction dominates (not a normal directional stop).

    Cases:
    - Gross profit eaten by fees: pnl >= 0 and commission > pnl
    - Small spread/slip loss with fees dominating: net < 0 and |pnl| <= k * fee
    """
    commission = max(0.0, float(commission_usd))
    if commission <= 0.0:
        return False
    pnl = float(realized_pnl_usd)
    net = pnl - commission
    if pnl >= 0.0:
        return commission > pnl
    if net >= 0.0:
        return False
    return abs(pnl) <= float(loss_dominance_k) * commission


def is_closing_execution_report(report: Mapping[str, Any]) -> bool:
    """True when the fill likely closes/reduces exposure (round-trip leg)."""
    if bool(report.get("reduce_only") or report.get("close_position")):
        return True
    prot = str(report.get("protection_type") or "").strip().lower()
    if prot in _CLOSING_PROTECTION_TYPES:
        return True
    try:
        rp = float(report.get("realized_pnl") or 0.0)
    except (TypeError, ValueError):
        rp = 0.0
    return abs(rp) > 0.0


@dataclass(frozen=True)
class FeeChurnGuardConfig:
    enabled: bool = True
    bad_trade_count: int = 3
    window_minutes: int = 60
    loss_dominance_k: float = 2.0
    scope: str = "per_strategy"  # per_strategy | per_lane
    recovery: str = "manual"  # manual | cooldown
    cooldown_minutes: int = 240

    @classmethod
    def from_constitution_dict(cls, raw: Mapping[str, Any]) -> FeeChurnGuardConfig:
        section = dict(raw.get("fee_churn_guard") or {})
        if not section:
            return cls(enabled=False)
        return cls(
            enabled=bool(section.get("enabled", True)),
            bad_trade_count=max(1, int(section.get("bad_trade_count", 3) or 3)),
            window_minutes=max(1, int(section.get("window_minutes", 60) or 60)),
            loss_dominance_k=float(section.get("loss_dominance_k", 2.0) or 2.0),
            scope=str(section.get("scope", "per_strategy") or "per_strategy"),
            recovery=str(section.get("recovery", "manual") or "manual"),
            cooldown_minutes=max(0, int(section.get("cooldown_minutes", 240) or 240)),
        )

    @classmethod
    def from_constitution_yaml(cls, path: str | Path) -> FeeChurnGuardConfig:
        return cls.from_constitution_dict(load_constitution_dict(path))


@dataclass
class BadTradeRecord:
    ts: datetime
    realized_pnl_usd: float
    commission_usd: float
    symbol: str = ""
    order_id: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "ts": self.ts.astimezone(timezone.utc).replace(microsecond=0).isoformat(),
            "realized_pnl_usd": float(self.realized_pnl_usd),
            "commission_usd": float(self.commission_usd),
            "symbol": str(self.symbol or ""),
            "order_id": str(self.order_id or ""),
        }

    @staticmethod
    def from_dict(obj: Mapping[str, Any]) -> BadTradeRecord:
        ts_raw = obj.get("ts")
        ts = datetime.now(timezone.utc)
        if ts_raw:
            try:
                ts = datetime.fromisoformat(str(ts_raw).replace("Z", "+00:00"))
            except ValueError:
                pass
        return BadTradeRecord(
            ts=ts,
            realized_pnl_usd=float(obj.get("realized_pnl_usd") or 0.0),
            commission_usd=float(obj.get("commission_usd") or 0.0),
            symbol=str(obj.get("symbol") or ""),
            order_id=str(obj.get("order_id") or ""),
        )


@dataclass
class PendingConfirmRecord:
    """A closing round-trip whose PnL is WS-confirmed but commission is not.

    Binance user-stream execution reports do not reliably carry ``commission``
    for multi-leg (see ``multi_leg_orchestrator._sync_commissions_from_binance``
    docstring and ``binance_api.get_user_trades`` docstring). ``realized_pnl``
    *is* reliable on the WS path. So we stage the pnl here and defer the
    bad-trade verdict until a REST-confirmed commission arrives via
    :meth:`FeeChurnGuard.confirm_commission`.
    """

    ts: datetime
    order_id: str
    symbol: str
    realized_pnl_usd: float
    fallback_commission_usd: float

    def as_dict(self) -> Dict[str, Any]:
        return {
            "ts": self.ts.astimezone(timezone.utc).replace(microsecond=0).isoformat(),
            "order_id": str(self.order_id or ""),
            "symbol": str(self.symbol or ""),
            "realized_pnl_usd": float(self.realized_pnl_usd),
            "fallback_commission_usd": float(self.fallback_commission_usd),
        }

    @staticmethod
    def from_dict(obj: Mapping[str, Any]) -> PendingConfirmRecord:
        ts_raw = obj.get("ts")
        ts = datetime.now(timezone.utc)
        if ts_raw:
            try:
                ts = datetime.fromisoformat(str(ts_raw).replace("Z", "+00:00"))
            except ValueError:
                pass
        return PendingConfirmRecord(
            ts=ts,
            order_id=str(obj.get("order_id") or ""),
            symbol=str(obj.get("symbol") or ""),
            realized_pnl_usd=float(obj.get("realized_pnl_usd") or 0.0),
            fallback_commission_usd=float(obj.get("fallback_commission_usd") or 0.0),
        )


@dataclass
class _PendingOrderAgg:
    strategy: str
    symbol: str
    order_id: str
    realized_pnl_usd: float = 0.0
    commission_usd: float = 0.0
    first_ts: Optional[datetime] = None
    last_ts: Optional[datetime] = None


@dataclass
class FeeChurnStrategyState:
    lane: str
    strategy: str
    bad_trades: List[BadTradeRecord] = field(default_factory=list)
    blocked: bool = False
    blocked_since: Optional[datetime] = None
    finalized_order_ids: List[str] = field(default_factory=list)
    pending_confirm: List[PendingConfirmRecord] = field(default_factory=list)

    def key(self) -> str:
        return f"{self.lane}:{self.strategy}"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "lane": self.lane,
            "strategy": self.strategy,
            "bad_trades": [b.as_dict() for b in self.bad_trades],
            "blocked": bool(self.blocked),
            "blocked_since": (
                self.blocked_since.astimezone(timezone.utc)
                .replace(microsecond=0)
                .isoformat()
                if self.blocked_since
                else None
            ),
            "finalized_order_ids": list(self.finalized_order_ids[-500:]),
            "pending_confirm": [p.as_dict() for p in self.pending_confirm],
        }

    @staticmethod
    def from_dict(obj: Mapping[str, Any]) -> FeeChurnStrategyState:
        blocked_since = None
        raw_bs = obj.get("blocked_since")
        if raw_bs:
            try:
                blocked_since = datetime.fromisoformat(
                    str(raw_bs).replace("Z", "+00:00")
                )
            except ValueError:
                blocked_since = None
        return FeeChurnStrategyState(
            lane=str(obj.get("lane") or ""),
            strategy=str(obj.get("strategy") or ""),
            bad_trades=[
                BadTradeRecord.from_dict(b) for b in (obj.get("bad_trades") or [])
            ],
            blocked=bool(obj.get("blocked", False)),
            blocked_since=blocked_since,
            finalized_order_ids=[
                str(x) for x in (obj.get("finalized_order_ids") or [])
            ],
            pending_confirm=[
                PendingConfirmRecord.from_dict(p)
                for p in (obj.get("pending_confirm") or [])
            ],
        )


class FeeChurnGuard:
    """Window-based bad-trade counter with round-trip order aggregation."""

    def __init__(
        self,
        *,
        lane: str,
        config: FeeChurnGuardConfig,
        state_path: Optional[Path] = None,
        on_block: Optional[Any] = None,
    ) -> None:
        self.lane = str(lane or "unknown")
        self.config = config
        self.state_path = Path(state_path) if state_path else None
        self.on_block = on_block
        self._strategies: Dict[str, FeeChurnStrategyState] = {}
        self._pending: Dict[str, _PendingOrderAgg] = {}
        self._process_start = datetime.now(timezone.utc)

    def load(self) -> None:
        if self.state_path is None or not self.state_path.exists():
            return
        try:
            raw = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.warning(
                "fee-churn: failed to load %s", self.state_path, exc_info=True
            )
            return
        for item in raw.get("strategies") or []:
            st = FeeChurnStrategyState.from_dict(item)
            if st.strategy:
                self._strategies[st.strategy] = st

    def save(self) -> None:
        if self.state_path is None:
            return
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "strategies": [st.as_dict() for st in self._strategies.values()],
        }
        self.state_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _state_for(self, strategy: str) -> FeeChurnStrategyState:
        key = str(strategy or "").strip() or "_lane"
        st = self._strategies.get(key)
        if st is None:
            st = FeeChurnStrategyState(lane=self.lane, strategy=key)
            self._strategies[key] = st
        return st

    def clear_blocked(self, strategy: str) -> None:
        st = self._state_for(strategy)
        st.blocked = False
        st.blocked_since = None
        st.bad_trades.clear()
        self.save()

    def blocks_strategy(self, strategy: str, *, now: Optional[datetime] = None) -> bool:
        if not self.config.enabled:
            return False
        now = now or datetime.now(timezone.utc)
        if self.config.scope == "per_lane":
            return any(
                self._is_blocked(st, now=now) for st in self._strategies.values()
            )
        st = self._state_for(strategy)
        return self._is_blocked(st, now=now)

    def block_reason(self, strategy: str) -> Optional[str]:
        if not self.blocks_strategy(strategy):
            return None
        return "fee_churn_streak"

    def recent_bad_trades(
        self, strategy: str, *, now: Optional[datetime] = None
    ) -> List[BadTradeRecord]:
        now = now or datetime.now(timezone.utc)
        st = self._state_for(strategy)
        window = timedelta(minutes=int(self.config.window_minutes))
        return [b for b in st.bad_trades if now - b.ts <= window]

    def record_execution_report(
        self,
        *,
        strategy: str,
        report: Mapping[str, Any],
        now: Optional[datetime] = None,
    ) -> bool:
        """Ingest a user-stream execution report for a closing fill.

        WS-reported ``commission`` is not reliable (Binance user-stream
        execution reports do not consistently carry it — see
        ``multi_leg_orchestrator._sync_commissions_from_binance`` and
        ``BinanceAPI.get_user_trades`` docstrings). ``realized_pnl`` *is*
        reliable on the WS path, so this only stages the pnl via
        :meth:`stage_pending_close`; the bad-trade verdict is deferred until
        :meth:`confirm_commission` / :meth:`confirm_commission_batch` supplies
        a REST-confirmed commission (or :meth:`expire_stale_pending` falls
        back to the WS value after a timeout). Always returns ``False``
        (kept for interface/back-compat; nothing is decided synchronously).
        """
        if not self.config.enabled:
            return False
        if not is_closing_execution_report(report):
            return False

        now = now or datetime.now(timezone.utc)
        order_id = str(report.get("order_id") or report.get("orderId") or "")
        if not order_id:
            return False

        st = self._state_for(strategy)
        if order_id in st.finalized_order_ids:
            return False

        status = str(report.get("status") or "").upper()
        try:
            rp = float(report.get("realized_pnl") or 0.0)
        except (TypeError, ValueError):
            rp = 0.0
        try:
            comm = float(report.get("commission") or 0.0)
        except (TypeError, ValueError):
            comm = 0.0

        pend_key = f"{strategy}:{order_id}"
        agg = self._pending.get(pend_key)
        if agg is None:
            agg = _PendingOrderAgg(
                strategy=strategy,
                symbol=str(report.get("symbol") or ""),
                order_id=order_id,
                first_ts=now,
            )
            self._pending[pend_key] = agg
        agg.realized_pnl_usd += rp
        agg.commission_usd += comm
        agg.last_ts = now
        if agg.first_ts is None:
            agg.first_ts = now

        if status not in {
            "FILLED",
            "PARTIALLY_FILLED",
            "CANCELED",
            "EXPIRED",
            "REJECTED",
        }:
            return False
        if status == "PARTIALLY_FILLED":
            return False
        if status in {"CANCELED", "EXPIRED", "REJECTED"}:
            # Partial-fill-then-cancel is a real (partial) round-trip close;
            # only skip if nothing was ever actually filled for this order.
            if agg.realized_pnl_usd == 0.0 and agg.commission_usd == 0.0:
                self._pending.pop(pend_key, None)
                return False

        self._pending.pop(pend_key, None)
        st.finalized_order_ids.append(order_id)
        st.finalized_order_ids = st.finalized_order_ids[-500:]

        self.stage_pending_close(
            strategy=strategy,
            realized_pnl_usd=agg.realized_pnl_usd,
            symbol=agg.symbol,
            order_id=order_id,
            fallback_commission_usd=agg.commission_usd,
            ts=agg.last_ts or now,
        )
        return False

    def stage_pending_close(
        self,
        *,
        strategy: str,
        realized_pnl_usd: float,
        symbol: str = "",
        order_id: str = "",
        fallback_commission_usd: float = 0.0,
        ts: Optional[datetime] = None,
    ) -> None:
        """Record a closing round-trip's WS-confirmed pnl, deferring the
        commission-dependent bad-trade verdict to a later REST confirmation.
        """
        if not self.config.enabled:
            return
        ts = ts or datetime.now(timezone.utc)
        st = self._state_for(strategy)
        st.pending_confirm.append(
            PendingConfirmRecord(
                ts=ts,
                order_id=str(order_id or ""),
                symbol=str(symbol or ""),
                realized_pnl_usd=float(realized_pnl_usd),
                fallback_commission_usd=float(fallback_commission_usd),
            )
        )
        self.save()

    def confirm_commission(
        self,
        *,
        order_id: str,
        commission_usd: float,
        strategy: Optional[str] = None,
    ) -> Optional[bool]:
        """Resolve a pending (WS-staged) close with REST-confirmed commission.

        If ``strategy`` is omitted, searches all strategies in this lane for
        a matching pending ``order_id`` (useful when the REST sync source
        doesn't carry strategy attribution). Returns ``None`` when no pending
        entry matches ``order_id`` (caller may fall back to
        :meth:`record_round_trip_close` if it has an independent pnl source,
        e.g. ``/fapi/v1/userTrades`` also carries ``realizedPnl``), else True
        if this call newly blocks the strategy.
        """
        oid = str(order_id or "").strip()
        if not oid:
            return None
        candidates = (
            [self._state_for(strategy)] if strategy else list(self._strategies.values())
        )
        for st in candidates:
            for i, rec in enumerate(st.pending_confirm):
                if rec.order_id == oid:
                    st.pending_confirm.pop(i)
                    return self._record_round_trip_close(
                        strategy=st.strategy,
                        realized_pnl_usd=rec.realized_pnl_usd,
                        commission_usd=float(commission_usd),
                        symbol=rec.symbol,
                        order_id=rec.order_id,
                        ts=rec.ts,
                    )
        return None

    def confirm_commission_batch(
        self,
        rows: Iterable[Mapping[str, Any]],
        *,
        strategy: Optional[str] = None,
    ) -> int:
        """Confirm ``{order_id|exchange_order_id, commission, ...}`` rows from
        a periodic REST userTrades sync.

        Two-tier resolution per row:
        1. If a WS-staged pending pnl exists for this ``order_id``, resolve it
           (fast path — most fills arrive via WS first).
        2. Otherwise, if the row itself carries ``realized_pnl_usd`` /
           ``realized_pnl`` (``/fapi/v1/userTrades`` rows do), record the
           round-trip directly. This also covers fills the WS user-stream
           missed entirely (e.g. routed only through REST backfill), since
           ``record_round_trip_close`` dedupes on the same
           ``finalized_order_ids`` registry.

        Returns the number of confirmations that newly blocked a strategy.
        """
        newly_blocked = 0
        for row in rows:
            oid = str(row.get("order_id") or row.get("exchange_order_id") or "").strip()
            if not oid:
                continue
            try:
                comm = float(row.get("commission") or 0.0)
            except (TypeError, ValueError):
                continue
            resolved = self.confirm_commission(
                order_id=oid, commission_usd=comm, strategy=strategy
            )
            if resolved is not None:
                if resolved:
                    newly_blocked += 1
                continue

            rp_raw = row.get("realized_pnl_usd", row.get("realized_pnl"))
            if rp_raw is None:
                continue
            try:
                rp = float(rp_raw)
            except (TypeError, ValueError):
                continue
            if rp == 0.0:
                # ``/fapi/v1/userTrades`` returns *every* fill, including
                # opening/adding legs. An opening fill has ``realizedPnl == 0``
                # (nothing was closed) yet still carries a taker/maker
                # ``commission`` > 0. Recording it here would mislabel grid
                # ENTRY fills as fee-dominated round-trip closes and trip the
                # guard on legitimate entries (see is_closing_execution_report,
                # which requires realizedPnl != 0 absent reduce_only/protection).
                # Only fills that actually realized PnL (closed/reduced a
                # position) can be a fee-churn round-trip. WS-staged closes are
                # resolved by tier 1 (confirm_commission) above and unaffected.
                continue
            if not strategy:
                # WS never staged this order (missed entirely) *and* the
                # caller didn't know which strategy placed it — bucketed
                # under "_lane". Under the default scope="per_strategy" this
                # bucket is invisible to blocks_strategy(<real strategy>),
                # so a fee-churn event hiding in WS-missed fills for a
                # specific strategy could go undetected. See
                # docs/architecture/account_safety_gate_CN.md §10.6.
                logger.warning(
                    "fee-churn: REST-only fill order=%s has no pending match "
                    "and no strategy attribution — recording under '_lane' "
                    "bucket (scope=%s); if a specific strategy is churning "
                    "fees via WS-missed fills this may under-detect",
                    oid,
                    self.config.scope,
                )
            ts_raw = row.get("ts")
            ts: Optional[datetime] = None
            if ts_raw:
                try:
                    ts_ms = float(ts_raw)
                    ts = datetime.fromtimestamp(
                        ts_ms / 1000.0 if ts_ms > 10**12 else ts_ms,
                        tz=timezone.utc,
                    )
                except (TypeError, ValueError, OverflowError, OSError):
                    ts = None
            if self.record_round_trip_close(
                strategy=strategy or "_lane",
                realized_pnl_usd=rp,
                commission_usd=comm,
                symbol=str(row.get("symbol") or ""),
                order_id=oid,
                ts=ts,
            ):
                newly_blocked += 1
        return newly_blocked

    def expire_stale_pending(
        self, *, now: Optional[datetime] = None, max_wait_minutes: int = 15
    ) -> int:
        """Finalize pending closes that never received a REST confirmation,
        falling back to the (less reliable) WS-reported commission so the
        guard degrades gracefully instead of silently losing detection.
        Returns the number of closes finalized this way.
        """
        now = now or datetime.now(timezone.utc)
        max_wait = timedelta(minutes=max(1, int(max_wait_minutes)))
        finalized = 0
        for st in self._strategies.values():
            stale = [r for r in st.pending_confirm if now - r.ts > max_wait]
            if not stale:
                continue
            st.pending_confirm = [
                r for r in st.pending_confirm if now - r.ts <= max_wait
            ]
            for rec in stale:
                logger.warning(
                    "fee-churn: no REST commission confirmation within %dm "
                    "lane=%s strategy=%s order=%s — falling back to WS "
                    "commission=%.4f",
                    max_wait_minutes,
                    self.lane,
                    st.strategy,
                    rec.order_id,
                    rec.fallback_commission_usd,
                )
                if self._record_round_trip_close(
                    strategy=st.strategy,
                    realized_pnl_usd=rec.realized_pnl_usd,
                    commission_usd=rec.fallback_commission_usd,
                    symbol=rec.symbol,
                    order_id=rec.order_id,
                    ts=rec.ts,
                ):
                    finalized += 1
        return finalized

    def record_round_trip_close(
        self,
        *,
        strategy: str,
        realized_pnl_usd: float,
        commission_usd: float,
        symbol: str = "",
        order_id: str = "",
        ts: Optional[datetime] = None,
    ) -> bool:
        """Record an aggregated closing round-trip (B-layer or tests)."""
        if not self.config.enabled:
            return False
        ts = ts or datetime.now(timezone.utc)
        if order_id:
            st = self._state_for(strategy)
            if order_id in st.finalized_order_ids:
                return False
            st.finalized_order_ids.append(order_id)
            st.finalized_order_ids = st.finalized_order_ids[-500:]
        return self._record_round_trip_close(
            strategy=strategy,
            realized_pnl_usd=realized_pnl_usd,
            commission_usd=commission_usd,
            symbol=symbol,
            order_id=order_id,
            ts=ts,
        )

    def _record_round_trip_close(
        self,
        *,
        strategy: str,
        realized_pnl_usd: float,
        commission_usd: float,
        symbol: str,
        order_id: str,
        ts: datetime,
    ) -> bool:
        st = self._state_for(strategy)
        was_blocked = bool(st.blocked)

        if not is_bad_trade(
            realized_pnl_usd=realized_pnl_usd,
            commission_usd=commission_usd,
            loss_dominance_k=self.config.loss_dominance_k,
        ):
            self._prune_window(st, now=ts)
            self.save()
            return False

        st.bad_trades.append(
            BadTradeRecord(
                ts=ts,
                realized_pnl_usd=float(realized_pnl_usd),
                commission_usd=float(commission_usd),
                symbol=str(symbol or ""),
                order_id=str(order_id or ""),
            )
        )
        self._prune_window(st, now=ts)

        if len(st.bad_trades) >= int(self.config.bad_trade_count):
            if not st.blocked:
                st.blocked = True
                st.blocked_since = ts
                logger.warning(
                    "fee-churn BLOCK lane=%s strategy=%s bad_trades=%d window=%dm",
                    self.lane,
                    strategy,
                    len(st.bad_trades),
                    self.config.window_minutes,
                )
                if self.on_block is not None:
                    try:
                        self.on_block(
                            lane=self.lane,
                            strategy=strategy,
                            bad_trades=list(st.bad_trades),
                        )
                    except Exception:
                        logger.warning(
                            "fee-churn on_block callback failed", exc_info=True
                        )
        self.save()
        return bool(st.blocked and not was_blocked)

    def _prune_window(self, st: FeeChurnStrategyState, *, now: datetime) -> None:
        window = timedelta(minutes=int(self.config.window_minutes))
        st.bad_trades = [b for b in st.bad_trades if now - b.ts <= window]

    def _is_blocked(self, st: FeeChurnStrategyState, *, now: datetime) -> bool:
        if not st.blocked:
            return False
        if (
            self.config.recovery == "cooldown"
            and st.blocked_since is not None
            and self.config.cooldown_minutes > 0
        ):
            until = st.blocked_since + timedelta(
                minutes=int(self.config.cooldown_minutes)
            )
            if now >= until:
                st.blocked = False
                st.blocked_since = None
                st.bad_trades.clear()
                self.save()
                return False
        return True
