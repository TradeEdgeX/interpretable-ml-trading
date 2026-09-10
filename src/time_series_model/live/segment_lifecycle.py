"""Shared segment lifecycle helpers for multi-leg live engines."""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any, Iterable, Mapping, Protocol

import pandas as pd

logger = logging.getLogger(__name__)

# 默认决策 bar 节拍（分钟）。chop_grid / dual_add_trend prod 均为 120T (2h)。
DEFAULT_SIGNAL_BAR_MINUTES = 120


class SegmentState(str, Enum):
    IDLE = "idle"
    ENTERING = "entering"
    ACTIVE = "active"
    CLOSING = "closing"
    # Reserved for audit trails; transitions today go CLOSING → IDLE via _deactivate().
    CLOSED = "closed"


def segment_occupies_slot(segment_state: str) -> bool:
    return segment_state in {
        SegmentState.ENTERING.value,
        SegmentState.ACTIVE.value,
        SegmentState.CLOSING.value,
    }


def segment_allows_new_entry(segment_state: str) -> bool:
    return segment_state == SegmentState.IDLE.value


def entry_signal_timestamp(features: Mapping[str, Any], bar_timestamp: str) -> str:
    """Stable 2h (or primary signal) timestamp for entry/reseed decisions."""
    raw = features.get("_signal_timestamp")
    if raw is not None and str(raw).strip():
        return str(raw)
    logger.debug(
        "entry_signal_timestamp: _signal_timestamp missing in features, "
        "falling back to bar_timestamp=%s — dedup will be per-execution-bar",
        bar_timestamp,
    )
    return str(bar_timestamp)


def _signal_bar_minutes(features: Mapping[str, Any]) -> int:
    """Decision bar 分钟数：优先用特征里的 timeframe，缺省回落到 2h。"""
    tf = features.get("_signal_timeframe") or features.get("_feature_timeframe")
    if tf:
        try:
            from src.live_data_stream.feature_publisher_stack import (
                timeframe_to_bar_minutes,
            )

            return timeframe_to_bar_minutes(str(tf))
        except Exception:
            pass
    return DEFAULT_SIGNAL_BAR_MINUTES


def _floor_to_bucket(ts_raw: str, bar_minutes: int) -> str:
    """把时间戳字符串 floor 到 bar 边界，作为 canonical bucket id。无法解析时原样返回。"""
    if not str(ts_raw).strip():
        return ""
    try:
        return pd.Timestamp(ts_raw).floor(f"{int(bar_minutes)}min").isoformat()
    except Exception:
        return str(ts_raw)


def signal_decision_bucket(features: Mapping[str, Any], bar_timestamp: str = "") -> str:
    """Enter/regime 决策用的 canonical bucket id — floor signal ts 到 bar 收盘。

    2h_close 模式下 publisher 已把 ``_signal_timestamp`` 对齐到 2h 边界；此处再
    floor 一次是 defense in depth：即便偶发 intra-bar 快照漏进来，也只产生同一个
    bucket，保证「一个 2h bar 一次入场」与回测一致。
    """
    return _floor_to_bucket(
        entry_signal_timestamp(features, bar_timestamp),
        _signal_bar_minutes(features),
    )


def entry_decision_allowed_for_signal(
    *,
    last_entry_signal_ts: str,
    features: Mapping[str, Any],
    bar_timestamp: str,
) -> bool:
    """One entry/reseed attempt per signal bar (ignore intra-2h feature churn)."""
    current_bucket = signal_decision_bucket(features, bar_timestamp)
    last_bucket = _floor_to_bucket(
        str(last_entry_signal_ts or ""), _signal_bar_minutes(features)
    )
    return current_bucket != last_bucket


def migrate_segment_state_from_legacy(
    *,
    active: bool,
    segment_state_raw: str | None,
    has_inventory_or_pending: bool = False,
) -> str:
    if segment_state_raw:
        return str(segment_state_raw)
    if not active:
        return SegmentState.IDLE.value
    # Legacy active=True: only preserve as ACTIVE if there's real inventory or
    # pending orders.  Ghost segments (stale active with empty local state) go
    # to IDLE so the concurrency gate can recycle the slot.
    if not has_inventory_or_pending:
        return SegmentState.IDLE.value
    return SegmentState.ACTIVE.value


class _SegmentLifecycleState(Protocol):
    symbol: str
    active: bool
    segment_state: str
    pending_orders: list[Any]
    inventory: list[Any]
    current_regime: str
    last_entry_signal_ts: str


class SegmentLifecycleMixin:
    """Common deactivate / ghost / slot logic for chop_grid and trend_scalp."""

    _engine_name: str = ""
    _exchange_open_orders: bool = False

    state: _SegmentLifecycleState

    def save_state(self) -> None:  # noqa: E704
        ...

    def _entry_decision_allowed(
        self, features: Mapping[str, Any], bar_timestamp: str
    ) -> bool:
        return entry_decision_allowed_for_signal(
            last_entry_signal_ts=str(
                getattr(self.state, "last_entry_signal_ts", "") or ""
            ),
            features=features,
            bar_timestamp=bar_timestamp,
        )

    def _mark_entry_signal_used(
        self, features: Mapping[str, Any], bar_timestamp: str
    ) -> None:
        self.state.last_entry_signal_ts = entry_signal_timestamp(
            features, bar_timestamp
        )

    def _log_stale_active_reset(self) -> None:
        raise NotImplementedError

    def _reconcile_legacy_active_flag(self) -> None:
        if not (
            self.state.active and self.state.segment_state == SegmentState.IDLE.value
        ):
            return
        # Legacy state: active=True but segment_state was never persisted.
        # Only preserve as ACTIVE when real inventory/pending exists; otherwise
        # clear the stale active flag so the slot can be recycled.
        if self.state.inventory or self.state.pending_orders:
            self.state.segment_state = SegmentState.ACTIVE.value
        else:
            self.state.active = False

    def _sync_active_from_segment_state(self) -> None:
        self.state.active = segment_occupies_slot(self.state.segment_state)

    def _segment_winding_down(self) -> bool:
        """True when segment is winding down (inactive or closing)."""
        return (
            not self.state.active
            or self.state.segment_state == SegmentState.CLOSING.value
        )

    def _needs_late_fill_cleanup(self) -> bool:
        """True when segment is winding down (trend: skip promote/protection on late fills)."""
        return (
            not self.state.active
            or self.state.segment_state == SegmentState.CLOSING.value
        )

    def _deactivate(self, reason: str) -> None:
        logger.info(
            "%s deactivate: symbol=%s reason=%s",
            self._engine_name,
            self.state.symbol,
            reason,
        )
        self.state.segment_state = SegmentState.IDLE.value
        self.state.active = False
        if hasattr(self.state, "current_regime"):
            self.state.current_regime = "idle"
        self.save_state()
        gate = getattr(self, "_concurrency_gate", None)
        if gate is not None:
            gate.notify_deactivation(self.state.symbol, self._engine_name)
        try:
            from src.time_series_model.live.metrics_exporter import METRICS

            METRICS.record_strategy_event(
                scope="hedge",
                strategy=self._engine_name,
                symbol=self.state.symbol,
                event=f"segment_{reason}",
                side="na",
            )
        except Exception:
            logger.debug(
                "%s segment event metrics skipped: symbol=%s reason=%s",
                self._engine_name,
                self.state.symbol,
                reason,
                exc_info=True,
            )

    def _enter_segment(self) -> None:
        self.state.segment_state = SegmentState.ENTERING.value
        self.state.active = True

    def _promote_to_active(self) -> None:
        if (
            self.state.segment_state == SegmentState.ENTERING.value
            and self.state.inventory
        ):
            self.state.segment_state = SegmentState.ACTIVE.value
            self.state.active = True

    def _begin_closing(self, reason: str) -> None:
        logger.info(
            "%s begin_closing: symbol=%s reason=%s",
            self._engine_name,
            self.state.symbol,
            reason,
        )
        self.state.segment_state = SegmentState.CLOSING.value
        self.state.active = True

    def _maybe_deactivate_if_fully_closed(self) -> None:
        """Deactivate when local state is empty *and* the exchange book is clear.

        Exchange-side remnants (entry limits or protection) must keep the
        segment occupying the slot until reconcile cancels them. Going IDLE
        while cg_/cg3_ orders still rest allowed ``_start_grid`` to mint a new
        ``grid_id`` on top of the old book (ETH 2026-07-18 stuck entries).
        """
        if not (
            self.state.active
            and not self.state.inventory
            and not self.state.pending_orders
        ):
            return
        if self._exchange_has_open_activity():
            return
        self._deactivate("fully_closed")

    def is_stale_active_ghost(self) -> bool:
        if self._exchange_has_open_activity():
            return False
        return bool(
            self.state.active
            and not self.state.pending_orders
            and not self.state.inventory
        )

    def clear_stale_active_if_ghost(self) -> bool:
        if not self.is_stale_active_ghost():
            return False
        self._log_stale_active_reset()
        self._deactivate("ghost_cleared")
        return True

    def on_place_rejected(
        self,
        actions: Iterable[Mapping[str, Any]],
        *,
        reason: str = "place_rejected",
    ) -> int:
        """Drop optimistic pending rows when place never reached the exchange.

        Engines append local ``pending_orders`` before the orchestrator runs.
        Exposure lock / portfolio governor / symbol-owner vetoes therefore leave
        unmapped ghosts that reconcile reports as ``missing_exchange_orders``
        until ghost-TTL. Roll those rows back immediately and deactivate if the
        segment never acquired inventory.
        """
        reject_ids = {
            str(action.get("order_id") or action.get("local_order_id") or "").strip()
            for action in actions
            if str(action.get("action") or "").lower() == "place"
        }
        reject_ids.discard("")
        if not reject_ids:
            return 0
        before = len(self.state.pending_orders)
        self.state.pending_orders = [
            order
            for order in self.state.pending_orders
            if str(getattr(order, "order_id", "") or "") not in reject_ids
        ]
        dropped = before - len(self.state.pending_orders)
        if not dropped:
            return 0
        logger.warning(
            "%s on_place_rejected: dropped %d pending order(s) reason=%s ids=%s",
            self._engine_name,
            dropped,
            reason,
            sorted(reject_ids)[:12],
        )
        self._maybe_deactivate_if_fully_closed()
        self.save_state()
        return dropped

    def holds_real_grid_slot(self) -> bool:
        if not segment_occupies_slot(
            getattr(self.state, "segment_state", SegmentState.IDLE.value)
        ):
            return False
        if not bool(getattr(self.state, "active", False)):
            return False
        return bool(self.state.pending_orders or self.state.inventory)

    def _exchange_has_open_activity(self) -> bool:
        return bool(self._exchange_open_orders)
