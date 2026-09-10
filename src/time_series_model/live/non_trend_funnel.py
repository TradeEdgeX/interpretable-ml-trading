"""Funnel helpers for non-trend (spot / multi_leg) live runners.

The trend stack writes 15min funnel rows via ``StatsCollector`` from
``run_live.py``; this module gives the spot accumulation and multi-leg
runners the same hook so the console "策略漏斗" panel can show A/C-layer
counts. The runners construct a ``funnel`` dict per evaluation and call
``StatsCollector.record_strategy_eval``; this file only owns the small
shared bits (a wall-clock 15min flusher and per-domain funnel mappers).
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional


# ── Path resolution ────────────────────────────────────────────

COIN_M_STRATEGY_SUFFIX = "@coin_m"
DEFAULT_COIN_TREND_MONITOR_DB = Path("data/bc_coin_trend_live/live_monitor.db")
DEFAULT_COIN_MULTILEG_MONITOR_DB = Path("data/bc_coin_multileg_live/live_monitor.db")


def coin_m_funnel_strategy_id(strategy: str) -> str:
    """Normalize coin-m funnel strategy ids to match CMS matrix (``srb@coin_m``)."""
    base = str(strategy or "").strip().lower()
    if not base:
        return base
    if base.endswith(COIN_M_STRATEGY_SUFFIX):
        return base
    return f"{base}{COIN_M_STRATEGY_SUFFIX}"


def lane_tag_funnel_strategy(strategy: str) -> str:
    """Tag strategy ids when this process is a coin_m lane (fail-closed env)."""
    mk = os.getenv("MLBOT_MARGIN_KIND", "").strip().lower()
    suffix = os.getenv("MLBOT_FUNNEL_STRATEGY_SUFFIX", "").strip()
    if mk == "coin_m" or suffix == COIN_M_STRATEGY_SUFFIX:
        return coin_m_funnel_strategy_id(strategy)
    return str(strategy or "").strip().lower()


def default_live_monitor_db_path() -> Path:
    """U-m / shared highcap stats path (or ``MLBOT_STATS_DB_PATH`` override)."""
    return resolve_live_monitor_db_path(margin_kind="usd_m")


def resolve_live_monitor_db_path(
    *,
    margin_kind: Optional[str] = None,
    account_scope: Optional[str] = None,
) -> Path:
    """Resolve stats_15min DB for this process.

    Coin-m B/C use independent persistent paths under ``data/bc_coin_*_live/``
    so CMS can merge without concurrent writers colliding on one SQLite file.
    """
    override = os.getenv("MLBOT_STATS_DB_PATH")
    if override:
        return Path(override)
    mk = (
        margin_kind
        if margin_kind is not None
        else os.getenv("MLBOT_MARGIN_KIND", "usd_m")
    )
    mk = str(mk or "usd_m").strip().lower()
    if mk == "coin_m":
        scope = (
            account_scope
            if account_scope is not None
            else os.getenv("MLBOT_ACCOUNT_SCOPE", "")
        )
        scope = str(scope or "").strip().lower()
        if scope == "multi_leg":
            return DEFAULT_COIN_MULTILEG_MONITOR_DB
        return DEFAULT_COIN_TREND_MONITOR_DB
    base = os.getenv("MLBOT_LIVE_BASE", "live/highcap")
    return Path(base) / "data" / "db" / "live_monitor.db"


# ── Funnel dict builders ───────────────────────────────────────

# Spot has no regime/prefilter chain; multi-leg engines apply their own gating
# inside ``engine.on_bar`` and the portfolio risk governor. Both report into a
# single shape so the console can render them next to trend.


def funnel_for_spot_decision(
    *,
    has_intent: bool,
    can_submit: bool,
    blocker: Optional[str] = None,
) -> Dict[str, Any]:
    """``record_strategy_eval``-shaped dict for one spot bar evaluation."""
    gate_reasons: List[str] = []
    if has_intent and not can_submit and blocker:
        gate_reasons.append(str(blocker)[:60])
    return {
        "regime": True,
        "prefilter": True,
        "direction": bool(has_intent),
        "direction_value": 1 if has_intent else 0,
        "gate": bool(has_intent and can_submit),
        "gate_reasons": gate_reasons,
        "entry_filter": bool(has_intent and can_submit),
        "evidence": bool(has_intent and can_submit),
    }


def funnel_for_multileg_bar(
    *,
    strategy: str = "",
    engine_audit: Optional[Mapping[str, Any]] = None,
    actions: Iterable[Any],
    approved_actions: Iterable[Any],
    rejected: Iterable[Any],
) -> Dict[str, Any]:
    """``record_strategy_eval``-shaped dict for one multi-leg bar evaluation."""
    from src.time_series_model.live.multileg_funnel import (
        funnel_for_multileg_bar as _build,
    )

    return _build(
        strategy=strategy,
        engine_audit=engine_audit,
        actions=actions,
        approved_actions=approved_actions,
        rejected=rejected,
    )


# ── 15min wall-clock flusher ───────────────────────────────────


class FifteenMinFlusher:
    """Calls ``StatsCollector.flush`` once per ``interval_s`` of wall time.

    Trend uses the bar listener's per-bar flush hook; spot/multi-leg loops
    don't have an equivalent symbol-aligned cadence, so a simple monotonic
    timer is used. ``maybe_flush`` is cheap and idempotent — call it once per
    poll iteration.
    """

    def __init__(
        self,
        stats_collector: Any,
        *,
        interval_s: float = 900.0,
        regime: str = "NORMAL",
    ) -> None:
        self.stats_collector = stats_collector
        self.interval_s = max(1.0, float(interval_s))
        self.regime = str(regime or "NORMAL")
        self._last_flush_mono = time.monotonic()

    def maybe_flush(
        self,
        *,
        symbol: str = "ALL",
        positions: Optional[Dict[str, Any]] = None,
        system_health: Optional[Dict[str, Any]] = None,
    ) -> bool:
        if self.stats_collector is None:
            return False
        now = time.monotonic()
        if now - self._last_flush_mono < self.interval_s:
            return False
        try:
            self.stats_collector.flush(
                regime=self.regime,
                positions=positions or {},
                system_health=system_health or {},
                symbol=symbol,
            )
        finally:
            self._last_flush_mono = now
        return True

    def force_flush(
        self,
        *,
        symbol: str = "ALL",
        positions: Optional[Dict[str, Any]] = None,
        system_health: Optional[Dict[str, Any]] = None,
    ) -> bool:
        if self.stats_collector is None:
            return False
        try:
            self.stats_collector.flush(
                regime=self.regime,
                positions=positions or {},
                system_health=system_health or {},
                symbol=symbol,
            )
        finally:
            self._last_flush_mono = time.monotonic()
        return True
