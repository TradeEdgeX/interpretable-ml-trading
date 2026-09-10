"""Chain debug logging for live decision paths (trend PCM, multi-leg engines)."""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

import pandas as pd

logger = logging.getLogger(__name__)

# One log per (scope, symbol, 2h bar bucket) by default.
_BAR_DEDUPE_KEYS: Set[str] = set()
_BAR_DEDUPE_MAX = 4000

TREND_FEATURE_KEYS: tuple[str, ...] = (
    "close",
    "ema_200",
    "atr",
    "bpc_semantic_chop",
    "weekly_ema_200_position",
    "trend_confidence",
    "trend_direction",
    "box_prefilter",
)

SPOT_FEATURE_KEYS: tuple[str, ...] = (
    "close",
    "weekly_ema_200_position",
    "atr_percentile",
)

MULTILEG_FEATURE_KEYS: tuple[str, ...] = (
    "bpc_semantic_chop",
    "box_prefilter",
    "trend_confidence",
    "trend_direction",
)


def chain_debug_enabled(scope: str) -> bool:
    """True when MLBOT_CHAIN_DEBUG or MLBOT_{SCOPE}_CHAIN_DEBUG is truthy."""
    if _env_truthy("MLBOT_CHAIN_DEBUG"):
        return True
    key = f"MLBOT_{str(scope or '').strip().upper()}_CHAIN_DEBUG"
    return _env_truthy(key)


def _env_truthy(name: str) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return False
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _dedupe_bucket() -> str:
    """``MLBOT_CHAIN_DEBUG_BUCKET``: ``2h`` (default) or ``15min``."""
    raw = (os.getenv("MLBOT_CHAIN_DEBUG_BUCKET") or "2h").strip().lower()
    if raw in {"15m", "15min", "15"}:
        return "15min"
    return "2h"


def _bar_dedupe_key(ts: Any) -> str:
    """Floor timestamp to dedupe bucket (2h for trend/spot, 15min optional)."""
    if ts is None:
        return "unknown"
    try:
        import pandas as pd

        t = pd.Timestamp(ts)
        if t.tzinfo is None:
            t = t.tz_localize("UTC")
        else:
            t = t.tz_convert("UTC")
        freq = "15min" if _dedupe_bucket() == "15min" else "2h"
        return t.floor(freq).isoformat()
    except Exception:
        return str(ts)


def throttle_allows(
    scope: str,
    symbol: str,
    bar_ts: Any,
    *,
    strategy: str = "",
) -> bool:
    """At most one chain-debug line per scope/symbol/(strategy)/2h bar."""
    if not chain_debug_enabled(scope):
        return False
    sym = str(symbol or "").upper().strip()
    strat = str(strategy or "").strip().lower()
    parts = [str(scope).lower(), sym, _bar_dedupe_key(bar_ts)]
    if strat:
        parts.append(strat)
    key = ":".join(parts)
    if key in _BAR_DEDUPE_KEYS:
        return False
    if len(_BAR_DEDUPE_KEYS) >= _BAR_DEDUPE_MAX:
        _BAR_DEDUPE_KEYS.clear()
    _BAR_DEDUPE_KEYS.add(key)
    return True


def _compact_funnel(funnel: Any) -> Dict[str, Any]:
    if not isinstance(funnel, dict):
        return {}
    out: Dict[str, Any] = {}
    for k, v in funnel.items():
        key = str(k)
        if isinstance(v, (bool, int, float)) or v is None:
            out[key] = v
        elif isinstance(v, str):
            out[key] = v if len(v) <= 200 else v[:200] + "…"
        elif isinstance(v, list):
            text = ", ".join(str(x) for x in v[:8])
            if len(v) > 8:
                text += f", …(+{len(v) - 8})"
            out[key] = text
        elif isinstance(v, dict):
            out[key] = {str(sk): v[sk] for sk in list(v.keys())[:6]}
    return out


def infer_block_stage(funnel: Dict[str, Any]) -> str:
    """Best-effort stage where decide() stopped or PCM pre-filtered."""
    f = funnel or {}
    if not f:
        return "not_evaluated"
    if f.get("pcm_direction_filter") is False:
        return "pcm_ema_filter"
    if f.get("simple_deep_bear") is False:
        return "simple_not_deep_bear"
    if str(f.get("accumulation_policy") or "").startswith("bull"):
        return "accumulation_policy"
    if f.get("regime") is False:
        return "regime_deny"
    if f.get("prefilter") is False:
        return "prefilter_deny"
    if f.get("regime_side_block"):
        return "regime_side_deny"
    if f.get("direction") is False or f.get("direction_value") == 0:
        return "no_direction"
    if f.get("gate") is False:
        return "gate_deny"
    if f.get("entry_filter") is False:
        return "entry_filter_deny"
    if f.get("reject_srb_wide_sr_too_close"):
        return "srb_wide_guard"
    if f.get("prefilter") is True and f.get("direction") is True:
        if f.get("gate") is True and f.get("entry_filter") is not False:
            return "strategy_layers_passed"
    return "unknown"


def summarize_layers(funnel: Dict[str, Any]) -> Dict[str, str]:
    """Fixed layer slots: pass / fail / n/a for quick scanning."""
    f = funnel or {}

    def _mark(present: bool, passed: Any) -> str:
        if not present:
            return "n/a"
        if passed is True:
            return "pass"
        if passed is False:
            return "fail"
        return str(passed)

    return {
        "simple_deep_bear": _mark("simple_deep_bear" in f, f.get("simple_deep_bear")),
        "regime": _mark("regime" in f, f.get("regime")),
        "prefilter": _mark("prefilter" in f, f.get("prefilter")),
        "direction": _mark("direction" in f, f.get("direction")),
        "gate": _mark("gate" in f, f.get("gate")),
        "entry_filter": _mark("entry_filter" in f, f.get("entry_filter")),
    }


def pick_features(features: Dict[str, Any], keys: tuple[str, ...]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key in keys:
        if key not in features:
            continue
        val = features.get(key)
        if val is None:
            out[key] = None
        elif isinstance(val, float):
            out[key] = round(val, 6)
        else:
            out[key] = val
    return out


def log_trend_no_intent(
    symbol: str,
    decision_handler: Any,
    features: Dict[str, Any],
) -> None:
    """Log why trend/PCM returned no TradeIntent on this decision cycle."""
    ts = features.get("timestamp")
    if not throttle_allows("trend", symbol, ts):
        return

    from src.time_series_model.portfolio.live_pcm import LivePCM

    feat = pick_features(features, TREND_FEATURE_KEYS)
    if isinstance(decision_handler, LivePCM):
        trace = dict(getattr(decision_handler, "_last_decide_trace", None) or {})
        per_strategy: Dict[str, Any] = {}
        for arch, strat in (
            getattr(decision_handler, "_strategies", None) or {}
        ).items():
            funnel = _compact_funnel(getattr(strat, "_last_funnel", None))
            per_strategy[str(arch)] = {
                "block": infer_block_stage(funnel),
                "layers": summarize_layers(funnel),
                "funnel": funnel,
            }
        logger.info(
            "[%s] signal-check no intent ts=%s features=%s pcm_trace=%s strategies=%s",
            symbol,
            ts,
            feat,
            trace,
            per_strategy,
        )
        return

    funnel = _compact_funnel(getattr(decision_handler, "_last_funnel", None))
    logger.info(
        "[%s] signal-check no intent ts=%s handler=%s block=%s layers=%s "
        "features=%s funnel=%s",
        symbol,
        ts,
        type(decision_handler).__name__,
        infer_block_stage(funnel),
        summarize_layers(funnel),
        feat,
        funnel,
    )


def spot_eligibility_log_enabled() -> bool:
    """Default on; set MLBOT_SPOT_ELIGIBILITY_LOG=false to silence."""
    raw = os.getenv("MLBOT_SPOT_ELIGIBILITY_LOG")
    if raw is None:
        return True
    return _env_truthy("MLBOT_SPOT_ELIGIBILITY_LOG")


def collect_spot_new_buy_report(
    *,
    symbol: str,
    ts: Any,
    features: Dict[str, Any],
    strategy: Any,
    deploy_schedule_cfg: Dict[str, Any],
    budget: Any,
    positions: Dict[str, Any],
    ledger: Any,
    day_key: str,
    intents: List[Any],
    om_shadow: bool,
    planned_usdt: float = 0.0,
    free_usdt: Optional[float] = None,
) -> Dict[str, Any]:
    """Dry-run all spot new-buy gates; used for one-line eligibility logging."""
    from src.time_series_model.live.spot_accum_simple import (
        deploy_schedule_allows_new_buy,
    )
    from src.order_management.spot_live_recovery import (
        floor_spot_buy_ts,
        has_blocking_pending_buy,
        pending_buy_count_for_day,
    )

    sym = str(symbol or "").upper()
    wk_raw = features.get("weekly_ema_200_position")
    try:
        weekly_ema = float(wk_raw) if wk_raw is not None else None
    except (TypeError, ValueError):
        weekly_ema = None
    below_weekly_ema200 = (
        weekly_ema is not None and weekly_ema == weekly_ema and weekly_ema < 0.0
    )

    pf_pass: Optional[bool] = None
    pf_reason: Optional[str] = None
    archetype = getattr(strategy, "archetype", None)
    if archetype is not None and getattr(archetype, "prefilter", None) is not None:
        pf_pass, pf_reason = archetype.prefilter.evaluate(features, latch_id=symbol)

    sched_ok, sched_detail = deploy_schedule_allows_new_buy(
        (
            pd.Timestamp(ts).to_pydatetime()
            if ts is not None
            else datetime.now(timezone.utc)
        ),
        deploy_schedule_cfg or {},
    )
    if sched_ok and not sched_detail:
        sched_detail = "in_window_or_schedule_disabled"

    signal_long = (
        bool(intents) and str(getattr(intents[0], "action", "") or "").upper() == "LONG"
    )
    funnel = dict(getattr(strategy, "_last_funnel", None) or {})

    blockers: List[str] = []
    # Buy-environment gate is prefilter.yaml only (latch / skip_symbols / operators).
    # Do not re-hardcode weekly_ema < 0 here — that ignored latch and skip_symbols.
    if pf_pass is False:
        blockers.append(f"prefilter_deny:{pf_reason or 'unknown'}")
    if not signal_long:
        blockers.append(f"no_long_signal:{infer_block_stage(funnel)}")
    if not sched_ok:
        blockers.append(f"deploy_schedule:{sched_detail}")

    scope = str(getattr(budget, "max_new_entries_scope", "account") or "account")
    scope = scope.strip().lower()
    if scope == "symbol":
        try:
            entries_today = int(ledger.buy_entries_today(day_key, symbol=sym))
        except TypeError:
            entries_today = int(ledger.buy_entries_today(day_key))
        pending_today = int(
            pending_buy_count_for_day(positions, day_key=day_key, symbol=sym)
        )
    else:
        entries_today = int(ledger.buy_entries_today(day_key))
        pending_today = int(pending_buy_count_for_day(positions, day_key=day_key))
    if entries_today + pending_today >= int(budget.max_new_entries_per_day):
        blockers.append(
            f"day_limit:scope={scope} entries={entries_today} pending={pending_today} "
            f"max={budget.max_new_entries_per_day}"
        )

    cur = positions.get(sym) or {}
    raw_last = cur.get("_last_buy_ts")
    if raw_last and budget.min_order_interval_minutes > 0:
        try:
            last_buy_ts = floor_spot_buy_ts(raw_last)
            bar_ts = floor_spot_buy_ts(ts)
            if last_buy_ts is not None and bar_ts is not None:
                mins = (bar_ts - last_buy_ts).total_seconds() / 60.0
                if mins < float(budget.min_order_interval_minutes):
                    blockers.append(
                        f"min_interval:mins={mins:.0f}<{budget.min_order_interval_minutes}"
                    )
        except Exception:
            pass

    if has_blocking_pending_buy(cur):
        blockers.append("open_pending_limit_buy")

    close_px = float(features.get("close") or 0.0)
    planned = max(0.0, float(planned_usdt or 0.0))
    if signal_long and not blockers and planned <= 0.0:
        blockers.append("planned_quote_zero(budget_cap_or_decay)")

    free_quote: Optional[float] = None
    if free_usdt is not None:
        try:
            free_quote = max(0.0, float(free_usdt))
        except (TypeError, ValueError):
            free_quote = None
    # Live (non-shadow): balance REST must succeed before a new buy.
    # free_usdt=None used to fail-open and size off the offline equity anchor.
    if not om_shadow and free_quote is None:
        blockers.append("balance_unavailable")
    elif planned > 0.0 and free_quote is not None and planned > free_quote + 1e-6:
        blockers.append(
            f"insufficient_free_usdt:need={planned:.2f} free={free_quote:.2f}"
        )

    can_submit = signal_long and not blockers and planned > 0.0 and close_px > 0.0
    return {
        "symbol": sym,
        "ts": ts,
        "weekly_ema_200_position": weekly_ema,
        "below_weekly_ema200": below_weekly_ema200,
        "prefilter_pass": pf_pass,
        "prefilter_reason": pf_reason,
        "deploy_schedule_ok": sched_ok,
        "deploy_schedule_detail": sched_detail,
        "signal_long": signal_long,
        "layers": summarize_layers(funnel),
        "block_stage": infer_block_stage(funnel),
        "planned_usdt": round(planned, 2),
        "free_usdt": round(free_quote, 2) if free_quote is not None else None,
        "close": round(close_px, 6) if close_px > 0 else None,
        "can_submit_new_buy": can_submit,
        "shadow_mode": bool(om_shadow),
        "blockers": blockers,
        "features": pick_features(features, SPOT_FEATURE_KEYS),
    }


def log_spot_new_buy_eligibility(report: Dict[str, Any]) -> None:
    """One INFO line per symbol per 2h bar: feature regime + can submit + reasons."""
    if not spot_eligibility_log_enabled():
        return
    sym = str(report.get("symbol") or "")
    ts = report.get("ts")
    key = f"spot_elig:{sym}:{_bar_dedupe_key(ts)}"
    if key in _BAR_DEDUPE_KEYS:
        return
    if len(_BAR_DEDUPE_KEYS) >= _BAR_DEDUPE_MAX:
        _BAR_DEDUPE_KEYS.clear()
    _BAR_DEDUPE_KEYS.add(key)
    can = bool(report.get("can_submit_new_buy"))
    wk = report.get("weekly_ema_200_position")
    wk_s = "nan" if wk is None else f"{float(wk):.4f}"
    below = report.get("below_weekly_ema200")
    sched_ok = report.get("deploy_schedule_ok")
    sched_d = report.get("deploy_schedule_detail") or ""
    blockers = report.get("blockers") or []
    if can:
        logger.info(
            "[%s] spot-eligibility ts=%s CAN_SUBMIT_NEW_BUY weekly_ema=%s below_wk_ema200=%s "
            "prefilter=%s schedule_ok=%s planned_usdt=%.2f close=%s shadow=%s features=%s",
            sym,
            ts,
            wk_s,
            below,
            report.get("prefilter_pass"),
            sched_ok,
            float(report.get("planned_usdt") or 0.0),
            report.get("close"),
            report.get("shadow_mode"),
            report.get("features"),
        )
    else:
        logger.info(
            "[%s] spot-eligibility ts=%s NO_NEW_BUY weekly_ema=%s below_wk_ema200=%s "
            "prefilter=%s prefilter_reason=%s schedule_ok=%s schedule=%s signal_long=%s "
            "layers=%s block_stage=%s shadow=%s reasons=%s features=%s",
            sym,
            ts,
            wk_s,
            below,
            report.get("prefilter_pass"),
            report.get("prefilter_reason"),
            sched_ok,
            sched_d,
            report.get("signal_long"),
            report.get("layers"),
            report.get("block_stage"),
            report.get("shadow_mode"),
            "; ".join(str(x) for x in blockers) or "unknown",
            report.get("features"),
        )


def log_spot_no_intent(
    symbol: str,
    strategy: Any,
    features: Dict[str, Any],
) -> None:
    ts = features.get("timestamp")
    if not throttle_allows("spot", symbol, ts):
        return
    funnel = _compact_funnel(getattr(strategy, "_last_funnel", None) or {})
    logger.info(
        "[%s] signal-check no intent ts=%s block=%s layers=%s features=%s funnel=%s",
        symbol,
        ts,
        infer_block_stage(funnel),
        summarize_layers(funnel),
        pick_features(features, SPOT_FEATURE_KEYS),
        funnel,
    )


def _multileg_regime_snapshot(engine: Any, features: Dict[str, Any]) -> Dict[str, Any]:
    snap: Dict[str, Any] = {}
    state = getattr(engine, "state", None)
    if state is not None:
        snap["active"] = getattr(state, "active", None)
        inv = getattr(state, "inventory", None)
        pending = getattr(state, "pending_orders", None)
        snap["inventory"] = len(inv) if inv is not None else 0
        snap["pending_orders"] = len(pending) if pending is not None else 0
        snap["trend_side"] = getattr(state, "trend_side", None)
        snap["grid_id"] = getattr(state, "grid_id", None)
        snap["spacing"] = getattr(state, "spacing", None)
    chop = features.get("bpc_semantic_chop")
    snap["features"] = pick_features(features, MULTILEG_FEATURE_KEYS)
    cfg = getattr(engine, "cfg", None)
    if cfg is not None:
        if hasattr(cfg, "entry_chop_min"):
            snap["regime"] = {
                "chop": chop,
                "entry_chop_min": getattr(cfg, "entry_chop_min", None),
                "exit_chop_below": getattr(cfg, "exit_chop_below", None),
                "wanted_enter": (
                    float(chop or 0) >= float(getattr(cfg, "entry_chop_min", 0))
                    if chop is not None
                    else None
                ),
                "should_exit_chop": (
                    float(chop or 1) < float(getattr(cfg, "exit_chop_below", 0))
                    if chop is not None
                    else None
                ),
            }
        elif hasattr(cfg, "entry_trend_min"):
            snap["regime"] = {
                "chop": chop,
                "trend_conf": features.get("trend_confidence"),
                "entry_trend_min": getattr(cfg, "entry_trend_min", None),
                "exit_trend_below": getattr(cfg, "exit_trend_below", None),
                "max_entry_chop": getattr(cfg, "max_entry_chop", None),
                "max_hold_chop": getattr(cfg, "max_hold_chop", None),
                "box_blocked": (
                    bool(getattr(cfg, "exclude_box_prefilter", False))
                    and bool(features.get("box_prefilter"))
                ),
            }
    return snap


def log_multileg_bar_no_actions(
    *,
    strategy: str,
    symbol: str,
    timestamp: str,
    engine: Any,
    features: Dict[str, Any],
) -> None:
    """Log once per processed bar when the multi-leg engine emitted no actions."""
    if not throttle_allows("multi_leg", symbol, timestamp, strategy=strategy):
        return
    snap = _multileg_regime_snapshot(engine, features)
    block = "flat_inactive"
    if snap.get("active"):
        block = "active_no_action"
    regime = snap.get("regime") or {}
    if isinstance(regime, dict):
        if regime.get("should_exit_chop") is True:
            block = "would_exit_chop"
        elif regime.get("wanted_enter") is False:
            block = "chop_below_entry"
        elif regime.get("box_blocked"):
            block = "box_prefilter"
        elif regime.get("trend_conf") is not None:
            try:
                tc = float(regime.get("trend_conf"))
                if tc < float(regime.get("entry_trend_min", 0)):
                    block = "trend_conf_low"
                elif float(regime.get("chop") or 1) > float(
                    regime.get("max_entry_chop", 1)
                ):
                    block = "chop_above_entry_cap"
            except (TypeError, ValueError):
                pass
    logger.info(
        "[%s] %s bar-check no actions ts=%s block=%s snapshot=%s",
        symbol,
        strategy,
        timestamp,
        block,
        snap,
    )
