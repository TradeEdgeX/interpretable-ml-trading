"""Spot accumulator budget, deploy legs, and fill simulation for event backtest."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import pandas as pd

from src.time_series_model.live.spot_accum_simple import (
    deploy_decay_multiplier,
    is_spot_accum_archetype,
)
from src.time_series_model.live.spot_budget import (
    UNIT_MODE_BUDGET_TRANCHES,
    parse_symbol_budget_pct,
    parse_symbol_budgets_usdt,
    parse_tranches_per_symbol,
    parse_unit_mode,
    resolve_symbol_budgets_usdt,
    resolve_symbol_units_usdt,
)

if TYPE_CHECKING:
    from scripts.event_backtest.engine import PositionSimulator


def clamp01(x: float) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, v))


def build_spot_capital_budget_or_none(
    *,
    constitution_raw: Dict[str, Any],
    strategy_names: List[str],
    equity_anchor_usdt: Optional[float],
) -> Optional[Dict[str, Any]]:
    """从 constitution YAML 顶层 ``spot`` 域生成 ``spot_accum`` 事件回测资金预算."""
    try:
        eq = float(equity_anchor_usdt if equity_anchor_usdt is not None else 0.0)
    except (TypeError, ValueError):
        eq = 0.0
    if eq <= 0:
        return None
    _spot_names = {str(x or "").strip().lower() for x in (strategy_names or [])}
    spot_requested = bool(_spot_names & {"spot_accum", "spot_accum_simple"})
    if not spot_requested:
        return None
    spot = constitution_raw.get("spot")
    if not isinstance(spot, dict):
        raise ValueError(
            "spot_accum event backtest requires constitution top-level 'spot' budget"
        )
    acc = spot.get("accumulation") if isinstance(spot.get("accumulation"), dict) else {}
    rl = spot.get("risk_limits") if isinstance(spot.get("risk_limits"), dict) else {}
    target_deploy_pct = clamp01(acc.get("target_deploy_pct", 1.0))
    try:
        unit_notional = float(acc.get("unit_notional", 0.0) or 0.0)
    except (TypeError, ValueError):
        unit_notional = 0.0
    try:
        tranche_count = int(acc.get("tranche_count") or 0)
    except (TypeError, ValueError):
        tranche_count = 0
    tranches_per_symbol = parse_tranches_per_symbol(acc.get("tranches_per_symbol"))
    unit_mode = parse_unit_mode(acc.get("unit_mode"))
    symbol_budgets = resolve_symbol_budgets_usdt(
        equity_usdt=eq,
        target_deploy_pct=target_deploy_pct,
        symbol_budget_pct=parse_symbol_budget_pct(acc.get("symbol_budget_pct")),
        symbol_budgets_fallback=parse_symbol_budgets_usdt(
            acc.get("symbol_budgets_usdt")
        ),
    )
    symbol_unit_fallback = parse_symbol_budgets_usdt(
        acc.get("symbol_unit_notional_usdt")
    )
    if not symbol_budgets:
        raise ValueError(
            "spot_accum event backtest requires non-empty spot.accumulation "
            "symbol_budget_pct or symbol_budgets_usdt"
        )
    if unit_mode == UNIT_MODE_BUDGET_TRANCHES:
        symbol_unit = resolve_symbol_units_usdt(
            symbol_budgets_usdt=symbol_budgets,
            unit_mode=UNIT_MODE_BUDGET_TRANCHES,
            tranches_per_symbol=tranches_per_symbol,
        )
        tranche_count = tranches_per_symbol
        if unit_notional <= 0.0 and symbol_unit:
            unit_notional = float(min(symbol_unit.values()))
    else:
        symbol_unit = dict(symbol_unit_fallback)
        if symbol_budgets and tranches_per_symbol > 0:
            for sk, sb in symbol_budgets.items():
                symbol_unit.setdefault(sk, max(1.0, sb / float(tranches_per_symbol)))
            tranche_count = tranches_per_symbol
            if unit_notional <= 0.0 and symbol_unit:
                unit_notional = float(min(symbol_unit.values()))
        elif unit_notional <= 0:
            unit_notional = max(eq * target_deploy_pct / max(tranche_count, 1), 1.0)
    if tranche_count <= 0:
        tranche_count = max(1, int(round(eq * target_deploy_pct / unit_notional)))
    max_gross = max(1e-9, float(rl.get("max_gross_notional_pct", 1.0) or 1.0))
    max_symbol = max(1e-9, float(rl.get("max_symbol_gross_notional_pct", 1.0) or 1.0))
    max_daily = max(0.0, float(rl.get("max_daily_deploy_pct", 1.0) or 1.0))

    return {
        "equity_usdt": eq,
        "target_deploy_pct": float(target_deploy_pct),
        "unit_notional_usdt": float(unit_notional),
        "tranche_count": int(tranche_count),
        "tranches_per_symbol": int(tranches_per_symbol),
        "unit_mode": str(unit_mode),
        "symbol_budgets_usdt": dict(symbol_budgets),
        "symbol_unit_notional_usdt": dict(symbol_unit),
        "max_gross_notional_pct": float(max_gross),
        "max_symbol_gross_notional_pct": float(max_symbol),
        "max_daily_deploy_pct": float(max_daily),
        "dust_frac": 0.05,
    }


def spot_peer_sims(sim: "PositionSimulator") -> List["PositionSimulator"]:
    if sim._spot_peer_sims:
        return list(sim._spot_peer_sims)
    return [sim]


def portfolio_spot_accum_deployed_quote_usd(sim: "PositionSimulator") -> float:
    tot = 0.0
    for s in _spot_peer_sims(sim):
        for pos in s._positions.values():
            if bool(pos.get("_is_add_position", False)):
                continue
            if not is_spot_accum_archetype(str(pos.get("archetype", "") or "")):
                continue
            tot += float(pos.get("_spot_quote_deployed", 0.0) or 0.0)
    return float(tot)


def utc_calendar_day_str(ts: Any) -> str:
    if isinstance(ts, str):
        dt = pd.Timestamp(ts).to_pydatetime()
    elif isinstance(ts, pd.Timestamp):
        dt = ts.to_pydatetime()
    elif isinstance(ts, datetime):
        dt = ts
    else:
        dt = datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return pd.Timestamp(dt).strftime("%Y-%m-%d")


def spot_entry_fill_price(
    entry_bar: Dict[str, Any],
    exec_cons: Dict[str, Any],
    *,
    is_long: bool = True,
) -> Tuple[Optional[float], str]:
    """Spot entry fill: market @ close, or limit offset below/above close if bar touches."""
    close_px = float(entry_bar.get("close", 0) or 0.0)
    if close_px <= 0:
        return None, "spot_entry_no_close"
    eo = (exec_cons or {}).get("entry_order") or {}
    if not isinstance(eo, dict):
        eo = {}
    otype = str(eo.get("type") or "market").strip().lower()
    if otype != "limit":
        return close_px, "market"
    try:
        bps = float(eo.get("limit_offset_bps", 0) or 0.0)
    except (TypeError, ValueError):
        bps = 0.0
    bps = max(0.0, bps)
    if is_long:
        limit_px = close_px * (1.0 - bps / 10000.0)
        bar_touch = float(entry_bar.get("low", close_px) or close_px)
        if bar_touch <= limit_px:
            return limit_px, "limit_filled"
        return None, "spot_limit_not_filled"
    limit_px = close_px * (1.0 + bps / 10000.0)
    bar_touch = float(entry_bar.get("high", close_px) or close_px)
    if bar_touch >= limit_px:
        return limit_px, "limit_filled"
    return None, "spot_limit_not_filled"


def spot_symbol_deploy_legs_today(
    sim: "PositionSimulator", *, symbol: str, day_key: str
) -> int:
    dm = getattr(sim, "_spot_symbol_daily_leg_counts", None)
    if not isinstance(dm, dict):
        return 0
    return int(dm.get(f"{symbol}|{day_key}", 0) or 0)


def spot_regime_leg_kwargs(
    features: Dict[str, Any], execution_profile: Dict[str, Any]
) -> Dict[str, Any]:
    profile = execution_profile or {}
    nested = profile.get("execution_constraints")
    exec_cons = nested if isinstance(nested, dict) else profile
    decay = profile.get("deploy_decay")
    if not isinstance(decay, dict):
        decay = exec_cons.get("deploy_decay")
    out_extra: Dict[str, Any] = {}
    if isinstance(decay, dict):
        out_extra["deploy_decay_scale"] = decay
    scale = (exec_cons or {}).get("regime_deploy_scale")
    if not isinstance(scale, dict):
        return {"regime_score": None, "regime_deploy_scale": None, **out_extra}
    feat = str(scale.get("feature") or "abc_macro_regime_score")
    raw = features.get(feat)
    try:
        sc = float(raw) if raw is not None else None
    except (TypeError, ValueError):
        sc = None
    if sc is not None and sc != sc:
        sc = None
    return {"regime_score": sc, "regime_deploy_scale": scale, **out_extra}


def spot_regime_unit_multiplier(
    regime_score: Optional[float], scale_cfg: Optional[Dict[str, Any]]
) -> float:
    """Map abc_macro_regime_score → deploy unit multiplier (lower score → larger size)."""
    if not isinstance(scale_cfg, dict) or not scale_cfg.get("enabled", False):
        return 1.0
    try:
        sc = float(regime_score if regime_score is not None else 999.0)
    except (TypeError, ValueError):
        return 1.0
    if sc != sc:
        return 1.0
    tiers = scale_cfg.get("tiers") or []
    if not isinstance(tiers, list):
        return 1.0
    ordered = sorted(
        tiers,
        key=lambda t: float(
            (t or {}).get("max_score_exclusive", (t or {}).get("max_score", 999.0))
            or 999.0
        ),
    )
    for tier in ordered:
        if not isinstance(tier, dict):
            continue
        bound = tier.get("max_score_exclusive", tier.get("max_score"))
        if bound is None:
            continue
        try:
            if sc < float(bound):
                return max(0.0, float(tier.get("unit_multiplier", 1.0) or 1.0))
        except (TypeError, ValueError):
            continue
    return 1.0


def record_spot_symbol_deploy_leg(
    sim: "PositionSimulator", *, symbol: str, day_key: str
) -> None:
    dm = getattr(sim, "_spot_symbol_daily_leg_counts", None)
    if not isinstance(dm, dict):
        return
    k = f"{symbol}|{day_key}"
    dm[k] = int(dm.get(k, 0) or 0) + 1


def mother_spot_deployed_usd_on_symbol(sim: "PositionSimulator", symbol: str) -> float:
    tot = 0.0
    for s in _spot_peer_sims(sim):
        for pos in s._positions.values():
            if bool(pos.get("_is_add_position", False)):
                continue
            if not is_spot_accum_archetype(str(pos.get("archetype", "") or "")):
                continue
            if str(pos.get("symbol", "") or "") != symbol:
                continue
            tot += float(pos.get("_spot_quote_deployed", 0.0) or 0.0)
    return float(tot)


def allocate_spot_accum_leg(
    sim: "PositionSimulator",
    *,
    archetype_lc: str,
    symbol: str,
    intent_base_add_m: float,
    parent_pos_for_merge: Optional[Dict[str, Any]],
    now_ts: datetime,
    regime_score: Optional[float] = None,
    regime_deploy_scale: Optional[Dict[str, Any]] = None,
    deploy_decay_scale: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, float, float, str, str]:
    """在 ``spot`` 预算内确定缩放倍数 + 账本计价 USD。

    Returns:
        ok, scaled_add_m, planned_leg_quote_usdt, daily_key_or_empty, reason_suffix

    Caller 仅在成功写入持仓后更新 ``daily_key`` 下的日预算。
    """
    b = getattr(sim, "_spot_capital_budget", None)
    if not isinstance(b, dict):
        if is_spot_accum_archetype(archetype_lc):
            raise RuntimeError(
                "spot_accum event backtest has no capital budget; "
                "refusing unbounded generic sizing"
            )
        bm = float(intent_base_add_m or 1.0)
        return True, bm if bm > 0 else 1.0, 0.0, "", ""

    if not is_spot_accum_archetype(archetype_lc):
        bm = float(intent_base_add_m or 1.0)
        return True, bm if bm > 0 else 1.0, 0.0, "", ""

    eq = float(b.get("equity_usdt", 0.0) or 0.0)
    if eq <= 0.0:
        return False, 0.0, 0.0, "", "spot_budget_no_equity"

    unit_usd = float(b.get("unit_notional_usdt", 1.0) or 1.0)
    if unit_usd <= 0:
        unit_usd = 1.0

    tg = max(1e-9, min(float(b.get("target_deploy_pct", 1.0) or 1.0), 1.0))
    cap_g_pct = float(b.get("max_gross_notional_pct", 1.0) or 1.0)
    cap_sy_pct = float(b.get("max_symbol_gross_notional_pct", 1.0) or 1.0)
    cap_daily_pct = float(b.get("max_daily_deploy_pct", 1.0) or 1.0)
    dust_frac = float(b.get("dust_frac", 0.05) or 0.05)

    deployed_cap_tgt = eq * tg
    cap_global = deployed_cap_tgt * cap_g_pct
    cap_symbol = eq * cap_sy_pct
    cap_daily = eq * max(0.0, cap_daily_pct)

    deployed_global = _portfolio_spot_accum_deployed_quote_usd(sim)
    if isinstance(parent_pos_for_merge, dict):
        sym_deployed_mother = float(
            parent_pos_for_merge.get("_spot_quote_deployed", 0.0) or 0.0
        )
    else:
        sym_deployed_mother = float(_mother_spot_deployed_usd_on_symbol(sim, symbol))

    symbol_budgets = (
        b.get("symbol_budgets_usdt")
        if isinstance(b.get("symbol_budgets_usdt"), dict)
        else {}
    )
    symbol_units = (
        b.get("symbol_unit_notional_usdt")
        if isinstance(b.get("symbol_unit_notional_usdt"), dict)
        else {}
    )
    if symbol_budgets:
        sb = float(symbol_budgets.get(str(symbol).upper(), 0.0) or 0.0)
        if sb <= 0.0:
            return False, 0.0, 0.0, "", "spot_budget_symbol_missing"
        cap_symbol = sb
        su = float(symbol_units.get(str(symbol).upper(), 0.0) or 0.0)
        if su > 0.0:
            unit_usd = su
        cap_global = max(cap_global, float(sum(symbol_budgets.values())))

    remain_global = max(0.0, cap_global - deployed_global + 1e-12)
    remain_symbol = max(0.0, cap_symbol - sym_deployed_mother)

    unit_mult = _spot_regime_unit_multiplier(regime_score, regime_deploy_scale)
    leg_unit = float(unit_usd) * float(unit_mult)
    if isinstance(deploy_decay_scale, dict) and deploy_decay_scale.get("enabled"):
        decay_mult = deploy_decay_multiplier(
            sym_deployed_mother, cap_symbol, deploy_decay_scale
        )
        leg_unit *= float(decay_mult)
    planned = min(leg_unit, remain_global, remain_symbol)

    dk = ""
    dm = getattr(sim, "_spot_daily_deploy_totals", None)
    if dm is not None:
        tsu = pd.Timestamp(now_ts)
        if tsu.tzinfo is None:
            tsu = tsu.tz_localize("UTC")
        else:
            tsu = tsu.tz_convert("UTC")
        dk = tsu.strftime("%Y-%m-%d")
        spent = float(dm.get(dk, 0.0) or 0.0)
        remain_daily = max(0.0, cap_daily - spent)
        planned = min(planned, remain_daily)

    if planned <= 0.0 or planned < dust_frac * leg_unit:
        return False, 0.0, 0.0, "", "spot_budget_room"

    base_m = float(intent_base_add_m or 1.0)
    if base_m <= 0.0:
        base_m = 1.0
    frac = planned / leg_unit if leg_unit > 0 else 0.0
    if frac <= 0.0:
        return False, 0.0, 0.0, "", "spot_budget_frac"

    scaled = base_m * frac
    return True, float(scaled), float(planned), dk, ""


build_spot_capital_budget_or_none = build_spot_capital_budget_or_none
_build_spot_capital_budget_or_none = build_spot_capital_budget_or_none
_spot_peer_sims = spot_peer_sims
_portfolio_spot_accum_deployed_quote_usd = portfolio_spot_accum_deployed_quote_usd
_utc_calendar_day_str = utc_calendar_day_str
_spot_entry_fill_price = spot_entry_fill_price
_spot_symbol_deploy_legs_today = spot_symbol_deploy_legs_today
_spot_regime_leg_kwargs = spot_regime_leg_kwargs
_spot_regime_unit_multiplier = spot_regime_unit_multiplier
_record_spot_symbol_deploy_leg = record_spot_symbol_deploy_leg
_mother_spot_deployed_usd_on_symbol = mother_spot_deployed_usd_on_symbol
_allocate_spot_accum_leg = allocate_spot_accum_leg
