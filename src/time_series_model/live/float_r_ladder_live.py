"""Live float_r_ladder_only add path — parity with event_backtest/backtester.py.

Research-only triggers (e.g. rejected ``pullback_near_ema1200``) stay in
event_backtest / add_position_rules for experiment replay — **not** live.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import pandas as pd

from src.config.strategy_layout import resolve_strategy_package_under_root
from src.time_series_model.core.constitution.add_position_rules import (
    resolve_float_r_ladder_only,
)
from src.time_series_model.core.trade_intent import TradeIntent
from src.time_series_model.live.position_execution_sync import load_execution_raw
from src.time_series_model.live.srb_regime import resolve_srb_add_path

logger = logging.getLogger(__name__)


def load_float_ladder_meta(
    strategies_root: Path | str,
    archetypes: Iterable[str],
) -> Dict[str, Dict[str, Any]]:
    """Build per-archetype float ladder metadata (mirrors event_backtest setup)."""
    root = Path(strategies_root)
    out: Dict[str, Dict[str, Any]] = {}
    for arch in archetypes:
        key = str(arch or "").strip().lower()
        if not key or key in out:
            continue
        pkg = resolve_strategy_package_under_root(root, key, allow_bad_candidates=False)
        exec_path = pkg / "archetypes" / "execution.yaml"
        if not exec_path.is_file():
            continue
        raw = load_execution_raw(root, key)
        add_position = raw.get("add_position")
        if not isinstance(add_position, dict):
            continue
        if not resolve_float_r_ladder_only(add_position):
            continue
        out[key] = {
            "strategy": key,
            "add_position": dict(add_position),
            "execution_constraints": dict(raw.get("execution_constraints") or {}),
            "execution_raw": raw,
            "srb_add_position_policy": dict(raw.get("srb_add_position_policy") or {}),
        }
    return out


def float_ladder_archetypes(meta: Mapping[str, Mapping[str, Any]]) -> frozenset[str]:
    return frozenset(str(k).lower() for k in meta.keys())


def _minutes_since_iso_utc(iso_ref: Optional[str]) -> Optional[float]:
    if not iso_ref:
        return None
    try:
        ts = pd.Timestamp(iso_ref)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        now = pd.Timestamp.now(tz="UTC")
        return float((now - ts).total_seconds() / 60.0)
    except Exception:
        return None


def _position_symbol_matches(pos: Mapping[str, Any], symbol: str) -> bool:
    sym_u = str(symbol or "").upper().strip()
    raw = str(pos.get("symbol") or "").upper().strip()
    if not sym_u or not raw:
        return False
    if raw == sym_u:
        return True
    if raw.endswith("USDT") and raw[:-4] == sym_u.removesuffix("USDT"):
        return True
    if sym_u.endswith("USDT") and sym_u[:-4] == raw.removesuffix("USDT"):
        return True
    return False


def _last_add_timestamp(
    *,
    parent_pid: str,
    parent_pos: Mapping[str, Any],
    runtime_state: Any,
) -> Optional[str]:
    last = parent_pos.get("_last_float_ladder_add_ts")
    if last:
        return str(last)
    rec = None
    try:
        rec = runtime_state.add_position.positions.get(parent_pid)
    except Exception:
        rec = None
    if rec is not None:
        ref = getattr(rec, "last_add_at", None) or getattr(rec, "updated_at", None)
        if ref:
            return str(ref)
    return None


def attempt_float_ladder_adds(
    *,
    symbol: str,
    features: Mapping[str, Any],
    tracker: Any,
    executor: Any,
    float_ladder_meta: Mapping[str, Mapping[str, Any]],
    runtime_state: Any = None,
) -> List[str]:
    """Try float-R ladder adds for open mother positions (one add per archetype/bar)."""
    if not float_ladder_meta:
        return []
    runtime_state = runtime_state or getattr(executor, "runtime_state", None)
    positions = tracker.all_positions() if hasattr(tracker, "all_positions") else {}
    added: List[str] = []
    ts_now = features.get("timestamp")

    for arch_lc, meta in float_ladder_meta.items():
        ladder_done = False
        for pid, pos in list(positions.items()):
            if ladder_done:
                break
            if not isinstance(pos, dict):
                continue
            if bool(pos.get("_is_add_position", False)):
                continue
            if str(pos.get("archetype", "")).strip().lower() != arch_lc:
                continue
            if not _position_symbol_matches(pos, symbol):
                continue
            if (
                resolve_srb_add_path(
                    features,
                    meta.get("srb_add_position_policy"),
                    default_path="float_ladder",
                )
                != "float_ladder"
            ):
                continue

            min_gap_m = float(
                (meta.get("execution_constraints") or {}).get(
                    "min_order_interval_minutes", 0
                )
                or 0
            )
            if min_gap_m > 0:
                ref_iso = _last_add_timestamp(
                    parent_pid=str(pid),
                    parent_pos=pos,
                    runtime_state=runtime_state,
                )
                if ref_iso is not None:
                    elapsed = _minutes_since_iso_utc(ref_iso)
                    if elapsed is not None and elapsed < min_gap_m:
                        logger.debug(
                            "[%s] float_ladder skip %s: min_interval %.0fm (%.1fm elapsed)",
                            symbol,
                            arch_lc,
                            min_gap_m,
                            elapsed,
                        )
                        continue

            side = str(pos.get("side", "LONG")).upper()
            action = "LONG" if side in {"LONG", "BUY"} else "SHORT"
            intent = TradeIntent(
                action=action,
                symbol=str(pos.get("symbol") or symbol),
                archetype=str(pos.get("archetype") or arch_lc),
                execution_strategy=str(meta.get("strategy") or arch_lc),
                add_position=True,
                parent_position_id=str(pid),
                size_multiplier=float(pos.get("_size_multiplier", 1.0) or 1.0),
                execution_profile={
                    "add_position": dict(meta.get("add_position") or {})
                },
            )
            try:
                ok = bool(executor.execute(intent=intent, features=dict(features)))
            except Exception:
                logger.warning(
                    "[%s] float_ladder add failed for %s pid=%s",
                    symbol,
                    arch_lc,
                    pid,
                    exc_info=True,
                )
                ok = False
            if ok:
                positions = tracker.all_positions()
                mother = positions.get(pid) or pos
                if ts_now is not None:
                    mother["_last_float_ladder_add_ts"] = ts_now
                added.append(str(pid))
                ladder_done = True
                logger.info(
                    "[%s] float_ladder add ok archetype=%s parent=%s",
                    symbol,
                    arch_lc,
                    pid,
                )
    return added
