"""Sync open-position execution state from archetype execution.yaml.

Keeps live PositionTracker fields (breakeven_measure, trailing, etc.) aligned
with backtest ``build_position_dict`` / ``ExecutionParamGenerator`` output.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import yaml

from src.config.strategy_layout import resolve_strategy_package_under_root
from src.time_series_model.live.execution_profile_apply import (
    rr_constraints_from_exec_params,
)
from src.time_series_model.live.generic_live_strategy import ExecutionParamGenerator

logger = logging.getLogger(__name__)

_EXECUTION_FIELD_KEYS = (
    "breakeven_enabled",
    "breakeven_trigger_r",
    "breakeven_lock_level_r",
    "breakeven_measure",
    "allow_trailing",
    "activation_r",
    "trail_r",
    "trail_r_far",
    "trail_r_near",
    "l3_near_threshold_atr",
    "trail_expand_primary_atr",
    "structural_exit",
    "regime_lifecycle_exit",
    "max_holding_bars",
    "time_stop_uncap_mfe_r",
    "l3_structural_exit_enabled",
    "l3_structural_exit_buffer_atr",
)


def execution_yaml_path(strategies_root: Path | str, archetype: str) -> Optional[Path]:
    root = Path(strategies_root)
    pkg = resolve_strategy_package_under_root(
        root, str(archetype or "").strip().lower(), allow_bad_candidates=False
    )
    path = pkg / "archetypes" / "execution.yaml"
    return path if path.is_file() else None


def load_execution_raw(strategies_root: Path | str, archetype: str) -> Dict[str, Any]:
    path = execution_yaml_path(strategies_root, archetype)
    if path is None:
        return {}
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        logger.warning("failed to read execution.yaml for %s", archetype, exc_info=True)
        return {}
    return dict(raw) if isinstance(raw, dict) else {}


def load_execution_raw_by_archetype(
    strategies_root: Path | str,
    archetypes: Iterable[str],
) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for arch in archetypes:
        key = str(arch or "").strip().lower()
        if not key or key in out:
            continue
        raw = load_execution_raw(strategies_root, key)
        if raw:
            out[key] = raw
    return out


def _rr_fields_from_execution(
    execution_raw: Mapping[str, Any],
    *,
    features: Optional[Mapping[str, Any]] = None,
    regime_label: str = "neutral",
) -> Dict[str, Any]:
    gen = ExecutionParamGenerator(dict(execution_raw))
    params = gen.generate_params(
        0.5,
        features=dict(features or {}),
        regime_label=regime_label,
    )
    return rr_constraints_from_exec_params(params)


def apply_execution_state_to_position(
    pos: Dict[str, Any],
    execution_raw: Mapping[str, Any],
    *,
    features: Optional[Mapping[str, Any]] = None,
    regime_label: str = "neutral",
    force_measure: bool = False,
) -> bool:
    """Fill missing (or wrong-default) execution fields on an open position."""
    if not execution_raw or not isinstance(pos, dict):
        return False
    rr = _rr_fields_from_execution(
        execution_raw, features=features, regime_label=regime_label
    )
    modified = False
    for key in _EXECUTION_FIELD_KEYS:
        val = rr.get(key)
        if val is None:
            continue
        if pos.get(key) is None:
            pos[key] = val
            modified = True
    cfg_measure = rr.get("breakeven_measure")
    if (
        force_measure
        and cfg_measure
        and str(pos.get("breakeven_measure") or "initial_risk").strip().lower()
        != str(cfg_measure).strip().lower()
    ):
        pos["breakeven_measure"] = cfg_measure
        modified = True
    if pos.get("breakeven_enabled") and not pos.get("breakeven_locked"):
        if pos.get("low_water_mark") is None and str(pos.get("side", "")).upper() in {
            "SHORT",
            "SELL",
        }:
            pos["low_water_mark"] = float(pos.get("entry_price") or 0.0)
            modified = True
        if pos.get("high_water_mark") is None and str(pos.get("side", "")).upper() in {
            "LONG",
            "BUY",
        }:
            pos["high_water_mark"] = float(pos.get("entry_price") or 0.0)
            modified = True
    return modified


def sync_open_positions_execution_state(
    positions: Mapping[str, Dict[str, Any]],
    execution_raw_by_archetype: Mapping[str, Mapping[str, Any]],
    *,
    features: Optional[Mapping[str, Any]] = None,
    force_measure: bool = True,
) -> int:
    """Apply execution.yaml fields to all open positions; returns change count."""
    changed = 0
    for pos in positions.values():
        if not isinstance(pos, dict):
            continue
        arch = str(pos.get("archetype") or "").strip().lower()
        raw = execution_raw_by_archetype.get(arch)
        if not raw:
            continue
        if apply_execution_state_to_position(
            pos,
            raw,
            features=features,
            force_measure=force_measure,
        ):
            changed += 1
    return changed
