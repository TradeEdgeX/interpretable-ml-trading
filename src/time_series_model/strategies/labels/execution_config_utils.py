"""Normalize archetype execution.yaml dicts for simulate_rr / live paths."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[4]

# Canonical patches (match prepare_fast_scalp_alpha_snapshots.py)
EXEC_PROFILES: dict[str, dict[str, Any]] = {
    "g5_tight": {
        "stop_loss": {"initial_r": 1.5, "trailing": {"enabled": False}},
        "take_profit": {"enabled": True, "target_r": 1.0},
        "holding": {"max_holding_bars": 6, "time_stop_bars": 6},
    },
    "g10_wide_tight": {
        "stop_loss": {"initial_r": 8.0, "trailing": {"enabled": False}},
        "take_profit": {"enabled": True, "target_r": 0.12},
        "holding": {"max_holding_bars": 24, "time_stop_bars": 24},
    },
}


def normalize_take_profit_block(tp: dict[str, Any] | None) -> dict[str, Any]:
    """Map legacy ``r`` → ``target_r`` for take_profit blocks."""
    if not tp:
        return {}
    out = dict(tp)
    if "target_r" not in out and "r" in out:
        out["target_r"] = out.pop("r")
    return out


def normalize_execution_config(raw: dict[str, Any]) -> dict[str, Any]:
    """Deep-copy execution config with normalized take_profit keys."""
    cfg = copy.deepcopy(raw)
    tp = cfg.get("take_profit")
    if isinstance(tp, dict):
        cfg["take_profit"] = normalize_take_profit_block(tp)
    return cfg


def load_exec_profile(
    *,
    exec_profile: str | None = None,
    execution_yaml: str | Path | None = None,
    strategies_root: str | Path = "config/strategies/tree_strategies",
    strategy: str = "fast_scalp",
) -> dict[str, Any]:
    """Load execution config from named profile or yaml path."""
    if exec_profile:
        key = exec_profile.strip().lower()
        if key not in EXEC_PROFILES:
            raise ValueError(
                f"unknown exec_profile {exec_profile!r}; have {sorted(EXEC_PROFILES)}"
            )
        return normalize_execution_config(EXEC_PROFILES[key])
    if execution_yaml:
        path = Path(execution_yaml)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return normalize_execution_config(raw)
    path = Path(strategies_root) / strategy / "archetypes" / "execution.yaml"
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return normalize_execution_config(raw)
