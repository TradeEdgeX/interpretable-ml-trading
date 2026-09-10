"""Entry-filter threshold scan with bar-by-bar RR execution (snotio KPI)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from scripts.backtest_execution_layer import (
    apply_direction_rules,
    load_direction_config,
    load_execution_config,
    simulate_rr_execution,
)
from src.research.expr import OPS
from src.research.stat_kernels.snotio_calc import compute_snotio

_OHLC_ATR = ("high", "low", "close", "atr")
_DIRECTION_FALLBACK_COLS = (
    "entry_direction",
    "bpc_breakout_direction",
    "tpc_breakout_direction",
)


def _direction_fallback_cols(strategy: str) -> tuple[str, ...]:
    """Prefer strategy-specific breakout direction before generic fallbacks."""
    strat = (strategy or "").lower()
    preferred = f"{strat}_breakout_direction"
    ordered: list[str] = []
    for col in (preferred, "entry_direction", "bpc_breakout_direction", "tpc_breakout_direction"):
        if col not in ordered:
            ordered.append(col)
    return tuple(ordered)


def entry_rr_requirements(df: pd.DataFrame) -> List[str]:
    """Columns missing for entry RR simulation."""
    missing = [c for c in _OHLC_ATR if c not in df.columns]
    if not any(c in df.columns for c in _DIRECTION_FALLBACK_COLS):
        missing.append("entry_direction")
    return missing


def prepare_entry_rr_frame(
    df: pd.DataFrame,
    strategy: str,
    *,
    strategies_root: str = "config/strategies",
    apply_gate: bool = True,
) -> pd.DataFrame:
    """Prepare dataframe for entry RR threshold scans (direction + optional gate)."""
    missing = entry_rr_requirements(df)
    if missing:
        raise ValueError(f"entry RR sim missing columns: {missing}")

    if "symbol" in df.columns:
        merged = df.sort_values(["symbol"]).reset_index(drop=True)
    else:
        merged = df.reset_index(drop=True).copy()

    if "entry_direction" not in merged.columns:
        for col in _direction_fallback_cols(strategy):
            if col == "entry_direction":
                continue
            if col in merged.columns:
                merged = merged.copy()
                merged["entry_direction"] = pd.to_numeric(
                    merged[col], errors="coerce"
                ).fillna(0.0)
                break
        if "entry_direction" not in merged.columns:
            dir_cfg = load_direction_config(strategy, strategies_root)
            if dir_cfg:
                apply_direction_rules(merged, strategy, dir_cfg)
        if "entry_direction" not in merged.columns:
            raise ValueError(
                "could not derive entry_direction from direction rules or breakout columns"
            )

    merged = merged.copy()
    merged["entry_direction"] = pd.to_numeric(
        merged["entry_direction"], errors="coerce"
    ).fillna(0.0)

    if apply_gate:
        if "gate_decision" in merged.columns:
            merged.loc[merged["gate_decision"].astype(str).str.lower() != "allow", "entry_direction"] = 0.0
        elif "gate_ok" in merged.columns:
            merged.loc[~merged["gate_ok"].astype(bool), "entry_direction"] = 0.0

    if int((merged["entry_direction"] != 0).sum()) == 0:
        raise ValueError("no entry signals after gate/direction prep")

    return merged


def scan_snotio_entry_rr_thresholds(
    prepared: pd.DataFrame,
    feature: str,
    operator: str,
    grid: List[float],
    base_mask: pd.Series,
    exec_config: Dict[str, Any],
    *,
    min_trades: int = 20,
    atr_col: str = "atr",
    direction_col: str = "entry_direction",
) -> List[Dict[str, Any]]:
    """Scan thresholds with simulate_rr_execution (full entry RR path)."""
    if feature not in prepared.columns:
        raise KeyError(f"Feature missing: {feature}")
    op_fn = OPS.get(operator)
    if op_fn is None:
        raise ValueError(f"Unsupported operator: {operator!r}")

    feat_s = pd.to_numeric(prepared[feature], errors="coerce")
    baseline_dir = prepared[direction_col].copy()
    mask = base_mask.reindex(prepared.index).fillna(False)
    results: List[Dict[str, Any]] = []

    for thr in grid:
        work = prepared.copy()
        hit = op_fn(feat_s, thr).fillna(False) & mask
        work[direction_col] = baseline_dir.copy()
        work.loc[~hit, direction_col] = 0.0
        n_entries = int((work[direction_col] != 0).sum())
        if n_entries < min_trades:
            results.append(
                {
                    "threshold": float(thr),
                    "trades": n_entries,
                    "snotio": 0.0,
                    "too_few": True,
                    "sim": "entry_rr",
                }
            )
            continue

        raw = simulate_rr_execution(
            work,
            exec_config,
            atr_col=atr_col,
            direction_col=direction_col,
            use_tier_params=False,
        )
        exec_returns = raw[0] if isinstance(raw, tuple) else raw
        valid = exec_returns.dropna()
        if len(valid) < min_trades:
            results.append(
                {
                    "threshold": float(thr),
                    "trades": len(valid),
                    "snotio": 0.0,
                    "too_few": True,
                    "sim": "entry_rr",
                }
            )
            continue

        snotio_val = compute_snotio(valid)
        results.append(
            {
                "threshold": float(thr),
                "trades": int(len(valid)),
                "snotio": snotio_val,
                "too_few": False,
                "sim": "entry_rr",
            }
        )

    return results


def load_strategy_exec_config(
    strategy: str,
    *,
    strategies_root: str = "config/strategies",
    simple: bool = False,
) -> Dict[str, Any]:
    if simple:
        return {
            "stop_loss": {"type": "fixed", "initial_r": 1.5},
            "take_profit": {"enabled": True, "target_r": 3.0},
            "holding": {"max_holding_bars": 50, "time_stop_bars": 50},
        }
    return load_execution_config(strategy, strategies_root)
