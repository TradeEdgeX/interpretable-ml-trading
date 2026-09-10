from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple


_ALLOWED_PPATH_USAGES = {"rotation", "add_on", "risk_release"}


def assert_ppath_usage(usage: str) -> None:
    """
    Guardrail: ppath is only allowed for rotation/add-on/risk-release decisions.
    """
    u = str(usage or "").strip().lower()
    if u not in _ALLOWED_PPATH_USAGES:
        raise ValueError(
            f"ppath_usage_not_allowed={u}; allowed={sorted(_ALLOWED_PPATH_USAGES)}"
        )


@dataclass(frozen=True)
class PPathConfig:
    target_mfe_atr: float = 2.0
    target_pnl_r: float = 1.0
    target_duration_bars: int = 48
    weight_mfe: float = 0.4
    weight_pnl: float = 0.3
    weight_time: float = 0.2
    weight_structure: float = 0.1
    max_ppath: float = 1.0
    structure_keys: Tuple[str, ...] = (
        "breakout_confirmed",
        "pullback_confirmed",
        "absorption_confirmed",
        "retest_confirmed",
    )


@dataclass(frozen=True)
class PositionState:
    """
    Minimal ex-post state for ppath computation.
    """

    realized_mfe_atr: float = 0.0
    realized_pnl_r: float = 0.0
    floating_pnl_r: float = 0.0
    bars_held: int = 0
    structure_flags: Optional[Dict[str, bool]] = None


def _clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, float(x))))


def _structure_score(flags: Optional[Dict[str, bool]], *, keys: Iterable[str]) -> float:
    if not flags:
        return 0.0
    ks = [str(k) for k in keys]
    if not ks:
        return 0.0
    hits = [1.0 if bool(flags.get(k, False)) else 0.0 for k in ks]
    return float(sum(hits) / float(len(ks)))


def compute_ppath(state: PositionState, *, cfg: PPathConfig = PPathConfig()) -> float:
    """
    ppath = ex-post progress only; no predictive fields.
    """
    mfe = _clamp01(state.realized_mfe_atr / max(1e-9, float(cfg.target_mfe_atr)))
    pnl = _clamp01(
        (state.realized_pnl_r + state.floating_pnl_r)
        / max(1e-9, float(cfg.target_pnl_r))
    )
    time = _clamp01(float(state.bars_held) / max(1.0, float(cfg.target_duration_bars)))
    struct = _structure_score(state.structure_flags, keys=cfg.structure_keys)

    w_sum = float(
        cfg.weight_mfe + cfg.weight_pnl + cfg.weight_time + cfg.weight_structure
    )
    if w_sum <= 0:
        return 0.0
    raw = (
        cfg.weight_mfe * mfe
        + cfg.weight_pnl * pnl
        + cfg.weight_time * time
        + cfg.weight_structure * struct
    ) / w_sum
    return float(max(0.0, min(float(cfg.max_ppath), raw)))


def compute_remaining_ppath(ppath: float, *, cfg: PPathConfig = PPathConfig()) -> float:
    return float(max(0.0, float(cfg.max_ppath) - float(ppath)))


def compute_ppath_and_remaining(
    state: PositionState, *, cfg: PPathConfig = PPathConfig()
) -> Tuple[float, float]:
    p = compute_ppath(state, cfg=cfg)
    return p, compute_remaining_ppath(p, cfg=cfg)
