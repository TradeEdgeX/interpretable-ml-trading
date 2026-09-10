from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass(frozen=True)
class RevivePhase:
    size_multiplier: float
    duration_bars: int


@dataclass(frozen=True)
class OODConfig:
    version: int
    name: str

    ood_horizon_bars: int
    survival_horizon_bars: int

    y_ood_or_sources: List[str]

    ood_degrade_ge: float
    ood_halt_ge: float
    survival_degrade_le: float
    survival_halt_le: float

    ood_revive_le: float
    survival_revive_ge: float
    revive_phases: Dict[str, RevivePhase]

    use_power_formula: bool
    survival_power: float
    ood_power: float
    min_cap: float
    max_cap: float

    dashboard_keys: List[str]


def _f(x: Any, default: float) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _i(x: Any, default: int) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)


def _default_ood_config() -> OODConfig:
    """In-code default when config/ood was removed. Safety is handled by constitution/slots only."""
    return OODConfig(
        version=1,
        name="ood_config_default",
        ood_horizon_bars=20,
        survival_horizon_bars=50,
        y_ood_or_sources=[],
        ood_degrade_ge=0.6,
        ood_halt_ge=0.8,
        survival_degrade_le=0.4,
        survival_halt_le=0.25,
        ood_revive_le=0.35,
        survival_revive_ge=0.6,
        revive_phases={},
        use_power_formula=True,
        survival_power=2.0,
        ood_power=2.0,
        min_cap=0.0,
        max_cap=1.0,
        dashboard_keys=[
            "ood_score",
            "top_archetype_survival_prob",
            "active_archetype",
            "size_cap",
            "kill_switch_state",
        ],
    )


def load_ood_config(
    path: str | Path = "config/ood/ood_config.yaml",
) -> OODConfig:
    p = Path(path)
    if not p.exists():
        return _default_ood_config()

    obj = yaml.safe_load(p.read_text(encoding="utf-8")) or {}

    labels = obj.get("labels") or {}
    agg = obj.get("aggregation") or {}
    thr = obj.get("thresholds") or {}
    rev = obj.get("revive") or {}
    sc = obj.get("size_cap") or {}
    dash = obj.get("dashboard") or {}

    phases_obj = (rev.get("phases") or {}) if isinstance(rev, dict) else {}
    phases: Dict[str, RevivePhase] = {}
    if isinstance(phases_obj, dict):
        for k, v in phases_obj.items():
            if not isinstance(v, dict):
                continue
            phases[str(k)] = RevivePhase(
                size_multiplier=_f(v.get("size_multiplier"), 0.0),
                duration_bars=_i(v.get("duration_bars"), 0),
            )

    keys = []
    if isinstance(dash, dict):
        xs = dash.get("keys") or []
        if isinstance(xs, list):
            keys = [str(x) for x in xs]
    if not keys:
        keys = list(_default_ood_config().dashboard_keys)

    return OODConfig(
        version=int(obj.get("version", 1)),
        name=str(obj.get("name", "ood_config")),
        ood_horizon_bars=_i(labels.get("ood_horizon_bars"), 20),
        survival_horizon_bars=_i(labels.get("survival_horizon_bars"), 50),
        y_ood_or_sources=[str(x) for x in (agg.get("y_ood_or_sources") or [])],
        ood_degrade_ge=_f(thr.get("ood_degrade_ge"), 0.6),
        ood_halt_ge=_f(thr.get("ood_halt_ge"), 0.8),
        survival_degrade_le=_f(thr.get("survival_degrade_le"), 0.4),
        survival_halt_le=_f(thr.get("survival_halt_le"), 0.25),
        ood_revive_le=_f(rev.get("ood_revive_le"), 0.35),
        survival_revive_ge=_f(rev.get("survival_revive_ge"), 0.6),
        revive_phases=phases,
        use_power_formula=bool(sc.get("use_power_formula", True)),
        survival_power=_f(sc.get("survival_power"), 2.0),
        ood_power=_f(sc.get("ood_power"), 2.0),
        min_cap=_f(sc.get("min_cap"), 0.0),
        max_cap=_f(sc.get("max_cap"), 1.0),
        dashboard_keys=keys,
    )


def compute_size_cap_multiplier(
    *, cfg: OODConfig, ood_score: float, survival_prob: float
) -> float:
    """
    Deterministic mapping used by both research and live:
      cap = clamp( (survival_prob^a) * ((1-ood_score)^b), [min_cap, max_cap] )
    """
    o = float(ood_score)
    s = float(survival_prob)
    if not cfg.use_power_formula:
        # Fallback: linear (still monotone)
        cap = max(0.0, min(1.0, s)) * max(0.0, min(1.0, 1.0 - o))
    else:
        cap = (max(0.0, min(1.0, s)) ** float(cfg.survival_power)) * (
            max(0.0, min(1.0, 1.0 - o)) ** float(cfg.ood_power)
        )
    cap = max(float(cfg.min_cap), min(float(cfg.max_cap), float(cap)))
    return float(cap)
