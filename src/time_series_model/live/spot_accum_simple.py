"""Stub: spot accumulation is not a public family. ma_cross never hits these."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple

SPOT_ACCUM_ARCHETYPES = frozenset({"spot_accum", "spot_accum_simple"})


def is_spot_accum_archetype(name: str) -> bool:
    return str(name or "").strip().lower() in SPOT_ACCUM_ARCHETYPES


def maybe_spot_regime_exit(*_a: Any, **_k: Any) -> Optional[str]:
    return None


def maybe_spot_simple_partial_sell(*_a: Any, **_k: Any) -> Optional[Any]:
    return None


def deploy_decay_multiplier(*_a: Any, **_k: Any) -> float:
    return 1.0


def apply_partial_sell_to_position(*_a: Any, **_k: Any) -> None:
    return None


def simple_accumulation_policy(raw_execution: Mapping[str, Any]) -> Dict[str, Any]:
    policy = raw_execution.get("simple_accumulation_policy") or {}
    return dict(policy) if isinstance(policy, dict) else {}


def deploy_schedule_policy(raw_execution: Mapping[str, Any]) -> Dict[str, Any]:
    policy = raw_execution.get("deploy_schedule") or {}
    return dict(policy) if isinstance(policy, dict) else {}


def deploy_schedule_allows_new_buy(
    *_a: Any, **_k: Any
) -> Tuple[bool, str]:
    return True, ""
