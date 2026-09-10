"""Stub: spot cycle machine is not used by ma_cross."""

from __future__ import annotations

from typing import Any, Mapping, Optional


class SpotCycleBook:
    def __init__(self, *, min_weeks_above: int = 4) -> None:
        self.phases: dict[str, str] = {}
        self.above_weeks: dict[str, list[str]] = {}
        self.block_until: dict[str, str] = {}
        self.winding_down: set[str] = set()
        self.min_weeks_above = max(1, int(min_weeks_above))

    def set_wind_down(self, symbol: Any, active: bool) -> None:
        return None


def maybe_spot_cycle_close(*_a: Any, **_k: Any) -> Optional[str]:
    return None


def apply_flatten_cooldown(*_a: Any, **_k: Any) -> None:
    return None


def sync_wind_down_buy_block(*_a: Any, **_k: Any) -> bool:
    return False


def attach_cycle_book(*_a: Any, **_k: Any) -> Any:
    return None


def iso_week_key(now: Any) -> str:
    return ""


def cycle_close_cfg(pos: Mapping[str, Any]) -> dict[str, Any]:
    return {}
