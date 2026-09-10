"""Stub: exogenous spot clock. ma_cross never arms it."""

from __future__ import annotations

from typing import Any


def combine_any_armed(*_a: Any, **_k: Any) -> bool:
    return False


def load_exogenous_clock(*_a: Any, **_k: Any) -> None:
    return None


def mother_lot_mtm(*_a: Any, **_k: Any) -> float:
    return 0.0


def mother_lot_qty_cost(*_a: Any, **_k: Any) -> tuple[float, float]:
    return 0.0, 0.0


def refresh_clock(*_a: Any, **_k: Any) -> None:
    return None


def refresh_quorum_clock(*_a: Any, **_k: Any) -> None:
    return None


def refresh_sleeve_clock(*_a: Any, **_k: Any) -> None:
    return None
