"""Lean CI must import OrderFlowListener without research ML deps.

requirements-ci.txt intentionally omits sklearn / talib / pywt / scipy.
OMS and coin_m G0 tests import OrderFlowListener → IncrementalFeatureComputer;
those modules must not import the research stack at module load time.
"""

from __future__ import annotations

import sys

import pytest


@pytest.fixture
def block_research_stack(monkeypatch):
    blocked = {"sklearn", "talib", "pywt", "scipy"}

    class _Blocker:
        def find_spec(self, fullname, path, target=None):  # noqa: ANN001
            if fullname.split(".")[0] in blocked:
                raise ModuleNotFoundError(fullname)
            return None

    # Drop already-imported packages so the blocker is effective.
    for name in list(sys.modules):
        root = name.split(".")[0]
        if root in blocked or name.startswith(
            (
                "src.live_data_stream",
                "src.time_series_model.live",
                "src.features.time_series.ema_talib",
                "live_data_stream",
                "time_series_model.live",
                "features.time_series.ema_talib",
            )
        ):
            monkeypatch.delitem(sys.modules, name, raising=False)

    monkeypatch.setattr(
        sys,
        "meta_path",
        [_Blocker(), *sys.meta_path],
    )


def test_order_flow_listener_imports_without_research_stack(block_research_stack):
    from live_data_stream.order_flow_listener import OrderFlowListener
    from src.live_data_stream.multi_symbol_manager import MultiSymbolManager

    assert OrderFlowListener is not None
    assert MultiSymbolManager is not None


def test_rolling_live_imports_without_talib(block_research_stack):
    """Champion recompute is lazy; collection must not require TA-Lib."""
    from time_series_model.live import rolling_champion_levels
    from time_series_model.live.rolling_coin_features import (
        build_rolling_features_from_bus,
    )

    assert rolling_champion_levels.apply_recomputed_cross is not None
    assert build_rolling_features_from_bus is not None
