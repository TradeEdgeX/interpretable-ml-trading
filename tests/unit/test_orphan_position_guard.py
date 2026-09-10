"""Unit tests for orphan_position_guard detection logic."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest


class _FakeAPI:
    """Minimal fake that returns positions / orders from dicts."""

    def __init__(
        self, positions: List[Dict], open_orders: List[Dict], algo_orders: List[Dict]
    ):
        self._positions = positions
        self._open_orders = open_orders
        self._algo_orders = algo_orders

    def _fapi_signed_get(self, path: str, params: Dict[str, Any] | None = None) -> Any:
        if "openAlgoOrders" in path:
            return self._algo_orders
        if "openOrders" in path:
            return self._open_orders
        return {"positions": self._positions}


def _write_state(state_dir: Path, filename: str, active: bool, inventory: List[Dict]):
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / filename).write_text(
        json.dumps({"active": active, "inventory": inventory})
    )


# ── detect_orphans ───────────────────────────────────────────────────────────


def test_no_positions_returns_empty(tmp_path: Path) -> None:
    from scripts.monitoring.orphan_position_guard import detect_orphans

    api = _FakeAPI([], [], [])
    orphans = detect_orphans(api=api, state_dir=tmp_path)
    assert orphans == []


def test_position_covered_by_active_inventory(tmp_path: Path) -> None:
    from scripts.monitoring.orphan_position_guard import detect_orphans

    _write_state(
        tmp_path,
        "chop_grid_BTCUSDT.json",
        active=True,
        inventory=[{"symbol": "BTCUSDT", "side": "SHORT", "quantity": 0.071}],
    )
    api = _FakeAPI(
        [
            {
                "symbol": "BTCUSDT",
                "positionAmt": -0.071,
                "entryPrice": 62880.6,
                "markPrice": 60767.4,
                "unrealizedProfit": 150.0,
            }
        ],
        [],
        [],
    )
    orphans = detect_orphans(api=api, state_dir=tmp_path)
    assert orphans == []


def test_orphan_when_no_inventory(tmp_path: Path) -> None:
    from scripts.monitoring.orphan_position_guard import detect_orphans

    # No state files at all
    api = _FakeAPI(
        [
            {
                "symbol": "BTCUSDT",
                "positionAmt": -0.071,
                "entryPrice": 62880.6,
                "markPrice": 60767.4,
                "unrealizedProfit": 150.0,
            }
        ],
        [],
        [],
    )
    orphans = detect_orphans(api=api, state_dir=tmp_path)
    assert len(orphans) == 1
    o = orphans[0]
    assert o["symbol"] == "BTCUSDT"
    assert o["side"] == "SHORT"
    assert o["quantity"] == 0.071
    assert o["gap"] == 0.071
    assert o["all_engines_inactive"] is True
    assert o["has_tp"] is False
    assert o["has_sl"] is False


def test_orphan_when_engine_inactive(tmp_path: Path) -> None:
    from scripts.monitoring.orphan_position_guard import detect_orphans

    _write_state(
        tmp_path,
        "chop_grid_BTCUSDT.json",
        active=False,
        inventory=[{"symbol": "BTCUSDT", "side": "SHORT", "quantity": 0.071}],
    )
    api = _FakeAPI(
        [
            {
                "symbol": "BTCUSDT",
                "positionAmt": -0.071,
                "entryPrice": 62880.6,
                "markPrice": 60767.4,
                "unrealizedProfit": 150.0,
            }
        ],
        [],
        [],
    )
    orphans = detect_orphans(api=api, state_dir=tmp_path)
    assert len(orphans) == 1
    assert orphans[0]["all_engines_inactive"] is True


def test_orphan_has_tp_detected(tmp_path: Path) -> None:
    from scripts.monitoring.orphan_position_guard import detect_orphans

    # SHORT position → TP close order is BUY reduce-only
    api = _FakeAPI(
        [
            {
                "symbol": "BTCUSDT",
                "positionAmt": -0.071,
                "entryPrice": 62880.6,
                "markPrice": 60767.4,
                "unrealizedProfit": 150.0,
            }
        ],
        open_orders=[
            {
                "symbol": "BTCUSDT",
                "side": "BUY",
                "type": "LIMIT",
                "origQty": 0.071,
                "reduceOnly": True,
                "price": 62000.0,
            },
        ],
        algo_orders=[],
    )
    orphans = detect_orphans(api=api, state_dir=tmp_path)
    assert len(orphans) == 1
    assert orphans[0]["has_tp"] is True
    assert orphans[0]["has_sl"] is False


def test_orphan_has_sl_detected_via_algo(tmp_path: Path) -> None:
    from scripts.monitoring.orphan_position_guard import detect_orphans

    # SHORT position → SL close order is BUY stop
    api = _FakeAPI(
        [
            {
                "symbol": "SOLUSDT",
                "positionAmt": -64.62,
                "entryPrice": 69.98,
                "markPrice": 67.67,
                "unrealizedProfit": 140.0,
            }
        ],
        open_orders=[],
        algo_orders=[
            {
                "symbol": "SOLUSDT",
                "side": "BUY",
                "orderType": "STOP_MARKET",
                "quantity": 64.62,
                "stopPrice": 72.0,
            },
        ],
    )
    orphans = detect_orphans(api=api, state_dir=tmp_path)
    assert len(orphans) == 1
    assert orphans[0]["has_tp"] is False
    assert orphans[0]["has_sl"] is True


def test_tp_wrong_side_not_counted(tmp_path: Path) -> None:
    """SELL reduce-only order should NOT count as protection for a SHORT position."""
    from scripts.monitoring.orphan_position_guard import detect_orphans

    api = _FakeAPI(
        [
            {
                "symbol": "BTCUSDT",
                "positionAmt": -0.071,
                "entryPrice": 62880.6,
                "markPrice": 60767.4,
                "unrealizedProfit": 150.0,
            }
        ],
        open_orders=[
            {
                "symbol": "BTCUSDT",
                "side": "SELL",
                "type": "LIMIT",
                "origQty": 0.071,
                "reduceOnly": True,
                "price": 59000.0,
            },
        ],
        algo_orders=[],
    )
    orphans = detect_orphans(api=api, state_dir=tmp_path)
    assert len(orphans) == 1
    # SELL reduce-only is for closing LONG, not SHORT
    assert orphans[0]["has_tp"] is False


def test_reduce_only_string_true_handled(tmp_path: Path) -> None:
    """Binance sometimes returns reduceOnly as string 'true'."""
    from scripts.monitoring.orphan_position_guard import detect_orphans

    api = _FakeAPI(
        [
            {
                "symbol": "BTCUSDT",
                "positionAmt": -0.071,
                "entryPrice": 62880.6,
                "markPrice": 60767.4,
                "unrealizedProfit": 150.0,
            }
        ],
        open_orders=[
            {
                "symbol": "BTCUSDT",
                "side": "BUY",
                "type": "LIMIT",
                "origQty": 0.071,
                "reduceOnly": "true",
                "price": 62000.0,
            },
        ],
        algo_orders=[],
    )
    orphans = detect_orphans(api=api, state_dir=tmp_path)
    assert len(orphans) == 1
    assert orphans[0]["has_tp"] is True


def test_partial_coverage_detected(tmp_path: Path) -> None:
    from scripts.monitoring.orphan_position_guard import detect_orphans

    # Active engine covers only part of exchange qty
    _write_state(
        tmp_path,
        "chop_grid_BTCUSDT.json",
        active=True,
        inventory=[{"symbol": "BTCUSDT", "side": "SHORT", "quantity": 0.03}],
    )
    api = _FakeAPI(
        [
            {
                "symbol": "BTCUSDT",
                "positionAmt": -0.071,
                "entryPrice": 62880.6,
                "markPrice": 60767.4,
                "unrealizedProfit": 150.0,
            }
        ],
        [],
        [],
    )
    orphans = detect_orphans(api=api, state_dir=tmp_path)
    assert len(orphans) == 1
    assert orphans[0]["gap"] == pytest.approx(0.041, rel=1e-3)
    assert orphans[0]["all_engines_inactive"] is False


def test_multiple_symbols_mixed(tmp_path: Path) -> None:
    from scripts.monitoring.orphan_position_guard import detect_orphans

    _write_state(
        tmp_path,
        "chop_grid_BTCUSDT.json",
        active=True,
        inventory=[{"symbol": "BTCUSDT", "side": "SHORT", "quantity": 0.071}],
    )
    api = _FakeAPI(
        [
            {
                "symbol": "BTCUSDT",
                "positionAmt": -0.071,
                "entryPrice": 62880.6,
                "markPrice": 60767.4,
                "unrealizedProfit": 150.0,
            },
            {
                "symbol": "HYPEUSDT",
                "positionAmt": -71.06,
                "entryPrice": 63.73,
                "markPrice": 63.05,
                "unrealizedProfit": 48.0,
            },
        ],
        [],
        [],
    )
    orphans = detect_orphans(api=api, state_dir=tmp_path)
    # BTC covered, HYPE orphan
    assert len(orphans) == 1
    assert orphans[0]["symbol"] == "HYPEUSDT"


def test_other_engine_covers(tmp_path: Path) -> None:
    from scripts.monitoring.orphan_position_guard import detect_orphans

    # trend_scalp covers the position, chop_grid is inactive
    _write_state(
        tmp_path,
        "chop_grid_BTCUSDT.json",
        active=False,
        inventory=[{"symbol": "BTCUSDT", "side": "SHORT", "quantity": 0.071}],
    )
    _write_state(
        tmp_path,
        "trend_scalp_BTCUSDT.json",
        active=True,
        inventory=[{"symbol": "BTCUSDT", "side": "SHORT", "quantity": 0.071}],
    )
    api = _FakeAPI(
        [
            {
                "symbol": "BTCUSDT",
                "positionAmt": -0.071,
                "entryPrice": 62880.6,
                "markPrice": 60767.4,
                "unrealizedProfit": 150.0,
            }
        ],
        [],
        [],
    )
    orphans = detect_orphans(api=api, state_dir=tmp_path)
    assert orphans == []


# ── format_report ────────────────────────────────────────────────────────────


def test_format_report_includes_protection_status() -> None:
    from scripts.monitoring.orphan_position_guard import format_report

    orphans = [
        {
            "symbol": "BTCUSDT",
            "side": "SHORT",
            "quantity": 0.071,
            "entry_price": 62880.6,
            "mark_price": 60767.4,
            "unrealized_pnl": 150.0,
            "local_qty": 0.0,
            "gap": 0.071,
            "all_engines_inactive": True,
            "has_tp": True,
            "has_sl": False,
        }
    ]
    console, tg = format_report(orphans)
    assert "TP" in console
    assert "BTCUSDT" in console
    assert "BTCUSDT" in tg
    assert "150.0" in console or "150.00" in console


# ── cooldown ─────────────────────────────────────────────────────────────────


def test_cooldown_prevents_alert(tmp_path: Path, monkeypatch) -> None:
    from scripts.monitoring.orphan_position_guard import _cooldown_ok, _touch_cooldown

    monkeypatch.setenv("ORPHAN_GUARD_COOLDOWN_FILE", str(tmp_path / "cooldown.txt"))
    monkeypatch.setenv("ORPHAN_GUARD_COOLDOWN_S", "3600")  # 1 hour

    # First call: no file → ok
    assert _cooldown_ok(force=False) is True

    # Touch cooldown
    _touch_cooldown()

    # Second call within cooldown → blocked
    assert _cooldown_ok(force=False) is False

    # --force bypasses
    assert _cooldown_ok(force=True) is True


def test_detect_orphans_dapi_normalizes_signal_symbol(tmp_path) -> None:
    from scripts.monitoring.orphan_position_guard import detect_orphans_dapi

    api = MagicMock()
    api.get_position_risk.return_value = [
        {
            "symbol": "SOLUSD_PERP",
            "positionAmt": "2",
            "entryPrice": "150",
            "markPrice": "155",
            "unRealizedProfit": "0.01",
        }
    ]
    api.get_open_orders.return_value = []
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    orphans = detect_orphans_dapi(api=api, state_dir=state_dir)
    assert len(orphans) == 1
    assert orphans[0]["symbol"] == "SOLUSDT"
