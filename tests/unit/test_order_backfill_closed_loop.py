"""Regression: trend OrderManager/storage backfill + multi-leg order row updates."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import Mock

import pytest

from src.order_management.models import Order, OrderSide, OrderStatus, OrderType
from src.order_management.order_manager import OrderManager
from src.order_management.storage import Storage
from src.order_management.multi_leg_storage import MultiLegStorage


def test_storage_recent_orders_for_backfill_includes_gap_rows(tmp_path) -> None:
    db_path = tmp_path / "om.db"
    storage = Storage(str(db_path))

    def _insert_gap_filled(order_id_suffix: str) -> None:
        o = Order(
            order_id=f"order_{order_id_suffix}",
            binance_order_id=f"ex_{order_id_suffix}",
            client_order_id=f"cl_{order_id_suffix}",
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=1.0,
            status=OrderStatus.FILLED,
            filled_quantity=1.0,
            average_price=None,
            filled_at=None,
            commission=0.0,
            created_at=datetime.now(),
            updated_at=None,
            error_message=None,
        )
        assert storage.create_order(o)

    _insert_gap_filled("gap1")
    _insert_gap_filled("gap2")
    cand = storage.get_recent_orders_for_backfill(lookback_hours=24, limit=50)
    assert len(cand) >= 2
    assert all(x.status == OrderStatus.FILLED for x in cand)
    assert all(x.binance_order_id for x in cand)


def test_order_manager_reconcile_recent_terminal_orders_updates(tmp_path) -> None:
    db_path = tmp_path / "om.db"
    storage = Storage(str(db_path))
    o = Order(
        order_id="order_u1",
        binance_order_id="888",
        client_order_id="tl_testcid",
        symbol="ETHUSDT",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=1.5,
        status=OrderStatus.FILLED,
        filled_quantity=1.5,
        average_price=None,
        filled_at=None,
        commission=0.0,
        created_at=datetime.now(),
    )
    assert storage.create_order(o)

    snap = {
        "status": "filled",
        "filled": 1.5,
        "average_price": 3210.5,
        "timestamp": 1_717_776_000.0,
    }

    api = Mock()
    api.get_order = Mock(return_value=snap)

    om = OrderManager(storage, api, shadow=False)
    om.storage.get_recent_orders_for_backfill = (  # type: ignore[method-assign]
        lambda **kw: [storage.get_order("order_u1")]
    )
    updated = om.reconcile_recent_terminal_orders(lookback_hours=1, limit=10)
    assert len(updated) == 1
    again = storage.get_order("order_u1")
    assert again is not None
    assert again.average_price == pytest.approx(3210.5)
    assert again.filled_at is not None


def test_reconcile_recent_terminal_orders_keeps_algo_stop_open(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MLBOT_TERMINAL_STALE_OPEN_GRACE_SECONDS", "0")
    db_path = tmp_path / "om_algo.db"
    storage = Storage(str(db_path))
    o = Order(
        order_id="order_algo_1",
        binance_order_id="4000001352133210",
        client_order_id="tl_bea7e8e7234f452892b331faf013f43b",
        symbol="ETHUSDT",
        side=OrderSide.BUY,
        order_type=OrderType.STOP_MARKET,
        quantity=0.2,
        status=OrderStatus.PENDING,
        stop_price=2244.96,
        created_at=datetime.now(),
    )
    assert storage.create_order(o)

    api = Mock()
    api.get_order = Mock(return_value=None)
    api.get_open_orders_for_sl_cleanup = Mock(
        return_value=[
            {
                "order_id": "4000001352133210",
                "client_order_id": "tl_bea7e8e7234f452892b331faf013f43b",
                "symbol": "ETHUSDT",
            }
        ]
    )

    om = OrderManager(storage, api, shadow=False)
    om.storage.get_recent_orders_for_backfill = (  # type: ignore[method-assign]
        lambda **kw: [storage.get_order("order_algo_1")]
    )
    updated = om.reconcile_recent_terminal_orders(lookback_hours=1, limit=10)
    assert updated == []
    loaded = storage.get_order("order_algo_1")
    assert loaded is not None
    assert loaded.status == OrderStatus.PENDING


def test_reconcile_recent_terminal_orders_requires_confirm_n(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MLBOT_TERMINAL_STALE_OPEN_GRACE_SECONDS", "0")
    monkeypatch.setenv("MLBOT_MULTI_LEG_MISSING_ORDER_CONFIRM_CYCLES", "2")
    db_path = tmp_path / "om_confirm.db"
    storage = Storage(str(db_path))
    o = Order(
        order_id="order_confirm_1",
        binance_order_id="555444333",
        client_order_id="tl_confirm_1",
        symbol="ETHUSDT",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=0.1,
        status=OrderStatus.PENDING,
        created_at=datetime.now(),
    )
    assert storage.create_order(o)

    api = Mock()
    api.get_order = Mock(return_value=None)
    api.get_open_orders_for_sl_cleanup = Mock(return_value=[])

    om = OrderManager(storage, api, shadow=False)
    om.storage.get_recent_orders_for_backfill = (  # type: ignore[method-assign]
        lambda **kw: [storage.get_order("order_confirm_1")]
    )
    first = om.reconcile_recent_terminal_orders(lookback_hours=1, limit=10)
    assert first == []
    loaded = storage.get_order("order_confirm_1")
    assert loaded is not None
    assert loaded.status == OrderStatus.PENDING

    second = om.reconcile_recent_terminal_orders(lookback_hours=1, limit=10)
    assert len(second) == 1
    loaded = storage.get_order("order_confirm_1")
    assert loaded is not None
    assert loaded.status == OrderStatus.EXPIRED


def test_reconcile_skips_stale_when_open_orders_snapshot_fails(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MLBOT_TERMINAL_STALE_OPEN_GRACE_SECONDS", "0")
    db_path = tmp_path / "om_open_fail.db"
    storage = Storage(str(db_path))
    o = Order(
        order_id="order_open_fail",
        binance_order_id="999888777",
        client_order_id="tl_open_fail",
        symbol="ETHUSDT",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=0.1,
        status=OrderStatus.PENDING,
        created_at=datetime.now(),
    )
    assert storage.create_order(o)

    api = Mock()
    api.get_order = Mock(return_value=None)
    api.get_open_orders_for_sl_cleanup = Mock(
        side_effect=RuntimeError("open orders unavailable")
    )

    om = OrderManager(storage, api, shadow=False)
    om.storage.get_recent_orders_for_backfill = (  # type: ignore[method-assign]
        lambda **kw: [storage.get_order("order_open_fail")]
    )
    updated = om.reconcile_recent_terminal_orders(lookback_hours=1, limit=10)
    assert updated == []
    loaded = storage.get_order("order_open_fail")
    assert loaded is not None
    assert loaded.status == OrderStatus.PENDING
    stats = getattr(om, "_last_terminal_backfill_stats", {})
    assert int(stats.get("api_error", 0) or 0) >= 1
    assert int(stats.get("stale_marked", 0) or 0) == 0


def test_handle_execution_report_commissions_accumulate(tmp_path) -> None:
    db_path = tmp_path / "om2.db"
    storage = Storage(str(db_path))

    api = Mock()
    om = OrderManager(storage, api, shadow=False)

    placed = Order(
        order_id="order_tr1",
        binance_order_id="bin_1",
        client_order_id="cid_tr1",
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=10.0,
        price=100.0,
        status=OrderStatus.PARTIALLY_FILLED,
        filled_quantity=0.0,
        commission=0.0,
        created_at=datetime.now(),
    )
    storage.create_order(placed)

    om.handle_execution_report(
        {
            "order_id": "bin_1",
            "client_order_id": "cid_tr1",
            "symbol": "BTCUSDT",
            "status": "partially_filled",
            "filled_qty": 5.0,
            "avg_price": 100.5,
            "commission": 0.01,
            "commission_asset": "USDT",
            "trade_time": 1717776000,
        }
    )
    om.handle_execution_report(
        {
            "order_id": "bin_1",
            "client_order_id": "cid_tr1",
            "symbol": "BTCUSDT",
            "status": "filled",
            "filled_qty": 10.0,
            "avg_price": 100.6,
            "commission": 0.02,
            "commission_asset": "USDT",
            "trade_time": 1717776100,
        }
    )

    loaded = storage.get_order("order_tr1")
    assert loaded is not None
    assert loaded.commission == pytest.approx(0.03)
    assert loaded.status == OrderStatus.FILLED


def test_multi_leg_storage_apply_execution_report_updates_row(tmp_path) -> None:
    db_path = tmp_path / "multi.db"
    mls = MultiLegStorage(str(db_path))
    mls.upsert_order(
        {
            "local_order_id": "loc_a",
            "run_id": "run_x",
            "strategy": "chop_grid",
            "symbol": "BTCUSDT",
            "leg_id": "leg_a",
            "side": "BUY",
            "position_side": "LONG",
            "order_type": "limit",
            "purpose": "entry",
            "quantity": 0.05,
            "price": 100_000.0,
            "client_order_id": "cg_deadbeef",
            "exchange_order_id": "ex_4242",
            "status": "new",
            "raw": {"k": "v"},
        }
    )

    changed = mls.apply_execution_report(
        {
            "order_id": "ex_4242",
            "client_order_id": "cg_deadbeef",
            "strategy": "chop_grid",
            "symbol": "BTCUSDT",
            "status": "FILLED",
            "filled_qty": 0.05,
            "avg_price": 99999.0,
            "commission": 0.001,
            "commission_asset": "USDT",
            "trade_time": 1717776000,
            "reject_reason": None,
            "run_id": "run_x",
            "raw": {"e": 1},
        }
    )
    assert changed >= 1

    conn_row = mls._connect()
    try:
        cur = conn_row.execute(
            "SELECT status, filled_quantity, average_price, commission FROM "
            "multi_leg_orders WHERE local_order_id = ?",
            ("loc_a",),
        ).fetchone()
    finally:
        conn_row.close()
    assert cur is not None
    assert cur["status"] == "filled"
    assert pytest.approx(float(cur["filled_quantity"]), rel=1e-9) == 0.05
    assert pytest.approx(float(cur["average_price"]), rel=1e-9) == 99999.0
    assert pytest.approx(float(cur["commission"]), rel=1e-9) == 0.001
