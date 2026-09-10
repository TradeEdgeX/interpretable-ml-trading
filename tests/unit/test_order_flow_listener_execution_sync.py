from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

from src.live_data_stream.order_flow_listener import OrderFlowListener
from src.live_data_stream.system_mode import SystemModeManager, SystemMode
from src.order_management.models import Order, OrderType, OrderStatus, OrderSide
from src.time_series_model.core.constitution.runtime_state import (
    ConstitutionRuntimeState,
    SlotRecord,
    AddPositionRecord,
)


def _make_listener():
    om = MagicMock()
    om.binance_api = MagicMock()
    ce = MagicMock()
    rs = ConstitutionRuntimeState()
    mm = SystemModeManager()
    listener = OrderFlowListener(
        symbol="BTCUSDT",
        storage_manager=MagicMock(),
        feature_computer=MagicMock(),
        constitution_executor=ce,
        runtime_state=rs,
        order_manager=om,
        mode_manager=mm,
    )
    return listener, om, ce, rs, mm


def test_exchange_stop_fill_closes_local_position_and_releases_slot():
    listener, om, ce, rs, mm = _make_listener()
    pid = "BTCUSDT:1"
    listener._position_tracker.add(
        pid,
        {
            "symbol": "BTCUSDT",
            "side": "LONG",
            "entry_time": datetime.now(timezone.utc) - timedelta(seconds=5),
            "qty": 0.01,
        },
    )
    rs.slots.active[pid] = SlotRecord(
        position_id=pid, symbol="BTCUSDT", archetype="bpc"
    )
    rs.add_position.positions[pid] = AddPositionRecord(position_id=pid, add_count=2)
    om.handle_execution_report.return_value = Order(
        order_id="o1",
        position_id=pid,
        symbol="BTCUSDT",
        side=OrderSide.SELL,
        order_type=OrderType.STOP_MARKET,
        status=OrderStatus.FILLED,
    )

    listener.on_execution_report(
        {"symbol": "BTCUSDT", "trade_time": int(datetime.now(timezone.utc).timestamp())}
    )

    assert listener._position_tracker.get(pid) is None
    ce.release_slot.assert_called()
    ce.save_runtime_state.assert_called()
    assert pid not in rs.add_position.positions
    assert mm.get_current_mode() == SystemMode.ABNORMAL


def test_execution_report_without_position_id_reconciles_when_exchange_flat():
    listener, om, ce, rs, _ = _make_listener()
    pid = "BTCUSDT:2"
    listener._position_tracker.add(
        pid,
        {
            "symbol": "BTCUSDT",
            "side": "LONG",
            "entry_time": datetime.now(timezone.utc) - timedelta(minutes=5),
            "qty": 0.01,
        },
    )
    rs.slots.active[pid] = SlotRecord(
        position_id=pid, symbol="BTCUSDT", archetype="bpc"
    )
    om.handle_execution_report.return_value = Order(
        order_id="o2",
        position_id=None,
        symbol="BTCUSDT",
        side=OrderSide.SELL,
        order_type=OrderType.STOP_MARKET,
        status=OrderStatus.FILLED,
    )
    om.binance_api.get_position.return_value = {"size": 0.0}

    listener.on_execution_report({"symbol": "BTCUSDT"})

    assert listener._position_tracker.get(pid) is None
    ce.release_slot.assert_called()


def test_account_update_zero_position_does_not_force_close_local_position():
    listener, _, ce, rs, _ = _make_listener()
    pid = "BTCUSDT:3"
    listener._position_tracker.add(
        pid,
        {
            "symbol": "BTCUSDT",
            "side": "LONG",
            "entry_time": datetime.now(timezone.utc) - timedelta(minutes=5),
            "qty": 0.02,
        },
    )
    rs.slots.active[pid] = SlotRecord(
        position_id=pid, symbol="BTCUSDT", archetype="bpc"
    )

    listener.on_account_update(
        {
            "positions": [
                {"symbol": "BTCUSDT", "position_amount": 0.0},
            ],
            "wallet_balance": 1234.5,
            "available_balance": 1100.0,
            "unrealized_pnl_total": 12.3,
            "event_time": 1710000000,
        }
    )

    # 账户推送只做快照更新，不直接关本地仓（避免秒进秒出）。
    assert listener._position_tracker.get(pid) is not None
    ce.release_slot.assert_not_called()


def test_account_update_accepts_raw_binance_position_keys():
    """positions 可直接用交易所原始字段 s / pa（不经 UserStream 规范化）。"""
    listener, _, ce, rs, _ = _make_listener()
    pid = "BTCUSDT:raw"
    listener._position_tracker.add(
        pid,
        {
            "symbol": "BTCUSDT",
            "side": "LONG",
            "entry_time": datetime.now(timezone.utc) - timedelta(minutes=1),
            "qty": 0.01,
        },
    )
    rs.slots.active[pid] = SlotRecord(
        position_id=pid, symbol="BTCUSDT", archetype="bpc"
    )

    listener.on_account_update(
        {
            "positions": [{"s": "BTCUSDT", "pa": "0"}],
            "wallet_balance": 100.0,
        }
    )

    assert listener._position_tracker.get(pid) is not None
    ce.release_slot.assert_not_called()


def test_account_update_injects_equity_features():
    listener, _, _, _, _ = _make_listener()
    features = {"close": 50000.0}
    # REST_BALANCE (or USDT B[]) is required for G0 equity authority.
    listener.on_account_update(
        {
            "event_type": "REST_BALANCE",
            "wallet_balance": 2345.6,
            "available_balance": 2100.0,
            "unrealized_pnl_total": -15.2,
            "event_time": 1710001234,
            "positions": [],
        }
    )

    listener._inject_account_features(features)

    assert features["equity"] == 2345.6
    assert features["account_available_balance"] == 2100.0
    assert features["account_unrealized_pnl"] == -15.2
    assert features["account_event_time"] == 1710001234


def test_position_only_account_update_preserves_wallet_balance():
    listener, _, _, _, _ = _make_listener()
    listener.on_account_update(
        {
            "event_type": "REST_BALANCE",
            "wallet_balance": 2345.6,
            "available_balance": 2100.0,
            "event_time": 1710000000,
            "positions": [],
        }
    )
    # Position-only (no settlement B[] / not REST) must not wipe wallet.
    listener.on_account_update(
        {
            "wallet_balance": 0.0,
            "available_balance": 0.0,
            "event_time": 1710000060,
            "positions": [
                {"symbol": "BTCUSDT", "position_amount": 0.01, "unrealized_pnl": 1.2}
            ],
        }
    )

    features = {"close": 50000.0}
    listener._inject_account_features(features)

    assert listener._latest_account_update["wallet_balance"] == 2345.6
    assert listener._latest_account_update["available_balance"] == 2100.0
    assert features["equity"] == 2345.6


def test_exchange_fill_notifies_pcm_slot_release_with_archetype():
    listener, om, ce, rs, _ = _make_listener()
    listener.decision_handler = MagicMock()
    pid = "BTCUSDT:notify"
    listener._position_tracker.add(
        pid,
        {
            "symbol": "BTCUSDT",
            "archetype": "bpc",
            "side": "LONG",
            "entry_time": datetime.now(timezone.utc) - timedelta(seconds=5),
            "qty": 0.01,
        },
    )
    rs.slots.active[pid] = SlotRecord(
        position_id=pid, symbol="BTCUSDT", archetype="bpc"
    )
    om.handle_execution_report.return_value = Order(
        order_id="o_notify",
        position_id=pid,
        symbol="BTCUSDT",
        side=OrderSide.SELL,
        order_type=OrderType.STOP_MARKET,
        status=OrderStatus.FILLED,
    )

    listener.on_execution_report({"symbol": "BTCUSDT"})

    listener.decision_handler.notify_position_closed.assert_called_once_with(
        "BTCUSDT", "bpc"
    )
    ce.release_slot.assert_called()


def test_exchange_fill_notifies_pcm_slot_release_without_archetype():
    listener, om, ce, rs, _ = _make_listener()
    listener.decision_handler = MagicMock()
    pid = "BTCUSDT:missing-pos"
    rs.slots.active[pid] = SlotRecord(position_id=pid, symbol="BTCUSDT", archetype="me")
    om.handle_execution_report.return_value = Order(
        order_id="o_missing_pos",
        position_id=pid,
        symbol="BTCUSDT",
        side=OrderSide.SELL,
        order_type=OrderType.MARKET,
        status=OrderStatus.FILLED,
    )

    listener.on_execution_report({"symbol": "BTCUSDT"})

    listener.decision_handler.notify_position_closed.assert_called_once_with(
        "BTCUSDT", ""
    )
    ce.release_slot.assert_called()
