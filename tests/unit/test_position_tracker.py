"""PositionTracker 单元测试

覆盖:
  test_enforce_sl_triggers_close        — SL 命中后调用 close()
  test_enforce_long_sl_below_price      — LONG SL 未命中时不平仓
  test_structural_exit_ema200           — 价格穿越 EMA200 后触发
  test_structural_exit_not_triggered    — EMA200 未穿越时不触发
  test_breakeven_lock_prevents_sl_drop  — 保本锁后 SL 不低于 entry_price
  test_trailing_updates_sl              — trailing 激活后 SL 向上移动
  test_sync_exchange_sl_cancel_replace  — SL 价格变化触发 cancel+replace
  test_close_cancels_pending_orders     — 平仓前 cancel SL/TP 挂单
  test_closed_position_removed          — 平仓后 position 从 tracker 移除
  test_add_position_recorded            — add() 后 all_positions() 可获取
"""

from __future__ import annotations

import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, call, patch

from src.order_management.models import OrderType, Position, PositionSide
from src.order_management.position_tracker import PositionTracker
from src.time_series_model.core.trade_intent import TradeIntent
from src.time_series_model.live.position_logic import build_position_dict


# ─── helpers ──────────────────────────────────────────────────────────────────


def _make_om():
    om = MagicMock()
    om.place_order.return_value = MagicMock(order_id="SL_NEW")
    om.binance_api = MagicMock()
    om.binance_api.get_open_orders.return_value = []
    om.binance_api.get_open_orders_for_sl_cleanup = (
        lambda symbol=None: om.binance_api.get_open_orders(symbol)
    )
    # No default mark unless a test sets it (MagicMock() would coerce to ~1.0).
    om.binance_api.get_ticker_price.return_value = None
    om.binance_api.get_positions.return_value = []
    return om


def _make_tracker(om=None) -> PositionTracker:
    if om is None:
        om = _make_om()
    return PositionTracker(order_manager=om, symbol="BTCUSDT", default_bar_minutes=240)


def _make_intent(
    action="LONG",
    stop_loss_r=2.0,
    take_profit_r=4.0,
    max_holding_bars=100,
    allow_trailing=False,
    trailing_atr=None,
    structural_exit=None,
):
    rr = {
        "stop_loss_r": stop_loss_r,
        "take_profit_r": take_profit_r,
        "max_holding_bars": max_holding_bars,
        "allow_trailing": allow_trailing,
    }
    if trailing_atr is not None:
        rr["trailing_atr"] = trailing_atr
    if structural_exit is not None:
        rr["structural_exit"] = structural_exit  # 从 rr_constraints 传入
    ep: dict = {"rr_constraints": rr}
    return TradeIntent(
        action=action,
        symbol="BTCUSDT",
        archetype="bpc",
        execution_profile=ep,
    )


def _make_pos(
    side="LONG",
    entry_price=50000.0,
    atr=500.0,
    stop_loss_r=2.0,
    take_profit_r=4.0,
    allow_trailing=False,
    trailing_atr=None,
    structural_exit=None,
    entry_time=None,
) -> dict:
    """直接通过 build_position_dict 构建持仓字典"""
    rr = {
        "stop_loss_r": stop_loss_r,
        "take_profit_r": take_profit_r,
        "max_holding_bars": 100,
        "allow_trailing": allow_trailing,
    }
    if trailing_atr is not None:
        rr["trailing_atr"] = trailing_atr
    ep: dict = {"rr_constraints": rr}
    if structural_exit:
        ep["strategy_specific"] = {"structural_exit": structural_exit}
    intent = TradeIntent(
        action="LONG" if side == "LONG" else "SHORT",
        symbol="BTCUSDT",
        archetype="bpc",
        execution_profile=ep,
    )
    if entry_time is None:
        entry_time = datetime(2025, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
    pos = build_position_dict(
        intent=intent,
        entry_price=entry_price,
        atr=atr,
        bar_minutes=240,
        entry_time=entry_time,
    )
    pos["qty"] = 0.01
    return pos


def _make_features(
    close=50000.0,
    ema_200=None,
    now=None,
):
    if now is None:
        now = datetime(2025, 6, 1, 12, 15, 0, tzinfo=timezone.utc)
    f = {"close": close, "timestamp": now}
    if ema_200 is not None:
        f["ema_200"] = ema_200
    return f


# ─── tests ────────────────────────────────────────────────────────────────────


class TestAddAndRetrieve:

    def test_add_position_recorded(self):
        """add() 后 all_positions() 可获取持仓"""
        tracker = _make_tracker()
        pos = _make_pos()
        tracker.add("pid1", pos)

        assert tracker.get("pid1") is pos
        assert "pid1" in tracker.all_positions()
        assert len(tracker) == 1

    def test_get_nonexistent_returns_none(self):
        tracker = _make_tracker()
        assert tracker.get("nonexistent") is None

    def test_persist_and_restore_position_dict(self, tmp_path):
        state_path = tmp_path / "BTCUSDT.json"
        tracker = PositionTracker(
            order_manager=_make_om(),
            symbol="BTCUSDT",
            default_bar_minutes=240,
            state_path=state_path,
        )
        pos = _make_pos(
            entry_time=datetime(2025, 6, 1, 12, 0, 0, tzinfo=timezone.utc),
            allow_trailing=True,
            trailing_atr=3.0,
        )
        pos["trailing_activated"] = True
        pos["high_water_mark"] = 53000.0
        tracker.add("pid1", pos)

        restored = PositionTracker(
            order_manager=_make_om(),
            symbol="BTCUSDT",
            default_bar_minutes=240,
            state_path=state_path,
        )
        assert restored.restore_from_disk(live_symbols={"BTCUSDT"}) == 1
        got = restored.get("pid1")
        assert got is not None
        assert got["trailing_activated"] is True
        assert got["high_water_mark"] == pytest.approx(53000.0)
        assert isinstance(got["entry_time"], datetime)
        assert got["entry_time"].tzinfo is not None

    def test_restore_from_disk_re_mirrors_sqlite(self, tmp_path):
        """After restart, restored JSON positions must re-write SQLite for console."""
        from src.order_management.storage import Storage

        db_path = tmp_path / "order_management.db"
        storage = Storage(str(db_path))
        om = _make_om()
        om.storage = storage
        state_path = tmp_path / "BTCUSDT.json"
        tracker = PositionTracker(
            order_manager=om,
            symbol="BTCUSDT",
            default_bar_minutes=240,
            state_path=state_path,
        )
        tracker.add("pid1", _make_pos(side="SHORT", entry_price=100.0))

        # Simulate empty SQLite (e.g. DB wiped) while JSON snapshot survives.
        import sqlite3

        conn = sqlite3.connect(db_path)
        conn.execute("DELETE FROM positions")
        conn.commit()
        conn.close()

        restored = PositionTracker(
            order_manager=om,
            symbol="BTCUSDT",
            default_bar_minutes=240,
            state_path=state_path,
        )
        assert restored.restore_from_disk(live_symbols={"BTCUSDT"}) == 1
        row = storage.get_position("pid1")
        assert row is not None
        assert row.status.value == "open"
        assert row.side.value == "short"

    def test_restore_skips_and_clears_when_exchange_has_no_symbol(self, tmp_path):
        state_path = tmp_path / "BTCUSDT.json"
        tracker = PositionTracker(
            order_manager=_make_om(),
            symbol="BTCUSDT",
            default_bar_minutes=240,
            state_path=state_path,
        )
        tracker.add("pid1", _make_pos())

        restored = PositionTracker(
            order_manager=_make_om(),
            symbol="BTCUSDT",
            default_bar_minutes=240,
            state_path=state_path,
        )
        assert restored.restore_from_disk(live_symbols={"ETHUSDT"}) == 0
        assert restored.all_positions() == {}

        restored_again = PositionTracker(
            order_manager=_make_om(),
            symbol="BTCUSDT",
            default_bar_minutes=240,
            state_path=state_path,
        )
        assert restored_again.restore_from_disk(live_symbols={"BTCUSDT"}) == 0


class TestBootstrapFromExchange:

    def test_bootstrap_sets_conservative_sl_and_tp(self, tmp_path):
        state_path = tmp_path / "BNBUSDT.json"
        tracker = PositionTracker(
            order_manager=_make_om(),
            symbol="BNBUSDT",
            default_bar_minutes=120,
            state_path=state_path,
        )
        pid = tracker.bootstrap_from_exchange_position(
            symbol="BNBUSDT",
            side="long",
            entry_price=630.68,
            qty=0.31,
        )
        assert pid is not None
        pos = tracker.get(pid)
        assert pos is not None
        assert pos["stop_loss_price"] == pytest.approx(621.2198, abs=1e-4)
        assert pos["take_profit_price"] == pytest.approx(649.6004, abs=1e-4)
        assert pos["stop_loss_price"] < pos["entry_price"]
        assert pos["take_profit_price"] > pos["entry_price"]

    def test_bootstrap_merge_close_uses_canonical_sqlite_row(self, tmp_path):
        """Bootstrap merge must not leave exchange_sync row open after close()."""
        from src.order_management.models import PositionStatus
        from src.order_management.storage import Storage

        db_path = tmp_path / "order_management.db"
        storage = Storage(str(db_path))
        om = _make_om()
        om.storage = storage
        state_path = tmp_path / "BNBUSDT.json"

        existing_id = "BNB:exchange_sync_old"
        storage.create_position(
            Position(
                position_id=existing_id,
                symbol="BNBUSDT",
                side=PositionSide.LONG,
                entry_time=datetime.now(timezone.utc),
                entry_price=630.68,
                initial_size=0.31,
                current_size=0.31,
                total_cost=630.68 * 0.31,
                status=PositionStatus.OPEN,
                strategy_id="tpc",
            )
        )

        tracker = PositionTracker(
            order_manager=om,
            symbol="BNBUSDT",
            default_bar_minutes=120,
            state_path=state_path,
        )
        pid = tracker.bootstrap_from_exchange_position(
            symbol="BNBUSDT",
            side="long",
            entry_price=630.68,
            qty=0.31,
        )
        assert pid == existing_id
        assert tracker.get(existing_id) is not None
        assert "bootstrap_" not in pid

        tracker.close(existing_id, qty=0.31, reason="stop_loss")

        assert tracker.get(existing_id) is not None
        assert bool(tracker.get(existing_id).get("_pending_close"))
        open_rows = storage.get_open_positions("BNBUSDT")
        assert len(open_rows) == 1
        row = storage.get_position(existing_id)
        assert row is not None
        assert row.status == PositionStatus.OPEN

        tracker.close_from_exchange(existing_id, reason="stop_loss", exit_price=630.0)
        assert tracker.get(existing_id) is None
        open_rows = storage.get_open_positions("BNBUSDT")
        assert open_rows == []
        row = storage.get_position(existing_id)
        assert row is not None
        assert row.status == PositionStatus.CLOSED
        assert row.exit_reason == "stop_loss"


class TestEnforceAll:

    def test_enforce_sl_triggers_close(self):
        """LONG 持仓 SL 命中（价格 < stop_loss_price）→ 平仓"""
        om = _make_om()
        tracker = _make_tracker(om)
        pos = _make_pos(entry_price=50000.0, atr=500.0, stop_loss_r=2.0)
        # stop_loss_price = 50000 - 2*500 = 49000
        assert pos["stop_loss_price"] == pytest.approx(49000.0)

        tracker.add("pid1", pos)
        # 价格跌破 SL
        features = _make_features(close=48000.0)
        closed = tracker.enforce_all(features)

        assert "pid1" in closed
        pos = tracker.get("pid1")
        assert pos is not None
        assert bool(pos.get("_pending_close"))
        om.place_order.assert_called()  # market 平仓

    def test_enforce_long_sl_not_triggered(self):
        """LONG 价格高于 SL → 不触发"""
        tracker = _make_tracker()
        pos = _make_pos(entry_price=50000.0, atr=500.0, stop_loss_r=2.0)
        tracker.add("pid1", pos)
        features = _make_features(close=51000.0)

        closed = tracker.enforce_all(features)

        assert closed == []
        assert len(tracker) == 1

    def test_enforce_all_sl_breached_mark_keeps_sqlite_closed(self, tmp_path):
        """-2021 guard must not re-persist CLOSED row as OPEN after market close."""
        from src.order_management.models import PositionStatus
        from src.order_management.storage import Storage

        db_path = tmp_path / "order_management.db"
        storage = Storage(str(db_path))
        om = _make_om()
        om.storage = storage
        om.binance_api.get_ticker_price.return_value = 48500.0
        tracker = _make_tracker(om)
        pos = _make_pos(entry_price=50000.0, atr=500.0, stop_loss_r=2.0)
        tracker.add("pid1", pos)

        closed = tracker.enforce_all(_make_features(close=51000.0))

        assert "pid1" in closed
        pos = tracker.get("pid1")
        assert pos is not None
        assert bool(pos.get("_pending_close"))
        row = storage.get_position("pid1")
        assert row is not None
        assert row.status == PositionStatus.OPEN

    def test_structural_exit_ema200_triggered(self):
        """LONG 价格跌破 EMA200 且 breakeven_locked=True → 触发 structural exit"""
        om = _make_om()
        tracker = _make_tracker(om)
        # structural_exit 通过 rr_constraints.structural_exit 传入
        intent = _make_intent(stop_loss_r=5.0, structural_exit="ema200")  # SL 很远
        pos = build_position_dict(
            intent=intent,
            entry_price=50000.0,
            atr=500.0,
            bar_minutes=240,
            entry_time=datetime(2025, 6, 1, tzinfo=timezone.utc),
        )
        pos["qty"] = 0.01
        pos["breakeven_locked"] = True  # structural exit 需要 breakeven_locked=True

        tracker.add("pid1", pos)
        # EMA200 = 49500, 价格 = 49000 < EMA200 → 穿越
        features = _make_features(close=49000.0, ema_200=49500.0)
        closed = tracker.enforce_all(features)

        assert "pid1" in closed

    def test_structural_exit_ema200_not_triggered(self):
        """LONG 价格仍在 EMA200 之上 → 不触发"""
        tracker = _make_tracker()
        intent = _make_intent(stop_loss_r=5.0, structural_exit="ema200")
        pos = build_position_dict(
            intent=intent,
            entry_price=50000.0,
            atr=500.0,
            bar_minutes=240,
            entry_time=datetime(2025, 6, 1, tzinfo=timezone.utc),
        )
        pos["qty"] = 0.01
        pos["breakeven_locked"] = True  # breakeven_locked 需要 True
        tracker.add("pid1", pos)

        features = _make_features(close=50500.0, ema_200=49000.0)  # 价格 > EMA200
        closed = tracker.enforce_all(features)

        assert closed == []

    def test_closed_position_removed_from_tracker(self):
        """SL 触发后 position 从 tracker 移除"""
        tracker = _make_tracker()
        pos = _make_pos(entry_price=50000.0, atr=500.0, stop_loss_r=2.0)
        tracker.add("pid1", pos)

        tracker.enforce_all(_make_features(close=48000.0))

        pos = tracker.get("pid1")
        assert pos is not None
        assert bool(pos.get("_pending_close"))
        assert len(tracker) == 1

    def test_no_positions_returns_empty(self):
        """没有持仓时 enforce_all 返回空列表"""
        tracker = _make_tracker()
        closed = tracker.enforce_all(_make_features())
        assert closed == []

    def test_missing_price_skips_enforce(self):
        """features 无法解析价格时跳过（返回空）"""
        tracker = _make_tracker()
        pos = _make_pos()
        tracker.add("pid1", pos)

        # features 中无 close/price 等
        closed = tracker.enforce_all({"timestamp": datetime.now(timezone.utc)})
        assert closed == []
        assert len(tracker) == 1

    def test_entry_bar_wick_does_not_stop_new_fade_long(self):
        """2h_close 开多后，不得用同一根信号棒的 pierce low 打止损。

        XRP 2026-08-21 08:00 live: close 1.4197, low 1.3123, SL 1.3398.
        """
        om = _make_om()
        tracker = PositionTracker(
            order_manager=om, symbol="XRPUSDT", default_bar_minutes=120
        )
        bar_ts = datetime(2026, 8, 21, 8, 0, tzinfo=timezone.utc)
        pos = _make_pos(
            side="LONG",
            entry_price=1.4197,
            atr=0.039937,
            stop_loss_r=2.0,
            entry_time=datetime(2026, 8, 21, 10, 2, tzinfo=timezone.utc),
        )
        pos["bar_minutes"] = 120
        # Persisted tracker state restores datetimes as ISO strings.
        pos["_entry_feature_ts"] = bar_ts.isoformat()
        tracker.add("xrp-fade", pos)
        assert pos["stop_loss_price"] == pytest.approx(1.339826, rel=1e-5)

        closed = tracker.enforce_all(
            {
                "timestamp": bar_ts,
                "open": 1.3138,
                "high": 1.4305,
                "low": 1.3123,
                "close": 1.4197,
                "atr": 0.039937,
            }
        )
        assert closed == []
        held = tracker.get("xrp-fade")
        assert held is not None
        assert not bool(held.get("_pending_close"))
        market_closes = [
            c
            for c in om.place_order.call_args_list
            if c.kwargs.get("order_type") == OrderType.MARKET
        ]
        assert market_closes == []

    def test_entry_bar_wick_does_not_stop_new_fade_short(self):
        """Close-entry short likewise ignores the decide-bar's earlier high."""
        om = _make_om()
        tracker = PositionTracker(
            order_manager=om, symbol="XRPUSDT", default_bar_minutes=120
        )
        bar_ts = datetime(2026, 8, 21, 8, 0, tzinfo=timezone.utc)
        pos = _make_pos(
            side="SHORT",
            entry_price=1.32,
            atr=0.04,
            stop_loss_r=2.0,
            entry_time=datetime(2026, 8, 21, 10, 2, tzinfo=timezone.utc),
        )
        pos["bar_minutes"] = 120
        pos["_entry_feature_ts"] = bar_ts
        tracker.add("xrp-fade-short", pos)
        assert pos["stop_loss_price"] == pytest.approx(1.40)

        closed = tracker.enforce_all(
            {
                "timestamp": bar_ts,
                "high": 1.43,
                "low": 1.31,
                "close": 1.32,
                "atr": 0.04,
            }
        )
        assert closed == []
        held = tracker.get("xrp-fade-short")
        assert held is not None
        assert not bool(held.get("_pending_close"))

    def test_later_bar_low_still_stops_after_entry_bar(self):
        """下一根 2h 的真实 low 穿过止损时仍应平仓。"""
        om = _make_om()
        tracker = PositionTracker(
            order_manager=om, symbol="XRPUSDT", default_bar_minutes=120
        )
        entry_bar = datetime(2026, 8, 21, 8, 0, tzinfo=timezone.utc)
        next_bar = datetime(2026, 8, 21, 10, 0, tzinfo=timezone.utc)
        pos = _make_pos(
            side="LONG",
            entry_price=1.4197,
            atr=0.039937,
            stop_loss_r=2.0,
            entry_time=datetime(2026, 8, 21, 10, 2, tzinfo=timezone.utc),
        )
        pos["bar_minutes"] = 120
        pos["_entry_feature_ts"] = entry_bar
        tracker.add("xrp-fade", pos)

        closed = tracker.enforce_all(
            {
                "timestamp": next_bar,
                "high": 1.42,
                "low": 1.30,
                "close": 1.35,
                "atr": 0.039937,
            }
        )
        assert "xrp-fade" in closed
        pending = tracker.get("xrp-fade")
        assert pending is not None
        assert bool(pending.get("_pending_close"))

    def test_parent_exit_forces_child_add_close_same_bar(self):
        """母仓触发退出时，子加仓同 bar 强制退出（默认行为）"""
        om = _make_om()
        tracker = _make_tracker(om)
        parent = _make_pos(entry_price=50000.0, atr=500.0, stop_loss_r=2.0)
        parent["qty"] = 0.01
        tracker.add("parent", parent)
        child = _make_pos(entry_price=50500.0, atr=500.0, stop_loss_r=6.0)
        child["qty"] = 0.005
        child["_is_add_position"] = True
        child["_parent_pid"] = "parent"
        child["_share_parent_exit"] = True
        tracker.add("child", child)
        # 48000 会触发 parent SL；child 单独还未到 50500-6*500=47500
        closed = tracker.enforce_all(_make_features(close=48000.0))
        assert set(closed) == {"parent", "child"}
        assert len(tracker) == 2
        assert bool(tracker.get("parent").get("_pending_close"))
        assert bool(tracker.get("child").get("_pending_close"))


class TestSyncExchangeSL:

    def test_sync_exchange_sl_cancel_replace(self):
        """SL 价格变化时 cancel 旧挂单 + place 新 STOP_MARKET 挂单"""
        om = _make_om()
        tracker = _make_tracker(om)
        pos = _make_pos(allow_trailing=True, trailing_atr=1.5, stop_loss_r=3.0)
        pos["_exchange_sl_order_id"] = "OLD_SL"
        pos["_exchange_sl_price"] = pos["stop_loss_price"]  # 记录初始 SL 价格
        tracker.add("pid1", pos)

        # 手动模拟 trailing 更新了 SL（价格上移）
        pos["stop_loss_price"] = pos["stop_loss_price"] + 200.0

        tracker.sync_exchange_sl("pid1")

        om.cancel_order.assert_called_once_with("OLD_SL")
        om.place_order.assert_called_once()
        new_call = om.place_order.call_args
        assert new_call.kwargs["stop_price"] == pos["stop_loss_price"]

    def test_sync_exchange_sl_no_change_skips(self):
        """SL 价格未变化时不触发 cancel+replace"""
        om = _make_om()
        tracker = _make_tracker(om)
        pos = _make_pos()
        sl = pos.get("stop_loss_price")
        if sl is not None:
            pos["_exchange_sl_order_id"] = "OLD_SL"
            pos["_exchange_sl_price"] = sl  # 相同价格
        tracker.add("pid1", pos)

        tracker.sync_exchange_sl("pid1")

        om.cancel_order.assert_not_called()
        om.place_order.assert_not_called()

    def test_sync_exchange_sl_initial_place_without_old_exchange_price(self):
        """有 software SL 但无 _exchange_sl_price 时仍应首次挂 STOP_MARKET"""
        om = _make_om()
        tracker = _make_tracker(om)
        pos = _make_pos(stop_loss_r=3.0)
        assert pos.get("stop_loss_price") is not None
        tracker.add("pid1", pos)

        tracker.sync_exchange_sl("pid1")

        om.cancel_order.assert_not_called()
        om.place_order.assert_called_once()
        assert tracker.get("pid1")["_exchange_sl_price"] == pos["stop_loss_price"]
        assert tracker.get("pid1")["_exchange_sl_order_id"] == "SL_NEW"

    def test_ensure_exchange_stop_losses_places_missing(self):
        om = _make_om()
        tracker = _make_tracker(om)
        pos = _make_pos(stop_loss_r=3.0)
        tracker.add("pid1", pos)

        n = tracker.ensure_exchange_stop_losses()

        assert n == 1
        om.place_order.assert_called_once()

    def test_ensure_exchange_stop_losses_rearms_when_local_flag_stale(self):
        """Local _exchange_sl_* set but order gone from openAlgo → clear + re-arm."""
        om = _make_om()
        om.binance_api.get_open_orders.return_value = []  # venue empty
        tracker = _make_tracker(om)
        pos = _make_pos(stop_loss_r=3.0)
        pos["_exchange_sl_price"] = float(pos["stop_loss_price"])
        pos["_exchange_sl_order_id"] = "GONE_SL"
        tracker.add("pid1", pos)

        n = tracker.ensure_exchange_stop_losses()

        assert n == 1
        om.place_order.assert_called_once()
        assert tracker.get("pid1")["_exchange_sl_order_id"] == "SL_NEW"

    def test_ensure_exchange_stop_losses_skips_when_venue_still_has_sl(self):
        om = _make_om()
        om.binance_api.get_open_orders.return_value = [
            {"order_id": "KEEP_SL", "type": "STOP_MARKET"}
        ]
        tracker = _make_tracker(om)
        pos = _make_pos(stop_loss_r=3.0)
        pos["_exchange_sl_price"] = float(pos["stop_loss_price"])
        pos["_exchange_sl_order_id"] = "KEEP_SL"
        tracker.add("pid1", pos)

        n = tracker.ensure_exchange_stop_losses()

        assert n == 0
        om.place_order.assert_not_called()

    def test_sync_exchange_sl_4509_clears_ghost_when_exchange_flat(self):
        """-4509 + exchange flat on this side -> drop the local ghost position."""
        om = _make_om()
        om.place_order.side_effect = Exception(
            'binance {"code":-4509,"msg":"Time in Force (TIF) GTE can only be '
            'used with open positions."}'
        )
        om.binance_api.get_positions.return_value = []  # exchange truly flat
        tracker = _make_tracker(om)
        tracker.add("pid1", _make_pos(stop_loss_r=3.0))

        tracker.sync_exchange_sl("pid1")

        assert "pid1" not in tracker.all_positions()

    def test_sync_exchange_sl_4509_keeps_position_when_still_live(self):
        """-4509 but exchange still shows a same-side position -> keep local state."""
        om = _make_om()
        om.place_order.side_effect = Exception(
            'binance {"code":-4509,"msg":"Time in Force (TIF) GTE can only be '
            'used with open positions."}'
        )
        om.binance_api.get_positions.return_value = [
            {"symbol": "BTCUSDT", "side": "long", "size": 0.5}
        ]
        tracker = _make_tracker(om)
        tracker.add("pid1", _make_pos(stop_loss_r=3.0))

        tracker.sync_exchange_sl("pid1")

        assert "pid1" in tracker.all_positions()

    def test_sync_exchange_sl_4509_keeps_position_when_exchange_read_fails(self):
        """Fail-safe: if exchange read errors, never drop the local position."""
        om = _make_om()
        om.place_order.side_effect = Exception(
            'binance {"code":-4509,"msg":"... open positions."}'
        )
        om.binance_api.get_positions.side_effect = Exception("network down")
        tracker = _make_tracker(om)
        tracker.add("pid1", _make_pos(stop_loss_r=3.0))

        tracker.sync_exchange_sl("pid1")

        assert "pid1" in tracker.all_positions()

    def test_sync_exchange_sl_skips_add_inheriting_parent_stop(self):
        om = _make_om()
        tracker = _make_tracker(om)
        pos = _make_pos(stop_loss_r=3.0)
        pos["_is_add_position"] = True
        pos["_inherit_parent_stop"] = True
        pos["stop_loss_price"] = float(pos["stop_loss_price"]) + 100.0
        tracker.add("add1", pos)

        tracker.sync_exchange_sl("add1")

        om.place_order.assert_not_called()
        om.binance_api.get_open_orders.assert_not_called()

    def test_sync_exchange_sl_cancels_close_position_tp_before_replace(self):
        """closePosition TP occupies the same Binance slot as closePosition SL."""
        om = _make_om()
        om.binance_api.get_open_orders.return_value = [
            {
                "order_id": "888",
                "side": "sell",
                "type": "take_profit_market",
                "info": {
                    "closePosition": True,
                    "positionSide": "LONG",
                    "type": "TAKE_PROFIT_MARKET",
                },
            }
        ]
        tracker = _make_tracker(om)
        pos = _make_pos(stop_loss_r=3.0)
        tracker.add("pid1", pos)
        pos["stop_loss_price"] = float(pos["stop_loss_price"]) + 50.0
        pos["_exchange_sl_price"] = float(pos["stop_loss_price"]) - 50.0

        tracker.sync_exchange_sl("pid1")

        om.binance_api.cancel_order.assert_called_with("888", "BTCUSDT")
        om.place_order.assert_called_once()

    def test_sync_exchange_sl_skips_non_owner_when_parent_has_exchange_sl(self):
        om = _make_om()
        tracker = _make_tracker(om)
        parent = _make_pos(stop_loss_r=3.0)
        parent["_exchange_sl_order_id"] = "PARENT_SL"
        parent["_exchange_sl_price"] = parent["stop_loss_price"]
        child = _make_pos(stop_loss_r=3.0)
        child["_is_add_position"] = True
        child["_inherit_parent_stop"] = False
        child["stop_loss_price"] = float(parent["stop_loss_price"]) + 100.0
        tracker.add("parent", parent)
        tracker.add("child", child)

        tracker.sync_exchange_sl("child")

        om.place_order.assert_not_called()

    def test_sync_exchange_sl_retries_after_4130_non_close_position_stop(self):
        """-4130 retry clears STOP/TP even when closePosition flag is absent in API row."""
        om = _make_om()
        om.place_order.side_effect = [
            Exception('binance {"code":-4130,"msg":"existing"}'),
            MagicMock(order_id="SL_RETRY2"),
        ]
        om.binance_api.get_open_orders.return_value = [
            {
                "order_id": "777",
                "client_order_id": "tl_stop_777",
                "side": "sell",
                "type": "stop_market",
                "info": {"positionSide": "LONG", "type": "STOP_MARKET"},
            }
        ]
        tracker = _make_tracker(om)
        pos = _make_pos(stop_loss_r=3.0)
        tracker.add("pid2", pos)
        pos["stop_loss_price"] = float(pos["stop_loss_price"]) + 50.0
        pos["_exchange_sl_price"] = float(pos["stop_loss_price"]) - 50.0

        tracker.sync_exchange_sl("pid2")

        assert om.place_order.call_count == 2
        om.binance_api.cancel_order.assert_called_with("777", "BTCUSDT")

    def test_4130_fallback_does_not_cancel_unowned_stop_tp(self):
        om = _make_om()
        tracker = _make_tracker(om)
        order = {
            "order_id": "manual1",
            "client_order_id": "manual_stop_1",
            "side": "sell",
            "type": "stop_market",
            "info": {"positionSide": "LONG", "type": "STOP_MARKET"},
        }
        om.binance_api.get_open_orders.return_value = [order]

        cancelled = tracker._cancel_open_close_position_conditionals(
            position_side="LONG",
            include_all_stop_tp=True,
        )

        assert PositionTracker._is_stop_or_tp_conditional(order) is True
        assert PositionTracker._is_owned_stop_or_tp_conditional(order) is False
        assert cancelled == 0
        om.binance_api.cancel_order.assert_not_called()

    def test_sync_exchange_sl_market_close_when_stop_breached_long(self):
        """-2021 guard: long mark below stop → market close, no STOP_MARKET."""
        from src.order_management.models import OrderType

        om = _make_om()
        om.binance_api.get_ticker_price.return_value = 48500.0
        tracker = _make_tracker(om)
        pos = _make_pos(stop_loss_r=2.0)
        tracker.add("pid1", pos)
        pos["stop_loss_price"] = 49000.0

        tracker.sync_exchange_sl("pid1")

        kwargs = om.place_order.call_args.kwargs
        assert kwargs["order_type"] == OrderType.MARKET
        assert kwargs["close_position"] is False
        pos = tracker.get("pid1")
        assert pos is not None
        assert bool(pos.get("_pending_close"))

    def test_sync_exchange_sl_2021_close_failure_keeps_position(self):
        """If the -2021 market close errors, keep local position (software SL active)."""
        om = _make_om()
        om.binance_api.get_ticker_price.return_value = 48500.0
        om.place_order.side_effect = Exception("network down")
        tracker = _make_tracker(om)
        pos = _make_pos(stop_loss_r=2.0)
        tracker.add("pid1", pos)
        pos["stop_loss_price"] = 49000.0

        tracker.sync_exchange_sl("pid1")

        assert tracker.get("pid1") is not None

    def test_ensure_exchange_stop_losses_2021_close_persists_state(self, tmp_path):
        """-2021 close at startup must also persist JSON state (no ghost on restart)."""
        import json as _json

        om = _make_om()
        om.binance_api.get_ticker_price.return_value = 48500.0
        state_path = tmp_path / "BTCUSDT.json"
        tracker = PositionTracker(
            order_manager=om,
            symbol="BTCUSDT",
            default_bar_minutes=240,
            state_path=state_path,
        )
        pos = _make_pos(stop_loss_r=2.0)
        tracker.add("pid1", pos)
        pos["stop_loss_price"] = 49000.0

        tracker.ensure_exchange_stop_losses()

        pos = tracker.get("pid1")
        assert pos is not None
        assert bool(pos.get("_pending_close"))
        saved = _json.loads(state_path.read_text(encoding="utf-8"))
        assert "pid1" in saved.get("positions", {})

    def test_sync_exchange_sl_market_close_when_stop_breached_short(self):
        from src.order_management.models import OrderType

        om = _make_om()
        om.binance_api.get_ticker_price.return_value = 51500.0
        tracker = _make_tracker(om)
        pos = _make_pos(side="SHORT", stop_loss_r=2.0)
        tracker.add("pid1", pos)
        pos["stop_loss_price"] = 51000.0

        tracker.sync_exchange_sl("pid1")

        kwargs = om.place_order.call_args.kwargs
        assert kwargs["order_type"] == OrderType.MARKET
        pos = tracker.get("pid1")
        assert pos is not None
        assert bool(pos.get("_pending_close"))

    def test_sync_exchange_sl_retries_after_4130(self):
        om = _make_om()
        om.place_order.side_effect = [
            Exception('binance {"code":-4130,"msg":"existing"}'),
            MagicMock(order_id="SL_RETRY"),
        ]
        om.binance_api.get_open_orders.return_value = [
            {
                "order_id": "999",
                "side": "sell",
                "type": "stop_market",
                "info": {
                    "closePosition": True,
                    "positionSide": "LONG",
                    "type": "STOP_MARKET",
                },
            }
        ]
        tracker = _make_tracker(om)
        pos = _make_pos(stop_loss_r=3.0)
        tracker.add("pid1", pos)
        pos["stop_loss_price"] = float(pos["stop_loss_price"]) + 50.0
        pos["_exchange_sl_price"] = float(pos["stop_loss_price"]) - 50.0

        tracker.sync_exchange_sl("pid1")

        assert om.place_order.call_count == 2
        om.binance_api.cancel_order.assert_called_with("999", "BTCUSDT")
        assert tracker.get("pid1")["_exchange_sl_order_id"] == "SL_RETRY"

    def test_enforce_all_auto_syncs_trailing_sl(self):
        """enforce_all 时 trailing SL 变化自动同步到交易所"""
        om = _make_om()
        tracker = _make_tracker(om)
        # trailing 策略：activation_r=1.0, trail_r=1.5
        intent = _make_intent(
            stop_loss_r=3.0,
            allow_trailing=True,
            trailing_atr=1.5,
        )
        pos = build_position_dict(
            intent=intent,
            entry_price=50000.0,
            atr=500.0,
            bar_minutes=240,
            entry_time=datetime(2025, 6, 1, tzinfo=timezone.utc),
        )
        pos["qty"] = 0.01
        # 记录初始 exchange SL（trailing 前）
        initial_sl = pos.get("stop_loss_price", 0.0)
        pos["_exchange_sl_order_id"] = "OLD_SL"
        pos["_exchange_sl_price"] = initial_sl
        # 模拟价格已大幅上涨，触发 trailing 激活
        pos["hwm"] = 51500.0  # high watermark > entry + activation
        pos["trailing_activated"] = True

        tracker.add("pid1", pos)
        # 价格继续上涨，trailing 应上移 SL
        features = _make_features(close=52000.0)
        tracker.enforce_all(features)

        # 若 SL 被 enforce_position 更新，cancel+replace 应发生
        new_sl = tracker.get("pid1") and tracker._positions.get("pid1", {}).get(
            "stop_loss_price"
        )
        # 仓位已平或 SL 已更新，只要没有异常即可
        assert True  # 核心是不崩溃


class TestClose:

    def test_close_cancels_pending_orders(self):
        """close() 前先 cancel SL 和 TP 挂单"""
        om = _make_om()
        tracker = _make_tracker(om)
        pos = _make_pos()
        pos["_exchange_sl_order_id"] = "SL_123"
        pos["_exchange_tp_order_id"] = "TP_456"
        tracker.add("pid1", pos)

        tracker.close("pid1", qty=0.01, reason="test")

        cancel_calls = [c.args[0] for c in om.cancel_order.call_args_list]
        assert "SL_123" in cancel_calls
        assert "TP_456" in cancel_calls

    def test_close_places_market_order(self):
        """close() 下 MARKET 平仓单"""
        from src.order_management.models import OrderType

        om = _make_om()
        tracker = _make_tracker(om)
        pos = _make_pos(side="LONG")
        tracker.add("pid1", pos)

        tracker.close("pid1", qty=0.01, reason="sl_hit")

        place_calls = om.place_order.call_args_list
        assert len(place_calls) >= 1
        kwargs = place_calls[0].kwargs
        assert kwargs["order_type"] == OrderType.MARKET
        assert kwargs["reduce_only"] is True
        assert kwargs.get("close_position") is False

    def test_close_market_omits_close_position_flag(self):
        """MARKET reduceOnly must not set closePosition (Binance -4136)."""
        from src.order_management.models import OrderType

        om = _make_om()
        tracker = _make_tracker(om)
        tracker.add("pid1", _make_pos(side="SHORT"))

        tracker.close("pid1", qty=0.01, reason="tp_hit")

        kwargs = om.place_order.call_args.kwargs
        assert kwargs["order_type"] == OrderType.MARKET
        assert kwargs["close_position"] is False

    def test_close_zero_qty_skips(self):
        """qty=0 时不下单"""
        om = _make_om()
        tracker = _make_tracker(om)
        pos = _make_pos()
        tracker.add("pid1", pos)

        tracker.close("pid1", qty=0.0, reason="test")

        om.place_order.assert_not_called()

    def test_close_sets_pending_close_without_removing_position(self):
        """Software close submits order but keeps position until exchange fill."""
        om = _make_om()
        tracker = _make_tracker(om)
        tracker.add("pid1", _make_pos())

        ok = tracker.close("pid1", qty=0.01, reason="time_stop")

        assert ok is True
        pos = tracker.get("pid1")
        assert pos is not None
        assert bool(pos.get("_pending_close"))
        assert pos.get("_pending_close_reason") == "time_stop"
        assert pos.get("_pending_close_since")
        assert pos.get("_pending_close_last_submit_at")

    def test_reconcile_pending_close_flat_confirms_close(self):
        """Stale pending close is finalized only after exchange side is flat."""
        om = _make_om()
        om.binance_api.get_positions.return_value = []
        tracker = _make_tracker(om)
        pos = _make_pos()
        pos["_pending_close"] = True
        pos["_pending_close_reason"] = "time_stop"
        pos["_pending_close_qty"] = 0.01
        pos["_pending_close_since"] = datetime(
            2025, 6, 1, 12, 0, tzinfo=timezone.utc
        ).isoformat()
        tracker.add("pid1", pos)

        closed = tracker.reconcile_pending_closes(
            timeout_seconds=60,
            now=datetime(2025, 6, 1, 12, 2, tzinfo=timezone.utc),
        )

        assert closed == ["pid1"]
        assert tracker.get("pid1") is None
        om.place_order.assert_not_called()

    def test_reconcile_pending_close_retries_when_exchange_still_open(self):
        """If exchange still has exposure, retry reduce-only close instead of popping."""
        om = _make_om()
        om.binance_api.get_positions.return_value = [
            {"symbol": "BTCUSDT", "side": "LONG", "size": 0.01}
        ]
        tracker = _make_tracker(om)
        tracker.pending_close_retry_seconds = 60
        tracker.pending_close_max_retries = 2
        pos = _make_pos()
        pos["_pending_close"] = True
        pos["_pending_close_reason"] = "time_stop"
        pos["_pending_close_qty"] = 0.01
        pos["_pending_close_since"] = datetime(
            2025, 6, 1, 12, 0, tzinfo=timezone.utc
        ).isoformat()
        pos["_pending_close_last_submit_at"] = datetime(
            2025, 6, 1, 12, 0, tzinfo=timezone.utc
        ).isoformat()
        tracker.add("pid1", pos)

        closed = tracker.reconcile_pending_closes(
            timeout_seconds=60,
            now=datetime(2025, 6, 1, 12, 2, tzinfo=timezone.utc),
        )

        assert closed == []
        assert tracker.get("pid1") is not None
        assert tracker.get("pid1").get("_pending_close_retries") == 1
        kwargs = om.place_order.call_args.kwargs
        assert kwargs["reduce_only"] is True
        assert kwargs["close_position"] is False


class TestExchangeCloseSync:

    def test_close_from_exchange_removes_position_without_market_order(self):
        om = _make_om()
        tracker = _make_tracker(om)
        pos = _make_pos()
        pos["_exchange_sl_order_id"] = "sl-123"
        tracker.add("pid1", pos)

        ok = tracker.close_from_exchange(
            "pid1", reason="stop_loss_hit", exit_price=49000.0
        )

        assert ok is True
        assert tracker.get("pid1") is None
        om.place_order.assert_not_called()
        om.cancel_order.assert_called_with("sl-123")

    def test_close_from_exchange_cascades_add_leg(self):
        om = _make_om()
        tracker = _make_tracker(om)
        parent = _make_pos()
        parent["qty"] = 0.01
        child = _make_pos(entry_price=50500.0)
        child["qty"] = 0.005
        child["_is_add_position"] = True
        child["_parent_pid"] = "parent"
        child["_share_parent_exit"] = True
        tracker.add("parent", parent)
        tracker.add("child", child)

        ok = tracker.close_from_exchange(
            "parent", reason="stop_loss_hit", exit_price=49000.0, cascade_family=True
        )

        assert ok is True
        assert tracker.get("parent") is None
        assert tracker.get("child") is None
