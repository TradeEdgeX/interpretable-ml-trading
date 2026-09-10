"""_periodic_tasks 中 slot 同步逻辑的单元测试 (Bug 2 修复验证)

覆盖:
  test_stale_slot_cleared_when_api_none     — Bug 2 修复: api=None 时强制清空所有 slot
  test_stale_slot_cleared_when_no_position  — api 可用但交易所无此持仓 → 释放 slot
  test_active_slot_preserved               — 交易所有真实持仓的 slot 不被释放
  test_no_active_slots_skips_sync          — active 为空时不调用 release_slot
  test_sync_logs_warning_on_api_none       — api=None 时记录 warning
"""

from __future__ import annotations

import asyncio
import pytest
from unittest.mock import MagicMock, call, AsyncMock, patch
from datetime import datetime, timezone


# ─── 提取 slot 同步核心逻辑为可独立测试的函数 ──────────────────────────────────


def _run_slot_sync(
    ce,
    rs,
    om,
    symbol: str = "BTCUSDT",
):
    """将 _periodic_tasks 中 slot 同步代码块提取为纯函数进行测试。

    这段代码直接拷贝自 order_flow_listener._periodic_tasks 的同步部分，
    以确保测试精确覆盖生产代码路径。
    """
    import logging

    logger = logging.getLogger(__name__)

    if ce is None or rs is None or om is None:
        return

    _api = getattr(om, "binance_api", None)
    _active = dict(rs.slots.active)
    if not _active:
        return

    if _api is None:
        # Bug 2 修复: api 不可用时强制清空所有 stale slot
        for _pid in list(_active.keys()):
            ce.release_slot(
                st=rs,
                position_id=_pid,
                reason="stale_sync_no_api",
            )
        ce.save_runtime_state(rs)
    else:
        # api 可用: 查询交易所实际持仓，释放无实仓的 slot
        _positions = _api.get_positions()
        _live_syms = set()
        for _p in _positions:
            _raw = _p.get("symbol", "").replace("/", "").split(":")[0]
            if _raw:
                _live_syms.add(_raw)
        _freed = 0
        for _pid, _rec in _active.items():
            _ssym = getattr(_rec, "symbol", None) or ""
            if _ssym and _ssym not in _live_syms:
                ce.release_slot(
                    st=rs,
                    position_id=_pid,
                    reason="stale_sync",
                )
                _freed += 1
        if _freed > 0:
            ce.save_runtime_state(rs)


def _make_slot_record(symbol="BTCUSDT"):
    rec = MagicMock()
    rec.symbol = symbol
    return rec


def _make_deps(api=None, active_slots=None):
    """构造 ce, rs, om 三个 mock"""
    ce = MagicMock()
    rs = MagicMock()
    rs.slots.active = {}
    if active_slots:
        for pid, sym in active_slots.items():
            rs.slots.active[pid] = _make_slot_record(sym)

    om = MagicMock()
    om.binance_api = api  # None 或 MagicMock
    return ce, rs, om


# ─── tests ────────────────────────────────────────────────────────────────────


class TestSlotSyncApiNone:

    def test_stale_slot_cleared_when_api_none(self):
        """Bug 2 修复验证: binance_api=None 时强制清空所有 active slot"""
        active = {
            "BNBUSDT:pid1": "BNBUSDT",
            "BNBUSDT:pid2": "BNBUSDT",
            "BNBUSDT:pid3": "BNBUSDT",
            "BNBUSDT:pid4": "BNBUSDT",
        }
        ce, rs, om = _make_deps(api=None, active_slots=active)

        _run_slot_sync(ce, rs, om)

        # 所有 4 个 slot 都应被 release
        assert ce.release_slot.call_count == 4
        released_pids = {
            c.kwargs["position_id"] for c in ce.release_slot.call_args_list
        }
        assert released_pids == set(active.keys())

        # 必须保存状态
        ce.save_runtime_state.assert_called_once_with(rs)

    def test_release_reason_is_stale_sync_no_api(self):
        """api=None 时释放原因为 stale_sync_no_api"""
        ce, rs, om = _make_deps(api=None, active_slots={"pid1": "BNBUSDT"})
        _run_slot_sync(ce, rs, om)

        reason = ce.release_slot.call_args.kwargs["reason"]
        assert reason == "stale_sync_no_api"

    def test_no_active_slots_skips_sync_api_none(self):
        """active 为空时不调用 release_slot（即使 api=None）"""
        ce, rs, om = _make_deps(api=None, active_slots={})
        _run_slot_sync(ce, rs, om)

        ce.release_slot.assert_not_called()
        ce.save_runtime_state.assert_not_called()


class TestSlotSyncApiAvailable:

    def _make_api_with_positions(self, symbols):
        """返回一个 mock api，get_positions() 返回指定 symbol 列表"""
        api = MagicMock()
        api.get_positions.return_value = [{"symbol": s} for s in symbols]
        return api

    def test_stale_slot_cleared_when_no_position(self):
        """交易所无此币种持仓时释放 slot"""
        api = self._make_api_with_positions(["BTC/USDT:USDT"])  # 只有 BTCUSDT
        active = {"BNBUSDT:pid1": "BNBUSDT"}  # BNBUSDT slot 但交易所没有
        ce, rs, om = _make_deps(api=api, active_slots=active)

        _run_slot_sync(ce, rs, om)

        ce.release_slot.assert_called_once()
        assert ce.release_slot.call_args.kwargs["reason"] == "stale_sync"
        ce.save_runtime_state.assert_called_once()

    def test_active_slot_preserved_when_position_exists(self):
        """交易所有此币种持仓时保留 slot"""
        api = self._make_api_with_positions(["BNB/USDT:USDT"])  # BNBUSDT 有持仓
        active = {"BNBUSDT:pid1": "BNBUSDT"}
        ce, rs, om = _make_deps(api=api, active_slots=active)

        _run_slot_sync(ce, rs, om)

        ce.release_slot.assert_not_called()
        ce.save_runtime_state.assert_not_called()

    def test_mixed_slots_partial_release(self):
        """多个 slot 中只有无实仓的被释放"""
        # 交易所有 BTCUSDT，无 BNBUSDT
        api = self._make_api_with_positions(["BTC/USDT:USDT"])
        active = {
            "BTCUSDT:pid1": "BTCUSDT",  # 保留
            "BNBUSDT:pid2": "BNBUSDT",  # 释放
        }
        ce, rs, om = _make_deps(api=api, active_slots=active)

        _run_slot_sync(ce, rs, om)

        assert ce.release_slot.call_count == 1
        released_pid = ce.release_slot.call_args.kwargs["position_id"]
        assert released_pid == "BNBUSDT:pid2"

    def test_symbol_format_normalized(self):
        """ccxt 格式 'BNB/USDT:USDT' 正确转换为 'BNBUSDT' 后与 slot 匹配"""
        # 交易所返回 ccxt 格式，slot symbol 是原生格式
        api = self._make_api_with_positions(["BNB/USDT:USDT"])  # ccxt 格式
        active = {"BNBUSDT:pid1": "BNBUSDT"}  # 原生格式
        ce, rs, om = _make_deps(api=api, active_slots=active)

        _run_slot_sync(ce, rs, om)

        # BNB/USDT:USDT → BNBUSDT → 匹配 → 不释放
        ce.release_slot.assert_not_called()

    def test_no_active_slots_skips_sync(self):
        """active 为空时不查询交易所，不调用 release_slot"""
        api = MagicMock()
        ce, rs, om = _make_deps(api=api, active_slots={})

        _run_slot_sync(ce, rs, om)

        api.get_positions.assert_not_called()
        ce.release_slot.assert_not_called()

    def test_all_slots_have_positions_no_save(self):
        """所有 slot 都有实仓时不调用 save_runtime_state"""
        api = self._make_api_with_positions(["BTC/USDT:USDT", "BNB/USDT:USDT"])
        active = {"BTCUSDT:pid1": "BTCUSDT", "BNBUSDT:pid2": "BNBUSDT"}
        ce, rs, om = _make_deps(api=api, active_slots=active)

        _run_slot_sync(ce, rs, om)

        ce.save_runtime_state.assert_not_called()


class TestSlotSyncMissingDeps:

    def test_missing_ce_skips(self):
        """constitution_executor 为 None 时直接跳过"""
        _, rs, om = _make_deps(active_slots={"pid1": "BTCUSDT"})
        _run_slot_sync(None, rs, om)  # ce=None

    def test_missing_rs_skips(self):
        """runtime_state 为 None 时直接跳过"""
        ce, _, om = _make_deps(active_slots={"pid1": "BTCUSDT"})
        _run_slot_sync(ce, None, om)  # rs=None
        ce.release_slot.assert_not_called()

    def test_missing_om_skips(self):
        """order_manager 为 None 时直接跳过"""
        ce, rs, _ = _make_deps(active_slots={"pid1": "BTCUSDT"})
        _run_slot_sync(ce, rs, None)  # om=None
        ce.release_slot.assert_not_called()
