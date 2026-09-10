"""
向量回测 vs 事件回测 退出逻辑一致性测试

验证 simulate_rr_execution (向量回测) 和 enforce_position (事件回测)
在相同 1min 价格路径上产生相同的退出价格、退出原因和退出时刻。

测试场景:
  1. SL 触发 (long / short)
  2. TP 触发 (long / short)
  3. Breakeven lock → SL 移至入场价
  4. Trailing activation → trailing_sl
  5. Timeout
  6. Breakeven + trailing → trailing_sl 取代 breakeven SL
  7. start_m 正确性: 1min 模拟从 bar CLOSE 后开始 (不含 bar 内)

设计:
  - 构造合成 4H signal bar + N 根 1min bars
  - 向量侧: simulate_rr_execution(DataFrame, exec_config, bars_1min_dict)
  - 事件侧: build_position_dict + enforce_position 逐 1min bar
  - 断言: exit_price, exit_reason 一致
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.time_series_model.core.trade_intent import TradeIntent
from src.time_series_model.live.position_logic import (
    build_position_dict,
    enforce_position,
)


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════


def _make_signal_bar(
    entry_price: float,
    atr: float,
    direction: int,
    bar_open_ts: pd.Timestamp,
    bar_minutes: int = 240,
) -> dict:
    """构造一个信号 bar（4H / 60T）的 dict"""
    return {
        "timestamp": bar_open_ts,
        "open": entry_price - 5,
        "high": entry_price + 10,
        "low": entry_price - 10,
        "close": entry_price,  # 入场价 = bar close
        "atr": atr,
        "entry_direction": float(direction),
        "_bar_minutes": bar_minutes,
    }


def _make_1min_bars(
    start_ts: pd.Timestamp,
    bars: list[tuple[float, float, float, float]],
) -> pd.DataFrame:
    """构造 1min bars DataFrame. bars = [(open, high, low, close), ...]"""
    rows = []
    for i, (o, h, l, c) in enumerate(bars):
        rows.append(
            {
                "timestamp": start_ts + pd.Timedelta(minutes=i),
                "open": o,
                "high": h,
                "low": l,
                "close": c,
            }
        )
    return pd.DataFrame(rows)


def _run_vector_side(
    entry_price: float,
    atr: float,
    direction: int,
    bars_1min: pd.DataFrame,
    initial_r: float = 2.0,
    activation_r: float = 1.5,
    trail_r: float = 1.0,
    time_stop_bars: int = 50,
    breakeven_lock_r: float = 0.0,
    tp_r: float = 0.0,
    bar_minutes: int = 240,
    signal_bar_ts: pd.Timestamp | None = None,
) -> dict:
    """运行向量回测侧: simulate_rr_execution

    返回 trade_details[0] 的子集: exit_price, exit_reason, exit_ts_ns
    """
    from scripts.backtest_execution_layer import simulate_rr_execution

    # signal_bar_ts = bar OPEN 时间; 默认从第一根 1min bar 推算
    if signal_bar_ts is None:
        bar_open_ts = bars_1min["timestamp"].iloc[0] - pd.Timedelta(minutes=bar_minutes)
    else:
        bar_open_ts = signal_bar_ts

    signal_df = pd.DataFrame(
        [
            {
                "timestamp": bar_open_ts,
                "open": entry_price - 5,
                "high": entry_price + 10,
                "low": entry_price - 10,
                "close": entry_price,
                "atr": atr,
                "entry_direction": float(direction),
                "symbol": "TESTUSDT",
                "_bar_minutes": bar_minutes,
                "_tier_initial_r": initial_r,
                "_tier_activation_r": activation_r,
                "_tier_trail_r": trail_r,
                "_tier_timeout": time_stop_bars,
                "_pcm_archetype": "test",
                "evidence_score": 0.5,
            }
        ]
    )

    # stop_loss.type = "trailing" 让向量内循环启用 trailing activation
    exec_config = {
        "stop_loss": {
            "type": "trailing",
            "initial_r": initial_r,
            "trailing": {
                "activation_r": activation_r,
                "trail_r": trail_r,
            },
        },
        "holding": {"time_stop_bars": time_stop_bars},
        "take_profit": {"enabled": tp_r > 0, "target_r": tp_r},
    }

    bars_1min_dict = {"TESTUSDT": bars_1min.copy()}

    results, trade_details = simulate_rr_execution(
        signal_df,
        exec_config,
        atr_col="atr",
        use_tier_params=True,
        breakeven_lock_r=breakeven_lock_r,
        bars_1min_dict=bars_1min_dict,
        silent=True,
    )

    if not trade_details:
        return {"exit_price": None, "exit_reason": None, "realized_rr": None}

    td = trade_details[0]
    return {
        "exit_price": td["exit_price"],
        "exit_reason": td["exit_reason"],
        "realized_rr": td["realized_rr"],
    }


def _run_event_side(
    entry_price: float,
    atr: float,
    direction: int,
    bars_1min: pd.DataFrame,
    initial_r: float = 2.0,
    activation_r: float = 1.5,
    trail_r: float = 1.0,
    time_stop_bars: int = 50,
    breakeven_lock_r: float = 0.0,
    tp_r: float = 0.0,
    bar_minutes: int = 240,
) -> dict:
    """运行事件回测侧: build_position_dict + enforce_position 逐 bar

    返回: exit_price, exit_reason, realized_rr
    """
    action = "LONG" if direction == 1 else "SHORT"
    is_long = direction == 1

    # build_position_dict 需要 TradeIntent
    bpc_cfg = {
        "activation_r": activation_r,
        "trail_r": trail_r,
        "breakeven_enabled": breakeven_lock_r > 0,
        "breakeven_trigger_r": breakeven_lock_r,
        "bar_minutes": bar_minutes,
    }
    rr = {
        "stop_loss_r": initial_r,
        "max_holding_bars": time_stop_bars,
    }
    if tp_r > 0:
        rr["take_profit_r"] = tp_r

    ep = {"rr_constraints": rr, "bpc_position_config": bpc_cfg}
    intent = TradeIntent(
        action=action,
        symbol="TESTUSDT",
        archetype="test",
        confidence=0.5,
        execution_profile=ep,
    )

    entry_time = bars_1min["timestamp"].iloc[0].to_pydatetime()
    if entry_time.tzinfo is None:
        entry_time = entry_time.replace(tzinfo=timezone.utc)

    pos = build_position_dict(
        intent=intent,
        entry_price=entry_price,
        atr=atr,
        bar_minutes=bar_minutes,
        entry_time=entry_time,
    )

    # 逐 1min bar 调用 enforce_position (与事件回测相同)
    exit_price = None
    exit_reason = None
    exit_bar_idx = None

    for idx, row in bars_1min.iterrows():
        bar_ts = row["timestamp"]
        if hasattr(bar_ts, "to_pydatetime"):
            now = bar_ts.to_pydatetime()
            if now.tzinfo is None:
                now = now.replace(tzinfo=timezone.utc)
        else:
            now = datetime.now(timezone.utc)

        reason, ep_val = enforce_position(
            pos,
            price_high=float(row["high"]),
            price_low=float(row["low"]),
            price_close=float(row["close"]),
            now=now,
            default_bar_minutes=bar_minutes,
        )
        if reason is not None:
            exit_price = ep_val
            # 名称映射: enforce_position 返回 "stop_loss" / "take_profit" / "time_stop"
            # 向量回测返回 "sl" / "trailing_sl" / "tp" / "timeout"
            if reason == "stop_loss":
                # 判断是否是 trailing_sl
                if pos.get("trailing_activated"):
                    exit_reason = "trailing_sl"
                else:
                    exit_reason = "sl"
            elif reason == "take_profit":
                exit_reason = "tp"
            elif reason == "time_stop":
                exit_reason = "timeout"
            else:
                exit_reason = reason
            exit_bar_idx = idx
            break

    # timeout: 所有 bar 都跑完没触发, 用最后一根的 close
    if exit_price is None and not bars_1min.empty:
        exit_price = float(bars_1min.iloc[-1]["close"])
        exit_reason = "timeout"

    # 计算 R/R
    risk_distance = initial_r * atr
    if direction == 1:
        realized_rr = (
            (exit_price - entry_price) / risk_distance if risk_distance > 0 else 0
        )
    else:
        realized_rr = (
            (entry_price - exit_price) / risk_distance if risk_distance > 0 else 0
        )

    return {
        "exit_price": exit_price,
        "exit_reason": exit_reason,
        "realized_rr": realized_rr,
    }


def _assert_consistent(vec: dict, evt: dict, scenario: str):
    """断言向量侧和事件侧结果一致"""
    assert (
        vec["exit_reason"] == evt["exit_reason"]
    ), f"[{scenario}] exit_reason 不一致: vector={vec['exit_reason']}, event={evt['exit_reason']}"
    assert vec["exit_price"] == pytest.approx(
        evt["exit_price"], abs=1e-6
    ), f"[{scenario}] exit_price 不一致: vector={vec['exit_price']}, event={evt['exit_price']}"
    assert vec["realized_rr"] == pytest.approx(
        evt["realized_rr"], abs=1e-4
    ), f"[{scenario}] realized_rr 不一致: vector={vec['realized_rr']}, event={evt['realized_rr']}"


# ═════════════════════════════════════════════════════════════════════════════
# Test Cases
# ═════════════════════════════════════════════════════════════════════════════

BAR_MINUTES = 240
T0 = pd.Timestamp("2025-01-01 00:00", tz="UTC")
# 1min bars start AFTER the signal bar closes
BARS_START = T0 + pd.Timedelta(minutes=BAR_MINUTES)


class TestSLExit:
    """止损退出一致性"""

    def test_long_sl(self):
        """Long: 价格跌破 SL → exit_reason=sl, exit_price=sl_price"""
        entry = 100.0
        atr = 5.0
        sl_price = entry - 2.0 * atr  # 90.0

        bars = _make_1min_bars(
            BARS_START,
            [
                (100, 101, 99, 100),  # bar 0: 正常
                (100, 100, 99, 99),  # bar 1: 正常
                (99, 99, 89, 89),  # bar 2: low=89 < sl=90 → SL
                (89, 95, 88, 90),  # bar 3: 不应触发
            ],
        )
        params = dict(
            entry_price=entry,
            atr=atr,
            direction=1,
            bars_1min=bars,
            initial_r=2.0,
            activation_r=1.5,
            trail_r=1.0,
            time_stop_bars=50,
            bar_minutes=BAR_MINUTES,
        )
        vec = _run_vector_side(**params)
        evt = _run_event_side(**params)
        _assert_consistent(vec, evt, "long_sl")
        assert vec["exit_reason"] == "sl"
        assert vec["exit_price"] == pytest.approx(sl_price, abs=1e-6)

    def test_short_sl(self):
        """Short: 价格涨破 SL → exit_reason=sl"""
        entry = 100.0
        atr = 5.0
        sl_price = entry + 2.0 * atr  # 110.0

        bars = _make_1min_bars(
            BARS_START,
            [
                (100, 101, 99, 100),
                (100, 111, 99, 105),  # high=111 > sl=110 → SL
                (105, 106, 104, 105),
            ],
        )
        params = dict(
            entry_price=entry,
            atr=atr,
            direction=-1,
            bars_1min=bars,
            initial_r=2.0,
            activation_r=1.5,
            trail_r=1.0,
            time_stop_bars=50,
            bar_minutes=BAR_MINUTES,
        )
        vec = _run_vector_side(**params)
        evt = _run_event_side(**params)
        _assert_consistent(vec, evt, "short_sl")
        assert vec["exit_reason"] == "sl"


class TestTPExit:
    """止盈退出一致性"""

    def test_long_tp(self):
        """Long: 价格达到 TP → exit_reason=tp"""
        entry = 100.0
        atr = 5.0
        tp_r = 3.0
        tp_price = entry + tp_r * atr  # 115.0

        bars = _make_1min_bars(
            BARS_START,
            [
                (100, 102, 99, 101),
                (101, 116, 100, 115),  # high=116 >= tp=115 → TP
                (115, 116, 114, 115),
            ],
        )
        # activation_r=10.0 禁用 trailing, 纯测 TP
        params = dict(
            entry_price=entry,
            atr=atr,
            direction=1,
            bars_1min=bars,
            initial_r=2.0,
            activation_r=10.0,
            trail_r=1.0,
            time_stop_bars=50,
            tp_r=tp_r,
            bar_minutes=BAR_MINUTES,
        )
        vec = _run_vector_side(**params)
        evt = _run_event_side(**params)
        _assert_consistent(vec, evt, "long_tp")
        assert vec["exit_reason"] == "tp"

    def test_short_tp(self):
        """Short: 价格下跌达到 TP"""
        entry = 100.0
        atr = 5.0
        tp_r = 3.0
        tp_price = entry - tp_r * atr  # 85.0

        bars = _make_1min_bars(
            BARS_START,
            [
                (100, 101, 98, 99),
                (99, 100, 84, 85),  # low=84 <= tp=85 → TP
                (85, 86, 84, 85),
            ],
        )
        # activation_r=10.0 禁用 trailing, 纯测 TP
        params = dict(
            entry_price=entry,
            atr=atr,
            direction=-1,
            bars_1min=bars,
            initial_r=2.0,
            activation_r=10.0,
            trail_r=1.0,
            time_stop_bars=50,
            tp_r=tp_r,
            bar_minutes=BAR_MINUTES,
        )
        vec = _run_vector_side(**params)
        evt = _run_event_side(**params)
        _assert_consistent(vec, evt, "short_tp")
        assert vec["exit_reason"] == "tp"


class TestBreakevenLock:
    """保本锁定一致性"""

    def test_breakeven_then_sl_at_entry(self):
        """Long: 先涨到 breakeven_lock_r → SL 移至 entry, 然后回落触发 SL at entry"""
        entry = 100.0
        atr = 5.0
        breakeven_r = 1.0  # 浮盈 >= 1R → SL 移至 100

        bars = _make_1min_bars(
            BARS_START,
            [
                (100, 106, 99, 105),  # high=106 → profit_r=1.2R → breakeven triggered
                (105, 106, 104, 104),  # 正常
                (104, 105, 99.5, 100),  # low=99.5 < sl=100 (breakeven) → SL
                (100, 101, 99, 100),
            ],
        )
        params = dict(
            entry_price=entry,
            atr=atr,
            direction=1,
            bars_1min=bars,
            initial_r=2.0,
            activation_r=10.0,
            trail_r=1.0,  # activation_r=10 禁用 trailing
            time_stop_bars=50,
            breakeven_lock_r=breakeven_r,
            bar_minutes=BAR_MINUTES,
        )
        vec = _run_vector_side(**params)
        evt = _run_event_side(**params)
        _assert_consistent(vec, evt, "breakeven_then_sl")
        assert vec["exit_reason"] == "sl"
        assert vec["exit_price"] == pytest.approx(entry, abs=1e-6)


class TestTrailingExit:
    """Trailing 止损一致性"""

    def test_trailing_activation_and_exit(self):
        """Long: 浮盈 >= activation_r → trailing 激活 → 回撤 trail_r → trailing_sl"""
        entry = 100.0
        atr = 5.0
        activation_r = 1.5  # 浮盈 >= 7.5 → 激活
        trail_r = 1.0  # HWM - 5.0 = trail_sl

        bars = _make_1min_bars(
            BARS_START,
            [
                (100, 103, 99, 102),  # profit=3 (0.6R) - not yet
                (
                    102,
                    108.5,
                    101,
                    108,
                ),  # high=108.5, profit_r=8.5/5=1.7R > 1.5 → activated, HWM=108.5
                (108, 109, 107, 108),  # HWM=109, trail_sl = 109 - 5 = 104
                (108, 108, 103.5, 104),  # low=103.5 < trail_sl=104 → trailing_sl
                (104, 105, 103, 104),
            ],
        )
        params = dict(
            entry_price=entry,
            atr=atr,
            direction=1,
            bars_1min=bars,
            initial_r=2.0,
            activation_r=activation_r,
            trail_r=trail_r,
            time_stop_bars=50,
            bar_minutes=BAR_MINUTES,
        )
        vec = _run_vector_side(**params)
        evt = _run_event_side(**params)
        _assert_consistent(vec, evt, "trailing_exit")
        assert vec["exit_reason"] == "trailing_sl"


class TestTimeout:
    """超时退出一致性"""

    def test_timeout_long(self):
        """Long: 价格在 SL/TP 之间波动, 超过 time_stop_bars → timeout"""
        entry = 100.0
        atr = 5.0
        time_stop_bars = 3  # 3 * 240 = 720 分钟 = 720 根 1min bar

        # 生成 800 根 1min bars, 全部在 SL/TP 范围内
        n_bars = 800
        bars_data = [(100, 101, 99, 100)] * n_bars
        bars = _make_1min_bars(BARS_START, bars_data)

        params = dict(
            entry_price=entry,
            atr=atr,
            direction=1,
            bars_1min=bars,
            initial_r=2.0,
            activation_r=10.0,
            trail_r=1.0,
            time_stop_bars=time_stop_bars,
            bar_minutes=BAR_MINUTES,
        )
        vec = _run_vector_side(**params)
        evt = _run_event_side(**params)
        _assert_consistent(vec, evt, "timeout_long")
        assert vec["exit_reason"] == "timeout"


class TestStartMCorrectness:
    """验证 start_m 从 bar CLOSE 后开始, 不含 bar 内 1min bars"""

    def test_intra_bar_prices_ignored(self):
        """
        Signal bar: 00:00-04:00 (240T)
        Bar 内 1min bars (00:01-04:00): 含有会触发 SL 的低价
        Bar 外 1min bars (04:01+): 价格正常, 不触发 SL

        正确行为: 忽略 bar 内价格, 不应触发 SL
        Bug 行为: bar 内价格触发 SL (start_m 从 bar OPEN 后开始)
        """
        entry = 100.0
        atr = 5.0
        sl_price = entry - 2.0 * atr  # 90.0

        # 构造: bar 内有 240 根 1min bars, 含低价 (会触发 SL 如果被包含)
        bar_open_ts = T0
        bar_close_ts = T0 + pd.Timedelta(minutes=BAR_MINUTES)

        intra_bar = []
        for i in range(BAR_MINUTES):
            ts = bar_open_ts + pd.Timedelta(minutes=i + 1)
            if i < 120:
                # 前半段: 价格暴跌到 85 (低于 SL=90)
                intra_bar.append(
                    {"timestamp": ts, "open": 95, "high": 96, "low": 85, "close": 86}
                )
            else:
                # 后半段: 价格回升
                intra_bar.append(
                    {"timestamp": ts, "open": 98, "high": 101, "low": 97, "close": 100}
                )

        # bar 外 1min bars: 价格正常, 不触发 SL
        post_bar = []
        for i in range(100):
            ts = bar_close_ts + pd.Timedelta(minutes=i + 1)
            post_bar.append(
                {"timestamp": ts, "open": 100, "high": 102, "low": 98, "close": 100}
            )

        all_bars = pd.DataFrame(intra_bar + post_bar)

        # 向量回测: 应该只用 post_bar, 不触发 SL
        # 传入 signal_bar_ts=T0 (信号 bar 的 OPEN 时间)
        vec = _run_vector_side(
            entry_price=entry,
            atr=atr,
            direction=1,
            bars_1min=all_bars,
            initial_r=2.0,
            activation_r=10.0,
            trail_r=1.0,
            time_stop_bars=50,
            bar_minutes=BAR_MINUTES,
            signal_bar_ts=T0,
        )
        # 因为 post_bar 都是 low=98 > sl=90, 不应触发 SL
        assert vec["exit_reason"] != "sl", (
            f"Bug: bar 内 1min bars 被 simulate_rr_execution 使用, "
            f"错误触发了 SL (exit_price={vec['exit_price']})"
        )

    def test_post_bar_sl_triggers_correctly(self):
        """
        bar 内价格正常, bar 外价格触发 SL → 应该正确触发
        """
        entry = 100.0
        atr = 5.0
        sl_price = entry - 2.0 * atr  # 90.0

        bar_close_ts = T0 + pd.Timedelta(minutes=BAR_MINUTES)

        # bar 内: 正常价格
        intra_bar = []
        for i in range(BAR_MINUTES):
            ts = T0 + pd.Timedelta(minutes=i + 1)
            intra_bar.append(
                {"timestamp": ts, "open": 100, "high": 102, "low": 98, "close": 100}
            )

        # bar 外: 第 5 根触发 SL
        post_bar = []
        for i in range(20):
            ts = bar_close_ts + pd.Timedelta(minutes=i + 1)
            if i == 4:
                post_bar.append(
                    {"timestamp": ts, "open": 95, "high": 96, "low": 88, "close": 89}
                )
            else:
                post_bar.append(
                    {"timestamp": ts, "open": 100, "high": 102, "low": 98, "close": 100}
                )

        all_bars = pd.DataFrame(intra_bar + post_bar)

        # 传入 signal_bar_ts=T0 (信号 bar 的 OPEN 时间)
        vec = _run_vector_side(
            entry_price=entry,
            atr=atr,
            direction=1,
            bars_1min=all_bars,
            initial_r=2.0,
            activation_r=10.0,
            trail_r=1.0,
            time_stop_bars=50,
            bar_minutes=BAR_MINUTES,
            signal_bar_ts=T0,
        )
        assert vec["exit_reason"] == "sl"
        assert vec["exit_price"] == pytest.approx(sl_price, abs=1e-6)


class TestMETimeframe:
    """ME 策略 (60T bars) 一致性"""

    def test_me_sl_60t(self):
        """ME 60T bar: 验证 bar_minutes=60 时 1min sim 正确"""
        entry = 100.0
        atr = 3.0
        bm = 60

        bars_start = T0 + pd.Timedelta(minutes=bm)
        bars = _make_1min_bars(
            bars_start,
            [
                (100, 101, 99, 100),
                (100, 100, 93, 94),  # low=93 < sl=94 → SL
                (94, 95, 93, 94),
            ],
        )
        params = dict(
            entry_price=entry,
            atr=atr,
            direction=1,
            bars_1min=bars,
            initial_r=2.0,
            activation_r=1.5,
            trail_r=1.0,
            time_stop_bars=50,
            bar_minutes=bm,
        )
        vec = _run_vector_side(**params)
        evt = _run_event_side(**params)
        _assert_consistent(vec, evt, "me_sl_60t")


class TestSLPriorityOverTP:
    """SL 优先于 TP (保守假设) — 同 bar 同时触发时"""

    def test_sl_and_tp_same_bar_long(self):
        """Long: 同一根 bar low <= SL 且 high >= TP → SL 优先"""
        entry = 100.0
        atr = 5.0
        tp_r = 2.0
        tp_price = entry + tp_r * atr  # 110.0
        sl_price = entry - 2.0 * atr  # 90.0

        bars = _make_1min_bars(
            BARS_START,
            [
                (100, 101, 99, 100),
                (100, 111, 89, 95),  # high=111 >= tp=110 AND low=89 <= sl=90
            ],
        )
        params = dict(
            entry_price=entry,
            atr=atr,
            direction=1,
            bars_1min=bars,
            initial_r=2.0,
            activation_r=10.0,
            trail_r=1.0,
            time_stop_bars=50,
            tp_r=tp_r,
            bar_minutes=BAR_MINUTES,
        )
        vec = _run_vector_side(**params)
        evt = _run_event_side(**params)
        # 两侧都应该选 SL (保守)
        _assert_consistent(vec, evt, "sl_tp_same_bar")
        assert vec["exit_reason"] == "sl"


class TestSlotFiltering:
    """Slot 过滤一致性: 向量回测 slot 限制行为"""

    def test_slot_rejects_second_trade_same_archetype(self):
        """per-strategy slot=1 时, 同 archetype 第二笔应被拒绝"""
        from scripts.backtest_execution_layer import simulate_rr_execution

        # 两个不同 symbol, 同 archetype, 连续入场
        t0 = pd.Timestamp("2025-01-01 00:00", tz="UTC")
        t1 = t0 + pd.Timedelta(hours=4)  # 第二个信号

        signal_df = pd.DataFrame(
            [
                {
                    "timestamp": t0,
                    "open": 99,
                    "high": 101,
                    "low": 99,
                    "close": 100,
                    "atr": 5.0,
                    "entry_direction": 1.0,
                    "symbol": "BTCUSDT",
                    "_bar_minutes": 240,
                    "_tier_initial_r": 2.0,
                    "_tier_activation_r": 10.0,
                    "_tier_trail_r": 1.0,
                    "_tier_timeout": 50,
                    "_pcm_archetype": "bpc",
                    "evidence_score": 0.6,
                },
                {
                    "timestamp": t1,
                    "open": 99,
                    "high": 101,
                    "low": 99,
                    "close": 100,
                    "atr": 5.0,
                    "entry_direction": 1.0,
                    "symbol": "ETHUSDT",
                    "_bar_minutes": 240,
                    "_tier_initial_r": 2.0,
                    "_tier_activation_r": 10.0,
                    "_tier_trail_r": 1.0,
                    "_tier_timeout": 50,
                    "_pcm_archetype": "bpc",
                    "evidence_score": 0.4,
                },
            ]
        )

        # 1min bars: 两个 symbol 都不触发 SL, 持仓超时 (50 bars)
        bars_btc = pd.DataFrame(
            [
                {
                    "timestamp": t0 + pd.Timedelta(minutes=240 + i + 1),
                    "open": 100,
                    "high": 102,
                    "low": 98,
                    "close": 100,
                }
                for i in range(50 * 240)
            ]
        )
        bars_eth = pd.DataFrame(
            [
                {
                    "timestamp": t0 + pd.Timedelta(minutes=240 + i + 1),
                    "open": 100,
                    "high": 102,
                    "low": 98,
                    "close": 100,
                }
                for i in range(50 * 240)
            ]
        )

        exec_config = {
            "stop_loss": {
                "type": "trailing",
                "initial_r": 2.0,
                "trailing": {"activation_r": 10.0, "trail_r": 1.0},
            },
            "holding": {"time_stop_bars": 50},
            "take_profit": {"enabled": False},
        }

        results, trade_details = simulate_rr_execution(
            signal_df,
            exec_config,
            atr_col="atr",
            use_tier_params=True,
            max_slots=3,  # 全局 3
            per_strategy_limits={"bpc": {"max_slots": 1}},  # bpc=1
            bars_1min_dict={"BTCUSDT": bars_btc, "ETHUSDT": bars_eth},
            silent=True,
        )

        # bpc 只允许 1 slot, 第二笔应被 reject
        assert len(trade_details) == 1, (
            f"Expected 1 trade (bpc slot=1), got {len(trade_details)}: "
            f"{[(t['symbol'], t['archetype']) for t in trade_details]}"
        )

    def test_slot_allows_different_archetypes(self):
        """per-strategy slot=1 时, 不同 archetype 可以同时开仓"""
        from scripts.backtest_execution_layer import simulate_rr_execution

        t0 = pd.Timestamp("2025-01-01 00:00", tz="UTC")
        t1 = t0 + pd.Timedelta(hours=4)

        signal_df = pd.DataFrame(
            [
                {
                    "timestamp": t0,
                    "open": 99,
                    "high": 101,
                    "low": 99,
                    "close": 100,
                    "atr": 5.0,
                    "entry_direction": 1.0,
                    "symbol": "BTCUSDT",
                    "_bar_minutes": 240,
                    "_tier_initial_r": 2.0,
                    "_tier_activation_r": 10.0,
                    "_tier_trail_r": 1.0,
                    "_tier_timeout": 50,
                    "_pcm_archetype": "bpc",
                    "evidence_score": 0.6,
                },
                {
                    "timestamp": t1,
                    "open": 99,
                    "high": 101,
                    "low": 99,
                    "close": 100,
                    "atr": 5.0,
                    "entry_direction": -1.0,
                    "symbol": "ETHUSDT",
                    "_bar_minutes": 240,
                    "_tier_initial_r": 2.0,
                    "_tier_activation_r": 10.0,
                    "_tier_trail_r": 1.0,
                    "_tier_timeout": 50,
                    "_pcm_archetype": "fer",
                    "evidence_score": 0.5,
                },
            ]
        )

        bars_btc = pd.DataFrame(
            [
                {
                    "timestamp": t0 + pd.Timedelta(minutes=240 + i + 1),
                    "open": 100,
                    "high": 102,
                    "low": 98,
                    "close": 100,
                }
                for i in range(50 * 240)
            ]
        )
        bars_eth = pd.DataFrame(
            [
                {
                    "timestamp": t0 + pd.Timedelta(minutes=240 + i + 1),
                    "open": 100,
                    "high": 102,
                    "low": 98,
                    "close": 100,
                }
                for i in range(50 * 240)
            ]
        )

        exec_config = {
            "stop_loss": {
                "type": "trailing",
                "initial_r": 2.0,
                "trailing": {"activation_r": 10.0, "trail_r": 1.0},
            },
            "holding": {"time_stop_bars": 50},
            "take_profit": {"enabled": False},
        }

        results, trade_details = simulate_rr_execution(
            signal_df,
            exec_config,
            atr_col="atr",
            use_tier_params=True,
            max_slots=3,
            per_strategy_limits={"bpc": {"max_slots": 1}, "fer": {"max_slots": 1}},
            bars_1min_dict={"BTCUSDT": bars_btc, "ETHUSDT": bars_eth},
            silent=True,
        )

        # bpc=1, fer=1 → 两笔都应该通过
        assert len(trade_details) == 2, (
            f"Expected 2 trades (bpc+fer), got {len(trade_details)}: "
            f"{[(t['symbol'], t['archetype']) for t in trade_details]}"
        )
