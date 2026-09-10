"""
position_logic 共享模块单元测试

覆盖:
  1. build_position_dict: 基础构建 / BPC 扩展 / 通用 trailing / 默认值
  2. enforce_position: 7 步持仓管理
     - time_stop
     - breakeven_lock
     - HWM/LWM 更新
     - activation trailing
     - SL hit (LONG/SHORT)
     - TP hit (LONG/SHORT)
     - SL 优先于 TP (同 bar 同时触发)
     - 持仓继续 (无触发)
  3. 实盘 vs 回测调用方式: 实盘传同一 price, 回测传 OHLC
"""

from __future__ import annotations

import pytest
from datetime import datetime, timedelta, timezone

from src.time_series_model.core.trade_intent import TradeIntent
from src.time_series_model.live.position_logic import (
    build_position_dict,
    enforce_position,
)


# ─── helpers ───


def _make_intent(
    action="LONG",
    symbol="BTCUSDT",
    archetype="bpc",
    confidence=0.75,
    stop_loss_r=2.0,
    take_profit_r=2.5,
    max_holding_bars=50,
    allow_trailing=False,
    trailing_atr=None,
    bpc_position_config=None,
    strategy_specific=None,
    structural_sl=None,
    activation_r=None,
    trail_r=None,
    extra_rr=None,
) -> TradeIntent:
    """构造一个带 execution_profile 的 TradeIntent"""
    rr = {
        "stop_loss_r": stop_loss_r,
        "take_profit_r": take_profit_r,
        "max_holding_bars": max_holding_bars,
        "allow_trailing": allow_trailing,
    }
    if trailing_atr is not None:
        rr["trailing_atr"] = trailing_atr
    if structural_sl is not None:
        rr["structural_sl"] = structural_sl
    if activation_r is not None:
        rr["activation_r"] = activation_r
    if trail_r is not None:
        rr["trailing_atr"] = trail_r
    if extra_rr is not None:
        rr.update(extra_rr)
    ep = {"rr_constraints": rr}
    if bpc_position_config is not None:
        ep["bpc_position_config"] = bpc_position_config
    if strategy_specific is not None:
        ep["strategy_specific"] = strategy_specific
    return TradeIntent(
        action=action,
        symbol=symbol,
        archetype=archetype,
        confidence=confidence,
        execution_profile=ep,
    )


def _now():
    return datetime(2025, 6, 1, 12, 0, 0, tzinfo=timezone.utc)


# ═════════════════════════════════════════════════════════════════════════════
# build_position_dict tests
# ═════════════════════════════════════════════════════════════════════════════


class TestBuildPositionDict:

    def test_basic_long(self):
        intent = _make_intent(action="LONG", stop_loss_r=2.0, take_profit_r=3.0)
        pos = build_position_dict(
            intent, entry_price=50000, atr=500, bar_minutes=240, entry_time=_now()
        )
        assert pos["side"] == "LONG"
        assert pos["symbol"] == "BTCUSDT"
        assert pos["entry_price"] == 50000
        assert pos["atr_at_entry"] == 500
        assert pos["bar_minutes"] == 240
        # SL = entry - 2.0 * atr = 50000 - 1000 = 49000
        assert pos["stop_loss_price"] == pytest.approx(49000, abs=1)
        # TP = entry + 3.0 * atr = 50000 + 1500 = 51500
        assert pos["take_profit_price"] == pytest.approx(51500, abs=1)
        assert pos["max_holding_bars"] == 50
        assert pos["bars_counted"] == 0

    def test_basic_short(self):
        intent = _make_intent(action="SHORT", stop_loss_r=1.5, take_profit_r=2.0)
        pos = build_position_dict(
            intent, entry_price=3000, atr=100, bar_minutes=60, entry_time=_now()
        )
        assert pos["side"] == "SHORT"
        # SHORT SL = entry + 1.5 * atr = 3000 + 150 = 3150
        assert pos["stop_loss_price"] == pytest.approx(3150, abs=1)
        # SHORT TP = entry - 2.0 * atr = 3000 - 200 = 2800
        assert pos["take_profit_price"] == pytest.approx(2800, abs=1)

    def test_crf_box_edge_long_uses_box_stop_and_opposite_tp(self):
        intent = _make_intent(
            action="LONG",
            archetype="crf",
            stop_loss_r=1.5,
            take_profit_r=1.2,
            extra_rr={
                "stop_loss_type": "box_edge",
                "take_profit_type": "opposite_edge",
                "box_hi_120": 120.0,
                "box_lo_120": 100.0,
                "box_stop_buffer_frac": 0.25,
                "box_target_edge_frac": 0.15,
            },
        )
        pos = build_position_dict(
            intent, entry_price=104.0, atr=2.0, bar_minutes=120, entry_time=_now()
        )

        assert pos["stop_loss_price"] == pytest.approx(95.0)
        assert pos["take_profit_price"] == pytest.approx(117.0)
        assert pos["sizing_stop_source"] == "box_edge"
        assert pos["initial_risk_distance"] == pytest.approx(9.0)

    def test_crf_box_edge_short_uses_box_stop_and_opposite_tp(self):
        intent = _make_intent(
            action="SHORT",
            archetype="crf",
            stop_loss_r=1.5,
            take_profit_r=1.2,
            extra_rr={
                "stop_loss_type": "box_edge",
                "take_profit_type": "opposite_edge",
                "box_hi_120": 120.0,
                "box_lo_120": 100.0,
                "box_stop_buffer_frac": 0.25,
                "box_target_edge_frac": 0.15,
            },
        )
        pos = build_position_dict(
            intent, entry_price=116.0, atr=2.0, bar_minutes=120, entry_time=_now()
        )

        assert pos["stop_loss_price"] == pytest.approx(125.0)
        assert pos["take_profit_price"] == pytest.approx(103.0)
        assert pos["sizing_stop_source"] == "box_edge"
        assert pos["initial_risk_distance"] == pytest.approx(9.0)

    def test_no_sl_tp_when_atr_zero(self):
        intent = _make_intent(stop_loss_r=2.0, take_profit_r=2.5)
        pos = build_position_dict(
            intent, entry_price=50000, atr=0, bar_minutes=240, entry_time=_now()
        )
        assert pos["stop_loss_price"] is None
        assert pos["take_profit_price"] is None

    def test_no_sl_tp_when_rr_zero(self):
        intent = _make_intent(stop_loss_r=0, take_profit_r=0)
        pos = build_position_dict(
            intent, entry_price=50000, atr=500, bar_minutes=240, entry_time=_now()
        )
        assert pos["stop_loss_price"] is None
        assert pos["take_profit_price"] is None

    def test_bpc_trailing_config(self):
        bpc_cfg = {
            "activation_r": 1.0,
            "trail_r": 0.8,
            "breakeven_enabled": True,
            "breakeven_trigger_r": 0.5,
            "bar_minutes": 240,
        }
        intent = _make_intent(bpc_position_config=bpc_cfg)
        pos = build_position_dict(
            intent, entry_price=50000, atr=500, bar_minutes=240, entry_time=_now()
        )
        assert pos["activation_r"] == 1.0
        assert pos["trail_r"] == 0.8
        assert pos["trailing_activated"] is False
        assert pos["breakeven_enabled"] is True
        assert pos["breakeven_trigger_r"] == 0.5
        assert pos["breakeven_locked"] is False
        assert pos["high_water_mark"] == 50000  # LONG → HWM = entry

    def test_generic_trailing(self):
        intent = _make_intent(allow_trailing=True, trailing_atr=1.5)
        pos = build_position_dict(
            intent, entry_price=50000, atr=500, bar_minutes=240, entry_time=_now()
        )
        assert pos["activation_r"] == 1.5
        assert pos["trail_r"] == 1.5
        assert pos["breakeven_enabled"] is False

    def test_strategy_specific_unknown_keys_are_not_promoted_to_top_level(self):
        """strategy_specific 里除 SRB 等白名单字段外不会自动变成仓位顶层字段。"""
        intent = _make_intent(strategy_specific={"legacy_tier_note": "x"})
        pos = build_position_dict(
            intent, entry_price=50000, atr=500, bar_minutes=240, entry_time=_now()
        )
        assert "legacy_tier_note" not in pos

    def test_entry_time_preserved(self):
        t = datetime(2025, 1, 15, 8, 0, 0, tzinfo=timezone.utc)
        intent = _make_intent()
        pos = build_position_dict(
            intent, entry_price=50000, atr=500, bar_minutes=240, entry_time=t
        )
        assert pos["entry_time"] == t

    def test_entry_time_default(self):
        intent = _make_intent()
        pos = build_position_dict(intent, entry_price=50000, atr=500)
        assert isinstance(pos["entry_time"], datetime)
        assert pos["entry_time"].tzinfo is not None

    def test_initial_risk_distance(self):
        """initial_risk_distance = stop_loss_r * atr (用于 R-multiple 归一化)"""
        intent = _make_intent(stop_loss_r=2.0, take_profit_r=3.0)
        pos = build_position_dict(
            intent, entry_price=50000, atr=500, bar_minutes=240, entry_time=_now()
        )
        assert pos["initial_risk_distance"] == pytest.approx(1000)  # 2.0 * 500

    def test_initial_risk_distance_fallback_to_atr(self):
        """stop_loss_r=0 时 initial_risk_distance 退化为 raw ATR"""
        intent = _make_intent(stop_loss_r=0, take_profit_r=0)
        pos = build_position_dict(
            intent, entry_price=50000, atr=500, bar_minutes=240, entry_time=_now()
        )
        assert pos["initial_risk_distance"] == pytest.approx(500)

    def test_activation_r_from_rr_constraints(self):
        """rr_constraints.activation_r 优先于 trailing_atr 作为 activation_r"""
        rr = {
            "stop_loss_r": 2.0,
            "take_profit_r": 3.0,
            "max_holding_bars": 50,
            "allow_trailing": True,
            "activation_r": 1.5,
            "trailing_atr": 0.5,
        }
        ep = {"rr_constraints": rr}
        intent = TradeIntent(
            action="LONG",
            symbol="BTCUSDT",
            archetype="test",
            confidence=0.75,
            execution_profile=ep,
        )
        pos = build_position_dict(
            intent, entry_price=50000, atr=500, bar_minutes=240, entry_time=_now()
        )
        # activation_r 应该取 rr_constraints.activation_r (1.5)，不是 trailing_atr (0.5)
        assert pos["activation_r"] == pytest.approx(1.5)
        # trail_r 应该取 trailing_atr (0.5)
        assert pos["trail_r"] == pytest.approx(0.5)

    def test_stop_guardrail_clips_with_max_stop_pct(self):
        rr = {
            "stop_loss_r": 3.5,  # atr_stop_pct=3.5%
            "take_profit_r": 0.0,
            "max_holding_bars": 50,
            "max_stop_pct": 0.02,
        }
        ep = {"rr_constraints": rr}
        intent = TradeIntent(
            action="LONG",
            symbol="BTCUSDT",
            archetype="bpc",
            confidence=0.8,
            execution_profile=ep,
        )
        pos = build_position_dict(
            intent, entry_price=10000, atr=100, bar_minutes=240, entry_time=_now()
        )
        assert pos["atr_stop_pct"] == pytest.approx(0.035)
        assert pos["effective_stop_pct"] == pytest.approx(0.02)
        assert pos["sizing_stop_source"] == "guardrail_clip"
        # effective_stop_r = 0.02*10000/100 = 2.0 -> risk_distance=200
        assert pos["initial_risk_distance"] == pytest.approx(200)

    def test_stop_guardrail_uses_atr_when_not_clipped(self):
        rr = {
            "stop_loss_r": 2.0,
            "take_profit_r": 0.0,
            "max_holding_bars": 50,
            "min_stop_pct": 0.005,
            "max_stop_pct": 0.03,
        }
        ep = {"rr_constraints": rr}
        intent = TradeIntent(
            action="LONG",
            symbol="BTCUSDT",
            archetype="me",
            confidence=0.8,
            execution_profile=ep,
        )
        pos = build_position_dict(
            intent, entry_price=10000, atr=100, bar_minutes=240, entry_time=_now()
        )
        assert pos["atr_stop_pct"] == pytest.approx(0.02)
        assert pos["effective_stop_pct"] == pytest.approx(0.02)
        assert pos["sizing_stop_source"] == "atr"


# ═════════════════════════════════════════════════════════════════════════════
# enforce_position tests
# ═════════════════════════════════════════════════════════════════════════════


class TestEnforcePositionTimeStop:

    def test_time_stop_triggers(self):
        """持仓超过 max_holding_bars * bar_minutes 后应触发 time_stop"""
        entry_time = _now()
        pos = {
            "side": "LONG",
            "entry_price": 50000,
            "entry_time": entry_time,
            "atr_at_entry": 500,
            "max_holding_bars": 10,
            "bar_minutes": 240,
            "stop_loss_price": 49000,
            "take_profit_price": 51500,
        }
        # 10 bars * 240 min = 2400 min = 40 hours
        future = entry_time + timedelta(hours=41)
        reason, exit_price = enforce_position(
            pos,
            price_high=50100,
            price_low=49900,
            price_close=50050,
            now=future,
        )
        assert reason == "time_stop"
        assert exit_price == 50050  # time stop 用 close

    def test_no_time_stop_within_limit(self):
        entry_time = _now()
        pos = {
            "side": "LONG",
            "entry_price": 50000,
            "entry_time": entry_time,
            "atr_at_entry": 500,
            "max_holding_bars": 10,
            "bar_minutes": 240,
            "stop_loss_price": 49000,
            "take_profit_price": 51500,
        }
        within = entry_time + timedelta(hours=20)
        reason, _ = enforce_position(
            pos,
            price_high=50100,
            price_low=49900,
            price_close=50050,
            now=within,
        )
        assert reason is None

    def test_time_stop_uncap_when_mfe_above_threshold(self):
        """E1: 母仓 MFE ≥ time_stop_uncap_mfe_r × R 时，time_stop 跳过（趋势在跑）"""
        entry_time = _now()
        pos = {
            "side": "LONG",
            "entry_price": 50000,
            "entry_time": entry_time,
            "atr_at_entry": 500,
            "max_holding_bars": 10,
            "bar_minutes": 240,
            "stop_loss_price": 49000,
            "initial_risk_distance": 1000,  # 1R = 1000
            "time_stop_uncap_mfe_r": 2.0,
            "breakeven_measure": "initial_risk",
            "high_water_mark": 53000,  # MFE = 3R ≥ 2R → uncap
        }
        future = entry_time + timedelta(hours=100)
        reason, _ = enforce_position(
            pos,
            price_high=52500,
            price_low=52000,
            price_close=52200,
            now=future,
        )
        assert reason is None  # time_stop 被 uncap 跳过

    def test_time_stop_triggers_when_mfe_below_uncap(self):
        """E1: MFE < uncap 阈值时保持原 time_stop 行为"""
        entry_time = _now()
        pos = {
            "side": "LONG",
            "entry_price": 50000,
            "entry_time": entry_time,
            "atr_at_entry": 500,
            "max_holding_bars": 10,
            "bar_minutes": 240,
            "stop_loss_price": 49000,
            "initial_risk_distance": 1000,
            "time_stop_uncap_mfe_r": 2.0,
            "breakeven_measure": "initial_risk",
            "high_water_mark": 50500,  # MFE = 0.5R < 2R
        }
        future = entry_time + timedelta(hours=100)
        reason, exit_price = enforce_position(
            pos,
            price_high=50100,
            price_low=49900,
            price_close=50050,
            now=future,
        )
        assert reason == "time_stop"

    def test_l3_structural_exit_long_breaks_lower_l3(self):
        """E2: LONG 价格跌破 wide_sr_lower_px - buffer × ATR 时触发 L3 结构退出"""
        pos = {
            "side": "LONG",
            "entry_price": 50000,
            "entry_time": _now(),
            "atr_at_entry": 500,
            "stop_loss_price": 48000,
            "l3_structural_exit_enabled": True,
            "l3_structural_exit_buffer_atr": 0.25,
        }
        # wide_sr_lower_px = 49000；buffer = 0.25 × 500 = 125；threshold = 48875
        reason, exit_price = enforce_position(
            pos,
            price_high=49500,
            price_low=48500,
            price_close=48800,  # < 48875 → exit
            now=_now() + timedelta(hours=10),
            wide_sr_upper_px=55000,
            wide_sr_lower_px=49000,
        )
        assert reason == "structural_exit_l3"
        assert exit_price == 48800

    def test_l3_structural_exit_short_breaks_upper_l3(self):
        """E2: SHORT 价格涨破 wide_sr_upper_px + buffer × ATR 时触发"""
        pos = {
            "side": "SHORT",
            "entry_price": 50000,
            "entry_time": _now(),
            "atr_at_entry": 500,
            "stop_loss_price": 52000,
            "l3_structural_exit_enabled": True,
            "l3_structural_exit_buffer_atr": 0.25,
        }
        # wide_sr_upper_px = 51000；buffer = 125；threshold = 51125
        reason, exit_price = enforce_position(
            pos,
            price_high=51500,
            price_low=50800,
            price_close=51200,  # > 51125 → exit
            now=_now() + timedelta(hours=10),
            wide_sr_upper_px=51000,
            wide_sr_lower_px=45000,
        )
        assert reason == "structural_exit_l3"

    def test_l3_structural_exit_no_trigger_within_buffer(self):
        """E2: 未穿越 L3 ± buffer 时不触发"""
        pos = {
            "side": "LONG",
            "entry_price": 50000,
            "entry_time": _now(),
            "atr_at_entry": 500,
            "stop_loss_price": 48000,
            "l3_structural_exit_enabled": True,
            "l3_structural_exit_buffer_atr": 0.25,
        }
        # threshold = 49000 - 125 = 48875；close 48900 ≥ threshold
        reason, _ = enforce_position(
            pos,
            price_high=49500,
            price_low=48850,
            price_close=48900,
            now=_now() + timedelta(hours=10),
            wide_sr_upper_px=55000,
            wide_sr_lower_px=49000,
        )
        assert reason is None


class TestEnforcePositionSL:

    def test_long_stop_loss_hit(self):
        """LONG: low <= SL → 止损"""
        pos = {
            "side": "LONG",
            "entry_price": 50000,
            "entry_time": _now(),
            "atr_at_entry": 500,
            "max_holding_bars": 50,
            "bar_minutes": 240,
            "stop_loss_price": 49000,
            "take_profit_price": 51500,
        }
        reason, exit_price = enforce_position(
            pos,
            price_high=50200,
            price_low=48900,  # low < SL (49000)
            price_close=49500,
            now=_now() + timedelta(hours=1),
        )
        assert reason == "stop_loss"
        assert exit_price == pytest.approx(49000)  # 精确 SL 价

    def test_short_stop_loss_hit(self):
        """SHORT: high >= SL → 止损"""
        pos = {
            "side": "SHORT",
            "entry_price": 3000,
            "entry_time": _now(),
            "atr_at_entry": 100,
            "max_holding_bars": 50,
            "bar_minutes": 240,
            "stop_loss_price": 3150,
            "take_profit_price": 2800,
        }
        reason, exit_price = enforce_position(
            pos,
            price_high=3200,  # high > SL (3150)
            price_low=2950,
            price_close=3100,
            now=_now() + timedelta(hours=1),
        )
        assert reason == "stop_loss"
        assert exit_price == pytest.approx(3150)

    def test_long_sl_not_hit(self):
        """LONG: low > SL → 不触发"""
        pos = {
            "side": "LONG",
            "entry_price": 50000,
            "entry_time": _now(),
            "atr_at_entry": 500,
            "max_holding_bars": 50,
            "bar_minutes": 240,
            "stop_loss_price": 49000,
            "take_profit_price": 55000,  # 远离 TP
        }
        reason, _ = enforce_position(
            pos,
            price_high=50500,
            price_low=49100,  # low > SL
            price_close=50200,
            now=_now() + timedelta(hours=1),
        )
        assert reason is None


class TestEnforcePositionTP:

    def test_long_take_profit_hit(self):
        """LONG: high >= TP → 止盈"""
        pos = {
            "side": "LONG",
            "entry_price": 50000,
            "entry_time": _now(),
            "atr_at_entry": 500,
            "max_holding_bars": 50,
            "bar_minutes": 240,
            "stop_loss_price": 49000,
            "take_profit_price": 51500,
        }
        reason, exit_price = enforce_position(
            pos,
            price_high=51600,  # high > TP
            price_low=50800,
            price_close=51200,
            now=_now() + timedelta(hours=1),
        )
        assert reason == "take_profit"
        assert exit_price == pytest.approx(51500)

    def test_short_take_profit_hit(self):
        """SHORT: low <= TP → 止盈"""
        pos = {
            "side": "SHORT",
            "entry_price": 3000,
            "entry_time": _now(),
            "atr_at_entry": 100,
            "max_holding_bars": 50,
            "bar_minutes": 240,
            "stop_loss_price": 3150,
            "take_profit_price": 2800,
        }
        reason, exit_price = enforce_position(
            pos,
            price_high=2950,
            price_low=2790,  # low < TP (2800)
            price_close=2850,
            now=_now() + timedelta(hours=1),
        )
        assert reason == "take_profit"
        assert exit_price == pytest.approx(2800)


class TestEnforcePositionSLPriority:

    def test_sl_priority_over_tp_long(self):
        """同一根 bar SL 和 TP 都触发时, SL 优先 (保守假设)"""
        pos = {
            "side": "LONG",
            "entry_price": 50000,
            "entry_time": _now(),
            "atr_at_entry": 500,
            "max_holding_bars": 50,
            "bar_minutes": 240,
            "stop_loss_price": 49000,
            "take_profit_price": 51500,
        }
        # 极端 bar: low 穿 SL, high 穿 TP
        reason, exit_price = enforce_position(
            pos,
            price_high=52000,
            price_low=48500,
            price_close=50000,
            now=_now() + timedelta(hours=1),
        )
        assert reason == "stop_loss"
        assert exit_price == pytest.approx(49000)

    def test_sl_priority_over_tp_short(self):
        """SHORT: 同一 bar SL 和 TP 都触发, SL 优先"""
        pos = {
            "side": "SHORT",
            "entry_price": 3000,
            "entry_time": _now(),
            "atr_at_entry": 100,
            "max_holding_bars": 50,
            "bar_minutes": 240,
            "stop_loss_price": 3150,
            "take_profit_price": 2800,
        }
        reason, exit_price = enforce_position(
            pos,
            price_high=3200,  # triggers SL
            price_low=2700,  # triggers TP
            price_close=2950,
            now=_now() + timedelta(hours=1),
        )
        assert reason == "stop_loss"
        assert exit_price == pytest.approx(3150)


class TestEnforcePositionBreakeven:

    def test_breakeven_lock_triggers(self):
        """当利润 >= breakeven_trigger_r, SL 应锁到入场价"""
        pos = {
            "side": "LONG",
            "entry_price": 50000,
            "entry_time": _now(),
            "atr_at_entry": 500,
            "max_holding_bars": 50,
            "bar_minutes": 240,
            "stop_loss_price": 49000,
            "take_profit_price": 52000,
            "breakeven_enabled": True,
            "breakeven_trigger_r": 1.0,
            "breakeven_locked": False,
        }
        # profit_r = (51000 - 50000) / 500 = 2.0 >= trigger 1.0
        reason, _ = enforce_position(
            pos,
            price_high=51000,
            price_low=50500,
            price_close=50800,
            now=_now() + timedelta(hours=1),
        )
        assert reason is None  # 不关仓
        assert pos["breakeven_locked"] is True
        assert pos["stop_loss_price"] == 50000  # SL 锁到入场价

    def test_breakeven_not_triggered_below_threshold(self):
        pos = {
            "side": "LONG",
            "entry_price": 50000,
            "entry_time": _now(),
            "atr_at_entry": 500,
            "max_holding_bars": 50,
            "bar_minutes": 240,
            "stop_loss_price": 49000,
            "take_profit_price": 52000,
            "breakeven_enabled": True,
            "breakeven_trigger_r": 1.0,
            "breakeven_locked": False,
        }
        # profit_r = (50200 - 50000) / 500 = 0.4 < trigger 1.0
        enforce_position(
            pos,
            price_high=50200,
            price_low=50000,
            price_close=50100,
            now=_now() + timedelta(hours=1),
        )
        assert pos["breakeven_locked"] is False
        assert pos["stop_loss_price"] == 49000  # 未变


class TestEnforcePositionTrailing:

    def test_activation_trailing_long(self):
        """LONG: 利润达到 activation_r 后, trailing SL 上移"""
        pos = {
            "side": "LONG",
            "entry_price": 50000,
            "entry_time": _now(),
            "atr_at_entry": 500,
            "max_holding_bars": 50,
            "bar_minutes": 240,
            "stop_loss_price": 49000,
            "take_profit_price": 55000,
            "activation_r": 1.0,
            "trail_r": 0.8,
            "trailing_activated": False,
            "high_water_mark": 50000,
            "low_water_mark": None,
            "breakeven_enabled": False,
            "breakeven_locked": False,
        }
        # bar 1: price high=51000 → profit_r = (51000-50000)/500 = 2.0 >= activation 1.0
        # HWM = 51000, trail_sl = 51000 - 0.8*500 = 50600
        enforce_position(
            pos,
            price_high=51000,
            price_low=50500,
            price_close=50800,
            now=_now() + timedelta(hours=1),
        )
        assert pos["trailing_activated"] is True
        assert pos["high_water_mark"] == 51000
        assert pos["stop_loss_price"] == pytest.approx(50600)

        # bar 2: HWM goes to 51500 → trail_sl = 51500 - 400 = 51100
        enforce_position(
            pos,
            price_high=51500,
            price_low=50900,
            price_close=51200,
            now=_now() + timedelta(hours=2),
        )
        assert pos["high_water_mark"] == 51500
        assert pos["stop_loss_price"] == pytest.approx(51100)

    def test_breakeven_initial_risk_triggers_long(self):
        """LONG: measure=initial_risk, MFE ≥ trigger_r × R → SL 抬到 entry + lock*R。"""
        pos = {
            "side": "LONG",
            "entry_price": 100.0,
            "entry_time": _now(),
            "atr_at_entry": 2.0,
            "initial_risk_distance": 2.0,  # 1R = $2
            "max_holding_bars": 50,
            "bar_minutes": 240,
            "stop_loss_price": 98.0,
            "take_profit_price": 120.0,
            "activation_r": None,
            "breakeven_enabled": True,
            "breakeven_trigger_r": 3.0,
            "breakeven_lock_level_r": 0.0,
            "breakeven_measure": "initial_risk",
            "breakeven_locked": False,
        }
        # price_high=106 → MFE_r = (106-100)/2 = 3.0 >= 3.0 → SL 锁到 entry=100
        enforce_position(
            pos,
            price_high=106.0,
            price_low=100.5,
            price_close=105.0,
            now=_now() + timedelta(hours=1),
        )
        assert pos["breakeven_locked"] is True
        assert pos["stop_loss_price"] == pytest.approx(100.0)

    def test_breakeven_initial_risk_uses_structural_sl_distance(self):
        """measure=initial_risk 使用 initial_risk_distance（可能 != atr，如 structural_sl）。"""
        pos = {
            "side": "LONG",
            "entry_price": 100.0,
            "entry_time": _now(),
            "atr_at_entry": 2.0,  # ATR=2
            "initial_risk_distance": 4.0,  # 结构化 SL 距离：1R=$4（≠ ATR）
            "max_holding_bars": 50,
            "bar_minutes": 240,
            "stop_loss_price": 96.0,
            "take_profit_price": 120.0,
            "activation_r": None,
            "breakeven_enabled": True,
            "breakeven_trigger_r": 3.0,
            "breakeven_lock_level_r": 0.0,
            "breakeven_measure": "initial_risk",
            "breakeven_locked": False,
        }
        # 基于 initial_risk_distance=4: MFE_r = (112-100)/4 = 3.0 → 触发
        enforce_position(
            pos,
            price_high=112.0,
            price_low=100.5,
            price_close=111.0,
            now=_now() + timedelta(hours=1),
        )
        assert pos["breakeven_locked"] is True
        assert pos["stop_loss_price"] == pytest.approx(100.0)

    def test_breakeven_atr_measure_matches_legacy(self):
        """measure=atr 保持 BPC 历史口径：profit_r = (price-entry)/atr。"""
        pos = {
            "side": "LONG",
            "entry_price": 100.0,
            "entry_time": _now(),
            "atr_at_entry": 2.0,
            "initial_risk_distance": 4.0,  # 设置但 measure=atr 时应该忽略
            "max_holding_bars": 50,
            "bar_minutes": 240,
            "stop_loss_price": 96.0,
            "take_profit_price": 120.0,
            "activation_r": None,
            "breakeven_enabled": True,
            "breakeven_trigger_r": 3.0,
            "breakeven_lock_level_r": 1.0,  # 锁 1R = 1*ATR = $2
            "breakeven_measure": "atr",
            "breakeven_locked": False,
        }
        # MFE_r = (106-100)/2 = 3.0 → 触发；SL = 100 + 1*2 = 102
        enforce_position(
            pos,
            price_high=106.0,
            price_low=100.5,
            price_close=105.0,
            now=_now() + timedelta(hours=1),
        )
        assert pos["breakeven_locked"] is True
        assert pos["stop_loss_price"] == pytest.approx(102.0)

    def test_breakeven_tighten_only_does_not_retract_sl(self):
        """tighten-only（硬编码）：若 SL 已超过 lock_level，不回撤。"""
        pos = {
            "side": "LONG",
            "entry_price": 100.0,
            "entry_time": _now(),
            "atr_at_entry": 2.0,
            "initial_risk_distance": 2.0,
            "max_holding_bars": 50,
            "bar_minutes": 240,
            # SL 已被 trailing 抬到 105（entry+2.5R）
            "stop_loss_price": 105.0,
            "take_profit_price": 120.0,
            "activation_r": None,
            "breakeven_enabled": True,
            "breakeven_trigger_r": 3.0,
            "breakeven_lock_level_r": 0.0,
            "breakeven_measure": "initial_risk",
            "breakeven_locked": False,
        }
        enforce_position(
            pos,
            price_high=108.0,
            price_low=104.0,
            price_close=107.0,
            now=_now() + timedelta(hours=1),
        )
        # trigger 激活但 tighten-only: 新 SL = entry = 100 < 老 SL 105 → 保留 105
        assert pos["breakeven_locked"] is True
        assert pos["stop_loss_price"] == pytest.approx(105.0)

    def test_breakeven_short_with_lock_profit(self):
        """SHORT MFE ≥ trigger 时 SL 下移到 entry - lock_level_r × R。"""
        pos = {
            "side": "SHORT",
            "entry_price": 100.0,
            "entry_time": _now(),
            "atr_at_entry": 2.0,
            "initial_risk_distance": 2.0,
            "max_holding_bars": 50,
            "bar_minutes": 240,
            "stop_loss_price": 102.0,
            "take_profit_price": 80.0,
            "activation_r": None,
            "breakeven_enabled": True,
            "breakeven_trigger_r": 3.0,
            "breakeven_lock_level_r": 0.5,  # 锁 0.5R 利润
            "breakeven_measure": "initial_risk",
            "breakeven_locked": False,
        }
        # price_low=94 → MFE_r = (100-94)/2 = 3.0 >= 3.0
        # new_sl = 100 - 0.5*2 = 99
        enforce_position(
            pos,
            price_high=100.5,
            price_low=94.0,
            price_close=95.0,
            now=_now() + timedelta(hours=1),
        )
        assert pos["breakeven_locked"] is True
        assert pos["stop_loss_price"] == pytest.approx(99.0)

    def test_trailing_l3_dynamic_long_near_reverse_l3(self):
        """LONG: 价距反向 L3（上沿）< threshold 时使用 trail_r_near（收紧）。"""
        pos = {
            "side": "LONG",
            "entry_price": 50000,
            "entry_time": _now(),
            "atr_at_entry": 500,
            "max_holding_bars": 50,
            "bar_minutes": 240,
            "stop_loss_price": 49000,
            "take_profit_price": 60000,
            "activation_r": 1.0,
            "trail_r": 5.0,
            "trail_r_far": 7.0,
            "trail_r_near": 3.0,
            "l3_near_threshold_atr": 2.0,
            "trailing_activated": False,
            "high_water_mark": 50000,
            "low_water_mark": None,
            "breakeven_enabled": False,
            "breakeven_locked": False,
        }
        # price_close=51000, wide_sr_upper_px=51500 → rev_dist = (51500-51000)/500 = 1.0 < 2.0
        # → 用 trail_r_near=3.0 → trail_sl = 51200 - 3.0*500 = 49700
        enforce_position(
            pos,
            price_high=51200,
            price_low=50900,
            price_close=51000,
            now=_now() + timedelta(hours=1),
            wide_sr_upper_px=51500,
            wide_sr_lower_px=48000,
        )
        assert pos["trailing_activated"] is True
        assert pos["stop_loss_price"] == pytest.approx(49700)

    def test_trailing_l3_dynamic_long_far_from_reverse_l3(self):
        """LONG: 价距反向 L3（上沿）>= threshold 时使用 trail_r_far（放宽）。"""
        pos = {
            "side": "LONG",
            "entry_price": 50000,
            "entry_time": _now(),
            "atr_at_entry": 500,
            "max_holding_bars": 50,
            "bar_minutes": 240,
            "stop_loss_price": 49000,
            "take_profit_price": 60000,
            "activation_r": 1.0,
            "trail_r": 5.0,
            "trail_r_far": 7.0,
            "trail_r_near": 3.0,
            "l3_near_threshold_atr": 2.0,
            "trailing_activated": False,
            "high_water_mark": 50000,
            "low_water_mark": None,
            "breakeven_enabled": False,
            "breakeven_locked": False,
        }
        # price_close=51000, wide_sr_upper_px=55000 → rev_dist = (55000-51000)/500 = 8.0 >= 2.0
        # → 用 trail_r_far=7.0 → trail_sl = 51200 - 7.0*500 = 47700 < old 49000, 不会更新
        enforce_position(
            pos,
            price_high=51200,
            price_low=50900,
            price_close=51000,
            now=_now() + timedelta(hours=1),
            wide_sr_upper_px=55000,
            wide_sr_lower_px=48000,
        )
        assert pos["trailing_activated"] is True
        # trail_sl 47700 < old 49000, 保持原 SL
        assert pos["stop_loss_price"] == pytest.approx(49000)

    def test_trailing_l3_dynamic_short(self):
        """SHORT: 价距反向 L3（下沿）< threshold 时使用 trail_r_near（收紧）。"""
        pos = {
            "side": "SHORT",
            "entry_price": 3000,
            "entry_time": _now(),
            "atr_at_entry": 100,
            "max_holding_bars": 50,
            "bar_minutes": 240,
            "stop_loss_price": 3200,
            "take_profit_price": 2000,
            "activation_r": 1.0,
            "trail_r": 5.0,
            "trail_r_far": 7.0,
            "trail_r_near": 3.0,
            "l3_near_threshold_atr": 2.0,
            "trailing_activated": False,
            "high_water_mark": None,
            "low_water_mark": 3000,
            "breakeven_enabled": False,
            "breakeven_locked": False,
        }
        # price_close=2900, wide_sr_lower_px=2750 → rev_dist = (2900-2750)/100 = 1.5 < 2.0
        # → 用 trail_r_near=3.0 → trail_sl = 2800 + 3.0*100 = 3100
        enforce_position(
            pos,
            price_high=2950,
            price_low=2800,
            price_close=2900,
            now=_now() + timedelta(hours=1),
            wide_sr_upper_px=3300,
            wide_sr_lower_px=2750,
        )
        assert pos["trailing_activated"] is True
        assert pos["low_water_mark"] == 2800
        assert pos["stop_loss_price"] == pytest.approx(3100)

    def test_trailing_l3_dynamic_falls_back_without_wide_levels(self):
        """缺少 wide_sr_upper_px / wide_sr_lower_px 时回退到 trail_r（原有行为）。"""
        pos = {
            "side": "LONG",
            "entry_price": 50000,
            "entry_time": _now(),
            "atr_at_entry": 500,
            "max_holding_bars": 50,
            "bar_minutes": 240,
            "stop_loss_price": 49000,
            "take_profit_price": 60000,
            "activation_r": 1.0,
            "trail_r": 5.0,
            "trail_r_far": 7.0,
            "trail_r_near": 3.0,
            "l3_near_threshold_atr": 2.0,
            "trailing_activated": False,
            "high_water_mark": 50000,
            "low_water_mark": None,
            "breakeven_enabled": False,
            "breakeven_locked": False,
        }
        enforce_position(
            pos,
            price_high=51200,
            price_low=50900,
            price_close=51000,
            now=_now() + timedelta(hours=1),
            wide_sr_upper_px=None,
            wide_sr_lower_px=None,
        )
        assert pos["trailing_activated"] is True
        # 回退 trail_r=5.0 → trail_sl = 51200 - 5*500 = 48700 < old 49000 → 保持 49000
        assert pos["stop_loss_price"] == pytest.approx(49000)

    def test_activation_trailing_short(self):
        """SHORT: 利润达到 activation_r 后, trailing SL 下移"""
        pos = {
            "side": "SHORT",
            "entry_price": 3000,
            "entry_time": _now(),
            "atr_at_entry": 100,
            "max_holding_bars": 50,
            "bar_minutes": 240,
            "stop_loss_price": 3200,
            "take_profit_price": 2500,
            "activation_r": 1.5,
            "trail_r": 1.0,
            "trailing_activated": False,
            "high_water_mark": None,
            "low_water_mark": 3000,
            "breakeven_enabled": False,
            "breakeven_locked": False,
        }
        # profit_r = (3000 - 2800) / 100 = 2.0 >= 1.5
        # LWM = 2800, trail_sl = 2800 + 1.0*100 = 2900
        enforce_position(
            pos,
            price_high=2950,
            price_low=2800,
            price_close=2850,
            now=_now() + timedelta(hours=1),
        )
        assert pos["trailing_activated"] is True
        assert pos["low_water_mark"] == 2800
        assert pos["stop_loss_price"] == pytest.approx(2900)

    def test_trailing_sl_never_moves_backwards_long(self):
        """LONG trailing: SL 只能上移不能下移"""
        pos = {
            "side": "LONG",
            "entry_price": 50000,
            "entry_time": _now(),
            "atr_at_entry": 500,
            "max_holding_bars": 50,
            "bar_minutes": 240,
            "stop_loss_price": 50600,  # 已经较高
            "take_profit_price": 55000,
            "activation_r": 1.0,
            "trail_r": 0.8,
            "trailing_activated": True,
            "high_water_mark": 51000,
            "low_water_mark": None,
            "breakeven_enabled": False,
            "breakeven_locked": False,
        }
        # 回调: high=50800, profit_r = (50800-50000)/500 = 1.6 >= 1.0
        # 但 HWM 不变, trail_sl = 51000 - 400 = 50600, 不低于当前 SL
        enforce_position(
            pos,
            price_high=50800,
            price_low=50400,
            price_close=50600,
            now=_now() + timedelta(hours=1),
        )
        # SL 应保持不变 (不会下移)
        assert pos["stop_loss_price"] == pytest.approx(50600)


class TestEnforcePositionLiveMode:
    """验证实盘模式: price_high = price_low = price_close = current_price"""

    def test_live_sl_hit(self):
        pos = {
            "side": "LONG",
            "entry_price": 50000,
            "entry_time": _now(),
            "atr_at_entry": 500,
            "max_holding_bars": 50,
            "bar_minutes": 240,
            "stop_loss_price": 49000,
            "take_profit_price": 51500,
        }
        current = 48800  # 低于 SL
        reason, exit_price = enforce_position(
            pos,
            price_high=current,
            price_low=current,
            price_close=current,
            now=_now() + timedelta(hours=1),
        )
        assert reason == "stop_loss"
        assert exit_price == pytest.approx(49000)

    def test_live_tp_hit(self):
        pos = {
            "side": "LONG",
            "entry_price": 50000,
            "entry_time": _now(),
            "atr_at_entry": 500,
            "max_holding_bars": 50,
            "bar_minutes": 240,
            "stop_loss_price": 49000,
            "take_profit_price": 51500,
        }
        current = 52000
        reason, exit_price = enforce_position(
            pos,
            price_high=current,
            price_low=current,
            price_close=current,
            now=_now() + timedelta(hours=1),
        )
        assert reason == "take_profit"
        assert exit_price == pytest.approx(51500)


class TestEnforcePositionNoClose:

    def test_position_stays_open(self):
        """价格在 SL/TP 之间且未超时 → 无关仓"""
        pos = {
            "side": "LONG",
            "entry_price": 50000,
            "entry_time": _now(),
            "atr_at_entry": 500,
            "max_holding_bars": 50,
            "bar_minutes": 240,
            "stop_loss_price": 49000,
            "take_profit_price": 51500,
        }
        reason, _ = enforce_position(
            pos,
            price_high=50300,
            price_low=49800,
            price_close=50100,
            now=_now() + timedelta(hours=1),
        )
        assert reason is None

    def test_no_sl_tp_prices(self):
        """无 SL/TP 时, 只有 time_stop 能触发"""
        pos = {
            "side": "LONG",
            "entry_price": 50000,
            "entry_time": _now(),
            "atr_at_entry": 500,
            "max_holding_bars": 50,
            "bar_minutes": 240,
            "stop_loss_price": None,
            "take_profit_price": None,
        }
        reason, _ = enforce_position(
            pos,
            price_high=40000,
            price_low=30000,
            price_close=35000,
            now=_now() + timedelta(hours=1),
        )
        assert reason is None  # 无 SL/TP 则不触发


class TestEnforcePositionMutatesPos:
    """验证 enforce_position 就地修改 pos 字典 (与实盘行为一致)"""

    def test_hwm_updated(self):
        pos = {
            "side": "LONG",
            "entry_price": 50000,
            "entry_time": _now(),
            "atr_at_entry": 500,
            "max_holding_bars": 50,
            "bar_minutes": 240,
            "stop_loss_price": 49000,
            "take_profit_price": 55000,
            "high_water_mark": 50500,
            "low_water_mark": None,
        }
        enforce_position(
            pos,
            price_high=51000,
            price_low=50200,
            price_close=50800,
            now=_now() + timedelta(hours=1),
        )
        assert pos["high_water_mark"] == 51000

    def test_lwm_updated_short(self):
        pos = {
            "side": "SHORT",
            "entry_price": 3000,
            "entry_time": _now(),
            "atr_at_entry": 100,
            "max_holding_bars": 50,
            "bar_minutes": 240,
            "stop_loss_price": 3150,
            "take_profit_price": 2800,
            "high_water_mark": None,
            "low_water_mark": 2950,
        }
        enforce_position(
            pos,
            price_high=2980,
            price_low=2900,
            price_close=2920,
            now=_now() + timedelta(hours=1),
        )
        assert pos["low_water_mark"] == 2900


# ═══════════════════════════════════════════════════════════════════════
# Structural Exit (EMA200) — Multi-Alpha Holding
# ═══════════════════════════════════════════════════════════════════════


class TestBuildPositionDictStructuralExit:
    """build_position_dict 应正确存储 structural_exit 字段"""

    def test_structural_exit_stored(self):
        """rr_constraints 含 structural_exit → pos 中出现该字段"""
        rr = {
            "stop_loss_r": 4.0,
            "take_profit_r": 0,
            "max_holding_bars": 0,
            "allow_trailing": True,
            "activation_r": 2.5,
            "trailing_atr": 7.0,
            "structural_exit": "ema200",
        }
        ep = {"rr_constraints": rr}
        intent = TradeIntent(
            action="LONG",
            symbol="BTCUSDT",
            archetype="bpc",
            confidence=0.8,
            execution_profile=ep,
        )
        pos = build_position_dict(intent, entry_price=50000, atr=500, entry_time=_now())
        assert pos["structural_exit"] == "ema200"

    def test_structural_exit_absent_when_not_set(self):
        """rr_constraints 无 structural_exit → pos 中不出现该字段"""
        intent = _make_intent(stop_loss_r=2.0, take_profit_r=3.0)
        pos = build_position_dict(intent, entry_price=50000, atr=500, entry_time=_now())
        assert "structural_exit" not in pos

    def test_structural_exit_absent_when_none(self):
        """structural_exit=None → 不存储"""
        rr = {
            "stop_loss_r": 4.0,
            "max_holding_bars": 0,
            "structural_exit": None,
        }
        ep = {"rr_constraints": rr}
        intent = TradeIntent(
            action="LONG",
            symbol="BTCUSDT",
            archetype="bpc",
            confidence=0.8,
            execution_profile=ep,
        )
        pos = build_position_dict(intent, entry_price=50000, atr=500, entry_time=_now())
        assert "structural_exit" not in pos

    def test_vwap1200_no_execution_inner_key(self):
        """vwap1200 仅标记 structural_exit；近场由 gate，执行层不写 vwap_exit_inner_abs"""
        rr = {
            "stop_loss_r": 4.0,
            "max_holding_bars": 0,
            "structural_exit": "vwap1200",
        }
        ep = {"rr_constraints": rr}
        intent = TradeIntent(
            action="LONG",
            symbol="BTCUSDT",
            archetype="bpc",
            confidence=0.8,
            execution_profile=ep,
        )
        pos = build_position_dict(intent, entry_price=50000, atr=500, entry_time=_now())
        assert pos["structural_exit"] == "vwap1200"
        assert "vwap_exit_inner_abs" not in pos


def _bpc_trend_hold_pos(
    side="LONG",
    entry_price=50000,
    atr=500,
    breakeven_locked=True,
    structural_exit="ema200",
):
    """构造 BPC trend_hold 持仓: breakeven + structural_exit + 宽 trailing"""
    is_long = side == "LONG"
    return {
        "side": side,
        "entry_price": entry_price,
        "entry_time": _now(),
        "atr_at_entry": atr,
        "max_holding_bars": 0,  # fat tail: 无时间止损
        "bar_minutes": 240,
        "stop_loss_price": (
            entry_price - 4.0 * atr if is_long else entry_price + 4.0 * atr
        ),
        "take_profit_price": None,  # 无 TP
        "activation_r": 2.5,
        "trail_r": 7.0,  # 灾难保护
        "trailing_activated": False,
        "high_water_mark": entry_price if is_long else None,
        "low_water_mark": entry_price if not is_long else None,
        "breakeven_enabled": True,
        "breakeven_trigger_r": 1.0,
        "breakeven_locked": breakeven_locked,
        "structural_exit": structural_exit,
    }


class TestStructuralExitLong:
    """LONG + structural_exit=ema200: close < EMA200 触发退出"""

    def test_long_close_below_ema200_exits(self):
        """breakeven locked + close < EMA200 → structural_exit_ema200"""
        pos = _bpc_trend_hold_pos(side="LONG", entry_price=50000)
        reason, exit_price = enforce_position(
            pos,
            price_high=49500,
            price_low=49000,
            price_close=49200,  # < EMA200 (49800)
            now=_now() + timedelta(hours=4),
            structural_price=49800,
        )
        assert reason == "structural_exit_ema200"
        assert exit_price == 49200

    def test_long_close_above_ema200_holds(self):
        """close > EMA200 → 不退出"""
        pos = _bpc_trend_hold_pos(side="LONG", entry_price=50000)
        reason, _ = enforce_position(
            pos,
            price_high=51000,
            price_low=50500,
            price_close=50800,  # > EMA200 (49000)
            now=_now() + timedelta(hours=4),
            structural_price=49000,
        )
        assert reason is None

    def test_long_close_equals_ema200_holds(self):
        """close == EMA200 → 不退出 (需严格穿越)"""
        pos = _bpc_trend_hold_pos(side="LONG", entry_price=50000)
        reason, _ = enforce_position(
            pos,
            price_high=50500,
            price_low=49800,
            price_close=49800,  # == EMA200
            now=_now() + timedelta(hours=4),
            structural_price=49800,
        )
        assert reason is None


class TestStructuralExitShort:
    """SHORT + structural_exit=ema200: close > EMA200 触发退出"""

    def test_short_close_above_ema200_exits(self):
        pos = _bpc_trend_hold_pos(side="SHORT", entry_price=3000, atr=100)
        reason, exit_price = enforce_position(
            pos,
            price_high=3150,
            price_low=3050,
            price_close=3120,  # > EMA200 (3100)
            now=_now() + timedelta(hours=4),
            structural_price=3100,
        )
        assert reason == "structural_exit_ema200"
        assert exit_price == 3120

    def test_short_close_below_ema200_holds(self):
        pos = _bpc_trend_hold_pos(side="SHORT", entry_price=3000, atr=100)
        reason, _ = enforce_position(
            pos,
            price_high=2950,
            price_low=2850,
            price_close=2900,  # < EMA200 (3100)
            now=_now() + timedelta(hours=4),
            structural_price=3100,
        )
        assert reason is None


class TestStructuralExitPreconditions:
    """structural_exit 前置条件: breakeven_locked + 有效 structural_price"""

    def test_structural_ema200_triggers_even_if_breakeven_not_locked(self):
        """EMA200 穿越与 breakeven 并行：未锁保本仍可结构出场"""
        pos = _bpc_trend_hold_pos(
            side="LONG", entry_price=50000, breakeven_locked=False
        )
        reason, px = enforce_position(
            pos,
            price_high=49500,
            price_low=49000,
            price_close=49200,  # < EMA200
            now=_now() + timedelta(hours=4),
            structural_price=49800,
        )
        assert reason == "structural_exit_ema200"
        assert px == 49200

    def test_no_exit_when_structural_price_none(self):
        """structural_price=None → 不触发"""
        pos = _bpc_trend_hold_pos(side="LONG", entry_price=50000)
        reason, _ = enforce_position(
            pos,
            price_high=49500,
            price_low=49000,
            price_close=49200,
            now=_now() + timedelta(hours=4),
            structural_price=None,
        )
        assert reason is None

    def test_no_exit_when_structural_price_zero(self):
        """structural_price=0 → 不触发"""
        pos = _bpc_trend_hold_pos(side="LONG", entry_price=50000)
        reason, _ = enforce_position(
            pos,
            price_high=49500,
            price_low=49000,
            price_close=49200,
            now=_now() + timedelta(hours=4),
            structural_price=0.0,
        )
        assert reason is None

    def test_no_exit_without_structural_exit_field(self):
        """pos 无 structural_exit 字段 → 即使传 structural_price 也不触发"""
        pos = _bpc_trend_hold_pos(side="LONG", entry_price=50000, structural_exit=None)
        del pos["structural_exit"]  # 模拟 ME 仓位 (无 structural_exit)
        reason, _ = enforce_position(
            pos,
            price_high=49500,
            price_low=49000,
            price_close=49200,
            now=_now() + timedelta(hours=4),
            structural_price=49800,
        )
        assert reason is None

    def test_no_exit_when_structural_exit_not_ema200(self):
        """structural_exit 值不是 'ema200' → 不触发"""
        pos = _bpc_trend_hold_pos(
            side="LONG", entry_price=50000, structural_exit="sma200"
        )
        reason, _ = enforce_position(
            pos,
            price_high=49500,
            price_low=49000,
            price_close=49200,
            now=_now() + timedelta(hours=4),
            structural_price=49800,
        )
        assert reason is None


class TestStructuralExitIntegration:
    """Structural exit 与其他退出机制的交互"""

    def test_breakeven_then_structural_exit_sequence(self):
        """完整流程: bar1 触发 breakeven → bar2 EMA200 穿越 → structural exit"""
        pos = _bpc_trend_hold_pos(
            side="LONG",
            entry_price=50000,
            breakeven_locked=False,
        )
        # bar1: 价格涨到 50600, profit_r = (50600-50000)/500 = 1.2 >= 1.0 → breakeven lock
        reason1, _ = enforce_position(
            pos,
            price_high=50600,
            price_low=50200,
            price_close=50400,
            now=_now() + timedelta(hours=4),
            structural_price=49000,  # 远低于价格, 不触发
        )
        assert reason1 is None
        assert pos["breakeven_locked"] is True
        assert pos["stop_loss_price"] == 50000  # SL 锁到入场价

        # bar2: 价格跌穿 EMA200 (49000)
        reason2, exit_price2 = enforce_position(
            pos,
            price_high=49200,
            price_low=48800,
            price_close=48900,  # < EMA200 (49000)
            now=_now() + timedelta(hours=8),
            structural_price=49000,
        )
        assert reason2 == "structural_exit_ema200"
        assert exit_price2 == 48900

    def test_structural_exit_before_trailing_sl(self):
        """structural exit 优先于 trailing SL (Step 3b 在 Step 4 之前)"""
        pos = _bpc_trend_hold_pos(side="LONG", entry_price=50000)
        pos["trailing_activated"] = True
        pos["high_water_mark"] = 55000
        # trail_sl = 55000 - 7.0 * 500 = 51500 (宽 trailing)
        pos["stop_loss_price"] = 51500

        # close < EMA200 → structural exit 优先触发
        reason, exit_price = enforce_position(
            pos,
            price_high=51800,
            price_low=51300,
            price_close=51400,  # < EMA200 (51600)
            now=_now() + timedelta(hours=4),
            structural_price=51600,
        )
        assert reason == "structural_exit_ema200"
        assert exit_price == 51400

    def test_sl_still_works_without_structural_price(self):
        """BPC 仓位不传 structural_price 时, 仍然有 trailing SL 保底"""
        pos = _bpc_trend_hold_pos(side="LONG", entry_price=50000)
        pos["trailing_activated"] = True
        pos["high_water_mark"] = 55000
        pos["stop_loss_price"] = 51500  # trail_sl

        # 不传 structural_price, low 触发 SL
        reason, exit_price = enforce_position(
            pos,
            price_high=51400,
            price_low=51400,  # <= SL (51500)
            price_close=51400,
            now=_now() + timedelta(hours=4),
            structural_price=None,
        )
        assert reason == "stop_loss"
        assert exit_price == pytest.approx(51500)

    def test_me_momentum_hold_unaffected(self):
        """ME 仓位 (无 structural_exit): 只受 trailing stop 管控"""
        pos = {
            "side": "LONG",
            "entry_price": 50000,
            "entry_time": _now(),
            "atr_at_entry": 500,
            "max_holding_bars": 0,
            "bar_minutes": 60,
            "stop_loss_price": 48000,
            "take_profit_price": None,
            "activation_r": 1.0,
            "trail_r": 0.5,  # 紧 trailing
            "trailing_activated": False,
            "high_water_mark": 50000,
            "low_water_mark": None,
            "breakeven_enabled": True,
            "breakeven_trigger_r": 1.0,
            "breakeven_locked": True,
            # 无 structural_exit 字段
        }
        # 即使传了 structural_price, ME 不受影响
        reason, _ = enforce_position(
            pos,
            price_high=49800,
            price_low=49200,
            price_close=49500,  # < EMA200
            now=_now() + timedelta(hours=1),
            structural_price=49800,
        )
        assert reason is None  # ME 不触发 structural exit


class TestEnforceVwap1200StructuralExit:
    def test_vwap1200_no_exit_when_slightly_above_vwap(self):
        """多仓 pv>0（价在 VWAP 上方，含贴边小正 pv）→ 不因 VWAP 规则平仓"""
        pos = {
            "side": "LONG",
            "entry_price": 50000,
            "entry_time": _now(),
            "atr_at_entry": 500,
            "max_holding_bars": 0,
            "bar_minutes": 120,
            "stop_loss_price": 48000,
            "structural_exit": "vwap1200",
            "breakeven_locked": True,
        }
        reason, _ = enforce_position(
            pos,
            price_high=50100,
            price_low=50000,
            price_close=50050,
            now=_now() + timedelta(hours=1),
            macro_tp_vwap_position=0.002,
        )
        assert reason is None

    def test_vwap1200_no_exit_when_above_vwap(self):
        """多仓 pv>0 → 不因 VWAP 规则平仓"""
        pos = {
            "side": "LONG",
            "entry_price": 50000,
            "entry_time": _now(),
            "atr_at_entry": 500,
            "max_holding_bars": 0,
            "bar_minutes": 120,
            "stop_loss_price": 48000,
            "structural_exit": "vwap1200",
            "breakeven_locked": False,
        }
        reason, _ = enforce_position(
            pos,
            price_high=50100,
            price_low=50000,
            price_close=50050,
            now=_now() + timedelta(hours=1),
            macro_tp_vwap_position=0.015,
        )
        assert reason is None

    def test_vwap1200_cross_long_when_price_below_vwap(self):
        pos = {
            "side": "LONG",
            "entry_price": 50000,
            "entry_time": _now(),
            "atr_at_entry": 500,
            "max_holding_bars": 0,
            "bar_minutes": 120,
            "stop_loss_price": 48000,
            "structural_exit": "vwap1200",
            "breakeven_locked": True,
        }
        reason, px = enforce_position(
            pos,
            price_high=50100,
            price_low=47000,
            price_close=47000,
            now=_now() + timedelta(hours=1),
            macro_tp_vwap_position=-0.05,
        )
        assert reason == "structural_exit_vwap1200"
        assert px == 47000

    def test_vwap1200_cross_short_when_price_above_vwap(self):
        pos = {
            "side": "SHORT",
            "entry_price": 50000,
            "entry_time": _now(),
            "atr_at_entry": 500,
            "max_holding_bars": 0,
            "bar_minutes": 120,
            "stop_loss_price": 52000,
            "structural_exit": "vwap1200",
            "breakeven_locked": True,
        }
        reason, px = enforce_position(
            pos,
            price_high=53000,
            price_low=49900,
            price_close=53000,
            now=_now() + timedelta(hours=1),
            macro_tp_vwap_position=0.05,
        )
        assert reason == "structural_exit_vwap1200"
        assert px == 53000


# ═════════════════════════════════════════════════════════════════════════════
# SRB structural_sl tests — SL 锚定"对面 SR"
# 语义：LONG 突破 resistance → SL 放在下方 support；SHORT 突破 support → SL 放在上方 resistance。
#       SL 过远时不 clip（sizing 公式会用 stop_pct 自动缩小仓位）；过近时兜底到 ATR-based。
# ═════════════════════════════════════════════════════════════════════════════


class TestStructuralSL:
    def test_long_sl_anchored_to_support(self):
        """LONG entry=101 atr=0.5 initial_r=6 → ATR-based SL=98（距 entry 6 ATR）。
        对面 support=95（远，距 entry 12 ATR） → 结构化 SL = 95 - 0.5×0.5 = 94.75。
        不 clip（min_distance_atr=2 满足：12 ATR >> 2）→ 最终 SL=94.75。
        """
        intent = _make_intent(
            action="LONG",
            stop_loss_r=6.0,
            take_profit_r=0,
            structural_sl={
                "enabled": True,
                "opposite_sr_buffer_atr": 0.5,
                "min_distance_atr": 2.0,
            },
            strategy_specific={"srb_opposite_sr_level": 95.0},
        )
        pos = build_position_dict(intent, entry_price=101.0, atr=0.5)
        assert pos["stop_loss_price"] == pytest.approx(94.75, rel=1e-6)
        assert pos["sizing_stop_source"] == "structural_opposite_sr"
        # effective_stop_pct = (101-94.75)/101 ≈ 0.0619
        assert pos["effective_stop_pct"] == pytest.approx(0.0618811881, rel=1e-4)

    def test_long_sl_fallback_when_support_too_close(self):
        """LONG entry=101 atr=0.5。对面 support=100.5（距 entry 1 ATR，紧贴）。
        结构化距离 = (101 - 100.25) / 0.5 = 1.5 ATR < min_distance_atr=2 → 兜底到 ATR-based。
        ATR-based SL = 101 - 6×0.5 = 98。
        """
        intent = _make_intent(
            action="LONG",
            stop_loss_r=6.0,
            take_profit_r=0,
            structural_sl={
                "enabled": True,
                "opposite_sr_buffer_atr": 0.5,
                "min_distance_atr": 2.0,
            },
            strategy_specific={"srb_opposite_sr_level": 100.5},
        )
        pos = build_position_dict(intent, entry_price=101.0, atr=0.5)
        assert pos["stop_loss_price"] == pytest.approx(98.0, rel=1e-6)
        assert pos["sizing_stop_source"] != "structural_opposite_sr"

    def test_short_sl_anchored_to_resistance(self):
        """SHORT entry=99 atr=0.5 initial_r=6 → ATR-based SL=102。
        对面 resistance=105（距 entry 12 ATR）→ 结构化 SL = 105 + 0.5×0.5 = 105.25。
        不 clip → 最终 SL=105.25。
        """
        intent = _make_intent(
            action="SHORT",
            stop_loss_r=6.0,
            take_profit_r=0,
            structural_sl={
                "enabled": True,
                "opposite_sr_buffer_atr": 0.5,
                "min_distance_atr": 2.0,
            },
            strategy_specific={"srb_opposite_sr_level": 105.0},
        )
        pos = build_position_dict(intent, entry_price=99.0, atr=0.5)
        assert pos["stop_loss_price"] == pytest.approx(105.25, rel=1e-6)
        assert pos["sizing_stop_source"] == "structural_opposite_sr"

    def test_disabled_when_no_opposite_sr(self):
        """无 srb_opposite_sr_level → 走原 ATR-based。"""
        intent = _make_intent(
            action="LONG",
            stop_loss_r=6.0,
            take_profit_r=0,
            structural_sl={"enabled": True},
            strategy_specific={},
        )
        pos = build_position_dict(intent, entry_price=101.0, atr=0.5)
        assert pos["stop_loss_price"] == pytest.approx(98.0, rel=1e-6)
        assert pos["sizing_stop_source"] == "atr"


class TestStructuralExitEma50:
    def test_long_close_below_ema50_exits(self):
        pos = _bpc_trend_hold_pos(side="LONG", structural_exit="ema50")
        pos["breakeven_enabled"] = False
        pos["trailing_activated"] = False
        reason, exit_price = enforce_position(
            pos,
            price_high=50100,
            price_low=49800,
            price_close=49900,
            now=_now() + timedelta(hours=2),
            ema_50_position=-0.01,
        )
        assert reason == "structural_exit_ema50"
        assert exit_price == 49900

    def test_long_close_above_ema50_holds(self):
        pos = _bpc_trend_hold_pos(side="LONG", structural_exit="ema50")
        pos["breakeven_enabled"] = False
        reason, _ = enforce_position(
            pos,
            price_high=51000,
            price_low=50500,
            price_close=50800,
            now=_now() + timedelta(hours=2),
            ema_50_position=0.01,
        )
        assert reason is None


class TestStructuralExitFundingZscore0:
    def test_short_exits_when_funding_z_back_to_zero(self):
        pos = _bpc_trend_hold_pos(side="SHORT", structural_exit="funding_zscore0")
        pos["breakeven_enabled"] = False
        pos["trailing_activated"] = False
        reason, exit_price = enforce_position(
            pos,
            price_high=50100,
            price_low=49800,
            price_close=49900,
            now=_now() + timedelta(hours=2),
            funding_rate_zscore_50=-0.01,
        )
        assert reason == "structural_exit_funding_zscore0"
        assert exit_price == 49900

    def test_short_holds_while_funding_still_crowded(self):
        pos = _bpc_trend_hold_pos(side="SHORT", structural_exit="funding_zscore0")
        pos["breakeven_enabled"] = False
        reason, _ = enforce_position(
            pos,
            price_high=50100,
            price_low=49800,
            price_close=49900,
            now=_now() + timedelta(hours=2),
            funding_rate_zscore_50=1.8,
        )
        assert reason is None
