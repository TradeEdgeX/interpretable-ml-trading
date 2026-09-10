from __future__ import annotations

import pytest

from src.time_series_model.core.constitution.add_position_rules import (
    apply_latest_add_stop_ratchet,
    latest_add_stop_price,
    resolve_add_position_max_times,
    resolve_add_position_size_multiplier,
    resolve_float_r_ladder_only,
    tighten_stop_price,
    validate_add_position_trigger,
)


def test_resolve_add_position_max_times_infers_longest_non_empty_vector():
    assert (
        resolve_add_position_max_times(
            {
                "max_add_times": 7,
                "add_size_multipliers": [1, 2],
                "min_current_r_by_add": [2, 4],
            }
        )
        == 2
    )
    assert resolve_add_position_max_times({"max_add_times": 3}) == 3


def test_resolve_add_position_max_times_fallback_without_vectors():
    assert resolve_add_position_max_times(None) == 1
    assert resolve_add_position_max_times({}) == 1


def test_resolve_add_position_max_times_respects_explicit_zero():
    """Regression (2026-08-05): `int(cfg.get("max_add_times", 1) or 1)` silently
    coerced an explicit ``max_add_times: 0`` (disable adds) back to 1, because 0
    is falsy in Python. Only a missing/None value should fall back to 1."""
    assert resolve_add_position_max_times({"max_add_times": 0}) == 0
    cfg = {
        "sizing_mode": "target_leverage_gap",
        "target_leverage_by_add": [3.0],
    }
    signal = {
        "current_leverage": 1.8,
        "base_leverage_unit": 1.2,
    }
    mult = resolve_add_position_size_multiplier(cfg, 1, signal)
    # gap = 1.2, base_unit=1.2 -> 1.0x
    assert mult == pytest.approx(1.0)


def test_target_leverage_gap_respects_max_total_leverage():
    cfg = {
        "sizing_mode": "target_leverage_gap",
        "target_leverage_by_add": [5.0],
        "max_total_leverage": 3.0,
    }
    signal = {
        "current_leverage": 2.8,
        "base_leverage_unit": 1.0,
    }
    mult = resolve_add_position_size_multiplier(cfg, 1, signal)
    # target gap=2.2, but max_total room=0.2
    assert mult == pytest.approx(0.2)


def test_target_leverage_gap_falls_back_when_gap_non_positive():
    cfg = {
        "sizing_mode": "target_leverage_gap",
        "target_leverage_by_add": [2.0],
        "add_size_multipliers": [0.35],
    }
    signal = {
        "current_leverage": 2.2,
        "base_leverage_unit": 1.0,
    }
    mult = resolve_add_position_size_multiplier(cfg, 1, signal)
    assert mult == pytest.approx(0.35)


def test_target_leverage_gap_applies_notional_caps():
    cfg = {
        "sizing_mode": "target_leverage_gap",
        "target_leverage_by_add": [5.0],
        "max_add_notional_frac": 0.30,
    }
    signal = {
        "current_leverage": 1.0,
        "base_leverage_unit": 1.0,
        "base_notional_frac": 0.10,
        "current_notional_frac": 0.25,
    }
    mult = resolve_add_position_size_multiplier(cfg, 1, signal)
    # notional room = 0.05, base_notional_frac=0.10 -> 0.5x
    assert mult == pytest.approx(0.5)


def test_resolve_float_r_ladder_only_from_trigger_type_only():
    assert (
        resolve_float_r_ladder_only({"trigger": {"type": "float_r_ladder_only"}})
        is True
    )
    assert resolve_float_r_ladder_only({}) is False
    assert resolve_float_r_ladder_only({"trigger": {}}) is False
    assert resolve_float_r_ladder_only({"float_r_ladder_only": True}) is False


def test_resolve_passive_add_ladder_includes_pullback_near_ema():
    from src.time_series_model.core.constitution.add_position_rules import (
        resolve_passive_add_ladder,
        resolve_pullback_near_ema1200,
    )

    pb = {
        "trigger": {"type": "pullback_near_ema1200", "max_abs_ema_1200_position": 0.02}
    }
    assert resolve_pullback_near_ema1200(pb) is True
    assert resolve_passive_add_ladder(pb) is True
    assert resolve_float_r_ladder_only(pb) is False
    assert (
        resolve_passive_add_ladder({"trigger": {"type": "float_r_ladder_only"}}) is True
    )
    assert resolve_passive_add_ladder({}) is False


def test_pullback_near_ema1200_allows_abs_band_and_rejects_far():
    from src.time_series_model.core.constitution.add_position_rules import (
        pullback_near_ema1200_allows,
    )

    cfg = {
        "trigger": {"type": "pullback_near_ema1200", "max_abs_ema_1200_position": 0.02}
    }
    ok, why = pullback_near_ema1200_allows({"ema_1200_position": 0.01}, cfg)
    assert ok is True and why == ""
    ok, why = pullback_near_ema1200_allows({"ema_1200_position": -0.019}, cfg)
    assert ok is True
    ok, why = pullback_near_ema1200_allows({"ema_1200_position": 0.05}, cfg)
    assert ok is False and "ema_pos_abs" in why
    ok, why = pullback_near_ema1200_allows({}, cfg)
    assert ok is False and "missing" in why


def test_pullback_near_ema1200_optional_dist_atr():
    from src.time_series_model.core.constitution.add_position_rules import (
        pullback_near_ema1200_allows,
    )

    cfg = {
        "trigger": {
            "type": "pullback_near_ema1200",
            "max_abs_ema_1200_position": 0.05,
            "max_dist_atr": 0.5,
        }
    }
    # |ep|*close/atr = 0.02*100/2 = 1.0 > 0.5 → reject
    ok, why = pullback_near_ema1200_allows(
        {"ema_1200_position": 0.02, "close": 100.0, "atr": 2.0}, cfg
    )
    assert ok is False and "dist_atr" in why
    ok, why = pullback_near_ema1200_allows(
        {"ema_1200_position": 0.005, "close": 100.0, "atr": 2.0}, cfg
    )
    assert ok is True


def test_validate_add_trigger_pullback_checks_near_ema():
    cfg = {
        "min_current_r_by_add": [0.5],
        "trigger": {"type": "pullback_near_ema1200", "max_abs_ema_1200_position": 0.02},
    }
    assert (
        validate_add_position_trigger(
            archetype="srb",
            direction=1,
            signal={"add_position_seq": 1, "ema_1200_position": 0.01},
            add_position_cfg=cfg,
            current_r=0.6,
        )
        is True
    )
    assert (
        validate_add_position_trigger(
            archetype="srb",
            direction=1,
            signal={"add_position_seq": 1, "ema_1200_position": 0.08},
            add_position_cfg=cfg,
            current_r=0.6,
        )
        is False
    )


def test_validate_add_trigger_float_r_ladder_only_only_checks_min_r():
    cfg = {
        "min_current_r_by_add": [0.5],
        "trigger": {"type": "float_r_ladder_only"},
    }
    signal = {"add_position_seq": 1}
    assert (
        validate_add_position_trigger(
            archetype="bpc-long-120T",
            direction=1,
            signal=signal,
            add_position_cfg=cfg,
            current_r=0.6,
        )
        is True
    )


def test_validate_add_trigger_without_trigger_still_checks_min_current_r():
    cfg = {
        "min_current_r_by_add": [0.5, 1.0, 1.5],
    }
    signal = {"add_position_seq": 2}
    ok = validate_add_position_trigger(
        archetype="bpc-long-120T",
        direction=1,
        signal=signal,
        add_position_cfg=cfg,
        current_r=0.8,  # below add #2 threshold 1.0
    )
    assert ok is False


def test_validate_add_trigger_without_trigger_passes_when_min_current_r_met():
    cfg = {
        "min_current_r_by_add": [0.5, 1.0, 1.5],
    }
    signal = {"add_position_seq": 2}
    ok = validate_add_position_trigger(
        archetype="bpc-long-120T",
        direction=1,
        signal=signal,
        add_position_cfg=cfg,
        current_r=1.05,  # above add #2 threshold 1.0
    )
    assert ok is True


# ── signal_add 路径回归测试（2026-06-10 backtester.py 修复后）──


def test_no_trigger_type_means_signal_add_path():
    """无 trigger.type 时 resolve_float_r_ladder_only 返回 False → signal_add 路径。

    事件回测中 _strats_float_ladder_meta 仅包含 float_r_ladder_only archetype，
    无 trigger 的 archetype 走 signal_add（PCM 再信号时加仓）。
    此前 backtester.py 的 _dup_open elif 在 _add_pos_enabled 之前拦截导致 signal_add 永为 0。
    """
    cfg_no_trigger = {
        "add_size_multipliers": [0.5, 0.25],
        "min_current_r_by_add": [0.5, 1],
    }
    assert resolve_float_r_ladder_only(cfg_no_trigger) is False

    cfg_empty_trigger = {
        "add_size_multipliers": [0.5],
        "trigger": {},
    }
    assert resolve_float_r_ladder_only(cfg_empty_trigger) is False


def test_signal_add_path_trigger_validates_min_r_only():
    """signal_add 路径（无 trigger.type）仅检查 min_current_r_by_add，不检查特征。

    这是设计意图：signal_add 的信号已经通过了 PCM 的 entry pipeline
    （prefilter→gate→direction→entry_filter→PCM仲裁），加仓时只需确认浮盈门槛。
    """
    cfg = {
        "min_current_r_by_add": [0.5, 1.0],
        # 无 trigger — signal_add 路径
    }
    signal = {"add_position_seq": 1, "position_action": "LONG"}
    # current_r 低于门槛 → 拒绝
    assert (
        validate_add_position_trigger(
            archetype="tpc",
            direction=1,
            signal=signal,
            add_position_cfg=cfg,
            current_r=0.3,
        )
        is False
    )
    # current_r 高于门槛 → 通过（不检查任何特征）
    assert (
        validate_add_position_trigger(
            archetype="tpc",
            direction=1,
            signal=signal,
            add_position_cfg=cfg,
            current_r=0.6,
        )
        is True
    )


def test_signal_add_vs_float_ladder_trigger_distinction():
    """确保 signal_add（无 trigger）和 float_r_ladder_only 的区分不会退化。

    两者都只检查 min_current_r_by_add，但路径不同：
    - signal_add: PCM 每次给新信号时触发一次（sparse）
    - float_r_ladder_only: 每 bar 检查（dense）
    """
    signal_cfg = {"min_current_r_by_add": [0.5]}
    float_cfg = {
        "min_current_r_by_add": [0.5],
        "trigger": {"type": "float_r_ladder_only"},
    }

    assert resolve_float_r_ladder_only(signal_cfg) is False
    assert resolve_float_r_ladder_only(float_cfg) is True

    signal = {"add_position_seq": 1}
    # 相同 current_r 下两者 trigger 验证结果一致
    assert (
        validate_add_position_trigger(
            archetype="tpc",
            direction=1,
            signal=signal,
            add_position_cfg=signal_cfg,
            current_r=0.6,
        )
        is True
    )
    assert (
        validate_add_position_trigger(
            archetype="tpc",
            direction=1,
            signal=signal,
            add_position_cfg=float_cfg,
            current_r=0.6,
        )
        is True
    )


def test_latest_add_stop_price_long_and_short():
    assert latest_add_stop_price(
        side="LONG", add_entry=76.7, sl_distance=12.36
    ) == pytest.approx(64.34)
    assert latest_add_stop_price(
        side="SHORT", add_entry=76.7, sl_distance=12.36
    ) == pytest.approx(89.06)
    assert latest_add_stop_price(side="LONG", add_entry=0.0, sl_distance=12.0) is None


def test_tighten_stop_price_never_loosens():
    assert tighten_stop_price(
        side="LONG", current=63.0, candidate=64.3
    ) == pytest.approx(64.3)
    assert tighten_stop_price(
        side="LONG", current=64.3, candidate=63.0
    ) == pytest.approx(64.3)
    assert tighten_stop_price(
        side="SHORT", current=80.0, candidate=78.0
    ) == pytest.approx(78.0)
    assert tighten_stop_price(
        side="SHORT", current=78.0, candidate=80.0
    ) == pytest.approx(78.0)


def test_apply_latest_add_stop_ratchet_moves_parent_and_add():
    parent = {
        "entry_price": 73.0,
        "stop_loss_price": 63.0,
        "side": "LONG",
    }
    add = {
        "_parent_pid": "p1",
        "entry_price": 76.7,
        "stop_loss_price": 63.0,
        "side": "LONG",
    }
    other = {"entry_price": 10.0, "stop_loss_price": 9.0, "side": "LONG"}
    positions = {"p1": parent, "a1": add, "x": other}
    new_sl = apply_latest_add_stop_ratchet(
        positions, parent_pid="p1", add_entry=76.7, side="LONG"
    )
    assert new_sl == pytest.approx(66.7)
    assert parent["stop_loss_price"] == pytest.approx(66.7)
    assert add["stop_loss_price"] == pytest.approx(66.7)
    assert other["stop_loss_price"] == pytest.approx(9.0)
    apply_latest_add_stop_ratchet(
        positions, parent_pid="p1", add_entry=77.5, side="LONG"
    )
    assert parent["stop_loss_price"] == pytest.approx(67.5)
    assert parent["_origin_sl_distance"] == pytest.approx(10.0)
