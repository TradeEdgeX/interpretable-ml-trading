"""Unit tests for SRB scale_out_bank research helper."""

from __future__ import annotations

from src.time_series_model.live.scale_out_bank import (
    apply_partial_sell_to_sim_position,
    evaluate_scale_out_bank,
    mother_return_pct,
)


def test_mother_return_pct_long():
    assert mother_return_pct(entry_price=100.0, close=125.0, is_long=True) == 0.25


def test_pct_mode_requires_full_adds():
    cfg = {
        "enabled": True,
        "mode": "pct",
        "min_profit_pct": 0.20,
        "require_adds_full": True,
        "min_add_count": 3,
        "sell_fraction": 0.5,
    }
    frac, reason = evaluate_scale_out_bank(
        cfg=cfg,
        entry_price=100.0,
        close=125.0,
        is_long=True,
        add_leg_count=2,
        already_done=False,
    )
    assert frac is None
    assert reason == ""

    frac, reason = evaluate_scale_out_bank(
        cfg=cfg,
        entry_price=100.0,
        close=125.0,
        is_long=True,
        add_leg_count=3,
        already_done=False,
    )
    assert frac == 0.5
    assert "mode=pct" in reason


def test_pct_mode_once_blocks_repeat():
    cfg = {"enabled": True, "mode": "pct", "min_profit_pct": 0.20, "sell_fraction": 0.5}
    frac, _ = evaluate_scale_out_bank(
        cfg=cfg,
        entry_price=100.0,
        close=130.0,
        is_long=True,
        add_leg_count=3,
        already_done=True,
    )
    assert frac is None


def test_wyckoff_mode_fail_closed_without_features():
    cfg = {
        "enabled": True,
        "mode": "wyckoff",
        "min_profit_pct": 0.15,
        "volume_ratio_pct_min": 0.80,
        "sell_fraction": 0.5,
        "min_add_count": 3,
    }
    frac, reason = evaluate_scale_out_bank(
        cfg=cfg,
        entry_price=100.0,
        close=120.0,
        is_long=True,
        add_leg_count=3,
        already_done=False,
        features=None,
    )
    assert frac is None
    assert reason == ""


def test_wyckoff_mode_triggers_on_high_vol_and_weak_efficiency():
    cfg = {
        "enabled": True,
        "mode": "wyckoff",
        "min_profit_pct": 0.15,
        "volume_ratio_pct_min": 0.80,
        "sell_fraction": 0.5,
        "min_add_count": 3,
    }
    feats = {
        "volume_ratio_pct": 0.95,
        "fer_signed_efficiency_pct": 0.20,
    }
    frac, reason = evaluate_scale_out_bank(
        cfg=cfg,
        entry_price=100.0,
        close=118.0,
        is_long=True,
        add_leg_count=3,
        already_done=False,
        features=feats,
    )
    assert frac == 0.5
    assert "mode=wyckoff" in reason


def test_apply_partial_sell_scales_book():
    pos = {
        "_qty_base": 2.0,
        "_entry_notional_usdt": 200.0,
        "_entry_fee_usdt": 1.0,
        "_size_multiplier": 4.0,
        "_contracts": 10.0,
    }
    apply_partial_sell_to_sim_position(pos, sell_qty=1.0, exit_price=110.0)
    assert pos["_qty_base"] == 1.0
    assert pos["_entry_notional_usdt"] == 100.0
    assert pos["_size_multiplier"] == 2.0
    assert pos["_contracts"] == 5.0
