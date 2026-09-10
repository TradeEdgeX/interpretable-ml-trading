"""Unit tests for event_backtest soft day/week KS policy helper."""

from __future__ import annotations

from scripts.event_backtest.ks_period_policy import (
    effective_risk_per_slot,
    hard_period_limits_for_evaluate,
    load_ks_period_soft_config,
)


def test_hard_limits_passthrough_when_no_soft() -> None:
    soft = load_ks_period_soft_config({})
    d, w = hard_period_limits_for_evaluate(
        daily_loss_limit=0.06, weekly_loss_limit=0.08, soft=soft
    )
    assert d == 0.06 and w == 0.08


def test_soft_disables_hard_day_week() -> None:
    soft = load_ks_period_soft_config(
        {
            "daily_soft_loss_limit": 0.06,
            "weekly_soft_loss_limit": 0.08,
            "derated_risk_per_slot": 0.005,
        }
    )
    d, w = hard_period_limits_for_evaluate(
        daily_loss_limit=1.0, weekly_loss_limit=1.0, soft=soft
    )
    assert d == 1.0 and w == 1.0


def test_effective_risk_derates_on_soft_hit() -> None:
    soft = {
        "daily_soft_loss_limit": 0.06,
        "weekly_soft_loss_limit": 0.08,
        "derated_risk_per_slot": 0.005,
    }
    risk, hit = effective_risk_per_slot(
        base_risk=0.01, daily_loss=0.07, weekly_loss=0.0, soft=soft
    )
    assert hit is True
    assert risk == 0.005
    risk2, hit2 = effective_risk_per_slot(
        base_risk=0.01, daily_loss=0.01, weekly_loss=0.0, soft=soft
    )
    assert hit2 is False
    assert risk2 == 0.01
