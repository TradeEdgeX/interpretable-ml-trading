"""Soft day/week period-loss policy (shared live + EB)."""

from __future__ import annotations

from types import SimpleNamespace

from src.time_series_model.core.constitution.period_loss_policy import (
    effective_risk_per_slot,
    hard_period_limits_for_evaluate,
    safety_limits_from_cfg,
    soft_config_from_cfg,
    soft_peak_dd_block_adds,
)
from src.time_series_model.core.constitution.safety_runtime import (
    SafetyRuntimeState,
    evaluate_safety_state,
)
from datetime import datetime, timezone


def test_soft_disables_hard_day_week_in_limits() -> None:
    cfg = SimpleNamespace(
        max_dd=0.30,
        daily_loss_limit=0.08,
        weekly_loss_limit=0.12,
        monthly_loss_limit=0.12,
        max_turnover_mean=1.0,
        max_cost_mean=1.0,
        daily_soft_loss_limit=0.08,
        weekly_soft_loss_limit=0.12,
        derated_risk_per_slot=0.005,
    )
    limits = safety_limits_from_cfg(cfg)
    assert limits["daily_loss_limit"] == 1.0
    assert limits["weekly_loss_limit"] == 1.0
    assert limits["max_dd"] == 0.30
    assert limits["monthly_loss_limit"] == 0.12


def test_soft_day_loss_does_not_halt_evaluate() -> None:
    cfg = SimpleNamespace(
        max_dd=0.30,
        daily_loss_limit=0.08,
        weekly_loss_limit=0.12,
        monthly_loss_limit=0.12,
        max_turnover_mean=1.0,
        max_cost_mean=1.0,
        daily_soft_loss_limit=0.08,
        weekly_soft_loss_limit=0.12,
        derated_risk_per_slot=0.005,
    )
    decision = evaluate_safety_state(
        state=SafetyRuntimeState(),
        now=datetime(2026, 7, 26, 12, 0, tzinfo=timezone.utc),
        cooldown_minutes=720,
        daily_reset_tz="UTC",
        daily_loss=0.10,
        weekly_loss=0.0,
        monthly_loss=0.0,
        drawdown=0.05,
        hard_violation=False,
        data_bad=False,
        daily_cost_mean=None,
        daily_turnover_mean=None,
        limits=safety_limits_from_cfg(cfg),
    )
    assert decision.ok is True
    assert decision.reasons == []


def test_max_dd_still_hard_halts_under_soft_day_week() -> None:
    cfg = SimpleNamespace(
        max_dd=0.30,
        daily_loss_limit=1.0,
        weekly_loss_limit=1.0,
        monthly_loss_limit=0.12,
        max_turnover_mean=1.0,
        max_cost_mean=1.0,
        daily_soft_loss_limit=0.08,
        weekly_soft_loss_limit=0.12,
        derated_risk_per_slot=0.005,
    )
    decision = evaluate_safety_state(
        state=SafetyRuntimeState(),
        now=datetime(2026, 7, 26, 12, 0, tzinfo=timezone.utc),
        cooldown_minutes=720,
        daily_reset_tz="UTC",
        daily_loss=0.10,
        weekly_loss=0.0,
        monthly_loss=0.0,
        drawdown=0.31,
        hard_violation=False,
        data_bad=False,
        daily_cost_mean=None,
        daily_turnover_mean=None,
        limits=safety_limits_from_cfg(cfg),
    )
    assert decision.ok is False
    assert "max_dd" in decision.reasons


def test_effective_risk_derates_on_soft_hit() -> None:
    soft = soft_config_from_cfg(
        SimpleNamespace(
            daily_soft_loss_limit=0.08,
            weekly_soft_loss_limit=0.12,
            derated_risk_per_slot=0.005,
        )
    )
    risk, hit = effective_risk_per_slot(
        base_risk=0.01, daily_loss=0.09, weekly_loss=0.0, soft=soft
    )
    assert hit is True
    assert risk == 0.005
    d, w = hard_period_limits_for_evaluate(
        daily_loss_limit=0.08, weekly_loss_limit=0.12, soft=soft
    )
    assert d == 1.0 and w == 1.0


def test_soft_peak_dd_block_adds_and_derate() -> None:
    soft = {
        "daily_soft_loss_limit": 0.0,
        "weekly_soft_loss_limit": 0.0,
        "derated_risk_per_slot": 0.005,
        "soft_max_dd_block_adds": 0.15,
        "soft_max_dd_derate": 0.15,
    }
    assert soft_peak_dd_block_adds(peak_drawdown=0.14, soft=soft) is False
    assert soft_peak_dd_block_adds(peak_drawdown=0.15, soft=soft) is True
    risk, hit = effective_risk_per_slot(
        base_risk=0.01,
        daily_loss=0.0,
        weekly_loss=0.0,
        soft=soft,
        peak_drawdown=0.16,
    )
    assert hit is True
    assert risk == 0.005
    risk2, hit2 = effective_risk_per_slot(
        base_risk=0.01,
        daily_loss=0.0,
        weekly_loss=0.0,
        soft=soft,
        peak_drawdown=0.10,
    )
    assert hit2 is False
    assert risk2 == 0.01
