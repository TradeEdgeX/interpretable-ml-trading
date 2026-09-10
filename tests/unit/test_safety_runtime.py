from datetime import datetime, timedelta, timezone

import pytest

from src.time_series_model.core.constitution.safety_runtime import (
    SafetyRuntimeState,
    evaluate_safety_state,
)


@pytest.mark.unit
def test_safety_daily_loss_requires_next_day_to_recover():
    st = SafetyRuntimeState()
    limits = {
        "max_dd": 0.2,
        "daily_loss_limit": 0.04,
        "weekly_loss_limit": 0.08,
        "monthly_loss_limit": 0.12,
        "max_turnover_mean": 0.35,
        "max_cost_mean": 0.002,
    }
    now = datetime(2026, 1, 27, 12, 0, tzinfo=timezone.utc)

    decision = evaluate_safety_state(
        state=st,
        now=now,
        cooldown_minutes=0,
        daily_reset_tz="UTC",
        daily_loss=0.05,
        weekly_loss=0.0,
        monthly_loss=0.0,
        drawdown=0.0,
        hard_violation=False,
        data_bad=False,
        daily_cost_mean=None,
        daily_turnover_mean=None,
        limits=limits,
    )
    assert decision.ok is False
    assert decision.state.halted is True
    assert "daily_loss_limit" in decision.reasons
    assert decision.state.last_daily_halt_date == "2026-01-27"

    # Same day, metrics recover -> should still halt
    later_same_day = now + timedelta(hours=6)
    decision_same_day = evaluate_safety_state(
        state=decision.state,
        now=later_same_day,
        cooldown_minutes=0,
        daily_reset_tz="UTC",
        daily_loss=0.0,
        weekly_loss=0.0,
        monthly_loss=0.0,
        drawdown=0.0,
        hard_violation=False,
        data_bad=False,
        daily_cost_mean=None,
        daily_turnover_mean=None,
        limits=limits,
    )
    assert decision_same_day.ok is False
    assert decision_same_day.state.halted is True

    # Next day -> recover
    next_day = now + timedelta(days=1, minutes=1)
    decision_next_day = evaluate_safety_state(
        state=decision_same_day.state,
        now=next_day,
        cooldown_minutes=0,
        daily_reset_tz="UTC",
        daily_loss=0.0,
        weekly_loss=0.0,
        monthly_loss=0.0,
        drawdown=0.0,
        hard_violation=False,
        data_bad=False,
        daily_cost_mean=None,
        daily_turnover_mean=None,
        limits=limits,
    )
    assert decision_next_day.ok is True
    assert decision_next_day.state.halted is False


@pytest.mark.unit
def test_safety_cooldown_and_metric_recovery():
    """Test that cooldown + metric recovery both required to lift halt."""
    st = SafetyRuntimeState()
    limits = {
        "max_dd": 0.2,
        "daily_loss_limit": 0.04,
        "weekly_loss_limit": 0.08,
        "monthly_loss_limit": 0.12,
        "max_turnover_mean": 0.35,
        "max_cost_mean": 0.002,
    }
    now = datetime(2026, 1, 27, 12, 0, tzinfo=timezone.utc)

    # Trigger halt with max_dd
    decision = evaluate_safety_state(
        state=st,
        now=now,
        cooldown_minutes=60,  # 1 hour cooldown
        daily_reset_tz="UTC",
        daily_loss=0.0,
        weekly_loss=0.0,
        monthly_loss=0.0,
        drawdown=0.25,  # Exceeds 0.2 limit
        hard_violation=False,
        data_bad=False,
        daily_cost_mean=None,
        daily_turnover_mean=None,
        limits=limits,
    )
    assert decision.ok is False
    assert decision.state.halted is True
    assert "max_dd" in decision.reasons
    assert decision.state.cooldown_until is not None

    # Metrics recover but cooldown not expired -> still halt
    later_before_cooldown = now + timedelta(minutes=30)
    decision_before_cooldown = evaluate_safety_state(
        state=decision.state,
        now=later_before_cooldown,
        cooldown_minutes=60,
        daily_reset_tz="UTC",
        daily_loss=0.0,
        weekly_loss=0.0,
        monthly_loss=0.0,
        drawdown=0.1,  # Recovered below limit
        hard_violation=False,
        data_bad=False,
        daily_cost_mean=None,
        daily_turnover_mean=None,
        limits=limits,
    )
    assert decision_before_cooldown.ok is False
    assert decision_before_cooldown.state.halted is True

    # Cooldown expired + metrics recovered -> should recover
    later_after_cooldown = now + timedelta(minutes=61)
    decision_after_cooldown = evaluate_safety_state(
        state=decision_before_cooldown.state,
        now=later_after_cooldown,
        cooldown_minutes=60,
        daily_reset_tz="UTC",
        daily_loss=0.0,
        weekly_loss=0.0,
        monthly_loss=0.0,
        drawdown=0.1,  # Still recovered
        hard_violation=False,
        data_bad=False,
        daily_cost_mean=None,
        daily_turnover_mean=None,
        limits=limits,
    )
    assert decision_after_cooldown.ok is True
    assert decision_after_cooldown.state.halted is False
    assert decision_after_cooldown.state.cooldown_until is None
    assert decision_after_cooldown.state.halt_since is None


@pytest.mark.unit
def test_safety_cooldown_only_not_sufficient():
    """Test that cooldown alone is not sufficient if metrics still violate."""
    st = SafetyRuntimeState()
    limits = {
        "max_dd": 0.2,
        "daily_loss_limit": 0.04,
        "weekly_loss_limit": 0.08,
        "monthly_loss_limit": 0.12,
        "max_turnover_mean": 0.35,
        "max_cost_mean": 0.002,
    }
    now = datetime(2026, 1, 27, 12, 0, tzinfo=timezone.utc)

    # Trigger halt
    decision = evaluate_safety_state(
        state=st,
        now=now,
        cooldown_minutes=30,
        daily_reset_tz="UTC",
        daily_loss=0.0,
        weekly_loss=0.0,
        monthly_loss=0.0,
        drawdown=0.25,
        hard_violation=False,
        data_bad=False,
        daily_cost_mean=None,
        daily_turnover_mean=None,
        limits=limits,
    )
    assert decision.ok is False
    assert decision.state.halted is True

    # Cooldown expired but metrics still violate -> should still halt
    later_after_cooldown = now + timedelta(minutes=31)
    decision_after_cooldown = evaluate_safety_state(
        state=decision.state,
        now=later_after_cooldown,
        cooldown_minutes=30,
        daily_reset_tz="UTC",
        daily_loss=0.0,
        weekly_loss=0.0,
        monthly_loss=0.0,
        drawdown=0.25,  # Still violates
        hard_violation=False,
        data_bad=False,
        daily_cost_mean=None,
        daily_turnover_mean=None,
        limits=limits,
    )
    assert decision_after_cooldown.ok is False
    assert decision_after_cooldown.state.halted is True
    assert "max_dd" in decision_after_cooldown.reasons
