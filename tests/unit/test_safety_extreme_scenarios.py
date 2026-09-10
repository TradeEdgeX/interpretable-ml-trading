"""
Extreme scenario tests for safety module: verify it prevents excessive drawdown
by halting trading when market conditions are extreme.

These tests construct extreme market data sequences (flash crashes, continuous drawdowns)
and verify that safety halts trading before drawdown exceeds limits.
"""

from datetime import datetime, timedelta, timezone

import pytest

from src.time_series_model.core.constitution.constitution_executor import (
    ConstitutionExecutor,
)
from src.time_series_model.core.constitution.violation import ConstitutionViolation
from src.time_series_model.live.enforcement import enforce_before_order
from src.time_series_model.core.constitution.runtime_state import (
    ConstitutionRuntimeState,
)


@pytest.mark.unit
def test_safety_halts_on_flash_crash(tmp_path):
    """
    Scenario: Flash crash - price drops 30% in 1 hour.
    Expected: Safety halts after drawdown exceeds max_dd, preventing further losses.
    """
    cy = tmp_path / "constitution.yaml"
    cy.write_text(
        """
version: 1
name: "C_EXTREME"
kill_switch:
  enabled: true
  max_dd: 0.20
  daily_loss_limit: 0.10
  weekly_loss_limit: 0.15
  monthly_loss_limit: 0.20
  kill_on_any_hard_violation: true
  cooldown_minutes: 240
safety_state:
  persist_to: "{tmp_path}/safety.db"
slots:
  enabled: true
  slot_count: 2
  risk_per_slot: 0.01
add_position:
  enabled: false
""".replace(
            "{tmp_path}", str(tmp_path)
        ),
        encoding="utf-8",
    )

    ex = ConstitutionExecutor(constitution_yaml=str(cy))
    st = ex.load_runtime_state()

    # Simulate normal trading (drawdown within limits)
    res1 = enforce_before_order(
        executor=ex,
        runtime_state=st,
        position_id="p1",
        symbol="BTCUSDT",
        archetype="TrendContinuationTC",
        drawdown=0.05,  # 5% drawdown, OK
        daily_loss=0.02,
    )
    assert res1.ok is True

    # Flash crash: drawdown jumps to 25% (exceeds 20% limit)
    st2 = ex.load_runtime_state()
    with pytest.raises(ConstitutionViolation) as exc_info:
        enforce_before_order(
            executor=ex,
            runtime_state=st2,
            position_id="p2",
            symbol="BTCUSDT",
            archetype="TrendContinuationTC",
            drawdown=0.25,  # 25% drawdown, exceeds 20% limit
            daily_loss=0.08,
        )
    assert "max_dd" in str(exc_info.value.message) or "SAFETY_HALT" in str(
        exc_info.value.code
    )

    # After halt, no new orders should be allowed even if drawdown recovers slightly
    st3 = ex.load_runtime_state()
    with pytest.raises(ConstitutionViolation):
        enforce_before_order(
            executor=ex,
            runtime_state=st3,
            position_id="p3",
            symbol="ETHUSDT",
            archetype="MomentumExpansion",
            drawdown=0.18,  # Still high but below limit
            daily_loss=0.05,
        )


@pytest.mark.unit
def test_safety_halts_on_continuous_drawdown(tmp_path):
    """
    Scenario: Continuous drawdown over multiple days (death spiral).
    Expected: Safety halts after daily loss limit, preventing further exposure.
    """
    cy = tmp_path / "constitution.yaml"
    cy.write_text(
        """
version: 1
name: "C_CONTINUOUS"
kill_switch:
  enabled: true
  max_dd: 0.20
  daily_loss_limit: 0.05
  weekly_loss_limit: 0.10
  monthly_loss_limit: 0.15
  kill_on_any_hard_violation: true
  cooldown_minutes: 240
safety_state:
  persist_to: "{tmp_path}/safety.db"
slots:
  enabled: true
  slot_count: 2
add_position:
  enabled: false
""".replace(
            "{tmp_path}", str(tmp_path)
        ),
        encoding="utf-8",
    )

    ex = ConstitutionExecutor(constitution_yaml=str(cy))
    st = ex.load_runtime_state()

    # Day 1: Small loss, OK
    res1 = enforce_before_order(
        executor=ex,
        runtime_state=st,
        position_id="p1",
        symbol="BTCUSDT",
        archetype="TrendContinuationTC",
        drawdown=0.03,
        daily_loss=0.02,  # 2% daily loss, OK
    )
    assert res1.ok is True

    # Day 2: Loss accumulates
    st2 = ex.load_runtime_state()
    res2 = enforce_before_order(
        executor=ex,
        runtime_state=st2,
        position_id="p2",
        symbol="ETHUSDT",
        archetype="MomentumExpansion",
        drawdown=0.08,
        daily_loss=0.04,  # 4% daily loss, still OK
    )
    assert res2.ok is True

    # Day 3: Daily loss exceeds limit -> halt
    st3 = ex.load_runtime_state()
    with pytest.raises(ConstitutionViolation) as exc_info:
        enforce_before_order(
            executor=ex,
            runtime_state=st3,
            position_id="p3",
            symbol="BTCUSDT",
            archetype="FailedBreakoutFade",
            drawdown=0.12,
            daily_loss=0.06,  # 6% daily loss, exceeds 5% limit
        )
    assert "daily_loss_limit" in str(exc_info.value.message) or "SAFETY_HALT" in str(
        exc_info.value.code
    )


@pytest.mark.unit
def test_safety_halts_on_weekly_loss_accumulation(tmp_path):
    """
    Scenario: Multiple days of losses accumulate to exceed weekly limit.
    Expected: Safety halts when weekly loss exceeds limit.
    """
    cy = tmp_path / "constitution.yaml"
    cy.write_text(
        """
version: 1
name: "C_WEEKLY"
kill_switch:
  enabled: true
  max_dd: 0.20
  daily_loss_limit: 0.10
  weekly_loss_limit: 0.08
  monthly_loss_limit: 0.15
  kill_on_any_hard_violation: true
safety_state:
  persist_to: "{tmp_path}/safety.db"
slots:
  enabled: true
  slot_count: 2
add_position:
  enabled: false
""".replace(
            "{tmp_path}", str(tmp_path)
        ),
        encoding="utf-8",
    )

    ex = ConstitutionExecutor(constitution_yaml=str(cy))
    st = ex.load_runtime_state()

    # Day 1-2: Small daily losses, each below daily limit, weekly loss accumulating
    for day in range(2):
        st_day = ex.load_runtime_state()
        res = enforce_before_order(
            executor=ex,
            runtime_state=st_day,
            position_id=f"p{day+1}",
            symbol="BTCUSDT",
            archetype="TrendContinuationTC",
            drawdown=0.05 + day * 0.01,
            daily_loss=0.03,  # Each day 3% loss, below 10% daily limit
            weekly_loss=0.03 * (day + 1),  # Day 1: 3%, Day 2: 6%
        )
        assert res.ok is True

    # Day 3: Weekly loss exceeds 8% limit -> halt (6% + 3% = 9% > 8%)
    st3 = ex.load_runtime_state()
    with pytest.raises(ConstitutionViolation) as exc_info:
        enforce_before_order(
            executor=ex,
            runtime_state=st3,
            position_id="p3",
            symbol="BTCUSDT",
            archetype="MomentumExpansion",
            drawdown=0.10,
            daily_loss=0.03,
            weekly_loss=0.09,  # 9% weekly loss, exceeds 8% limit
        )
    assert "weekly_loss_limit" in str(exc_info.value.message) or "SAFETY_HALT" in str(
        exc_info.value.code
    )


@pytest.mark.unit
def test_safety_prevents_excessive_drawdown_via_slots(tmp_path):
    """
    Scenario: Multiple positions open during drawdown period.
    Expected: Slot cap prevents opening too many positions, limiting exposure.
    """
    cy = tmp_path / "constitution.yaml"
    cy.write_text(
        """
version: 1
name: "C_SLOTS"
kill_switch:
  enabled: true
  max_dd: 0.20
  daily_loss_limit: 0.10
safety_state:
  persist_to: "{tmp_path}/safety.db"
slots:
  enabled: true
  slot_count: 2
  risk_per_slot: 0.01
  slot_state_tracking:
    persist_to: "{tmp_path}/state/slots.json"
add_position:
  enabled: false
""".replace(
            "{tmp_path}", str(tmp_path)
        ),
        encoding="utf-8",
    )

    ex = ConstitutionExecutor(constitution_yaml=str(cy))
    st = ex.load_runtime_state()

    # Open first position (drawdown moderate)
    res1 = enforce_before_order(
        executor=ex,
        runtime_state=st,
        position_id="p1",
        symbol="BTCUSDT",
        archetype="TrendContinuationTC",
        drawdown=0.10,
    )
    assert res1.ok is True
    # State is saved by enforce_before_order, reload to get updated slots
    st = ex.load_runtime_state()
    assert st.slots.active_count() == 1

    # Open second position (drawdown still moderate)
    res2 = enforce_before_order(
        executor=ex,
        runtime_state=st,
        position_id="p2",
        symbol="ETHUSDT",
        archetype="MomentumExpansion",
        drawdown=0.12,
    )
    assert res2.ok is True
    # State is saved, reload to get updated slots
    st = ex.load_runtime_state()
    assert st.slots.active_count() == 2

    # Try third position -> slot cap prevents it (2 slots already full)
    with pytest.raises(ConstitutionViolation) as exc_info:
        enforce_before_order(
            executor=ex,
            runtime_state=st,
            position_id="p3",
            symbol="DOGEUSDT",
            archetype="FailedBreakoutFade",
            drawdown=0.15,  # Still below 20% max_dd, but slots full
        )
    assert "slot" in str(exc_info.value.message).lower() or "SLOT" in str(
        exc_info.value.code
    )
