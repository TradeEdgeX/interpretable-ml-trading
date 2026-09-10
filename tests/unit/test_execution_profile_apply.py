from datetime import datetime, timedelta, timezone

from src.time_series_model.live.execution_profile_apply import (
    compute_rr_prices,
    compute_trailing_stop,
    holding_expired,
    rr_constraints_from_exec_params,
)
from src.time_series_model.live.generic_live_strategy import ExecutionParamGenerator


def test_compute_rr_prices_long() -> None:
    sl, tp = compute_rr_prices(
        side="LONG", entry_price=100.0, atr=2.0, stop_loss_r=1.0, take_profit_r=2.5
    )
    assert sl == 98.0
    assert tp == 105.0


def test_compute_rr_prices_short() -> None:
    sl, tp = compute_rr_prices(
        side="SHORT", entry_price=100.0, atr=2.0, stop_loss_r=1.0, take_profit_r=2.5
    )
    assert sl == 102.0
    assert tp == 95.0


def test_compute_trailing_stop_long() -> None:
    stop = compute_trailing_stop(
        side="LONG", current_price=110.0, atr=2.0, trailing_atr=0.5
    )
    assert stop == 109.0


def test_compute_trailing_stop_short() -> None:
    stop = compute_trailing_stop(
        side="SHORT", current_price=90.0, atr=2.0, trailing_atr=0.5
    )
    assert stop == 91.0


def test_holding_expired() -> None:
    entry = datetime(2026, 1, 1, tzinfo=timezone.utc)
    now = entry + timedelta(hours=8)
    assert holding_expired(
        entry_time=entry, now=now, max_holding_bars=2, bar_minutes=240
    )


def test_rr_constraints_from_exec_params_includes_stop_and_structural() -> None:
    execution_cfg = {
        "stop_loss": {
            "initial_r": 4.0,
            "structural_exit": "ema1200",
            "trailing": {"enabled": True, "activation_r": 3.5, "trail_r": 6.0},
            "guardrails": {"min_stop_pct": 0.01, "max_stop_pct": 0.2},
            "breakeven": {"enabled": True, "trigger_r": 10.0, "lock_level_r": 2},
        },
        "take_profit": {"enabled": False},
        "execution_constraints": {"allow_add_on": True},
    }
    exec_params = ExecutionParamGenerator(execution_cfg).generate_params(0.5)
    rr = rr_constraints_from_exec_params(exec_params)
    assert rr["stop_loss_r"] == 4.0
    assert rr["structural_exit"] == "ema1200"
    assert rr["allow_trailing"] is True
    assert rr["activation_r"] == 3.5


def test_rr_constraints_from_exec_params_includes_scale_out_bank() -> None:
    execution_cfg = {
        "stop_loss": {"initial_r": 4.0, "structural_exit": "ema1200"},
        "take_profit": {"enabled": False},
        "scale_out_bank": {"enabled": True, "mode": "pct", "min_profit_pct": 0.2},
    }
    exec_params = ExecutionParamGenerator(execution_cfg).generate_params(0.5)
    rr = rr_constraints_from_exec_params(exec_params)
    assert rr["scale_out_bank"]["enabled"] is True
    assert rr["scale_out_bank"]["mode"] == "pct"


def test_rr_constraints_copy_cycle_close() -> None:
    execution_cfg = {
        "stop_loss": {
            "initial_r": 0.0,
            "structural_exit": "spot_simple_profit_ladder",
            "cycle_close": {"enabled": True, "min_weeks_above": 4},
        },
        "take_profit": {"enabled": False},
    }
    exec_params = ExecutionParamGenerator(execution_cfg).generate_params(0.5)
    rr = rr_constraints_from_exec_params(exec_params)
    assert rr["cycle_close"]["enabled"] is True
    assert rr["cycle_close"]["min_weeks_above"] == 4
