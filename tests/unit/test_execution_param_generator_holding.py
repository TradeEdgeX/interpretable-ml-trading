"""ExecutionParamGenerator.holding：与向量回测一致的 time_stop / max_holding 别名。"""

from __future__ import annotations

from src.time_series_model.live.generic_live_strategy import ExecutionParamGenerator


def test_holding_prefers_time_stop_bars() -> None:
    gen = ExecutionParamGenerator(
        {
            "stop_loss": {"initial_r": 2.0, "trailing": {"enabled": False}},
            "take_profit": {"enabled": False},
            "holding": {"time_stop_bars": 40, "max_holding_bars": 99},
        }
    )
    p = gen.generate_params(0.5)
    assert p["time_stop_bars"] == 40
    assert p["max_holding_bars"] == 40


def test_holding_falls_back_to_max_holding_bars_when_time_stop_missing() -> None:
    gen = ExecutionParamGenerator(
        {
            "stop_loss": {"initial_r": 2.0, "trailing": {"enabled": False}},
            "take_profit": {"enabled": False},
            "holding": {"max_holding_bars": 33},
        }
    )
    p = gen.generate_params(0.5)
    assert p["time_stop_bars"] == 33
    assert p["max_holding_bars"] == 33


def test_holding_time_stop_zero_still_uses_positive_max_holding() -> None:
    """time_stop_bars: 0 且 max_holding_bars>0 时应用 max（BPC/TPC 旧 yaml 语义）。"""
    gen = ExecutionParamGenerator(
        {
            "stop_loss": {"initial_r": 2.0, "trailing": {"enabled": False}},
            "take_profit": {"enabled": False},
            "holding": {"time_stop_bars": 0, "max_holding_bars": 50},
        }
    )
    p = gen.generate_params(0.5)
    assert p["time_stop_bars"] == 50
    assert p["max_holding_bars"] == 50


def test_holding_max_zero_disables_time_stop() -> None:
    gen = ExecutionParamGenerator(
        {
            "stop_loss": {"initial_r": 2.0, "trailing": {"enabled": False}},
            "take_profit": {"enabled": False},
            "holding": {"max_holding_bars": 0},
        }
    )
    p = gen.generate_params(0.5)
    assert p["time_stop_bars"] == 0
    assert p["max_holding_bars"] == 0
