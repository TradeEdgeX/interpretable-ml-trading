import pytest

from src.time_series_model.nnmultihead.strategy_profile import resolve_execution_profile


@pytest.mark.unit
def test_strategy_meta_execution_present_for_core_strategies():
    ex = resolve_execution_profile(strategy_id="mean_failed_trend_liquidation")
    assert ex is not None
    assert ex.router_mode in ("MEAN", "TREND", "NO_TRADE")
    assert isinstance(ex.execution_strategy_id, str) and ex.execution_strategy_id
    assert isinstance(ex.evidence_rules or [], list)
