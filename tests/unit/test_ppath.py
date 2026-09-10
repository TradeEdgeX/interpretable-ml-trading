import pytest

from src.time_series_model.portfolio.ppath import (
    PPathConfig,
    PositionState,
    assert_ppath_usage,
    compute_ppath,
    compute_ppath_and_remaining,
)


def test_ppath_clamps_and_returns_remaining() -> None:
    cfg = PPathConfig(target_mfe_atr=1.0, target_pnl_r=1.0, target_duration_bars=10)
    st = PositionState(
        realized_mfe_atr=10.0,
        realized_pnl_r=5.0,
        floating_pnl_r=1.0,
        bars_held=100,
        structure_flags={"breakout_confirmed": True},
    )
    p, rem = compute_ppath_and_remaining(st, cfg=cfg)
    assert 0.0 <= p <= 1.0
    assert 0.0 <= rem <= 1.0
    assert rem == pytest.approx(1.0 - p, abs=1e-9)


def test_ppath_structure_increases_score() -> None:
    cfg = PPathConfig(target_mfe_atr=2.0, target_pnl_r=2.0, target_duration_bars=20)
    st0 = PositionState(realized_mfe_atr=1.0, realized_pnl_r=1.0, bars_held=10)
    st1 = PositionState(
        realized_mfe_atr=1.0,
        realized_pnl_r=1.0,
        bars_held=10,
        structure_flags={
            "breakout_confirmed": True,
            "pullback_confirmed": True,
            "absorption_confirmed": True,
            "retest_confirmed": True,
        },
    )
    p0 = compute_ppath(st0, cfg=cfg)
    p1 = compute_ppath(st1, cfg=cfg)
    assert p1 >= p0


def test_ppath_usage_guard_raises_on_entry() -> None:
    with pytest.raises(ValueError):
        assert_ppath_usage("entry")
