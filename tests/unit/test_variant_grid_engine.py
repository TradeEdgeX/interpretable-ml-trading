"""Smoke test: variant_grid dispatcher chooses the right backend per `engine`."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.event_backtest.variant_grid import (
    _VALID_ENGINES,
    _build_chop_grid_cmd,
    _build_event_backtest_cmd,
    _build_multileg_timeline_cmd,
    _resolve_engine,
)


def test_resolve_engine_defaults_to_event_backtest() -> None:
    assert _resolve_engine({}, {}) == "event_backtest"


def test_resolve_engine_uses_run_then_grid() -> None:
    assert _resolve_engine({"engine": "chop_grid"}, {}) == "chop_grid"
    assert _resolve_engine({}, {"engine": "chop_grid"}) == "chop_grid"
    assert (
        _resolve_engine({"engine": "event_backtest"}, {"engine": "chop_grid"})
        == "event_backtest"
    )


def test_resolve_engine_rejects_unknown() -> None:
    with pytest.raises(ValueError):
        _resolve_engine({"engine": "ftw"}, {})


def test_engines_set_locked() -> None:
    assert set(_VALID_ENGINES) == {
        "event_backtest",
        "chop_grid",
        "trend_scalp",
        "multileg_joint",
    }


def test_event_backtest_cmd_uses_strategies_root() -> None:
    cmd = _build_event_backtest_cmd(
        run={
            "variant": "A",
            "strategy": "tpc",
            "strategies_root": "config_experiments/A_strategies",
            "start_date": "2025-04-01",
            "end_date": "2025-05-01",
        },
        grid={"strategy": "tpc"},
        out_path=Path("/tmp/out"),
        extra_argv=["--quiet-signal-logs"],
    )
    assert "scripts.event_backtest" in cmd
    assert "config_experiments/A_strategies" in cmd
    assert cmd[-1] == "--quiet-signal-logs"
    assert "--no-kill-switch" in cmd


def test_chop_grid_cmd_dispatches_to_chop_grid_backtest(tmp_path: Path) -> None:
    cmd = _build_chop_grid_cmd(
        run={
            "variant": "baseline_recent",
            "config": "config/experiments/20260615_chop_grid_emergency_sl/variants/baseline/meta.yaml",
            "start_date": "2025-04-01",
            "end_date": "2026-04-01",
        },
        grid={"engine": "chop_grid"},
        out_path=tmp_path,
        extra_argv=[],
    )
    assert "scripts.chop_grid_backtest" in cmd
    assert "--config" in cmd
    assert "--out-dir" in cmd
    assert "--no-maps" in cmd


def test_chop_grid_cmd_requires_config() -> None:
    with pytest.raises(ValueError):
        _build_chop_grid_cmd(
            run={
                "variant": "X",
                "start_date": "2025-04-01",
                "end_date": "2026-04-01",
            },
            grid={},
            out_path=Path("/tmp/out"),
            extra_argv=[],
        )


def test_multileg_timeline_cmd_uses_live_engine_backtest(tmp_path: Path) -> None:
    cmd = _build_multileg_timeline_cmd(
        seg_id="recent_6m_oos",
        seg={"start_date": "2025-12-01", "end_date": "2026-05-31"},
        grid={
            "chop_config": "variants/chop_prod/meta.yaml",
            "trend_config": "variants/trend_prod/meta.yaml",
            "symbols": ["BTCUSDT", "ETHUSDT"],
            "equity": 10000,
        },
        run={"chop_config": "variants/chop_alt/meta.yaml"},
        out_path=tmp_path,
        extra_argv=[],
        preload_path=tmp_path / "preload.pkl",
        save_preload=True,
    )
    assert any("backtest_multileg_timeline.py" in part for part in cmd)
    assert "variants/chop_alt/meta.yaml" in cmd
    assert "--summary-json" in cmd
    assert "--save-preload" in cmd
    assert cmd[cmd.index("--equity") + 1] == "10000.0"
