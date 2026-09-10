"""Backtest-only segment → MLBOT_TREND_POOL_ACTIVE_REGIME wiring."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.event_backtest.market_segment import (
    apply_segment_pool_regime,
    load_market_segments,
    resolve_segment_run,
    segment_pool_regime_context,
    trend_pool_active_regime_from_segment,
)
from scripts.event_backtest.variant_grid import _build_event_backtest_cmd

_REPO = Path(__file__).resolve().parents[2]
_ENV = "MLBOT_TREND_POOL_ACTIVE_REGIME"


@pytest.fixture(autouse=True)
def _clear_pool_regime_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(_ENV, raising=False)


def test_canonical_segments_have_trend_pool_active_regime() -> None:
    segs = load_market_segments()
    assert trend_pool_active_regime_from_segment(segs["bear_2022"]) == "bear"
    assert trend_pool_active_regime_from_segment(segs["bull_2023_2024"]) == "bull"
    assert (
        trend_pool_active_regime_from_segment(segs["recent_range_to_bear"]) == "range"
    )
    assert trend_pool_active_regime_from_segment(segs["recent_6m_oos"]) == "range"


def test_resolve_segment_run_includes_regime() -> None:
    run = resolve_segment_run({"variant": "x", "segment": "bull_2023_2024"})
    assert run["start_date"] == "2023-06-01"
    assert run["end_date"] == "2025-01-01"
    assert run["trend_pool_active_regime"] == "bull"


def test_apply_segment_pool_regime_sets_env() -> None:
    regime = apply_segment_pool_regime("bear_2022")
    assert regime == "bear"
    assert os.environ[_ENV] == "bear"


def test_segment_pool_regime_context_restores_env() -> None:
    os.environ[_ENV] = "bull"
    with segment_pool_regime_context("bear_2022") as regime:
        assert regime == "bear"
        assert os.environ[_ENV] == "bear"
    assert os.environ[_ENV] == "bull"


def test_variant_grid_passes_seg_id() -> None:
    run = resolve_segment_run(
        {
            "variant": "v",
            "segment": "recent_range_to_bear",
            "start_date": "2025-01-01",
            "end_date": "2026-05-31",
        }
    )
    cmd = _build_event_backtest_cmd(
        run=run,
        grid={},
        out_path=Path("/tmp/out"),
        extra_argv=[],
    )
    assert "--seg-id" in cmd
    idx = cmd.index("--seg-id")
    assert cmd[idx + 1] == "recent_range_to_bear"


def test_cli_seg_id_applies_regime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(_ENV, raising=False)
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.event_backtest",
            "--seg-id",
            "bull_2023_2024",
            "--help",
        ],
        cwd=str(_REPO),
        env={**os.environ, "PYTHONPATH": "src"},
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "--seg-id" in proc.stdout
