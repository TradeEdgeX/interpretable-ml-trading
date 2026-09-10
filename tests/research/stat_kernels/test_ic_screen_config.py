"""Tests for strategy ic_screen.yaml loading and ic-prune param merge."""

from pathlib import Path

import pytest

from src.research.stat_kernels.ic_screen_config import (
    load_ic_screen,
    resolve_ic_prune_params,
    resolve_strategy_config_dir,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def test_load_fast_scalp_ic_screen() -> None:
    cfg_dir = PROJECT_ROOT / "config/strategies/tree_strategies/fast_scalp"
    screen = load_ic_screen(cfg_dir)
    assert screen.label_horizon_bars == 3
    assert screen.allowed_best_lags == [1, 3, 5]
    assert screen.reject_peak_at == 50
    assert 50 in screen.horizons
    assert "peak ∈ {1,3,5}" in screen.summary_line()


def test_load_short_term_swing_ic_screen() -> None:
    cfg_dir = PROJECT_ROOT / "config/strategies/tree_strategies/short_term_swing"
    screen = load_ic_screen(cfg_dir)
    assert screen.label_horizon_bars == 20
    assert screen.allowed_best_lags == [10, 20]
    assert screen.min_ic == 0.015
    assert screen.writeback.top_n_columns == 20


def test_resolve_ic_prune_params_step_overrides() -> None:
    params = resolve_ic_prune_params(
        strategy="fast_scalp",
        overrides={"min_ic": 0.03, "top_n_columns": 15},
        project_root=PROJECT_ROOT,
    )
    assert params["min_ic"] == 0.03
    assert params["top_n_columns"] == 15
    assert params["allowed_best_lags"] == "1,3,5"
    assert "_ic_screen_summary" in params


def test_resolve_strategy_config_dir_by_slug() -> None:
    path = resolve_strategy_config_dir(strategy="fast_scalp", project_root=PROJECT_ROOT)
    assert path is not None
    assert path.name == "fast_scalp"
    assert (path / "ic_screen.yaml").is_file()


def test_missing_ic_screen_raises() -> None:
    with pytest.raises(FileNotFoundError, match="ic_screen.yaml"):
        load_ic_screen(
            PROJECT_ROOT / "config/strategies/tree_strategies/trend_following"
        )
