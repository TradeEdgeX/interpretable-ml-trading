"""grid_semantic_chop feature: registration, math, and entry_feature wiring."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.chop_grid_config import (
    grid_config_from_strategy_yaml,
    merge_chop_grid_yaml,
)
from src.features.grid_semantic_chop import (
    GRID_SEMANTIC_CHOP_COLUMN,
    grid_semantic_chop_series,
)


def _close(n: int = 400, seed: int = 7) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="2h", tz="UTC")
    return pd.Series(np.cumsum(rng.standard_normal(n)) + 100.0, index=idx)


def test_registered_in_feature_registry() -> None:
    from src.features.registry import ensure_features_registered, get_compute_func

    ensure_features_registered()
    func = get_compute_func("compute_grid_semantic_chop_from_series")
    out = func(close=_close())
    assert list(out.columns) == [GRID_SEMANTIC_CHOP_COLUMN]


def test_output_bounded_and_finite() -> None:
    s = grid_semantic_chop_series(_close())
    assert s.name == GRID_SEMANTIC_CHOP_COLUMN
    assert s.min() >= 0.0 and s.max() <= 1.0
    assert np.isfinite(s.to_numpy()).all()


def test_drops_when_trend_emerges() -> None:
    """A clean monotonic trend must score lower chop than a flat noisy series."""
    idx = pd.date_range("2024-01-01", periods=300, freq="2h", tz="UTC")
    trend = pd.Series(np.linspace(100.0, 200.0, 300), index=idx)
    flat = pd.Series(100.0 + np.tile([0.0, 0.05, -0.05, 0.02], 75), index=idx)
    assert (
        grid_semantic_chop_series(trend).tail(50).mean()
        < grid_semantic_chop_series(flat).tail(50).mean()
    )


def test_matches_legacy_heuristic_definition() -> None:
    """Must equal the legacy diagnose_crf_edge._semantic_chop math exactly."""
    from scripts.diagnose_crf_edge import _bb_width_pctile, _semantic_chop

    close = _close()
    legacy = _semantic_chop(close, _bb_width_pctile(close))
    got = grid_semantic_chop_series(close)
    pd.testing.assert_series_equal(
        got.rename("x"), legacy.rename("x"), check_exact=False, atol=1e-12
    )


_DDROOT = Path("config/experiments/20260624_chop_dd_restore/variants")


def test_entry_feature_grid_uses_heuristic_column() -> None:
    """grid_semantic_chop variant reads grid_semantic_chop from FeatureStore when configured."""
    yd = merge_chop_grid_yaml(_DDROOT / "heuristic_052_033/meta.yaml")
    assert yd["entry_feature"] == "grid_semantic_chop"
    cfg = grid_config_from_strategy_yaml(_DDROOT / "heuristic_052_033/meta.yaml")
    assert cfg.entry_feature == "grid_semantic_chop"
    assert cfg.feature_store_dir is None  # variant omits FS; legacy offline path only
    assert cfg.chop_min == pytest.approx(0.52)
    assert cfg.exit_chop_min == pytest.approx(0.33)


def test_chop_grid_features_yaml_includes_grid_semantic_chop_f() -> None:
    from scripts.chop_grid_config import _chop_grid_store_requested_features

    feats = _chop_grid_store_requested_features()
    assert "grid_semantic_chop_f" in feats
    assert "bpc_soft_phase_f" in feats


def test_entry_feature_defaults_to_bpc() -> None:
    cfg = grid_config_from_strategy_yaml(_DDROOT / "bpc_088_080/meta.yaml")
    assert cfg.entry_feature == "bpc_semantic_chop"
