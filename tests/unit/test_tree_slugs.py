"""Smoke test: fast_scalp / short_term_swing slug skeletons are loadable + label fn works."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.time_series_model.strategies.labels.forward_rr_signed_label import (
    compute_raw_signed_forward_rr,
    compute_signed_forward_rr_label,
    forward_rr_column_name,
)


_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("slug", ["fast_scalp", "short_term_swing"])
@pytest.mark.parametrize(
    "fname",
    ["meta.yaml", "features.yaml", "labels.yaml", "model.yaml", "backtest.yaml"],
)
def test_tree_slug_yaml_loads(slug: str, fname: str) -> None:
    path = _ROOT / "config" / "strategies" / slug / fname
    assert path.exists(), f"missing {path}"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict)


@pytest.mark.parametrize(
    "slug, expected_horizon",
    [("fast_scalp", 3), ("short_term_swing", 20)],
)
def test_tree_slug_label_points_at_signed_forward_rr(
    slug: str, expected_horizon: int
) -> None:
    labels = yaml.safe_load(
        (_ROOT / "config" / "strategies" / slug / "labels.yaml").read_text(
            encoding="utf-8"
        )
    )
    gen = labels["label_generator"]
    assert gen["module"] == (
        "src.time_series_model.strategies.labels.forward_rr_signed_label"
    )
    assert gen["function"] == "compute_signed_forward_rr_label"
    assert gen["params"]["horizon"] == expected_horizon


@pytest.mark.parametrize("slug", ["fast_scalp", "short_term_swing"])
def test_tree_slug_has_requested_features(slug: str) -> None:
    from src.time_series_model.strategy_config.loader import StrategyConfigLoader

    cfg_dir = _ROOT / "config" / "strategies" / slug
    loader = StrategyConfigLoader(cfg_dir)
    features = loader.load().features
    assert features.requested_features
    assert "feature_groups" not in (
        yaml.safe_load((cfg_dir / "features.yaml").read_text(encoding="utf-8")) or {}
    )


@pytest.mark.parametrize(
    "slug, expected_max_hold",
    [("fast_scalp", 6), ("short_term_swing", 40)],
)
def test_tree_slug_backtest_max_holding_aligned_with_horizon(
    slug: str, expected_max_hold: int
) -> None:
    bt = yaml.safe_load(
        (_ROOT / "config" / "strategies" / slug / "backtest.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert bt["backtest"]["params"]["max_holding_bars"] == expected_max_hold
    assert bt["backtest"]["params"]["use_signal_direction"] is True


def test_tree_slug_model_uses_regression() -> None:
    for slug in ("fast_scalp", "short_term_swing"):
        model = yaml.safe_load(
            (_ROOT / "config" / "strategies" / slug / "model.yaml").read_text(
                encoding="utf-8"
            )
        )
        params = model["trainer"]["params"]
        assert params["task_type"] == "regression"
        assert params["target_col"] == "label"


def test_forward_rr_h_column_helpers() -> None:
    assert forward_rr_column_name(3) == "forward_rr_h3"
    assert forward_rr_column_name(20) == "forward_rr_h20"
    n = 50
    rng = np.random.default_rng(42)
    close = 100 + np.cumsum(rng.normal(0, 1, size=n))
    df = pd.DataFrame({"close": close, "atr14": np.full(n, 1.0)})
    raw = compute_raw_signed_forward_rr(df, horizon=3)
    floored = compute_signed_forward_rr_label(df, horizon=3, rr_floor=0.5)
    assert raw.name == "forward_rr_h3"
    assert raw.notna().sum() > floored.notna().sum()


def test_signed_forward_rr_label_basic() -> None:
    n = 50
    rng = np.random.default_rng(42)
    close = 100 + np.cumsum(rng.normal(0, 1, size=n))
    df = pd.DataFrame(
        {
            "close": close,
            "atr14": np.full(n, 1.0),
        }
    )
    s = compute_signed_forward_rr_label(df, horizon=3, rr_floor=0.0)
    assert len(s) == n
    assert s.iloc[: n - 3].notna().any()
    assert s.iloc[-3:].isna().all()

    s_floor = compute_signed_forward_rr_label(df, horizon=3, rr_floor=0.5)
    nonzero = s_floor.dropna()
    assert (nonzero.abs() >= 0.5).all()


def test_signed_forward_rr_label_validates_horizon() -> None:
    df = pd.DataFrame({"close": [1.0, 2.0, 3.0], "atr14": [1.0, 1.0, 1.0]})
    with pytest.raises(ValueError):
        compute_signed_forward_rr_label(df, horizon=0)
    with pytest.raises(KeyError):
        compute_signed_forward_rr_label(
            df.drop(columns=["atr14"]), horizon=1, atr_col="atr14"
        )
