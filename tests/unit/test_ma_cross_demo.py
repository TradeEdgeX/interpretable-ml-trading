"""Public ma_cross pack: loads, stays off live, uses event_backtest."""

from __future__ import annotations

from pathlib import Path

import yaml

from src.research.court_run import convention_results_root
from src.research.harness_registry import event_backtest_families, required_harness

_ROOT = Path(__file__).resolve().parents[2]
_PACK = _ROOT / "config/strategies/ma_cross"


def test_ma_cross_meta_is_research_demo() -> None:
    raw = yaml.safe_load((_PACK / "meta.yaml").read_text(encoding="utf-8"))
    st = raw["strategy"]
    assert st["timeframe"] == "120T"
    assert st.get("live_enabled") is False
    assert st["name"].startswith("ma_cross")


def test_ma_cross_direction_is_ema_sign() -> None:
    raw = yaml.safe_load(
        (_PACK / "archetypes" / "direction.yaml").read_text(encoding="utf-8")
    )
    rule = raw["direction_rules"][0]
    assert rule["feature"] == "ema_1200_position"
    assert rule["transform"] == "sign"


def test_ma_cross_prefilter_leaves_deadzone() -> None:
    raw = yaml.safe_load(
        (_PACK / "archetypes" / "prefilter.yaml").read_text(encoding="utf-8")
    )
    any_of = raw["rules"][0]["any_of"]
    feats = {row["feature"] for row in any_of}
    assert feats == {"ema_1200_position"}


def test_ma_cross_constitution_is_research_only() -> None:
    raw = yaml.safe_load(
        (_ROOT / "config/constitution/constitution.yaml").read_text(encoding="utf-8")
    )
    assert raw["resource_allocation"]["enabled_archetypes"] == ["ma_cross"]
    assert raw["kill_switch"]["enabled"] is False
    meta = yaml.safe_load((_PACK / "meta.yaml").read_text(encoding="utf-8"))
    assert meta["strategy"].get("live_enabled") is False


def test_ma_cross_requests_cross_feature() -> None:
    raw = yaml.safe_load((_PACK / "features.yaml").read_text(encoding="utf-8"))
    req = raw["feature_pipeline"]["requested_features"]
    assert "ema_50_200_cross_f" in req
    alt = _PACK / "archetypes" / "direction_ema50_200_cross.yaml"
    assert alt.is_file()


def test_ma_cross_harness_is_event_backtest() -> None:
    assert required_harness("ma_cross") == "event_backtest"
    assert required_harness("ma") == "event_backtest"


def test_example_families_are_not_ma_cross() -> None:
    assert required_harness("funding_fade") == "event_backtest"
    assert required_harness("tenbagger_smallcap") == "cohort_hold"
    families = event_backtest_families()
    assert "ma_cross" in families
    assert "funding_fade" in families
    assert "tenbagger_smallcap" not in families
    assert convention_results_root("funding_fade", "20260910_funding_fade") == (
        "results/funding_fade/experiments/20260910_funding_fade"
    )
