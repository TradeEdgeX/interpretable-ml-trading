"""Unit tests for RegimeConfig + StrategyArchetype regime integration."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from src.time_series_model.archetype.loader import (
    PrefilterConfig,
    RegimeConfig,
    load_strategy_archetype,
)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(text).lstrip("\n"), encoding="utf-8")


# ---------------------------------------------------------------------------
# RegimeConfig.from_yaml roundtrip
# ---------------------------------------------------------------------------


def test_regime_config_defaults_when_file_missing(tmp_path: Path):
    cfg = RegimeConfig.from_yaml(tmp_path / "nonexistent.yaml")
    assert cfg.rules == []
    assert cfg.allowed_regimes == ["bull", "bear", "neutral"]
    assert cfg.allowed_sides == ["long", "short"]
    assert cfg.is_empty is True


def test_regime_allows_side_for_bar_side_mask_disabled_matches_allows_side(
    tmp_path: Path,
):
    cfg = RegimeConfig.from_mapping(
        {
            "allowed_sides": ["long", "short"],
            "side_mask": {"enabled": False, "long_when": {"all_of": []}},
        }
    )
    feats = {"macro_tp_vwap_1200_position": 0.05}
    assert cfg.allows_side_for_bar(1, feats) is True
    assert cfg.allows_side_for_bar(-1, feats) is True


def test_regime_config_yaml_roundtrip_side_mask(tmp_path: Path):
    regime_yaml = tmp_path / "regime.yaml"
    _write(
        regime_yaml,
        """
        allowed_sides: [long, short]
        side_mask:
          enabled: true
          long_when:
            all_of:
              - ema_1200_slope_10:
                  value_gt: 0.0
        """,
    )
    cfg = RegimeConfig.from_yaml(regime_yaml)
    assert cfg.side_mask.get("enabled") is True
    assert "long_when" in cfg.side_mask
    assert cfg.allows_side_for_bar(1, {"ema_1200_slope_10": 0.01}) is True
    assert cfg.allows_side_for_bar(1, {"ema_1200_slope_10": -0.01}) is False


def test_regime_allows_side_for_bar_side_mask(tmp_path: Path):
    cfg = RegimeConfig.from_mapping(
        {
            "allowed_sides": ["long", "short"],
            "side_mask": {
                "enabled": True,
                "long_when": {
                    "all_of": [
                        {"macro_tp_vwap_1200_position": {"value_gte": 0.10}},
                        {"ema_1200_slope_10": {"value_gt": 0.0}},
                    ]
                },
                "short_when": {
                    "all_of": [
                        {"macro_tp_vwap_1200_position": {"value_lte": -0.10}},
                        {"ema_1200_slope_10": {"value_lt": 0.0}},
                    ]
                },
            },
        }
    )
    feats_bull = {
        "macro_tp_vwap_1200_position": 0.15,
        "ema_1200_slope_10": 0.002,
    }
    feats_bear = {
        "macro_tp_vwap_1200_position": -0.15,
        "ema_1200_slope_10": -0.002,
    }
    feats_mid = {"macro_tp_vwap_1200_position": 0.05, "ema_1200_slope_10": 0.01}
    assert cfg.allows_side_for_bar(1, feats_bull) is True
    assert cfg.allows_side_for_bar(-1, feats_bear) is True
    assert cfg.allows_side_for_bar(1, feats_bear) is False
    assert cfg.allows_side_for_bar(-1, feats_bull) is False
    assert cfg.allows_side_for_bar(1, feats_mid) is False


def test_regime_config_loads_rules_and_masks(tmp_path: Path):
    regime_yaml = tmp_path / "regime.yaml"
    _write(
        regime_yaml,
        """
        allowed_regimes: [bear, neutral]
        allowed_sides: [short]
        rules:
          - feature: tpc_semantic_chop
            operator: "<="
            value: 0.4
        """,
    )
    cfg = RegimeConfig.from_yaml(regime_yaml)
    assert cfg.allowed_regimes == ["bear", "neutral"]
    assert cfg.allowed_sides == ["short"]
    assert len(cfg.rules) == 1
    assert cfg.rules[0]["feature"] == "tpc_semantic_chop"
    assert cfg.is_empty is False


# ---------------------------------------------------------------------------
# RegimeConfig.evaluate semantics
# ---------------------------------------------------------------------------


def test_regime_evaluate_passes_when_no_rules():
    cfg = RegimeConfig()
    passed, reason = cfg.evaluate({"x": 0.5})
    assert passed is True
    assert reason is None


def test_regime_evaluate_reject_prefixed_with_regime():
    cfg = RegimeConfig(
        rules=[{"feature": "tpc_semantic_chop", "operator": "<=", "value": 0.4}]
    )
    passed, reason = cfg.evaluate({"tpc_semantic_chop": 0.6})
    assert passed is False
    assert reason is not None
    assert reason.startswith("regime_")
    # 必须不漏出 prefilter 前缀
    assert "prefilter_" not in reason


def test_regime_evaluate_passes_when_threshold_satisfied():
    cfg = RegimeConfig(
        rules=[{"feature": "tpc_semantic_chop", "operator": "<=", "value": 0.4}]
    )
    passed, reason = cfg.evaluate({"tpc_semantic_chop": 0.2})
    assert passed is True
    assert reason is None


def test_regime_any_of_rule_evaluates_like_prefilter():
    cfg = RegimeConfig(
        rules=[
            {
                "any_of": [
                    {"feature": "box_pos_120", "operator": "<=", "value": 0.15},
                    {"feature": "box_pos_120", "operator": ">=", "value": 0.85},
                ]
            }
        ]
    )
    # box mid → reject
    passed, reason = cfg.evaluate({"box_pos_120": 0.5})
    assert passed is False
    assert reason is not None and reason.startswith("regime_")

    # box edge → pass
    passed, _ = cfg.evaluate({"box_pos_120": 0.1})
    assert passed is True


# ---------------------------------------------------------------------------
# allowed_sides masking
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "allowed,direction,expected",
    [
        (["long", "short"], 1, True),
        (["long", "short"], -1, True),
        (["long", "short"], 0, True),
        (["long"], 1, True),
        (["long"], -1, False),
        (["short"], 1, False),
        (["short"], -1, True),
        ([], 1, False),
        ([], -1, False),
    ],
)
def test_allowed_sides_masking(allowed, direction, expected):
    cfg = RegimeConfig(allowed_sides=list(allowed))
    assert cfg.allows_side(direction) is expected


# ---------------------------------------------------------------------------
# RegimeConfig is a PrefilterConfig subclass — substitutable
# ---------------------------------------------------------------------------


def test_regime_config_is_prefilter_subclass():
    cfg = RegimeConfig(rules=[{"feature": "x", "operator": ">=", "value": 0.0}])
    assert isinstance(cfg, PrefilterConfig)


def test_prefilter_latch_stays_open_after_first_pass():
    rule = {
        "feature": "weekly_ema_200_position",
        "operator": "<",
        "value": 0.0,
        "latch": True,
    }
    cfg = PrefilterConfig(rules=[rule])
    assert (
        cfg.evaluate({"weekly_ema_200_position": 0.10}, latch_id="BTCUSDT")[0] is False
    )
    assert (
        cfg.evaluate({"weekly_ema_200_position": -0.01}, latch_id="BTCUSDT")[0] is True
    )
    assert (
        cfg.evaluate({"weekly_ema_200_position": 0.20}, latch_id="BTCUSDT")[0] is True
    )
    assert (
        cfg.evaluate({"weekly_ema_200_position": 0.20}, latch_id="BNBUSDT")[0] is False
    )

    plain = PrefilterConfig(
        rules=[
            {
                "feature": "weekly_ema_200_position",
                "operator": "<",
                "value": 0.0,
            }
        ]
    )
    assert (
        plain.evaluate({"weekly_ema_200_position": -0.01}, latch_id="BTCUSDT")[0]
        is True
    )
    assert (
        plain.evaluate({"weekly_ema_200_position": 0.20}, latch_id="BTCUSDT")[0]
        is False
    )


def test_prefilter_skip_symbols_bypasses_missing_feature():
    rule = {
        "feature": "weekly_ema_200_position",
        "operator": "<",
        "value": 0.0,
        "skip_symbols": ["PUMPUSDT"],
    }
    cfg = PrefilterConfig(rules=[rule])
    assert cfg.evaluate({}, latch_id="PUMPUSDT")[0] is True
    assert (
        cfg.evaluate({"weekly_ema_200_position": 0.20}, latch_id="BTCUSDT")[0] is False
    )


def test_prefilter_latch_persists_to_json(tmp_path):
    rule = {
        "feature": "weekly_ema_200_position",
        "operator": "<",
        "value": 0.0,
        "latch": True,
    }
    cfg = PrefilterConfig(rules=[rule])
    cfg.evaluate({"weekly_ema_200_position": -0.01}, latch_id="BTCUSDT")
    path = tmp_path / "latch.json"
    cfg.persist_latched(path)
    other = PrefilterConfig(rules=[rule])
    assert other.restore_latched(path) == 1
    assert (
        other.evaluate({"weekly_ema_200_position": 0.20}, latch_id="BTCUSDT")[0] is True
    )


# ---------------------------------------------------------------------------
# load_strategy_archetype wires regime.yaml when present
# ---------------------------------------------------------------------------


def _minimal_archetype_dir(root: Path, *, with_regime: bool) -> Path:
    pkg = root / "demo_strat"
    arch = pkg / "archetypes"
    arch.mkdir(parents=True)
    _write(arch / "gate.yaml", "hard_gates: []\n")
    _write(arch / "evidence.yaml", "evidence: []\n")
    _write(arch / "execution.yaml", "execution_constraints: {}\n")
    _write(
        arch / "prefilter.yaml",
        """
        rules:
          - feature: pf_feat
            operator: ">="
            value: 0.0
        """,
    )
    if with_regime:
        _write(
            arch / "regime.yaml",
            """
            allowed_sides: [long]
            rules:
              - feature: tpc_semantic_chop
                operator: "<="
                value: 0.4
            """,
        )
    return pkg


def test_load_strategy_archetype_without_regime_file(tmp_path: Path):
    _minimal_archetype_dir(tmp_path, with_regime=False)
    arch = load_strategy_archetype("demo_strat", strategies_root=str(tmp_path))
    assert isinstance(arch.regime, RegimeConfig)
    assert arch.regime.is_empty is True
    assert arch.regime.allows_side(1) is True
    assert arch.regime.allows_side(-1) is True


def test_load_strategy_archetype_with_regime_file(tmp_path: Path):
    _minimal_archetype_dir(tmp_path, with_regime=True)
    arch = load_strategy_archetype("demo_strat", strategies_root=str(tmp_path))
    assert arch.regime.allowed_sides == ["long"]
    assert arch.regime.allows_side(1) is True
    assert arch.regime.allows_side(-1) is False
    # Regime evaluator independent from prefilter
    passed, reason = arch.regime.evaluate({"tpc_semantic_chop": 0.6})
    assert passed is False
    assert reason is not None and reason.startswith("regime_")


def test_regime_reject_neutral_flag():
    allowed = {
        "bull": {
            "rules": [{"feature": "ema_1200_position", "operator": ">=", "value": 0.08}]
        },
        "bear": {
            "rules": [
                {"feature": "ema_1200_position", "operator": "<=", "value": -0.08}
            ]
        },
    }
    open_n = RegimeConfig.from_mapping(
        {"allowed_regimes": allowed, "reject_neutral": False}
    )
    deny_n = RegimeConfig.from_mapping(
        {"allowed_regimes": allowed, "reject_neutral": True}
    )
    assert open_n.allows_classified_label("neutral") is True
    assert deny_n.allows_classified_label("neutral") is False
    assert deny_n.allows_classified_label("bull") is True
    assert deny_n.reject_neutral is True


def test_prod_fade_regime_box_edge_contract():
    """largebar_fade: box_pos_60 edge · reject_neutral true (20260726 promote)."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    path = root / "config/strategies/largebar_fade/archetypes/regime.yaml"
    rg = RegimeConfig.from_yaml(path)
    assert rg.reject_neutral is True
    assert "box_edge" in rg.allowed_regimes
    assert rg.classify({"box_pos_60": 0.10}) == "box_edge"
    assert rg.classify({"box_pos_60": 0.90}) == "box_edge"
    assert rg.classify({"box_pos_60": 0.50}) == "neutral"
    assert rg.allows_classified_label("box_edge") is True
    assert rg.allows_classified_label("neutral") is False
