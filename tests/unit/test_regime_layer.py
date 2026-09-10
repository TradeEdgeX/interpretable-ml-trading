from pathlib import Path

import yaml

from src.config.multileg_config import load_multileg_effective_config
from src.config.regime_layer import (
    load_regime_layer,
    multileg_regime_section,
    parse_regime_layer,
    regime_layer_effective_fragment,
)
from src.time_series_model.archetype.loader import RegimeConfig


def test_multileg_regime_section_reads_extensions() -> None:
    raw = {
        "allowed_regimes": ["bull"],
        "rules": [{"feature": "bpc_semantic_chop", "operator": ">=", "value": 0.52}],
        "extensions": {"multileg": {"entry_chop_min": 0.52, "exit_chop_below": 0.33}},
    }
    assert multileg_regime_section(raw)["entry_chop_min"] == 0.52


def test_multileg_regime_section_legacy_nested() -> None:
    raw = {"regime": {"entry_chop_min": 0.4}}
    assert multileg_regime_section(raw)["entry_chop_min"] == 0.4


def test_regime_layer_effective_fragment_synthesises_rules() -> None:
    """extensions.multileg.entry_min → auto-synthesised rule; no manual rules needed."""
    raw = {
        "allowed_sides": ["long"],
        "extensions": {
            "multileg": {
                "entry_feature": "bpc_semantic_chop",
                "entry_min": 0.52,
                "exit_below": 0.33,
            }
        },
    }
    frag = regime_layer_effective_fragment(raw)
    assert frag["regime"]["entry_min"] == 0.52
    assert frag["regime_rules"][0]["feature"] == "bpc_semantic_chop"
    assert frag["regime_rules"][0]["value"] == 0.52
    assert frag["allowed_sides"] == ["long"]


def test_regime_layer_synthesises_rule_for_trend_scalp() -> None:
    raw = {
        "extensions": {
            "multileg": {
                "entry_feature": "trend_confidence",
                "entry_min": 0.7,
                "exit_below": 0.4,
            }
        }
    }
    config, multileg = parse_regime_layer(raw)
    assert config.rules[0]["feature"] == "trend_confidence"
    assert config.rules[0]["value"] == 0.7
    assert multileg["exit_below"] == 0.4


def test_regime_layer_legacy_entry_chop_min_fallback() -> None:
    """Old YAML / sweep-script dicts with entry_chop_min still parse correctly."""
    raw = {
        "extensions": {
            "multileg": {
                "entry_feature": "bpc_semantic_chop",
                "entry_chop_min": 0.45,
                "exit_chop_below": 0.28,
            }
        }
    }
    config, multileg = parse_regime_layer(raw)
    assert config.rules[0]["value"] == 0.45
    assert multileg["exit_chop_below"] == 0.28


def test_explicit_rules_not_overridden_by_synthesis() -> None:
    """TPC strategies write rules explicitly; they must not be replaced."""
    raw = {
        "rules": [{"feature": "ema_1200_position", "operator": ">=", "value": 0.1}],
    }
    config, _ = parse_regime_layer(raw)
    assert config.rules[0]["feature"] == "ema_1200_position"


def test_load_regime_layer_from_chop_grid_archetype(tmp_path: Path) -> None:
    cfg_dir = tmp_path / "archetypes"
    cfg_dir.mkdir(parents=True)
    (cfg_dir / "regime.yaml").write_text(
        yaml.safe_dump(
            {
                "extensions": {
                    "multileg": {
                        "entry_feature": "bpc_semantic_chop",
                        "entry_min": 0.52,
                        "exit_below": 0.33,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    config, multileg = load_regime_layer(cfg_dir / "regime.yaml")
    assert isinstance(config, RegimeConfig)
    assert config.rules[0]["value"] == 0.52
    assert multileg["exit_below"] == 0.33


def test_effective_config_merges_multileg_regime(tmp_path: Path) -> None:
    cfg_dir = tmp_path / "chop_grid"
    (cfg_dir / "archetypes").mkdir(parents=True)
    (cfg_dir / "archetypes/regime.yaml").write_text(
        yaml.safe_dump(
            {
                "extensions": {
                    "multileg": {
                        "entry_feature": "bpc_semantic_chop",
                        "entry_min": 0.52,
                        "exit_below": 0.33,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    (cfg_dir / "archetypes/prefilter.yaml").write_text(
        yaml.safe_dump({"rules": []}), encoding="utf-8"
    )
    (cfg_dir / "archetypes/execution.yaml").write_text(
        yaml.safe_dump({"inventory": {}}), encoding="utf-8"
    )
    got = load_multileg_effective_config(config_dir=cfg_dir, strategy_type="grid")
    assert got["regime"]["entry_min"] == 0.52
    # rules auto-synthesised — not present as top-level key in effective config
    assert got["regime_rules"][0]["value"] == 0.52
    assert "extensions" not in got
