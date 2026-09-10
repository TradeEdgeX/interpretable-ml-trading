"""Regime.yaml drives box_structure_f in live feature plan (Feature Bus readiness)."""

from __future__ import annotations

from pathlib import Path

from time_series_model.live.live_feature_plan import extract_features_from_archetypes


def test_regime_yaml_pulls_box_structure_node(tmp_path: Path) -> None:
    arch = tmp_path / "archetypes"
    arch.mkdir()
    (arch / "regime.yaml").write_text(
        """
rules:
  - feature: tpc_semantic_chop
    operator: "<="
    value: 0.40
  - any_of:
      - feature: box_pos_120
        operator: "<="
        value: 0.15
      - feature: box_breakout_up
        operator: ">="
        value: 0.5
""",
        encoding="utf-8",
    )
    cols, nodes = extract_features_from_archetypes(
        arch,
        feature_deps_path=Path("config/feature_dependencies.yaml"),
    )
    assert "box_pos_120" in cols
    assert "tpc_semantic_chop" in cols
    assert "box_structure_f" in nodes


def test_live_tpc_has_regime_yaml() -> None:
    live_regime = Path("config/strategies/tpc/archetypes/regime.yaml")
    assert live_regime.is_file(), "deploy regime.yaml to live/highcap first"


def test_multileg_trend_scalp_regime_pulls_trend_confidence_node() -> None:
    arch = Path("config/strategies/trend_scalp/archetypes")
    _cols, nodes = extract_features_from_archetypes(
        arch,
        feature_deps_path=Path("config/feature_dependencies.yaml"),
    )
    assert "trend_confidence_f" in nodes


def test_regime_side_mask_pulls_ema_slope_node(tmp_path: Path) -> None:
    arch = tmp_path / "archetypes"
    arch.mkdir()
    (arch / "regime.yaml").write_text(
        """
rules:
  - feature: macro_tp_vwap_1200_position
    operator: ">="
    value: 0.10
side_mask:
  enabled: true
  long_when:
    all_of:
      - macro_tp_vwap_1200_position:
          value_gte: 0.10
      - ema_1200_slope_10:
          value_gt: 0.0
""",
        encoding="utf-8",
    )
    cols, nodes = extract_features_from_archetypes(
        arch,
        feature_deps_path=Path("config/feature_dependencies.yaml"),
    )
    assert "ema_1200_slope_10" in cols
    assert "ema_1200_slope_f" in nodes


def test_multileg_chop_grid_regime_pulls_soft_phase_node() -> None:
    # chop_grid regime.yaml uses entry_feature: grid_semantic_chop (NOT bpc_*),
    # so the live plan must pull grid_semantic_chop_f via the archetype path.
    # The feature is selected from regime rules directly, not from the research
    # features.yaml requested_features pool (which we intentionally do not read
    # into the live plan — see live_feature_plan._feature_nodes_from_strategy_features_yaml).
    arch = Path("config/strategies/chop_grid/archetypes")
    cols, nodes = extract_features_from_archetypes(
        arch,
        feature_deps_path=Path("config/feature_dependencies.yaml"),
    )
    assert "grid_semantic_chop" in cols
    assert "grid_semantic_chop_f" in nodes
