"""Meta prefilter: features.yaml forbidden_requested_features strips module columns."""

from pathlib import Path

from scripts.analyze_archetype_feature_stratification import (
    PROJECT_ROOT,
    _resolve_features_from_prefilter_yaml,
)


def test_resolve_prefilter_yaml_me_keeps_oi_scene_when_not_forbidden(
    tmp_path: Path,
) -> None:
    """ME features.yaml 已允许 oi_scene_semantic_scores_f 进主池时，meta 解析不应再剔除 OI 语义列。"""
    fp = tmp_path / "features_prefilter.yaml"
    fp.write_text(
        """
feature_pipeline:
  requested_features:
    - oi_scene_semantic_scores_f
    - me_soft_phase_f
""",
        encoding="utf-8",
    )
    deps = PROJECT_ROOT / "config" / "feature_dependencies.yaml"
    cols = [
        "oi_compression_score",
        "oi_ignition_score",
        "me_atr_pct",
        "me_cvd_alignment",
        "symbol",
        "datetime",
    ]
    out = _resolve_features_from_prefilter_yaml(str(fp), str(deps), cols, strategy="me")
    assert "oi_compression_score" in out
    assert "oi_ignition_score" in out
    assert "me_atr_pct" in out


def test_resolve_prefilter_yaml_me_forbidden_prefilter_meta_strips_vol_regime(
    tmp_path: Path,
) -> None:
    """ME features.yaml lists me_vol_regime under forbidden_prefilter_meta_columns."""
    fp = tmp_path / "features_prefilter.yaml"
    fp.write_text(
        """
feature_pipeline:
  requested_features:
    - me_soft_phase_f
""",
        encoding="utf-8",
    )
    deps = PROJECT_ROOT / "config" / "feature_dependencies.yaml"
    cols = [
        "me_atr_pct",
        "me_vol_regime",
        "me_accel_2k",
        "me_cvd_alignment",
    ]
    out = _resolve_features_from_prefilter_yaml(str(fp), str(deps), cols, strategy="me")
    assert "me_vol_regime" not in out
    assert "me_atr_pct" in out


def test_resolve_prefilter_yaml_no_strategy_keeps_me_vol_regime(tmp_path: Path) -> None:
    fp = tmp_path / "features_prefilter.yaml"
    fp.write_text(
        """
feature_pipeline:
  requested_features:
    - me_soft_phase_f
""",
        encoding="utf-8",
    )
    deps = PROJECT_ROOT / "config" / "feature_dependencies.yaml"
    cols = ["me_atr_pct", "me_vol_regime", "me_accel_2k"]
    out = _resolve_features_from_prefilter_yaml(str(fp), str(deps), cols, strategy=None)
    assert "me_vol_regime" in out


def test_resolve_prefilter_yaml_no_strategy_keeps_oi_columns(tmp_path: Path) -> None:
    fp = tmp_path / "features_prefilter.yaml"
    fp.write_text(
        """
feature_pipeline:
  requested_features:
    - oi_scene_semantic_scores_f
""",
        encoding="utf-8",
    )
    deps = PROJECT_ROOT / "config" / "feature_dependencies.yaml"
    cols = ["oi_compression_score", "oi_ignition_score"]
    out = _resolve_features_from_prefilter_yaml(str(fp), str(deps), cols, strategy=None)
    assert "oi_compression_score" in out
    assert "oi_ignition_score" in out
