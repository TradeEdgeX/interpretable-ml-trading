from pathlib import Path

from src.config.strategy_validation import validate_pipeline_strategy_packages


def _mk(path: Path, text: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_validate_pipeline_multileg_requires_profiles_and_archetypes(
    tmp_path: Path,
) -> None:
    strategy_dir = tmp_path / "strategies/chop_grid"
    _mk(strategy_dir / "features.yaml", "features: []\n")
    _mk(strategy_dir / "research/calibrate_roll.default.yaml", "strategy_type: grid\n")
    cfg = {
        "strategies": {
            "chop_grid": {"strategy_type": "grid", "config": str(strategy_dir)}
        }
    }
    issues = validate_pipeline_strategy_packages(
        pipeline_cfg=cfg,
        project_root=tmp_path,
        allow_strategy_types={"grid", "dual_add_trend"},
    )
    missing = [it.path for it in issues if it.code == "missing_required_file"]
    assert str(strategy_dir / "research/research_roll.features_on.yaml") in missing
    assert str(strategy_dir / "research/validate_static.full_study.yaml") in missing
    assert str(strategy_dir / "archetypes/prefilter.yaml") in missing
    assert str(strategy_dir / "archetypes/execution.yaml") in missing


def test_validate_pipeline_trend_does_not_require_multileg_archetypes(
    tmp_path: Path,
) -> None:
    strategy_dir = tmp_path / "strategies/bpc"
    _mk(strategy_dir / "features.yaml", "features: []\n")
    _mk(strategy_dir / "research/calibrate_roll.default.yaml", "strategy_type: bpc\n")
    _mk(
        strategy_dir / "research/research_roll.features_on.yaml", "strategy_type: bpc\n"
    )
    _mk(
        strategy_dir / "research/validate_static.full_study.yaml",
        "strategy_type: bpc\n",
    )
    cfg = {"strategies": {"bpc": {"strategy_type": "bpc", "config": str(strategy_dir)}}}
    issues = validate_pipeline_strategy_packages(
        pipeline_cfg=cfg,
        project_root=tmp_path,
    )
    assert issues == []


def test_validate_multileg_prefilter_feature_file_when_enabled(tmp_path: Path) -> None:
    strategy_dir = tmp_path / "strategies/chop_grid"
    _mk(strategy_dir / "features.yaml", "features: []\n")
    _mk(strategy_dir / "research/calibrate_roll.default.yaml", "strategy_type: grid\n")
    _mk(
        strategy_dir / "research/research_roll.features_on.yaml",
        "strategy_type: grid\n",
    )
    _mk(
        strategy_dir / "research/validate_static.full_study.yaml",
        "strategy_type: grid\n",
    )
    _mk(strategy_dir / "archetypes/prefilter.yaml", "regime: {}\n")
    _mk(strategy_dir / "archetypes/execution.yaml", "inventory: {}\n")
    cfg = {
        "strategies": {
            "chop_grid": {
                "strategy_type": "grid",
                "config": str(strategy_dir),
                "has_prefilter": True,
            }
        }
    }
    issues = validate_pipeline_strategy_packages(
        pipeline_cfg=cfg,
        project_root=tmp_path,
        allow_strategy_types={"grid", "dual_add_trend"},
    )
    missing = [it.path for it in issues if it.code == "missing_required_file"]
    assert str(strategy_dir / "features_prefilter.yaml") in missing
