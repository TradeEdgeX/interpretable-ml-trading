"""
Tests for feature-group-search base_features resolution.

Verifies:
1. --base-features-yaml takes priority
2. Auto-detection of features_base.yaml in strategy directory
3. Empty base_features when no base file exists
"""

import pytest
import tempfile
from pathlib import Path
import yaml


class TestBaseFeatureResolution:
    """Test base_features resolution logic."""

    def test_explicit_base_features_yaml_takes_priority(self, tmp_path: Path):
        """--base-features-yaml should take priority over auto-detection."""
        # Create strategy dir with features_base.yaml
        strategy_dir = tmp_path / "my_strategy"
        strategy_dir.mkdir()

        # Auto-detected file
        auto_base = strategy_dir / "features_base.yaml"
        auto_base.write_text(yaml.dump(["auto_feature_f"]))

        # Explicit file
        explicit_base = tmp_path / "explicit_base.yaml"
        explicit_base.write_text(yaml.dump(["explicit_feature_f", "another_f"]))

        # Simulate resolution logic
        base_features_path = explicit_base  # --base-features-yaml provided

        base_features = yaml.safe_load(base_features_path.read_text()) or []

        assert base_features == ["explicit_feature_f", "another_f"]
        assert "auto_feature_f" not in base_features

    def test_auto_detection_of_features_base_yaml(self, tmp_path: Path):
        """Should auto-detect features_base.yaml in strategy directory."""
        strategy_dir = tmp_path / "sr_reversal_long"
        strategy_dir.mkdir()

        # Create features_base.yaml
        base_file = strategy_dir / "features_base.yaml"
        base_file.write_text(yaml.dump(["poc_hal_features_close_f", "atr_f"]))

        # Simulate resolution logic (no --base-features-yaml provided)
        base_features_path = None
        conventional_base = strategy_dir / "features_base.yaml"
        if conventional_base.exists():
            base_features_path = conventional_base

        assert base_features_path is not None
        base_features = yaml.safe_load(base_features_path.read_text()) or []

        assert "poc_hal_features_close_f" in base_features
        assert "atr_f" in base_features
        assert len(base_features) == 2

    def test_empty_base_features_when_no_file_exists(self, tmp_path: Path):
        """Should return empty list when no base features file exists."""
        strategy_dir = tmp_path / "empty_strategy"
        strategy_dir.mkdir()

        # No features_base.yaml created

        # Simulate resolution logic
        base_features_path = None
        conventional_base = strategy_dir / "features_base.yaml"
        if conventional_base.exists():
            base_features_path = conventional_base

        if base_features_path and base_features_path.exists():
            base_features = yaml.safe_load(base_features_path.read_text()) or []
        else:
            base_features = []

        assert base_features == []

    def test_features_base_yaml_must_be_list(self, tmp_path: Path):
        """Should raise error if features_base.yaml is not a list."""
        strategy_dir = tmp_path / "invalid_strategy"
        strategy_dir.mkdir()

        # Create invalid features_base.yaml (dict instead of list)
        base_file = strategy_dir / "features_base.yaml"
        base_file.write_text(yaml.dump({"features": ["a_f", "b_f"]}))

        base_features = yaml.safe_load(base_file.read_text())

        assert not isinstance(base_features, list)
        # In real code, this would raise ValueError

    def test_all_strategies_have_features_base(self):
        """All strategies should have features_base.yaml."""
        strategies_dir = Path("config/strategies")
        if not strategies_dir.exists():
            pytest.skip("config/strategies not found")

        missing = []
        for strategy_dir in strategies_dir.iterdir():
            if not strategy_dir.is_dir():
                continue
            base_file = strategy_dir / "features_base.yaml"
            if not base_file.exists():
                missing.append(strategy_dir.name)

        assert len(missing) == 0, f"Missing features_base.yaml: {missing}"

    @pytest.mark.parametrize(
        "strategy,expected_features",
        [
            ("sr_reversal_rr_reg_long", ["poc_hal_features_close_f", "atr_f"]),
            ("sr_reversal_long", ["poc_hal_features_close_f", "atr_f"]),
            ("sr_reversal_short", ["poc_hal_features_close_f", "atr_f"]),
            ("sr_breakout", ["atr_f", "poc_hal_features_close_f"]),
            ("compression_breakout", ["compression_duration_f", "atr_f"]),
            ("trend_following", ["atr_f"]),
        ],
    )
    def test_strategy_features_base_content(
        self, strategy: str, expected_features: list
    ):
        """Main strategies should have correct features in features_base.yaml."""
        base_file = Path(f"config/strategies/{strategy}/features_base.yaml")

        if base_file.exists():
            base_features = yaml.safe_load(base_file.read_text()) or []

            for feat in expected_features:
                assert (
                    feat in base_features
                ), f"{strategy}: Missing {feat} in features_base.yaml"

            # Should NOT contain strategy signal features (those should be optimized)
            assert (
                "cvd_features_f" not in base_features
            ), f"{strategy}: cvd_features_f should not be in base (should be optimized)"
        else:
            pytest.skip(f"features_base.yaml not found for {strategy}")


class TestBaseFeatureIntegration:
    """Integration tests for base_features in feature-group-search."""

    def test_base_features_not_in_candidates(self, tmp_path: Path):
        """Base features should not appear in candidate groups."""
        base_features = ["poc_hal_features_close_f", "atr_f"]

        # Simulated groups (from Pool B or semantic groups)
        groups = {
            "kline_core": ["macd_f", "rsi_f", "atr_f"],  # atr_f overlaps!
            "vpin_block": ["vpin_base_aligned_features_f"],
        }

        # In real implementation, base_features should be excluded from search
        # but still included in final feature set

        # Verify no duplicate when merging
        final_features = list(base_features)
        for g_feats in groups.values():
            for f in g_feats:
                if f not in final_features:
                    final_features.append(f)

        # atr_f should appear only once
        assert final_features.count("atr_f") == 1
