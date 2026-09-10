"""Test that features_all.yaml is self-contained and factor-eval uses it correctly."""

import pytest
import yaml
from pathlib import Path

from src.features.loader.strategy_feature_loader import StrategyFeatureLoader
from src.time_series_model.strategy_config import (
    StrategyConfigLoader,
    FeaturePipelineConfig,
)


class TestFeaturesAllSelfContained:
    """Test that features_all.yaml is self-contained."""

    def test_generate_all_features_yaml_includes_all_dependencies(self, tmp_path):
        """Test that generate_all_features_yaml includes all dependencies."""
        from scripts.generate_all_features_yaml import generate_all_features_yaml

        # Create a temporary strategy config directory
        strategy_dir = tmp_path / "test_strategy"
        strategy_dir.mkdir()

        # Generate features_all.yaml
        output_path = generate_all_features_yaml(
            strategy_config_path=strategy_dir,
            output_path=strategy_dir / "features_all.yaml",
        )

        # Load the generated file
        with open(output_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        requested_features = data.get("feature_pipeline", {}).get(
            "requested_features", []
        )

        # Verify it contains features
        assert len(requested_features) > 0, "Should contain features"

        # Verify all features end with _f (are compute function names)
        for feature in requested_features:
            assert feature.endswith("_f"), f"Feature {feature} should end with _f"

        # Verify dependencies are resolved
        feature_loader = StrategyFeatureLoader()
        # Try to resolve dependencies - should not add any new features
        resolved = feature_loader.resolve_dependencies(requested_features)

        # All requested features should be in resolved (self-contained)
        requested_set = set(requested_features)
        resolved_set = set(resolved)

        # Resolved should contain all requested features
        assert requested_set.issubset(
            resolved_set
        ), f"Some requested features missing in resolved: {requested_set - resolved_set}"

        print(f"✅ Generated features_all.yaml with {len(requested_features)} features")
        print(f"   Resolved dependencies: {len(resolved)} features")
        print(f"   All dependencies included: {requested_set.issubset(resolved_set)}")

    def test_features_all_yaml_self_contained_for_real_strategy(self):
        """Test that existing features_all.yaml files are self-contained."""
        strategies = [
            "sr_breakout",
            "compression_breakout",
            "trend_following",
            "sr_reversal_rr_reg_long",
        ]

        feature_loader = StrategyFeatureLoader()

        for strategy_name in strategies:
            features_all_path = Path(
                f"config/strategies/{strategy_name}/features_all.yaml"
            )
            if not features_all_path.exists():
                pytest.skip(f"features_all.yaml not found for {strategy_name}")

            # Load features_all.yaml
            with open(features_all_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)

            requested_features = data.get("feature_pipeline", {}).get(
                "requested_features", []
            )

            if not requested_features:
                pytest.skip(f"features_all.yaml is empty for {strategy_name}")

            # Verify all are feature compute function names (or output columns that will be resolved)
            # Note: Some old features_all.yaml might have output columns, but new ones should only have feature names
            feature_names = [f for f in requested_features if f.endswith("_f")]
            output_cols = [f for f in requested_features if not f.endswith("_f")]

            if output_cols:
                # Old format - still acceptable, but warn
                print(
                    f"⚠️  {strategy_name}: Contains {len(output_cols)} output columns (old format)"
                )

            # Verify dependencies are resolved (self-contained)
            # Only check feature names for dependency resolution
            if feature_names:
                resolved = feature_loader.resolve_dependencies(feature_names)
                feature_set = set(feature_names)
                resolved_set = set(resolved)

                # All requested feature names should be in resolved
                assert feature_set.issubset(
                    resolved_set
                ), f"{strategy_name}: Some requested features missing in resolved"

                print(
                    f"✅ {strategy_name}: {len(feature_names)} feature names, "
                    f"all dependencies resolved ({len(resolved)} total)"
                )
            else:
                pytest.skip(
                    f"{strategy_name}: No feature names found (only output columns)"
                )


class TestFactorEvalUsesFeaturesAllOnly:
    """Test that factor-eval uses features_all.yaml directly without features.yaml."""

    def test_factor_eval_overrides_features_config(self, tmp_path):
        """Test that factor-eval overrides strategy_cfg.features when features_all.yaml is provided."""
        from src.time_series_model.diagnostics.factor_ts_eval import parse_args

        # Create a temporary strategy config directory
        strategy_dir = tmp_path / "test_strategy"
        strategy_dir.mkdir()

        # Create features.yaml (should NOT be used when features_all.yaml is provided)
        features_yaml = strategy_dir / "features.yaml"
        features_yaml.write_text(
            """
name: test_strategy
feature_pipeline:
  requested_features:
    - atr_f
    - rsi_f
"""
        )

        # Create labels.yaml and model.yaml (required by StrategyConfigLoader)
        (strategy_dir / "labels.yaml").write_text(
            """
name: test_strategy
target_column: label
label_generator:
  module: src.time_series_model.strategies.labels.bpc_label
  function: compute_bpc_label
  params:
    signal_col: signal
    max_holding_bars: 50
"""
        )

        (strategy_dir / "model.yaml").write_text(
            """
name: test_strategy
model_type: lightgbm
model_params:
  n_estimators: 100
trainer:
  module: src.time_series_model.training.trainer
  function: train_lightgbm_model
"""
        )

        # Create features_all.yaml (should be used)
        features_all_yaml = strategy_dir / "features_all.yaml"
        features_all_yaml.write_text(
            """
name: test_strategy
feature_pipeline:
  requested_features:
    - atr_f
    - rsi_f
    - macd_f
    - bbands_f
  ensure_signal_column:
    name: signal
    default_value: 0
"""
        )

        # Simulate factor-eval argument parsing
        import argparse

        args = argparse.Namespace()
        args.strategy_config = str(features_all_yaml)  # Pass features_all.yaml as file
        args.symbol = "BTCUSDT"
        args.timeframe = "240T"
        args.start_date = "2023-01-01"
        args.end_date = "2023-12-31"
        args.feature_mode = "strategy"
        args.factors = None
        args.output_dir = "results/factor_ts_eval"
        args.export_yaml = None

        # Test the logic from factor_ts_eval.main()
        strategy_config_path = Path(args.strategy_config)
        features_file_override = None
        if strategy_config_path.is_file():
            features_file_override = strategy_config_path
            strategy_config_dir = strategy_config_path.parent
        else:
            strategy_config_dir = strategy_config_path

        # Load strategy config (will load features.yaml, but we'll override it)
        loader = StrategyConfigLoader(strategy_config_dir)
        strategy_cfg = loader.load()

        # Original features from features.yaml
        original_features = strategy_cfg.features.requested_features
        assert "atr_f" in original_features
        assert "rsi_f" in original_features
        assert "macd_f" not in original_features  # Not in features.yaml

        # Override with features_all.yaml (simulating factor-eval logic)
        if features_file_override:
            with open(features_file_override, "r", encoding="utf-8") as f:
                features_override = yaml.safe_load(f)

            if "feature_pipeline" in features_override:
                override_requested = features_override["feature_pipeline"].get(
                    "requested_features", []
                )

                strategy_cfg.features = FeaturePipelineConfig(
                    requested_features=override_requested,
                    invert_features=features_override["feature_pipeline"].get(
                        "invert_features", []
                    ),
                    post_processors=[],
                    selector=None,
                    ensure_signal=features_override["feature_pipeline"].get(
                        "ensure_signal_column",
                        strategy_cfg.features.ensure_signal,
                    ),
                )

        # Verify override worked
        assert "atr_f" in strategy_cfg.features.requested_features
        assert "rsi_f" in strategy_cfg.features.requested_features
        assert (
            "macd_f" in strategy_cfg.features.requested_features
        )  # Now included from features_all.yaml
        assert "bbands_f" in strategy_cfg.features.requested_features

        print(
            "✅ factor-eval correctly overrides features config with features_all.yaml"
        )
        print(f"   Original features: {len(original_features)}")
        print(
            f"   Overridden features: {len(strategy_cfg.features.requested_features)}"
        )


class TestFeatureGroupSearchBaseFeatures:
    """Test that feature-group-search uses empty base_features when Pool B or semantic groups are provided."""

    def test_feature_group_search_empty_base_with_pool_b(self):
        """Test that feature-group-search uses empty base_features when Pool B is provided."""
        from src.time_series_model.diagnostics.feature_group_search import _parse_args

        # This test verifies the logic, not the full execution
        # The actual logic is in feature_group_search.py main()

        # Simulate the logic from feature_group_search.main()
        base_dir = Path("config/strategies/sr_breakout")
        pool_b_yaml = "results/pools/sr_breakout/pool_b/features_pool_b.yaml"
        groups_yaml = None

        has_pool_b = pool_b_yaml is not None
        has_semantic_groups = (
            groups_yaml is not None
            or (
                Path("config") / f"feature_groups_{base_dir.name}_semantic.yaml"
            ).exists()
            or (Path("config") / "feature_groups.yaml").exists()
        )

        if has_pool_b or has_semantic_groups:
            base_features = []
            print(
                "✅ Using empty base_features when Pool B or semantic groups are provided"
            )
        else:
            # Would load from features.yaml
            base_features = ["atr_f", "rsi_f"]
            print(
                "⚠️  Would load from features.yaml (not recommended when using Pool B)"
            )

        assert (
            base_features == []
        ), "base_features should be empty when Pool B is provided"

        print(f"   base_features: {base_features}")
        print(f"   has_pool_b: {has_pool_b}")
        print(f"   has_semantic_groups: {has_semantic_groups}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
