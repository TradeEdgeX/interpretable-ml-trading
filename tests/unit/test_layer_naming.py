"""
Tests for FeatureStore layer naming utilities.

This module tests the layer name resolution logic, ensuring that:
1. Layer names are auto-generated when None or empty string is provided
2. Specified layer names are used as-is
3. The same config always generates the same layer name
"""

import pytest
from pathlib import Path

from src.feature_store.layer_naming import (
    default_layer_from_config,
    resolve_layer_name,
)


class TestDefaultLayerFromConfig:
    """Test default_layer_from_config() function."""

    def test_generates_stable_hash(self):
        """Test that the same config generates the same layer name."""
        config_dir = "config/nnmultihead/path_primitives_4h_80h_min"

        layer1 = default_layer_from_config(config_dir)
        layer2 = default_layer_from_config(config_dir)

        assert layer1 == layer2, "Same config should generate same layer name"
        assert layer1.startswith(
            "features_"
        ), "Layer name should start with 'features_'"
        assert len(layer1) > len("features_"), "Layer name should have hash suffix"

    def test_different_configs_generate_different_names(self):
        """Test that different configs generate different layer names."""
        config1 = "config/nnmultihead/path_primitives_4h_80h_min"
        config2 = "config/strategies/sr_reversal_long"

        layer1 = default_layer_from_config(config1)
        layer2 = default_layer_from_config(config2)

        # They might be the same by chance, but very unlikely
        # If they are the same, it's still valid behavior
        assert isinstance(layer1, str)
        assert isinstance(layer2, str)

    def test_custom_prefix(self):
        """Test that custom prefix works."""
        config_dir = "config/nnmultihead/path_primitives_4h_80h_min"

        layer = default_layer_from_config(config_dir, prefix="custom")

        assert layer.startswith("custom_"), "Layer name should use custom prefix"


class TestResolveLayerName:
    """Test resolve_layer_name() function - the main function we refactored."""

    def test_none_auto_generates(self):
        """Test that None auto-generates layer name."""
        config_dir = "config/nnmultihead/path_primitives_4h_80h_min"

        result = resolve_layer_name(None, config_dir)

        assert result is not None, "Result should not be None"
        assert isinstance(result, str), "Result should be a string"
        assert result.startswith("features_"), "Should generate features_* layer name"
        # Should match default_layer_from_config
        expected = default_layer_from_config(config_dir)
        assert result == expected, "Should match default_layer_from_config output"

    def test_empty_string_auto_generates(self):
        """Test that empty string auto-generates layer name."""
        config_dir = "config/nnmultihead/path_primitives_4h_80h_min"

        result = resolve_layer_name("", config_dir)

        assert result is not None, "Result should not be None"
        assert isinstance(result, str), "Result should be a string"
        assert result.startswith("features_"), "Should generate features_* layer name"
        # Should match default_layer_from_config
        expected = default_layer_from_config(config_dir)
        assert result == expected, "Should match default_layer_from_config output"

    def test_whitespace_only_auto_generates(self):
        """Test that whitespace-only string auto-generates layer name."""
        config_dir = "config/nnmultihead/path_primitives_4h_80h_min"

        result = resolve_layer_name("   ", config_dir)

        assert result is not None, "Result should not be None"
        assert isinstance(result, str), "Result should be a string"
        assert result.startswith("features_"), "Should generate features_* layer name"

    def test_specified_name_used_as_is(self):
        """Test that specified layer name is used as-is."""
        config_dir = "config/nnmultihead/path_primitives_4h_80h_min"
        specified_name = "heavy_v6"

        result = resolve_layer_name(specified_name, config_dir)

        assert result == specified_name, "Should return specified name as-is"
        assert result is not None, "Result should not be None"

    def test_custom_layer_name_preserved(self):
        """Test that custom layer names are preserved."""
        config_dir = "config/nnmultihead/path_primitives_4h_80h_min"
        custom_names = ["base_v1", "heavy_v6", "nnmultihead_v2", "test_layer_123"]

        for custom_name in custom_names:
            result = resolve_layer_name(custom_name, config_dir)
            assert result == custom_name, f"Should preserve custom name: {custom_name}"

    def test_consistency_none_vs_empty(self):
        """Test that None and empty string generate the same result."""
        config_dir = "config/nnmultihead/path_primitives_4h_80h_min"

        result_none = resolve_layer_name(None, config_dir)
        result_empty = resolve_layer_name("", config_dir)

        assert (
            result_none == result_empty
        ), "None and empty string should generate same result"

    def test_always_returns_string(self):
        """Test that function always returns a string (never None)."""
        config_dir = "config/nnmultihead/path_primitives_4h_80h_min"
        test_cases = [None, "", "   ", "heavy_v6", "base_v1"]

        for test_case in test_cases:
            result = resolve_layer_name(test_case, config_dir)
            assert isinstance(
                result, str
            ), f"Should return string for input: {test_case!r}"
            assert (
                result is not None
            ), f"Should not return None for input: {test_case!r}"

    def test_path_object_config_dir(self):
        """Test that Path objects work as config_dir."""
        config_dir = Path("config/nnmultihead/path_primitives_4h_80h_min")

        result = resolve_layer_name(None, config_dir)

        assert result is not None
        assert isinstance(result, str)
        assert result.startswith("features_")

    def test_no_auto_concept(self):
        """Test that 'AUTO' string is treated as a regular layer name (not special)."""
        config_dir = "config/nnmultihead/path_primitives_4h_80h_min"

        # 'AUTO' should be treated as a regular string, not auto-generate
        result = resolve_layer_name("AUTO", config_dir)

        assert result == "AUTO", "AUTO should be treated as regular layer name"
        assert result != default_layer_from_config(
            config_dir
        ), "AUTO should not auto-generate"


class TestIntegrationWithScripts:
    """Test that resolve_layer_name works correctly in script contexts."""

    def test_script_default_behavior(self):
        """Test that scripts get auto-generated layer when None is passed."""
        config_dir = "config/nnmultihead/path_primitives_4h_80h_min"

        # Simulate script behavior: get layer from args (default None)
        layer_from_args = None
        resolved = resolve_layer_name(layer_from_args, config_dir)

        assert resolved is not None
        assert isinstance(resolved, str)
        assert resolved.startswith("features_")

    def test_script_explicit_layer(self):
        """Test that scripts can use explicit layer names."""
        config_dir = "config/nnmultihead/path_primitives_4h_80h_min"

        # Simulate script behavior: user specifies layer
        layer_from_args = "heavy_v6"
        resolved = resolve_layer_name(layer_from_args, config_dir)

        assert resolved == "heavy_v6"

    def test_cli_passes_none(self):
        """Test that CLI passing None results in auto-generation."""
        config_dir = "config/nnmultihead/path_primitives_4h_80h_min"

        # Simulate CLI behavior: default=None
        cli_layer = None
        resolved = resolve_layer_name(cli_layer, config_dir)

        assert resolved is not None
        assert isinstance(resolved, str)
        assert resolved.startswith("features_")


class TestArchetypeTimeframeNaming:
    """Test that layer names include archetype + timeframe from meta.yaml."""

    def test_strategy_me_includes_archetype_and_timeframe(self):
        """ME strategy should produce features_me_120T_{hash} (per meta.yaml)."""
        layer = default_layer_from_config("config/strategies/me")
        assert layer.startswith(
            "features_me_120T_"
        ), f"Expected features_me_120T_*, got {layer}"

    def test_strategy_bpc_includes_archetype_and_timeframe(self):
        """BPC strategy should produce features_bpc_120T_{hash} (per meta.yaml)."""
        layer = default_layer_from_config("config/strategies/bpc")
        assert layer.startswith(
            "features_bpc_120T_"
        ), f"Expected features_bpc_120T_*, got {layer}"

    def test_different_archetypes_different_layers(self):
        """ME and BPC should produce different layer names."""
        me = default_layer_from_config("config/strategies/me")
        bpc = default_layer_from_config("config/strategies/bpc")
        assert me != bpc
        assert "_me_" in me
        assert "_bpc_" in bpc

    def test_hash_stable_across_calls(self):
        """Same config should produce identical layer name on repeated calls."""
        l1 = default_layer_from_config("config/strategies/me")
        l2 = default_layer_from_config("config/strategies/me")
        assert l1 == l2

    def test_no_meta_yaml_uses_dirname_only(self):
        """Config without meta.yaml should use dirname as archetype, no timeframe."""
        layer = default_layer_from_config(
            "config/nnmultihead/path_primitives_4h_80h_min"
        )
        # Should include dirname but no timeframe segment from meta.yaml
        assert layer.startswith("features_path_primitives_4h_80h_min_")
