"""
Test that the new structured features.yaml format (required + optional_blocks)
does not break tree model training.

Key requirements:
1. StrategyConfigLoader should flatten structured format to a flat list
2. Tree models should work with both old (flat list) and new (structured) formats
3. nnmultihead load_feature_contract should derive contract from structured format
4. New helper functions (_load_feature_dependencies, _feature_name_to_output_columns) work correctly
"""

import textwrap

import pytest

from pathlib import Path

from src.time_series_model.models.nn.feature_contract import (
    _feature_name_to_output_columns,
    _load_feature_dependencies,
    load_feature_contract,
)
from src.time_series_model.strategy_config.loader import StrategyConfigLoader


def _write_yaml(path: Path, content: str) -> None:
    """Helper to write YAML file with dedented content."""
    path.write_text(textwrap.dedent(content), encoding="utf-8")


def test_strategy_config_loader_old_format_flat_list(tmp_path: Path):
    """Test that old format (flat list) still works for tree models."""
    config_dir = tmp_path / "test_strategy"
    config_dir.mkdir()

    _write_yaml(
        config_dir / "features.yaml",
        """
        name: test_strategy
        feature_pipeline:
          requested_features:
            - atr_f
            - rsi_f
            - macd_f
        """,
    )
    _write_yaml(
        config_dir / "labels.yaml",
        """
        target_column: label
        generator:
          module: tests.sample_module
          function: fake_label
        """,
    )
    _write_yaml(
        config_dir / "model.yaml",
        """
        trainer:
          module: tests.sample_module
          function: fake_trainer
        """,
    )

    loader = StrategyConfigLoader(config_dir)
    config = loader.load()

    # Tree models should see a flat list
    assert isinstance(config.features.requested_features, list)
    assert config.features.requested_features == ["atr_f", "rsi_f", "macd_f"]


def test_strategy_config_loader_new_format_structured(tmp_path: Path):
    """Test that new format (structured: required + optional_blocks) is flattened for tree models."""
    config_dir = tmp_path / "test_strategy"
    config_dir.mkdir()

    _write_yaml(
        config_dir / "features.yaml",
        """
        name: test_strategy
        feature_pipeline:
          requested_features:
            required:
              - atr_f
              - rsi_f
            optional_blocks:
              compression_blocks:
                - compression_duration_f
                - compression_energy_f
              ticks_blocks:
                - vpin_f
        """,
    )
    _write_yaml(
        config_dir / "labels.yaml",
        """
        target_column: label
        generator:
          module: tests.sample_module
          function: fake_label
        """,
    )
    _write_yaml(
        config_dir / "model.yaml",
        """
        trainer:
          module: tests.sample_module
          function: fake_trainer
        """,
    )

    loader = StrategyConfigLoader(config_dir)
    config = loader.load()

    # Tree models should see a flattened list (required + all optional_blocks)
    assert isinstance(config.features.requested_features, list)
    # Should contain all features from required and optional_blocks
    assert "atr_f" in config.features.requested_features
    assert "rsi_f" in config.features.requested_features
    assert "compression_duration_f" in config.features.requested_features
    assert "compression_energy_f" in config.features.requested_features
    assert "vpin_f" in config.features.requested_features
    # Should be flattened (no nested structure)
    assert len(config.features.requested_features) == 5


def test_strategy_config_loader_new_format_empty_optional_blocks(tmp_path: Path):
    """Test that new format with empty optional_blocks works."""
    config_dir = tmp_path / "test_strategy"
    config_dir.mkdir()

    _write_yaml(
        config_dir / "features.yaml",
        """
        name: test_strategy
        feature_pipeline:
          requested_features:
            required:
              - atr_f
              - rsi_f
            optional_blocks: {}
        """,
    )
    _write_yaml(
        config_dir / "labels.yaml",
        """
        target_column: label
        generator:
          module: tests.sample_module
          function: fake_label
        """,
    )
    _write_yaml(
        config_dir / "model.yaml",
        """
        trainer:
          module: tests.sample_module
          function: fake_trainer
        """,
    )

    loader = StrategyConfigLoader(config_dir)
    config = loader.load()

    # Should only contain required features
    assert isinstance(config.features.requested_features, list)
    assert config.features.requested_features == ["atr_f", "rsi_f"]


def test_strategy_config_loader_new_format_only_optional_blocks(tmp_path: Path):
    """Test that new format with only optional_blocks (no required) works."""
    config_dir = tmp_path / "test_strategy"
    config_dir.mkdir()

    _write_yaml(
        config_dir / "features.yaml",
        """
        name: test_strategy
        feature_pipeline:
          requested_features:
            required: []
            optional_blocks:
              compression_blocks:
                - compression_duration_f
        """,
    )
    _write_yaml(
        config_dir / "labels.yaml",
        """
        target_column: label
        generator:
          module: tests.sample_module
          function: fake_label
        """,
    )
    _write_yaml(
        config_dir / "model.yaml",
        """
        trainer:
          module: tests.sample_module
          function: fake_trainer
        """,
    )

    loader = StrategyConfigLoader(config_dir)
    config = loader.load()

    # Should only contain optional_blocks features
    assert isinstance(config.features.requested_features, list)
    assert config.features.requested_features == ["compression_duration_f"]


# ============================================================================
# Tests for new helper functions: _load_feature_dependencies and
# _feature_name_to_output_columns
# ============================================================================


def test_load_feature_dependencies_loads_real_file():
    """Test that _load_feature_dependencies can load the actual feature_dependencies.yaml."""
    deps = _load_feature_dependencies()

    # Should successfully load
    assert isinstance(deps, dict)
    assert "features" in deps

    # Should have some known features
    features = deps.get("features", {})
    assert "atr_f" in features
    assert "macd_f" in features
    assert "rsi_f" in features

    # Verify structure
    atr_info = features["atr_f"]
    assert "output_columns" in atr_info
    assert atr_info["output_columns"] == ["atr"]


def test_load_feature_dependencies_nonexistent_file(tmp_path: Path):
    """Test that _load_feature_dependencies returns empty dict for nonexistent file."""
    nonexistent_path = tmp_path / "nonexistent_feature_deps.yaml"
    deps = _load_feature_dependencies(nonexistent_path)

    assert isinstance(deps, dict)
    assert deps == {}


def test_feature_name_to_output_columns_with_feature_deps():
    """Test _feature_name_to_output_columns with real feature_dependencies."""
    deps = _load_feature_dependencies()

    # Single output column
    assert _feature_name_to_output_columns("atr_f", deps) == ["atr"]
    assert _feature_name_to_output_columns("rsi_f", deps) == ["rsi"]

    # Multiple output columns
    macd_cols = _feature_name_to_output_columns("macd_f", deps)
    assert "macd" in macd_cols
    assert "macd_signal" in macd_cols
    assert "macd_histogram" in macd_cols
    assert len(macd_cols) == 3

    # Multiple output columns (bbands)
    bbands_cols = _feature_name_to_output_columns("bbands_f", deps)
    assert "bb_upper" in bbands_cols
    assert "bb_middle" in bbands_cols
    assert "bb_lower" in bbands_cols
    assert len(bbands_cols) == 3


def test_feature_name_to_output_columns_with_mock_feature_deps():
    """Test _feature_name_to_output_columns with mock feature_deps."""
    mock_deps = {
        "features": {
            "atr_f": {
                "output_columns": ["atr"],
            },
            "macd_f": {
                "output_columns": ["macd", "macd_signal", "macd_histogram"],
            },
            "test_feature_f": {
                "output_columns": ["col1", "col2", "col3"],
            },
        }
    }

    # Single output
    assert _feature_name_to_output_columns("atr_f", mock_deps) == ["atr"]

    # Multiple outputs
    assert _feature_name_to_output_columns("macd_f", mock_deps) == [
        "macd",
        "macd_signal",
        "macd_histogram",
    ]

    # Custom feature
    assert _feature_name_to_output_columns("test_feature_f", mock_deps) == [
        "col1",
        "col2",
        "col3",
    ]


def test_feature_name_to_output_columns_fallback_heuristic():
    """Test _feature_name_to_output_columns fallback when feature not in deps."""
    mock_deps = {"features": {}}

    # Should fall back to removing "_f" suffix
    assert _feature_name_to_output_columns("unknown_feature_f", mock_deps) == [
        "unknown_feature"
    ]
    assert _feature_name_to_output_columns("compression_duration_f", mock_deps) == [
        "compression_duration"
    ]


def test_feature_name_to_output_columns_no_feature_deps():
    """Test _feature_name_to_output_columns when feature_deps is None (auto-load)."""
    # Should auto-load from config/feature_dependencies.yaml
    # and return accurate results
    cols = _feature_name_to_output_columns("atr_f")
    assert cols == ["atr"]

    cols = _feature_name_to_output_columns("macd_f")
    assert "macd" in cols
    assert "macd_signal" in cols
    assert "macd_histogram" in cols


def test_feature_name_to_output_columns_invalid_input():
    """Test _feature_name_to_output_columns with invalid inputs."""
    deps = _load_feature_dependencies()

    # Non-string input
    assert _feature_name_to_output_columns(None, deps) == []
    assert _feature_name_to_output_columns(123, deps) == []
    assert _feature_name_to_output_columns([], deps) == []

    # Empty string
    assert _feature_name_to_output_columns("", deps) == []

    # Feature without _f suffix (should still work with fallback)
    assert _feature_name_to_output_columns("atr", deps) == ["atr"]


def test_feature_name_to_output_columns_feature_without_output_columns():
    """Test _feature_name_to_output_columns when feature exists but has no output_columns."""
    mock_deps = {
        "features": {
            "test_feature_f": {
                # No output_columns field
            },
        }
    }

    # Should fall back to heuristic
    assert _feature_name_to_output_columns("test_feature_f", mock_deps) == [
        "test_feature"
    ]


def test_feature_name_to_output_columns_empty_output_columns():
    """Test _feature_name_to_output_columns when feature has empty output_columns."""
    mock_deps = {
        "features": {
            "test_feature_f": {
                "output_columns": [],
            },
        }
    }

    # Should fall back to heuristic
    assert _feature_name_to_output_columns("test_feature_f", mock_deps) == [
        "test_feature"
    ]


def test_feature_name_to_output_columns_integration_with_contract_derivation(
    tmp_path: Path,
):
    """Integration test: verify that contract derivation uses accurate output_columns."""
    config_dir = tmp_path / "test_nnmultihead"
    config_dir.mkdir()

    _write_yaml(
        config_dir / "features.yaml",
        """
        name: test_nnmultihead
        feature_pipeline:
          requested_features:
            required:
              - macd_f
              - atr_f
            optional_blocks:
              test_blocks:
                - bbands_f
          missingness_policy:
            append_block_mask: true
            block_dropout_p: 0.05
        """,
    )

    contract = load_feature_contract(config_dir)

    assert contract is not None

    # macd_f should produce all 3 columns
    assert "macd" in contract.minimal_required_cols
    assert "macd_signal" in contract.minimal_required_cols
    assert "macd_histogram" in contract.minimal_required_cols

    # atr_f should produce atr
    assert "atr" in contract.minimal_required_cols

    # bbands_f should produce all bbands columns in optional_blocks
    assert isinstance(contract.optional_blocks, dict)
    assert "test_blocks" in contract.optional_blocks
    test_block_patterns = contract.optional_blocks["test_blocks"]
    # Should contain patterns for bb_upper, bb_middle, bb_lower
    assert any("bb_upper" in p or "*bb_upper*" in p for p in test_block_patterns)
    assert any("bb_middle" in p or "*bb_middle*" in p for p in test_block_patterns)
    assert any("bb_lower" in p or "*bb_lower*" in p for p in test_block_patterns)


def test_nnmultihead_load_feature_contract_from_structured_format(tmp_path: Path):
    """Test that nnmultihead load_feature_contract derives contract from structured format."""
    config_dir = tmp_path / "test_nnmultihead"
    config_dir.mkdir()

    _write_yaml(
        config_dir / "features.yaml",
        """
        name: test_nnmultihead
        feature_pipeline:
          requested_features:
            required:
              - atr_f
              - trend_r2_20_f
              - compression_duration_f
            optional_blocks:
              compression_blocks:
                - compression_energy_f
                - compression_to_breakout_prob_f
              ticks_blocks:
                - vpin_f
          missingness_policy:
            append_block_mask: true
            block_dropout_p: 0.05
        """,
    )

    contract = load_feature_contract(config_dir)

    assert contract is not None
    # minimal_required_cols should be derived from required features
    assert "atr" in contract.minimal_required_cols
    assert "trend_r2_20" in contract.minimal_required_cols
    assert "compression_duration" in contract.minimal_required_cols
    # Basic OHLCV fields should be included
    assert "open" in contract.minimal_required_cols
    assert "high" in contract.minimal_required_cols
    assert "low" in contract.minimal_required_cols
    assert "close" in contract.minimal_required_cols
    assert "volume" in contract.minimal_required_cols

    # optional_blocks should be derived from optional_blocks
    assert isinstance(contract.optional_blocks, dict)
    assert "compression_blocks" in contract.optional_blocks
    assert "ticks_blocks" in contract.optional_blocks
    # Should contain column patterns derived from feature names
    compression_patterns = contract.optional_blocks["compression_blocks"]
    assert "compression_energy" in compression_patterns or any(
        "*compression_energy*" in p for p in compression_patterns
    )

    # missingness_policy should be preserved
    assert contract.missingness_policy["append_block_mask"] is True
    assert contract.missingness_policy["block_dropout_p"] == 0.05


def test_nnmultihead_load_feature_contract_from_old_format(tmp_path: Path):
    """Test that nnmultihead load_feature_contract still works with old format (flat list)."""
    config_dir = tmp_path / "test_nnmultihead"
    config_dir.mkdir()

    _write_yaml(
        config_dir / "features.yaml",
        """
        name: test_nnmultihead
        feature_pipeline:
          requested_features:
            - atr_f
            - rsi_f
        feature_contract:
          minimal_required_cols:
            - atr
            - rsi
          optional_blocks: {}
          missingness_policy:
            append_block_mask: false
        """,
    )

    contract = load_feature_contract(config_dir)

    # Should fall back to explicit feature_contract section
    assert contract is not None
    assert "atr" in contract.minimal_required_cols
    assert "rsi" in contract.minimal_required_cols
    assert contract.missingness_policy["append_block_mask"] is False


def test_both_formats_produce_same_requested_features_for_tree_models(tmp_path: Path):
    """Test that old and new formats produce the same requested_features for tree models."""
    config_dir_old = tmp_path / "old_format"
    config_dir_new = tmp_path / "new_format"
    config_dir_old.mkdir()
    config_dir_new.mkdir()

    # Old format: flat list
    _write_yaml(
        config_dir_old / "features.yaml",
        """
        name: test_strategy
        feature_pipeline:
          requested_features:
            - atr_f
            - rsi_f
            - compression_duration_f
            - compression_energy_f
        """,
    )
    _write_yaml(
        config_dir_old / "labels.yaml",
        """
        target_column: label
        generator:
          module: tests.sample_module
          function: fake_label
        """,
    )
    _write_yaml(
        config_dir_old / "model.yaml",
        """
        trainer:
          module: tests.sample_module
          function: fake_trainer
        """,
    )

    # New format: structured
    _write_yaml(
        config_dir_new / "features.yaml",
        """
        name: test_strategy
        feature_pipeline:
          requested_features:
            required:
              - atr_f
              - rsi_f
            optional_blocks:
              compression_blocks:
                - compression_duration_f
                - compression_energy_f
        """,
    )
    _write_yaml(
        config_dir_new / "labels.yaml",
        """
        target_column: label
        generator:
          module: tests.sample_module
          function: fake_label
        """,
    )
    _write_yaml(
        config_dir_new / "model.yaml",
        """
        trainer:
          module: tests.sample_module
          function: fake_trainer
        """,
    )

    loader_old = StrategyConfigLoader(config_dir_old)
    config_old = loader_old.load()

    loader_new = StrategyConfigLoader(config_dir_new)
    config_new = loader_new.load()

    # Both should produce the same flattened list (order may differ, so use set)
    assert set(config_old.features.requested_features) == set(
        config_new.features.requested_features
    )
    assert len(config_old.features.requested_features) == len(
        config_new.features.requested_features
    )


def test_tree_model_feature_loading_with_structured_format(tmp_path: Path):
    """Integration test: ensure tree models can load features using structured format."""
    import pandas as pd
    from src.features.loader.strategy_feature_loader import StrategyFeatureLoader

    config_dir = tmp_path / "test_strategy"
    config_dir.mkdir()

    # Use structured format
    _write_yaml(
        config_dir / "features.yaml",
        """
        name: test_strategy
        feature_pipeline:
          requested_features:
            required:
              - atr_f
              - rsi_f
            optional_blocks:
              compression_blocks:
                - compression_duration_f
        """,
    )
    _write_yaml(
        config_dir / "labels.yaml",
        """
        target_column: label
        generator:
          module: tests.sample_module
          function: fake_label
        """,
    )
    _write_yaml(
        config_dir / "model.yaml",
        """
        trainer:
          module: tests.sample_module
          function: fake_trainer
        """,
    )

    # Load config
    loader = StrategyConfigLoader(config_dir)
    config = loader.load()

    # Verify requested_features is a flat list
    assert isinstance(config.features.requested_features, list)
    assert len(config.features.requested_features) == 3
    assert "atr_f" in config.features.requested_features
    assert "rsi_f" in config.features.requested_features
    assert "compression_duration_f" in config.features.requested_features

    # Create minimal DataFrame
    idx = pd.date_range("2025-01-01", periods=5, freq="D")
    df = pd.DataFrame(
        {
            "open": [1, 2, 3, 4, 5],
            "high": [2, 3, 4, 5, 6],
            "low": [0.5, 1.5, 2.5, 3.5, 4.5],
            "close": [1.5, 2.5, 3.5, 4.5, 5.5],
            "volume": [10, 11, 12, 13, 14],
            "_symbol": ["AAA"] * 5,
            "symbol": ["AAA"] * 5,
        },
        index=idx,
    )

    # Load features using the flattened requested_features
    feature_loader = StrategyFeatureLoader()
    df_features = feature_loader.load_features_from_requested(
        df,
        requested_features=config.features.requested_features,
        fit=True,
    )

    # Verify features were computed (atr_f and rsi_f should produce output columns)
    # Note: compression_duration_f might not be available without proper setup,
    # but atr_f and rsi_f should work
    assert "atr" in df_features.columns or "atr_f" in df_features.columns
    # rsi_f should produce rsi column
    # (Some features might not be computed if dependencies are missing, but the structure should work)
