"""Tests for shared tree-core FeatureStore layer naming."""

from src.feature_store.tree_core_layer import (
    default_tree_core_layer,
    resolve_tree_core_layer,
)


def test_tree_core_layer_auto_format():
    layer = default_tree_core_layer()
    assert layer.startswith("features_tree_core_120T_")
    assert len(layer.split("_")[-1]) == 10


def test_resolve_tree_core_layer_explicit():
    assert resolve_tree_core_layer("my_custom_layer") == "my_custom_layer"
