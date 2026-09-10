"""Shared FeatureStore layer id for core-4 (bpc/tpc/me/srb) union config."""

from __future__ import annotations

from pathlib import Path

from src.feature_store.layer_naming import default_layer_from_config, resolve_layer_name

TREE_CORE_CONFIG_DIR = Path("config/strategies/_shared")


def resolve_tree_core_layer(explicit: str | None = None) -> str:
    """Return FeatureStore layer name for the shared tree-core config."""
    return resolve_layer_name(explicit, TREE_CORE_CONFIG_DIR)


def default_tree_core_layer() -> str:
    """AUTO layer id: ``features_tree_core_120T_<hash10>``."""
    return default_layer_from_config(TREE_CORE_CONFIG_DIR, prefix="features")
