"""Shared FeatureStore layer id for full registry (~940 output columns)."""

from __future__ import annotations

from pathlib import Path

from src.feature_store.layer_naming import default_layer_from_config, resolve_layer_name

TREE_FULL_CONFIG_DIR = Path("config/strategies/_shared")
TREE_FULL_FEATURES_FILE = "features_all.yaml"


def resolve_tree_full_layer(explicit: str | None = None) -> str:
    """Return FeatureStore layer for ``features_all.yaml`` (full registry build)."""
    if explicit:
        return str(explicit)
    return default_layer_from_config(
        TREE_FULL_CONFIG_DIR,
        prefix="features",
        features_file=TREE_FULL_FEATURES_FILE,
    )


def resolve_tree_full_layer_name(layer: str | None = None) -> str:
    """Resolve explicit or AUTO layer for full registry builds."""
    return resolve_layer_name(
        layer,
        TREE_FULL_CONFIG_DIR,
        features_file=TREE_FULL_FEATURES_FILE,
    )
