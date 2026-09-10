from src.feature_store.tree_core_layer import resolve_tree_core_layer
from src.feature_store.tree_full_layer import resolve_tree_full_layer


def test_tree_core_and_tree_full_layers_differ() -> None:
    core = resolve_tree_core_layer()
    full = resolve_tree_full_layer()
    assert core.startswith("features_tree_core_120T_")
    assert full.startswith("features_tree_full_120T_")
    assert core != full


def test_tree_full_layer_stable_hash() -> None:
    assert resolve_tree_full_layer() == "features_tree_full_120T_958f665062"
