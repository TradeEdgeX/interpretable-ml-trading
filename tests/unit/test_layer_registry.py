from src.feature_store.tree_core_layer import default_tree_core_layer
from src.research.layer_registry import writeback_path


def test_writeback_path_gate():
    p = writeback_path("bpc", "gate")
    assert p.name == "gate.yaml"
    assert "bpc" in str(p)


def test_tree_core_layer():
    layer = default_tree_core_layer()
    assert "tree_core" in layer
