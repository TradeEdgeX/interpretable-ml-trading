"""Tests for src.config.strategy_layout."""

from pathlib import Path

from src.config.strategy_layout import (
    deep_merge_dicts,
    resolve_default_pipeline_config,
    resolve_strategy_package_under_root,
    strategy_packaged_root,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_strategy_packaged_root_ma_cross():
    root = _repo_root()
    p = strategy_packaged_root(root, "ma_cross")
    assert p == root / "config" / "strategies" / "ma_cross"
    assert p.is_dir()
    assert (
        resolve_strategy_package_under_root(root / "config" / "strategies", "ma_cross")
        == p.resolve()
    )


def test_resolve_explicit_config_no_warnings():
    root = _repo_root()
    explicit = root / "config" / "pipelines" / "research_pipeline_template.yaml"
    p, warns = resolve_default_pipeline_config(root, "ma_cross", explicit)
    assert p == explicit.resolve()
    assert warns == []


def test_resolve_unknown_strategy_fallback():
    root = _repo_root()
    p, warns = resolve_default_pipeline_config(
        root, "___no_such_strategy_slug___", None
    )
    assert p == (root / "config" / "pipelines" / "pcm_orchestrate_2h.yaml").resolve()
    assert warns and "falling back" in warns[0]


def test_deep_merge_golden():
    base = {"a": 1, "nested": {"x": 1, "y": 2}, "list": [1]}
    override = {"b": 2, "nested": {"y": 9, "z": 3}, "list": [2, 3]}
    got = deep_merge_dicts(base, override)
    assert got["a"] == 1
    assert got["b"] == 2
    assert got["nested"] == {"x": 1, "y": 9, "z": 3}
    assert got["list"] == [2, 3]
