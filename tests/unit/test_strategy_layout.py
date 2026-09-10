"""Tests for src.config.strategy_layout."""

from pathlib import Path

import pytest

from src.config.strategy_layout import (
    deep_merge_dicts,
    resolve_default_pipeline_config,
    resolve_strategy_package_under_root,
    strategy_packaged_root,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_strategy_packaged_root():
    root = _repo_root()
    p = strategy_packaged_root(root, "bpc")
    assert p == root / "config" / "strategies" / "bpc"


def test_strategy_packaged_root_me_from_root():
    root = _repo_root()
    p = strategy_packaged_root(root, "me")
    assert p == root / "config" / "strategies" / "me"


def test_resolve_me_turbo_single_file_from_root():
    root = _repo_root()
    p, warns = resolve_default_pipeline_config(root, "me", None)
    assert p.name == "calibrate_roll.default.yaml"
    assert p.parts[-3:-1] == ("me", "research")
    assert not warns


def test_resolve_chop_grid_turbo_is_single_file():
    root = _repo_root()
    p, warns = resolve_default_pipeline_config(root, "chop_grid", None)
    assert p.name == "calibrate_roll.default.yaml"
    assert p.parent.name == "research"
    assert not any("Resolved pipeline" in w for w in warns)


def test_resolve_bpc_turbo_single_file():
    root = _repo_root()
    p, warns = resolve_default_pipeline_config(root, "bpc", None)
    assert p.name == "calibrate_roll.default.yaml"
    assert p.parts[-3:-1] == ("bpc", "research")
    assert not warns


def test_resolve_explicit_config_no_warnings():
    root = _repo_root()
    explicit = root / "config" / "pipelines" / "research_pipeline.yaml"
    p, warns = resolve_default_pipeline_config(root, "bpc", explicit)
    assert p == explicit.resolve()
    assert warns == []


def test_resolve_unknown_strategy_fallback():
    root = _repo_root()
    p, warns = resolve_default_pipeline_config(
        root, "___no_such_strategy_slug___", None
    )
    assert p == (root / "config" / "pipelines" / "pcm_orchestrate_2h.yaml").resolve()
    assert warns and "falling back" in warns[0]


def test_resolve_largebar_fade_package_slug_aliases():
    root = _repo_root()
    strategies = root / "config" / "strategies"
    canonical = resolve_strategy_package_under_root(strategies, "largebar_fade")
    assert canonical == (strategies / "largebar_fade").resolve()
    for legacy in ("chop_largebar_fade", "fade"):
        assert resolve_strategy_package_under_root(strategies, legacy) == canonical


def test_deep_merge_golden():
    base = {"a": 1, "nested": {"x": 1, "y": 2}, "list": [1]}
    override = {"b": 2, "nested": {"y": 9, "z": 3}, "list": [2, 3]}
    got = deep_merge_dicts(base, override)
    assert got["a"] == 1
    assert got["b"] == 2
    assert got["nested"] == {"x": 1, "y": 9, "z": 3}
    assert got["list"] == [2, 3]
