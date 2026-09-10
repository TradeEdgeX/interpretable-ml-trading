"""Smoke test: _new_decision_doc.build_markdown templates."""

from __future__ import annotations

import pytest

from scripts._new_decision_doc import _TEMPLATES, build_markdown


_RUNS = [
    {
        "variant": "baseline_recent",
        "engine": "chop_grid",
        "period": "2025-04-01/2026-04-01",
        "strategies_root": "config/strategies",
        "dir": "results/chop_grid/experiments/baseline_recent",
    },
    {
        "variant": "proxy_tpc_recent",
        "engine": "chop_grid",
        "period": "2025-04-01/2026-04-01",
        "strategies_root": "config_experiments/chop_grid_proxy_tpc",
        "dir": "results/chop_grid/experiments/proxy_tpc_recent",
    },
]


def test_default_template_has_side_breakdown_section() -> None:
    md = build_markdown(
        topic="tpc_demo",
        experiment_id="exp1",
        runs=_RUNS,
        promoted_variant=None,
    )
    assert "## 2.3 按 side 分解" in md
    assert "离线 label / IC" in md
    assert "## 1. 变体定义" in md
    assert "**baseline_recent**" in md


def test_c_semantic_proxy_template_lists_entry_features() -> None:
    md = build_markdown(
        topic="chop_grid_proxy",
        experiment_id="exp2",
        runs=_RUNS,
        promoted_variant=None,
        template="c_semantic_proxy",
    )
    assert "语义代理 vs C KPI" in md
    assert "_build_grid_segment_labels.py" in md
    assert "seg_total_r_over_dd" in md
    assert "**baseline_recent**" in md
    assert "**proxy_tpc_recent**" in md
    assert "## 4. Plateau 宽度" in md
    assert "## 2.3" not in md


def test_tree_slug_template_lists_variants_and_pcm() -> None:
    md = build_markdown(
        topic="fast_scalp_initial",
        experiment_id="exp3",
        runs=[
            {
                "variant": "fast_scalp_recent",
                "engine": "event_backtest",
                "period": "2025-04-01/2026-04-01",
                "dir": "results/fast_scalp/experiments/recent",
            },
            {
                "variant": "fast_scalp_bull",
                "engine": "event_backtest",
                "period": "2024-01-01/2025-01-01",
                "dir": "results/fast_scalp/experiments/bull",
            },
        ],
        promoted_variant="fast_scalp_recent",
        template="tree_slug",
    )
    assert "Promote **fast_scalp_recent**" in md
    assert "IC 对齐与 τ plateau" in md
    assert "B/C PCM" in md
    assert "**fast_scalp_recent**" in md
    assert "**fast_scalp_bull**" in md


def test_unknown_template_raises() -> None:
    with pytest.raises(ValueError):
        build_markdown(
            topic="x",
            experiment_id="x",
            runs=[],
            promoted_variant=None,
            template="not_a_real_template",
        )


def test_all_templates_listed() -> None:
    assert set(_TEMPLATES) == {"default", "c_semantic_proxy", "tree_slug"}
