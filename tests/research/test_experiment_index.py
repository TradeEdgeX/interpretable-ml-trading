"""Experiment lineage: DECISION.md front-matter → machine-readable index."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.research.experiment_index import (
    KPI_KEYS,
    VERDICTS,
    build_index,
    filter_rows,
    infer_verdict,
    infer_verdict_detail,
    load_known_segments,
    parse_front_matter,
    read_experiment_meta,
    validate_meta,
)

DECLARED = """---
topic: 20260910_ma_cross_demo
strategy: ma_cross
harness: event_backtest
segments: [bear_2022, bull_2023_2024]
kill_switch: false
verdict: reject
kpi: {cagr: 12.2, calmar: 0.31, maxdd: -39.0}
supersedes: [20260901_ma_cross_prior]
tags: [ma-cross, demo]
---

# DECISION — ma_cross demo

正文里出现 PROMOTE 这个词也不应该改变判决。
"""

LEGACY = """# DECISION — old experiment

结论：REJECT，三段回测全面变差。
"""


def _write_exp(root: Path, name: str, decision: str | None) -> Path:
    exp = root / name
    exp.mkdir(parents=True)
    if decision is not None:
        (exp / "DECISION.md").write_text(decision, encoding="utf-8")
    return exp


def test_parse_front_matter_splits_body() -> None:
    meta, body = parse_front_matter(DECLARED)
    assert meta["verdict"] == "reject"
    assert body.lstrip().startswith("# DECISION")


def test_parse_front_matter_absent_returns_full_text() -> None:
    meta, body = parse_front_matter(LEGACY)
    assert meta == {}
    assert body == LEGACY


def test_parse_front_matter_ignores_mid_document_fence() -> None:
    text = "# Title\n\n---\nverdict: promote\n---\n"
    meta, _ = parse_front_matter(text)
    assert meta == {}


def test_declared_verdict_beats_prose(tmp_path: Path) -> None:
    exp = _write_exp(tmp_path, "20260910_ma_cross_demo", DECLARED)
    meta = read_experiment_meta(exp)
    assert meta.verdict == "reject"
    assert meta.verdict_source == "declared"
    assert meta.declared is True
    assert meta.record_class == "trusted"
    assert meta.harness == "event_backtest"
    assert meta.kill_switch is False
    assert meta.kpi["cagr"] == pytest.approx(12.2)
    assert meta.supersedes == ("20260901_ma_cross_prior",)
    assert meta.issues == ()


def test_legacy_doc_falls_back_to_inferred(tmp_path: Path) -> None:
    exp = _write_exp(tmp_path, "20260101_legacy_topic", LEGACY)
    meta = read_experiment_meta(exp)
    assert meta.verdict == "reject"
    assert meta.verdict_source == "inferred"
    assert meta.declared is False
    assert meta.record_class == "legacy"


def test_missing_decision_doc(tmp_path: Path) -> None:
    exp = _write_exp(tmp_path, "20260101_no_decision", None)
    meta = read_experiment_meta(exp)
    assert meta.verdict is None
    assert meta.verdict_source == "missing"


@pytest.mark.parametrize("verdict", VERDICTS)
def test_all_verdicts_accepted(verdict: str) -> None:
    meta = validate_meta(
        {
            "verdict": verdict,
            "harness": "event_backtest",
            "segments": ["bear_2022"],
            "kill_switch": False,
        }
    )
    assert meta.verdict == verdict
    assert meta.issues == ()


def test_unknown_verdict_reported_not_raised() -> None:
    meta = validate_meta({"verdict": "looks-good"})
    assert meta.verdict is None
    assert any("unknown" in i for i in meta.issues)


def test_declared_verdict_requires_evidence_fields() -> None:
    meta = validate_meta({"verdict": "promote"})
    joined = " ".join(meta.issues)
    assert "harness" in joined
    assert "segments" in joined
    assert "kill_switch" in joined


def test_chinese_tag_is_invalid() -> None:
    meta = validate_meta(
        {
            "verdict": "reject",
            "harness": "event_backtest",
            "segments": ["bear_2021"],
            "kill_switch": False,
            "tags": ["止损"],
        },
        known_segments={"bear_2021"},
    )
    assert any("ASCII slug" in i for i in meta.issues)


def test_total_r_rejected_as_headline_kpi() -> None:
    meta = validate_meta(
        {
            "verdict": "promote",
            "harness": "event_backtest",
            "segments": ["bear_2022"],
            "kill_switch": False,
            "kpi": {"total_r": 242, "cagr": 18.0},
        }
    )
    assert "total_r" not in meta.kpi
    assert meta.kpi["cagr"] == pytest.approx(18.0)
    assert any("total_r" in i for i in meta.issues)


def test_unknown_harness_and_segment_flagged() -> None:
    meta = validate_meta(
        {
            "verdict": "promote",
            "harness": "my_own_backtest",
            "segments": ["made_up_segment"],
            "kill_switch": False,
        },
        known_segments={"bear_2022"},
    )
    joined = " ".join(meta.issues)
    assert "harness" in joined
    assert "made_up_segment" in joined


def test_non_bool_kill_switch_flagged() -> None:
    meta = validate_meta(
        {
            "verdict": "promote",
            "harness": "event_backtest",
            "segments": ["bear_2022"],
            "kill_switch": "off",
        }
    )
    assert meta.kill_switch is None
    assert any("kill_switch" in i for i in meta.issues)


def test_infer_verdict_prefers_explicit_reject() -> None:
    assert infer_verdict("满足 promote 三条杠 → REJECT") == "reject"
    assert infer_verdict("") is None


def test_mixed_verdicts_are_ambiguous_not_guessed() -> None:
    """A doc that PROMOTEs one track and REJECTs another has no single verdict."""
    text = "Track A: PROMOTE 排序字段。Track B/C: REJECT 软惩罚与加权仓位。"
    assert infer_verdict_detail(text) == (None, "ambiguous")
    assert infer_verdict(text) is None


def test_declared_verdict_survives_mixed_prose(tmp_path: Path) -> None:
    body = "\n正文 Track A PROMOTE，Track B REJECT。\n"
    exp = _write_exp(
        tmp_path,
        "20260805_mixed",
        "---\nverdict: promote\nharness: event_backtest\n"
        "segments: [bear_2022]\nkill_switch: false\n---\n" + body,
    )
    meta = read_experiment_meta(exp)
    assert (meta.verdict, meta.verdict_source) == ("promote", "declared")


@pytest.mark.parametrize(
    "text",
    [
        "## Promote\n\n- [ ] 三条杠（LAYER_PROMOTION_CRITERIA §1）",
        "跑 `mlbot research promote-baseline` 完成",
        "见 LAYER_PROMOTION_CRITERIA.md",
        "combo COMBO_NOT_PROMOTE 见另一实验",
    ],
)
def test_checklist_boilerplate_is_not_a_verdict(text: str) -> None:
    """Headings and command names must not be mistaken for a conclusion."""
    assert infer_verdict(text) is None


def test_build_index_shape_and_summary(tmp_path: Path) -> None:
    root = tmp_path / "config" / "experiments"
    root.mkdir(parents=True)
    _write_exp(root, "20260818_ashare_full_tp15", DECLARED)
    _write_exp(root, "20260101_legacy_topic", LEGACY)
    _write_exp(root, "20260102_no_decision", None)
    _write_exp(root, "_template", DECLARED)  # underscore dirs are skipped

    index = build_index(repo_root=tmp_path, experiments_root=root)

    assert index["summary"]["count"] == 3
    assert index["summary"]["by_source"]["declared"] == 1
    assert index["summary"]["by_source"]["inferred"] == 1
    assert index["summary"]["by_source"]["missing"] == 1
    assert index["summary"]["by_record_class"]["trusted"] == 1
    assert index["summary"]["by_record_class"]["legacy"] == 2
    assert index["summary"]["invalid_count"] == 0
    ids = {r["id"] for r in index["rows"]}
    assert "_template" not in ids
    json.dumps(index)  # must stay serialisable


def test_build_index_reports_invalid_rows(tmp_path: Path) -> None:
    root = tmp_path / "config" / "experiments"
    root.mkdir(parents=True)
    _write_exp(root, "20260103_broken", "---\nverdict: promote\n---\n\n# x\n")

    index = build_index(repo_root=tmp_path, experiments_root=root)
    assert index["invalid"] == ["20260103_broken"]
    assert index["summary"]["invalid_count"] == 1


def test_filter_rows_by_verdict_tag_and_declared(tmp_path: Path) -> None:
    root = tmp_path / "config" / "experiments"
    root.mkdir(parents=True)
    _write_exp(root, "20260910_ma_cross_demo", DECLARED)
    _write_exp(root, "20260101_legacy_topic", LEGACY)

    rows = build_index(repo_root=tmp_path, experiments_root=root)["rows"]

    assert len(filter_rows(rows, verdict="reject")) == 2
    assert len(filter_rows(rows, verdict="reject", declared_only=True)) == 1
    assert len(filter_rows(rows, tag="ma-cross")) == 1
    assert len(filter_rows(rows, strategy="ma_cross")) == 1
    assert len(filter_rows(rows, query="ma_cross")) == 1
    assert len(filter_rows(rows, query="nonexistent")) == 0
    assert len(filter_rows(rows, record_classes=["trusted"])) == 1
    assert len(filter_rows(rows, record_classes=["legacy"])) == 1


def test_auto_scaffold_and_open_classes(tmp_path: Path) -> None:
    scaffold = (
        "---\nstrategy: tpc\nharness: event_backtest\n"
        "verdict:\nstandardization: auto-scaffold\n---\n\n# old\nREJECT\n"
    )
    opened = (
        "---\nstrategy: ma_cross\nharness: event_backtest\n"
        "segments: [bear_2022, bull_2023_2024, recent_range_to_bear]\n"
        "kill_switch: false\nverdict:\n---\n\n# DECISION\n"
    )
    a = _write_exp(tmp_path, "20260701_tpc_old", scaffold)
    b = _write_exp(tmp_path, "20260823_new_open", opened)
    assert read_experiment_meta(a).record_class == "legacy"
    assert read_experiment_meta(b).record_class == "open"


def test_known_segments_include_canonical_three() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    segments = load_known_segments(repo_root)
    assert {"bear_2022", "bull_2023_2024", "recent_range_to_bear"} <= segments


def test_template_decision_front_matter_is_parseable() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    template = repo_root / "config" / "experiments" / "_template" / "DECISION.md"
    raw, body = parse_front_matter(template.read_text(encoding="utf-8"))
    assert raw["verdict"] is None, "template must ship with an unset verdict"
    assert "kill_switch" in raw
    assert set(raw) >= {"harness", "segments", "kill_switch", "verdict", "kpi"}
    assert "# DECISION" in body
    # Placeholder KPI table must reference headline KPIs, not Total R.
    assert "Total R" not in body
    assert all(k in {"cagr", "calmar", "win_rate", "maxdd", "sharpe"} for k in KPI_KEYS)


def test_scaffolded_experiment_yields_open_verdict(tmp_path: Path) -> None:
    """`mlbot research init` output must parse and start as an open experiment."""
    from scripts.research.init_experiment import _render

    repo_root = Path(__file__).resolve().parents[2]
    template = repo_root / "config" / "experiments" / "_template" / "DECISION.md"
    rendered = _render(
        template.read_text(encoding="utf-8"),
        topic="20260901_ma_cross_demo",
        strategy="ma_cross",
        segment="recent_6m_oos",
        date="20260901",
    )
    exp = _write_exp(tmp_path, "20260901_ma_cross_demo", rendered)

    meta = read_experiment_meta(exp)
    assert meta.verdict is None
    assert meta.verdict_source == "missing"
    assert meta.strategy == "ma_cross"
    assert meta.issues == (), "a fresh scaffold must not report front-matter issues"
