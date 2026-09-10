"""Scaffold front-matter: format only, never a declared verdict."""

from __future__ import annotations

from pathlib import Path

from src.research.experiment_index import parse_front_matter, read_experiment_meta
from src.research.standardize_experiments import (
    apply_scaffold,
    propose_scaffold,
)


def _write_exp(root: Path, name: str, body: str, **files: str) -> Path:
    exp = root / name
    exp.mkdir(parents=True)
    (exp / "DECISION.md").write_text(body, encoding="utf-8")
    for rel, text in files.items():
        path = exp / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return exp


def test_legacy_decision_gets_empty_verdict_scaffold(tmp_path: Path) -> None:
    exp = _write_exp(
        tmp_path,
        "20260910_ma_cross_wall",
        "# DECISION\n\n结论：REJECT\n",
        **{
            "ma_cross_wall_grid.yaml": (
                "strategy: ma_cross\n"
                "segment_matrix:\n"
                "  segments: [bear_2022, bull_2023_2024]\n"
            )
        },
    )
    plan = propose_scaffold(exp)
    assert plan.action == "write"
    assert plan.fields["verdict"] is None
    assert plan.fields["strategy"] == "ma_cross"
    assert plan.fields["harness"] == "event_backtest"
    assert plan.fields["segments"] == ["bear_2022", "bull_2023_2024"]
    assert plan.fields["kill_switch"] is False

    assert apply_scaffold(plan, write=True)
    meta = read_experiment_meta(exp)
    assert meta.verdict == "reject"
    assert meta.verdict_source == "inferred"
    assert meta.declared is False
    assert meta.strategy == "ma_cross"
    assert meta.harness == "event_backtest"


def test_unregistered_name_does_not_invent_a_family(tmp_path: Path) -> None:
    exp = _write_exp(tmp_path, "20260802_rolling_ladder", "# notes\n")
    plan = propose_scaffold(exp)
    assert plan.fields["strategy"] is None
    assert plan.fields["harness"] is None


def test_combo_yaml_strategy_is_not_stamped(tmp_path: Path) -> None:
    exp = _write_exp(
        tmp_path,
        "20260802_combo",
        "# notes\n",
        **{"combo_grid.yaml": "strategy: ma_cross,other\n"},
    )
    plan = propose_scaffold(exp)
    assert plan.fields["strategy"] is None
    assert plan.fields["harness"] is None


def test_unknown_segments_dropped_when_known_set(tmp_path: Path) -> None:
    exp = _write_exp(
        tmp_path,
        "20260802_ma_cross_custom_seg",
        "# notes\n",
        **{
            "ma_cross_grid.yaml": (
                "strategy: ma_cross\nsegments: [bear_2022, luna_crash_2022]\n"
            )
        },
    )
    plan = propose_scaffold(exp, known_segments={"bear_2022"})
    assert plan.fields["segments"] == ["bear_2022"]


def test_dict_segments_use_id_only(tmp_path: Path) -> None:
    exp = _write_exp(
        tmp_path,
        "20260802_ma_cross_seg",
        "# notes\n",
        **{
            "ma_cross_grid.yaml": (
                "strategy: ma_cross\n"
                "segments:\n"
                "  - id: bear_2022\n"
                "    start_date: '2022-01-01'\n"
                "  - bull_2023_2024\n"
            )
        },
    )
    plan = propose_scaffold(exp)
    assert plan.fields["segments"] == ["bear_2022", "bull_2023_2024"]


def test_auto_scaffold_can_be_refreshed(tmp_path: Path) -> None:
    exp = _write_exp(tmp_path, "20260802_ma_cross_refresh", "# notes\n")
    assert apply_scaffold(propose_scaffold(exp), write=True)
    (exp / "ma_cross_grid.yaml").write_text(
        "strategy: ma_cross\nsegments: [bear_2022]\n", encoding="utf-8"
    )
    plan = propose_scaffold(exp)
    assert plan.action == "write"
    assert apply_scaffold(plan, write=True)
    raw, body = parse_front_matter((exp / "DECISION.md").read_text(encoding="utf-8"))
    assert raw["segments"] == ["bear_2022"]
    assert raw["verdict"] is None
    assert body.lstrip().startswith("# notes")


def test_kill_switch_defaults_false(tmp_path: Path) -> None:
    exp = _write_exp(tmp_path, "20260802_ma_cross_notes", "KS on in a sentence\n")
    plan = propose_scaffold(exp)
    assert plan.fields["kill_switch"] is False


def test_declared_front_matter_is_not_overwritten(tmp_path: Path) -> None:
    body = (
        "---\n"
        "topic: already\n"
        "strategy: ma_cross\n"
        "harness: event_backtest\n"
        "segments: [bear_2022]\n"
        "kill_switch: false\n"
        "verdict: reject\n"
        "---\n\n# done\n"
    )
    exp = _write_exp(tmp_path, "20260803_ma_cross_done", body)
    plan = propose_scaffold(exp)
    assert plan.action == "skip"
    assert apply_scaffold(plan, write=True) is False
    raw, _ = parse_front_matter((exp / "DECISION.md").read_text(encoding="utf-8"))
    assert raw["verdict"] == "reject"


def test_dry_run_does_not_write(tmp_path: Path) -> None:
    exp = _write_exp(tmp_path, "20260804_ma_cross_deadzone", "# open\n")
    original = (exp / "DECISION.md").read_text(encoding="utf-8")
    plan = propose_scaffold(exp)
    assert apply_scaffold(plan, write=False) is True
    assert (exp / "DECISION.md").read_text(encoding="utf-8") == original
