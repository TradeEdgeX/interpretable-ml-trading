"""Regression: find_output_dir accepts train_final output layouts (canonical + legacy)."""

from __future__ import annotations

from pathlib import Path

from scripts.pipeline.steps import find_output_dir


def test_find_output_dir_from_stdout_canonical_layout() -> None:
    text = (
        "Training done.\n"
        "📂 Output directory: results/train_final/bpc/train_final_20260101_120000_rr_extreme\n"
        "Wrote results/train_final/bpc/train_final_20260101_120000_rr_extreme/bpc/model.pkl\n"
    )
    got = find_output_dir(text, "bpc")
    assert got == "results/train_final/bpc/train_final_20260101_120000_rr_extreme/bpc"


def test_find_output_dir_from_stdout_nested_strategy_layout() -> None:
    text = (
        "Training done.\n"
        "📂 Output directory: results/bpc/train_final_20260101_120000_rr_extreme\n"
        "Wrote results/bpc/train_final_20260101_120000_rr_extreme/bpc/model.pkl\n"
    )
    got = find_output_dir(text, "bpc")
    assert got == "results/bpc/train_final_20260101_120000_rr_extreme/bpc"


def test_find_output_dir_from_stdout_legacy_layout() -> None:
    text = "Wrote results/train_final_20260101_120000_rr_extreme/bpc/model.pkl\n"
    got = find_output_dir(text, "bpc")
    assert got == "results/train_final_20260101_120000_rr_extreme/bpc"


def test_find_output_dir_glob_canonical_layout(tmp_path: Path, monkeypatch) -> None:
    import scripts.pipeline.steps as steps

    monkeypatch.setattr(steps, "PROJECT_ROOT", tmp_path)
    leaf = (
        tmp_path
        / "results"
        / "train_final"
        / "me"
        / "train_final_20260202_101010_rr_extreme"
        / "me"
    )
    leaf.mkdir(parents=True)
    got = find_output_dir("", "me")
    assert got == "results/train_final/me/train_final_20260202_101010_rr_extreme/me"


def test_find_output_dir_glob_nested_strategy_layout(
    tmp_path: Path, monkeypatch
) -> None:
    import scripts.pipeline.steps as steps

    monkeypatch.setattr(steps, "PROJECT_ROOT", tmp_path)
    leaf = tmp_path / "results" / "me" / "train_final_20260202_101010_rr_extreme" / "me"
    leaf.mkdir(parents=True)
    got = find_output_dir("", "me")
    assert got == "results/me/train_final_20260202_101010_rr_extreme/me"


def test_find_output_dir_glob_legacy_layout(tmp_path: Path, monkeypatch) -> None:
    import scripts.pipeline.steps as steps

    monkeypatch.setattr(steps, "PROJECT_ROOT", tmp_path)
    leaf = tmp_path / "results" / "train_final_20260202_101010_rr_extreme" / "me"
    leaf.mkdir(parents=True)
    got = find_output_dir("", "me")
    assert got == "results/train_final_20260202_101010_rr_extreme/me"
