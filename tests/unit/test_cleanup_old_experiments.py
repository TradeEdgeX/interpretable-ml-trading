"""Tests for scripts/cleanup_old_experiments.get_experiment_dirs."""

from pathlib import Path

from scripts.cleanup_old_experiments import get_experiment_dirs


def test_get_experiment_dirs_uses_history_root(tmp_path: Path) -> None:
    pr = tmp_path
    hist = pr / "results" / "demo" / "research_roll.features_on"
    (hist / "bpc" / "20250101_120000").mkdir(parents=True)
    (hist / "bpc" / "20250102_030000").mkdir(parents=True)
    got = get_experiment_dirs("bpc", history_root=hist, project_root=pr)
    assert len(got) == 2
    assert {d.name for d in got} == {"20250101_120000", "20250102_030000"}


def test_get_experiment_dirs_dedupes_legacy_and_history(tmp_path: Path) -> None:
    pr = tmp_path
    hist = pr / "results" / "x" / "hist"
    same = hist / "me" / "20260101_120000"
    legacy = pr / "results" / "research_history" / "me" / "20260101_120000"
    same.mkdir(parents=True)
    legacy.mkdir(parents=True)
    got = get_experiment_dirs("me", history_root=hist, project_root=pr)
    assert len(got) == 1
