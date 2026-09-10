from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.research.court_run import (
    CourtRunError,
    argv_for_grid,
    check_runnable,
    resolve_court_grid,
    run_experiment,
    validate_court_grid,
)
from src.research.experiment_gate import find_segment_kpi_rows
from src.research.replay import scaffold_replay
from src.research.scorecard import harvest_scorecard


def _decision(exp: Path, *, strategy: str = "ma_cross", verdict: str = "") -> None:
    exp.mkdir(parents=True, exist_ok=True)
    (exp / "DECISION.md").write_text(
        "---\n"
        f"strategy: {strategy}\n"
        "harness: event_backtest\n"
        "segments: [bear_2022, bull_2023_2024, recent_range_to_bear]\n"
        "kill_switch: false\n"
        f"verdict: {verdict}\n"
        "---\n# demo\n",
        encoding="utf-8",
    )


def _grid_body(experiment_id: str, family: str = "ma_cross") -> dict:
    return {
        "experiment_id": experiment_id,
        "strategy": family,
        "segment_matrix": {
            "segments": ["bear_2022", "bull_2023_2024", "recent_range_to_bear"],
            "variants": [
                {
                    "suffix": "baseline",
                    "output_dir": (
                        f"results/{family}/experiments/{experiment_id}/baseline"
                    ),
                }
            ],
        },
    }


def _write_grid(exp: Path, name: str = "court_grid.yaml") -> Path:
    path = exp / name
    path.write_text(
        yaml.safe_dump(_grid_body(exp.name), sort_keys=False), encoding="utf-8"
    )
    return path


def test_argv_is_variant_grid_not_family_hint(tmp_path: Path) -> None:
    exp = tmp_path / "config" / "experiments" / "20260823_ma_cross_demo"
    _decision(exp)
    grid = _write_grid(exp)
    argv = argv_for_grid(grid)
    assert "--variant-grid" in argv
    assert str(grid) in argv
    assert "--quiet-signal-logs" in argv
    assert "--strategy" not in argv


def test_run_dry_run_prints_and_does_not_call(tmp_path: Path) -> None:
    exp = tmp_path / "config" / "experiments" / "20260823_ma_cross_demo"
    _decision(exp)
    _write_grid(exp)
    called = []

    def _boom(*_a, **_k):
        called.append(True)
        raise AssertionError("harness must not start")

    rc = run_experiment(
        exp.name,
        repo_root=tmp_path,
        execute=False,
        runner=_boom,
    )
    assert rc == 0
    assert called == []


def test_run_executes_argv_via_runner(tmp_path: Path) -> None:
    exp = tmp_path / "config" / "experiments" / "20260823_ma_cross_demo"
    _decision(exp)
    grid = _write_grid(exp)
    seen: list[list[str]] = []

    def _ok(argv, **_k):
        seen.append(list(argv))
        return 0

    rc = run_experiment(
        exp.name,
        repo_root=tmp_path,
        runner=_ok,
        refresh_index=False,
    )
    assert rc == 0
    assert seen[0] == argv_for_grid(grid)
    text = (exp / "DECISION.md").read_text(encoding="utf-8")
    assert "results/ma_cross/experiments/20260823_ma_cross_demo" in text


def test_refuse_rolling(tmp_path: Path) -> None:
    exp = tmp_path / "config" / "experiments" / "20260823_roll"
    _decision(exp, strategy="rolling_trend")
    with pytest.raises(CourtRunError, match="event_backtest"):
        check_runnable(exp, repo_root=tmp_path)


def test_refuse_trusted_without_force(tmp_path: Path) -> None:
    exp = tmp_path / "config" / "experiments" / "20260823_done"
    _decision(exp, verdict="reject")
    with pytest.raises(CourtRunError, match="--force"):
        check_runnable(exp, repo_root=tmp_path)
    assert check_runnable(exp, repo_root=tmp_path, force=True) == "ma_cross"


def test_refuse_no_grid_and_run_grid_py(tmp_path: Path) -> None:
    exp = tmp_path / "config" / "experiments" / "20260823_script"
    _decision(exp)
    (exp / "run_grid.py").write_text("# leftover\n", encoding="utf-8")
    with pytest.raises(CourtRunError, match="run_grid.py is not the court"):
        resolve_court_grid(exp)


def test_refuse_two_grids(tmp_path: Path) -> None:
    exp = tmp_path / "config" / "experiments" / "20260823_two"
    _decision(exp)
    _write_grid(exp, "a_grid.yaml")
    _write_grid(exp, "b_grid.yaml")
    with pytest.raises(CourtRunError, match="multiple"):
        resolve_court_grid(exp)


def test_refuse_flat_output_dir(tmp_path: Path) -> None:
    data = _grid_body("20260823_flat")
    data["segment_matrix"]["variants"][0][
        "output_dir"
    ] = "results/event_backtest/20260823_flat"
    with pytest.raises(CourtRunError, match="must start with"):
        validate_court_grid(data, family="ma_cross", experiment_id="20260823_flat")


def test_replay_copies_and_rewrites_grid(tmp_path: Path) -> None:
    parent = tmp_path / "config" / "experiments" / "20260101_srb_old"
    _decision(parent)
    (parent / "old_grid.yaml").write_text(
        yaml.safe_dump(_grid_body("20260101_srb_old"), sort_keys=False),
        encoding="utf-8",
    )
    dest = scaffold_replay("20260101_srb_old", repo_root=tmp_path, today="20260823")
    copied = dest / "old_grid.yaml"
    assert copied.is_file()
    data = yaml.safe_load(copied.read_text(encoding="utf-8"))
    assert data["experiment_id"] == "20260823_replay_srb_old"
    assert data["segment_matrix"]["variants"][0]["output_dir"] == (
        "results/ma_cross/experiments/20260823_replay_srb_old/baseline"
    )
    text = (dest / "DECISION.md").read_text(encoding="utf-8")
    assert "mlbot research run 20260823_replay_srb_old" in text


def test_segment_rows_without_decision_cite(tmp_path: Path) -> None:
    exp = tmp_path / "config" / "experiments" / "20260823_nocite"
    _decision(exp, strategy="bpc")
    bear = (
        tmp_path
        / "results"
        / "bpc"
        / "experiments"
        / "20260823_nocite"
        / "baseline"
        / "bear_2022"
    )
    bear.mkdir(parents=True)
    (bear / "capital_report.json").write_text(
        '{"cagr": 0.11, "max_drawdown_pct": -0.1, "calmar": 1.1}\n',
        encoding="utf-8",
    )
    rows = find_segment_kpi_rows(exp, repo_root=tmp_path)
    assert len(rows) == 1
    assert rows[0]["segment"] == "bear_2022"
    assert rows[0]["cagr"] == 0.11
    card = harvest_scorecard(exp, repo_root=tmp_path)
    assert card["phase3_segments"][0]["cagr"] == 0.11
