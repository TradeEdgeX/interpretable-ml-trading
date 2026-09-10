"""Program court: agents do not write verdict."""

from __future__ import annotations

import json
from pathlib import Path

from src.research.experiment_gate import allowed_declare, evaluate_experiment
from src.research.experiment_index import ExperimentMeta
from src.research.experiment_gate import evaluate_court


def test_wrong_harness_is_court_fail() -> None:
    meta = ExperimentMeta(
        strategy="ma_cross",
        harness="phase1_scan_only",
        segments=("bear_2022", "bull_2023_2024", "recent_range_to_bear"),
        kill_switch=False,
    )
    status, reasons = evaluate_court(meta)
    assert status == "court_fail"
    assert any("does not match" in r for r in reasons)


def test_missing_segments_is_artifacts_missing() -> None:
    meta = ExperimentMeta(
        strategy="ma_cross",
        harness="event_backtest",
        segments=(),
        kill_switch=False,
    )
    status, _ = evaluate_court(meta)
    assert status == "artifacts_missing"


def test_promote_requires_court_ok_and_yes(tmp_path: Path) -> None:
    exp = tmp_path / "20260823_demo"
    exp.mkdir()
    (exp / "DECISION.md").write_text(
        "---\nstrategy: ma_cross\nharness: event_backtest\n"
        "segments: [bear_2022, bull_2023_2024, recent_range_to_bear]\n"
        "kill_switch: false\nverdict:\n---\n",
        encoding="utf-8",
    )
    gate = evaluate_experiment(exp, repo_root=tmp_path)
    assert gate.gate_status == "artifacts_missing"
    assert allowed_declare(gate, "promote", yes=True)
    assert allowed_declare(gate, "reject", yes=False) is None


def test_close_cli_refuses_missing_experiment() -> None:
    from scripts.research.close import main

    assert main(["no_such_exp", "--declare", "promote", "--yes"]) == 2


def test_spot_daily_mtm_kpis_take_precedence_over_realized_report(
    tmp_path: Path,
) -> None:
    exp = tmp_path / "20260903_spot_mtm"
    result_dir = exp / "quick_scan" / "h0"
    result_dir.mkdir(parents=True)
    (exp / "DECISION.md").write_text(
        "---\nstrategy: spot_accum_simple\nharness: spot_accum\n"
        "segments: [bear_2022, bull_2023_2024, recent_range_to_bear]\n"
        "kill_switch: false\nverdict:\n---\n",
        encoding="utf-8",
    )
    (result_dir / "capital_report.json").write_text(
        json.dumps({"cagr": 0.34, "max_drawdown_pct": -0.03}),
        encoding="utf-8",
    )
    (result_dir / "spot_daily_mtm_report.json").write_text(
        json.dumps(
            {
                "cagr": 0.34,
                "calmar": 1.7,
                "win_rate": 0.49,
                "max_drawdown_pct": -0.20,
                "sharpe_daily_annualized": 1.38,
            }
        ),
        encoding="utf-8",
    )

    gate = evaluate_experiment(exp, repo_root=tmp_path)

    assert gate.gate_status == "court_ok"
    assert gate.kpi["maxdd"] == -0.20
    assert gate.kpi["sharpe"] == 1.38
