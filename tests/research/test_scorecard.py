from __future__ import annotations

import json
from pathlib import Path

from src.research.experiment_gate import find_kpi_artifacts
from src.research.scorecard import build_scorecard_census, harvest_scorecard


def test_empty_slots_when_no_artifacts(tmp_path: Path) -> None:
    exp = tmp_path / "20260101_empty"
    exp.mkdir()
    (exp / "DECISION.md").write_text("# x\n", encoding="utf-8")
    card = harvest_scorecard(exp, repo_root=tmp_path)
    assert card["phase1"]["ic"] is None
    assert card["phase3"]["cagr"] is None
    assert card["can_close"] is False


def test_harvests_phase1_and_phase3_json(tmp_path: Path) -> None:
    exp = tmp_path / "20260101_has"
    scan = exp / "quick_scan"
    scan.mkdir(parents=True)
    (scan / "ic.json").write_text(
        json.dumps({"ic": 0.12, "icir": 0.4, "lift_pp": 3.2}), encoding="utf-8"
    )
    (exp / "capital_report.json").write_text(
        json.dumps(
            {
                "cagr": 0.18,
                "calmar": 0.6,
                "win_rate": 0.52,
                "max_drawdown_pct": -0.22,
                "sharpe_r": 0.7,
            }
        ),
        encoding="utf-8",
    )
    card = harvest_scorecard(exp, repo_root=tmp_path)
    assert card["phase1"]["ic"] == 0.12
    assert card["phase1"]["auc"] is None
    assert card["phase3"]["cagr"] == 0.18
    assert card["phase3"]["maxdd"] == -0.22
    assert card["can_close"] is True


def test_declared_kpi_fills_empty_phase3(tmp_path: Path) -> None:
    exp = tmp_path / "20260101_declared"
    exp.mkdir()
    card = harvest_scorecard(
        exp, repo_root=tmp_path, declared_kpi={"cagr": 0.1, "maxdd": -0.2}
    )
    assert card["phase3"]["cagr"] == 0.1
    assert card["phase3"]["maxdd"] == -0.2
    assert card["can_close"] is False


def test_phase3_merges_trades_csv_and_derives_calmar(tmp_path: Path) -> None:
    """capital_report only has CAGR/MaxDD; WR/Sharpe come from event_trades."""
    exp = tmp_path / "20260101_merge"
    results = tmp_path / "results" / "bpc" / "seg"
    results.mkdir(parents=True)
    exp.mkdir()
    (results / "capital_report.json").write_text(
        json.dumps({"cagr": -0.045, "max_drawdown_pct": -0.12}),
        encoding="utf-8",
    )
    (results / "event_trades_bpc.csv").write_text(
        "pnl_r\n-1.0\n1.0\n1.0\n",
        encoding="utf-8",
    )
    (exp / "DECISION.md").write_text(
        "产物：`results/bpc/seg/capital_report.json`\n",
        encoding="utf-8",
    )
    card = harvest_scorecard(exp, repo_root=tmp_path)
    assert card["phase3"]["cagr"] == -0.045
    assert card["phase3"]["maxdd"] == -0.12
    assert card["phase3"]["calmar"] == -0.045 / 0.12
    assert card["phase3"]["win_rate"] == 2 / 3
    assert card["phase3"]["sharpe"] is not None
    assert card["can_close"] is True


def test_harvests_phase3_by_canonical_segment(tmp_path: Path) -> None:
    exp = tmp_path / "20260101_segs"
    results = tmp_path / "results" / "bpc" / "experiments" / "20260101_segs"
    exp.mkdir()
    for variant, segs in (
        (
            "baseline",
            {
                "bear_2022": {
                    "cagr": -0.04,
                    "max_drawdown_pct": -0.12,
                    "win_rate": 0.29,
                    "sharpe_r": -0.22,
                    "calmar": -0.33,
                },
                "bull_2023_2024": {
                    "cagr": -0.02,
                    "max_drawdown_pct": -0.10,
                    "win_rate": 0.40,
                    "sharpe_r": -0.07,
                    "calmar": -0.20,
                },
                "recent_range_to_bear": {
                    "cagr": 0.06,
                    "max_drawdown_pct": -0.04,
                    "win_rate": 0.56,
                    "sharpe_r": 0.37,
                    "calmar": 1.50,
                },
            },
        ),
        (
            "v2",
            {
                "bear_2022": {
                    "cagr": -0.01,
                    "max_drawdown_pct": -0.05,
                    "win_rate": 0.36,
                    "sharpe_r": -0.05,
                    "calmar": -0.20,
                },
                "bull_2023_2024": {
                    "cagr": 0.00,
                    "max_drawdown_pct": -0.06,
                    "win_rate": 0.41,
                    "sharpe_r": 0.02,
                    "calmar": 0.00,
                },
                "recent_range_to_bear": {
                    "cagr": -0.02,
                    "max_drawdown_pct": -0.05,
                    "win_rate": 0.39,
                    "sharpe_r": -0.33,
                    "calmar": -0.40,
                },
            },
        ),
    ):
        for seg, payload in segs.items():
            d = results / variant / seg
            d.mkdir(parents=True)
            (d / "capital_report.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )
    (exp / "DECISION.md").write_text(
        "产物：`results/bpc/experiments/20260101_segs/baseline/bear_2022/capital_report.json`\n",
        encoding="utf-8",
    )
    card = harvest_scorecard(exp, repo_root=tmp_path)
    rows = card["phase3_segments"]
    assert [r["variant"] for r in rows] == ["baseline"] * 3 + ["v2"] * 3
    assert [r["segment"] for r in rows[:3]] == [
        "bear_2022",
        "bull_2023_2024",
        "recent_range_to_bear",
    ]
    assert rows[0]["cagr"] == -0.04
    assert rows[5]["sharpe"] == -0.33


def test_segment_harvest_does_not_mix_sibling_experiments(tmp_path: Path) -> None:
    exp = tmp_path / "20260101_keep"
    other = tmp_path / "results" / "bpc" / "experiments" / "20260102_other"
    mine = tmp_path / "results" / "bpc" / "experiments" / "20260101_keep"
    exp.mkdir()
    for root, cagr in ((mine, 0.11), (other, 0.99)):
        d = root / "bear_2022"
        d.mkdir(parents=True)
        (d / "capital_report.json").write_text(
            json.dumps({"cagr": cagr, "max_drawdown_pct": -0.1, "calmar": 1.1}),
            encoding="utf-8",
        )
        d2 = root / "bull_2023_2024"
        d2.mkdir()
        (d2 / "capital_report.json").write_text(
            json.dumps({"cagr": cagr, "max_drawdown_pct": -0.1, "calmar": 1.1}),
            encoding="utf-8",
        )
    (exp / "DECISION.md").write_text(
        "产物：`results/bpc/experiments/20260101_keep/bear_2022/capital_report.json`\n",
        encoding="utf-8",
    )
    card = harvest_scorecard(exp, repo_root=tmp_path)
    assert [r["cagr"] for r in card["phase3_segments"]] == [0.11, 0.11]


def test_variants_dir_is_not_harvested(tmp_path: Path) -> None:
    exp = tmp_path / "20260101_big"
    nested = exp / "variants" / "a" / "b"
    nested.mkdir(parents=True)
    (nested / "capital_report.json").write_text(
        json.dumps({"cagr": 0.99}), encoding="utf-8"
    )
    files, kpi = find_kpi_artifacts(exp, repo_root=tmp_path)
    assert files == []
    assert kpi == {}


def test_census_counts_empty_and_filled(tmp_path: Path) -> None:
    root = tmp_path / "config" / "experiments"
    empty = root / "20260101_empty"
    filled = root / "20260102_filled"
    empty.mkdir(parents=True)
    filled.mkdir()
    (empty / "DECISION.md").write_text("# empty\n", encoding="utf-8")
    (filled / "DECISION.md").write_text("# filled\n", encoding="utf-8")
    (filled / "capital_report.json").write_text(
        json.dumps(
            {
                "cagr": 0.1,
                "calmar": 0.2,
                "win_rate": 0.5,
                "max_drawdown_pct": -0.1,
                "sharpe_r": 0.3,
            }
        ),
        encoding="utf-8",
    )
    census = build_scorecard_census(repo_root=tmp_path, experiments_root=root)
    assert census["count"] == 2
    assert census["buckets"]["empty"] == 1
    assert census["buckets"]["phase3_complete"] == 1
