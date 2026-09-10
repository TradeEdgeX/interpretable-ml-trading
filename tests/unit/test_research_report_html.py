"""report.html 与 research run 的 report.json 同步生成."""

from __future__ import annotations

from pathlib import Path

import scripts.auto_research_pipeline as arp


def test_write_research_run_report_html_writes_sibling(tmp_path: Path) -> None:
    run_dir = tmp_path / "bpc" / "20260108_120000"
    run_dir.mkdir(parents=True)
    report_path = run_dir / "report.json"
    report_path.write_text(
        """{
  "version": 2,
  "strategy": "bpc",
  "timestamp": "20260108_120000",
  "data_range": {"start_date": "2022-01-01", "end_date": "2026-01-01"},
  "backtest_metrics": {"total_trades": 100, "sharpe_per_trade": 0.12},
  "comparison": {"decision": "ADOPT", "reasons": ["ok"]},
  "thresholds": {},
  "artifacts": {"evidence_dir": "results/x"}
}""",
        encoding="utf-8",
    )

    arp.write_research_run_report_html(report_path)

    html_path = run_dir / "report.html"
    assert html_path.is_file()
    text = html_path.read_text(encoding="utf-8")
    assert "Research · bpc · 20260108_120000" in text
    assert "compare_runs" in text
    assert "ADOPT" in text


def test_write_research_run_report_html_skips_other_json(tmp_path: Path) -> None:
    p = tmp_path / "other.json"
    p.write_text('{"strategy": "x", "timestamp": "t"}', encoding="utf-8")
    arp.write_research_run_report_html(p)
    assert not (tmp_path / "report.html").exists()
