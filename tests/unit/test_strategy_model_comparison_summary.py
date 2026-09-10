import json
from pathlib import Path

from src.time_series_model.diagnostics.strategy_model_comparison import (
    _summarize_results,
)


def test_summarize_results_extracts_backtest_and_diagnostics(tmp_path: Path):
    results = {
        "cv": {"Sharpe_mean": 1.23},
        "backtest": {
            "sharpe": 2.0,
            "total_return_pct": 10.0,
            "max_drawdown_pct": -5.0,
            "total_trades": 42,
            "diagnostics": {"entries_exits": {"total_entries": 40}},
        },
        "diagnostics": {
            "labels": {"value_counts": {"0": 100, "1": 50}},
            "predictions": {"mean": 0.52},
        },
    }
    p = tmp_path / "results.json"
    p.write_text(json.dumps(results), encoding="utf-8")

    s = _summarize_results("demo_strategy", results, p)
    assert s.strategy == "demo_strategy"
    assert s.cv_score == 1.23
    assert s.sharpe == 2.0
    assert s.total_trades == 42
    assert s.entries_exits == {"total_entries": 40}
    assert isinstance(s.label_summary, dict)
    assert isinstance(s.pred_summary, dict)
