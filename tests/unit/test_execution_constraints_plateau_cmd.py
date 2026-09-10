import json
import sys

import pandas as pd


def test_execution_constraints_plateau_smoke(tmp_path):
    idx = pd.date_range("2024-01-01", periods=6, freq="4H")
    logs = pd.DataFrame(
        {
            "timestamp": idx,
            "symbol": ["BTCUSDT"] * len(idx),
            "mode": ["TREND"] * len(idx),
            "ret_mean": [0.0] * len(idx),
            "ret_trend": [0.01] * len(idx),
        }
    )
    logs_path = tmp_path / "logs.parquet"
    logs.to_parquet(logs_path)

    out_dir = tmp_path / "out"
    out_dir.mkdir()

    from scripts.diagnose_execution_constraints_plateau import main as plateau_main

    args = [
        "--logs",
        str(logs_path),
        "--min-interval-grid",
        "0,480",
        "--out",
        str(out_dir),
    ]

    old = sys.argv
    try:
        sys.argv = ["diagnose_execution_constraints_plateau.py"] + args
        plateau_main()
    finally:
        sys.argv = old

    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert (out_dir / "plateau.csv").exists()
    assert (out_dir / "report.md").exists()
    assert summary["selected_min_order_interval_minutes"] in [0, 480]
