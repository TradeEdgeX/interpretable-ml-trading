import json
import sys

import pandas as pd


def test_archetype_trade_counts_smoke(tmp_path):
    idx = pd.date_range("2024-01-01", periods=6, freq="4H")
    mode = pd.DataFrame(
        {
            "timestamp": idx,
            "symbol": ["BTCUSDT"] * len(idx),
            "mode": ["NO_TRADE", "MEAN", "MEAN", "NO_TRADE", "MEAN", "NO_TRADE"],
            "gate_decision": [
                "no_trade",
                "allow",
                "allow",
                "no_trade",
                "allow",
                "no_trade",
            ],
            "gate_archetype": [
                "",
                "FailureReversionFR",
                "FailureReversionFR",
                "",
                "ExhaustionTurnET",
                "",
            ],
        }
    )
    mode_path = tmp_path / "mode.parquet"
    mode.to_parquet(mode_path)

    out_dir = tmp_path / "out"
    out_dir.mkdir()

    from scripts.diagnose_archetype_trade_counts import main as counts_main

    args = [
        "--mode",
        str(mode_path),
        "--out",
        str(out_dir),
    ]

    old = sys.argv
    try:
        sys.argv = ["diagnose_archetype_trade_counts.py"] + args
        counts_main()
    finally:
        sys.argv = old

    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert (out_dir / "entries.csv").exists()
    assert summary["total_entries"] == 2
    assert summary["by_archetype"]["FailureReversionFR"] == 1
    assert summary["by_archetype"]["ExhaustionTurnET"] == 1
