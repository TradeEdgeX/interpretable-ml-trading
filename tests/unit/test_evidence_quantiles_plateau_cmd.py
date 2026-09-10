import json

import pandas as pd

from src.feature_store.feature_store import FeatureStore, FeatureStoreSpec


def test_evidence_quantiles_plateau_smoke(tmp_path):
    # Minimal feature store for one symbol
    store = FeatureStore(tmp_path)
    spec = FeatureStoreSpec(layer="test_layer", symbol="BTCUSDT", timeframe="240T")
    idx = pd.date_range("2024-01-01", periods=5, freq="4H")
    df = pd.DataFrame(
        {"vpin": [0.01, 0.05, 0.08, 0.10, 0.12], "cvd_change_5": [1, 2, 3, 4, 5]},
        index=idx,
    )
    store.write_month(
        spec,
        "2024-01",
        df,
        base_columns=[],
        feature_columns=list(df.columns),
        overwrite=True,
    )

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    # Import script function by executing as module
    from scripts.diagnose_evidence_quantiles_plateau import main as plateau_main

    # Run with basic args (no logs)
    args = [
        "--feature-store-root",
        str(tmp_path),
        "--layer",
        "test_layer",
        "--symbols",
        "BTCUSDT",
        "--timeframe",
        "240T",
        "--start-date",
        "2024-01-01",
        "--end-date",
        "2024-01-02",
        "--registry",
        "config/nnmultihead/execution_archetypes.yaml",
        "--archetype",
        "TrendContinuationTC",
        "--sweep-key",
        "vpin",
        "--q-grid",
        "0.5,0.6",
        "--out",
        str(out_dir),
    ]
    import sys

    old = sys.argv
    try:
        sys.argv = ["diagnose_evidence_quantiles_plateau.py"] + args
        plateau_main()
    finally:
        sys.argv = old

    assert (out_dir / "plateau.csv").exists()
    assert (out_dir / "report.md").exists()
    assert (out_dir / "summary.json").exists()


def test_evidence_quantiles_plateau_gate_selection(tmp_path):
    # Feature store + logs so gate can evaluate
    store = FeatureStore(tmp_path)
    spec = FeatureStoreSpec(layer="test_layer", symbol="BTCUSDT", timeframe="240T")
    idx = pd.date_range("2024-01-01", periods=6, freq="4H")
    df = pd.DataFrame(
        {
            "vpin": [0.01, 0.02, 0.03, 0.20, 0.21, 0.22],
            "cvd_change_5": [1, 2, 3, 4, 5, 6],
        },
        index=idx,
    )
    store.write_month(
        spec,
        "2024-01",
        df,
        base_columns=[],
        feature_columns=list(df.columns),
        overwrite=True,
    )

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

    gate_yaml = tmp_path / "gate.yaml"
    gate_yaml.write_text(
        "layer: execution_layer\nhard_fail:\n  router_diag__trade_rate: {min: 0.2, max: 1.0}\n  router_diag__trade_win_rate: {min: 0.4, max: 1.0}\n  router_diag__trade_avg_ret: {min: 0.0, max: 1.0}\n",
        encoding="utf-8",
    )

    out_dir = tmp_path / "out2"
    out_dir.mkdir()
    from scripts.diagnose_evidence_quantiles_plateau import main as plateau_main

    args = [
        "--feature-store-root",
        str(tmp_path),
        "--layer",
        "test_layer",
        "--symbols",
        "BTCUSDT",
        "--timeframe",
        "240T",
        "--start-date",
        "2024-01-01",
        "--end-date",
        "2024-01-02",
        "--registry",
        "config/nnmultihead/execution_archetypes.yaml",
        "--archetype",
        "TrendContinuationTC",
        "--sweep-key",
        "vpin",
        "--q-grid",
        "0.5,0.9",
        "--logs",
        str(logs_path),
        "--gate-yaml",
        str(gate_yaml),
        "--require-gate",
        "--out",
        str(out_dir),
    ]
    import sys

    old = sys.argv
    try:
        sys.argv = ["diagnose_evidence_quantiles_plateau.py"] + args
        plateau_main()
    finally:
        sys.argv = old

    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    plateau_qs = summary.get("plateau_qs") or []
    assert plateau_qs
    if len(plateau_qs) % 2 == 1:
        expected = float(sorted(plateau_qs)[len(plateau_qs) // 2])
    else:
        qs = sorted(plateau_qs)
        expected = 0.5 * (qs[len(qs) // 2 - 1] + qs[len(qs) // 2])
    assert summary["selected_q"] == expected
