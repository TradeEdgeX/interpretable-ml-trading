import json
import sys

import pandas as pd

from src.feature_store.feature_store import FeatureStore, FeatureStoreSpec


def test_execution_gate_plateau_smoke(tmp_path):
    # Minimal feature store for one symbol
    store = FeatureStore(tmp_path)
    spec = FeatureStoreSpec(layer="test_layer", symbol="BTCUSDT", timeframe="240T")
    idx = pd.date_range("2024-01-01", periods=6, freq="4H")
    df = pd.DataFrame(
        {
            "vpin": [0.01, 0.02, 0.03, 0.20, 0.21, 0.22],
            "volume_ratio": [1, 2, 3, 4, 5, 6],
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

    mode = pd.DataFrame(
        {
            "timestamp": idx,
            "symbol": ["BTCUSDT"] * len(idx),
            "mode": ["TREND"] * len(idx),
        }
    )
    mode_path = tmp_path / "mode.parquet"
    mode.to_parquet(mode_path)

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

    registry = tmp_path / "registry.yaml"
    registry.write_text(
        "\n".join(
            [
                "version: 1",
                "name: test_registry",
                "regimes:",
                "  TREND:",
                "    archetypes:",
                "      TrendContinuationTC:",
                "        required_conditions: []",
                "        required_evidence: []",
                "        evidence_rules: []",
                "        gate_rules:",
                "          rules:",
                "            - name: g1",
                "              kind: quantile_gt",
                "              key: vpin",
                "              quantile: 0.5",
                "              on_missing: false",
                "            - name: g2",
                "              kind: quantile_gt",
                "              key: volume_ratio",
                "              quantile: 0.5",
                "              on_missing: false",
                "          deny_if: []",
                "          allow_if: [g1, g2]",
                "          allow_mode: min2",
                "          default_action: deny",
            ]
        ),
        encoding="utf-8",
    )

    # Live config from YAML (no database)
    live_config_yaml = tmp_path / "live_config.yaml"
    live_config_yaml.write_text(
        "enabled_archetypes:\n  - TrendContinuationTC\nsize_multipliers: {}\nwindow_minutes: 10\nmin_order_interval_minutes: 10\nnnmultihead_inference: {}\n",
        encoding="utf-8",
    )

    out_dir = tmp_path / "out"
    out_dir.mkdir()

    from scripts.diagnose_execution_gate_plateau import main as plateau_main

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
        "--mode",
        str(mode_path),
        "--logs",
        str(logs_path),
        "--registry",
        str(registry),
        "--live-config",
        str(live_config_yaml),
        "--sweep-key",
        "vpin",
        "--q-grid",
        "0.5,0.9",
        "--quantiles",
        "0.5,0.9",
        "--out",
        str(out_dir),
    ]

    old = sys.argv
    try:
        sys.argv = ["diagnose_execution_gate_plateau.py"] + args
        plateau_main()
    finally:
        sys.argv = old

    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert (out_dir / "plateau.csv").exists()
    assert (out_dir / "report.md").exists()
    assert summary["selected_q"] in [0.5, 0.9]
