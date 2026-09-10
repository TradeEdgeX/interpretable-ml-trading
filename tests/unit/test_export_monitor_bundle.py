"""Unit tests for monitor_bundle export / promote."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.monitoring.export_monitor_bundle import (
    compute_rules_hash,
    export_monitor_bundle,
    infer_schema,
    promote_monitor_bundle,
)
from tests.unit.test_regime_health import TPC_LABELED_REGIME

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _synthetic_tpc_parquet(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "adx_50": rng.uniform(10, 35, n),
            "ema_1200_position": rng.uniform(-0.2, 0.3, n),
            "vol_persistence": rng.uniform(0, 1, n),
            "vol_leverage_asymmetry": rng.uniform(-1, 1, n),
            "forward_rr": rng.normal(0, 1, n),
        }
    )


def test_compute_rules_hash_stable():
    h1 = compute_rules_hash(TPC_LABELED_REGIME)
    h2 = compute_rules_hash(dict(TPC_LABELED_REGIME))
    assert h1 == h2
    assert h1 and len(h1) == 40


def test_infer_schema_labeled():
    assert infer_schema(TPC_LABELED_REGIME) == "labeled"


def test_export_monitor_bundle_draft(tmp_path, monkeypatch):
    pq = tmp_path / "features.parquet"
    _synthetic_tpc_parquet().to_parquet(pq, index=False)
    out = tmp_path / "monitor_bundle"

    regime_dir = tmp_path / "strategies" / "tpc" / "archetypes"
    regime_dir.mkdir(parents=True)
    import yaml

    regime_dir.joinpath("regime.yaml").write_text(
        yaml.dump(TPC_LABELED_REGIME, allow_unicode=True),
        encoding="utf-8",
    )

    # Skip heavy smoke (watchdog imports gate yaml)
    monkeypatch.setattr(
        "scripts.monitoring.export_monitor_bundle.run_bundle_smoke",
        lambda **kw: {"drift_exit": 0, "watchdog_exit": 0, "skipped": True},
    )

    result = export_monitor_bundle(
        strategy="tpc",
        layer="regime",
        parquet=pq,
        out_dir=out,
        strategies_root=tmp_path / "strategies",
        run_smoke=True,
    )
    bundle = result["bundle"]
    assert bundle["schema"] == "labeled"
    assert "regime_shares" in bundle["regime"]
    assert bundle["regime"]["rules_hash"]
    assert (out / "bundle.json").is_file()
    assert (out / "reference" / "tpc_psi_ref.parquet").is_file()


def test_promote_monitor_bundle_dry_run(tmp_path):
    bundle = {
        "version": 1,
        "strategy": "tpc",
        "layer": "regime",
        "schema": "labeled",
        "calibration": {"parquet": "results/test.parquet", "n_rows": 100},
        "regime": {
            "regime_shares": {"bull": 0.1, "bear": 0.4, "neutral": 0.5},
            "rules_hash": "abc",
        },
        "psi": {
            "features": ["ema_1200_position"],
            "reference_parquet": "monitor_bundle/reference/tpc_psi_ref.parquet",
        },
    }
    mb = tmp_path / "monitor_bundle"
    ref = mb / "reference"
    ref.mkdir(parents=True)
    _synthetic_tpc_parquet(50).to_parquet(ref / "tpc_psi_ref.parquet", index=False)

    result = promote_monitor_bundle(
        bundle,
        dry_run=True,
        bundle_dir=mb,
    )
    assert result["dry_run"] is True
    assert any("regime_watchdog_baseline.json" in a for a in result["actions"])
