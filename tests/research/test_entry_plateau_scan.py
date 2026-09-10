"""Tests for entry_plateau_scan shared module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from scripts.research.entry_plateau_scan import (
    generate_scan_range,
    list_entry_plateau_jobs,
    run_entry_plateau_batch,
    scan_entry_condition,
)
from src.research.execution_kernel.entry_rr_scan import prepare_entry_rr_frame


def _synthetic_logs(n: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    close = 100 + np.cumsum(rng.normal(0, 0.5, size=n))
    feat = rng.normal(size=n)
    direction = np.where(feat > 0, 1.0, -1.0)
    return pd.DataFrame(
        {
            "symbol": ["BTC"] * n,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "atr": np.full(n, 1.0),
            "entry_direction": direction,
            "gate_decision": ["allow"] * n,
            "pulse_z": feat,
        }
    )


def test_generate_scan_range_normalized():
    grid = generate_scan_range(0.5, ">=", n_steps=5)
    assert len(grid) == 5
    assert min(grid) >= 0.0
    assert max(grid) <= 1.0


def test_list_entry_plateau_jobs_from_fixture(tmp_path: Path) -> None:
    arch = tmp_path / "strategies" / "tpc" / "archetypes"
    arch.mkdir(parents=True)
    (arch / "entry_filters.yaml").write_text(
        yaml.dump(
            {
                "filters": [
                    {
                        "id": "f1",
                        "enabled": True,
                        "conditions": [
                            {
                                "feature": "pulse_z",
                                "operator": "<=",
                                "value": 0.0,
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    jobs = list_entry_plateau_jobs("tpc", strategies_root=str(tmp_path / "strategies"))
    assert len(jobs) == 1
    assert jobs[0]["filter_id"] == "f1"


def test_scan_entry_condition_runs(tmp_path: Path) -> None:
    arch = tmp_path / "strategies" / "srb" / "archetypes"
    arch.mkdir(parents=True)
    (arch / "entry_filters.yaml").write_text(
        yaml.dump(
            {
                "filters": [
                    {
                        "id": "test_filter",
                        "enabled": True,
                        "conditions": [
                            {
                                "feature": "pulse_z",
                                "operator": "<=",
                                "value": 0.0,
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    df = _synthetic_logs(200)
    prepared = prepare_entry_rr_frame(
        df, "srb", strategies_root=str(tmp_path / "strategies")
    )
    fdef = {
        "conditions": [{"feature": "pulse_z", "operator": "<=", "value": 0.0}],
    }
    result = scan_entry_condition(
        prepared,
        "srb",
        fdef,
        {"index": 0, "feature": "pulse_z", "operator": "<=", "value": 0.0},
        snotio_mode="entry_rr",
        steps=5,
        min_trades=5,
        strategies_root=str(tmp_path / "strategies"),
        simple_execution=True,
    )
    assert result["payload"]["snotio_mode"] == "entry_rr"
    assert len(result["scan_results"]) == 5


def test_run_entry_plateau_batch_writes_summary(tmp_path: Path) -> None:
    arch = tmp_path / "strategies" / "srb" / "archetypes"
    arch.mkdir(parents=True)
    (arch / "entry_filters.yaml").write_text(
        yaml.dump(
            {
                "filters": [
                    {
                        "id": "test_filter",
                        "enabled": True,
                        "locked": True,
                        "conditions": [
                            {
                                "feature": "pulse_z",
                                "operator": "<=",
                                "value": 0.0,
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    pq = tmp_path / "logs.parquet"
    _synthetic_logs(200).to_parquet(pq)
    out = tmp_path / "out"
    summary = run_entry_plateau_batch(
        pq,
        "srb",
        steps=5,
        min_trades=5,
        strategies_root=str(tmp_path / "strategies"),
        simple_execution=True,
        out_dir=out,
    )
    assert summary["snotio_mode"] == "entry_rr"
    assert (out / "entry_plateau_summary.json").is_file()
    assert summary["all_results"]["test_filter"]["scanned_conditions"]
