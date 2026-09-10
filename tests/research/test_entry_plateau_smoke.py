"""Integration smoke: entry_rr plateau on real logs_gated.parquet when available."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from scripts.research.entry_plateau_scan import scan_entry_condition
from src.research.execution_kernel.entry_rr_scan import prepare_entry_rr_frame

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve_logs_gated() -> Path | None:
    env = os.environ.get("LOGS_GATED_PARQUET")
    if env:
        p = Path(env)
        if p.is_file():
            return p
    matches = sorted(
        PROJECT_ROOT.glob("results/train_final/srb/**/logs_gated.parquet"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return matches[0] if matches else None


@pytest.mark.integration
def test_entry_rr_smoke_on_logs_gated() -> None:
    pq = _resolve_logs_gated()
    if pq is None:
        pytest.skip(
            "no logs_gated.parquet (set LOGS_GATED_PARQUET or run SRB train_final)"
        )
    import pandas as pd

    df = pd.read_parquet(pq)
    if "srb_sr_success_breakout_score" not in df.columns:
        pytest.skip("SRB entry feature column missing in parquet")
    prepared = prepare_entry_rr_frame(df, "srb")
    fdef = {
        "conditions": [
            {
                "feature": "srb_sr_success_breakout_score",
                "operator": ">=",
                "value": 0.12,
            }
        ],
    }
    result = scan_entry_condition(
        prepared,
        "srb",
        fdef,
        {
            "index": 0,
            "feature": "srb_sr_success_breakout_score",
            "operator": ">=",
            "value": 0.12,
        },
        snotio_mode="entry_rr",
        steps=5,
        min_trades=3,
    )
    payload = result["payload"]
    assert payload.get("snotio_mode") == "entry_rr"
    assert payload.get("rows")
    assert any(not r.get("too_few") for r in payload["rows"])
