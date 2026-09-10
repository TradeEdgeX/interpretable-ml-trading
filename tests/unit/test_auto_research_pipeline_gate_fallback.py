from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from scripts.auto_research_pipeline import (
    _materialize_gate_fallback_predictions,
    _select_gate_fallback_columns,
)


def _write_src_parquet(path: Path) -> None:
    table = pa.table(
        {
            "timestamp": ["2026-05-01", "2026-05-02"],
            "open": [1.0, 2.0],
            "high": [1.1, 2.1],
            "low": [0.9, 1.9],
            "close": [1.0, 2.0],
            "volume": [10.0, 20.0],
            "forward_rr": [0.1, -0.2],
            "success_no_rr_extreme": [1, 0],
            "me_semantic_chop_ts_q": [0.4, 0.1],
            "unused_feature_zzz": [99, 88],
        }
    )
    pq.write_table(table, path)


def _write_gate_yaml(path: Path) -> None:
    payload = {
        "hard_gates": [
            {
                "id": "g1",
                "when": {"me_semantic_chop_ts_q": {"value_gte": 0.35}},
                "then": {"action": "deny"},
            }
        ]
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def test_select_gate_fallback_columns_includes_gate_features(tmp_path: Path) -> None:
    src = tmp_path / "features_labeled.parquet"
    gate = tmp_path / "gate_draft.yaml"
    _write_src_parquet(src)
    _write_gate_yaml(gate)
    keep = _select_gate_fallback_columns(src, gate)
    assert "timestamp" in keep
    assert "success_no_rr_extreme" in keep
    assert "me_semantic_chop_ts_q" in keep
    assert "unused_feature_zzz" not in keep


def test_materialize_gate_fallback_predictions_creates_readable_file(
    tmp_path: Path,
) -> None:
    src = tmp_path / "features_labeled.parquet"
    dst = tmp_path / "predictions.parquet"
    gate = tmp_path / "gate_draft.yaml"
    _write_src_parquet(src)
    _write_gate_yaml(gate)
    mode = _materialize_gate_fallback_predictions(
        fallback_src=src,
        gate_pred=dst,
        gate_path=gate,
    )
    assert dst.exists()
    assert mode in {"hardlink", "symlink", "compact_copy[8 cols]", "full_copy"} or (
        mode.startswith("compact_copy[")
    )
    cols = pq.read_table(dst).column_names
    assert "timestamp" in cols
