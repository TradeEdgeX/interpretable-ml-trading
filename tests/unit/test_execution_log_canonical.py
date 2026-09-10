import pandas as pd

from src.time_series_model.diagnostics.execution_log import build_execution_log_record
from src.time_series_model.diagnostics.execution_log_canonical import (
    build_canonical_from_pipeline,
    build_stage_logs_from_pipeline,
)
from src.time_series_model.diagnostics.execution_log_aggregate import (
    aggregate_stage_logs,
)
from src.time_series_model.diagnostics.execution_log import ExecutionStageLogWriter


def test_build_execution_log_record_basic():
    rec = build_execution_log_record(
        source="pipeline",
        symbol="BTCUSDT",
        timestamp=pd.Timestamp("2024-01-01T00:00:00Z"),
    )
    assert rec["schema_version"] == "v1"
    assert rec["source"] == "pipeline"
    assert rec["symbol"] == "BTCUSDT"
    assert "timestamp" in rec


def test_build_canonical_from_pipeline():
    ts = pd.Timestamp("2024-01-01T00:00:00Z")
    preds_df = pd.DataFrame(
        {
            "symbol": ["BTCUSDT"],
            "timestamp": [ts],
            "pred_dir_prob": [0.6],
            "pred_mfe_atr": [1.0],
            "pred_mae_atr": [0.5],
            "pred_t_to_mfe": [10.0],
        }
    )
    mode_df = pd.DataFrame(
        {
            "symbol": ["BTCUSDT"],
            "timestamp": [ts],
            "mode": ["TREND"],
        }
    )
    logs_df = pd.DataFrame(
        {
            "symbol": ["BTCUSDT"],
            "timestamp": [ts],
            "head_dir_score": [0.2],
            "head_mfe_atr": [1.0],
            "head_mae_atr": [0.5],
            "head_t_to_mfe": [10.0],
            "ret_mean": [0.01],
            "ret_trend": [0.02],
            "drawdown": [0.1],
        }
    )

    records = build_canonical_from_pipeline(
        preds_df=preds_df,
        mode_df=mode_df,
        logs_df=logs_df,
        run_id="test_run",
        timeframe="240T",
    )
    assert len(records) == 1
    rec = records[0]
    assert rec["source"] == "pipeline"
    assert rec["router"]["mode"] == "TREND"
    assert rec["execution"]["intent"] is True
    assert rec["execution"]["submit_order"] is False
    assert rec["returns"]["ret_trend"] == 0.02


def test_build_stage_logs_and_aggregate(tmp_path):
    ts = pd.Timestamp("2024-01-01T00:00:00Z")
    preds_df = pd.DataFrame(
        {
            "symbol": ["BTCUSDT"],
            "timestamp": [ts],
            "pred_dir_prob": [0.6],
            "pred_mfe_atr": [1.0],
            "pred_mae_atr": [0.5],
            "pred_t_to_mfe": [10.0],
        }
    )
    mode_df = pd.DataFrame(
        {
            "symbol": ["BTCUSDT"],
            "timestamp": [ts],
            "mode": ["TREND"],
        }
    )
    logs_df = pd.DataFrame(
        {
            "symbol": ["BTCUSDT"],
            "timestamp": [ts],
            "head_dir_score": [0.2],
            "head_mfe_atr": [1.0],
            "head_mae_atr": [0.5],
            "head_t_to_mfe": [10.0],
            "ret_mean": [0.01],
            "ret_trend": [0.02],
            "drawdown": [0.1],
        }
    )
    records = build_stage_logs_from_pipeline(
        preds_df=preds_df,
        mode_df=mode_df,
        logs_df=logs_df,
        run_id="test_run",
        timeframe="240T",
        strategy_name="pipeline",
    )
    writers = {}
    for rec in records:
        stage = rec["stage"]
        writer = writers.get(stage)
        if writer is None:
            writer = ExecutionStageLogWriter(base_dir=tmp_path, stage=stage)
            writers[stage] = writer
        writer.write(rec, decision_ts_ns=int(rec["decision_ts_ns"]))

    canonical = aggregate_stage_logs(tmp_path)
    assert len(canonical) == 1
    rec = canonical[0]
    assert rec["router"]["mode"] == "TREND"
    assert rec["returns"]["ret_trend"] == 0.02
