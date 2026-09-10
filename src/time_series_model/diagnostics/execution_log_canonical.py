from __future__ import annotations

from typing import Any, Dict, List, Optional
from pathlib import Path
import pandas as pd

from src.time_series_model.diagnostics.execution_log import (
    build_execution_log_record,
    build_decision_id,
    build_stage_record,
)


def _read_any(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _collect_files(p: Path, *, prefix: str) -> List[Path]:
    if p.is_dir():
        files = sorted(p.glob(f"{prefix}_*.parquet"))
        if not files:
            files = sorted(p.glob("*.parquet"))
        if not files:
            files = sorted(p.glob("*.csv"))
        return files
    return [p]


def _load_multi(p: Path, *, prefix: str) -> pd.DataFrame:
    parts = []
    for f in _collect_files(p, prefix=prefix):
        df = _read_any(f)
        if "timestamp" not in df.columns and isinstance(df.index, pd.DatetimeIndex):
            df = df.copy()
            df["timestamp"] = pd.to_datetime(
                df.index, utc=True, errors="coerce"
            ).tz_convert(None)
            df = df.reset_index(drop=True)
        if "symbol" not in df.columns:
            df = df.copy()
            df["symbol"] = f.stem.replace(f"{prefix}_", "")
        parts.append(df)
    return pd.concat(parts, axis=0, ignore_index=False) if parts else pd.DataFrame()


def _normalize_timestamp(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if df is None or df.empty or "timestamp" not in df.columns:
        return df
    out = df.copy()
    ts = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out["timestamp"] = ts.dt.tz_convert(None)
    return out


def _normalize_symbol(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if df is None or df.empty or "symbol" not in df.columns:
        return df
    out = df.copy()
    out["symbol"] = out["symbol"].astype(str)
    return out


def build_canonical_from_pipeline(
    *,
    preds_df: pd.DataFrame,
    mode_df: Optional[pd.DataFrame],
    logs_df: Optional[pd.DataFrame],
    run_id: Optional[str] = None,
    timeframe: Optional[str] = None,
) -> List[Dict[str, Any]]:
    if preds_df.empty:
        return []
    df = _normalize_symbol(_normalize_timestamp(preds_df)).copy()
    if mode_df is not None and not mode_df.empty:
        mode_df = _normalize_symbol(_normalize_timestamp(mode_df))
        df = df.merge(
            mode_df[["symbol", "timestamp", "mode"]],
            on=["symbol", "timestamp"],
            how="left",
        )
    if logs_df is not None and not logs_df.empty:
        logs_df = _normalize_symbol(_normalize_timestamp(logs_df))
        cols = [
            c
            for c in [
                "symbol",
                "timestamp",
                "head_dir_score",
                "head_mfe_atr",
                "head_mae_atr",
                "head_t_to_mfe",
                "ret_mean",
                "ret_trend",
                "drawdown",
            ]
            if c in logs_df.columns
        ]
        df = df.merge(logs_df[cols], on=["symbol", "timestamp"], how="left")

    records: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        router_mode = row.get("mode")
        execution_intent = (
            str(router_mode).upper() != "NO_TRADE" if router_mode is not None else False
        )
        record = build_execution_log_record(
            source="pipeline",
            run_id=run_id,
            symbol=row.get("symbol"),
            timestamp=row.get("timestamp"),
            timeframe=timeframe,
            preds={
                k: row.get(k)
                for k in [
                    "pred_dir_prob",
                    "pred_mfe_atr",
                    "pred_mae_atr",
                    "pred_t_to_mfe",
                ]
                if k in row
            },
            router={
                "mode": str(router_mode).upper() if router_mode is not None else None,
                "thresholds": None,
                "scores": {
                    "head_dir_score": row.get("head_dir_score"),
                    "head_mfe_atr": row.get("head_mfe_atr"),
                    "head_mae_atr": row.get("head_mae_atr"),
                    "head_t_to_mfe": row.get("head_t_to_mfe"),
                },
            },
            gate={"blocked": False, "decisions": []},
            evidence=None,
            execution={
                "intent": bool(execution_intent),
                "submit_order": False,
                "side": None,
                "qty": None,
                "price": None,
                "reason": "pipeline_offline",
            },
            returns={
                "ret_mean": row.get("ret_mean"),
                "ret_trend": row.get("ret_trend"),
                "drawdown": row.get("drawdown"),
            },
        )
        records.append(record)
    return records


def build_stage_logs_from_pipeline(
    *,
    preds_df: pd.DataFrame,
    mode_df: Optional[pd.DataFrame],
    logs_df: Optional[pd.DataFrame],
    gated_df: Optional[pd.DataFrame] = None,
    run_id: Optional[str] = None,
    timeframe: Optional[str] = None,
    strategy_name: str = "pipeline",
) -> List[Dict[str, Any]]:
    if preds_df.empty:
        return []
    df = _normalize_symbol(_normalize_timestamp(preds_df)).copy()
    if mode_df is not None and not mode_df.empty:
        mode_df = _normalize_symbol(_normalize_timestamp(mode_df))
        df = df.merge(
            mode_df[["symbol", "timestamp", "mode"]],
            on=["symbol", "timestamp"],
            how="left",
        )
    if logs_df is not None and not logs_df.empty:
        logs_df = _normalize_symbol(_normalize_timestamp(logs_df))
        cols = [
            c
            for c in [
                "symbol",
                "timestamp",
                "head_dir_score",
                "head_mfe_atr",
                "head_mae_atr",
                "head_t_to_mfe",
                "ret_mean",
                "ret_trend",
                "drawdown",
            ]
            if c in logs_df.columns
        ]
        df = df.merge(logs_df[cols], on=["symbol", "timestamp"], how="left")

    # Merge gated logs if provided
    if gated_df is not None and not gated_df.empty:
        gated_df = _normalize_symbol(_normalize_timestamp(gated_df))
        gate_cols = [
            c
            for c in [
                "symbol",
                "timestamp",
                "gate_ok",
                "gate_decision",
                "gate_reasons",
                "gate_archetype",
            ]
            if c in gated_df.columns
        ]
        if gate_cols:
            df = df.merge(gated_df[gate_cols], on=["symbol", "timestamp"], how="left")

    records: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        ts = pd.to_datetime(row.get("timestamp"), utc=True, errors="coerce")
        decision_ts_ns = int(ts.value) if not pd.isna(ts) else 0
        decision_id = build_decision_id(
            strategy_name=str(strategy_name),
            symbol=str(row.get("symbol")),
            decision_ts_ns=decision_ts_ns,
        )
        preds = {
            k: row.get(k)
            for k in [
                "pred_dir_prob",
                "pred_mfe_atr",
                "pred_mae_atr",
                "pred_t_to_mfe",
            ]
            if k in row
        }
        router = {
            "mode": (
                str(row.get("mode")).upper() if row.get("mode") is not None else None
            ),
            "thresholds": None,
            "scores": {
                "head_dir_score": row.get("head_dir_score"),
                "head_mfe_atr": row.get("head_mfe_atr"),
                "head_mae_atr": row.get("head_mae_atr"),
                "head_t_to_mfe": row.get("head_t_to_mfe"),
            },
        }
        returns = {
            "ret_mean": row.get("ret_mean"),
            "ret_trend": row.get("ret_trend"),
            "drawdown": row.get("drawdown"),
        }
        records.append(
            build_stage_record(
                stage="preds",
                decision_id=decision_id,
                decision_ts_ns=decision_ts_ns,
                source="pipeline",
                run_id=run_id,
                symbol=str(row.get("symbol")),
                timeframe=timeframe,
                strategy_name=str(strategy_name),
                instrument_id=None,
                data=preds or None,
            )
        )
        records.append(
            build_stage_record(
                stage="router",
                decision_id=decision_id,
                decision_ts_ns=decision_ts_ns,
                source="pipeline",
                run_id=run_id,
                symbol=str(row.get("symbol")),
                timeframe=timeframe,
                strategy_name=str(strategy_name),
                instrument_id=None,
                data=router,
            )
        )
        records.append(
            build_stage_record(
                stage="returns",
                decision_id=decision_id,
                decision_ts_ns=decision_ts_ns,
                source="pipeline",
                run_id=run_id,
                symbol=str(row.get("symbol")),
                timeframe=timeframe,
                strategy_name=str(strategy_name),
                instrument_id=None,
                data=returns,
            )
        )

        # Gate stage (from gated logs)
        gate_blocked = False
        gate_decisions = []
        gate_reasons = {}
        gate_archetype = None

        if "gate_ok" in row and pd.notna(row.get("gate_ok")):
            gate_ok_val = bool(row.get("gate_ok"))
            gate_blocked = not gate_ok_val

            if "gate_decision" in row and pd.notna(row.get("gate_decision")):
                gate_decisions = [str(row.get("gate_decision"))]

            if "gate_reasons" in row and pd.notna(row.get("gate_reasons")):
                reasons_str = str(row.get("gate_reasons"))
                # Parse gate_reasons (could be string or list)
                if reasons_str.startswith("[") or reasons_str.startswith("{"):
                    import json

                    try:
                        gate_reasons = json.loads(reasons_str)
                    except:
                        gate_reasons = {"gate_rules": [reasons_str]}
                else:
                    gate_reasons = {"gate_rules": [reasons_str]} if reasons_str else {}

            if "gate_archetype" in row and pd.notna(row.get("gate_archetype")):
                gate_archetype = str(row.get("gate_archetype"))

        gate_data = {
            "blocked": gate_blocked,
            "decisions": gate_decisions,
            "reasons": gate_reasons,
            "archetype": gate_archetype,
        }

        records.append(
            build_stage_record(
                stage="gate",
                decision_id=decision_id,
                decision_ts_ns=decision_ts_ns,
                source="pipeline",
                run_id=run_id,
                symbol=str(row.get("symbol")),
                timeframe=timeframe,
                strategy_name=str(strategy_name),
                instrument_id=None,
                data=gate_data,
            )
        )

        # Execution stage
        router_mode = str(row.get("mode", "NO_TRADE")).upper()
        execution_intent = router_mode != "NO_TRADE" and not gate_blocked

        execution_data = {
            "intent": execution_intent,
            "submit_order": False,  # pipeline is offline evaluation
            "side": None,
            "qty": None,
            "price": None,
            "reason": "pipeline_offline",
            "gate_blocked": gate_blocked,
            "archetype": gate_archetype,
        }

        records.append(
            build_stage_record(
                stage="execution",
                decision_id=decision_id,
                decision_ts_ns=decision_ts_ns,
                source="pipeline",
                run_id=run_id,
                symbol=str(row.get("symbol")),
                timeframe=timeframe,
                strategy_name=str(strategy_name),
                instrument_id=None,
                data=execution_data,
            )
        )
    return records


def load_pipeline_inputs(
    preds_path: Path, mode_path: Optional[Path], logs_path: Optional[Path]
) -> tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    preds_df = _load_multi(preds_path, prefix="preds")
    mode_df = _load_multi(mode_path, prefix="mode") if mode_path else None
    logs_df = _load_multi(logs_path, prefix="logs") if logs_path else None
    return preds_df, mode_df, logs_df
