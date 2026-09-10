from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
from datetime import datetime, timezone


SCHEMA_VERSION = "v1"


def _to_iso(ts: Any) -> Optional[str]:
    if ts is None:
        return None
    if isinstance(ts, datetime):
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return ts.isoformat()
    try:
        dt = datetime.fromtimestamp(float(ts) / 1e9, tz=timezone.utc)
        return dt.isoformat()
    except Exception:
        try:
            dt = datetime.fromtimestamp(float(ts), tz=timezone.utc)
            return dt.isoformat()
        except Exception:
            return None


def build_execution_log_record(
    *,
    source: str,
    symbol: str,
    timestamp: Any,
    run_id: Optional[str] = None,
    timeframe: Optional[str] = None,
    strategy_name: Optional[str] = None,
    instrument_id: Optional[str] = None,
    features: Optional[Dict[str, Any]] = None,
    preds: Optional[Dict[str, Any]] = None,
    router: Optional[Dict[str, Any]] = None,
    gate: Optional[Dict[str, Any]] = None,
    evidence: Optional[Dict[str, Any]] = None,
    execution: Optional[Dict[str, Any]] = None,
    returns: Optional[Dict[str, Any]] = None,
    observability: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "source": str(source),
        "run_id": run_id,
        "symbol": str(symbol),
        "timestamp": _to_iso(timestamp) or str(timestamp),
        "timeframe": timeframe,
        "strategy_name": strategy_name,
        "instrument_id": instrument_id,
        "features": features,
        "preds": preds,
        "router": router,
        "gate": gate,
        "evidence": evidence,
        "execution": execution,
        "returns": returns,
        "observability": observability,
    }


@dataclass
class ExecutionLogWriter:
    path: Path

    def __post_init__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: Dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_decision_id(*, strategy_name: str, symbol: str, decision_ts_ns: int) -> str:
    return f"{strategy_name}:{symbol}:{decision_ts_ns}"


def decision_month_key(decision_ts_ns: int) -> str:
    dt = datetime.fromtimestamp(decision_ts_ns / 1e9, tz=timezone.utc)
    return f"{dt.year:04d}-{dt.month:02d}"


def build_stage_record(
    *,
    stage: str,
    decision_id: str,
    decision_ts_ns: int,
    source: str,
    symbol: str,
    run_id: Optional[str] = None,
    timeframe: Optional[str] = None,
    strategy_name: Optional[str] = None,
    instrument_id: Optional[str] = None,
    data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "stage": str(stage),
        "decision_id": str(decision_id),
        "decision_ts_ns": int(decision_ts_ns),
        "source": str(source),
        "run_id": run_id,
        "symbol": str(symbol),
        "timestamp": _to_iso(decision_ts_ns) or str(decision_ts_ns),
        "timeframe": timeframe,
        "strategy_name": strategy_name,
        "instrument_id": instrument_id,
        "data": data,
    }


@dataclass
class ExecutionStageLogWriter:
    base_dir: Path
    stage: str

    def __post_init__(self) -> None:
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def write(self, record: Dict[str, Any], *, decision_ts_ns: int) -> None:
        month_key = decision_month_key(int(decision_ts_ns))
        out_dir = self.base_dir / self.stage
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{month_key}.jsonl"
        with out_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
