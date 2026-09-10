from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from src.time_series_model.diagnostics.execution_log import build_execution_log_record


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_stage_logs(stage_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for p in sorted(stage_dir.glob("*.jsonl")):
        rows.extend(list(_read_jsonl(p)))
    return rows


def aggregate_stage_logs(stage_root: Path) -> List[Dict[str, Any]]:
    stage_root = Path(stage_root)
    agg: Dict[str, Dict[str, Any]] = {}

    for stage_dir in stage_root.iterdir():
        if not stage_dir.is_dir():
            continue
        stage = stage_dir.name
        for row in _read_jsonl_from_dir(stage_dir):
            decision_id = row.get("decision_id")
            if not decision_id:
                continue
            entry = agg.setdefault(
                decision_id,
                {
                    "source": row.get("source"),
                    "run_id": row.get("run_id"),
                    "symbol": row.get("symbol"),
                    "timestamp": row.get("timestamp"),
                    "timeframe": row.get("timeframe"),
                    "strategy_name": row.get("strategy_name"),
                    "instrument_id": row.get("instrument_id"),
                    "stages": {},
                },
            )
            entry["stages"][stage] = row.get("data")

    records: List[Dict[str, Any]] = []
    for entry in agg.values():
        stages = entry.get("stages") or {}
        rec = build_execution_log_record(
            source=entry.get("source") or "unknown",
            run_id=entry.get("run_id"),
            symbol=entry.get("symbol") or "",
            timestamp=entry.get("timestamp"),
            timeframe=entry.get("timeframe"),
            strategy_name=entry.get("strategy_name"),
            instrument_id=entry.get("instrument_id"),
            features=stages.get("features"),
            preds=stages.get("preds"),
            router=stages.get("router"),
            gate=stages.get("gate"),
            evidence=stages.get("evidence"),
            execution=stages.get("execution"),
            returns=stages.get("returns"),
            observability=stages.get("observability"),
        )
        records.append(rec)
    return records


def _read_jsonl_from_dir(stage_dir: Path) -> Iterable[Dict[str, Any]]:
    for p in sorted(stage_dir.glob("*.jsonl")):
        yield from _read_jsonl(p)
