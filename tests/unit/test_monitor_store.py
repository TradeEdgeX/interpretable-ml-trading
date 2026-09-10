"""Unit tests for monitor file/SQLite index."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from src.monitoring.store import (
    index_monitor_run,
    init_registry_db,
    update_monitoring_index,
    upsert_monitor_events_from_run,
)


def _fake_run_dir(tmp_path: Path) -> Path:
    out = tmp_path / "weekly_rule_stack" / "20260101_1200"
    (out / "watchdog" / "wd1").mkdir(parents=True)
    (out / "drift" / "dr1").mkdir(parents=True)
    (out / "heartbeat.json").write_text(
        json.dumps(
            {
                "task": "weekly_rule_stack",
                "status": "ALERT",
                "strategies": "tpc",
            }
        ),
        encoding="utf-8",
    )
    (out / "watchdog" / "wd1" / "report.json").write_text(
        json.dumps(
            {
                "any_alert": True,
                "reports": [{"strategy": "tpc", "any_alert": True, "alerts": ["X"]}],
                "factor_health": {"any_alert": False, "items": []},
            }
        ),
        encoding="utf-8",
    )
    (out / "drift" / "dr1" / "drift_report.json").write_text(
        json.dumps(
            {
                "any_alert": False,
                "report": [{"strategy": "bpc", "any_alert": False, "items": []}],
            }
        ),
        encoding="utf-8",
    )
    return out


def test_update_monitoring_index(tmp_path):
    out = _fake_run_dir(tmp_path)
    idx = update_monitoring_index(
        cadence="weekly",
        run_ts="20260101_1200",
        exit_code=1,
        output_dir=out,
        manifest_path="config/monitoring/weekly_rule_stack.yaml",
        index_path=tmp_path / "index.json",
    )
    data = json.loads(idx.read_text(encoding="utf-8"))
    row = data["cadences"]["weekly"]
    assert row["status"] == "ALERT"
    assert row["drift_any_alert"] is False
    assert row["watchdog_any_alert"] is True
    assert (tmp_path / "latest_weekly.json").is_file()


def test_upsert_monitor_events_sqlite(tmp_path):
    out = _fake_run_dir(tmp_path)
    db = init_registry_db(tmp_path / "registry.sqlite")
    n = upsert_monitor_events_from_run(
        cadence="weekly",
        run_ts="20260101_1200",
        output_dir=out,
        db_path=db,
    )
    assert n >= 1
    conn = sqlite3.connect(db)
    try:
        rows = conn.execute(
            "SELECT strategy, status, source FROM monitor_event ORDER BY strategy"
        ).fetchall()
    finally:
        conn.close()
    strategies = {r[0]: (r[1], r[2]) for r in rows}
    assert "bpc" not in strategies
    assert strategies["tpc"][0] == "ALERT"


def test_drift_no_plateaus_status_in_sqlite(tmp_path):
    out = tmp_path / "out" / "20260101_1200"
    (out / "drift" / "dr1").mkdir(parents=True)
    (out / "heartbeat.json").write_text(
        json.dumps({"task": "monthly", "status": "OK", "strategies": "tpc"}),
        encoding="utf-8",
    )
    (out / "drift" / "dr1" / "drift_report.json").write_text(
        json.dumps(
            {
                "any_alert": False,
                "report": [
                    {
                        "strategy": "tpc",
                        "any_alert": False,
                        "status": "NO_PLATEAUS",
                        "skipped": "plateaus empty",
                        "items": [],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    db = init_registry_db(tmp_path / "registry.sqlite")
    upsert_monitor_events_from_run(
        cadence="monthly",
        run_ts="20260101_1200",
        output_dir=out,
        db_path=db,
    )
    conn = sqlite3.connect(db)
    try:
        row = conn.execute(
            "SELECT status FROM monitor_event WHERE strategy = 'tpc'"
        ).fetchone()
    finally:
        conn.close()
    assert row is None


def test_index_monitor_run_combined(tmp_path):
    out = _fake_run_dir(tmp_path)
    meta = index_monitor_run(
        cadence="weekly",
        run_ts="20260101_1200",
        exit_code=1,
        output_dir=out,
        manifest_path="weekly_rule_stack.yaml",
        registry_db=tmp_path / "registry.sqlite",
        index_path=tmp_path / "index.json",
    )
    assert meta["sqlite_rows"] >= 1
    assert Path(meta["index_path"]).is_file()
