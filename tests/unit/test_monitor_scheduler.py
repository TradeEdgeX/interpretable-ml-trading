"""Unit tests for monitor scheduler hooks (no subprocess)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.monitoring.scheduler import post_run_hooks, run_cadence


def test_run_cadence_dry_run_skips_index_and_hooks(tmp_path, monkeypatch):
    monkeypatch.setattr("src.monitoring.scheduler.post_run_hooks", MagicMock())
    execute = MagicMock(return_value=(0, "20260101_1200", tmp_path / "out"))
    load_manifest = MagicMock(return_value={"monitor_id": "t", "steps": []})
    cfg_path = tmp_path / "schedules.yaml"
    cfg_path.write_text(
        "schedules:\n  weekly:\n    manifest: weekly.yaml\n",
        encoding="utf-8",
    )
    rc = run_cadence(
        "weekly",
        execute_manifest=execute,
        load_manifest=load_manifest,
        schedules_path=cfg_path,
        repo_root=tmp_path,
        dry_run=True,
    )
    assert rc == 0
    execute.assert_called_once()


def test_post_run_hooks_calls_telegram_on_alert(monkeypatch, tmp_path):
    notify = MagicMock(return_value=True)
    staleness = MagicMock(return_value=0)
    monkeypatch.setattr("src.monitoring.scheduler.notify_cadence_result", notify)
    monkeypatch.setattr("src.monitoring.scheduler.run_staleness_check", staleness)

    index_path = tmp_path / "results/monitoring"
    index_path.mkdir(parents=True)
    (index_path / "index.json").write_text(
        '{"cadences":{"weekly":{"run_ts":"20260101_1200","status":"ALERT","watchdog_any_alert":true}}}',
        encoding="utf-8",
    )
    sched = tmp_path / "config/monitoring/schedules.yaml"
    sched.parent.mkdir(parents=True)
    sched.write_text("schedules:\n  weekly:\n    manifest: w.yaml\n", encoding="utf-8")

    post_run_hooks(
        cadence="weekly",
        exit_code=1,
        schedules_path=sched,
        repo_root=tmp_path,
    )
    notify.assert_called_once()
    staleness.assert_not_called()


def test_post_run_hooks_daily_runs_staleness(monkeypatch, tmp_path):
    monkeypatch.setattr("src.monitoring.scheduler.notify_cadence_result", MagicMock())
    staleness = MagicMock(return_value=0)
    monkeypatch.setattr("src.monitoring.scheduler.run_staleness_check", staleness)
    (tmp_path / "results/monitoring").mkdir(parents=True)
    (tmp_path / "results/monitoring/index.json").write_text(
        '{"cadences":{}}', encoding="utf-8"
    )

    post_run_hooks(
        cadence="daily", exit_code=0, schedules_path=None, repo_root=tmp_path
    )
    staleness.assert_called_once()
