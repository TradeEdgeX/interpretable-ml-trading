"""Unit tests for staleness_check module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.monitoring.staleness_check import run_staleness_check


def test_run_staleness_check_all_ok(tmp_path):
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    run_ts = now.strftime("%Y%m%d_%H%M")
    sched = tmp_path / "schedules.yaml"
    sched.write_text(
        "staleness_hours:\n  weekly: 48\nschedules:\n  weekly:\n    manifest: w.yaml\n",
        encoding="utf-8",
    )
    (tmp_path / "results/monitoring").mkdir(parents=True)
    (tmp_path / "results/monitoring/index.json").write_text(
        f'{{"cadences":{{"weekly":{{"run_ts":"{run_ts}","status":"OK"}}}}}}',
        encoding="utf-8",
    )
    assert run_staleness_check(repo_root=tmp_path, schedules_path=sched) == 0


@patch("src.monitoring.staleness_check.notify_stale_cadences", return_value=True)
def test_run_staleness_check_notifies(mock_notify, tmp_path):
    sched = tmp_path / "schedules.yaml"
    sched.write_text(
        "staleness_hours:\n  weekly: 1\nschedules:\n  weekly:\n    manifest: w.yaml\n",
        encoding="utf-8",
    )
    (tmp_path / "results/monitoring").mkdir(parents=True)
    (tmp_path / "results/monitoring/index.json").write_text(
        '{"cadences":{"weekly":{"run_ts":"20200101_0000","status":"OK"}}}',
        encoding="utf-8",
    )
    assert run_staleness_check(repo_root=tmp_path, schedules_path=sched) == 1
    mock_notify.assert_called_once()
