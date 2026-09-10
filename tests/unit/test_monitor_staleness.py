"""Unit tests for monitor staleness + telegram message formatting."""

from __future__ import annotations

from datetime import datetime, timezone

from src.monitoring.staleness import (
    build_cadence_cards,
    evaluate_cadence_health,
    hours_since_run,
    list_stale_cadences,
)
from src.monitoring.telegram import format_alert_message


def test_hours_since_run_parses_compact_ts():
    now = datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)
    age = hours_since_run("20260601_0600", now=now)
    assert age is not None
    assert 5.9 < age < 6.1


def test_evaluate_missed_when_no_row():
    card = evaluate_cadence_health("weekly", None, max_age_hours=24.0)
    assert card["display_status"] == "MISSED"
    assert card["missed"] is True


def test_evaluate_alert_when_recent_alert():
    now = datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)
    row = {
        "run_ts": "20260601_1100",
        "status": "ALERT",
        "watchdog_any_alert": True,
        "drift_any_alert": False,
    }
    card = evaluate_cadence_health("weekly", row, max_age_hours=48.0, now=now)
    assert card["display_status"] == "ALERT"
    assert card["missed"] is False


def test_list_stale_cadences():
    cfg = {
        "schedules": {"weekly": {}, "monthly": {}},
        "staleness_hours": {"weekly": 1, "monthly": 1},
    }
    index = {
        "cadences": {
            "weekly": {"run_ts": "20200101_0000", "status": "OK"},
        }
    }
    stale = list_stale_cadences(
        index, cfg, now=datetime(2026, 6, 1, tzinfo=timezone.utc)
    )
    names = {c["cadence"] for c in stale}
    assert "weekly" in names
    assert "monthly" in names


def test_format_alert_message_includes_strategies():
    msg = format_alert_message(
        cadence="weekly",
        card={"display_status": "ALERT", "run_ts": "20260601_1200", "exit_code": 1},
        alert_events=[
            {"source": "drift", "strategy": "tpc", "status": "ALERT", "detail": ""}
        ],
        host="testhost",
    )
    assert "weekly" in msg
    assert "drift/tpc" in msg
