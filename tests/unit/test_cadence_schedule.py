"""Unit tests for cadence schedule date helpers."""

from __future__ import annotations

from datetime import datetime, timezone

from src.monitoring.cadence_schedule import (
    attach_schedule_dates,
    next_on_calendar,
    valid_until_at,
)
from src.monitoring.staleness import build_cadence_cards, evaluate_cadence_health


def test_valid_until_at_from_run_ts():
    iso = valid_until_at("20260611_1000", max_age_hours=192.0)
    assert iso is not None
    assert "2026-06-19" in iso


def test_next_on_calendar_weekly_sunday():
    after = datetime(2026, 6, 11, 10, 0, tzinfo=timezone.utc)  # Wed
    nxt = next_on_calendar("Sun *-*-* 08:00:00", after=after)
    assert nxt.weekday() == 6
    assert nxt.hour == 8
    assert nxt > after


def test_next_on_calendar_daily():
    after = datetime(2026, 6, 11, 7, 0, tzinfo=timezone.utc)
    nxt = next_on_calendar("*-*-* 06:30:00", after=after)
    assert nxt.day == 12
    assert nxt.hour == 6
    assert nxt.minute == 30


def test_build_cadence_cards_includes_schedule_dates():
    cfg = {
        "staleness_hours": {"weekly": 192},
        "schedules": {"weekly": {"manifest": "w.yaml"}},
        "timers": {"weekly": "Sun *-*-* 08:00:00"},
    }
    index = {
        "cadences": {
            "weekly": {"run_ts": "20260611_1000", "status": "OK"},
        }
    }
    now = datetime(2026, 6, 11, 12, 0, tzinfo=timezone.utc)
    cards = build_cadence_cards(index, cfg, now=now)
    assert len(cards) == 1
    card = cards[0]
    assert card.get("valid_until_at")
    assert card.get("next_run_at")
    assert card.get("last_run_at")


def test_attach_schedule_dates_missed_cadence():
    card = evaluate_cadence_health("monthly_c", None, max_age_hours=840.0)
    cfg = {
        "timers": {"monthly_c": "*-*-01 09:30:00"},
        "schedules": {"monthly_c": {}},
    }
    out = attach_schedule_dates(
        card,
        schedules_cfg=cfg,
        now=datetime(2026, 6, 11, tzinfo=timezone.utc),
    )
    assert out["valid_until_at"] is None
    assert out["next_run_at"] is not None


def test_prod_schedules_keep_only_useful_cadences():
    from src.monitoring.store import enabled_schedule_names, load_schedules

    names = set(enabled_schedule_names(load_schedules()))
    assert names == {
        "daily",
        "weekly",
        "monthly",
        "quarterly",
    }
