"""Unit tests for CMS dashboard aggregation."""

from __future__ import annotations

import json
from pathlib import Path

from src.monitoring.dashboard import (
    build_monitoring_dashboard,
    compute_alert_streaks,
    compute_feature_alert_streaks,
    enrich_cards_with_details,
    extract_alert_features,
    messages_from_monitor_detail,
    sort_cadence_cards,
    strategy_alerts_by_cadence,
    strategy_uncalibrated_by_cadence,
)
from src.monitoring.store import init_registry_db, upsert_monitor_events_from_run


def test_build_monitoring_dashboard_skips_disabled_cadences(tmp_path):
    sched = tmp_path / "config/monitoring/schedules.yaml"
    sched.parent.mkdir(parents=True)
    sched.write_text(
        """
staleness_hours:
  weekly: 10000
  weekly_c: 10000
schedules:
  weekly:
    manifest: w.yaml
  weekly_c:
    enabled: false
    manifest: c.yaml
""",
        encoding="utf-8",
    )
    idx_dir = tmp_path / "results/monitoring"
    idx_dir.mkdir(parents=True)
    (idx_dir / "index.json").write_text(
        json.dumps(
            {
                "updated_at": "2026-06-01T00:00:00+00:00",
                "cadences": {
                    "weekly": {
                        "run_ts": "20260601_1200",
                        "status": "OK",
                        "exit_code": 0,
                    },
                    "weekly_c": {
                        "run_ts": "20260601_1200",
                        "status": "ALERT",
                        "exit_code": 1,
                        "drift_any_alert": True,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    db = init_registry_db(tmp_path / "registry.sqlite")
    dash = build_monitoring_dashboard(tmp_path, db, schedules_path=sched)
    assert dash["summary"]["n_cards"] == 1
    assert dash["cards"][0]["cadence"] == "weekly"
    assert "weekly_c" not in dash["stale_cadences"]


def test_build_monitoring_dashboard_cards(tmp_path):
    sched = tmp_path / "config/monitoring/schedules.yaml"
    sched.parent.mkdir(parents=True)
    sched.write_text(
        """
staleness_hours:
  weekly: 10000
schedules:
  weekly:
    manifest: w.yaml
""",
        encoding="utf-8",
    )
    idx_dir = tmp_path / "results/monitoring"
    idx_dir.mkdir(parents=True)
    (idx_dir / "index.json").write_text(
        json.dumps(
            {
                "updated_at": "2026-06-01T00:00:00+00:00",
                "cadences": {
                    "weekly": {
                        "run_ts": "20260601_1200",
                        "status": "OK",
                        "exit_code": 0,
                        "watchdog_any_alert": False,
                        "drift_any_alert": False,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    db = init_registry_db(tmp_path / "registry.sqlite")
    dash = build_monitoring_dashboard(tmp_path, db, schedules_path=sched)
    assert dash["summary"]["n_cards"] == 1
    assert dash["cards"][0]["display_status"] == "OK"


def test_strategy_alerts_filters_by_latest_run_ts():
    cards = [{"cadence": "weekly", "run_ts": "20260102_0000"}]
    events = [
        {
            "cadence": "weekly",
            "run_ts": "20260102_0000",
            "source": "drift",
            "strategy": "tpc",
            "status": "ALERT",
        },
        {
            "cadence": "weekly",
            "run_ts": "20260101_0000",
            "source": "drift",
            "strategy": "bpc",
            "status": "ALERT",
        },
    ]
    out = strategy_alerts_by_cadence(events, cards)
    assert len(out["weekly"]) == 1
    assert out["weekly"][0]["strategy"] == "tpc"


def test_strategy_uncalibrated_filters_no_plateaus():
    cards = [{"cadence": "weekly", "run_ts": "20260102_0000"}]
    events = [
        {
            "cadence": "weekly",
            "run_ts": "20260102_0000",
            "source": "drift",
            "strategy": "tpc",
            "status": "NO_PLATEAUS",
        },
        {
            "cadence": "weekly",
            "run_ts": "20260102_0000",
            "source": "drift",
            "strategy": "bpc",
            "status": "OK",
        },
    ]
    out = strategy_uncalibrated_by_cadence(events, cards)
    assert len(out["weekly"]) == 1
    assert out["weekly"][0]["strategy"] == "tpc"


def test_compute_alert_streaks_counts_consecutive_and_resets():
    # tpc alerts newest 3 runs; bpc alerted only 2 runs ago then OK last run.
    events = [
        # newest run
        {
            "cadence": "weekly",
            "run_ts": "20260703_0800",
            "strategy": "tpc",
            "status": "ALERT",
        },
        {
            "cadence": "weekly",
            "run_ts": "20260703_0800",
            "strategy": "bpc",
            "status": "OK",
        },
        {
            "cadence": "weekly",
            "run_ts": "20260626_0800",
            "strategy": "tpc",
            "status": "ALERT",
        },
        {
            "cadence": "weekly",
            "run_ts": "20260626_0800",
            "strategy": "bpc",
            "status": "ALERT",
        },
        {
            "cadence": "weekly",
            "run_ts": "20260619_0800",
            "strategy": "tpc",
            "status": "OK",
        },
        {
            "cadence": "weekly",
            "run_ts": "20260619_0800",
            "strategy": "bpc",
            "status": "ALERT",
        },
    ]
    streaks = compute_alert_streaks(events)
    assert streaks["weekly"]["tpc"]["streak"] == 2
    assert streaks["weekly"]["tpc"]["first_run_ts"] == "20260626_0800"
    # bpc did not alert in newest observed run → no active streak
    assert "bpc" not in streaks.get("weekly", {})


def test_build_dashboard_attaches_streak_to_alert_rows(tmp_path):
    sched = tmp_path / "config/monitoring/schedules.yaml"
    sched.parent.mkdir(parents=True)
    sched.write_text(
        "staleness_hours:\n  weekly: 100000\nschedules:\n  weekly:\n    manifest: w.yaml\n",
        encoding="utf-8",
    )
    idx_dir = tmp_path / "results/monitoring"
    idx_dir.mkdir(parents=True)
    (idx_dir / "index.json").write_text(
        json.dumps(
            {
                "updated_at": "2026-07-03T08:00:00+00:00",
                "cadences": {
                    "weekly": {
                        "run_ts": "20260703_0800",
                        "status": "ALERT",
                        "exit_code": 2,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    db = init_registry_db(tmp_path / "registry.sqlite")
    conn = __import__("sqlite3").connect(db)
    detail = json.dumps({"alerts": ["PSI_DRIFT: ema_1200_position psi=1.4 > 0.25"]})
    runs = [
        ("20260703_0800", "ALERT"),
        ("20260626_0800", "ALERT"),
        ("20260619_0800", "OK"),
    ]
    with conn:
        for rts, st in runs:
            conn.execute(
                "INSERT INTO monitor_event (id, cadence, source, strategy, status, detail_json, run_ts, ts) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (
                    f"weekly:{rts}:watchdog:_factor_health",
                    "weekly",
                    "watchdog",
                    "_factor_health",
                    st,
                    detail if st == "ALERT" else json.dumps({"items": []}),
                    rts,
                    rts,
                ),
            )
    conn.close()
    dash = build_monitoring_dashboard(tmp_path, db, schedules_path=sched)
    rows = dash["strategy_alerts"]["weekly"]
    row = next(r for r in rows if r["strategy"] == "_factor_health")
    assert row["alert_streak"] == 2
    assert row["alert_runs_considered"] == 3
    card = next(c for c in dash["cards"] if c["cadence"] == "weekly")
    assert card["alert_streak_max"] == 2
    assert card["alert_runs_considered"] == 3


def test_extract_alert_features_from_alerts_and_items():
    detail = json.dumps(
        {
            "alerts": [
                "PSI_DRIFT: ema_1200_position psi=5.3 > 0.25",
                "IC_DRIFT: forward_rr ic dropped",
            ],
            "items": [{"kind": "psi", "feature": "adx_50", "psi": 0.4}],
        }
    )
    feats = extract_alert_features(detail)
    assert feats == {"ema_1200_position", "forward_rr", "adx_50"}


def test_compute_feature_alert_streaks_per_feature():
    # ema_1200_position alerts in newest 2 runs; adx_50 only in the oldest.
    d_ema = json.dumps({"alerts": ["PSI_DRIFT: ema_1200_position psi=1.4 > 0.25"]})
    d_both = json.dumps(
        {
            "alerts": [
                "PSI_DRIFT: ema_1200_position psi=1.1 > 0.25",
                "PSI_DRIFT: adx_50 psi=0.9 > 0.25",
            ]
        }
    )
    events = [
        {
            "cadence": "weekly",
            "run_ts": "20260703_0800",
            "strategy": "_factor_health",
            "status": "ALERT",
            "detail_json": d_ema,
        },
        {
            "cadence": "weekly",
            "run_ts": "20260626_0800",
            "strategy": "_factor_health",
            "status": "ALERT",
            "detail_json": d_both,
        },
        {
            "cadence": "weekly",
            "run_ts": "20260619_0800",
            "strategy": "bpc",
            "status": "OK",
        },
    ]
    fs = compute_feature_alert_streaks(events)
    assert fs["weekly"]["_factor_health"]["ema_1200_position"] == 2
    # adx_50 did not alert in the newest run → streak breaks → not active
    assert "adx_50" not in fs["weekly"]["_factor_health"]


def test_sort_cadence_cards_daily_first():
    cards = [
        {"cadence": "yearly"},
        {"cadence": "weekly"},
        {"cadence": "daily"},
    ]
    ordered = [c["cadence"] for c in sort_cadence_cards(cards)]
    assert ordered == ["daily", "weekly", "yearly"]


def test_messages_from_factor_health_detail():
    detail = json.dumps(
        {
            "alerts": ["PSI_DRIFT: ema_1200_position psi=5.351 > 0.25"],
            "items": [
                {
                    "kind": "ic_drift",
                    "skipped": "target 'forward_rr' not in window",
                }
            ],
        }
    )
    msgs = messages_from_monitor_detail(
        detail, source="watchdog", strategy="_factor_health", status="ALERT"
    )
    assert any("PSI_DRIFT" in m for m in msgs)
    assert any("forward_rr" in m for m in msgs)


def test_enrich_cards_with_alert_details():
    cards = [
        {"cadence": "weekly", "run_ts": "20260611_0912", "display_status": "ALERT"}
    ]
    events = [
        {
            "cadence": "weekly",
            "run_ts": "20260611_0912",
            "source": "watchdog",
            "strategy": "_factor_health",
            "status": "ALERT",
            "detail_json": json.dumps(
                {
                    "alerts": ["PSI_DRIFT: ema_1200_position psi=5.351 > 0.25"],
                    "items": [],
                }
            ),
        }
    ]
    out = enrich_cards_with_details(cards, events)
    assert out[0]["alert_details"][0].startswith("[因子健康 (PSI/IC)]")


def test_load_monitoring_index_from_store(tmp_path):
    from src.monitoring.store import load_monitoring_index

    p = tmp_path / "results/monitoring"
    p.mkdir(parents=True)
    (p / "index.json").write_text('{"cadences":{}}', encoding="utf-8")
    assert load_monitoring_index(tmp_path)["cadences"] == {}
